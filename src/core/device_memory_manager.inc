// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <algorithm>
#include <vector>

#include <boost/container/deque.hpp>

#include "common/address_space.h"
#include "common/address_space.inc"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/div_ceil.h"
#include "common/fritchii_write_ring.h"
#include "common/nx2_uniform_keep_pages.h"
#include "common/paged_scalar.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/device_memory.h"
#include "core/device_memory_manager.h"
#include "core/memory.h"
#include "core/nx2_protect_stats.h"
#include "video_core/back_read_audit.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/rasterizer_download_area.h"

// The read's caller as an offset in the executable (resolve it with the build's PDB or map):
// the "caller" of the unmapped read lines, for reads no ReadSiteScope names. With LTO the
// read may be inlined into its caller, which then names that caller's caller.
#if defined(__clang__) || defined(__GNUC__)
#define FRITCHII_READ_CALLER() __builtin_return_address(0)
#elif defined(_MSC_VER)
#include <intrin.h>
#define FRITCHII_READ_CALLER() _ReturnAddress()
#else
#define FRITCHII_READ_CALLER() nullptr
#endif
#ifdef _WIN32
extern "C" const char __ImageBase; // the executable's base (MSVC, lld, MinGW)
#endif

namespace Core {

namespace {

// Fritchii: reads of unmapped device memory, one line a read. The per-page lines this replaces
// shared one cap of 16 for the whole session, so a single 1 MiB read used it up on its first 16
// pages (AMD tester t2057: "16 pages at a block start" was the cap, not the read) and every
// later read went unlogged. Now: the read's size, how much of it was unmapped, the thread and
// the write ring's site, and the session totals; the first 32 reads, then every 1024th.
std::atomic<u64> g_unmapped_reads{};
std::atomic<u64> g_unmapped_read_bytes{};

#ifdef _WIN32
u64 CallerOffset(const void* caller) {
    const u64 base = reinterpret_cast<u64>(&::__ImageBase);
    const u64 at = reinterpret_cast<u64>(caller);
    return at >= base ? at - base : at;
}
#else
u64 CallerOffset(const void* caller) {
    return reinterpret_cast<u64>(caller);
}
#endif

void ReportUnmappedRead(const char* what, DAddr address, size_t size, DAddr first_unmapped,
                        size_t unmapped_bytes, const void* caller) {
    const u64 reads = g_unmapped_reads.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 total = g_unmapped_read_bytes.fetch_add(unmapped_bytes, std::memory_order_relaxed) +
                      unmapped_bytes;
    if (reads > 32 && (reads & 1023) != 0) {
        return;
    }
    std::array<char, 17> thread{};
    std::memcpy(thread.data(), Common::WriteRing::t_thread.data(), 16);
    // The read's own site (ReadSiteScope), else the write ring's (on with its switch only).
    const char* const read_site = Common::WriteRing::t_read_site;
    const char* const site = read_site != nullptr ? read_site : Common::WriteRing::t_site;
    const char* const context =
        read_site != nullptr ? Common::WriteRing::t_read_context : Common::WriteRing::t_context;
    LOG_ERROR(HW_Memory,
              "Unmapped Device {} @ {:#016x} size {:#x}: {} KiB unmapped, first at {:#016x} "
              "(read as zeros; thread {}, site {}{}{}, caller exe+{:#x}); {} such reads, {} KiB "
              "unmapped this session",
              what, address, size, unmapped_bytes >> 10, first_unmapped,
              thread[0] != '\0' ? thread.data() : "?", site != nullptr ? site : "-",
              context != nullptr && context != site ? " in " : "",
              context != nullptr && context != site ? context : "", CallerOffset(caller), reads,
              total >> 10);
}

class MultiAddressContainer {
public:
    MultiAddressContainer() = default;
    ~MultiAddressContainer() = default;

    void GatherValues(u32 start_entry, Common::ScratchBuffer<u32>& buffer) {
        buffer.resize(8);
        const auto add_value = [&buffer](u32 value, size_t index) {
            if (buffer.size() < index + 1)
                buffer.resize(index + 8);
            buffer[index++] = value;
            return index;
        };

        size_t index = 0;
        u32 iter_entry = start_entry;
        Entry* current = &storage[iter_entry - 1];
        index = add_value(current->value, index);
        while (current->next_entry != 0) {
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
            index = add_value(current->value, index);
        }
        buffer.resize(index);
    }

    u32 Register(u32 value) {
        return RegisterImplementation(value);
    }

    void Register(u32 value, u32 start_entry) {
        auto entry_id = RegisterImplementation(value);
        u32 iter_entry = start_entry;
        Entry* current = &storage[iter_entry - 1];
        while (current->next_entry != 0) {
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
        }
        current->next_entry = entry_id;
    }

    std::pair<bool, u32> Unregister(u32 value, u32 start_entry) {
        u32 iter_entry = start_entry;
        Entry* previous{};
        Entry* current = &storage[iter_entry - 1];
        Entry* next{};
        bool more_than_one_remaining = false;
        u32 result_start{start_entry};
        size_t count = 0;
        while (current->value != value) {
            count++;
            previous = current;
            iter_entry = current->next_entry;
            current = &storage[iter_entry - 1];
        }
        // Find next
        u32 next_entry = current->next_entry;
        if (next_entry != 0) {
            next = &storage[next_entry - 1];
            more_than_one_remaining = next->next_entry != 0 || previous != nullptr;
        }
        if (previous) {
            previous->next_entry = next_entry;
        } else {
            result_start = next_entry;
        }
        free_entries.emplace_back(iter_entry);
        return std::make_pair(more_than_one_remaining || count > 1, result_start);
    }

    u32 ReleaseEntry(u32 start_entry) {
        Entry* current = &storage[start_entry - 1];
        free_entries.emplace_back(start_entry);
        return current->value;
    }

private:
    u32 RegisterImplementation(u32 value) {
        auto entry_id = GetNewEntry();
        auto& entry = storage[entry_id - 1];
        entry.next_entry = 0;
        entry.value = value;
        return entry_id;
    }
    u32 GetNewEntry() {
        if (!free_entries.empty()) {
            u32 result = free_entries.front();
            free_entries.pop_front();
            return result;
        }
        storage.emplace_back();
        u32 new_entry = static_cast<u32>(storage.size());
        return new_entry;
    }

    struct Entry {
        u32 next_entry{};
        u32 value{};
    };

    std::vector<Entry> storage;
    boost::container::deque<u32> free_entries;
};

struct EmptyAllocator {
    EmptyAllocator([[maybe_unused]] DAddr address) {}
};

} // namespace

template <typename DTraits>
struct DeviceMemoryManagerAllocator {
    static constexpr size_t device_virtual_bits = DTraits::device_virtual_bits;
    static constexpr DAddr first_address = 1ULL << Memory::YUZU_PAGEBITS;
    static constexpr DAddr max_device_area = 1ULL << device_virtual_bits;

    DeviceMemoryManagerAllocator() : main_allocator(first_address) {}

    Common::FlatAllocator<DAddr, 0, device_virtual_bits> main_allocator;
    MultiAddressContainer multi_dev_address;

    /// Returns true when vaddr -> vaddr+size is fully contained in the buffer
    template <bool pin_area>
    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= 0 && addr + size <= max_device_area;
    }

    DAddr Allocate(size_t size) {
        return main_allocator.Allocate(size);
    }

    void AllocateFixed(DAddr b_address, size_t b_size) {
        main_allocator.AllocateFixed(b_address, b_size);
    }

    void Free(DAddr b_address, size_t b_size) {
        main_allocator.Free(b_address, b_size);
    }
};

template <typename Traits>
DeviceMemoryManager<Traits>::DeviceMemoryManager(const DeviceMemory& device_memory_)
    : physical_base{uintptr_t(device_memory_.buffer.BackingBasePointer())}
    , device_memory_ptr{&device_memory_}
    , device_inter{nullptr}
    , compressed_device_addr(1ULL << ((Settings::values.memory_layout_mode.GetValue() == Settings::MemoryLayout::Memory_4Gb ? physical_min_bits : physical_max_bits) - Memory::YUZU_PAGEBITS))
    , tracked_entries(device_as_size >> Memory::YUZU_PAGEBITS)
{
    impl = std::make_unique<DeviceMemoryManagerAllocator<Traits>>();
    cached_pages = std::make_unique<CachedPages>();
    // FRITCHII_MEM_CENSUS (no-ops when off)
    compressed_device_addr.SetCensusName("SMMU address map");
    tracked_entries.SetCensusName("SMMU page entries");
}

template <typename Traits>
DeviceMemoryManager<Traits>::~DeviceMemoryManager() = default;

template <typename Traits>
void DeviceMemoryManager<Traits>::HintCold(DAddr address, size_t size) {
    struct Run {
        u64 physical_offset;
        VAddr virtual_address; ///< 0: no CPU mapping known
        u64 size;
    };
    std::vector<Run> runs;
    const size_t page_end = Common::DivCeil(address + size, Memory::YUZU_PAGESIZE);
    for (size_t page = address >> Memory::YUZU_PAGEBITS; page < page_end; ++page) {
        const u32 compressed = tracked_entries[page].compressed_physical_ptr;
        if (compressed == 0) {
            continue;
        }
        const u64 physical = u64{compressed - 1} << Memory::YUZU_PAGEBITS;
        const VAddr vaddr = ExtractCPUBacking(page).second;
        if (!runs.empty()) {
            Run& last = runs.back();
            const bool contiguous_view =
                last.virtual_address == 0 ? vaddr == 0 : last.virtual_address + last.size == vaddr;
            if (last.physical_offset + last.size == physical && contiguous_view) {
                last.size += Memory::YUZU_PAGESIZE;
                continue;
            }
        }
        runs.push_back({physical, vaddr, Memory::YUZU_PAGESIZE});
    }
    if (runs.empty()) {
        return;
    }
    if (!cold_hint_worker) {
        cold_hint_worker = std::make_unique<Common::ThreadWorker>(1, "DeviceMemory:ColdHint");
    }
    cold_hint_worker->QueueWork([this, runs = std::move(runs)] {
        for (const Run& run : runs) {
            device_memory_ptr->buffer.HintCold(
                run.physical_offset, run.size,
                run.virtual_address != 0 ? std::optional<size_t>{run.virtual_address}
                                         : std::nullopt);
        }
    });
}

template <typename Traits>
void DeviceMemoryManager<Traits>::BindInterface(DeviceInterface* device_inter_) {
    device_inter = device_inter_;
}

template <typename Traits>
DAddr DeviceMemoryManager<Traits>::Allocate(size_t size) {
    return impl->Allocate(size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::AllocateFixed(DAddr start, size_t size) {
    return impl->AllocateFixed(start, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Free(DAddr start, size_t size) {
    impl->Free(start, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Map(DAddr address, VAddr virtual_address, size_t size, Asid asid,
                                      bool track) {
    Core::Memory::Memory* process_memory = registered_processes[asid.id];
    size_t start_page_d = address >> Memory::YUZU_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::YUZU_PAGESIZE) >> Memory::YUZU_PAGEBITS;
    // EXP92: watches the uniform keeps hold on these device pages are given back while the old
    // guest address is known (before EXP89's release, which uncaches a page this lets linger).
    UnwatchUniformKeepPages(address, size);
    // EXP89: a page still lingering here is protected at its old guest address, which
    // InsertCPUBacking below replaces: uncache it there first.
    ReleaseLingeringRange(address, size);
    std::scoped_lock lk(mapping_guard);

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages);
    CutContinuityAt(start_page_d);
    for (size_t i = 0; i < num_pages; i++) {
        const VAddr new_vaddress = virtual_address + i * Memory::YUZU_PAGESIZE;
        auto* ptr = process_memory->GetPointerSilent(Common::ProcessAddress(new_vaddress));
        // One page per run unless tracked below: a run left from an earlier mapping of these
        // device pages would make WalkBlock copy the host memory that followed that mapping.
        tracked_entries.GetUnchecked(start_page_d + i).continuity_tracker = 0;
        if (ptr == nullptr) [[unlikely]] {
            tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = 0;
            continue;
        }
        auto phys_addr = static_cast<u32>(GetRawPhysicalAddr(ptr) >> Memory::YUZU_PAGEBITS) + 1U;
        tracked_entries.GetUnchecked(start_page_d + i).compressed_physical_ptr = phys_addr;
        InsertCPUBacking(start_page_d + i, new_vaddress, asid);
        const u32 base_dev = compressed_device_addr[phys_addr - 1U];
        const u32 new_dev = static_cast<u32>(start_page_d + i);
        if (base_dev == 0) [[likely]] {
            compressed_device_addr.GetAndFault(phys_addr - 1U) = new_dev;
            continue;
        }
        u32 start_id = base_dev & MULTI_MASK;
        if ((base_dev >> MULTI_FLAG_BITS) == 0) {
            start_id = impl->multi_dev_address.Register(base_dev);
            compressed_device_addr.GetAndFault(phys_addr - 1U) = MULTI_FLAG | start_id;
        }
        impl->multi_dev_address.Register(new_dev, start_id);
    }
    translation_generation.fetch_add(1, std::memory_order_release);
    if (track) {
        TrackContinuityImpl(address, virtual_address, size, asid);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::Unmap(DAddr address, size_t size) {
    size_t start_page_d = address >> Memory::YUZU_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::YUZU_PAGESIZE) >> Memory::YUZU_PAGEBITS;
    device_inter->InvalidateRegion(address, size);
    // FRITCHII_FENCE_COPY_UNLOCKED: a fence download the fencing thread is writing here without
    // the buffer cache's lock ends before the backing is erased (with the lock held for the whole
    // pop, the invalidation above waited for it).
    VideoCore::DownloadLanding::AwaitUnlockedFenced(address, size);
    // EXP92: as in Map, before the backing is erased (and before EXP89's release).
    UnwatchUniformKeepPages(address, size);
    // EXP89: the pages the caches just dropped may linger; the uncache needs their guest
    // address, which the loop below erases.
    ReleaseLingeringRange(address, size);
    SCOPE_EXIT {
        // EXP89 verify (fable52_answer_after88 1.6): no page of an unmapped range may be left
        // lingering. Such a page can't be uncached any more (no guest address): drop the note.
        if (VideoCore::Nx2Debug::Experiment(89) != 0 && Core::Nx2ProtectStats::Enabled()) {
            u64 left = 0;
            {
                std::scoped_lock linger_lock{linger_mutex};
                for (size_t i = 0; i < num_pages && !lingering.empty(); ++i) {
                    if (lingering.erase(start_page_d + i) != 0) {
                        lingering_count.fetch_sub(1, std::memory_order_relaxed);
                        ++left;
                    }
                }
            }
            if (left != 0) {
                Core::Nx2ProtectStats::AddLinger(
                    &Core::Nx2ProtectStats::State::linger_left_after_unmap, left);
            }
        }
    };
    std::scoped_lock lk(mapping_guard);

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages); // should already be committed, but just in case
    CutContinuityAt(start_page_d);
    for (size_t i = 0; i < num_pages; i++) {
        auto& entry = tracked_entries.GetUnchecked(start_page_d + i);
        auto phys_addr = entry.compressed_physical_ptr;
        entry.compressed_physical_ptr = 0;
        entry.cpu_backing_address = 0;
        entry.continuity_tracker = 0;
        if (phys_addr != 0) [[likely]] {
            u32& base_dev = compressed_device_addr.GetAndFault(phys_addr - 1U);
            if ((base_dev >> MULTI_FLAG_BITS) == 0) [[likely]] {
                base_dev = 0;
                continue;
            }
            const auto [more_entries, new_start] = impl->multi_dev_address.Unregister(
                static_cast<u32>(start_page_d + i), base_dev & MULTI_MASK);
            if (!more_entries) {
                base_dev = impl->multi_dev_address.ReleaseEntry(new_start);
                continue;
            }
            base_dev = new_start | MULTI_FLAG;
        }
    }
    translation_generation.fetch_add(1, std::memory_order_release);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::CutContinuityAt(size_t page) {
    // Runs counted from the pages before `page` end there: the pages from `page` on are being
    // mapped again or unmapped. A run is counted down page by page (TrackContinuityImpl), so the
    // first page whose run already ends before `page` ends the walk.
    for (size_t index = page; index-- > 0;) {
        const size_t last_count = page - 1 - index;
        if (tracked_entries[index].continuity_tracker <= last_count) {
            break;
        }
        tracked_entries.GetUnchecked(index).continuity_tracker = static_cast<u32>(last_count);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::TrackContinuityImpl(DAddr address, VAddr virtual_address,
                                                      size_t size, Asid asid) {
    Core::Memory::Memory* process_memory = registered_processes[asid.id];
    size_t start_page_d = address >> Memory::YUZU_PAGEBITS;
    size_t num_pages = Common::AlignUp(size, Memory::YUZU_PAGESIZE) >> Memory::YUZU_PAGEBITS;
    uintptr_t last_ptr = 0;
    size_t page_count = 1;

    tracked_entries.CommitRegion(start_page_d, start_page_d + num_pages);
    for (size_t i = num_pages; i > 0; i--) {
        size_t index = i - 1;
        const VAddr new_vaddress = virtual_address + index * Memory::YUZU_PAGESIZE;
        const uintptr_t new_ptr = reinterpret_cast<uintptr_t>(
            process_memory->GetPointerSilent(Common::ProcessAddress(new_vaddress)));
        if (new_ptr + page_size == last_ptr) {
            page_count++;
        } else {
            page_count = 1;
        }
        last_ptr = new_ptr;
        tracked_entries.GetUnchecked(start_page_d + index).continuity_tracker = static_cast<u32>(page_count) - 1;
    }
}
// FRITCHII_BACK_READ_AUDIT (video_core/back_read_audit.cpp): the read functions below count
// their caller on the GPU back thread. One branch on a static bool each while off.
template <typename Traits>
u8* DeviceMemoryManager<Traits>::GetSpan(const DAddr src_addr, const std::size_t size) {
    const auto body = [&]() -> u8* {
        size_t page_index = src_addr >> page_bits;
        size_t subbits = src_addr & page_mask;
        if ((static_cast<size_t>(tracked_entries[page_index].continuity_tracker+1) << page_bits) >= size + subbits) {
            return GetPointer<u8>(src_addr);
        }
        return nullptr;
    };
    FRITCHII_BACK_READ_AUDITED(size, VideoCommon::BackReadAudit::Kind::DevGetSpanMut, body);
}

template <typename Traits>
const u8* DeviceMemoryManager<Traits>::GetSpan(const DAddr src_addr, const std::size_t size) const {
    const auto body = [&]() -> const u8* {
        size_t page_index = src_addr >> page_bits;
        size_t subbits = src_addr & page_mask;
        if ((static_cast<size_t>(tracked_entries[page_index].continuity_tracker+1) << page_bits) >= size + subbits) {
            return GetPointer<u8>(src_addr);
        }
        return nullptr;
    };
    FRITCHII_BACK_READ_AUDITED(size, VideoCommon::BackReadAudit::Kind::DevGetSpan, body);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::InnerGatherDeviceAddresses(Common::ScratchBuffer<u32>& buffer,
                                                             PAddr address) {
    size_t phys_addr = address >> page_bits;
    std::scoped_lock lk(mapping_guard);
    u32 backing = compressed_device_addr[phys_addr];
    if ((backing >> MULTI_FLAG_BITS) != 0) {
        impl->multi_dev_address.GatherValues(backing & MULTI_MASK, buffer);
        return;
    }
    buffer.resize(1);
    buffer[0] = backing;
}

template <typename Traits>
template <typename T>
T* DeviceMemoryManager<Traits>::GetPointer(DAddr address) {
    FRITCHII_BACK_READ_NOTE(0, VideoCommon::BackReadAudit::Kind::DevGetPointerMut);
    const size_t index = address >> Memory::YUZU_PAGEBITS;
    const size_t offset = address & Memory::YUZU_PAGEMASK;
    auto phys_addr = tracked_entries[index].compressed_physical_ptr;
    if (phys_addr == 0) [[unlikely]]
        return nullptr;
    return GetPointerFromRaw<T>((PAddr(phys_addr - 1) << Memory::YUZU_PAGEBITS) + offset);
}

template <typename Traits>
template <typename T>
const T* DeviceMemoryManager<Traits>::GetPointer(DAddr address) const {
    FRITCHII_BACK_READ_NOTE(0, VideoCommon::BackReadAudit::Kind::DevGetPointer);
    const size_t index = address >> Memory::YUZU_PAGEBITS;
    const size_t offset = address & Memory::YUZU_PAGEMASK;
    auto phys_addr = tracked_entries[index].compressed_physical_ptr;
    if (phys_addr == 0)
        return nullptr;
    return GetPointerFromRaw<T>((PAddr(phys_addr - 1) << Memory::YUZU_PAGEBITS) + offset);
}

template <typename Traits>
template <typename T>
void DeviceMemoryManager<Traits>::Write(DAddr address, T value) {
    // A write: its pointer getter is hidden from the back read audit.
    VideoCommon::BackReadAudit::Unaudited([&] {
        Common::WritePagedScalar(address, value, [&](DAddr page_address) {
            return page_address < device_as_size ? GetPointer<u8>(page_address) : nullptr;
        });
    });
    Common::UniformKeepPages::Note(address, sizeof(T)); // EXP92, after the store
    Common::WriteRing::Note(address, sizeof(T), "device Write");
}

template <typename Traits>
template <typename T>
T DeviceMemoryManager<Traits>::Read(DAddr address) const {
    const auto body = [&]() -> T {
        return Common::ReadPagedScalar<T>(address, [&](DAddr page_address) {
            return page_address < device_as_size ? GetPointer<u8>(page_address) : nullptr;
        });
    };
    FRITCHII_BACK_READ_AUDITED(sizeof(T), VideoCommon::BackReadAudit::Kind::DevRead, body);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WalkBlock(DAddr addr, std::size_t size, auto on_unmapped, auto on_memory, auto increment) {
    std::size_t remaining_size = size;
    std::size_t page_index = addr >> Memory::YUZU_PAGEBITS;
    std::size_t page_offset = addr & Memory::YUZU_PAGEMASK;
    while (remaining_size) {
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        // An unmapped page says nothing about the pages after it: its run cannot be trusted, so
        // only this page is handed to on_unmapped before the walk moves on.
        const size_t next_pages =
            phys_addr == 0 ? 1 : std::size_t(tracked_entries[page_index].continuity_tracker + 1);
        const std::size_t copy_amount = (std::min)((next_pages << Memory::YUZU_PAGEBITS) - page_offset, remaining_size);
        const auto current_vaddr = u64((page_index << Memory::YUZU_PAGEBITS) + page_offset);
        SCOPE_EXIT{
            page_index += next_pages;
            page_offset = 0;
            increment(copy_amount);
            remaining_size -= copy_amount;
        };

        if (phys_addr == 0) {
            on_unmapped(copy_amount, current_vaddr);
            continue;
        }
        auto* mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::YUZU_PAGEBITS) + page_offset);
        on_memory(copy_amount, mem_ptr);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReadBlock(DAddr address, void* dest_pointer, size_t size) {
    FRITCHII_BACK_READ_NOTE(size, VideoCommon::BackReadAudit::Kind::DevReadBlock);
    device_inter->FlushRegion(address, size);
    const std::size_t page_offset = address & Memory::YUZU_PAGEMASK;
    if (size <= Memory::YUZU_PAGESIZE - page_offset) {
        const DAddr guest_page = address & ~static_cast<DAddr>(Memory::YUZU_PAGEMASK);
        TranslationCache& cache = Translations();
        for (const TranslationEntry& entry : cache.slots) {
            if (entry.guest_page == guest_page && entry.host_ptr != nullptr) {
                std::memcpy(dest_pointer, entry.host_ptr + page_offset, size);
                return;
            }
        }

        const std::size_t page_index = address >> Memory::YUZU_PAGEBITS;
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        if (phys_addr != 0) {
            auto* const mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::YUZU_PAGEBITS));
            cache.slots[cache.cursor] = TranslationEntry{.guest_page = guest_page, .host_ptr = mem_ptr};
            cache.cursor = (cache.cursor + 1) & 3U;
            std::memcpy(dest_pointer, mem_ptr + page_offset, size);
            return;
        }
    }
    size_t unmapped_bytes = 0;
    DAddr first_unmapped = 0;
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (unmapped_bytes == 0) {
                first_unmapped = current_vaddr;
            }
            unmapped_bytes += copy_amount;
            std::memset(dest_pointer, 0, copy_amount);
        },
        [&](size_t copy_amount, const u8* const src_ptr) {
            std::memcpy(dest_pointer, src_ptr, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            dest_pointer = static_cast<u8*>(dest_pointer) + copy_amount;
        });
    if (unmapped_bytes != 0) [[unlikely]] {
        ReportUnmappedRead(__func__, address, size, first_unmapped, unmapped_bytes,
                           FRITCHII_READ_CALLER());
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WriteBlock(DAddr address, const void* src_pointer, size_t size) {
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(
                    HW_Memory,
                    "Unmapped Device WriteBlock @ {:#016x} (start address = {:#016x}, size = {})",
                    current_vaddr, address, size);
            }
        },
        [&](size_t copy_amount, u8* const dst_ptr) {
            std::memcpy(dst_ptr, src_pointer, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            src_pointer = static_cast<const u8*>(src_pointer) + copy_amount;
        });
    Common::UniformKeepPages::Note(address, size); // EXP92, after the stores
    Common::WriteRing::Note(address, size, "device WriteBlock");
    device_inter->InvalidateRegion(address, size);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReadBlockUnsafe(DAddr address, void* dest_pointer, size_t size) {
    FRITCHII_BACK_READ_NOTE(size, VideoCommon::BackReadAudit::Kind::DevReadBlockUnsafe);
    const std::size_t page_offset = address & Memory::YUZU_PAGEMASK;
    if (size <= Memory::YUZU_PAGESIZE - page_offset) {
        const DAddr guest_page = address & ~static_cast<DAddr>(Memory::YUZU_PAGEMASK);
        TranslationCache& cache = Translations();
        for (const TranslationEntry& entry : cache.slots) {
            if (entry.guest_page == guest_page && entry.host_ptr != nullptr) {
                std::memcpy(dest_pointer, entry.host_ptr + page_offset, size);
                return;
            }
        }

        const std::size_t page_index = address >> Memory::YUZU_PAGEBITS;
        const auto phys_addr = tracked_entries[page_index].compressed_physical_ptr;
        if (phys_addr != 0) {
            auto* const mem_ptr = GetPointerFromRaw<u8>((PAddr(phys_addr - 1) << Memory::YUZU_PAGEBITS));
            cache.slots[cache.cursor] = TranslationEntry{.guest_page = guest_page, .host_ptr = mem_ptr};
            cache.cursor = (cache.cursor + 1) & 3U;
            std::memcpy(dest_pointer, mem_ptr + page_offset, size);
            return;
        }
    }
    size_t unmapped_bytes = 0;
    DAddr first_unmapped = 0;
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            if (unmapped_bytes == 0) {
                first_unmapped = current_vaddr;
            }
            unmapped_bytes += copy_amount;
            std::memset(dest_pointer, 0, copy_amount);
        },
        [&](size_t copy_amount, const u8* const src_ptr) {
            std::memcpy(dest_pointer, src_ptr, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            dest_pointer = static_cast<u8*>(dest_pointer) + copy_amount;
        });
    if (unmapped_bytes != 0) [[unlikely]] {
        ReportUnmappedRead(__func__, address, size, first_unmapped, unmapped_bytes,
                           FRITCHII_READ_CALLER());
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::WriteBlockUnsafe(DAddr address, const void* src_pointer,
                                                   size_t size) {
    WalkBlock(
        address, size,
        [&](size_t copy_amount, DAddr current_vaddr) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(
                    HW_Memory,
                    "Unmapped Device WriteBlock @ {:#016x} (start address = {:#016x}, size = {})",
                    current_vaddr, address, size);
            }
        },
        [&](size_t copy_amount, u8* const dst_ptr) {
            std::memcpy(dst_ptr, src_pointer, copy_amount);
        },
        [&](const std::size_t copy_amount) {
            src_pointer = static_cast<const u8*>(src_pointer) + copy_amount;
        });
    Common::UniformKeepPages::Note(address, size); // EXP92, after the stores
    Common::WriteRing::Note(address, size, "device WriteBlockUnsafe");
}

template <typename Traits>
Asid DeviceMemoryManager<Traits>::RegisterProcess(Memory::Memory* memory_device_inter) {
    size_t new_id{};
    if (!id_pool.empty()) {
        new_id = id_pool.front();
        id_pool.pop_front();
        registered_processes[new_id] = memory_device_inter;
    } else {
        registered_processes.emplace_back(memory_device_inter);
        new_id = registered_processes.size() - 1U;
    }
    return Asid{new_id};
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UnregisterProcess(Asid asid) {
    // EXP89: lingering pages are uncached while their process interface is still known.
    ReleaseAllLingering();
    registered_processes[asid.id] = nullptr;
    id_pool.push_front(asid.id);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedCountNoLock(DAddr addr, size_t size, s32 delta) {
    u64 uncache_begin = 0;
    u64 cache_begin = 0;
    u64 uncache_bytes = 0;
    u64 cache_bytes = 0;
    // EDEN_NX2_PROTECT_STATS=1: every call counted and timed (Core::Nx2ProtectStats).
    const auto MarkRegionCaching = [this](Memory::Memory* memory_inter, u64 vaddr, u64 bytes,
                                          bool cached) {
        MarkCaching(memory_inter, vaddr, bytes, cached);
    };
    // EXP89 = N > 0: a page whose count falls to 0 stays protected for N frames (lingering
    // protection, fable52_answer_after88 1.3); read live.
    const int linger = VideoCore::Nx2Debug::Experiment(89);

    std::atomic_thread_fence(std::memory_order_acquire);
    const size_t page_end = Common::DivCeil(addr + size, Memory::YUZU_PAGESIZE);
    size_t page = addr >> Memory::YUZU_PAGEBITS;
    auto [asid, base_vaddress] = ExtractCPUBacking(page);
    auto* memory_device_inter =
        asid.id < registered_processes.size() ? registered_processes[asid.id] : nullptr;
    const auto release_pending = [&] {
        if (uncache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, uncache_begin << Memory::YUZU_PAGEBITS,
                                  uncache_bytes, false);
            }
            uncache_bytes = 0;
        }
        if (cache_bytes > 0) {
            if (memory_device_inter != nullptr) {
                MarkRegionCaching(memory_device_inter, cache_begin << Memory::YUZU_PAGEBITS,
                                  cache_bytes, true);
            }
            cache_bytes = 0;
        }
    };
    size_t old_vpage = (base_vaddress >> Memory::YUZU_PAGEBITS) - 1;
    for (; page != page_end; ++page) {
        CounterAtomicType& count = cached_pages->at(page >> subentries_shift).Count(page);
        auto [asid_2, vpage] = ExtractCPUBacking(page);
        vpage >>= Memory::YUZU_PAGEBITS;

        if (vpage == 0) [[unlikely]] {
            release_pending();
            continue;
        }

        if (asid.id != asid_2.id) [[unlikely]] {
            release_pending();
            asid = asid_2;
            memory_device_inter =
                asid_2.id < registered_processes.size() ? registered_processes[asid_2.id] : nullptr;
        }

        if (vpage != old_vpage + 1) [[unlikely]] {
            release_pending();
        }

        old_vpage = vpage;

        // Adds or subtracts 1, as count is a unsigned 8-bit value
        count.fetch_add(static_cast<CounterType>(delta), std::memory_order_release);
        const auto now = count.load(std::memory_order::relaxed);

        const auto flush_uncache_run = [&] {
            if (uncache_bytes > 0) {
                if (memory_device_inter != nullptr) {
                    MarkRegionCaching(memory_device_inter, uncache_begin << Memory::YUZU_PAGEBITS,
                                      uncache_bytes, false);
                }
                uncache_bytes = 0;
            }
        };
        const auto flush_cache_run = [&] {
            if (cache_bytes > 0) {
                if (memory_device_inter != nullptr) {
                    MarkRegionCaching(memory_device_inter, cache_begin << Memory::YUZU_PAGEBITS,
                                      cache_bytes, true);
                }
                cache_bytes = 0;
            }
        };

        // Assume delta is either -1 or 1
        if (now == 0) {
            if (linger > 0 && delta < 0 && NoteLingering(page, memory_device_inter)) {
                // EXP89: stays protected (the sweep or Map/Unmap uncaches it later).
                flush_uncache_run();
            } else {
                if (uncache_bytes == 0) {
                    uncache_begin = vpage;
                }
                uncache_bytes += Memory::YUZU_PAGESIZE;
            }
        } else {
            flush_uncache_run();
        }
        if (now == 1 && delta > 0) {
            if (TakeLingering(page)) {
                // EXP89: still protected from before, nothing to call.
                flush_cache_run();
            } else {
                if (cache_bytes == 0) {
                    cache_begin = vpage;
                }
                cache_bytes += Memory::YUZU_PAGESIZE;
            }
        } else {
            flush_cache_run();
        }
    }
    release_pending();
    if (delta > 0) {
        // Before the caller reads the pages' guest memory: a CPU write whose check after its
        // store (NoteStoredWrite) misses this move stored its bytes before that read.
        g_pages_tracked_generation.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::MarkCaching(Memory::Memory* memory_inter, u64 vaddr, u64 bytes,
                                              bool cached) {
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::Count(vaddr, bytes, cached, [&] {
            DeviceMethods::MarkRegionCaching(memory_inter, vaddr, bytes, cached);
        });
        return;
    }
    DeviceMethods::MarkRegionCaching(memory_inter, vaddr, bytes, cached);
}

template <typename Traits>
bool DeviceMemoryManager<Traits>::NoteLingering(size_t page, Memory::Memory* memory_inter) {
    // Without a process interface the page was never protected; without a sweep (no Vulkan
    // TickFrame) nothing would ever uncache it.
    const u32 frame = linger_frame.load(std::memory_order_relaxed);
    if (memory_inter == nullptr || frame == 0) {
        return false;
    }
    {
        std::scoped_lock lock{linger_mutex};
        const auto [it, inserted] = lingering.try_emplace(page, frame);
        if (inserted) {
            lingering_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            it->second = frame;
        }
    }
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::AddLinger(&Core::Nx2ProtectStats::State::linger_noted, 1);
    }
    return true;
}

template <typename Traits>
bool DeviceMemoryManager<Traits>::TakeLingering(size_t page) {
    // A page's note is made and dropped under counter_guard for that page, which the caller
    // holds: a relaxed 0 here means this page is not noted.
    if (lingering_count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    {
        std::scoped_lock lock{linger_mutex};
        if (lingering.erase(page) == 0) {
            return false;
        }
        lingering_count.fetch_sub(1, std::memory_order_relaxed);
    }
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::AddLinger(&Core::Nx2ProtectStats::State::linger_cancelled, 1);
    }
    return true;
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UncacheDevicePages(std::span<const u64> pages) {
    Memory::Memory* run_inter = nullptr;
    u64 run_begin = 0;
    u64 run_pages = 0;
    const auto release_run = [&] {
        if (run_pages > 0 && run_inter != nullptr) {
            MarkCaching(run_inter, run_begin << Memory::YUZU_PAGEBITS,
                        run_pages << Memory::YUZU_PAGEBITS, false);
        }
        run_pages = 0;
    };
    for (const u64 page : pages) {
        const auto [asid, vaddr] = ExtractCPUBacking(page);
        const u64 vpage = vaddr >> Memory::YUZU_PAGEBITS;
        if (vpage == 0) {
            release_run();
            continue;
        }
        Memory::Memory* const inter =
            asid.id < registered_processes.size() ? registered_processes[asid.id] : nullptr;
        if (inter != run_inter || vpage != run_begin + run_pages) {
            release_run();
            run_inter = inter;
            run_begin = vpage;
        }
        ++run_pages;
    }
    release_run();
}

template <typename Traits>
size_t DeviceMemoryManager<Traits>::ReleaseLingeringPages(
    std::span<const std::pair<u64, u32>> pages, bool match_frame) {
    if (pages.empty()) {
        return 0;
    }
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::BeginBatch();
    }
    size_t released = 0;
    std::vector<u64> uncache;
    size_t begin = 0;
    while (begin < pages.size()) {
        size_t end = begin + 1;
        while (end < pages.size() && pages[end].first == pages[end - 1].first + 1) {
            ++end;
        }
        const u64 first_page = pages[begin].first;
        const u64 run_pages = pages[end - 1].first - first_page + 1;
        // counter_guard first, then linger_mutex; counter_guard is held through the uncache so
        // no 0 -> 1 of these pages can protect in between and be undone by it.
        Common::ScopedRangeLock lk(counter_guard, first_page << Memory::YUZU_PAGEBITS,
                                   run_pages << Memory::YUZU_PAGEBITS);
        uncache.clear();
        {
            std::scoped_lock lock{linger_mutex};
            for (size_t i = begin; i < end; ++i) {
                const u64 page = pages[i].first;
                const auto it = lingering.find(page);
                if (it == lingering.end() || (match_frame && it->second != pages[i].second)) {
                    continue; // taken (or taken and noted again) since it was collected
                }
                lingering.erase(it);
                lingering_count.fetch_sub(1, std::memory_order_relaxed);
                if (cached_pages->at(page >> subentries_shift)
                        .Count(page)
                        .load(std::memory_order_relaxed) == 0) {
                    uncache.push_back(page);
                }
            }
        }
        UncacheDevicePages(uncache);
        released += uncache.size();
        begin = end;
    }
    return released;
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReleaseLingering() {
    const u32 now = linger_frame.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool stats = Core::Nx2ProtectStats::Enabled();
    if (lingering_count.load(std::memory_order_relaxed) == 0) {
        if (stats) [[unlikely]] {
            Core::Nx2ProtectStats::Get().linger_held.store(0, std::memory_order_relaxed);
        }
        return;
    }
    // EXP89 = 0 (switched off live): every noted page is released now.
    const u32 frames = static_cast<u32>(std::max(VideoCore::Nx2Debug::Experiment(89), 0));
    std::vector<std::pair<u64, u32>> due;
    {
        std::scoped_lock lock{linger_mutex};
        for (const auto& [page, frame] : lingering) {
            if (now - frame > frames) {
                due.emplace_back(page, frame);
            }
        }
    }
    std::ranges::sort(due);
    const size_t released = ReleaseLingeringPages(due, true);
    if (stats) [[unlikely]] {
        Core::Nx2ProtectStats::AddLinger(&Core::Nx2ProtectStats::State::linger_swept, released);
        Core::Nx2ProtectStats::Get().linger_held.store(
            lingering_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReleaseLingeringRange(DAddr address, size_t size) {
    const u64 first_page = address >> Memory::YUZU_PAGEBITS;
    const u64 end_page = Common::DivCeil(address + size, Memory::YUZU_PAGESIZE);
    std::vector<std::pair<u64, u32>> noted;
    {
        std::scoped_lock lock{linger_mutex};
        if (lingering.empty()) {
            return;
        }
        if (lingering.size() < end_page - first_page) {
            for (const auto& [page, frame] : lingering) {
                if (page >= first_page && page < end_page) {
                    noted.emplace_back(page, frame);
                }
            }
        } else {
            for (u64 page = first_page; page < end_page; ++page) {
                if (const auto it = lingering.find(page); it != lingering.end()) {
                    noted.emplace_back(page, it->second);
                }
            }
        }
    }
    std::ranges::sort(noted);
    const size_t released = ReleaseLingeringPages(noted, false);
    if (released != 0 && Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::AddLinger(&Core::Nx2ProtectStats::State::linger_mapping, released);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ReleaseAllLingering() {
    std::vector<std::pair<u64, u32>> noted;
    {
        std::scoped_lock lock{linger_mutex};
        noted.assign(lingering.begin(), lingering.end());
    }
    std::ranges::sort(noted);
    const size_t released = ReleaseLingeringPages(noted, false);
    if (released != 0 && Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::AddLinger(&Core::Nx2ProtectStats::State::linger_mapping, released);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta) {
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::BeginBatch();
    }
    Common::ScopedRangeLock lk(counter_guard, addr, size);
    UpdatePagesCachedCountNoLock(addr, size, delta);
}

template <typename Traits>
void DeviceMemoryManager<Traits>::NoteHostStore(DAddr address, size_t size) {
    Common::UniformKeepPages::Note(address, size);
    Common::WriteRing::Note(address, size, "device span store");
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UnwatchUniformKeepPages(DAddr address, size_t size) {
    static_assert(Common::UniformKeepPages::PageBits == Memory::YUZU_PAGEBITS);
    Common::UniformKeepPages::Unwatch(address, size, [this](DAddr page_address, s32 delta) {
        UpdatePagesCachedCount(page_address, Memory::YUZU_PAGESIZE, delta);
    });
}

template <typename Traits>
void DeviceMemoryManager<Traits>::UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta) {
    if (ranges.empty()) {
        return;
    }
    if (Core::Nx2ProtectStats::Enabled()) [[unlikely]] {
        Core::Nx2ProtectStats::BeginBatch();
    }
    // Runs already sorted with gaps between them (the buffer tracker's) are what the sort and
    // merge below would produce: use them as they are instead of two heap copies.
    const bool separated =
        VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::SortedPageBatches) &&
        std::ranges::adjacent_find(ranges, [](const auto& a, const auto& b) {
            return a.first > b.first || a.first + a.second >= b.first;
        }) == ranges.end();
    if (separated && !VideoCommon::Nx2Perf::Verify()) {
        const DAddr lock_begin = ranges.front().first;
        const DAddr lock_end = ranges.back().first + ranges.back().second;
        Common::ScopedRangeLock lk(counter_guard, lock_begin,
                                   static_cast<size_t>(lock_end - lock_begin));
        for (const auto& [addr, size] : ranges) {
            UpdatePagesCachedCountNoLock(addr, size, delta);
        }
        return;
    }
    // Make a local copy and sort by address
    std::vector<std::pair<DAddr, size_t>> tmp(ranges.begin(), ranges.end());
    std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Coalesce adjacent/overlapping ranges
    std::vector<std::pair<DAddr, size_t>> coalesced;
    DAddr cur_addr = tmp[0].first;
    size_t cur_size = tmp[0].second;
    for (size_t i = 1; i < tmp.size(); ++i) {
        DAddr next_addr = tmp[i].first;
        size_t next_size = tmp[i].second;
        if (cur_addr + cur_size >= next_addr) {
            // overlapping or contiguous
            const DAddr end = std::max(cur_addr + cur_size, next_addr + next_size);
            cur_size = end - cur_addr;
        } else {
            coalesced.emplace_back(cur_addr, cur_size);
            cur_addr = next_addr;
            cur_size = next_size;
        }
    }
    coalesced.emplace_back(cur_addr, cur_size);
    if (separated) {
        static std::atomic<u32> logged{};
        if (!std::ranges::equal(ranges, coalesced) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: cached page runs from {:#x} merged differently",
                      coalesced.front().first);
        }
    }

    const DAddr lock_begin = coalesced.front().first;
    const DAddr lock_end = coalesced.back().first + coalesced.back().second;
    Common::ScopedRangeLock lk(counter_guard, lock_begin, static_cast<size_t>(lock_end - lock_begin));

    for (const auto& [addr, size] : coalesced) {
        UpdatePagesCachedCountNoLock(addr, size, delta);
    }
}

template <typename Traits>
void DeviceMemoryManager<Traits>::ArmReadTraps(DAddr addr, size_t size) {
    const auto MarkRegionReadTrap = &DeviceMemoryManager<Traits>::DeviceMethods::MarkRegionReadTrap;
    Memory::Memory* run_inter = nullptr;
    u64 run_begin = 0;
    u64 run_pages = 0;
    const auto release_run = [&] {
        if (run_pages > 0 && run_inter != nullptr) {
            MarkRegionReadTrap(run_inter, run_begin << Memory::YUZU_PAGEBITS,
                               run_pages << Memory::YUZU_PAGEBITS);
        }
        run_pages = 0;
    };
    const size_t page_end = Common::DivCeil(addr + size, Memory::YUZU_PAGESIZE);
    for (size_t page = addr >> Memory::YUZU_PAGEBITS; page != page_end; ++page) {
        auto [asid, vaddr] = ExtractCPUBacking(page);
        const u64 vpage = vaddr >> Memory::YUZU_PAGEBITS;
        if (vpage == 0) {
            release_run();
            continue;
        }
        auto* const inter = registered_processes[asid.id];
        if (inter != run_inter || vpage != run_begin + run_pages) {
            release_run();
            run_inter = inter;
            run_begin = vpage;
        }
        ++run_pages;
    }
    release_run();
}

} // namespace Core
