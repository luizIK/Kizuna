// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <span>

#include "common/common_types.h"
#include "common/range_mutex.h"
#include "common/scratch_buffer.h"
#include "common/thread_worker.h"
#include "common/sparse_large_vector.h"

namespace Core {

constexpr size_t DEVICE_PAGEBITS = 12ULL;
constexpr size_t DEVICE_PAGESIZE = 1ULL << DEVICE_PAGEBITS;
constexpr size_t DEVICE_PAGEMASK = DEVICE_PAGESIZE - 1ULL;

class DeviceMemory;

namespace Memory {
class Memory;
}

/// Moved each time a GPU cache takes guest memory's bytes as clean again (pages tracked again by
/// UpdatePagesCachedCount with a positive delta, collected CPU writes taken by a dirty manager's
/// Gather). A CPU write marks its pages before it stores its bytes (Core::Memory's Write), so a
/// cache that takes the marks in between can upload the old bytes and treat them as current; a
/// write that sees this value move across its store marks its pages again (NoteStoredWrite in
/// core/memory.cpp, FRITCHII_WRITE_REMARK). Only equality is compared.
inline std::atomic<u64> g_pages_tracked_generation{};

template <typename DTraits>
struct DeviceMemoryManagerAllocator;

struct Asid {
    size_t id;
};

template <typename Traits>
class DeviceMemoryManager {
    using DeviceInterface = typename Traits::DeviceInterface;
    using DeviceMethods = typename Traits::DeviceMethods;

public:
    DeviceMemoryManager(const DeviceMemory& device_memory);
    ~DeviceMemoryManager();

    static constexpr bool HAS_FLUSH_INVALIDATION = true;
    static constexpr size_t AS_BITS = Traits::device_virtual_bits;

    void BindInterface(DeviceInterface* device_inter);

    /// Marks the memory behind [address, address + size) cold (Common::HostMemory::HintCold),
    /// from a low-priority thread: data the host GPU has its own copy of, e.g. uploaded textures.
    void HintCold(DAddr address, size_t size);

    DAddr Allocate(size_t size);
    void AllocateFixed(DAddr start, size_t size);
    void Free(DAddr start, size_t size);

    void Map(DAddr address, VAddr virtual_address, size_t size, Asid asid, bool track = false);

    void Unmap(DAddr address, size_t size);

    void TrackContinuityImpl(DAddr address, VAddr virtual_address, size_t size, Asid asid);
    void TrackContinuity(DAddr address, VAddr virtual_address, size_t size, Asid asid) {
        std::scoped_lock lk(mapping_guard);
        TrackContinuityImpl(address, virtual_address, size, asid);
    }

    // Write / Read
    template <typename T>
    T* GetPointer(DAddr address);

    template <typename T>
    const T* GetPointer(DAddr address) const;

    template <typename Func>
    void ApplyOpOnPAddr(PAddr address, Common::ScratchBuffer<u32>& buffer, Func&& operation) {
        DAddr subbits = DAddr(address & page_mask);
        const u32 base = compressed_device_addr[(address >> page_bits)];
        if ((base >> MULTI_FLAG_BITS) == 0) [[likely]] {
            const DAddr d_address = (DAddr(base) << page_bits) + subbits;
            operation(d_address);
            return;
        }
        InnerGatherDeviceAddresses(buffer, address);
        for (u32 value : buffer) {
            operation((DAddr(value) << page_bits) + subbits);
        }
    }

    /// Whether p points into the emulated physical memory, the only memory with device addresses.
    bool IsPhysicalPointer(const u8* p) const {
        return (GetRawPhysicalAddr(p) >> page_bits) < compressed_device_addr.size();
    }

    template <typename Func>
    void ApplyOpOnPointer(const u8* p, Common::ScratchBuffer<u32>& buffer, Func&& operation) {
        PAddr address = GetRawPhysicalAddr<u8>(p);
        ApplyOpOnPAddr(address, buffer, operation);
    }

    PAddr GetPhysicalRawAddressFromDAddr(DAddr address) const {
        PAddr subbits = PAddr(address & page_mask);
        auto paddr = tracked_entries[(address >> page_bits)].compressed_physical_ptr;
        if (paddr == 0) {
            return 0;
        }
        return (PAddr(paddr - 1) << page_bits) + subbits;
    }

    template <typename T>
    void Write(DAddr address, T value);

    template <typename T>
    T Read(DAddr address) const;

    u8* GetSpan(const DAddr src_addr, const std::size_t size);
    const u8* GetSpan(const DAddr src_addr, const std::size_t size) const;

    /// Changes whenever Map or Unmap changes which host memory a device page reads: host
    /// pointers kept from before are stale.
    [[nodiscard]] u64 TranslationGeneration() const {
        return translation_generation.load(std::memory_order_acquire);
    }

    /// The bytes from `address` (at most `size`) before the first device page with no host
    /// backing, which ReadBlock reads as zeros (FRITCHII_BARE_SSBO_CLAMP). One table read a page.
    [[nodiscard]] size_t BackedPrefix(DAddr address, size_t size) const {
        const DAddr end = address + size;
        for (DAddr page = address & ~DAddr{page_mask}; page < end; page += page_size) {
            if (tracked_entries[page >> page_bits].compressed_physical_ptr == 0) {
                return page <= address ? 0 : static_cast<size_t>(page - address);
            }
        }
        return size;
    }

    void ReadBlock(DAddr address, void* dest_pointer, size_t size);
    void ReadBlockUnsafe(DAddr address, void* dest_pointer, size_t size);
    void WriteBlock(DAddr address, const void* src_pointer, size_t size);
    void WriteBlockUnsafe(DAddr address, const void* src_pointer, size_t size);
    /// EXP92: bytes stored in place through a span (Core::Memory::GuestMemoryScoped), after
    /// they landed: noted for the uniform keeps (Common::UniformKeepPages).
    void NoteHostStore(DAddr address, size_t size);

    Asid RegisterProcess(Memory::Memory* memory);
    void UnregisterProcess(Asid id);

    void UpdatePagesCachedCount(DAddr addr, size_t size, s32 delta);

    // New batch API to update multiple ranges with a single lock acquisition.
    void UpdatePagesCachedBatch(std::span<const std::pair<DAddr, size_t>> ranges, s32 delta);

    /// Traps the next guest CPU read of these pages, to learn which GPU writes the CPU reads.
    void ArmReadTraps(DAddr addr, size_t size);

    /// EXP89 (lingering protection, fable52_answer_after88 1.3): once a frame, from the
    /// rasterizer's TickFrame. Raises the lingering frame and really uncaches the pages that
    /// lingered more than EXP89 frames (all of them with EXP89 = 0).
    void ReleaseLingering();

private:
    struct TranslationEntry {
        DAddr guest_page{};
        u8* host_ptr{};
    };

    /// A thread's last four page translations of one manager. Several threads read through
    /// them (the GPU thread, the fencing thread, service threads): one cache shared by all was
    /// cleared by Map on a service thread between the GPU thread's null check of an entry and
    /// its use, and the GPU thread copied from address 0 (title 1 crashed every few minutes).
    struct TranslationCache {
        const void* owner{};
        u64 generation{};
        std::array<TranslationEntry, 4> slots{};
        u32 cursor{};
    };
    static inline thread_local TranslationCache t_cache{};

    /// This thread's translations, emptied when Map or Unmap ran since they were made.
    TranslationCache& Translations() {
        const u64 current = translation_generation.load(std::memory_order_acquire);
        if (t_cache.owner != this || t_cache.generation != current) {
            t_cache = TranslationCache{.owner = this, .generation = current};
        }
        return t_cache;
    }

    // Internal helper that performs the update assuming the caller already holds the necessary lock.
    void UpdatePagesCachedCountNoLock(DAddr addr, size_t size, s32 delta);

    /// DeviceMethods::MarkRegionCaching, counted and timed under EDEN_NX2_PROTECT_STATS=1.
    void MarkCaching(Memory::Memory* memory_inter, u64 vaddr, u64 bytes, bool cached);

    // EXP89 lingering protection. The caller holds counter_guard for the page; linger_mutex is
    // only ever taken after counter_guard, never before.
    /// Count 1 -> 0: leaves the page protected and notes it with the lingering frame. False when
    /// it can't linger (no process interface, or no sweep ran yet: no TickFrame calls it).
    bool NoteLingering(size_t page, Memory::Memory* memory_inter);
    /// Count 0 -> 1: true when the page was lingering (still protected; the note is dropped).
    bool TakeLingering(size_t page);
    /// Uncaches the noted pages of `pages` (sorted device pages with the frame they were seen
    /// noted with) that are still noted (with that frame when `match_frame`), per run of
    /// consecutive device pages under counter_guard. Returns the pages uncached.
    size_t ReleaseLingeringPages(std::span<const std::pair<u64, u32>> pages, bool match_frame);
    /// Uncaches the noted pages of a device range, whatever their age (Map, Unmap).
    void ReleaseLingeringRange(DAddr address, size_t size);
    /// EXP92 (Common::UniformKeepPages): gives back the page watches the uniform keeps hold on a
    /// device range whose backing is about to change (Map, Unmap). One load while none is held.
    void UnwatchUniformKeepPages(DAddr address, size_t size);
    /// Uncaches every noted page (UnregisterProcess).
    void ReleaseAllLingering();
    /// MarkRegionCaching(false) for sorted device pages, per run of consecutive guest pages of
    /// one process. The caller holds counter_guard for them.
    void UncacheDevicePages(std::span<const u64> pages);

private:
    static constexpr size_t device_virtual_bits = Traits::device_virtual_bits;
    static constexpr size_t device_as_size = 1ULL << device_virtual_bits;
    static constexpr size_t physical_min_bits = 32;
    static constexpr size_t physical_max_bits = 35; // covers the 12 GiB layout plus nx2 graphics server memory
    static constexpr size_t page_bits = 12;
    static constexpr size_t page_size = 1ULL << page_bits;
    static constexpr size_t page_mask = page_size - 1ULL;
    static constexpr u32 physical_address_base = 1U << page_bits;
    static constexpr u32 MULTI_FLAG_BITS = 31;
    static constexpr u32 MULTI_FLAG = 1U << MULTI_FLAG_BITS;
    static constexpr u32 MULTI_MASK = ~MULTI_FLAG;

    template <typename T>
    T* GetPointerFromRaw(PAddr addr) {
        return reinterpret_cast<T*>(physical_base + addr);
    }

    template <typename T>
    const T* GetPointerFromRaw(PAddr addr) const {
        return reinterpret_cast<T*>(physical_base + addr);
    }

    template <typename T>
    PAddr GetRawPhysicalAddr(const T* ptr) const {
        return static_cast<PAddr>(reinterpret_cast<uintptr_t>(ptr) - physical_base);
    }

    void WalkBlock(const DAddr addr, const std::size_t size, auto on_unmapped, auto on_memory,
                   auto increment);

    void CutContinuityAt(size_t page);

    void InnerGatherDeviceAddresses(Common::ScratchBuffer<u32>& buffer, PAddr address);

    std::unique_ptr<DeviceMemoryManagerAllocator<Traits>> impl;

    const uintptr_t physical_base;
    const DeviceMemory* device_memory_ptr;
    DeviceInterface* device_inter;
    std::unique_ptr<Common::ThreadWorker> cold_hint_worker;

    struct TrackedEntry {
        VAddr cpu_backing_address;
        u32 continuity_tracker;
        u32 compressed_physical_ptr;
    };
    Common::SparseLargeVector<u32> compressed_device_addr;
    Common::SparseLargeVector<TrackedEntry> tracked_entries;

    // Process memory interfaces

    std::deque<size_t> id_pool;
    std::deque<Memory::Memory*> registered_processes;

    // Memory protection management

    static constexpr size_t guest_max_as_bits = 39;
    static constexpr size_t guest_as_size = 1ULL << guest_max_as_bits;
    static constexpr size_t guest_mask = guest_as_size - 1ULL;
    static constexpr size_t asid_start_bit = guest_max_as_bits;

    std::pair<Asid, VAddr> ExtractCPUBacking(size_t page_index) {
        auto content = tracked_entries[page_index].cpu_backing_address;
        const VAddr address = content & guest_mask;
        const Asid asid{static_cast<size_t>(content >> asid_start_bit)};
        return std::make_pair(asid, address);
    }

    constexpr void InsertCPUBacking(size_t page_index, VAddr address, Asid asid) {
        tracked_entries.GetUnchecked(page_index).cpu_backing_address = address | (asid.id << asid_start_bit);
    }

    /// Starts past every earlier manager's: one built where a destroyed one was must not match
    /// the translations a thread kept for that one (Translations compares owner and generation).
    static inline std::atomic<u64> next_generation_base{1};
    /// Bumped by Map and Unmap
    std::atomic<u64> translation_generation{next_generation_base.fetch_add(1ULL << 32)};
    using CounterType = u8;
    using CounterAtomicType = std::atomic_uint8_t;
    static constexpr size_t subentries = 8 / sizeof(CounterType);
    static constexpr size_t subentries_mask = subentries - 1;
    static constexpr size_t subentries_shift =
        std::countr_zero(sizeof(u64)) - std::countr_zero(sizeof(CounterType));
    class CounterEntry final {
    public:
        CounterEntry() = default;

        CounterAtomicType& Count(std::size_t page) {
            return values[page & subentries_mask];
        }

        const CounterAtomicType& Count(std::size_t page) const {
            return values[page & subentries_mask];
        }

    private:
        std::array<CounterAtomicType, subentries> values{};
    };
    static_assert(sizeof(CounterEntry) == subentries * sizeof(CounterType),
                  "CounterEntry should be 8 bytes!");

    static constexpr size_t num_counter_entries =
        (1ULL << (device_virtual_bits - page_bits)) / subentries;
    using CachedPages = std::array<CounterEntry, num_counter_entries>;
    std::unique_ptr<CachedPages> cached_pages;
    Common::RangeMutex counter_guard;
    std::mutex mapping_guard;

    // EXP89 lingering protection: device page number -> lingering frame it was noted in. Pages
    // here have a cached count of 0 but are still protected. Dropped without calls when the
    // manager is destroyed (the process interfaces may be gone by then).
    std::mutex linger_mutex;
    std::unordered_map<u64, u32> lingering;
    /// lingering.size(), changed under linger_mutex: TakeLingering's lock-free "nothing noted".
    std::atomic<size_t> lingering_count{0};
    /// Raised once a frame by ReleaseLingering; 0 until the first sweep.
    std::atomic<u32> linger_frame{0};


};

} // namespace Core
