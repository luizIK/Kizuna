// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/multi_level_page_table.h"
#include "common/range_map.h"
#include "common/scratch_buffer.h"
#include "common/sparse_large_vector.h"
#include "video_core/invalidation_accumulator.h"
#include "video_core/cache_types.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/pte_kind.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Core {
class System;
} // namespace Core

namespace Tegra {

/// The small-page entry words (2 bits per 4 KiB page: 64 MiB for a 40-bit address space).
/// FRITCHII_RAM_GPU_ENTRIES (boot, default 0; docs/fritchii/switches.md): 1 = from zero pages
/// the host fills in on first touch (VirtualAlloc / anonymous mmap) instead of a zero-filled
/// std::vector, which made every page of all 64 MiB resident per address space.
class GpuEntryWords {
public:
    GpuEntryWords() = default;
    ~GpuEntryWords();
    GpuEntryWords(const GpuEntryWords&) = delete;
    GpuEntryWords& operator=(const GpuEntryWords&) = delete;

    void Allocate(size_t count);

    u64& operator[](size_t index) noexcept {
        return words[index];
    }
    const u64& operator[](size_t index) const noexcept {
        return words[index];
    }
    [[nodiscard]] const void* Data() const noexcept {
        return words;
    }
    [[nodiscard]] size_t Bytes() const noexcept {
        return count * sizeof(u64);
    }

private:
    void Release();

    std::vector<u64> vector;
    u64* words{};
    size_t count{};
    bool zero_pages{};
};

class MemoryManager final {
public:
    explicit MemoryManager(Core::System& system_, u64 address_space_bits_ = 40,
                           GPUVAddr split_address = 1ULL << 34, u64 big_page_bits_ = 16,
                           u64 page_bits_ = 12);
    explicit MemoryManager(Core::System& system_, MaxwellDeviceMemoryManager& memory_,
                           u64 address_space_bits_ = 40, GPUVAddr split_address = 1ULL << 34,
                           u64 big_page_bits_ = 16, u64 page_bits_ = 12);
    ~MemoryManager();

    // Switch 2: GPU address of the read-only syncpoint shim (u32 per syncpoint); 0 = none.
    void SetNativeSyncpointShim(GPUVAddr address);
    static constexpr u32 NumShimSyncpoints = 192; // Tegra::Host1x::SyncpointManager
    /// Whether [address, address + size) reads syncpoint values through the shim (they change
    /// when the GPU thread signals a syncpoint, not through guest memory).
    [[nodiscard]] bool OverlapsNativeSyncpointShim(GPUVAddr address, u64 size) const {
        const GPUVAddr shim = native_syncpoint_shim.load(std::memory_order_acquire);
        return shim != 0 && address < shim + NumShimSyncpoints * sizeof(u32) &&
               address + size > shim;
    }

    /// Switch 2 lays out block-linear surfaces with the console's GOB swizzle.
    void EnableNativeTiling() {
        native_tiling = true;
    }
    bool UsesNativeTiling() const {
        return native_tiling;
    }
    bool IsShuttingDown() const;

    static constexpr bool HAS_FLUSH_INVALIDATION = true;

    inline size_t GetID() const noexcept {
        return unique_identifier;
    }

    /// Binds a renderer to the memory manager.
    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    [[nodiscard]] std::optional<DAddr> GpuToCpuAddress(GPUVAddr addr) const;

    [[nodiscard]] std::optional<DAddr> GpuToCpuAddress(GPUVAddr addr, std::size_t size) const;

    template <typename T>
    [[nodiscard]] T Read(GPUVAddr addr) const;

    template <typename T>
    void Write(GPUVAddr addr, T data);

    [[nodiscard]] u8* GetPointer(GPUVAddr addr);
    [[nodiscard]] const u8* GetPointer(GPUVAddr addr) const;

    /// EDEN_NX2_HANDLE_PTR_CACHE (default on, 0 = off): the texture handle words and the texture and
    /// sampler descriptor entries every draw reads go through this thread's host pointers of the
    /// 16 device pages (4 KiB) it read last, instead of a page table walk each. They are dropped
    /// when this manager maps, remaps or unmaps anything, or the device memory does. The value
    /// is still read from guest memory every time.
    [[nodiscard]] static bool HandlePointerCacheEnabled();
    /// Read<u32> through the cached page pointers when HandlePointerCacheEnabled.
    [[nodiscard]] u32 ReadHandleWord(GPUVAddr addr) const;
    /// GetPointer through the cached page pointers when HandlePointerCacheEnabled (callers read
    /// within the page of `addr` only).
    [[nodiscard]] const u8* GetPointerForDescriptor(GPUVAddr addr) const;

    template <typename T>
    [[nodiscard]] inline T* GetPointer(GPUVAddr addr) noexcept {
        const auto address = GpuToCpuAddress(addr);
        if (!address)
            return {};
        return memory.GetPointer<T>(*address);
    }

    template <typename T>
    [[nodiscard]] inline const T* GetPointer(GPUVAddr addr) const noexcept {
        return GetPointer<T*>(addr);
    }

    /**
     * ReadBlock and WriteBlock are full read and write operations over virtual
     * GPU Memory. It's important to use these when GPU memory may not be continuous
     * in the Host Memory counterpart. Note: This functions cause Host GPU Memory
     * Flushes and Invalidations, respectively to each operation.
     */
    void ReadBlock(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size, VideoCommon::CacheType which = VideoCommon::CacheType::All) const;
    void WriteBlock(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, VideoCommon::CacheType which = VideoCommon::CacheType::All);
    void CopyBlock(GPUVAddr gpu_dest_addr, GPUVAddr gpu_src_addr, std::size_t size, VideoCommon::CacheType which = VideoCommon::CacheType::All);

    /**
     * ReadBlockUnsafe and WriteBlockUnsafe are special versions of ReadBlock and
     * WriteBlock respectively. In this versions, no flushing or invalidation is actually
     * done and their performance is similar to a memcpy. This functions can be used
     * on either of this 2 scenarios instead of their safe counterpart:
     * - Memory which is sure to never be represented in the Host GPU.
     * - Memory Managed by a Cache Manager. Example: Texture Flushing should use
     * WriteBlockUnsafe instead of WriteBlock since it shouldn't invalidate the texture
     * being flushed.
     */
    void ReadBlockUnsafe(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size) const;
    void WriteBlockUnsafe(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size);

    /// EXP92 (Common::UniformKeepPages): bytes were stored in place at [gpu_addr, gpu_addr +
    /// size) through a span (Core::Memory::GuestMemoryScoped): notes their device pages for the
    /// uniform keeps. One load until a keep was ever made.
    void NoteHostStore(GPUVAddr gpu_addr, std::size_t size);

    /// ReadBlockUnsafe for a thread that is not the GPU thread (EDEN_NX2_ASYNC_TRANSLATE: the
    /// translation workers' texture descriptor reads). Holds the page table lock shared, so Map,
    /// MapSparse and Unmap can't change the tables under the copy. Gives up (false, nothing read)
    /// when a writer keeps the lock for 2 ms: a writer can wait on the GPU thread (the caches'
    /// unmap callbacks), so a worker never waits on one without a limit.
    [[nodiscard]] bool ReadBlockGuarded(GPUVAddr gpu_src_addr, void* dest_buffer,
                                        std::size_t size) const;
    /// IsFullyMappedRange under the same lock, with the same limit (nullopt when it gave up).
    [[nodiscard]] std::optional<bool> IsFullyMappedRangeGuarded(GPUVAddr gpu_addr,
                                                                std::size_t size) const;

    /// Reads `count` consecutive words as Read<u32> would (Switch 2 syncpoint shim included),
    /// without a translation per word.
    void ReadWordsUnsafe(GPUVAddr gpu_src_addr, u32* dest, std::size_t count) const;
    void WriteBlockCached(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size);
    /// WriteBlockCached's bytes without its range (the memory stage collects its own).
    void WriteBlockStage(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size);
    /// Where WriteBlockStage's page walk writes a range within one 4 KiB GPU page.
    enum class StageWrite : u8 {
        None,          ///< Nowhere: unmapped, or a small page under a reserved big page
        BigContinuous, ///< A big page backed by one host range
        BigSplit,      ///< A big page whose host memory is split (the device write)
        Small,         ///< A small page (nothing written where it has no host pointer)
    };
    /// The memory stage's constant buffer words (FRITCHII_MEMSTAGE_THREAD=2): one page lookup
    /// for a range within one 4 KiB GPU page, giving both what GpuToCpuAddress(gpu_addr) returns
    /// and where WriteBlockStage writes, which would each look the page up again.
    struct StageTarget {
        std::optional<DAddr> device; ///< GpuToCpuAddress(gpu_addr)
        DAddr write_device{};        ///< The device address WriteBlockStage writes at
        StageWrite write{StageWrite::None};
    };
    [[nodiscard]] StageTarget ResolveStage(GPUVAddr gpu_addr) const;
    /// WriteBlockStage(gpu_dest_addr, src_buffer, size) through `target`, which is
    /// ResolveStage(gpu_dest_addr); the range lies within one 4 KiB GPU page.
    void WriteBlockStageAt(const StageTarget& target, GPUVAddr gpu_dest_addr,
                           const void* src_buffer, std::size_t size);
    /// WriteBlockCached of `count` consecutive words within one device page, in a single write.
    void WriteWordsCached(GPUVAddr gpu_dest_addr, const u32* words, std::size_t count);
    /// EXP88 & 2: constant buffer data written at
    /// `device_addr`, which the caller translated from `gpu_addr` (one mapped device range, not
    /// Switch 2 sparse memory): what WriteBlockCached does, without its page walk, and with the
    /// device range logged for FlushCaching instead of the GPU range accumulated (32-byte
    /// rounded as the accumulator rounds, merged with the last range when they touch).
    void WriteConstantBufferData(DAddr device_addr, GPUVAddr gpu_addr, const void* data,
                                 std::size_t size, bool verify);

    /**
     * Checks if a gpu region can be simply read with a pointer.
     */
    [[nodiscard]] bool IsGranularRange(GPUVAddr gpu_addr, std::size_t size) const;

    /**
     * Checks if a gpu region is mapped by a single range of device addresses.
     */
    [[nodiscard]] bool IsContinuousRange(GPUVAddr gpu_addr, std::size_t size) const;

    /**
     * Checks if a gpu region is mapped entirely.
     */
    [[nodiscard]] bool IsFullyMappedRange(GPUVAddr gpu_addr, std::size_t size) const;

    /**
     * Returns a vector with all the subranges of device addresses mapped beneath.
     * if the region is continuous, a single pair will be returned. If it's unmapped, an empty
     * vector will be returned;
     */
    boost::container::small_vector<std::pair<GPUVAddr, std::size_t>, 32> GetSubmappedRange(
        GPUVAddr gpu_addr, std::size_t size) const;

    /// The device ranges [gpu_addr, gpu_addr + size) maps to (contiguous pages merged), up to the
    /// end of the address space.
    boost::container::small_vector<std::pair<DAddr, std::size_t>, 32> GetSubmappedDeviceRange(
        GPUVAddr gpu_addr, std::size_t size) const;

    GPUVAddr Map(GPUVAddr gpu_addr, DAddr dev_addr, std::size_t size,
                 PTEKind kind = PTEKind::INVALID, bool is_big_pages = true);
    GPUVAddr MapSparse(GPUVAddr gpu_addr, std::size_t size, bool is_big_pages = true);
    void Unmap(GPUVAddr gpu_addr, std::size_t size);

    void FlushRegion(GPUVAddr gpu_addr, size_t size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) const;

    void InvalidateRegion(GPUVAddr gpu_addr, size_t size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) const;

    bool IsMemoryDirty(GPUVAddr gpu_addr, size_t size,
                       VideoCommon::CacheType which = VideoCommon::CacheType::All) const;

    size_t MaxContinuousRange(GPUVAddr gpu_addr, size_t size) const;

    /// Bumped at the start and at the end of each page table change: lookups cached under one
    /// value stay valid until then.
    [[nodiscard]] u64 MapSerial() const {
        return map_serial.load(std::memory_order_acquire);
    }

    /// A lookup that will be cached under the map counter (a sequence lock's read side): the
    /// counter and whether a page table change is running, loaded before the lookup.
    struct MapRead {
        u64 serial;
        bool busy;
    };
    [[nodiscard]] MapRead BeginMapRead() const {
        const u64 serial = map_serial.load(std::memory_order_acquire);
        return {serial, map_changes_running.load(std::memory_order_acquire) != 0};
    }
    /// After the lookup: whether it saw whole tables and may be cached under `read.serial`. No
    /// change was running when it started, and none started since (that bumps the counter).
    [[nodiscard]] bool MapReadStable(const MapRead& read) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return !read.busy && map_serial.load(std::memory_order_relaxed) == read.serial;
    }

    /// FRITCHII_BIND_ADDR_MEMO: one remembered translation of one small GPU page (page_bits),
    /// kept by its user (a vertex stream, the index buffer, a uniform slot).
    struct TranslationMemo {
        size_t as_id{~size_t{0}}; ///< unique_identifier of the manager it was looked up in
        u64 serial{};             ///< map_serial of a stable lookup (MapReadStable)
        u64 page{~u64{0}};        ///< gpu_addr >> page_bits
        DAddr device_page{};      ///< What GpuToCpuAddress gives for the page's first byte
    };
    /// Counts for the "Bind address memo" line (the caller's thread only).
    struct TranslationMemoStats {
        u64 hits{};
        u64 lookups{};  ///< Misses: GpuToCpuAddress ran
        u64 unstable{}; ///< Misses not kept: a page table change overlapped the lookup
    };
    /// GpuToCpuAddress(gpu_addr) through `memo`. A hit needs the same manager, the same small
    /// page and an unmoved map_serial: every page table change (PageTableOp, BigPageTableOp:
    /// Map, MapSparse, Unmap, remaps) bumps it before its first table write and after its last,
    /// so a hit returns what the walk returns now. Within one small page the translation is
    /// linear (the page and big page tables map whole small pages), so the page's base plus the
    /// offset is the walk's result. Unmapped results are not kept. A lookup is kept only when no
    /// change overlapped it (BeginMapRead, MapReadStable), as the EXP63 page cache does.
    [[nodiscard]] std::optional<DAddr> GpuToCpuAddressMemo(GPUVAddr gpu_addr, TranslationMemo& memo,
                                                          TranslationMemoStats& stats) const {
        const u64 page = gpu_addr >> page_bits;
        if (memo.page == page && memo.as_id == unique_identifier &&
            memo.serial == map_serial.load(std::memory_order_acquire)) {
            ++stats.hits;
            return memo.device_page + (gpu_addr & page_mask);
        }
        ++stats.lookups;
        const MapRead read = BeginMapRead();
        const std::optional<DAddr> device_addr = GpuToCpuAddress(gpu_addr);
        if (device_addr) {
            if (MapReadStable(read)) {
                memo = TranslationMemo{
                    .as_id = unique_identifier,
                    .serial = read.serial,
                    .page = page,
                    .device_page = *device_addr - (gpu_addr & page_mask),
                };
            } else {
                ++stats.unstable;
                memo.page = ~u64{0};
            }
        }
        return device_addr;
    }

    /// Changes whenever this address space or the device memory behind it is mapped or unmapped:
    /// a host pointer GetPointer returned is only good while it stays the same.
    [[nodiscard]] u64 HostPointerSerial() const {
        return MapSerial() + memory.TranslationGeneration();
    }
    /// MaxContinuousRange by walking the page tables (the answer with Nx2Perf off).
    size_t MaxContinuousRangeUncached(GPUVAddr gpu_addr, size_t size) const;

    bool IsWithinGPUAddressRange(GPUVAddr gpu_addr) const {
        return gpu_addr < address_space_size;
    }

    PTEKind GetPageKind(GPUVAddr gpu_addr) const;

    size_t GetMemoryLayoutSize(GPUVAddr gpu_addr,
                               size_t max_size = (std::numeric_limits<size_t>::max)()) const;

    /// GetMemoryLayoutSize from the kind map under its lock (the answer with Nx2Perf off).
    size_t GetMemoryLayoutSizeUncached(GPUVAddr gpu_addr) const;

    void FlushCaching();

    /// Leaves the caches' invalidation of a range already written for the next FlushCaching
    /// (before every draw, dispatch, clear and copy), merged with neighbouring ranges.
    void AccumulateInvalidation(GPUVAddr gpu_addr, std::size_t size) {
        accumulator.Add(gpu_addr, size);
        NoteCachedWrite();
    }

    /// FRITCHII_ESPLIT (fritchii_e_split_thread.h, FlushPlace): cached writes added since the
    /// start, so E1 can tell whether a piece of work left any for the next FlushCaching. One
    /// writer at a time (the thread that writes game memory for E); read by E1.
    [[nodiscard]] u64 CachedWriteCount() const noexcept {
        return cached_writes.load(std::memory_order_relaxed);
    }

    /// FRITCHII_ESPLIT (ESplitLateShader): whether a range the next FlushCaching invalidates lands
    /// where `hit(device address, size)` says (each accumulated range by its first page's
    /// translation, each constant buffer range as logged). The drawing thread's, read only.
    template <typename Hit>
    [[nodiscard]] bool PendingWritesHit(Hit&& hit) const {
        bool found = false;
        accumulator.ForEachRange([&](GPUVAddr address, std::size_t size) {
            if (!found) {
                if (const std::optional<DAddr> device = GpuToCpuAddress(address)) {
                    found = hit(*device, static_cast<u64>(size));
                }
            }
        });
        for (const auto& [device, size] : constant_buffer_writes) {
            if (found) {
                break;
            }
            found = hit(device, static_cast<u64>(size));
        }
        return found;
    }

    /// EXP91: whether cached writes wait for the next FlushCaching (accumulated GPU ranges or
    /// EXP88's constant buffer device ranges). Drawing thread.
    [[nodiscard]] bool HasPendingCachedWrites() const noexcept {
        return !accumulator.Empty() || !constant_buffer_writes.empty();
    }

    const u8* GetSpan(const GPUVAddr src_addr, const std::size_t size) const;
    u8* GetSpan(const GPUVAddr src_addr, const std::size_t size);

private:
    template <typename FuncMapped, typename FuncReserved, typename FuncUnmapped>
    inline void MemoryOperation(GPUVAddr gpu_src_addr, std::size_t size, bool is_big_page, FuncMapped&& func_mapped, FuncReserved&& func_reserved, FuncUnmapped&& func_unmapped) const;

    /// Immutable copy of the device-contiguous runs and the kind map that readers query without
    /// a lock; each thread swaps its copy when map_serial changes.
    struct MappingSnapshot {
        struct Run {
            GPUVAddr begin;
            GPUVAddr end;
        };
        std::vector<Run> runs;
        std::vector<std::pair<s64, PTEKind>> kinds;
    };

    /// Rebuilds the runs through the big pages a page table operation touched.
    void UpdateContinuousRuns(GPUVAddr gpu_addr, size_t size);
    const MappingSnapshot& CurrentSnapshot() const;

    void ReadBlockImpl(GPUVAddr gpu_src_addr, void* dest_buffer, std::size_t size, VideoCommon::CacheType which, bool unsafe) const;
    /// `note_keeps` false: a cached write (WriteBlockCached, WriteWordsCached), whose range the
    /// uniform keeps get exactly at FlushCaching (EXP92 tier 1); every other write notes its
    /// device pages after the copy (tier 2).
    void WriteBlockImpl(GPUVAddr gpu_dest_addr, const void* src_buffer, std::size_t size, VideoCommon::CacheType which, bool unsafe, bool note_keeps = true);

    [[nodiscard]] std::size_t PageEntryIndex(GPUVAddr gpu_addr, bool is_big_page) const {
        if (is_big_page) {
            return (gpu_addr >> big_page_bits) & big_page_table_mask;
        } else {
            return (gpu_addr >> page_bits) & page_table_mask;
        }
    }

    inline bool IsBigPageContinuous(size_t big_page_index) const;
    inline void SetBigPageContinuous(size_t big_page_index, bool value);

    template <bool is_gpu_address>
    void GetSubmappedRangeImpl(
        GPUVAddr gpu_addr, std::size_t size,
        boost::container::small_vector<std::pair<std::conditional_t<is_gpu_address, GPUVAddr, DAddr>, std::size_t>, 32>& result) const;

    /// Host pointer of `addr` through this thread's cached page pointers (HandlePointerCacheEnabled).
    const u8* CachedPagePointer(GPUVAddr addr) const;

    std::atomic<GPUVAddr> native_syncpoint_shim{};
    Core::System& system;
    MaxwellDeviceMemoryManager& memory;

    const u64 address_space_bits;
    GPUVAddr split_address;
    const u64 page_bits;
    u64 address_space_size;
    u64 page_size;
    u64 page_mask;
    u64 page_table_mask;
    static constexpr u64 cpu_page_bits{12};

    const u64 big_page_bits;
    u64 big_page_size;
    u64 big_page_mask;
    u64 big_page_table_mask;

    VideoCore::RasterizerInterface* rasterizer = nullptr;

    enum class EntryType : u64 {
        Free = 0,
        Reserved = 1,
        Mapped = 2,
    };

    GpuEntryWords entries;
    std::vector<u64> big_entries;

    GPUVAddr PageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, EntryType entry_type);
    GPUVAddr BigPageTableOp(GPUVAddr gpu_addr, [[maybe_unused]] DAddr dev_addr, size_t size, PTEKind kind, EntryType entry_type);

    inline EntryType GetEntry(size_t position, bool is_big_page) const;
    inline void SetEntry(size_t position, EntryType entry, bool is_big_page);

    Common::MultiLevelPageTable<u32> page_table;
    Common::RangeMap<GPUVAddr, PTEKind> kind_map;
    Common::SparseLargeVector<u32> big_page_table_dev;

    std::vector<u64> big_page_continuous;
    boost::container::small_vector<std::pair<DAddr, std::size_t>, 32> page_stash2{};

    bool native_tiling{};
    mutable std::mutex guard;
    /// Taken exclusively by Map, MapSparse and Unmap around their page table updates while
    /// translation workers read guest memory (VideoCommon::AsyncTranslateWorkers), shared by the
    /// guarded reads. Untouched otherwise.
    mutable std::shared_mutex page_table_lock;
    /// Shared lock of page_table_lock, waiting 2 ms at most.
    [[nodiscard]] bool TryLockPageTableShared() const;
    /// Bumped at the start and at the end of each page table change (MapChange); readers keep
    /// their MappingSnapshot until then.
    std::atomic<u64> map_serial{};
    /// Page table changes running now (MapChange). A count, not the counter's parity: Switch 2
    /// maps run on the GPU thread (MapInOrder, RunInOrder) while other mappings run on the
    /// guest's thread, so two changes can overlap.
    std::atomic<u32> map_changes_running{};
    /// Around PageTableOp and BigPageTableOp: the write side of BeginMapRead/MapReadStable.
    class MapChange {
    public:
        explicit MapChange(MemoryManager& manager_) : manager{manager_} {
            manager.map_changes_running.fetch_add(1, std::memory_order_seq_cst);
            manager.map_serial.fetch_add(1, std::memory_order_seq_cst);
            // The table writes below stay after both.
            std::atomic_thread_fence(std::memory_order_release);
        }
        ~MapChange() {
            manager.map_serial.fetch_add(1, std::memory_order_release);
            manager.map_changes_running.fetch_sub(1, std::memory_order_release);
        }
        MapChange(const MapChange&) = delete;
        MapChange& operator=(const MapChange&) = delete;

    private:
        MemoryManager& manager;
    };

    struct ContinuousRun {
        GPUVAddr end;
        DAddr device_addr;
    };
    /// Maximal GPU ranges, keyed by their start, that are mapped to one device range: what
    /// MaxContinuousRangeUncached walks to. Kept only with Nx2Perf on.
    std::map<GPUVAddr, ContinuousRun> continuous_runs;
    mutable std::shared_ptr<const MappingSnapshot> snapshot;
    mutable u64 snapshot_serial{};
    mutable std::mutex runs_guard;

    static constexpr size_t continuous_bits = 64;

    const size_t unique_identifier;
    VideoCommon::InvalidationAccumulator accumulator;
    /// CachedWriteCount's counter (a plain load and store: one writer at a time).
    std::atomic<u64> cached_writes{};
    void NoteCachedWrite() noexcept {
        cached_writes.store(cached_writes.load(std::memory_order_relaxed) + 1,
                            std::memory_order_relaxed);
    }
    /// EXP88 & 2: device ranges WriteConstantBufferData wrote since the last FlushCaching.
    boost::container::small_vector<std::pair<DAddr, std::size_t>, 32> constant_buffer_writes;
    /// EXP88 & 4: those writes one by one, translated again at FlushCaching.
    struct ConstantBufferVerify {
        GPUVAddr gpu_addr;
        DAddr device_addr;
        std::size_t size;
        u64 map_serial;
    };
    std::vector<ConstantBufferVerify> constant_buffer_verify;

    static std::atomic<size_t> unique_identifier_generator;

    Common::ScratchBuffer<u8> tmp_buffer;
};

} // namespace Tegra
