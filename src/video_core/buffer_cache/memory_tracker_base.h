// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <deque>
#include <limits>
#include <type_traits>
#include "common/container/unordered_set.h"
#include <utility>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/fritchii_bind_skip.h"
#include "video_core/buffer_cache/word_manager.h"
#include "video_core/page_filter.h"

namespace VideoCommon {

template <typename DeviceTracker>
class MemoryTrackerBase {
    static constexpr size_t MAX_CPU_PAGE_BITS = 36; // device addresses (MaxwellDeviceTraits)
    static constexpr size_t HIGHER_PAGE_BITS = 22;
    static constexpr size_t HIGHER_PAGE_SIZE = 1ULL << HIGHER_PAGE_BITS;
    static constexpr size_t HIGHER_PAGE_MASK = HIGHER_PAGE_SIZE - 1ULL;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;
    static constexpr size_t WORDS_STACK_NEEDED = HIGHER_PAGE_SIZE / BYTES_PER_WORD;
    using Manager = WordManager<DeviceTracker, WORDS_STACK_NEEDED, HIGHER_PAGE_SIZE>;

public:
    MemoryTrackerBase(DeviceTracker& device_tracker_) : device_tracker{&device_tracker_} {}
    ~MemoryTrackerBase() = default;

    /// Returns the inclusive CPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedCpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<true>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::CPU, offset, size);
        });
    }

    /// Returns the inclusive GPU modified range in a begin end pair
    [[nodiscard]] std::pair<u64, u64> ModifiedGpuRegion(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePairs<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->ModifiedRegion(Type::GPU, offset, size);
        });
    }

    /// Returns true if a region has been modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::CPU, offset, size);
        });
    }

    /// The memory stage's P3 (fable_stage_p3 4.2): whether no page of the region is CPU-modified
    /// or untracked (a cached write's page), so an upload walk over it (SynchronizeBuffer's, any
    /// of its forms) finds and changes nothing: it uploads the first and re-tracks both. Missing
    /// managers are created, as the walk creates them (all dirty: not clean).
    [[nodiscard]] bool IsRegionSyncClean(VAddr query_cpu_addr, u64 query_size) noexcept {
        return !IteratePages<true>(query_cpu_addr, query_size,
                                   [](Manager* manager, u64 offset, size_t size) {
                                       return manager->IsRegionModified(Type::CPU, offset, size) ||
                                              manager->IsRegionModified(Type::Untracked, offset,
                                                                        size);
                                   });
    }

    /// Returns true if every page of a region is CPU-modified and none is GPU-modified or
    /// Preflushable. False where the tracker has no manager yet (nothing to prove it).
    [[nodiscard]] bool IsRegionWhollyCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        if (query_size == 0) {
            return false;
        }
        const size_t first = query_cpu_addr >> HIGHER_PAGE_BITS;
        const size_t last = (query_cpu_addr + query_size - 1) >> HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            if (top_tier[page_index] == nullptr) {
                return false;
            }
        }
        return !IteratePages<false>(query_cpu_addr, query_size,
                                    [](Manager* manager, u64 offset, size_t size) {
                                        return !manager->IsRegionWhollyCpuModified(offset, size);
                                    });
    }

    /// Returns true if a region has been modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::GPU, offset, size);
        });
    }

    /// Returns true if a region has been marked as Preflushable
    [[nodiscard]] bool IsRegionPreflushable(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            return manager->IsRegionModified(Type::Preflushable, offset, size);
        });
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::CPU, true, manager->cpu_addr + offset, size);
        });
    }

    /// EXP108: as MarkRegionAsCpuModified, for data the GPU's own engines
    /// wrote to guest memory (inline constant buffer data): the pages stay tracked, as no guest CPU
    /// write needs to be let through, and the next upload has nothing to track again.
    void MarkRegionAsWrittenTracked(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->MarkWrittenTracked(manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as CPU modified, notifying the device_tracker about this change
    void UnmarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        ever_synchronized_pages.Mark(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::CPU, false, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the host GPU
    void MarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        ever_synchronized_pages.Mark(dirty_cpu_addr, query_size);
        ever_gpu_modified_pages.Mark(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::GPU, true, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the host GPU
    void MarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        ever_synchronized_pages.Mark(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, true, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::GPU, false, manager->cpu_addr + offset, size);
        });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsPreflushable(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<true>(dirty_cpu_addr, query_size, [](Manager* manager, u64 offset, size_t size) {
            manager->ChangeRegionState(Type::Preflushable, false, manager->cpu_addr + offset, size);
        });
    }

    /// Mark region as modified from the CPU
    /// but don't mark it as modified until FlusHCachedWrites is called.
    void CachedCpuWrite(VAddr dirty_cpu_addr, u64 query_size) {
        ever_synchronized_pages.Mark(dirty_cpu_addr, query_size);
        IteratePages<true>(dirty_cpu_addr, query_size, [this](Manager* manager, u64 offset, size_t size) {
            const VAddr cpu_address = manager->cpu_addr + offset;
            manager->ChangeRegionState(Type::CachedCPU, true, cpu_address, size);
            cached_pages.insert(u32(cpu_address >> HIGHER_PAGE_BITS));
        });
    }

    /// Flushes cached CPU writes, and notify the device_tracker about the deltas
    void FlushCachedWrites(VAddr query_cpu_addr, u64 query_size) noexcept {
        IteratePages<false>(query_cpu_addr, query_size, [](Manager* manager, [[maybe_unused]] u64 offset, [[maybe_unused]] size_t size) {
            manager->FlushCachedWrites();
        });
    }

    void FlushCachedWrites() noexcept {
        for (auto id : cached_pages) {
            top_tier[id]->FlushCachedWrites();
        }
        cached_pages.clear();
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    template <typename Func>
    void ForEachUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        ever_synchronized_pages.Mark(query_cpu_range, query_size);
        IteratePages<true>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + offset, size, func);
        });
    }

    /// Call 'func' for each range marked as Preflushable
    template <typename Func>
    void ForEachPreflushableRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::Preflushable, false, manager->cpu_addr + offset, size, func);
        });
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <typename Func>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, bool clear, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func, clear](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, clear, manager->cpu_addr + offset, size, func);
        });
    }

    template <typename Func>
    void ForEachDownloadRangeAndClear(VAddr query_cpu_range, u64 query_size, Func&& func) {
        IteratePages<false>(query_cpu_range, query_size, [&func](Manager* manager, u64 offset, size_t size) {
            manager->ForEachModifiedRange(Type::GPU, true, manager->cpu_addr + offset, size, func);
        });
    }

    /// ForEachUploadRange without the clean words (word_manager.h) at either end of each manager's
    /// part of the range: they have nothing to find or change. A per-word form of a ported skip.
    /// Returns false when every word was clean (nothing walked at all).
    template <typename Func>
    bool ForEachDirtyUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        ever_synchronized_pages.Mark(query_cpu_range, query_size);
        bool walked = false;
        IteratePages<true>(query_cpu_range, query_size, [&func, &walked](Manager* manager, u64 offset, size_t size) {
            const auto [begin, end] = manager->DirtyWindow(offset, size);
            if (begin >= end) {
                return;
            }
            walked = true;
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + begin,
                                          static_cast<s64>(end - begin), func);
        });
        return walked;
    }

    /// EDEN_NX2_STORAGE_SYNC_SCAN: ForEachUploadRange over only each manager's part of the range
    /// from its first to its last word with a CPU or Untracked page, found by reading the words
    /// (WordManager::ScannedDirtyWindow; clean_words is not used). The words left out have
    /// nothing the full walk would upload or re-protect. Missing managers are created, as the
    /// full walk creates them (every page CPU-modified, so walked whole). Returns false when
    /// nothing was walked.
    template <typename Func>
    bool ForEachScannedUploadRange(VAddr query_cpu_range, u64 query_size, Func&& func) {
        ever_synchronized_pages.Mark(query_cpu_range, query_size);
        bool walked = false;
        IteratePages<true>(query_cpu_range, query_size, [&func, &walked](Manager* manager, u64 offset, size_t size) {
            const auto [begin, end] = manager->ScannedDirtyWindow(offset, size);
            if (begin >= end) {
                return;
            }
            walked = true;
            manager->ForEachModifiedRange(Type::CPU, true, manager->cpu_addr + begin,
                                          static_cast<s64>(end - begin), func);
        });
        return walked;
    }

    /// EDEN_NX2_STORAGE_SYNC_MEMO: the sum of the dirty generations (WordManager::dirty_gen) of
    /// the managers over a range. The generations only grow, so the sum is unchanged exactly
    /// when none of them moved: no page of those managers gained a CPU or Untracked bit. False
    /// when a manager is missing (nothing to prove; the walk would create it all dirty).
    [[nodiscard]] bool DirtyGenerationSum(VAddr query_cpu_addr, u64 query_size,
                                          u64& sum) const noexcept {
        if (query_size == 0) {
            return false;
        }
        const size_t first = query_cpu_addr >> HIGHER_PAGE_BITS;
        const size_t last = (query_cpu_addr + query_size - 1) >> HIGHER_PAGE_BITS;
        if (last >= NUM_HIGH_PAGES) {
            return false;
        }
        u64 total = 0;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            const Manager* const manager = top_tier[page_index];
            if (manager == nullptr) {
                return false;
            }
            total += manager->dirty_gen;
        }
        sum = total;
        return true;
    }

    struct DirtyWalkCheck {
        u64 cpu_pages = 0;       ///< CPU pages the full walk uploads and the dirty-word walk would not
        u64 untracked_pages = 0; ///< Untracked pages it re-tracks and the dirty-word walk would not
        bool walks = false;      ///< The dirty-word walk would walk anything (a manager to create counts)
    };
    /// EDEN_NX2_EXP59 & 8: what ForEachDirtyUploadRange would leave out of the range that
    /// ForEachUploadRange finds or changes. Changes nothing; call it before the full walk.
    [[nodiscard]] DirtyWalkCheck CheckDirtyWalk(VAddr query_cpu_range, u64 query_size) {
        DirtyWalkCheck check;
        if (query_size == 0) {
            return check;
        }
        // A missing manager is created by both walks with every word dirty: no difference.
        const size_t first = query_cpu_range >> HIGHER_PAGE_BITS;
        const size_t last =
            (query_cpu_range + query_size - 1) >> HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            check.walks |= top_tier[page_index] == nullptr;
        }
        IteratePages<false>(query_cpu_range, query_size, [&check](Manager* manager, u64 offset, size_t size) {
            const auto [begin, end] = manager->DirtyWindow(offset, size);
            check.walks |= begin < end;
            const auto [cpu_pages, untracked_pages] = manager->DirtyWalkMisses(offset, size);
            check.cpu_pages += cpu_pages;
            check.untracked_pages += untracked_pages;
        });
        return check;
    }
    /// EDEN_NX2_STORAGE_SYNC_SCAN=2: what ForEachScannedUploadRange would leave out of the range
    /// that ForEachUploadRange finds or changes (counted over every word). Changes nothing; call
    /// it before the full walk. A missing manager is created whole by both walks: no difference.
    [[nodiscard]] DirtyWalkCheck CheckScannedWalk(VAddr query_cpu_range, u64 query_size) {
        DirtyWalkCheck check;
        if (query_size == 0) {
            return check;
        }
        IteratePages<false>(query_cpu_range, query_size, [&check](Manager* manager, u64 offset, size_t size) {
            const auto [begin, end] = manager->ScannedDirtyWindow(offset, size);
            check.walks |= begin < end;
            const auto [cpu_pages, untracked_pages] = manager->ScannedWalkMisses(offset, size);
            check.cpu_pages += cpu_pages;
            check.untracked_pages += untracked_pages;
        });
        return check;
    }

    /// Verify mode: true when the clean words of the managers over a range match a recount.
    [[nodiscard]] bool CleanWordsHold(VAddr query_cpu_addr, u64 query_size) const noexcept {
        const size_t first = query_cpu_addr >> HIGHER_PAGE_BITS;
        const size_t last = (query_cpu_addr + (std::max<u64>)(query_size, 1) - 1) >> HIGHER_PAGE_BITS;
        for (size_t page_index = first; page_index <= last; ++page_index) {
            if (top_tier[page_index] && !top_tier[page_index]->CleanWordsHold()) {
                return false;
            }
        }
        return true;
    }

    /// EXP88: whether a page of the range may ever have left
    /// the CPU-modified state: an upload walk asked for it, an unmark, a cached write, a GPU or
    /// Preflushable mark. Pages without any are CPU-modified and untracked as their managers
    /// are made, so a CPU mark there changes nothing. Bits never clear. Any thread, no lock.
    [[nodiscard]] bool IsRegionEverSynchronized(VAddr query_cpu_addr, u64 query_size) const noexcept {
        return ever_synchronized_pages.Any(query_cpu_addr, query_size);
    }

    /// EXP88: whether a page of the range was ever marked GPU-modified (IsRegionGpuModified can
    /// only be true there). Bits never clear. Any thread, no lock.
    [[nodiscard]] bool IsRegionEverGpuModified(VAddr query_cpu_addr, u64 query_size) const noexcept {
        return ever_gpu_modified_pages.Any(query_cpu_addr, query_size);
    }

    /// EXP88 verify: a CPU mark of the range would change no page's state (missing managers are
    /// made with every page CPU-modified and untracked). Under the buffer cache's lock.
    [[nodiscard]] bool IsRegionUntouched(VAddr query_cpu_addr, u64 query_size) noexcept {
        return !IteratePages<false>(query_cpu_addr, query_size,
                                    [](Manager* manager, u64 offset, size_t size) {
                                        return !manager->IsRegionUntouched(offset, size);
                                    });
    }

private:
    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, Manager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    template <bool create_region_on_fail, typename Func>
    std::pair<u64, u64> IteratePairs(VAddr cpu_address, size_t size, Func&& func) {
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & HIGHER_PAGE_MASK};
        u64 begin = (std::numeric_limits<u64>::max)();
        u64 end = 0;
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            const auto execute = [&] {
                auto [new_begin, new_end] = func(manager, page_offset, copy_amount);
                if (new_begin != 0 || new_end != 0) {
                    const u64 base_address = page_index << HIGHER_PAGE_BITS;
                    begin = (std::min)(new_begin + base_address, begin);
                    end = (std::max)(new_end + base_address, end);
                }
            };
            if (manager) {
                execute();
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                execute();
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        if (begin < end) {
            return std::make_pair(begin, end);
        } else {
            return std::make_pair(0ULL, 0ULL);
        }
    }

    void CreateRegion(std::size_t page_index) {
        const VAddr base_cpu_addr = page_index << HIGHER_PAGE_BITS;
        top_tier[page_index] = GetNewManager(base_cpu_addr);
        // FRITCHII_BIND_SKIP: its pages start CPU-modified and untracked, no hook saw it.
        ::Common::BindSkip::NoteTrackerWide();
    }

    Manager* GetNewManager(VAddr base_cpu_address) {
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                new (&last_pool[i]) Manager(0, *device_tracker);
                free_managers.push_back(&last_pool[i]);
            }
        }
        Manager* new_manager = free_managers.front();
        new_manager->cpu_addr = base_cpu_address;
        free_managers.pop_front();
        return new_manager;
    }

    std::array<Manager*, NUM_HIGH_PAGES> top_tier{};
    std::deque<std::array<Manager, MANAGER_POOL_SIZE>> manager_pool;
    std::deque<Manager*> free_managers;
    ::Common::unordered_set<u32> cached_pages;
    PageFilter ever_synchronized_pages;  ///< EXP88: IsRegionEverSynchronized
    PageFilter ever_gpu_modified_pages;  ///< EXP88: IsRegionEverGpuModified
    DeviceTracker* device_tracker = nullptr;
};

} // namespace VideoCommon
