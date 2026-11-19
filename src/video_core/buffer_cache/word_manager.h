// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "video_core/nx2_debug_control.h"
#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/fritchii_bind_skip.h"
#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2_perf.h"

namespace VideoCommon {

constexpr u64 PAGES_PER_WORD = 64;
constexpr u64 BYTES_PER_PAGE = Core::DEVICE_PAGESIZE;
constexpr u64 BYTES_PER_WORD = PAGES_PER_WORD * BYTES_PER_PAGE;

enum class Type {
    CPU,
    GPU,
    CachedCPU,
    Untracked,
    Preflushable,
    Hot,   ///< Pages the CPU rewrites again and again: kept modified, never write-protected
    Flip,  ///< Pages the CPU modified once, while tracked, since the last reset
    NoHot, ///< Pages the GPU modified: never hot
    HotClean, ///< Hot pages already uploaded this frame
    Max
};

/// Frames since start, moved by BufferCache::TickFrame. Hot page state is dropped every 128.
inline std::atomic<u32> g_hot_page_epoch{0};

/// Ported fix: back thread 14.5 -> 13.8 ms a frame in OoT, protect calls
/// -71%. Fritchii: behind EDEN_NX2_EXP48 (live, default off) until a fight A/B and a flicker
/// check: a hot page the CPU writes again after its upload in the same frame reaches the GPU
/// one frame later. EDEN_NX2_HOT_PAGES=0 also turns it off.
[[nodiscard]] inline bool HotPagesEnabled() {
    static const bool allowed = [] {
        const char* value = std::getenv("EDEN_NX2_HOT_PAGES");
        return value == nullptr || value[0] != '0';
    }();
    return allowed && VideoCore::Nx2Debug::Experiment(48) != 0;
}

/// HotPagesEnabled() as of the last BufferCache::TickFrame, which refreshes it with the epoch: the
/// tracker walks test this one relaxed load instead of a guarded static and two atomics per walk
/// (a live EXP48 toggle applies from the next frame).
inline std::atomic<bool> g_hot_pages_on{false};

[[nodiscard]] inline bool HotPagesOn() {
    return g_hot_pages_on.load(std::memory_order_relaxed);
}

/// EXP109: page tracking changes a thread
/// applies itself once it drops the buffer cache's lock (RasterizerVulkan::OnCPUWrite): changing a
/// page's host protection is a system call, and every other thread drawing or reading waited for
/// it. Ours collects only untracking (ApplyCollectedRanges).
class DeferredPageCaching {
public:
    using Ranges = std::vector<std::pair<VAddr, size_t>>;
    using Apply = void (*)(void* tracker, std::span<const std::pair<VAddr, size_t>>, s32 delta);

    void Add(void* tracker, Apply apply, const Ranges& ranges, s32 delta) {
        batches.push_back(Batch{tracker, apply, delta, collected.size(), ranges.size()});
        collected.insert(collected.end(), ranges.begin(), ranges.end());
    }

    /// In the order they were made.
    void ApplyAll() {
        for (const Batch& batch : batches) {
            batch.apply(batch.tracker,
                        std::span<const std::pair<VAddr, size_t>>(collected).subspan(batch.begin,
                                                                                    batch.count),
                        batch.delta);
        }
        batches.clear();
        collected.clear();
    }

private:
    struct Batch {
        void* tracker;
        Apply apply;
        s32 delta;
        size_t begin;
        size_t count;
    };
    std::vector<Batch> batches;
    Ranges collected;
};

/// Set while the calling thread collects its page untracking instead of applying it.
inline thread_local DeferredPageCaching* t_deferred_page_caching{};

template <class DeviceTracker, size_t stack_words, size_t size_bytes>
struct WordManager {
    using CollectedRanges = std::vector<std::pair<VAddr, size_t>>;

    static constexpr size_t num_words = Common::DivCeil(size_bytes, BYTES_PER_WORD);
    static_assert(num_words <= 64, "clean_words holds a bit per word");

    explicit WordManager(VAddr cpu_addr_, DeviceTracker& tracker_) : tracker{&tracker_}, cpu_addr{cpu_addr_} {
        std::fill_n(heap.data() + size_t(Type::CPU) * num_words, num_words, ~u64{0});
        std::fill_n(heap.data() + size_t(Type::Untracked) * num_words, num_words, ~u64{0});
        // Clean up tailing bits
        u64 const last_word_size = size_bytes % BYTES_PER_WORD;
        u64 const last_local_page = Common::DivCeil(last_word_size, BYTES_PER_PAGE);
        u64 const shift = (PAGES_PER_WORD - last_local_page) % PAGES_PER_WORD;
        u64 const last_word = (~u64{0} << shift) >> shift;
        heap[num_words * size_t(Type::CPU) + num_words - 1] = last_word;
        heap[num_words * size_t(Type::Untracked) + num_words - 1] = last_word;
        for (size_t index = 0; index < num_words; ++index) {
            NoteCleanWord(index);
        }
    }
    explicit WordManager() = default;

    static u64 ExtractBits(u64 word, size_t page_start, size_t page_end) {
        constexpr size_t number_bits = sizeof(u64) * 8;
        const size_t limit_page_end = number_bits - (std::min)(page_end, number_bits);
        u64 bits = (word >> page_start) << page_start;
        bits = (bits << limit_page_end) >> limit_page_end;
        return bits;
    }

    static std::pair<size_t, size_t> GetWordPage(VAddr address) {
        const size_t converted_address = size_t(address);
        const size_t word_number = converted_address / BYTES_PER_WORD;
        const size_t amount_pages = converted_address % BYTES_PER_WORD;
        return std::make_pair(word_number, amount_pages / BYTES_PER_PAGE);
    }

    template <typename Func>
    void IterateWords(size_t offset, size_t size, Func&& func) const {
        using FuncReturn = std::invoke_result_t<Func, std::size_t, u64>;
        const size_t start = size_t(std::max<s64>(s64(offset), 0LL));
        const size_t end = size_t(std::max<s64>(s64(offset + size), 0LL));
        if (!(start >= size_bytes || end <= start)) {
            auto [start_word, start_page] = GetWordPage(start);
            auto [end_word, end_page] = GetWordPage(end + BYTES_PER_PAGE - 1ULL);
            start_word = (std::min)(start_word, num_words);
            end_word = (std::min)(end_word, num_words);
            const size_t diff = end_word - start_word;
            end_word += (end_page + PAGES_PER_WORD - 1ULL) / PAGES_PER_WORD;
            end_word = (std::min)(end_word, num_words);
            end_page += diff * PAGES_PER_WORD;
            constexpr u64 base_mask{~0ULL};
            for (size_t word_index = start_word; word_index < end_word; word_index++) {
                const u64 mask = ExtractBits(base_mask, start_page, end_page);
                start_page = 0;
                end_page -= PAGES_PER_WORD;
                if constexpr (std::is_same_v<FuncReturn, bool>) { // bool return
                    if (func(word_index, mask))
                        return;
                } else {
                    func(word_index, mask);
                }
            }
        }
    }

    template <typename Func>
    void IteratePages(u64 mask, Func&& func) const {
        size_t offset = 0;
        while (mask != 0) {
            const size_t empty_bits = std::countr_zero(mask);
            offset += empty_bits;
            mask = mask >> empty_bits;

            const size_t continuous_bits = std::countr_one(mask);
            func(offset, continuous_bits);
            mask = continuous_bits < PAGES_PER_WORD ? (mask >> continuous_bits) : 0;
            offset += continuous_bits;
        }
    }

    /// @brief Change the state of a range of pages
    /// @param type          Type of the page
    /// @param enable        If enabling or disabling
    /// @param dirty_addr    Base address to mark or unmark as modified
    /// @param size          Size in bytes to mark or unmark as modified
    /// Forgets which pages are hot every 128 frames, so a page that stopped being rewritten gets
    /// tracked again at its next upload.
    void RefreshHotEpoch() noexcept {
        const u32 frame = g_hot_page_epoch.load(std::memory_order_relaxed);
        if (frame != hot_frame) {
            hot_frame = frame;
            std::fill_n(heap.data() + size_t(Type::HotClean) * num_words, num_words, u64{0});
        }
        const u32 now = frame >> 7;
        if (now != hot_epoch) {
            hot_epoch = now;
            std::fill_n(heap.data() + size_t(Type::Hot) * num_words, num_words, u64{0});
            std::fill_n(heap.data() + size_t(Type::Flip) * num_words, num_words, u64{0});
        }
    }

    /// EXP108 (ported): ChangeRegionState(Type::CPU, true,
    /// ...) leaving the pages tracked (Untracked unchanged): pages tracked now stay tracked, so
    /// clearing the bits at the upload tracks nothing again. Ours also moves dirty_gen where a
    /// page gains its CPU bit (EDEN_NX2_STORAGE_SYNC_MEMO) and, with EXP48, drops the pages'
    /// hot-clean bits, so an upload this frame doesn't pass them over.
    void MarkWrittenTracked(u64 dirty_addr, u64 size) noexcept {
        const bool hot_pages = HotPagesOn();
        if (hot_pages) [[unlikely]] {
            RefreshHotEpoch();
        }
        std::span<u64> cpu_words = Span(Type::CPU);
        std::span<u64> cached_words = Span(Type::CachedCPU);
        std::span<u64> hot_clean_words = Span(Type::HotClean);
        const bool skip_gens = Common::BindSkip::On();
        IterateWords(dirty_addr - cpu_addr, size, [&](size_t index, u64 mask) {
            if ((cpu_words[index] & mask) != mask) {
                ++dirty_gen;
            }
            if (skip_gens) [[unlikely]] {
                // FRITCHII_BIND_SKIP: pages gaining their CPU bit or losing a cached write.
                const u64 changed = (~cpu_words[index] | cached_words[index]) & mask;
                if (changed != 0) {
                    Common::BindSkip::NoteTrackerWord(cpu_addr + index * BYTES_PER_WORD, changed);
                }
            }
            cpu_words[index] |= mask;
            cached_words[index] &= ~mask;
            if (hot_pages) [[unlikely]] {
                hot_clean_words[index] &= ~mask;
            }
            NoteCleanWord(index);
        });
    }

    void ChangeRegionState(Type type, bool enable, u64 dirty_addr, u64 size) noexcept {
        // EXP48 off (the default): the walk without a hot-page span or branch (the code before
        // d20c32704c).
        if (HotPagesOn()) [[unlikely]] {
            ChangeRegionStateImpl<true>(type, enable, dirty_addr, size);
        } else {
            ChangeRegionStateImpl<false>(type, enable, dirty_addr, size);
        }
    }

    template <bool hot_pages>
    void ChangeRegionStateImpl(Type type, bool enable, u64 dirty_addr, u64 size) noexcept {
        if constexpr (hot_pages) {
            RefreshHotEpoch();
        }
        std::span<u64> state_words = Span(type);
        [[maybe_unused]] std::span<u64> untracked_words = Span(Type::Untracked);
        [[maybe_unused]] std::span<u64> cached_words = Span(Type::CachedCPU);
        CollectedRanges ranges;
        // FRITCHII_BIND_SKIP: the pages whose CPU, Untracked, cached-write or GPU bit changes.
        const bool skip_gens = Common::BindSkip::On() &&
                               (type == Type::CPU || type == Type::CachedCPU || type == Type::GPU);
        IterateWords(dirty_addr - cpu_addr, size, [&](size_t index, u64 mask) {
            if (skip_gens) [[unlikely]] {
                u64 changed = (enable ? ~state_words[index] : state_words[index]) & mask;
                if (type != Type::GPU) {
                    changed |= (enable ? ~untracked_words[index] : untracked_words[index]) & mask;
                }
                if (changed != 0) {
                    Common::BindSkip::NoteTrackerWord(cpu_addr + index * BYTES_PER_WORD, changed);
                }
            }
            if (enable && (type == Type::CPU || type == Type::CachedCPU)) {
                // A page gains a CPU or an Untracked bit (EDEN_NX2_STORAGE_SYNC_MEMO): a CPU
                // mark sets both, a cached write the Untracked one. Pages holding them already
                // cannot be in a range a memo holds as clean.
                const u64 held = type == Type::CPU ? state_words[index] & untracked_words[index]
                                                   : untracked_words[index];
                if ((held & mask) != mask) {
                    ++dirty_gen;
                }
            }
            if constexpr (hot_pages) {
                if (enable) {
                    HotPagesOnChange(type, index, mask, untracked_words[index]);
                }
            }
            if (type == Type::CPU || type == Type::CachedCPU) {
                CollectChangedRanges(!enable, index, untracked_words[index], mask, ranges);
            }
            if (enable) {
                state_words[index] |= mask;
                if (type == Type::CPU || type == Type::CachedCPU)
                    untracked_words[index] |= mask;
                if (type == Type::CPU)
                    cached_words[index] &= ~mask;
            } else {
                if (type == Type::CPU)
                    cached_words[index] &= ~(state_words[index] & mask);
                state_words[index] &= ~mask;
                if (type == Type::CPU || type == Type::CachedCPU)
                    untracked_words[index] &= ~mask;
            }
            if (type == Type::CPU || type == Type::CachedCPU) {
                NoteCleanWord(index);
            }
        });
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, (!enable) ? 1 : -1);
        }
    }

    /// EXP48: the hot-page bookkeeping of a page range marked modified (ChangeRegionState with
    /// `enable`), before the state words change. `untracked` is the word's Untracked bits then.
    void HotPagesOnChange(Type type, size_t index, u64 mask, u64 untracked) noexcept {
        std::span<u64> hot_words = Span(Type::Hot);
        std::span<u64> flip_words = Span(Type::Flip);
        std::span<u64> no_hot_words = Span(Type::NoHot);
        std::span<u64> hot_clean_words = Span(Type::HotClean);
        if (type == Type::CPU) {
            // Whoever marks a hot page modified (a write made by the GPU in the middle of
            // the frame, a bank re-upload) wants it uploaded again, whatever was sent.
            hot_clean_words[index] &= ~mask;
            // Tracked pages the CPU just modified: the second time within the reset
            // period makes a page hot, unless the GPU ever wrote it.
            const u64 flipped = ~untracked & mask;
            if (flipped != 0) {
                hot_words[index] |= flip_words[index] & flipped & ~no_hot_words[index];
                flip_words[index] |= flipped;
            }
        } else if (type == Type::GPU) {
            no_hot_words[index] |= mask;
            hot_words[index] &= ~mask;
            hot_clean_words[index] &= ~mask;
        }
    }

    /// @brief Loop over each page in the given range.
    /// Turn off those bits and notify the tracker if needed. Call the given function on each turned off range.
    /// @param type            Type of the address
    /// @param clear           Whetever to clear
    /// @param query_cpu_range Base CPU address to loop over
    /// @param size            Size in bytes of the CPU range to loop over
    /// @param func            Function to call for each turned off region
    template <typename Func>
    void ForEachModifiedRange(Type type, bool clear, VAddr query_cpu_range, s64 size, Func&& func) {
        //static_assert(type != Type::Untracked);
        // EXP48 off (the default), or not a CPU clear: the walk without a hot-page span, mask or
        // branch (the code before d20c32704c).
        if (clear && type == Type::CPU && HotPagesOn()) [[unlikely]] {
            ForEachModifiedRangeImpl<true>(type, clear, query_cpu_range, size, func);
        } else {
            ForEachModifiedRangeImpl<false>(type, clear, query_cpu_range, size, func);
        }
    }

    template <bool hot_pages, typename Func>
    void ForEachModifiedRangeImpl(Type type, bool clear, VAddr query_cpu_range, s64 size,
                                  Func& func) {
        if constexpr (hot_pages) {
            RefreshHotEpoch();
        }
        [[maybe_unused]] std::span<u64> hot_words = Span(Type::Hot);
        [[maybe_unused]] std::span<u64> hot_clean_words = Span(Type::HotClean);
        std::span<u64> state_words = Span(type);
        std::span<u64> untracked_words = Span(Type::Untracked);
        std::span<u64> cached_words = Span(Type::CachedCPU);
        size_t const offset = query_cpu_range - cpu_addr;
        bool pending = false;
        size_t pending_offset{};
        size_t pending_pointer{};
        const auto release = [&]() {
            func(cpu_addr + pending_offset * BYTES_PER_PAGE,
                 (pending_pointer - pending_offset) * BYTES_PER_PAGE);
        };
        CollectedRanges ranges;
        // FRITCHII_BIND_SKIP: the pages a clear takes a CPU, Untracked, cached-write or GPU bit
        // from (a re-protect at an upload, a download's clear).
        const bool skip_gens = clear && Common::BindSkip::On() &&
                               (type == Type::CPU || type == Type::CachedCPU || type == Type::GPU);
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            if (skip_gens) [[unlikely]] {
                // A superset of what either walk below clears (the hot one keeps hot pages).
                u64 changed = state_words[index] & mask;
                if (type != Type::GPU) {
                    changed |= (untracked_words[index] | cached_words[index]) & mask;
                }
                if (changed != 0) {
                    Common::BindSkip::NoteTrackerWord(cpu_addr + index * BYTES_PER_WORD, changed);
                }
            }
            u64 word;
            if constexpr (hot_pages) {
                // Only for CPU clears (ForEachModifiedRange's dispatch).
                // Hot pages uploaded since the frame began are not modified again until the next
                // frame.
                const u64 hot_clean = hot_words[index] & hot_clean_words[index] & mask;
                word = state_words[index] & mask & ~hot_clean;
                // Hot pages stay modified and untracked: their next use uploads them again,
                // with no write protection to set and no write fault to take.
                const u64 hot = hot_words[index] & mask;
                hot_clean_words[index] |= hot & word;
                const u64 clear_mask = mask & ~hot;
                CollectChangedRanges(true, index, untracked_words[index], clear_mask, ranges);
                state_words[index] &= ~clear_mask;
                untracked_words[index] &= ~clear_mask;
                cached_words[index] &= ~(word & ~hot);
                NoteCleanWord(index);
            } else {
                word = state_words[index] & mask;
                if (clear) {
                    if (type == Type::CPU || type == Type::CachedCPU) {
                        CollectChangedRanges(true, index, untracked_words[index], mask, ranges);
                    }
                    state_words[index] &= ~mask;
                    if (type == Type::CPU || type == Type::CachedCPU)
                        untracked_words[index] &= ~mask;
                    if (type == Type::CPU)
                        cached_words[index] &= ~word;
                    if (type == Type::CPU || type == Type::CachedCPU)
                        NoteCleanWord(index);
                }
            }
            const size_t base_offset = index * PAGES_PER_WORD;
            IteratePages(word, [&](size_t pages_offset, size_t pages_size) {
                if (!pending) {
                    pending_offset = base_offset + pages_offset;
                    pending_pointer = base_offset + pages_offset + pages_size;
                    pending = true;
                } else if (pending_pointer == base_offset + pages_offset) {
                    pending_pointer += pages_size;
                } else {
                    func(cpu_addr + pending_offset * BYTES_PER_PAGE, (pending_pointer - pending_offset) * BYTES_PER_PAGE);
                    pending_offset = base_offset + pages_offset;
                    pending_pointer = base_offset + pages_offset + pages_size;
                }
            });
        });
        if (pending) {
            release();
        }
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, 1);
        }
    }

    /// @brief Returns true when a region has been modified
    /// @param type   Type of region
    /// @param offset Offset in bytes from the start of the buffer
    /// @param size   Size in bytes of the region to query for modifications
    [[nodiscard]] bool IsRegionModified(Type type, u64 offset, u64 size) const noexcept {
        //static_assert(type != Type::Untracked);
        const std::span<const u64> state_words = Span(type);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        bool result = false;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            return (state_words[index] & mask) != 0 ? (result = true) : false;
        });
        return result;
    }

    /// Every page of the region CPU-modified, and none GPU-modified or Preflushable (the raw bits,
    /// Untracked not masked out): its host contents are all stale (BufferCache::JoinOverlap).
    [[nodiscard]] bool IsRegionWhollyCpuModified(u64 offset, u64 size) const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> gpu_words = Span(Type::GPU);
        const std::span<const u64> preflush_words = Span(Type::Preflushable);
        bool whole = true;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if ((cpu_words[index] & mask) != mask || (gpu_words[index] & mask) != 0 ||
                (preflush_words[index] & mask) != 0) {
                whole = false;
                return true;
            }
            return false;
        });
        return whole;
    }

    /// EXP88 verify: every page of the range CPU-modified and untracked, none cached-written,
    /// GPU-modified, Preflushable or hot and clean, so a CPU mark there changes no word.
    [[nodiscard]] bool IsRegionUntouched(u64 offset, u64 size) const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        const std::span<const u64> cached_words = Span(Type::CachedCPU);
        const std::span<const u64> gpu_words = Span(Type::GPU);
        const std::span<const u64> preflush_words = Span(Type::Preflushable);
        const std::span<const u64> hot_clean_words = Span(Type::HotClean);
        bool untouched = true;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if ((cpu_words[index] & mask) != mask || (untracked_words[index] & mask) != mask ||
                ((cached_words[index] | gpu_words[index] | preflush_words[index] |
                  hot_clean_words[index]) &
                 mask) != 0) {
                untouched = false;
                return true;
            }
            return false;
        });
        return untouched;
    }

    /// @brief Returns a begin end pair with the inclusive modified region
    /// @param offset Offset in bytes from the start of the buffer
    /// @param size   Size in bytes of the region to query for modifications
    [[nodiscard]] std::pair<u64, u64> ModifiedRegion(Type type, u64 offset, u64 size) const noexcept {
        //static_assert(type != Type::Untracked);
        const std::span<const u64> state_words = Span(type);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        u64 begin = (std::numeric_limits<u64>::max)(), end = 0;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (type == Type::GPU)
                mask &= ~untracked_words[index];
            const u64 word = state_words[index] & mask;
            if (word != 0) {
                const u64 local_page_begin = std::countr_zero(word);
                const u64 local_page_end = PAGES_PER_WORD - std::countl_zero(word);
                const u64 page_index = index * PAGES_PER_WORD;
                begin = (std::min)(begin, page_index + local_page_begin);
                end = page_index + local_page_end;
            }
        });
        return begin < end ? std::make_pair<u64, u64>(begin * BYTES_PER_PAGE, end * BYTES_PER_PAGE)
            : std::make_pair<u64, u64>(0, 0);
    }

    void FlushCachedWrites() noexcept {
        auto const cached_words = Span(Type::CachedCPU);
        auto const untracked_words = Span(Type::Untracked);
        auto const cpu_words = Span(Type::CPU);
        CollectedRanges ranges;
        const bool skip_gens = Common::BindSkip::On();
        for (u64 word_index = 0; word_index < num_words; ++word_index) {
            const u64 cached_bits = cached_words[word_index];
            // The cached pages turn CPU-modified (EDEN_NX2_STORAGE_SYNC_MEMO).
            if (cached_bits != 0) {
                ++dirty_gen;
                if (skip_gens) [[unlikely]] {
                    // FRITCHII_BIND_SKIP: and change state.
                    Common::BindSkip::NoteTrackerWord(cpu_addr + word_index * BYTES_PER_WORD,
                                                      cached_bits);
                }
            }
            CollectChangedRanges(false, word_index, untracked_words[word_index], cached_bits, ranges);
            untracked_words[word_index] |= cached_bits;
            cpu_words[word_index] |= cached_bits;
            cached_words[word_index] = 0;
            NoteCleanWord(word_index);
        }
        if (!ranges.empty()) {
            ApplyCollectedRanges(ranges, -1);
        }
    }

    /// @brief Notify tracker about changes in the CPU tracking state of a word in the buffer
    /// @param add_to_tracker If add to tracker (selects changed bits)
    /// @param word_index     Index to the word to notify to the tracker
    /// @param current_bits   Current state of the word
    /// @param new_bits       New state of the word
    /// @tparam add_to_tracker True when the tracker should start tracking the new pages
    void CollectChangedRanges(bool add_to_tracker, u64 word_index, u64 current_bits, u64 new_bits, CollectedRanges& out_ranges) const {
        u64 changed_bits = (add_to_tracker ? current_bits : ~current_bits) & new_bits;
        VAddr addr = cpu_addr + word_index * BYTES_PER_WORD;
        IteratePages(changed_bits, [&](size_t offset, size_t size) {
            out_ranges.emplace_back(addr + offset * BYTES_PER_PAGE, size * BYTES_PER_PAGE);
        });
    }

    void ApplyCollectedRanges(CollectedRanges& ranges, int delta) const {
        if (ranges.empty())
            return;
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // EXP109: untracking, collected for after the lock. Only decrements:
        // a page left protected a little longer only traps once more, while an increment applied
        // after another thread's decrement could let a page other users track drop its
        // protection in between.
        if (delta < 0 && t_deferred_page_caching != nullptr && !Nx2Perf::Verify()) [[unlikely]] {
            CoalesceInPlace(ranges);
            t_deferred_page_caching->Add(
                tracker,
                [](void* deferred_tracker, std::span<const std::pair<VAddr, size_t>> runs,
                   s32 run_delta) {
                    static_cast<DeviceTracker*>(deferred_tracker)
                        ->UpdatePagesCachedBatch(runs, run_delta);
                },
                ranges, delta);
            ranges.clear();
            return;
        }
        if (Nx2Perf::On(Nx2Perf::Opt::SortedPageBatches) && !Nx2Perf::Verify()) {
            CoalesceInPlace(ranges);
            tracker->UpdatePagesCachedBatch(
                std::span<const std::pair<VAddr, size_t>>(ranges.data(), ranges.size()), delta);
            ranges.clear();
            return;
        }
        // Coalesce adjacent/contiguous ranges
        std::vector<std::pair<VAddr, size_t>> coalesced;
        coalesced.reserve(ranges.size());
        VAddr cur_addr = ranges[0].first;
        size_t cur_size = static_cast<size_t>(ranges[0].second);
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (cur_addr + cur_size == ranges[i].first) {
                cur_size += static_cast<size_t>(ranges[i].second);
            } else {
                coalesced.emplace_back(cur_addr, cur_size);
                cur_addr = ranges[i].first;
                cur_size = static_cast<size_t>(ranges[i].second);
            }
        }
        coalesced.emplace_back(cur_addr, cur_size);
        if (Nx2Perf::On(Nx2Perf::Opt::SortedPageBatches)) {
            CoalesceInPlace(ranges);
            static std::atomic<u32> logged{};
            if (!std::ranges::equal(ranges, coalesced) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: tracker runs from {:#x} coalesced differently",
                          coalesced.front().first);
            }
        }
        // Use batch API to reduce lock acquisitions and contention.
        tracker->UpdatePagesCachedBatch(coalesced, delta);
        ranges.clear();
    }

    /// Merges the contiguous neighbours of sorted, disjoint runs where they lie (no copy).
    static void CoalesceInPlace(CollectedRanges& ranges) noexcept {
        size_t last = 0;
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[last].first + ranges[last].second == ranges[i].first) {
                ranges[last].second += ranges[i].second;
            } else {
                ranges[++last] = ranges[i];
            }
        }
        ranges.resize(last + 1);
    }

    /// @brief Notify tracker about changes in the CPU tracking state of a word in the buffer
    /// @param add_to_tracker True when the tracker should start tracking the new pages
    /// @param word_index   Index to the word to notify to the tracker
    /// @param current_bits Current state of the word
    /// @param new_bits     New state of the word
    void NotifyRasterizer(bool add_to_tracker, u64 word_index, u64 current_bits, u64 new_bits) const {
        u64 changed_bits = (add_to_tracker ? current_bits : ~current_bits) & new_bits;
        VAddr addr = cpu_addr + word_index * BYTES_PER_WORD;
        IteratePages(changed_bits, [&](size_t offset, size_t size) {
            tracker->UpdatePagesCachedCount(addr + offset * BYTES_PER_PAGE, size * BYTES_PER_PAGE, add_to_tracker ? 1 : -1);
        });
    }

    std::span<u64> Span(Type type) noexcept {
        return std::span<u64>(heap.data() + num_words * size_t(type), num_words);
    }

    std::span<const u64> Span(Type type) const noexcept {
        return std::span<const u64>(heap.data() + num_words * size_t(type), num_words);
    }

    /// Words of [first_word, last_word] that are not clean, one bit each. A clean word has no CPU
    /// or Untracked page, so an upload over it finds and changes nothing.
    [[nodiscard]] u64 DirtyWords(size_t first_word, size_t last_word) const noexcept {
        const u64 mask = (~u64{0} >> (63 - (last_word - first_word))) << first_word;
        return mask & ~clean_words;
    }

    /// The part of [offset, offset + size) that the dirty-word walk visits: from its first dirty
    /// word to its last (MemoryTrackerBase::ForEachDirtyUploadRange). Empty when all are clean.
    [[nodiscard]] std::pair<u64, u64> DirtyWindow(u64 offset, u64 size) const noexcept {
        const u64 dirty =
            DirtyWords(offset / BYTES_PER_WORD, (offset + size - 1) / BYTES_PER_WORD);
        if (dirty == 0) {
            return {offset + size, offset + size};
        }
        const u64 begin = (std::max<u64>)(offset, std::countr_zero(dirty) * BYTES_PER_WORD);
        const u64 end =
            (std::min<u64>)(offset + size, (64 - std::countl_zero(dirty)) * BYTES_PER_WORD);
        return {begin, end};
    }

    /// EDEN_NX2_EXP59 & 8: the CPU and Untracked pages of [offset, offset + size) outside
    /// DirtyWindow, counted from the words themselves (not clean_words). The full walk uploads
    /// the first and re-tracks the second; the dirty-word walk would do neither.
    [[nodiscard]] std::pair<u64, u64> DirtyWalkMisses(u64 offset, u64 size) const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        const auto [begin, end] = DirtyWindow(offset, size);
        u64 cpu_pages = 0;
        u64 untracked_pages = 0;
        const auto count = [&](u64 from, u64 to) {
            if (from >= to) {
                return;
            }
            IterateWords(from, to - from, [&](size_t index, u64 mask) {
                cpu_pages += std::popcount(cpu_words[index] & mask);
                untracked_pages += std::popcount(untracked_words[index] & mask);
            });
        };
        count(offset, begin);
        count(end, offset + size);
        return {cpu_pages, untracked_pages};
    }

    /// EDEN_NX2_STORAGE_SYNC_SCAN: the part of [offset, offset + size) from its first to its last
    /// word holding a CPU or Untracked page, read from those words themselves (not clean_words,
    /// which the CleanWords walk trusts). A CPU clear (ForEachModifiedRange) over a word with
    /// neither uploads nothing, re-protects nothing and stores back what every word holds, so
    /// the walk over this window does what the walk over the whole range does. Empty when no word
    /// has either. Read only: two loads a word, against the clear's stores and NoteCleanWord.
    [[nodiscard]] std::pair<u64, u64> ScannedDirtyWindow(u64 offset, u64 size) const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        size_t first = num_words;
        size_t last = 0;
        IterateWords(offset, size, [&](size_t index, u64 mask) {
            if (((cpu_words[index] | untracked_words[index]) & mask) != 0) {
                first = (std::min)(first, index);
                last = index;
            }
        });
        if (first == num_words) {
            return {offset + size, offset + size};
        }
        const u64 begin = (std::max<u64>)(offset, first * BYTES_PER_WORD);
        const u64 end = (std::min<u64>)(offset + size, (last + 1) * BYTES_PER_WORD);
        return {begin, end};
    }

    /// EDEN_NX2_STORAGE_SYNC_SCAN=2: the CPU and Untracked pages of [offset, offset + size)
    /// outside ScannedDirtyWindow, counted word by word over the whole range. Both must stay 0.
    [[nodiscard]] std::pair<u64, u64> ScannedWalkMisses(u64 offset, u64 size) const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        const auto [begin, end] = ScannedDirtyWindow(offset, size);
        u64 cpu_pages = 0;
        u64 untracked_pages = 0;
        const auto count = [&](u64 from, u64 to) {
            if (from >= to) {
                return;
            }
            IterateWords(from, to - from, [&](size_t index, u64 mask) {
                // A word the window ends inside belongs to it: count only the pages outside.
                u64 outside = mask;
                if (begin < end) {
                    const u64 word_begin = index * BYTES_PER_WORD;
                    const u64 word_end = word_begin + BYTES_PER_WORD;
                    if (word_begin < end && word_end > begin) {
                        outside = 0;
                    }
                }
                cpu_pages += std::popcount(cpu_words[index] & outside);
                untracked_pages += std::popcount(untracked_words[index] & outside);
            });
        };
        count(offset, begin);
        count(end, offset + size);
        return {cpu_pages, untracked_pages};
    }

    /// Recounts the clean words: true when clean_words matches the CPU and Untracked words.
    [[nodiscard]] bool CleanWordsHold() const noexcept {
        const std::span<const u64> cpu_words = Span(Type::CPU);
        const std::span<const u64> untracked_words = Span(Type::Untracked);
        for (size_t index = 0; index < num_words; ++index) {
            const bool clean = (cpu_words[index] | untracked_words[index]) == 0;
            if (((clean_words >> index) & 1) != u64{clean}) {
                return false;
            }
        }
        return true;
    }

    void NoteCleanWord(size_t index) noexcept {
        const bool clean = (heap[index] | heap[num_words * size_t(Type::Untracked) + index]) == 0;
        clean_words = (clean_words & ~(u64{1} << index)) | (u64{clean} << index);
    }

    std::array<u64, size_t(Type::Max) * num_words> heap = {};
    /// Bit per word, set while the word has no CPU and no Untracked page (DirtyWords). Every
    /// change to those words updates it: a per-word form of a ported upload_words count.
    u64 clean_words = 0;
    /// EDEN_NX2_STORAGE_SYNC_MEMO: bumped whenever a page of this manager gains a CPU or an
    /// Untracked bit (ChangeRegionStateImpl, FlushCachedWrites; the constructor's all-dirty
    /// state comes before any memo). Only grows. Nothing else sets those bits, so a range left
    /// clean at one value is still clean while it holds.
    u64 dirty_gen = 0;
    DeviceTracker* tracker = nullptr;
    VAddr cpu_addr = 0;
    u32 hot_epoch = 0;
    u32 hot_frame = 0;
};

} // namespace VideoCommon
