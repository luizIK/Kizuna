// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>

#include "common/common_types.h"

namespace VideoCommon {

/// The 4 KiB device pages a cache has ever registered objects on. Bits are never cleared, so a
/// clear bit proves the cache holds nothing there and an invalidation of those pages does nothing.
class PageFilter {
public:
    PageFilter() : words{std::make_unique<std::atomic<u64>[]>(NumWords)} {}

    /// Records that the cache holds objects on [address, address + size).
    void Mark(u64 address, u64 size) {
        if (size == 0) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumWords * 64) {
            overflow.store(true, std::memory_order_relaxed);
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_relaxed);
            }
            page += count;
        }
        if (also != nullptr) {
            also->Mark(address, size);
        }
    }

    /// One follower (fable_b 14.4 cut 5): every later Mark also marks `union_filter`, which must
    /// outlive this filter; the bits this filter holds now are copied into it first. A clear bit
    /// in the union then proves a clear bit in every filter that feeds it.
    void AlsoMark(PageFilter* union_filter) {
        also = union_filter;
        for (u64 i = 0; i < NumWords; ++i) {
            if (const u64 bits = words[i].load(std::memory_order_relaxed); bits != 0) {
                union_filter->words[i].fetch_or(bits, std::memory_order_relaxed);
            }
        }
        if (overflow.load(std::memory_order_relaxed)) {
            union_filter->overflow.store(true, std::memory_order_relaxed);
        }
    }

    /// Whether the cache may hold objects on [address, address + size).
    [[nodiscard]] bool Any(u64 address, u64 size) const {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumWords * 64 || overflow.load(std::memory_order_relaxed)) {
            return true;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if (words[page / 64].load(std::memory_order_relaxed) & mask) {
                return true;
            }
            page += count;
        }
        return false;
    }

private:
    static constexpr u64 PageBits = 12;
    static constexpr u64 NumWords = (u64{1} << (36 - PageBits)) / 64; // 2 MiB of bits
    std::unique_ptr<std::atomic<u64>[]> words;
    std::atomic<bool> overflow{};
    PageFilter* also{}; ///< AlsoMark's union, set before the threads that mark exist
};

/// EXP84 (fable52_answer_microcuts 3.4): how many registered objects cover each 64 KiB page now,
/// so "nothing here now" is answered without the cache's lock. PageFilter's bits never clear, and
/// the game's render targets move every frame, so its bits pile up over memory that holds no image
/// any more. Add at register, Remove at unregister with the same range. Conservative: a page any
/// object touches counts it; out of range answers "maybe".
class CountingPageFilter {
public:
    CountingPageFilter() : counts{std::make_unique<std::atomic<u32>[]>(NumPages)} {}

    void Add(u64 address, u64 size) {
        Update(address, size, true);
    }
    void Remove(u64 address, u64 size) {
        Update(address, size, false);
    }

    [[nodiscard]] bool Any(u64 address, u64 size) const {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumPages || overflow.load(std::memory_order_relaxed)) {
            return true;
        }
        for (u64 page = first; page <= last; ++page) {
            if (counts[page].load(std::memory_order_acquire) != 0) {
                return true;
            }
        }
        return false;
    }

private:
    void Update(u64 address, u64 size, bool add) {
        if (size == 0) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumPages) {
            overflow.store(true, std::memory_order_relaxed);
            return;
        }
        for (u64 page = first; page <= last; ++page) {
            if (add) {
                counts[page].fetch_add(1, std::memory_order_release);
            } else {
                counts[page].fetch_sub(1, std::memory_order_release);
            }
        }
    }

    static constexpr u64 PageBits = 16;
    static constexpr u64 NumPages = u64{1} << (36 - PageBits); // 1 Mi counters, 4 MiB
    std::unique_ptr<std::atomic<u32>[]> counts;
    std::atomic<bool> overflow{};
};

} // namespace VideoCommon
