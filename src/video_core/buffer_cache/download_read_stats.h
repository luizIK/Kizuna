// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_DOWNLOAD_READ_STATS=1 (diagnostic, read once; fable52_answer_imagereads 7, F2 of
// fable52_answer_chicken 5): whether the buffer downloads the fences commit are ever read by the
// guest. PopAsyncBuffers sets a per-page "landed, not read yet" bit for the pages it writes; a
// page that lands while its bit is still set counts "landed again unread". A guest read that
// reaches the buffer cache (GetFlushArea, which also disarms the read traps, and EXP86's
// lock-free clean answer) clears the bits of its pages, one "read" per bit cleared.
// CommitAsyncFlushesHigh counts the downloads and their bytes. One log line per 600 frames.
// Off: one test of g_enabled at each site.
namespace VideoCommon::DownloadReadStats {

inline const bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_DOWNLOAD_READ_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

/// One bit per 4 KiB device page over the 36-bit device space (2 MiB, allocated only when on).
/// Relaxed atomics: the fencing thread sets bits while guest threads clear them.
class LandedPages {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 NumPages = u64{1} << (36 - PageBits);
    static constexpr u64 NumWords = NumPages / 64;

    LandedPages()
        : words{g_enabled ? std::make_unique<std::atomic<u64>[]>(NumWords) : nullptr} {}

    /// Sets the bits of pages [first, last]; returns how many were already set.
    u64 Set(u64 first, u64 last) {
        u64 again = 0;
        ForEachWord(first, last, [&](std::atomic<u64>& word, u64 mask) {
            const u64 old = word.fetch_or(mask, std::memory_order_relaxed);
            again += static_cast<u64>(std::popcount(old & mask));
        });
        return again;
    }

    /// Clears the bits of pages [first, last]; returns how many were set.
    u64 Clear(u64 first, u64 last) {
        u64 cleared = 0;
        ForEachWord(first, last, [&](std::atomic<u64>& word, u64 mask) {
            if ((word.load(std::memory_order_relaxed) & mask) == 0) {
                return;
            }
            const u64 old = word.fetch_and(~mask, std::memory_order_relaxed);
            cleared += static_cast<u64>(std::popcount(old & mask));
        });
        return cleared;
    }

private:
    template <typename Func>
    void ForEachWord(u64 first, u64 last, Func&& func) {
        if (!words || first > last || last >= NumPages) {
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            func(words[page / 64], mask);
            page += count;
        }
    }

    std::unique_ptr<std::atomic<u64>[]> words;
};

struct State {
    LandedPages pages;
    std::atomic<u64> fences{};          ///< CommitAsyncFlushesHigh calls
    std::atomic<u64> fences_download{}; ///< of those, with downloads to commit
    std::atomic<u64> downloads{};       ///< copies committed
    std::atomic<u64> bytes{};           ///< bytes committed
    std::atomic<u64> landed{};          ///< pages landed in guest memory
    std::atomic<u64> landed_again{};    ///< pages landed while still unread since their last landing
    std::atomic<u64> read{};            ///< landed pages read by the guest (bits cleared)
};

inline State& Get() {
    static State state;
    return state;
}

/// CommitAsyncFlushesHigh: one fence's commit (copies == 0: nothing to download).
inline void Committed(u64 copies, u64 bytes) {
    State& s = Get();
    s.fences.fetch_add(1, std::memory_order_relaxed);
    if (copies != 0) {
        s.fences_download.fetch_add(1, std::memory_order_relaxed);
        s.downloads.fetch_add(copies, std::memory_order_relaxed);
        s.bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

/// PopAsyncBuffers: the ranges one fence wrote to guest memory, [begin, end) each, in any order.
/// A page two of them touch counts once.
inline void Landed(std::vector<std::pair<u64, u64>>& ranges) {
    if (ranges.empty()) {
        return;
    }
    std::sort(ranges.begin(), ranges.end());
    State& s = Get();
    u64 landed = 0;
    u64 again = 0;
    u64 next_page = 0; // first page not handled yet
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) {
            continue;
        }
        const u64 first = (std::max)(begin >> LandedPages::PageBits, next_page);
        const u64 last = (end - 1) >> LandedPages::PageBits;
        if (first > last) {
            continue;
        }
        landed += last - first + 1;
        again += s.pages.Set(first, last);
        next_page = last + 1;
    }
    s.landed.fetch_add(landed, std::memory_order_relaxed);
    s.landed_again.fetch_add(again, std::memory_order_relaxed);
}

/// A guest read of [begin, end) reaching the buffer cache.
inline void Read(u64 begin, u64 end) {
    if (end <= begin) {
        return;
    }
    const u64 cleared =
        Get().pages.Clear(begin >> LandedPages::PageBits, (end - 1) >> LandedPages::PageBits);
    if (cleared != 0) {
        Get().read.fetch_add(cleared, std::memory_order_relaxed);
    }
}

/// TickFrame, every 600 frames.
inline void Log() {
    State& s = Get();
    const u64 fences = s.fences.exchange(0, std::memory_order_relaxed);
    const u64 fences_download = s.fences_download.exchange(0, std::memory_order_relaxed);
    const u64 downloads = s.downloads.exchange(0, std::memory_order_relaxed);
    const u64 bytes = s.bytes.exchange(0, std::memory_order_relaxed);
    const u64 landed = s.landed.exchange(0, std::memory_order_relaxed);
    const u64 again = s.landed_again.exchange(0, std::memory_order_relaxed);
    const u64 read = s.read.exchange(0, std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "Download reads (F2) last 600 frames: {} downloads committed ({} KiB) by {} of {} "
             "fences; {} pages landed, {} read, {} landed again unread ({:.1f}% of landed unread)",
             downloads, bytes >> 10, fences_download, fences, landed, read, again,
             landed != 0 ? 100.0 * static_cast<double>(again) / static_cast<double>(landed) : 0.0);
}

} // namespace VideoCommon::DownloadReadStats
