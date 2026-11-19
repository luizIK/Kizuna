// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "common/common_types.h"
#include "common/logging.h"

/// FRITCHII_REPORT_WAIT (helper CN, work\helper_cn_answer.md): the counters and the 10 s line of
/// RasterizerVulkan::FlushQueryResults, the wait a macro read of report memory makes for query
/// results the host GPU still owes (the front's "macro dma flush" site; with one follower it is
/// lent, so R waits and E is parked). At Kakariko it is 25 to 30 waits a second of ~1.8 ms with
/// the front, E and R all stopped.
///
/// The old wait queues an operation behind a new fence and waits for it: for every piece of GPU
/// work sent so far, including what was sent after the reports. The lean wait (EXP30's) ends
/// when the reports in the read's range are written (Nx2WaitAsyncReportsSettled). Every report is
/// issued and completed through those counters (QueryCacheBase::CounterReport: the run entry and
/// the fence operation both), so a settled range holds the words the old wait would leave there.
/// Verify (2) proves it where it runs: the lean wait, the words read, then the old wait, the words
/// read again; any difference is counted (must be 0) and the first 16 are logged.
///
/// Every value times its waits (two clock reads, ~30 a second), so the line reads the same in a
/// 0 window and a 1 window. One writer at a time (the thread that owns the query cache: R when
/// lent, else the back), so relaxed atomics only keep the reader (the line) sane.
namespace VideoCommon::ReportWait {

struct Stats {
    std::atomic<u64> waits{};      ///< FlushQueryResults calls that found pending reports
    std::atomic<u64> wait_ns{};    ///< their wait, as the mode in force made it
    std::atomic<u64> longest_ns{}; ///< the longest of them in the window
    std::atomic<u64> lean{};       ///< of them, lean waits (1, EXP30, or 2's first half)
    // Verify (2): both waits of the same read.
    std::atomic<u64> compared{};      ///< reads compared
    std::atomic<u64> lean_ns{};       ///< the lean wait's part
    std::atomic<u64> full_ns{};       ///< start to the end of the old wait (lean + rest)
    std::atomic<u64> differ{};        ///< reads whose words changed in the old wait's rest
    std::atomic<u64> differ_total{};  ///< since boot
    std::atomic<s64> window_start_ns{};
    std::atomic<int> last_mode{-1};
};
inline Stats g_stats;

[[nodiscard]] inline u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void Add(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

/// One wait: its time and whether it was lean.
inline void NoteWait(u64 ns, bool lean) noexcept {
    Add(g_stats.waits);
    Add(g_stats.wait_ns, ns);
    if (lean) {
        Add(g_stats.lean);
    }
    if (ns > g_stats.longest_ns.load(std::memory_order_relaxed)) {
        g_stats.longest_ns.store(ns, std::memory_order_relaxed);
    }
}

/// Verify: one read's two waits and whether its words changed after the lean one. True for the
/// first 16 differences since boot (log those).
[[nodiscard]] inline bool NoteVerify(u64 lean_ns, u64 full_ns, bool differs) noexcept {
    Add(g_stats.compared);
    Add(g_stats.lean_ns, lean_ns);
    Add(g_stats.full_ns, full_ns);
    if (!differs) {
        return false;
    }
    Add(g_stats.differ);
    Add(g_stats.differ_total);
    return g_stats.differ_total.load(std::memory_order_relaxed) <= 16;
}

/// The line, at most every 10 s, from a wait (no waits: no line, nothing to say).
inline void MaybeReport(int mode) {
    const u64 now = NowNs();
    s64 start = g_stats.window_start_ns.load(std::memory_order_relaxed);
    if (start == 0) {
        g_stats.window_start_ns.store(static_cast<s64>(now), std::memory_order_relaxed);
        g_stats.last_mode.store(mode, std::memory_order_relaxed);
        return;
    }
    const u64 elapsed = now - static_cast<u64>(start);
    // A switch flip starts a new window (profile_toggle's windows then hold one value each).
    const bool flipped = g_stats.last_mode.load(std::memory_order_relaxed) != mode;
    if (elapsed < 10'000'000'000ULL && !flipped) {
        return;
    }
    const double seconds = static_cast<double>(elapsed) / 1e9;
    const u64 waits = g_stats.waits.exchange(0, std::memory_order_relaxed);
    const u64 wait_ns = g_stats.wait_ns.exchange(0, std::memory_order_relaxed);
    const u64 longest = g_stats.longest_ns.exchange(0, std::memory_order_relaxed);
    const u64 lean = g_stats.lean.exchange(0, std::memory_order_relaxed);
    const u64 compared = g_stats.compared.exchange(0, std::memory_order_relaxed);
    const u64 lean_ns = g_stats.lean_ns.exchange(0, std::memory_order_relaxed);
    const u64 full_ns = g_stats.full_ns.exchange(0, std::memory_order_relaxed);
    const u64 differ = g_stats.differ.exchange(0, std::memory_order_relaxed);
    const int window_mode = g_stats.last_mode.exchange(mode, std::memory_order_relaxed);
    g_stats.window_start_ns.store(static_cast<s64>(now), std::memory_order_relaxed);
    if (waits == 0 || seconds <= 0.0) {
        return;
    }
    const auto per_s = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto ms_s = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    LOG_INFO(HW_GPU,
             "Report waits (FRITCHII_REPORT_WAIT={}) {:.1f} s: {:.1f} waits/s, {:.1f} ms/s "
             "({:.0f} us each, longest {:.2f} ms), {:.1f}/s lean; verify: {} compared, lean "
             "{:.1f} ms/s, old wait {:.1f} ms/s ({:.0f} vs {:.0f} us each), words changed in the "
             "old wait's rest {} (must be 0; {} since boot)",
             window_mode, seconds, per_s(waits), ms_s(wait_ns),
             static_cast<double>(wait_ns) / 1e3 / static_cast<double>(waits),
             static_cast<double>(longest) / 1e6, per_s(lean), compared, ms_s(lean_ns),
             ms_s(full_ns),
             compared != 0 ? static_cast<double>(lean_ns) / 1e3 / static_cast<double>(compared)
                           : 0.0,
             compared != 0 ? static_cast<double>(full_ns) / 1e3 / static_cast<double>(compared)
                           : 0.0,
             differ, g_stats.differ_total.load(std::memory_order_relaxed));
}

} // namespace VideoCommon::ReportWait
