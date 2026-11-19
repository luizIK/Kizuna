// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

// EXP119, report runs: the report side of the occlusion tests (fable52_answer_potroom3 sections 2
// and 4, step 2). Docs: docs/fritchii/switches.md, EXP119.
//
// Today every sample count report (the one after each occlusion test draw) queues its own
// operation (a std::function in the fence manager's list) and every counter reset queues one
// more. With the switch on, a sample count report that is not a fence, and the counter reset's
// amend reset, become entries of a run: one operation in the list that writes all of them in
// one loop when the fence's results are in. An entry is only added to a run while that run is
// still the last operation queued and no fence took the list since (the fence manager's stamp),
// so every entry runs exactly where its own operation would have, between the same neighbours.
// The report is still listed in the query cache's map (the split's front, guest reads and
// render enable look there), with one hash fewer; its slot is unlisted and freed as before.
// Render enable: the Conditional mode skips the map lookup (its answer was never used), the
// IfEqual / IfNotEqual modes do their two lookups under one lock.
//
// Values: 0 = off; 1 = on; 2 = verify: today's operations write and stay in charge, the runs only
// compute (value, address, outcome) and every report's own operation compares; render enable
// does both forms and compares. Mismatches must be 0.
namespace VideoCommon::FritchiiReportRuns {

/// The switch: EDEN_NX2_EXP119.
inline constexpr int Switch = 119;
inline constexpr u64 FramesPerLine = 600;

/// Frames presented (RasterizerVulkan::TickFrame, while the switch is on).
inline std::atomic<u64> g_frames{};

/// Why a report took today's path with the switch on.
enum class Fallback : u8 {
    Fence,     ///< Fence-flagged (a short release): its operation goes with the fence
    OtherType, ///< Not a sample count report, or no streamer for it
    Traced,    ///< EDEN_NX2_QUERY_TRACE or EDEN_NX2_SEM_WATCH on its address
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> fallback_names{
    "fence",
    "other type",
    "traced",
};

/// Counted by the fencing thread (the runs and today's operations), read at each line.
struct RunCounters {
    std::atomic<u64> run_ops{};       ///< Runs that ran
    std::atomic<u64> run_entries{};   ///< Entries they ran
    std::atomic<u64> compared{};      ///< EXP119=2: reports compared with their own operation
    std::atomic<u64> mismatches{};    ///< ...that differed (value, address, outcome or order)
    std::atomic<u64> dropped_runs{};  ///< Runs dropped without running (emulation stopped)
};
inline RunCounters g_run_counters{};

/// Counts on the thread that issues the reports (the split's back), logged every
/// FramesPerLine frames while the switch is on.
struct Stats {
    u64 reports{};        ///< Reports seen with the switch on
    u64 in_runs{};        ///< Sample count reports added to a run (1: written by it; 2: computed)
    u64 runs{};           ///< Runs started
    u64 resets{};         ///< Counter resets with the switch on
    u64 resets_in_runs{}; ///< ...whose amend reset went into a run
    u64 unmapped{};       ///< Reports to an unmapped address (dropped, as today)
    std::array<u64, static_cast<size_t>(Fallback::Count)> fallbacks{};
    u64 enable_calls{};      ///< Render enable calls with the switch on
    u64 enable_no_lookup{};  ///< Conditional mode: no lookup
    u64 enable_one_lock{};   ///< IfEqual / IfNotEqual: both lookups under one lock
    u64 enable_compared{};   ///< EXP119=2: render enable calls done both ways
    u64 enable_mismatches{}; ///< ...whose lookups differed (must stay 0)
    u64 window_frame{};
    int mode{};
    bool started{};

    void Fall(Fallback reason) {
        ++fallbacks[static_cast<size_t>(reason)];
    }

    /// Logs and starts a new window every FramesPerLine frames; a switch change starts one too.
    void MaybeLog(int mode_) {
        const u64 frames = g_frames.load(std::memory_order_relaxed);
        if (!started || mode != mode_) {
            *this = Stats{.window_frame = frames, .mode = mode_, .started = true};
            TakeRunCounters();
            return;
        }
        if (frames - window_frame < FramesPerLine) {
            return;
        }
        const RunTotals totals = TakeRunCounters();
        const double frame_count = static_cast<double>(frames - window_frame);
        const auto per_frame = [frame_count](u64 count) {
            return static_cast<double>(count) / frame_count;
        };
        std::string falls;
        for (size_t reason = 0; reason < fallbacks.size(); ++reason) {
            if (fallbacks[reason] != 0) {
                falls += fmt::format("{}{} {:.1f}", falls.empty() ? "" : ", ",
                                     fallback_names[reason], per_frame(fallbacks[reason]));
            }
        }
        std::string verify;
        if (mode == 2) {
            verify = fmt::format(" | verify: {:.1f} reports compared, {} mismatches; {:.1f} "
                                 "render enables compared, {} mismatches",
                                 per_frame(totals.compared), totals.mismatches,
                                 per_frame(enable_compared), enable_mismatches);
        }
        LOG_INFO(HW_GPU,
                 "Report runs (EXP119={}) per frame over {:.0f} frames: {:.1f} reports, {:.1f} "
                 "in runs ({:.1f} runs started, {:.1f} ran with {:.1f} entries), {:.1f} counter "
                 "resets ({:.1f} in runs), {:.1f} unmapped | today's path: {} | render enable: "
                 "{:.1f} calls, {:.1f} without lookup, {:.1f} with one lock{}{}",
                 mode, frame_count, per_frame(reports), per_frame(in_runs), per_frame(runs),
                 per_frame(totals.run_ops), per_frame(totals.run_entries), per_frame(resets),
                 per_frame(resets_in_runs), per_frame(unmapped),
                 falls.empty() ? std::string{"none"} : falls, per_frame(enable_calls),
                 per_frame(enable_no_lookup), per_frame(enable_one_lock), verify,
                 totals.dropped_runs != 0
                     ? fmt::format(" | {} runs dropped unrun", totals.dropped_runs)
                     : std::string{});
        *this = Stats{.window_frame = frames, .mode = mode, .started = true};
    }

private:
    struct RunTotals {
        u64 run_ops;
        u64 run_entries;
        u64 compared;
        u64 mismatches;
        u64 dropped_runs;
    };

    static RunTotals TakeRunCounters() {
        RunCounters& counters = g_run_counters;
        return RunTotals{
            .run_ops = counters.run_ops.exchange(0, std::memory_order_relaxed),
            .run_entries = counters.run_entries.exchange(0, std::memory_order_relaxed),
            .compared = counters.compared.exchange(0, std::memory_order_relaxed),
            .mismatches = counters.mismatches.exchange(0, std::memory_order_relaxed),
            .dropped_runs = counters.dropped_runs.exchange(0, std::memory_order_relaxed),
        };
    }
};

/// EXP119=2: the first mismatches are logged one by one, at most this many a session.
inline constexpr u64 MismatchLogCap = 16;
inline std::atomic<u64> g_mismatches_logged{};

} // namespace VideoCommon::FritchiiReportRuns
