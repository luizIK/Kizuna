// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

/// FRITCHII_OCCLUSION_VISIBLE (live, default 0; docs/fritchii/switches.md). OoT tests objects
/// for visibility with two 64-bit sample count reports, before and after the object's test
/// geometry, and its 0xe2c macro reads both (six words: count, timestamp, count) and draws the
/// object when they differ (or, with p0 = 1, when they are equal). With the switch on:
/// - the GPU front does not read the words: the macro gets six words that say "differ" (1) or
///   "equal" (2), so it never waits for the back;
/// - the back writes every sample count report at once with a made-up value instead of running
///   a host occlusion query (no queries, no result copies, no fence work), skips the sample
///   counter resets, and the render thread keeps the host sample counter stopped.
/// 1: everything is visible. The made-up count grows by 4096 at every report, so any two reports
///    differ, whoever reads them (the front, the back's macro path, a CPU read).
/// 2: every report reads the same 4096. Two reports then read as equal, which OoT's 0xe2c takes
///    as "no samples passed": objects drawn on p0 = 0 are skipped. For comparison only: it is
///    what a constant made-up count does.
namespace VideoCommon::FritchiiOcclusion {

[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_occlusion_visible,
                                            "FRITCHII_OCCLUSION_VISIBLE");
}

inline constexpr u32 SamplesPerReport = 4096;

struct Counters {
    alignas(64) std::atomic<u64> reports{};     ///< Back: sample reports written at once
    alignas(64) std::atomic<u64> resets{};      ///< Back: sample counter resets skipped
    alignas(64) std::atomic<u64> front_reads{}; ///< Front: 0xe2c reads not made
    alignas(64) std::atomic<u64> ticks{};       ///< Calls since the last time check
    std::atomic<s64> last_log{};                ///< steady_clock nanoseconds, 0 = not started
};
inline Counters g_counters;
inline std::atomic<u32> g_count{};

/// The value a sample count report gets with the switch on (`mode` 1 or 2). Never 0.
[[nodiscard]] inline u32 ReportValue(int mode) {
    if (mode != 1) {
        return SamplesPerReport;
    }
    u32 value = g_count.fetch_add(SamplesPerReport, std::memory_order_relaxed) + SamplesPerReport;
    if (value == 0) [[unlikely]] {
        // Wrapped: a zero would read as "nothing drawn yet" to a reader after a reset.
        value = g_count.fetch_add(SamplesPerReport, std::memory_order_relaxed) + SamplesPerReport;
    }
    return value;
}

/// The 10 s line, from whichever thread counts past the time first.
inline void MaybeLog() {
    auto& c = g_counters;
    if ((c.ticks.fetch_add(1, std::memory_order_relaxed) & 1023) != 0) [[likely]] {
        return;
    }
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = c.last_log.load(std::memory_order_relaxed);
    if (last == 0) {
        c.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < s64{10'000'000'000} ||
        !c.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const double seconds = static_cast<double>(now - last) / 1e9;
    const auto rate = [seconds](std::atomic<u64>& counter) {
        return static_cast<double>(counter.exchange(0, std::memory_order_relaxed)) / seconds;
    };
    const double reports = rate(c.reports);
    const double resets = rate(c.resets);
    const double front_reads = rate(c.front_reads);
    LOG_INFO(HW_GPU,
             "Occlusion (FRITCHII_OCCLUSION_VISIBLE={}) last {:.0f} s: {:.0f} sample reports/s "
             "answered without a host query, {:.0f} counter resets/s skipped, {:.0f} 0xe2c "
             "reads/s skipped on the front",
             Mode(), seconds, reports, resets, front_reads);
}

inline void Count(std::atomic<u64>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
    MaybeLog();
}

} // namespace VideoCommon::FritchiiOcclusion
