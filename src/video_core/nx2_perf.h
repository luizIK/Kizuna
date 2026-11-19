// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"

// EXPERIMENT (not for nx2): GPU back-thread optimizations for Ocarina of Time.
// split_overlap is quarantined. Frizti build: the options reviewed as exact rewrites are on by
// default; those that change timing or reuse ideas reverted with a ported change (its
// slab of wrong geometry in Kokiri Forest) are off until EDEN_NX2_PERF_ON=<name>,... turns them
// on: frames_ahead, clean_words, index_rebind, fence_flush, fragment_barrier (submit_budget is
// on since 2026-11-14),
// front_sleep, async_pipelines, and bounds (this tree keeps its own range cache).
// EDEN_NX2_PERF_OPT=0 turns every one off for the A/B; EDEN_NX2_PERF_OFF=<name>,... turns off
// only those named (Opt below). EDEN_NX2_PERF_VERIFY=1 computes each optimized answer the old
// way too, logs any difference and uses the old answer.
namespace VideoCommon::Nx2Perf {

enum class Opt : unsigned {
    Bounds,            ///< MaxContinuousRange / GetMemoryLayoutSize from map-time runs
    Method,            ///< Maxwell3D fast method dispatch on the back
    CbBatch,           ///< Consecutive constant buffer words uploaded in one write
    CleanWords,        ///< SynchronizeBuffer skips clean tracker words
    IndexRebind,       ///< Repeated vkCmdBindIndexBuffer skipped
    UsageTracker,      ///< UsageTracker::Track skips pages known to be used
    BankCheck,         ///< Native 64 KiB banks: per-frame set checked first
    SplitOverlap,      ///< Quarantined: pending writes are not a proof of safe early MME reads
    FrontSleep,        ///< The split front sleeps instead of yielding while it waits
    SubmitBudget,      ///< 4096-draw native submission budget outside MoltenVK
    FragmentBarrier,   ///< FragmentBarrier deferred to the next draw that reads the pass
    FenceFlush,        ///< 5 ms DeferFlush, command list copies submitted at once
    FramesAhead,       ///< Ocarina presents a frame ahead
    PipelineRetention, ///< Sixteen sessions of used pipelines, saved every 15 s
    WriteIntersections,
    SortedPageBatches,
    AsyncPipelines,    ///< Native graphics SPIR-V emitted by the pipeline worker, not the back
    UnswizzleWorkers,
    Count,
};

inline constexpr std::array<std::string_view, static_cast<size_t>(Opt::Count)> opt_names{
    "bounds",         "method",          "cb_batch",    "clean_words",   "index_rebind",
    "usage_tracker",  "bank_check",      "split_overlap", "front_sleep", "submit_budget",
    "fragment_barrier", "fence_flush",   "frames_ahead", "pipeline_retention",
    "write_intersections", "sorted_page_batches", "async_pipelines", "unswizzle_workers",
};

namespace Detail {
/// Enabled()'s and On()'s answers, each computed at its first use as before, then kept in an
/// atomic word instead of a function's static: MSVC checks a static's initialization on every
/// call through a thread-local, which cost the GPU thread 3 to 6 ms/s (fable_b 21.2 cut 4).
/// 0 = not computed yet; Enabled: 1 off, 2 on; On: bit 31 set, the off mask below it.
inline std::atomic<unsigned> g_enabled_word{0};
inline std::atomic<unsigned> g_off_word{0};
inline constexpr unsigned OffWordReady = 1U << 31;
static_assert(static_cast<unsigned>(Opt::Count) < 31);
inline unsigned ComputeOffMask();
} // namespace Detail

inline bool Enabled() {
    unsigned word = Detail::g_enabled_word.load(std::memory_order_relaxed);
    if (word == 0) [[unlikely]] {
        const char* value = std::getenv("EDEN_NX2_PERF_OPT");
        unsigned expected = 0;
        Detail::g_enabled_word.compare_exchange_strong(
            expected, value == nullptr || value[0] != '0' ? 2U : 1U, std::memory_order_relaxed);
        word = Detail::g_enabled_word.load(std::memory_order_relaxed);
    }
    return word == 2;
}

inline bool On(Opt opt) {
    unsigned word = Detail::g_off_word.load(std::memory_order_relaxed);
    if (word == 0) [[unlikely]] {
        const unsigned mask = Detail::ComputeOffMask();
        unsigned expected = 0;
        Detail::g_off_word.compare_exchange_strong(expected, Detail::OffWordReady | mask,
                                                   std::memory_order_relaxed);
        word = Detail::g_off_word.load(std::memory_order_relaxed);
    }
    return Enabled() && (word >> static_cast<unsigned>(opt) & 1) == 0;
}

/// The off mask On() keeps (its first use only; it logs the state line).
inline unsigned Detail::ComputeOffMask() {
    static const unsigned off = [] {
        const auto bit = [](Opt option) { return 1U << static_cast<unsigned>(option); };
        const auto names = [](const char* variable) {
            unsigned bits = 0;
            std::string_view list = std::getenv(variable) ? std::getenv(variable) : "";
            while (!list.empty()) {
                const size_t comma = list.find(',');
                const std::string_view name = list.substr(0, comma);
                bool known = false;
                for (size_t i = 0; i < opt_names.size(); ++i) {
                    if (name == opt_names[i]) {
                        bits |= 1U << i;
                        known = true;
                    }
                }
                if (!known && !name.empty()) {
                    LOG_WARNING(HW_GPU, "Nx2Perf: unknown name {} in {}", name, variable);
                }
                list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            }
            return bits;
        };
        // FramesAhead is on: with Ocarina's CPU readbacks preflushed it is worth ~1 fps (48.3 ->
        // 49.5 on the Kokiri porch) and it waits for this frame too whenever the GPU falls behind.
        // SubmitBudget is on (2026-11-14, owner's call): fresh market A/B (2 off, 2 on, 90 s):
        // median 53.4 -> 55.1 fps, 1% low 39.8 -> 40.8; a submit boundary changes when the card
        // starts, not what it draws (fable_submits_brief 2). EDEN_NX2_PERF_OFF=submit_budget.
        const unsigned default_off = bit(Opt::CleanWords) |
                                     bit(Opt::IndexRebind) |
                                     bit(Opt::FenceFlush) | bit(Opt::FragmentBarrier) |
                                     bit(Opt::FrontSleep) | bit(Opt::AsyncPipelines) |
                                     bit(Opt::Bounds);
        unsigned mask = (default_off & ~names("EDEN_NX2_PERF_ON")) | names("EDEN_NX2_PERF_OFF");
        mask |= bit(Opt::SplitOverlap);
        if (const char* value = std::getenv("EDEN_NX2_PERF_FRAMES_AHEAD"); value && value[0] == '0') {
            mask |= bit(Opt::FramesAhead);
        }
        std::string state;
        for (size_t i = 0; i < opt_names.size(); ++i) {
            state += fmt::format(" {}={}", opt_names[i], Enabled() && (mask >> i & 1) == 0 ? "on" : "off");
        }
        LOG_INFO(HW_GPU, "Nx2Perf:{}", state);
        return mask;
    }();
    return off;
}

[[nodiscard]] inline bool VerifySwitchSet() {
    const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
    return value != nullptr && value[0] == '1';
}

/// EDEN_NX2_PERF_VERIFY=1. A plain flag: the buffer cache, the memory tracker and the 3D engine
/// check it on hot paths, and a function's static is checked for its initialization on every
/// call (MSVC reads a thread-local for it).
inline bool g_verify = VerifySwitchSet();

/// Called by VideoCore::CreateGPU: static initialization runs before the frontend loads
/// nx2_switches.txt, so a switch set only there takes effect from that point on.
inline void ReadSwitch() {
    g_verify = VerifySwitchSet();
}

inline bool Verify() {
    return g_verify;
}

/// The frame period the game asks for (nx2:vi2 present word 12 at the speed limit), 0 while
/// unknown or unpaced. The host frame pacer (vk_host_pacer.h) never holds frames longer.
inline std::atomic<long long> g_guest_frame_interval_ns{0};

/// The display's measured refresh period while the host frame pacer runs, else 0. The game's
/// present pacing (nx2_gsv.h) times its frames to whole multiples of it (genlock).
inline std::atomic<long long> g_host_refresh_ns{0};

/// The display's refresh period once the host frame pacer has measured it, paced or not, else
/// 0. Ocarina of Time's high refresh rate cap (nx2_gsv.h) stays at or below it.
inline std::atomic<long long> g_display_refresh_ns{0};

/// The swapchain's present mode as a Settings::VSyncMode value (0 Immediate, 1 Mailbox, 2 FIFO,
/// 3 FIFO relaxed; what the driver gave, not the setting), -1 before the first swapchain.
/// Ocarina of Time's high refresh rate cap only stays at or below the display rate with FIFO.
inline std::atomic<int> g_present_mode{-1};

/// True while the game presents as fast as it can on purpose (Ocarina of Time's high refresh
/// rate with no frame period). With Mailbox the host frame pacer then holds no frames: holding
/// one per refresh capped the game at the display rate.
inline std::atomic<bool> g_guest_unpaced{false};

/// Whether the Vulkan device runs the guest's tensor operations on native cooperative matrices
/// (set at device creation). Ocarina's movie upscaler setting picks by it on Automatic.
inline std::atomic<bool> g_native_tensor_ops{false};

/// At 30 FPS and below the host frame pacer presents frames as soon as they are rendered and
/// publishes how long before the display's next refresh the last one was ready (ns), or -1
/// while it isn't pacing that way. The game's present pacing (nx2_gsv.h) moves its frames until
/// that is ~6 ms: each lands on a refresh with little wait (input lag) and evenly spaced.
inline std::atomic<long long> g_host_phase_ns{-1};

/// Host frame times: the interval between frames handed to the swapchain, counted in 0.25 ms
/// buckets (the last holds 63.75 ms and up). The debug server's status returns the counts; the
/// difference between two calls gives that window's percentiles (1% and 0.1% lows).
inline constexpr size_t FrameTimeBuckets = 256;
inline constexpr long long FrameTimeBucketNs = 250'000;
inline std::array<std::atomic<unsigned long long>, FrameTimeBuckets> g_frame_times{};

inline void NoteFrameTime(long long interval_ns) {
    // Longer gaps are loading screens or a paused game, not frames.
    if (interval_ns <= 0 || interval_ns > 1'000'000'000) {
        return;
    }
    const auto bucket =
        (std::min)(static_cast<size_t>(interval_ns / FrameTimeBucketNs), FrameTimeBuckets - 1);
    g_frame_times[bucket].fetch_add(1, std::memory_order_relaxed);
}

} // namespace VideoCommon::Nx2Perf
