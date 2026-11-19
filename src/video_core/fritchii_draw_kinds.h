// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_profile.h"

// FRITCHII_DRAW_KINDS=1 (diagnostic, read once when the rasterizer is made; fable52_answer_potroom2
// sections 3 and 4b): the drawing thread's draws and the 3D register writes before them, split by
// kind, one line per 600 frames. A draw can be of several kinds:
//   shadow: a depth-only pass (no colour target) whose depth target is at most 1,920 a side,
//   test:   sample counting on (ZPass counting enabled) and every colour write off,
//   report: a ZPass report (type 2 or 21) comes before the next draw or clear,
//   other:  none of the three.
// Each draw takes the 3D register writes made on its thread since the previous draw or clear (its
// setup; the report after a draw goes to the next one). Clears are counted apart, with their writes.
// With EDEN_NX2_GPU_PROFILE on, the line also has each kind's busy time a draw: inside the draw and
// between the previous draw or clear and this one, the profiler's wait zones taken out.
// Each kind also counts its draws whose pipeline samples nothing (no texture, image, texture
// buffer or image buffer descriptor: the delta path's first cut, fable52_answer_potroom3
// section 2) and, while EDEN_NX2_FASTDRAW_COUNT counts, its would-be delta draws (of them, those
// sampling nothing).
// Off: one test of g_on at each site.
namespace VideoCommon::FritchiiDrawKinds {

/// FRITCHII_DRAW_KINDS == "1". A plain bool: every site tests it and nothing else when off.
inline bool g_on{};
/// Frames presented (RasterizerVulkan::TickFrame); the drawing thread logs every 600.
inline std::atomic<u64> g_frames{};

inline constexpr u32 FramesPerLine = 600;
inline constexpr u32 MaxShadowSide = 1920;

enum Bits : u8 {
    Shadow = 1U << 0,
    Test = 1U << 1,
    ReportAfter = 1U << 2,
};
inline constexpr size_t NumMasks = 8;

/// Reads the switch once (the rasterizer's constructor: after main() applied the switch file).
inline void Init() {
    static const bool once = [] {
        const char* const value = std::getenv("FRITCHII_DRAW_KINDS");
        g_on = value != nullptr && value[0] == '1';
        if (g_on) {
            LOG_INFO(HW_GPU, "Draw kinds (FRITCHII_DRAW_KINDS): counting, a line per {} frames",
                     FramesPerLine);
        }
        return true;
    }();
    (void)once;
}

struct Bucket {
    u64 draws{};
    u64 writes{};
    u64 draw_ticks{}; ///< Busy inside the draw (profiled runs)
    u64 gap_ticks{};  ///< Busy since the previous draw or clear ended (profiled runs)
    u64 square{};     ///< Of the draws: a square depth target (shadow draws only)
    u64 no_samples{}; ///< Of the draws: the pipeline samples nothing
    u64 would_delta{};            ///< Of the draws: would-be delta draws (the count's decision)
    u64 would_delta_no_samples{}; ///< Of those: the pipeline samples nothing
};

struct ThreadState {
    std::array<Bucket, NumMasks> buckets{};
    Bucket clears{};
    u64 writes_since{}; ///< 3D register writes since the previous draw or clear began
    // The draw being made or made last: committed when the next draw or clear begins, so a
    // report after it can still mark it.
    bool pending{};
    bool pending_clear{};
    bool pending_square{};
    bool pending_no_samples{};
    bool pending_would_delta{};
    u8 pending_mask{};
    u64 pending_writes{};
    u64 pending_draw_ticks{};
    u64 pending_gap_ticks{};
    // Profiled runs: where the previous draw or clear ended, and the profiler's wait time then.
    u64 last_end_ticks{};
    u64 last_end_waits{};
    u64 begin_ticks{};
    u64 begin_waits{};
    // The window.
    bool started{};
    u64 window_frame{};
    u64 window_ns{};
    u64 window_ticks{};
};

inline thread_local ThreadState t_state;

/// The 3D register writes made (Maxwell3D, on the thread that makes them).
inline void NoteWrites(u64 count) {
    t_state.writes_since += count;
}

/// A ZPass report (Maxwell3D::ProcessQueryGet): it marks the draw before it.
inline void NoteZPassReport() {
    auto& t = t_state;
    if (t.pending && !t.pending_clear) {
        t.pending_mask |= ReportAfter;
    }
}

/// The profiler's wait zones on this thread (ticks), so busy time leaves them out.
[[nodiscard]] inline u64 WaitTicks() {
    using Nx2Profile::Zone;
    const auto& data = Nx2Profile::t_data;
    u64 total{};
    for (const Zone zone : {Zone::Idle, Zone::PendingSleep, Zone::ResumePending, Zone::SplitDrain,
                            Zone::SplitWait, Zone::WriterWait, Zone::ListSubmitWait,
                            Zone::ListGpuWait, Zone::QueryDmaWait, Zone::MemoryHandOff}) {
        total += data.exclusive_ns[static_cast<size_t>(zone)];
    }
    return total;
}

/// Ticks between two marks less the waits between them. The profiler starts its sums again
/// every 10 s: a sum below its mark means it started again in between (the gap is then taken
/// as all busy, less the waits since).
[[nodiscard]] inline u64 BusyTicks(u64 from_ticks, u64 to_ticks, u64 from_waits, u64 to_waits) {
    const u64 span = to_ticks > from_ticks ? to_ticks - from_ticks : 0;
    const u64 waits = to_waits >= from_waits ? to_waits - from_waits : to_waits;
    return span > waits ? span - waits : 0;
}

inline void Commit() {
    auto& t = t_state;
    if (!t.pending) {
        return;
    }
    t.pending = false;
    Bucket& bucket = t.pending_clear ? t.clears : t.buckets[t.pending_mask];
    ++bucket.draws;
    bucket.writes += t.pending_writes;
    bucket.draw_ticks += t.pending_draw_ticks;
    bucket.gap_ticks += t.pending_gap_ticks;
    bucket.square += t.pending_square ? 1 : 0;
    bucket.no_samples += t.pending_no_samples ? 1 : 0;
    bucket.would_delta += t.pending_would_delta ? 1 : 0;
    bucket.would_delta_no_samples += t.pending_would_delta && t.pending_no_samples ? 1 : 0;
}

/// The draw's pipeline (RasterizerVulkan::PrepareDraw, once it has one): whether it samples
/// nothing.
inline void NotePipeline(bool samples_nothing) {
    auto& t = t_state;
    if (t.pending && !t.pending_clear) {
        t.pending_no_samples = samples_nothing;
    }
}

/// The delta draw count (EDEN_NX2_FASTDRAW_COUNT) found the draw a would-be delta draw.
inline void NoteWouldBeDelta() {
    auto& t = t_state;
    if (t.pending && !t.pending_clear) {
        t.pending_would_delta = true;
    }
}

/// What a draw made with these registers is (all but ReportAfter, known later).
[[nodiscard]] inline u8 Classify(const Tegra::Engines::Maxwell3D::Regs& regs, bool& square) {
    constexpr u32 NumColour = Tegra::Engines::Maxwell3D::Regs::NumRenderTargets;
    const u32 colour_count = (std::min)(static_cast<u32>(regs.rt_control.count), NumColour);
    bool any_colour = false;
    bool colour_writes = false;
    for (u32 index = 0; index < colour_count; ++index) {
        const auto& rt = regs.rt[index];
        if (rt.Address() == 0 || rt.format == Tegra::RenderTargetFormat::NONE) {
            continue;
        }
        any_colour = true;
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        if ((mask.raw & 0x1111) != 0) {
            colour_writes = true;
            break;
        }
    }
    u8 mask = 0;
    square = false;
    if (!any_colour && regs.zeta_enable != 0 && regs.zeta.Address() != 0) {
        const u32 width = regs.zeta_size.width;
        const u32 height = regs.zeta_size.height;
        if (width != 0 && height != 0 && width <= MaxShadowSide && height <= MaxShadowSide) {
            mask |= Shadow;
            square = width == height;
        }
    }
    if (regs.zpass_pixel_count_enable != 0 && !colour_writes) {
        mask |= Test;
    }
    return mask;
}

inline void LogAndReset(ThreadState& t, u64 frames);

inline void Begin(const Tegra::Engines::Maxwell3D::Regs& regs, bool is_clear) {
    auto& t = t_state;
    Commit();
    const u64 frames = g_frames.load(std::memory_order_relaxed);
    const bool profile = Nx2Profile::Enabled();
    if (!t.started) {
        t = ThreadState{.started = true,
                        .window_frame = frames,
                        .window_ns = Nx2Profile::Now(),
                        .window_ticks = Nx2Profile::Ticks()};
    } else if (frames - t.window_frame >= FramesPerLine) {
        LogAndReset(t, frames);
    }
    t.pending = true;
    t.pending_clear = is_clear;
    t.pending_square = false;
    t.pending_no_samples = false;
    t.pending_would_delta = false;
    t.pending_mask = is_clear ? 0 : Classify(regs, t.pending_square);
    t.pending_writes = t.writes_since;
    t.writes_since = 0;
    t.pending_draw_ticks = 0;
    t.pending_gap_ticks = 0;
    if (profile) {
        t.begin_ticks = Nx2Profile::Ticks();
        t.begin_waits = WaitTicks();
        if (t.last_end_ticks != 0) {
            t.pending_gap_ticks =
                BusyTicks(t.last_end_ticks, t.begin_ticks, t.last_end_waits, t.begin_waits);
        }
    } else {
        t.begin_ticks = 0;
    }
}

inline void End() {
    auto& t = t_state;
    if (t.begin_ticks == 0) {
        t.last_end_ticks = 0;
        return;
    }
    const u64 now = Nx2Profile::Ticks();
    const u64 waits = WaitTicks();
    t.pending_draw_ticks = BusyTicks(t.begin_ticks, now, t.begin_waits, waits);
    t.last_end_ticks = now;
    t.last_end_waits = waits;
}

/// Around RasterizerVulkan::Draw, DrawIndirect and Clear.
class Scope {
public:
    explicit Scope(const Tegra::Engines::Maxwell3D::Regs& regs, bool is_clear = false) {
        if (g_on) [[unlikely]] {
            active = true;
            Begin(regs, is_clear);
        }
    }
    ~Scope() {
        if (active) [[unlikely]] {
            End();
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    bool active{};
};

inline void LogAndReset(ThreadState& t, u64 frames) {
    const u64 now_ns = Nx2Profile::Now();
    const u64 now_ticks = Nx2Profile::Ticks();
    const double frame_count = static_cast<double>((std::max<u64>)(frames - t.window_frame, 1));
    const double us_per_tick =
        now_ticks > t.window_ticks
            ? static_cast<double>(now_ns - t.window_ns) / 1000.0 /
                  static_cast<double>(now_ticks - t.window_ticks)
            : 0.0;
    Bucket all{};
    std::array<Bucket, 4> kinds{}; // shadow, test, report, other (inclusive)
    for (size_t mask = 0; mask < NumMasks; ++mask) {
        const Bucket& b = t.buckets[mask];
        const auto add = [&b](Bucket& into) {
            into.draws += b.draws;
            into.writes += b.writes;
            into.draw_ticks += b.draw_ticks;
            into.gap_ticks += b.gap_ticks;
            into.square += b.square;
            into.no_samples += b.no_samples;
            into.would_delta += b.would_delta;
            into.would_delta_no_samples += b.would_delta_no_samples;
        };
        add(all);
        if (mask == 0) {
            add(kinds[3]);
        }
        for (size_t bit = 0; bit < 3; ++bit) {
            if ((mask & (size_t{1} << bit)) != 0) {
                add(kinds[bit]);
            }
        }
    }
    const bool timed = us_per_tick > 0.0 && (all.draw_ticks != 0 || all.gap_ticks != 0);
    // Would-be delta draws exist only while the delta draw count runs.
    const bool counted = VideoCommon::Nx2FastDrawCount::g_count;
    const auto describe = [&](const Bucket& b) {
        const double draws = static_cast<double>(b.draws);
        std::string text = fmt::format("{:.1f} draws {:.1f} writes", draws / frame_count,
                                       static_cast<double>(b.writes) / frame_count);
        if (timed && b.draws != 0) {
            text += fmt::format(" {:.2f}+{:.2f} us",
                                static_cast<double>(b.draw_ticks) * us_per_tick / draws,
                                static_cast<double>(b.gap_ticks) * us_per_tick / draws);
        }
        if (b.draws != 0) {
            text += fmt::format(" ({:.1f} sample nothing",
                                static_cast<double>(b.no_samples) / frame_count);
            if (counted) {
                text += fmt::format(", {:.1f} would-be delta, {:.1f} of them sample nothing",
                                    static_cast<double>(b.would_delta) / frame_count,
                                    static_cast<double>(b.would_delta_no_samples) / frame_count);
            }
            text += ')';
        }
        return text;
    };
    static constexpr std::array<const char*, NumMasks> mask_names{
        "other",         "shadow",       "test",         "shadow+test",
        "report",        "shadow+report", "test+report", "shadow+test+report",
    };
    std::string exclusive;
    for (size_t mask = 0; mask < NumMasks; ++mask) {
        if (t.buckets[mask].draws == 0) {
            continue;
        }
        exclusive += fmt::format("{}{} {}", exclusive.empty() ? "" : "; ", mask_names[mask],
                                 describe(t.buckets[mask]));
    }
    LOG_INFO(HW_GPU,
             "Draw kinds (FRITCHII_DRAW_KINDS) per frame over {:.0f} frames{}: all {} | shadow "
             "(depth-only, at most {} a side) {}, {:.1f} square | test (counting on, colour off) "
             "{} | report after {} | other {} | clears {} | exclusive: {}",
             frame_count, timed ? " (busy us a draw: in the draw + before it)" : "",
             describe(all), MaxShadowSide, describe(kinds[0]),
             static_cast<double>(kinds[0].square) / frame_count, describe(kinds[1]),
             describe(kinds[2]), describe(kinds[3]), describe(t.clears),
             exclusive.empty() ? "none" : exclusive);
    for (Bucket& b : t.buckets) {
        b = {};
    }
    t.clears = {};
    t.window_frame = frames;
    t.window_ns = now_ns;
    t.window_ticks = now_ticks;
}

} // namespace VideoCommon::FritchiiDrawKinds
