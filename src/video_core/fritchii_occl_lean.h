// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_debug_control.h"

/// FRITCHII_OCCL_LEAN (helper CS, work\helper_cs_answer.md; live, default 0).
///
/// Kakariko sends ~2,570 occlusion test draws a frame (counting on, every colour write off) of
/// ~3,420, and E (GPUBack) fills each one in full. The pixel program of a draw that writes no
/// colour can change the picture and the samples-passed count only through: a discard (demote),
/// a depth or sample mask write, a store (storage buffer, image, global memory, atomics), or its
/// colour 0 alpha when alpha to coverage or the alpha test (emulated in the pixel program) is on.
/// With none of them the draw's depth, stencil and counted samples are those of the same
/// pipeline without a pixel program (Vulkan: no fragment shader, the per-fragment tests and the
/// query count run on the rasterized coverage and the interpolated depth; colour writes are off).
///
/// So for such a draw E takes, after its own lookup (rule 3), the same key with the pixel program
/// hash 0 (a "sibling"): the key a draw with the pixel program disabled makes, built by the same
/// code. E's fill then has no pixel stage (its uniform buffers, storage binds and texture handle
/// values) and R binds no pixel textures. The sibling is translated on E from the draw's own
/// environments (at most one a frame), never stored in the disk cache, and used only once it is
/// built and its vertex side (uniform masks and sizes, storage, texture, image and texel
/// descriptors of stages 0 to 3, the vertex program) matches the full pipeline's; else the full
/// pipeline draws (refused for good when they differ).
///
/// Values: 0 off (one relaxed load a draw). 1 on. 2 verify: the siblings are made and their
/// vertex side compared, the full pipeline still draws (picture unchanged). 3 dry: the census
/// and the timer only. 1, 2 and 3 log the census line every 10 s and at a flip; the fill timer
/// (1 draw in 16 of the eligible ones, fill and push) runs at 1, 2 and 3, so 3/1 toggles compare
/// E's time a draw directly.
namespace VideoCommon::OcclLean {

enum Value : int { Off = 0, On = 1, Verify = 2, Dry = 3 };

/// Why an eligible draw's pipeline keeps its pixel program (counted per draw).
enum class PipelineClass : u8 {
    Lean,         ///< the pixel program can change nothing: the sibling may draw
    NoPixel,      ///< no pixel program already
    Discards,     ///< demote / discard (alpha-tested cut-outs)
    DepthOrMask,  ///< writes depth or the sample mask
    Stores,       ///< storage, image or global writes, atomics
    Mesh,         ///< task/mesh or emulated mesh pipeline
    KeyState,     ///< the key's alpha test is not Always, or dual source blending
    VertexDiffers,///< the sibling's vertex side differs from the full pipeline's (refused)
    Failed,       ///< the sibling did not translate or build
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(PipelineClass::Count)> ClassNames{
    "lean", "no pixel program", "discard", "depth or sample mask", "stores", "mesh",
    "alpha test or dual source key", "vertex side differs", "failed"};

[[nodiscard]] inline int Mode() noexcept {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_occl_lean,
                                                       "FRITCHII_OCCL_LEAN");
    return value >= 0 && value <= 3 ? value : 0;
}

/// E is the only writer; relaxed atomics keep the line's reads sane.
struct Stats {
    std::atomic<u64> draws{};       ///< E's looked-up draws seen with the switch on
    std::atomic<u64> counting{};    ///< sample counting on (zpass_pixel_count_enable)
    std::atomic<u64> colour_off{};  ///< no colour write (any bound target, any slot)
    std::atomic<u64> tests{};       ///< counting on and colour off: the occlusion test draws
    std::atomic<u64> regs_refused{};///< colour off, but alpha to coverage or the alpha test on
    std::array<std::atomic<u64>, static_cast<size_t>(PipelineClass::Count)> by_class{};
    // Of the Lean-class draws: what E's fill and R's configure skip with the sibling.
    std::atomic<u64> pixel_uniforms{};  ///< enabled pixel uniform buffers
    std::atomic<u64> pixel_textures{};  ///< pixel texture and image descriptors (elements)
    std::atomic<u64> pixel_storage{};   ///< pixel storage buffers
    std::atomic<u64> pixel_programs{};  ///< pixel handle programs
    std::atomic<u64> same_pipeline{};   ///< the same full pipeline as the previous Lean draw
    std::atomic<u64> lean{};            ///< drawn with the sibling (1)
    std::atomic<u64> not_ready{};       ///< sibling not made yet (budget, pending) or not built
    std::atomic<u64> ready_unused{};    ///< verify (2): the sibling was ready, the full one drew
    std::atomic<u64> made{};            ///< siblings translated
    std::atomic<u64> found{};           ///< siblings found in the cache already
    std::atomic<u64> failed{};          ///< siblings that did not translate or build
    std::atomic<u64> compared{};        ///< siblings whose vertex side was compared
    std::atomic<u64> differ{};          ///< of them, a vertex side that differs (refused)
    std::atomic<u64> made_ns{};         ///< E's time translating siblings
    // The timer: 1 in 16 Lean-class draws, E's fill and push.
    std::atomic<u64> timed_lean{};
    std::atomic<u64> timed_lean_ns{};
    std::atomic<u64> timed_full{};
    std::atomic<u64> timed_full_ns{};
    std::atomic<u64> window_start_ns{};
    std::atomic<int> last_mode{-1};
    u32 sample{};        ///< E: the timer's sample counter
    u32 clock_check{};   ///< E: draws since the line's clock was read
    u64 last_seen_ns{};  ///< E: the line's last clock read
    const void* last_lean_full{}; ///< E: the previous Lean draw's full pipeline
};
inline Stats g_stats;

inline void Add(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

[[nodiscard]] inline u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// The draw's register facts the cut depends on (E's engine at the draw; R's view equals it).
struct RegsVerdict {
    bool counting{};
    bool colour_off{};
    bool refused{}; ///< alpha to coverage or a live alpha test
};

/// Colour off: no colour target with a format takes a write, read both ways the code maps slots
/// (the key's attachment slot i with its mask i, and the render target rt_control.Map(i)); a
/// superset of both readings, so a draw passes only when nothing can be written.
[[nodiscard]] inline RegsVerdict ClassifyRegs(const Tegra::Engines::Maxwell3D::Regs& regs) noexcept {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    constexpr u32 Targets = Regs::NumRenderTargets;
    RegsVerdict verdict{};
    verdict.counting = regs.zpass_pixel_count_enable != 0;
    const auto mask_of = [&regs](u32 index) {
        return regs.color_mask[regs.color_mask_common ? 0 : index].raw & 0x1111U;
    };
    bool writes = false;
    for (u32 index = 0; index < Targets && !writes; ++index) {
        writes = regs.rt[index].format != Tegra::RenderTargetFormat::NONE && mask_of(index) != 0;
    }
    const u32 count = (std::min)(static_cast<u32>(regs.rt_control.count), Targets);
    for (u32 slot = 0; slot < count && !writes; ++slot) {
        const u32 target = regs.rt_control.Map(slot);
        writes = target < Targets && regs.rt[target].format != Tegra::RenderTargetFormat::NONE &&
                 mask_of(slot) != 0;
    }
    verdict.colour_off = !writes;
    const bool alpha_test =
        regs.alpha_test_enabled != 0 &&
        regs.alpha_test_func != Regs::ComparisonOp::Always_GL &&
        regs.alpha_test_func != Regs::ComparisonOp::Always_D3D;
    verdict.refused = regs.anti_alias_alpha_control.alpha_to_coverage != 0 || alpha_test;
    return verdict;
}

/// What the call site gets back: whether the draw is timed and drew lean.
struct DrawNote {
    bool timed{};
    bool lean{};
};

inline void NoteFill(const DrawNote& note, u64 ns) noexcept {
    if (note.lean) {
        Add(g_stats.timed_lean);
        Add(g_stats.timed_lean_ns, ns);
    } else {
        Add(g_stats.timed_full);
        Add(g_stats.timed_full_ns, ns);
    }
}

/// The line, every 10 s and at a flip (E, at a counted draw; the clock read every 64 draws).
inline void MaybeReport(int mode) {
    Stats& s = g_stats;
    const bool flipped = s.last_mode.load(std::memory_order_relaxed) != mode;
    if (!flipped && ++s.clock_check < 64) {
        return;
    }
    s.clock_check = 0;
    const u64 now = NowNs();
    // A switch at 0 calls nothing: half a second without a clock read (E draws ~170k a second)
    // means it was off, so the window ends at the last read (0/1 toggles keep one value a line).
    const u64 last_seen = std::exchange(s.last_seen_ns, now);
    const bool gap = last_seen != 0 && now - last_seen > 500'000'000ULL;
    const u64 start = s.window_start_ns.load(std::memory_order_relaxed);
    if (start == 0) {
        s.window_start_ns.store(now, std::memory_order_relaxed);
        s.last_mode.store(mode, std::memory_order_relaxed);
        return;
    }
    const u64 end = (flipped || gap) && last_seen > start ? last_seen : now;
    const u64 elapsed = end - start;
    if (elapsed < 10'000'000'000ULL && !flipped && !gap) {
        return;
    }
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const double seconds = static_cast<double>(elapsed) / 1e9;
    const auto per_s = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const u64 draws = take(s.draws);
    const u64 counting = take(s.counting);
    const u64 colour_off = take(s.colour_off);
    const u64 tests = take(s.tests);
    const u64 regs_refused = take(s.regs_refused);
    std::array<u64, static_cast<size_t>(PipelineClass::Count)> classes{};
    for (size_t i = 0; i < classes.size(); ++i) {
        classes[i] = take(s.by_class[i]);
    }
    const u64 lean_class = classes[static_cast<size_t>(PipelineClass::Lean)];
    const u64 uniforms = take(s.pixel_uniforms);
    const u64 textures = take(s.pixel_textures);
    const u64 storage = take(s.pixel_storage);
    const u64 programs = take(s.pixel_programs);
    const u64 same = take(s.same_pipeline);
    const u64 lean = take(s.lean);
    const u64 not_ready = take(s.not_ready);
    const u64 ready_unused = take(s.ready_unused);
    const u64 made = take(s.made);
    const u64 found = take(s.found);
    const u64 failed = take(s.failed);
    const u64 compared = take(s.compared);
    const u64 differ = take(s.differ);
    const u64 made_ns = take(s.made_ns);
    const u64 timed_lean = take(s.timed_lean);
    const u64 timed_lean_ns = take(s.timed_lean_ns);
    const u64 timed_full = take(s.timed_full);
    const u64 timed_full_ns = take(s.timed_full_ns);
    const int window_mode = s.last_mode.exchange(mode, std::memory_order_relaxed);
    s.window_start_ns.store(now, std::memory_order_relaxed);
    if (draws == 0 || seconds <= 0.0) {
        return;
    }
    const auto each = [](u64 sum, u64 count) {
        return count != 0 ? static_cast<double>(sum) / static_cast<double>(count) : 0.0;
    };
    std::string refused;
    for (size_t i = 1; i < classes.size(); ++i) {
        if (classes[i] != 0) {
            refused += fmt::format("{}{} {:.0f}/s", refused.empty() ? "" : ", ", ClassNames[i],
                                   per_s(classes[i]));
        }
    }
    LOG_INFO(HW_GPU,
             "Occlusion lean (FRITCHII_OCCL_LEAN={}) {:.1f} s: E draws {:.0f}/s, counting {:.0f}/s, "
             "colour off {:.0f}/s, tests (counting, colour off) {:.0f}/s; colour off but alpha to "
             "coverage or alpha test {:.0f}/s; pixel program can change nothing {:.0f}/s (same "
             "pipeline as the previous {:.0f}%), each skips {:.2f} uniform buffers, {:.2f} "
             "textures, {:.2f} storage buffers, {:.2f} handle programs; kept full: {}; drawn lean "
             "{:.0f}/s, sibling not ready {:.0f}/s, verify ready but full drew {:.0f}/s; siblings "
             "made {} ({:.1f} ms on E), found {}, failed {}, vertex side compared {}, differs {} "
             "(refused); E fill+push a timed draw: lean {:.2f} us ({}), full {:.2f} us ({})",
             window_mode, seconds, per_s(draws), per_s(counting), per_s(colour_off), per_s(tests),
             per_s(regs_refused), per_s(lean_class), 100.0 * each(same, lean_class),
             each(uniforms, lean_class), each(textures, lean_class), each(storage, lean_class),
             each(programs, lean_class), refused.empty() ? std::string{"none"} : refused,
             per_s(lean), per_s(not_ready), per_s(ready_unused), made,
             static_cast<double>(made_ns) / 1e6, found, failed, compared, differ,
             each(timed_lean_ns, timed_lean) / 1e3, timed_lean,
             each(timed_full_ns, timed_full) / 1e3, timed_full);
}

} // namespace VideoCommon::OcclLean
