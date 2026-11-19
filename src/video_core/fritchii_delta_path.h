// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

// EXP118, the delta path for draws that sample nothing (fable52_answer_potroom3 sections 2 and 4,
// step 1: the delta draw plan's D4 cut down to pipelines with no texture, image, texture buffer
// or image buffer descriptor). Docs: docs/fritchii/switches.md, EXP118.
//
// A draw whose previous draw used the same pipeline, with only light state changed in between
// (the delta draw count's serial, nx2_fastdraw_count.h), keeps that pipeline instead of looking
// it up, and configures it without the steps that only serve textures (the descriptor pools'
// sync, handle reads, sampler lookups, image views). Uniform, storage, vertex and index binds,
// the descriptor push, render targets, dynamic state, the host query, the draw record and the
// reports run as for every draw. The breaks it needs are raised only when EXP118 is on at boot
// (Nx2FastDrawCount::Init); with them on, the switch is live: 0 = off, 1 = on, 2 = verify only
// (decide, then look the pipeline up in full and compare the pointer), 3 = on, and every delta
// draw also looks the pipeline up (the pointer must match, else the draw is drawn the full way)
// and synchronizes the descriptor pools afterwards (nothing may change).
namespace VideoCommon::FritchiiDeltaPath {

/// The switch: EDEN_NX2_EXP118.
inline constexpr int Switch = 118;
inline constexpr u64 FramesPerLine = 600;

/// Frames presented (RasterizerVulkan::TickFrame, while the breaks are on); the drawing thread
/// logs every FramesPerLine.
inline std::atomic<u64> g_frames{};

/// Why a draw was drawn the full way with the switch on.
enum class Fallback : u8 {
    NoCandidate,   ///< The draw before can't be followed (ineligible, none, another thread)
    Kind,          ///< Indirect, indexed or not, engine, topology, engine hint, inline index, dump
    StateChanged,  ///< A break since the draw before (a register, an invalidation, a fence...)
    Map,           ///< GPU memory mappings changed
    Submitted,     ///< The command buffer was submitted since
    RenderPass,    ///< Another render pass or bound pipeline
    Samples,       ///< The pipeline samples something (not this cut) or is a mesh pipeline
    MemoSwitches,  ///< EXP32, EXP87 or EDEN_NX2_TEX_BANK_GEN on
    NotCurrent,    ///< The pipeline cache's current pipeline is another one
    LookupInputs,  ///< The programs or the viewport swizzles are marked for a fresh lookup
    VertexInput,   ///< Vertex input moved and the pipeline's key would change (or is static)
    Late,          ///< A break between the decision and the cache locks
    LookupDiffers, ///< EXP118=3: the full lookup gave another pipeline
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> fallback_names{
    "no draw to follow", "other kind",    "state changed",  "memory map",
    "submitted",         "render pass",   "samples",        "memo switches",
    "not current",       "lookup inputs", "vertex input",   "late break",
    "lookup differs",
};

/// Counts on the drawing thread, logged every FramesPerLine frames while the switch is on.
struct Stats {
    u64 draws{};        ///< Draws decided with the switch on
    u64 taken{};        ///< Delta draws (1 and 3)
    u64 would{};        ///< EXP118=2: draws that would have been delta draws
    u64 compared{};     ///< Lookups compared with the kept pipeline (2: after the late test; 3)
    u64 differences{};  ///< ...that gave another pipeline (must stay 0)
    u64 sync_checked{}; ///< EXP118=3: delta draws whose descriptor pools were synchronized after
    u64 sync_changed{}; ///< ...and changed something (must stay 0)
    u64 vertex_kept{};  ///< Delta draws with the vertex input moved, the key unchanged
    std::array<u64, static_cast<size_t>(Fallback::Count)> fallbacks{};
    u64 total_differences{}; ///< Since boot (differences and sync changes), for the log cap
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
            *this = Stats{.total_differences = total_differences,
                          .window_frame = frames,
                          .mode = mode_,
                          .started = true};
            return;
        }
        if (frames - window_frame < FramesPerLine) {
            return;
        }
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
        if (mode == 2 || mode == 3) {
            verify = fmt::format(" | verify: {:.1f} compared, {} differences", per_frame(compared),
                                 differences);
            if (mode == 3) {
                verify += fmt::format(", pools synchronized after {:.1f}, {} changed",
                                      per_frame(sync_checked), sync_changed);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "Delta path (EXP118={}) per frame over {:.0f} frames: {:.1f} draws decided, "
                 "{:.1f} {} ({:.1f} with the vertex input moved) | full: {}{}",
                 mode, frame_count, per_frame(draws), per_frame(mode == 2 ? would : taken),
                 mode == 2 ? "would be delta draws" : "delta draws", per_frame(vertex_kept),
                 falls.empty() ? std::string{"none"} : falls, verify);
        *this = Stats{.total_differences = total_differences,
                      .window_frame = frames,
                      .mode = mode,
                      .started = true};
    }
};

} // namespace VideoCommon::FritchiiDeltaPath
