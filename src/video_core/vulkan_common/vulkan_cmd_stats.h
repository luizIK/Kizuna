// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan::vk {

/// FRITCHII_SLOW_GPU (docs/fritchii/switches.md): what one main-queue submission's command
/// buffers hold, counted by the vk::CommandBuffer wrappers on the thread that records them (the
/// Vulkan worker) while t_cmd_stats points here. Every other thread leaves t_cmd_stats null and
/// pays one thread_local load and a branch a command. Reset for each submission.
struct CmdStats {
    static constexpr u32 MaxPrograms = 8;

    /// A pipeline bound in the submission, and the draws or dispatches recorded while it was.
    struct Program {
        VkPipeline handle{};
        u64 hash_a{};          ///< graphics: the vertex program; compute: the compute program
        u64 hash_b{};          ///< graphics: the fragment program
        const char* label{};   ///< an internal pass that names itself ("astc", ...)
        u32 uses{};
    };

    u32 draws{};
    u32 indirect_draws{};
    u32 dispatches{};
    u32 indirect_dispatches{};
    u32 render_passes{};
    u32 clears{};
    u32 fills{};
    u32 buffer_copies{};
    u32 image_uploads{};   ///< buffer to image copies
    u32 image_downloads{}; ///< image to buffer copies
    u32 image_copies{};    ///< image copies, blits and resolves
    u32 query_ops{};       ///< begin, end, reset, copy results, timestamps
    u32 barriers{};        ///< vkCmdPipelineBarrier calls
    u32 barrier_items{};   ///< memory, buffer and image barriers in them
    u32 pipeline_binds{};
    u32 texture_passes{};  ///< ASTC decodes, BC encodes
    u32 presents{};        ///< BlitScreen frames (window adapt, upscalers, anti-aliasing)
    u32 present_draws{};
    u32 present_dispatches{};
    u32 untracked_binds{}; ///< binds of pipelines past MaxPrograms
    u64 buffer_copy_bytes{};
    u64 fill_bytes{};
    bool in_present{};
    s8 graphics_current{-1};
    s8 compute_current{-1};
    u8 graphics_count{};
    u8 compute_count{};
    std::array<Program, MaxPrograms> graphics{};
    std::array<Program, MaxPrograms> compute{};

    void Bind(VkPipelineBindPoint point, VkPipeline handle) noexcept {
        ++pipeline_binds;
        const bool is_compute = point == VK_PIPELINE_BIND_POINT_COMPUTE;
        auto& table = is_compute ? compute : graphics;
        u8& count = is_compute ? compute_count : graphics_count;
        s8& current = is_compute ? compute_current : graphics_current;
        for (u8 i = 0; i < count; ++i) {
            if (table[i].handle == handle) {
                current = static_cast<s8>(i);
                return;
            }
        }
        if (count == MaxPrograms) {
            ++untracked_binds;
            current = -1;
            return;
        }
        table[count] = Program{.handle = handle};
        current = static_cast<s8>(count++);
    }

    /// Names the pipeline bound last at `point` (its program hashes, or a pass label).
    void Name(VkPipelineBindPoint point, u64 hash_a, u64 hash_b, const char* label) noexcept {
        const bool is_compute = point == VK_PIPELINE_BIND_POINT_COMPUTE;
        const s8 current = is_compute ? compute_current : graphics_current;
        if (current < 0) {
            return;
        }
        Program& program = (is_compute ? compute : graphics)[current];
        program.hash_a = hash_a;
        program.hash_b = hash_b;
        program.label = label;
    }

    void Draw(bool indirect) noexcept {
        ++draws;
        indirect_draws += indirect ? 1 : 0;
        present_draws += in_present ? 1 : 0;
        if (graphics_current >= 0) {
            ++graphics[graphics_current].uses;
        }
    }

    void Dispatch(bool indirect) noexcept {
        ++dispatches;
        indirect_dispatches += indirect ? 1 : 0;
        present_dispatches += in_present ? 1 : 0;
        if (compute_current >= 0) {
            ++compute[compute_current].uses;
        }
    }
};

/// Set on the Vulkan worker while FRITCHII_SLOW_GPU counts (vk_submit_times.cpp); null elsewhere.
inline thread_local CmdStats* t_cmd_stats{};

/// From a pipeline's bind (on the recording thread, after cmdbuf.BindPipeline): its programs.
inline void NoteBoundProgram(VkPipelineBindPoint point, u64 hash_a, u64 hash_b) noexcept {
    if (CmdStats* const stats = t_cmd_stats) [[unlikely]] {
        stats->Name(point, hash_a, hash_b, nullptr);
    }
}

/// From an internal pass's bind: its label; `texture_pass` counts it as a texture decode/encode.
inline void NoteBoundPass(VkPipelineBindPoint point, const char* label,
                          bool texture_pass) noexcept {
    if (CmdStats* const stats = t_cmd_stats) [[unlikely]] {
        stats->Name(point, 0, 0, label);
        stats->texture_passes += texture_pass ? 1 : 0;
    }
}

/// Brackets a frame's presentation passes (BlitScreen::DrawToFrame) in recording order.
inline void NotePresentPass(bool begin) noexcept {
    if (CmdStats* const stats = t_cmd_stats) [[unlikely]] {
        stats->presents += begin ? 1 : 0;
        stats->in_present = begin;
    }
}

} // namespace Vulkan::vk
