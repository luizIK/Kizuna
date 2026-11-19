// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"

namespace Vulkan {

/// FRITCHII_MEMSTAGE_SHADOW (memstage_shadow.cpp, compare c): a digest of what the memory stage's
/// fill takes from a graphics pipeline (MemStage::DrawFacts, spec 4.2's StageFacts): the uniform
/// masks and sizes, the stages, the flags, and per stage the descriptors whose constant buffer
/// words the fill reads (storage buffer pointers, texel buffers, handle words and programs).
/// Values, never pointers: two pipelines with the same facts give the same digest.
[[nodiscard]] inline u64 ShadowFactsDigest(const MemStage::DrawFacts& facts,
                                           bool dynamic_vertex_input) {
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](u64 value) {
        hash = (hash ^ value) * 0x100000001b3ULL;
        hash ^= hash >> 29;
    };
    for (const u32 mask : *facts.uniform_masks) {
        mix(mask);
    }
    for (const auto& stage : *facts.uniform_sizes) {
        for (const u32 size : stage) {
            mix(size);
        }
    }
    mix(facts.stages);
    mix((facts.has_texel_buffers ? 1 : 0) | (facts.uses_descriptor_buffer ? 2 : 0) |
        (dynamic_vertex_input ? 4 : 0));
    for (u32 stage = 0; stage < 5; ++stage) {
        if ((facts.stages & (1U << stage)) == 0) {
            continue;
        }
        const Shader::Info& info = facts.stage_infos[stage];
        mix(0x5747ULL + stage);
        for (const auto& desc : info.storage_buffers_descriptors) {
            mix(desc.cbuf_index);
            mix(desc.cbuf_offset);
            mix(desc.count);
            mix(desc.is_written ? 1 : 0);
            mix(desc.indirect_offset);
        }
        mix(0x7b);
        for (const auto& desc : info.texture_buffer_descriptors) {
            mix(desc.cbuf_index);
            mix(desc.cbuf_offset);
            mix(desc.count);
            mix(desc.has_secondary ? 1 : 0);
            mix(desc.secondary_cbuf_index);
            mix(desc.secondary_cbuf_offset);
            mix(desc.handle_program);
        }
        mix(0x1b);
        for (const auto& desc : info.image_buffer_descriptors) {
            mix(desc.cbuf_index);
            mix(desc.cbuf_offset);
            mix(desc.count);
            mix(desc.handle_program);
        }
        mix(0x7d);
        for (const auto& desc : info.texture_descriptors) {
            mix(desc.cbuf_index);
            mix(desc.cbuf_offset);
            mix(desc.count);
            mix(desc.has_secondary ? 1 : 0);
            mix(desc.secondary_cbuf_index);
            mix(desc.secondary_cbuf_offset);
            mix(desc.handle_program);
        }
        mix(0x1d);
        for (const auto& desc : info.image_descriptors) {
            mix(desc.cbuf_index);
            mix(desc.cbuf_offset);
            mix(desc.count);
            mix(desc.handle_program);
        }
        mix(0x9f);
        for (const auto& program : info.handle_programs) {
            mix(program.size());
            for (const u32 word : program) {
                mix(word);
            }
        }
    }
    return hash;
}

} // namespace Vulkan
