// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <list>
#include <vector>

#include "common/math_util.h"
#include "video_core/renderer_vulkan/present/present_push_constants.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Tegra {
struct FramebufferConfig;
}

namespace Vulkan {

class Device;
struct Frame;
class Layer;
class MemoryAllocator;
class Scheduler;
class RasterizerVulkan;

class WindowAdaptPass final {
public:
    explicit WindowAdaptPass(const Device& device, VkFormat frame_format, vk::Sampler&& sampler,
                             vk::ShaderModule&& fragment_shader);
    ~WindowAdaptPass();

    void Draw(const Device& device, MemoryAllocator& memory_allocator, RasterizerVulkan& rasterizer,
              Scheduler& scheduler, size_t image_index, std::list<Layer>& layers,
              std::span<const Tegra::FramebufferConfig> configs,
              const Layout::FramebufferLayout& layout, Frame* dst);

    VkDescriptorSetLayout GetDescriptorSetLayout();
    VkRenderPass GetRenderPass();

private:
    void CreateDescriptorSetLayout(const Device& device);
    void CreatePipelineLayout(const Device& device);
    void CreateVertexShader(const Device& device);
    void CreateRenderPass(const Device& device, VkFormat frame_format);
    void CreatePipelines(const Device& device);

    /// Stamped test copies, FRITCHII_TEST_MARK=1: brings the mark's texture up to date with
    /// TestWatermark::Build (records an upload only when its pixels changed) and returns whether
    /// there is a mark to draw this frame (test_mark_push and test_mark.sets[0] are then set).
    bool PrepareTestMark(const Device& device, MemoryAllocator& memory_allocator,
                         Scheduler& scheduler, const Layout::FramebufferLayout& layout, u32 width,
                         u32 height);

    /// The mark's pixels over its bounding box, one per frame pixel, with the descriptor set that
    /// samples them. Replaced ones wait in test_mark_retired until the GPU is past their tick.
    struct TestMarkTexture {
        vk::Image image;
        vk::ImageView view;
        vk::DescriptorPool pool;
        vk::DescriptorSets sets;
        vk::Buffer staging;
        u32 width = 0;
        u32 height = 0;
        u64 tick = 0;
    };

    vk::DescriptorSetLayout descriptor_set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::Sampler sampler;
    vk::ShaderModule vertex_shader;
    vk::ShaderModule fragment_shader;
    vk::RenderPass render_pass;
    vk::Pipeline opaque_pipeline;
    vk::Pipeline premultiplied_pipeline;
    vk::Pipeline coverage_pipeline;

    // The test mark (created on the first frame with a mark: none in plain builds).
    vk::Sampler test_mark_sampler;
    vk::ShaderModule test_mark_fragment_shader;
    vk::Pipeline test_mark_pipeline;
    TestMarkTexture test_mark;
    std::vector<TestMarkTexture> test_mark_retired;
    std::vector<float> test_mark_texels; ///< RGBA, what test_mark holds
    u64 test_mark_version = 0;           ///< TestWatermark::Blocks::version last seen
    bool test_mark_ready = false;        ///< test_mark_push and test_mark hold the current mark
    PresentPushConstants test_mark_push{};
};

} // namespace Vulkan
