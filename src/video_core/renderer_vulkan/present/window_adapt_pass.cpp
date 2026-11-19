// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "core/frontend/framebuffer_layout.h"
#include "video_core/framebuffer_config.h"
#include "video_core/host_shaders/vulkan_present_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_vert_spv.h"
#include "video_core/renderer_vulkan/present/layer.h"
#include "video_core/renderer_vulkan/present/present_push_constants.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/present/window_adapt_pass.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/test_watermark.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

WindowAdaptPass::WindowAdaptPass(const Device& device, VkFormat frame_format, vk::Sampler&& sampler_, vk::ShaderModule&& fragment_shader_)
    : sampler(std::move(sampler_))
    , fragment_shader(std::move(fragment_shader_))
{
    CreateDescriptorSetLayout(device);
    CreatePipelineLayout(device);
    CreateVertexShader(device);
    CreateRenderPass(device, frame_format);
    CreatePipelines(device);
}

WindowAdaptPass::~WindowAdaptPass() = default;

void WindowAdaptPass::Draw(const Device& device, MemoryAllocator& memory_allocator,
                           RasterizerVulkan& rasterizer, Scheduler& scheduler, size_t image_index,
                           std::list<Layer>& layers,
                           std::span<const Tegra::FramebufferConfig> configs,
                           const Layout::FramebufferLayout& layout, Frame* dst) {

    const VkFramebuffer host_framebuffer{*dst->framebuffer};
    const VkRenderPass renderpass{*render_pass};
    const VkPipelineLayout graphics_pipeline_layout{*pipeline_layout};
    const VkExtent2D render_area{
        .width = dst->width,
        .height = dst->height,
    };

    const size_t layer_count = configs.size();
    std::vector<PresentPushConstants> push_constants(layer_count);
    std::vector<VkDescriptorSet> descriptor_sets(layer_count);
    std::vector<VkPipeline> graphics_pipelines(layer_count);

    auto layer_it = layers.begin();
    for (size_t i = 0; i < layer_count; i++) {
        switch (configs[i].blending) {
        case Tegra::BlendMode::Opaque:
        default:
            graphics_pipelines[i] = *opaque_pipeline;
            break;
        case Tegra::BlendMode::Premultiplied:
            graphics_pipelines[i] = *premultiplied_pipeline;
            break;
        case Tegra::BlendMode::Coverage:
            graphics_pipelines[i] = *coverage_pipeline;
            break;
        }

        layer_it->ConfigureDraw(device, &push_constants[i], &descriptor_sets[i], rasterizer, *sampler,
                                image_index, configs[i], layout);
        layer_it++;
    }

    // Stamped test copies: their mark, inside the game picture (video_core/test_watermark.cpp).
    // FRITCHII_TEST_MARK=1 (default): one textured draw. 3: two clears over all its blocks (AMD's
    // driver took ~10 ms of GPU time a frame for those at 4K). 0: no mark drawn.
    std::vector<VkClearRect> mark_shadow;
    std::vector<VkClearRect> mark_text;
    bool mark_draw = false;
    const int mark_mode = VideoCore::TestWatermark::Mode();
    if (mark_mode == 0) {
        static bool logged = false; // the log line only (Build is never called again after it)
        if (!logged &&
            !VideoCore::TestWatermark::Build(layout.screen, dst->width, dst->height).text.empty()) {
            logged = true;
            VideoCore::TestWatermark::LogMode("off");
        }
    } else if (mark_mode == 1) {
        mark_draw = PrepareTestMark(device, memory_allocator, scheduler, layout, dst->width,
                                    dst->height);
    } else {
        const auto& mark = VideoCore::TestWatermark::Build(layout.screen, dst->width, dst->height);
        const auto to_rects = [](const std::vector<VideoCore::TestWatermark::Block>& blocks,
                                 std::vector<VkClearRect>& out) {
            out.reserve(blocks.size());
            for (const auto& b : blocks) {
                out.push_back(VkClearRect{
                    .rect = {.offset = {static_cast<s32>(b.x), static_cast<s32>(b.y)},
                             .extent = {b.width, b.height}},
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                });
            }
        };
        to_rects(mark.shadow, mark_shadow);
        to_rects(mark.text, mark_text);
        if (!mark_text.empty()) {
            VideoCore::TestWatermark::LogMode("two clears");
        }
    }
    const VkPipeline mark_pipeline = mark_draw ? *test_mark_pipeline : VkPipeline{};
    const VkDescriptorSet mark_set = mark_draw ? test_mark.sets[0] : VkDescriptorSet{};
    const PresentPushConstants mark_push = test_mark_push;

    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const f32 bg_red = Settings::values.bg_red.GetValue() / 255.0f;
        const f32 bg_green = Settings::values.bg_green.GetValue() / 255.0f;
        const f32 bg_blue = Settings::values.bg_blue.GetValue() / 255.0f;
        const VkClearAttachment clear_attachment{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .colorAttachment = 0,
            .clearValue =
                {
                    .color = {.float32 = {bg_red, bg_green, bg_blue, 1.0f}},
                },
        };
        const VkClearRect clear_rect{
            .rect =
                {
                    .offset = {0, 0},
                    .extent = render_area,
                },
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        BeginRenderPass(cmdbuf, renderpass, host_framebuffer, render_area);
        cmdbuf.ClearAttachments({clear_attachment}, {clear_rect});

        for (size_t i = 0; i < layer_count; i++) {
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipelines[i]);
            cmdbuf.PushConstants(graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                                 push_constants[i]);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0,
                                      descriptor_sets[i], {});
            cmdbuf.Draw(4, 1, 0, 0);
        }

        if (!mark_text.empty()) {
            const auto attachment = [](const float (&c)[4]) {
                return VkClearAttachment{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .colorAttachment = 0,
                    .clearValue = {.color = {.float32 = {c[0], c[1], c[2], c[3]}}},
                };
            };
            cmdbuf.ClearAttachments({attachment(VideoCore::TestWatermark::ShadowColor)},
                                    mark_shadow);
            cmdbuf.ClearAttachments({attachment(VideoCore::TestWatermark::TextColor)}, mark_text);
        }
        if (mark_pipeline != VK_NULL_HANDLE) {
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, mark_pipeline);
            cmdbuf.PushConstants(graphics_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, mark_push);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0,
                                      mark_set, {});
            cmdbuf.Draw(4, 1, 0, 0);
        }

        cmdbuf.EndRenderPass();
    });
}

VkDescriptorSetLayout WindowAdaptPass::GetDescriptorSetLayout() {
    return *descriptor_set_layout;
}

VkRenderPass WindowAdaptPass::GetRenderPass() {
    return *render_pass;
}

void WindowAdaptPass::CreateDescriptorSetLayout(const Device& device) {
    descriptor_set_layout =
        CreateWrappedDescriptorSetLayout(device, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
}

void WindowAdaptPass::CreatePipelineLayout(const Device& device) {
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PresentPushConstants),
    };

    pipeline_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = descriptor_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    });
}

void WindowAdaptPass::CreateVertexShader(const Device& device) {
    vertex_shader = BuildShader(device, VULKAN_PRESENT_VERT_SPV);
}

void WindowAdaptPass::CreateRenderPass(const Device& device, VkFormat frame_format) {
    render_pass = CreateWrappedRenderPass(device, frame_format, VK_IMAGE_LAYOUT_UNDEFINED);
}

void WindowAdaptPass::CreatePipelines(const Device& device) {
    opaque_pipeline = CreateWrappedPipeline(device, render_pass, pipeline_layout,
                                            std::tie(vertex_shader, fragment_shader));
    premultiplied_pipeline = CreateWrappedPremultipliedBlendingPipeline(
        device, render_pass, pipeline_layout, std::tie(vertex_shader, fragment_shader));
    coverage_pipeline = CreateWrappedCoverageBlendingPipeline(
        device, render_pass, pipeline_layout, std::tie(vertex_shader, fragment_shader));
}

bool WindowAdaptPass::PrepareTestMark(const Device& device, MemoryAllocator& memory_allocator,
                                      Scheduler& scheduler,
                                      const Layout::FramebufferLayout& layout, u32 width,
                                      u32 height) {
    using namespace VideoCore::TestWatermark;
    const Blocks& mark = Build(layout.screen, width, height);
    if (mark.text.empty()) {
        // Plain builds always (nothing is ever created); a stamped copy in a tiny window.
        test_mark_version = mark.version;
        test_mark_ready = false;
        return false;
    }
    if (!test_mark_retired.empty()) {
        std::erase_if(test_mark_retired,
                      [&](const TestMarkTexture& old) { return scheduler.IsFree(old.tick); });
    }
    if (test_mark_ready && mark.version == test_mark_version) {
        return true;
    }
    test_mark_version = mark.version;
    LogMode("one textured draw");

    // The mark's pixels over its bounding box: the shadow blocks, the text over them, alpha 0
    // elsewhere. Float texels, so the drawn colours are exactly the clear colours of mode 3.
    u32 x0 = ~0u;
    u32 y0 = ~0u;
    u32 x1 = 0;
    u32 y1 = 0;
    for (const auto* list : {&mark.shadow, &mark.text}) {
        for (const Block& b : *list) {
            x0 = (std::min)(x0, b.x);
            y0 = (std::min)(y0, b.y);
            x1 = (std::max)(x1, b.x + b.width);
            y1 = (std::max)(y1, b.y + b.height);
        }
    }
    const u32 mark_width = x1 - x0;
    const u32 mark_height = y1 - y0;
    std::vector<float> texels(std::size_t{mark_width} * mark_height * 4, 0.0f);
    const auto fill = [&](const std::vector<Block>& blocks, const float (&color)[4]) {
        for (const Block& b : blocks) {
            for (u32 y = b.y - y0; y < b.y - y0 + b.height; ++y) {
                float* row = texels.data() + (std::size_t{y} * mark_width + (b.x - x0)) * 4;
                for (u32 x = 0; x < b.width; ++x) {
                    std::memcpy(row + std::size_t{x} * 4, color, sizeof(color));
                }
            }
        }
    };
    fill(mark.shadow, ShadowColor);
    fill(mark.text, TextColor);

    // One quad over the bounding box, one texel per frame pixel (nearest sampling).
    const f32 left = static_cast<f32>(x0);
    const f32 top = static_cast<f32>(y0);
    const f32 right = static_cast<f32>(x1);
    const f32 bottom = static_cast<f32>(y1);
    test_mark_push.modelview_matrix =
        MakeOrthographicMatrix(static_cast<f32>(width), static_cast<f32>(height));
    test_mark_push.vertices[0] = ScreenRectVertex(left, top, 0.0f, 0.0f);
    test_mark_push.vertices[1] = ScreenRectVertex(right, top, 1.0f, 0.0f);
    test_mark_push.vertices[2] = ScreenRectVertex(left, bottom, 0.0f, 1.0f);
    test_mark_push.vertices[3] = ScreenRectVertex(right, bottom, 1.0f, 1.0f);

    if (!test_mark_pipeline) {
        test_mark_sampler = CreateNearestNeighborSampler(device);
        test_mark_fragment_shader = BuildShader(device, VULKAN_PRESENT_FRAG_SPV);
        test_mark_pipeline = CreateWrappedOverBlendingPipeline(
            device, render_pass, pipeline_layout,
            std::tie(vertex_shader, test_mark_fragment_shader));
    }

    const bool same_size = test_mark.image && test_mark.width == mark_width &&
                           test_mark.height == mark_height;
    if (same_size && texels == test_mark_texels) {
        // Moved to its next spot with the same pixels: only the quad moves.
        test_mark_ready = true;
        return true;
    }

    constexpr VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    if (!same_size) {
        if (test_mark.image) {
            test_mark.tick = scheduler.CurrentTick();
            test_mark_retired.push_back(std::move(test_mark));
            test_mark = TestMarkTexture{};
        }
        test_mark.image = memory_allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {.width = mark_width, .height = mark_height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        test_mark.view = CreateWrappedImageView(device, test_mark.image, format);
        test_mark.pool = CreateWrappedDescriptorPool(device, 1, 1);
        const std::array<VkDescriptorSetLayout, 1> set_layouts{*descriptor_set_layout};
        test_mark.sets = CreateWrappedDescriptorSets(test_mark.pool, set_layouts);
        const VkDescriptorImageInfo image_info{
            .sampler = *test_mark_sampler,
            .imageView = *test_mark.view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = test_mark.sets[0],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        device.GetLogical().UpdateDescriptorSets(write, {});
        test_mark.width = mark_width;
        test_mark.height = mark_height;
    } else if (test_mark.staging) {
        // The last upload's buffer may still be read: a new one, the old one freed later.
        TestMarkTexture old;
        old.staging = std::move(test_mark.staging);
        old.tick = scheduler.CurrentTick();
        test_mark_retired.push_back(std::move(old));
    }

    const std::size_t bytes = texels.size() * sizeof(float);
    test_mark.staging = memory_allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::Upload);
    std::memcpy(test_mark.staging.Mapped().data(), texels.data(), bytes);
    test_mark.staging.Flush();

    // Outside the present render pass, before it: the whole image is rewritten (earlier frames'
    // sampling of it finishes first), then read by this frame's draw.
    const VkBuffer src = *test_mark.staging;
    const VkImage image = *test_mark.image;
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset{.x = 0, .y = 0, .z = 0},
        .imageExtent{.width = mark_width, .height = mark_height, .depth = 1},
    };
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([src, image, copy](vk::CommandBuffer cmdbuf) {
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
        cmdbuf.CopyBufferToImage(src, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, copy);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, barrier);
    });
    test_mark_texels = std::move(texels);
    test_mark_ready = true;
    return true;
}

} // namespace Vulkan
