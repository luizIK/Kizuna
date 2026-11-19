// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <memory>
#include <variant>
#include <vector>
#include "video_core/renderer_vulkan/vk_mem_stage_hook.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct DescriptorAddress {
    VkDeviceAddress address;
    VkDeviceSize range;
    VkFormat format;
};

union DescriptorUpdateEntry {
    DescriptorUpdateEntry() = default;
    DescriptorUpdateEntry(VkDescriptorImageInfo image_) : image{image_} {}
    DescriptorUpdateEntry(VkDescriptorBufferInfo buffer_) : buffer{buffer_} {}
    DescriptorUpdateEntry(VkBufferView texel_buffer_) : texel_buffer{texel_buffer_} {}
    DescriptorUpdateEntry(DescriptorAddress address_) : address{address_} {}
    std::monostate empty{};
    VkDescriptorImageInfo image;
    VkDescriptorBufferInfo buffer;
    VkBufferView texel_buffer;
    DescriptorAddress address;
};

class UpdateDescriptorQueue final {
    // This should be plenty for the vast majority of cases. Most desktop platforms only
    // provide up to 3 swapchain images.
    static constexpr size_t FRAMES_IN_FLIGHT = 8;

public:
    static constexpr size_t GUEST_FRAME_PAYLOAD_SIZE = 0x80000;
    static constexpr size_t COMPUTE_FRAME_PAYLOAD_SIZE = 0x20000;

    explicit UpdateDescriptorQueue(const Device& device_, size_t frame_payload_size_,
                                   bool supports_descriptor_buffer_ = false);
    ~UpdateDescriptorQueue();

    [[nodiscard]] bool UsesDescriptorBuffer() const noexcept {
        return use_descriptor_buffer;
    }

    [[nodiscard]] bool FormFor(bool use_descriptor_buffer_) const noexcept {
        return supports_descriptor_buffer && use_descriptor_buffer_;
    }

    /// The entry AddBuffer adds, in the form `descriptor_buffer` asks for.
    [[nodiscard]] static DescriptorUpdateEntry MakeBufferEntry(bool descriptor_buffer,
                                                               VkBuffer buffer,
                                                               VkDeviceAddress base_address,
                                                               VkDeviceSize offset,
                                                               VkDeviceSize size) {
        if (!descriptor_buffer) {
            // A null buffer is a null descriptor only as (0, VK_WHOLE_SIZE); with a real offset
            // or range the driver may build one at device address 0 instead.
            return VkDescriptorBufferInfo{
                .buffer = buffer,
                .offset = buffer == VK_NULL_HANDLE ? 0 : offset,
                .range = buffer == VK_NULL_HANDLE ? VK_WHOLE_SIZE : size,
            };
        }
        return DescriptorAddress{
            .address = base_address == 0 ? 0 : base_address + offset,
            .range = base_address == 0 ? VK_WHOLE_SIZE : size,
            .format = VK_FORMAT_UNDEFINED,
        };
    }

    /// The entry AddTexelBuffer adds, in the form `descriptor_buffer` asks for.
    [[nodiscard]] static DescriptorUpdateEntry MakeTexelEntry(bool descriptor_buffer,
                                                              VkBufferView texel_buffer,
                                                              VkDeviceAddress base_address,
                                                              VkDeviceSize offset,
                                                              VkDeviceSize size, VkFormat format) {
        if (!descriptor_buffer) {
            return texel_buffer;
        }
        return DescriptorAddress{
            .address = base_address == 0 ? 0 : base_address + offset,
            .range = base_address == 0 ? VK_WHOLE_SIZE : size,
            .format = format,
        };
    }

    void TickFrame();
    void Acquire(Scheduler& scheduler, size_t required_entries = 0,
                 bool use_descriptor_buffer_ = false);

    const DescriptorUpdateEntry* UpdateData() const noexcept {
        return upload_start;
    }

    size_t UpdateCount() const noexcept {
        return upload_start ? static_cast<size_t>(payload_cursor - upload_start) : 0;
    }

    void AddSampledImage(VkImageView image_view, VkSampler sampler) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) = VkDescriptorImageInfo{
            .sampler = sampler,
            .imageView = image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    void AddImage(VkImageView image_view) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) = VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    void AddBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) = VkDescriptorBufferInfo{
            .buffer = buffer,
            .offset = buffer == VK_NULL_HANDLE ? 0 : offset,
            .range = buffer == VK_NULL_HANDLE ? VK_WHOLE_SIZE : size,
        };
    }

    void AddBuffer(VkBuffer buffer, VkDeviceAddress base_address, VkDeviceSize offset,
                   VkDeviceSize size) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) =
            MakeBufferEntry(use_descriptor_buffer, buffer, base_address, offset, size);
    }

    void AddTexelBuffer(VkBufferView texel_buffer) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) = texel_buffer;
    }

    /// Adds entries recorded from an earlier payload (a pipeline's remembered texture bindings,
    /// EDEN_NX2_EXP32; ported).
    void AddEntries(const DescriptorUpdateEntry* entries, size_t count) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        payload_cursor = std::copy_n(entries, count, payload_cursor);
    }

    void AddTexelBuffer(VkBufferView texel_buffer, VkDeviceAddress base_address,
                        VkDeviceSize offset, VkDeviceSize size, VkFormat format) {
        SeamHook::Watch(SeamHook::Use::DescriptorAdd);
        *(payload_cursor++) = MakeTexelEntry(use_descriptor_buffer, texel_buffer, base_address,
                                             offset, size, format);
    }

private:
    const Device& device;
    size_t frame_payload_size;
    const size_t maximum_frame_payload_size;
    const bool supports_descriptor_buffer;
    bool use_descriptor_buffer{false};
    size_t frame_index{0};
    DescriptorUpdateEntry* payload_cursor = nullptr;
    DescriptorUpdateEntry* payload_start = nullptr;
    const DescriptorUpdateEntry* upload_start = nullptr;
    std::unique_ptr<DescriptorUpdateEntry[]> payload;
    bool growth_disabled{};
    bool fallback_reported{};
};

// TODO: should these be separate classes instead?
using GuestDescriptorQueue = UpdateDescriptorQueue;
using ComputePassDescriptorQueue = UpdateDescriptorQueue;

} // namespace Vulkan
