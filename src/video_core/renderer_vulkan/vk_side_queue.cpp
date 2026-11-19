// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_side_queue.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
constexpr size_t NumSlots = 64;
}

SideQueue::SideQueue(const Device& device_, MasterSemaphore& master_)
    : device{device_}, master{master_} {
    pool = device.GetLogical().CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags =
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    cmdbufs = pool.Allocate(NumSlots);
    slots.resize(NumSlots);
    for (size_t i = 0; i < NumSlots; ++i) {
        slots[i].cmdbuf = cmdbufs[i];
    }
    static constexpr VkSemaphoreTypeCreateInfo semaphore_type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    semaphore = device.GetLogical().CreateSemaphore({
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semaphore_type_ci,
        .flags = 0,
    });
    master.SetSideSemaphore(&semaphore);
    LOG_INFO(Render_Vulkan, "FRITCHII_READ_SIDE_COPY: side queue ready ({} command buffers)",
             NumSlots);
}

SideQueue::~SideQueue() {
    // Every side value before the device goes: the command buffers and the semaphore.
    const u64 last = next_value - 1;
    if (last != 0) {
        Wait(last);
    }
    master.SetSideSemaphore(nullptr);
}

u64 SideQueue::Submit(u64 main_tick_, std::span<const Copy> copies) {
    std::scoped_lock lock{mutex};
    // Never lower than the previous submission's: in order on this queue, see the header.
    u64 main_tick = main_tick_;
    if (main_tick < last_main_tick) {
        main_tick = last_main_tick;
        raised.fetch_add(1, std::memory_order_relaxed);
    }
    if (main_tick > master.QueueSubmittedTick()) {
        // Handed to the worker but not to the queue: the caller copies on the main queue.
        not_queued.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    last_main_tick = main_tick;
    Slot& slot = slots[next_slot];
    if (slot.value != 0 && !IsDone(slot.value)) {
        Wait(slot.value);
    }
    const vk::CommandBuffer cmdbuf{slot.cmdbuf, device.GetDispatchLoader()};
    cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    for (const Copy& copy : copies) {
        cmdbuf.CopyBuffer(copy.src, copy.dst, copy.region);
    }
    static constexpr VkMemoryBarrier HOST_READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           HOST_READ_BARRIER);
    cmdbuf.End();

    const u64 value = next_value;
    const VkSemaphore wait_semaphore = master.TimelineHandle();
    const VkSemaphore signal_semaphore = *semaphore;
    static constexpr VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkTimelineSemaphoreSubmitInfo timeline_si{
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &main_tick,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &value,
    };
    const VkCommandBuffer raw_cmdbuf = slot.cmdbuf;
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &raw_cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    // The side queue is used only here, under `mutex`: externally synchronized.
    const VkResult result = device.GetSideQueue().Submit(submit_info);
    if (result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "FRITCHII_READ_SIDE_COPY: side submit failed ({})",
                  static_cast<int>(result));
        return 0;
    }
    slot.value = value;
    ++next_value;
    next_slot = (next_slot + 1) % slots.size();
    // Every main submission after `main_tick` waits for it until the side semaphore reports it
    // (none up to it does: the copy waits for those).
    master.RequireSide(value, main_tick, value);
    return value;
}

bool SideQueue::IsDone(u64 value) {
    if (known.load(std::memory_order_acquire) >= value) {
        return true;
    }
    const u64 counter = semaphore.GetCounter();
    u64 old = known.load(std::memory_order_relaxed);
    while (counter > old &&
           !known.compare_exchange_weak(old, counter, std::memory_order_release,
                                        std::memory_order_relaxed)) {
    }
    return counter >= value;
}

void SideQueue::Wait(u64 value) {
    if (IsDone(value)) {
        return;
    }
    // In 2 s steps: a side value that does not come is logged (with the main queue's state)
    // instead of blocking without a word.
    constexpr u64 Step = 2'000'000'000ULL;
    for (u32 steps = 0; !semaphore.Wait(value, Step); ++steps) {
        if (steps == 0) {
            slow_waits.fetch_add(1, std::memory_order_relaxed);
        }
        master.Refresh();
        LOG_ERROR(Render_Vulkan,
                  "FRITCHII_READ_SIDE_COPY: side value {} not done after {} s (side reports {}, "
                  "newest asked {}; its main tick wait {}, main known {}, main current {})",
                  value, (steps + 1) * 2, semaphore.GetCounter(), next_value - 1, last_main_tick,
                  master.KnownGpuTick(), master.CurrentTick());
    }
    (void)IsDone(value);
}

SideQueue::Stats SideQueue::TakeStats() {
    return Stats{
        .raised = raised.exchange(0, std::memory_order_relaxed),
        .slow_waits = slow_waits.exchange(0, std::memory_order_relaxed),
        .not_queued = not_queued.exchange(0, std::memory_order_relaxed),
    };
}

} // namespace Vulkan
