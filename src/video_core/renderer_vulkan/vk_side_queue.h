// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <mutex>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// FRITCHII_READ_SIDE_COPY >= 2 (helper_cu_answer.md Round 3): a guest read's new download
/// copied on a second queue of the graphics family (Device::GetSideQueue) that waits only for the
/// main-queue tick of the newest write over the copied bytes, instead of at the end of the main
/// queue behind every submission since. Same family: the buffers need no ownership transfer.
///
/// Ordering:
///   - the side submission waits on the master timeline at `main_tick` (stage TRANSFER): every
///     main-queue write before it is done and visible to the copy (a semaphore signal makes its
///     batch's writes available, the wait makes them visible to the waiting stage);
///   - nothing submitted to the main queue after `main_tick` writes the copied bytes before the
///     copy reads them: the newest write over them is in `main_tick` (the read's caller checks),
///     and every main submission from the next one on waits for the side value
///     (MasterSemaphore::RequireSide) until the side semaphore reports it;
///   - the copy ends with a TRANSFER_WRITE -> HOST_READ barrier, the reader waits on the side
///     semaphore before it reads the staging memory;
///   - the main ticks the side submissions wait for never go down (each waits for the highest
///     asked so far). The side queue runs its submissions in order, so a later copy whose writer
///     is older would otherwise sit behind an earlier copy waiting for a newer main tick, while
///     main submissions between the two ticks waited for the later copy: build 26 hung so, in
///     both modes, ~3 s into the fight (helper_cu_answer.md Round 4). Waiting longer than the
///     writer needs is still exact: nothing recorded before the read writes its bytes.
class SideQueue {
public:
    struct Copy {
        VkBuffer src;
        VkBuffer dst;
        VkBufferCopy region;
    };

    explicit SideQueue(const Device& device, MasterSemaphore& master);
    ~SideQueue();

    SideQueue(const SideQueue&) = delete;
    SideQueue& operator=(const SideQueue&) = delete;

    /// Records and submits `copies`, after the main timeline reached `main_tick`. Returns the
    /// side value to wait for (0: nothing was submitted: the submit failed, or `main_tick` is
    /// not given to the main queue yet, only to the worker: the side submission never waits for
    /// a signal not submitted yet, whatever a driver does with such waits).
    [[nodiscard]] u64 Submit(u64 main_tick, std::span<const Copy> copies);

    [[nodiscard]] bool IsDone(u64 value);
    void Wait(u64 value);

    /// Must-be-0 and diagnostic counts, taken (reset) by the caller's log line.
    struct Stats {
        u64 raised;       ///< submissions whose main tick was raised to keep the order
        u64 slow_waits;   ///< host waits for a side value that took 2 s or more (logged)
        u64 not_queued;   ///< refused: the main tick was not given to vkQueueSubmit yet
    };
    [[nodiscard]] Stats TakeStats();

private:
    struct Slot {
        VkCommandBuffer cmdbuf{};
        u64 value{}; ///< the side value its last submission signals (0: free)
    };

    const Device& device;
    MasterSemaphore& master;
    vk::CommandPool pool;
    vk::CommandBuffers cmdbufs;
    vk::Semaphore semaphore;
    std::mutex mutex;
    std::vector<Slot> slots;
    size_t next_slot{};
    u64 next_value{1};
    u64 last_main_tick{0}; ///< the highest main tick a side submission waits for (`mutex`)
    std::atomic<u64> known{0};
    std::atomic<u64> raised{0};
    std::atomic<u64> slow_waits{0};
    std::atomic<u64> not_queued{0};
};

} // namespace Vulkan
