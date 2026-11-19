// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan {

/// FRITCHII_KEEP_BLOCKS (live, MiB, default 0 = off; 1 = 256 MiB; at most 512 MiB): VMA's
/// device-local memory blocks are kept when VMA frees them and handed back, without a driver
/// call, when VMA next asks the driver for a block of the same memory type, size and flags.
/// Area loads evict images (their blocks empty and go back to the driver) and then make new
/// ones (new blocks from the driver): vkAllocateMemory took 29-50% of the GPU back thread in
/// OoT's transition frames on Blitchii (DahBard/Blitchii r64). Only plain blocks are kept (a
/// power-of-two size from 1 to 64 MiB, no chain but the device address flags), never memory
/// dedicated to an image or a buffer, never host-visible memory.
///
/// How much it keeps: the switch's cap, and at most what the device budget leaves above the
/// live usage (SetLimit, from the texture cache's budget check). It gives everything back when
/// an allocation fails, when an image does not fit the budget, when images stop fitting
/// (memory pressure), and when the switch goes to 0. The caches count kept blocks as free.
///
/// FRITCHII_KEEP_BLOCKS_PREMAKE=1 (boot, default 0): the cap's worth of blocks is made at boot
/// (VMA's default block size, in the memory type images take), so the first area loads find
/// them too.
class KeptBlocks {
public:
    struct Totals {
        u64 reused{};
        u64 reused_bytes{};
        u64 kept{};
        u64 kept_bytes{};
        u64 released{};
        u64 released_bytes{};
        u64 refused{}; ///< Freed blocks not kept: over the cap or the limit
    };

    explicit KeptBlocks(VkDevice device, PFN_vkAllocateMemory allocate, PFN_vkFreeMemory free,
                        const VkPhysicalDeviceMemoryProperties& properties);
    ~KeptBlocks();

    KeptBlocks(const KeptBlocks&) = delete;
    KeptBlocks& operator=(const KeptBlocks&) = delete;

    /// The kept blocks VMA's memory calls for `device` go through, or null.
    [[nodiscard]] static KeptBlocks* Active(VkDevice device) noexcept;

    /// FRITCHII_KEEP_BLOCKS's cap in bytes (0 = off).
    [[nodiscard]] static u64 SwitchCap();

    /// VMA's memory calls go through these kept blocks until Close.
    void Activate() noexcept;

    /// Whether blocks of this memory type can be kept (device-local, not host-visible).
    [[nodiscard]] bool Keeps(u32 memory_type) const noexcept;

    /// Makes up to `count` blocks of `size` bytes in `memory_type` now, with the flags VMA gives
    /// its blocks, and keeps them; returns how many it made.
    u32 Make(u32 memory_type, VkDeviceSize size, u32 count, VkMemoryAllocateFlags flags);

    /// A kept block for `info` (true, `memory` set): the driver is not called.
    bool Take(const VkMemoryAllocateInfo& info, VkDeviceMemory& memory);

    /// `memory`, just made by the driver for `info`, is remembered if it could be kept.
    void Track(const VkMemoryAllocateInfo& info, VkDeviceMemory memory);

    /// True when `memory`, which VMA frees, is kept instead of given back to the driver.
    bool Keep(VkDeviceMemory memory);

    /// The most the blocks may hold from now on, besides the switch's cap.
    void SetLimit(u64 bytes) noexcept;

    /// Once a frame: gives one block back while they hold more than allowed (all of them when
    /// the switch is 0), and logs the last 10 s.
    void Tick();

    /// Gives every kept block back to the driver now; returns how many bytes.
    u64 Release();

    /// Gives every kept block back and keeps nothing more (device teardown).
    void Close();

    [[nodiscard]] u64 KeptBytes() const noexcept {
        return kept_total.load(std::memory_order_relaxed);
    }
    [[nodiscard]] u64 KeptBytes(u32 heap) const noexcept {
        return heap < VK_MAX_MEMORY_HEAPS ? kept_by_heap[heap].load(std::memory_order_relaxed)
                                          : 0;
    }

    /// The driver's calls under VMA, for the 10 s line.
    void CountDriverAllocation(u64 bytes, u64 ns) noexcept;
    void CountDriverFree(u64 ns) noexcept;

private:
    struct Key {
        u32 memory_type{};
        VkMemoryAllocateFlags flags{};
        VkDeviceSize size{};
        bool operator==(const Key&) const = default;
    };
    struct Block {
        VkDeviceMemory memory;
        Key key;
    };

    [[nodiscard]] std::optional<Key> KeyOf(const VkMemoryAllocateInfo& info) const noexcept;
    [[nodiscard]] u64 Allowed() const;
    void AddKept(const Key& key);
    void RemoveKept(const Key& key);
    void FreeTimed(VkDeviceMemory memory);
    void Log(u64 cap);

    VkDevice device;
    PFN_vkAllocateMemory allocate;
    PFN_vkFreeMemory free_memory;
    u32 keep_types{};
    std::array<u32, VK_MAX_MEMORY_TYPES> heap_of_type{};
    std::atomic<u64> limit{~u64{0}};
    std::atomic<u64> kept_total{0};
    std::atomic<u64> kept_blocks{0};
    std::array<std::atomic<u64>, VK_MAX_MEMORY_HEAPS> kept_by_heap{};
    std::mutex mutex;
    std::vector<Block> blocks; ///< Kept, oldest first
    std::unordered_map<VkDeviceMemory, Key> tracked; ///< In VMA's hands, kept when freed
    Totals totals;
    bool closed{};

    // The 10 s line.
    std::atomic<u64> driver_allocations{0};
    std::atomic<u64> driver_allocation_bytes{0};
    std::atomic<u64> driver_allocation_ns{0};
    std::atomic<u64> driver_longest_ns{0};
    std::atomic<u64> driver_frees{0};
    std::atomic<u64> driver_free_ns{0};
    s64 last_log_ns{};
};

} // namespace Vulkan
