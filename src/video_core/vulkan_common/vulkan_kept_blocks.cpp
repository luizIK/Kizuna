// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// Kept device memory blocks (FRITCHII_KEEP_BLOCKS). Ported from DahBard/Blitchii's device memory
// reserve (r64 cccc1ff9fc, vulkan_memory_reserve.cpp), capped for 8 GB cards.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/vulkan_common/vulkan_kept_blocks.h"

namespace Vulkan {
namespace {

constexpr u64 MiB = u64{1} << 20;
/// FRITCHII_KEEP_BLOCKS=1, and the most any value keeps (no brute-force VRAM: 8 GB cards).
constexpr u64 DefaultCap = 256 * MiB;
constexpr u64 MaxCap = 512 * MiB;
/// The block sizes kept: VMA's default blocks (4-32 MiB, EDEN_NX2_VMA_BLOCK_MIB) and the separate
/// pools' 16 MiB ones.
constexpr VkDeviceSize MinBlock = 1 * MiB;
constexpr VkDeviceSize MaxBlock = 64 * MiB;

std::atomic<KeptBlocks*> g_active{nullptr};

/// The flags of a block allocation's chain (empty, or one VkMemoryAllocateFlagsInfo for buffer
/// device addresses), or nothing for any other chain: memory dedicated to an image or a buffer,
/// exported memory, priorities, device masks.
std::optional<VkMemoryAllocateFlags> BlockFlags(const VkMemoryAllocateInfo& info) {
    VkMemoryAllocateFlags flags = 0;
    for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next != nullptr;
         next = next->pNext) {
        if (next->sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO) {
            return std::nullopt;
        }
        const auto& flags_info = *reinterpret_cast<const VkMemoryAllocateFlagsInfo*>(next);
        if ((flags_info.flags & VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT) != 0) {
            return std::nullopt;
        }
        flags |= flags_info.flags;
    }
    return flags;
}

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // Anonymous namespace

KeptBlocks::KeptBlocks(VkDevice device_, PFN_vkAllocateMemory allocate_, PFN_vkFreeMemory free_,
                       const VkPhysicalDeviceMemoryProperties& properties)
    : device{device_}, allocate{allocate_}, free_memory{free_} {
    constexpr VkMemoryPropertyFlags excluded =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |
        VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
        VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    for (u32 type = 0; type < properties.memoryTypeCount; ++type) {
        const VkMemoryType& memory_type = properties.memoryTypes[type];
        heap_of_type[type] = memory_type.heapIndex;
        if ((memory_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
            (memory_type.propertyFlags & excluded) == 0) {
            keep_types |= 1U << type;
        }
    }
}

KeptBlocks::~KeptBlocks() {
    Close();
}

KeptBlocks* KeptBlocks::Active(VkDevice device) noexcept {
    KeptBlocks* const kept = g_active.load(std::memory_order_acquire);
    return kept != nullptr && kept->device == device ? kept : nullptr;
}

u64 KeptBlocks::SwitchCap() {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_keep_blocks,
                                                       "FRITCHII_KEEP_BLOCKS");
    if (value <= 0) {
        return 0;
    }
    return value == 1 ? DefaultCap : (std::min)(u64{static_cast<u32>(value)} * MiB, MaxCap);
}

void KeptBlocks::Activate() noexcept {
    g_active.store(this, std::memory_order_release);
}

bool KeptBlocks::Keeps(u32 memory_type) const noexcept {
    return memory_type < VK_MAX_MEMORY_TYPES && ((keep_types >> memory_type) & 1) != 0;
}

std::optional<KeptBlocks::Key> KeptBlocks::KeyOf(const VkMemoryAllocateInfo& info) const noexcept {
    const VkDeviceSize size = info.allocationSize;
    if (!Keeps(info.memoryTypeIndex) || size < MinBlock || size > MaxBlock ||
        !std::has_single_bit(size)) {
        return std::nullopt;
    }
    const std::optional<VkMemoryAllocateFlags> flags = BlockFlags(info);
    if (!flags) {
        return std::nullopt;
    }
    return Key{
        .memory_type = info.memoryTypeIndex,
        .flags = *flags,
        .size = size,
    };
}

u64 KeptBlocks::Allowed() const {
    return (std::min)(SwitchCap(), limit.load(std::memory_order_relaxed));
}

void KeptBlocks::AddKept(const Key& key) {
    kept_total.fetch_add(key.size, std::memory_order_relaxed);
    kept_blocks.fetch_add(1, std::memory_order_relaxed);
    kept_by_heap[heap_of_type[key.memory_type]].fetch_add(key.size, std::memory_order_relaxed);
}

void KeptBlocks::RemoveKept(const Key& key) {
    kept_total.fetch_sub(key.size, std::memory_order_relaxed);
    kept_blocks.fetch_sub(1, std::memory_order_relaxed);
    kept_by_heap[heap_of_type[key.memory_type]].fetch_sub(key.size, std::memory_order_relaxed);
}

u32 KeptBlocks::Make(u32 memory_type, VkDeviceSize size, u32 count, VkMemoryAllocateFlags flags) {
    // VMA's blocks: VkMemoryAllocateFlagsInfo with the device address bit when the allocator uses
    // buffer device addresses (VmaBlockVector::CreateBlock).
    const VkMemoryAllocateFlagsInfo flags_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = nullptr,
        .flags = flags,
        .deviceMask = 0,
    };
    const VkMemoryAllocateInfo info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = flags != 0 ? &flags_info : nullptr,
        .allocationSize = size,
        .memoryTypeIndex = memory_type,
    };
    const std::optional<Key> key = KeyOf(info);
    if (!key) {
        return 0;
    }
    u32 made = 0;
    for (; made < count; ++made) {
        VkDeviceMemory memory{};
        if (allocate(device, &info, nullptr, &memory) != VK_SUCCESS) {
            break;
        }
        std::scoped_lock lock{mutex};
        blocks.push_back({memory, *key});
        AddKept(*key);
    }
    return made;
}

bool KeptBlocks::Take(const VkMemoryAllocateInfo& info, VkDeviceMemory& memory) {
    if (kept_total.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    const std::optional<Key> key = KeyOf(info);
    if (!key) {
        return false;
    }
    std::scoped_lock lock{mutex};
    if (closed) {
        return false;
    }
    // The newest first: the one most likely still resident.
    const auto it = std::find_if(blocks.rbegin(), blocks.rend(),
                                 [&key](const Block& block) { return block.key == *key; });
    if (it == blocks.rend()) {
        return false;
    }
    memory = it->memory;
    blocks.erase(std::next(it).base());
    RemoveKept(*key);
    tracked.insert_or_assign(memory, *key);
    ++totals.reused;
    totals.reused_bytes += key->size;
    return true;
}

void KeptBlocks::Track(const VkMemoryAllocateInfo& info, VkDeviceMemory memory) {
    if (const std::optional<Key> key = KeyOf(info)) {
        std::scoped_lock lock{mutex};
        if (!closed) {
            tracked.insert_or_assign(memory, *key);
        }
    }
}

bool KeptBlocks::Keep(VkDeviceMemory memory) {
    const u64 allowed = Allowed();
    std::scoped_lock lock{mutex};
    const auto it = tracked.find(memory);
    if (it == tracked.end()) {
        return false;
    }
    const Key key = it->second;
    tracked.erase(it);
    if (closed || allowed == 0) {
        return false;
    }
    if (kept_total.load(std::memory_order_relaxed) + key.size > allowed) {
        ++totals.refused;
        return false;
    }
    blocks.push_back({memory, key});
    AddKept(key);
    ++totals.kept;
    totals.kept_bytes += key.size;
    return true;
}

void KeptBlocks::SetLimit(u64 bytes) noexcept {
    limit.store(bytes, std::memory_order_relaxed);
}

void KeptBlocks::FreeTimed(VkDeviceMemory memory) {
    const s64 start = NowNs();
    free_memory(device, memory, nullptr);
    CountDriverFree(static_cast<u64>(NowNs() - start));
}

void KeptBlocks::Tick() {
    const u64 cap = SwitchCap();
    if (cap == 0 && kept_total.load(std::memory_order_relaxed) != 0) {
        // Switched off: everything goes at once, so off is off from this frame on.
        Release();
    }
    const u64 allowed = (std::min)(cap, limit.load(std::memory_order_relaxed));
    if (kept_total.load(std::memory_order_relaxed) > allowed) {
        Block block{};
        bool found = false;
        {
            std::scoped_lock lock{mutex};
            const u64 kept = kept_total.load(std::memory_order_relaxed);
            if (!blocks.empty() && kept > allowed) {
                // One a frame, each a driver call: the oldest block of the smallest size that
                // brings the blocks under the limit, else the oldest of the largest.
                const u64 excess = kept - allowed;
                auto chosen = blocks.begin();
                for (auto it = std::next(blocks.begin()); it != blocks.end(); ++it) {
                    const bool fits = it->key.size >= excess;
                    const bool chosen_fits = chosen->key.size >= excess;
                    if (fits ? !chosen_fits || it->key.size < chosen->key.size
                             : !chosen_fits && it->key.size > chosen->key.size) {
                        chosen = it;
                    }
                }
                block = *chosen;
                blocks.erase(chosen);
                RemoveKept(block.key);
                ++totals.released;
                totals.released_bytes += block.key.size;
                found = true;
            }
        }
        if (found) {
            FreeTimed(block.memory);
        }
    }
    Log(cap);
}

u64 KeptBlocks::Release() {
    std::vector<Block> released;
    {
        std::scoped_lock lock{mutex};
        released.swap(blocks);
        for (const Block& block : released) {
            RemoveKept(block.key);
            ++totals.released;
            totals.released_bytes += block.key.size;
        }
    }
    u64 bytes = 0;
    for (const Block& block : released) {
        FreeTimed(block.memory);
        bytes += block.key.size;
    }
    return bytes;
}

void KeptBlocks::Close() {
    {
        std::scoped_lock lock{mutex};
        closed = true;
        tracked.clear();
    }
    KeptBlocks* self = this;
    g_active.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
    Release();
}

void KeptBlocks::CountDriverAllocation(u64 bytes, u64 ns) noexcept {
    driver_allocations.fetch_add(1, std::memory_order_relaxed);
    driver_allocation_bytes.fetch_add(bytes, std::memory_order_relaxed);
    driver_allocation_ns.fetch_add(ns, std::memory_order_relaxed);
    u64 longest = driver_longest_ns.load(std::memory_order_relaxed);
    while (ns > longest &&
           !driver_longest_ns.compare_exchange_weak(longest, ns, std::memory_order_relaxed)) {
    }
}

void KeptBlocks::CountDriverFree(u64 ns) noexcept {
    driver_frees.fetch_add(1, std::memory_order_relaxed);
    driver_free_ns.fetch_add(ns, std::memory_order_relaxed);
}

void KeptBlocks::Log(u64 cap) {
    // Logged while the switch is on, or was set at boot (so a live 0/1 toggle has the line in its
    // off windows too, as the baseline).
    static const bool set_at_boot = std::getenv("FRITCHII_KEEP_BLOCKS") != nullptr;
    const s64 now = NowNs();
    if (last_log_ns == 0) {
        last_log_ns = now;
        return;
    }
    if (now - last_log_ns < 10'000'000'000) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    const u64 allocations = driver_allocations.exchange(0, std::memory_order_relaxed);
    const u64 allocation_bytes = driver_allocation_bytes.exchange(0, std::memory_order_relaxed);
    const u64 allocation_ns = driver_allocation_ns.exchange(0, std::memory_order_relaxed);
    const u64 longest_ns = driver_longest_ns.exchange(0, std::memory_order_relaxed);
    const u64 frees = driver_frees.exchange(0, std::memory_order_relaxed);
    const u64 free_ns = driver_free_ns.exchange(0, std::memory_order_relaxed);
    Totals taken;
    {
        std::scoped_lock lock{mutex};
        taken = std::exchange(totals, Totals{});
    }
    if ((cap == 0 && !set_at_boot) ||
        allocations + frees + taken.reused + taken.kept + taken.released == 0) {
        return;
    }
    const u64 limit_bytes = limit.load(std::memory_order_relaxed);
    const std::string limit_text =
        limit_bytes == ~u64{0} ? std::string{"not set yet"}
                               : fmt::format("{} MiB", limit_bytes / MiB);
    LOG_INFO(Render_Vulkan,
             "Kept blocks in {:.1f} s (FRITCHII_KEEP_BLOCKS cap {} MiB, limit {}): {} driver "
             "allocations under VMA ({} MiB) in {:.1f} ms, longest {:.2f} ms; {} driver frees in "
             "{:.1f} ms; {} blocks reused ({} MiB), {} kept ({} MiB), {} not kept (over the "
             "cap or limit), {} given back ({} MiB); holds {} MiB in {} blocks",
             seconds, cap / MiB, limit_text, allocations,
             allocation_bytes / MiB, static_cast<double>(allocation_ns) / 1e6,
             static_cast<double>(longest_ns) / 1e6, frees, static_cast<double>(free_ns) / 1e6,
             taken.reused, taken.reused_bytes / MiB, taken.kept, taken.kept_bytes / MiB,
             taken.refused, taken.released, taken.released_bytes / MiB, KeptBytes() / MiB,
             kept_blocks.load(std::memory_order_relaxed));
}

} // namespace Vulkan
