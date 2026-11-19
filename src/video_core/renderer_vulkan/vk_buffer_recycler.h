// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <span>
#include <utility>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Scheduler;

/// EXP104 (Fable's ideal pipeline design, step 3b): a pool of VkBuffers (each with its VMA
/// allocation) by size class, for the buffer cache's buffers and the staging pool's buffers.
/// A buffer that would be destroyed is handed here instead, and a buffer that would be created
/// is taken from here when one of its class is free; OoT's buffer cache creates and deletes
/// 40 to 200 buffers a second, each a driver call (and VMA work) on the back thread.
///
/// Exact: an entry is stamped with the scheduler tick current when it was returned (every
/// command that can still name it was recorded at or before that tick), and is handed out
/// again only once the host GPU passed that tick (Scheduler::IsFree). Which VkBuffer backs a
/// range is invisible to the guest; a larger one is addressed by offset and size as today.
/// Evicted entries (older than the age, over the byte cap, or EXP104 switched off) wait for
/// the same check, then are destroyed by the back in its split idle hook when it runs out of
/// work (like EXP99's pool images), or at the frame's end without the split.
///
/// Thread: the buffer cache and the staging pool create and destroy on the back (the memory
/// allocator is externally synchronized), and so does every destroy here. The lists have a
/// mutex anyway, in case a guest-thread read-back path ever takes a staging buffer.
///
/// FRITCHII_MEMSTAGE_SPLIT_ALLOC: each staging pool has its own recycler on its own allocator;
/// the byte caps count all recyclers together.
class BufferRecycler {
public:
    enum class Family : u32 {
        Cache,              ///< buffer cache buffers (device local, the cache's usage flags)
        StagingUpload,      ///< staging buffers by memory usage: separate memory types
        StagingDownload,
        StagingDeviceLocal,
        Count,
    };

    /// The buffer cache's smallest class (Fable: powers of two from 64 KiB).
    static constexpr u32 MinCacheLog2 = 16;
    /// The largest class pooled (32 MiB): rounding bigger buffers up wastes too much.
    static constexpr u32 MaxLog2 = 25;
    /// Bytes held for reuse, all families.
    static constexpr u64 MaxHeldBytes = u64{256} << 20;
    /// Evicted bytes waiting for an idle back before they are destroyed at the frame's end.
    static constexpr u64 MaxDoomedBytes = u64{256} << 20;
    /// Frames an entry may wait to be taken (EXP104 = 1; N > 1 sets N), like the image pool's.
    static constexpr u64 DefaultAgeFrames = 180;

    explicit BufferRecycler(Scheduler& scheduler_);
    ~BufferRecycler();

    BufferRecycler(const BufferRecycler&) = delete;
    BufferRecycler& operator=(const BufferRecycler&) = delete;

    /// The buffer cache's clock (packet stamps with FRITCHII_MEMSTAGE_TICKS), before any use.
    void UseClock(const SeamClock& clock_) {
        std::scoped_lock lock{mutex};
        clock = clock_;
    }

    /// Whether EXP104 is on (live).
    [[nodiscard]] static bool Enabled() noexcept;

    [[nodiscard]] static Family StagingFamily(MemoryUsage usage) noexcept;

    /// A pooled buffer of `family` of class `log2` (at least 2^log2 bytes; its real size in
    /// `bytes`) that the GPU finished with, or an empty one. Counts taken or missed.
    [[nodiscard]] vk::Buffer Take(Family family, u32 log2, u64* bytes = nullptr);

    /// Pools `buffer` (`bytes`: its real size) under class floor(log2(bytes)) and returns true,
    /// leaving `buffer` empty; false, `buffer` untouched, while EXP104 is off or it is too large.
    bool Give(Family family, vk::Buffer& buffer, u64 bytes);

    /// The back, once a frame: ages entries out, and destroys evicted ones the idle hook will not.
    void TickFrame();

    /// The back's split idle hook: destroys up to `max` evicted entries the GPU finished with.
    void DestroyIdle(u32 max);

    [[nodiscard]] bool HasDoomed() const noexcept {
        return doomed_count.load(std::memory_order_relaxed) != 0;
    }

    /// Out of device memory: evicts every entry and destroys those the GPU finished with now.
    /// Returns how many were destroyed.
    size_t DropAll();

private:
    struct Entry {
        vk::Buffer buffer;
        u64 bytes{};
        u64 tick{};  ///< scheduler tick current when returned
        u64 frame{}; ///< frame it was returned in
        Family family{};
    };

    static constexpr size_t NumClasses = 64;
    static constexpr size_t NumLists = static_cast<size_t>(Family::Count) * NumClasses;

    [[nodiscard]] static size_t ListIndex(Family family, u32 log2) noexcept {
        return static_cast<size_t>(family) * NumClasses + log2;
    }

    /// Moves an entry from the lists to the doomed queue (mutex held).
    void Doom(Entry&& entry);
    /// Destroys doomed entries the GPU finished with, oldest first, until at most `keep_bytes`
    /// are left; counted as destroyed here. Returns how many.
    size_t DestroyDoomed(u64 keep_bytes);
    void PublishGauges() noexcept;

    SeamClock clock;
    std::mutex mutex;
    std::array<std::deque<Entry>, NumLists> lists;
    std::deque<Entry> doomed;
    u64 held_bytes{};
    u64 held_entries{};
    u64 doomed_bytes{};
    std::atomic<u64> doomed_count{};
    u64 frame{};
};

/// A buffer cache buffer's VkBuffer: when its Buffer goes (the buffer cache's delayed
/// destruction ring, 8 frames after DeleteBuffer, so every stale use the ring covers today
/// is still covered), the VkBuffer is offered to the pool (EXP104) instead of destroyed.
class RecycledBuffer {
public:
    RecycledBuffer() = default;
    /// Never pooled (the null buffer, sparse-compatible buffers).
    explicit RecycledBuffer(vk::Buffer buffer_) noexcept : buffer{std::move(buffer_)} {}
    RecycledBuffer(vk::Buffer buffer_, BufferRecycler* recycler_, u64 bytes_) noexcept
        : buffer{std::move(buffer_)}, recycler{recycler_}, bytes{bytes_} {}

    RecycledBuffer(const RecycledBuffer&) = delete;
    RecycledBuffer& operator=(const RecycledBuffer&) = delete;

    RecycledBuffer(RecycledBuffer&& rhs) noexcept
        : buffer{std::move(rhs.buffer)}, recycler{std::exchange(rhs.recycler, nullptr)},
          bytes{rhs.bytes} {}

    RecycledBuffer& operator=(RecycledBuffer&& rhs) noexcept {
        if (this != &rhs) {
            Recycle();
            buffer = std::move(rhs.buffer);
            recycler = std::exchange(rhs.recycler, nullptr);
            bytes = rhs.bytes;
        }
        return *this;
    }

    ~RecycledBuffer() {
        Recycle();
    }

    VkBuffer operator*() const noexcept {
        return *buffer;
    }

    explicit operator bool() const noexcept {
        return static_cast<bool>(buffer);
    }

    [[nodiscard]] std::span<u8> Mapped() noexcept {
        return buffer.Mapped();
    }

    [[nodiscard]] std::span<const u8> Mapped() const noexcept {
        return buffer.Mapped();
    }

    [[nodiscard]] bool IsHostCoherent() const noexcept {
        return buffer.IsHostCoherent();
    }

    [[nodiscard]] vk::MemoryLocation Location() const noexcept {
        return buffer.Location();
    }

    void SetObjectNameEXT(const char* name) const {
        buffer.SetObjectNameEXT(name);
    }

    /// The VkBuffer's real size (at least the Buffer's size).
    [[nodiscard]] u64 Bytes() const noexcept {
        return bytes;
    }

private:
    /// Offers the VkBuffer to the pool; when it refuses, the vk::Buffer destroys it as before.
    void Recycle() noexcept {
        if (recycler != nullptr && buffer) {
            recycler->Give(BufferRecycler::Family::Cache, buffer, bytes);
        }
        recycler = nullptr;
    }

    vk::Buffer buffer;
    BufferRecycler* recycler{};
    u64 bytes{};
};

} // namespace Vulkan
