// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

// EDEN_NX2_BUFFER_CHURN_STATS=1 (live, diagnostic; Fable's ideal pipeline design 8.3, "not
// established: why the cache churns"): every buffer the buffer cache creates or deletes, by cause,
// and every staging buffer the staging pool creates, by caller. Each is a driver call (and VMA
// work) on the GPU back thread. Read on first use (SwitchValue): nx2_switches.txt is applied after
// static initialization. "Buffer churn (5 s" every 5 s from the rasterizer's TickFrame. Off: one
// relaxed load at each site.
namespace VideoCommon::BufferChurnStats {

[[nodiscard]] inline bool Enabled() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_buffer_churn_stats,
                                            "EDEN_NX2_BUFFER_CHURN_STATS", 0) > 0;
}

/// Why BufferCache::CreateBuffer made a buffer.
enum class Created : u32 {
    NewRange, ///< no cached buffer overlapped the range
    Grow,     ///< one cached buffer overlapped: it is replaced by a larger one
    Join,     ///< two or more cached buffers overlapped: they are joined into one
    Inline,   ///< the inline index buffer (made or grown)
    Count,
};

/// Why a cached buffer went. There is no deletion on unmap in this tree: unmapped ranges keep
/// their buffers until a join or the collector takes them.
enum class Deleted : u32 {
    Join,      ///< joined into a new buffer (BufferCache::JoinOverlap)
    Collector, ///< evicted by the garbage collector (BufferCache::RunGarbageCollector)
    Inline,    ///< the inline index buffer, replaced by a larger one
    Count,
};

/// Who asked the staging pool for the buffer it had to create.
enum class StagingCaller : u32 {
    BufferUpload,    ///< buffer cache uploads (BufferCacheRuntime::UploadStagingBuffer)
    BufferDownload,  ///< buffer cache downloads and read-backs
    UniformStream,   ///< small uniforms streamed when the stream ring was busy
    TextureUpload,   ///< texture cache uploads
    TextureDownload, ///< texture cache downloads
    Query,           ///< query cache
    ComputePass,     ///< compute passes (quad and uint8 index conversion, vertex repack)
    Rasterizer,      ///< the rasterizer (mesh emulation, clears)
    Other,
    Count,
};

constexpr size_t NumCreated = static_cast<size_t>(Created::Count);
constexpr size_t NumDeleted = static_cast<size_t>(Deleted::Count);
constexpr size_t NumCallers = static_cast<size_t>(StagingCaller::Count);

struct Counter {
    std::atomic<u64> count{};
    std::atomic<u64> bytes{};

    void Add(u64 size) noexcept {
        count.fetch_add(1, std::memory_order_relaxed);
        bytes.fetch_add(size, std::memory_order_relaxed);
    }
};

struct State {
    std::array<Counter, NumCreated> created;
    std::array<Counter, NumDeleted> deleted;
    std::array<Counter, NumCallers> staging;
    std::atomic<u64> staging_stream_full{}; ///< of those, made because the stream ring was busy
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

inline State& Get() {
    static State state;
    return state;
}

/// EXP104's pool (renderer_vulkan/vk_buffer_recycler.h). Counted whenever the pool is used,
/// logged on the churn line.
enum class PoolGroup : u32 {
    Cache,   ///< the buffer cache's buffers
    Staging, ///< staging buffers (upload, download, device local)
    Count,
};

struct PoolCounters {
    std::atomic<u64> taken{};    ///< creates served from the pool (no driver call)
    std::atomic<u64> missed{};   ///< creates the pool could not serve (a driver call)
    std::atomic<u64> returned{}; ///< buffers given to the pool instead of destroyed
    std::atomic<u64> refused{};  ///< too large to pool: destroyed as before
    std::atomic<u64> evicted{};  ///< left (age, byte cap, EXP104 off), to be destroyed
};

struct Pool {
    std::array<PoolCounters, static_cast<size_t>(PoolGroup::Count)> group;
    std::atomic<u64> destroyed_idle{}; ///< evicted buffers destroyed in the back's idle wait
    std::atomic<u64> destroyed_here{}; ///< ...at the frame's end (no idle wait, or too many)
    std::atomic<u64> held_bytes{};     ///< gauges, kept by the pool
    std::atomic<u64> held_entries{};
    std::atomic<u64> doomed_bytes{};   ///< evicted, not destroyed yet
};

inline Pool& GetPool() {
    static Pool pool;
    return pool;
}

inline void PoolAdd(std::atomic<u64>& counter, u64 value = 1) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

inline void BufferCreated(Created cause, u64 bytes) {
    if (!Enabled()) {
        return;
    }
    Get().created[static_cast<size_t>(cause)].Add(bytes);
}

inline void BufferDeleted(Deleted cause, u64 bytes) {
    if (!Enabled()) {
        return;
    }
    Get().deleted[static_cast<size_t>(cause)].Add(bytes);
}

/// A staging buffer the pool had to create for `caller`; `stream_full`: the request wanted the
/// stream ring, which was busy.
inline void StagingCreated(StagingCaller caller, u64 bytes, bool stream_full) {
    if (!Enabled()) {
        return;
    }
    Get().staging[static_cast<size_t>(caller)].Add(bytes);
    if (stream_full) {
        Get().staging_stream_full.fetch_add(1, std::memory_order_relaxed);
    }
}

namespace Detail {
struct Taken {
    u64 count;
    u64 bytes;
};

inline Taken Take(Counter& counter) {
    return {counter.count.exchange(0, std::memory_order_relaxed),
            counter.bytes.exchange(0, std::memory_order_relaxed)};
}

/// "N/s (K KiB)": the rate over `seconds` and the average size.
inline std::string Rate(const Taken& taken, double seconds) {
    const double average_kib =
        taken.count != 0 ? static_cast<double>(taken.bytes) / 1024.0 / taken.count : 0.0;
    return fmt::format("{:.1f}/s ({:.0f} KiB)", taken.count / seconds, average_kib);
}

/// The causes of one family, "name N/s (K KiB)" each, and their sum.
template <size_t N>
inline std::string Family(std::array<Counter, N>& counters, const char* const (&names)[N],
                          double seconds, Taken& total) {
    std::string text;
    for (size_t i = 0; i < N; ++i) {
        const Taken taken = Take(counters[i]);
        total.count += taken.count;
        total.bytes += taken.bytes;
        if (!text.empty()) {
            text += ", ";
        }
        text += fmt::format("{} {}", names[i], Rate(taken, seconds));
    }
    return text;
}
} // namespace Detail

/// The rasterizer's TickFrame (the back): logs every 5 s while on.
inline void TickFrame() {
    if (!Enabled()) {
        return;
    }
    State& s = Get();
    const auto now = std::chrono::steady_clock::now();
    if (now - s.last < std::chrono::seconds{5}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.last).count();
    s.last = now;

    static constexpr const char* created_names[NumCreated] = {"new range", "grow", "join",
                                                              "inline"};
    static constexpr const char* deleted_names[NumDeleted] = {"join", "collector", "inline"};
    static constexpr const char* caller_names[NumCallers] = {
        "buffer upload", "buffer download", "uniform stream", "texture upload",
        "texture download", "query", "compute pass", "rasterizer", "other"};
    Detail::Taken created{};
    Detail::Taken deleted{};
    Detail::Taken staging{};
    const std::string created_text = Detail::Family(s.created, created_names, seconds, created);
    const std::string deleted_text = Detail::Family(s.deleted, deleted_names, seconds, deleted);
    const std::string staging_text = Detail::Family(s.staging, caller_names, seconds, staging);
    const u64 stream_full = s.staging_stream_full.exchange(0, std::memory_order_relaxed);

    // EXP104: per group taken / missed / returned / refused / evicted per second.
    Pool& pool = GetPool();
    std::string pool_text;
    static constexpr const char* group_names[] = {"cache", "staging"};
    for (size_t i = 0; i < pool.group.size(); ++i) {
        PoolCounters& group = pool.group[i];
        const auto rate = [seconds](std::atomic<u64>& counter) {
            return counter.exchange(0, std::memory_order_relaxed) / seconds;
        };
        const double taken = rate(group.taken);
        const double missed = rate(group.missed);
        const double returned = rate(group.returned);
        const double refused = rate(group.refused);
        const double evicted = rate(group.evicted);
        pool_text += fmt::format("{} taken {:.1f}/s, missed {:.1f}/s, returned {:.1f}/s, "
                                 "refused {:.1f}/s, evicted {:.1f}/s; ",
                                 group_names[i], taken, missed, returned, refused, evicted);
    }
    const double destroyed_idle =
        pool.destroyed_idle.exchange(0, std::memory_order_relaxed) / seconds;
    const double destroyed_here =
        pool.destroyed_here.exchange(0, std::memory_order_relaxed) / seconds;
    pool_text += fmt::format(
        "destroyed while idle {:.1f}/s, here {:.1f}/s; held {} buffers, {:.1f} MiB ({:.1f} MiB "
        "waiting to be destroyed)",
        destroyed_idle, destroyed_here, pool.held_entries.load(std::memory_order_relaxed),
        pool.held_bytes.load(std::memory_order_relaxed) / 1048576.0,
        pool.doomed_bytes.load(std::memory_order_relaxed) / 1048576.0);

    LOG_INFO(HW_GPU,
             "Buffer churn (5 s, {:.1f} s): created {} [{}]; deleted {} [{}] (no unmap path); "
             "staging created {} [{}], {:.1f}/s of them as the stream ring was busy; "
             "pool (EXP104={}): {}",
             seconds, Detail::Rate(created, seconds), created_text, Detail::Rate(deleted, seconds),
             deleted_text, Detail::Rate(staging, seconds), staging_text, stream_full / seconds,
             VideoCore::Nx2Debug::Experiment(104), pool_text);
}

} // namespace VideoCommon::BufferChurnStats
