// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/page_filter.h"

// EDEN_NX2_MEMSTAGE_STATS=1 (diagnostic, read once; Fable's ideal pipeline design, build step 1):
// counters that size the memory stage before any code moves (fable_design_ideal_pipeline.md 2.4,
// 5). Nothing here changes behaviour. Counted:
// - per draw and per dispatch, the ranges and bytes the buffer side reads from guest memory or
//   resolves for a binding: uniform stream copies, uniform binds from the cached buffer, the native
//   bank check, index, vertex, indirect, storage and texture buffer ranges, texture handle words
//   and handle programs read from constant buffers, storage buffer address words, the QMD; plus
//   the buffer uploads those binds make from guest memory and the query cache's buffer requests;
// - stream writes (constant buffer words, inline uploads, DMA copies) and how many land on pages
//   the texture cache ever registered (page_filter) or holds an image on now (join 2);
// - first draws of a program set and first dispatches of a compute program (join 1);
// - the stream ring running full and staging buffers created (join 3);
// - buffer creates and deletes;
// - draws and clears skipped by render-enable, and how the conditions were decided (section 4);
// - command lists and draws per channel.
// A few lines starting "Memory stage counters (5 s" every 5 s, from the rasterizer's TickFrame.
// Off: one test of g_enabled at each site.
namespace VideoCommon::MemStageStats {

inline const bool g_enabled = [] {
    const char* value = std::getenv("EDEN_NX2_MEMSTAGE_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}();

[[nodiscard]] inline bool Enabled() noexcept {
    return g_enabled;
}

/// What a read of the buffer side is for. Graphics kinds count per draw, compute kinds per dispatch.
enum class Kind : u32 {
    UniformStream,     ///< small uniforms copied from guest memory into the stream (per bind)
    UniformCached,     ///< uniforms bound from the cached host buffer (synchronized)
    BankCheck,         ///< native bank ranges compared with their snapshots
    Index,             ///< index buffer range
    Vertex,            ///< vertex buffer ranges
    Indirect,          ///< indirect draw (and count) buffer ranges
    Storage,           ///< storage buffer ranges
    TextureBuffer,     ///< texture buffer ranges (address from a texture entry)
    HandleWord,        ///< texture handle words read from constant buffers
    HandleProgram,     ///< texture handles computed by a handle program (storage reads)
    StorageAddress,    ///< storage buffer pointer, size and table words read from constant buffers
    CUniformStream,    ///< compute: uniforms streamed
    CUniformCached,    ///< compute: uniforms from the cached buffer
    CStorage,          ///< compute: storage buffer ranges
    CTextureBuffer,    ///< compute: texture buffer ranges
    CHandleWord,       ///< compute: texture handle words
    CHandleProgram,    ///< compute: handle programs
    CStorageAddress,   ///< compute: storage buffer address words
    Qmd,               ///< compute: the launch descriptor (QMD) read from guest memory
    Upload,            ///< buffer uploads from guest memory (UploadMemory copies), any caller
    QueryBuffer,       ///< query cache: buffer cache ranges it obtains (conditional rendering, syncs)
    Count,
};

/// Writes the command stream makes to guest memory (the memory stage would own them).
enum class Write : u32 {
    CbWords,  ///< constant buffer words (CB_DATA, the constant buffer word fast path)
    Inline,   ///< inline uploads (upload engine)
    DmaLinear, ///< DMA copies and fills between linear memory
    DmaImage,  ///< DMA copies between an image and a buffer (the buffer side only)
    Count,
};

constexpr size_t NumKinds = static_cast<size_t>(Kind::Count);
constexpr size_t NumWrites = static_cast<size_t>(Write::Count);
/// Channels with their own counters; any further one counts under the last slot ("other").
constexpr size_t NumChannelSlots = 8;

struct RangeCounter {
    std::atomic<u64> ranges{};
    std::atomic<u64> bytes{};
};

struct WriteCounter {
    std::atomic<u64> writes{};
    std::atomic<u64> bytes{};
    std::atomic<u64> image_pages_ever{}; ///< landed on a page the texture cache ever registered
    std::atomic<u64> image_pages_now{};  ///< landed on a 64 KiB page an image is registered on now
    std::atomic<u64> unmapped{};         ///< no device address (not tested)
};

struct ChannelCounter {
    std::atomic<s32> id{INT_MIN};
    std::atomic<u64> lists{};
    std::atomic<u64> entries{};
    std::atomic<u64> draws{};
    std::atomic<u64> skipped{};
    std::atomic<u64> dispatches{};
};

struct State {
    std::array<RangeCounter, NumKinds> reads;
    std::array<WriteCounter, NumWrites> writes;
    std::array<ChannelCounter, NumChannelSlots> channels;
    std::atomic<s32> back_channel{INT_MIN}; ///< the rasterizer's bound channel (back thread)

    std::atomic<u64> frames{};
    std::atomic<u64> draws{};          ///< draws that reached the rasterizer (direct and indirect)
    std::atomic<u64> draws_skipped{};  ///< draws render-enable skipped
    std::atomic<u64> clears{};
    std::atomic<u64> clears_skipped{};
    std::atomic<u64> dispatches{};
    std::atomic<u64> condition_host{};  ///< render-enable decided by the host (predicated draws)
    std::atomic<u64> condition_true{};  ///< decided on the CPU: draw
    std::atomic<u64> condition_false{}; ///< decided on the CPU: skip

    std::atomic<u64> uploads{};         ///< UploadMemory calls (Kind::Upload counts their copies)
    std::atomic<u64> stream_full{};     ///< stream ring busy, fell back to a staging buffer
    std::atomic<u64> staging_created{}; ///< staging buffers created (no free one), any user
    std::atomic<u64> staging_created_bytes{};
    std::atomic<u64> buffers_created{};
    std::atomic<u64> buffers_created_bytes{};
    std::atomic<u64> buffers_deleted{};
    std::atomic<u64> buffers_deleted_bytes{};
    std::atomic<u64> new_program_sets{};
    std::atomic<u64> new_compute_programs{};

    std::atomic<const PageFilter*> image_ever{};
    std::atomic<const CountingPageFilter*> image_now{};

    std::mutex programs_mutex;
    std::unordered_set<u64> program_sets;
    std::unordered_set<u64> compute_programs;

    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

inline State& Get() {
    static State state;
    return state;
}

inline void Add(std::atomic<u64>& counter, u64 value = 1) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

/// One range of `bytes` read or resolved for `kind`.
inline void Read(Kind kind, u64 bytes, u64 ranges = 1) {
    if (!g_enabled) {
        return;
    }
    RangeCounter& counter = Get().reads[static_cast<size_t>(kind)];
    Add(counter.ranges, ranges);
    Add(counter.bytes, bytes);
}

/// The texture cache's page filters, for the stream write test (the rasterizer, at creation).
inline void SetImagePageFilters(const PageFilter* ever, const CountingPageFilter* now) {
    if (!g_enabled) {
        return;
    }
    Get().image_ever.store(ever, std::memory_order_relaxed);
    Get().image_now.store(now, std::memory_order_relaxed);
}

/// A stream write of `bytes` at device address `device` (0: not mapped), over `span` bytes from
/// there (0: `bytes`; pitched DMA lines). DMA image copies are counted only: the texture cache
/// (the back) writes their memory, not the stream.
inline void StreamWrite(Write write, u64 device, u64 bytes, u64 span = 0) {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    WriteCounter& counter = s.writes[static_cast<size_t>(write)];
    Add(counter.writes);
    Add(counter.bytes, bytes);
    if (write == Write::DmaImage) {
        return;
    }
    if (device == 0) {
        Add(counter.unmapped);
        return;
    }
    const u64 size = span != 0 ? span : bytes;
    if (const PageFilter* ever = s.image_ever.load(std::memory_order_relaxed);
        ever != nullptr && ever->Any(device, size)) {
        Add(counter.image_pages_ever);
    }
    if (const CountingPageFilter* now = s.image_now.load(std::memory_order_relaxed);
        now != nullptr && now->Any(device, size)) {
        Add(counter.image_pages_now);
    }
}

inline ChannelCounter& ChannelSlot(s32 id) {
    State& s = Get();
    for (size_t i = 0; i + 1 < NumChannelSlots; ++i) {
        ChannelCounter& slot = s.channels[i];
        s32 current = slot.id.load(std::memory_order_relaxed);
        if (current == id) {
            return slot;
        }
        if (current == INT_MIN &&
            (slot.id.compare_exchange_strong(current, id, std::memory_order_relaxed) ||
             current == id)) {
            return slot;
        }
    }
    return s.channels[NumChannelSlots - 1];
}

/// The front: a command list of `entries` for `channel`.
inline void ListSubmitted(s32 channel, u64 entries) {
    if (!g_enabled) {
        return;
    }
    ChannelCounter& slot = ChannelSlot(channel);
    Add(slot.lists);
    Add(slot.entries, entries);
}

/// The back: the rasterizer bound `channel`.
inline void ChannelBound(s32 channel) {
    if (!g_enabled) {
        return;
    }
    Get().back_channel.store(channel, std::memory_order_relaxed);
}

/// A draw that reached the rasterizer, or one render-enable skipped.
inline void Draw(bool skipped) {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    ChannelCounter& slot = ChannelSlot(s.back_channel.load(std::memory_order_relaxed));
    if (skipped) {
        Add(s.draws_skipped);
        Add(slot.skipped);
    } else {
        Add(s.draws);
        Add(slot.draws);
    }
}

inline void Clear(bool skipped) {
    if (!g_enabled) {
        return;
    }
    Add(skipped ? Get().clears_skipped : Get().clears);
}

inline void Dispatch() {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    Add(s.dispatches);
    Add(ChannelSlot(s.back_channel.load(std::memory_order_relaxed)).dispatches);
}

/// A render-enable condition: decided by the host, or on the CPU with `execute`.
inline void Condition(bool host, bool execute) {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    Add(host ? s.condition_host : (execute ? s.condition_true : s.condition_false));
}

/// UploadMemory: `copies` ranges, `bytes` in all, read from guest memory.
inline void Uploaded(u64 copies, u64 bytes) {
    if (!g_enabled) {
        return;
    }
    Add(Get().uploads);
    Read(Kind::Upload, bytes, copies);
}

inline void StreamFull() {
    if (!g_enabled) {
        return;
    }
    Add(Get().stream_full);
}

inline void StagingCreated(u64 bytes) {
    if (!g_enabled) {
        return;
    }
    Add(Get().staging_created);
    Add(Get().staging_created_bytes, bytes);
}

inline void BufferCreated(u64 bytes) {
    if (!g_enabled) {
        return;
    }
    Add(Get().buffers_created);
    Add(Get().buffers_created_bytes, bytes);
}

inline void BufferDeleted(u64 bytes) {
    if (!g_enabled) {
        return;
    }
    Add(Get().buffers_deleted);
    Add(Get().buffers_deleted_bytes, bytes);
}

/// A graphics draw's shader set (its stages' unique hashes): counts the first draw of each.
inline void ProgramSet(const std::array<u64, 6>& hashes) {
    if (!g_enabled) {
        return;
    }
    u64 key = 0xcbf29ce484222325ULL;
    for (const u64 hash : hashes) {
        key = (key ^ hash) * 0x100000001b3ULL;
        key ^= key >> 29;
    }
    State& s = Get();
    std::scoped_lock lock{s.programs_mutex};
    if (s.program_sets.insert(key).second) {
        Add(s.new_program_sets);
    }
}

/// A dispatch's compute program: counts the first dispatch of each.
inline void ComputeProgram(u64 hash) {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    std::scoped_lock lock{s.programs_mutex};
    if (s.compute_programs.insert(hash).second) {
        Add(s.new_compute_programs);
    }
}

inline u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

inline double PerEach(u64 value, u64 count) {
    return count != 0 ? static_cast<double>(value) / static_cast<double>(count) : 0.0;
}

/// The rasterizer's TickFrame: counts the frame, logs every 5 s.
inline void TickFrame() {
    if (!g_enabled) {
        return;
    }
    State& s = Get();
    Add(s.frames);
    const auto now = std::chrono::steady_clock::now();
    if (now - s.last < std::chrono::seconds{5}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.last).count();
    s.last = now;

    const u64 frames = Take(s.frames);
    const u64 draws = Take(s.draws);
    const u64 skipped = Take(s.draws_skipped);
    const u64 clears = Take(s.clears);
    const u64 clears_skipped = Take(s.clears_skipped);
    const u64 dispatches = Take(s.dispatches);
    const u64 condition_host = Take(s.condition_host);
    const u64 condition_true = Take(s.condition_true);
    const u64 condition_false = Take(s.condition_false);

    std::string channels;
    for (size_t i = 0; i < NumChannelSlots; ++i) {
        ChannelCounter& slot = s.channels[i];
        const s32 id = slot.id.load(std::memory_order_relaxed);
        const u64 lists = Take(slot.lists);
        const u64 entries = Take(slot.entries);
        const u64 channel_draws = Take(slot.draws);
        const u64 channel_skipped = Take(slot.skipped);
        const u64 channel_dispatches = Take(slot.dispatches);
        if (lists == 0 && channel_draws == 0 && channel_skipped == 0 && channel_dispatches == 0) {
            continue;
        }
        const std::string name = i + 1 == NumChannelSlots
                                     ? std::string{"other"}
                                     : (id == INT_MIN ? std::string{"none"} : fmt::format("{}", id));
        channels += fmt::format("; ch {}: {} lists ({} entries), {} draws, {} skipped, {} dispatches",
                                name, lists, entries, channel_draws, channel_skipped,
                                channel_dispatches);
    }
    LOG_INFO(HW_GPU,
             "Memory stage counters (5 s, {:.1f} s): {} frames, {} draws ({:.1f}/frame), {} skipped "
             "by render-enable ({:.1f}/frame, {:.1f}% of draws); clears {} ({} skipped); "
             "render-enable decided on host {}, on CPU {} draw / {} skip; {} dispatches "
             "({:.1f}/frame){}",
             seconds, frames, draws, PerEach(draws, frames), skipped, PerEach(skipped, frames),
             100.0 * PerEach(skipped, draws + skipped), clears, clears_skipped, condition_host,
             condition_true, condition_false, dispatches, PerEach(dispatches, frames), channels);

    std::array<u64, NumKinds> ranges{};
    std::array<u64, NumKinds> bytes{};
    for (size_t i = 0; i < NumKinds; ++i) {
        ranges[i] = Take(s.reads[i].ranges);
        bytes[i] = Take(s.reads[i].bytes);
    }
    const auto per_draw = [&](Kind kind, u64 count) {
        const size_t i = static_cast<size_t>(kind);
        return fmt::format("{:.2f} / {:.0f} B", PerEach(ranges[i], count),
                           PerEach(bytes[i], count));
    };
    u64 draw_bytes = 0;
    for (size_t i = 0; i <= static_cast<size_t>(Kind::StorageAddress); ++i) {
        draw_bytes += bytes[i];
    }
    u64 dispatch_bytes = 0;
    for (size_t i = static_cast<size_t>(Kind::CUniformStream); i <= static_cast<size_t>(Kind::Qmd);
         ++i) {
        dispatch_bytes += bytes[i];
    }
    LOG_INFO(HW_GPU,
             "Memory stage counters (5 s) per draw, ranges / bytes: uniform stream {}, uniform "
             "cached {}, bank check {}, index {}, vertex {}, indirect {}, storage {}, texture buffer "
             "{}, handle words {}, handle programs {}, storage address words {}; {} KiB in all "
             "({:.0f} B/draw)",
             per_draw(Kind::UniformStream, draws), per_draw(Kind::UniformCached, draws),
             per_draw(Kind::BankCheck, draws), per_draw(Kind::Index, draws),
             per_draw(Kind::Vertex, draws), per_draw(Kind::Indirect, draws),
             per_draw(Kind::Storage, draws), per_draw(Kind::TextureBuffer, draws),
             per_draw(Kind::HandleWord, draws), per_draw(Kind::HandleProgram, draws),
             per_draw(Kind::StorageAddress, draws), draw_bytes >> 10, PerEach(draw_bytes, draws));
    const u64 uploads = Take(s.uploads);
    const size_t upload = static_cast<size_t>(Kind::Upload);
    const size_t query = static_cast<size_t>(Kind::QueryBuffer);
    LOG_INFO(HW_GPU,
             "Memory stage counters (5 s) per dispatch, ranges / bytes: QMD {}, uniform stream {}, "
             "uniform cached {}, storage {}, texture buffer {}, handle words {}, handle programs "
             "{}, storage address words {}; {} KiB in all; buffer uploads from guest memory {} "
             "({} copies, {} KiB, {:.0f} B/draw+dispatch); query cache buffer ranges {} ({} KiB)",
             per_draw(Kind::Qmd, dispatches), per_draw(Kind::CUniformStream, dispatches),
             per_draw(Kind::CUniformCached, dispatches), per_draw(Kind::CStorage, dispatches),
             per_draw(Kind::CTextureBuffer, dispatches), per_draw(Kind::CHandleWord, dispatches),
             per_draw(Kind::CHandleProgram, dispatches),
             per_draw(Kind::CStorageAddress, dispatches), dispatch_bytes >> 10, uploads,
             ranges[upload], bytes[upload] >> 10, PerEach(bytes[upload], draws + dispatches),
             ranges[query], bytes[query] >> 10);

    std::string writes;
    constexpr std::array<const char*, NumWrites> write_names{"constant buffer words", "inline",
                                                            "DMA linear", "DMA image/buffer"};
    for (size_t i = 0; i < NumWrites; ++i) {
        WriteCounter& counter = s.writes[i];
        const u64 count = Take(counter.writes);
        const u64 write_bytes = Take(counter.bytes);
        const u64 ever = Take(counter.image_pages_ever);
        const u64 image_now = Take(counter.image_pages_now);
        const u64 unmapped = Take(counter.unmapped);
        if (static_cast<Write>(i) == Write::DmaImage) {
            writes += fmt::format("; {} {} ({} KiB)", write_names[i], count, write_bytes >> 10);
            continue;
        }
        writes += fmt::format("{}{} {} ({} KiB, image pages ever {} / now {}, unmapped {})",
                              i == 0 ? "" : "; ", write_names[i], count, write_bytes >> 10, ever,
                              image_now, unmapped);
    }
    const u64 created = Take(s.buffers_created);
    const u64 created_bytes = Take(s.buffers_created_bytes);
    const u64 deleted = Take(s.buffers_deleted);
    const u64 deleted_bytes = Take(s.buffers_deleted_bytes);
    const u64 staging = Take(s.staging_created);
    const u64 staging_bytes = Take(s.staging_created_bytes);
    size_t total_sets = 0;
    size_t total_compute = 0;
    {
        std::scoped_lock lock{s.programs_mutex};
        total_sets = s.program_sets.size();
        total_compute = s.compute_programs.size();
    }
    LOG_INFO(HW_GPU,
             "Memory stage counters (5 s) joins: stream writes: {}; first draws of a program set "
             "{} ({} seen), first dispatches of a compute program {} ({} seen); stream ring full "
             "{} (fell to staging), staging buffers created {} ({} KiB); buffers created {} "
             "({:.1f}/s, {} KiB), deleted {} ({:.1f}/s, {} KiB)",
             writes, Take(s.new_program_sets), total_sets, Take(s.new_compute_programs),
             total_compute, Take(s.stream_full), staging, staging_bytes >> 10, created,
             static_cast<double>(created) / seconds, created_bytes >> 10, deleted,
             static_cast<double>(deleted) / seconds, deleted_bytes >> 10);
}

} // namespace VideoCommon::MemStageStats
