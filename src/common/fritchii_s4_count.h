// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"

/// FRITCHII_S4_COUNT (live, default 0): the dry counter that decides whether S4 (uniform binds
/// served from a per-list-entry page mirror, work\fable_b2_answers.md 3.3) is worth building.
/// The counting itself is video_core/fritchii_s4_count.cpp; this part is in common because the
/// writers of game memory that tell the causes apart live in common and core too (the write
/// ring's Note, the buffer cache's downloads, MemoryManager's word writes).
///
/// Off: every hook below is one relaxed load of g_on and a branch.
namespace Common::S4Count {

/// True while FRITCHII_S4_COUNT is non-zero (kept in step by VideoCommon::S4Count::Poll, once a
/// draw on the engine thread).
inline std::atomic<bool> g_on{false};
/// The switch's value while on (1 or 2).
inline std::atomic<int> g_mode{0};

[[nodiscard]] inline bool On() noexcept {
    return g_on.load(std::memory_order_relaxed);
}

/// Who wants the front's list entry marks (VideoCommon::S4Count::FrontEntry), one bit each: the
/// counter while on, and FRITCHII_BIND_SKIP (its in-entry step, VideoCommon::S4Count::
/// SetBindSkipEntries). The front tests this once per list entry.
inline constexpr u32 EntryUserCounter = 1;
inline constexpr u32 EntryUserBindSkip = 2;
inline std::atomic<u32> g_entry_users{0};

[[nodiscard]] inline bool EntryMarks() noexcept {
    return g_entry_users.load(std::memory_order_relaxed) != 0;
}

/// Set on the engine thread (E: the thread that runs the stream's methods in order and fills the
/// draws) by the counter: its writes to game memory are stream uploads (constant buffer words,
/// inline uploads, the 3D engine's semaphore words), the ones a page mirror would apply in order.
inline thread_local bool t_engine = false;
/// Set on the render thread (R, one follower) by the counter: its writes land GPU results.
inline thread_local bool t_render = false;

/// Who wrote a page, for the cause of a change between two binds.
enum class Writer : u32 {
    Stream, ///< E's own writes, in stream order (constant buffer words, uploads, releases)
    Gpu,    ///< GPU results landing in game memory (buffer downloads, report and semaphore words
            ///< written off E)
    Other,  ///< emulator writes on any other thread (services, unsafe stores)
    Count,
};

/// Write generations per writer: a hash of the device page to 65,536 buckets, raised by each
/// write while on. A collision only makes a range look written by that writer.
inline constexpr size_t GenBuckets = 65536;
inline std::array<std::array<std::atomic<u32>, GenBuckets>, static_cast<size_t>(Writer::Count)>
    g_gens{};

[[nodiscard]] inline size_t GenBucket(u64 page) noexcept {
    return static_cast<size_t>((page * 0x9E3779B97F4A7C15ULL) >> 48);
}

inline void MarkGens(Writer writer, u64 address, u64 size) noexcept {
    if (size == 0) {
        return;
    }
    auto& gens = g_gens[static_cast<size_t>(writer)];
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    // A huge write (over 1 MiB) raises its first 256 pages only: no uniform range is that large,
    // and the counter must not stall a big download.
    const u64 stop = (last - first >= 256) ? first + 255 : last;
    for (u64 page = first; page <= stop; ++page) {
        gens[GenBucket(page)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// The generations of [address, address + size) summed for one writer (they only grow: an equal
/// sum over the same range means no write by that writer since).
[[nodiscard]] inline u64 GenSum(Writer writer, u64 address, u64 size) noexcept {
    if (size == 0) {
        return 0;
    }
    const auto& gens = g_gens[static_cast<size_t>(writer)];
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    u64 sum = 0;
    for (u64 page = first; page <= last; ++page) {
        sum += gens[GenBucket(page)].load(std::memory_order_relaxed);
    }
    return sum;
}

/// An emulator write to game memory, after its bytes landed (the write ring's Note): E's own
/// writes are stream uploads, R's are GPU results, every other thread's are "other".
inline void NoteWrite(u64 address, u64 size) noexcept {
    MarkGens(t_engine ? Writer::Stream : (t_render ? Writer::Gpu : Writer::Other), address, size);
}

/// A GPU word (report, semaphore) written to game memory: on E a stream write in order, on any
/// other thread (R landing a report) a GPU result.
inline void NoteGpuWord(u64 address, u64 size) noexcept {
    MarkGens(t_engine ? Writer::Stream : Writer::Gpu, address, size);
}

} // namespace Common::S4Count
