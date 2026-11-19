// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

// EDEN_NX2_RUNAHEAD (live; Ocarina of Time 0400486024DEC000 only; fable_cpu_pipeline_plan.md
// section 6): run-ahead with a stamped write log.
//
// A semaphore acquire the split's front answers as met because a release it already sent writes
// a value that meets it (GpuSplit::QueuedReleasePayload, or the stricter check in
// puller.cpp's ReleaseFencesForReadable) is a boundary. At a boundary in mode 1 the front writes
// the released word to guest memory at once instead of waiting for the back to run the release,
// so the guest CPU sees the GPU's progress early and goes on with the next frame. The CPU may then
// overwrite data that draws still queued for the back read. So before the word is written, the
// front takes every guest CPU write collected since the last boundary (EXP35=2's dirty managers)
// with its current bytes: a chunk. The back applies a chunk once it has run everything before the
// previous boundary's release (the chunk's stamp), straight into the host buffers
// (BufferCache::Nx2ApplyCpuBytes), so the draws in between see the bytes as they were when the
// CPU was let go, and the memory tracker never re-reads those ranges from guest memory.
//
// Positions: the frontier G is the newest release whose word the front wrote early. Draws sent
// before G may have their data overwritten by the CPU from then on; draws sent after G can't
// (the CPU was not told they ran). The chunk taken at boundary k holds the writes made between
// boundaries k-1 and k, so it must not reach draws before G(k-1) and must reach every draw after
// it: its stamp is G(k-1). While the back is behind G, it never gathers collected writes itself
// (GPU::InvalidateGPUCache), never compares uniform bank pages with guest memory and never
// streams small uniform buffers from guest memory (BufferCache): guest memory may already hold
// the next frame. Once past G it gathers again, and applies what it gathers at once from guest
// memory (no CPU-modified marks left behind for a later boundary to race).
//
// 0 = off, exactly as before. 1 = on. 2 = verify: at each boundary the front only reads the
// pending writes (nothing is taken from the back) and the acquire goes the old way (drains
// included); when the back reaches the chunk's stamp it compares the chunk's bytes with guest
// memory there (what the old path uploads) and logs the first 16 differences and a count every
// 10 s. 3 = early answers without writing the word (no chunks: the back writes the word in
// order, so the CPU never runs ahead); tells the front's own gain apart from the CPU's.
// EDEN_NX2_RUNAHEAD_MS (live, default 16): early answers only while the back is less than this
// many ms behind the oldest early answer it has not reached; past it the acquire drains as before.
// Needs the default (not delayed) fences, EXP33 = 0, and for mode 1 EXP35 = 2.
//
// Guest stores in flight (nx2_write_track.h): a store's dirty mark is collected before its bytes
// land. A gather that copies bytes takes the in-flight writes' ranges again at the next gather
// (their stores have landed by then), so no write is lost. Verify mode records them per chunk
// and journals every tracked write, so a difference says whether the bytes were in flight at the
// snapshot, rewritten after it (by which thread), or changed by no tracked CPU write.
//
// Predicted boundaries (EXP13 = 3 or 4, the QMD prediction's feed): an acquire the feed answers
// with a queued plain release of a simple QMD (or a copy semaphore) is a boundary too, with the
// predicted payload. Landings of such words are never skipped (the back writes the made value in
// order, which repairs a wrong prediction; the feed turns itself off at the first mismatch).
namespace VideoCommon::Runahead {

inline constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;

enum class Mode : int {
    Off = 0,
    On = 1,
    Verify = 2,
    NoWrite = 3,
};

/// One guest CPU write in a chunk: device bytes [address, address + size), at `offset` in bytes.
struct Range {
    u64 address;
    u32 size;
    u32 offset;
};

struct Chunk {
    u64 stamp{};      ///< The back applies it once it ran everything before this ring position
    u64 frontier{};   ///< The frontier after the boundary that took it (for the log)
    u64 release{};    ///< The boundary's release position (for the log)
    u64 created_ns{}; ///< steady clock
    bool verify{};    ///< Mode 2: compare only
    bool own{};       ///< Gathered by the back itself (due at once)
    u64 epoch{};      ///< Mode 2: the write journal's epoch of the snapshot
    std::vector<Range> ranges;
    std::vector<u8> bytes;
    /// Mode 2: writes in flight at the snapshot (address, size; their bytes may be missing).
    std::vector<std::pair<u64, u64>> in_flight;
};

/// The rasterizer's side (RasterizerVulkan sets them; null without a Vulkan rasterizer).
struct Hooks {
    void* argument{};
    /// Any thread: the guest bytes at a device address, or null when unmapped.
    const u8* (*guest_pointer)(void* argument, u64 device){};
    /// Any thread, lock-free: whether an image may have been registered on these bytes.
    bool (*may_hold_images)(void* argument, u64 device, u64 size){};
    /// Back thread: applies the chunk's bytes to the caches.
    void (*apply)(void* argument, const Chunk& chunk){};
    /// Back thread: the bound vertex and pixel programs' GPU addresses (verify log).
    std::array<u64, 2> (*bound_programs)(void* argument){};
};

/// The oldest unapplied chunk's stamp, ~0 for none: GpuSplit::BackLoop compares its position with
/// it before every entry (one relaxed load).
inline std::atomic<u64> g_apply_at{~u64{0}};
/// The frontier: the newest release position whose word the front wrote early this session.
inline std::atomic<u64> g_frontier{0};
/// Set before the first gather into a chunk this session (GPU::ForEachPendingCpuWrite then asks
/// for the unapplied chunks' ranges too; set before the gather, so a reader that missed the
/// writes in the dirty managers sees it).
inline std::atomic<bool> g_exact_used{false};
/// Set once mode 1 wrote a word early this session: the landing check runs from then on.
inline std::atomic<bool> g_words_written{false};

/// The mode now (the live switch, gated on the title, the split and the fence settings).
[[nodiscard]] Mode CurrentMode();
/// GpuSplit::MaybeStart: the running title.
void SetProgramId(u64 program_id);
void SetHooks(const Hooks* hooks);
/// GpuSplit::Stop: ring positions start over; queued chunks are dropped (their bytes are in
/// guest memory).
void Reset();

// Front thread.

/// What an unmet readable acquire (one that would drain in ReleaseFencesForReadable) waits for,
/// counted in the log line for every mode but 0.
enum class Unmet : u32 {
    QueuedLaterWriter, ///< A queued release meets it, but work sent after it may write the word
    ReleaseRan,        ///< The newest release the front knows already ran: it was overwritten
    WrongPayload,      ///< The newest release the front knows writes a value that doesn't meet it
    UntrackedDispatch, ///< No release known; a native compute launch is queued (its QMD release
                       ///< or reduction may write it)
    UntrackedWriter,   ///< No release known; copies, unranged writers or maps are queued
    Nothing,           ///< Nothing queued can write it: the CPU or another channel will
    Count,
};
void CountUnmet(Unmet kind);

/// Whether the front may answer early now (the back is under EDEN_NX2_RUNAHEAD_MS behind the
/// oldest early answer it has not reached). `consumed`: the back's position.
[[nodiscard]] bool WithinBound(u64 consumed);
/// The front answered an acquire early at ring position `sent` (for the bound).
void NoteEarlyAnswer(u64 sent, bool by_drain_path);
/// The front answers (or would answer) an acquire as met by the release at `release_position`
/// that writes `payload` to the u32 at device `device` (`word`: its guest pointer, null when it
/// must not be written: GPU-written page, syncpoint shim). Mode 1: takes a chunk and, within the
/// bound and when no chunk range may be an image, writes the word and moves the frontier.
/// Mode 2: takes a verify chunk. Returns whether the word was written. `predicted`: the payload
/// is the QMD prediction's (its landing is not skipped).
bool AtBoundary(Core::System& system, u64 device, u8* word, u32 payload, u64 release_position,
                u64 consumed, bool predicted = false);
/// The EXP13 feed answered an acquire (counted).
void NotePredictedAnswer();

// Back thread (and GPU::InvalidateGPUCache's callers).

/// Applies the chunks whose stamp `position` reached (GpuSplit::BackLoop, InvalidateGPUCache).
void ApplyDue(u64 position);
/// GPU::InvalidateGPUCache: true when run-ahead handled it (gathered and applied exactly, or
/// skipped because the caller is behind the frontier); false: take the old path.
bool HandleInvalidate(Core::System& system);
/// BufferCache, on the back: whether guest memory may be newer than what the draw being prepared
/// must see (the back is behind the frontier). Counts `what` when it is.
enum class Hold : u32 {
    BankCompare,   ///< A uniform bank check skipped its guest compares
    FastUniform,   ///< A small uniform bind used the host buffer instead of streaming guest memory
    AlignedStream, ///< A misaligned uniform bind had to stream guest memory anyway (a risk)
    Count,
};
[[nodiscard]] bool BackHolds(Hold what);
/// The query cache, on the back, before it writes a payload release to the u32 at `device`:
/// true when the front already wrote that word for this release or a later one (it must not be
/// written again, over what the CPU did since).
[[nodiscard]] bool LandingSuperseded(u64 device);
/// GPU::ForEachPendingCpuWrite: the ranges of chunks not applied yet and of writes taken again at
/// the next gather (their bytes are the CPU's: a read download must not land over them).
void ForEachUnappliedRange(void (*callback)(void* argument, u64 address, u64 size), void* argument);

} // namespace VideoCommon::Runahead
