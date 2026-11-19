// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "video_core/nx2_profile.h"

/// The dry counters of work\fable_stage_levelpoints.md 18.4 and the worker load of 19.4. A cut's
/// price is its share (a delay test) times its size: these give the size, the time of exactly the
/// items the cut would remove, not a count times a zone's average. Only with the GPU profile on
/// (EDEN_NX2_GPU_PROFILE >= 1); off, each site pays the profile's own enabled test (a load and a
/// branch). One line per group every 10 s (MaybeReport, from the back thread). Every counter has
/// one writing thread (the back, the front or the Vulkan worker, named at each group), so a
/// plain load and store adds to it; the report only reads. Times are in time stamp ticks, as the
/// profile's zones, converted in the report.
namespace VideoCommon::FritchiiPrice {

[[nodiscard]] inline bool On() noexcept {
    return Nx2Profile::Enabled();
}

[[nodiscard]] inline u64 Ticks() noexcept {
    return Nx2Profile::Ticks();
}

using Counter = std::atomic<u64>;

/// One writer: a plain load and store.
inline void Add(Counter& counter, u64 amount) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

template <size_t N>
struct Group {
    std::array<Counter, N> count{};
    std::array<Counter, N> ticks{};
    void Note(size_t index, u64 amount, u64 elapsed) noexcept {
        Add(count[index], amount);
        Add(ticks[index], elapsed);
    }
};

// ---------------------------------------------------------------------------------------------
// Row 1: uniform binds by verdict (the back; BindSmallUniformReusing's compare).

enum class Uniform : u32 {
    Changed,   ///< the compare found other bytes (the copy must be made)
    Unflagged, ///< same bytes, same range as the slot's last bind, no write flag since: the cut
    Flagged,   ///< same bytes, but a write flag was raised on its pages since the last bind
    Moved,     ///< same bytes, but another range than the slot's last bind (or none before)
    NoFlags,   ///< same bytes, the bank marks are off (FRITCHII_BANK_CENSUS or _FLAGGED unset)
    /// fable_b 14.4 cut 7's split (dry): as Unflagged, but the buffer cache's tracker holds the
    /// range's pages CPU-modified, so a guest store there traps nothing and raises no flag: an
    /// unflagged verdict does not prove the bytes untouched (not skippable without watching).
    UnflaggedCpuModified,
    Count,
};
inline Group<static_cast<size_t>(Uniform::Count)> g_uniform;
/// The bind NoteUniformBind is called for: its range's pages are CPU-modified in the buffer
/// cache's tracker (BufferCache::BindHostGraphicsUniformBuffer sets it while On(), before the
/// small-uniform path).
inline thread_local bool t_uniform_cpu_modified{};

/// "Flagged since" for rows 1 and 2 uses Common::PriceGenSum (common/fritchii_page_flags.h): page
/// write generations raised by every mark of the two flags FRITCHII_BANK_FLAGGED takes. The
/// marks are only made while the bank marks are on (Common::BankMarks: FRITCHII_BANK_CENSUS or
/// FRITCHII_BANK_FLAGGED non-zero); with them off the verdict is "marks off".

/// A uniform bind's compare took `elapsed` and found the bytes `same`; `sum` is GenSum of its
/// range read before the compare; `slot_key` = stage * 32 + constant buffer index.
void NoteUniformBind(u32 slot_key, u64 device_addr, u32 size, bool same, u64 elapsed,
                     u64 sum) noexcept;

// ---------------------------------------------------------------------------------------------
// Row 2: draws by their handle range since that pipeline's last draw (the back).

enum class Handles : u32 {
    Unflagged, ///< same constant buffer range and no write flag on its pages: the cut
    Flagged,   ///< same range, written since
    Moved,     ///< another range (a constant buffer rebound) or the pipeline's first draw
    NoFlags,   ///< the bank marks are off
    Count,
};
/// [0] the seam's handle fill (FillHandleValues, EXP123 = 3), [1] ConfigureImpl's handle loop
/// (EXP123 below 3: the words are read there, with the samplers and views' indices).
inline std::array<Group<static_cast<size_t>(Handles::Count)>, 2> g_handles;

/// A draw's verdict from its handle range's key (the constant buffer addresses its handle words
/// come from) and GenSum over them, against the pipeline's last draw (`last_key`, `last_sum`,
/// updated).
[[nodiscard]] Handles ClassifyHandles(u64 key, u64 sum, u64& last_key, u64& last_sum) noexcept;

// ---------------------------------------------------------------------------------------------
// Row 3: constant buffer word calls on the back by run length, the write half's time
// (Maxwell3D::CallConstBufferWords and ProcessCBMultiData past the BackServed test).

inline constexpr size_t CbBuckets = 6; ///< 1, 2, 3-4, 5-8, 9-16, 17 and more words
inline Group<CbBuckets> g_cb_runs;
[[nodiscard]] inline size_t CbBucket(u32 words) noexcept {
    return words <= 1 ? 0 : words == 2 ? 1 : words <= 4 ? 2 : words <= 8 ? 3 : words <= 16 ? 4 : 5;
}

/// The write half: from construction to destruction, on the back thread, profile on.
class CbWriteScope {
public:
    explicit CbWriteScope(u32 words_) {
        if (On() && GpuSplit::OnBackThread()) [[unlikely]] {
            words = words_;
            start = Ticks();
        }
    }
    ~CbWriteScope() {
        if (start != 0) [[unlikely]] {
            g_cb_runs.Note(CbBucket(words), 1, Ticks() - start);
            Add(g_cb_words[CbBucket(words)], words);
        }
    }
    CbWriteScope(const CbWriteScope&) = delete;
    CbWriteScope& operator=(const CbWriteScope&) = delete;

    static inline std::array<Counter, CbBuckets> g_cb_words{};

private:
    u64 start{};
    u32 words{};
};

// ---------------------------------------------------------------------------------------------
// Row 4: the back's ring entries by kind (GpuSplit::Execute): entries, and the profile's
// "method" zone time each kind added (the zone's own estimate: sampled for 3D methods).

enum class Entry : u32 {
    CbRun,       ///< 3D constant buffer data words in a row (CallConstBufferWords; entries = words)
    Method3D,    ///< another single 3D method
    MethodOther, ///< a single method of another engine
    Multi,       ///< a multi-word method (uploads, macro words)
    Call,        ///< a deferred call (draws, fences, flushes are calls or methods' children)
    Pairs,       ///< EDEN_NX2_RING_PAIRS blocks (entries = pairs)
    StateDelta,  ///< EXP22 state deltas
    DrawBatch,   ///< EXP22 draw batches
    CbWords,     ///< FRITCHII_CB_WORDS_BLOCK blocks (entries = words)
    Other,       ///< engine switches, ring wraps
    Count,
};
inline Group<static_cast<size_t>(Entry::Count)> g_entries;

// ---------------------------------------------------------------------------------------------
// Row 2b (fable_b_one_follower 23.1 cut 4, E; dry): unlent draws by their handle values against
// the same pipeline's last unlent draw (OneFollowerDraw::FillRing's fill, timed).

enum class HandleValues : u32 {
    Changed,         ///< other values than the pipeline's last draw (or its first draw)
    SameKeep,        ///< same values, same ranges, no flag on their pages, pages tracked: the cut
    SameCpuModified, ///< as SameKeep, but a page is CPU-modified (guest stores raise no flag)
    SameFlagged,     ///< same values, same ranges, a page flagged since (page, not word, grain)
    SameMoved,       ///< same values from other ranges (a constant buffer rebound)
    Program,         ///< a handle program (reads storage): no range key
    /// Must be 0: same ranges, unflagged, tracked, yet other values (a keep would be stale).
    KeepWouldDiffer,
    Count,
};
inline Group<static_cast<size_t>(HandleValues::Count)> g_handle_values;
/// E, at an unlent draw's handle fill (GPU profile on).
void NoteHandleValues(HandleValues verdict, u64 ticks) noexcept;

// ---------------------------------------------------------------------------------------------
// Row 4b (fable_b_one_follower 22.3 step 2 and 23, the back; dry): the "3D single" entries of row
// 4 by method. Each costs what Execute ran for it in the "method" zone: its own handler and the
// state writes sunk before it, which it consumes first (ConsumeSink); `sunk` counts those.

inline constexpr size_t Method3DSlots = 0x1000;
struct Method3D {
    std::array<Counter, Method3DSlots> count{};
    std::array<Counter, Method3DSlots> ticks{};
    std::array<Counter, Method3DSlots> sunk{};
};
inline Method3D g_method3d;
/// The back, a single 3D method entry: its method, the zone time it added, the sunk writes it
/// consumed.
inline void NoteMethod3D(u32 method, u64 ticks, u64 sunk) noexcept {
    const size_t slot = method < Method3DSlots ? method : Method3DSlots - 1;
    Add(g_method3d.count[slot], 1);
    Add(g_method3d.ticks[slot], ticks);
    Add(g_method3d.sunk[slot], sunk);
}

// ---------------------------------------------------------------------------------------------
// Front waits split (rows 7, 8): until the back reached the wait's target, then after (the
// back's position stores are stamped; the front looks the target up when its wait ends).

/// The back: it stored `position` as consumed (GpuSplit::BackLoop), profile on.
void BackReached(u64 position) noexcept;
/// When the back first stored a position at or past `target` (Ticks), if the stamps still hold
/// it.
[[nodiscard]] bool ReachedAt(u64 target, u64& ticks) noexcept;

/// Sites: the drain sites (DrainSite), then the suspended acquires.
inline constexpr size_t WaitSites = static_cast<size_t>(DrainSite::Count) + 1;
inline constexpr size_t AcquireSite = static_cast<size_t>(DrainSite::Count);
struct WaitSplit {
    std::array<Counter, WaitSites> count{};
    std::array<Counter, WaitSites> until{};
    std::array<Counter, WaitSites> after{};
    std::array<Counter, WaitSites> unknown{}; ///< target no longer in the stamps: all "until"
};
inline WaitSplit g_waits;

/// The front: a wait at `site` for `target` from `start` (Ticks) to now ended.
void NoteFrontWait(size_t site, u64 target, u64 start) noexcept;

// ---------------------------------------------------------------------------------------------
// Worker load (19.4): the Vulkan worker's busy time (its chunks, the worker writes), and on the
// back the time inside Scheduler::Record calls (lambda capture, chunk append, a full chunk's
// hand-off), inside draws and outside, sampled about one call in 8.

struct WorkerLoad {
    std::array<Counter, 2> chunks{}; ///< [0] record only, [1] with a submit
    std::array<Counter, 2> busy{};   ///< ExecuteAll (and the command buffer reallocation)
    Counter delay_ticks{};           ///< FRITCHII_DELAY_WORKER_US's spins (not in busy)
};
inline WorkerLoad g_worker;

struct BackRecord {
    std::array<Counter, 2> calls{};  ///< [0] outside a draw, [1] inside Draw/DrawIndirect
    std::array<Counter, 2> ticks{};  ///< estimated from the sampled calls
    Counter draws{};
    Counter draw_ticks{};            ///< Draw/DrawIndirect inclusive, outside the D spin
};
inline BackRecord g_record;
inline thread_local u32 t_draw_depth{};
inline thread_local u32 t_record_left{1};
inline thread_local u32 t_record_weight{1};
inline thread_local u32 t_record_seed{0x9E3779B9};

/// Scheduler::RecordCommand on the back: one in about 8 calls timed, standing for the rest.
class RecordScope {
public:
    RecordScope() {
        if (On() && GpuSplit::OnBackThread()) [[unlikely]] {
            Begin();
        }
    }
    ~RecordScope() {
        if (start != 0) [[unlikely]] {
            End();
        }
    }
    RecordScope(const RecordScope&) = delete;
    RecordScope& operator=(const RecordScope&) = delete;

private:
    void Begin() noexcept;
    void End() noexcept;
    u64 start{};
    u32 weight{};
    u32 in_draw{};
};

/// RasterizerVulkan::Draw and DrawIndirect on the back: the draw path's whole time.
class DrawScope {
public:
    DrawScope() {
        if (On() && GpuSplit::OnBackThread()) [[unlikely]] {
            ++t_draw_depth;
            start = Ticks();
        }
    }
    ~DrawScope() {
        if (start != 0) [[unlikely]] {
            --t_draw_depth;
            if (t_draw_depth == 0) {
                Add(g_record.draws, 1);
                Add(g_record.draw_ticks, Ticks() - start);
            }
        }
    }
    DrawScope(const DrawScope&) = delete;
    DrawScope& operator=(const DrawScope&) = delete;

private:
    u64 start{};
};

/// The back, between batches with the profile on: logs the groups every 10 s.
void MaybeReport();

} // namespace VideoCommon::FritchiiPrice
