// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Common {
class SpinRecursiveMutex;
}

namespace VideoCommon {

struct MemStageShadowEngine;

/// The memory stage's thread, GPUMem, and the back's link to it (memory_stage_link.cpp).
class MemoryStageLink {
public:
    using Function = void (*)(void* argument);

    enum class HandOff : u8 { Draw, Dispatch, Idle, Served, Count };

    /// Counters that must read 0 (memory_stage_link.cpp).
    enum class Check : u8 {
        AllocatorUser,
        StagingUser,
        PushUser,
        BudgetUser,
        StageScheduler,
        PushNowhere,
        PushWhilePending,
        PlayerOnStage,
        FrontWaitOnStage,
        HandOffUnlocked,
        BackInCache,
        BackInvalidation,
        PacketMissing,
        PacketUnplayed,
        RingFull,
        DrawCount,
        PacketOrphan,
        StageException,
        PacketMismatch,
        BackHandles,
        SyncAhead,
        ServedAheadJoin,
        FutureWriteUnderImage,
        ServedOverStageWrite,
        FactsKeyCollision,
        ListReadBack,
        ListArgsMissing,
        ListArgsUnasked,
        GatherMissing,
        GatherOrphan,
        GatherDrawBetween,
        ReleaseMissing,
        ReleaseOrphan,
        ReleaseDrawBetween,
        RunOverlap,
        PassQuerySync,
        PassFlushes,
        PassLate,
        PassCommand,
        PassWaitMissing,
        PassListedFreed,
        PassListedLeft,
        PassSyncTarget,
        PassFenceMissing,
        SetupKeepDisagree,
        Count,
    };

    /// FRITCHII_MEMSTAGE_RELEASE: what became of a puller semaphore release closure GPUMem reached
    /// (StageCacheCall::SemaphoreRelease; counted).
    enum class ReleaseOutcome : u8 {
        Served,   ///< the back ran it at its entry on GPUMem's request, GPUMem level, no run
        Closed,   ///< level: the switch off or the window closed (a run starts)
        NextHard, ///< level: a hard closure follows, which the release's run takes in
        NoRoom,   ///< level: no room for its record
        Stopping, ///< level: GPUMem is stopping
        InRun,    ///< inside a level run (followed; the back runs it lent)
        Count,
    };

    /// FRITCHII_MEMSTAGE_GATHER: what became of a gather (3D invalidate_texture_data_cache) GPUMem's
    /// copy reached (counted).
    enum class GatherOutcome : u8 {
        Served,       ///< the back ran it at its entry on GPUMem's request, GPUMem level, no run
        Closed,       ///< level: the switch off or the window closed (a run starts)
        NotFirst,     ///< level: not a lone Direct method, or a draw before it in its entry
        DeferredDraw, ///< level: the back had a deferred draw it would issue first at the entry
        NextHard,     ///< level: a hard closure follows, which the gather's run takes in
        NoBackend,    ///< level: no ring backend runs gathers (not the Vulkan rasterizer)
        NoRoom,       ///< level: no room for its record
        Stopping,     ///< level: GPUMem is stopping
        InRun,        ///< inside a level run (followed; the back runs it lent)
        Late,         ///< in an entry that was hard already
        Count,
    };

    /// FRITCHII_MEMSTAGE_LIST_DRAWS: what became of a list draw GPUMem's copy issued (counted).
    enum class ListDraw : u8 {
        Served,   ///< filled on GPUMem, the list words' buffer in the packet
        Closed,   ///< level: the window closed (AHEAD 0) or the switch off
        ReadBack, ///< level: the back reads the two words on the CPU (DrawListWords' on_cpu)
        Refused,  ///< level: the fill refused it (facts miss, texel buffers, no room)
        Forced,   ///< level: every draw is (a frame dump or a trace, g_ring_level_draws)
        Count,
    };

    /// Step 7: why GPUMem came level inside its window (StageLevel, counted).
    enum class LevelReason : u8 {
        Copy,        ///< a full copy of the back's engine (a new or stale copy)
        WritingDraw, ///< a draw whose fill marks buffers GPU-written (storage writes, xfb)
        Claim,       ///< collected constant buffer ranges over GPU-written buffer data
        Gather,      ///< a gather the back runs at its entry (FRITCHII_MEMSTAGE_GATHER)
        Release,     ///< a semaphore release the back runs at its entry (FRITCHII_MEMSTAGE_RELEASE)
        Count,
    };

    /// THREAD = 2: what a served call of the back is for (counted, not a must-be-0).
    enum class Served : u8 {
        FlushRegion,
        FlushRegionForRead,
        Gather,
        Invalidation,
        InnerInvalidation,
        MustFlush,
        Fallback,
        Count,
    };

    /// FRITCHII_MEMSTAGE_PASS bit 2 (fable_stage_p3 4.2): one maximal run of report pages GPUMem
    /// resolved at a sync point, [begin, begin + bytes) of device memory, and the buffer the
    /// back's query result sync copies into there (a VkBuffer, as u64) at `offset` for `begin`.
    struct SyncTarget {
        u64 begin;
        u64 bytes;
        u64 buffer;
        u64 offset;
    };

    /// THREAD = 2: what the Vulkan side does for the stage (memory_stage_link.cpp).
    struct RingBackend {
        void* context{};
        /// GPUMem: fills the draw the stage's copy issues and puts its packet in ring B. False:
        /// the stage cannot serve it (no facts, texel buffers): the draw is level.
        bool (*fill_draw)(void* context, MemStageShadowEngine& copy, bool indexed, u64 position,
                          u64 seq, u64 hash,
                          std::vector<std::pair<u64, std::size_t>>& ranges){};
        void (*bind_channel)(void* context, s32 bind_id){};
        void (*bind_uniform)(void* context, size_t stage, u32 slot, u64 gpu_address, u32 size,
                             bool enabled){};
        /// The back at a run's start: the buffer half of the stage's collected ranges.
        void (*apply_ranges)(void* context, const std::pair<u64, std::size_t>* ranges,
                             size_t count){};
        /// GPUMem, the back parked: the buffer cache's idle work.
        void (*idle)(void* context){};
        /// The back at a run's end: FlushCaching.
        void (*end_run)(void* context){};
        /// The back: plays whole the packets of entries before `position`; returns how many.
        u64 (*drop_packets)(void* context, u64 position){};
        /// The bound channel as the back has it (the stage's start, after an unfollowed entry).
        s32 (*back_channel)(void* context){};
        /// Step 7: packets in ring B the back has not taken yet; whether ring B has room.
        u64 (*ring_packets)(void* context){};
        bool (*ring_room)(void* context){};
        /// The back, both threads parked at the ring's start (`end` false) or end: GPUMem's fill
        /// side starts clean (its unplayed mark 0; at the end its pending entries played first).
        void (*reset_stage)(void* context, bool end){};
        /// GPUMem, inside an entry: whether these ranges cover GPU-written buffer data (the page
        /// bits, then the cache's answer under its lock).
        bool (*ranges_gpu_written)(void* context, const std::pair<u64, std::size_t>* ranges,
                                   size_t count){};
        /// GPUMem, at a list draw of its copy: whether the back reads the two words on the CPU
        /// (RasterizerVulkan::DrawListWords' test, on the copy's registers and draw state).
        bool (*list_reads_back)(void* context, const MemStageShadowEngine& copy){};
        /// GPUMem, level at a gather's entry (the back parked there; FRITCHII_MEMSTAGE_GATHER):
        /// the back runs the gather now, as a request GPUMem waits for (ServeNow), unless its
        /// engine `engine` (a Maxwell3D) has a deferred draw. True when it ran; `draw_seq` is the
        /// back's draw count then.
        bool (*gather)(void* context, void* engine, u64& draw_seq){};
        /// GPUMem, at a sync point or fence entry outside any fill (fable_stage_levelpoints 6.2
        /// step 4): a PassReady. Tracking: read downloads are tracked (EXP49, or one pending);
        /// Flushes: the buffer cache has uncommitted flushes (its note, then under its lock the
        /// sets, refreshing the note).
        u8 (*pass_ready)(void* context){};
        /// GPUMem, at a passed gather (FRITCHII_MEMSTAGE_PASS bit 1, outside any fill): takes the
        /// game's collected writes into `ranges` (nothing when every collector is empty) and,
        /// unless one lies on a page with GPU-written buffer data, applies their buffer half at
        /// once. A PassTake.
        u8 (*pass_gather)(void* context, std::vector<std::pair<u64, u64>>& ranges){};
        /// The back: a pass record's half for the other caches (`whole`: the full gather path,
        /// the buffer half too: the entry ran as a level run, or the record is late and lent).
        /// `listed`: the ranges are on GPU's applying_writes (rule T): unlisted right after the
        /// apply, also when it throws.
        void (*apply_pass)(void* context, const std::vector<std::pair<u64, u64>>& ranges,
                           bool whole, bool listed){};
        /// The back, at the ring's end: how many taken lists are still on applying_writes.
        size_t (*listed_writes)(void* context){};
        /// GPUMem, P3's dry counters (fable_stage_p3 6): whether a query sync over these device
        /// runs [begin, bytes) would be pure lookups (BufferCache::ProbeSyncTarget, under the
        /// lock); the first failure, a P3Outcome (Passed when all are pure). Changes nothing.
        u8 (*sync_probe)(void* context, const std::pair<u64, u64>* runs, size_t count){};
        /// GPUMem, P3 (FRITCHII_MEMSTAGE_PASS bit 2, fable_stage_p3 4.2), outside any fill: as
        /// sync_probe, and when every run is a pure lookup, the sync's ObtainCPUBuffer
        /// (FullSynchronize) on each, here at the entry's place in the cache's order (it then only
        /// marks the usage), into `out[0, count)`. A P3Outcome; nothing is obtained when it is not
        /// Passed.
        u8 (*sync_targets)(void* context, const std::pair<u64, u64>* runs, size_t count,
                           SyncTarget* out){};
        /// GPUMem, in a catch-up while the back waits to promote the marks a packet's fill made
        /// ahead (FRITCHII_MEMSTAGE_MARKS_AHEAD): promotes them for it (GPUMem may hold the lock).
        void (*promote_marks)(void* context){};
    };
    /// pass_gather's answer.
    enum class PassTake : u8 { Empty, Applied, GpuWritten, Recorded };

    [[nodiscard]] static MemoryStageLink& Get() noexcept {
        return instance;
    }

    void Start(Core::System& system);
    void Stop();

    /// Back, at a frame end: 0 fills on the back, 1 hand-off, 2 the ring-fed stage. Returns the
    /// mode it runs.
    u32 SetMode(u32 wanted);
    [[nodiscard]] u32 Mode() const noexcept {
        return mode;
    }

    void Call(HandOff kind, Function function, void* argument);
    template <typename F>
    void Call(HandOff kind, F& function) {
        Call(kind, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    void Serve(Function function, void* argument);
    template <typename F>
    void Serve(F& function) {
        Serve([](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }
    /// Serve without the window's catch-up: GPUMem in its entry, level (StageLevel), so the back is
    /// parked at the entry with every packet before it played.
    void ServeNow(Function function, void* argument);
    template <typename F>
    void ServeNow(F& function) {
        ServeNow([](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }

    [[nodiscard]] static bool OnStageThread() noexcept {
        return t_on_stage;
    }
    [[nodiscard]] static bool InCall() noexcept {
        return t_in_call;
    }
    [[nodiscard]] static bool IsCacheUser() noexcept {
        return !g_checking.load(std::memory_order_relaxed) || IsCacheUserSlow();
    }
    static void CheckCacheUser(Check check) noexcept {
        if (g_checking.load(std::memory_order_relaxed)) [[unlikely]] {
            CheckCacheUserSlow(check);
        }
    }
    static void Note(Check check, const char* where) noexcept;
    [[nodiscard]] static bool Checking() noexcept {
        return g_checking.load(std::memory_order_relaxed);
    }

    class AllocatorOp {
    public:
        explicit AllocatorOp(const void* allocator) noexcept
            : watched{allocator != nullptr &&
                      allocator == g_buffer_allocator.load(std::memory_order_relaxed)} {
            if (watched) [[unlikely]] {
                CheckCacheUser(Check::AllocatorUser);
            }
        }
        ~AllocatorOp() {
            if (watched) [[unlikely]] {
                g_buffer_allocator_ops.fetch_add(1, std::memory_order_release);
            }
        }
        AllocatorOp(const AllocatorOp&) = delete;
        AllocatorOp& operator=(const AllocatorOp&) = delete;

    private:
        bool watched;
    };
    static void WatchBufferAllocator(const void* allocator) noexcept {
        g_buffer_allocator.store(allocator, std::memory_order_relaxed);
    }
    [[nodiscard]] static u64 BufferAllocatorOps() noexcept {
        return g_buffer_allocator_ops.load(std::memory_order_acquire);
    }

    [[nodiscard]] static bool FullVertexBinds() noexcept {
        return g_full_vertex_binds.load(std::memory_order_relaxed);
    }
    static void SetFullVertexBinds(bool on) noexcept {
        g_full_vertex_binds.store(on, std::memory_order_relaxed);
    }

    // --- THREAD = 2 ---

    void SetRingBackend(const RingBackend& backend) noexcept {
        ring_backend = backend;
    }
    /// The buffer cache's lock: a thread that holds it and starts to wait for the other thread
    /// (a request, a call, a catch-up) wakes that thread if it sleeps on the lock (step 7).
    void SetCacheLock(Common::SpinRecursiveMutex* lock) noexcept {
        cache_lock = lock;
    }
    [[nodiscard]] const RingBackend& Backend() const noexcept {
        return ring_backend;
    }
    /// The back's limit is `staged` (gpu_split.cpp).
    [[nodiscard]] static bool RingActive() noexcept {
        return g_ring_active.load(std::memory_order_acquire);
    }
    /// The back runs an entry the stage serves (not lent, not serving a request).
    [[nodiscard]] static bool BackServed() noexcept {
        return t_back_served && !t_back_serving;
    }
    /// The back runs a level run (the cache is lent to it).
    [[nodiscard]] static bool BackInRun() noexcept {
        return t_back_run;
    }
    /// At a buffer cache entry point: the back in the cache while not lent (must be 0).
    static void CheckBackCache(const char* where) noexcept {
        if (t_back_served && !t_back_serving && !t_back_cache) [[unlikely]] {
            Note(Check::BackInCache, where);
        }
    }
    /// GPUMem runs a call it serves for the back (it acts at the back's position).
    [[nodiscard]] static bool ServingBack() noexcept {
        return t_on_stage && t_serving_back;
    }
    /// FRITCHII_MEMSTAGE_MARKS_AHEAD: the back holds the buffer cache's lock for the fence
    /// commit (FenceManager::CommitCacheFlushes, SignalOrdering), also while served: it is the
    /// cache's user there (the user checks and CheckBackCache accept it).
    class BackCacheScope {
    public:
        BackCacheScope() noexcept : was{t_back_cache} {
            t_back_cache = true;
        }
        ~BackCacheScope() {
            t_back_cache = was;
        }
        BackCacheScope(const BackCacheScope&) = delete;
        BackCacheScope& operator=(const BackCacheScope&) = delete;

    private:
        bool was;
    };
    /// FRITCHII_MEMSTAGE_MARKS_AHEAD, the back after a packet whose fill marked writes ahead:
    /// calls `promote` (a try of the cache's lock; true when nothing is due any more) until it
    /// succeeds, serving GPUMem's requests meanwhile. It never blocks inside the lock: GPUMem may
    /// hold it in a fill and wait for the back (StageCatchUp promotes for it then).
    void BackPromoteMarks(bool (*promote)(void* argument), void* argument);
    /// The back, served: runs `function` on GPUMem, which holds the cache.
    void ServeOnStage(Served cause, Function function, void* argument);
    template <typename F>
    void ServeOnStage(Served cause, F& function) {
        ServeOnStage(cause, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
    }
    /// Ring B's counters, kept by the Vulkan side. `ahead`: packets in ring B after the push.
    void NotePacketSent(u64 bytes, u64 ahead) noexcept;
    /// FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h): E's push counts the packet and its bytes
    /// only (the window's counters are GPUMem's question).
    void NotePacketSentLean(u64 bytes) noexcept;
    void NotePacketTaken() noexcept;
    /// The back took or dropped a packet: wakes a stage that waits for the window or for room.
    void NoteRingFreed() noexcept;

    // --- Step 7: the window (FRITCHII_MEMSTAGE_AHEAD) ---

    /// Values at or above this are unlimited (spec 1.4: 65,536).
    static constexpr u32 AheadUnlimited = 65536;
    /// The back, at a frame end (a level run: GPUMem is parked): the window, in packets. 0 is
    /// level mode (6b). `blocked`: a setting the window does not run with (it runs 0 then).
    void SetAhead(u32 packets, const char* blocked = nullptr);
    /// The window GPUMem uses (read at its loop top; it changes only while GPUMem is parked).
    [[nodiscard]] static u32 Ahead() noexcept {
        return g_ahead.load(std::memory_order_relaxed);
    }
    /// GPUMem, inside an entry at `position`, outside any fill: waits until the back has run every
    /// entry before it (so every packet before it is played), serving the back's calls meanwhile.
    /// Does nothing in level mode (the stage is level before each entry there) or off GPUMem.
    /// False when GPUMem is stopping (the caller stops its work).
    bool StageLevel(u64 position, LevelReason reason);
    /// GPUMem, before a draw's fill at `position` (outside any fill and lock): waits for the
    /// window and for room in ring B (fps fix 1: not before every entry). False when stopping.
    bool StageBeforeFill(u64 position);
    /// The staging pool's stream was busy on GPUMem with the window open: it took a new buffer
    /// (the pool never waits for a region, so a stamp not played yet is never waited on).
    static void NoteStagingFallback() noexcept {
        if (t_on_stage && g_ahead.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            auto& counter = instance.counters.staging_fallbacks;
            counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        }
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_LIST_DRAWS (live, default on).
    void SetListDraws(bool on);
    /// GPUMem serves list draws: the switch on and the window open (AHEAD 0 keeps them level).
    [[nodiscard]] static bool ListDrawsServed() noexcept {
        return g_list_draws.load(std::memory_order_relaxed) &&
               g_ahead.load(std::memory_order_relaxed) != 0;
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_FAST_LOOP (live, default on).
    void SetFastLoop(bool on);
    /// GPUMem's fast entry loop (memstage_shadow.cpp, PairsFast; the input's end read once per
    /// stretch in the loop): the switch on and the window open (AHEAD 0 keeps today's loop).
    [[nodiscard]] static bool FastLoop() noexcept {
        return g_fast_loop.load(std::memory_order_relaxed) &&
               g_ahead.load(std::memory_order_relaxed) != 0;
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_ENTRY_FAST (live, default off).
    void SetEntryFast(bool on);
    /// GPUMem's short path for 3D Method and CbWords entries (memstage_shadow.cpp, EntryFast):
    /// the switch on and the window open (AHEAD 0 keeps Process's path).
    [[nodiscard]] static bool EntryFast() noexcept {
        return g_entry_fast.load(std::memory_order_relaxed) &&
               g_ahead.load(std::memory_order_relaxed) != 0;
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_CB_RUNS (live, default off).
    void SetCbRuns(bool on);
    /// GPUMem writes runs of constant buffer words as one (memstage_shadow.cpp, CbWords): the
    /// switch on and the window open (AHEAD 0 keeps a write per call).
    [[nodiscard]] static bool CbRuns() noexcept {
        return g_cb_runs.load(std::memory_order_relaxed) &&
               g_ahead.load(std::memory_order_relaxed) != 0;
    }
    /// GPUMem: a list draw of its copy, served or level (and why).
    void NoteListDraw(ListDraw what) noexcept {
        auto& counter = counters.list_draws[static_cast<size_t>(what)];
        counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }
    /// The back drew a served list draw from its packet's list words buffer.
    void NoteListDrawn() noexcept {
        counters.list_drawn.store(counters.list_drawn.load(std::memory_order_relaxed) + 1,
                                  std::memory_order_relaxed);
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_GATHER (live, default off).
    void SetGather(bool on);
    /// GPUMem has the back run gathers at their entry: the switch on and the window open, as
    /// GPUMem's loop read it (ring_window, what StageLevel goes by; g_ahead may have moved since).
    /// GPUMem only.
    [[nodiscard]] static bool GatherServed() noexcept {
        return g_gather.load(std::memory_order_relaxed) && t_on_stage &&
               instance.ring_window != 0;
    }
    /// GPUMem: a gather of its copy, and what became of it (`ns`: GPUMem's time for a served one).
    void NoteGather(GatherOutcome what, u64 ns = 0) noexcept {
        BumpOne(counters.gathers[static_cast<size_t>(what)], 1);
        BumpOne(counters.gather_ns, ns);
    }
    /// GPUMem, after a served gather: what the back's run of it found (counted).
    void NoteGatherRun(bool synced_queries, u64 ranges, bool gpu_written) noexcept {
        gather_last_synced = synced_queries;
        BumpOne(counters.gather_synced, synced_queries ? 1 : 0);
        BumpOne(counters.gather_ranges, ranges);
        BumpOne(counters.gather_gpu_written, gpu_written ? 1 : 0);
    }
    /// GPUMem: room for one more gather record.
    [[nodiscard]] bool GatherRoom() const noexcept {
        return gather_tail.load(std::memory_order_relaxed) -
                   gather_head.load(std::memory_order_acquire) <
               GatherSlots;
    }
    /// GPUMem, before it asks the back to run the gather of the entry at `position`: its record,
    /// so that the back's own run of that entry skips it (BackTakeGather) even when the request
    /// throws and the ring ends at the entry. The back is parked at the entry and takes nothing
    /// until GPUMem publishes past it: FinishGather sets the back's draw count then, RetractGather
    /// takes the record back when the request declined.
    void PushGather(u64 position) noexcept;
    void FinishGather(u64 draw_seq) noexcept;
    void RetractGather() noexcept;
    /// The back: whether a gather record waits (any mode: one can outlive the ring at its end).
    [[nodiscard]] bool GatherPending() const noexcept {
        return gather_head.load(std::memory_order_relaxed) !=
               gather_tail.load(std::memory_order_acquire);
    }
    /// The back, at a gather (RasterizerVulkan::InvalidateTextureDataCache): true when GPUMem had
    /// it run already at this entry. Counts the must-be-0 cases (a served one with no record, a
    /// record of an entry the back passed, a draw in between).
    [[nodiscard]] bool BackTakeGather(u64 position) noexcept;

    /// The back, at a frame end: FRITCHII_MEMSTAGE_RELEASE (live, default off).
    void SetRelease(bool on);
    /// GPUMem has the back run semaphore releases at their entry: the switch on, the window open
    /// (ring_window, as GatherServed). GPUMem only.
    [[nodiscard]] static bool ReleaseServed() noexcept {
        return g_release.load(std::memory_order_relaxed) && t_on_stage &&
               instance.ring_window != 0;
    }
    /// GPUMem: a semaphore release of the ring, and what became of it (`ns` for a served one).
    void NoteRelease(ReleaseOutcome what, u64 ns = 0) noexcept {
        BumpOne(counters.releases[static_cast<size_t>(what)], 1);
        BumpOne(counters.release_ns, ns);
    }
    [[nodiscard]] bool ReleaseRoom() const noexcept {
        return release_tail.load(std::memory_order_relaxed) -
                   release_head.load(std::memory_order_acquire) <
               ReleaseSlots;
    }
    /// GPUMem, before it asks the back to run the release closure of the entry at `position`: as
    /// PushGather (a record the back's own run of that entry skips, BackTakeRelease).
    void PushRelease(u64 position) noexcept;
    void FinishRelease(u64 draw_seq) noexcept;
    [[nodiscard]] bool ReleasePending() const noexcept {
        return release_head.load(std::memory_order_relaxed) !=
               release_tail.load(std::memory_order_acquire);
    }
    /// The back, at a release closure's Call entry (GpuSplit::Execute): true when GPUMem had it run
    /// already at this entry (the call is skipped). Counts the must-be-0 cases as BackTakeGather.
    [[nodiscard]] bool BackTakeRelease(u64 position) noexcept;
    // --- fable_stage_levelpoints 5 (P0's counters) and 6 (P1, FRITCHII_MEMSTAGE_PASS) ---

    /// What became of a lone gather (3D 0x3dd) or wait for idle (0x44) GPUMem reached: passed
    /// (dry: it would pass P1's tests up to the flush test), or the first test it failed.
    enum class PassOutcome : u8 {
        Passed,      ///< passed (dry: would pass)
        Closed,      ///< the window closed (level mode)
        InRun,       ///< inside a level run (followed; the back runs it lent)
        Late,        ///< in an entry hard already
        NotFirst,    ///< not a lone method, or a draw before it in its entry
        SyncPending, ///< a query result sync may be pending at the entry (the mirror)
        Tracking,    ///< read download tracking on (EXP49, or a read download pending)
        Flushes,     ///< the buffer cache has uncommitted flushes
        NoRoom,      ///< no room for its record
        GpuWritten,  ///< a taken game write lies on a page with GPU-written data: a run
        Count,
    };
    enum class PassMethod : u8 { Gather, WaitForIdle, Count };
    /// The backend's pass_ready answer (rule C and read tracking, at GPUMem's position).
    enum class PassReady : u8 { Ready, Tracking, Flushes };
    /// GPUMem: a pass decision.
    void NotePass(PassMethod method, PassOutcome what) noexcept {
        BumpOne(pass_stats.outcomes[static_cast<size_t>(method)][static_cast<size_t>(what)], 1);
    }
    /// GPUMem, after the back ran a gather at its entry (FRITCHII_MEMSTAGE_GATHER): whether the
    /// mirror said no sync could be pending, against what the back found (fable_stage_levelpoints
    /// 12.3: "clear, but the back synced" must be 0).
    void NoteMirrorCheck(bool clear, bool synced) noexcept {
        auto& stats = pass_stats;
        BumpOne(stats.mirror_checks, 1);
        BumpOne(stats.mirror_clear, clear ? 1 : 0);
        BumpOne(stats.mirror_wrong, clear && synced ? 1 : 0);
    }
    /// GPUMem: whether the back's run of the gather it last asked for synced query results.
    [[nodiscard]] bool LastGatherSynced() const noexcept {
        return gather_last_synced;
    }
    /// GPUMem: a fence entry outside a run (0: the puller's release closure, 1: a 3D fence report)
    /// and whether rule C would hold there (P3's pass rate).
    void NoteFenceEntry(u32 kind, PassReady ready) noexcept {
        BumpOne(pass_stats.fence_entries[kind & 1][static_cast<size_t>(ready)], 1);
    }
    /// The front, at a puller semaphore release (WriteLong): fence-flagged or timestamped, and
    /// whether its address is mapped.
    void NoteFrontRelease(bool fence, bool mapped) noexcept {
        BumpOne(pass_stats.front_releases[(fence ? 2 : 0) + (mapped ? 1 : 0)], 1);
    }
    /// GPUMem: the switch's bit `bit` (1: gathers and waits for idle with no query sync pending;
    /// 2: P3, sync points with a pending sync, release closures and 3D fence reports; 3: either)
    /// is on, the window open as GPUMem's loop read it, and no must-be-0 of the passes turned it
    /// off. GPUMem only.
    [[nodiscard]] static bool PassServed(u32 bit) noexcept {
        return (g_pass.load(std::memory_order_relaxed) & bit) != 0 && t_on_stage &&
               instance.ring_window != 0;
    }
    /// GPUMem: the window is open (P0's dry counters only count there).
    [[nodiscard]] static bool StageWindowOpen() noexcept {
        return t_on_stage && instance.ring_window != 0;
    }
    /// The back (any thread): a bit of `bit` of the switch is on and the window open, as last set
    /// at a frame end (the back's own gathers while served go to GPUMem then, levelpoints 6.3).
    [[nodiscard]] static bool PassOn(u32 bit) noexcept {
        return (g_pass.load(std::memory_order_relaxed) & bit) != 0 &&
               g_ahead.load(std::memory_order_relaxed) != 0;
    }
    /// The back, at a frame end: FRITCHII_MEMSTAGE_PASS (live, default 0). Runs as 0 for the
    /// session after a must-be-0 of the passes.
    void SetPass(u32 bits);

    /// A passed entry's record (levelpoints 6.4, fable_stage_p3 1): its position, and for a
    /// gather the game's writes GPUMem took there (`whole`: nothing of them applied, the entry
    /// runs as a level run). One writer (GPUMem), one reader (the back), 256 slots; a record
    /// lives until the back asks for a later entry's.
    struct PassRecord {
        u64 position{};
        bool whole{};
        bool seen{};           ///< The back found it at its entry (BackPassRecord)
        bool ranges_applied{}; ///< The back applied its half (BackApplyPasses)
        /// Rule T (fable_stage_levelpoints 13.3): `ranges` is on GPU's applying_writes until the
        /// back applies it (a whole record). It must not move or change meanwhile.
        bool listed{};
        /// P3 (bit 2, fable_stage_p3 4 and 5.1): GPUMem resolved the entry's query result sync:
        /// `targets` holds a buffer for every report page that may be pending there (none when
        /// none is). The back's SyncValues copies into them and stays out of the buffer cache.
        bool sync{};
        std::vector<std::pair<u64, u64>> ranges;
        std::vector<SyncTarget> targets;
    };
    static constexpr u64 PassSlots = 256;
    /// GPUMem: room for one more record.
    [[nodiscard]] bool PassRoom() const noexcept {
        return pass_tail.load(std::memory_order_relaxed) -
                   pass_head.load(std::memory_order_acquire) <
               PassSlots;
    }
    /// GPUMem (PassRoom said yes): the next record, cleared, for the entry at `position`; it is
    /// the back's once PassPush publishes it.
    PassRecord& PassBegin(u64 position) noexcept {
        PassRecord& record = pass_records[pass_tail.load(std::memory_order_relaxed) % PassSlots];
        record.position = position;
        record.whole = false;
        record.seen = false;
        record.ranges_applied = false;
        record.listed = false;
        record.sync = false;
        record.ranges.clear();
        record.targets.clear();
        return record;
    }
    void PassPush() noexcept;
    /// The back: whether a record waits (any mode: one can outlive the ring at its end).
    [[nodiscard]] bool PassPending() const noexcept {
        return pass_head.load(std::memory_order_relaxed) !=
               pass_tail.load(std::memory_order_acquire);
    }
    /// The back: frees the records of entries before `position` (one it never found is late:
    /// must be 0, its half applied there), then returns the record of `position` if there is
    /// one, without freeing it (it may be asked for twice in an entry).
    PassRecord* BackPassRecord(u64 position) noexcept;
    /// The back, at a passed gather's entry (RasterizerVulkan::InvalidateGPUCache): applies the
    /// half of the record of `position` the back owes, once.
    void BackApplyPasses(u64 position) noexcept;
    /// The back: BackPassRecord of the entry it runs now (GpuSplit's BackEntryPosition); null when
    /// it has none, or no record waits at all.
    PassRecord* BackCurrentPassRecord() noexcept;
    /// fable_stage_p3 section 6, dry (decided and counted, behaving as today): would P3 pass the
    /// entry, and if not the first reason, per kind. With bit 2 on, the same counters hold the
    /// live decisions (the entries that reached them).
    enum class P3Outcome : u8 {
        Passed,         ///< would pass (a sync point, a fence entry with pure targets)
        PassedUnmapped, ///< a release to an unmapped address (the back's Query returns at once)
        PassedNoFence,  ///< a release that is not fence-flagged (no fence, no cache)
        NotLone,        ///< in a run, late, or not a lone method entry
        Accurate,       ///< the fence behaviour is Accurate (a lookup may wait)
        Trim,           ///< EXP28 & 1 (deferred payload syncs: the fence may not sync)
        Tracking,       ///< read download tracking on
        Flushes,        ///< uncommitted buffer flushes
        ReportsUnknown, ///< reports GPUMem did not see may be pending (after a run, a hard entry)
        TooManyRuns,    ///< the report pages make more than 16 runs
        NoBuffer,       ///< a report page no buffer holds (the sync would create one)
        NotInBounds,    ///< a run past the buffer on its first page
        CpuModified,    ///< a report page with CPU-modified (or untracked) bytes: it would upload
        NoRoom,         ///< live (bit 2): no room for the record
        Count,
    };
    enum class P3Kind : u8 { Gather, WaitForIdle, Release, FenceReport, Count };
    void NoteP3(P3Kind kind, P3Outcome what) noexcept {
        BumpOne(pass_stats.p3[static_cast<size_t>(kind)][static_cast<size_t>(what)], 1);
    }
    /// GPUMem, bit 2: an entry of `kind` P3 passed (pushed its record), with `targets` runs.
    void NoteP3Passed(P3Kind kind, u64 targets) noexcept {
        BumpOne(pass_stats.p3_passed[static_cast<size_t>(kind)], 1);
        BumpOne(pass_stats.p3_targets, targets);
    }
    /// The back, served, at a query result sync (QueryCacheRuntime::SyncValues): from a pass
    /// record's targets (no buffer cache), or not (no record of P3's: today's block, which is not
    /// a must-be-0, a read flush's own fence syncs there).
    void NoteP3BackSync(bool from_record) noexcept {
        BumpOne(pass_stats.p3_back_syncs[from_record ? 1 : 0], 1);
    }
    /// P3 dry: the report pages of a sync point that reached the target test.
    void NoteP3Pages(u64 pages) noexcept {
        BumpOne(pass_stats.p3_sync_points, 1);
        BumpOne(pass_stats.p3_pages, pages);
        if (pages > pass_stats.p3_pages_max.load(std::memory_order_relaxed)) {
            pass_stats.p3_pages_max.store(pages, std::memory_order_relaxed);
        }
    }

    /// The back, served: one of its own gathers taken on GPUMem (levelpoints 6.3), and whether it
    /// found writes.
    void NoteServedTake(bool found) noexcept {
        BumpOne(pass_stats.served_takes[found ? 1 : 0], 1);
    }

    /// The entry turned hard at a constant buffer write onto GPU-written data (counted).
    void NoteWriteLevel() noexcept {
        counters.write_levels.store(counters.write_levels.load(std::memory_order_relaxed) + 1,
                                    std::memory_order_relaxed);
    }
    /// The back, at a frame end: the window's verify runs (FRITCHII_MEMSTAGE_READ_LOG or
    /// FRITCHII_MEMSTAGE_RING_VERIFY): GPUMem keeps its own writes, images look for them.
    static void SetVerify(bool on) noexcept {
        g_verify.store(on, std::memory_order_relaxed);
    }
    /// GPUMem keeps its own writes in its ring (the window open and its verify on).
    [[nodiscard]] static bool StageWritesKept() noexcept {
        return g_ahead.load(std::memory_order_relaxed) != 0 &&
               g_verify.load(std::memory_order_relaxed);
    }

    /// GPUMem, around a lock of the buffer cache taken outside any fill and outside a served
    /// call: while it waits for the lock, held by the back, it serves the back's call (step 7).
    class CleanPoint {
    public:
        CleanPoint() noexcept : was{t_stage_clean} {
            t_stage_clean = t_on_stage;
        }
        ~CleanPoint() {
            t_stage_clean = was;
        }
        CleanPoint(const CleanPoint&) = delete;
        CleanPoint& operator=(const CleanPoint&) = delete;

    private:
        bool was;
    };
    /// The buffer cache joined a buffer at [address, address + size) into a new one
    /// (BufferCache::JoinOverlap): with the window open GPUMem keeps where (the entry it fills).
    static void NoteBufferJoin(u64 address, u64 size) {
        if (t_on_stage && g_ahead.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            instance.NoteBufferJoinSlow(address, size);
        }
    }
    /// BufferCache::JoinOverlap, before a join: whether the window needs the join's GPU-written
    /// test (GPUMem's own fill, the window open), and then the catch-up it asks for.
    [[nodiscard]] static bool JoinNeedsTest() noexcept {
        return t_on_stage && !t_serving_back && g_ahead.load(std::memory_order_relaxed) != 0;
    }
    static void JoinOverGpuWritten() {
        instance.StageJoinCatchUpSlow();
    }
    /// GPUMem, in a call it serves for the back at entry `back_position`: whether a buffer over
    /// [address, address + size) was joined by a fill of a later entry (its copy is in a packet
    /// the back has not played).
    enum class Joined : u8 { No, Yes, Unknown };
    [[nodiscard]] Joined JoinedAhead(u64 address, u64 size, u64 back_position) const noexcept;
    /// GPUMem, in a download it serves for the back: the join ring could not answer (every kept
    /// join is ahead of the back). Counted, not a failure.
    void NoteJoinUnknown() noexcept {
        counters.joins_unknown.store(counters.joins_unknown.load(std::memory_order_relaxed) + 1,
                                     std::memory_order_relaxed);
    }
    /// GPUMem processes the entry at `position` (the joins' and the stage writes' position).
    [[nodiscard]] u64 StageEntry() const noexcept {
        return stage_entry;
    }

    // --- Step 7: GPUMem's own writes to game memory (spec 6.4's "short ring of the stage's own
    // stream writes with positions"): constant buffer words, by device range and entry ---

    /// GPUMem, the window open: it wrote [begin, begin + size) of device memory for the entry at
    /// `position`. Positions only grow (the stream's order).
    void NoteStageWrite(u64 begin, u64 size, u64 position) noexcept;
    /// How many writes were noted (the number the next one gets).
    [[nodiscard]] u64 StageWriteCount() const noexcept {
        return stage_write_count.load(std::memory_order_acquire);
    }
    /// Any thread: calls func(begin, end, position) for each write numbered `since` or later,
    /// newest first, until it returns true. False when some of them were overwritten already.
    template <typename F>
    bool ForEachStageWriteSince(u64 since, F&& func) const {
        if (!stage_writes) {
            return true;
        }
        const u64 count = stage_write_count.load(std::memory_order_acquire);
        if (count < since) {
            return true;
        }
        bool complete = count - since <= StageWriteSlots;
        const u64 oldest = complete ? since : count - StageWriteSlots;
        for (u64 n = count; n > oldest; --n) {
            const StageWrite& slot = stage_writes[(n - 1) % StageWriteSlots];
            if (slot.seq.load(std::memory_order_acquire) != n) {
                complete = false;
                continue;
            }
            const u64 begin = slot.begin.load(std::memory_order_relaxed);
            const u64 end = slot.end.load(std::memory_order_relaxed);
            const u64 position = slot.position.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (slot.seq.load(std::memory_order_relaxed) != n) {
                complete = false;
                continue;
            }
            if (func(begin, end, position)) {
                break;
            }
        }
        return complete;
    }
    /// The back registers an image over [address, address + size) (the texture cache): risk 9.1,
    /// a constant buffer write GPUMem made for a later entry under it (must be 0). Only while the
    /// window's verify runs (StageWritesKept): the fps gate measures play without the walk.
    static void NoteImageRegistered(u64 address, u64 size) {
        if (g_ring_active.load(std::memory_order_relaxed) && StageWritesKept()) [[unlikely]] {
            instance.NoteImageRegisteredSlow(address, size);
        }
    }
    void NoteLevelDraw(u32 kind) noexcept;
    /// The verify of compare b with two threads (FRITCHII_MEMSTAGE_RING_VERIFY).
    void NoteHash(bool same) noexcept;
    [[nodiscard]] static bool RingVerify() noexcept;

    /// The back's loop (gpu_split.cpp).
    struct BackLimits {
        u64 available;
        u64 lent_from;
        u64 lent_until;
    };
    /// At the top of the back's loop: starts or ends ring mode at `position` (everything before
    /// it consumed). True while it is on.
    bool BackRingTop(u64 position);
    /// Whether the back should leave its batch so the ring starts at its loop top.
    [[nodiscard]] bool BackRingStartWanted() const noexcept {
        return ring_wanted.load(std::memory_order_relaxed) &&
               !g_ring_active.load(std::memory_order_relaxed);
    }
    [[nodiscard]] BackLimits BackLoad(u64 published) const noexcept;
    /// Sets the back's thread-locals for the entry at `position`; starts and ends runs.
    void BackEntry(u64 position, const BackLimits& limits);
    /// After an entry: true when the run's last entry ran (the back then consumes its method
    /// sink and calls BackEndRun).
    [[nodiscard]] bool BackAfter(u64 position, const BackLimits& limits);
    void BackEndRun();
    [[nodiscard]] bool BackRequestPending() const noexcept;
    void NoteBackWait(u64 ns) noexcept;
    /// The back stored `consumed`; the front published (wakes a stage that waits for them).
    void NoteConsumed(u64 position) noexcept;
    void NotePublished() noexcept;
    /// The back waits for the stage (phase 3 of the split wait): serves its requests. True if it
    /// served one.
    bool BackServeRequests();
    /// The back's idle hook is about to run: false when the stage is busy (the hook is skipped).
    bool BackEnterHook();
    void BackLeaveHook();
    /// Step 7: the back ran its idle hook from the wait though the stage was busy (a request).
    void NoteHookFlush() noexcept;
    /// The back drew (DrawManager, every draw, before the render-enable test).
    void NoteBackDraw(const void* engine) noexcept;
    void NoteBackDrawState(const void* engine) noexcept;
    [[nodiscard]] u64 BackDrawSeq() const noexcept {
        return back_draw_seq;
    }

    void LogStats();

private:
    static bool IsCacheUserSlow() noexcept;
    static void CheckCacheUserSlow(Check check) noexcept;

    void StageMain(std::stop_token stop);
    void RunCall(u32 seq);
    void WaitForStage(u32 seq);
    void ServeOne();
    void WakeStage();
    void WakeBack();

    // THREAD = 2: the stage's loop (memory_stage_link.cpp).
    void RingLoop(std::stop_token& stop, u64 start);
    bool BackRingTopImpl(u64 position);
    void PublishStaged(u64 position);
    /// Input: the front; Back: level mode; Run: a level run; Window: AHEAD packets in ring B; Room:
    /// ring B full; Level: StageLevel inside the window.
    enum class StageWait : u8 { Input, Back, Run, Window, Room, Level, Count };
    /// `waited`: when not null, the wait's nanoseconds are added to it (the clock reads the
    /// wait makes anyway).
    bool StageWaitFor(std::stop_token& stop, StageWait reason, u64 target, u64* waited = nullptr);
    /// Step 7, GPUMem in its own fill, before a sync goes to the back: the back first plays every
    /// packet before the published position (unless it waits in a call GPUMem must serve first).
    void StageSyncCatchUp();
    void StageJoinCatchUpSlow();
    /// The back runs up to the published position; false when it could not (counted).
    bool StageCatchUp(const char* what);
    bool StageServeCalls();
    bool BecomeBusy(std::stop_token& stop);
    void StageIdle(std::stop_token& stop);
    void Lend(u64 from, u64 until);
    u64 RingFailed(std::stop_token& stop, const char* what);

    static MemoryStageLink instance;

    static inline std::atomic<bool> g_checking{false};
    static inline std::atomic<const void*> g_buffer_allocator{nullptr};
    static inline std::atomic<u64> g_buffer_allocator_ops{0};
    static inline std::atomic<bool> g_full_vertex_binds{false};
    static inline std::atomic<bool> g_ring_active{false};
    static inline std::atomic<u32> g_ahead{0};
    static inline std::atomic<bool> g_verify{false};
    static inline std::atomic<bool> g_list_draws{false};
    static inline std::atomic<bool> g_fast_loop{false};
    static inline std::atomic<bool> g_entry_fast{false};
    static inline std::atomic<bool> g_cb_runs{false};
    static inline std::atomic<bool> g_gather{false};
    static inline std::atomic<bool> g_release{false};
    /// FRITCHII_MEMSTAGE_PASS's bits as GPUMem uses them (0 after a must-be-0 of the passes).
    static inline std::atomic<u32> g_pass{0};

    /// A stats counter: a plain load and store (one writer each, or a lost count is fine).
    static void BumpOne(std::atomic<u64>& counter, u64 amount) noexcept {
        counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
    }

    static inline thread_local bool t_on_stage{};
    /// GPUMem runs a call the back posted (a served call): its syncs go to the back at once.
    static inline thread_local bool t_serving_back{};
    /// GPUMem is at a lock site that is a clean point (CleanPoint).
    static inline thread_local bool t_stage_clean{};
    static inline thread_local bool t_in_call{};
    static inline thread_local bool t_stage_user{};
    static inline thread_local bool t_back_lent{};
    static inline thread_local bool t_back_serving{};
    static inline thread_local bool t_back_run{};
    static inline thread_local bool t_back_served{};
    /// The back holds the cache's lock for a fence commit (BackCacheScope).
    static inline thread_local bool t_back_cache{};

    Core::System* system{};
    u32 mode{};
    bool refused_logged{};

    // Written by the back before `calls`, read by the stage after it.
    Function call_function{};
    void* call_argument{};
    bool call_recording{};
    bool call_served{};
    std::exception_ptr call_error;
    u32 call_sent{};
    u32 calls_seen{};
    // Written by the stage before `requests`, read by the back after it.
    Function request_function{};
    void* request_argument{};
    std::exception_ptr request_error;
    u32 request_sent{};
    u32 answered_local{};

    alignas(64) std::atomic<u32> calls{0};
    alignas(64) std::atomic<u32> done{0};
    alignas(64) std::atomic<u32> requests{0};
    alignas(64) std::atomic<u32> answered{0};
    alignas(64) std::atomic<u32> stage_word{0};
    std::atomic<bool> stage_sleeping{false};
    /// Step 7: GPUMem's sleep waits for the window or for room in ring B, the only waits a take
    /// ends (set before stage_sleeping, both seq_cst: a back that sees GPUMem asleep sees it).
    std::atomic<bool> stage_ring_wait{false};
    alignas(64) std::atomic<u32> back_word{0};
    std::atomic<bool> back_sleeping{false};

    // THREAD = 2.
    RingBackend ring_backend{};
    Common::SpinRecursiveMutex* cache_lock{};
    void WakeLockWaiters() noexcept;
    alignas(64) std::atomic<bool> ring_wanted{false};
    /// Stage: inside a batch (between a park of the back and the next publish), or its idle work.
    alignas(64) std::atomic<bool> stage_busy{false};
    /// Back: in its idle hook.
    std::atomic<bool> back_in_hook{false};
    /// Set by the back at the loop top: where the stage starts reading.
    std::atomic<u64> ring_start{~u64{0}};
    std::atomic<bool> ring_running{false};
    std::atomic<bool> ring_failed{false};
    u64 staged_local{};
    /// The stage's ranges handed to the back at a lend (read by the back at the run's start).
    std::vector<std::pair<u64, std::size_t>> lend_ranges;
    /// Step 7: the lend's flag hand-over waits for the back's run start (the back is not parked
    /// at a lend with the window open). Written by the stage before it publishes the run.
    bool lend_flags_deferred{};
    /// Step 7: the window is refused for the session (a sync went to a back that had not played
    /// the packets ahead of it); the last blocking setting logged.
    std::atomic<bool> ahead_refused{false};
    const char* ahead_blocked_logged{};
    /// GPUMem's loop: its stop token while the ring runs, and the window it read last.
    std::stop_token* ring_stop{};
    u32 ring_window{};
    /// The entry GPUMem processes (GPUMem only).
    u64 stage_entry{};
    /// GPUMem's busy time is timed per publish batch: its start, 0 when no batch is open.
    u64 busy_batch_start{};
    void CloseBusyBatch() noexcept;
    /// Buffers joined by GPUMem's fills, with the entry (GPUMem only; served calls run there too).
    struct Join {
        u64 begin;
        u64 end;
        u64 position;
    };
    static constexpr size_t JoinSlots = 4096;
    std::array<Join, JoinSlots> joins{};
    u64 join_count{};
    void NoteBufferJoinSlow(u64 address, u64 size) noexcept;
    /// GPUMem's own writes (NoteStageWrite): one writer, any reader (a sequence word per slot).
    struct StageWrite {
        std::atomic<u64> seq{};
        std::atomic<u64> begin{};
        std::atomic<u64> end{};
        std::atomic<u64> position{};
    };
    static constexpr u64 StageWriteSlots = u64{1} << 17;
    std::unique_ptr<StageWrite[]> stage_writes;
    std::atomic<u64> stage_write_count{0};
    void NoteImageRegisteredSlow(u64 address, u64 size);
    /// FRITCHII_MEMSTAGE_GATHER: the gathers GPUMem had the back run at their entry, oldest first
    /// (one writer, GPUMem; one reader, the back). Emptied at the ring's start and end.
    struct GatherRecord {
        u64 position;
        u64 draw_seq; ///< the back's draw count when it ran the gather
    };
    static constexpr u64 GatherSlots = 256;
    std::array<GatherRecord, GatherSlots> gather_records{};
    alignas(64) std::atomic<u64> gather_tail{0}; ///< pushed (GPUMem)
    alignas(64) std::atomic<u64> gather_head{0}; ///< taken (the back)
    /// The back: drops the records of entries before `before` (each one must-be-0 orphan).
    void BackDropGathers(u64 before, const char* where) noexcept;
    /// FRITCHII_MEMSTAGE_GATHER's last value logged (-1: none yet).
    int gather_logged{-1};
    /// FRITCHII_MEMSTAGE_RELEASE: the semaphore releases GPUMem had the back run, as the gathers'.
    static constexpr u64 ReleaseSlots = 256;
    std::array<GatherRecord, ReleaseSlots> release_records{};
    alignas(64) std::atomic<u64> release_tail{0}; ///< pushed (GPUMem)
    alignas(64) std::atomic<u64> release_head{0}; ///< taken (the back)
    void BackDropReleases(u64 before, const char* where) noexcept;
    int release_logged{-1};
    /// GPUMem's thread id (the back serves it while it waits for a lock GPUMem holds).
    std::atomic<std::thread::id> stage_thread{};
    /// Bumped while the back waits for a lock GPUMem holds (StageSyncCatchUp gives up then).
    std::atomic<u64> back_contended{0};
    /// The back waits to promote marks made ahead (BackPromoteMarks; a catch-up promotes then).
    std::atomic<bool> back_promote_wait{false};
    static bool BackContendedHook(std::thread::id holder, bool first);
    /// The back's thread id (GPUMem serves the back's call while it waits for a lock it holds).
    std::atomic<std::thread::id> back_thread{};
    static bool StageContendedHook(std::thread::id holder, bool first);
    /// Back: the run it is in.
    u64 back_run_from{};
    /// The limits the back's last BackEntry had (its batch's) and that entry's position: the
    /// must-be-0 notes of the gathers, releases and runs log them (BackRingState).
    BackLimits back_entry_limits{};
    u64 back_entry_at{};
    /// The back's ring state for a must-be-0 note at `position` (formatted; that path only).
    std::string BackRingState(u64 position) const;
    u64 back_draw_seq{};
    std::atomic<const void*> back_drew_in_run{};
    std::atomic<const void*> back_state_in_run{};

    /// The window's fill histogram: 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-255, 256 and more.
    static constexpr size_t WindowBuckets = 8;

    /// Grouped by the thread that bumps them, each group on its own cache lines (step 7: with
    /// the window open both threads bump theirs for every packet; mixed lines bounced).
    struct alignas(64) Counters {
        // GPUMem's.
        std::atomic<u64> stage_ns{};
        std::atomic<u64> stage_sleeps{};
        std::atomic<u64> stage_input_sleeps{}; ///< of stage_sleeps, waiting for the front
        std::array<std::atomic<u64>, 4> entries{}; ///< pass, work, level, neutral
        std::atomic<u64> runs{};
        std::atomic<u64> run_entries{};
        std::atomic<u64> run_ns{};       ///< the stage, from the lend to the return
        std::array<std::atomic<u64>, static_cast<size_t>(StageWait::Count)> stage_wait_ns{};
        std::atomic<u64> stage_busy_ns{};
        std::atomic<u64> fill_ns{};
        std::atomic<u64> packets{};
        std::atomic<u64> packet_bytes{};
        std::atomic<u64> publishes{};
        std::atomic<u64> copies{};
        std::atomic<u64> flag_handovers{};
        std::array<std::atomic<u64>, 8> level_draws{};
        std::atomic<u64> ring_ahead_max{};
        std::array<std::atomic<u64>, WindowBuckets> window{}; ///< packets in ring B at each push
        std::atomic<u64> ahead_sum{};
        std::array<std::atomic<u64>, static_cast<size_t>(LevelReason::Count)> levels{};
        std::atomic<u64> catchups{};
        std::atomic<u64> catchup_ns{};
        std::atomic<u64> deferred_lends{};
        std::atomic<u64> stage_lock_serves{}; ///< GPUMem served a call while waiting for it
        std::atomic<u64> join_catchups{};
        std::atomic<u64> joins_unknown{};     ///< a served download past the join ring's reach
        std::atomic<u64> write_levels{};      ///< words onto GPU-written data: the entry hard
        std::atomic<u64> staging_fallbacks{}; ///< GPUMem took a new staging buffer, stream busy
        std::atomic<u64> stage_idles{};
        std::atomic<u64> run_input_waits{};   ///< fps fix 3: a run kept lent at the input's end
        std::array<std::atomic<u64>, static_cast<size_t>(ListDraw::Count)> list_draws{};
        std::array<std::atomic<u64>, static_cast<size_t>(GatherOutcome::Count)> gathers{};
        std::atomic<u64> gather_ns{};          ///< GPUMem in served gathers (level wait and run)
        std::atomic<u64> gather_synced{};      ///< served ones whose wait for idle synced queries
        std::atomic<u64> gather_ranges{};      ///< game write ranges the served ones applied
        std::atomic<u64> gather_gpu_written{}; ///< served ones with a range by GPU-written data
        std::array<std::atomic<u64>, static_cast<size_t>(ReleaseOutcome::Count)> releases{};
        std::atomic<u64> release_ns{};         ///< GPUMem in served releases (level wait and run)
        // The back's.
        alignas(64) std::array<std::atomic<u64>, static_cast<size_t>(HandOff::Count)> calls{};
        std::atomic<u64> call_ns{};
        std::atomic<u64> served{};
        std::atomic<u64> served_ns{};
        std::atomic<u64> back_sleeps{};
        std::array<std::atomic<u64>, static_cast<size_t>(Served::Count)> served_calls{};
        std::atomic<u64> served_ahead{};
        std::atomic<u64> back_run_ns{};  ///< the back, from the lend seen to the next packet taken
        std::atomic<u64> back_wait{};
        std::atomic<u64> back_wait_ns{};
        std::atomic<u64> taken{};
        std::atomic<u64> hashes{};
        std::atomic<u64> hash_mismatches{};
        std::atomic<u64> back_lock_waits{};   ///< the back waited for the lock GPUMem held
        std::atomic<u64> hook_flushes{};      ///< the back's skipped idle hook, its flush run
        std::atomic<u64> ring_wakes{};        ///< takes that woke GPUMem (window or room wait)
        std::atomic<u64> list_drawn{};        ///< served list draws drawn from the packet's buffer
        std::atomic<u64> gathers_taken{};     ///< gathers the back found run at their entry
        std::atomic<u64> gathers_taken_run{}; ///< ... of them inside a level run
        std::atomic<u64> releases_taken{};    ///< releases the back found run at their entry
        // Either thread's, rare (the must-be-0 checks).
        alignas(64) std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> checks{};
    };
    Counters counters;
    /// fable_memstage_compute_brief 3: GPUMem's level runs by channel (0-6, then any other) and
    /// by size (entries, compute launches), with their time. Taken and reset at each line.
    static constexpr size_t RunChannels = 8;
    static constexpr size_t RunBuckets = 6; ///< 0, 1, 2-7, 8-63, 64-511, 512 and more
    struct alignas(64) RunShape {
        std::array<std::atomic<u64>, RunChannels> runs{};
        std::array<std::atomic<u64>, RunChannels> entries{};
        std::array<std::atomic<u64>, RunChannels> launches{};
        std::array<std::atomic<u64>, RunChannels> ns{};
        std::array<std::atomic<u64>, RunBuckets> by_entries{};
        std::array<std::atomic<u64>, RunBuckets> by_entries_ns{};
        std::array<std::atomic<u64>, RunBuckets> by_launches{};
        std::array<std::atomic<u64>, RunBuckets> by_launches_ns{};
    };
    RunShape run_shape;
    void NoteRunShape(s32 channel, u64 entries, u64 launches, u64 ns) noexcept;
    void LogRunShape(double seconds);
    /// fable_stage_levelpoints 14.4 row 4 and 14.9: GPUMem's sleeps in StageWaitFor by reason
    /// (always counted: a sleep is a system call anyway), and the "Memory stage run waits" line's
    /// counts. Taken and reset at each line (a lost count is fine).
    struct alignas(64) RunWaitStats {
        // GPUMem's.
        std::array<std::atomic<u64>, static_cast<size_t>(StageWait::Count)> sleeps{};
        /// Of the sleeps, those ended after a wake call that found GPUMem asleep, and the time
        /// from that call to GPUMem running again (the wake latency).
        std::array<std::atomic<u64>, static_cast<size_t>(StageWait::Count)> woken{};
        std::array<std::atomic<u64>, static_cast<size_t>(StageWait::Count)> wake_ns{};
        /// GPU profile on: the run's wait for the back (StageWait::Run), split where the back
        /// reached the run's first entry: before it (the back's backlog) and after (the run).
        std::atomic<u64> split_runs{};
        std::atomic<u64> reach_ns{};
        std::atomic<u64> run_ns{};
        std::atomic<u64> split_other{}; ///< not split: the back's mark was another run's
        // The back's: where and when it reached a run's first entry (BackEntry).
        alignas(64) std::atomic<u64> reached_from{~u64{0}};
        std::atomic<u64> reached_ns{};
        // The waker's (the back or the front): when a wake call found GPUMem asleep.
        alignas(64) std::atomic<u64> wake_call_ns{};
    };
    RunWaitStats run_waits;
    /// GPUMem, GPU profile on: lend to return by the run's first hard kind (MemStageShadow's
    /// StageEntryKey). Under its lock (one take a run; the line swaps the map out).
    struct RunKind {
        u64 runs{};
        u64 ns{};
        u64 back_ns{};
        u64 front_ns{};
    };
    std::mutex run_kinds_mutex;
    std::unordered_map<u64, RunKind> run_kinds;
    void NoteRunWaitSplit(u64 run_from, u64 wait_from, u64 wait_ns) noexcept;
    void NoteRunKind(u64 key, u64 ns, u64 back_ns, u64 front_ns);
    void LogRunWaits(double seconds);
    /// fable_stage_levelpoints 5.2 (P0) and 6.6 (P1): the "Memory stage passes" line's counts,
    /// taken and reset at each line (a lost count is fine). Grouped by the thread that bumps them.
    struct alignas(64) PassStats {
        // GPUMem's.
        std::array<std::array<std::atomic<u64>, static_cast<size_t>(PassOutcome::Count)>,
                   static_cast<size_t>(PassMethod::Count)>
            outcomes{};
        std::atomic<u64> mirror_checks{};
        std::atomic<u64> mirror_clear{};
        std::atomic<u64> mirror_wrong{};
        std::array<std::array<std::atomic<u64>, 3>, 2> fence_entries{}; ///< [kind][PassReady]
        /// GPUMem from the lend to the return: waiting for the back (the run's end), for the
        /// front while the run is kept lent; by the run's compute launches (RunBucket), with the
        /// whole time (following = whole - back - front).
        std::array<std::atomic<u64>, RunBuckets> split_runs{};
        std::array<std::atomic<u64>, RunBuckets> split_ns{};
        std::array<std::atomic<u64>, RunBuckets> split_back_ns{};
        std::array<std::atomic<u64>, RunBuckets> split_front_ns{};
        std::atomic<u64> delay_runs{};
        std::atomic<u64> delay_ns{};
        std::atomic<u64> records{};
        std::atomic<u64> record_ranges{};
        std::array<std::array<std::atomic<u64>, static_cast<size_t>(P3Outcome::Count)>,
                   static_cast<size_t>(P3Kind::Count)>
            p3{};
        std::atomic<u64> p3_sync_points{};
        std::atomic<u64> p3_pages{};
        std::atomic<u64> p3_pages_max{};
        std::array<std::atomic<u64>, static_cast<size_t>(P3Kind::Count)> p3_passed{};
        std::atomic<u64> p3_targets{};
        // The back's: its waits for the stage after a run's end, before its next packet.
        alignas(64) std::atomic<u64> idle_waits{};
        std::atomic<u64> idle_ns{};
        std::array<std::atomic<u64>, 2> served_takes{}; ///< [found writes]
        std::atomic<u64> records_applied{};
        std::array<std::atomic<u64>, 2> p3_back_syncs{}; ///< [from a record's targets]
        // The front's: puller releases [timestamped unmapped, mapped, fence unmapped, mapped].
        alignas(64) std::array<std::atomic<u64>, 4> front_releases{};
    };
    PassStats pass_stats;
    /// Set by NoteGatherRun (GPUMem, after the back's run of a gather it asked for).
    bool gather_last_synced{};
    /// FRITCHII_MEMSTAGE_PASS: the records (one writer, GPUMem; one reader, the back).
    std::array<PassRecord, PassSlots> pass_records{};
    alignas(64) std::atomic<u64> pass_tail{0}; ///< pushed (GPUMem)
    alignas(64) std::atomic<u64> pass_head{0}; ///< freed (the back)
    /// The back: frees every record of an entry before `before` (late ones applied).
    void BackDropPasses(u64 before) noexcept;
    /// The back: a record's half, once (apply_pass), unlisting it when listed (rule T).
    void BackApplyRecord(PassRecord& record) noexcept;
    /// A must-be-0 of the passes was counted: the switch runs as 0 from the next frame end.
    std::atomic<bool> pass_refused{false};
    int pass_logged{-1};
    /// The back: a run ended and no packet was taken since (its waits then are idle).
    bool back_after_run{};
    /// GPUMem, right after a run's end: FRITCHII_DELAY_RUN_US / _KIND's spin.
    void DelayRun(u64 launches) noexcept;
    /// A hand-over delay test: FRITCHII_DELAY_HANDOVER_US / _KIND's spin (1: GPUMem after a wait
    /// for the front that waited, 2: the back after a wait for the stage, 3: GPUMem after any
    /// other stage wait that waited). Counted in delay_runs / delay_ns.
    void DelayHandover(int kind_here) noexcept;
    void LogPasses(double seconds);
    struct Snapshot {
        std::array<u64, static_cast<size_t>(HandOff::Count)> calls{};
        u64 call_ns{};
        u64 stage_ns{};
        u64 served{};
        u64 served_ns{};
        u64 back_sleeps{};
        u64 stage_sleeps{};
        u64 stage_input_sleeps{};
        std::array<u64, static_cast<size_t>(Check::Count)> checks{};
        std::array<u64, static_cast<size_t>(Served::Count)> served_calls{};
        std::array<u64, 4> entries{};
        u64 runs{};
        u64 run_entries{};
        u64 run_ns{};
        u64 back_run_ns{};
        u64 back_wait{};
        u64 back_wait_ns{};
        std::array<u64, static_cast<size_t>(StageWait::Count)> stage_wait_ns{};
        u64 stage_busy_ns{};
        u64 fill_ns{};
        u64 packets{};
        u64 packet_bytes{};
        u64 taken{};
        u64 publishes{};
        u64 copies{};
        u64 flag_handovers{};
        std::array<u64, 8> level_draws{};
        u64 hashes{};
        u64 hash_mismatches{};
        std::array<u64, WindowBuckets> window{};
        u64 ahead_sum{};
        std::array<u64, static_cast<size_t>(LevelReason::Count)> levels{};
        u64 served_ahead{};
        u64 catchups{};
        u64 catchup_ns{};
        u64 deferred_lends{};
        u64 back_lock_waits{};
        u64 stage_lock_serves{};
        u64 join_catchups{};
        u64 joins_unknown{};
        u64 write_levels{};
        u64 staging_fallbacks{};
        u64 hook_flushes{};
        u64 ring_wakes{};
        u64 stage_idles{};
        u64 run_input_waits{};
        std::array<u64, static_cast<size_t>(ListDraw::Count)> list_draws{};
        u64 list_drawn{};
        std::array<u64, static_cast<size_t>(GatherOutcome::Count)> gathers{};
        u64 gather_ns{};
        u64 gather_synced{};
        u64 gather_ranges{};
        u64 gather_gpu_written{};
        u64 gathers_taken{};
        u64 gathers_taken_run{};
        std::array<u64, static_cast<size_t>(ReleaseOutcome::Count)> releases{};
        u64 release_ns{};
        u64 releases_taken{};
    };
    void Take(Snapshot& out) const noexcept;
    Snapshot reported{};
    /// FRITCHII_MEMSTAGE_LIST_DRAWS's last value logged (-1: none yet).
    int list_draws_logged{-1};
    /// FRITCHII_MEMSTAGE_FAST_LOOP's last value logged (-1: none yet).
    int fast_loop_logged{-1};
    /// FRITCHII_MEMSTAGE_ENTRY_FAST's last value logged (-1: none yet).
    int entry_fast_logged{-1};
    /// FRITCHII_MEMSTAGE_CB_RUNS's last value logged (-1: none yet).
    int cb_runs_logged{-1};
    u64 last_log_ns{};
    u64 back_run_seen_ns{};

    std::jthread thread;
};

} // namespace VideoCommon
