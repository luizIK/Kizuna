// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#ifdef _MSC_VER
#include <intrin.h>
#elif defined(ARCHITECTURE_x86_64)
#include <x86intrin.h>
#endif
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/gpu_split.h"
#include "common/logging.h"
#include "common/cpu_features.h"
#include "common/thread_cpu_clock.h"
#include "common/nx2_hitch.h"
#include "common/nx2_timeline.h"

// EDEN_NX2_GPU_PROFILE: time where the GPU thread spends its time. Each zone counts its own
// (exclusive) time, nested zones excluded; every 10 s the thread logs each zone's share.
namespace VideoCommon::Nx2Profile {

enum class Zone : u32 {
    Other, ///< GPU thread time outside every zone below (register writes, command parsing)
    Idle,
    PendingSleep,
    ResumePending,
    Push,
    Tick,
    Run,
    FlushInvalidate,
    Macro,
    Draw,
    Clear,
    Compute,
    FlushWork,
    PipelineLookup,
    Configure,
    SyncDescriptors,
    ReadHandles,
    FillImageViews,
    TextureBuffers,
    UpdateBuffers,
    DescriptorAcquire,
    BindStageBuffers,
    PushDescriptors,
    RenderTargets,
    ConfigureDraw,
    DynamicState,
    DrawRecord,
    TextureUpload,
    Fence,        ///< Fences made (SignalFence)
    FenceSkipped, ///< Empty fences not made
    Method,       ///< Engine method calls (what a split GPU thread's back half would run)
    SplitDrain,   ///< Split GPU thread: the front waits for the back (gpu_split.h)
    SplitWait,    ///< Split GPU thread: the back waits for work
    WriterWait,
    ListSubmitWait,
    ListGpuWait,
    MemoryQuery, ///< Device::QueryDeviceMemoryUsage, the driver's memory budget query (EXP29)
    QueryDmaWait, ///< Macro DMA reads of reports waiting for query results (FlushQueryResults)
    ReadFlush, ///< Guest reads of GPU-written memory run by TickWork (FlushRegion, Finish)
    // Of compute: the steps of a dispatch (RasterizerVulkan::DispatchCompute, Configure).
    ComputeLookup,   ///< CurrentComputePipeline
    ComputePointers, ///< Storage buffer pointers read (BindComputeStorageBuffer)
    ComputeViews,    ///< Texture and image handles read, image views found (FillImageViews)
    ComputeResolve,  ///< Bindings resolved to buffers (UpdateComputeBuffers)
    ComputeSync,     ///< Buffers synchronized and bound (BindHostComputeBuffers)
    // TEST: fine zones, EDEN_NX2_GPU_PROFILE=2 only (FineScope): they split the zones above.
    CbBind,          ///< Of method: constant buffer binds (Maxwell3D::ProcessCBBind)
    CbUpload,        ///< Of method: constant buffer words written to memory
    QueryReport,     ///< Of method: report semaphores (Maxwell3D::ProcessQueryGet)
    CounterReset,    ///< Of method: counter resets (Maxwell3D::ProcessCounterReset)
    RenderEnable,    ///< Of method: render enable conditions (Maxwell3D::ProcessQueryCondition)
    Sink,            ///< State writes held for the next executable method (ConsumeSinkImpl)
    MacroEnd,        ///< A front macro's end on the back (Maxwell3D::FinishFrontMacro)
    DrawSetup,       ///< Of draw: after the pipeline lookup, before Configure (cache locks)
    DrawQueries,     ///< Of draw: query and transform feedback state before recording
    TickWork,        ///< Of draw and compute: the sync operations after them (GPU::TickWork)
    UniformBuffers,  ///< Of bind stage buffers: the uniform buffers
    GeometryBuffers, ///< Of update buffers: index and vertex buffer binds
    StorageBindings, ///< Of read handles: storage buffer addresses from constant buffers
    // Of bind stage buffers (BufferCache::BindHostStageBuffers): where its ~140 ms/s at the
    // Castle Town market go. "uniform buffers" keeps the per-slot rest (touch, offset, tracker).
    UniformBank,     ///< Of uniform buffers: the native 64 KiB bank check (page compares)
    UniformStream,   ///< Of uniform buffers: small buffers streamed (rebound or copied)
    UniformClassic,  ///< Of uniform buffers: the host buffer synchronized and bound
    StageStorage,    ///< Of bind stage buffers: the storage buffers
    StageTexBuffers, ///< Of bind stage buffers: the texture buffers
    // Of stage storage buffers (BufferCache::BindHostGraphicsStorageBuffers): a sample capture
    // put 91% of it in SynchronizeBuffer's tracker walk. The rest (touch, usage, host bind) stays
    // in "stage storage buffers".
    StorageSync,     ///< Of stage storage buffers: SynchronizeBuffer (tracker walk and uploads)
    StorageMark,     ///< Of stage storage buffers: written bindings marked GPU-modified
    // Texture cache images (fable52_answer_potroom 4): made and destroyed on the back.
    ImageCreate,        ///< A VkImage made for a texture cache image (not taken from the pool)
    PoolDestroyHere,    ///< An evicted pool image destroyed on the spot (TickFrame, eviction)
    PoolDestroyIdle,    ///< EXP99=1: evicted pool images destroyed in the back's idle hook
    PoolDestroyBacklog, ///< EXP110: the idle backlog destroyed at the frame's end
    MemoryFill,
    MemoryPlay,
    MemoryHandOff,
    MemorySyncWait,
    StageWait,
    LevelRun,
    StageEntries, ///< GPUMem's entry loop, the entries themselves (fine zone)
    StageCbWords, ///< ... their constant buffer words (fine zone, inside StageEntries)
    // Of stage entries (fine zones): what GPUMem's entry loop spends per kind of entry. What
    // stays in "stage entries" is the per-entry bookkeeping (classify, the copy's lookup, runs,
    // counters) and the closures.
    StagePairs,     ///< Pairs and StateDelta blocks' writes on the copy
    StageMethods,   ///< Method and Multi entries' handlers on the copy (words and draws excluded)
    StagePackets,   ///< Writes-tagged draw packets and DrawBatch entries on the copy
    StageDrawSetup, ///< A draw the copy issued: serve or level it (its fill excluded)
    // Of draw, the back with FRITCHII_MEMSTAGE_THREAD=2 (fine zones):
    RingTake,     ///< The ring B packet's take (TakeRingDraw: peek, counters, wakes, ticks)
    RingTakeCopy, ///< ... the packet's bytes copied out of ring B (inside RingTake)
    DrawLocks,    ///< The caches' locks taken for the draw (the texture cache's when served)
    DrawAfter,    ///< From Configure's end to dynamic state: trackers, traces, dumps
    RingFacts,    ///< Of draw setup, THREAD=2: the facts published under the stage's key
    RestTail,     ///< Of draw: PlayRest after the play (held flags, ticks mark, the watch count)
    // The three halves of a dispatch (fable_memstage_compute_brief 3; fine zones):
    InlineUpload, ///< Of method: an inline upload's write (AccelerateInlineToMemoryImpl)
    DispatchFill, ///< Of compute: the dispatch's buffer fill (MemStage::FillDispatchHere)
    QmdSignal,    ///< Of method: a QMD signal (scheduler, releases; the dispatch is "compute")
    // fable_stage_levelpoints 15.7 (fine zones). GPUMem's draw setup (StageFillDraw, inside
    // "stage draw setup"):
    SetupWindow,  ///< The window and ring B's room tested before the fill (StageBeforeFill)
    SetupProgram, ///< The program id and the facts lookup (or FRITCHII_MEMSTAGE_SETUP_KEEP's keep)
    SetupLevel,   ///< The window's tests: a writing draw, the ranges over GPU-written data
    SetupLock,    ///< The buffer cache's lock taken (a served call in the wait is its own zone)
    SetupInputs,  ///< The copy's cache inputs, the repack layout, the draw's number
    SetupHandles, ///< The draw's raw handle values read (SEAM 3, FillGraphicsHandleValues)
    // The fill (MemStage::FillDrawHere, inside "memory fill"; the brackets stay there):
    FillRanges,    ///< The stage's collected constant buffer ranges applied (ring_pre)
    FillBindSetup, ///< The uniform state, the storage binds, the texture buffer unbinds
    FillGeometry,  ///< The vertex and index buffers (UpdateGraphicsBuffers, BindHostGeometryBuffers)
    FillStages,    ///< Each stage's buffers (BindHostStageBuffers; its uniform and storage zones out)
    FillPacket,    ///< The trackers' record and the packet's push (PushRingDraw)
    // The front (15.5 row 5: its "push" doubles with the stage on):
    FrontMethods, ///< Of push and resume pending: a GP entry's methods (ProcessCommands; macros out)
    Publish,      ///< The split's publishes (GpuSplit::Publish; the wakes out)
    PublishWake,  ///< Of publish: a wake of the sleeping back or GPUMem (notify)
    OnefParked,   ///< One follower: E parked in a lend (it waits for R to play the call)
    // FRITCHII_ESPLIT=1 (fritchii_e_split_thread.h):
    ESplitJob,      ///< E2 runs a piece of E's work E1 handed over
    ESplitWait,     ///< E2 waits for work
    ESplitHandOver, ///< E1 hands work to E2 and waits for it, or for E2's room (S2)
    // Helper CP (Kakariko): E's draw method outside the fill (0x585's "method" share, 1.1 us a
    // draw), fine zones. E's lookup is "pipeline lookup", its TickWork "tick work".
    DrawRoute,     ///< E: RouteDraw's tests before the fill (Classify, the mask, UnlentBlocker)
    DrawFlush,     ///< E: the stream's invalidations before the fill (FlushCaching)
    DrawHandles,   ///< E: the draw's raw handle values read (FillGraphicsHandleValues)
    DrawFillSetup, ///< E: FillRing's own work (facts, room tests, the cache lock, repack, record)
    DrawPush,      ///< E: R's packet packed and pushed (OneFollower::PushDraw)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Zone::Count)> zone_names{
    "other",         "idle",          "pending sleep",  "resume pending", "push",
    "tick",          "run",           "flush/inval",    "macro",          "draw",
    "clear",         "compute",       "flush work",     "pipeline lookup", "configure",
    "sync descriptors", "read handles", "fill image views", "texture buffers", "update buffers",
    "descriptor acquire", "bind stage buffers", "push descriptors", "render targets",
    "configure draw", "dynamic state", "draw record", "texture upload", "fence",
    "fence skipped", "method", "split drain", "split wait",
    "writer wait", "list submit wait", "list GPU wait", "memory query",
    "query DMA wait", "read flush",
    "compute lookup", "compute pointers", "compute views", "compute resolve",
    "compute synchronize",
    "cb bind", "cb upload", "query report", "counter reset", "render enable", "sink",
    "macro end", "draw setup", "draw queries", "tick work", "uniform buffers",
    "geometry buffers", "storage bindings",
    "uniform bank check", "uniform stream", "uniform classic", "stage storage buffers",
    "stage texture buffers", "storage synchronize", "storage written marks",
    "image create", "pool destroy here", "pool destroy idle", "pool destroy backlog",
    "memory fill", "memory play", "memory hand-off", "memory sync wait",
    "stage wait", "level run", "stage entries", "stage cb words",
    "stage pairs", "stage methods", "stage packets", "stage draw setup",
    "ring take", "ring take copy", "draw locks", "draw after configure", "ring facts",
    "rest tail", "inline upload", "dispatch fill", "qmd signal",
    "setup window", "setup program", "setup level tests", "setup lock", "setup inputs",
    "setup handle values", "fill ranges", "fill bind setup", "fill geometry", "fill stage buffers",
    "fill packet", "front methods", "publish", "publish wake", "parked in lends",
    "e split job", "e split E2 wait", "e split hand-over",
    "draw route", "draw flush caching", "draw handle values", "draw fill setup", "draw push",
};
static_assert(zone_names.back() != nullptr, "a zone without a name");

// Read once, when the GPU thread starts (Initialize): Enabled() runs several times per method
// write (no guard to check). Not at static initialization: nx2_switches.txt is only applied in
// main(). Off while EDEN_NX2_SAMPLE_PROFILE samples (common/nx2_sample_profile.h): the zone timers
// would show up in the samples.
inline bool g_enabled = false;
/// TEST: EDEN_NX2_GPU_PROFILE=2 adds the fine zones (FineScope) and counters (FineCounter), at
/// the cost of more time stamp reads: =1 keeps the zones runs compare with.
inline bool g_fine = false;

inline void Initialize() {
    g_enabled = std::getenv("EDEN_NX2_GPU_PROFILE") != nullptr &&
                std::getenv("EDEN_NX2_SAMPLE_PROFILE") == nullptr;
    const char* value = std::getenv("EDEN_NX2_GPU_PROFILE");
    g_fine = g_enabled && value != nullptr && value[0] == '2';
    // Scope's entry test (one load for the profiler and the timeline).
    if (g_enabled) {
        Common::Nx2Timeline::g_zone_gate.fetch_or(Common::Nx2Timeline::ZoneGateProfile,
                                                  std::memory_order_release);
    } else {
        Common::Nx2Timeline::g_zone_gate.fetch_and(
            static_cast<u8>(~Common::Nx2Timeline::ZoneGateProfile), std::memory_order_release);
    }
}

inline bool Enabled() {
    return g_enabled;
}

inline bool Fine() {
    return g_fine;
}

// Nanoseconds; from the host's own counter where the wall clock reads it directly (the zones
// run ~150,000 times a second, and QueryPerformanceCounter cost ~5% of the GPU thread).
inline u64 Now() {
    return static_cast<u64>(Common::g_wall_clock.GetTimeNS().count());
}

/// What a front macro memory read (MME DMA) had to wait for: the newest unexecuted writer.
enum class DmaWaitCause : u32 {
    ComputeCopy,       ///< Compute or copy work
    HostSync,          ///< The work before a host-level wait
    ConstBufferUpload, ///< A 3D constant buffer upload (const_buffer.buffer)
    InlineUpload,      ///< 3D inline data
    Macro3D,           ///< A 3D macro register
    Report,            ///< A 3D report (report_semaphore.query)
    DmaLaunch,         ///< 3D launch_dma
    SyncInfo,          ///< 3D sync_info
    WaitForIdle,       ///< 3D wait_for_idle
    Other3D,           ///< Another memory-writing 3D method
    OtherEngine,       ///< A compute, DMA, inline-to-memory or 2D engine method
    Deferred,          ///< Puller semaphore releases and maps
    AvoidedByRange,    ///< Not waited: the pending writes' ranges miss the read (RangeLastWrite)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DmaWaitCause::Count)>
    dma_wait_cause_names{"compute/copy",  "host sync",    "cbuf upload", "inline upload",
                         "3D macro",      "report",       "dma launch",  "sync info",
                         "wait for idle", "other 3D",     "other engine", "deferred",
                         "avoided (range)"};
// What the zones and drain waits measure in: the time stamp counter without the fences the wall
// clock reads it with (those fences cost ~10% of the GPU back thread with ~45,000 draws a
// second). Report converts it with the wall clock.
inline u64 Ticks() {
#ifdef ARCHITECTURE_x86_64
    return __rdtsc();
#else
    return Now();
#endif
}

/// TEST: amounts counted with EDEN_NX2_GPU_PROFILE=2, logged per second in a line of their own.
enum class FineCounter : u32 {
    UniformRebinds,  ///< Small uniform buffers bound again from the same stream allocation
    UniformStreamed, ///< Small uniform buffers copied into the stream buffer
    UniformCached,   ///< Uniform buffers bound from their host buffer
    UniformOther,    ///< Uniform buffers bound otherwise (root tables, OpenGL, fallbacks)
    UniformBytes,    ///< Bytes of the uniform buffers bound (all paths)
    StorageBinds,    ///< Graphics storage buffers bound
    TextureHandles,  ///< Graphics texture handles read
    StorageKiB,      ///< KiB of the graphics storage bindings synchronized (one range each)
    StorageWritten,  ///< Graphics storage bindings marked GPU-written
    StorageMulti,    ///< Graphics storage bindings bound as several ranges (BindMultiRangeStorage)
    StorageNoUpload, ///< Graphics storage synchronizes that uploaded nothing
    StorageUploads,  ///< Graphics storage synchronizes that uploaded something
    StorageMemoHits, ///< Graphics storage synchronizes skipped (EDEN_NX2_STORAGE_SYNC_MEMO)
    // fable_memstage_compute_brief 3 (compute and the upload words):
    UploadWordsMacro,  ///< Front: compute upload data words a macro emitted
    UploadWordsMethod, ///< Front: compute upload data words sent by FrontCallMethod
    UploadWordsMulti,  ///< Front: compute upload data words sent by FrontCallMultiMethod
    InlineSize4,       ///< Inline uploads (AccelerateInlineToMemoryImpl) of 4 bytes or less
    InlineSize64,      ///< ... of 5-64 bytes
    InlineSize256,     ///< ... of 65-256 bytes
    InlineSizeMore,    ///< ... of more
    InlineNeverWritten, ///< ... to pages never GPU-written (EXP85's branch: no cache lock)
    InlineInPlace,     ///< ... uploaded in place into GPU-written buffer data
    InlineNotUploaded, ///< ... under the cache's lock, not uploaded (invalidations follow)
    InlineUnmapped,    ///< ... to an unmapped destination
    DispatchMarksBuffer, ///< Dispatches whose fill marks a buffer GPU-written
    DispatchMarkBytes,   ///< ... the bytes those marks cover
    DispatchImagesOnly,  ///< Dispatches that mark no buffer but write images
    DispatchMarksNothing, ///< Dispatches that mark no buffer and write no image
    DispatchIndirect,    ///< Indirect dispatches
    DispatchTexelBuffers, ///< Dispatches with texel buffers (their fill comes late)
    DispatchSkipped,     ///< Dispatches DispatchCompute ended before the record
    DispatchReplaced,    ///< Dispatches the upscaler replaced
    DispatchListCopies,  ///< Dispatches whose written command list pages were copied
    DispatchExtentProbes, ///< Dispatches that left extent probes to record
    // Served calls made inside a dispatch (OnStage), in MemoryStageLink::Served's order.
    ServedInDispatch,
    ServedInDispatchForRead,
    ServedInDispatchGather,
    ServedInDispatchInvalidation,
    ServedInDispatchInner,
    ServedInDispatchMustFlush,
    ServedInDispatchFallback,
    // EXP124 (fable_memstage_compute_brief 4.4): a macro's upload words as one entry.
    UploadEntries,        ///< Front: word batches sent (FrontSendUploadWords; GpuSplit::MacroWords
                          ///< makes one ring entry per 8,192 words of a batch)
    UploadEntryWords,     ///< Front: ... their data words
    InlinePiecesOne,      ///< Back: pieces (a 4 KiB page each) written in one call
    InlinePiecesWritten,  ///< Back: pieces word by word, the page may be GPU-written (or EXP85 off)
    InlinePiecesUnmapped, ///< Back: pieces word by word, unmapped or on sparse memory
    InlineWordsGrid,      ///< Back: flushes word by word, a grid placeholder live
    InlineWordsUnaligned, ///< Back: flushes word by word, not on a 4-byte boundary
    InlinePiecesChecked,  ///< Back, EXP124=2: one-call pieces whose words were each asked
    InlinePiecesDisagree, ///< Back, EXP124=2: ... whose words disagreed (must be 0)
    // fable_stage_compute 14: QMD releases by kind (KeplerCompute's release callback, in its
    // branch order) and QMD signals by the dispatches each one ran.
    QmdReleaseReduction, ///< Releases with the reduction bit (26)
    QmdReleaseFourWords, ///< ... else four-word (report) releases
    QmdReleaseMaximum,   ///< ... else maximum releases (bit 19)
    QmdReleasePlain,     ///< ... else plain one-word releases
    QmdSignalDispatches0, ///< Signals that ran no dispatch
    QmdSignalDispatches1, ///< ... one
    QmdSignalDispatches2, ///< ... 2-3
    QmdSignalDispatches4, ///< ... 4-7
    QmdSignalDispatches8, ///< ... 8 or more
    // fable_stage_levelpoints 15.7, for 15.3 row 4's census: the 3D register writes GPUMem's copy
    // follows (memstage_shadow.cpp, CountWriteGroup), by register group, in this order.
    StageWritesProgram,     ///< pipelines, program_region
    StageWritesVertexStreams, ///< vertex_streams, vertex_stream_limits, _instances, _substitute
    StageWritesVertexAttribs, ///< vertex_attrib_format
    StageWritesIndexBuffer, ///< index_buffer and its first/subsequent words
    StageWritesDrawParams,  ///< draw, vertex_buffer, bases, instances, topology, inline indices
    StageWritesCbBind,      ///< const_buffer's address, size and offset, bind_groups
    StageWritesCbData,      ///< const_buffer.buffer (the constant buffer data words)
    StageWritesTextures,    ///< tex_header, tex_sampler, sampler_binding, the bindless slot
    StageWritesRenderTargets, ///< rt, zeta, surface clip, rt_control, layers, color masks
    StageWritesViewports,   ///< viewport_transform, viewports, windows, scissors, window origin
    StageWritesBlend,       ///< blend, blend_per_target, blend color, logic op
    StageWritesDepthStencil, ///< depth, stencil, depth bias, depth bounds, polygon offset
    StageWritesRaster,      ///< cull, front face, polygon modes, point and line, alpha test
    StageWritesTransformFeedback, ///< transform_feedback, stream_out_layout, its enable
    StageWritesShadowScratch, ///< shadow_scratch, falcon
    StageWritesOther,       ///< everything else below the macro registers
    StageWritesExecutable,  ///< Of the above: executable methods (CallMethod's handlers)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(FineCounter::Count)>
    fine_counter_names{
        "uniform rebinds", "uniform streamed", "uniform cached",  "uniform other",
        "uniform bytes",   "storage binds",    "texture handles", "storage KiB",
        "storage written binds", "storage multi-range binds", "storage syncs without upload",
        "storage syncs uploading", "storage sync memo hits",
        "upload words (macro emit)", "upload words (method)", "upload words (multi-method)",
        "inline uploads 4 B", "inline uploads 5-64 B", "inline uploads 65-256 B",
        "inline uploads over 256 B", "inline uploads to never GPU-written pages",
        "inline uploads uploaded in place", "inline uploads not uploaded",
        "inline uploads unmapped", "dispatches marking a buffer GPU-written",
        "dispatch marked bytes", "dispatches writing images only",
        "dispatches marking nothing", "indirect dispatches", "dispatches with texel buffers",
        "dispatches skipped", "dispatches replaced", "dispatches copying list pages",
        "dispatches with extent probes", "served in a dispatch: FlushRegion",
        "served in a dispatch: FlushRegionForRead", "served in a dispatch: gather",
        "served in a dispatch: single invalidation", "served in a dispatch: InnerInvalidation",
        "served in a dispatch: MustFlushRegion", "served in a dispatch: fallback fill",
        "upload word batches (EXP124)", "upload batch words", "upload pieces in one call",
        "upload pieces word by word (page may be GPU-written)",
        "upload pieces word by word (unmapped or sparse)",
        "upload flushes word by word (grid placeholder live)",
        "upload flushes word by word (not word-aligned)", "upload pieces checked (EXP124=2)",
        "upload pieces whose words disagree (must be 0)",
        "qmd releases reduction", "qmd releases four words", "qmd releases maximum",
        "qmd releases plain", "qmd signals with 0 dispatches", "qmd signals with 1 dispatch",
        "qmd signals with 2-3 dispatches", "qmd signals with 4-7 dispatches",
        "qmd signals with 8+ dispatches",
        "stage writes: program", "stage writes: vertex streams", "stage writes: vertex attributes",
        "stage writes: index buffer", "stage writes: draw parameters",
        "stage writes: constant buffer binds", "stage writes: constant buffer data",
        "stage writes: textures and samplers", "stage writes: render targets",
        "stage writes: viewports and scissors", "stage writes: blend",
        "stage writes: depth and stencil", "stage writes: raster", "stage writes: transform feedback",
        "stage writes: shadow scratch", "stage writes: other",
        "stage writes: of them executable",
    };
static_assert(fine_counter_names.back() != nullptr, "a fine counter without a name");

/// fable_stage_levelpoints 15.7: what the front did while GPUMem waited for its next publish.
/// The drain sites first (the front finished a wait for the back at that site), then these.
enum class FrontDid : u32 {
    Idle = static_cast<u32>(DrainSite::Count), ///< finished an idle wait (no command list)
    Pending,     ///< finished a pending sleep (a suspended channel's poll gap)
    WorkingInto, ///< no wait; published as it began a wait for the back (Drain, WaitFor)
    Working,     ///< no wait; published while working
    Count,
};
inline constexpr size_t FrontWaitKinds = static_cast<size_t>(FrontDid::Count);

struct ThreadData {
    std::array<u64, static_cast<size_t>(FineCounter::Count)> fine_counters{}; ///< TEST
    /// GPUMem's waits for the front by FrontDid (StageInputWaited): ns, waits; and those
    /// in which the front finished more than one wait.
    std::array<u64, FrontWaitKinds> front_wait_ns{};
    std::array<u64, FrontWaitKinds> front_wait_count{};
    u64 front_wait_several{};
    std::array<u64, static_cast<size_t>(Zone::Count)> exclusive_ns{};
    std::array<u64, static_cast<size_t>(Zone::Count)> count{};
    std::array<u64, 64> counters{}; ///< See Counter (static_assert below)
    std::array<u64, static_cast<size_t>(DmaWaitCause::Count)> dma_wait_causes{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_ns{};
    std::array<u64, static_cast<size_t>(DrainSite::Count)> drain_count{};
    u64 child_ns{};    ///< Time of the zones nested in the current one
    u64 window_start{};
    u64 window_start_ticks{};
    std::optional<u64> cpu_start_ns;
    u32 depth{};          ///< Zones open (Scope, timed SampledScope)
    u32 samples_left{1};  ///< SampledScope: entries until the next timed one
    u32 sample_weight{1}; ///< SampledScope: the entries the next timed one stands for
    u32 sample_seed{0x2545F491};
};

inline thread_local ThreadData t_data;

/// The zones a timeline records without detail: waits, whole draws, dispatches and macros.
[[nodiscard]] constexpr bool IsCoarse(Zone zone) {
    switch (zone) {
    case Zone::Idle:
    case Zone::PendingSleep:
    case Zone::ResumePending:
    case Zone::Push:
    case Zone::Macro:
    case Zone::Draw:
    case Zone::Clear:
    case Zone::Compute:
    case Zone::FlushWork:
    case Zone::TextureUpload:
    case Zone::Fence:
    case Zone::SplitDrain:
    case Zone::SplitWait:
    case Zone::WriterWait:
    case Zone::ListSubmitWait:
    case Zone::ListGpuWait:
    case Zone::QueryDmaWait:
    case Zone::ReadFlush:
    case Zone::ImageCreate:
    case Zone::PoolDestroyHere:
        return true;
    default:
        return false;
    }
}

/// The zones that mark EDEN_NX2_STALL_SNAPSHOT's span slot: the ones that can hold a thread up
/// (waits, sleeps, host GPU waits, read flushes, host image creates and destroys) and the front's
/// per-list push, so a busy front reads "push" rather than "(no span)". Per-draw, per-dispatch,
/// per-macro and per-fence work zones (draw, compute, macro, fence, flush work...) do not: with
/// the profile off they cost the inline gate check only, as with the snapshot off (fable_front_
/// drains 8.4). Waits inside them have zones or timeline spans of their own (split drain, "wait
/// for host GPU", "fence: wait for fence", "wait for Vulkan worker"). The split's wait for work
/// ("split wait") is marked by its timeline span.
[[nodiscard]] constexpr bool IsStallZone(Zone zone) {
    switch (zone) {
    case Zone::Idle:
    case Zone::PendingSleep:
    case Zone::Push:
    case Zone::SplitDrain:
    case Zone::WriterWait:
    case Zone::ListSubmitWait:
    case Zone::ListGpuWait:
    case Zone::QueryDmaWait:
    case Zone::ReadFlush:
    case Zone::ImageCreate:
    case Zone::PoolDestroyHere:
        return true;
    default:
        return false;
    }
}

#if defined(_MSC_VER)
#define NX2_PROFILE_FORCEINLINE __forceinline
#define NX2_PROFILE_NOINLINE __declspec(noinline)
#else
#define NX2_PROFILE_FORCEINLINE inline __attribute__((always_inline))
#define NX2_PROFILE_NOINLINE __attribute__((noinline))
#endif

/// A zone. With the profiler off and no timeline recording (every normal run), entering one is
/// one relaxed load of Common::Nx2Timeline::g_zone_gate and a not-taken branch, inlined at the
/// call site; leaving it tests one byte of the object. The timing and the timeline span are in
/// Begin and End, out of line. A stall zone (IsStallZone) also loads Nx2Hitch::t_stall_slot and,
/// with the stall snapshot on, writes the slot without reading a clock.
class Scope {
public:
    NX2_PROFILE_FORCEINLINE explicit Scope(Zone zone_) {
        if (Common::Nx2Timeline::ZoneGateOpen()) [[unlikely]] {
            Begin(zone_, Enabled(), true);
        } else if (IsStallZone(zone_)) {
            // A constant at every call site: no code at all for the other zones.
            BeginStall(zone_);
        }
    }

    /// A zone timed only if `on` (a profiled run's choice, such as Fine()); on the timeline only
    /// if `timeline_on` too.
    NX2_PROFILE_FORCEINLINE Scope(Zone zone_, bool on, bool timeline_on) {
        if (on || (timeline_on && Common::Nx2Timeline::ZoneGateOpen())) [[unlikely]] {
            Begin(zone_, on, timeline_on);
        } else if (timeline_on && IsStallZone(zone_)) {
            BeginStall(zone_);
        }
    }

    NX2_PROFILE_FORCEINLINE ~Scope() {
        if (state != 0) [[unlikely]] {
            End();
        }
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    static constexpr u8 StateTimed = 1;
    static constexpr u8 StateTimeline = 2;
    static constexpr u8 StateStall = 4;

    /// EDEN_NX2_STALL_SNAPSHOT with the gate closed (profile off, no recording): a wait zone marks
    /// the thread's span slot, if it has one. No clock read (Nx2Hitch::StallTickNs).
    NX2_PROFILE_FORCEINLINE void BeginStall(Zone zone_) {
        Common::Nx2Hitch::StallSlot* const slot = Common::Nx2Hitch::t_stall_slot;
        if (slot != nullptr) {
            stall_mark = Common::Nx2Hitch::StallEnterSlow(
                slot, zone_names[static_cast<size_t>(zone_)]);
            state = StateStall;
        }
    }

    NX2_PROFILE_NOINLINE void Begin(Zone zone_, bool on, bool timeline_on) {
        zone = zone_;
        const bool timeline = timeline_on && Common::Nx2Timeline::Active() &&
                              (Common::Nx2Timeline::Detail() || IsCoarse(zone_));
        state = static_cast<u8>((on ? StateTimed : 0) | (timeline ? StateTimeline : 0));
        // EDEN_NX2_STALL_SNAPSHOT: the wait zones also mark the thread's span slot.
        if (Common::Nx2Hitch::t_stall_slot != nullptr && IsStallZone(zone_)) [[unlikely]] {
            stall_mark = Common::Nx2Hitch::StallEnter(zone_names[static_cast<size_t>(zone_)]);
            state = static_cast<u8>(state | StateStall);
        }
        if (timeline) {
            timeline_start = Common::Nx2Timeline::Now();
        }
        if (!on) {
            return;
        }
        auto& data = t_data;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    NX2_PROFILE_NOINLINE void End() {
        if ((state & StateTimeline) != 0) {
            Common::Nx2Timeline::Record(zone_names[static_cast<size_t>(zone)], timeline_start,
                                        Common::Nx2Timeline::Now());
        }
        if ((state & StateStall) != 0) {
            Common::Nx2Hitch::StallLeave(stall_mark);
        }
        if ((state & StateTimed) == 0) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        const auto index = static_cast<size_t>(zone);
        data.exclusive_ns[index] += elapsed - data.child_ns;
        ++data.count[index];
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    // Only `state` is written on the off path; the rest is set by Begin before End reads it.
    u8 state{}; ///< StateTimed | StateTimeline | StateStall; 0: nothing to do on exit
    Zone zone;
    u64 start;
    u64 parent_child_ns;
    u64 timeline_start;
    Common::Nx2Hitch::StallMark stall_mark; ///< Set by Begin with StateStall only
};

/// TEST: a fine zone (EDEN_NX2_GPU_PROFILE=2), on the timeline only then too.
class FineScope : public Scope {
public:
    explicit FineScope(Zone zone_) : Scope{zone_, Fine(), Fine()} {}
};

/// The ticks two time stamp reads in a row take: what a timed entry measures on top of its own
/// time (the least of a few tries).
inline u64 TimingOverheadTicks() {
    static const u64 overhead = [] {
        u64 least = ~u64{0};
        for (int i = 0; i < 64; ++i) {
            const u64 first = Ticks();
            const u64 second = Ticks();
            least = (std::min)(least, second - first);
        }
        return least;
    }();
    return overhead;
}

/// A zone entered millions of times a second, such as the split's back's 3D register writes
/// (Maxwell3D::CallMethod): timing every entry took two time stamp reads each, which cost more
/// than most entries and slowed the very runs it measured. Entered outside every other zone, one
/// entry in about SampleEvery is timed and stands for the entries since the last timed one; all
/// are counted. Entered inside another zone it does nothing: the entry is that zone's time.
/// Not recorded on the Nx2Timeline (a span per sampled method would say nothing there).
class SampledScope {
public:
    static constexpr u32 SampleEvery = 32;
    /// About 25 us at 4 GHz (time stamp ticks): far longer than these entries take.
    static constexpr u64 LongTicks = 100'000;

    /// `entries`: what this entry counts as (a run of constant buffer words counts each word).
    explicit SampledScope(Zone zone, u32 entries = 1) {
        if (!Enabled()) [[likely]] {
            return;
        }
        auto& data = t_data;
        if (data.depth != 0) {
            return;
        }
        index = static_cast<size_t>(zone);
        data.count[index] += entries;
        if (--data.samples_left != 0) {
            return;
        }
        weight = data.sample_weight;
        // The next interval: 1 to 2 * SampleEvery - 1 entries, so that no fixed period lines up
        // with a repeating pattern of entries (xorshift32).
        u32 seed = data.sample_seed;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        data.sample_seed = seed;
        data.sample_weight = 1 + seed % (2 * SampleEvery - 1);
        data.samples_left = data.sample_weight;
        ++data.depth;
        parent_child_ns = data.child_ns;
        data.child_ns = 0;
        start = Ticks();
    }

    ~SampledScope() {
        if (weight == 0) {
            return;
        }
        const u64 elapsed = Ticks() - start;
        auto& data = t_data;
        // Less the reads' own cost, which the untimed entries did not pay.
        const u64 overhead = TimingOverheadTicks();
        const u64 own = elapsed - data.child_ns;
        const u64 measured = own > overhead ? own - overhead : 0;
        // An entry this long was most likely interrupted or preempted: it counts once, not for
        // the untimed entries too.
        data.exclusive_ns[index] += measured > LongTicks ? measured : measured * weight;
        data.child_ns = parent_child_ns + elapsed;
        --data.depth;
    }

    SampledScope(const SampledScope&) = delete;
    SampledScope& operator=(const SampledScope&) = delete;

private:
    size_t index{};
    u64 weight{}; ///< 0: not timed
    u64 start{};
    u64 parent_child_ns{};
};

/// Amounts counted on the GPU thread, logged per second with the zones.
enum class Counter : u32 {
    DownloadBytes,  ///< Buffer data copied back from the host GPU at fences
    DownloadCopies, ///< Buffer ranges copied back
    DmaReadBytes,   ///< Guest memory the native MME read with DMA
    DownloadBytesOnDmaPages, ///< Of DownloadBytes: on pages the MME has DMA-read
    MethodWrites,   ///< 3D register writes (Maxwell3D::CallMethod below the macro range)
    SameValueWrites, ///< Of MethodWrites: the register already held the value
    AcquireChecks,  ///< Semaphore acquires evaluated
    AcquireWaits,   ///< Of AcquireChecks: not met yet (the channel waits)
    AcquiresFast,   ///< EDEN_NX2_ACQUIRE_PRECHECK: acquires met without draining the split
    AcquiresQueued, ///< Acquires met by a release still queued for the split's back
    PtrCacheHits,   ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page pointer cached
    PtrCacheMisses, ///< EDEN_NX2_HANDLE_PTR_CACHE: handle/descriptor reads, page looked up
    UboInPlaceSame, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers unchanged, nothing copied
    UboInPlaceChanged, ///< EDEN_NX2_UBO_INPLACE: small uniform buffers changed, copied once
    AcquireNoRelease,  ///< Drained acquires: no release of the word was sent this session
    AcquireReleaseRan, ///< Drained acquires: the last release of the word already ran on the back
    AcquireLaterWrite, ///< Drained acquires: work sent after its release may write the word
    AcquirePayloadMiss, ///< Drained acquires: the queued release's payload does not meet it
    AcquireMetAfterDrain, ///< Drained acquires met at once after the drain
    DmaReadsForwarded, ///< MME DMA reads served from a queued constant buffer upload
    DmaReadsEarly,  ///< MME DMA reads without a drain (Maxwell3D::FrontDmaReadEarly)
    DmaReadsWaitUnknown, ///< Of those, waited for a write the front can't place
    DmaReadsWaitGpu,     ///< waited for all queued GPU work with unknown destinations
    DmaReadsWaitWrite,   ///< waited for a noted write to the words' pages
    AcquirePolls,   ///< Polls of a waiting acquire (Puller::CanResumeAcquire)
    AcquirePollsFast, ///< Of AcquirePolls: answered from the word without draining the back
    SemNoFastPath,    ///< SemaphoreReadableNow false: split off or EDEN_NX2_SEM_FAST=0
    SemGpuWrittenPage, ///< SemaphoreReadableNow false: the word's page is marked GPU-written
    SemAsyncPending,  ///< SemaphoreReadableNow false: a counter report or timed release is pending
    SemWritersWait,   ///< SemaphoreReadableNow false: WaitForWritesTo gave up
    AcquireNoDrain,   ///< Acquires of a current word not met yet that registered without a drain
    // The compute bindings line (Report): what a dispatch's storage buffers cost.
    ComputeStorageBindings,  ///< Compute storage buffers bound (BindHostComputeStorageBuffers)
    ComputeStorageBytes,     ///< Their bound sizes
    ComputeSyncUploads,      ///< Their SynchronizeBuffer calls that uploaded something
    ComputeSyncBytes,        ///< Bytes those uploaded
    ComputeSyncDirtyWalks,   ///< Their synchronizes under EDEN_NX2_EXP59 & 4 or & 8 (dirty-word walk)
    ComputeSyncDirtySkips,   ///< Of those, all tracker words clean: the binding skipped entirely
    RangeCacheHits,          ///< Compute VirtualRangeCache::Query answered from the cache
    RangeCacheMisses,        ///< Compute VirtualRangeCache::Query that walked the page table
    StorageBindingHits,      ///< Compute StorageBufferBinding answered from its 4-way cache
    StorageBindingMisses,    ///< Compute StorageBufferBinding worked out again
    Count,
};
static_assert(static_cast<size_t>(Counter::Count) <= 64, "Counter outgrew the counters array");
static_assert(static_cast<size_t>(Counter::Count) <= std::tuple_size_v<decltype(ThreadData::counters)>);

/// Method calls on this thread by method (3D registers on the back, macros on the front), and
/// same-value writes among them; MME DMA reads by the macro that made them. The most frequent
/// are logged with the zones.
inline thread_local std::array<u32, 0x2000> t_method_count{};
inline thread_local std::array<u32, 0x2000> t_method_same{};
inline thread_local std::array<u32, 0x2000> t_dma_reads{};

inline void CountMethod(u32 method, bool same) {
    if (Enabled() && method < t_method_count.size()) {
        ++t_method_count[method];
        t_method_same[method] += same ? 1 : 0;
    }
}

/// A 3D register write (Maxwell3D::CallMethod below the macro range), `same` if the register
/// held the value already: the write counters and the per-method counts.
inline void CountRegisterWrite(u32 method, bool same) {
    if (!Enabled()) {
        return;
    }
    auto& counters = t_data.counters;
    ++counters[static_cast<size_t>(Counter::MethodWrites)];
    counters[static_cast<size_t>(Counter::SameValueWrites)] += same ? 1 : 0;
    CountMethod(method, same);
}

inline void CountDmaRead(u32 macro) {
    if (Enabled() && macro < t_dma_reads.size()) {
        ++t_dma_reads[macro];
    }
}

/// Compute dispatches on this thread by program (shader hash); the most frequent are logged
/// with the zones.
inline thread_local std::unordered_map<u64, u64> t_compute_programs;

inline void CountComputeProgram(u64 shader_hash) {
    if (Enabled()) {
        ++t_compute_programs[shader_hash];
    }
}

inline std::string TopMethods(const std::array<u32, 0x2000>& counts,
                              const std::array<u32, 0x2000>* same, u64 window) {
    std::array<u32, 0x2000> order;
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    constexpr size_t top = 24;
    std::partial_sort(order.begin(), order.begin() + top, order.end(),
                      [&counts](u32 a, u32 b) { return counts[a] > counts[b]; });
    std::string text;
    for (size_t i = 0; i < top && counts[order[i]] != 0; ++i) {
        const u32 method = order[i];
        text += fmt::format(" {:#x} {:.0f}/s", method,
                            static_cast<double>(counts[method]) * 1e9 / window);
        if (same != nullptr && (*same)[method] != 0) {
            text += fmt::format(" ({:.0f} same)", static_cast<double>((*same)[method]) * 1e9 / window);
        }
        text += ',';
    }
    if (!text.empty()) {
        text.pop_back();
    }
    return text;
}

/// Device pages (4 KiB) the native MME has DMA-read since start (GPU thread only; filled only
/// while profiling): which buffer downloads anything on the GPU thread reads back.
inline std::unordered_set<u64>& DmaReadPages() {
    static std::unordered_set<u64> pages;
    return pages;
}

inline void Add(Counter counter, u64 amount) {
    if (Enabled()) {
        t_data.counters[static_cast<size_t>(counter)] += amount;
    }
}

/// TEST: EDEN_NX2_GPU_PROFILE=2 only.
inline void AddFine(FineCounter counter, u64 amount) {
    if (Fine()) [[unlikely]] {
        t_data.fine_counters[static_cast<size_t>(counter)] += amount;
    }
}

/// fable_stage_levelpoints 15.7 (profile on): the front's last finished wait, as
/// (waits finished << 8) | FrontDid, and the site + 1 of a wait for the back it is in (0: none).
/// Written by the front alone; GPUMem reads them around its waits for the front.
inline std::atomic<u64> g_front_wait_mark{0};
inline std::atomic<u32> g_front_in_wait{0};

/// The front: a wait for the back at `site` begins (before the publish that goes with it).
inline void FrontWaitBegin(DrainSite site) {
    if (Enabled()) {
        g_front_in_wait.store(static_cast<u32>(site) + 1, std::memory_order_relaxed);
    }
}

/// The front: a wait ended (`kind`: a DrainSite or FrontDid::Idle, ::Pending).
inline void FrontWaitEnd(u32 kind) {
    if (Enabled()) {
        g_front_in_wait.store(0, std::memory_order_relaxed);
        const u64 waits = (g_front_wait_mark.load(std::memory_order_relaxed) >> 8) + 1;
        g_front_wait_mark.store((waits << 8) | kind, std::memory_order_release);
    }
}

/// The front: marks the end of an idle or pending wait (gpu_thread.cpp) at scope exit.
class FrontWaitScope {
public:
    explicit FrontWaitScope(FrontDid kind_) : kind{kind_} {}
    ~FrontWaitScope() {
        FrontWaitEnd(static_cast<u32>(kind));
    }
    FrontWaitScope(const FrontWaitScope&) = delete;
    FrontWaitScope& operator=(const FrontWaitScope&) = delete;

private:
    FrontDid kind;
};

/// GPUMem, before a wait for the front: the mark to compare with after it.
[[nodiscard]] inline u64 FrontWaitMark() {
    return Enabled() ? g_front_wait_mark.load(std::memory_order_acquire) : 0;
}

/// GPUMem waited `ns` for the front's publish; `before`: FrontWaitMark() at the wait's start.
/// The front finished a wait since: that wait's kind (the last one, if several). None: it was
/// working, and it published either as it began a wait for the back or while working.
inline void StageInputWaited(u64 before, u64 ns) {
    if (!Enabled()) {
        return;
    }
    const u64 after = g_front_wait_mark.load(std::memory_order_acquire);
    size_t kind;
    if ((after >> 8) != (before >> 8)) {
        kind = (std::min)(static_cast<size_t>(after & 0xff), FrontWaitKinds - 1);
        t_data.front_wait_several += (after >> 8) - (before >> 8) > 1 ? 1 : 0;
    } else if (g_front_in_wait.load(std::memory_order_relaxed) != 0) {
        kind = static_cast<size_t>(FrontDid::WorkingInto);
    } else {
        kind = static_cast<size_t>(FrontDid::Working);
    }
    t_data.front_wait_ns[kind] += ns;
    ++t_data.front_wait_count[kind];
}

/// The front waited for the back since `start` (Ticks) for `site`.
inline void AddDrain(DrainSite site, u64 start) {
    if (Enabled()) {
        const auto index = static_cast<size_t>(site);
        t_data.drain_ns[index] += Ticks() - start;
        ++t_data.drain_count[index];
        FrontWaitEnd(static_cast<u32>(site));
    }
}

inline void AddDmaWaitCause(DmaWaitCause cause) {
    if (Enabled()) {
        ++t_data.dma_wait_causes[static_cast<size_t>(cause)];
    }
}

/// Called by the GPU thread (and the split's back thread) between commands: logs and resets
/// the totals every 10 s.
inline void Report(const char* thread_name = "GPU thread") {
    if (!Enabled()) {
        return;
    }
    auto& data = t_data;
    const u64 now = Now();
    if (data.window_start == 0) {
        data.window_start = now;
        data.window_start_ticks = Ticks();
        data.cpu_start_ns = Common::CurrentThreadCpuTimeNs();
        data.child_ns = 0;
        return;
    }
    const u64 window = now - data.window_start;
    if (window < 10'000'000'000ULL) {
        return;
    }
    const auto cpu_now = Common::CurrentThreadCpuTimeNs();
    const u64 now_ticks = Ticks();
    const u64 window_ticks = (std::max<u64>)(now_ticks - data.window_start_ticks, 1);
    // Time in no zone at all is the wall time minus the zones' totals.
    u64 zoned{};
    for (size_t i = 1; i < data.exclusive_ns.size(); ++i) {
        zoned += data.exclusive_ns[i];
    }
    data.exclusive_ns[0] = window_ticks > zoned ? window_ticks - zoned : 0;
    std::string line;
    for (size_t i = 0; i < data.exclusive_ns.size(); ++i) {
        if (data.exclusive_ns[i] == 0 && data.count[i] == 0) {
            continue;
        }
        // Per second of wall time: milliseconds spent and calls made.
        const double ms = static_cast<double>(data.exclusive_ns[i]) * 1000.0 / window_ticks;
        const double calls = static_cast<double>(data.count[i]) * 1e9 / window;
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", zone_names[i], ms, calls);
    }
    line += fmt::format(" | buffer downloads {:.0f} KiB in {:.0f} copies ({:.0f} KiB on pages "
                        "MME DMA reads) | MME DMA reads {:.0f} KiB",
                        static_cast<double>(data.counters[0]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[1]) * 1e9 / window,
                        static_cast<double>(data.counters[3]) * 1e9 / 1024.0 / window,
                        static_cast<double>(data.counters[2]) * 1e9 / 1024.0 / window);
    line += fmt::format(" | 3D register writes {:.0f} ({:.0f} same value)",
                        static_cast<double>(data.counters[4]) * 1e9 / window,
                        static_cast<double>(data.counters[5]) * 1e9 / window);
    line += fmt::format(" | acquires {:.0f} ({:.0f} waited)",
                        static_cast<double>(data.counters[6]) * 1e9 / window,
                        static_cast<double>(data.counters[7]) * 1e9 / window);
    if (const u64 queued = data.counters[static_cast<size_t>(Counter::AcquiresQueued)]; queued != 0) {
        line += fmt::format(" | acquires met by a queued release {:.0f}",
                            static_cast<double>(queued) * 1e9 / window);
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquireChecks)] != 0) {
            line += fmt::format(" | drained acquires: no release {:.0f}, release ran {:.0f}, "
                                "later write {:.0f}, payload miss {:.0f}, met after drain {:.0f}",
                                rate(Counter::AcquireNoRelease), rate(Counter::AcquireReleaseRan),
                                rate(Counter::AcquireLaterWrite),
                                rate(Counter::AcquirePayloadMiss),
                                rate(Counter::AcquireMetAfterDrain));
        }
    }
    if (const u64 forwarded = data.counters[static_cast<size_t>(Counter::DmaReadsForwarded)];
        forwarded != 0) {
        line += fmt::format(" | MME DMA reads from queued uploads {:.0f}",
                            static_cast<double>(forwarded) * 1e9 / window);
    }
    if (const u64 fast = data.counters[static_cast<size_t>(Counter::AcquiresFast)]; fast != 0) {
        line += fmt::format(" | acquires without a drain {:.0f}",
                            static_cast<double>(fast) * 1e9 / window);
    }
    if (const u64 hits = data.counters[static_cast<size_t>(Counter::PtrCacheHits)],
        misses = data.counters[static_cast<size_t>(Counter::PtrCacheMisses)];
        hits != 0 || misses != 0) {
        line += fmt::format(" | handle page pointers {:.0f} cached, {:.0f} looked up",
                            static_cast<double>(hits) * 1e9 / window,
                            static_cast<double>(misses) * 1e9 / window);
    }
    if (const u64 same = data.counters[static_cast<size_t>(Counter::UboInPlaceSame)],
        changed = data.counters[static_cast<size_t>(Counter::UboInPlaceChanged)];
        same != 0 || changed != 0) {
        line += fmt::format(" | uniform buffers compared in place {:.0f} unchanged, {:.0f} changed",
                            static_cast<double>(same) * 1e9 / window,
                            static_cast<double>(changed) * 1e9 / window);
    }
    if (const u64 early = data.counters[static_cast<size_t>(Counter::DmaReadsEarly)]; early != 0) {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        line += fmt::format(" | MME DMA reads without a drain {:.0f} (waited for: unknown writes "
                            "{:.0f}, GPU work {:.0f}, writes to the words {:.0f})",
                            static_cast<double>(early) * 1e9 / window,
                            rate(Counter::DmaReadsWaitUnknown), rate(Counter::DmaReadsWaitGpu),
                            rate(Counter::DmaReadsWaitWrite));
    }
    {
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        if (data.counters[static_cast<size_t>(Counter::AcquirePolls)] != 0 ||
            data.counters[static_cast<size_t>(Counter::AcquireNoDrain)] != 0) {
            line += fmt::format(" | acquire polls {:.0f} ({:.0f} without a drain), acquires "
                                "waiting without a drain {:.0f}, semaphore word not readable: no "
                                "fast path {:.0f}, GPU-written page {:.0f}, report pending {:.0f}, "
                                "writers {:.0f}",
                                rate(Counter::AcquirePolls), rate(Counter::AcquirePollsFast),
                                rate(Counter::AcquireNoDrain), rate(Counter::SemNoFastPath),
                                rate(Counter::SemGpuWrittenPage), rate(Counter::SemAsyncPending),
                                rate(Counter::SemWritersWait));
        }
    }
    std::string drains;
    for (size_t i = 0; i < data.drain_count.size(); ++i) {
        if (data.drain_count[i] != 0) {
            drains += fmt::format(" {} {:.1f} ms ({:.0f}/s),", drain_site_names[i],
                                  static_cast<double>(data.drain_ns[i]) * 1000.0 / window_ticks,
                                  static_cast<double>(data.drain_count[i]) * 1e9 / window);
        }
    }
    if (!drains.empty()) {
        drains.pop_back();
        line += " | waits for the back by site:" + drains;
    }
    std::string causes;
    for (size_t i = 0; i < data.dma_wait_causes.size(); ++i) {
        if (data.dma_wait_causes[i] != 0) {
            causes += fmt::format(" {} {:.0f}/s,", dma_wait_cause_names[i],
                                  static_cast<double>(data.dma_wait_causes[i]) * 1e9 / window);
        }
    }
    if (!causes.empty()) {
        causes.pop_back();
        line += " | macro dma reads waited for:" + causes;
    }
    if (cpu_now && data.cpu_start_ns && *cpu_now >= *data.cpu_start_ns) {
        line += fmt::format(" | host_cpu_busy_ms_per_s={:.3f} window_wall_ms={:.3f}",
                            static_cast<double>(*cpu_now - *data.cpu_start_ns) * 1000.0 / window,
                            static_cast<double>(window) / 1e6);
    } else {
        line += fmt::format(" | host_cpu_busy_ms_per_s=unavailable window_wall_ms={:.3f}",
                            static_cast<double>(window) / 1e6);
    }
    LOG_INFO(HW_GPU, "{} per second:{}", thread_name, line);
    if (std::ranges::any_of(data.front_wait_count, [](u64 count) { return count != 0; })) {
        // fable_stage_levelpoints 15.7 (GPUMem): its waits for the front's next publish by what
        // the front did meanwhile (StageInputWaited).
        static constexpr std::array<const char*, 4> own{
            "idle (no command list)", "pending sleep", "working, published as it began a wait for "
            "the back", "working"};
        static_assert(own.size() == FrontWaitKinds - static_cast<size_t>(DrainSite::Count));
        std::string kinds;
        for (size_t i = 0; i < FrontWaitKinds; ++i) {
            if (data.front_wait_count[i] == 0) {
                continue;
            }
            constexpr size_t Sites = static_cast<size_t>(DrainSite::Count);
            const std::string name = i < Sites ? fmt::format("waited for the back at {}",
                                                             drain_site_names[i])
                                               : std::string{own[i - Sites]};
            kinds += fmt::format(" {} {:.1f} ms ({:.0f}/s),", name,
                                 static_cast<double>(data.front_wait_ns[i]) * 1000.0 / window,
                                 static_cast<double>(data.front_wait_count[i]) * 1e9 / window);
        }
        kinds.pop_back();
        LOG_INFO(HW_GPU,
                 "{} waits for the front by what the front did before its publish, per second:{}; "
                 "{:.0f}/s spanned several front waits (the last one counted)",
                 thread_name, kinds,
                 static_cast<double>(data.front_wait_several) * 1e9 / window);
    }
    if (Fine()) {
        // TEST: the fine counters.
        std::string fine;
        for (size_t i = 0; i < data.fine_counters.size(); ++i) {
            if (data.fine_counters[i] != 0) {
                fine += fmt::format(" {} {:.0f},", fine_counter_names[i],
                                    static_cast<double>(data.fine_counters[i]) * 1e9 / window);
            }
        }
        if (!fine.empty()) {
            fine.pop_back();
            LOG_INFO(HW_GPU, "{} fine counters per second:{}", thread_name, fine);
        }
    }
    if (const std::string methods = TopMethods(t_method_count, &t_method_same, window); !methods.empty()) {
        LOG_INFO(HW_GPU, "{} methods per second:{}", thread_name, methods);
    }
    if (const std::string reads = TopMethods(t_dma_reads, nullptr, window); !reads.empty()) {
        LOG_INFO(HW_GPU, "{} MME DMA reads per second by macro:{}", thread_name, reads);
    }
    {
        // What a dispatch's storage buffers cost: bound against uploaded bytes, and how often
        // the binding caches answered (see the compute zones for the time).
        const auto rate = [&](Counter counter) {
            return static_cast<double>(data.counters[static_cast<size_t>(counter)]) * 1e9 / window;
        };
        const u64 dispatches = data.count[static_cast<size_t>(Zone::Compute)];
        if (dispatches != 0 || data.counters[static_cast<size_t>(Counter::ComputeStorageBindings)] != 0) {
            constexpr double MiB = 1024.0 * 1024.0;
            // EDEN_NX2_EXP59 & 4 / & 8: synchronizes that found every tracker word clean.
            const std::string dirty_walk =
                data.counters[static_cast<size_t>(Counter::ComputeSyncDirtyWalks)] == 0
                    ? std::string{}
                    : fmt::format(", dirty-word walk (EXP59) skipped {:.0f} of {:.0f}",
                                  rate(Counter::ComputeSyncDirtySkips),
                                  rate(Counter::ComputeSyncDirtyWalks));
            LOG_INFO(HW_GPU,
                     "{} compute bindings per second: {:.0f} dispatches, storage bound {:.1f} MiB "
                     "in {:.0f} bindings, synchronized/uploaded {:.2f} MiB in {:.0f} uploads, "
                     "range cache {:.0f} hits / {:.0f} misses, pointer cache {:.0f} hits / "
                     "{:.0f} misses{}",
                     thread_name, static_cast<double>(dispatches) * 1e9 / window,
                     rate(Counter::ComputeStorageBytes) / MiB,
                     rate(Counter::ComputeStorageBindings), rate(Counter::ComputeSyncBytes) / MiB,
                     rate(Counter::ComputeSyncUploads), rate(Counter::RangeCacheHits),
                     rate(Counter::RangeCacheMisses), rate(Counter::StorageBindingHits),
                     rate(Counter::StorageBindingMisses), dirty_walk);
        }
    }
    if (!t_compute_programs.empty()) {
        std::vector<std::pair<u64, u64>> programs; // count, shader hash
        programs.reserve(t_compute_programs.size());
        for (const auto& [hash, count] : t_compute_programs) {
            programs.emplace_back(count, hash);
        }
        const size_t top = (std::min<size_t>)(programs.size(), 8);
        std::partial_sort(programs.begin(), programs.begin() + top, programs.end(),
                          [](const auto& a, const auto& b) { return a > b; });
        std::string text;
        for (size_t i = 0; i < top; ++i) {
            text += fmt::format(" {:016x} {:.0f},", programs[i].second,
                                static_cast<double>(programs[i].first) * 1e9 / window);
        }
        text.pop_back();
        LOG_INFO(HW_GPU, "{} compute programs per second ({} programs, top {}):{}", thread_name,
                 programs.size(), top, text);
        t_compute_programs.clear();
    }
    t_method_count = {};
    t_method_same = {};
    t_dma_reads = {};
    // The zones open around this call (Report runs inside the split's back loop) stay open.
    const u32 depth = data.depth;
    data = {};
    data.depth = depth;
    data.window_start = now;
    data.window_start_ticks = now_ticks;
    data.cpu_start_ns = cpu_now;
}

} // namespace VideoCommon::Nx2Profile
