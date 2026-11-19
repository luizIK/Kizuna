// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/fritchii_delta_path.h"
#include "video_core/nx2_debug_control.h"

/// EDEN_NX2_FASTDRAW_COUNT=1 (read once, by the first Maxwell3D): counts the draws that could
/// take the original delta ("fast") draw path, which keeps the previous draw's
/// pipeline, image views and samplers when only light state changed in between. Counting only:
/// no draw is resolved or skipped differently. While on, each pipeline also keeps its last draw
/// (GraphicsPipeline::KeptLastDraw, Fable's port plan step D2; EXP106 does that alone) for the original
/// two checks that need it (sampled images to prepare, computed handles changed). Off, each site
/// costs one test of g_on.
///
/// The rules are the ported change's (nx2_fast_draw.h, Maxwell3D::IsFastDrawParameter and its lists,
/// RasterizerVulkan::TakeReusableDraw / DeltaStillHolds / NoteResolvedDraw), with its later
/// commits 792e12a47b (vertex_id_base, vertex formats, report operands and neutral methods are
/// light; the write list widens instead of ending reuse; handle words compared one by one),
/// e05a72847e (pipelines with computed handles take delta draws) and 7824123e58 (back macro
/// calls are neutral). Every break moves a serial, as there; it also ORs the break's group into a
/// mask, cleared at each draw, so a draw that state ended shows every group that ended it, not
/// only the last one (Fable plan, fable_answer_backthread_threads.md section 8).
///
/// Step D1 of Fable's port plan (section 10.2) adds the five breaks our tree needs and the original
/// doesn't raise (writes applied over the pools, the pools written through an alias, EXP43
/// relocations, channel switches, and the render pass serial counted always), marks as neutral
/// the deferred calls whose effects now break where they act (channel bind, fence releases,
/// collected writes applied, macro read flushes, FlushRegion, list end), and logs every break
/// raised by reason and the deferred calls that still break by thunk.
namespace VideoCommon::Nx2FastDrawCount {

enum class Reason : u8 {
    Register,
    Method,
    InlineData,
    OtherEngine,
    Call,
    TickWork,
    Invalidation,
    Fence,
    IndirectReset,
    Frame,
    TextureHeaders,
    Map,
    HleMacro,
    // Fable's port plan step D1 (fable_answer_backthread_threads.md 10.2, the five breaks):
    PoolWrite, ///< A guest CPU (or cached GPU) write applied over the TIC or TSC pool
    PoolAlias, ///< A stream write through another GPU address onto the pools
    Relocate,  ///< EXP43 moved an image (TextureCache::RelocateImage)
    Channel,   ///< The rasterizer switched to another channel
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "register",       "method",        "inline data",   "other engine",        "deferred call",
    "tick work",      "invalidation",  "fence release", "indirect reset",      "frame",
    "texture header data", "memory map", "HLE macro",   "pool write",          "pool alias write",
    "image relocated", "channel switch",
};

/// The groups a draw's breaks are sorted into, a bit each. Register and method breaks go by
/// method (Maxwell3D::FastDrawCountGroup); with the original lists most of the first three are light
/// and stay 0: the mask shows what is left.
enum Group : u32 {
    VertexIdBase = 1U << 0,  ///< 0x446
    VertexLayout = 1U << 1,  ///< Vertex stream word 0 (enable, stride), attribute formats
    QueryMethods = 1U << 2,  ///< Counter reset, render enable, report operands, MME memory
    ListOverflow = 1U << 3,  ///< The 16-range write list was widened (not a break in the original)
    RealState = 1U << 4,     ///< Any other 3D register or method
    OtherEngine = 1U << 5,
    Invalidation = 1U << 6,  ///< Texture cache writes and deletions, rescales, shader removals,
                             ///< writes applied over the pools, image relocations
    Fence = 1U << 7,
    TickWork = 1U << 8,
    Call = 1U << 9,          ///< Deferred calls on the split's back not marked neutral
    Frame = 1U << 10,
    Map = 1U << 11,          ///< GPU memory map changes, unmapped images
    Other = 1U << 12,        ///< Inline data, texture header data (also through an alias),
                             ///< indirect reset, HLE macros, channel switches
};
inline constexpr u32 NumGroups = 13;
inline constexpr u32 BreakGroups = ((1U << NumGroups) - 1) & ~u32{ListOverflow};
inline constexpr std::array<const char*, NumGroups> group_names{
    "vertex_id_base", "vertex layout", "query methods", "write list overflow", "real state",
    "other engine",   "invalidation",  "fence",         "tick work",           "deferred call",
    "frame",          "memory map",    "other",
};

[[nodiscard]] constexpr u32 GroupOf(Reason reason) {
    switch (reason) {
    case Reason::OtherEngine:
        return OtherEngine;
    case Reason::Call:
        return Call;
    case Reason::TickWork:
        return TickWork;
    case Reason::Invalidation:
    case Reason::PoolWrite:
    case Reason::Relocate:
        return Invalidation;
    case Reason::Fence:
        return Fence;
    case Reason::Frame:
        return Frame;
    case Reason::Map:
        return Map;
    case Reason::Register:
    case Reason::Method:
        return RealState;
    default:
        return Other;
    }
}

/// Breaks are raised (the serial and everything below): EDEN_NX2_FASTDRAW_COUNT == "1", or
/// Fritchii's delta path (EXP118, fritchii_delta_path.h) on at boot. A plain bool: every site
/// tests it and nothing else when off.
inline bool g_on{};
/// EDEN_NX2_FASTDRAW_COUNT == "1": the count itself and its 5 s lines.
inline bool g_count{};

/// Reads the switches once (Maxwell3D's constructor: after main() applied nx2_switches.txt).
inline void Init() {
    static const bool once = [] {
        const char* const value = std::getenv("EDEN_NX2_FASTDRAW_COUNT");
        g_count = value != nullptr && value[0] == '1';
        // EXP118 is live, but its breaks must run from the start: on at boot, they stay on
        // (the switch can then go 0 to 3 and back); off at boot, it does nothing this session.
        const bool delta_path =
            VideoCore::Nx2Debug::Experiment(VideoCommon::FritchiiDeltaPath::Switch) != 0;
        g_on = g_count || delta_path;
        if (g_count) {
            LOG_INFO(HW_GPU, "Delta draw count on (EDEN_NX2_FASTDRAW_COUNT=1): counting only, "
                             "ported rules");
        }
        if (delta_path) {
            LOG_INFO(HW_GPU, "Delta path (EXP118): on at boot, its breaks are raised this session");
        }
        return true;
    }();
    (void)once;
}

[[nodiscard]] inline bool On() {
    return g_on;
}

/// A Call entry's flag on the split's ring (bit 7, free for Call entries): a deferred call that
/// can't change what a draw resolves to (the original GpuSplit::FastDrawNeutral). Set only when on.
inline constexpr u64 NeutralCallBit = u64{1} << 7;
[[nodiscard]] inline u64 Neutral() {
    return g_on ? NeutralCallBit : 0;
}

/// Whether vertex stream strides and enables are dynamic state (extended dynamic state and
/// dynamic vertex input): then they and the attribute formats are draw parameters. Set by the
/// pipeline cache.
inline bool g_vertex_streams_dynamic{};

inline std::atomic<u64> g_serial{1};
/// Groups broken since the last draw took its snapshot.
inline std::atomic<u32> g_pending{};
/// The last break's reason and, for Register and Method, its 3D method (bits 8 and up).
inline std::atomic<u32> g_last_break{};
/// Every break raised, by reason (whether or not a draw followed it), for the 5 s log.
inline std::array<std::atomic<u64>, static_cast<size_t>(Reason::Count)> g_raised{};

/// A break, when on (callers test g_on). Breaks come from several threads.
inline void BreakOn(Reason reason, u32 group, u32 method = 0) {
    g_raised[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    g_serial.fetch_add(1, std::memory_order_relaxed);
    g_last_break.store(static_cast<u32>(reason) | (method << 8), std::memory_order_relaxed);
    if ((g_pending.load(std::memory_order_relaxed) & group) != group) {
        g_pending.fetch_or(group, std::memory_order_relaxed);
    }
}

inline void Break(Reason reason) {
    if (g_on) [[unlikely]] {
        BreakOn(reason, GroupOf(reason));
    }
}

/// The deferred calls that still break (Call entries without the neutral bit), by thunk address,
/// for the 5 s log's top 8 (Fable section 9.1: which call ends the chains). Recorded by
/// GpuSplit::Execute when on.
struct CallSites {
    static constexpr size_t Size = 64;
    std::array<std::atomic<u64>, Size> thunks{};
    std::array<std::atomic<u64>, Size> counts{};
    std::atomic<u64> overflow{}; ///< Calls of thunks the table had no room for
};
inline CallSites g_call_sites;
/// The base of the module the thunks are in (lichii.exe; set by GpuSplit on Windows, else 0):
/// the log prints thunks as offsets from it, which a symbol lookup on the PDB resolves.
inline std::atomic<u64> g_module_base{};

inline void NoteCallSite(u64 thunk) {
    CallSites& sites = g_call_sites;
    size_t index = static_cast<size_t>(((thunk >> 4) * 0x9E3779B97F4A7C15ULL) >> 58);
    for (size_t probe = 0; probe < CallSites::Size; ++probe) {
        u64 key = sites.thunks[index].load(std::memory_order_relaxed);
        if (key == 0 &&
            sites.thunks[index].compare_exchange_strong(key, thunk, std::memory_order_relaxed)) {
            key = thunk;
        }
        if (key == thunk) {
            sites.counts[index].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        index = (index + 1) % CallSites::Size;
    }
    sites.overflow.fetch_add(1, std::memory_order_relaxed);
}

/// The device memory a GPU range maps to (the original DeviceSpans): a write through another GPU
/// address that maps the same device memory changes the range too.
class DeviceSpans {
public:
    /// Translates [address, address + size) again unless it was translated last and the memory
    /// manager's mappings haven't changed since. Returns whether the spans may have changed.
    template <typename MemoryManager>
    bool Update(const MemoryManager& memory_, GPUVAddr address_, u64 size_) {
        const u64 serial = memory_.MapSerial();
        if (memory == &memory_ && address == address_ && size == size_ && map_serial == serial) {
            return false;
        }
        memory = &memory_;
        address = address_;
        size = size_;
        map_serial = serial;
        spans.clear();
        for (const auto& [device, bytes] : memory_.GetSubmappedDeviceRange(address_, size_)) {
            spans.emplace_back(device, device + bytes);
        }
        std::sort(spans.begin(), spans.end());
        size_t merged = 0;
        for (const auto& span : spans) {
            if (merged != 0 && span.first <= spans[merged - 1].second) {
                spans[merged - 1].second = (std::max)(spans[merged - 1].second, span.second);
            } else {
                spans[merged++] = span;
            }
        }
        spans.resize(merged);
        return true;
    }

    /// Whether [begin, end) of device memory overlaps the range's.
    [[nodiscard]] bool Intersects(DAddr begin, DAddr end) const {
        const auto span = std::upper_bound(
            spans.begin(), spans.end(), begin,
            [](DAddr value, const std::pair<DAddr, DAddr>& item) { return value < item.second; });
        return span != spans.end() && span->first < end;
    }

    /// Sorted, disjoint [begin, end) pairs.
    [[nodiscard]] const std::vector<std::pair<DAddr, DAddr>>& Spans() const noexcept {
        return spans;
    }

private:
    const void* memory{};
    GPUVAddr address{};
    u64 size{};
    u64 map_serial{};
    std::vector<std::pair<DAddr, DAddr>> spans;
};

/// The device memory of the bound 3D engine's texture header (TIC) and sampler (TSC) pools,
/// published by the thread that runs it (Maxwell3D::UpdateFastDrawCountPools) for the threads
/// that apply guest writes (a sequence lock: one writer, any readers). Pools are not images, so
/// TextureCache::WriteMemory's break does not cover a write there (Fable 10.2, row 1).
struct PoolSpans {
    static constexpr size_t Max = 8;
    std::atomic<u32> generation{};
    std::atomic<u32> count{};
    std::array<std::atomic<u64>, Max> begin{};
    std::array<std::atomic<u64>, Max> end{};
};
inline PoolSpans g_pools;
/// The engine whose pools were published last (the publishing thread only).
inline const void* g_pools_owner{};

inline void PublishPools(const void* owner, const DeviceSpans& tic, const DeviceSpans& tsc) {
    g_pools_owner = owner;
    PoolSpans& pools = g_pools;
    const u32 generation = pools.generation.load(std::memory_order_relaxed);
    pools.generation.store(generation + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    // Too many pieces: each pool's hull (more breaks, never fewer).
    const bool fits = tic.Spans().size() + tsc.Spans().size() <= PoolSpans::Max;
    u32 count = 0;
    const auto add = [&pools, &count](DAddr begin, DAddr end) {
        pools.begin[count].store(begin, std::memory_order_relaxed);
        pools.end[count].store(end, std::memory_order_relaxed);
        ++count;
    };
    for (const DeviceSpans* pool : {&tic, &tsc}) {
        const auto& spans = pool->Spans();
        if (spans.empty()) {
            continue;
        }
        if (fits) {
            for (const auto& [begin, end] : spans) {
                add(begin, end);
            }
        } else {
            add(spans.front().first, spans.back().second);
        }
    }
    pools.count.store(count, std::memory_order_relaxed);
    pools.generation.store(generation + 2, std::memory_order_release);
}

[[nodiscard]] inline bool PoolsIntersect(DAddr begin, DAddr end) {
    const PoolSpans& pools = g_pools;
    for (;;) {
        const u32 generation = pools.generation.load(std::memory_order_acquire);
        if ((generation & 1) != 0) {
            continue; // Being published (a few stores)
        }
        const u32 count = (std::min)(pools.count.load(std::memory_order_relaxed),
                                     static_cast<u32>(PoolSpans::Max));
        bool hit = false;
        for (u32 i = 0; i < count; ++i) {
            hit |= pools.begin[i].load(std::memory_order_relaxed) < end &&
                   begin < pools.end[i].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (pools.generation.load(std::memory_order_relaxed) == generation) {
            return hit;
        }
    }
}

/// When on: a write of [addr, addr + size) of device memory applied to the caches (guest CPU
/// writes, collected or not, and the GPU's cached writes). Over a pool: a break.
inline void NotePoolWrite(DAddr addr, u64 size) {
    if (size != 0 && PoolsIntersect(addr, addr + size)) {
        BreakOn(Reason::PoolWrite, Invalidation);
    }
}

/// Light changes since the last draw that can reach texture handles: constant buffer binds and
/// inline constant buffer data by GPU address. Written and read on the thread that runs the 3D
/// engine and the rasterizer.
struct Delta {
    static constexpr size_t MaxWrites = 16;

    std::array<u32, 5> binds{}; ///< Constant buffer slots bound per stage
    std::array<u64, MaxWrites> write_begin{};
    std::array<u64, MaxWrites> write_end{};
    u32 num_writes{};
    bool widened{}; ///< The list was full and a range widened (the original: no uniform keeps then)

    [[nodiscard]] bool Empty() const {
        return num_writes == 0 && (binds[0] | binds[1] | binds[2] | binds[3] | binds[4]) == 0;
    }

    [[nodiscard]] bool Reaches(u64 address, u64 size) const {
        for (u32 write = 0; write < num_writes; ++write) {
            if (write_begin[write] < address + size && address < write_end[write]) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool ReachesWords(u64 base, std::span<const std::pair<u32, u32>> words) const {
        for (u32 write = 0; write < num_writes; ++write) {
            for (const auto& [first, second] : words) {
                if (write_begin[write] < base + second && base + first < write_end[write]) {
                    return true;
                }
            }
        }
        return false;
    }
};

inline Delta g_delta;

/// The original AppendRange: extends a range the new one touches or overlaps, else appends it; a full
/// list widens the range that grows least (true then).
inline bool AppendRange(std::array<u64, Delta::MaxWrites>& begins,
                        std::array<u64, Delta::MaxWrites>& ends, u32& count, u64 begin, u64 end) {
    if (count != 0 && ends[count - 1] == begin) {
        ends[count - 1] = end;
        return false;
    }
    for (u32 i = 0; i < count; ++i) {
        if (begin <= ends[i] && begins[i] <= end) {
            begins[i] = (std::min)(begins[i], begin);
            ends[i] = (std::max)(ends[i], end);
            return false;
        }
    }
    if (count < Delta::MaxWrites) {
        begins[count] = begin;
        ends[count] = end;
        ++count;
        return false;
    }
    u32 best = 0;
    u64 best_growth = ~u64{0};
    for (u32 i = 0; i < count; ++i) {
        const u64 growth = ((std::max)(ends[i], end) - (std::min)(begins[i], begin)) -
                           (ends[i] - begins[i]);
        if (growth < best_growth) {
            best = i;
            best_growth = growth;
        }
    }
    begins[best] = (std::min)(begins[best], begin);
    ends[best] = (std::max)(ends[best], end);
    return true;
}

/// When on: a constant buffer bind that changed the slot.
inline void NoteBind(size_t stage, u32 slot) {
    if (stage < g_delta.binds.size()) {
        g_delta.binds[stage] |= 1U << slot;
    }
}

/// EXP107, counted when on: inline constant buffer writes left out of the
/// delta because memory held their words already, and those of them not written at all.
inline std::atomic<u64> g_unchanged_writes{};
inline std::atomic<u64> g_skipped_writes{};

/// When on: inline constant buffer data written to GPU addresses [begin, end).
inline void NoteConstantData(u64 begin, u64 end) {
    Delta& delta = g_delta;
    delta.widened |= AppendRange(delta.write_begin, delta.write_end, delta.num_writes, begin, end);
}

/// What a pipeline's handles are read from (the original GraphicsPipeline handle masks), built on its
/// first counted draw.
struct PipelineInfo {
    static constexpr size_t NumStages = 5;
    static constexpr size_t NumSlots = 18;

    std::array<u32, NumStages> handle_cbufs{}; ///< Slots handles are read from, per stage
    /// The bytes [first, second) of a slot handles are read from...
    std::array<std::array<std::pair<u32, u32>, NumSlots>, NumStages> ranges{};
    /// ...and the same bytes word by word (sorted, disjoint), by (first, count) into words.
    std::vector<std::pair<u32, u32>> words;
    std::array<std::array<std::pair<u16, u16>, NumSlots>, NumStages> word_index{};
    bool writes_images{};
    bool mesh{};

    [[nodiscard]] std::span<const std::pair<u32, u32>> Words(size_t stage, u32 slot) const {
        const auto [first, count] = word_index[stage][slot];
        return std::span{words}.subspan(first, count);
    }
};

/// Why a draw would be resolved in full (the original FullDraw).
enum class Full : u8 {
    NoCandidate,
    Kind,
    StateChanged,
    Submitted,
    RenderPass,
    DeltaPipeline,
    VertexInput,
    HandleBind,
    HandleData,
    /// The previous pipeline kept no whole draw (its handles came from a TEX_BANK_GEN memo)
    NoKeptDraw,
    /// A kept view's image would be uploaded or get a newer alias copied in (SampledViewsChange)
    SampledImages,
    /// A kept computed handle evaluates to another value (ComputedHandlesChanged)
    ComputedHandles,
    Count,
};

/// Why a draw can't be the previous draw of a delta draw (the original Ineligible).
enum class Ineligible : u8 {
    Indirect,
    Mesh,
    NotBuilt,
    WritesImages,
    TransformFeedback,
    InlineIndex,
    Quads,
    Repacked,
    NoRenderPass,
    PipelineNotBound,
    OffBackThread,
    Count,
};

/// The previous eligible draw (the original ReusableDraw).
struct PreviousDraw {
    const void* pipeline{};
    const void* maxwell3d{};
    u64 serial{};
    u64 map_serial{};
    u64 tick{};
    u64 render_pass{};
    u32 topology{};
    u32 engine_state{}; ///< Maxwell3D::engine_state (the HLE macro hint the pipeline key reads)
    bool is_indexed{};
    bool valid{};
};

struct Stats {
    std::chrono::steady_clock::time_point window_start{};
    u64 draws{};     ///< PrepareDraw calls
    u64 not_drawn{}; ///< Returned before the decision (no pipeline, mesh draws skipped)
    u64 delta{};
    u64 other_pipeline{}; ///< Would-be delta draws whose full resolve picked another pipeline
    std::array<u64, static_cast<size_t>(Full::Count)> full{};
    std::array<u64, static_cast<size_t>(Ineligible::Count)> ineligible{};
    std::array<u64, static_cast<size_t>(Reason::Count)> breaks{}; ///< Last break, state changed
    std::vector<u32> break_methods = std::vector<u32>(0x1000);
    std::unordered_map<u32, u64> masks; ///< Break groups of the draws state changed
    std::array<u64, NumGroups> any{};
    std::array<u64, NumGroups> alone{};
};

/// The rasterizer's counting state (one per RasterizerVulkan, on its drawing thread).
struct Counter {
    PreviousDraw previous;
    // The draw being prepared, from Begin to Decide.
    PreviousDraw taken;
    Delta taken_delta;
    u64 take_serial{};
    u32 take_last_break{};
    bool vertex_input_dirty{};
    bool open{};
    // From Decide to Resolved.
    u64 draw_serial{};
    u64 draw_map_serial{};
    Stats stats;

    void CountStateChanged(u32 mask, u32 last_break) {
        ++stats.full[static_cast<size_t>(Full::StateChanged)];
        ++stats.breaks[(std::min)(last_break & 0xff, static_cast<u32>(Reason::Count) - 1)];
        if (const u32 method = last_break >> 8;
            method != 0 && method < stats.break_methods.size()) {
            ++stats.break_methods[method];
        }
        ++stats.masks[mask];
        const u32 breaks = mask & BreakGroups;
        for (u32 group = 0; group < NumGroups; ++group) {
            if ((mask >> group) & 1) {
                ++stats.any[group];
                if (breaks == (1U << group)) {
                    ++stats.alone[group];
                }
            }
        }
    }

    static std::string MaskName(u32 mask) {
        if (mask == 0) {
            return "none";
        }
        std::string name;
        for (u32 group = 0; group < NumGroups; ++group) {
            if ((mask >> group) & 1) {
                if (!name.empty()) {
                    name += '+';
                }
                name += group_names[group];
            }
        }
        return name;
    }

    /// Logs the window every 5 s (called once a frame).
    void MaybeLog() {
        const auto now = std::chrono::steady_clock::now();
        if (stats.window_start == std::chrono::steady_clock::time_point{}) {
            stats.window_start = now;
            // The global counts start with the window too.
            for (auto& count : g_raised) {
                count.store(0, std::memory_order_relaxed);
            }
            for (auto& count : g_call_sites.counts) {
                count.store(0, std::memory_order_relaxed);
            }
            g_call_sites.overflow.store(0, std::memory_order_relaxed);
            g_unchanged_writes.store(0, std::memory_order_relaxed);
            g_skipped_writes.store(0, std::memory_order_relaxed);
            return;
        }
        if (now - stats.window_start < std::chrono::seconds{5}) {
            return;
        }
        const double seconds = std::chrono::duration<double>(now - stats.window_start).count();
        const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
        const auto full = [&](Full reason) { return rate(stats.full[static_cast<size_t>(reason)]); };
        const u64 decided = stats.draws - stats.not_drawn;
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s, per second): {:.0f} draws, {:.0f} would-be delta "
                 "({:.1f}% of decided draws) | full: {:.0f} after an ineligible draw, {:.0f} "
                 "different kind, {:.0f} state changed, {:.0f} after a submit, {:.0f} render pass "
                 "or pipeline changed, {:.0f} pipeline takes no delta draws, {:.0f} vertex "
                 "attribute types changed, {:.0f} handle buffer bound, {:.0f} handle data "
                 "written, {:.0f} no kept draw, {:.0f} sampled image to prepare, {:.0f} computed "
                 "handle changed | {:.0f} not drawn (no pipeline or skipped) | {:.0f} would-be "
                 "delta resolved to another pipeline | constant buffer writes that changed "
                 "nothing {:.0f}, not written {:.0f} (EXP107)",
                 rate(stats.draws), rate(stats.delta),
                 decided == 0 ? 0.0
                              : 100.0 * static_cast<double>(stats.delta) /
                                    static_cast<double>(decided),
                 full(Full::NoCandidate), full(Full::Kind), full(Full::StateChanged),
                 full(Full::Submitted), full(Full::RenderPass), full(Full::DeltaPipeline),
                 full(Full::VertexInput), full(Full::HandleBind), full(Full::HandleData),
                 full(Full::NoKeptDraw), full(Full::SampledImages), full(Full::ComputedHandles),
                 rate(stats.not_drawn), rate(stats.other_pipeline),
                 rate(g_unchanged_writes.exchange(0, std::memory_order_relaxed)),
                 rate(g_skipped_writes.exchange(0, std::memory_order_relaxed)));
        static constexpr std::array<const char*, static_cast<size_t>(Ineligible::Count)>
            ineligible_names{"indirect",       "mesh",          "not built",
                             "writes images",  "transform feedback", "inline index",
                             "quads",          "repacked vertices",  "no render pass",
                             "pipeline not bound", "off the back thread"};
        std::string ineligible;
        for (size_t reason = 0; reason < stats.ineligible.size(); ++reason) {
            if (stats.ineligible[reason] != 0) {
                ineligible += fmt::format(" {} {:.0f},", ineligible_names[reason],
                                          rate(stats.ineligible[reason]));
            }
        }
        if (!ineligible.empty()) {
            ineligible.pop_back();
        }
        LOG_INFO(Render_Vulkan, "Delta draw count (5 s) ineligible draws per second:{}",
                 ineligible.empty() ? std::string{" none"} : ineligible);
        std::string breaks;
        for (size_t reason = 0; reason < stats.breaks.size(); ++reason) {
            if (stats.breaks[reason] != 0) {
                breaks += fmt::format(" {} {:.0f},", reason_names[reason],
                                      rate(stats.breaks[reason]));
            }
        }
        if (!breaks.empty()) {
            breaks.pop_back();
        }
        std::vector<std::pair<u32, u32>> methods;
        for (u32 method = 1; method < stats.break_methods.size(); ++method) {
            if (stats.break_methods[method] != 0) {
                methods.emplace_back(stats.break_methods[method], method);
            }
        }
        const size_t shown = (std::min)(methods.size(), size_t{12});
        std::partial_sort(methods.begin(), methods.begin() + shown, methods.end(),
                          std::greater{});
        std::string top;
        for (size_t i = 0; i < shown; ++i) {
            top += fmt::format(" {:#x}:{:.0f}", methods[i].second, rate(methods[i].first));
        }
        // Every break raised (D1: the pool write, alias, relocate and channel rates), and the
        // deferred calls that still break by thunk, top 8.
        std::string raised;
        for (size_t reason = 0; reason < g_raised.size(); ++reason) {
            if (const u64 count = g_raised[reason].exchange(0, std::memory_order_relaxed);
                count != 0) {
                raised += fmt::format(" {} {:.0f},", reason_names[reason], rate(count));
            }
        }
        if (!raised.empty()) {
            raised.pop_back();
        }
        std::vector<std::pair<u64, u64>> calls;
        for (size_t i = 0; i < CallSites::Size; ++i) {
            const u64 thunk = g_call_sites.thunks[i].load(std::memory_order_relaxed);
            if (const u64 count = g_call_sites.counts[i].exchange(0, std::memory_order_relaxed);
                thunk != 0 && count != 0) {
                calls.emplace_back(count, thunk);
            }
        }
        const size_t shown_calls = (std::min)(calls.size(), size_t{8});
        std::partial_sort(calls.begin(), calls.begin() + shown_calls, calls.end(),
                          std::greater{});
        const u64 module_base = g_module_base.load(std::memory_order_relaxed);
        std::string call_list;
        for (size_t i = 0; i < shown_calls; ++i) {
            const u64 thunk = calls[i].second;
            call_list += module_base != 0 && thunk >= module_base
                             ? fmt::format(" +{:#x}:{:.0f}", thunk - module_base,
                                           rate(calls[i].first))
                             : fmt::format(" {:#x}:{:.0f}", thunk, rate(calls[i].first));
        }
        if (const u64 overflow = g_call_sites.overflow.exchange(0, std::memory_order_relaxed);
            overflow != 0) {
            call_list += fmt::format(" other:{:.0f}", rate(overflow));
        }
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s) state changed per second, by last break:{} | top "
                 "methods by last break:{} | breaks raised per second:{} | deferred calls "
                 "that broke per second, top thunks (offset from the module base):{}",
                 breaks.empty() ? std::string{" none"} : breaks,
                 top.empty() ? std::string{" none"} : top,
                 raised.empty() ? std::string{" none"} : raised,
                 call_list.empty() ? std::string{" none"} : call_list);
        std::string any;
        std::string alone;
        for (u32 group = 0; group < NumGroups; ++group) {
            if (stats.any[group] != 0) {
                any += fmt::format(" {} {:.0f},", group_names[group], rate(stats.any[group]));
            }
            if (stats.alone[group] != 0) {
                alone += fmt::format(" {} {:.0f},", group_names[group], rate(stats.alone[group]));
            }
        }
        if (!any.empty()) {
            any.pop_back();
        }
        if (!alone.empty()) {
            alone.pop_back();
        }
        std::vector<std::pair<u64, u32>> masks;
        masks.reserve(stats.masks.size());
        for (const auto& [mask, count] : stats.masks) {
            masks.emplace_back(count, mask);
        }
        const size_t shown_masks = (std::min)(masks.size(), size_t{16});
        std::partial_sort(masks.begin(), masks.begin() + shown_masks, masks.end(),
                          std::greater{});
        std::string mask_list;
        for (size_t i = 0; i < shown_masks; ++i) {
            mask_list += fmt::format(" [{}] {:.0f},", MaskName(masks[i].second),
                                     rate(masks[i].first));
        }
        if (!mask_list.empty()) {
            mask_list.pop_back();
        }
        LOG_INFO(Render_Vulkan,
                 "Delta draw count (5 s) break groups of the state-changed draws per second, "
                 "group set:{} | group alone:{} | top masks:{}",
                 any.empty() ? std::string{" none"} : any,
                 alone.empty() ? std::string{" none"} : alone,
                 mask_list.empty() ? std::string{" none"} : mask_list);
        stats = Stats{};
        stats.window_start = now;
    }
};

} // namespace VideoCommon::Nx2FastDrawCount
