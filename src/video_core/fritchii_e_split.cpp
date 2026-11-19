// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_ESPLIT_COUNT: the E split's dry step S0 (fritchii_e_split.h; work\E_SPLIT_DESIGN.md).
//
// Everything here runs on E (GpuSplit's back thread), or on R inside a lent call while E is
// parked (the record hooks take R's writes to E's engine there, as the one follower's delta
// does). Counters have that one writer at a time: plain fields.
//
// What E2 reads of the 3D engine (W2, Watched2 below), from the code: the buffer cache reads the
// engine only through CacheInputs (cache_inputs.h; no direct engine read is left in
// buffer_cache.h or vk_mem_stage.cpp), whose register accessors reach the vertex streams and
// their limits, the native stream sizes (0x600), the transform feedback block and its enable,
// and the vertex attribute formats with the streams' strides (the repack layout,
// VertexRepackMask); the index buffer block raises the IndexBuffer flag. FillRing reads the render target formats (FRITCHII_FILL_BRACKET_LEAN). The
// E2 entries read the constant buffer selection (size, address, offset: the words' destination),
// the report semaphore block, sync_info, clear_report_value, the render enable block and its
// override (a condition decided on E2), and a bind's group and selection (its buffer cache
// half). Beside the registers the fill reads by value: the constant buffer bindings (state), the
// root tables, the draw manager's draw state and a list draw's indirect parameters, and the
// buffer flags (IndexBuffer, VertexBuffers, VertexBuffer0..31), which E2 owns in the split.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/buffer_cache/cache_inputs.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"
#include "video_core/one_follower_hooks.h"

namespace VideoCommon::ESplit {

namespace {

namespace OneF = VideoCommon::OneFollower;
using Regs = Maxwell3D::Regs;
using Flags = Maxwell3D::DirtyState::Flags;

constexpr u32 NumRegs = static_cast<u32>(Regs::NUM_REGS);
constexpr size_t ClassCount = static_cast<size_t>(Class::Count);
constexpr size_t JoinCount = static_cast<size_t>(JoinCause::Count);
constexpr size_t StallCount = static_cast<size_t>(StallCause::Count);
constexpr u32 Slots = static_cast<u32>(Regs::MaxConstBuffers);
constexpr u32 Bindings = static_cast<u32>(Regs::MaxShaderStage) * Slots;
constexpr u32 RootWords = 8 * 64;
/// 1 in 64 entries (and hand-overs) timed, picked by a xorshift (no aliasing with the stream's
/// own periods); sums are scaled by 64.
constexpr u64 SampleScale = 64;

// --- W2: the register words E2 reads ---------------------------------------------------------------

struct Watched2 {
    std::bitset<NumRegs> words;
    std::vector<std::pair<u32, u32>> ranges; ///< [begin, end), merged

    Watched2() {
        const auto add = [this](size_t first, size_t count) {
            for (size_t i = first; i < first + count && i < NumRegs; ++i) {
                words.set(i);
            }
        };
        constexpr size_t Word = sizeof(u32);
        add(MAXWELL3D_REG_INDEX(vertex_streams), sizeof(Regs::vertex_streams) / Word);
        add(MAXWELL3D_REG_INDEX(vertex_stream_limits), sizeof(Regs::vertex_stream_limits) / Word);
        add(MAXWELL3D_REG_INDEX(vertex_attrib_format), sizeof(Regs::vertex_attrib_format) / Word);
        add(0x600 / Word, 2 * Regs::NumVertexArrays); // NativeVertexStreamSize
        add(MAXWELL3D_REG_INDEX(transform_feedback), sizeof(Regs::transform_feedback) / Word);
        add(MAXWELL3D_REG_INDEX(transform_feedback_enabled), 1);
        // The fill reads the index buffer from the draw state; its registers are here for the
        // IndexBuffer flag their writes raise (dirty_flags.cpp), which E2 owns.
        add(MAXWELL3D_REG_INDEX(index_buffer), sizeof(Regs::index_buffer) / Word);
        add(MAXWELL3D_REG_INDEX(const_buffer.size), 1);
        add(MAXWELL3D_REG_INDEX(const_buffer.address_high), 1);
        add(MAXWELL3D_REG_INDEX(const_buffer.address_low), 1);
        add(MAXWELL3D_REG_INDEX(const_buffer.offset), 1);
        const size_t rt_words = sizeof(Regs::rt) / Regs::NumRenderTargets / Word;
        for (size_t i = 0; i < Regs::NumRenderTargets; ++i) {
            add(MAXWELL3D_REG_INDEX(rt) + i * rt_words +
                    offsetof(Regs::RenderTargetConfig, format) / Word,
                1);
        }
        add(MAXWELL3D_REG_INDEX(report_semaphore), sizeof(Regs::report_semaphore) / Word);
        add(MAXWELL3D_REG_INDEX(sync_info), 1);
        add(MAXWELL3D_REG_INDEX(clear_report_value), 1);
        add(MAXWELL3D_REG_INDEX(render_enable), sizeof(Regs::render_enable) / Word);
        add(MAXWELL3D_REG_INDEX(render_enable_override), 1);
        add(MAXWELL3D_REG_INDEX(bind_groups), sizeof(Regs::bind_groups) / Word);
        for (u32 i = 0; i < NumRegs;) {
            if (!words.test(i)) {
                ++i;
                continue;
            }
            u32 end = i;
            while (end < NumRegs && words.test(end)) {
                ++end;
            }
            ranges.emplace_back(i, end);
            i = end;
        }
    }
};

const Watched2& W2() {
    static const Watched2 watched;
    return watched;
}

Flags BufferFlagMask() {
    Flags mask;
    mask.set(Dirty::IndexBuffer);
    mask.set(Dirty::VertexBuffers);
    for (u32 i = 0; i < Regs::NumVertexArrays; ++i) {
        mask.set(Dirty::VertexBuffer0 + i);
    }
    return mask;
}

const Flags& BufferFlags() {
    static const Flags mask = BufferFlagMask();
    return mask;
}

// The buffer flags are one run of the flag numbers: VertexBuffers, VertexBuffer0..31, IndexBuffer.
constexpr u32 FlagFirst = Dirty::VertexBuffers;
constexpr u32 FlagLast = Dirty::IndexBuffer;
static_assert(Dirty::VertexBuffer0 == FlagFirst + 1 && Dirty::VertexBuffer31 == FlagFirst + 32 &&
              FlagLast == FlagFirst + 33);
static_assert(FlagProbe::FlagIndex == Dirty::VertexBuffers);
constexpr u64 PackedVertexBuffers = 1;
constexpr u64 PackedBindings = u64{0xffff'ffff} << 1;
constexpr u64 PackedIndexBuffer = u64{1} << 33;

/// The buffer flags as 34 bits (bit 0 VertexBuffers, 1..32 VertexBuffer0..31, 33 IndexBuffer).
[[nodiscard]] u64 PackFlags(const Flags& flags) noexcept {
    static const Flags low = [] {
        Flags bits;
        for (u32 i = 0; i <= FlagLast - FlagFirst; ++i) {
            bits.set(i);
        }
        return bits;
    }();
    return ((flags >> FlagFirst) & low).to_ullong();
}

[[nodiscard]] Flags UnpackFlags(u64 packed) noexcept {
    return Flags{packed} << FlagFirst;
}

/// The packed buffer flag a register word's dirty table entry names, 0 for any other flag.
[[nodiscard]] u64 PackedBit(u8 flag) noexcept {
    return flag >= FlagFirst && flag <= FlagLast ? u64{1} << (flag - FlagFirst) : 0;
}

// --- Counters ------------------------------------------------------------------------------------

enum class Diff : u8 { Register, Binding, RootWord, DrawState, Indirect, BufferFlag, Count };
constexpr size_t DiffCount = static_cast<size_t>(Diff::Count);
constexpr std::array<const char*, DiffCount> DiffNames{"register",   "binding",  "root word",
                                                       "draw state", "indirect", "buffer flag"};

struct Counts {
    std::array<u64, ClassCount> entries{};
    std::array<u64, ClassCount> ticks{}; ///< timed entries only
    u64 timed{};
    std::array<u64, JoinCount> joins{};
    u64 lends{};
    std::array<u64, StallCount> stalls{};
    u64 conditions{};
    u64 back_raises{};
    u64 draws{};
    // 2 and 3
    std::array<u64, 2> records{}; ///< [0] E2 entry, [1] draw
    u64 record_pairs{};
    u64 record_bytes{};
    u64 draw_pairs{};
    u64 build_ticks{};
    u64 consume_ticks{};
    u64 compare_ticks{};
    std::array<u64, DiffCount> differs{};
    u64 compares{};
    u64 seeds{};
    u64 untracked{};
    u64 flag_syncs{};
    // The buffer flag take (draw records): records that carried one, and its flags by kind.
    u64 take_records{};
    u64 take_vertex_buffers{};
    u64 take_bindings{}; ///< per-binding flags (bits)
    u64 take_index_buffer{};
    /// VertexBuffers 0 to 1 edges by site: FlagSite, then an E2 entry, a join, a lend.
    std::array<u64, static_cast<size_t>(FlagSite::Count) + 3> rises{};
    // Channel calls (E1's; E2 takes the bind in order): all, and the ones whose bind lent.
    u64 channel_calls{};
    u64 channel_lends{};
    // 3
    u64 view_fills{};
    u64 fallback_differ{};
    u64 fallback_inline{};
    u64 after_checks{};
    u64 after_differs{};
};

Counts g_c;
u64 g_differs_total{};
u64 g_after_total{};
int g_level{};
u64 g_batches{};
u64 g_since_ns{};
u64 g_since_ticks{};
u32 g_rng = 0x2545f491u;
thread_local bool t_back{}; ///< set on E (BatchLevel's thread)
/// FRITCHII_ESPLIT_DEPTH > 0: BuildInto takes every buffer flag off the engine (E1's take).
bool g_take_all{};

constexpr size_t RiseE2Entry = static_cast<size_t>(FlagSite::Count);
constexpr size_t RiseJoin = RiseE2Entry + 1;
constexpr size_t RiseLend = RiseE2Entry + 2;
/// The viewed engine's flags, for the state tracker hook on any thread (written on E at a seed).
std::atomic<const void*> g_view_flags{};
/// Its counts by thread: E, R in a lent call (E parked), any other (unordered with E).
std::array<std::atomic<u64>, 3> g_tracker_raises{};
bool g_entry_vb{};   ///< the engine's VertexBuffers before an E2 entry or a join (2 and 3)
bool g_lend_vb{};    ///< ... before a lend

[[nodiscard]] bool Take() noexcept {
    u32 x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return (x & (SampleScale - 1)) == 0;
}

[[nodiscard]] u64 Ticks() noexcept {
    return Nx2Profile::Ticks();
}

// --- The running entry and its carves -------------------------------------------------------------

EntryInfo g_entry{};
Class g_entry_class{};
Class g_cur{};
u64 g_seg_start{};
std::array<Class, 16> g_stack{};
u32 g_depth{};
bool g_lend_pushed{};
bool g_shader_window{}; ///< an E2 stream write onto shader pages since the last lookup or sync

// --- Engines -------------------------------------------------------------------------------------

const Tegra::Engines::EngineInterface* g_kind_engine{};
EngineKind g_kind{EngineKind::None};
const Maxwell3D* g_engine3d{};

// --- Records (2 and 3) ---------------------------------------------------------------------------

std::bitset<NumRegs> g_noted;
std::vector<u16> g_touched;
std::array<u64, 2> g_binds{}; ///< bit stage * 18 + slot
std::array<u64, RootWords / 64> g_roots{}; ///< bit table * 64 + index
View g_view;
std::vector<u8> g_ring;
size_t g_ring_at{};
bool g_own_mark_hook{};

struct RecordHeader {
    u32 bytes;
    u32 pairs;
    u16 binds;
    u16 roots;
    u8 draw;
    u8 indirect;
    u8 take; ///< a draw record: the buffer flag take (one u64) follows the roots
    u8 reserved;
    u64 position;
};
static_assert(sizeof(RecordHeader) % 8 == 0);
struct RecordPair {
    u32 index;
    u32 value;
};
struct RecordBind {
    u8 stage;
    u8 slot;
    u8 enabled;
    u8 reserved;
    u32 size;
    u64 address;
};
static_assert(sizeof(RecordBind) == 16);
struct RecordRoot {
    u16 table;
    u16 index;
    u32 value;
};
/// The draw manager's state a fill reads, by value (DrawManager::State without its byte vector).
struct DrawPart {
    Regs::VertexBuffer vertex_buffer;
    Regs::IndexBuffer index_buffer;
    u64 list_draw_address;
    u64 list_keep_id;
    u32 topology;
    u32 draw_mode;
    u32 draw_element_count;
    u32 base_index;
    u32 base_instance;
    u32 instance_count;
    u32 drawn_instances;
    u8 draw_indexed;
    u8 reserved[3];
};
constexpr size_t RingBytes = size_t{1} << 18;
constexpr size_t MaxRecordBytes = sizeof(RecordHeader) + NumRegs * sizeof(RecordPair) +
                                  Bindings * sizeof(RecordBind) + RootWords * sizeof(RecordRoot) +
                                  sizeof(u64) + sizeof(DrawPart) +
                                  sizeof(Maxwell3D::DrawManager::IndirectParams) + 64;
static_assert(MaxRecordBytes < RingBytes);

[[nodiscard]] constexpr size_t Padded(size_t bytes) noexcept {
    return (bytes + 7) & ~size_t{7};
}

[[nodiscard]] bool Tracking() noexcept {
    return OneF::g_track_writes_always || OneF::g_track_writes.load(std::memory_order_relaxed);
}

[[nodiscard]] bool OnE() noexcept {
    return t_back || OneF::InLent();
}

void ClearPending() noexcept {
    for (const u16 index : g_touched) {
        g_noted.reset(index);
    }
    g_touched.clear();
    g_binds = {};
    g_roots = {};
}

void TakeDraw(Maxwell3D::DrawManager::State& out, const Maxwell3D::DrawManager::State& in) {
    out.topology = in.topology;
    out.draw_mode = in.draw_mode;
    out.draw_indexed = in.draw_indexed;
    out.draw_element_count = in.draw_element_count;
    out.base_index = in.base_index;
    out.vertex_buffer = in.vertex_buffer;
    out.index_buffer = in.index_buffer;
    out.base_instance = in.base_instance;
    out.instance_count = in.instance_count;
    out.drawn_instances = in.drawn_instances;
    out.list_draw_address = in.list_draw_address;
    out.list_keep_id = in.list_keep_id;
}

void Seed(const Maxwell3D& engine) {
    std::memcpy(&g_view.regs, &engine.regs, sizeof(Regs));
    g_view.state = engine.state;
    g_view.root_tables = engine.root_tables;
    TakeDraw(g_view.draw_state, engine.draw_manager.draw_state);
    g_view.indirect = engine.draw_manager.indirect_state;
    g_view.flags = engine.dirty.flags & BufferFlags();
    g_view.engine = &engine;
    g_view.seeded = true;
    g_view_flags.store(&engine.dirty.flags, std::memory_order_relaxed);
    ClearPending();
    ++g_c.seeds;
}

size_t BuildInto(u8* base, const Maxwell3D& engine, bool draw, bool list_args);

/// E1: the record of everything noted since the last one, into the ring.
const u8* Build(const Maxwell3D& engine, bool draw, bool list_args) {
    if (g_ring.size() != RingBytes) {
        g_ring.assign(RingBytes, 0);
        g_ring_at = 0;
    }
    if (g_ring_at + MaxRecordBytes > RingBytes) {
        g_ring_at = 0;
    }
    u8* const base = g_ring.data() + g_ring_at;
    g_ring_at += BuildInto(base, engine, draw, list_args);
    return base;
}

/// The record at `base` (room for MaxRecordBytes); returns its size.
size_t BuildInto(u8* base, const Maxwell3D& engine, bool draw, bool list_args) {
    RecordHeader header{};
    size_t at = sizeof(RecordHeader);
    const auto& table0 = engine.dirty.tables[0];
    const auto& table1 = engine.dirty.tables[1];
    u64 table_bits = 0; ///< the buffer flags this record's words raise (Consume raises them)
    for (const u16 index : g_touched) {
        g_noted.reset(index);
        const RecordPair pair{index, engine.regs.reg_array[index]};
        std::memcpy(base + at, &pair, sizeof(pair));
        at += sizeof(pair);
        ++header.pairs;
        table_bits |= PackedBit(table0[index]) | PackedBit(table1[index]);
    }
    g_touched.clear();
    at = Padded(at);
    for (u32 word = 0; word < g_binds.size(); ++word) {
        for (u64 bits = g_binds[word]; bits != 0; bits &= bits - 1) {
            const u32 bit = word * 64 + static_cast<u32>(std::countr_zero(bits));
            const auto& info = engine.state.shader_stages[bit / Slots].const_buffers[bit % Slots];
            const RecordBind bind{
                .stage = static_cast<u8>(bit / Slots),
                .slot = static_cast<u8>(bit % Slots),
                .enabled = static_cast<u8>(info.enabled ? 1 : 0),
                .reserved = 0,
                .size = info.size,
                .address = info.address,
            };
            std::memcpy(base + at, &bind, sizeof(bind));
            at += sizeof(bind);
            ++header.binds;
        }
    }
    g_binds = {};
    for (u32 word = 0; word < g_roots.size(); ++word) {
        for (u64 bits = g_roots[word]; bits != 0; bits &= bits - 1) {
            const u32 bit = word * 64 + static_cast<u32>(std::countr_zero(bits));
            const RecordRoot root{static_cast<u16>(bit / 64), static_cast<u16>(bit % 64),
                                  engine.root_tables.tables[bit / 64][bit % 64]};
            std::memcpy(base + at, &root, sizeof(root));
            at += sizeof(root);
            ++header.roots;
        }
    }
    g_roots = {};
    at = Padded(at);
    if (draw) {
        // The buffer flag take. Every buffer flag on the engine is E2's in the split, but writers
        // other than the register words raise them on E1's side too: the Vulkan state tracker's
        // invalidations (new command buffer, InvalidateState), a lent call's flags (both views
        // take the engine's after it), the cache's own marks. In the split E1 takes the raised
        // ones off its engine at each draw record (read and clear: E2 never writes E1's engine,
        // its fill clears its own copy) and E2 raises them in its view. Here the engine is one
        // object, so the take is what the engine holds and the view lacks: between two places
        // where the view takes the engine's flags whole (a fill, a lend, a seed) nothing but a
        // fill clears a buffer flag, so that is exactly what was raised (the compare checks it:
        // a flag the view holds and the engine does not would count). E2 reads the buffer flags
        // only in a fill, which a draw record precedes, so raises arriving here instead of when
        // they were made change nothing it reads (they are ORs); a raise by E2's own work (in the
        // split straight into its copy) comes the same way here. FlagSite counts where they rose.
        // FRITCHII_ESPLIT_DEPTH > 0 (the thread, async): the split's own take. E2's view runs
        // behind, so E1 takes every buffer flag raised on its engine and clears it there
        // (ThreadBuildAsync); E2 ORs them into its copy.
        const u64 take = g_take_all ? PackFlags(engine.dirty.flags)
                                    : PackFlags(engine.dirty.flags) &
                                          ~(PackFlags(g_view.flags) | table_bits);
        if (g_take_all) {
            const_cast<Maxwell3D&>(engine).dirty.flags &= ~BufferFlags();
        }
        std::memcpy(base + at, &take, sizeof(take));
        at += sizeof(take);
        header.take = 1;
        if (take != 0) {
            ++g_c.take_records;
            g_c.take_vertex_buffers += (take & PackedVertexBuffers) != 0 ? 1 : 0;
            g_c.take_bindings += static_cast<u64>(std::popcount(take & PackedBindings));
            g_c.take_index_buffer += (take & PackedIndexBuffer) != 0 ? 1 : 0;
        }
        const auto& state = engine.draw_manager.draw_state;
        DrawPart part{};
        part.vertex_buffer = state.vertex_buffer;
        part.index_buffer = state.index_buffer;
        part.list_draw_address = state.list_draw_address;
        part.list_keep_id = state.list_keep_id;
        part.topology = static_cast<u32>(state.topology);
        part.draw_mode = static_cast<u32>(state.draw_mode);
        part.draw_element_count = state.draw_element_count;
        part.base_index = state.base_index;
        part.base_instance = state.base_instance;
        part.instance_count = state.instance_count;
        part.drawn_instances = state.drawn_instances;
        part.draw_indexed = state.draw_indexed ? 1 : 0;
        std::memcpy(base + at, &part, sizeof(part));
        at = Padded(at + sizeof(part));
        header.draw = 1;
        if (list_args) {
            const auto& indirect = engine.draw_manager.indirect_state;
            std::memcpy(base + at, &indirect, sizeof(indirect));
            at = Padded(at + sizeof(indirect));
            header.indirect = 1;
        }
    }
    header.bytes = static_cast<u32>(at);
    header.position = GpuSplit::Get().BackEntryPosition();
    std::memcpy(base, &header, sizeof(header));
    ++g_c.records[draw ? 1 : 0];
    g_c.record_pairs += header.pairs;
    g_c.record_bytes += at;
    if (draw) {
        g_c.draw_pairs += header.pairs;
    }
    return at;
}

/// E2: the record applied to the view. The buffer flags rise as the engine's did for each word
/// (its dirty tables are fixed at boot).
void Consume(const u8* record, const Maxwell3D& engine) {
    RecordHeader header;
    std::memcpy(&header, record, sizeof(header));
    size_t at = sizeof(RecordHeader);
    const auto& table0 = engine.dirty.tables[0];
    const auto& table1 = engine.dirty.tables[1];
    for (u32 i = 0; i < header.pairs; ++i) {
        RecordPair pair;
        std::memcpy(&pair, record + at, sizeof(pair));
        at += sizeof(pair);
        g_view.regs.reg_array[pair.index] = pair.value;
        g_view.flags[table0[pair.index]] = true;
        g_view.flags[table1[pair.index]] = true;
    }
    at = Padded(at);
    for (u32 i = 0; i < header.binds; ++i) {
        RecordBind bind;
        std::memcpy(&bind, record + at, sizeof(bind));
        at += sizeof(bind);
        auto& info = g_view.state.shader_stages[bind.stage].const_buffers[bind.slot];
        info.address = bind.address;
        info.size = bind.size;
        info.enabled = bind.enabled != 0;
    }
    for (u32 i = 0; i < header.roots; ++i) {
        RecordRoot root;
        std::memcpy(&root, record + at, sizeof(root));
        at += sizeof(root);
        g_view.root_tables.tables[root.table][root.index] = root.value;
        g_view.root_tables.loaded = true;
    }
    at = Padded(at);
    if (header.take != 0) {
        u64 take;
        std::memcpy(&take, record + at, sizeof(take));
        at += sizeof(take);
        if (take != 0) {
            g_view.flags |= UnpackFlags(take);
        }
    }
    if (header.draw != 0) {
        DrawPart part;
        std::memcpy(&part, record + at, sizeof(part));
        at = Padded(at + sizeof(part));
        auto& state = g_view.draw_state;
        state.vertex_buffer = part.vertex_buffer;
        state.index_buffer = part.index_buffer;
        state.list_draw_address = part.list_draw_address;
        state.list_keep_id = part.list_keep_id;
        state.topology = static_cast<Regs::PrimitiveTopology>(part.topology);
        state.draw_mode = static_cast<Maxwell3D::DrawManager::DrawMode>(part.draw_mode);
        state.draw_element_count = part.draw_element_count;
        state.base_index = part.base_index;
        state.base_instance = part.base_instance;
        state.instance_count = part.instance_count;
        state.drawn_instances = part.drawn_instances;
        state.draw_indexed = part.draw_indexed != 0;
        if (header.indirect != 0) {
            std::memcpy(&g_view.indirect, record + at, sizeof(g_view.indirect));
        }
    }
}

/// Counts and reports a difference (the first 16 since boot logged); `after`: the check after a
/// fill that read the view.
void NoteDiff(Diff kind, u32 index, u64 view, u64 engine, bool draw, bool after) {
    u64& total = after ? g_after_total : g_differs_total;
    if (after) {
        ++g_c.after_differs;
    } else {
        ++g_c.differs[static_cast<size_t>(kind)];
    }
    if (++total <= 16) {
        LOG_ERROR(HW_GPU,
                  "E split view (FRITCHII_ESPLIT_COUNT={}): {} {:#x} differs{}: view {:#x}, "
                  "engine {:#x} (at {}, ring position {:#x}, {} since boot)",
                  g_level, DiffNames[static_cast<size_t>(kind)], index,
                  after ? " after a fill that read the view" : "", view, engine,
                  draw ? "a draw fill" : "an E2 entry", GpuSplit::Get().BackEntryPosition(),
                  total);
    }
}

[[nodiscard]] bool SameBinding(const Tegra::Engines::ConstBufferInfo& a,
                               const Tegra::Engines::ConstBufferInfo& b) noexcept {
    return a.address == b.address && a.size == b.size && a.enabled == b.enabled;
}

[[nodiscard]] bool SameIndirect(const Maxwell3D::DrawManager::IndirectParams& a,
                                const Maxwell3D::DrawManager::IndirectParams& b) noexcept {
    return a.is_byte_count == b.is_byte_count && a.is_indexed == b.is_indexed &&
           a.include_count == b.include_count && a.count_start_address == b.count_start_address &&
           a.indirect_start_address == b.indirect_start_address &&
           a.buffer_size == b.buffer_size && a.max_draw_counts == b.max_draw_counts &&
           a.stride == b.stride && a.list_words == b.list_words &&
           a.list_keep_id == b.list_keep_id;
}

/// The view against the engine: W2 always; at a draw also the bindings, the root tables, the
/// draw state, a list draw's indirect parameters and (not after a fill) the buffer flags. A
/// difference is counted and taken from the engine, so one miss is one count. True: equal.
bool Compare(const Maxwell3D& engine, bool draw, bool list_args, bool after) {
    u32 differing = 0;
    for (const auto& [begin, end] : W2().ranges) {
        if (std::memcmp(&g_view.regs.reg_array[begin], &engine.regs.reg_array[begin],
                        (end - begin) * sizeof(u32)) == 0) {
            continue;
        }
        for (u32 i = begin; i < end; ++i) {
            if (g_view.regs.reg_array[i] != engine.regs.reg_array[i]) {
                NoteDiff(Diff::Register, i, g_view.regs.reg_array[i], engine.regs.reg_array[i],
                         draw, after);
                g_view.regs.reg_array[i] = engine.regs.reg_array[i];
                ++differing;
            }
        }
    }
    if (!draw) {
        return differing == 0;
    }
    for (u32 bit = 0; bit < Bindings; ++bit) {
        auto& view = g_view.state.shader_stages[bit / Slots].const_buffers[bit % Slots];
        const auto& real = engine.state.shader_stages[bit / Slots].const_buffers[bit % Slots];
        if (!SameBinding(view, real)) {
            NoteDiff(Diff::Binding, bit, view.address, real.address, draw, after);
            view = real;
            ++differing;
        }
    }
    if (std::memcmp(&g_view.root_tables.tables, &engine.root_tables.tables,
                    sizeof(engine.root_tables.tables)) != 0 ||
        g_view.root_tables.loaded != engine.root_tables.loaded) {
        for (u32 bit = 0; bit < RootWords; ++bit) {
            u32& view = g_view.root_tables.tables[bit / 64][bit % 64];
            const u32 real = engine.root_tables.tables[bit / 64][bit % 64];
            if (view != real) {
                NoteDiff(Diff::RootWord, bit, view, real, draw, after);
                view = real;
                ++differing;
            }
        }
        if (g_view.root_tables.loaded != engine.root_tables.loaded) {
            NoteDiff(Diff::RootWord, RootWords, g_view.root_tables.loaded ? 1 : 0,
                     engine.root_tables.loaded ? 1 : 0, draw, after);
            g_view.root_tables.loaded = engine.root_tables.loaded;
            ++differing;
        }
    }
    const auto& view = g_view.draw_state;
    const auto& real = engine.draw_manager.draw_state;
    const auto field = [&](u32 id, u64 a, u64 b) {
        if (a != b) {
            NoteDiff(Diff::DrawState, id, a, b, draw, after);
            ++differing;
        }
    };
    const u32 before = differing;
    field(0, static_cast<u64>(view.topology), static_cast<u64>(real.topology));
    field(1, static_cast<u64>(view.draw_mode), static_cast<u64>(real.draw_mode));
    field(2, view.draw_indexed ? 1 : 0, real.draw_indexed ? 1 : 0);
    field(3, view.draw_element_count, real.draw_element_count);
    field(4, view.base_index, real.base_index);
    field(5, view.base_instance, real.base_instance);
    field(6, view.instance_count, real.instance_count);
    field(7, view.drawn_instances, real.drawn_instances);
    field(8, view.list_draw_address, real.list_draw_address);
    field(9, view.list_keep_id, real.list_keep_id);
    field(10,
          std::memcmp(&view.vertex_buffer, &real.vertex_buffer, sizeof(real.vertex_buffer)) == 0
              ? 0
              : 1,
          0);
    field(11,
          std::memcmp(&view.index_buffer, &real.index_buffer, sizeof(real.index_buffer)) == 0
              ? 0
              : 1,
          0);
    if (differing != before) {
        TakeDraw(g_view.draw_state, real);
    }
    if (list_args && !SameIndirect(g_view.indirect, engine.draw_manager.indirect_state)) {
        NoteDiff(Diff::Indirect, 0, g_view.indirect.indirect_start_address,
                 engine.draw_manager.indirect_state.indirect_start_address, draw, after);
        g_view.indirect = engine.draw_manager.indirect_state;
        ++differing;
    }
    if (!after) {
        const Flags flags = (g_view.flags ^ engine.dirty.flags) & BufferFlags();
        if (flags.any()) {
            for (u32 flag = 0; flag < flags.size(); ++flag) {
                if (flags.test(flag)) {
                    NoteDiff(Diff::BufferFlag, flag, g_view.flags.test(flag) ? 1 : 0,
                             engine.dirty.flags.test(flag) ? 1 : 0, draw, after);
                    ++differing;
                }
            }
            g_view.flags = (g_view.flags & ~BufferFlags()) | (engine.dirty.flags & BufferFlags());
        }
    }
    return differing == 0;
}

/// One hand-over: E1's record built into the ring, consumed into the view at once, the view
/// compared with the engine. True when the view equals the engine on what was compared.
bool HandOver(const Maxwell3D& engine, bool draw, bool list_args) {
    if (!Tracking()) {
        // Words written now go unseen (the one follower tracks no writes): seed when it does.
        ++g_c.untracked;
        g_view.seeded = false;
        ClearPending();
        return false;
    }
    if (!g_view.seeded || g_view.engine != &engine) {
        Seed(engine);
        return true;
    }
    const bool timed = Take();
    const u64 t0 = timed ? Ticks() : 0;
    const u8* const record = Build(engine, draw, list_args);
    const u64 t1 = timed ? Ticks() : 0;
    Consume(record, engine);
    const u64 t2 = timed ? Ticks() : 0;
    const bool equal = Compare(engine, draw, list_args, false);
    ++g_c.compares;
    if (timed) {
        const u64 t3 = Ticks();
        g_c.build_ticks += t1 - t0;
        g_c.consume_ticks += t2 - t1;
        g_c.compare_ticks += t3 - t2;
    }
    return equal;
}

/// CacheInputs' marks on an engine's buffer flags outside a stage copy (a buffer the cache bound
/// went away or was repacked): E2's own flags in the split, so the view takes them too.
void OnCacheMark(const Maxwell3D* engine, bool index_buffer, u32 vertex_mask) {
    if (!g_records || !g_view.seeded || engine != g_view.engine) {
        return;
    }
    if (index_buffer) {
        g_view.flags[Dirty::IndexBuffer] = true;
        return;
    }
    g_view.flags[Dirty::VertexBuffers] = true;
    for (u32 binding = 0; binding < 32; ++binding) {
        if ((vertex_mask >> binding) & 1) {
            g_view.flags[Dirty::VertexBuffer0 + binding] = true;
        }
    }
}

void SetLevel(int level) {
    const int old = g_level;
    g_level = level;
    t_back = true;
    g_c = {};
    g_since_ns = Nx2Profile::Now();
    g_since_ticks = Ticks();
    g_state = level != 0 ? StateOn : 0;
    g_records = level >= 2;
    g_depth = 0;
    g_lend_pushed = false;
    g_shader_window = false;
    if (g_records) {
        OneF::g_write_watch |= OneF::WriteWatchSplit;
        if (CacheInputs::g_mark_hook == nullptr) {
            CacheInputs::g_mark_hook = &OnCacheMark;
            g_own_mark_hook = true;
        }
        g_touched.reserve(NumRegs);
    } else {
        OneF::g_write_watch &= static_cast<u8>(~OneF::WriteWatchSplit);
        if (g_own_mark_hook) {
            CacheInputs::g_mark_hook = nullptr;
            g_own_mark_hook = false;
        }
    }
    ClearPending();
    g_view.seeded = false;
    g_view_flags.store(nullptr, std::memory_order_relaxed);
    for (auto& count : g_tracker_raises) {
        count.store(0, std::memory_order_relaxed);
    }
    LOG_INFO(HW_GPU,
             "E split count (FRITCHII_ESPLIT_COUNT={}): {} (was {}); W2 {} register words in {} "
             "ranges",
             level,
             level == 0   ? "off"
             : level == 1 ? "classes, timers and stall counts"
             : level == 2 ? "with the record rehearsal and the view compare"
                          : "with the rehearsal, and fills read the view",
             old, W2().words.count(), W2().ranges.size());
}

void Report(u64 now_ns) {
    const Counts c = g_c;
    const double window = static_cast<double>(now_ns - g_since_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now_ns - g_since_ns) /
        static_cast<double>((std::max)(Ticks() - g_since_ticks, u64{1}));
    const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * static_cast<double>(SampleScale) * ns_per_tick / 1e6 /
               window;
    };
    const auto cls = [&](Class k) { return static_cast<size_t>(k); };
    const double e1 = ms(c.ticks[cls(Class::E1)]);
    const double e2 = ms(c.ticks[cls(Class::E2)]);
    const double join = ms(c.ticks[cls(Class::Join)]);
    const double lend = ms(c.ticks[cls(Class::Lend)]);
    const bool profiled = Nx2Profile::Enabled();
    const int level = g_level;
    LOG_INFO(HW_GPU,
             "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: classes in ms/s (1 in 64 entries "
             "timed, {} timed, GPU profile {}): E1 {:.1f} ({:.0f} entries/s), E2 {:.1f} ({:.0f} "
             "entries/s), join {:.1f} ({:.0f} entries/s), lend {:.1f} ({:.0f} lends/s); all "
             "entries {:.1f} ms/s",
             level, c.timed, profiled ? "on" : "off", e1, rate(c.entries[cls(Class::E1)]), e2,
             rate(c.entries[cls(Class::E2)]), join, rate(c.entries[cls(Class::Join)]), lend,
             rate(c.lends), e1 + e2 + join + lend);
    const u64 joins = c.joins[0] + c.joins[1] + c.joins[2];
    const u64 decisions = c.stalls[0] + c.stalls[1];
    const auto join_rate = [&](JoinCause k) { return rate(c.joins[static_cast<size_t>(k)]); };
    const auto stall_rate = [&](StallCause k) { return rate(c.stalls[static_cast<size_t>(k)]); };
    LOG_INFO(HW_GPU,
             "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: stall points {:.0f}/s = joins "
             "{:.0f} (upload {:.0f}, other engine {:.0f}, untagged call {:.0f}) + E1 decisions "
             "{:.0f} (game memory reads {:.0f}, lookups after a shader page write {:.0f}); "
             "channel calls {:.0f}/s, no join (E1 runs them, E2 takes the bind in order): {:.0f} "
             "switch (their bind lends, E2 parks in the lend), {:.0f} bind the bound channel "
             "again; conditions from memory {:.0f}/s (forwarded, decided on E2); E2 to E1 flag "
             "raises {:.0f}/s; draws filled {:.0f}/s",
             level, rate(joins + decisions), rate(joins), join_rate(JoinCause::Upload),
             join_rate(JoinCause::OtherEngine), join_rate(JoinCause::UntaggedCall),
             rate(decisions), stall_rate(StallCause::MemoryDecision),
             stall_rate(StallCause::ShaderWrite), rate(c.channel_calls), rate(c.channel_lends),
             rate(c.channel_calls - (std::min)(c.channel_lends, c.channel_calls)),
             rate(c.conditions), rate(c.back_raises), rate(c.draws));
    // The hand-over: estimated at 1 (fritchii design 2b: 0.15 us a filled draw, 0.05 us an E2
    // entry), measured at 2 and 3 (build + consume, 1 in 64 hand-overs timed).
    const double estimated =
        (rate(c.draws) * 0.15 + rate(c.entries[cls(Class::E2)]) * 0.05) / 1000.0;
    const double build = ms(c.build_ticks);
    const double consume = ms(c.consume_ticks);
    const double handover = level >= 2 ? build + consume : estimated;
    const double join_cost = rate(joins) * 2.0 / 1000.0;
    const double decision_cost = rate(decisions) * 4.0 / 1000.0;
    const double lend_cost = rate(c.lends) * 1.5 / 1000.0;
    const double net = e1 - handover - join_cost - decision_cost - lend_cost;
    LOG_INFO(HW_GPU,
             "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: net movable {:.1f} ms/s = E1 {:.1f} "
             "- hand-over {:.1f} ({}) - joins x 2 us {:.1f} - E1 decisions x 4 us {:.1f} - lends "
             "x 1.5 us {:.1f}; S1 bar: net >= 60 ms/s profiled, stall points <= 2,000/s, view "
             "differs 0",
             level, net, e1, handover,
             level >= 2 ? fmt::format("measured: build {:.1f} + consume {:.1f}", build, consume)
                        : std::string{"estimated: 0.15 us a filled draw + 0.05 us an E2 entry"},
             join_cost, decision_cost, lend_cost);
    if (level >= 2) {
        const u64 records = c.records[0] + c.records[1];
        const u64 differs = [&] {
            u64 sum = 0;
            for (const u64 d : c.differs) {
                sum += d;
            }
            return sum;
        }();
        const auto per = [](u64 part, u64 whole) {
            return whole != 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: records {:.0f}/s (draws {:.0f}, "
                 "E2 entries {:.0f}), {:.1f} W2 words and {:.0f} bytes a record, {:.1f} W2 words "
                 "a draw record; build {:.2f} ms/s, consume {:.2f} ms/s, compare {:.2f} ms/s (1 "
                 "in 64 timed); view differs (must be 0): {} in {} compares (registers {}, "
                 "bindings {}, root words {}, draw state {}, indirect {}, buffer flags {}), {} "
                 "since boot; seeds {}, untracked hand-overs {}, buffer flag syncs at lends "
                 "{:.0f}/s",
                 level, rate(records), rate(c.records[1]), rate(c.records[0]),
                 per(c.record_pairs, records), per(c.record_bytes, records),
                 per(c.draw_pairs, c.records[1]), build, consume, ms(c.compare_ticks), differs,
                 c.compares, c.differs[0], c.differs[1], c.differs[2], c.differs[3], c.differs[4],
                 c.differs[5], g_differs_total, c.seeds, c.untracked, rate(c.flag_syncs));
        const auto rise = [&](size_t site) { return rate(c.rises[site]); };
        const auto tracker = [&](size_t who) {
            return rate(g_tracker_raises[who].exchange(0, std::memory_order_relaxed));
        };
        LOG_INFO(HW_GPU,
                 "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: buffer flag take (raised on the "
                 "engine outside the register records, handed over with the draw record) "
                 "{:.0f}/s draw records: VertexBuffers {:.0f}/s, per-binding flags {:.0f}/s, "
                 "IndexBuffer {:.0f}/s; VertexBuffers rose (0 to 1) in: the draw's stream flush "
                 "{:.0f}/s, its pipeline lookup {:.0f}/s, its TickWork {:.0f}/s, E2 entries "
                 "{:.0f}/s, joins {:.0f}/s, lends {:.0f}/s; state tracker invalidations on the "
                 "viewed engine: on E {:.0f}/s, R in a lent call {:.0f}/s, other threads {:.0f}/s "
                 "(must be 0: unordered with E)",
                 level, rate(c.take_records), rate(c.take_vertex_buffers), rate(c.take_bindings),
                 rate(c.take_index_buffer), rise(static_cast<size_t>(FlagSite::StreamFlush)),
                 rise(static_cast<size_t>(FlagSite::Lookup)),
                 rise(static_cast<size_t>(FlagSite::TickWork)), rise(RiseE2Entry),
                 rise(RiseJoin), rise(RiseLend), tracker(0), tracker(1), tracker(2));
    }
    if (level >= 3) {
        LOG_INFO(HW_GPU,
                 "E split count (FRITCHII_ESPLIT_COUNT={}) 10 s: fills from the view {:.0f}/s, "
                 "from the engine {:.0f}/s (the view differed or was just seeded {}, inline "
                 "index bytes {}); inputs differ after a fill from the view (must be 0): {} in "
                 "{} checks, {} since boot",
                 level, rate(c.view_fills), rate(c.fallback_differ + c.fallback_inline),
                 c.fallback_differ, c.fallback_inline, c.after_differs, c.after_checks,
                 g_after_total);
    }
}

} // Anonymous namespace

int BatchLevel() noexcept {
    int level = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_esplit_count,
                                                 "FRITCHII_ESPLIT_COUNT");
    if (level < 0 || level > 3) {
        level = 0;
    }
    // FRITCHII_ESPLIT=1 (the thread mode) has its own records and lines: the dry step needs one
    // thread, so it stays off.
    if (level != 0 && g_thread) [[unlikely]] {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_WARNING(HW_GPU, "E split count: FRITCHII_ESPLIT_COUNT={} ignored, FRITCHII_ESPLIT=1 "
                                "runs the E2 thread (its own lines, \"E split thread\")",
                        level);
        }
        level = 0;
    }
    if (level != g_level) [[unlikely]] {
        SetLevel(level);
    }
    if (level != 0 && (++g_batches & 63) == 0) [[unlikely]] {
        const u64 now = Nx2Profile::Now();
        if (now - g_since_ns >= 10'000'000'000ULL) {
            Report(now);
            g_c = {};
            g_since_ns = now;
            g_since_ticks = Ticks();
        }
    }
    return level;
}

EngineKind KindOf(const Tegra::Engines::EngineInterface* engine) noexcept {
    if (engine == g_kind_engine) [[likely]] {
        return g_kind;
    }
    g_kind_engine = engine;
    if (engine == nullptr) {
        g_kind = EngineKind::None;
    } else if (const auto* maxwell3d = dynamic_cast<const Maxwell3D*>(engine)) {
        g_kind = EngineKind::Maxwell3D;
        g_engine3d = maxwell3d;
    } else if (dynamic_cast<const Tegra::Engines::KeplerCompute*>(engine) != nullptr) {
        g_kind = EngineKind::Compute;
    } else {
        g_kind = EngineKind::Other;
    }
    return g_kind;
}

void BeforeEntry(const EntryInfo& info) noexcept {
    g_entry = info;
    g_entry_class = info.cls;
    if (info.cls == Class::Join) {
        ++g_c.joins[static_cast<size_t>(info.cause)];
        g_shader_window = false; // E2 drained
    }
    if (info.channel) {
        ++g_c.channel_calls;
    }
    // The rehearsal's own work before the timer starts: the entry's time stays its own. A
    // channel call is handed over too: E2 takes the bind in order, after the records before it.
    if (g_records && (info.cls == Class::E2 || info.channel) && g_engine3d != nullptr) {
        (void)HandOver(*g_engine3d, false, false);
    }
    // Where the engine's VertexBuffers rises outside the records: E2 entries and joins.
    g_entry_vb = !(g_records && info.cls != Class::E1 && g_view.seeded) ||
                 g_view.engine->dirty.flags[Dirty::VertexBuffers]; // no edge counted then
    if (Take()) {
        g_state |= StateSampling;
        g_cur = info.cls;
        g_depth = 0;
        g_seg_start = Ticks();
    }
}

void AfterEntry(std::size_t size) noexcept {
    if ((g_state & StateSampling) != 0) {
        g_c.ticks[static_cast<size_t>(g_cur)] += Ticks() - g_seg_start;
        g_depth = 0;
        g_state &= static_cast<u8>(~StateSampling);
        ++g_c.timed;
    }
    ++g_c.entries[static_cast<size_t>(g_entry_class)];
    if (g_records && g_entry.cls != Class::E1 && g_view.seeded && !g_entry_vb &&
        g_view.engine->dirty.flags[Dirty::VertexBuffers]) {
        ++g_c.rises[g_entry.cls == Class::Join ? RiseJoin : RiseE2Entry];
    }
    // E2 runs the constant buffer words itself in the split: its offset advances by them.
    if (g_records && g_entry.cb_entry && g_view.seeded) {
        const u64 words = g_entry.cb_words != 0 ? g_entry.cb_words : size;
        g_view.regs.const_buffer.offset += static_cast<u32>(words * sizeof(u32));
    }
}

bool PushCarve(Class cls) noexcept {
    if (!t_back || g_depth >= g_stack.size()) {
        return false; // R in a lent call (E parked inside the timed entry), or too deep
    }
    const u64 now = Ticks();
    g_c.ticks[static_cast<size_t>(g_cur)] += now - g_seg_start;
    g_stack[g_depth++] = g_cur;
    g_cur = cls;
    g_seg_start = now;
    return true;
}

void PopCarve() noexcept {
    if (g_depth == 0 || (g_state & StateSampling) == 0) {
        return;
    }
    const u64 now = Ticks();
    g_c.ticks[static_cast<size_t>(g_cur)] += now - g_seg_start;
    g_cur = g_stack[--g_depth];
    g_seg_start = now;
}

bool BeginLend() noexcept {
    if (!t_back) {
        return false;
    }
    ++g_c.lends;
    if (g_entry.channel) {
        ++g_c.channel_lends; // a real switch: RasterizerVulkan::BindChannel lent
    }
    g_shader_window = false; // E2 drains to the lend's place
    g_lend_vb = !(g_records && g_view.seeded) ||
                g_view.engine->dirty.flags[Dirty::VertexBuffers]; // unseeded: no edge counted
    g_lend_pushed = (g_state & StateSampling) != 0 && PushCarve(Class::Lend);
    return true;
}

void EndLend() noexcept {
    if (std::exchange(g_lend_pushed, false)) {
        PopCarve();
    }
    // R's run left the engine's buffer flags as it used them; both stages are parked in a lend,
    // so E2 takes them (the registers R wrote come in the next record). Not a lend E2 makes
    // while E1 runs ahead (FRITCHII_ESPLIT_DEPTH > 0): E1's engine is past E2's place then.
    if (g_records && g_view.seeded && g_view.engine != nullptr && !(g_async && OnE2())) {
        const Flags& flags = g_view.engine->dirty.flags;
        if (!g_lend_vb && flags[Dirty::VertexBuffers]) {
            ++g_c.rises[RiseLend];
        }
        g_view.flags = (g_view.flags & ~BufferFlags()) | (flags & BufferFlags());
        ++g_c.flag_syncs;
    }
}

void NoteStallImpl(StallCause cause) noexcept {
    if (OnE()) {
        ++g_c.stalls[static_cast<size_t>(cause)];
    }
}

void NoteConditionImpl() noexcept {
    if (OnE()) {
        ++g_c.conditions;
    }
}

void NoteStreamWriteImpl(bool shader_pages) noexcept {
    if (shader_pages && t_back) {
        g_shader_window = true;
    }
}

void NoteLookupImpl() noexcept {
    if (g_shader_window && t_back) {
        ++g_c.stalls[static_cast<size_t>(StallCause::ShaderWrite)];
        g_shader_window = false;
    }
}

void NoteBackRaiseImpl() noexcept {
    if (t_back) {
        ++g_c.back_raises;
    }
}

void NoteDrawFilledImpl() noexcept {
    if (t_back) {
        ++g_c.draws;
    }
}

void NoteFlagRiseImpl(FlagSite site) noexcept {
    if (t_back) {
        ++g_c.rises[static_cast<size_t>(site)];
    }
}

void NoteTrackerInvalidationImpl(const void* flags) noexcept {
    if (flags == nullptr || flags != g_view_flags.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t who = t_back ? 0 : OneF::InLent() ? 1 : 2;
    g_tracker_raises[who].fetch_add(1, std::memory_order_relaxed);
}

void SetEntryClassImpl(Class cls) noexcept {
    if (!t_back) {
        return;
    }
    g_entry_class = cls;
    if ((g_state & StateSampling) != 0 && g_depth == 0) {
        const u64 now = Ticks();
        g_c.ticks[static_cast<size_t>(g_cur)] += now - g_seg_start;
        g_cur = cls;
        g_seg_start = now;
    }
}

void NoteWord(u32 index) noexcept {
    // Called by NoteRegWrite on E or R in a lent call (E parked). Only W2 words go to E2.
    if (!g_records || index >= NumRegs || !W2().words.test(index) || g_noted.test(index)) {
        return;
    }
    g_noted.set(index);
    g_touched.push_back(static_cast<u16>(index));
}

void NoteCbBindImpl(std::size_t stage, u32 slot) noexcept {
    if (stage >= Regs::MaxShaderStage || slot >= Slots || !OnE()) {
        return;
    }
    const u32 bit = static_cast<u32>(stage) * Slots + slot;
    g_binds[bit / 64] |= u64{1} << (bit % 64);
}

void NoteRootWordImpl(u32 table, u32 index) noexcept {
    if (table >= 8 || index >= 64 || !OnE()) {
        return;
    }
    const u32 bit = table * 64 + index;
    g_roots[bit / 64] |= u64{1} << (bit % 64);
}

const View* DrawHandOver(const Maxwell3D& engine, bool list_args) noexcept {
    if (!t_back) {
        return nullptr;
    }
    const bool seeded_before = g_view.seeded && g_view.engine == &engine;
    const bool equal = HandOver(engine, true, list_args);
    if (g_level < 3) {
        return nullptr;
    }
    if (!equal || !seeded_before) {
        ++g_c.fallback_differ; // a record that failed, or a view just taken from the engine
        return nullptr;
    }
    if (!engine.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        ++g_c.fallback_inline;
        return nullptr;
    }
    ++g_c.view_fills;
    return &g_view;
}

void AfterFill(const Maxwell3D& engine, const View* used) noexcept {
    if (!t_back || !g_view.seeded || g_view.engine != &engine) {
        return;
    }
    // The fill (E2's) consumed and raised buffer flags: E2's own after it.
    g_view.flags = (g_view.flags & ~BufferFlags()) | (engine.dirty.flags & BufferFlags());
    if (used != nullptr) {
        ++g_c.after_checks;
        (void)Compare(engine, true, engine.draw_manager.indirect_state.list_words, true);
    }
}

// --- The thread mode (FRITCHII_ESPLIT=1) ------------------------------------------------------------

namespace {
/// E1's side of the records in the thread mode: the engine its records follow, and whether a
/// write went untracked since (the next record is a seed then).
const Maxwell3D* g_t_engine{};
bool g_t_lost = true;

/// E2: its view taken from the engine (E1 waits), without touching E1's pending sets.
void ViewFromEngine(const Maxwell3D& engine) {
    std::memcpy(&g_view.regs, &engine.regs, sizeof(Regs));
    g_view.state = engine.state;
    g_view.root_tables = engine.root_tables;
    TakeDraw(g_view.draw_state, engine.draw_manager.draw_state);
    g_view.indirect = engine.draw_manager.indirect_state;
    g_view.flags = engine.dirty.flags & BufferFlags();
    g_view.engine = &engine;
    g_view.seeded = true;
    g_view_flags.store(&engine.dirty.flags, std::memory_order_relaxed);
    ++g_c.seeds;
}

/// FNV-1a over 64-bit lanes, as the hash pair needs it (both sides take the same fields).
struct Hasher {
    u64 value = 0xcbf29ce484222325ULL;
    void Add(u64 lane) noexcept {
        value = (value ^ lane) * 0x100000001b3ULL;
    }
};

u64 HashParts(const Regs& regs, const Maxwell3D::State& state,
              const Maxwell3D::RootTables& roots, const Maxwell3D::DrawManager::State& draw,
              const Maxwell3D::DrawManager::IndirectParams* indirect) noexcept {
    Hasher h;
    for (const auto& [begin, end] : W2().ranges) {
        for (u32 i = begin; i < end; ++i) {
            h.Add(regs.reg_array[i]);
        }
    }
    for (u32 bit = 0; bit < Bindings; ++bit) {
        const auto& info = state.shader_stages[bit / Slots].const_buffers[bit % Slots];
        h.Add(info.address);
        h.Add((u64{info.size} << 1) | (info.enabled ? 1 : 0));
    }
    for (u32 bit = 0; bit < RootWords; ++bit) {
        h.Add(roots.tables[bit / 64][bit % 64]);
    }
    h.Add(roots.loaded ? 1 : 0);
    h.Add(static_cast<u64>(draw.topology));
    h.Add(static_cast<u64>(draw.draw_mode));
    h.Add(draw.draw_indexed ? 1 : 0);
    h.Add(draw.draw_element_count);
    h.Add(draw.base_index);
    h.Add(draw.base_instance);
    h.Add(draw.instance_count);
    h.Add(draw.drawn_instances);
    h.Add(draw.list_draw_address);
    h.Add(draw.list_keep_id);
    u64 lanes[(sizeof(draw.vertex_buffer) + sizeof(draw.index_buffer) + 7) / 8]{};
    std::memcpy(lanes, &draw.vertex_buffer, sizeof(draw.vertex_buffer));
    std::memcpy(reinterpret_cast<u8*>(lanes) + sizeof(draw.vertex_buffer), &draw.index_buffer,
                sizeof(draw.index_buffer));
    for (const u64 lane : lanes) {
        h.Add(lane);
    }
    if (indirect != nullptr) {
        h.Add(indirect->is_byte_count ? 1 : 0);
        h.Add(indirect->is_indexed ? 1 : 0);
        h.Add(indirect->include_count ? 1 : 0);
        h.Add(indirect->count_start_address);
        h.Add(indirect->indirect_start_address);
        h.Add(indirect->buffer_size);
        h.Add(indirect->max_draw_counts);
        h.Add(indirect->stride);
        h.Add(indirect->list_words ? 1 : 0);
        h.Add(indirect->list_keep_id);
    }
    return h.value;
}
} // Anonymous namespace

void ThreadMark() noexcept {
    t_back = true;
}

void ThreadRecordsOn() noexcept {
    g_records = true;
    OneF::g_write_watch |= OneF::WriteWatchSplit;
    if (CacheInputs::g_mark_hook == nullptr) {
        CacheInputs::g_mark_hook = &OnCacheMark;
        g_own_mark_hook = true;
    }
    g_touched.reserve(NumRegs);
    ClearPending();
    g_view.seeded = false;
    g_t_engine = nullptr;
    g_t_lost = true;
}

std::size_t RecordRoom() noexcept {
    return MaxRecordBytes;
}

RecordKind ThreadBuild(const Maxwell3D& engine, bool draw, bool list_args, u8* out,
                       u32& bytes) noexcept {
    bytes = 0;
    if (!Tracking()) {
        ++g_c.untracked;
        ClearPending();
        g_t_lost = true;
        return RecordKind::Lost;
    }
    if (g_t_lost || g_t_engine != &engine) {
        ClearPending();
        g_t_lost = false;
        g_t_engine = &engine;
        return RecordKind::Seed;
    }
    bytes = static_cast<u32>(BuildInto(out, engine, draw, list_args));
    return RecordKind::Record;
}

u64 ThreadConsume(RecordKind kind, const u8* record, const Maxwell3D& engine, bool compare,
                  bool draw, bool list_args) noexcept {
    switch (kind) {
    case RecordKind::None:
        return 0;
    case RecordKind::Lost:
        g_view.seeded = false;
        return 0;
    case RecordKind::Seed:
        ViewFromEngine(engine);
        return 0;
    case RecordKind::Record:
        break;
    }
    if (!g_view.seeded || g_view.engine != &engine) {
        return 0; // stale until the next seed (E1 sends one after a loss or a channel change)
    }
    Consume(record, engine);
    if (!compare) {
        return 0;
    }
    const u64 before = g_differs_total;
    ++g_c.compares;
    (void)Compare(engine, draw, list_args, false);
    return g_differs_total - before;
}

void ThreadCbAdvance(u64 words) noexcept {
    if (g_view.seeded) {
        g_view.regs.const_buffer.offset += static_cast<u32>(words * sizeof(u32));
    }
}

View& ThreadView() noexcept {
    return g_view;
}

u64 ThreadAfterDiffers() noexcept {
    return g_after_total;
}

bool ThreadRecordReady(const Maxwell3D& engine) noexcept {
    return Tracking() && !g_t_lost && g_t_engine == &engine;
}

u32 ThreadBuildAsync(Maxwell3D& engine, bool list_args, u8* out) noexcept {
    g_take_all = true;
    const u32 bytes = static_cast<u32>(BuildInto(out, engine, true, list_args));
    g_take_all = false;
    return bytes;
}

// The queued mode's buffer flags (FRITCHII_ESPLIT_DEPTH > 0): between transfers the engine holds
// only the flags E1's work raised since the last one (they are cleared at each transfer, so a raise
// after E2's fill cleared its copy is a 0 to 1 edge again), E2's view the whole state. Overwriting
// the engine with the view (as S2 did first) dropped E1's raises made since the last record: the
// view then took them from the record's words and held a flag the engine lacked (IndexBuffer,
// flag 0x30, the step-in of 15 Nov). Both follow the view's own engine: after a channel switch
// the engine the view left gets its whole state back before the new one is seeded.
void SyncFlagsToEngine() noexcept {
    if (g_view.seeded && g_view.engine != nullptr) {
        Maxwell3D& engine = const_cast<Maxwell3D&>(*g_view.engine);
        engine.dirty.flags |= g_view.flags & BufferFlags();
    }
}

void SyncFlagsToView() noexcept {
    if (g_view.seeded && g_view.engine != nullptr) {
        Maxwell3D& engine = const_cast<Maxwell3D&>(*g_view.engine);
        g_view.flags = (g_view.flags & ~BufferFlags()) | (engine.dirty.flags & BufferFlags());
        engine.dirty.flags &= ~BufferFlags();
    }
}

u64 HashEngine(const Maxwell3D& engine, bool list_args) noexcept {
    return HashParts(engine.regs, engine.state, engine.root_tables, engine.draw_manager.draw_state,
                     list_args ? &engine.draw_manager.indirect_state : nullptr);
}

u64 HashView(bool list_args) noexcept {
    return HashParts(g_view.regs, g_view.state, g_view.root_tables, g_view.draw_state,
                     list_args ? &g_view.indirect : nullptr);
}

} // namespace VideoCommon::ESplit
