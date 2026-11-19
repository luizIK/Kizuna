// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part (b): R's register view, the delta E sends, the dirty flag partition on the
// engine side, step 2's view compare and the stream-write guard (one_follower.h).
//
// The delta. Maxwell3D's register writes on E (ProcessDirtyRegisters, ApplyDeltaWrites) note
// their word while step 2+ is on (g_track_writes); each packet carries the watched words noted
// since the last one with E's values at the packet. A lent call R runs writes the engine with E
// parked; those writes are noted the same way (R in a lent call stands in for E).
//
// The flags. No raise site is hooked: E keeps `after`, the forwarded flags as they stood after
// the last packet (or after the last lent call, or E's fill at step 4), and a packet forwards
// what is raised now and was not then. Between packets E only raises forwarded flags (its buffer
// fill, which clears the shared ones, runs inside the draw, after the pack). Step 2's compare
// catches any clear on E between packets (a raise R would miss).
//
// The partition (section 2), from the code: the buffer cache alone reads and clears IndexBuffer
// and VertexBuffer0..31 (cache_inputs.h), so they stay on E. VertexBuffers is read by both (the
// cache's vertex buffer update and RasterizerVulkan's vertex input update), so it exists on both.
// Every other flag (render targets, descriptors, shaders, the Vulkan state tracker's) only R
// reads. Flag 0 is the tables' "no flag" slot: neither side.

#include <array>
#include <algorithm>
#include <bit>
#include <bitset>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "video_core/dirty_flags.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/one_follower.h"
#include "video_core/one_follower_stats.h"
#include "video_core/page_filter.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

namespace VideoCommon::OneFollower {

namespace {

constexpr u32 NumRegs = static_cast<u32>(Maxwell3D::Regs::NUM_REGS);

struct Partition {
    std::bitset<NumRegs> watched;
    std::bitset<NumRegs> pool_words;
    FlagBits engine_only; ///< never forwarded
    FlagBits shared;      ///< on both sides
    FlagBits forward;     ///< R-only and shared: what packets carry and R's view holds
    FlagBits render_only;
    /// R-only flags E's own pipeline lookup reads (rule 3, FRITCHII_ONEF_E_PIPELINE): the
    /// programs, the blend and vertex input parts of the key, the viewport swizzles.
    FlagBits lookup;

    Partition() {
        watched.set();
        const auto unwatch = [this](u32 first, u32 count) {
            for (u32 i = first; i < first + count && i < NumRegs; ++i) {
                watched.reset(i);
            }
        };
        // Words R never reads and E writes all the time: the constant buffer data words and its
        // write offset (ProcessCBMultiData advances it in place), the firmware and shadow scratch.
        unwatch(MAXWELL3D_REG_INDEX(const_buffer.offset), 1);
        unwatch(MAXWELL3D_REG_INDEX(const_buffer.buffer), 16);
        unwatch(MAXWELL3D_REG_INDEX(falcon), 32);
        unwatch(MAXWELL3D_REG_INDEX(shadow_scratch), 0x100);
        const auto pool = [this](u32 first, u32 count) {
            for (u32 i = first; i < first + count; ++i) {
                pool_words.set(i);
            }
        };
        pool(MAXWELL3D_REG_INDEX(tex_header), sizeof(Maxwell3D::Regs::TexHeader) / sizeof(u32));
        pool(MAXWELL3D_REG_INDEX(tex_sampler), sizeof(Maxwell3D::Regs::TexSampler) / sizeof(u32));
        pool(MAXWELL3D_REG_INDEX(sampler_binding), 1);

        engine_only.set(Dirty::NullEntry);
        engine_only.set(Dirty::IndexBuffer);
        for (u32 i = 0; i < 32; ++i) {
            engine_only.set(Dirty::VertexBuffer0 + i);
        }
        shared.set(Dirty::VertexBuffers);
        forward = ~engine_only;
        render_only = forward & ~shared;
        lookup.set(VideoCommon::Dirty::Shaders);
        lookup.set(Vulkan::Dirty::Blending);
        lookup.set(Vulkan::Dirty::VertexInput);
        lookup.set(Vulkan::Dirty::ViewportSwizzles);
        lookup &= render_only;
    }
};

const Partition& Parts() {
    static const Partition parts;
    return parts;
}

/// E's side of the delta (E only, or R inside a lent call with E parked).
struct EngineSide {
    std::bitset<NumRegs> noted;
    std::vector<u16> touched;
    FlagBits after;  ///< the forwarded flags at the last packet / lent call / fill
    bool started{};  ///< the view was seeded since tracking (re)started
    std::atomic<bool> restart{true};
    u64 on_texture{}; ///< stream writes onto texture pages (StreamWritesOnTexturePages)
    /// The TIC and TSC pools' device ranges [begin, end) at E's position (PackSpans), which R
    /// reads for its descriptors (fable_b 10.1 row 8).
    std::array<std::pair<DAddr, DAddr>, 2> pools{};
    bool stream_hit{}; ///< a guard hit since the last draw (TakeStreamWriteLend)
    // FRITCHII_ONEF_UNLENT_VALUES (fable_b 19 step 2): EngineValuesMode, the values R's view has
    // from E (the last lent packet's or the last span's), the bindings bound since the last pack
    // (bit stage * MaxConstBuffers + slot).
    u8 values{};
    u32 sent_engine_state{};
    u8 sent_native{};
    std::array<u64, 2> binds{};
};

EngineSide g_e;
RegisterView g_view;
const PageFilter* g_texture_pages{};
const PageFilter* g_shader_pages{};
const PageFilter* g_union_pages{}; ///< SetUnionPages (fable_b 14.4 cut 5's union)


/// R: the last applied packet carried a draw span (VerifyView compares it then).
thread_local bool t_draw_fresh{};

Pools PoolsOf(const Maxwell3D::Regs& regs) {
    Pools pools{};
    pools.tic_address = regs.tex_header.Address();
    pools.tic_limit = regs.tex_header.limit;
    pools.tsc_address = regs.tex_sampler.Address();
    pools.tsc_limit = regs.tex_sampler.limit;
    pools.tsc_linked =
        regs.sampler_binding == Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding ? 1U : 0U;
    return pools;
}

/// E: the pools' device ranges from the engine's registers (contiguous from their first page:
/// a pool is one mapping in practice; a write past a discontinuity is missed, not misread).
void NotePools(const Maxwell3D& engine) {
    const Pools pools = PoolsOf(engine.regs);
    auto& memory = const_cast<Maxwell3D&>(engine).GetMemoryManager();
    const auto range = [&memory](GPUVAddr address, u64 bytes) -> std::pair<DAddr, DAddr> {
        if (address == 0) {
            return {};
        }
        const std::optional<DAddr> begin = memory.GpuToCpuAddress(address);
        return begin ? std::pair<DAddr, DAddr>{*begin, *begin + bytes} : std::pair<DAddr, DAddr>{};
    };
    constexpr u64 EntryBytes = 0x20; // TICEntry and TSCEntry
    g_e.pools[0] = range(pools.tic_address, (u64{pools.tic_limit} + 1) * EntryBytes);
    const u32 tsc_limit = pools.tsc_linked != 0 ? pools.tic_limit : pools.tsc_limit;
    g_e.pools[1] = range(pools.tsc_address, (u64{tsc_limit} + 1) * EntryBytes);
}

void FillDraw(DrawValues& values, const Maxwell3D& engine) {
    const auto& state = engine.draw_manager.draw_state;
    values.topology = state.topology;
    values.draw_mode = static_cast<u32>(state.draw_mode);
    values.base_index = state.base_index;
    values.base_instance = state.base_instance;
    values.instance_count = state.instance_count;
    values.drawn_instances = state.drawn_instances;
    values.draw_element_count = state.draw_element_count;
    values.vertex_buffer = state.vertex_buffer;
    values.index_buffer = state.index_buffer;
    values.draw_indexed = state.draw_indexed ? 1 : 0;
    // A list draw (inside DrawEnd's list branch): what R's view needs of the indirect state.
    const auto& indirect = engine.draw_manager.indirect_state;
    values.list_words = indirect.list_words ? 1 : 0;
    values.list_keep_id = indirect.list_words ? indirect.list_keep_id : 0;
    values.list_address = indirect.list_words ? indirect.indirect_start_address : 0;
}

bool SameDraw(const DrawValues& view, const Maxwell3D& engine) {
    DrawValues now = view; // the fields only the caller knows (kind, indexed, ...) are the view's
    FillDraw(now, engine);
    return std::memcmp(&now, &view, sizeof(DrawValues)) == 0;
}

/// E at a lent packet (or R inside a lent bind, E parked): R's view takes the engine's values
/// there (SyncFromEngine), so the by-value tracking starts over, on or off as the switch says.
void ResetValues(const Maxwell3D& engine) {
    g_e.values = UnlentValuesOn() ? 1 : 0;
    g_e.sent_engine_state = static_cast<u32>(engine.engine_state);
    g_e.sent_native = const_cast<Maxwell3D&>(engine).GetNativeMacroEngine().native_turing ? 1 : 0;
    g_e.binds = {};
    g_track_binds.store(g_e.values != 0, std::memory_order_relaxed);
}

/// R's note text for a must-be-0 (Note keeps the pointer only for its log call).
thread_local std::string t_note;

} // Anonymous namespace

bool Watched(u32 index) noexcept {
    return index < NumRegs && Parts().watched.test(index);
}

FlagOwner OwnerOf(u8 flag) noexcept {
    const Partition& parts = Parts();
    return parts.engine_only.test(flag) ? FlagOwner::E
           : parts.shared.test(flag)    ? FlagOwner::Both
                                        : FlagOwner::R;
}

RegisterView& View() noexcept {
    return g_view;
}

void NoteRegWrite(u32 index) noexcept {
    if (!Detail::t_engine && !InLent()) {
        return; // the front's copy, or another thread: not E's engine
    }
    if (const u8 watch = g_write_watch; watch != 0) [[unlikely]] {
        // FRITCHII_ONEF_KEY_KEEP: every write of a lookup word, before the per-packet dedupe.
        if ((watch & WriteWatchKey) != 0 && index < KeyWordSlots &&
            ((g_key_words[index / 64] >> (index % 64)) & 1) != 0) {
            ++g_key_writes;
        }
        // FRITCHII_ESPLIT_COUNT >= 2: the word goes in E1's next record (its own dedupe).
        if ((watch & WriteWatchSplit) != 0) {
            ESplit::NoteWord(index);
        }
    }
    if (index >= NumRegs || g_e.noted.test(index)) {
        return;
    }
    g_e.noted.set(index);
    g_e.touched.push_back(static_cast<u16>(index));
}

void NoteCbBind(size_t stage, u32 slot) noexcept {
    if (!Detail::t_engine || g_e.values == 0 || stage >= Maxwell3D::Regs::MaxShaderStage ||
        slot >= Maxwell3D::Regs::MaxConstBuffers) {
        return;
    }
    const size_t bit = stage * Maxwell3D::Regs::MaxConstBuffers + slot;
    g_e.binds[bit / 64] |= u64{1} << (bit % 64);
}

u8 EngineValuesMode() noexcept {
    return g_e.values;
}

void NoteFlagRaised(u8 flag) noexcept {
    // Not needed with the `after` diff (see the top of this file); kept for raise sites that
    // clear on E between packets, should step 2's compare find one: forwarding it again is safe.
    if (Detail::t_engine && Parts().forward.test(flag)) {
        g_e.after.reset(flag);
    }
}

namespace Detail {
void ViewTrackingStopped() noexcept {
    g_e.restart.store(true, std::memory_order_release);
    ++g_key_writes; // FRITCHII_ONEF_KEY_KEEP: writes from now on may go unseen
}

void SeedAfterBind(RegisterView& view, const Maxwell3D& engine) {
    // PackSpans' seed branch (E's side; R stands in for E, which is parked in this lent call),
    // then Play's SeedView of the generic view.
    g_e.restart.store(false, std::memory_order_relaxed);
    g_e.started = true;
    for (const u16 index : g_e.touched) {
        g_e.noted.reset(index);
    }
    g_e.touched.clear();
    g_e.after = engine.dirty.flags & Parts().forward;
    NotePools(engine);
    ResetValues(engine);
    SeedView(view, engine);
}
} // namespace Detail

bool ViewReady() noexcept {
    return g_e.started && !g_e.restart.load(std::memory_order_acquire);
}

std::size_t PackSpans(const Maxwell3D& engine, PacketHeader& header, std::span<u8> out) {
    const Partition& parts = Parts();
    std::size_t used = 0;
    const auto put = [&](const void* data, std::size_t bytes) {
        std::memcpy(out.data() + used, data, bytes);
        used += (bytes + 7) & ~std::size_t{7};
    };
    // Every draw packet and lent draw carries the draw span (part c's PlayDraw and Compare read
    // it); SetNextDraw adds what only the caller knows (kind, indexed, the instance argument).
    const bool is_draw = header.kind == PacketKind::Draw ||
                         (header.kind == PacketKind::Lent &&
                          header.op == static_cast<u8>(LentOp::Draw));
    const auto draw_span = [&] {
        if (!is_draw) {
            return;
        }
        const DrawValues* const next = Detail::NextDraw();
        DrawValues values = next != nullptr ? *next : DrawValues{};
        FillDraw(values, engine);
        header.spans |= SpanDraw;
        put(&values, sizeof(values));
    };
    if (g_e.restart.load(std::memory_order_acquire) || !g_e.started) {
        if (header.kind != PacketKind::Lent) {
            return 0; // a seed needs E parked: the next lent packet carries it
        }
        // Tracking starts over: R takes the engine in full under this packet.
        g_e.restart.store(false, std::memory_order_relaxed);
        g_e.started = true;
        for (const u16 index : g_e.touched) {
            g_e.noted.reset(index);
        }
        g_e.touched.clear();
        g_e.after = engine.dirty.flags & parts.forward;
        NotePools(engine);
        ResetValues(engine);
        header.flags |= PacketSeed;
        draw_span();
        return used;
    }
    // The delta: the watched words written since the last packet, with their values now.
    u32 count = 0;
    bool pools_changed = false;
    for (const u16 index : g_e.touched) {
        g_e.noted.reset(index);
        if (!parts.watched.test(index)) {
            continue;
        }
        const RegPair pair{index, engine.regs.reg_array[index]};
        std::memcpy(out.data() + used + count * sizeof(RegPair), &pair, sizeof(pair));
        ++count;
        pools_changed |= parts.pool_words.test(index);
    }
    g_e.touched.clear();
    if (count != 0) {
        header.spans |= SpanDelta;
        header.delta_count = count;
        used += count * sizeof(RegPair);
    }
    // The forwarded flags. fable_b 10.2 rule 1, one home for each R-only flag: an unlent draw's
    // packet takes every R-only flag set on E's engine and clears it there (R consumes them on
    // its view; a later raise is then a real 0 to 1 edge again). A difference against a baseline
    // missed the second raise of a flag the engine still held (Shaders: the step 4 trips). A
    // lent packet forwards by difference (the lent call runs on the engine, and both views take
    // the engine's flags after it). The shared flags always by difference: E keeps its copy.
    const FlagBits now = engine.dirty.flags & parts.forward;
    FlagBits raised;
    const bool draw_pack = header.kind == PacketKind::Draw;
    const bool op_pack =
        header.kind == PacketKind::Op && (header.flags & PacketRenderState) != 0;
    if (draw_pack || op_pack) {
        raised = (now & parts.render_only) | (now & parts.shared & ~g_e.after);
        // fable_b 18.3 (the lean darkness, 15 Nov): a flag with a reader on E leaves E's engine
        // only in a pack that follows that read. A draw packs after its lookup (FillAndPush);
        // a posted clear packs with no lookup before it, so it keeps the lookup's flags on E
        // for the next draw (R takes them now and again at that draw: one rebind too many).
        const FlagBits taken = draw_pack ? parts.render_only : parts.render_only & ~parts.lookup;
        const_cast<Maxwell3D&>(engine).dirty.flags &= ~taken; // E's engine
    } else {
        raised = now & ~g_e.after;
    }
    g_e.after = engine.dirty.flags & parts.forward;
    if (raised.any()) {
        header.spans |= SpanFlags;
        put(&raised, sizeof(raised));
    }
    draw_span();
    // Programs: not sent. The six hashes come from ShaderCache::RefreshStages, which is the
    // pipeline cache's (R's) and reads the program words R's view has; R computes them itself.
    if (pools_changed) {
        NotePools(engine);
        const Pools pools = PoolsOf(engine.regs);
        header.spans |= SpanPools;
        put(&pools, sizeof(pools));
    }
    // FRITCHII_ONEF_UNLENT_VALUES: what SyncFromEngine copies at a lent packet besides the
    // registers, by value when it changed. Sent while the mode is 1 or 2 (2: the switch went off,
    // the view stays exact until the next lent packet, which takes the engine's values anyway).
    const bool lent_pack = header.kind == PacketKind::Lent;
    if (g_e.values != 0) {
        if (g_e.values == 1 && !UnlentValuesOn()) {
            g_e.values = 2; // draws are lent until the next lent packet (UnlentBlocker)
        }
        Maxwell3D& e = const_cast<Maxwell3D&>(engine);
        const u32 engine_state = static_cast<u32>(e.engine_state);
        const u8 native = e.GetNativeMacroEngine().native_turing ? 1 : 0;
        if (engine_state != g_e.sent_engine_state || native != g_e.sent_native) {
            g_e.sent_engine_state = engine_state;
            g_e.sent_native = native;
            const EngineValues values{engine_state, native, {}};
            header.spans |= SpanEngine;
            put(&values, sizeof(values));
        }
        if ((g_e.binds[0] | g_e.binds[1]) != 0) {
            const std::size_t at = used;
            used += sizeof(BindsHeader);
            BindsHeader binds{};
            constexpr u32 Slots = Maxwell3D::Regs::MaxConstBuffers;
            for (u32 word = 0; word < 2; ++word) {
                u64 bits = g_e.binds[word];
                while (bits != 0) {
                    const u32 bit = word * 64 + static_cast<u32>(std::countr_zero(bits));
                    bits &= bits - 1;
                    const auto& info = e.state.shader_stages[bit / Slots].const_buffers[bit % Slots];
                    const BindValue value{
                        .stage = static_cast<u8>(bit / Slots),
                        .slot = static_cast<u8>(bit % Slots),
                        .enabled = static_cast<u8>(info.enabled ? 1 : 0),
                        .reserved = 0,
                        .size = info.size,
                        .address = info.address,
                    };
                    std::memcpy(out.data() + used, &value, sizeof(value));
                    used += sizeof(value);
                    ++binds.count;
                }
            }
            std::memcpy(out.data() + at, &binds, sizeof(binds));
            g_e.binds = {};
            header.spans |= SpanBinds;
        }
        if (lent_pack && g_e.values == 1) {
            header.flags |= PacketValues; // R may compare before it takes the engine's values
        }
    }
    if (lent_pack) {
        ResetValues(engine); // R's view takes the engine's values at this packet
    }
    return used;
}

void ApplySpans(RegisterView& view, const PacketHeader& header, const u8* spans) {
    std::size_t at = 0;
    const auto take = [&](void* data, std::size_t bytes) {
        std::memcpy(data, spans + at, bytes);
        at += (bytes + 7) & ~std::size_t{7};
    };
    if ((header.spans & SpanDelta) != 0) {
        for (u32 i = 0; i < header.delta_count; ++i) {
            RegPair pair;
            std::memcpy(&pair, spans + at + i * sizeof(RegPair), sizeof(pair));
            if (pair.index < NumRegs) {
                view.regs.reg_array[pair.index] = pair.value;
            }
        }
        at += header.delta_count * sizeof(RegPair);
    }
    if ((header.spans & SpanFlags) != 0) {
        FlagBits raised;
        take(&raised, sizeof(raised));
        view.flags |= raised;
    }
    t_draw_fresh = (header.spans & SpanDraw) != 0;
    if (t_draw_fresh) {
        take(&view.draw, sizeof(view.draw));
    }
    if ((header.spans & SpanPrograms) != 0) {
        take(&view.programs, sizeof(view.programs));
    }
    if ((header.spans & SpanPools) != 0) {
        take(&view.pools, sizeof(view.pools));
    }
    ++view.packets;
}

void SeedView(RegisterView& view, const Maxwell3D& engine) {
    std::memcpy(&view.regs, &engine.regs, sizeof(view.regs));
    view.flags = engine.dirty.flags & Parts().forward;
    FillDraw(view.draw, engine);
    view.pools = PoolsOf(engine.regs);
    view.seeded = true;
    t_draw_fresh = false;
}

void VerifyView(const RegisterView& view, const Maxwell3D& engine) {
    Stats::ViewCompared();
    const Partition& parts = Parts();
    // Words: 64 at a time, the differing blocks word by word.
    u32 differing = 0;
    u32 first = NumRegs;
    constexpr u32 Block = 64;
    for (u32 begin = 0; begin < NumRegs; begin += Block) {
        const u32 words = (std::min)(Block, NumRegs - begin);
        if (std::memcmp(&view.regs.reg_array[begin], &engine.regs.reg_array[begin],
                        words * sizeof(u32)) == 0) {
            continue;
        }
        for (u32 i = begin; i < begin + words; ++i) {
            if (parts.watched.test(i) && view.regs.reg_array[i] != engine.regs.reg_array[i]) {
                first = first == NumRegs ? i : first;
                ++differing;
            }
        }
    }
    if (differing != 0) {
        t_note = fmt::format("lent draw: {} words, first {:#x} view {:#x} engine {:#x}",
                             differing, first, view.regs.reg_array[first],
                             engine.regs.reg_array[first]);
        Note(Check::ViewWords, t_note.c_str());
    }
    // At step 4 E's buffer fill clears E's copies of the shared flags (VertexBuffers) for its
    // unlent draws while the view keeps them raised: only the R-only flags must agree then.
    // And with rule 1 the R-only flags live on R's draw view (part c) at step 4, not here: the
    // engine takes that view's before the call. So no flags compare at step 4.
    const FlagBits flag_mask = Step() == StepLevel::Unlend ? FlagBits{} : parts.forward;
    const FlagBits flags = (view.flags ^ engine.dirty.flags) & flag_mask;
    if (flags.any()) {
        u32 flag = 0;
        while (!flags.test(flag)) {
            ++flag;
        }
        t_note = fmt::format("lent draw: {} flags, first {} (view {}, engine {})", flags.count(),
                             flag, view.flags.test(flag), engine.dirty.flags.test(flag));
        Note(Check::ViewFlags, t_note.c_str());
    }
    if (t_draw_fresh && !SameDraw(view.draw, engine)) {
        Note(Check::ViewDraw, "lent draw: the draw span against the draw manager");
    }
    const Pools pools = PoolsOf(engine.regs);
    if (std::memcmp(&pools, &view.pools, sizeof(Pools)) != 0) {
        Note(Check::ViewPools, "lent draw: the TIC/TSC pools");
    }
}

void BeforeLentCall(RegisterView& view, Maxwell3D& engine) {
    // Nothing: the engine's flags stay E's. Part (c)'s model (vk_one_follower_draw.h): a side
    // that did not consume a flag keeps it raised, so a lent draw after unlent ones redoes that
    // part from registers equal to the view's (a rebind too many, never one too few).
    (void)view;
    (void)engine;
}

void AfterLentCall(RegisterView& view, Maxwell3D& engine) {
    // The lent code ran both halves on the engine: the view and E's baseline take what it left.
    const FlagBits now = engine.dirty.flags & Parts().forward;
    view.flags = now;
    g_e.after = now; // E is parked
}

void NoteEngineFlags(const Maxwell3D& engine) noexcept {
    g_e.after = engine.dirty.flags & Parts().forward;
}

void SetTexturePages(const PageFilter* filter) noexcept {
    g_texture_pages = filter;
}

void SetShaderPages(const PageFilter* filter) noexcept {
    g_shader_pages = filter;
}

void SetUnionPages(const PageFilter* filter) noexcept {
    g_union_pages = filter;
}

int CbLean() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_cb_lean,
                                            "FRITCHII_ONEF_CB_LEAN");
}

bool StreamWriteHits(DAddr address, u64 size) noexcept {
    return g_texture_pages != nullptr && g_texture_pages->Any(address, size) && Behind() != 0;
}

void GuardStreamWrite(DAddr address, u64 size) noexcept {
    // FRITCHII_ESPLIT=1: a game memory write on E1 outside a join (E1Touch, must be 0).
    ESplit::NoteMemoryWrite();
    if (!Detail::t_engine || size == 0 || g_texture_pages == nullptr) {
        return;
    }
    // FRITCHII_ESPLIT, E2 while E1 runs ahead, a write E1 did not check (not an S4 constant
    // buffer record's: E2ExclusiveBegin skips those): the guard reads E1's pool ranges and may
    // drain R, so E1 is held at a safe point for it.
    struct Exclusive {
        bool held = ESplit::g_async && ESplit::E2ExclusiveBegin();
        ~Exclusive() {
            if (held) {
                ESplit::E2ExclusiveEnd();
            }
        }
    } const esplit_exclusive;
    // FRITCHII_ESPLIT_COUNT: an E2 write onto shader pages opens the window in which E1's next
    // pipeline lookup would have to wait for it.
    if (ESplit::g_state != 0) [[unlikely]] {
        ESplit::NoteStreamWriteImpl(g_shader_pages != nullptr && g_shader_pages->Any(address, size));
    }
    // FRITCHII_ESPLIT_CB_ASYNC (S4): E2 writes constant buffer words E1 queued after it found
    // these pages clear (StreamPagesClear, with E's pools at E1's place: E2 does not read them,
    // they are E1's). A filter bit set since then is ESplitCbGuard (must be 0): R still plays
    // everything before the write first, but the next draw's lend (stream_hit, E1's) is not
    // asked for, E1 routed it already.
    if (ESplit::t_async_cb) [[unlikely]] {
        if (g_texture_pages->Any(address, size) ||
            (g_shader_pages != nullptr && g_shader_pages->Any(address, size))) {
            ESplit::NoteCbGuardHit();
            if (Behind() != 0) {
                Drain();
            }
        }
        return;
    }
    Stats::StreamWriteChecked();
    const DAddr end = address + size;
    const auto in_pools = [&] {
        for (const auto& [begin, pool_end] : g_e.pools) {
            if (begin < end && address < pool_end) {
                return true;
            }
        }
        return false;
    };
    // FRITCHII_ONEF_CB_LEAN & 2 (fable_b 23.1 cut 3): the union holds every texture and shader
    // page bit (they mark it in the same Mark), so a clear union is both filters clear.
    if (g_union_pages != nullptr && (CbLean() & 2) != 0 && !g_union_pages->Any(address, size) &&
        !in_pools()) {
        return;
    }
    if (!g_texture_pages->Any(address, size) &&
        (g_shader_pages == nullptr || !g_shader_pages->Any(address, size)) && !in_pools()) {
        return;
    }
    // FRITCHII_ESPLIT: a write onto shader pages: E1's next lookup drains and flushes first.
    if (ESplit::g_thread && g_shader_pages != nullptr && g_shader_pages->Any(address, size))
        [[unlikely]] {
        ESplit::g_shader_pending.store(true, std::memory_order_relaxed);
    }
    ++g_e.on_texture;
    if (Behind() == 0) {
        return; // R has nothing left that could read the page before this write: today's order
    }
    // R may still read this page for an earlier draw (texture data, TIC or TSC entries): the
    // write waits for R to play everything before it, which is the write run lent.
    t_note = fmt::format("stream write {:#x}+{:#x} with {} packets unplayed", address, size,
                         Behind());
    Note(Check::StreamWrite, t_note.c_str());
    Drain();
    Stats::StreamWriteHandled(Behind() == 0);
    // Part (c): the next draw is lent too (LentReason::StreamWrite), so its build reads E's
    // engine with the write landed.
    g_e.stream_hit = true;
}

bool TakeStreamWriteLend() noexcept {
    return std::exchange(g_e.stream_hit, false);
}

bool OnShaderPages(DAddr address, u64 size) noexcept {
    return g_shader_pages != nullptr && size != 0 && g_shader_pages->Any(address, size);
}

bool StreamPagesClear(DAddr address, u64 size) noexcept {
    // GuardStreamWrite's test without the act: no page that ever held an image or shader code,
    // not E's pools. No filters: nothing can be shown clear.
    if (g_texture_pages == nullptr || g_shader_pages == nullptr) {
        return false;
    }
    if (g_texture_pages->Any(address, size) || g_shader_pages->Any(address, size)) {
        return false;
    }
    const DAddr end = address + size;
    for (const auto& [begin, pool_end] : g_e.pools) {
        if (begin < end && address < pool_end) {
            return false;
        }
    }
    return true;
}

u64 StreamWritesOnTexturePages() noexcept {
    return g_e.on_texture;
}

} // namespace VideoCommon::OneFollower
