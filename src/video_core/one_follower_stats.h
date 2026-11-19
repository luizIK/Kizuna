// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower's counters and 10 s log lines (part d of work\fable_b_one_follower.md, sections 5,
// 7 and 8). The names of what is counted are one_follower.h's (LentOp, LentReason, UnlendKind,
// Check); this file adds the storage, the calls that feed it and four log lines:
//
//   always while FRITCHII_ONEF=1 (onefollow_gate.py and memstage_gate.py read the first):
//     "One follower checks (must be 0; this window / session): view words 0 / 0, ..."
//     "One follower verify (step ...) 10 s: view compares ..., stream-write guard: ..."
//   with the GPU profile on (EDEN_NX2_GPU_PROFILE):
//     "One follower hand-overs (step ...) 10 s: packets ..., lent by op ..., the hop ..."
//     "One follower threads (FRITCHII_ONEF_AHEAD=...) 10 s: E's ahead window ..., R busy ..."
//
// Cheap by construction: counters written by one thread only (E's draw packets, R's times) are
// bumped with a relaxed load and store (no locked instruction); counters any thread may write
// (lends, must-be-0s) use a relaxed fetch_add, and a lend costs a round trip anyway. The logger
// only reads and keeps its own snapshot, so nothing is ever reset under a writer. Counts run
// always; times only while Timing() (the GPU profile), so callers read the clock only then.
//
// Check::StreamWrite is the guard's hit count (a hit is handled: the write is made lent), so it is
// logged on the verify line, not the must-be-0 line. Its must-be-0 is "stream write not lent" (a
// hit whose write still ran unlent), added on the must-be-0 line after one_follower.h's names.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"

namespace VideoCommon::OneFollower::Stats {

/// The draw kinds counted: UnlendKind's eight bits, then a draw without a kind bit.
inline constexpr size_t KindSlots = 9;
inline constexpr std::array<const char*, KindSlots> kind_names{
    "plain indexed", "plain arrays", "instance loop", "indirect", "list",
    "mesh",          "GPU count",    "inline index",  "no kind",
};

/// Packets R had not played when E pushed one (Behind() before the push), bucketed.
inline constexpr size_t BehindSlots = 6;
inline constexpr std::array<const char*, BehindSlots> behind_names{"0",    "1",      "2-7",
                                                                   "8-63", "64-511", "512+"};

/// Where R's time went (R only).
enum class RenderTime : u8 {
    DrawPackets,  ///< unlent draw packets: take, apply the spans, build, record (step 4)
    LentCalls,    ///< inside lent calls (E parked); LentRan's call_ns, counted here as well
    OtherPackets, ///< delta packets, seeds, ring upkeep
    Idle,         ///< no packet: spinning or asleep (RenderSlept counts the sleeps)
    WaitForE,     ///< R waiting for something E owns (a served landing, a ring B packet late)
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(RenderTime::Count)> render_time_names{
    "draw packets", "lent calls", "other packets", "idle", "waited for E",
};

/// The extra must-be-0 this file adds (logged after Check's names on the must-be-0 line).
inline constexpr const char* stream_not_lent_name = "stream write not lent";

struct Counters {
    // E only (unlent draw packets: E pushes them, nobody else does).
    std::atomic<u64> draw_packets{};
    std::atomic<u64> draw_bytes{};
    std::atomic<u64> draw_words{}; ///< RegPair count of their deltas
    std::array<std::atomic<u64>, KindSlots> unlent_kind{};
    std::array<std::atomic<u64>, BehindSlots> behind_draw{};
    std::atomic<u64> behind_max_draw{};
    std::atomic<u64> window_waits{};
    std::atomic<u64> window_ns{};
    std::atomic<u64> stream_checked{};

    // Lent and delta packets: E only (Pushed bumps them); drains: fetch_add.
    std::atomic<u64> lent_packets{};
    std::atomic<u64> delta_packets{};
    std::atomic<u64> other_bytes{};
    std::atomic<u64> other_words{};
    std::array<std::atomic<u64>, BehindSlots> behind_other{};
    std::atomic<u64> behind_max_other{};
    std::array<std::atomic<u64>, static_cast<size_t>(LentOp::Count)> lent_op{};
    std::array<std::atomic<u64>, static_cast<size_t>(LentReason::Count)> lent_reason{};
    std::array<std::atomic<u64>, KindSlots> lent_kind{};
    std::array<std::atomic<u64>, static_cast<size_t>(LentOp::Count)> parked_ns{};
    std::atomic<u64> drains{};
    std::atomic<u64> drain_ns{};
    std::atomic<u64> stream_lent{};

    // R only.
    std::array<std::atomic<u64>, static_cast<size_t>(LentOp::Count)> lent_call_ns{};
    std::atomic<u64> lent_reach_ns{};
    std::atomic<u64> lents_ran{};
    std::array<std::atomic<u64>, static_cast<size_t>(RenderTime::Count)> render_ns{};
    std::atomic<u64> render_sleeps{};
    std::atomic<u64> view_compares{};
    std::atomic<u64> build_compares{};

    // Must-be-0s and the guard's hits (any thread, fetch_add).
    std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> checks{};
    std::atomic<u64> stream_not_lent{};
};

inline Counters g_counters;

/// A counter only one thread writes: a relaxed load and store, no locked instruction.
inline void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
/// A counter several threads may write.
inline void Add(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

[[nodiscard]] inline size_t KindSlot(u32 kind) noexcept {
    for (size_t i = 0; i < KindSlots - 1; ++i) {
        if ((kind >> i) & 1u) {
            return i;
        }
    }
    return KindSlots - 1;
}

[[nodiscard]] inline size_t BehindSlot(u64 behind) noexcept {
    return behind == 0 ? 0 : behind == 1 ? 1 : behind < 8 ? 2 : behind < 64 ? 3 : behind < 512 ? 4 : 5;
}

/// Times are counted only with the GPU profile on: read the clock only when this is true.
[[nodiscard]] inline bool Timing() noexcept {
    return Nx2Profile::Enabled();
}
/// Nanoseconds, the profiler's clock.
[[nodiscard]] inline u64 Now() noexcept {
    return Nx2Profile::Now();
}

// --- E (part a: the ring and Lend) -----------------------------------------------------------------

/// Every packet pushed to R, right after the push. `kind` is the draw's UnlendKind bit (0 for
/// anything but a draw), `behind` is Behind() just before the push. A lent packet counts its op,
/// and for a draw its reason (header.reason) and its kind: these are the lends, "hand-overs".
inline void Pushed(const PacketHeader& header, u32 kind, u64 behind) noexcept {
    Counters& c = g_counters;
    const size_t slot = BehindSlot(behind);
    if (header.kind == PacketKind::Draw) {
        Bump(c.draw_packets);
        Bump(c.draw_bytes, header.bytes);
        Bump(c.draw_words, header.delta_count);
        Bump(c.unlent_kind[KindSlot(kind)]);
        Bump(c.behind_draw[slot]);
        if (behind > c.behind_max_draw.load(std::memory_order_relaxed)) {
            c.behind_max_draw.store(behind, std::memory_order_relaxed);
        }
        return;
    }
    // Pushed runs on E only (Lend and PushPacked, both behind t_engine): single-writer bumps,
    // not locked adds, for these as for the draw counters (about 200,000 op packets a second).
    Bump(header.kind == PacketKind::Lent ? c.lent_packets : c.delta_packets);
    Bump(c.other_bytes, header.bytes);
    Bump(c.other_words, header.delta_count);
    Bump(c.behind_other[slot]);
    u64 seen = c.behind_max_other.load(std::memory_order_relaxed);
    while (behind > seen &&
           !c.behind_max_other.compare_exchange_weak(seen, behind, std::memory_order_relaxed)) {
    }
    if (header.kind != PacketKind::Lent) {
        return;
    }
    const size_t op = (std::min)(static_cast<size_t>(header.op),
                                 static_cast<size_t>(LentOp::Count) - 1);
    Bump(c.lent_op[op]);
    if (static_cast<LentOp>(op) == LentOp::Draw) {
        Bump(c.lent_reason[(std::min)(static_cast<size_t>(header.reason),
                                      static_cast<size_t>(LentReason::Count) - 1)]);
        Bump(c.lent_kind[KindSlot(kind)]);
    }
}

/// E, after R released it: the whole time E was parked for one lend of `op` (push to release).
inline void Parked(LentOp op, u64 ns) noexcept {
    Add(g_counters.parked_ns[static_cast<size_t>(op)], ns);
}

/// E waited `ns` for R because FRITCHII_ONEF_AHEAD's window was full (E only).
inline void WindowWait(u64 ns) noexcept {
    Bump(g_counters.window_waits);
    Bump(g_counters.window_ns, ns);
}

/// Drain(): E waited `ns` until R had played every packet.
inline void Drained(u64 ns) noexcept {
    Add(g_counters.drains);
    Add(g_counters.drain_ns, ns);
}

// --- R (parts a and c) -------------------------------------------------------------------------------

/// R ran a lent packet's call: `reach_ns` from E's push to the call's start (R's catch-up on the
/// packets before it and its wake), `call_ns` the call itself. E's Parked less these two is what
/// the hop costs on top (the release and E's wake).
inline void LentRan(LentOp op, u64 reach_ns, u64 call_ns) noexcept {
    Counters& c = g_counters;
    Bump(c.lents_ran);
    Bump(c.lent_reach_ns, reach_ns);
    Bump(c.lent_call_ns[static_cast<size_t>(op)], call_ns);
    Bump(c.render_ns[static_cast<size_t>(RenderTime::LentCalls)], call_ns);
}

/// R's time outside lent calls (R only; LentRan counts the lent calls' own time).
inline void RenderTimeSpent(RenderTime what, u64 ns) noexcept {
    Bump(g_counters.render_ns[static_cast<size_t>(what)], ns);
}

/// R went to sleep for want of packets (R only).
inline void RenderSlept() noexcept {
    Bump(g_counters.render_sleeps);
}

// --- Verify (parts b and c) ---------------------------------------------------------------------------

/// R ran VerifyView once (step 2's compare, every lent draw with PacketVerify): the gate's proof
/// that "view ... 0" means something. Mismatches go through Note(Check::View*).
inline void ViewCompared() noexcept {
    Bump(g_counters.view_compares);
}

/// R compared one draw built from the packet with the lent path's (step 3, and step 4's forced
/// lent draws). Mismatches go through Note(Check::Build*).
inline void BuildCompared() noexcept {
    Bump(g_counters.build_compares);
}

// --- The stream-write guard (section 5 / 7, whoever implements StreamWriteHits) --------------------

/// E checked one stream write against the texture pages (each StreamWriteHits call; E only).
/// A hit itself is Note(Check::StreamWrite, what).
inline void StreamWriteChecked() noexcept {
    Bump(g_counters.stream_checked);
}

/// After a hit: whether the write was made lent. `false` is the must-be-0 "stream write not lent".
inline void StreamWriteHandled(bool lent) noexcept {
    Add(lent ? g_counters.stream_lent : g_counters.stream_not_lent);
}

// --- The log lines ---------------------------------------------------------------------------------

/// What the lines print about the switches (part a passes the values it runs with).
struct Logged {
    StepLevel step{};   ///< Step(): the effective step after nesting
    u8 asked{};         ///< the step the switches asked for before nesting
    u32 unlend_mask{};  ///< FRITCHII_ONEF_UNLEND as used (0 below step 4)
    int ahead{};        ///< FRITCHII_ONEF_AHEAD as used (0 = the ring)
    int verify_every{}; ///< FRITCHII_ONEF_VERIFY_EVERY as used (0 = off)
};

/// Logs the lines when 10 s have passed since the last time (the first call only starts the
/// window). Call it from any thread at frame ends while FRITCHII_ONEF=1; a call while another
/// thread logs returns at once.
void MaybeLog(const Logged& now);

} // namespace VideoCommon::OneFollower::Stats
