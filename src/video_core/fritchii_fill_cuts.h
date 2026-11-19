// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <unordered_map>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"

/// The rest of E's fill, fable_b 25.2 cuts e to h (work\fable_b_one_follower.md), each behind its
/// own live switch: 0 = the code as before (one relaxed load and a branch a draw), 1 = on,
/// 2 = on with the old path run as well and compared (must be 0), 3 = dry count: the old path
/// only, with the calls the cut would skip counted and their time sampled (1 call in 64 timed,
/// scaled by 64). One line per cut every 10 s (MaybeReport, from the draw that has a cut on).
///
/// - FRITCHII_FILL_GEOM_KEEP (e): the vertex and index binds' synchronize, skipped while the
///   range's tracker managers kept the dirty generation sum a synchronize left it clean at.
/// - FRITCHII_FILL_UPDATE_KEEP (f): the index and vertex bindings (translation, buffer lookup)
///   kept while their inputs, the mappings, the channel and the set of buffers are unchanged.
/// - FRITCHII_HANDLE_PROG_KEEP (g): E's handle program results kept while every word they read
///   (and every constant buffer address they read through) has the value it had.
/// - FRITCHII_FILL_BRACKET_LEAN (h): E's tracker record only for the slots R can read, ring B's
///   push without the window counters and the room re-check RingRoom already made.
///
/// Counters have one writing thread (the buffer cache's holder, or E), so a plain load and store
/// adds to them; R's own tracker check counts in counters only R writes.
namespace VideoCommon::FillCuts {

enum class Cut : u32 {
    GeomKeep,
    UpdateKeep,
    HandleProgKeep,
    BracketLean,
    Count,
};

inline constexpr int Off = 0;
inline constexpr int On = 1;
inline constexpr int Verify = 2;
inline constexpr int Dry = 3;

/// A cut whose cheap must-be-0 check failed while it was on (value 1): off for the session.
inline std::array<std::atomic<bool>, static_cast<size_t>(Cut::Count)> g_stopped{};

/// The cut's value for this draw: 0 to 3 (anything else is 0; a stopped cut's 1 is 0).
[[nodiscard]] inline int Mode(Cut cut) noexcept {
    namespace Debug = VideoCore::Nx2Debug;
    int value = 0;
    switch (cut) {
    case Cut::GeomKeep:
        value = Debug::SwitchValue(Debug::g_fill_geom_keep, "FRITCHII_FILL_GEOM_KEEP");
        break;
    case Cut::UpdateKeep:
        value = Debug::SwitchValue(Debug::g_fill_update_keep, "FRITCHII_FILL_UPDATE_KEEP");
        break;
    case Cut::HandleProgKeep:
        value = Debug::SwitchValue(Debug::g_handle_prog_keep, "FRITCHII_HANDLE_PROG_KEEP");
        break;
    case Cut::BracketLean:
        value = Debug::SwitchValue(Debug::g_fill_bracket_lean, "FRITCHII_FILL_BRACKET_LEAN");
        break;
    case Cut::Count:
        break;
    }
    if (value == 0) [[likely]] {
        return Off;
    }
    if (value < 0 || value > Dry) {
        return Off;
    }
    if (value == On && g_stopped[static_cast<size_t>(cut)].load(std::memory_order_relaxed)) {
        return Off;
    }
    return value;
}

using Counter = std::atomic<u64>;

/// One writer: a plain load and store.
inline void Add(Counter& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

/// 1 in 64 of a site's calls are timed (time stamp ticks); the report scales by SampleScale.
inline constexpr u64 SampleScale = 64;
struct Sample {
    u32 n{};
    [[nodiscard]] bool Take() noexcept {
        return (++n & (SampleScale - 1)) == 0;
    }
};

[[nodiscard]] inline u64 Ticks() noexcept {
    return Nx2Profile::Ticks();
}

/// Must-be-0 compares: in the window, since boot, and how many were made.
struct Compare {
    Counter checked{};
    Counter differ{};
    Counter differ_total{};
    /// Counts a difference; true for the first 16 since boot (log those).
    bool Differ() noexcept {
        Add(differ);
        Add(differ_total);
        return differ_total.load(std::memory_order_relaxed) <= 16;
    }
};

/// Cut e. The buffer cache's holder.
struct GeomStats {
    Counter index_syncs{};  ///< index buffer synchronizes the memo looked at
    Counter vertex_syncs{}; ///< vertex buffer synchronizes the memo looked at
    Counter hits{};         ///< of them, memo hits: skipped at 1
    Counter check_ticks{};  ///< the memo's own look, sampled over every sync
    Counter hit_ticks{};    ///< the synchronize of a hit (what 1 removes), sampled over hits
    Compare compare{};      ///< a hit's synchronize uploaded or found a dirty page (2 and 3)
    Sample check_sample{};
    Sample hit_sample{};
};
inline GeomStats g_geom;

/// Cut f. The buffer cache's holder.
struct UpdateStats {
    Counter index_updates{};
    Counter index_hits{};
    Counter vertex_updates{};
    Counter vertex_hits{};
    Counter check_ticks{}; ///< the memo's own look, sampled over every update
    Counter hit_ticks{};   ///< the resolve of a hit (what 1 removes), sampled over hits
    Compare compare{};     ///< a hit's resolve gave another binding (2 and 3)
    Sample check_sample{};
    Sample hit_sample{};
};
inline UpdateStats g_update;

/// Cut g. E.
struct ProgramStats {
    Counter draws{};    ///< draws with handle programs whose fill went through the cut
    Counter programs{}; ///< program evaluations (a table counts once)
    Counter same{};     ///< every word and constant buffer address read unchanged: skipped at 1
    Counter moved{};    ///< a constant buffer the program reads through was rebound
    Counter changed{};  ///< a word it read changed
    Counter first{};    ///< no record yet (or the record could not hold the reads)
    Counter check_ticks{}; ///< the replay compare, sampled over every compare
    Counter hit_ticks{};   ///< the evaluation of a same-words program, sampled over them
    Compare compare{};     ///< same words, another result (2 and 3)
    Sample check_sample{};
    Sample hit_sample{};
};
inline ProgramStats g_program;

/// Cut h. E, except the `r_` counters (R).
struct BracketStats {
    Counter tracker_draws{}; ///< draws whose fill made the trackers' record
    Counter lean_slots{};    ///< record slots the lean record leaves out (of 3 a draw)
    Counter full_ticks{};    ///< the whole record, sampled over tracker draws
    Counter lean_ticks{};    ///< the lean record, sampled over the same draws
    Counter pushes{};        ///< ring B draw pushes on E
    Counter push_ticks{};    ///< the push parts the lean push leaves out, sampled
    Compare push_compare{};  ///< the lean room rule said "fits" and the full check disagreed
    Counter r_checked{};     ///< R: reads of a record with left-out slots (R writes)
    Counter r_differ{};      ///< R: a slot R read was one the lean record left out (must be 0)
    Counter r_differ_total{};
    Sample tracker_sample{};
    Sample push_sample{};
};
inline BracketStats g_bracket;

/// Cut h, R: a draw packet's tracker record whose slots `read` (bit i: slot i) R reads now. A
/// slot the lean record left out is a difference (must be 0); at 1 it turns the cut off for the
/// session (its record then is whole again from the next draw E fills).
inline void NoteTrackerRead(u8 lean_skipped, u8 read) noexcept {
    BracketStats& s = g_bracket;
    Add(s.r_checked);
    if ((lean_skipped & read) == 0) [[likely]] {
        return;
    }
    Add(s.r_differ);
    Add(s.r_differ_total);
    g_stopped[static_cast<size_t>(Cut::BracketLean)].store(true, std::memory_order_relaxed);
    if (s.r_differ_total.load(std::memory_order_relaxed) <= 16) {
        LOG_ERROR(HW_GPU,
                  "Fill bracket lean (FRITCHII_FILL_BRACKET_LEAN): R read tracker slots {:#x} of "
                  "a record that left out {:#x}; the cut's 1 is off for the session",
                  read, lean_skipped);
    }
}

namespace Detail {

struct ReportState {
    std::atomic<u64> calls{};
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
    u64 since_ns{};
    u64 since_ticks{};
    std::unordered_map<const Counter*, u64> last;
};
inline ReportState g_report;

inline const char* Label(int mode) {
    switch (mode) {
    case On:
        return "on";
    case Verify:
        return "verify";
    case Dry:
        return "dry count";
    default:
        return "off";
    }
}

} // namespace Detail

namespace Detail {
NX2_PROFILE_NOINLINE inline void Report() noexcept;
} // namespace Detail

/// From a draw that has a cut on: one line per cut every 10 s (the clock read 1 call in 1024).
inline void MaybeReport() noexcept {
    Detail::ReportState& r = Detail::g_report;
    const u64 calls = r.calls.load(std::memory_order_relaxed) + 1;
    r.calls.store(calls, std::memory_order_relaxed);
    if ((calls & 1023) == 0) [[unlikely]] {
        Detail::Report();
    }
}

NX2_PROFILE_NOINLINE inline void Detail::Report() noexcept {
    ReportState& r = g_report;
    if (r.busy.test_and_set(std::memory_order_acquire)) {
        return;
    }
    const u64 now_ns = Nx2Profile::Now();
    const u64 now_ticks = Ticks();
    if (r.since_ns == 0) {
        r.since_ns = now_ns;
        r.since_ticks = now_ticks;
    }
    if (now_ns - r.since_ns < 10'000'000'000ULL) {
        r.busy.clear(std::memory_order_release);
        return;
    }
    const double window = static_cast<double>(now_ns - r.since_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now_ns - r.since_ns) /
        static_cast<double>((std::max)(now_ticks - r.since_ticks, u64{1}));
    const auto delta = [&r](const Counter& counter) {
        const u64 value = counter.load(std::memory_order_relaxed);
        u64& last = r.last[&counter];
        const u64 moved = value - last;
        last = value;
        return moved;
    };
    const auto rate = [&](u64 count) { return static_cast<double>(count) / window; };
    // Sampled ticks (1 in 64) to ms/s.
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * static_cast<double>(SampleScale) * ns_per_tick / 1e6 /
               window;
    };
    const auto pct = [](u64 part, u64 whole) {
        return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };

    // Cut e.
    {
        GeomStats& s = g_geom;
        const int mode = Mode(Cut::GeomKeep);
        const u64 index = delta(s.index_syncs);
        const u64 vertex = delta(s.vertex_syncs);
        const u64 hits = delta(s.hits);
        const double hit_ms = ms(delta(s.hit_ticks));
        const double check_ms = ms(delta(s.check_ticks));
        const u64 checked = delta(s.compare.checked);
        const u64 differ = delta(s.compare.differ);
        if (mode != Off || index + vertex != 0) {
            LOG_INFO(HW_GPU,
                     "Fill geometry keep (FRITCHII_FILL_GEOM_KEEP={} {}), last {:.0f} s: "
                     "synchronizes {:.0f}/s index, {:.0f}/s vertex; memo hits {:.0f}/s "
                     "({:.0f}%{}); the hits' synchronize {:.2f} ms/s, the memo's look {:.2f} ms/s "
                     "(1 in 64 timed); net {:.2f} ms/s; hits that uploaded or found a dirty page "
                     "(must be 0): {} of {}, since boot {}",
                     mode, Label(mode), window, rate(index), rate(vertex), rate(hits),
                     pct(hits, index + vertex), mode == On ? ", skipped" : "", hit_ms, check_ms,
                     hit_ms - check_ms, differ, checked,
                     s.compare.differ_total.load(std::memory_order_relaxed));
        }
    }
    // Cut f.
    {
        UpdateStats& s = g_update;
        const int mode = Mode(Cut::UpdateKeep);
        const u64 index = delta(s.index_updates);
        const u64 index_hits = delta(s.index_hits);
        const u64 vertex = delta(s.vertex_updates);
        const u64 vertex_hits = delta(s.vertex_hits);
        const double hit_ms = ms(delta(s.hit_ticks));
        const double check_ms = ms(delta(s.check_ticks));
        const u64 checked = delta(s.compare.checked);
        const u64 differ = delta(s.compare.differ);
        if (mode != Off || index + vertex != 0) {
            LOG_INFO(HW_GPU,
                     "Fill update keep (FRITCHII_FILL_UPDATE_KEEP={} {}), last {:.0f} s: index "
                     "updates {:.0f}/s ({:.0f}% kept), vertex slot updates {:.0f}/s ({:.0f}% "
                     "kept){}; the kept ones' resolve {:.2f} ms/s, the memo's look {:.2f} ms/s "
                     "(1 in 64 timed); net {:.2f} ms/s; a kept binding the resolve gave otherwise "
                     "(must be 0): {} of {}, since boot {}",
                     mode, Label(mode), window, rate(index), pct(index_hits, index), rate(vertex),
                     pct(vertex_hits, vertex), mode == On ? ", skipped" : "", hit_ms, check_ms,
                     hit_ms - check_ms, differ, checked,
                     s.compare.differ_total.load(std::memory_order_relaxed));
        }
    }
    // Cut g.
    {
        ProgramStats& s = g_program;
        const int mode = Mode(Cut::HandleProgKeep);
        const u64 draws = delta(s.draws);
        const u64 programs = delta(s.programs);
        const u64 same = delta(s.same);
        const u64 moved = delta(s.moved);
        const u64 changed = delta(s.changed);
        const u64 first = delta(s.first);
        const double hit_ms = ms(delta(s.hit_ticks));
        const double check_ms = ms(delta(s.check_ticks));
        const u64 checked = delta(s.compare.checked);
        const u64 differ = delta(s.compare.differ);
        if (mode != Off || programs != 0) {
            LOG_INFO(HW_GPU,
                     "Handle program keep (FRITCHII_HANDLE_PROG_KEEP={} {}), last {:.0f} s: draws "
                     "with programs {:.0f}/s, programs {:.0f}/s: same words {:.0f}/s ({:.0f}%{}), "
                     "constant buffer rebound {:.0f}/s, words changed {:.0f}/s, no record "
                     "{:.0f}/s; the same-words evaluations {:.2f} ms/s, the replay compare "
                     "{:.2f} ms/s (1 in 64 timed); net {:.2f} ms/s; same words but another result "
                     "(must be 0): {} of {}, since boot {}",
                     mode, Label(mode), window, rate(draws), rate(programs), rate(same),
                     pct(same, programs), mode == On ? ", kept" : "", rate(moved), rate(changed),
                     rate(first), hit_ms, check_ms, hit_ms - check_ms, differ, checked,
                     s.compare.differ_total.load(std::memory_order_relaxed));
        }
    }
    // Cut h.
    {
        BracketStats& s = g_bracket;
        const int mode = Mode(Cut::BracketLean);
        const u64 tracker = delta(s.tracker_draws);
        const u64 lean_slots = delta(s.lean_slots);
        const double full_ms = ms(delta(s.full_ticks));
        const double lean_ms = ms(delta(s.lean_ticks));
        const u64 pushes = delta(s.pushes);
        const double push_ms = ms(delta(s.push_ticks));
        const u64 push_checked = delta(s.push_compare.checked);
        const u64 push_differ = delta(s.push_compare.differ);
        const u64 r_checked = delta(s.r_checked);
        const u64 r_differ = delta(s.r_differ);
        if (mode != Off || tracker + pushes != 0) {
            LOG_INFO(HW_GPU,
                     "Fill bracket lean (FRITCHII_FILL_BRACKET_LEAN={} {}), last {:.0f} s: "
                     "tracker records {:.0f}/s, slots the lean record leaves out {:.0f}/s ({:.0f}% "
                     "of 3 a draw); the record {:.2f} ms/s whole, {:.2f} ms/s lean; ring B pushes "
                     "{:.0f}/s, the parts the lean push leaves out {:.2f} ms/s (1 in 64 timed); "
                     "net {:.2f} ms/s; R read a slot the lean record leaves out (must be 0): {} "
                     "of {}, since boot {}; room rule disagreed (must be 0): {} of {}, since boot "
                     "{}",
                     mode, Label(mode), window, rate(tracker), rate(lean_slots),
                     pct(lean_slots, 3 * tracker), full_ms, lean_ms, rate(pushes), push_ms,
                     full_ms - lean_ms + push_ms, r_differ, r_checked,
                     s.r_differ_total.load(std::memory_order_relaxed), push_differ, push_checked,
                     s.push_compare.differ_total.load(std::memory_order_relaxed));
        }
    }
    r.since_ns = now_ns;
    r.since_ticks = now_ticks;
    r.busy.clear(std::memory_order_release);
}

} // namespace VideoCommon::FillCuts
