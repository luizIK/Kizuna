// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower (one_follower.h), part (b): the engine side's write tracking, in a header of its
// own so maxwell_3d.h can include it (one_follower.h includes maxwell_3d.h).

#include <array>
#include <atomic>

#include "common/common_types.h"

namespace VideoCommon::OneFollower {

/// Maxwell3D's register write paths test this before NoteRegWrite: set while step 2 or more is on
/// (FRITCHII_ONEF_VIEW). Off: one relaxed load and a branch per register write.
inline std::atomic<bool> g_track_writes{};
/// FRITCHII_ONEF_LEAN: the register writes are tracked from boot (set before E starts, never
/// changed), so the hooks test this plain bool first (fable_b 14.4 cut 2: an atomic load at
/// each of 2.6 million writes a second).
inline bool g_track_writes_always{};

/// FRITCHII_ONEF_E_PIPELINE=1 (boot, with FRITCHII_ONEF=1; fable_b_one_follower 10.2 rule 3): E
/// owns the pipeline lookup. The pipeline cache always reads E's engine (R's view never replaces
/// it), E looks a draw's pipeline up before its fill and the packet carries it, R never looks
/// one up outside a lent call. Set once by ReadSwitches.
inline bool g_e_pipeline{};

/// Register word `index` of the engine was written on E (one_follower_view.cpp). Words written on
/// another thread (the front's own copy) are not E's and are ignored there.
void NoteRegWrite(u32 index) noexcept;

/// FRITCHII_ONEF_KEY_KEEP (fable_b 23.1 cut 2): the clean pipeline lookup's watched register
/// words (PipelineCache::BuildCleanWatch sets them), and the count of their writes NoteRegWrite
/// saw on E (or R in a lent call), every write, not only a packet's first. Bumped too when the
/// write tracking stops (writes then go unseen). E's lookup keeps its register compare when the
/// count did not move since its snapshot last equalled the registers.
inline constexpr u32 KeyWordSlots = 0x1000;
inline std::array<u64, KeyWordSlots / 64> g_key_words{};
/// NoteRegWrite's extra watchers in one byte, so it pays one load while none is on (written on E,
/// or R in a lent call with E parked). WriteWatchKey: FRITCHII_ONEF_KEY_KEEP was seen on (key
/// writes counted from then). WriteWatchSplit: FRITCHII_ESPLIT_COUNT >= 2, the E split's records
/// take the word (fritchii_e_split.h).
inline constexpr u8 WriteWatchKey = 1;
inline constexpr u8 WriteWatchSplit = 2;
inline u8 g_write_watch{};
inline u64 g_key_writes{};

/// FRITCHII_ONEF_UNLENT_VALUES (fable_b 19 step 2): Maxwell3D::ProcessCBBind tests this before
/// NoteCbBind; set on E while the constant buffer bindings go to R's view by value.
inline std::atomic<bool> g_track_binds{};
/// E bound constant buffer `slot` of shader stage `stage` (one_follower_view.cpp).
void NoteCbBind(size_t stage, u32 slot) noexcept;

namespace Detail {
/// True on E while FRITCHII_ONEF=1 (set once by MarkEngineThread): ShouldLend's one load.
inline thread_local bool t_engine{};
/// True on R (its thread start).
inline thread_local bool t_render{};
} // namespace Detail

/// fable_b 11.3: acquire latches (Nx2SemaphoreWaiters::OnWrite) by the thread that ran the
/// release: R, E, any other. With one follower every release is a lent call or a render op, so
/// R's count is the latch; E's or another's is what the queries line names.
inline std::atomic<u64> g_latches[3]{};
inline void NoteLatch() noexcept {
    g_latches[Detail::t_render ? 0 : Detail::t_engine ? 1 : 2].fetch_add(
        1, std::memory_order_relaxed);
}

/// fable_b 19.5 step 3's dry price (FRITCHII_ONEF_DRY_RELEASE, live, default 0; E keeps this
/// current at its reports): for each resumed acquire, the release that met it as E passed it
/// and as it was written, and the front's stalled time between the two.
inline std::atomic<bool> g_dry_release{};
/// The kind of release E passed.
enum class DryKind : u8 {
    PayloadRelease, ///< a payload Query, not fence-flagged (puller, 3D, compute, copy)
    PayloadFence,   ///< a fence-flagged payload Query (QMD release, puller, Ampere)
    CounterReport,  ///< a samples or other counter report
    QmdReduction,   ///< a QMD release with a reduction (a SignalFence callback)
    QmdMax,         ///< a QMD max release (a SignalFence callback)
    Count,
};
/// E: it passes a release to the u32 at `device`.
void DryNotePass(u64 device, DryKind kind) noexcept;
/// Any thread, at a GPU write of a semaphore word (Nx2SemaphoreWaiters::OnWrite).
void DryNoteRun(u64 device) noexcept;
/// FRITCHII_ONEF_SEM_E on (E keeps it current): semaphore writes run on E; the front's lagged
/// check waits for E's position on a word whose newest release is a payload one; every write
/// on E or R is checked against the word's last ("semaphore order", must be 0).
inline std::atomic<bool> g_sem_e{};
/// E or R, at a semaphore write: its place against the word's last write's.
void NoteSemaphoreOrder(u64 device) noexcept;
inline void NoteSemaphoreWrite(u64 device) noexcept {
    if (g_dry_release.load(std::memory_order_relaxed)) [[unlikely]] {
        DryNoteRun(device);
    }
    if (g_sem_e.load(std::memory_order_relaxed) && (Detail::t_engine || Detail::t_render))
        [[unlikely]] {
        NoteSemaphoreOrder(device);
    }
}
/// The front: the time it was stalled so far (pending sleeps, lagged check waits), ns.
[[nodiscard]] u64 DryStallNow() noexcept;
/// The front: a stall (a pending sleep, a lagged check's wait) begins or ends.
void DryFrontStall(bool begin) noexcept;
struct DryStallScope {
    bool on = g_dry_release.load(std::memory_order_relaxed);
    DryStallScope() {
        if (on) [[unlikely]] {
            DryFrontStall(true);
        }
    }
    ~DryStallScope() {
        if (on) [[unlikely]] {
            DryFrontStall(false);
        }
    }
    DryStallScope(const DryStallScope&) = delete;
    DryStallScope& operator=(const DryStallScope&) = delete;
};
/// The front, at a suspended acquire's resume: `since_ns` the suspension (steady clock ns),
/// `stall_at_since` DryStallNow then.
void DryNoteResume(u64 device, u64 since_ns, u64 stall_at_since) noexcept;

/// Owner asserts (section 5), for the caches' entry points: one thread-local load when off.
void CheckRenderOwned(const char* where) noexcept;
void CheckEngineOwned(const char* where) noexcept;
/// An R-owned cache (texture, pipeline, query) is used here: counted on E unless R is parked or
/// drained ("owner on E").
inline void RenderOwned(const char* where) noexcept {
    if (Detail::t_engine) [[unlikely]] {
        CheckRenderOwned(where);
    }
}
/// FRITCHII_ONEF_INVAL_E (fable_b 11.1 item 2), R: before an image upload reads game memory,
/// whether E wrote a page of [address, address + size) after the packet R plays now (E's page
/// write stamps of the invalidations it ran itself). True counts "texture read ahead of R"
/// (must be 0) and the caller makes the image upload again at its next use. Off (no stamps
/// yet, or not R): false at once.
[[nodiscard]] bool WrittenAheadOfRender(u64 address, u64 size) noexcept;

/// fable_b 13.0, with FRITCHII_ONEF_POST_FENCE: a posted fence may run the reports queued
/// before it at R's place (a folded fence runs their landings at once), while E is ahead. E's
/// reads of game memory for the buffer cache (uploads, binds, compares) test their pages: a
/// page with a report a posted fence lands, not yet played by R, waits for R to play it.
inline std::atomic<bool> g_report_landing_watch{};
void WaitReportLanding(u64 address, u64 size) noexcept;
/// FRITCHII_QUERY_FLUSH_DRY (helper DE), E: no page of [address, address + size) has a report
/// since the last fence, or one a posted fence lands that R has not played. The same pages
/// WaitReportLanding would wait for, without waiting. Not proof that R's query cache holds no
/// report there (the dry line counts where they disagree).
[[nodiscard]] bool ReportLandingClear(u64 address, u64 size) noexcept;
inline void BeforeEngineRead(u64 address, u64 size) noexcept {
    if (g_report_landing_watch.load(std::memory_order_relaxed) && Detail::t_engine) [[unlikely]] {
        WaitReportLanding(address, size);
    }
}

/// The buffer cache (E's) is used here: counted on R outside a lent call ("owner on R").
inline void EngineOwned(const char* where) noexcept {
    if (Detail::t_render) [[unlikely]] {
        CheckEngineOwned(where);
    }
}

} // namespace VideoCommon::OneFollower
