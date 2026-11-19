// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "video_core/nx2_debug_control.h"

/// FRITCHII_FRONT_SPLIT_COUNT (live, default 0): step FS0 of the front split
/// (work\FRONT_SPLIT_DESIGN.md 4.3), dry. Nothing here changes what the front sends or when.
///
/// 1 = 1 list entry (DmaPusher::Step call) in 64, picked at random, is timed with time stamp
///     ticks and split by stage: parser (Step's bookkeeping, re-reads, entry checks, acquires,
///     header decode: everything below), 3D state (Maxwell3D::FrontCallMethod and
///     FrontCallMultiMethod, the macros included), compute (KeplerCompute's front calls) and
///     ring sends (FrontWriteSend's entry, the list's Method / MultiMethod entries). Each stage
///     is exclusive; the waits for the back inside a timed entry (GpuSplit::WaitFor and Drain,
///     their NoteSiteWait) are taken out of the stage they happened in. Scaled by entries /
///     timed entries. The same time per channel (ChannelState::bind_id).
///     Counted on every entry: headers (2a's chunks), list words, suspensions; the joins a
///     parse thread (2a) would need, by cause: an acquire check with engine methods decoded
///     since the last join, a list writer check with a writer sent since, a GPU-written list
///     that may need macro state (TakeListDraw, TakeListDispatch, FrontMacroCountsDraw), Step's
///     own Run; Step's Defers (encoded into 2a's queue, no join); the raw ring words 2c would
///     send for 3D and compute methods (a Method entry a word, a Multi entry 2 + words / 2)
///     against the ring words the front sent; the macro DMA reads (0xe2c apart) by answer;
///     list entries copied again by a second or later refresh before they were parsed.
///
/// Line every 10 s: "Front split dry: ...".
///
/// Off: per Step one relaxed load and a compare with a plain mirror; per hooked call (stage
/// boundaries, engine calls, writer notes, acquire checks, macro DMA reads, waits) one plain
/// load of g_state.
///
/// FRITCHII_ONEF_NULL_FILL=1 (test only, the picture breaks): E skips FlushCaching, FillRing and
/// PushDraw for every draw it would fill (no packet, R draws nothing); the rest as today. A line
/// every 10 s: draws null-filled. Off: one relaxed load per filled draw.
namespace VideoCommon::FrontSplit {

enum class Stage : u8 {
    Parser,   ///< Step, entry checks, acquires, header decode: the part 2a's P would run
    State3D,  ///< below FrontCallMethod / FrontCallMultiMethod (FrontWrite, macros, notes)
    Compute,  ///< below KeplerCompute's FrontCallMethod / FrontCallMultiMethod
    RingSend, ///< the ring entries themselves (FrontWriteSend, Method, MultiMethod)
    Count,
};

enum class JoinCause : u8 {
    Acquire,     ///< an acquire check with engine methods decoded since the last join
    WriterCheck, ///< a list writer check with a writer sent since the last join
    GpuList,     ///< a GPU-written list that may need macro state (list draw / dispatch)
    StepRun,     ///< Step's own Run (the GP sync fence)
    Count,
};

/// The answer a macro DMA read (Maxwell3D::FrontDmaRead) took: what 2c's E1 would join for.
enum class DmaAnswer : u8 {
    Forwarded,    ///< from a queued constant buffer upload (ForwardRead)
    SnapshotHit,  ///< the page snapshot was current
    NotedOverlap, ///< drained: a noted write since the snapshot over the bytes
    Writer,       ///< drained: compute, copy or an unranged writer since the snapshot
    NewPage,      ///< drained: no snapshot of the page (or one still pending)
    ReportPage,   ///< drained: a report listed over the bytes
    Drained,      ///< drained: no snapshot path (another macro, mode, a split range)
    Deferred,     ///< FRITCHII_E2C_ON_E: E reads it at its place
    Count,
};

inline constexpr u8 StateOn = 1;       ///< FRITCHII_FRONT_SPLIT_COUNT >= 1
inline constexpr u8 StateSampling = 2; ///< the front is inside a timed entry
/// Written by the front only (SetLevel, Begin, End). Every hook tests it first.
inline u8 g_state{};
/// The switch's value the state follows (a plain mirror: the hot test is a compare).
inline int g_level{};

/// Counted on every entry while on (front only).
struct Live {
    u64 entries{};
    u64 headers{};
    u64 words{};
    u64 suspensions{};
    u64 engine_calls{};
    u64 raw_ring_words{};
    u64 step_defers{};
    u64 reread_twice{};
    std::array<u64, static_cast<size_t>(JoinCause::Count)> joins{};
    std::array<u64, static_cast<size_t>(DmaAnswer::Count)> e2c{};
    std::array<u64, static_cast<size_t>(DmaAnswer::Count)> other_reads{};
    /// 2a's model of M's queue: engine methods decoded since the last join, a writer among them.
    bool backlog{};
    bool writer_backlog{};
};
inline Live g_live{};

void SetLevel(int level) noexcept;
/// A Step call while on: counts it, picks it for timing (true), writes the line when due.
[[nodiscard]] bool Begin(s32 channel) noexcept;
void End() noexcept;
[[nodiscard]] bool Push(Stage stage) noexcept;
void Pop() noexcept;
void NoteWaitSampled(u64 ns) noexcept;

/// DmaPusher::Step, for the whole call.
class EntryScope {
public:
    explicit EntryScope(s32 channel) noexcept {
        const int level =
            VideoCore::Nx2Debug::g_front_split_count.load(std::memory_order_relaxed);
        if (level != g_level) [[unlikely]] {
            SetLevel(level);
        }
        if (g_state != 0) [[unlikely]] {
            timed = Begin(channel);
        }
    }
    ~EntryScope() {
        if (timed) [[unlikely]] {
            End();
        }
    }
    EntryScope(const EntryScope&) = delete;
    EntryScope& operator=(const EntryScope&) = delete;

private:
    bool timed{};
};

/// A stage boundary: its time (inside a timed entry) goes to `stage` until it closes.
class StageScope {
public:
    explicit StageScope(Stage stage) noexcept {
        if ((g_state & StateSampling) != 0) [[unlikely]] {
            pushed = Push(stage);
        }
    }
    ~StageScope() {
        Close();
    }
    /// Ends the stage before the scope does.
    void Close() noexcept {
        if (pushed) [[unlikely]] {
            Pop();
            pushed = false;
        }
    }
    StageScope(const StageScope&) = delete;
    StageScope& operator=(const StageScope&) = delete;

private:
    bool pushed{};
};

/// GpuSplit's NoteSiteWait (front): a wait for the back of `ns` just ended.
inline void NoteWait(u64 ns) noexcept {
    if ((g_state & StateSampling) != 0) [[unlikely]] {
        NoteWaitSampled(ns);
    }
}

/// DmaPusher::ProcessCommands: headers decoded and words run.
inline void NoteParsed(u64 headers, u64 words) noexcept {
    if (g_state != 0) [[unlikely]] {
        g_live.headers += headers;
        g_live.words += words;
    }
}

inline void NoteSuspension() noexcept {
    if (g_state != 0) [[unlikely]] {
        ++g_live.suspensions;
    }
}

/// DmaPusher::CallMethod / CallMultiMethod, a method to an engine (with the split active):
/// it would sit in 2a's queue. `raw_words`: the ring words 2c would send for it (3D and compute
/// only, else 0).
inline void NoteEngineCall(u32 raw_words) noexcept {
    if (g_state != 0) [[unlikely]] {
        ++g_live.engine_calls;
        g_live.raw_ring_words += raw_words;
        g_live.backlog = true;
    }
}

/// GpuSplit's writer notes (NoteWriter, NoteMemoryWrite, NoteMemoryWriteData): a writer sent.
inline void NoteWriterSent() noexcept {
    if (g_state != 0) [[unlikely]] {
        g_live.writer_backlog = true;
        g_live.backlog = true;
    }
}

inline void Join(JoinCause cause) noexcept {
    ++g_live.joins[static_cast<size_t>(cause)];
    g_live.backlog = false;
    g_live.writer_backlog = false;
}

/// Puller: an acquire check that reads the front's records (first check, poll).
inline void AcquireCheck() noexcept {
    if (g_state != 0 && g_live.backlog) [[unlikely]] {
        Join(JoinCause::Acquire);
    }
}

/// DmaPusher::Step: the list writer checks of an entry begin.
inline void WriterCheck() noexcept {
    if (g_state != 0 && g_live.writer_backlog) [[unlikely]] {
        Join(JoinCause::WriterCheck);
    }
}

/// DmaPusher::Step: a GPU-written list on the list draw / dispatch path.
inline void GpuListCheck() noexcept {
    if (g_state != 0 && g_live.backlog) [[unlikely]] {
        Join(JoinCause::GpuList);
    }
}

/// DmaPusher::Step: one of its own Defers (2a's P encodes it into the queue: no join).
inline void StepDefer() noexcept {
    if (g_state != 0) [[unlikely]] {
        ++g_live.step_defers;
    }
}

/// DmaPusher::Step: its own Run (waits for the back: a join whatever the queue holds).
inline void StepRun() noexcept {
    if (g_state != 0) [[unlikely]] {
        Join(JoinCause::StepRun);
    }
}

/// Maxwell3D::FrontDmaRead: the answer a read took (`e2c`: macro 0xe2c's).
inline void NoteDmaRead(DmaAnswer answer, bool e2c) noexcept {
    if (g_state != 0) [[unlikely]] {
        ++(e2c ? g_live.e2c : g_live.other_reads)[static_cast<size_t>(answer)];
    }
}

/// DmaPusher::Step's refresh: `entries` not yet parsed copied again by a second or later
/// refresh of their list.
inline void NoteRereadTwice(u64 entries) noexcept {
    g_live.reread_twice += entries;
}

// --- FRITCHII_ONEF_NULL_FILL (E) -----------------------------------------------------------------

void NoteNullFill() noexcept;

/// OneFollowerDraw::FillAndPush: true when E skips the draw's flush, fill and push.
[[nodiscard]] inline bool NullFill() noexcept {
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_null_fill,
                                         "FRITCHII_ONEF_NULL_FILL") != 1) [[likely]] {
        return false;
    }
    NoteNullFill();
    return true;
}

} // namespace VideoCommon::FrontSplit
