// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// Two answers to the read-back waits of a fight (work\helper_cu_answer.md, Rounds 2 and 3).
// Early compute stays a dry count (Round 3: chained inputs rule it out); the side copy has its
// verify and on modes.
//
// FRITCHII_EARLY_COMPUTE (read at first use, default 0; 1 = dry count; 2 and 3 not built, they
// count as 1): a dispatch whose output the guest CPU reads back (its
// program learned from the reads' writers, FRITCHII_READ_RUNS' ring) could run on a second queue
// at once instead of behind the ~10 submissions the host GPU has not run yet, if nothing pending
// on the main queue touches its bound ranges: no unrun GPU write over any of them (read after
// write, write after write) and no unrun use of the ranges it writes (write after read). The
// dry count evaluates each feeder dispatch where its buffers are bound
// (BufferCache::BindHostComputeBuffers, the back or the one follower's E), keeps the verdict in
// its write marks, and at each read of those bytes adds the read's "until the writer ran" time
// to the verdict's saving. Images are judged where the dispatch's views are filled
// (ComputePipeline::Configure, the back or R): a sampled or storage image whose last GPU write is
// not run yet. The buffer verdict and the image verdict are counted apart (with dispatches
// posted, the two run on different threads); the line says both.
//
// FRITCHII_READ_SIDE_COPY (read at first use, default 0; 1 = dry count; 2 = verify, 3 = on; the
// side queue itself needs the value at boot, Device::WantsSideQueue): a read's new download
// (EXP49) copied on a second queue (vk_side_queue.h) that waits only for the newest write over
// the copied bytes. 2 records both copies, lands the main one and compares (must be 0); 3 records
// the side copy alone and its reader lands it as soon as it is there. At RecordReadDownload the newest write mark over every copied
// range gives that write's scheduler tick: eligible when that tick is submitted already (a write
// in the open chunk keeps today's path). The reader's wait is cut at it: the part after it is the
// saving (less the side copy's own run).
//
// Both turn FRITCHII_READ_RUNS=2's ring and stamps on (read_run_stats.h ReadRunsValue).
namespace VideoCommon::EarlyCompute {

[[nodiscard]] inline int ReadMode(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] < '0' || value[0] > '9') {
        return 0;
    }
    return value[0] - '0';
}

/// A mode switch read at its first test (static initializers run before main() applies
/// nx2_switches.txt). Constant-initialized; racing first tests read the same value.
class FirstUseMode {
public:
    using Reader = int (*)();
    constexpr explicit FirstUseMode(Reader reader_) noexcept : reader{reader_} {}

    [[nodiscard]] int Get() const noexcept {
        int value = state.load(std::memory_order_relaxed);
        if (value < 0) [[unlikely]] {
            value = reader();
            state.store(value, std::memory_order_relaxed);
        }
        return value;
    }

    explicit operator bool() const noexcept {
        return Get() != 0;
    }

private:
    Reader reader;
    mutable std::atomic<int> state{-1};
};

// FRITCHII_FEEDER_LANE (read at first use, default 0; 1 = dry; helper_dl_answer.md item 2): would
// a feeder dispatch on a second (async compute) lane, started right after its real
// dependencies, finish sooner than on the main queue? At each feeder's bind (JudgeEarlyCompute)
// D is the newest unrun main-queue work it really depends on: the newest unrun write mark over
// its bound ranges (storage bindings narrowed to their read extent plus their write mark when the
// read extent is proven, else the whole binding), the newest unrun use of the ranges it writes
// (WAR), and its own upload's chunk when the bind uploads. A mark or use of an earlier judged
// feeder gives that feeder's own D, not its tick (compute chains stay on the lane, in order).
// When either ring has wrapped past the known tick, D is the feeder's own chunk. The write
// marks carry D; at a read that waits for such a writer the reader waits for D first, then the
// writer (same timeline, the total wait is unchanged): "after D until the writer ran" is the
// predicted saving. Images are not part of D (counted apart: a compute-family lane can't take
// them). Turns FRITCHII_EARLY_COMPUTE's dry count, its rings and FRITCHII_READ_RUNS=2's stamps on.
// =2..9 are not built (they count as 1). Lines every 10 s: "Feeder lane".
inline FirstUseMode g_lane{[] { return ReadMode("FRITCHII_FEEDER_LANE"); }};
inline FirstUseMode g_early{[] {
    const int early = ReadMode("FRITCHII_EARLY_COMPUTE");
    return early != 0 ? early : (g_lane ? 1 : 0);
}};
inline FirstUseMode g_side_copy{[] { return ReadMode("FRITCHII_READ_SIDE_COPY"); }};
/// FRITCHII_SUBMIT_TIMES (vk_submit_times.h): GPU timestamps per submission, for the cuts.
inline FirstUseMode g_submit_times{[] { return ReadMode("FRITCHII_SUBMIT_TIMES"); }};

/// The buffer verdict of a dispatch, kept in its write marks.
enum class Verdict : u8 {
    NotJudged,    ///< not a feeder when bound (or the switch off)
    Qualifies,    ///< nothing unrun on the main queue touches its bound buffer ranges
    PendingWrite, ///< an unrun GPU write over one of its bound ranges (RAW or WAW)
    PendingUse,   ///< an unrun use of a range it writes (WAR)
    Count,
};

/// Set around a dispatch's binds (BindHostComputeBuffers), read by its write marks.
inline thread_local Verdict t_verdict{Verdict::NotJudged};

/// FRITCHII_FEEDER_LANE: what set a feeder's D.
enum class LaneKind : u8 {
    None,        ///< nothing unrun it depends on: the lane could start it at once
    DrawBuffer,  ///< an unrun draw's storage or texture buffer write over its inputs
    ComputeMain, ///< an unrun earlier dispatch (not a judged feeder) on the main queue
    ComputeLane, ///< an earlier judged feeder: its own D (it would be on the lane too)
    Other,       ///< an unrun copy, query, transform feedback or indirect write
    Use,         ///< an unrun use of a range it writes (WAR)
    Upload,      ///< its own bind uploaded CPU data in its chunk
    Wrapped,     ///< a ring wrapped past the known tick: its own chunk (not eligible)
    Count,
};

[[nodiscard]] inline const char* LaneKindName(LaneKind kind) {
    switch (kind) {
    case LaneKind::None:
        return "nothing unrun";
    case LaneKind::DrawBuffer:
        return "draw-written buffer";
    case LaneKind::ComputeMain:
        return "earlier dispatch on the main queue";
    case LaneKind::ComputeLane:
        return "earlier feeder on the lane";
    case LaneKind::Other:
        return "copy/query/other";
    case LaneKind::Use:
        return "unrun use of its written range (WAR)";
    case LaneKind::Upload:
        return "its own upload";
    case LaneKind::Wrapped:
        return "ring wrapped";
    case LaneKind::Count:
        break;
    }
    return "?";
}

/// FRITCHII_FEEDER_LANE: a judged feeder's D, kept in its write marks and uses. `d` counts its
/// own upload; `d_copy` is D if the lane took the upload copy too. Stamps of the buffer cache's
/// clock (0: nothing unrun).
struct Lane {
    u64 d{};
    u64 d_copy{};
    LaneKind kind{LaneKind::None};
    LaneKind copy_kind{LaneKind::None};
    bool judged{};
};

/// Set around a judged feeder's binds (BindHostComputeBuffers), read by its marks and uses.
inline thread_local Lane t_lane{};

// ---- Uses: every bound buffer range (Buffer::MarkUsage), for the write-after-read test.

struct Use {
    u64 begin{};
    u64 end{};
    u64 stamp{}; ///< the buffer cache clock's stamp of the use
    Lane lane{}; ///< FRITCHII_FEEDER_LANE: the using feeder's D (judged ones)
};

inline constexpr size_t UseRingSize = 65536;

struct UseRing {
    std::mutex mutex;
    std::vector<Use> uses;
    u64 count{};
};

inline UseRing& GetUseRing() {
    static UseRing ring;
    return ring;
}

/// Buffer::MarkUsage (g_early only).
inline void NoteUse(u64 begin, u64 size, u64 stamp) {
    if (size == 0) {
        return;
    }
    UseRing& r = GetUseRing();
    std::scoped_lock lock{r.mutex};
    if (r.uses.empty()) {
        r.uses.resize(UseRingSize);
    }
    r.uses[r.count % UseRingSize] = Use{begin, begin + size, stamp, t_lane};
    ++r.count;
}

/// FRITCHII_FEEDER_LANE: the newest dependency in a ring scan (a write mark's or a use's), for
/// D with its own upload and D with the copy on the lane.
struct LaneDep {
    u64 d{};
    u64 d_copy{};
    LaneKind kind{LaneKind::None};
    LaneKind copy_kind{LaneKind::None};
    bool incomplete{}; ///< the ring wrapped before a run entry: not every dependency is known

    /// One unrun entry over the ranges: its stamp, or a judged feeder's own D
    void Add(u64 stamp, const Lane& lane, LaneKind main_kind) {
        const u64 strict = lane.judged ? lane.d : stamp;
        const u64 copy = lane.judged ? lane.d_copy : stamp;
        const LaneKind kind_of = lane.judged ? LaneKind::ComputeLane : main_kind;
        if (strict > d) {
            d = strict;
            kind = kind_of;
        }
        if (copy > d_copy) {
            d_copy = copy;
            copy_kind = kind_of;
        }
    }

    void Merge(const LaneDep& other) {
        if (other.d > d) {
            d = other.d;
            kind = other.kind;
        }
        if (other.d_copy > d_copy) {
            d_copy = other.d_copy;
            copy_kind = other.copy_kind;
        }
        incomplete |= other.incomplete;
    }
};

/// FRITCHII_FEEDER_LANE: the newest unrun use over any of `ranges` ({begin, end}), newest first,
/// stopping at the first run one.
template <typename Ranges, typename IsFree>
[[nodiscard]] LaneDep NewestPendingUse(const Ranges& ranges, IsFree&& is_free) {
    LaneDep dep{};
    UseRing& r = GetUseRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, UseRingSize);
    for (u64 i = 0; i < n; ++i) {
        const Use& use = r.uses[(r.count - 1 - i) % UseRingSize];
        if (is_free(use.stamp)) {
            return dep;
        }
        for (const auto& [begin, end] : ranges) {
            if (use.begin < end && begin < use.end) {
                dep.Add(use.stamp, use.lane, LaneKind::Use);
                break;
            }
        }
    }
    dep.incomplete = n == UseRingSize;
    return dep;
}

/// Whether an unrun use overlaps [begin, end): newest first, stopping at the first run one
/// (stamps grow with the ring, near enough for a count).
template <typename IsFree>
[[nodiscard]] bool PendingUse(u64 begin, u64 end, IsFree&& is_free) {
    UseRing& r = GetUseRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, UseRingSize);
    for (u64 i = 0; i < n; ++i) {
        const Use& use = r.uses[(r.count - 1 - i) % UseRingSize];
        if (is_free(use.stamp)) {
            return false;
        }
        if (use.begin < end && begin < use.end) {
            return true;
        }
    }
    return false;
}

// ---- Feeders: programs whose writes the guest reads back.

struct Feeders {
    std::mutex mutex;
    std::unordered_set<u64> programs;
    std::atomic<u64> count{};
};

inline Feeders& GetFeeders() {
    static Feeders feeders;
    return feeders;
}

inline void LearnFeeder(u64 program) {
    Feeders& f = GetFeeders();
    std::scoped_lock lock{f.mutex};
    if (f.programs.insert(program).second) {
        f.count.fetch_add(1, std::memory_order_relaxed);
    }
}

[[nodiscard]] inline bool IsFeeder(u64 program) {
    Feeders& f = GetFeeders();
    if (f.count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::scoped_lock lock{f.mutex};
    return f.programs.contains(program);
}

// ---- FRITCHII_FEEDER_LANE counts per feeder program, logged every 10 s (ReadRunStats::LogDry).

inline constexpr size_t LaneKinds = static_cast<size_t>(LaneKind::Count);

struct LaneProgram {
    // Binds (JudgeEarlyCompute and after the binds)
    u64 judged{};
    u64 bind_kinds[LaneKinds]{}; ///< what set D (with its own upload) at the bind
    u64 uploads{};               ///< binds that uploaded CPU data
    u64 storage_bindings{};      ///< storage bindings of those binds
    u64 read_extents{};          ///< of them narrowed to a proven read extent
    // Images (ComputePipeline::Configure): not part of D
    u64 configured{};
    u64 with_images{};
    u64 image_pending{}; ///< an image's last GPU write unrun at the configure
    // Reads that waited for a write of one of its judged dispatches (the area mark's)
    u64 reads{};
    u64 wait_ns{};
    u64 cut{};             ///< the wait was cut at the writer's tick (writer submitted)
    u64 eligible{};        ///< of them, D's tick before the writer's
    u64 saving_ns{};       ///< of those, from D having run until the writer ran
    u64 copy_eligible{};   ///< the same with D_copy (the lane takes the upload too)
    u64 copy_saving_ns{};
    u64 read_kinds[LaneKinds]{}; ///< what set D, over the cut reads
    u64 behind_sum{};      ///< eligible: submissions from D's to the writer's
    u64 d_after_writer{};  ///< must be 0: D's tick past the writer's
    u64 d_unsubmitted{};   ///< D not submitted when waited for (no saving counted)
};

struct LaneStats {
    std::mutex mutex;
    std::unordered_map<u64, LaneProgram> programs;
};

inline LaneStats& GetLaneStats() {
    static LaneStats stats;
    return stats;
}

/// BufferCache::BindHostComputeBuffers, after a judged feeder's binds.
inline void NoteLaneJudged(u64 program, const Lane& lane, bool uploaded, u32 storage_bindings,
                           u32 read_extents) {
    LaneStats& s = GetLaneStats();
    std::scoped_lock lock{s.mutex};
    LaneProgram& p = s.programs[program];
    ++p.judged;
    ++p.bind_kinds[static_cast<size_t>(lane.kind)];
    p.uploads += uploaded ? 1 : 0;
    p.storage_bindings += storage_bindings;
    p.read_extents += read_extents;
}

/// One read that waited for a judged feeder's write (ReadRunStats::NoteWriter).
struct LaneRead {
    u64 wait_ns{};
    bool cut{};
    u64 writer_ns{};    ///< from the wait's start until the writer ran
    u64 write_tick{};
    u64 lane_tick{};    ///< D's tick (0: nothing unrun at the bind)
    u64 lane_ns{};      ///< from the wait's start until D ran (0: D had run)
    u64 copy_tick{};
    u64 copy_ns{};
    bool unsubmitted{}; ///< D (or D_copy) was not submitted when waited for
    LaneKind kind{};
};

inline void NoteLaneRead(u64 program, const LaneRead& read) {
    LaneStats& s = GetLaneStats();
    std::scoped_lock lock{s.mutex};
    LaneProgram& p = s.programs[program];
    ++p.reads;
    p.wait_ns += read.wait_ns;
    if (read.lane_tick > read.write_tick || read.copy_tick > read.write_tick) {
        ++p.d_after_writer;
    }
    if (!read.cut) {
        return;
    }
    ++p.cut;
    ++p.read_kinds[static_cast<size_t>(read.kind)];
    if (read.unsubmitted) {
        ++p.d_unsubmitted;
        return;
    }
    if (read.lane_tick < read.write_tick) {
        ++p.eligible;
        p.saving_ns += read.writer_ns > read.lane_ns ? read.writer_ns - read.lane_ns : 0;
        p.behind_sum += read.write_tick - read.lane_tick;
    }
    if (read.copy_tick < read.write_tick) {
        ++p.copy_eligible;
        p.copy_saving_ns += read.writer_ns > read.copy_ns ? read.writer_ns - read.copy_ns : 0;
    }
}

// ---- Counts, logged with FRITCHII_READ_RUNS' lines every 600 frames.

struct ProgramImages {
    u64 dispatches{};    ///< feeder dispatches configured
    u64 with_images{};   ///< of them with a sampled or storage image
    u64 image_pending{}; ///< of them with an image whose last GPU write was not run yet
    u64 writes_images{}; ///< of them writing a storage image (its WAR is not tested)
};

struct Counts {
    std::mutex mutex;
    // Binds (the buffer verdict).
    u64 judged[static_cast<size_t>(Verdict::Count)]{};
    u64 uploads{}; ///< qualifying dispatches whose bind uploaded CPU data (moves with it)
    // Reads, per read kind (flushed, read download, fence download), by the writer's verdict.
    u64 reads[3][static_cast<size_t>(Verdict::Count)]{};
    u64 saving_ns[3]{}; ///< "until the writer ran" of the reads whose writer qualified
    u64 cut_reads[3]{}; ///< of them with that cut
    std::unordered_map<u64, ProgramImages> images;
    // FRITCHII_READ_SIDE_COPY dry: new downloads (EXP49) by their newest writer's tick.
    u64 side_downloads{};
    u64 side_eligible{};     ///< the writer's tick submitted when the download was recorded
    u64 side_open{};         ///< in the chunk the download closes: today's path is as good
    u64 side_no_mark{};      ///< no mark over the copied bytes in the ring
    u64 side_cuts{};         ///< reader waits cut at that tick
    u64 side_writer_ns{};    ///< until the writer's submission ran
    u64 side_saving_ns{};    ///< then until the download was there
    u64 side_saving_max_ns{};
    // FRITCHII_READ_SIDE_COPY >= 2.
    u64 side_submits{};        ///< side copies submitted
    u64 side_submit_failed{};  ///< failed (the main copy was recorded instead)
    u64 verify_downloads{};    ///< =2: downloads compared at the landing
    u64 verify_copies{};
    u64 verify_differing{};    ///< must be 0
    u64 verify_bytes{};
    u64 verify_waits{};        ///< =2: readers that waited for both copies
    u64 verify_side_ns{};      ///< until the side copy was there
    u64 verify_main_ns{};      ///< until the main copy was there (today's wait)
    u64 landed{};              ///< =3: reads landed from the side copy alone
    u64 landed_waits{};
    u64 landed_ns{};
    u64 landed_max_ns{};
    // The queues (BufferCache::TickFrame, from the runtime, before the line).
    u64 queue_raised{};       ///< side waits raised to keep their main ticks in order
    u64 queue_slow_waits{};   ///< host waits for a side value of 2 s or more
    u64 queue_not_queued{};   ///< refused: the writer's tick not given to vkQueueSubmit yet
    u64 queue_main_waits{};   ///< main submissions that waited for a side value
    u64 queue_unsubmitted{};  ///< must be 0: a main wait for a side value never submitted
};

inline Counts& Get() {
    static Counts counts;
    return counts;
}

inline void NoteJudged(Verdict verdict, bool uploaded) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.judged[static_cast<size_t>(verdict)];
    if (verdict == Verdict::Qualifies && uploaded) {
        ++c.uploads;
    }
}

/// ReadRunStats::NoteWriter: a read whose pages' newest write mark came from a judged dispatch.
inline void NoteRead(size_t kind, Verdict verdict, bool cut, u64 writer_wait_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.reads[kind][static_cast<size_t>(verdict)];
    if (verdict == Verdict::Qualifies && cut) {
        ++c.cut_reads[kind];
        c.saving_ns[kind] += writer_wait_ns;
    }
}

/// ComputePipeline::Configure, for a feeder program.
inline void NoteImages(u64 program, bool with_images, bool pending, bool writes_images) {
    {
        Counts& c = Get();
        std::scoped_lock lock{c.mutex};
        ProgramImages& p = c.images[program];
        ++p.dispatches;
        p.with_images += with_images ? 1 : 0;
        p.image_pending += pending ? 1 : 0;
        p.writes_images += writes_images ? 1 : 0;
    }
    if (g_lane) [[unlikely]] {
        LaneStats& s = GetLaneStats();
        std::scoped_lock lock{s.mutex};
        LaneProgram& p = s.programs[program];
        ++p.configured;
        p.with_images += with_images ? 1 : 0;
        p.image_pending += pending ? 1 : 0;
    }
}

/// FRITCHII_FEEDER_LANE: the "Feeder lane" lines (every 10 s, ReadRunStats::LogDry): a header,
/// then the feeders with the longest read-back waits.
inline void LogLane(double frames, double seconds) {
    LaneStats& s = GetLaneStats();
    std::unordered_map<u64, LaneProgram> programs;
    {
        std::scoped_lock lock{s.mutex};
        programs.swap(s.programs);
    }
    const auto ms_frame = [frames](u64 ns) { return static_cast<double>(ns) / 1e6 / frames; };
    const auto ms_each = [](u64 ns, u64 n) {
        return n != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(n) : 0.0;
    };
    const auto share = [](u64 part, u64 whole) {
        return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    LaneProgram total{};
    for (const auto& [program, p] : programs) {
        total.judged += p.judged;
        total.reads += p.reads;
        total.wait_ns += p.wait_ns;
        total.cut += p.cut;
        total.eligible += p.eligible;
        total.saving_ns += p.saving_ns;
        total.copy_eligible += p.copy_eligible;
        total.copy_saving_ns += p.copy_saving_ns;
        total.d_after_writer += p.d_after_writer;
        total.d_unsubmitted += p.d_unsubmitted;
        total.uploads += p.uploads;
        total.storage_bindings += p.storage_bindings;
        total.read_extents += p.read_extents;
        for (size_t k = 0; k < LaneKinds; ++k) {
            total.bind_kinds[k] += p.bind_kinds[k];
        }
    }
    LOG_INFO(HW_GPU,
             "Feeder lane (FRITCHII_FEEDER_LANE={}, dry) last {:.0f} s: {} feeder programs, {} "
             "dispatches judged ({:.0f} a frame; {} uploaded; storage bindings narrowed to a "
             "proven read extent {} of {}) | D at the bind: nothing unrun {}, draw-written buffer "
             "{}, earlier dispatch on the main queue {}, earlier feeder on the lane {}, "
             "copy/query/other {}, unrun use of its written range {}, its own upload {}, ring "
             "wrapped {} | reads that waited for them {} ({:.2f} ms a frame), cut at the writer's "
             "tick {} | eligible (D before the writer's submission) {} = {:.0f}% of the cut "
             "reads | predicted saving {:.2f} ms a frame ({:.3f} a read), with the upload on the "
             "lane too {:.2f} ms a frame ({} eligible) | D not submitted when waited {} | must "
             "be 0: D after the writer {}",
             g_lane.Get(), seconds, programs.size(), total.judged,
             static_cast<double>(total.judged) / frames, total.uploads, total.read_extents,
             total.storage_bindings, total.bind_kinds[0], total.bind_kinds[1],
             total.bind_kinds[2], total.bind_kinds[3], total.bind_kinds[4], total.bind_kinds[5],
             total.bind_kinds[6], total.bind_kinds[7], total.reads, ms_frame(total.wait_ns),
             total.cut, total.eligible, share(total.eligible, total.cut),
             ms_frame(total.saving_ns), ms_each(total.saving_ns, total.eligible),
             ms_frame(total.copy_saving_ns), total.copy_eligible, total.d_unsubmitted,
             total.d_after_writer);
    std::vector<std::pair<u64, const LaneProgram*>> order;
    order.reserve(programs.size());
    for (const auto& [program, p] : programs) {
        if (p.reads != 0) {
            order.emplace_back(program, &p);
        }
    }
    const size_t shown = (std::min<size_t>)(6, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                      order.end(), [](const auto& a, const auto& b) {
                          return a.second->wait_ns > b.second->wait_ns;
                      });
    for (size_t i = 0; i < shown; ++i) {
        const LaneProgram& p = *order[i].second;
        LOG_INFO(HW_GPU,
                 "Feeder lane {}/{}: program {:016x} | reads waited {} ({:.2f} ms a frame), cut "
                 "{} | eligible share f {:.0f}% ({} of the cut reads), predicted saving {:.2f} ms "
                 "a frame ({:.3f} a read), with the upload on the lane {:.2f} ms a frame | D set "
                 "by (cut reads): nothing unrun {}, draw-written buffer {}, earlier dispatch on "
                 "the main queue {}, earlier feeder on the lane {}, copy/query/other {}, unrun use "
                 "of its written range {}, its own upload {}, ring wrapped {} | D {:.1f} "
                 "submissions before the writer | binds {} (uploaded {}, read extents {} of {}) | "
                 "images (not in D): configured {}, with images {}, an image's last write unrun {} "
                 "| D not submitted {}, must be 0: D after the writer {}",
                 i + 1, shown, order[i].first, p.reads, ms_frame(p.wait_ns), p.cut,
                 share(p.eligible, p.cut), p.eligible, ms_frame(p.saving_ns),
                 ms_each(p.saving_ns, p.eligible), ms_frame(p.copy_saving_ns), p.read_kinds[0],
                 p.read_kinds[1], p.read_kinds[2], p.read_kinds[3], p.read_kinds[4],
                 p.read_kinds[5], p.read_kinds[6], p.read_kinds[7],
                 p.eligible != 0
                     ? static_cast<double>(p.behind_sum) / static_cast<double>(p.eligible)
                     : 0.0,
                 p.judged, p.uploads, p.read_extents, p.storage_bindings, p.configured,
                 p.with_images, p.image_pending, p.d_unsubmitted, p.d_after_writer);
    }
}

/// BufferCache::RecordReadDownload: what the newest write over a new download's bytes was.
enum class SideWriter : u8 { NoMark, Open, Submitted };
inline void NoteSideDownload(SideWriter writer) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.side_downloads;
    c.side_eligible += writer == SideWriter::Submitted ? 1 : 0;
    c.side_open += writer == SideWriter::Open ? 1 : 0;
    c.side_no_mark += writer == SideWriter::NoMark ? 1 : 0;
}

/// BufferCache::FinishReadDownloads: a reader's wait for an eligible download, cut there.
inline void NoteSideCut(u64 writer_ns, u64 rest_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.side_cuts;
    c.side_writer_ns += writer_ns;
    c.side_saving_ns += rest_ns;
    c.side_saving_max_ns = (std::max)(c.side_saving_max_ns, rest_ns);
}

/// BufferCache::RecordReadDownload: a side copy submitted (or failed).
inline void NoteSideSubmit(bool submitted) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++(submitted ? c.side_submits : c.side_submit_failed);
}

/// BufferCache::LandReadDownload (=2): the side copy's bytes against the main copy's.
inline void NoteSideVerify(u64 copies, u64 differing, u64 bytes) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.verify_downloads;
    c.verify_copies += copies;
    c.verify_differing += differing;
    c.verify_bytes += bytes;
    if (differing != 0) {
        LOG_ERROR(HW_GPU, "Read side copy (FRITCHII_READ_SIDE_COPY=2): {} of {} copies differ "
                          "from the main queue's (must be 0)",
                  differing, copies);
    }
}

/// BufferCache::FinishReadDownloads (=2): a reader's wait for the side copy, then for both.
inline void NoteSideVerifyWait(u64 side_ns, u64 total_ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.verify_waits;
    c.verify_side_ns += side_ns;
    c.verify_main_ns += total_ns;
}

/// BufferCache::FinishReadDownloads (=3): a read landed from its side copy (`ns` waited, 0: it
/// was there).
inline void NoteSideLanded(u64 ns) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    ++c.landed;
    if (ns != 0) {
        ++c.landed_waits;
        c.landed_ns += ns;
        c.landed_max_ns = (std::max)(c.landed_max_ns, ns);
    }
}

/// BufferCache::TickFrame, right before the line: the queues' counts since the last line.
inline void NoteQueueStats(u64 raised, u64 slow_waits, u64 not_queued, u64 main_waits,
                           u64 unsubmitted) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    c.queue_raised += raised;
    c.queue_slow_waits += slow_waits;
    c.queue_not_queued += not_queued;
    c.queue_main_waits += main_waits;
    c.queue_unsubmitted += unsubmitted;
}

/// ReadRunStats::Log, every 600 frames.
inline void Log(double frames) {
    Counts& c = Get();
    std::scoped_lock lock{c.mutex};
    constexpr auto Q = static_cast<size_t>(Verdict::Qualifies);
    constexpr auto W = static_cast<size_t>(Verdict::PendingWrite);
    constexpr auto U = static_cast<size_t>(Verdict::PendingUse);
    constexpr auto N = static_cast<size_t>(Verdict::NotJudged);
    if (g_early) {
        u64 dispatches = 0, with_images = 0, image_pending = 0, writes_images = 0;
        for (const auto& [program, p] : c.images) {
            dispatches += p.dispatches;
            with_images += p.with_images;
            image_pending += p.image_pending;
            writes_images += p.writes_images;
        }
        const auto ms_frame = [frames](u64 ns) { return static_cast<double>(ns) / 1e6 / frames; };
        const auto ms_read = [](u64 ns, u64 n) {
            return n != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(n) : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Early compute (FRITCHII_EARLY_COMPUTE={}, dry) last 600 frames: {} feeder "
                 "programs | feeder dispatches' buffers: qualify {} ({} uploaded CPU data), an "
                 "unrun write over a bound range {}, an unrun use of a written range {} | their "
                 "images (configured {}): with images {}, an image's last write unrun {}, writing "
                 "a storage image {} | reads by the writer's buffer verdict as qualify/unrun "
                 "write/unrun use/not judged: flushed {}/{}/{}/{}, fence download {}/{}/{}/{} | "
                 "predicted saving (the qualifying reads' 'until the writer ran'): flushed {:.2f} "
                 "ms a frame ({:.3f} a read), fence download {:.2f} ({:.3f})",
                 g_early.Get(), GetFeeders().count.load(std::memory_order_relaxed), c.judged[Q],
                 c.uploads, c.judged[W], c.judged[U], dispatches, with_images, image_pending,
                 writes_images, c.reads[0][Q], c.reads[0][W], c.reads[0][U], c.reads[0][N],
                 c.reads[2][Q], c.reads[2][W], c.reads[2][U], c.reads[2][N],
                 ms_frame(c.saving_ns[0]), ms_read(c.saving_ns[0], c.cut_reads[0]),
                 ms_frame(c.saving_ns[2]), ms_read(c.saving_ns[2], c.cut_reads[2]));
        // The feeders with the most dispatches: whether images rule them out.
        std::vector<std::pair<u64, ProgramImages>> order(c.images.begin(), c.images.end());
        const size_t shown = (std::min<size_t>)(6, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                          order.end(), [](const auto& a, const auto& b) {
                              return a.second.dispatches > b.second.dispatches;
                          });
        for (size_t i = 0; i < shown; ++i) {
            const ProgramImages& p = order[i].second;
            LOG_INFO(HW_GPU,
                     "Early compute feeder {}/{}: program {:016x}, {} dispatches, with images {}, "
                     "an image's last write unrun {}, writing a storage image {}",
                     i + 1, shown, order[i].first, p.dispatches, p.with_images, p.image_pending,
                     p.writes_images);
        }
    }
    if (g_side_copy) {
        const u64 cuts = c.side_cuts;
        LOG_INFO(HW_GPU,
                 "Read side copy (FRITCHII_READ_SIDE_COPY={}, dry) last 600 frames: {} new "
                 "downloads; newest write over their bytes: submitted already {} (eligible), in "
                 "the chunk the download closes {}, no mark in the ring {} | eligible reader waits "
                 "cut there {}: until the writer ran {:.3f} ms each, then (the saving) {:.3f} ms "
                 "each, longest {:.2f}, {:.2f} ms a frame",
                 g_side_copy.Get(), c.side_downloads, c.side_eligible, c.side_open, c.side_no_mark,
                 cuts,
                 cuts != 0 ? static_cast<double>(c.side_writer_ns) / 1e6 / static_cast<double>(cuts)
                           : 0.0,
                 cuts != 0 ? static_cast<double>(c.side_saving_ns) / 1e6 / static_cast<double>(cuts)
                           : 0.0,
                 static_cast<double>(c.side_saving_max_ns) / 1e6,
                 static_cast<double>(c.side_saving_ns) / 1e6 / frames);
        if (g_side_copy.Get() >= 2) {
            const auto per = [](u64 ns, u64 n) {
                return n != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(n) : 0.0;
            };
            LOG_INFO(HW_GPU,
                     "Read side copy (FRITCHII_READ_SIDE_COPY={}) last 600 frames: {} side copies "
                     "submitted, {} failed | verify: {} downloads compared ({} copies, {} KiB), {} "
                     "copies differ (must be 0); {} readers waited for both: side copy there "
                     "after {:.3f} ms, main copy after {:.3f} ms | on: {} reads landed from the "
                     "side copy, {} waited {:.3f} ms each (longest {:.2f}), {:.2f} ms a frame",
                     g_side_copy.Get(), c.side_submits, c.side_submit_failed, c.verify_downloads,
                     c.verify_copies, c.verify_bytes >> 10, c.verify_differing, c.verify_waits,
                     per(c.verify_side_ns, c.verify_waits), per(c.verify_main_ns, c.verify_waits),
                     c.landed, c.landed_waits, per(c.landed_ns, c.landed_waits),
                     static_cast<double>(c.landed_max_ns) / 1e6,
                     static_cast<double>(c.landed_ns) / 1e6 / frames);
            LOG_INFO(HW_GPU,
                     "Read side queue (FRITCHII_READ_SIDE_COPY={}) last 600 frames: main "
                     "submissions that waited for a side copy {}, waits for a side value never "
                     "submitted {} (must be 0, dropped) | side copies refused, writer's tick not "
                     "given to the queue yet {} (copied on the main queue) | side waits raised to "
                     "keep the order {} | host waits for a side value over 2 s {} (must be 0)",
                     g_side_copy.Get(), c.queue_main_waits, c.queue_unsubmitted,
                     c.queue_not_queued, c.queue_raised, c.queue_slow_waits);
        }
    }
    std::fill(std::begin(c.judged), std::end(c.judged), 0);
    c.uploads = 0;
    for (auto& kind : c.reads) {
        std::fill(std::begin(kind), std::end(kind), 0);
    }
    std::fill(std::begin(c.saving_ns), std::end(c.saving_ns), 0);
    std::fill(std::begin(c.cut_reads), std::end(c.cut_reads), 0);
    c.images.clear();
    c.side_downloads = c.side_eligible = c.side_open = c.side_no_mark = 0;
    c.side_cuts = c.side_writer_ns = c.side_saving_ns = c.side_saving_max_ns = 0;
    c.side_submits = c.side_submit_failed = 0;
    c.verify_downloads = c.verify_copies = c.verify_differing = c.verify_bytes = 0;
    c.verify_waits = c.verify_side_ns = c.verify_main_ns = 0;
    c.landed = c.landed_waits = c.landed_ns = c.landed_max_ns = 0;
    c.queue_raised = c.queue_slow_waits = c.queue_not_queued = 0;
    c.queue_main_waits = c.queue_unsubmitted = 0;
}

} // namespace VideoCommon::EarlyCompute
