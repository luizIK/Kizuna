// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/storage_write_extent.h"
#include "video_core/early_compute.h"

// FRITCHII_READ_RUNS=1 (diagnostic, read once; fable52_answer_fountain 4): the guest reads of
// GPU-written memory that go to the back or wait for a download (GPU::OnCPURead past its
// preemptive answer), one line per 600 frames: whether each read is on the page after the
// previous one, how long the GPU-written run around its page is (BufferCache::GetFlushArea,
// from gpu_modified_ranges inside the read's buffer), whether its bytes were already on their
// way, how long the reader waited, and how many distinct pages a frame reads. It also counts
// EXP112's run flushes. Off: one test of g_enabled at each site.
// A second line, "Read overlap", says per read whether its own bytes [address, address + size)
// overlap the GPU-written bytes it waits for, or only share their page(s) (see ByteCheck).
// "Read writers" and "Read writer" lines (fable52_answer_readback 3): which dispatch wrote the
// bytes each read waits for. Every GPU write mark (BufferCache::MarkWrittenBuffer) goes into a
// ring with the writer the bind set (t_writer: the compute program's hash and binding, or a
// draw's stage and binding); at a read, GetFlushArea takes the newest mark over the read's bytes
// (FindWriter) and OnCPURead counts it per writer (NoteWriter). The "Read runs for B1" line
// sums what the GPU writes into the runs read in the last 16 frames.
// FRITCHII_READ_RUNS=2 adds the "Read writer ticks" line (helper_cu_answer.md): per read kind,
// the newest write mark over the read's pages (the bytes its download takes), where the host GPU
// was with that write's submission when the read asked (not submitted, submitted and not run,
// run), how many submissions the read's own tick is behind it, and, for every read that waits
// for the host GPU, the wait cut at the writer's tick: first until the writer's submission ran,
// then the rest. Same timeline, so the total wait is what it was; only the cut is new.
namespace VideoCommon::ReadRunStats {

/// A boot switch read at its first test, not by a static initializer: those run before main()
/// applies nx2_switches.txt, so the file's value was never seen (only a real environment
/// variable was). Constant-initialized; the first tests may race, they read the same value.
class FirstUseSwitch {
public:
    using Reader = bool (*)();
    constexpr explicit FirstUseSwitch(Reader reader_) noexcept : reader{reader_} {}

    operator bool() const noexcept {
        int value = state.load(std::memory_order_relaxed);
        if (value < 0) [[unlikely]] {
            value = reader() ? 1 : 0;
            state.store(value, std::memory_order_relaxed);
        }
        return value != 0;
    }

private:
    Reader reader;
    mutable std::atomic<int> state{-1};
};

// FRITCHII_STORE_FOOTPRINT (read at first use, default 0; helper_dl_answer.md item 1): any value
// at boot arms it (this switch); the live value picks 0, 1, 2 or 3 (store_footprint_live.h). 3 at
// boot also turns the read ring on for the dry count below: every native written storage
// binding's write mark carries its program's store footprint (the residues modulo a
// record stride its stores can land on, made at translation, so EDEN_NX2_PRECOMPILED_FAST=0).
// Each read that goes to the back is classed by the footprint of the newest mark over its bytes:
// inside it, outside it with no older mark in the ring still covering the bytes (it would be
// answered from guest memory), outside it but under an older mark, or a mark without a
// footprint. Waiting and marking are unchanged. Lines every 10 s: "Store footprint".
inline EarlyCompute::FirstUseMode g_store_footprint{
    [] { return EarlyCompute::ReadMode("FRITCHII_STORE_FOOTPRINT"); }};

inline bool ReadRunsValue(bool writer_ticks) {
    // FRITCHII_EARLY_COMPUTE and FRITCHII_READ_SIDE_COPY count on this ring and its stamps
    // (FRITCHII_FEEDER_LANE turns FRITCHII_EARLY_COMPUTE on).
    if (EarlyCompute::g_early || EarlyCompute::g_side_copy || EarlyCompute::g_submit_times) {
        return true;
    }
    // FRITCHII_EXTENT_CBUF=3 (dry) counts its reads on the ring too ("Extent cbuf" lines), and
    // FRITCHII_EXTENT_CENSUS ranks program bindings by their read-back waits.
    if (!writer_ticks) {
        // FRITCHII_STORE_FOOTPRINT: the dry count's ring only for 3 at boot (1 and 2 at boot arm
        // the live switch without it)
        if (g_store_footprint.Get() == 3) {
            return true;
        }
        const char* cbuf = std::getenv("FRITCHII_EXTENT_CBUF");
        if (cbuf != nullptr && cbuf[0] == '3') {
            return true;
        }
        const char* census = std::getenv("FRITCHII_EXTENT_CENSUS");
        if (census != nullptr && census[0] != '\0' && census[0] != '0') {
            return true;
        }
    }
    // FRITCHII_EXTENT_SHARED=3: the same dry count for the extents proven through shared memory
    // slots, with the write marks' ticks too (as FRITCHII_READ_RUNS=2, counter only): a read past
    // the proof is timed until the newest mark the proof leaves over its bytes ran
    const char* shared = std::getenv("FRITCHII_EXTENT_SHARED");
    if (shared != nullptr && shared[0] == '3') {
        return true;
    }
    // FRITCHII_EXTENT_GUARD=3: the same for the extents proven with the stores' path facts
    const char* guard = std::getenv("FRITCHII_EXTENT_GUARD");
    if (guard != nullptr && guard[0] == '3') {
        return true;
    }
    const char* value = std::getenv("FRITCHII_READ_RUNS");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') {
        return false;
    }
    return !writer_ticks || value[0] == '2';
}

inline FirstUseSwitch g_enabled{[] { return ReadRunsValue(false); }};

/// FRITCHII_READ_RUNS=2: the write marks carry their stamp and reads cut their host waits there.
inline FirstUseSwitch g_writer_ticks{[] { return ReadRunsValue(true); }};

/// Pages the run is counted up to (one more than the histogram's last bucket).
inline constexpr u64 RunCountPages = 65;

/// Set by BufferCache::GetFlushArea on the reading thread: the GPU-written pages in a row around
/// the read's pages inside its buffer (capped at RunCountPages); 0 when the buffer cache had no
/// GPU-written byte there (an image, or nothing). GPU::OnCPURead clears it before asking.
inline thread_local u64 t_run_pages = 0;

enum class Kind : u8 {
    Flushed,       ///< a new download through the back (one round trip)
    InFlight,      ///< its bytes were in a read download on its way: only waited
    FenceDownload, ///< its bytes were in a fence download on its way (EXP72): only waited
};

/// The read's own bytes against GPU-written byte ranges inside the read's pages.
enum class Overlap : u8 {
    Unknown,  ///< not answered from the buffer cache's ranges (an image, or no range there)
    Bytes,    ///< at least one of the read's bytes is GPU-written
    PageOnly, ///< none is: the read only shares the page(s) with GPU-written bytes
};

/// One answer: what the read overlaps and, for PageOnly, the bytes between the read and the
/// nearest GPU-written byte in its pages (0 = adjacent).
struct ByteAnswer {
    Overlap overlap{Overlap::Unknown};
    u64 gap{};
};

/// Fed the GPU-written ranges inside the read's pages (clipped to them), one by one.
struct ByteCheck {
    u64 begin{}; ///< the read's first byte
    u64 end{};   ///< one past its last byte
    bool any{};
    bool overlap{};
    u64 gap{std::numeric_limits<u64>::max()};

    void Add(u64 lower, u64 upper) {
        if (lower >= upper) {
            return;
        }
        any = true;
        if (lower < end && begin < upper) {
            overlap = true;
        } else {
            gap = (std::min)(gap, upper <= begin ? begin - upper : lower - end);
        }
    }

    ByteAnswer Answer() const {
        if (!any) {
            return {};
        }
        return overlap ? ByteAnswer{Overlap::Bytes, 0} : ByteAnswer{Overlap::PageOnly, gap};
    }
};

/// Set by BufferCache::GetFlushArea on the reading thread (its mutex held), cleared by
/// GPU::OnCPURead before asking. `modified`: against gpu_modified_ranges at the read (what a
/// flush downloads). `on_way`: against the download already on its way that the read waits for:
/// pending_download_ranges up to the newest one the read needs (EXP72's fence download), or
/// read_download_ranges (EXP49's read download, GetFlushArea's in_flight answer).
struct ReadBytes {
    ByteAnswer modified;
    ByteAnswer on_way;
};
inline thread_local ReadBytes t_read_bytes{};

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// What made a GPU write mark.
enum class WriterKind : u8 {
    Other,          ///< no bind set one: copies, queries, transform feedback, indirect buffers
    ComputeStorage, ///< a dispatch's storage buffer
    ComputeTexture, ///< a dispatch's texture or image buffer
    DrawStorage,    ///< a draw's storage buffer
    DrawTexture,    ///< a draw's texture or image buffer
};

struct Writer {
    u64 program{};  ///< the compute program's hash (0 for draws and Other)
    u32 binding{};  ///< the binding index; draws: stage * 64 + index
    WriterKind kind{WriterKind::Other};

    bool operator==(const Writer&) const = default;
};

/// Set on the thread that binds a dispatch's or draw's written buffers while it marks them
/// (BufferCache::BindHost*StorageBuffers and *TextureBuffers), Other otherwise.
inline thread_local Writer t_writer{};

/// Dispatches bound so far (BufferCache::BindHostComputeBuffers): a write mark's ring position.
inline std::atomic<u64> g_dispatches{};

struct WriteMark {
    u64 begin{};
    u64 end{};
    u64 time_ns{};
    u64 dispatch{}; ///< g_dispatches when it was made
    u64 stamp{};    ///< the buffer cache's clock stamp then (g_writer_ticks, else 0)
    Writer writer{};
    /// FRITCHII_EARLY_COMPUTE: the writing dispatch's buffer verdict at its bind
    EarlyCompute::Verdict verdict{EarlyCompute::Verdict::NotJudged};
    /// FRITCHII_EXTENT_CBUF=3 (dry): where the store extent proven with constant buffer words
    /// would have ended the mark (t_proven_end), 0 when the dispatch had no such proof
    u64 proven_end{};
    /// FRITCHII_STORE_FOOTPRINT: the binding's footprint (an id of the footprint table, 0: none
    /// recorded) and the device address its offsets count from (the pointer word's)
    u32 footprint_id{};
    u64 footprint_base{};
    /// FRITCHII_FEEDER_LANE: the writing feeder's D (judged ones)
    EarlyCompute::Lane lane{};
};

/// FRITCHII_EXTENT_CBUF=3: set with t_writer while a dispatch's storage binding marks
/// (BufferCache::BindHostComputeStorageBuffers), 0 otherwise.
inline thread_local u64 t_proven_end = 0;

/// FRITCHII_STORE_FOOTPRINT: set with t_writer while a dispatch's native storage binding marks.
struct FootprintRef {
    u32 id{};
    u64 base{};
};
inline thread_local FootprintRef t_footprint{};

/// FRITCHII_STORE_FOOTPRINT: the footprints of the program bindings seen, by id (from 1). Entries
/// never move nor change once published, so readers look them up without the lock.
struct FootprintTable {
    static constexpr u32 Capacity = 8192;
    std::mutex mutex;
    std::unordered_map<u64, u32> ids; ///< (program hash, binding) -> id
    std::unique_ptr<Shader::StorageStoreFootprint[]> entries{
        std::make_unique<Shader::StorageStoreFootprint[]>(Capacity)};
    std::unique_ptr<u64[]> programs{std::make_unique<u64[]>(Capacity)};
    std::atomic<u32> count{1}; ///< id 0 is "none"
};

inline FootprintTable& GetFootprints() {
    static FootprintTable table;
    return table;
}

[[nodiscard]] inline u64 FootprintKey(u64 program, u32 binding) {
    return program ^ (u64{binding} * 0x9e3779b97f4a7c15ULL);
}

/// BufferCache::BindComputeStorageBuffer (=3): the id of a program binding's footprint, added
/// the first time (0 when the table is full).
[[nodiscard]] inline u32 InternFootprint(u64 program, u32 binding,
                                         const Shader::StorageStoreFootprint& footprint) {
    FootprintTable& t = GetFootprints();
    std::scoped_lock lock{t.mutex};
    const u64 key = FootprintKey(program, binding);
    if (const auto it = t.ids.find(key); it != t.ids.end()) {
        return it->second;
    }
    const u32 id = t.count.load(std::memory_order_relaxed);
    if (id >= FootprintTable::Capacity) {
        return 0;
    }
    t.entries[id] = footprint;
    t.programs[id] = program;
    t.count.store(id + 1, std::memory_order_release);
    t.ids.emplace(key, id);
    return id;
}

/// The footprint of `id`, or null for 0 (or an id not published).
[[nodiscard]] inline const Shader::StorageStoreFootprint* FootprintOf(u32 id) {
    FootprintTable& t = GetFootprints();
    if (id == 0 || id >= t.count.load(std::memory_order_acquire)) {
        return nullptr;
    }
    return &t.entries[id];
}

/// The footprint of a program binding (null when none was recorded at a bind).
[[nodiscard]] inline const Shader::StorageStoreFootprint* FootprintOf(u64 program, u32 binding) {
    FootprintTable& t = GetFootprints();
    u32 id = 0;
    {
        std::scoped_lock lock{t.mutex};
        const auto it = t.ids.find(FootprintKey(program, binding));
        if (it == t.ids.end()) {
            return nullptr;
        }
        id = it->second;
    }
    return FootprintOf(id);
}

/// FRITCHII_STORE_FOOTPRINT: how a read that goes to the back stands to the footprint of the
/// newest mark over its bytes.
enum class FootRead : u8 {
    None,               ///< not classed (the switch off, or no mark over the read's bytes)
    NoFootprint,        ///< the newest mark has no footprint (or no valid one)
    Inside,             ///< the read's bytes touch the newest mark's footprint
    Outside,            ///< outside it, and no older mark in the ring covers them: guest memory
                        ///< would answer
    CoveredInside,      ///< outside it, but an older mark's footprint covers them
    CoveredNoFootprint, ///< outside it, but an older mark without a footprint covers them
    Count,
};
inline constexpr size_t FootReads = static_cast<size_t>(FootRead::Count);

/// FRITCHII_EXTENT_CBUF=3: a read whose newest mark over its bytes had a cbuf proof.
enum class DryRead : u8 {
    None,         ///< no such mark (or the read's newest mark had no proof)
    Inside,       ///< the read's bytes touch the proven part of the mark
    Freed,        ///< past the proof, and no older mark in the ring covers them (the proven
                  ///< part only, for marks with a proof)
    CoveredSame,  ///< past the proof, an older mark of the same writer still covers them
    CoveredOther, ///< past the proof, an older mark of another writer covers them
};

/// Marks kept (the "Read writers" line says how many a frame, so how many frames this covers).
inline constexpr size_t RingSize = 16384;

/// B1 (fable52_answer_readback 2): a GPU-written run around a read that went to the back, kept
/// until 16 frames after its last read. Runs do not overlap (merged when added).
struct ReadRun {
    u64 begin{};
    u64 end{};
    u64 frame{};
};
inline constexpr u64 RunKeepFrames = 16;

struct Ring {
    std::mutex mutex;
    std::vector<WriteMark> marks; ///< RingSize entries once the first mark comes
    u64 count{};                  ///< marks made; the newest is marks[(count - 1) % RingSize]
    std::vector<ReadRun> runs;
    u64 run_bytes{};         ///< bytes marked GPU-written inside the runs, all writers
    u64 run_compute_bytes{}; ///< of them by dispatches
    u64 run_count_sum{};     ///< runs kept, summed over the window's frames
    u64 run_size_sum{};      ///< their bytes, summed over the window's frames
    u64 run_size_max{};      ///< the most bytes kept at one frame
};

inline Ring& GetRing() {
    static Ring ring;
    return ring;
}

/// BufferCache::MarkWrittenBuffer (the thread that binds: the back, or the memory stage).
/// `stamp`: the runtime's CurrentTick for the write (g_writer_ticks), else 0.
inline void NoteWrite(u64 begin, u64 size, u64 stamp = 0) {
    if (size == 0) {
        return;
    }
    const u64 end = begin + size;
    const WriteMark mark{.begin = begin,
                         .end = end,
                         .time_ns = NowNs(),
                         .dispatch = g_dispatches.load(std::memory_order_relaxed),
                         .stamp = stamp,
                         .writer = t_writer,
                         .verdict = EarlyCompute::t_verdict,
                         .proven_end = t_proven_end,
                         .footprint_id = t_footprint.id,
                         .footprint_base = t_footprint.base,
                         .lane = EarlyCompute::t_lane};
    const bool compute = mark.writer.kind == WriterKind::ComputeStorage ||
                         mark.writer.kind == WriterKind::ComputeTexture;
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    if (r.marks.empty()) {
        r.marks.resize(RingSize);
    }
    r.marks[r.count % RingSize] = mark;
    ++r.count;
    for (const ReadRun& run : r.runs) {
        const u64 lower = (std::max)(begin, run.begin);
        const u64 upper = (std::min)(end, run.end);
        if (lower < upper) {
            r.run_bytes += upper - lower;
            if (compute) {
                r.run_compute_bytes += upper - lower;
            }
        }
    }
}

/// Bytes of the read compared before and after they land.
inline constexpr u32 CompareBytes = 64;

/// Set by BufferCache::GetFlushArea on the reading thread when the read's pages hold GPU-written
/// bytes (before anything lands), cleared by GPU::OnCPURead before asking.
struct WriterAnswer {
    bool asked{};     ///< GetFlushArea looked
    bool found{};     ///< a mark in the ring covers at least one of the read's bytes
    bool page_only{}; ///< none does, but one is on the read's page(s)
    WriteMark mark{};
    u64 read_ns{};          ///< when the read asked
    u64 dispatches_after{}; ///< dispatches bound between the mark and the read
    u64 address{};          ///< the read's first byte
    u64 run_begin{};        ///< the GPU-written run around the read's pages (B1)
    u64 run_end{};
    u32 compared{}; ///< bytes in `before` (0: not taken)
    std::array<u8, CompareBytes> before{};
    // FRITCHII_READ_RUNS=2 ("Read writer ticks"): the newest mark over the read's pages, the
    // bytes its download takes (GetFlushArea fills the ticks, the waits fill the cut).
    bool area_found{};
    WriteMark area_mark{};
    u64 write_tick{};     ///< the scheduler tick the area mark's stamp is recorded in
    u64 read_tick{};      ///< the scheduler's current tick when the read asked
    u64 known_tick{};     ///< the host GPU's completed tick then (refreshed)
    bool cut{};           ///< a host wait was cut at the writer's tick
    u64 writer_wait_ns{}; ///< until the writer's submission ran
    u64 rest_wait_ns{};   ///< then until the read's download (or fence download) was there
    // FRITCHII_SUBMIT_TIMES: the submissions from the one after the known tick to the writer's,
    // on the GPU (measured once the writer ran).
    u64 cut_tick{}; ///< the scheduler tick the wait was cut at
    u32 ahead_submissions{};
    u32 ahead_resolved{};
    u64 ahead_busy_ns{};
    u64 ahead_gap_ns{};
    u64 ahead_overlap_ns{};
    u64 ahead_span_ns{};
    // FRITCHII_EXTENT_CBUF=3 (FindWriter): the read against the cbuf proof of its newest mark.
    DryRead dry{DryRead::None};
    // FRITCHII_EXTENT_CBUF / _SHARED =3, a read past the proof under an older mark (Covered*):
    // that mark, the newest over the read's bytes once every mark with a proof is cut at it, so
    // the one the read would wait for with the proofs (FindWriter). With the write ticks
    // (FRITCHII_EXTENT_SHARED=3 or FRITCHII_READ_RUNS=2 at boot) GetFlushArea fills its tick and
    // WaitReadWriter the time from the host wait's start until it ran (timed).
    bool dry_cover_found{};
    WriteMark dry_cover{};
    u64 dry_cover_tick{};
    u64 dry_cover_ns{};
    bool dry_cover_timed{};
    // FRITCHII_STORE_FOOTPRINT (FindWriter): the read against its newest mark's footprint, and
    // the read's first byte past that mark's pointer modulo the footprint's modulus.
    FootRead foot{FootRead::None};
    u64 foot_residue{};
    // FRITCHII_FEEDER_LANE: the area mark's D as scheduler ticks (GetFlushArea; 0: nothing
    // unrun), and when the reader saw them run (WaitReadWriter, from its wait's start).
    u64 lane_tick{};
    u64 lane_copy_tick{};
    u64 lane_ns{};
    u64 lane_copy_ns{};
    bool lane_unsubmitted{};
};
inline thread_local WriterAnswer t_writer_answer{};

/// FRITCHII_EARLY_COMPUTE: whether an unrun write mark overlaps one of `ranges` ({begin, end}),
/// newest first, stopping at the first run one (stamps grow with the ring, near enough for a
/// count). `is_free(stamp)`: the buffer cache clock's test.
template <typename Ranges, typename IsFree>
[[nodiscard]] bool AnyPendingMark(const Ranges& ranges, IsFree&& is_free) {
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (is_free(mark.stamp)) {
            return false;
        }
        for (const auto& [begin, end] : ranges) {
            if (mark.begin < end && begin < mark.end) {
                return true;
            }
        }
    }
    return false;
}

/// FRITCHII_READ_SIDE_COPY: the stamp of the newest write mark over any of `ranges`
/// ({begin, size}); false when none is in the ring.
template <typename Ranges>
[[nodiscard]] bool NewestMarkOver(const Ranges& ranges, u64& stamp) {
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        for (const auto& [begin, size] : ranges) {
            if (mark.begin < begin + size && begin < mark.end) {
                stamp = mark.stamp;
                return true;
            }
        }
    }
    return false;
}

/// FRITCHII_FEEDER_LANE: the newest unrun write mark over any of `ranges` ({begin, end}), newest
/// first until the first run one: its stamp, or a judged feeder's own D.
template <typename Ranges, typename IsFree>
[[nodiscard]] EarlyCompute::LaneDep NewestPendingMark(const Ranges& ranges, IsFree&& is_free) {
    EarlyCompute::LaneDep dep{};
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (is_free(mark.stamp)) {
            return dep;
        }
        for (const auto& [begin, end] : ranges) {
            if (mark.begin < end && begin < mark.end) {
                EarlyCompute::LaneKind kind = EarlyCompute::LaneKind::Other;
                switch (mark.writer.kind) {
                case WriterKind::DrawStorage:
                case WriterKind::DrawTexture:
                    kind = EarlyCompute::LaneKind::DrawBuffer;
                    break;
                case WriterKind::ComputeStorage:
                case WriterKind::ComputeTexture:
                    kind = EarlyCompute::LaneKind::ComputeMain;
                    break;
                case WriterKind::Other:
                    break;
                }
                dep.Add(mark.stamp, mark.lane, kind);
                break;
            }
        }
    }
    dep.incomplete = n == RingSize;
    return dep;
}

/// FRITCHII_FEEDER_LANE: after a judged feeder's binds uploaded CPU data, its marks' D is its
/// own chunk (`stamp`); D_copy stays.
inline void PatchLaneUpload(u64 dispatch, u64 program, u64 stamp) {
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, 256);
    for (u64 i = 0; i < n; ++i) {
        WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (mark.dispatch == dispatch && mark.writer.program == program && mark.lane.judged) {
            mark.lane.d = (std::max)(mark.lane.d, stamp);
            mark.lane.kind = EarlyCompute::LaneKind::Upload;
        }
    }
}

/// FRITCHII_STORE_FOOTPRINT: whether `mark`'s footprint can hold any of the read's bytes in the
/// mark ([address, end) clipped to it); nullopt when the mark has no valid footprint.
[[nodiscard]] inline std::optional<bool> FootprintHits(const WriteMark& mark, u64 address,
                                                       u64 end) {
    const Shader::StorageStoreFootprint* footprint = FootprintOf(mark.footprint_id);
    if (footprint == nullptr || !footprint->Valid()) {
        return std::nullopt;
    }
    const u64 lower = (std::max)(address, mark.begin);
    const u64 upper = (std::min)(end, mark.end);
    if (lower >= upper) {
        return false;
    }
    const s64 offset = static_cast<s64>(lower - mark.footprint_base);
    return Shader::StoreFootprintHits(*footprint, offset, upper - lower);
}

/// FRITCHII_STORE_FOOTPRINT (r.mutex held): the read [address, end) against marks[newest - i]
/// (the newest over it), then the older marks over it while it stays outside the footprints.
[[nodiscard]] inline FootRead ClassifyFootprint(const Ring& r, u64 i, u64 address, u64 end) {
    const WriteMark& newest = r.marks[(r.count - 1 - i) % RingSize];
    const std::optional<bool> hits = FootprintHits(newest, address, end);
    if (!hits) {
        return FootRead::NoFootprint;
    }
    if (*hits) {
        return FootRead::Inside;
    }
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 j = i + 1; j < n; ++j) {
        const WriteMark& mark = r.marks[(r.count - 1 - j) % RingSize];
        if (!(mark.begin < end && address < mark.end)) {
            continue;
        }
        const std::optional<bool> older = FootprintHits(mark, address, end);
        if (!older) {
            return FootRead::CoveredNoFootprint;
        }
        if (*older) {
            return FootRead::CoveredInside;
        }
    }
    return FootRead::Outside;
}

/// FRITCHII_EXTENT_CBUF=3 (r.mutex held): the read [address, end) against marks[newest - i]
/// (the newest over it, with a cbuf proof): inside its proven part, or past it and then whether
/// an older mark in the ring still covers the bytes once every mark with a proof is cut there
/// (that mark in `cover`).
[[nodiscard]] inline DryRead ClassifyDryRead(const Ring& r, u64 i, u64 address, u64 end,
                                             const WriteMark*& cover) {
    cover = nullptr;
    const auto effective_end = [](const WriteMark& mark) {
        return mark.proven_end != 0 ? (std::min)(mark.end, mark.proven_end) : mark.end;
    };
    const WriteMark& newest = r.marks[(r.count - 1 - i) % RingSize];
    if (newest.begin < end && address < effective_end(newest)) {
        return DryRead::Inside;
    }
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 j = i + 1; j < n; ++j) {
        const WriteMark& mark = r.marks[(r.count - 1 - j) % RingSize];
        if (mark.begin < end && address < effective_end(mark)) {
            cover = &mark;
            return mark.writer == newest.writer ? DryRead::CoveredSame : DryRead::CoveredOther;
        }
    }
    return DryRead::Freed;
}

/// GetFlushArea: the newest mark over [address, address + size), else note one on its pages.
inline void FindWriter(u64 address, u64 size, u64 page_begin, u64 page_end) {
    WriterAnswer& a = t_writer_answer;
    a.asked = true;
    a.read_ns = NowNs();
    a.address = address;
    const u64 end = address + size;
    const u64 dispatches = g_dispatches.load(std::memory_order_relaxed);
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    const u64 n = (std::min<u64>)(r.count, RingSize);
    for (u64 i = 0; i < n; ++i) {
        const WriteMark& mark = r.marks[(r.count - 1 - i) % RingSize];
        if (!a.area_found && mark.begin < page_end && page_begin < mark.end) {
            a.area_found = true;
            a.area_mark = mark;
        }
        if (mark.begin < end && address < mark.end) {
            a.found = true;
            a.page_only = false;
            a.mark = mark;
            a.dispatches_after = dispatches - mark.dispatch;
            if (mark.proven_end != 0) [[unlikely]] {
                const WriteMark* cover = nullptr;
                a.dry = ClassifyDryRead(r, i, address, end, cover);
                if (cover != nullptr) {
                    a.dry_cover_found = true;
                    a.dry_cover = *cover;
                }
            }
            if (g_store_footprint) [[unlikely]] {
                a.foot = ClassifyFootprint(r, i, address, end);
                if (const auto* footprint = FootprintOf(mark.footprint_id);
                    footprint != nullptr && footprint->Valid()) {
                    const u64 lower = (std::max)(address, mark.begin);
                    const s64 offset = static_cast<s64>(lower - mark.footprint_base);
                    const s64 modulus = static_cast<s64>(footprint->modulus);
                    const s64 residue = offset % modulus;
                    a.foot_residue = static_cast<u64>(residue < 0 ? residue + modulus : residue);
                }
            }
            return;
        }
        if (!a.page_only && mark.begin < page_end && page_begin < mark.end) {
            a.page_only = true;
        }
    }
}

/// Per writer (program, binding, kind), reset after each line.
struct WriterStats {
    u64 reads[3]{}; ///< [Kind]
    u64 ages[5]{};  ///< the write's age at the read: < 0.5 ms, < 1, < 2, < 4, more
    u64 wait_ns{};
    u64 lowest_offset{std::numeric_limits<u64>::max()};
    u64 highest_offset{};
    u64 previous_offset{std::numeric_limits<u64>::max()};
    std::unordered_set<u64> offsets;           ///< capped at 256
    std::unordered_map<s64, u64> steps;        ///< offset minus the previous read's; capped at 64
    u64 compared{};
    u64 differed{};
    u64 dispatches_after{};
};

struct WriterHash {
    size_t operator()(const Writer& w) const noexcept {
        return std::hash<u64>{}(w.program ^ (u64{w.binding} << 48) ^
                                (u64{static_cast<u8>(w.kind)} << 56));
    }
};

/// What the line reports, reset after each line.
struct Counts {
    u64 reads[3]{};
    u64 next_page{};  ///< on the page after the previous read's
    u64 near_pages{}; ///< 2 to 16 pages after or before it
    u64 same_page{};  ///< on the previous read's page
    u64 runs[5]{};    ///< not in the buffer cache, 1 page, 2-16, 17-64, more
    u64 wait_ns{};
    u64 max_wait_ns{};
    u64 lowest{std::numeric_limits<u64>::max()};
    u64 highest{};
    u64 distinct_pages{}; ///< summed over the window's frames
    u64 repeat_reads{};   ///< reads of a page already read that frame
    // The "Read overlap" line.
    u64 overlap[3][3]{};  ///< [Kind][Overlap]
    u64 overlap_wait_ns[3]{}; ///< [Overlap], all kinds
    u64 gaps[5]{};        ///< PageOnly: < 64 B, < 256 B, < 1 KiB, < 4 KiB, more
    u64 sizes[5]{};       ///< <= 16 B, <= 64, <= 256, <= 4096, more
    u64 modified_differs{}; ///< waited for a download, and gpu_modified_ranges answers otherwise
    // The "Read writers" line.
    u64 writer_not_asked{}; ///< GetFlushArea found no GPU-written byte in the read's pages
    u64 writer_none{};      ///< no mark in the ring over the read's bytes nor its pages
    u64 writer_page_only{}; ///< none over its bytes, one on its pages
    u64 writer_ages[3][3]{}; ///< [Kind][age < 1 ms, 1 to 1.5 ms, 1.5 ms or more]
    u64 writer_compared{};
    u64 writer_differed{};
};

/// The "Read writer ticks" line (g_writer_ticks), per Kind; reset after each line.
struct TickCounts {
    u64 reads[3]{};      ///< asked, with a mark over the read's pages in the ring
    u64 no_mark[3]{};    ///< asked, none in the ring (written before its window)
    u64 state[3][3]{};   ///< the writer's submission at the read: not submitted, submitted, run
    u64 behind[3][5]{};  ///< submissions from the writer's to the read's: 0, 1-2, 3-5, 6-10, more
    u64 behind_sum[3]{};
    u64 unrun_sum[3]{};  ///< submitted after the writer's and not run when the read asked
    u64 kinds[3][5]{};   ///< [Kind][WriterKind] of the mark
    u64 wait_ns[3]{};    ///< the reads' whole wait (as the other lines count it)
    u64 cuts[3]{};       ///< host waits cut at the writer's tick
    u64 writer_ns[3]{};  ///< of them, until the writer's submission ran
    u64 rest_ns[3]{};    ///< then until the download was there
    u64 rest_max_ns[3]{};
    // FRITCHII_SUBMIT_TIMES, over the cut reads that had submissions measured.
    u64 ahead_reads[3]{};
    u64 ahead_submissions[3]{};
    u64 ahead_resolved[3]{};
    u64 ahead_busy_ns[3]{};
    u64 ahead_gap_ns[3]{};
    u64 ahead_overlap_ns[3]{};
    u64 ahead_span_ns[3]{};
    u64 ahead_writer_ns[3]{}; ///< those reads' "until the writer ran", to set against the above
};

/// FRITCHII_EXTENT_CBUF=3: per writer with a cbuf proof, the reads its marks sent to the back
/// (taken by the "Extent cbuf" lines every 10 s, BufferCache::LogExtentCensus).
struct DryStats {
    u64 reads[5]{};   ///< [DryRead]
    u64 wait_ns[5]{}; ///< [DryRead]
    /// Must be 0: Freed reads whose compared bytes changed when the wait landed (the GPU did
    /// write them: the proof would have served stale bytes).
    u64 freed_compared{};
    u64 freed_differed{};
    u64 inside_compared{};
    u64 inside_differed{};
    /// The marks the proof leaves over the Covered reads' bytes, by their writer (capped at 64
    /// writers): what those reads would still wait for
    struct Cover {
        u64 reads{};
        u64 wait_ns{};      ///< the reads' whole waits, as waited
        u64 timed{};        ///< of them, timed until the cover ran
        u64 timed_wait_ns{}; ///< the timed ones' whole waits
        u64 predicted_ns{}; ///< the timed ones' waits had they ended when the cover ran
    };
    std::unordered_map<Writer, Cover, WriterHash> covers;
    /// Every read's wait with the proof: Inside as waited, Freed 0 (no mark left over its
    /// bytes), Covered until the cover ran when timed, else as waited
    u64 proof_wait_ns{};
    u64 covered_timed{};
    u64 covered_untimed{};
};

/// FRITCHII_EXTENT_CENSUS: per writer, the reads its newest marks sent to the back (taken by
/// the "Extent census" lines every 10 s while g_wait_census is set).
struct WaitStats {
    u64 reads{};
    u64 wait_ns{};
};
inline std::atomic<bool> g_wait_census{false};

/// FRITCHII_STORE_FOOTPRINT: per writer (the newest mark's over the read's bytes), the reads to
/// the back by FootRead (taken by the "Store footprint" lines every 10 s).
struct FootStats {
    u64 reads[FootReads]{};
    u64 wait_ns[FootReads]{};
    /// Must be 0: Outside reads whose compared bytes changed when the wait landed (the GPU did
    /// write them: guest memory would have answered stale bytes)
    u64 outside_compared{};
    u64 outside_differed{};
    u64 inside_compared{};
    u64 inside_differed{};
    /// The read's first byte past the mark's pointer, modulo the footprint's modulus: reads
    std::unordered_map<u64, u64> residues; ///< capped at 32
};

struct State {
    std::mutex mutex;
    Counts counts;
    TickCounts ticks;
    std::unordered_map<Writer, FootStats, WriterHash> foot;
    std::unordered_map<Writer, DryStats, WriterHash> dry;
    std::unordered_map<Writer, WaitStats, WriterHash> waits;
    /// The "Read writer" lines: reads whose bytes a mark in the ring covers, by its writer.
    std::unordered_map<Writer, WriterStats, WriterHash> writers;
    u64 previous_page{std::numeric_limits<u64>::max()};
    /// Pages read in the frame `frame`.
    std::unordered_set<u64> frame_pages;
    u64 frame{};
};

inline State& Get() {
    static State state;
    return state;
}

/// BufferCache::TickFrame advances it once a frame.
inline std::atomic<u64> g_frame{};
/// EXP112's run flushes and the bytes they took beyond the read's own pages.
inline std::atomic<u64> g_run_flushes{};
inline std::atomic<u64> g_run_bytes{};
/// EXP112: read flushes it looked at (BufferCache::DownloadMemoryForRead), and of them those
/// whose own pages were Preflushable by then (GetFlushArea marks them first with preflushing on).
inline std::atomic<u64> g_run_asked{};
inline std::atomic<u64> g_run_own_marked{};

/// Why EXP112 took no run for a read flush.
enum class NoRun : u8 {
    NotGpuModified,        ///< the read's pages held no GPU-modified byte by the flush
    NoBuffer,              ///< no buffer holds the read's pages
    NeighbourPreflushable, ///< the run would go on, but the next page is Preflushable
    NoNeighbour,           ///< no GPU-written page next to the read's inside its buffer
    Count,
};
inline std::atomic<u64> g_no_run[static_cast<size_t>(NoRun::Count)]{};

inline void CountNoRun(NoRun why) {
    if (g_enabled) [[unlikely]] {
        g_no_run[static_cast<size_t>(why)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// GPU::OnCPURead, once per read past the preemptive answer. `address` is the read's area start
/// (page aligned), `size` the read's own size; `bytes` is t_read_bytes as GetFlushArea left it.
inline void NoteRead(Kind kind, u64 address, u64 run_pages, u64 wait_ns, u64 size,
                     const ReadBytes& bytes) {
    State& s = Get();
    const u64 page = address >> 12;
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    // A flush downloads gpu_modified_ranges' bytes; a wait waits for the download on its way.
    const ByteAnswer& answer = kind == Kind::Flushed ? bytes.modified : bytes.on_way;
    std::scoped_lock lock{s.mutex};
    Counts& c = s.counts;
    ++c.reads[static_cast<size_t>(kind)];
    ++c.overlap[static_cast<size_t>(kind)][static_cast<size_t>(answer.overlap)];
    c.overlap_wait_ns[static_cast<size_t>(answer.overlap)] += wait_ns;
    if (answer.overlap == Overlap::PageOnly) {
        ++c.gaps[answer.gap < 64     ? 0
                 : answer.gap < 256  ? 1
                 : answer.gap < 1024 ? 2
                 : answer.gap < 4096 ? 3
                                     : 4];
    }
    ++c.sizes[size <= 16 ? 0 : size <= 64 ? 1 : size <= 256 ? 2 : size <= 4096 ? 3 : 4];
    if (kind != Kind::Flushed && bytes.modified.overlap != answer.overlap) {
        ++c.modified_differs;
    }
    if (s.previous_page != std::numeric_limits<u64>::max()) {
        const u64 distance =
            page >= s.previous_page ? page - s.previous_page : s.previous_page - page;
        if (distance == 0) {
            ++c.same_page;
        } else if (page == s.previous_page + 1) {
            ++c.next_page;
        } else if (distance <= 16) {
            ++c.near_pages;
        }
    }
    s.previous_page = page;
    const size_t bucket = run_pages == 0    ? 0
                          : run_pages == 1  ? 1
                          : run_pages <= 16 ? 2
                          : run_pages <= 64 ? 3
                                            : 4;
    ++c.runs[bucket];
    c.wait_ns += wait_ns;
    c.max_wait_ns = (std::max)(c.max_wait_ns, wait_ns);
    c.lowest = (std::min)(c.lowest, address);
    c.highest = (std::max)(c.highest, address);
    if (frame != s.frame) {
        c.distinct_pages += s.frame_pages.size();
        s.frame_pages.clear();
        s.frame = frame;
    }
    if (!s.frame_pages.insert(page).second) {
        ++c.repeat_reads;
    }
}

/// GPU::OnCPURead, after NoteRead: the read's writer (t_writer_answer as GetFlushArea left it)
/// and, when `after` is given, the read's first a.compared bytes in guest memory now (landed).
inline void NoteWriter(Kind kind, u64 wait_ns, const WriterAnswer& a, const u8* after) {
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (a.run_begin < a.run_end) {
        // B1: this read's run, merged with the kept runs it touches or adjoins.
        Ring& r = GetRing();
        std::scoped_lock lock{r.mutex};
        ReadRun run{.begin = a.run_begin, .end = a.run_end, .frame = frame};
        for (size_t i = 0; i < r.runs.size();) {
            const ReadRun& kept = r.runs[i];
            if (kept.begin <= run.end && run.begin <= kept.end) {
                run.begin = (std::min)(run.begin, kept.begin);
                run.end = (std::max)(run.end, kept.end);
                r.runs[i] = r.runs.back();
                r.runs.pop_back();
            } else {
                ++i;
            }
        }
        r.runs.push_back(run);
    }
    const bool compared = a.compared != 0 && after != nullptr;
    const bool differed = compared && std::memcmp(a.before.data(), after, a.compared) != 0;
    const u64 age_ns = a.found && a.read_ns > a.mark.time_ns ? a.read_ns - a.mark.time_ns : 0;
    // FRITCHII_EARLY_COMPUTE: the programs whose writes are read back, and each read by its
    // writer's verdict, with what running that writer early would have saved this read.
    if (EarlyCompute::g_early && a.asked && a.area_found) [[unlikely]] {
        if (a.area_mark.writer.kind == WriterKind::ComputeStorage ||
            a.area_mark.writer.kind == WriterKind::ComputeTexture) {
            EarlyCompute::LearnFeeder(a.area_mark.writer.program);
        }
        EarlyCompute::NoteRead(static_cast<size_t>(kind), a.area_mark.verdict, a.cut,
                               a.writer_wait_ns);
        // FRITCHII_FEEDER_LANE: what the area mark's feeder would have saved on the lane
        if (EarlyCompute::g_lane && a.area_mark.lane.judged) {
            EarlyCompute::NoteLaneRead(a.area_mark.writer.program,
                                       EarlyCompute::LaneRead{
                                           .wait_ns = wait_ns,
                                           .cut = a.cut,
                                           .writer_ns = a.writer_wait_ns,
                                           .write_tick = a.write_tick,
                                           .lane_tick = a.lane_tick,
                                           .lane_ns = a.lane_ns,
                                           .copy_tick = a.lane_copy_tick,
                                           .copy_ns = a.lane_copy_ns,
                                           .unsubmitted = a.lane_unsubmitted,
                                           .kind = a.area_mark.lane.kind,
                                       });
        }
    }
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    if (g_writer_ticks && a.asked) {
        TickCounts& t = s.ticks;
        const size_t k = static_cast<size_t>(kind);
        t.wait_ns[k] += wait_ns;
        if (!a.area_found) {
            ++t.no_mark[k];
        } else {
            ++t.reads[k];
            const size_t state = a.write_tick >= a.read_tick    ? 0
                                 : a.write_tick <= a.known_tick ? 2
                                                                : 1;
            ++t.state[k][state];
            const u64 behind = a.read_tick > a.write_tick ? a.read_tick - a.write_tick : 0;
            ++t.behind[k][behind == 0    ? 0
                          : behind <= 2  ? 1
                          : behind <= 5  ? 2
                          : behind <= 10 ? 3
                                         : 4];
            t.behind_sum[k] += behind;
            const u64 floor = (std::max)(a.write_tick, a.known_tick);
            t.unrun_sum[k] += a.read_tick > floor + 1 ? a.read_tick - floor - 1 : 0;
            ++t.kinds[k][static_cast<size_t>(a.area_mark.writer.kind)];
            if (a.cut) {
                ++t.cuts[k];
                t.writer_ns[k] += a.writer_wait_ns;
                t.rest_ns[k] += a.rest_wait_ns;
                t.rest_max_ns[k] = (std::max)(t.rest_max_ns[k], a.rest_wait_ns);
                if (a.ahead_resolved != 0) {
                    ++t.ahead_reads[k];
                    t.ahead_submissions[k] += a.ahead_submissions;
                    t.ahead_resolved[k] += a.ahead_resolved;
                    t.ahead_busy_ns[k] += a.ahead_busy_ns;
                    t.ahead_gap_ns[k] += a.ahead_gap_ns;
                    t.ahead_overlap_ns[k] += a.ahead_overlap_ns;
                    t.ahead_span_ns[k] += a.ahead_span_ns;
                    t.ahead_writer_ns[k] += a.writer_wait_ns;
                }
            }
        }
    }
    Counts& c = s.counts;
    if (!a.asked) {
        ++c.writer_not_asked;
        return;
    }
    if (!a.found) {
        ++(a.page_only ? c.writer_page_only : c.writer_none);
        return;
    }
    if (g_wait_census.load(std::memory_order_relaxed)) [[unlikely]] {
        WaitStats& waited = s.waits[a.mark.writer];
        ++waited.reads;
        waited.wait_ns += wait_ns;
    }
    if (a.foot != FootRead::None) [[unlikely]] {
        FootStats& f = s.foot[a.mark.writer];
        const size_t foot = static_cast<size_t>(a.foot);
        ++f.reads[foot];
        f.wait_ns[foot] += wait_ns;
        if (a.foot == FootRead::Outside) {
            f.outside_compared += compared ? 1 : 0;
            f.outside_differed += differed ? 1 : 0;
        } else if (a.foot == FootRead::Inside) {
            f.inside_compared += compared ? 1 : 0;
            f.inside_differed += differed ? 1 : 0;
        }
        if (a.foot != FootRead::NoFootprint) {
            if (const auto it = f.residues.find(a.foot_residue); it != f.residues.end()) {
                ++it->second;
            } else if (f.residues.size() < 32) {
                f.residues.emplace(a.foot_residue, 1);
            }
        }
    }
    if (a.dry != DryRead::None) [[unlikely]] {
        DryStats& d = s.dry[a.mark.writer];
        const size_t dry = static_cast<size_t>(a.dry);
        ++d.reads[dry];
        d.wait_ns[dry] += wait_ns;
        if (a.dry == DryRead::Freed) {
            d.freed_compared += compared ? 1 : 0;
            d.freed_differed += differed ? 1 : 0;
        } else if (a.dry == DryRead::Inside) {
            d.inside_compared += compared ? 1 : 0;
            d.inside_differed += differed ? 1 : 0;
            d.proof_wait_ns += wait_ns;
        } else if (a.dry_cover_found) {
            // Covered: with the proof the read's host wait would have ended when the cover ran
            // (the rest of its wait, before the host wait, stays)
            const u64 host = a.writer_wait_ns + a.rest_wait_ns;
            const bool timed = a.cut && a.dry_cover_timed && host != 0;
            const u64 predicted =
                timed ? wait_ns - (std::min)(wait_ns, host) + (std::min)(a.dry_cover_ns, host)
                      : wait_ns;
            d.proof_wait_ns += predicted;
            ++(timed ? d.covered_timed : d.covered_untimed);
            auto it = d.covers.find(a.dry_cover.writer);
            if (it == d.covers.end() && d.covers.size() < 64) {
                it = d.covers.emplace(a.dry_cover.writer, DryStats::Cover{}).first;
            }
            if (it != d.covers.end()) {
                DryStats::Cover& cover = it->second;
                ++cover.reads;
                cover.wait_ns += wait_ns;
                if (timed) {
                    ++cover.timed;
                    cover.timed_wait_ns += wait_ns;
                    cover.predicted_ns += predicted;
                }
            }
        } else {
            d.proof_wait_ns += wait_ns;
        }
    }
    const size_t k = static_cast<size_t>(kind);
    ++c.writer_ages[k][age_ns < 1'000'000 ? 0 : age_ns < 1'500'000 ? 1 : 2];
    c.writer_compared += compared ? 1 : 0;
    c.writer_differed += differed ? 1 : 0;
    WriterStats& w = s.writers[a.mark.writer];
    ++w.reads[k];
    ++w.ages[age_ns < 500'000     ? 0
             : age_ns < 1'000'000 ? 1
             : age_ns < 2'000'000 ? 2
             : age_ns < 4'000'000 ? 3
                                  : 4];
    w.wait_ns += wait_ns;
    const u64 offset = a.address >= a.mark.begin ? a.address - a.mark.begin : 0;
    w.lowest_offset = (std::min)(w.lowest_offset, offset);
    w.highest_offset = (std::max)(w.highest_offset, offset);
    if (w.offsets.size() < 256) {
        w.offsets.insert(offset);
    }
    if (w.previous_offset != std::numeric_limits<u64>::max()) {
        const s64 step = static_cast<s64>(offset - w.previous_offset);
        if (const auto it = w.steps.find(step); it != w.steps.end()) {
            ++it->second;
        } else if (w.steps.size() < 64) {
            w.steps.emplace(step, 1);
        }
    }
    w.previous_offset = offset;
    w.compared += compared ? 1 : 0;
    w.differed += differed ? 1 : 0;
    w.dispatches_after += a.dispatches_after;
}

/// FRITCHII_EXTENT_CENSUS: the per-writer read-back waits since the last call.
[[nodiscard]] inline std::unordered_map<Writer, WaitStats, WriterHash> TakeWaits() {
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    std::unordered_map<Writer, WaitStats, WriterHash> taken;
    taken.swap(s.waits);
    return taken;
}

/// FRITCHII_EXTENT_CBUF=3: the per-writer dry read counts since the last call.
[[nodiscard]] inline std::unordered_map<Writer, DryStats, WriterHash> TakeDry() {
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    std::unordered_map<Writer, DryStats, WriterHash> taken;
    taken.swap(s.dry);
    return taken;
}

/// BufferCache::TickFrame, once a frame (after advancing g_frame): drops the runs not read for
/// RunKeepFrames frames and sums what is kept.
inline void FrameTick() {
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    Ring& r = GetRing();
    std::scoped_lock lock{r.mutex};
    u64 bytes = 0;
    for (size_t i = 0; i < r.runs.size();) {
        if (r.runs[i].frame + RunKeepFrames < frame) {
            r.runs[i] = r.runs.back();
            r.runs.pop_back();
        } else {
            bytes += r.runs[i].end - r.runs[i].begin;
            ++i;
        }
    }
    r.run_count_sum += r.runs.size();
    r.run_size_sum += bytes;
    r.run_size_max = (std::max)(r.run_size_max, bytes);
}

inline const char* WriterKindName(WriterKind kind) {
    switch (kind) {
    case WriterKind::ComputeStorage:
        return "dispatch storage buffer";
    case WriterKind::ComputeTexture:
        return "dispatch texture buffer";
    case WriterKind::DrawStorage:
        return "draw storage buffer";
    case WriterKind::DrawTexture:
        return "draw texture buffer";
    case WriterKind::Other:
        break;
    }
    return "other (copy, query, transform feedback)";
}

/// Log: the "Read writers", "Read writer" and "Read runs for B1" lines (s.mutex held).
inline void LogWriters(State& s, double frames) {
    Counts& c = s.counts;
    const u64 found = c.writer_ages[0][0] + c.writer_ages[0][1] + c.writer_ages[0][2] +
                      c.writer_ages[1][0] + c.writer_ages[1][1] + c.writer_ages[1][2] +
                      c.writer_ages[2][0] + c.writer_ages[2][1] + c.writer_ages[2][2];
    u64 marks = 0;
    double runs_a_frame = 0.0;
    double run_kib_a_frame = 0.0;
    u64 run_kib_max = 0;
    u64 run_bytes = 0;
    u64 run_compute_bytes = 0;
    {
        Ring& r = GetRing();
        std::scoped_lock lock{r.mutex};
        static u64 previous_count = 0;
        marks = r.count - previous_count;
        previous_count = r.count;
        runs_a_frame = static_cast<double>(r.run_count_sum) / frames;
        run_kib_a_frame = static_cast<double>(r.run_size_sum) / 1024.0 / frames;
        run_kib_max = r.run_size_max >> 10;
        run_bytes = r.run_bytes;
        run_compute_bytes = r.run_compute_bytes;
        r.run_count_sum = 0;
        r.run_size_sum = 0;
        r.run_size_max = 0;
        r.run_bytes = 0;
        r.run_compute_bytes = 0;
    }
    LOG_INFO(HW_GPU,
             "Read writers (FRITCHII_READ_RUNS) last 600 frames: {} reads with a write mark over "
             "their bytes, {} with one only on their page, {} with none in the ring ({} marks), "
             "{} not GPU-written by the buffer cache | the write's age at the read as <1 ms/1-1.5/"
             ">=1.5: flushed {}/{}/{}, read download {}/{}/{}, fence download {}/{}/{} | read "
             "bytes differed from guest memory at landing in {} of {} compared | {:.1f} marks a "
             "frame, {} writers",
             found, c.writer_page_only, c.writer_none, RingSize, c.writer_not_asked,
             c.writer_ages[0][0], c.writer_ages[0][1], c.writer_ages[0][2], c.writer_ages[1][0],
             c.writer_ages[1][1], c.writer_ages[1][2], c.writer_ages[2][0], c.writer_ages[2][1],
             c.writer_ages[2][2], c.writer_differed, c.writer_compared,
             static_cast<double>(marks) / frames, s.writers.size());
    std::vector<std::pair<const Writer*, const WriterStats*>> order;
    order.reserve(s.writers.size());
    for (const auto& [writer, stats] : s.writers) {
        order.emplace_back(&writer, &stats);
    }
    constexpr size_t Top = 8;
    const size_t shown = (std::min)(Top, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                      order.end(), [](const auto& a, const auto& b) {
                          return a.second->wait_ns > b.second->wait_ns;
                      });
    for (size_t i = 0; i < shown; ++i) {
        const Writer& writer = *order[i].first;
        const WriterStats& w = *order[i].second;
        const u64 reads = w.reads[0] + w.reads[1] + w.reads[2];
        s64 step = 0;
        u64 step_count = 0;
        for (const auto& [value, count] : w.steps) {
            if (count > step_count) {
                step = value;
                step_count = count;
            }
        }
        const bool draw =
            writer.kind == WriterKind::DrawStorage || writer.kind == WriterKind::DrawTexture;
        const std::string binding = draw ? fmt::format("stage {} binding {}", writer.binding / 64,
                                                       writer.binding % 64)
                                         : fmt::format("binding {}", writer.binding);
        LOG_INFO(HW_GPU,
                 "Read writer {}/{}: {} program {:016x} {} | reads flushed {}, read download {}, "
                 "fence download {} | the write's age at the read <0.5 ms {}, <1 {}, <2 {}, <4 "
                 "{}, more {} | waited {:.2f} ms a frame ({:.3f} ms a read) | the read's offset "
                 "in the marked range {:#x} to {:#x}, {} distinct, most common step {:+#x} ({} "
                 "times) | bytes differed at landing in {} of {} | {:.1f} dispatches bound after "
                 "the write",
                 i + 1, shown, WriterKindName(writer.kind), writer.program, binding, w.reads[0],
                 w.reads[1], w.reads[2], w.ages[0], w.ages[1], w.ages[2], w.ages[3], w.ages[4],
                 static_cast<double>(w.wait_ns) / 1e6 / frames,
                 reads != 0 ? static_cast<double>(w.wait_ns) / 1e6 / static_cast<double>(reads)
                            : 0.0,
                 w.lowest_offset, w.highest_offset, w.offsets.size(), step, step_count,
                 w.differed, w.compared,
                 reads != 0 ? static_cast<double>(w.dispatches_after) / static_cast<double>(reads)
                            : 0.0);
    }
    LOG_INFO(HW_GPU,
             "Read runs for B1 (FRITCHII_READ_RUNS) last 600 frames: {:.1f} distinct GPU-written "
             "runs read in the last {} frames ({:.0f} KiB, at most {} KiB) | marked GPU-written "
             "in them {:.0f} KiB a frame, {:.0f} KiB of it by dispatches",
             runs_a_frame, RunKeepFrames, run_kib_a_frame, run_kib_max,
             static_cast<double>(run_bytes) / 1024.0 / frames,
             static_cast<double>(run_compute_bytes) / 1024.0 / frames);
    s.writers.clear();
}

/// Log: the "Read writer ticks" lines, one per read kind that had reads (s.mutex held).
inline void LogTicks(State& s, double frames) {
    TickCounts& t = s.ticks;
    static constexpr const char* KindNames[3] = {"flushed (new download)", "read download on its way",
                                                 "fence download"};
    for (size_t k = 0; k < 3; ++k) {
        const u64 reads = t.reads[k];
        if (reads == 0 && t.no_mark[k] == 0) {
            continue;
        }
        const auto per_read = [reads](u64 value) {
            return reads != 0 ? static_cast<double>(value) / static_cast<double>(reads) : 0.0;
        };
        const auto ms_per_cut = [&](u64 ns) {
            return t.cuts[k] != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(t.cuts[k])
                                  : 0.0;
        };
        LOG_INFO(HW_GPU,
                 "Read writer ticks (FRITCHII_READ_RUNS=2) last 600 frames, {}: {} reads with a "
                 "write mark over their pages, {} without one in the ring | the mark's writer: "
                 "dispatch storage {}, dispatch texture {}, draw storage {}, draw texture {}, "
                 "other (copy, query, transform feedback) {} | its submission when the read "
                 "asked: not submitted {}, submitted not run {}, run {} | the read's tick is "
                 "{:.1f} submissions after the writer's (0: {}, 1-2: {}, 3-5: {}, 6-10: {}, more: "
                 "{}), {:.1f} of them submitted and not run | waited {:.2f} ms a frame | host "
                 "waits cut at the writer's tick {}: until the writer ran {:.3f} ms each, then "
                 "until the download was there {:.3f} ms each (longest {:.2f})",
                 KindNames[k], reads, t.no_mark[k],
                 t.kinds[k][static_cast<size_t>(WriterKind::ComputeStorage)],
                 t.kinds[k][static_cast<size_t>(WriterKind::ComputeTexture)],
                 t.kinds[k][static_cast<size_t>(WriterKind::DrawStorage)],
                 t.kinds[k][static_cast<size_t>(WriterKind::DrawTexture)],
                 t.kinds[k][static_cast<size_t>(WriterKind::Other)], t.state[k][0], t.state[k][1],
                 t.state[k][2], per_read(t.behind_sum[k]), t.behind[k][0], t.behind[k][1],
                 t.behind[k][2], t.behind[k][3], t.behind[k][4], per_read(t.unrun_sum[k]),
                 static_cast<double>(t.wait_ns[k]) / 1e6 / frames, t.cuts[k],
                 ms_per_cut(t.writer_ns[k]), ms_per_cut(t.rest_ns[k]),
                 static_cast<double>(t.rest_max_ns[k]) / 1e6);
        if (t.ahead_reads[k] != 0) {
            const double n = static_cast<double>(t.ahead_reads[k]);
            const auto ms = [n](u64 ns) { return static_cast<double>(ns) / 1e6 / n; };
            LOG_INFO(HW_GPU,
                     "Read writer run-up (FRITCHII_SUBMIT_TIMES) last 600 frames, {}: {} cut reads "
                     "measured | per read: {:.1f} submissions from the one after the known tick "
                     "to the writer's ({:.1f} with both GPU timestamps), on the GPU busy {:.3f} "
                     "ms, gaps between them {:.3f} ms, overlaps {:.3f} ms, first start to the "
                     "writer's end {:.3f} ms | the same reads waited {:.3f} ms until the writer "
                     "ran",
                     KindNames[k], t.ahead_reads[k],
                     static_cast<double>(t.ahead_submissions[k]) / n,
                     static_cast<double>(t.ahead_resolved[k]) / n, ms(t.ahead_busy_ns[k]),
                     ms(t.ahead_gap_ns[k]), ms(t.ahead_overlap_ns[k]), ms(t.ahead_span_ns[k]),
                     ms(t.ahead_writer_ns[k]));
        }
    }
    t = TickCounts{};
}

/// FRITCHII_STORE_FOOTPRINT: a footprint as text: the modulus and the stored residues.
[[nodiscard]] inline std::string FootprintText(const Shader::StorageStoreFootprint& f) {
    if (!f.Valid()) {
        return fmt::format("none ({})", Shader::StoreFootprintWhyName(f.why));
    }
    std::string text;
    if (f.modulus == (u64{1} << 32)) {
        text = "exact offsets";
    } else {
        text = fmt::format("modulus {:#x}{}", f.modulus,
                           f.PowerOfTwo() ? "" : " (not a power of two: holds while the 32-bit "
                                                 "offset doesn't wrap)");
    }
    text += f.covers_all ? ", stores cover every residue (claims nothing):" : ", stored bytes";
    for (size_t i = 0; i < f.num_intervals && !f.covers_all; ++i) {
        text += fmt::format(" +{:#x}..+{:#x}", f.intervals[i].begin, f.intervals[i].end - 1);
    }
    text += fmt::format(" | {} stores, {}{}", f.stores,
                        f.ids_only ? "ids only" : "with loaded/cbuf/loop multiples",
                        f.widened ? ", gaps filled" : "");
    return text;
}

/// FRITCHII_STORE_FOOTPRINT: the "Store footprint" lines (LogDry, every 10 s).
inline void LogFootprint(double frames, double seconds) {
    std::unordered_map<Writer, FootStats, WriterHash> foot;
    {
        State& s = Get();
        std::scoped_lock lock{s.mutex};
        foot.swap(s.foot);
    }
    const auto ms_frame = [frames](u64 ns) { return static_cast<double>(ns) / 1e6 / frames; };
    FootStats total{};
    for (const auto& [writer, f] : foot) {
        for (size_t k = 0; k < FootReads; ++k) {
            total.reads[k] += f.reads[k];
            total.wait_ns[k] += f.wait_ns[k];
        }
        total.outside_compared += f.outside_compared;
        total.outside_differed += f.outside_differed;
        total.inside_compared += f.inside_compared;
        total.inside_differed += f.inside_differed;
    }
    u64 reads = 0;
    u64 wait_ns = 0;
    for (size_t k = 0; k < FootReads; ++k) {
        reads += total.reads[k];
        wait_ns += total.wait_ns[k];
    }
    // The footprints recorded at binds so far (the whole session)
    u32 recorded = 0, valid = 0, claims_nothing = 0, power_of_two = 0, ids_only = 0;
    std::array<u32, 16> whys{};
    {
        FootprintTable& t = GetFootprints();
        const u32 count = t.count.load(std::memory_order_acquire);
        for (u32 id = 1; id < count; ++id) {
            const Shader::StorageStoreFootprint& f = t.entries[id];
            ++recorded;
            if (!f.Valid()) {
                ++whys[static_cast<size_t>(f.why) & 15];
                continue;
            }
            ++valid;
            claims_nothing += f.covers_all ? 1 : 0;
            power_of_two += f.PowerOfTwo() ? 1 : 0;
            ids_only += f.ids_only ? 1 : 0;
        }
    }
    using Why = Shader::StoreFootprintWhy;
    const auto why_count = [&](Why why) { return whys[static_cast<size_t>(why)]; };
    constexpr auto O = static_cast<size_t>(FootRead::Outside);
    constexpr auto I = static_cast<size_t>(FootRead::Inside);
    constexpr auto CI = static_cast<size_t>(FootRead::CoveredInside);
    constexpr auto CN = static_cast<size_t>(FootRead::CoveredNoFootprint);
    constexpr auto N = static_cast<size_t>(FootRead::NoFootprint);
    LOG_INFO(HW_GPU,
             "Store footprint (FRITCHII_STORE_FOOTPRINT={}, dry) last {:.0f} s: {} reads to the "
             "back with a write mark over their bytes, waited {:.2f} ms a frame | outside the "
             "newest mark's footprint and no older mark covers them (would be answered from guest "
             "memory) {} ({:.2f} ms a frame) | inside the footprint {} ({:.2f}) | outside, but an "
             "older mark covers them: inside its footprint {} ({:.2f}), without a footprint {} "
             "({:.2f}) | newest mark without a footprint {} ({:.2f}) | must be 0: outside reads "
             "whose bytes changed at landing {} of {} compared (inside: {} of {}) | program "
             "bindings bound {}: with a footprint {} ({} claim nothing, {} power-of-two moduli, {} "
             "ids only), without: not recorded {}, a store stays global {}, pointer picked at run "
             "time {}, global stores {}, address not split {}, no single pointer term {}, no "
             "stores {}",
             g_store_footprint.Get(), seconds, reads, ms_frame(wait_ns), total.reads[O],
             ms_frame(total.wait_ns[O]), total.reads[I], ms_frame(total.wait_ns[I]),
             total.reads[CI], ms_frame(total.wait_ns[CI]), total.reads[CN],
             ms_frame(total.wait_ns[CN]), total.reads[N], ms_frame(total.wait_ns[N]),
             total.outside_differed, total.outside_compared, total.inside_differed,
             total.inside_compared, recorded, valid, claims_nothing, power_of_two, ids_only,
             why_count(Why::NotRecorded), why_count(Why::UntrackedWrites),
             why_count(Why::RuntimeRouted), why_count(Why::GlobalStore),
             why_count(Why::AddressNotTracked), why_count(Why::NoPointerTerm),
             why_count(Why::NoStores));
    std::vector<std::pair<const Writer*, const FootStats*>> order;
    order.reserve(foot.size());
    for (const auto& [writer, f] : foot) {
        order.emplace_back(&writer, &f);
    }
    const auto total_wait = [](const FootStats& f) {
        u64 ns = 0;
        for (size_t k = 0; k < FootReads; ++k) {
            ns += f.wait_ns[k];
        }
        return ns;
    };
    const size_t shown = (std::min<size_t>)(8, order.size());
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                      order.end(), [&](const auto& a, const auto& b) {
                          return total_wait(*a.second) > total_wait(*b.second);
                      });
    for (size_t i = 0; i < shown; ++i) {
        const Writer& writer = *order[i].first;
        const FootStats& f = *order[i].second;
        std::string footprint = "no footprint recorded at a bind (not a native written binding)";
        if (writer.kind == WriterKind::ComputeStorage) {
            if (const auto* recorded_footprint = FootprintOf(writer.program, writer.binding)) {
                footprint = FootprintText(*recorded_footprint);
            }
        }
        // The most common read residues
        std::vector<std::pair<u64, u64>> residues(f.residues.begin(), f.residues.end());
        const size_t top = (std::min<size_t>)(4, residues.size());
        std::partial_sort(residues.begin(), residues.begin() + static_cast<std::ptrdiff_t>(top),
                          residues.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string read_residues;
        for (size_t j = 0; j < top; ++j) {
            read_residues += fmt::format("{}+{:#x} ({})", j == 0 ? "" : ", ", residues[j].first,
                                         residues[j].second);
        }
        if (read_residues.empty()) {
            read_residues = "-";
        }
        const bool draw =
            writer.kind == WriterKind::DrawStorage || writer.kind == WriterKind::DrawTexture;
        const std::string binding = draw ? fmt::format("stage {} binding {}", writer.binding / 64,
                                                       writer.binding % 64)
                                         : fmt::format("binding {}", writer.binding);
        LOG_INFO(HW_GPU,
                 "Store footprint {}/{}: {} program {:016x} {} | waited {:.2f} ms a frame | "
                 "outside {} ({:.2f} ms a frame), inside {} ({:.2f}), covered by an older mark: "
                 "inside its footprint {} ({:.2f}) / without one {} ({:.2f}), no footprint {} "
                 "({:.2f}) | must be 0: outside reads changed at landing {} of {} (inside {} of "
                 "{}) | read offsets mod the modulus: {} | footprint: {}",
                 i + 1, shown, WriterKindName(writer.kind), writer.program, binding,
                 ms_frame(total_wait(f)), f.reads[O], ms_frame(f.wait_ns[O]), f.reads[I],
                 ms_frame(f.wait_ns[I]), f.reads[CI], ms_frame(f.wait_ns[CI]), f.reads[CN],
                 ms_frame(f.wait_ns[CN]), f.reads[N], ms_frame(f.wait_ns[N]),
                 f.outside_differed, f.outside_compared, f.inside_differed, f.inside_compared,
                 read_residues, footprint);
    }
}

/// BufferCache::TickFrame, every frame while the ring is on: the 10 s lines of the dry switches
/// FRITCHII_STORE_FOOTPRINT and FRITCHII_FEEDER_LANE (nothing while both are off).
inline void LogDry() {
    if (!g_store_footprint && !EarlyCompute::g_lane) {
        return;
    }
    static std::chrono::steady_clock::time_point start{};
    static u64 start_frame = 0;
    const auto now = std::chrono::steady_clock::now();
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (start == std::chrono::steady_clock::time_point{}) {
        start = now;
        start_frame = frame;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - start).count();
    if (seconds < 10.0) {
        return;
    }
    const double frames = static_cast<double>((std::max<u64>)(frame - start_frame, 1));
    start = now;
    start_frame = frame;
    if (g_store_footprint) {
        LogFootprint(frames, seconds);
    }
    if (EarlyCompute::g_lane) {
        EarlyCompute::LogLane(frames, seconds);
    }
}

/// BufferCache::TickFrame, every 600 frames (after advancing g_frame).
inline void Log(int run_flush_mode) {
    State& s = Get();
    std::scoped_lock lock{s.mutex};
    Counts& c = s.counts;
    c.distinct_pages += s.frame_pages.size();
    s.frame_pages.clear();
    s.frame = g_frame.load(std::memory_order_relaxed);
    const u64 reads = c.reads[0] + c.reads[1] + c.reads[2];
    constexpr double Frames = 600.0;
    LOG_INFO(HW_GPU,
             "Read runs (FRITCHII_READ_RUNS, EXP112={}) last 600 frames: {} reads to the back "
             "({:.1f} a frame): {} flushed, {} only waited for a read download on its way, {} "
             "waited for a fence download | next page {}, within 16 pages {}, same page {} | "
             "GPU-written run around the page: not in the buffer cache {}, 1 page {}, 2-16 {}, "
             "17-64 {}, more {} | waited {:.2f} ms a frame, {:.3f} ms a read, longest {:.2f} ms | "
             "{:.1f} distinct pages a frame, {} reads of a page already read that frame | span "
             "[{:#x}, {:#x}] | run flushes {} ({} KiB beyond the reads' pages) of {} read "
             "flushes looked at ({} with their own pages Preflushable); no run: not GPU-modified "
             "{}, no buffer {}, next page Preflushable {}, no GPU-written neighbour {}",
             run_flush_mode, reads, static_cast<double>(reads) / Frames, c.reads[0], c.reads[1],
             c.reads[2], c.next_page, c.near_pages, c.same_page, c.runs[0], c.runs[1], c.runs[2],
             c.runs[3], c.runs[4], static_cast<double>(c.wait_ns) / 1e6 / Frames,
             reads != 0 ? static_cast<double>(c.wait_ns) / 1e6 / static_cast<double>(reads) : 0.0,
             static_cast<double>(c.max_wait_ns) / 1e6,
             static_cast<double>(c.distinct_pages) / Frames, c.repeat_reads,
             reads != 0 ? c.lowest : 0, c.highest,
             g_run_flushes.exchange(0, std::memory_order_relaxed),
             g_run_bytes.exchange(0, std::memory_order_relaxed) >> 10,
             g_run_asked.exchange(0, std::memory_order_relaxed),
             g_run_own_marked.exchange(0, std::memory_order_relaxed),
             g_no_run[0].exchange(0, std::memory_order_relaxed),
             g_no_run[1].exchange(0, std::memory_order_relaxed),
             g_no_run[2].exchange(0, std::memory_order_relaxed),
             g_no_run[3].exchange(0, std::memory_order_relaxed));
    constexpr size_t B = static_cast<size_t>(Overlap::Bytes);
    constexpr size_t P = static_cast<size_t>(Overlap::PageOnly);
    constexpr size_t U = static_cast<size_t>(Overlap::Unknown);
    const auto ms_frame = [&](size_t o) {
        return static_cast<double>(c.overlap_wait_ns[o]) / 1e6 / Frames;
    };
    const auto ms_read = [&](size_t o) {
        const u64 n = c.overlap[0][o] + c.overlap[1][o] + c.overlap[2][o];
        return n != 0 ? static_cast<double>(c.overlap_wait_ns[o]) / 1e6 / static_cast<double>(n)
                      : 0.0;
    };
    LOG_INFO(HW_GPU,
             "Read overlap (FRITCHII_READ_RUNS) last 600 frames, the read's own bytes against "
             "the GPU-written bytes in its pages it waits for (fence download: its pending "
             "download ranges; read download: read_download_ranges; flushed: gpu_modified_ranges) "
             "as overlap/page only/unknown: fence download {}/{}/{}, read download {}/{}/{}, "
             "flushed {}/{}/{} | page only, gap to the nearest GPU-written byte: <64 B {}, "
             "<256 B {}, <1 KiB {}, <4 KiB {}, more {} | read size <=16 B {}, <=64 {}, <=256 "
             "{}, <=4096 {}, more {} | waited: overlap {:.2f} ms a frame ({:.3f} ms a read), "
             "page only {:.2f} ({:.3f}), unknown {:.2f} ({:.3f}) | gpu_modified_ranges answers "
             "otherwise for {} of the waits",
             c.overlap[2][B], c.overlap[2][P], c.overlap[2][U], c.overlap[1][B], c.overlap[1][P],
             c.overlap[1][U], c.overlap[0][B], c.overlap[0][P], c.overlap[0][U], c.gaps[0],
             c.gaps[1], c.gaps[2], c.gaps[3], c.gaps[4], c.sizes[0], c.sizes[1], c.sizes[2],
             c.sizes[3], c.sizes[4], ms_frame(B), ms_read(B), ms_frame(P), ms_read(P),
             ms_frame(U), ms_read(U), c.modified_differs);
    LogWriters(s, Frames);
    if (g_writer_ticks) {
        LogTicks(s, Frames);
    }
    if (EarlyCompute::g_early || EarlyCompute::g_side_copy) {
        EarlyCompute::Log(Frames);
    }
    c = Counts{};
}

} // namespace VideoCommon::ReadRunStats
