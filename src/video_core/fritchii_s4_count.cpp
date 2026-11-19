// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/back_read_audit.h"
#include "video_core/fritchii_s4_count.h"
#include "video_core/gpu_split.h"
#include "video_core/one_follower_hooks.h"
#include "video_core/split_write_tracker.h"

namespace VideoCommon::S4Count {

namespace {

using Common::S4Count::GenSum;
using Common::S4Count::Writer;
using Clock = std::chrono::steady_clock;

constexpr u64 PageBytes = 4096;
/// Ranges up to this size are hashed whole at every bind (row 4); larger ones (native banks)
/// one page in BankPageSample, chosen by address so the same pages are always the ones watched.
constexpr u32 FullHashBytes = 16384;
constexpr u64 BankPageSample = 64;
/// Value 2's sample: one list entry in SampleEvery, up to SampleRanges ranges of at most
/// SampleBytes each (the front hashes them with its queued writes laid over).
constexpr u64 SampleEvery = 256;
constexpr u32 SampleRanges = 8;
constexpr u32 SampleBytes = 4096;
constexpr size_t RecentRanges = 64;
constexpr size_t EntryRingSize = size_t{1} << 16;
constexpr size_t SampleRingSize = 1024;

[[nodiscard]] u64 Mix(u64 value) noexcept {
    return (value * 0x9E3779B97F4A7C15ULL) >> 32;
}

[[nodiscard]] u32 SampleClip(u32 size) noexcept {
    return (std::min)(size, SampleBytes) & ~3U;
}

/// Reads [address, address + size) of game memory into `out` page by page (hidden from the back
/// read audit: the counter's reads are not the fill's). False if a page is unmapped.
bool ReadRange(Tegra::MaxwellDeviceMemoryManager& memory, u64 address, u32 size, u8* out) {
    if (size == 0 || ((address + size - 1) >> 36) != 0) {
        return false;
    }
    return BackReadAudit::Unaudited([&]() -> bool {
        u64 at = address;
        u32 left = size;
        while (left != 0) {
            const u32 chunk =
                static_cast<u32>((std::min<u64>)(left, PageBytes - (at & (PageBytes - 1))));
            const u8* const pointer = memory.GetPointer<u8>(at);
            if (pointer == nullptr) {
                return false;
            }
            std::memcpy(out, pointer, chunk);
            out += chunk;
            at += chunk;
            left -= chunk;
        }
        return true;
    });
}

[[nodiscard]] u64 HashBytes(const u8* bytes, u32 size) noexcept {
    return Common::CityHash64(reinterpret_cast<const char*>(bytes), size);
}

using Gens = std::array<u64, 3>;

[[nodiscard]] Gens ReadGens(u64 address, u32 size) noexcept {
    return {GenSum(Writer::Stream, address, size), GenSum(Writer::Gpu, address, size),
            GenSum(Writer::Other, address, size)};
}

// --- The front's side: list entry positions and value 2's samples ------------------------------

/// The front's ring position at the start of each list entry it parsed (one writer, the front;
/// E reads up to g_entry_head). `sample`: 1 + the index of value 2's sample taken there, or 0.
struct EntryMark {
    u64 position;
    u32 sample;
};
std::array<EntryMark, EntryRingSize> g_entries{};
std::atomic<u64> g_entry_head{0};

enum class SampleKind : u8 {
    Ready,         ///< hashed: E compares at its bind of the range in this entry
    Queued,        ///< a queued upload the front noted writes the range (a mirror applies it)
    QueuedUnknown, ///< queued words over it the front cannot lay over (OverlayQueuedWrites)
    Unmapped,
};

struct RangeSample {
    u64 address{};
    u64 hash{};
    Gens gens{};
    u32 size{};
    SampleKind kind{};
    bool unknown_writer{}; ///< a queued writer of unknown range was not run yet at the parse
};

struct Sample {
    u32 count{};
    std::array<RangeSample, SampleRanges> ranges{};
};
/// Written by the front at sample N % size, read by E when it crosses that entry (E is far less
/// than SampleRingSize samples, 262,144 entries, behind).
std::array<Sample, SampleRingSize> g_samples{};

/// Ranges E bound lately (address | size << 40), for the front to sample.
std::array<std::atomic<u64>, RecentRanges> g_recent{};

struct FrontState {
    u64 entries{};
    u64 samples{};
    u32 rotate{};
    std::array<u32, SampleBytes / 4> words{};
};
FrontState g_front;

u32 TakeSample(GpuSplit& split, Tegra::MaxwellDeviceMemoryManager& memory) {
    FrontState& f = g_front;
    const u64 index = f.samples % SampleRingSize;
    Sample& sample = g_samples[index];
    sample.count = 0;
    // What E had run when guest memory is read: writes up to there are in guest memory, later
    // ones the front either lays over (their words are known) or marks.
    const u64 since = split.EngineConsumedNow();
    const SplitWriteTracker& writes = split.Writes();
    const bool unknown_writer = writes.LastUnknown() > since;
    const u32 rotate = f.rotate++;
    for (u32 i = 0; i < SampleRanges; ++i) {
        const u64 key =
            g_recent[(rotate + i * (RecentRanges / SampleRanges)) % RecentRanges].load(
                std::memory_order_relaxed);
        if (key == 0) {
            continue;
        }
        const u64 address = key & ((u64{1} << 40) - 1);
        const u32 size = static_cast<u32>(key >> 40);
        bool seen = false;
        for (u32 j = 0; j < sample.count; ++j) {
            seen |= sample.ranges[j].address == address && sample.ranges[j].size == size;
        }
        if (seen || size == 0) {
            continue;
        }
        RangeSample& range = sample.ranges[sample.count++];
        range = RangeSample{};
        range.address = address;
        range.size = size;
        range.gens = ReadGens(address, size);
        range.unknown_writer = unknown_writer;
        if (writes.LastWriteTo(address, address + size) > since) {
            range.kind = SampleKind::Queued;
            continue;
        }
        u8* const bytes = reinterpret_cast<u8*>(f.words.data());
        if (!ReadRange(memory, address, size, bytes)) {
            range.kind = SampleKind::Unmapped;
            continue;
        }
        if (!split.OverlayQueuedWrites(since, address, f.words.data(), size / 4)) {
            range.kind = SampleKind::QueuedUnknown;
            continue;
        }
        range.hash = HashBytes(bytes, size);
        range.kind = SampleKind::Ready;
    }
    ++f.samples;
    return static_cast<u32>(index + 1);
}

// --- E's side ----------------------------------------------------------------------------------

enum PageClass : size_t {
    Tracked,    ///< the tracker holds it clean: a guest store traps and flags it
    Untrapped,  ///< CPU-modified: guest stores land unseen (a mirror must copy it without a flag)
    Stream,     ///< written by E's stream uploads while bound (the driver banks: by value)
    GpuWritten, ///< holds GPU-written data (the cached path binds the host buffer)
    PageClassCount,
};

/// Open-addressed set of device pages with two bits and a stream generation each (E alone).
class PageSet {
public:
    explicit PageSet(u32 bits_log2)
        : mask{(size_t{1} << bits_log2) - 1}, keys(mask + 1), flags(mask + 1), gens(mask + 1) {}

    struct Slot {
        u8* flags;
        u32* gen;
        bool inserted;
    };

    /// nullptr flags: the set is full (counted as overflowed).
    Slot Find(u64 page) {
        const u64 key = page + 1;
        size_t i = Mix(page) & mask;
        for (size_t probe = 0; probe <= mask; ++probe, i = (i + 1) & mask) {
            if (keys[i] == key) {
                return {&flags[i], &gens[i], false};
            }
            if (keys[i] == 0) {
                if (count >= (mask + 1) / 4 * 3) {
                    overflowed = true;
                    return {nullptr, nullptr, false};
                }
                keys[i] = key;
                flags[i] = 0;
                gens[i] = 0;
                ++count;
                return {&flags[i], &gens[i], true};
            }
        }
        overflowed = true;
        return {nullptr, nullptr, false};
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (size_t i = 0; i <= mask; ++i) {
            if (keys[i] != 0) {
                func(keys[i] - 1, flags[i], gens[i]);
            }
        }
    }

    void Clear() {
        if (count != 0) {
            std::fill(keys.begin(), keys.end(), u64{0});
        }
        count = 0;
        overflowed = false;
    }

    [[nodiscard]] size_t Count() const noexcept {
        return count;
    }
    [[nodiscard]] bool Overflowed() const noexcept {
        return overflowed;
    }

private:
    size_t mask;
    std::vector<u64> keys;
    std::vector<u8> flags;
    std::vector<u32> gens;
    size_t count{};
    bool overflowed{};
};

constexpr u8 FlagCpu = 1;
constexpr u8 FlagGpu = 2;

[[nodiscard]] PageClass ClassOf(u64 page, u8 flags, u32 gen_at_first) noexcept {
    const u32 now = Common::S4Count::g_gens[static_cast<size_t>(Writer::Stream)]
                                           [Common::S4Count::GenBucket(page)]
                                               .load(std::memory_order_relaxed);
    if (now != gen_at_first) {
        return Stream;
    }
    if ((flags & FlagGpu) != 0) {
        return GpuWritten;
    }
    return (flags & FlagCpu) != 0 ? Untrapped : Tracked;
}

/// Why a range's bytes changed between two binds (row 4) or between the front's parse and E's
/// bind (value 2).
enum Cause : size_t {
    CauseStream,     ///< E's stream uploads wrote the range between (a mirror applies them)
    CauseGpu,        ///< a GPU result landed on it (download, report or semaphore off E)
    CauseOther,      ///< an emulator write on another thread
    CauseUnknown,    ///< (value 2) a queued writer of unknown range was not run at the parse
    CauseTrapped,    ///< no writer seen: a guest store on a page the tracker held clean
    CauseUntrapped,  ///< no writer seen: a guest store on a CPU-modified page
    CauseCount,
};

struct Memo {
    u64 hash{};
    u64 entry{};
    Gens gens{};
    bool tracked{};
};

struct Window {
    // Row 1.
    u64 binds{};
    u64 stream_binds{};
    u64 cached_binds{};
    u64 bank_binds{};
    u64 stream_bytes{};
    u64 cached_upload_bytes{};
    u64 bank_bytes{};
    // Row 2.
    std::array<u64, PageClassCount> page_seconds{};
    u64 seconds{};
    bool page_overflow{};
    // Row 3.
    u64 entries{};
    u64 entries_lost{};
    u64 front_head_at_start{};
    std::array<u64, PageClassCount> page_entries{};
    u64 entry_page_overflow{};
    // Row 4.
    u64 again_in_entry{};
    u64 again_later{};
    std::array<u64, CauseCount> changed{};
    u64 bank_again{};
    u64 bank_changed_stream{};
    u64 bank_changed_other{};
    u64 gpu_range_binds{};
    u64 unread{};
    // Value 2.
    u64 s_ranges{};
    u64 s_compared{};
    u64 s_same{};
    std::array<u64, CauseCount> s_differ{};
    u64 s_queued{};
    u64 s_queued_unknown{};
    u64 s_unmapped{};
    u64 s_unbound{};
    u64 s_gpu_range{};
};

struct EngineState {
    u32 epoch{~0U};
    Clock::time_point window_start{};
    Clock::time_point second_start{};
    u64 tail{};
    u64 entry_index{};
    u64 last_back{};
    u64 checks{};
    u64 recent{};
    PageSet second_pages{16};
    PageSet window_pages{17};
    // The pages bound in the current entry: (page + 1, entry + 1) slots, no clearing needed.
    std::array<u64, 1024> entry_keys{};
    std::array<u64, 1024> entry_stamps{};
    u32 entry_count{};
    std::unordered_map<u64, Memo> memos;
    u32 active_sample{};
    u32 matched{};
    std::array<u8, FullHashBytes> bytes{};
    Window w;
    u64 session_changed{};
    u64 session_front{};
};

std::atomic<u32> g_epoch{0};
std::atomic<bool> g_busy{false};
/// Binds not counted because another thread held the counter (any thread adds).
std::atomic<u64> g_busy_binds{0};
EngineState* g_engine = nullptr;

EngineState& Engine() {
    if (g_engine == nullptr) {
        g_engine = new EngineState();
        g_engine->memos.reserve(size_t{1} << 16);
    }
    return *g_engine;
}

void ResetWindow(EngineState& e, Clock::time_point now) {
    e.w = Window{};
    g_busy_binds.store(0, std::memory_order_relaxed);
    e.w.front_head_at_start = g_entry_head.load(std::memory_order_acquire);
    e.window_start = now;
    e.second_start = now;
    e.second_pages.Clear();
    e.window_pages.Clear();
}

/// Resolves the active sample's ranges that were never bound in their entry.
void EndSample(EngineState& e) {
    if (e.active_sample == 0) {
        return;
    }
    const Sample& sample = g_samples[e.active_sample - 1];
    for (u32 i = 0; i < sample.count; ++i) {
        if (((e.matched >> i) & 1) == 0) {
            ++e.w.s_unbound;
        }
    }
    e.active_sample = 0;
    e.matched = 0;
}

void BeginSample(EngineState& e, u32 sample_index) {
    e.active_sample = sample_index;
    e.matched = 0;
    const Sample& sample = g_samples[sample_index - 1];
    e.w.s_ranges += sample.count;
    for (u32 i = 0; i < sample.count; ++i) {
        switch (sample.ranges[i].kind) {
        case SampleKind::Ready:
            continue;
        case SampleKind::Queued:
            ++e.w.s_queued;
            break;
        case SampleKind::QueuedUnknown:
            ++e.w.s_queued_unknown;
            break;
        case SampleKind::Unmapped:
            ++e.w.s_unmapped;
            break;
        }
        e.matched |= 1U << i;
    }
}

/// Crosses the list entries whose start E's current ring position passed.
void CrossEntries(EngineState& e) {
    const u64 back = GpuSplit::Get().BackEntryPosition();
    const u64 head = g_entry_head.load(std::memory_order_acquire);
    if (back < e.last_back) {
        // A new split session starts its positions over: the marks before it mean nothing now.
        EndSample(e);
        e.tail = head;
    }
    e.last_back = back;
    if (head - e.tail > EntryRingSize) {
        e.w.entries_lost += head - e.tail - EntryRingSize;
        e.tail = head - EntryRingSize;
    }
    while (e.tail < head) {
        const EntryMark mark = g_entries[e.tail % EntryRingSize];
        if (mark.position > back) {
            break;
        }
        ++e.tail;
        ++e.entry_index;
        ++e.w.entries;
        e.entry_count = 0;
        EndSample(e);
        if (mark.sample != 0) {
            BeginSample(e, mark.sample);
        }
    }
}

/// Whether `page` is new in the current entry (and records it).
bool NewInEntry(EngineState& e, u64 page) {
    const u64 key = page + 1;
    const u64 stamp = e.entry_index + 1;
    size_t i = Mix(page) & (e.entry_keys.size() - 1);
    for (size_t probe = 0; probe < e.entry_keys.size();
         ++probe, i = (i + 1) & (e.entry_keys.size() - 1)) {
        if (e.entry_stamps[i] != stamp) {
            if (e.entry_count >= e.entry_keys.size() / 4 * 3) {
                ++e.w.entry_page_overflow;
                return false;
            }
            e.entry_stamps[i] = stamp;
            e.entry_keys[i] = key;
            ++e.entry_count;
            return true;
        }
        if (e.entry_keys[i] == key) {
            return false;
        }
    }
    ++e.w.entry_page_overflow;
    return false;
}

[[nodiscard]] Cause CauseOf(const Gens& before, const Gens& now, bool unknown_writer,
                            bool tracked_before) noexcept {
    if (now[1] != before[1]) {
        return CauseGpu;
    }
    if (now[2] != before[2]) {
        return CauseOther;
    }
    if (now[0] != before[0]) {
        return CauseStream;
    }
    if (unknown_writer) {
        return CauseUnknown;
    }
    return tracked_before ? CauseTrapped : CauseUntrapped;
}

/// Row 4 for one hashed piece (a whole small range, or one sampled page of a large one).
/// Returns the cause if its bytes changed inside the entry, CauseCount otherwise.
Cause CompareMemo(EngineState& e, u64 key, u64 hash, const Gens& gens, bool tracked, bool& again,
                  bool& later) {
    auto [it, inserted] = e.memos.try_emplace(key);
    Cause cause = CauseCount;
    again = false;
    later = false;
    if (!inserted) {
        const Memo& memo = it->second;
        if (memo.entry == e.entry_index) {
            again = true;
            if (memo.hash != hash) {
                cause = CauseOf(memo.gens, gens, false, memo.tracked);
            }
        } else {
            later = true;
        }
    }
    it->second = Memo{hash, e.entry_index, gens, tracked};
    return cause;
}

void FoldSecond(EngineState& e) {
    e.second_pages.ForEach([&](u64 page, u8 flags, u32 gen) {
        ++e.w.page_seconds[ClassOf(page, flags, gen)];
    });
    e.w.page_overflow |= e.second_pages.Overflowed();
    ++e.w.seconds;
    e.second_pages.Clear();
}

void Log(EngineState& e, double seconds) {
    const Window& w = e.w;
    const int mode = Common::S4Count::g_mode.load(std::memory_order_relaxed);
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    const auto mbs = [seconds](u64 bytes) {
        return static_cast<double>(bytes) / seconds / (1024.0 * 1024.0);
    };
    const double folded = w.seconds != 0 ? static_cast<double>(w.seconds) : 1.0;
    const auto per_second = [folded](u64 count) { return static_cast<double>(count) / folded; };
    const u64 front_entries =
        g_entry_head.load(std::memory_order_acquire) - w.front_head_at_start;
    u64 page_entry_total = 0;
    for (const u64 count : w.page_entries) {
        page_entry_total += count;
    }
    const double entries_rate = rate(w.entries);
    const double untrapped_pages = per_second(w.page_seconds[Untrapped]);
    LOG_INFO(HW_GPU,
             "S4 count (FRITCHII_S4_COUNT={}) 10 s: uniform binds {:.0f}/s (stream path {:.0f}, "
             "cached path {:.0f}; native banks among them {:.0f}; {:.0f}/s not counted, the "
             "counter busy on another thread); bytes the fill reads and compares for them: "
             "stream path {:.1f} MB/s read, about as much compared; cached path uploads {:.1f} "
             "MB/s; bank check over {:.1f} MB/s of bank range (its compare's upper bound); list "
             "entries {:.0f}/s crossed by E (front parsed {:.0f}/s, {} lost); distinct 4 KiB "
             "pages a second {:.0f}: tracked {:.0f}, untrapped {:.0f}, stream-written (driver "
             "banks) {:.0f}, GPU-written {:.0f}; in 10 s {}{}; page-entries {:.0f}/s (tracked "
             "{:.0f}, untrapped {:.0f}, stream {:.0f}, GPU {:.0f}{}); mirror upkeep: the pages "
             "each entry binds {:.1f} MB/s, every untrapped page every entry {:.1f} MB/s",
             mode, rate(w.binds), rate(w.stream_binds), rate(w.cached_binds), rate(w.bank_binds),
             rate(g_busy_binds.load(std::memory_order_relaxed)), mbs(w.stream_bytes),
             mbs(w.cached_upload_bytes), mbs(w.bank_bytes),
             entries_rate, rate(front_entries), w.entries_lost,
             per_second(w.page_seconds[Tracked] + w.page_seconds[Untrapped] +
                        w.page_seconds[Stream] + w.page_seconds[GpuWritten]),
             per_second(w.page_seconds[Tracked]), untrapped_pages,
             per_second(w.page_seconds[Stream]), per_second(w.page_seconds[GpuWritten]),
             e.window_pages.Count(),
             (w.page_overflow || e.window_pages.Overflowed()) ? " (set full: more)" : "",
             rate(page_entry_total), rate(w.page_entries[Tracked]),
             rate(w.page_entries[Untrapped]), rate(w.page_entries[Stream]),
             rate(w.page_entries[GpuWritten]),
             w.entry_page_overflow != 0 ? ", some entries over 768 pages" : "",
             mbs(page_entry_total * PageBytes),
             entries_rate * untrapped_pages * static_cast<double>(PageBytes) /
                 (1024.0 * 1024.0));
    u64 bad = 0;
    for (size_t cause = 0; cause < CauseCount; ++cause) {
        if (cause != CauseStream) {
            bad += w.changed[cause];
        }
    }
    bad += w.bank_changed_other;
    const u64 bank_scale = BankPageSample;
    std::string front_part;
    if (mode >= 2) {
        u64 front_bad = 0;
        u64 differ = 0;
        for (size_t cause = 0; cause < CauseCount; ++cause) {
            differ += w.s_differ[cause];
            if (cause != CauseStream && cause != CauseUnknown) {
                front_bad += w.s_differ[cause];
            }
        }
        front_part = fmt::format(
            "; front parse vs E fill (1 entry in {}): ranges sampled {:.0f}/s, compared {:.0f}/s, "
            "same {:.0f}/s, differ {:.0f}/s (stream upload {:.0f}, GPU write {:.0f}, emulator "
            "write off E {:.0f}, a queued writer of unknown range {:.0f}, guest store {:.0f}); "
            "not compared: a queued upload on the range at the parse {:.0f}/s, queued words "
            "unknown {:.0f}/s, unmapped {:.0f}/s, GPU-written range {:.0f}/s, not bound in its "
            "entry {:.0f}/s; MUST BE 0: differ by GPU, emulator or guest write {:.0f}/s ({} this "
            "session)",
            SampleEvery, rate(w.s_ranges), rate(w.s_compared), rate(w.s_same), rate(differ),
            rate(w.s_differ[CauseStream]), rate(w.s_differ[CauseGpu]),
            rate(w.s_differ[CauseOther]), rate(w.s_differ[CauseUnknown]),
            rate(w.s_differ[CauseTrapped] + w.s_differ[CauseUntrapped]), rate(w.s_queued),
            rate(w.s_queued_unknown), rate(w.s_unmapped), rate(w.s_gpu_range),
            rate(w.s_unbound), rate(front_bad), e.session_front);
    }
    LOG_INFO(HW_GPU,
             "S4 count (FRITCHII_S4_COUNT={}) 10 s: same range bound again in its list entry "
             "{:.0f}/s (in a later entry {:.0f}/s); its bytes changed inside the entry {:.0f}/s: "
             "stream upload {:.0f}, GPU write {:.0f}, emulator write off E {:.0f}, guest store "
             "on a tracked page {:.0f}, guest store on an untrapped page {:.0f}; ranges over 16 "
             "KiB, 1 page in {} watched (x{}): again in the entry {:.0f}/s, changed by stream "
             "{:.0f}/s, by anything else {:.0f}/s; binds on GPU-written ranges (host data, not "
             "compared) {:.0f}/s, unreadable {:.0f}/s; MUST BE 0 for S4 to be exact: changed "
             "inside the entry outside stream uploads {:.0f}/s ({} this session){}",
             mode, rate(w.again_in_entry), rate(w.again_later),
             rate(w.changed[CauseStream] + w.changed[CauseGpu] + w.changed[CauseOther] +
                  w.changed[CauseTrapped] + w.changed[CauseUntrapped]),
             rate(w.changed[CauseStream]), rate(w.changed[CauseGpu]),
             rate(w.changed[CauseOther]), rate(w.changed[CauseTrapped]),
             rate(w.changed[CauseUntrapped]), BankPageSample, bank_scale,
             rate(w.bank_again * bank_scale), rate(w.bank_changed_stream * bank_scale),
             rate(w.bank_changed_other * bank_scale), rate(w.gpu_range_binds), rate(w.unread),
             rate(bad), e.session_changed, front_part);
}

void MaybeTick(EngineState& e) {
    if ((++e.checks & 255) != 0) {
        return;
    }
    const auto now = Clock::now();
    if (now - e.second_start >= std::chrono::seconds{1}) {
        FoldSecond(e);
        e.second_start = now;
    }
    if (now - e.window_start >= std::chrono::seconds{10}) {
        Log(e, std::chrono::duration<double>(now - e.window_start).count());
        ResetWindow(e, now);
        if (e.memos.size() > (size_t{1} << 18)) {
            e.memos.clear();
        }
    }
}

void CountPages(EngineState& e, const BindFacts& facts, PageTest cpu_page, PageTest gpu_page,
                void* context) {
    const u64 first = facts.device_addr / PageBytes;
    const u64 last = (facts.device_addr + facts.size - 1) / PageBytes;
    for (u64 page = first; page <= last; ++page) {
        const PageSet::Slot slot = e.second_pages.Find(page);
        if (slot.flags != nullptr && slot.inserted) {
            *slot.flags = static_cast<u8>((cpu_page(context, page * PageBytes) ? FlagCpu : 0) |
                                          (gpu_page(context, page * PageBytes) ? FlagGpu : 0));
            *slot.gen = Common::S4Count::g_gens[static_cast<size_t>(Writer::Stream)]
                                               [Common::S4Count::GenBucket(page)]
                                                   .load(std::memory_order_relaxed);
        }
        (void)e.window_pages.Find(page);
        if (NewInEntry(e, page)) {
            const PageClass page_class = slot.flags != nullptr
                                             ? ClassOf(page, *slot.flags, *slot.gen)
                                             : (facts.cpu_modified ? Untrapped : Tracked);
            ++e.w.page_entries[page_class];
        }
    }
}

void CompareChanges(EngineState& e, Tegra::MaxwellDeviceMemoryManager& memory,
                    const BindFacts& facts, bool& have_bytes) {
    have_bytes = false;
    const bool tracked = !facts.cpu_modified;
    if (facts.size <= FullHashBytes) {
        if (!ReadRange(memory, facts.device_addr, facts.size, e.bytes.data())) {
            ++e.w.unread;
            return;
        }
        have_bytes = true;
        const u64 hash = HashBytes(e.bytes.data(), facts.size);
        const Gens gens = ReadGens(facts.device_addr, facts.size);
        const u64 key = facts.device_addr | (u64{facts.size} << 40);
        bool again = false;
        bool later = false;
        const Cause cause = CompareMemo(e, key, hash, gens, tracked, again, later);
        e.w.again_in_entry += again ? 1 : 0;
        e.w.again_later += later ? 1 : 0;
        if (cause != CauseCount) {
            ++e.w.changed[cause];
            if (cause != CauseStream) {
                ++e.session_changed;
            }
        }
        return;
    }
    // A large range (native bank): its watched pages only, each as its own piece.
    const u64 end = facts.device_addr + facts.size;
    for (u64 page = facts.device_addr & ~(PageBytes - 1); page < end; page += PageBytes) {
        if (Mix(page / PageBytes) % BankPageSample != 0) {
            continue;
        }
        const u64 begin = (std::max)(page, facts.device_addr);
        const u32 bytes = static_cast<u32>((std::min)(page + PageBytes, end) - begin);
        if (!ReadRange(memory, begin, bytes, e.bytes.data())) {
            ++e.w.unread;
            continue;
        }
        const u64 hash = HashBytes(e.bytes.data(), bytes);
        const Gens gens = ReadGens(begin, bytes);
        const u64 key = begin | (u64{bytes} << 40) | (u64{1} << 63);
        bool again = false;
        bool later = false;
        const Cause cause = CompareMemo(e, key, hash, gens, tracked, again, later);
        e.w.bank_again += again ? 1 : 0;
        if (cause == CauseStream) {
            ++e.w.bank_changed_stream;
        } else if (cause != CauseCount) {
            ++e.w.bank_changed_other;
            ++e.session_changed;
        }
    }
}

void CompareSample(EngineState& e, Tegra::MaxwellDeviceMemoryManager& memory,
                   const BindFacts& facts, bool have_bytes) {
    const Sample& sample = g_samples[e.active_sample - 1];
    const u32 clip = SampleClip(facts.size);
    for (u32 i = 0; i < sample.count; ++i) {
        const RangeSample& range = sample.ranges[i];
        if (((e.matched >> i) & 1) != 0 || range.address != facts.device_addr ||
            range.size != clip) {
            continue;
        }
        e.matched |= 1U << i;
        if (facts.gpu_modified) {
            ++e.w.s_gpu_range;
            continue;
        }
        // The row 4 read holds the range's first bytes when it read the whole range.
        if (!have_bytes && !ReadRange(memory, range.address, clip, e.bytes.data())) {
            ++e.w.s_unmapped;
            continue;
        }
        ++e.w.s_compared;
        if (HashBytes(e.bytes.data(), clip) == range.hash) {
            ++e.w.s_same;
            continue;
        }
        const Cause cause = CauseOf(range.gens, ReadGens(range.address, clip),
                                    range.unknown_writer, !facts.cpu_modified);
        ++e.w.s_differ[cause];
        if (cause != CauseStream && cause != CauseUnknown) {
            ++e.session_front;
        }
    }
}

} // Anonymous namespace

void SetMode(int mode) noexcept {
    Common::S4Count::g_mode.store(mode, std::memory_order_relaxed);
    Common::S4Count::g_on.store(mode != 0, std::memory_order_relaxed);
    if (mode != 0) {
        Common::S4Count::g_entry_users.fetch_or(Common::S4Count::EntryUserCounter,
                                                std::memory_order_relaxed);
    } else {
        Common::S4Count::g_entry_users.fetch_and(~Common::S4Count::EntryUserCounter,
                                                 std::memory_order_relaxed);
    }
    g_epoch.fetch_add(1, std::memory_order_relaxed);
}

namespace {
/// FRITCHII_BIND_SKIP's own crossing of the front's entry marks (E alone; the counter's state
/// above is its own and may be off).
struct SkipEntryClock {
    u64 tail{};
    u64 last_back{};
    u64 index{1};
    bool started{};
};
SkipEntryClock g_skip_clock;
} // Anonymous namespace

void SetBindSkipEntries(bool want) noexcept {
    if (want) {
        Common::S4Count::g_entry_users.fetch_or(Common::S4Count::EntryUserBindSkip,
                                                std::memory_order_seq_cst);
    } else {
        Common::S4Count::g_entry_users.fetch_and(~Common::S4Count::EntryUserBindSkip,
                                                 std::memory_order_seq_cst);
    }
}

u64 EngineEntryIndex() noexcept {
    // E only: not R in a lent call, not the memory stage's thread.
    if (!GpuSplit::OnBackThread() || OneFollower::Detail::t_render) {
        return 0;
    }
    GpuSplit& split = GpuSplit::Get();
    if (!split.Active()) {
        return 0;
    }
    SkipEntryClock& c = g_skip_clock;
    const u64 back = split.BackEntryPosition();
    const u64 head = g_entry_head.load(std::memory_order_acquire);
    if (!c.started || back < c.last_back) {
        // First use, or a new split session (its positions start over): marks before now mean
        // nothing. A new index, so no memo from before matches.
        c.started = true;
        c.tail = head;
        c.index += 2;
    }
    c.last_back = back;
    if (head - c.tail > EntryRingSize) {
        // Marks overwritten before E crossed them: their entries are crossed all the same.
        c.index += head - c.tail - EntryRingSize;
        c.tail = head - EntryRingSize;
    }
    while (c.tail < head) {
        if (g_entries[c.tail % EntryRingSize].position > back) {
            break;
        }
        ++c.tail;
        ++c.index;
    }
    return c.index;
}

void Bind(Tegra::MaxwellDeviceMemoryManager& memory, const BindFacts& facts, PageTest cpu_page,
          PageTest gpu_page, void* context) {
    if (facts.size == 0 || facts.device_addr == 0) {
        return;
    }
    if (g_busy.exchange(true, std::memory_order_acquire)) {
        // Another thread binds at the same time (the memory stage's own thread): not counted.
        g_busy_binds.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    EngineState& e = Engine();
    // E's writes are stream uploads from here on (R binds only in lent calls, E parked).
    if (OneFollower::Detail::t_engine ||
        (GpuSplit::OnBackThread() && !OneFollower::Detail::t_render)) {
        Common::S4Count::t_engine = true;
    } else if (OneFollower::Detail::t_render) {
        Common::S4Count::t_render = true;
    }
    const u32 epoch = g_epoch.load(std::memory_order_relaxed);
    if (epoch != e.epoch) {
        // Switched on (or to another value): start clean.
        e.epoch = epoch;
        e.memos.clear();
        e.active_sample = 0;
        e.matched = 0;
        e.tail = g_entry_head.load(std::memory_order_acquire);
        e.last_back = GpuSplit::Get().BackEntryPosition();
        e.session_changed = 0;
        e.session_front = 0;
        ResetWindow(e, Clock::now());
    }
    CrossEntries(e);
    Window& w = e.w;
    // Row 1: today's binds and bytes.
    ++w.binds;
    if (facts.stream) {
        ++w.stream_binds;
        w.stream_bytes += facts.size;
    } else {
        ++w.cached_binds;
        if (facts.cpu_modified) {
            w.cached_upload_bytes += facts.size;
        }
    }
    if (facts.bank) {
        ++w.bank_binds;
        w.bank_bytes += facts.size;
    }
    // Rows 2 and 3: pages by class, a second, and per entry.
    CountPages(e, facts, cpu_page, gpu_page, context);
    // Row 4: bytes changed between two binds of the same range inside one entry.
    bool have_bytes = false;
    if (facts.gpu_modified) {
        ++w.gpu_range_binds;
    } else {
        CompareChanges(e, memory, facts, have_bytes);
    }
    // Value 2: the front's sample of this entry.
    if (e.active_sample != 0) {
        CompareSample(e, memory, facts, have_bytes);
    }
    // Ranges for the front to sample (value 2), aligned for its word overlay.
    if (!facts.gpu_modified && (facts.device_addr & 3) == 0 && SampleClip(facts.size) != 0) {
        g_recent[e.recent++ % RecentRanges].store(
            facts.device_addr | (u64{SampleClip(facts.size)} << 40), std::memory_order_relaxed);
    }
    MaybeTick(e);
    g_busy.store(false, std::memory_order_release);
}

void FrontEntry(GpuSplit& split, Tegra::MaxwellDeviceMemoryManager& memory) {
    u32 sample = 0;
    if (Common::S4Count::g_mode.load(std::memory_order_relaxed) >= 2 &&
        (++g_front.entries % SampleEvery) == 0) {
        sample = TakeSample(split, memory);
    }
    const u64 head = g_entry_head.load(std::memory_order_relaxed);
    g_entries[head % EntryRingSize] = EntryMark{split.WritePositionNow(), sample};
    g_entry_head.store(head + 1, std::memory_order_release);
}

} // namespace VideoCommon::S4Count
