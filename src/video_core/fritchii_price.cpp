// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/fritchii_page_flags.h"
#include "common/logging.h"
#include "video_core/fritchii_price.h"

namespace VideoCommon::FritchiiPrice {

namespace {

/// Row 1: each uniform slot's last bind (the back alone).
struct UniformSlot {
    u64 address{};
    u64 sum{};
    u32 size{};
    bool valid{};
};
std::array<UniformSlot, 5 * 32> g_uniform_slots{};

/// The back's position stores, stamped (BackReached); the front looks them up (ReachedAt).
constexpr size_t ReachedSlots = 256;
struct ReachedStamps {
    std::array<std::atomic<u64>, ReachedSlots> position{};
    std::array<std::atomic<u64>, ReachedSlots> ticks{};
    std::atomic<u64> count{};
};
ReachedStamps g_reached;

} // Anonymous namespace

namespace {
void GensOn() noexcept {
    if (!Common::g_price_gens_on.load(std::memory_order_relaxed)) [[unlikely]] {
        Common::g_price_gens_on.store(true, std::memory_order_relaxed);
    }
}
} // Anonymous namespace

void NoteUniformBind(u32 slot_key, u64 device_addr, u32 size, bool same, u64 elapsed,
                     u64 sum) noexcept {
    GensOn();
    UniformSlot& slot = g_uniform_slots[slot_key % g_uniform_slots.size()];
    Uniform verdict;
    if (!same) {
        verdict = Uniform::Changed;
    } else if (!Common::BankMarks()) {
        verdict = Uniform::NoFlags;
    } else if (!slot.valid || slot.address != device_addr || slot.size != size) {
        verdict = Uniform::Moved;
    } else {
        verdict = slot.sum == sum && sum != ~u64{0} ? Uniform::Unflagged : Uniform::Flagged;
        if (verdict == Uniform::Unflagged && t_uniform_cpu_modified) {
            verdict = Uniform::UnflaggedCpuModified; // guest stores there raise no flag
        }
    }
    slot = UniformSlot{device_addr, sum, size, true};
    g_uniform.Note(static_cast<size_t>(verdict), 1, elapsed);
}

void NoteHandleValues(HandleValues verdict, u64 ticks) noexcept {
    GensOn();
    g_handle_values.Note(static_cast<size_t>(verdict), 1, ticks);
}

Handles ClassifyHandles(u64 key, u64 sum, u64& last_key, u64& last_sum) noexcept {
    GensOn();
    Handles verdict;
    if (!Common::BankMarks()) {
        verdict = Handles::NoFlags;
    } else if (key != last_key || key == 0) {
        verdict = Handles::Moved;
    } else {
        verdict = sum == last_sum && sum != ~u64{0} ? Handles::Unflagged : Handles::Flagged;
    }
    last_key = key;
    last_sum = sum;
    return verdict;
}

void BackReached(u64 position) noexcept {
    const u64 n = g_reached.count.load(std::memory_order_relaxed);
    const size_t slot = static_cast<size_t>(n % ReachedSlots);
    g_reached.position[slot].store(~u64{0}, std::memory_order_relaxed); // being written
    g_reached.ticks[slot].store(Ticks(), std::memory_order_relaxed);
    g_reached.position[slot].store(position, std::memory_order_release);
    g_reached.count.store(n + 1, std::memory_order_release);
}

bool ReachedAt(u64 target, u64& ticks) noexcept {
    const u64 n = g_reached.count.load(std::memory_order_acquire);
    const u64 held = n < ReachedSlots ? n : ReachedSlots;
    bool found = false;
    for (u64 i = 0; i < held; ++i) {
        const size_t slot = static_cast<size_t>((n - 1 - i) % ReachedSlots);
        const u64 position = g_reached.position[slot].load(std::memory_order_acquire);
        if (position == ~u64{0}) {
            continue; // the back is writing it
        }
        if (position < target) {
            return found; // the stamp after this one is the first at or past the target
        }
        ticks = g_reached.ticks[slot].load(std::memory_order_relaxed);
        found = true;
    }
    // Every stamp held is past the target: the first one there may be older than the ring.
    return found && held < ReachedSlots;
}

void NoteFrontWait(size_t site, u64 target, u64 start) noexcept {
    const u64 end = Ticks();
    const u64 total = end > start ? end - start : 0;
    auto& waits = g_waits;
    Add(waits.count[site], 1);
    u64 reached = 0;
    if (!ReachedAt(target, reached)) {
        Add(waits.unknown[site], 1);
        Add(waits.until[site], total);
        return;
    }
    const u64 until = reached > start ? (reached < end ? reached - start : total) : 0;
    Add(waits.until[site], until);
    Add(waits.after[site], total - until);
}

void RecordScope::Begin() noexcept {
    // One call in 1 to 15 (about 8) is timed and stands for the calls since the last timed one
    // (the interval varies, so no fixed period lines up with the calls of a draw).
    in_draw = t_draw_depth != 0 ? 1 : 0;
    Add(g_record.calls[in_draw], 1);
    if (--t_record_left != 0) {
        return;
    }
    u32 seed = t_record_seed;
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    t_record_seed = seed;
    weight = t_record_weight;
    t_record_weight = 1 + seed % 15;
    t_record_left = t_record_weight;
    start = Ticks();
}

void RecordScope::End() noexcept {
    Add(g_record.ticks[in_draw], (Ticks() - start) * weight);
}

namespace {

/// The report's last totals of each counter (the back alone reads and keeps them).
struct ReportState {
    u64 since_ns{};
    u64 since_ticks{};
    u32 calls{};
    std::array<u64, 2 * static_cast<size_t>(Uniform::Count)> uniform{};
    std::array<u64, 4 * static_cast<size_t>(Handles::Count)> handles{};
    std::array<u64, 3 * CbBuckets> cb{};
    std::array<u64, 2 * static_cast<size_t>(Entry::Count)> entries{};
    std::array<u64, 4 * WaitSites> waits{};
    std::array<u64, 5> worker{};
    std::array<u64, 6> record{};
    std::array<u64, 3 * Method3DSlots> method3d{};
    std::array<u64, 2 * static_cast<size_t>(HandleValues::Count)> handle_values{};
};
ReportState g_report;

/// The amount `counter` moved since the last report (kept in `last`).
u64 Delta(const Counter& counter, u64& last) {
    const u64 now = counter.load(std::memory_order_relaxed);
    const u64 moved = now - last;
    last = now;
    return moved;
}

constexpr std::array<const char*, static_cast<size_t>(Uniform::Count)> uniform_names{
    "changed", "unchanged and unflagged since its last bind, pages tracked (the cut)",
    "unchanged but flagged", "unchanged, another range than its last bind",
    "unchanged, bank marks off",
    "unchanged and unflagged, pages CPU-modified (guest stores raise no flag: not the cut)"};
constexpr std::array<const char*, static_cast<size_t>(Handles::Count)> handle_names{
    "unflagged since the pipeline's last draw (the cut)", "flagged",
    "another range or first draw", "bank marks off"};
constexpr std::array<const char*, CbBuckets> cb_names{"1", "2", "3-4", "5-8", "9-16", "17+"};
constexpr std::array<const char*, static_cast<size_t>(Entry::Count)> entry_names{
    "cb word runs", "3D single", "other engine single", "multi", "call", "pairs",
    "state delta", "draw batch", "cb word blocks", "engine/wrap"};

} // Anonymous namespace

void MaybeReport() {
    ReportState& r = g_report;
    if ((++r.calls & 255) != 0) {
        return;
    }
    const u64 now_ns = Nx2Profile::Now();
    const u64 now_ticks = Ticks();
    if (r.since_ns == 0) {
        r.since_ns = now_ns;
        r.since_ticks = now_ticks;
    }
    if (now_ns - r.since_ns < 10'000'000'000ULL) {
        return;
    }
    const double window = static_cast<double>(now_ns - r.since_ns) / 1e9;
    const double ns_per_tick = static_cast<double>(now_ns - r.since_ns) /
                               static_cast<double>((std::max)(now_ticks - r.since_ticks, u64{1}));
    const auto ms = [&](u64 ticks) {
        return static_cast<double>(ticks) * ns_per_tick / 1e6 / window;
    };
    const auto rate = [&](u64 count) { return static_cast<double>(count) / window; };

    // Row 1.
    {
        std::string text;
        u64 any = 0;
        for (size_t i = 0; i < uniform_names.size(); ++i) {
            const u64 count = Delta(g_uniform.count[i], r.uniform[2 * i]);
            const u64 ticks = Delta(g_uniform.ticks[i], r.uniform[2 * i + 1]);
            any |= count;
            text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", text.empty() ? "" : "; ",
                                uniform_names[i], rate(count), ms(ticks));
        }
        if (any != 0) {
            LOG_INFO(HW_GPU,
                     "Price uniform binds (back, 10 s; the time is the guest read and the "
                     "compare): {}",
                     text);
        }
    }
    // Row 2.
    {
        std::string text;
        u64 any = 0;
        for (size_t site = 0; site < 2; ++site) {
            std::string part;
            for (size_t i = 0; i < handle_names.size(); ++i) {
                const size_t base = (site * handle_names.size() + i) * 2;
                const u64 count = Delta(g_handles[site].count[i], r.handles[base]);
                const u64 ticks = Delta(g_handles[site].ticks[i], r.handles[base + 1]);
                any |= count;
                part += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", part.empty() ? "" : ", ",
                                    handle_names[i], rate(count), ms(ticks));
            }
            text += fmt::format("{}{}: {}", site == 0 ? "" : "; ",
                                site == 0 ? "FillHandleValues (EXP123 3)"
                                          : "ConfigureImpl's handle loop (EXP123 below 3)",
                                part);
        }
        if (any != 0) {
            LOG_INFO(HW_GPU, "Price handle reads (back, draws, 10 s): {}", text);
        }
    }
    // Row 3.
    {
        std::string text;
        u64 any = 0;
        for (size_t i = 0; i < CbBuckets; ++i) {
            const u64 calls = Delta(g_cb_runs.count[i], r.cb[3 * i]);
            const u64 ticks = Delta(g_cb_runs.ticks[i], r.cb[3 * i + 1]);
            const u64 words = Delta(CbWriteScope::g_cb_words[i], r.cb[3 * i + 2]);
            any |= calls;
            text += fmt::format("{}{} words {:.0f}/s ({:.0f} words/s) {:.1f} ms/s",
                                text.empty() ? "" : "; ", cb_names[i], rate(calls), rate(words),
                                ms(ticks));
        }
        if (any != 0) {
            LOG_INFO(HW_GPU,
                     "Price constant buffer word calls (back, 10 s; the write half's time, "
                     "by run length): {}",
                     text);
        }
    }
    // Row 4.
    {
        std::string text;
        u64 any = 0;
        for (size_t i = 0; i < entry_names.size(); ++i) {
            const u64 count = Delta(g_entries.count[i], r.entries[2 * i]);
            const u64 ticks = Delta(g_entries.ticks[i], r.entries[2 * i + 1]);
            any |= count;
            text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", text.empty() ? "" : "; ",
                                entry_names[i], rate(count), ms(ticks));
        }
        if (any != 0) {
            LOG_INFO(HW_GPU,
                     "Price back ring entries by kind (10 s; entries a second, the \"method\" "
                     "zone's ms/s each kind added): {}",
                     text);
        }
    }
    // Row 2b: E's handle values by verdict.
    {
        static constexpr std::array<const char*, static_cast<size_t>(HandleValues::Count)> names{
            "changed or first", "same, unflagged, pages tracked (the cut)",
            "same, unflagged, a page CPU-modified", "same but a page flagged",
            "same values from other ranges", "handle program",
            "unflagged and tracked but changed (must be 0)"};
        std::string text;
        u64 any = 0;
        for (size_t i = 0; i < names.size(); ++i) {
            const u64 count = Delta(g_handle_values.count[i], r.handle_values[2 * i]);
            const u64 ticks = Delta(g_handle_values.ticks[i], r.handle_values[2 * i + 1]);
            any |= count;
            text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", text.empty() ? "" : "; ", names[i],
                                rate(count), ms(ticks));
        }
        if (any != 0) {
            LOG_INFO(HW_GPU,
                     "Price handle values (E, unlent draws, 10 s; against the pipeline's last, "
                     "the time is the fill): {}",
                     text);
        }
    }
    // Row 4b: the single 3D method entries by method, the costliest 24 (sunk: state writes the
    // entry consumed before its own handler).
    {
        struct Row {
            u32 method;
            u64 count;
            u64 ticks;
            u64 sunk;
        };
        std::vector<Row> rows;
        u64 total_ticks = 0;
        u64 total_count = 0;
        for (size_t i = 0; i < Method3DSlots; ++i) {
            const u64 count = Delta(g_method3d.count[i], r.method3d[3 * i]);
            const u64 ticks = Delta(g_method3d.ticks[i], r.method3d[3 * i + 1]);
            const u64 sunk = Delta(g_method3d.sunk[i], r.method3d[3 * i + 2]);
            if (count != 0) {
                rows.push_back({static_cast<u32>(i), count, ticks, sunk});
                total_ticks += ticks;
                total_count += count;
            }
        }
        if (!rows.empty()) {
            std::sort(rows.begin(), rows.end(),
                      [](const Row& a, const Row& b) { return a.ticks > b.ticks; });
            std::string text;
            for (size_t i = 0; i < rows.size() && i < 24; ++i) {
                text += fmt::format("{}{:#x} {:.0f}/s {:.1f} ms/s sunk {:.0f}/s",
                                    text.empty() ? "" : "; ", rows[i].method, rate(rows[i].count),
                                    ms(rows[i].ticks), rate(rows[i].sunk));
            }
            LOG_INFO(HW_GPU,
                     "Price back 3D methods (10 s; single 3D entries by method, {:.0f}/s {:.1f} "
                     "ms/s in all, costliest first; time = the \"method\" zone it added, sunk = "
                     "state writes it consumed first): {}",
                     rate(total_count), ms(total_ticks), text);
        }
    }
    // Rows 7 and 8.
    {
        std::string text;
        for (size_t site = 0; site < WaitSites; ++site) {
            const u64 count = Delta(g_waits.count[site], r.waits[4 * site]);
            const u64 until = Delta(g_waits.until[site], r.waits[4 * site + 1]);
            const u64 after = Delta(g_waits.after[site], r.waits[4 * site + 2]);
            const u64 unknown = Delta(g_waits.unknown[site], r.waits[4 * site + 3]);
            if (count == 0) {
                continue;
            }
            text += fmt::format("{}{} {:.0f}/s: until the back reached it {:.1f} ms/s, after "
                                "{:.1f} ms/s ({:.0f}/s not in the stamps)",
                                text.empty() ? "" : "; ",
                                site == AcquireSite ? "suspended acquires"
                                                    : drain_site_names[site],
                                rate(count), ms(until), ms(after), rate(unknown));
        }
        if (!text.empty()) {
            LOG_INFO(HW_GPU, "Price front waits split (10 s): {}", text);
        }
    }
    // Worker load (19.4).
    {
        const u64 plain = Delta(g_worker.chunks[0], r.worker[0]);
        const u64 submits = Delta(g_worker.chunks[1], r.worker[1]);
        const u64 plain_ticks = Delta(g_worker.busy[0], r.worker[2]);
        const u64 submit_ticks = Delta(g_worker.busy[1], r.worker[3]);
        const u64 delay_ticks = Delta(g_worker.delay_ticks, r.worker[4]);
        const u64 calls_out = Delta(g_record.calls[0], r.record[0]);
        const u64 calls_in = Delta(g_record.calls[1], r.record[1]);
        const u64 ticks_out = Delta(g_record.ticks[0], r.record[2]);
        const u64 ticks_in = Delta(g_record.ticks[1], r.record[3]);
        const u64 draws = Delta(g_record.draws, r.record[4]);
        const u64 draw_ticks = Delta(g_record.draw_ticks, r.record[5]);
        if (plain + submits + calls_out + calls_in + draws != 0) {
            const double busy = ms(plain_ticks + submit_ticks);
            LOG_INFO(HW_GPU,
                     "Price Vulkan worker (10 s): busy {:.1f} ms/s ({:.0f} chunks/s {:.1f} ms/s "
                     "record only, {:.0f}/s {:.1f} ms/s with a submit), idle {:.1f} ms/s, "
                     "FRITCHII_DELAY_WORKER_US spins {:.1f} ms/s; the back's Scheduler::Record "
                     "calls: in draws {:.0f}/s {:.1f} ms/s, outside {:.0f}/s {:.1f} ms/s; draws "
                     "{:.0f}/s {:.1f} ms/s (Draw and DrawIndirect whole)",
                     busy, rate(plain), ms(plain_ticks), rate(submits), ms(submit_ticks),
                     (std::max)(1000.0 - busy - ms(delay_ticks), 0.0), ms(delay_ticks),
                     rate(calls_in), ms(ticks_in), rate(calls_out), ms(ticks_out), rate(draws),
                     ms(draw_ticks));
        }
    }
    r.since_ns = now_ns;
    r.since_ticks = now_ticks;
}

} // namespace VideoCommon::FritchiiPrice
