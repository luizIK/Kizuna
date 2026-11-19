// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower's counters and log lines (one_follower_stats.h; work\fable_b_one_follower.md
// sections 5, 7 and 8). Also defines one_follower.h's Note().

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/one_follower.h"
#include "video_core/one_follower_stats.h"

namespace VideoCommon::OneFollower {

namespace {
/// The first reports of each must-be-0 name its place (`what`); later ones only count.
constexpr u64 FirstReports = 8;
} // namespace

void Note(Check check, const char* what) noexcept {
    const size_t index = (std::min)(static_cast<size_t>(check), check_names.size() - 1);
    const u64 count =
        Stats::g_counters.checks[index].fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= FirstReports) {
        // Not "must be 0" here: the 10 s line carries the count, this names where.
        LOG_ERROR(HW_GPU, "One follower check hit: {}: {} ({} this session{})", check_names[index],
                  what != nullptr ? what : "?", count,
                  count == FirstReports ? ", later ones only counted" : "");
    }
}

} // namespace VideoCommon::OneFollower

namespace VideoCommon::OneFollower::Stats {

namespace {

constexpr size_t Ops = static_cast<size_t>(LentOp::Count);
constexpr size_t Reasons = static_cast<size_t>(LentReason::Count);
constexpr size_t Checks = static_cast<size_t>(Check::Count);
constexpr size_t Times = static_cast<size_t>(RenderTime::Count);

/// Counters as plain numbers: what the logger read last time and reads now.
struct Snapshot {
    u64 draw_packets{}, draw_bytes{}, draw_words{};
    std::array<u64, KindSlots> unlent_kind{};
    std::array<u64, BehindSlots> behind{};
    u64 window_waits{}, window_ns{}, stream_checked{};
    u64 lent_packets{}, delta_packets{}, other_bytes{}, other_words{};
    std::array<u64, Ops> lent_op{};
    std::array<u64, Reasons> lent_reason{};
    std::array<u64, KindSlots> lent_kind{};
    std::array<u64, Ops> parked_ns{};
    u64 drains{}, drain_ns{}, stream_lent{};
    std::array<u64, Ops> lent_call_ns{};
    u64 lent_reach_ns{}, lents_ran{};
    std::array<u64, Times> render_ns{};
    u64 render_sleeps{}, view_compares{}, build_compares{};
    std::array<u64, Checks> checks{};
    u64 stream_not_lent{};
};

u64 Load(const std::atomic<u64>& value) {
    return value.load(std::memory_order_relaxed);
}

template <size_t N>
void LoadAll(std::array<u64, N>& out, const std::array<std::atomic<u64>, N>& in) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = Load(in[i]);
    }
}

void Take(Snapshot& s) {
    const Counters& c = g_counters;
    s.draw_packets = Load(c.draw_packets);
    s.draw_bytes = Load(c.draw_bytes);
    s.draw_words = Load(c.draw_words);
    LoadAll(s.unlent_kind, c.unlent_kind);
    for (size_t i = 0; i < BehindSlots; ++i) {
        s.behind[i] = Load(c.behind_draw[i]) + Load(c.behind_other[i]);
    }
    s.window_waits = Load(c.window_waits);
    s.window_ns = Load(c.window_ns);
    s.stream_checked = Load(c.stream_checked);
    s.lent_packets = Load(c.lent_packets);
    s.delta_packets = Load(c.delta_packets);
    s.other_bytes = Load(c.other_bytes);
    s.other_words = Load(c.other_words);
    LoadAll(s.lent_op, c.lent_op);
    LoadAll(s.lent_reason, c.lent_reason);
    LoadAll(s.lent_kind, c.lent_kind);
    LoadAll(s.parked_ns, c.parked_ns);
    s.drains = Load(c.drains);
    s.drain_ns = Load(c.drain_ns);
    s.stream_lent = Load(c.stream_lent);
    LoadAll(s.lent_call_ns, c.lent_call_ns);
    s.lent_reach_ns = Load(c.lent_reach_ns);
    s.lents_ran = Load(c.lents_ran);
    LoadAll(s.render_ns, c.render_ns);
    s.render_sleeps = Load(c.render_sleeps);
    s.view_compares = Load(c.view_compares);
    s.build_compares = Load(c.build_compares);
    LoadAll(s.checks, c.checks);
    s.stream_not_lent = Load(c.stream_not_lent);
}

/// The window maximum of Behind() at a push: read and restarted by the logger. A writer's store
/// racing the restart only loses one window's maximum, never a count.
u64 TakeBehindMax() {
    Counters& c = g_counters;
    const u64 draw = c.behind_max_draw.exchange(0, std::memory_order_relaxed);
    const u64 other = c.behind_max_other.exchange(0, std::memory_order_relaxed);
    return (std::max)(draw, other);
}

std::mutex log_mutex;
Snapshot reported;
std::chrono::steady_clock::time_point last_log{};
bool started = false;

const char* StepName(StepLevel step) {
    switch (step) {
    case StepLevel::Off:
        return "0 off";
    case StepLevel::Lent:
        return "1 lent";
    case StepLevel::View:
        return "2 view";
    case StepLevel::Build:
        return "3 build";
    case StepLevel::Unlend:
        return "4 unlend";
    }
    return "?";
}

} // namespace

void MaybeLog(const Logged& now) {
    std::unique_lock lock{log_mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
        return;
    }
    const auto time = std::chrono::steady_clock::now();
    if (!started) {
        started = true;
        last_log = time;
        Take(reported);
        TakeBehindMax();
        return;
    }
    if (time - last_log < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(time - last_log).count();
    last_log = time;
    Snapshot current;
    Take(current);
    const Snapshot& before = reported;
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto ms = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    const auto per_us = [](u64 ns, u64 count) {
        return count != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(count) : 0.0;
    };
    const auto each = [](u64 total, u64 count) {
        return count != 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
    };
    const auto sum = [](const auto& values) {
        u64 total = 0;
        for (const u64 value : values) {
            total += value;
        }
        return total;
    };
    // "name rate, name rate" for an array of counters against its earlier snapshot.
    const auto rates = [&](const auto& names, const auto& now_values, const auto& old_values) {
        std::string text;
        for (size_t i = 0; i < now_values.size(); ++i) {
            text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", names[i],
                                rate(now_values[i] - old_values[i]));
        }
        return text;
    };

    // 1. The must-be-0 line (always): one_follower.h's checks in its pair form, then this file's
    //    own. Check::StreamWrite is the guard's hits (handled), on the verify line instead.
    std::string checks_text;
    for (size_t i = 0; i < Checks; ++i) {
        if (static_cast<Check>(i) == Check::StreamWrite) {
            continue;
        }
        checks_text += fmt::format("{}{} {} / {}", checks_text.empty() ? "" : ", ",
                                   check_names[i], current.checks[i] - before.checks[i],
                                   current.checks[i]);
    }
    checks_text += fmt::format(", {} {} / {}", stream_not_lent_name,
                               current.stream_not_lent - before.stream_not_lent,
                               current.stream_not_lent);
    LOG_INFO(HW_GPU, "One follower checks (must be 0; this window / session): {}", checks_text);

    // 2. Coverage and the stream-write guard (always): a must-be-0 of 0 means something only
    //    with compares behind it.
    const size_t stream_index = static_cast<size_t>(Check::StreamWrite);
    const size_t verify_index = static_cast<size_t>(LentReason::Verify);
    const size_t draw_index = static_cast<size_t>(LentOp::Draw);
    const u64 lent_draws = current.lent_op[draw_index] - before.lent_op[draw_index];
    const u64 unlent_draws = current.draw_packets - before.draw_packets;
    LOG_INFO(HW_GPU,
             "One follower verify (step {}, asked {}, FRITCHII_ONEF_UNLEND={:#x}, "
             "FRITCHII_ONEF_VERIFY_EVERY={}) {:.0f} s: view compares {:.0f}/s ({} this session), "
             "build compares {:.0f}/s ({} this session), forced lent to verify {:.0f}/s ({} this "
             "session); lent draws {:.0f}/s, unlent draws {:.0f}/s ({:.1f}% unlent); stream-write "
             "guard: writes checked {:.0f}/s, hits {:.0f}/s ({} this session), made lent {:.0f}/s "
             "({} this session)",
             StepName(now.step), now.asked, now.unlend_mask, now.verify_every, seconds,
             rate(current.view_compares - before.view_compares), current.view_compares,
             rate(current.build_compares - before.build_compares), current.build_compares,
             rate(current.lent_reason[verify_index] - before.lent_reason[verify_index]),
             current.lent_reason[verify_index], rate(lent_draws), rate(unlent_draws),
             lent_draws + unlent_draws != 0
                 ? 100.0 * static_cast<double>(unlent_draws) /
                       static_cast<double>(lent_draws + unlent_draws)
                 : 0.0,
             rate(current.stream_checked - before.stream_checked),
             rate(current.checks[stream_index] - before.checks[stream_index]),
             current.checks[stream_index], rate(current.stream_lent - before.stream_lent),
             current.stream_lent);

    const u64 behind_max = TakeBehindMax();
    if (Timing()) {
        // 3. The hand-overs: packets, lends by op / reason / kind, unlent draws by kind, the hop.
        const u64 draw_packets = current.draw_packets - before.draw_packets;
        const u64 lent_packets = current.lent_packets - before.lent_packets;
        const u64 delta_packets = current.delta_packets - before.delta_packets;
        const u64 packets = draw_packets + lent_packets + delta_packets;
        const u64 bytes = current.draw_bytes - before.draw_bytes + current.other_bytes -
                          before.other_bytes;
        const u64 words = current.draw_words - before.draw_words + current.other_words -
                          before.other_words;
        const u64 lends = sum(current.lent_op) - sum(before.lent_op);
        const u64 parked = sum(current.parked_ns) - sum(before.parked_ns);
        const u64 ran = current.lents_ran - before.lents_ran;
        const u64 reach = current.lent_reach_ns - before.lent_reach_ns;
        const u64 call = sum(current.lent_call_ns) - sum(before.lent_call_ns);
        std::string parked_text;
        for (size_t i = 0; i < Ops; ++i) {
            parked_text += fmt::format("{}{} {:.1f}", i == 0 ? "" : ", ", lent_op_names[i],
                                       ms(current.parked_ns[i] - before.parked_ns[i]));
        }
        LOG_INFO(
            HW_GPU,
            "One follower hand-overs (step {}, FRITCHII_ONEF_UNLEND={:#x}) {:.0f} s: packets "
            "{:.0f}/s (draw {:.0f}, lent {:.0f}, delta {:.0f}; {:.0f} B and {:.1f} register "
            "words each); lent by op/s: {}; lent draws by reason/s: {}; lent draws by kind/s: "
            "{}; unlent draws by kind/s: {}; the hop: lends {:.0f}/s, E parked {:.1f} ms/s "
            "({:.2f} us a lend: R reached it after {:.2f}, ran it {:.2f}, E back after {:.2f}); "
            "E parked by op ms/s: {}",
            StepName(now.step), now.unlend_mask, seconds, rate(packets), rate(draw_packets),
            rate(lent_packets), rate(delta_packets), each(bytes, packets), each(words, packets),
            rates(lent_op_names, current.lent_op, before.lent_op),
            rates(lent_reason_names, current.lent_reason, before.lent_reason),
            rates(kind_names, current.lent_kind, before.lent_kind),
            rates(kind_names, current.unlent_kind, before.unlent_kind), rate(lends), ms(parked),
            per_us(parked, lends), per_us(reach, ran), per_us(call, ran),
            (std::max)(per_us(parked, lends) - per_us(reach, ran) - per_us(call, ran), 0.0),
            parked_text);

        // 4. The threads: E's ahead window and waits, R's busy / idle / wait.
        const auto render = [&](RenderTime what) {
            const size_t i = static_cast<size_t>(what);
            return current.render_ns[i] - before.render_ns[i];
        };
        const u64 busy = render(RenderTime::DrawPackets) + render(RenderTime::LentCalls) +
                         render(RenderTime::OtherPackets);
        LOG_INFO(HW_GPU,
                 "One follower threads (FRITCHII_ONEF_AHEAD={}) {:.0f} s: E's ahead window, "
                 "packets R had not played at each push, per second: {}; max {}; E waited for the "
                 "window {:.0f}/s {:.1f} ms/s, drained {:.0f}/s {:.1f} ms/s, parked in lends "
                 "{:.1f} ms/s; R busy {:.1f} ms/s (draw packets {:.1f}, lent calls {:.1f}, other "
                 "packets {:.1f}), idle {:.1f} ms/s (slept {:.0f}/s), waited for E {:.1f} ms/s",
                 now.ahead == 0 ? std::string{"0 ring"} : std::to_string(now.ahead), seconds,
                 rates(behind_names, current.behind, before.behind), behind_max,
                 rate(current.window_waits - before.window_waits),
                 ms(current.window_ns - before.window_ns), rate(current.drains - before.drains),
                 ms(current.drain_ns - before.drain_ns), ms(parked), ms(busy),
                 ms(render(RenderTime::DrawPackets)), ms(render(RenderTime::LentCalls)),
                 ms(render(RenderTime::OtherPackets)), ms(render(RenderTime::Idle)),
                 rate(current.render_sleeps - before.render_sleeps),
                 ms(render(RenderTime::WaitForE)));
    }
    reported = current;
}

} // namespace VideoCommon::OneFollower::Stats
