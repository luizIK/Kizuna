// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_FRONT_SPLIT_COUNT and FRITCHII_ONEF_NULL_FILL (fritchii_front_split.h).

#include <algorithm>
#include <array>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/fritchii_front_split.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"

namespace VideoCommon::FrontSplit {
namespace {

constexpr size_t StageCount = static_cast<size_t>(Stage::Count);
constexpr size_t MaxChannels = 16;
constexpr u64 ReportNs = 10'000'000'000ULL;

[[nodiscard]] u64 Ticks() noexcept {
    return Nx2Profile::Ticks();
}

struct ChannelTime {
    s32 id{};
    bool used{};
    u64 ticks{};
    u64 wait_ns{};
    u64 timed{};
};

/// The timed part (front only).
struct Timed {
    std::array<u64, StageCount> ticks{};
    std::array<u64, StageCount> wait_ns{};
    std::array<ChannelTime, MaxChannels> channels{};
    ChannelTime other_channels{};
    u64 entries{}; ///< timed entries
};
Timed g_timed{};

// The open timed entry.
Stage g_cur{Stage::Parser};
std::array<Stage, 8> g_stack{};
size_t g_depth{};
u64 g_seg_start{};
u64 g_entry_start{};
u64 g_entry_wait_ns{};
s32 g_entry_channel{};

u64 g_since_ns{};
u64 g_since_ticks{};
u64 g_since_position{};
/// FRITCHII_NOTED_SUMMARY's totals at the window's start (its part of the line).
GpuSplit::NotedSummaryStats g_since_noted{};
u32 g_rand{0x9e3779b9U};

[[nodiscard]] bool Take() noexcept {
    // xorshift32: no aliasing with the game's own patterns of entries.
    g_rand ^= g_rand << 13;
    g_rand ^= g_rand >> 17;
    g_rand ^= g_rand << 5;
    return (g_rand & 63) == 0;
}

void CloseSegment(u64 now) noexcept {
    g_timed.ticks[static_cast<size_t>(g_cur)] += now - g_seg_start;
    g_seg_start = now;
}

ChannelTime& ChannelOf(s32 id) noexcept {
    for (ChannelTime& channel : g_timed.channels) {
        if (channel.used && channel.id == id) {
            return channel;
        }
        if (!channel.used) {
            channel.used = true;
            channel.id = id;
            return channel;
        }
    }
    return g_timed.other_channels;
}

[[nodiscard]] u64 SentPosition() noexcept {
    auto& split = GpuSplit::Get();
    return split.Active() ? split.WritePositionNow() : 0;
}

void ResetWindow() noexcept {
    g_live = {};
    g_timed = {};
    g_since_ns = Nx2Profile::Now();
    g_since_ticks = Ticks();
    g_since_position = SentPosition();
    g_since_noted = GpuSplit::Get().GetNotedSummaryStats();
}

/// F4's part of the line: what the noted summary answered in this window.
std::string NotedSummaryText(double window, double ns_per_tick) {
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_noted_summary,
                                                      "FRITCHII_NOTED_SUMMARY");
    if (mode < 1 || mode > 3) {
        return fmt::format("noted summary off (FRITCHII_NOTED_SUMMARY={})", mode);
    }
    const auto& s = GpuSplit::Get().GetNotedSummaryStats();
    const auto& l = g_since_noted;
    const u64 checks = s.checks - l.checks;
    const u64 answered = s.answered - l.answered;
    const u64 timed = s.timed - l.timed;
    // Walks avoided: at 1 the answered checks less the timed ones (they walk); 2 and 3 walk all.
    const u64 avoided = mode == 1 ? answered - (s.timed_answered - l.timed_answered) : 0;
    const double saving =
        timed != 0
            ? (static_cast<double>(s.walk_ticks_answered - l.walk_ticks_answered) -
               static_cast<double>(s.summary_ticks - l.summary_ticks)) *
                  ns_per_tick * static_cast<double>(checks) / static_cast<double>(timed) / 1e6 /
                  window
            : 0.0;
    return fmt::format("noted summary (FRITCHII_NOTED_SUMMARY={}): answered {:.0f}/s of {:.0f}/s "
                       "checks, walks avoided {:.0f}/s, sampled saving {:.2f} ms/s, differ {}",
                       mode, static_cast<double>(answered) / window,
                       static_cast<double>(checks) / window,
                       static_cast<double>(avoided) / window, saving, s.differ);
}

void Report(u64 now_ns) {
    const Live c = g_live;
    const Timed& t = g_timed;
    const double window = static_cast<double>(now_ns - g_since_ns) / 1e9;
    const double ns_per_tick =
        static_cast<double>(now_ns - g_since_ns) /
        static_cast<double>((std::max)(Ticks() - g_since_ticks, u64{1}));
    // Random 1 in 64: scaled by the entries counted over the ones timed.
    const double scale = t.entries != 0 ? static_cast<double>(c.entries) /
                                              static_cast<double>(t.entries)
                                        : 0.0;
    const auto rate = [window](u64 count) { return static_cast<double>(count) / window; };
    const auto ms = [&](u64 ticks, u64 wait_ns) {
        const double ns = static_cast<double>(ticks) * ns_per_tick - static_cast<double>(wait_ns);
        return (std::max)(ns, 0.0) * scale / 1e6 / window;
    };
    const auto stage = [&](Stage s) {
        const auto i = static_cast<size_t>(s);
        return ms(t.ticks[i], t.wait_ns[i]);
    };
    u64 waits = 0;
    for (const u64 ns : t.wait_ns) {
        waits += ns;
    }
    const double waits_ms = static_cast<double>(waits) * scale / 1e6 / window;
    // Channels, largest first.
    std::array<const ChannelTime*, MaxChannels + 1> order{};
    size_t used = 0;
    for (const ChannelTime& channel : t.channels) {
        if (channel.used) {
            order[used++] = &channel;
        }
    }
    if (t.other_channels.timed != 0) {
        order[used++] = &t.other_channels;
    }
    std::sort(order.begin(), order.begin() + used, [](const ChannelTime* a, const ChannelTime* b) {
        return a->ticks > b->ticks;
    });
    std::string channels;
    double channels_total = 0.0;
    for (size_t i = 0; i < used; ++i) {
        channels_total += ms(order[i]->ticks, order[i]->wait_ns);
    }
    for (size_t i = 0; i < used; ++i) {
        const ChannelTime& channel = *order[i];
        const double value = ms(channel.ticks, channel.wait_ns);
        channels += fmt::format("{}{} {:.1f} ({:.0f}%)", i != 0 ? ", " : "",
                                &channel == &t.other_channels ? std::string{"others"}
                                                              : fmt::format("{}", channel.id),
                                value, channels_total > 0.0 ? value * 100.0 / channels_total : 0.0);
    }
    if (channels.empty()) {
        channels = "none timed";
    }
    const auto join = [&](JoinCause cause) { return rate(c.joins[static_cast<size_t>(cause)]); };
    u64 joins = 0;
    for (const u64 count : c.joins) {
        joins += count;
    }
    const auto answers = [&](const std::array<u64, static_cast<size_t>(DmaAnswer::Count)>& a) {
        const auto r = [&](DmaAnswer answer) { return rate(a[static_cast<size_t>(answer)]); };
        return fmt::format("forwarded {:.0f}, snapshot hit {:.0f}, drained: noted overlap "
                           "{:.0f}, writer {:.0f}, new page {:.0f}, report page {:.0f}, no "
                           "snapshot path {:.0f}; deferred to E {:.0f}",
                           r(DmaAnswer::Forwarded), r(DmaAnswer::SnapshotHit),
                           r(DmaAnswer::NotedOverlap), r(DmaAnswer::Writer), r(DmaAnswer::NewPage),
                           r(DmaAnswer::ReportPage), r(DmaAnswer::Drained),
                           r(DmaAnswer::Deferred));
    };
    const u64 position = SentPosition();
    const u64 sent_words = position >= g_since_position ? position - g_since_position : 0;
    const double parser = stage(Stage::Parser);
    const double state3d = stage(Stage::State3D);
    const double compute = stage(Stage::Compute);
    const double sends = stage(Stage::RingSend);
    LOG_INFO(HW_GPU,
             "Front split dry: parser {:.1f} ms/s, 3D state {:.1f}, compute {:.1f}, ring sends "
             "{:.1f} (sum {:.1f}; {} of {} entries timed, waits taken out {:.1f}{}); entries "
             "{:.0f}/s, headers {:.0f}/s, words {:.0f}/s, suspensions {:.0f}/s; 2a joins "
             "{:.0f}/s: acquire with a backlog {:.0f}, writer check with a writer in it {:.0f}, "
             "GPU-written list {:.0f}, Step run {:.0f}; Step defers {:.0f}/s (encoded, no "
             "join); engine calls {:.0f}/s; 2c raw ring words {:.0f}/s (today {:.0f}/s); 0xe2c "
             "by answer /s: {}; other macro reads /s: {}; channels ms/s: {}; re-read twice "
             "{:.0f}/s; {}",
             parser, state3d, compute, sends, parser + state3d + compute + sends, t.entries,
             c.entries, waits_ms, Nx2Profile::Enabled() ? ", profiled" : "", rate(c.entries),
             rate(c.headers), rate(c.words), rate(c.suspensions), rate(joins),
             join(JoinCause::Acquire), join(JoinCause::WriterCheck), join(JoinCause::GpuList),
             join(JoinCause::StepRun), rate(c.step_defers), rate(c.engine_calls),
             rate(c.raw_ring_words), rate(sent_words), answers(c.e2c), answers(c.other_reads),
             channels, rate(c.reread_twice), NotedSummaryText(window, ns_per_tick));
}

} // Anonymous namespace

void SetLevel(int level) noexcept {
    if (level < 0) {
        level = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_front_split_count,
                                                 "FRITCHII_FRONT_SPLIT_COUNT");
    }
    // The mirror holds the raw value, so the hot compare stays quiet for values other than 1.
    const int old = g_level;
    g_level = level;
    const bool on = level == 1;
    if (on == ((g_state & StateOn) != 0)) {
        return;
    }
    g_state = on ? StateOn : 0;
    g_depth = 0;
    ResetWindow();
    LOG_INFO(HW_GPU,
             "Front split dry (FRITCHII_FRONT_SPLIT_COUNT={}): {} (was {})", level,
             on ? "stage timers 1 in 64 entries, 2a/2c counts, 0xe2c answers, channels" : "off",
             old);
}

bool Begin(s32 channel) noexcept {
    ++g_live.entries;
    if ((g_live.entries & 255) == 0) [[unlikely]] {
        if (const u64 now = Nx2Profile::Now(); now - g_since_ns >= ReportNs) {
            Report(now);
            ResetWindow();
        }
    }
    if ((g_state & StateSampling) != 0 || !Take()) {
        return false; // nested (a Step inside a timed one), or not picked
    }
    g_state |= StateSampling;
    g_cur = Stage::Parser;
    g_depth = 0;
    g_entry_channel = channel;
    g_entry_wait_ns = 0;
    g_entry_start = Ticks();
    g_seg_start = g_entry_start;
    return true;
}

void End() noexcept {
    if ((g_state & StateSampling) == 0) {
        return;
    }
    const u64 now = Ticks();
    CloseSegment(now);
    ChannelTime& channel = ChannelOf(g_entry_channel);
    channel.ticks += now - g_entry_start;
    channel.wait_ns += g_entry_wait_ns;
    ++channel.timed;
    ++g_timed.entries;
    g_depth = 0;
    g_state &= static_cast<u8>(~StateSampling);
}

bool Push(Stage stage) noexcept {
    if (g_depth >= g_stack.size()) {
        return false;
    }
    CloseSegment(Ticks());
    g_stack[g_depth++] = g_cur;
    g_cur = stage;
    return true;
}

void Pop() noexcept {
    if (g_depth == 0 || (g_state & StateSampling) == 0) {
        return;
    }
    CloseSegment(Ticks());
    g_cur = g_stack[--g_depth];
}

void NoteWaitSampled(u64 ns) noexcept {
    g_timed.wait_ns[static_cast<size_t>(g_cur)] += ns;
    g_entry_wait_ns += ns;
}

void NoteNullFill() noexcept {
    // E only.
    static u64 draws{};
    static u64 logged{};
    static u64 since_ns{};
    ++draws;
    if ((draws & 1023) != 0) {
        return;
    }
    const u64 now = Nx2Profile::Now();
    if (since_ns == 0) {
        since_ns = now;
        logged = draws;
        return;
    }
    if (now - since_ns < ReportNs) {
        return;
    }
    const double window = static_cast<double>(now - since_ns) / 1e9;
    LOG_WARNING(HW_GPU,
                "Null fill (FRITCHII_ONEF_NULL_FILL=1, test only, the picture is broken): {:.0f} "
                "draws/s null-filled (E skipped the flush, the fill and the push), {} this session",
                static_cast<double>(draws - logged) / window, draws);
    since_ns = now;
    logged = draws;
}

} // namespace VideoCommon::FrontSplit
