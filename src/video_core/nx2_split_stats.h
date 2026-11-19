// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::Nx2SplitStats {

/// EDEN_NX2_SPLIT_STATS=1 (read once): the GPU split's hand-over, for Fable's heavyphase ask (c).
/// Front: words per Publish, bucketed. Back: each wait for work (the front had nothing published)
/// by length, the deepest phase it reached (spin, idle hook, sleep) and the words it found when it
/// woke. Logged every 5 s from the back ("GPU split hand-over (5 s").
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

inline constexpr std::array<u64, 6> WordBuckets{8, 32, 128, 512, 2048, ~u64{0}};
inline constexpr std::array<u64, 7> WaitBucketsNs{1'000,   4'000,     16'000,   64'000,
                                                  256'000, 1'000'000, ~u64{0}};

struct State {
    std::array<std::atomic<u64>, 6> publish_words{};
    std::atomic<u64> publishes{};
    std::atomic<u64> words{};
    std::array<u64, 7> wait_count{};    ///< Back thread only
    std::array<u64, 7> wait_ns{};
    std::array<u64, 3> wait_phase{};    ///< Deepest phase: 0 spin, 1 idle hook, 2 sleep
    std::array<u64, 6> woke_words{};    ///< Words available when the wait ended
    std::chrono::steady_clock::time_point since{};
};

inline State& Get() {
    static State state;
    return state;
}

template <size_t N>
inline size_t Bucket(const std::array<u64, N>& limits, u64 value) {
    for (size_t i = 0; i < N; ++i) {
        if (value <= limits[i]) {
            return i;
        }
    }
    return N - 1;
}

/// Front thread, in Publish.
inline void NotePublish(u64 words) {
    State& state = Get();
    state.publish_words[Bucket(WordBuckets, words)].fetch_add(1, std::memory_order_relaxed);
    state.publishes.fetch_add(1, std::memory_order_relaxed);
    state.words.fetch_add(words, std::memory_order_relaxed);
}

/// Back thread, at the end of one wait for work.
inline void NoteWait(u64 ns, u32 phase, u64 woke_words) {
    State& state = Get();
    const size_t bucket = Bucket(WaitBucketsNs, ns);
    ++state.wait_count[bucket];
    state.wait_ns[bucket] += ns;
    ++state.wait_phase[phase < 3 ? phase : 2];
    ++state.woke_words[Bucket(WordBuckets, woke_words)];
    const auto now = std::chrono::steady_clock::now();
    if (state.since == std::chrono::steady_clock::time_point{}) {
        state.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - state.since).count();
    if (seconds < 5.0) {
        return;
    }
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    std::array<u64, 6> publish_words{};
    for (size_t i = 0; i < publish_words.size(); ++i) {
        publish_words[i] = take(state.publish_words[i]);
    }
    const u64 publishes = take(state.publishes);
    const u64 words = take(state.words);
    u64 waits = 0;
    u64 total_ns = 0;
    for (size_t i = 0; i < state.wait_count.size(); ++i) {
        waits += state.wait_count[i];
        total_ns += state.wait_ns[i];
    }
    const auto per_s = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    LOG_INFO(HW_GPU,
             "GPU split hand-over (5 s, per second): front publishes {:.0f}, {:.1f} words each; "
             "by words <=8 {:.0f}, <=32 {:.0f}, <=128 {:.0f}, <=512 {:.0f}, <=2048 {:.0f}, more "
             "{:.0f}. Back waits {:.0f} ({:.1f} ms/s, {:.2f} us each); by length (count / ms/s) "
             "<1us {:.0f}/{:.1f}, 1-4 {:.0f}/{:.1f}, 4-16 {:.0f}/{:.1f}, 16-64 {:.0f}/{:.1f}, "
             "64-256 {:.0f}/{:.1f}, 256us-1ms {:.0f}/{:.1f}, more {:.0f}/{:.1f}; deepest phase "
             "spin {:.0f}, idle hook {:.0f}, sleep {:.0f}; words found on waking <=8 {:.0f}, <=32 "
             "{:.0f}, <=128 {:.0f}, <=512 {:.0f}, <=2048 {:.0f}, more {:.0f}",
             per_s(publishes), publishes != 0 ? static_cast<double>(words) / publishes : 0.0,
             per_s(publish_words[0]), per_s(publish_words[1]), per_s(publish_words[2]),
             per_s(publish_words[3]), per_s(publish_words[4]), per_s(publish_words[5]),
             per_s(waits), per_s(total_ns) / 1e6,
             waits != 0 ? static_cast<double>(total_ns) / waits / 1e3 : 0.0,
             per_s(state.wait_count[0]), per_s(state.wait_ns[0]) / 1e6, per_s(state.wait_count[1]),
             per_s(state.wait_ns[1]) / 1e6, per_s(state.wait_count[2]),
             per_s(state.wait_ns[2]) / 1e6, per_s(state.wait_count[3]),
             per_s(state.wait_ns[3]) / 1e6, per_s(state.wait_count[4]),
             per_s(state.wait_ns[4]) / 1e6, per_s(state.wait_count[5]),
             per_s(state.wait_ns[5]) / 1e6, per_s(state.wait_count[6]),
             per_s(state.wait_ns[6]) / 1e6, per_s(state.wait_phase[0]),
             per_s(state.wait_phase[1]), per_s(state.wait_phase[2]), per_s(state.woke_words[0]),
             per_s(state.woke_words[1]), per_s(state.woke_words[2]), per_s(state.woke_words[3]),
             per_s(state.woke_words[4]), per_s(state.woke_words[5]));
    state.wait_count = {};
    state.wait_ns = {};
    state.wait_phase = {};
    state.woke_words = {};
    state.since = now;
}

} // namespace VideoCommon::Nx2SplitStats
