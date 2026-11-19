// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/one_follower_hooks.h"

namespace Core::Nx2ProtectStats {

/// EDEN_NX2_PROTECT_STATS=1 (read once): counts the host page protection calls the device memory
/// manager makes when a page's cached count moves between 0 and 1 (protect = the page became
/// tracked, unprotect = it became untracked), how long they take, how many follow the previous
/// call of the same kind on the next page within one batch (one call could have done both), and
/// the pages that change most. Logged every 5 s ("Page protection (5 s"). Fable's next70 ask 3.
/// Also (fable52_answer_after88 1.6, with EXP89 off or on): for every page a protect call covers,
/// how many frames ago that guest page was last unprotected ("Page protection frames since the
/// last unprotect"), and EXP89's lingering protection counters ("Lingering protection (EXP89").
[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PROTECT_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

struct PageCount {
    u32 protects{};
    u32 unprotects{};
};

/// Frames presented so far: raised once a frame by RasterizerVulkan::TickFrame (TickFrame below),
/// the same place that runs EXP89's sweep. Counted with the stats off too.
inline std::atomic<u64> g_frame{0};

/// Buckets of "frames since this page was last unprotected" at a protect: 0 (same frame), 1, 2,
/// 3-4, 5-16, 17-64, more, never (or longer ago than the last 5 s report's 64-frame trim). A
/// lingering protection of N frames (EXP89=N) cancels the protects up to N frames after.
inline constexpr size_t FramesAgoBuckets = 8;

[[nodiscard]] inline size_t FramesAgoBucket(u64 frames) {
    if (frames <= 2) {
        return static_cast<size_t>(frames);
    }
    if (frames <= 4) {
        return 3;
    }
    if (frames <= 16) {
        return 4;
    }
    return frames <= 64 ? 5 : 6;
}

struct State {
    std::atomic<u64> calls[2]{};      ///< [0] unprotect, [1] protect
    std::atomic<u64> bytes[2]{};
    std::atomic<u64> ns[2]{};
    std::atomic<u64> mergeable[2]{};  ///< Calls that start where the previous same-kind call ended
    std::atomic<u64> batches{};
    std::mutex pages_mutex;
    std::unordered_map<u64, PageCount> pages; ///< Guest page number -> changes this window
    /// Guest page number -> frame of its last unprotect call; entries older than 64 frames are
    /// trimmed at each report. Under pages_mutex.
    std::unordered_map<u64, u64> last_unprotect;
    std::atomic<u64> frames_ago[FramesAgoBuckets]{}; ///< Protected pages by FramesAgoBucket
    // EXP89 (DeviceMemoryManager's lingering protection), pages:
    std::atomic<u64> linger_noted{};      ///< 1 -> 0 left protected: unprotects not made
    std::atomic<u64> linger_cancelled{};  ///< 0 -> 1 of a lingering page: protects not made
    std::atomic<u64> linger_swept{};      ///< Uncached by the once-a-frame sweep
    std::atomic<u64> linger_mapping{};    ///< Uncached at once by Map, Unmap or UnregisterProcess
    std::atomic<u64> linger_held{};       ///< Lingering after the last sweep
    std::atomic<u64> linger_left_after_unmap{}; ///< Verify: must stay 0
    // Round 5 (helper DE): the calls one follower's engine thread E (GPUBack) makes, and E's
    // protects of pages it unprotected 0-1 frames before (what EXP89=1 would not call, with
    // their unprotects): E's share of the cycle, the wall thread's ceiling for EXP89.
    std::atomic<u64> e_calls[2]{};
    std::atomic<u64> e_ns[2]{};
    std::atomic<u64> e_repeat_calls{};
    std::atomic<u64> e_repeat_ns{};
    std::atomic<s64> next_report_ns{};
};

inline State& Get() {
    static State state;
    return state;
}

/// EXP89 counters (pages); call only with Enabled().
inline void AddLinger(std::atomic<u64> State::*counter, u64 pages) {
    (Get().*counter).fetch_add(pages, std::memory_order_relaxed);
}

/// The previous call on this thread inside the current batch, for the mergeable count.
inline thread_local u64 t_last_end = ~u64{0};
inline thread_local bool t_last_cached = false;

inline void BeginBatch() {
    t_last_end = ~u64{0};
    Get().batches.fetch_add(1, std::memory_order_relaxed);
}

inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline void Report(State& state, s64 now) {
    s64 next = state.next_report_ns.load(std::memory_order_relaxed);
    if (next == 0) {
        state.next_report_ns.compare_exchange_strong(next, now + 5'000'000'000LL);
        return;
    }
    if (now < next || !state.next_report_ns.compare_exchange_strong(next, now + 5'000'000'000LL)) {
        return;
    }
    const auto take = [](std::atomic<u64>& value) {
        return value.exchange(0, std::memory_order_relaxed);
    };
    const u64 calls_u = take(state.calls[0]);
    const u64 calls_p = take(state.calls[1]);
    const u64 bytes_u = take(state.bytes[0]);
    const u64 bytes_p = take(state.bytes[1]);
    const u64 ns_u = take(state.ns[0]);
    const u64 ns_p = take(state.ns[1]);
    const u64 merge_u = take(state.mergeable[0]);
    const u64 merge_p = take(state.mergeable[1]);
    const u64 batches = take(state.batches);
    u64 frames_ago[FramesAgoBuckets]{};
    u64 frames_ago_total = 0;
    for (size_t i = 0; i < FramesAgoBuckets; ++i) {
        frames_ago[i] = take(state.frames_ago[i]);
        frames_ago_total += frames_ago[i];
    }
    const u64 linger_noted = take(state.linger_noted);
    const u64 linger_cancelled = take(state.linger_cancelled);
    const u64 linger_swept = take(state.linger_swept);
    const u64 linger_mapping = take(state.linger_mapping);
    const u64 linger_left = take(state.linger_left_after_unmap);
    const u64 linger_held = state.linger_held.load(std::memory_order_relaxed);
    std::vector<std::pair<u64, PageCount>> top;
    size_t distinct = 0;
    size_t cycled = 0;
    size_t last_unprotect_kept = 0;
    {
        std::scoped_lock lock{state.pages_mutex};
        distinct = state.pages.size();
        top.reserve(distinct);
        for (const auto& [page, count] : state.pages) {
            cycled += (count.protects > 0 && count.unprotects > 0) ? 1 : 0;
            top.emplace_back(page, count);
        }
        state.pages.clear();
        const u64 frame = g_frame.load(std::memory_order_relaxed);
        std::erase_if(state.last_unprotect,
                      [frame](const auto& entry) { return frame > entry.second + 64; });
        last_unprotect_kept = state.last_unprotect.size();
    }
    const size_t keep = std::min<size_t>(16, top.size());
    std::partial_sort(top.begin(), top.begin() + keep, top.end(), [](const auto& a, const auto& b) {
        return a.second.protects + a.second.unprotects > b.second.protects + b.second.unprotects;
    });
    std::string pages_text;
    for (size_t i = 0; i < keep; ++i) {
        pages_text += fmt::format(" {:#x}:{}/{}", top[i].first << 12, top[i].second.protects,
                                  top[i].second.unprotects);
    }
    LOG_INFO(HW_GPU,
             "Page protection (5 s, per second): protect {:.0f} calls {:.0f} pages {:.2f} ms, "
             "unprotect {:.0f} calls {:.0f} pages {:.2f} ms; could merge with the previous call: "
             "protect {:.0f}, unprotect {:.0f}; batches {:.0f}; pages changed {} (both ways {})",
             calls_p / 5.0, bytes_p / 4096.0 / 5.0, ns_p / 5e6, calls_u / 5.0,
             bytes_u / 4096.0 / 5.0, ns_u / 5e6, merge_p / 5.0, merge_u / 5.0, batches / 5.0,
             distinct, cycled);
    LOG_INFO(HW_GPU, "Page protection top pages (address:protects/unprotects in 5 s):{}",
             pages_text);
    // Share of the protected pages a lingering protection of N frames would have cancelled.
    const auto share = [&](size_t last_bucket) {
        u64 sum = 0;
        for (size_t i = 0; i <= last_bucket; ++i) {
            sum += frames_ago[i];
        }
        return frames_ago_total == 0 ? 0.0 : 100.0 * static_cast<double>(sum) / frames_ago_total;
    };
    LOG_INFO(HW_GPU,
             "Page protection frames since the last unprotect (5 s, protected pages): 0:{} 1:{} "
             "2:{} 3-4:{} 5-16:{} 17-64:{} more:{} never:{}; EXP89=N would cancel: N=1 {:.1f}%, "
             "N=2 {:.1f}%, N=4 {:.1f}%, N=16 {:.1f}%, N=64 {:.1f}%; pages remembered {}",
             frames_ago[0], frames_ago[1], frames_ago[2], frames_ago[3], frames_ago[4],
             frames_ago[5], frames_ago[6], frames_ago[7], share(1), share(2), share(3), share(4),
             share(5), last_unprotect_kept);
    {
        const u64 e_u = take(state.e_calls[0]);
        const u64 e_p = take(state.e_calls[1]);
        const u64 e_ns_u = take(state.e_ns[0]);
        const u64 e_ns_p = take(state.e_ns[1]);
        const u64 e_rep = take(state.e_repeat_calls);
        const u64 e_rep_ns = take(state.e_repeat_ns);
        // The unprotect each repeat protect pairs with, priced at E's average unprotect.
        const double pair_ns =
            static_cast<double>(e_rep_ns) +
            (e_u != 0 ? static_cast<double>(e_rep) * static_cast<double>(e_ns_u) / e_u : 0.0);
        LOG_INFO(HW_GPU,
                 "Page protection on GPUBack (E, 5 s, per second): protect {:.0f} calls {:.2f} ms, "
                 "unprotect {:.0f} calls {:.2f} ms; protects of a page E unprotected 0-1 frames "
                 "before {:.0f} calls {:.2f} ms, with their unprotects ~{:.2f} ms (EXP89=1's "
                 "ceiling on E; its cost is the guest's trapped stores on lingering pages, the "
                 "Store memo line)",
                 e_p / 5.0, e_ns_p / 5e6, e_u / 5.0, e_ns_u / 5e6, e_rep / 5.0, e_rep_ns / 5e6,
                 pair_ns / 5e6);
    }
    LOG_INFO(HW_GPU,
             "Lingering protection (EXP89, 5 s, pages per second): unprotects cancelled {:.0f}, "
             "protects cancelled {:.0f}, released by the sweep {:.0f}, by map/unmap {:.0f}; held "
             "{}; left after unmap {} (must be 0)",
             linger_noted / 5.0, linger_cancelled / 5.0, linger_swept / 5.0, linger_mapping / 5.0,
             linger_held, linger_left);
}

/// Once a frame from RasterizerVulkan::TickFrame, before EXP89's sweep.
inline void TickFrame() {
    g_frame.fetch_add(1, std::memory_order_relaxed);
    if (Enabled()) [[unlikely]] {
        Report(Get(), NowNs());
    }
}

/// Wraps one MarkRegionCaching call: `vaddr`/`bytes` are the guest range, `cached` true when
/// the pages became tracked (protected).
template <typename Func>
inline void Count(u64 vaddr, u64 bytes, bool cached, Func&& call) {
    State& state = Get();
    const s64 start = NowNs();
    call();
    const s64 end = NowNs();
    const size_t kind = cached ? 1 : 0;
    const bool on_e = VideoCommon::OneFollower::Detail::t_engine;
    bool repeat = false;
    state.calls[kind].fetch_add(1, std::memory_order_relaxed);
    state.bytes[kind].fetch_add(bytes, std::memory_order_relaxed);
    state.ns[kind].fetch_add(static_cast<u64>(end - start), std::memory_order_relaxed);
    if (t_last_end == vaddr && t_last_cached == cached) {
        state.mergeable[kind].fetch_add(1, std::memory_order_relaxed);
    }
    t_last_end = vaddr + bytes;
    t_last_cached = cached;
    {
        const u64 frame = g_frame.load(std::memory_order_relaxed);
        std::scoped_lock lock{state.pages_mutex};
        for (u64 page = vaddr >> 12; page < (vaddr + bytes + 4095) >> 12; ++page) {
            PageCount& count = state.pages[page];
            if (cached) {
                ++count.protects;
                const auto it = state.last_unprotect.find(page);
                size_t bucket = FramesAgoBuckets - 1; // never (or trimmed)
                if (it != state.last_unprotect.end()) {
                    bucket = FramesAgoBucket(frame > it->second ? frame - it->second : 0);
                    repeat |= frame <= it->second + 1;
                }
                state.frames_ago[bucket].fetch_add(1, std::memory_order_relaxed);
            } else {
                ++count.unprotects;
                state.last_unprotect[page] = frame;
            }
        }
    }
    if (on_e) {
        state.e_calls[kind].fetch_add(1, std::memory_order_relaxed);
        state.e_ns[kind].fetch_add(static_cast<u64>(end - start), std::memory_order_relaxed);
        if (repeat) {
            state.e_repeat_calls.fetch_add(1, std::memory_order_relaxed);
            state.e_repeat_ns.fetch_add(static_cast<u64>(end - start), std::memory_order_relaxed);
        }
    }
    Report(state, end);
}

} // namespace Core::Nx2ProtectStats
