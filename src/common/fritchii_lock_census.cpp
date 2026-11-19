// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/fritchii_lock_census.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/nx2_sampler.h"
#include "common/thread.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Common::LockCensus {
namespace {

using Clock = std::chrono::steady_clock;

constexpr unsigned MaxFrames = 24;
constexpr std::size_t MaxSamples = 400;            ///< Holder stacks kept per 10 s window
constexpr std::uint64_t SampleGapNs = 25'000'000; ///< At most 40 holder pauses a second

/// A thread the census has seen, by its std::thread::id. Never erased (a node's address is the
/// key of the pair counts and stays valid).
struct ThreadInfo {
    std::string name;
    unsigned os_id = 0;
    void* handle = nullptr; ///< For the holder stacks (modes 2 and 3), opened on first use
    bool opened = false;
};

std::mutex g_threads_mutex;

std::unordered_map<std::thread::id, ThreadInfo>& Threads() {
    // Leaked on purpose: threads may wait for a lock while statics are destroyed at exit.
    static auto* const threads = new std::unordered_map<std::thread::id, ThreadInfo>();
    return *threads;
}

unsigned OsIdOf([[maybe_unused]] std::thread::id id) {
#if defined(_WIN32) && defined(_MSVC_STL_VERSION)
    // The MSVC library's thread::id is the Win32 thread id.
    static_assert(sizeof(std::thread::id) == sizeof(unsigned));
    unsigned value;
    std::memcpy(&value, &id, sizeof(value));
    return value;
#else
    return 0;
#endif
}

/// The census entry of thread `id` (made with a placeholder name for a thread never named).
/// Called with g_threads_mutex held.
ThreadInfo& InfoLocked(std::thread::id id) {
    auto [it, is_new] = Threads().try_emplace(id);
    if (is_new) {
        it->second.os_id = OsIdOf(id);
        it->second.name = fmt::format("unnamed {}", it->second.os_id);
    }
    return it->second;
}

thread_local ThreadInfo* t_self = nullptr;

ThreadInfo* Self() {
    if (t_self == nullptr) {
        const std::scoped_lock lock{g_threads_mutex};
        t_self = &InfoLocked(std::this_thread::get_id());
    }
    return t_self;
}

struct Totals {
    std::uint64_t ns = 0;
    std::uint64_t count = 0;
};

struct StackSample {
    const ThreadInfo* waiter;
    const ThreadInfo* holder;
    const char* lock;
    unsigned depth;
    std::array<unsigned long long, MaxFrames> frames;
    std::uint64_t ns; ///< The wait it was taken in (0 until the wait ends)
};

std::mutex g_data_mutex;
std::map<std::tuple<const ThreadInfo*, const ThreadInfo*, const char*>, Totals> g_pairs;
std::vector<StackSample> g_samples;
Totals g_released;               ///< Waits whose holder had let go when the waiter looked
std::uint64_t g_pause_missed = 0; ///< Holder pauses that found the lock released
unsigned g_window = 1;

std::atomic<std::uint64_t> g_next_sample_ns{0};
std::once_flag g_reporter_once;

std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

struct PauseTest {
    const std::atomic<std::thread::id>* owner;
    std::thread::id holder;
};

/// Run while the holder is paused: nothing here locks or allocates.
bool StillHolds(const void* context) {
    const auto* test = static_cast<const PauseTest*>(context);
    return test->owner->load(std::memory_order_relaxed) == test->holder;
}

void Report(double seconds) {
    decltype(g_pairs) pairs;
    std::vector<StackSample> samples;
    Totals released;
    std::uint64_t pause_missed;
    {
        const std::scoped_lock lock{g_data_mutex};
        pairs.swap(g_pairs);
        samples.swap(g_samples);
        released = std::exchange(g_released, Totals{});
        pause_missed = std::exchange(g_pause_missed, 0);
        ++g_window;
    }
    const int mode = Mode();
    std::vector<std::pair<Totals, std::string>> rows;
    {
        const std::scoped_lock lock{g_threads_mutex}; // the names
        for (const auto& [key, totals] : pairs) {
            const auto& [waiter, holder, lock_name] = key;
            rows.emplace_back(totals, fmt::format("{} <- {} ({})", waiter->name, holder->name,
                                                  lock_name != nullptr ? lock_name : "lock"));
        }
    }
    std::ranges::sort(rows, [](const auto& a, const auto& b) { return a.first.ns > b.first.ns; });
    std::string text;
    for (std::size_t i = 0; i < rows.size() && i < 12; ++i) {
        const Totals& t = rows[i].first;
        text += fmt::format("{}{} {:.1f} ms/s ({:.0f}/s, {:.1f} us each)", text.empty() ? "" : "; ",
                            rows[i].second, static_cast<double>(t.ns) / 1e6 / seconds,
                            static_cast<double>(t.count) / seconds,
                            t.count != 0 ? static_cast<double>(t.ns) / 1e3 /
                                               static_cast<double>(t.count)
                                         : 0.0);
    }
    LOG_INFO(HW_GPU,
             "Lock census (FRITCHII_LOCK_CENSUS={}) {:.0f} s, contended cache lock waits, waiter "
             "<- holder (lock): {}; holder gone before the waiter looked {:.1f} ms/s ({:.0f}/s)",
             mode, seconds, text.empty() ? "none" : text,
             static_cast<double>(released.ns) / 1e6 / seconds,
             static_cast<double>(released.count) / seconds);
    if (samples.empty() && pause_missed == 0) {
        return;
    }
    // The holders' stacks: the first frames (where the holder is) and every function on them.
    std::unordered_map<unsigned long long, std::string> names;
    const auto name_of = [&names](unsigned long long address) -> const std::string& {
        auto [it, is_new] = names.try_emplace(address);
        if (is_new) {
            it->second = Nx2Sampler::SymbolName(address);
        }
        return it->second;
    };
    std::map<std::string, Totals> chains;
    std::map<std::string, std::uint64_t> inclusive;
    std::uint64_t taken = 0;
    for (const StackSample& sample : samples) {
        if (sample.depth == 0) {
            continue;
        }
        ++taken;
        std::string chain;
        {
            const std::scoped_lock lock{g_threads_mutex};
            chain = sample.holder->name + ": ";
        }
        for (unsigned i = 0; i < sample.depth && i < 5; ++i) {
            chain += (i == 0 ? "" : " < ") + name_of(sample.frames[i]);
        }
        Totals& c = chains[chain];
        c.ns += sample.ns;
        ++c.count;
        std::set<std::string> seen;
        for (unsigned i = 0; i < sample.depth; ++i) {
            const std::string& name = name_of(sample.frames[i]);
            if (seen.insert(name).second) {
                ++inclusive[name];
            }
        }
    }
    std::vector<std::pair<Totals, std::string>> chain_rows;
    for (auto& [chain, totals] : chains) {
        chain_rows.emplace_back(totals, chain);
    }
    std::ranges::sort(chain_rows,
                      [](const auto& a, const auto& b) { return a.first.count > b.first.count; });
    std::string chain_text;
    for (std::size_t i = 0; i < chain_rows.size() && i < 8; ++i) {
        const Totals& t = chain_rows[i].first;
        chain_text += fmt::format("{}{:.0f}% (waits {:.1f} us each) {}", chain_text.empty() ? "" : "; ",
                                  100.0 * static_cast<double>(t.count) / static_cast<double>(taken),
                                  t.count != 0 ? static_cast<double>(t.ns) / 1e3 /
                                                     static_cast<double>(t.count)
                                               : 0.0,
                                  chain_rows[i].second);
    }
    std::vector<std::pair<std::uint64_t, std::string>> inclusive_rows;
    for (auto& [name, count] : inclusive) {
        inclusive_rows.emplace_back(count, name);
    }
    std::ranges::sort(inclusive_rows,
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string inclusive_text;
    for (std::size_t i = 0; i < inclusive_rows.size() && i < 16; ++i) {
        inclusive_text += fmt::format("{}{:.0f}% {}", inclusive_text.empty() ? "" : "; ",
                                      100.0 * static_cast<double>(inclusive_rows[i].first) /
                                          static_cast<double>(taken),
                                      inclusive_rows[i].second);
    }
    LOG_INFO(HW_GPU,
             "Lock census holder stacks ({} taken, {} found the lock let go): where the holder "
             "was: {}; functions on the holders' stacks: {}",
             taken, pause_missed, chain_text.empty() ? "none" : chain_text,
             inclusive_text.empty() ? "none" : inclusive_text);
}

void ReporterLoop() {
    Common::SetCurrentThreadName("LockCensus");
    auto last = Clock::now();
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        const auto now = Clock::now();
        if (Mode() <= 0) {
            // Off: what was counted is dropped, the next window starts when it is on again.
            const std::scoped_lock lock{g_data_mutex};
            g_pairs.clear();
            g_samples.clear();
            g_released = {};
            g_pause_missed = 0;
            ++g_window;
            last = now;
            continue;
        }
        const double seconds = std::chrono::duration<double>(now - last).count();
        if (seconds < 10.0) {
            continue;
        }
        last = now;
        Report(seconds);
    }
}

} // Anonymous namespace

int ReadMode() {
    const char* const value = std::getenv("FRITCHII_LOCK_CENSUS");
    int unset = -1;
    g_mode.compare_exchange_strong(unset, value != nullptr ? (std::max)(0, std::atoi(value)) : 0,
                                   std::memory_order_relaxed);
    return g_mode.load(std::memory_order_relaxed);
}

WaitToken BeginWait(int mode, std::thread::id holder, const char* lock,
                    const std::atomic<std::thread::id>* owner) {
    std::call_once(g_reporter_once, [] { std::thread{ReporterLoop}.detach(); });
    WaitToken token{};
    token.waiter = Self();
    token.lock = lock;
    if (holder == std::thread::id{}) {
        return token; // let go between the failed try and this look
    }
    ThreadInfo* info;
    void* handle = nullptr;
    const bool stacks =
        mode >= 3 ||
        (mode == 2 && Nx2Hitch::t_role == Nx2Hitch::Role::GpuBack);
    {
        const std::scoped_lock threads_lock{g_threads_mutex};
        info = &InfoLocked(holder);
        if (stacks && !info->opened) {
            info->opened = true;
            info->handle = info->os_id != 0 ? Nx2Sampler::OpenThreadForCapture(info->os_id) : nullptr;
        }
        handle = info->handle;
    }
    token.holder = info;
    if (!stacks || handle == nullptr) {
        return token;
    }
    const std::uint64_t now = NowNs();
    std::uint64_t next = g_next_sample_ns.load(std::memory_order_relaxed);
    if (now < next ||
        !g_next_sample_ns.compare_exchange_strong(next, now + SampleGapNs,
                                                  std::memory_order_relaxed)) {
        return token;
    }
    // No census lock is held while the holder is paused (it may be in this code itself).
    const PauseTest test{owner, holder};
    std::array<unsigned long long, MaxFrames> frames{};
    const unsigned depth =
        Nx2Sampler::CaptureSuspended(handle, frames.data(), MaxFrames, &StillHolds, &test);
    const std::scoped_lock data_lock{g_data_mutex};
    if (depth == 0) {
        ++g_pause_missed;
        return token;
    }
    if (g_samples.size() < MaxSamples) {
        token.sample = static_cast<int>(g_samples.size());
        token.window = g_window;
        g_samples.push_back(StackSample{static_cast<const ThreadInfo*>(token.waiter), info, lock,
                                        depth, frames, 0});
    }
    return token;
}

void EndWait(const WaitToken& token, std::uint64_t ns) {
    const std::scoped_lock lock{g_data_mutex};
    if (token.holder == nullptr) {
        g_released.ns += ns;
        ++g_released.count;
        return;
    }
    Totals& totals = g_pairs[{static_cast<const ThreadInfo*>(token.waiter),
                              static_cast<const ThreadInfo*>(token.holder), token.lock}];
    totals.ns += ns;
    ++totals.count;
    if (token.sample >= 0 && token.window == g_window &&
        static_cast<std::size_t>(token.sample) < g_samples.size()) {
        g_samples[static_cast<std::size_t>(token.sample)].ns = ns;
    }
}

int ReadFenceYieldE() {
    const char* const value = std::getenv("FRITCHII_FENCE_YIELD_E");
    int unset = -1;
    g_fence_yield_e.compare_exchange_strong(
        unset, value != nullptr ? (std::max)(0, std::atoi(value)) : 0, std::memory_order_relaxed);
    return g_fence_yield_e.load(std::memory_order_relaxed);
}

void NoteFencePops(int mode, std::uint64_t pops, std::uint64_t chances, bool last_waited,
                   std::uint64_t yield_ns) {
    // The fencing thread only: plain counters.
    static std::uint64_t batches = 0;
    static std::uint64_t pop_count = 0;
    static std::uint64_t chance_count = 0;
    static std::uint64_t last_count = 0;
    static std::uint64_t yielded_ns = 0;
    static auto since = Clock::now();
    ++batches;
    pop_count += pops;
    chance_count += chances;
    last_count += last_waited ? 1 : 0;
    yielded_ns += yield_ns;
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - since).count();
    if (seconds < 10.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Fence yield to E (FRITCHII_FENCE_YIELD_E={}) {:.0f} s: pop batches {:.0f}/s, pops "
             "{:.0f}/s; E waited at the end of a pop with pops left {:.0f}/s ({}), at the "
             "batch's last pop {:.0f}/s (out of reach); handed to E for {:.2f} ms/s",
             mode, seconds, static_cast<double>(batches) / seconds,
             static_cast<double>(pop_count) / seconds,
             static_cast<double>(chance_count) / seconds, mode == 1 ? "yielded" : "dry, counted",
             static_cast<double>(last_count) / seconds,
             static_cast<double>(yielded_ns) / 1e6 / seconds);
    batches = pop_count = chance_count = last_count = yielded_ns = 0;
    since = now;
}

int ReadFenceCopyUnlocked() {
    const char* const value = std::getenv("FRITCHII_FENCE_COPY_UNLOCKED");
    if (value != nullptr) {
        // Set at all (0 too): the line runs from the start, for both arms of a toggle.
        g_fence_copy_watch.store(true, std::memory_order_relaxed);
    }
    int unset = -1;
    g_fence_copy_unlocked.compare_exchange_strong(
        unset, value != nullptr ? (std::max)(0, std::atoi(value)) : 0, std::memory_order_relaxed);
    return g_fence_copy_unlocked.load(std::memory_order_relaxed);
}

void NoteFenceCopy(const FenceCopyBatch& batch) {
    // The fencing thread only: plain totals.
    static FenceCopyBatch totals{};
    static std::uint64_t batches = 0;
    static auto since = Clock::now();
    ++batches;
    totals.pops += batch.pops;
    totals.groups_unlocked += batch.groups_unlocked;
    totals.groups_locked += batch.groups_locked;
    totals.pieces += batch.pieces;
    totals.bytes += batch.bytes;
    totals.unlocked_ns += batch.unlocked_ns;
    totals.relock_ns += batch.relock_ns;
    totals.held_ns += batch.held_ns;
    totals.holder_waits += batch.holder_waits;
    totals.holder_wait_ns += batch.holder_wait_ns;
    totals.verify_bytes += batch.verify_bytes;
    totals.verify_bad_bytes += batch.verify_bad_bytes;
    totals.verify_bad_pieces += batch.verify_bad_pieces;
    const auto now = Clock::now();
    const double seconds = std::chrono::duration<double>(now - since).count();
    if (seconds < 10.0) {
        return;
    }
    const auto per_s = [seconds](std::uint64_t value) {
        return static_cast<double>(value) / seconds;
    };
    const auto ms_s = [seconds](std::uint64_t ns) {
        return static_cast<double>(ns) / 1e6 / seconds;
    };
    const std::uint64_t behind_ns = g_back_behind_fence_ns.exchange(0, std::memory_order_relaxed);
    const std::uint64_t behind = g_back_behind_fence_waits.exchange(0, std::memory_order_relaxed);
    std::string verify;
    if (batch.mode == 2) {
        verify = fmt::format("; verify: {:.0f} KB/s compared, {} bytes differ in {} copies "
                             "(must be 0)",
                             per_s(totals.verify_bytes) / 1024.0, totals.verify_bad_bytes,
                             totals.verify_bad_pieces);
    }
    LOG_INFO(HW_GPU,
             "Fence copy unlocked (FRITCHII_FENCE_COPY_UNLOCKED={}) {:.0f} s: pop batches "
             "{:.0f}/s, pops {:.0f}/s; landings copied outside the lock {:.0f}/s, small ones "
             "under it {:.0f}/s; copies {:.0f}/s, {:.2f} MB/s, {:.2f} ms/s out of the lock; "
             "fencing thread held the buffer lock {:.2f} ms/s, waited to take it back {:.2f} "
             "ms/s; lock holders waited for a copy {:.0f}/s, {:.3f} ms/s; E's cache lock waits "
             "behind the fencing thread {:.2f} ms/s ({:.0f}/s, {:.1f} us each){}",
             batch.mode, seconds, per_s(batches), per_s(totals.pops),
             per_s(totals.groups_unlocked), per_s(totals.groups_locked), per_s(totals.pieces),
             per_s(totals.bytes) / 1e6, ms_s(totals.unlocked_ns), ms_s(totals.held_ns),
             ms_s(totals.relock_ns), per_s(totals.holder_waits), ms_s(totals.holder_wait_ns),
             ms_s(behind_ns), per_s(behind),
             behind != 0 ? static_cast<double>(behind_ns) / 1e3 / static_cast<double>(behind)
                         : 0.0,
             verify);
    totals = FenceCopyBatch{};
    batches = 0;
    since = now;
}

void NameThread(const char* name) {
    const std::scoped_lock lock{g_threads_mutex};
    ThreadInfo& info = InfoLocked(std::this_thread::get_id());
    info.name = name != nullptr ? name : "";
#ifdef _WIN32
    info.os_id = static_cast<unsigned>(GetCurrentThreadId());
#endif
}

} // namespace Common::LockCensus
