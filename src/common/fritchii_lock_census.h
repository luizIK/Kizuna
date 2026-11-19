// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

// FRITCHII_LOCK_CENSUS (live, default 0; helper CP, Kakariko): who holds a cache lock when a thread
// has to wait for it. The waits themselves are timed per role already ("cache lock waits" in the
// front's drain sites line); this names the holder.
// - 1: every contended wait of the two cache locks (SpinRecursiveMutex::LockContended) is counted
//   by waiter thread, holder thread (the owner at the start of the wait, by thread name) and lock;
//   one line every 10 s ("Lock census"): ms/s, waits/s, us each, per pair.
// - 2: as 1, plus the holder's stack for E's (GPUBack's) waits: the holder is paused (at most 40
//   a second, ~20-50 us each), its stack read while it still holds the lock, then resumed. A
//   second line names the holders' functions (DbgHelp; the first report loads the symbols, a
//   few seconds and a few hundred MiB on the census thread). Windows only.
// - 3: as 2, for every waiter.
// Off: one relaxed load in the contended path (never on an uncontended lock).
namespace Common::LockCensus {

inline std::atomic<int> g_mode{-1};

/// Reads FRITCHII_LOCK_CENSUS from the environment the first time (-1: not read yet).
int ReadMode();

[[nodiscard]] inline int Mode() {
    const int mode = g_mode.load(std::memory_order_relaxed);
    return mode >= 0 ? mode : ReadMode();
}

/// What a waiter noted at the start of its wait.
struct WaitToken {
    const void* waiter = nullptr; ///< The waiter's census entry
    const void* holder = nullptr; ///< The holder's census entry, null if it had let go already
    const char* lock = nullptr;
    int sample = -1;      ///< The holder stack sample taken for this wait, or -1
    unsigned window = 0;  ///< The report window the sample belongs to
};

/// The start of a contended wait (`owner` is the lock's owner word, re-read while the holder is
/// paused so that a stack is kept only if the holder still holds the lock).
[[nodiscard]] WaitToken BeginWait(int mode, std::thread::id holder, const char* lock,
                                  const std::atomic<std::thread::id>* owner);

/// The wait's end: `ns` waited.
void EndWait(const WaitToken& token, std::uint64_t ns);

/// From SetCurrentThreadName: the thread's name for the census (any thread, any time).
void NameThread(const char* name);

// FRITCHII_FENCE_YIELD_E (live, default 0; helper CP): the fencing thread's buffer and texture
// pops (FenceManager::ReleaseThreadFunc) hold the cache lock for the whole batch. With it, E
// (GPUBack) marks the lock while it waits (SpinRecursiveMutex::BackWaiting), and the fencing
// thread, between two pops of a batch, lets the lock go to E and takes it back once E has it
// (at most 50 us later): the same pops in the same order, each whole under the lock, as EXP74's
// hand-back (only when E waits, with no time limit). 1 = on; 3 = dry: E marks, the fencing thread
// only counts. Line "Fence yield to E" every 10 s from the fencing thread: batches and pops a
// second, pops E waited behind with pops left (the cut's chances) and at the batch's last pop
// (out of its reach). Not with EDEN_NX2_FENCE_RELEASE_STATS (that path is left as it was).
inline std::atomic<int> g_fence_yield_e{-1};

/// Reads FRITCHII_FENCE_YIELD_E from the environment the first time.
int ReadFenceYieldE();

[[nodiscard]] inline int FenceYieldE() {
    const int mode = g_fence_yield_e.load(std::memory_order_relaxed);
    return mode >= 0 ? mode : ReadFenceYieldE();
}

/// The fencing thread, after a batch's pops of one cache: how many pops, how many ended with E
/// waiting and pops left (yielded at 1), whether E waited at the last one; and the line.
void NoteFencePops(int mode, std::uint64_t pops, std::uint64_t chances, bool last_waited,
                   std::uint64_t yield_ns);

// FRITCHII_FENCE_COPY_UNLOCKED (live, default 0; helper CQ, Kakariko): the fencing thread's buffer
// pops copy the downloaded bytes into game memory without the buffer cache's lock
// (BufferCache::PopAsyncBuffersUnlocked; work\helper_cq_answer.md has the ordering argument).
// Under the lock: the landing, the collected CPU writes and the ranges to write; then the copy
// alone outside it; then, under the lock again, the GPU-modified ranges cleared and the fence's
// download marked landed. 1 = on; 2 = on and verify: after each copy, under the lock, the game
// memory is compared with what the locked pop would have written (mismatches must be 0).
// Line "Fence copy unlocked" every 10 s from the fencing thread. Not with
// EDEN_NX2_FENCE_RELEASE_STATS (that path is left as it was).
inline std::atomic<int> g_fence_copy_unlocked{-1};

/// Reads FRITCHII_FENCE_COPY_UNLOCKED from the environment the first time.
int ReadFenceCopyUnlocked();

[[nodiscard]] inline int FenceCopyUnlocked() {
    const int mode = g_fence_copy_unlocked.load(std::memory_order_relaxed);
    return mode >= 0 ? mode : ReadFenceCopyUnlocked();
}

/// FRITCHII_FENCE_COPY_UNLOCKED's line: once the switch was on, E's (GPUBack's) contended cache
/// lock waits that began with the fencing thread holding the lock are timed here
/// (SpinRecursiveMutex::LockContended), with the switch at 0 too, so both arms of a toggle get
/// the number.
inline std::atomic<bool> g_fence_copy_watch{false};
inline std::atomic<std::thread::id> g_fence_thread{};
inline std::atomic<std::uint64_t> g_back_behind_fence_ns{0};
inline std::atomic<std::uint64_t> g_back_behind_fence_waits{0};

/// One batch of the fencing thread's buffer pops, for the line.
struct FenceCopyBatch {
    int mode = 0;
    std::uint64_t pops = 0;
    std::uint64_t groups_unlocked = 0;  ///< landings copied outside the lock
    std::uint64_t groups_locked = 0;    ///< small landings copied under it, as before
    std::uint64_t pieces = 0;           ///< copies made outside the lock
    std::uint64_t bytes = 0;            ///< their bytes
    std::uint64_t unlocked_ns = 0;      ///< outside the lock (copies, marks)
    std::uint64_t relock_ns = 0;        ///< waiting to take the lock back
    std::uint64_t held_ns = 0;          ///< holding it
    std::uint64_t holder_waits = 0;     ///< lock holders that waited for a copy outside the lock
    std::uint64_t holder_wait_ns = 0;
    std::uint64_t verify_bytes = 0;     ///< 2: bytes compared
    std::uint64_t verify_bad_bytes = 0; ///< 2: bytes that differ (must be 0)
    std::uint64_t verify_bad_pieces = 0;
};

/// The fencing thread, after each batch (any mode once the switch was on): the line.
void NoteFenceCopy(const FenceCopyBatch& batch);

} // namespace Common::LockCensus
