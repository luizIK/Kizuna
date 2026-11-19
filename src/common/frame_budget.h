// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"

// Where each frame's time goes, for the performance overlay's time rows (Fritchii 2026-11-16).
// The overlay turns the rows on with SetWanted; until then nothing here runs but one relaxed
// load per Vulkan submission. Then:
//  - the Vulkan scheduler writes one timestamp pair per submission (start of its upload command
//    buffer, end of its main one) and, without waiting, reads back finished submissions. Their
//    union on the GPU's clock is the GPU's busy time (AddGpuBusy).
//  - the threads that set the frame rate register by name (RegisterThread, from
//    SetCurrentThreadName): the GPU front ("GPU"), the back ("GPUBack", "GPUBackE2"), the render
//    thread ("GPURender") and the game's cores ("CPUCore_N"). Take reads their CPU time.
// Take, from the overlay about once a second, returns busy shares of the wall time since the
// previous Take; times the frame time they are milliseconds per frame.
namespace Common::FrameBudget {

enum class Group : u8 { Front, Back, Render, Game, Count };
inline constexpr size_t Groups = static_cast<size_t>(Group::Count);

namespace Detail {
inline std::atomic<bool> g_wanted{};
inline std::atomic<u64> g_gpu_busy_ns{};
/// 0: no GPU measurement yet, 1: timestamps work, 2: the GPU or driver can't time the queue.
inline std::atomic<u8> g_gpu_state{};
} // namespace Detail

/// From the overlay: the time rows are on screen (true) or not.
inline void SetWanted(bool wanted) {
    Detail::g_wanted.store(wanted, std::memory_order_relaxed);
}

/// Whether anything shows the rows (the scheduler times submissions only then).
[[nodiscard]] inline bool Wanted() {
    return Detail::g_wanted.load(std::memory_order_relaxed);
}

/// From the scheduler: GPU busy time from submissions that finished.
inline void AddGpuBusy(u64 ns) {
    Detail::g_gpu_busy_ns.fetch_add(ns, std::memory_order_relaxed);
    Detail::g_gpu_state.store(1, std::memory_order_relaxed);
}

/// From the scheduler: timestamps are unsupported or failed; the overlay shows "n/a".
inline void SetGpuUnsupported() {
    Detail::g_gpu_state.store(2, std::memory_order_relaxed);
}

/// From SetCurrentThreadName: keeps the threads above, ignores the rest.
void RegisterThread(const char* name);

struct Sample {
    bool valid{};      ///< false on the first Take after a pause (no window yet)
    double seconds{};  ///< wall time since the previous Take
    double gpu = -1;   ///< GPU busy share of the wall time, -1 when not measured
    bool gpu_unsupported{};
    double cpu = -1;   ///< Fritchii's CPU time over (wall time x logical CPUs), -1 when unknown
    /// Busy share of the busiest thread in each group, -1 when the group has no thread.
    std::array<double, Groups> busy{-1, -1, -1, -1};
};

/// Busy shares since the previous call. One caller (the overlay, on the UI thread).
[[nodiscard]] Sample Take();

/// Forgets the previous Take, so the next one starts a new window (after a pause or a hide).
void Reset();

} // namespace Common::FrameBudget
