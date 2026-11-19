// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

/// EXP88 (ported, fable52_answer_next70 section 3): the
/// constant buffer path's switch bits and counters. See EXP88 in nx2_debug_control.h.
namespace VideoCommon::Nx2CbPath {

/// 1: collected writes skip BufferCache::WriteMemory where it would change nothing (part c).
inline constexpr int SkipBit = 1;
/// 2: constant buffer data goes to the bound buffer's device address, translated once per
/// bind, and its device ranges are logged for FlushCaching (parts a and b).
inline constexpr int DirectBit = 2;
/// 4: verify both: skipped ranges take the locked WriteMemory anyway after checking under the
/// lock that it changes nothing; direct writes are read back and their translation redone.
inline constexpr int VerifyBit = 4;

/// EXP88 now: one relaxed load once the switches are loaded, so it toggles within a session.
[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::Experiment(88);
}

struct Counters {
    // Part c, FlushCaching's ranges (RasterizerVulkan::InnerInvalidation).
    std::atomic<u64> ranges{};       ///< Ranges seen while bit 1 is on
    std::atomic<u64> skipped{};      ///< Ranges that skipped WriteMemory (or, verifying, would)
    std::atomic<u64> locked{};       ///< Ranges sent to WriteMemory under the lock
    std::atomic<u64> kept_synced{};  ///< ... because a page ever left the CPU-modified state
    std::atomic<u64> kept_gpu{};     ///< ... because a page may hold GPU-written data
    std::atomic<u64> kept_vertex{};  ///< ... because an enabled vertex binding overlaps it
    std::atomic<u64> lock_skips{};   ///< Calls that took no buffer lock at all
    // Part c on the collected guest writes (RasterizerVulkan::OnCacheInvalidations).
    std::atomic<u64> guest_ranges{};
    std::atomic<u64> guest_skipped{};
    std::atomic<u64> guest_claim_skips{}; ///< Batches whose ClaimCpuWrites needed no lock
    /// FRITCHII_GATHER_ONE_LOCK: gathers whose buffer part took the lock once for all ranges.
    std::atomic<u64> guest_one_lock_holds{};
    // Parts a and b (Maxwell3D::WriteBoundConstantBuffer).
    std::atomic<u64> direct_writes{};
    std::atomic<u64> fallback_device0{}; ///< The bound buffer isn't one mapped device range
    std::atomic<u64> fallback_sparse{};  ///< The write touches Switch 2 sparse memory
    // Verify (bit 4), whole session: both must stay 0.
    std::atomic<u64> verify_skip_checked{};
    std::atomic<u64> verify_skip_mismatch{}; ///< A skipped range's WriteMemory would change state
    std::atomic<u64> verify_direct_checked{};
    std::atomic<u64> verify_direct_mismatch{}; ///< Bytes read back differ, or translation differs
    std::atomic<u64> calls{};
    std::atomic<s64> last_log{};
};
inline Counters g_counters;

inline void Count(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

/// Logs the counters every 5 s while EXP88 is not 0. Called by the drawing thread at each
/// FlushCaching invalidation; looks at the clock every 64th call.
inline void MaybeLog(int mode) {
    if (mode == 0) {
        return;
    }
    Counters& c = g_counters;
    if ((c.calls.fetch_add(1, std::memory_order_relaxed) & 63) != 0) {
        return;
    }
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = c.last_log.load(std::memory_order_relaxed);
    if (last == 0) {
        c.last_log.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 5000 || !c.last_log.compare_exchange_strong(last, now)) {
        return;
    }
    const u64 ranges = c.ranges.exchange(0, std::memory_order_relaxed);
    const u64 skipped = c.skipped.exchange(0, std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "Constant buffer path (EXP88={}, 5 s): flush ranges {}, c skipped {} ({:.1f}%), "
             "locked {} (kept: ever synchronized {}, may be GPU-written {}, vertex binding {}), "
             "{} flushes without the buffer lock; guest write ranges {}, c skipped {}, {} claims "
             "without the lock, {} gathers under one buffer lock (FRITCHII_GATHER_ONE_LOCK); a+b "
             "direct writes {}, fallbacks device 0 {}, sparse {}; verify "
             "skip {} checked, {} would change (whole session, must be 0), direct {} checked, {} "
             "differ (whole session, must be 0)",
             mode, ranges, skipped, ranges != 0 ? 100.0 * double(skipped) / double(ranges) : 0.0,
             c.locked.exchange(0, std::memory_order_relaxed),
             c.kept_synced.exchange(0, std::memory_order_relaxed),
             c.kept_gpu.exchange(0, std::memory_order_relaxed),
             c.kept_vertex.exchange(0, std::memory_order_relaxed),
             c.lock_skips.exchange(0, std::memory_order_relaxed),
             c.guest_ranges.exchange(0, std::memory_order_relaxed),
             c.guest_skipped.exchange(0, std::memory_order_relaxed),
             c.guest_claim_skips.exchange(0, std::memory_order_relaxed),
             c.guest_one_lock_holds.exchange(0, std::memory_order_relaxed),
             c.direct_writes.exchange(0, std::memory_order_relaxed),
             c.fallback_device0.exchange(0, std::memory_order_relaxed),
             c.fallback_sparse.exchange(0, std::memory_order_relaxed),
             c.verify_skip_checked.exchange(0, std::memory_order_relaxed),
             c.verify_skip_mismatch.load(std::memory_order_relaxed),
             c.verify_direct_checked.exchange(0, std::memory_order_relaxed),
             c.verify_direct_mismatch.load(std::memory_order_relaxed));
}

} // namespace VideoCommon::Nx2CbPath
