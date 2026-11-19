// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex> // Kernel headers that use std::mutex get it through this one

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/hle/kernel/k_scoped_lock.h"

namespace Kernel {

class KSpinLock {
public:
    explicit KSpinLock() = default;

    YUZU_NON_COPYABLE(KSpinLock);
    YUZU_NON_MOVEABLE(KSpinLock);

    void Lock();
    void Unlock();
    bool TryLock();

    // EXP98: true when no one holds the lock. The acquire pairs with Unlock's release, so every
    // write of a completed critical section is visible after it reads true.
    bool IsFree() const {
        return m_state.load(std::memory_order_acquire) == 0;
    }

private:
    // 0 free, 1 held, 2 held with (possible) sleepers. The scheduler lock is taken by every
    // core thousands of times a frame and held for microseconds: a contended Lock spins briefly
    // before sleeping, instead of putting the host thread to sleep at once and paying its
    // wake-up latency (std::mutex did).
    std::atomic<u32> m_state{0};
};

// TODO(bunnei): Alias for now, in case we want to implement these accurately in the future.
using KAlignedSpinLock = KSpinLock;
using KNotAlignedSpinLock = KSpinLock;

using KScopedSpinLock = KScopedLock<KSpinLock>;
using KScopedAlignedSpinLock = KScopedLock<KAlignedSpinLock>;
using KScopedNotAlignedSpinLock = KScopedLock<KNotAlignedSpinLock>;

} // namespace Kernel
