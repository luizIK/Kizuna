// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/spin_recursive_mutex.h"

namespace VideoCommon {

/// EXP86 (fable52_answer_gpuwritten 2.1): the sequence counter of a cache's lock-free mirrors
/// (the buffer cache's page mirrors; EXP100, the texture cache's count of GPU-modified images).
/// Odd from the first mirror change in a hold of the cache's mutex until that hold releases the
/// mutex (SpinRecursiveMutex's release hook). A reader that sees the same even value before and
/// after its looks read the mirrors as they stood between two holds.
/// All operations are seq_cst: the counter's changes, every bit change and every reader load are
/// in one total order, so a reader's bit load that returns a value a hold wrote comes after that
/// hold's opening increment, and the reader's second load of the counter sees it odd or moved.
class MirrorSeq {
public:
    explicit MirrorSeq(const Common::SpinRecursiveMutex& mutex_,
                       const char* name_ = "GPU-written page mirror (EXP86)")
        : mutex{mutex_}, name{name_} {}

    /// Before a mirror bit changes. Only the mutex's holder may change bits: a change by any
    /// other thread (a site that does not hold the lock) breaks the mirror for good, so every
    /// later read takes the locked path, and the broken flag is stored before the bit changes.
    void BeforeChange() {
        if (!mutex.IsHeldByCurrentThread()) [[unlikely]] {
            Break();
            return;
        }
        if (!open) {
            seq.fetch_add(1);
            open = true;
        }
    }

    /// The release hook, run by the holder while it still holds the mutex.
    void OnRelease() {
        if (open) {
            open = false;
            seq.fetch_add(1);
        }
    }

    static void ReleaseHook(void* context) {
        static_cast<MirrorSeq*>(context)->OnRelease();
    }

    /// An address the mirror can't hold, or a change without the lock: never answer again.
    void Break() {
        if (!broken.exchange(true)) {
            LOG_ERROR(HW_GPU,
                      "{} broken: a change out of range or without its cache's lock; reads take "
                      "the locked path",
                      name);
        }
    }

    [[nodiscard]] u64 Load() const {
        return seq.load();
    }
    [[nodiscard]] bool Broken() const {
        return broken.load();
    }

private:
    const Common::SpinRecursiveMutex& mutex;
    const char* name;
    std::atomic<u64> seq{0};
    std::atomic<bool> broken{false};
    bool open = false; ///< Touched only by the mutex's holder
};

} // namespace VideoCommon
