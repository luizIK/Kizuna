// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// FRITCHII_ASYNC_COMPUTE_SKIP (renderer_vulkan/vk_async_compute_skip.h): every write to guest
// memory the caches see (a GPU write mark, an image modification, a guest CPU write) ends the
// "last written by a skipped dispatch" state of the ranges it covers, so the must-be-0 count of
// guest reads of skipped output counts only reads whose last writer really was skipped. Off (no
// skipped range held): two relaxed loads per write.
namespace VideoCommon::ComputeSkipHook {

/// Bounds of the ranges held, [g_lo, g_hi); empty when g_lo >= g_hi.
inline std::atomic<u64> g_lo{~u64{0}};
inline std::atomic<u64> g_hi{0};
/// Set by the rasterizer's AsyncComputeSkip while it lives.
inline std::atomic<void (*)(u64, u64)> g_slow{nullptr};

inline void NoteWrite(u64 addr, u64 size) {
    if (size == 0 || addr >= g_hi.load(std::memory_order_relaxed) ||
        addr + size <= g_lo.load(std::memory_order_relaxed)) {
        return;
    }
    if (const auto slow = g_slow.load(std::memory_order_acquire)) {
        slow(addr, size);
    }
}

} // namespace VideoCommon::ComputeSkipHook
