// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>

#include "common/common_types.h"
#include "common/fritchii_page_flags.h"

/// FRITCHII_BIND_SKIP (live, default 0): "S4-lite". A graphics uniform bind
/// (BufferCache::BindHostGraphicsUniformBuffer) skips the guest read, the compare, the copy and the
/// native bank check when the bound range's bytes provably did not change since the same (stage,
/// slot) bound the same range last. The proof is a write generation of the range's pages that
/// every writer of those bytes and every change of the tracker's state for them raises:
///  - content: Common::PriceGenSum (common/fritchii_page_flags.h), raised by every mark of the two
///    flags the native bank check takes: E's stream writes (constant buffer words, inline
///    uploads, the engines' semaphore and report words through MemoryManager::Write and the write
///    ring's Note), the buffer cache's own writes (WriteMemory, downloads landing, inline
///    memory), R's results landing (the write ring's Note), emulator writes on other threads
///    (Note), and guest stores on pages the tracker holds clean (they trap: the collected flag
///    before the store and again after the bytes landed, the after-store point). Those marks
///    that only the native bank check needed are behind the bank marks gate (BankMarks), which
///    this switch holds open while it is on.
///  - tracker: g_tracker_gens below, raised by the tracker's word managers wherever a page's CPU,
///    Untracked, cached-write or GPU bit changes (MarkRegionAsCpuModified, a re-protect at an
///    upload, MarkRegionAsGpuModified, a download's clear, cached writes flushed), and
///    g_tracker_wide whenever a manager is made (all its pages start CPU-modified).
/// The code that uses them is BufferCache's BindSkip* (buffer_cache.h).
///
/// Off: the tracker's hooks are one relaxed load per call; the bind path one member test.
namespace Common::BindSkip {

/// True while the switch is non-zero (set by the buffer cache once a draw, BindSkipPoll).
inline std::atomic<bool> g_on{false};

[[nodiscard]] inline bool On() noexcept {
    return g_on.load(std::memory_order_relaxed);
}

/// Tracker state generations: a hash of the device page to 65,536 buckets (the price gens' hash),
/// raised while on. A collision only makes a range look changed.
inline constexpr size_t TrackerBuckets = 65536;
inline std::array<std::atomic<u32>, TrackerBuckets> g_tracker_gens{};
/// Raised when a tracker manager is made (its pages start CPU-modified without a state change
/// the hooks see): every sum includes it.
inline std::atomic<u32> g_tracker_wide{0};

[[nodiscard]] inline size_t TrackerBucket(u64 page) noexcept {
    return static_cast<size_t>((page * 0x9E3779B97F4A7C15ULL) >> 48);
}

/// A word manager's word: the pages of `mask` (bit n = the page at word_address + n * 4 KiB)
/// changed state. The caller tests On() first.
inline void NoteTrackerWord(u64 word_address, u64 mask) noexcept {
    const u64 base = word_address >> 12;
    while (mask != 0) {
        const u64 bit = static_cast<u64>(std::countr_zero(mask));
        mask &= mask - 1;
        g_tracker_gens[TrackerBucket(base + bit)].fetch_add(1, std::memory_order_relaxed);
    }
}

inline void NoteTrackerWide() noexcept {
    if (On()) [[unlikely]] {
        g_tracker_wide.fetch_add(1, std::memory_order_relaxed);
    }
}

/// The range's write generation: content and tracker sums together. Each part only grows, so an
/// equal value over the same range means neither moved. ~0: no proof (over 64 pages).
[[nodiscard]] inline u64 RangeGen(u64 address, u64 size) noexcept {
    if (size == 0) {
        return ~u64{0};
    }
    const u64 content = PriceGenSum(address, size);
    if (content == ~u64{0}) {
        return ~u64{0};
    }
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    u64 tracker = u64{g_tracker_wide.load(std::memory_order_relaxed)} << 40;
    for (u64 page = first; page <= last; ++page) {
        tracker += g_tracker_gens[TrackerBucket(page)].load(std::memory_order_relaxed);
    }
    // The generations are read before the bytes they vouch for.
    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 gen = content + tracker;
    return gen == ~u64{0} ? gen - 1 : gen;
}

} // namespace Common::BindSkip
