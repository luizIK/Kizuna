// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// Switches for images made where a GPU-written image was dropped without a write-back (the
// lost-contents counter, FRITCHII_STALE_ZERO, FRITCHII_LOST_WRITEBACK). Kept out of
// nx2_debug_control.h so they rebuild only the texture cache and the switch lookup.

#pragma once

#include <atomic>

namespace VideoCore::Nx2Debug {

/// FRITCHII_STALE_ZERO (live, default 1): docs/fritchii/switches.md.
inline std::atomic<int> g_stale_zero{-1};
/// FRITCHII_LOST_WRITEBACK (live, default 1; N > 1 = only images up to N KiB):
/// docs/fritchii/switches.md.
inline std::atomic<int> g_lost_writeback{-1};
/// FRITCHII_LOST_KEEP (live, default 1; N > 1 = a budget of N MiB/s, 1 = 32): docs/fritchii/switches.md.
inline std::atomic<int> g_lost_keep{-1};

} // namespace VideoCore::Nx2Debug
