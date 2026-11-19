// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// Render target relocation guards kept across sessions (FRITCHII_RELOCATION_MEMORY).
//
// EXP43 moves a render target to the address a cleared target drifted to; the old address keeps
// no data. A shape whose contents are read in a later frame trips a guard and never relocates
// again, but only for the session, so each such shape lost its contents once per session: the
// first read came back cleared to zero (a black flash, e.g. when switching items fast; 1280x720
// colour targets and OoT's square D32 targets in the logs). The guards are written to a file and
// read at boot, and a seed from our own sessions starts a first boot where a warmed one stands.

#pragma once

#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::RelocationMemory {

/// FRITCHII_RELOCATION_MEMORY (boot, default 1; 0 = guards per session only, as before).
[[nodiscard]] bool Enabled();

/// The shapes (RelocationShapeKey) and their guard bits from the seed and earlier sessions.
[[nodiscard]] std::vector<std::pair<u64, u8>> Load();

/// A shape gained guard bits this session: written to the file (a handful per session).
void Note(u64 shape_key, u8 guard);

} // namespace VideoCommon::RelocationMemory
