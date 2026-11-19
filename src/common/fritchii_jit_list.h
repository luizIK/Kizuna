// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// FRITCHII_JIT_BLOCK_LIST (boot, default 0): a per-title list of the game code blocks the CPU
// recompiler compiled, kept between launches, so the next launch compiles them on idle cores
// before the game reaches them (EDEN_NX2_JIT_WARM's queues, its budget). Port of the idea of
// zortos #1225 (ne1i). In OoT, the first time a code path runs (a save load, an enemy type's first
// kill) compiles hundreds to thousands of blocks in one frame: a 721 ms frame after a save load,
// then ~20 frames of 40-450 ms.
//
// The file: <cache dir>/jit/<program id>-<build id>.blocks. A 64-byte header naming the image,
// then append-only chunks (count, checksum, entries). An entry is a block's location key with its
// pc taken relative to the code image (the image moves every launch). A chunk cut short or with a
// wrong checksum ends the list; the file is cut back to the good part before the next write.
//
// One file per code layout: <program id>-<build id>-<image size>.blocks (2026-11-17, helper DD).
// The single per-title file was rewritten from zero whenever the image size changed: a mod that
// adds a module (the cheat menu's subsdk8, +0x2434000) alternated with plain launches in the test
// root and wiped the list each time, so the owner's fight launch (killcap-v8owner, 02:02) warmed a
// list that had never seen a kill and compiled 77,816 blocks while playing (548 ms in hitch
// frames). An old <program id>-<build id>.blocks whose header matches is renamed to the new name.
//
// FRITCHII_JIT_LIST_SEED=<dir> (boot, default off; 1 = <exe dir>\jit on Windows): also warms the
// entries of <dir>\<same file name> (a list shipped with the build, merged from test sessions with
// work\jit_list_merge.py) that the user's own list lacks. Not written into the user's file.

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Common::FritchiiJitList {

/// FRITCHII_JIT_BLOCK_LIST (boot): 0 = off, 1 = record and warm up from the list, 2 = record only
/// (builds the list, compiles nothing ahead: the A side of an A/B).
[[nodiscard]] int Mode();

/// The application's code image is known (before its cores start): loads the list and starts
/// the writer. Does nothing when Mode() is 0.
void Begin(u64 program_id, std::span<const u8> build_id, u64 image_base, u64 image_size,
           const std::filesystem::path& cache_dir);

/// The application stops: writes what is left and stops the writer.
void End();

/// What Note knew of a block the game waited for.
enum class Seen : u8 {
    Off,      ///< not recording, single-stepping, or outside the code image
    New,      ///< not in the list: recorded now
    Listed,   ///< in the list (an earlier launch or the seed), the warm-up had not compiled it
    Recorded, ///< recorded earlier this launch
};

/// From the recompiler, any core: execution compiled block `key` (an A64 location key) while the
/// game waited; `compiled_earlier` = some core compiled it before in this launch (dropped since,
/// or another core was compiling it). Counts it for the writer's line ("compiles while the game
/// ran"). Cheap when the list is off; one hash lookup under the list's lock when on.
Seen Note(u64 key, bool compiled_earlier);

/// For `core`'s warm-up while it is idle: up to `max` listed keys it has not been handed yet, in
/// list order (appended to `out`). Returns how many.
std::size_t TakePreload(u32 core, std::vector<u64>& out, std::size_t max);

/// Whether a list is loaded for warm-up (Mode() 1 and Begin found entries).
[[nodiscard]] bool Preloading();

} // namespace Common::FritchiiJitList
