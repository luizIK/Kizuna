// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// EDEN_NX2_RUNAHEAD (nx2_runahead.h): guest CPU writes in flight.
//
// Core::Memory's HandleRasterizerWrite collects a write's dirty mark (GPUDirtyMemoryManager)
// before its caller stores the bytes. The old path never minds: a gather only marks pages
// CPU-modified, and the upload that reads guest memory comes much later. Run-ahead copies the
// bytes right at the gather (a boundary's chunk, the back's own gather). A store still on its way
// then leaves the old bytes in the chunk and no mark behind: the new bytes never reach the host
// buffer until the guest writes there again (a frame of stale bone data on a moving NPC).
//
// While armed, each thread that writes cached guest memory owns a slot: Open (before the mark is
// collected) makes its sequence odd and records the range, Close (after the store) makes it even.
// A gatherer calls ForEachInFlight after its gather and before copying bytes: every write whose
// mark it took either shows up there (its bytes are taken again at the next gather) or was stored
// before it looked (its bytes are in the copy). Both sides fence (seq_cst), so a write that missed
// the gather's exchange of the dirty manager left its mark for the next gather instead.
//
// Verify mode also keeps a short journal per thread (address, size, the epoch it was written in):
// a chunk that differs from guest memory at its stamp says whether its bytes were in flight at
// the snapshot, rewritten after it (and by which thread and core), or changed by no tracked CPU
// write at all (the GPU, an untracked writer).
namespace VideoCommon::WriteTrack {

/// Set while run-ahead is on or verifying (Runahead::CurrentMode): Open and Close do anything.
inline std::atomic<bool> g_armed{false};
/// Verify mode: Open also writes the journal.
inline std::atomic<bool> g_journal{false};

/// Core::Memory, HandleRasterizerWrite, before the mark is collected: the calling thread stores
/// `size` bytes at device `address` next. `core`: the CPU core index it collects for.
void Open(u64 address, u64 size, u32 core);
/// After the store (or right after Open where nothing is stored: cache maintenance).
void Close();
inline void CloseIfArmed() {
    if (g_armed.load(std::memory_order_relaxed)) [[unlikely]] {
        Close();
    }
}

/// A gatherer, right after its gather and before it copies bytes: `callback` for every write
/// opened and not closed (its bytes may not be in guest memory yet).
void ForEachInFlight(void (*callback)(void* argument, u64 address, u64 size, u32 slot, u32 core),
                     void* argument);

/// Verify mode, before a snapshot: a new epoch; journal entries written from now on carry it or a
/// later one.
u64 NewEpoch();

/// What the journal knows about [address, address + size) since `epoch`.
struct Rewrite {
    enum class Kind : u32 {
        Found,     ///< A tracked CPU write since: the first thread found is `slot`, `core`
        Unknown,   ///< None found, but a journal that may have held one wrapped
        None,      ///< No tracked CPU write since
    } kind;
    u32 slot;
    u32 core;
    u32 writers; ///< Threads that wrote the range since
};
[[nodiscard]] Rewrite FindRewrite(u64 address, u64 size, u64 epoch);

/// For the log: threads with a slot, and opens refused because every slot was taken.
[[nodiscard]] u32 SlotCount();
[[nodiscard]] u64 TakeUntrackedOpens();

} // namespace VideoCommon::WriteTrack
