// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"
#include "common/fritchii_write_ring.h"
#include "video_core/host1x/gpu_device_memory_manager.h"

namespace Tegra {
class MemoryManager;
}

namespace VideoCommon::MemReadLog {

enum class Kind : u8 {
    Uniform,
    Upload,
    Bound,
    StorageAddress,
    StorageSize,
    TableAddress,
    ComputePointer,
    InlinePageWrite,
    HandleProgramWord,
    HandleProgramStorageBase,
    HandleProgramStorageWord,
    Count,
};

inline std::atomic<bool> g_on{false};

void NoteBytes(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address, u64 size);
void NoteStaged(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address,
                const u8* staged, u64 size);
void NoteValue(Kind kind, Tegra::MemoryManager* memory, u64 address, u64 value, u32 size);
void NoteRange(Kind kind, u64 address, u64 size);

inline void Bytes(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address, u64 size) {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        NoteBytes(kind, memory, address, size);
    }
}

inline void Staged(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address,
                   const u8* staged, u64 size) {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        NoteStaged(kind, memory, address, staged, size);
    }
}

inline void Value(Kind kind, Tegra::MemoryManager* memory, u64 address, u64 value,
                  u32 size = 8) {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        NoteValue(kind, memory, address, value, size);
    }
}

inline void Range(Kind kind, u64 address, u64 size) {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        NoteRange(kind, address, size);
    }
}

/// A held packet exists (SEAM 2/3): the texture cache's guest writes are tested against its fill.
inline std::atomic<bool> g_held{false};

void NoteGuestWrite(Tegra::MemoryManager& memory, u64 gpu_addr, u64 size);

inline void GuestWrite(Tegra::MemoryManager& memory, u64 gpu_addr, u64 size) {
    if (g_held.load(std::memory_order_relaxed)) [[unlikely]] {
        NoteGuestWrite(memory, gpu_addr, size);
    }
}

} // namespace VideoCommon::MemReadLog
