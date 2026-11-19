// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/fritchii_s4_count.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2_debug_control.h"

/// FRITCHII_S4_COUNT (live, default 0): the S4 counter of work\fable_b2_answers.md 3.3, dry.
///
/// S4 would serve the fill's uniform (constant buffer) binds from a mirror of the pages they use,
/// brought up to date once per list entry (a GP entry the front parses), instead of reading and
/// comparing guest memory at every bind. It pays if the mirror's upkeep is under today's bytes,
/// and it is exact only if a bound range's bytes never change inside a list entry except by the
/// stream's own uploads (which a mirror applies in order). This counts both, on E, inside the
/// fill (BufferCache::BindHostGraphicsUniformBuffer), and logs two lines every 10 s, "S4 count
/// (FRITCHII_S4_COUNT=N) 10 s:". Nothing it does changes what is bound, drawn or written.
///
/// 1 = the binds and their bytes by path, distinct 4 KiB pages under them (by class), list
///     entries, and binds whose bytes changed inside their entry, by cause.
/// 2 = 1, and a sample: at 1 in 256 list entries the front hashes up to 8 ranges E bound
///     recently (with the queued writes it knows the words of laid over, as a mirror would be
///     made there); E hashes the same range at its bind in that entry and compares.
///
/// Off: one relaxed load and a branch per uniform bind, per list entry on the front, per noted
/// write; the switch is read once a draw (Poll).
namespace VideoCommon {
class GpuSplit;
}

namespace VideoCommon::S4Count {

/// E, once a draw (BufferCache::SetUniformBuffersState): Common::S4Count::g_on and g_mode follow
/// the live switch.
void SetMode(int mode) noexcept;
inline void Poll() noexcept {
    const int mode =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_s4_count, "FRITCHII_S4_COUNT");
    if (mode != Common::S4Count::g_mode.load(std::memory_order_relaxed)) [[unlikely]] {
        SetMode(mode);
    }
}

/// One uniform bind of the fill, before it reads anything (the bank check, the stream copy or the
/// cached path's synchronize).
struct BindFacts {
    u64 device_addr{};
    u32 size{};
    u32 stage{};
    u32 slot{};
    bool stream{}; ///< the stream path (copied and compared), else the cached path
    bool bank{};   ///< a native 64 KiB bank: the bank check compares its pages too
    /// The tracker holds a page of the range CPU-modified (guest stores there do not trap).
    bool cpu_modified{};
    /// The range holds GPU-written data (the cached path binds the host buffer).
    bool gpu_modified{};
};

/// Whether the tracker holds one 4 KiB device page CPU-modified (or GPU-modified).
using PageTest = bool (*)(void* context, u64 page);

/// E (or R in a lent call, E parked): counts one bind. `memory` is read (hashes only).
void Bind(Tegra::MaxwellDeviceMemoryManager& memory, const BindFacts& facts, PageTest cpu_page,
          PageTest gpu_page, void* context);

/// The front, at the start of each list entry it parses (DmaPusher::Step, not a resumed one):
/// its ring position, and with value 2 one sample in 256. Also while FRITCHII_BIND_SKIP wants
/// the marks (Common::S4Count::EntryMarks).
void FrontEntry(GpuSplit& split, Tegra::MaxwellDeviceMemoryManager& memory);

/// FRITCHII_BIND_SKIP: the front marks list entries while `want` (or while the counter is on).
void SetBindSkipEntries(bool want) noexcept;

/// FRITCHII_BIND_SKIP, at a uniform bind: the index of the list entry E's ring position is in
/// (grows by one per entry E crosses). 0 when not on E (R in a lent call, the memory stage's
/// thread) or without the split: unknown, no in-entry proof then.
[[nodiscard]] u64 EngineEntryIndex() noexcept;

} // namespace VideoCommon::S4Count
