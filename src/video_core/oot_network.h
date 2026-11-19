// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <mutex>

#include "common/common_types.h"
#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"

// The Legend of Zelda: Ocarina of Time's anti-aliasing. After tonemapping, the game runs a
// single-frame tensor network on the tonemapped image and an output pass that sharpens the result
// hard. With the "Ocarina of Time anti-aliasing" setting at No sharpening, the network is skipped
// and its output pass is replaced by a copy of the tonemapped image
// (docs/nx2/oot-network-bypass.md).
//
// Shader hashes, native program keys and constant buffer offsets are those of the game's main
// with build id E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348; a game update needs them re-checked.
namespace VideoCommon::OotNetwork {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

// Native program keys of the network's compute passes.
constexpr u64 PrepassProgramKey = 0x8475d671e6967428ULL;
constexpr u64 NetworkProgramKey = 0x42593194bb0b2b32ULL;
constexpr u64 OutputProgramKey = 0xda6df3e54c716df2ULL;

/// Constant buffer 0 offset of the output pass's texture handle for the tonemapped image it
/// samples (the network's input; FRITCHII_OOT_BYPASS_SAMPLED copies from it).
constexpr u32 InputImageHandleOffset = 0xc0;

/// Constant buffer 0 offset of the output pass's texture handle for the B10G11R11 image it writes
/// (the scene image the rest of the frame uses).
constexpr u32 OutputImageHandleOffset = 0xd0;

/// Fragment shaders of the game's tonemap passes; render target 0 of their draws is the
/// network's input.
constexpr bool IsTonemapShader(u64 hash) {
    return hash == 0xbded5ce917183214ULL || // outdoors
           hash == 0x6cb06edf14be1d79ULL || // indoors
           hash == 0x3ac011bd2d772994ULL;   // outdoors at another time of day
}

/// Owned by the rasterizer: the tonemap pass draws on the 3D engine, and the network's passes do
/// not all run on the same compute engine.
struct BypassState {
    std::mutex mutex;
    /// Render target 0 of the latest tonemap draw (address 0 until one is seen), under mutex.
    Tegra::Engines::Maxwell3D::Regs::RenderTargetConfig tonemapped{};
    std::atomic<bool> output_replaced{}; ///< the latest output pass was replaced
    /// FRITCHII_OOT_BYPASS_SAMPLED: the prepass was skipped and no output pass has been replaced
    /// since.
    std::atomic<bool> output_pending{};
    /// FRITCHII_OOT_BYPASS_SAMPLED: the first native program dispatched after the network, 0 until
    /// one is seen.
    std::atomic<u64> after_network{};
};

/// Set when one of the network's passes can't be built on this GPU (handheld PCs on RADV:
/// "Native compute pipeline unavailable"): the output pass would never write the scene, so the
/// picture stays black. No sharpening's copy draws it instead, for the rest of the session
/// (CheezyEmu 97a7e109e4). Never set while the network's passes build.
inline std::atomic<bool> forced_bypass{false};

inline bool BypassEnabled(u64 program_id) {
    return program_id == ProgramId &&
           (Settings::values.oot_anti_aliasing.GetValue() ==
                Settings::OotAntiAliasing::NoSharpening ||
            forced_bypass.load(std::memory_order_relaxed));
}

constexpr bool IsNetworkPass(u64 native_key) {
    return native_key == PrepassProgramKey || native_key == NetworkProgramKey ||
           native_key == OutputProgramKey;
}

} // namespace VideoCommon::OotNetwork
