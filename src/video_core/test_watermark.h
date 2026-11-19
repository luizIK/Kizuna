// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "common/math_util.h"

namespace VideoCore::TestWatermark {

/// One block of the mark, top-left origin, in frame pixels.
struct Block {
    u32 x;
    u32 y;
    u32 width;
    u32 height;
};

struct Blocks {
    std::vector<Block> shadow; ///< Drawn first, dark
    std::vector<Block> text;   ///< Drawn over it, light
    u64 version = 0;           ///< New for every rebuild (any thread); 0: never built
};

inline constexpr float ShadowColor[4]{0.05f, 0.05f, 0.05f, 1.0f};
inline constexpr float TextColor[4]{0.85f, 0.85f, 0.85f, 1.0f};

/// FRITCHII_TEST_MARK (boot): 0 = no on-screen mark (the stamp itself stays), 1 (default, also
/// any other value) = one textured draw (Vulkan; OpenGL draws clears), 3 = two clears over all
/// the blocks, as before (Vulkan A/B).
[[nodiscard]] int Mode();

/// Logs once per run how the mark is drawn ("off", "one textured draw", ...). Call it only on a
/// frame that has a mark, so plain builds never log it.
void LogMode(const char* how);

/// Copies that carry a package stamp show it on screen: "FRITCHII V6 TEST" and the copy's id,
/// inside the game picture (`screen`), moving to another spot every minute. Empty for plain
/// builds. `frame_width`/`frame_height` clamp the blocks.
[[nodiscard]] const Blocks& Build(const Common::Rectangle<u32>& screen, u32 frame_width,
                                  u32 frame_height);

} // namespace VideoCore::TestWatermark
