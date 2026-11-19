// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part c (work\fable_b_one_follower.md 3 and 5): the registry of R's draw player and
// E's 0xe2c note. The player itself is the Vulkan rasterizer's (vk_one_follower_draw.cpp).

#include <atomic>

#include "video_core/nx2_debug_control.h"
#include "video_core/one_follower.h"

namespace VideoCommon::OneFollower {

namespace {
std::atomic<DrawPlayer*> g_player{nullptr};
/// E: 0xe2c notes since the last compare at step 4 (only E touches it).
u32 g_condition_count = 0;
} // Anonymous namespace

void SetDrawPlayer(DrawPlayer* player) noexcept {
    g_player.store(player, std::memory_order_release);
}

DrawPlayer* GetDrawPlayer() noexcept {
    return g_player.load(std::memory_order_acquire);
}

void NoteCondition(bool execute_on) {
    if (!On()) {
        return;
    }
    const StepLevel step = Step();
    if (step < StepLevel::Build) {
        return;
    }
    DrawPlayer* const player = GetDrawPlayer();
    if (player == nullptr) {
        return;
    }
    if (OnRenderThread()) {
        // ProcessQueryCondition runs in a lent call (E parked): R's view is at this place.
        player->Condition(execute_on);
        return;
    }
    // On E. Step 3 compares every condition; step 4 every FRITCHII_ONEF_VERIFY_EVERY-th, so the
    // compare does not add a round trip to each 0xe2c (R's mode is only used by PacketConditional
    // draws, which step 5 sends).
    if (step == StepLevel::Unlend) {
        const int every = VideoCore::Nx2Debug::SwitchValue(
            VideoCore::Nx2Debug::g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY",
            DefaultVerifyEvery);
        if (every <= 0 || ++g_condition_count < static_cast<u32>(every)) {
            return;
        }
        g_condition_count = 0;
    }
    struct Argument {
        DrawPlayer* player;
        bool execute_on;
    } argument{player, execute_on};
    Lend(
        LentOp::Query, LentReason::Verify,
        [](void* raw) {
            const auto* const a = static_cast<Argument*>(raw);
            a->player->Condition(a->execute_on);
        },
        &argument);
}

} // namespace VideoCommon::OneFollower
