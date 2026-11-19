// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"

namespace VideoCommon {

/// The memory stage's copy of one Maxwell3D's state (piece 5b, memstage_shadow.cpp): what the
/// draw manager's code reads and writes, under the names it uses on Maxwell3D, so the same
/// DrawManager code runs on it (draw_manager.cpp instantiates both). Draws reach StageDraw instead
/// of the rasterizer; clears and texture draws reach nothing (they do not touch the buffer cache).
struct MemStageShadowEngine {
    using Maxwell3D = Tegra::Engines::Maxwell3D;

    /// What DrawManager calls on the rasterizer besides draws: nothing, for the stage.
    struct NoRasterizer {
        void Clear(u32) noexcept {}
        void DrawTexture() noexcept {}
    };

    struct Dirty {
        /// The copy's dirty flags. Flags 15-48 are the buffer cache's own state (piece 5c).
        Maxwell3D::DirtyState::Flags flags;
    };

    bool ShouldExecute() const noexcept {
        return execute_on;
    }

    /// The draw manager issued a draw (ProcessDraw, ProcessDrawIndirect), before the
    /// render-enable test. Defined in memstage_shadow.cpp.
    void StageDraw(bool indexed, u32 instance_count, bool indirect);
    /// The draw manager raised `flag` itself (not through a register store).
    void NoteRaised(u32 flag);
    /// The draw manager changed draw state the fill reads after a draw.
    void NoteChanged();

    Maxwell3D::Regs regs{};
    /// shadow_state of Maxwell3D: the shadow RAM and its control register.
    Maxwell3D::Regs shadow_state{};
    Dirty dirty{};
    Maxwell3D::DrawManager draw_manager{};
    /// The constant buffer bindings (ProcessCBBind).
    Maxwell3D::State state{};
    Maxwell3D::RootTables root_tables{};
    bool execute_on{true};
    NoRasterizer* rasterizer{&no_rasterizer};
    /// The engine this is a copy of.
    Maxwell3D* back{};

    static inline NoRasterizer no_rasterizer{};
};

} // namespace VideoCommon
