// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/assert.h"
#include "common/logging.h"
#include "core/core.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra::Engines {

KeplerMemory::KeplerMemory(MemoryManager& memory_manager)
    : upload_state{memory_manager, regs.upload}
{}

KeplerMemory::~KeplerMemory() = default;

void KeplerMemory::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    upload_state.BindRasterizer(rasterizer_);

    execution_mask.reset();
    execution_mask[KEPLERMEMORY_REG_INDEX(exec)] = true;
    execution_mask[KEPLERMEMORY_REG_INDEX(data)] = true;
}

void KeplerMemory::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        regs.reg_array[method] = value;
    }
    method_sink.clear();
}

void KeplerMemory::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    ASSERT_MSG(method < Regs::NUM_REGS, "Invalid KeplerMemory register, increase the size of the Regs structure");

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLERMEMORY_REG_INDEX(exec): {
        upload_state.ProcessExec(regs.exec.linear != 0);
        break;
    }
    case KEPLERMEMORY_REG_INDEX(data): {
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    }
}

void KeplerMemory::FrontNoteMethod(u32 method, u32 argument) {
    if (method - FrontFirst >= FrontCount) {
        return;
    }
    front_regs[method - FrontFirst] = argument;
    front_seen |= 1U << (method - FrontFirst);
    if (method == KEPLERMEMORY_REG_INDEX(exec)) {
        // Upload::State::ProcessExec.
        front_exec_size = u64{front_regs[0]} * front_regs[1];
        front_linear = (argument & 1) != 0;
    }
}

bool KeplerMemory::FrontDataRange(GPUVAddr& begin, u64& size) const {
    // line_length_in, line_count, address high and low, exec; pitch for more than one line.
    const u64 lines = (std::max)(front_regs[1], 1U);
    const u32 needed = 0xF | (lines > 1 ? 0x10 : 0) |
                       (1U << (KEPLERMEMORY_REG_INDEX(exec) - FrontFirst));
    if (!front_linear || (front_seen & needed) != needed) {
        return false;
    }
    // ProcessData reads the registers when the data arrives: lines of line_length_in bytes,
    // pitch apart, from the address; a partial transfer writes within the exec's copy_size.
    const u64 line_length = front_regs[0];
    const u64 pitch = front_regs[4];
    begin = (GPUVAddr{front_regs[2]} << 32) | front_regs[3];
    size = (std::max)(pitch * (lines - 1) + line_length, front_exec_size);
    return size != 0;
}

void KeplerMemory::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    switch (method) {
    case KEPLERMEMORY_REG_INDEX(data):
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

} // namespace Tegra::Engines
