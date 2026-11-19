// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>

#include "common/common_types.h"

#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"

namespace Dynarmic::Backend::X64 {

class BlockOfCode;

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)  // Structure was padded due to alignment specifier
#endif

struct A64JitState {
    using ProgramCounterType = u64;

    A64JitState() { ResetRSB(); }

    // Exclusive state stuff
    static constexpr u64 RESERVATION_GRANULE_MASK = 0xFFFF'FFFF'FFFF'FFF0ull;
    // Return stack buffer
    static constexpr size_t RSB_SIZE = 8;  // MUST be a power of 2.
    static constexpr size_t RSB_PTR_MASK = RSB_SIZE - 1;

    std::array<u64, 31> reg{};
    u64 sp = 0;
    u64 pc = 0;
    u32 cpsr_nzcv = 0;
    u32 fpsr_exc = 0;
    u32 fpsr_qc = 0;
    u32 fpcr = 0;

    alignas(16) std::array<u64, 64> vec{};  // Extension registers.

    // For internal use (See: BlockOfCode::RunCode)
    u32 guest_MXCSR = 0x00001f80;
    u32 asimd_MXCSR = 0x00009fc0;
    volatile u32 halt_reason = 0;

    // Exclusive state
    u8 exclusive_state = 0;
    u32 rsb_ptr = 0;
    std::array<u64, RSB_SIZE> rsb_location_descriptors;
    std::array<u64, RSB_SIZE> rsb_codeptrs;
    /// Fritchii (FRITCHII_JIT_VERIFY): pinned registers found different from this block's copy;
    /// must stay 0. Written by generated code only.
    u64 pin_mismatches = 0;
    /// Fritchii (FRITCHII_JIT_PIN_STATS): guest register reads (0..31, 31 = SP), then writes.
    std::array<u64, 64> pin_stats{};
    /// Fritchii (FRITCHII_JIT_VERIFY=2): while generated code runs, a pinned register's own slot
    /// (reg[i] / sp) holds PinPoison(i) and this shadow holds the write-through copy (one per pin,
    /// in PinConfig order). The counters below are written by generated code only; all must be 0.
    static constexpr size_t max_pins = 8;
    std::array<u64, max_pins> pin_shadow{};
    u64 pin_reads_checked = 0;    ///< pinned reads compared with their shadow (proof the check runs)
    u64 pin_read_mismatches = 0;  ///< a pinned read whose host register differed from the shadow
    u64 pin_slot_written = 0;     ///< a checkpoint found a pinned slot not holding the poison
    u64 pin_store_missed = 0;     ///< the reload after a call out found the poison (no store before it)
    /// The value a pinned register's slot holds while FRITCHII_JIT_VERIFY=2 generated code runs
    /// (index 0..30, 31 = SP): high half 0xBAD0BAD0, low half the register number.
    static constexpr u64 PinPoison(size_t index) {
        return 0xBAD0BAD000000000ull | u64(index);
    }
    /// Fritchii (FRITCHII_JIT_STATE_REFS): this core's own addresses (callbacks object, JIT,
    /// config, TPIDR registers, exclusive monitor slots, dispatch table, indirect slots and
    /// counters), read by generated code through the state register instead of being written
    /// into it as immediates. Indexed by BlockOfCode's CoreRef; set once by the JIT and kept
    /// across Reset. Last in the struct, so no other field moves. Two spare.
    static constexpr size_t core_ref_count = 13;
    std::array<u64, core_ref_count> core_refs{};

    u32 GetPstate() const {
        return NZCV::FromX64(cpsr_nzcv);
    }

    void SetPstate(u32 new_pstate) {
        cpsr_nzcv = NZCV::ToX64(new_pstate);
    }

    void ResetRSB() {
        rsb_location_descriptors.fill(0xFFFFFFFFFFFFFFFFull);
        rsb_codeptrs.fill(0);
    }

    u32 GetFpcr() const;
    u32 GetFpsr() const;
    void SetFpcr(u32 value);
    void SetFpsr(u32 value);

    u64 GetUniqueHash() const noexcept {
        const u64 fpcr_u64 = u64(fpcr & A64::LocationDescriptor::fpcr_mask) << A64::LocationDescriptor::fpcr_shift;
        const u64 pc_u64 = pc & A64::LocationDescriptor::pc_mask;
        return pc_u64 | fpcr_u64;
    }
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif

using CodePtr = const void*;

}  // namespace Dynarmic::Backend::X64
