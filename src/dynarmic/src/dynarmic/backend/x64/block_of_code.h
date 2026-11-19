// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include "common/common_types.h"
#include "common/x64/xbyak.h"
#include "dynarmic/mcl/bit.hpp"
#include "dynarmic/backend/x64/xbyak.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/callback.h"
#include "dynarmic/backend/x64/constant_pool.h"
#include "dynarmic/backend/x64/host_feature.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/cond.h"

namespace Dynarmic::Backend::X64 {

using CodePtr = const void*;

struct RunCodeCallbacks {
    ArgCallback LookupBlock;
    ArgCallback AddTicks;
    ArgCallback GetTicksRemaining;
    bool enable_cycle_counting;
};

/// Fritchii (FRITCHII_JIT_PIN): guest registers that live in host registers while generated code
/// runs. The state block's copy is current only outside generated code and around its calls out.
struct PinnedRegister {
    Xbyak::Reg64 host;
    u32 state_offset;  ///< of the guest register's qword in the state block (ABI_JIT_PTR)
    u32 shadow_offset = 0;  ///< FRITCHII_JIT_VERIFY=2: of its write-through copy
    u64 poison = 0;         ///< FRITCHII_JIT_VERIFY=2: what its own slot holds while pinned
};
struct PinConfig {
    std::vector<PinnedRegister> regs;  ///< empty = no pinning, old code
    /// FRITCHII_JIT_VERIFY (1 or 2): guest register writes also go to a copy, and the dispatcher,
    /// the exit and every call out compare each pinned register with it (a mismatch counts at
    /// mismatch_offset, a qword in the state block, and the copy's value wins).
    /// =1: the copy is the guest register's own slot (the state block is always current).
    bool verify = false;
    /// =2 (`shadow`): the copy is a shadow field; the own slot holds `poison` while the pins are
    /// live and gets the real value only from the store before a call out and at the exit, as
    /// without verify. Checkpoints also count a slot that lost its poison (slot_written_offset:
    /// a call out without the reload after it, or a write from outside), and the reload after a
    /// call out counts reading the poison back (store_missed_offset: no store before the call)
    /// and takes the shadow's value.
    bool shadow = false;
    u32 mismatch_offset = 0;
    u32 slot_written_offset = 0;
    u32 store_missed_offset = 0;
};

/// Fritchii (FRITCHII_JIT_STATE_REFS): the per-core addresses generated code needs. Each has a
/// qword in the state block (StateRefConfig::base_offset + 8 * index), so with the switch on a
/// block reads them through ABI_JIT_PTR instead of holding them as immediates, and its bytes no
/// longer depend on the core that compiled it.
enum class CoreRef : u32 {
    Callbacks,          ///< the UserCallbacks object (SVC, exceptions, cache ops, memory slow paths)
    Jit,                ///< the JIT itself (the dispatcher's block lookup)
    Conf,               ///< the emitter's config (exclusive accesses through the monitor in C++)
    Tpidr,              ///< &tpidr_el0
    Tpidrro,            ///< &tpidrro_el0
    MonitorAddress,     ///< this core's exclusive monitor reservation address
    MonitorValue,       ///< this core's exclusive monitor reservation value
    FastDispatchTable,  ///< the fast dispatch table
    IndirectSlots,      ///< FRITCHII_JIT_INDIRECT_CACHE's slot array
    IndirectCounters,   ///< FRITCHII_JIT_INDIRECT_CACHE's counters
    CntpctLast,         ///< FRITCHII_JIT_INLINE_CNTPCT's previous read (the callbacks object's)
    Count,
};
struct StateRefConfig {
    int mode = 0;          ///< 0 = immediates (old code), 1 = loads from the state block, 2 = 1 + scan
    u32 base_offset = 0;   ///< of the per-core qwords in the state block
    std::array<u64, size_t(CoreRef::Count)> values{};  ///< what the state block holds (this core's)
};

class BlockOfCode final : public Xbyak::CodeGenerator {
public:
    BlockOfCode(RunCodeCallbacks cb, JitStateInfo jsi, size_t total_code_size, std::function<void(BlockOfCode&)> rcp, PinConfig pins = {}, StateRefConfig state_refs = {});
    BlockOfCode(const BlockOfCode&) = delete;
    // Fritchii (FRITCHII_JIT_FAR_CODE): Xbyak's limit is the near region's end while near code is
    // emitted; the base class frees the whole space.
    ~BlockOfCode() {
        if (far_code_begin != 0) {
            maxSize_ = code_space_size;
        }
    }

    /// Call when external emitters have finished emitting their preludes.
    void PreludeComplete();

    /// Change permissions to RW. This is required to support systems with W^X enforced.
    void EnableWriting();
    /// Change permissions to RX. This is required to support systems with W^X enforced.
    void DisableWriting();

    /// Clears this block of code and resets code pointer to beginning.
    void ClearCache();
    /// Calculates how much space is remaining to use.
    size_t SpaceRemaining() const;
    /// Ensure at least codesize bytes of code cache memory are committed at the current code_ptr.
    void EnsureMemoryCommitted(size_t codesize);

    /// Runs emulated code from code_ptr.
    HaltReason RunCode(void* jit_state, CodePtr code_ptr) const;
    /// Runs emulated code from code_ptr for a single cycle.
    HaltReason StepCode(void* jit_state, CodePtr code_ptr) const;
    /// Code emitter: Returns to dispatcher
    void ReturnFromRunCode(bool mxcsr_already_exited = false);
    /// Code emitter: Returns to dispatcher, forces return to host
    void ForceReturnFromRunCode(bool mxcsr_already_exited = false);
    /// Code emitter: Makes guest MXCSR the current MXCSR
    void SwitchMxcsrOnEntry();
    /// Code emitter: Makes saved host MXCSR the current MXCSR
    void SwitchMxcsrOnExit();
    /// Code emitter: Enter standard ASIMD MXCSR region
    void EnterStandardASIMD();
    /// Code emitter: Leave standard ASIMD MXCSR region
    void LeaveStandardASIMD();
    /// Code emitter: Updates cycles remaining my calling cb.AddTicks and cb.GetTicksRemaining
    /// @note this clobbers ABI caller-save registers
    void UpdateTicks();
    /// Code emitter: Performs a block lookup based on current state
    /// @note this clobbers ABI caller-save registers
    void LookupBlock();

    /// Code emitter: Load required flags for conditional cond from rax into host rflags
    void LoadRequiredFlagsForCondFromRax(IR::Cond cond);

    /// @brief Code emitter: Calls the function
    template<typename F>
        requires std::is_pointer_v<F> && std::is_function_v<std::remove_pointer_t<F>>
    void CallFunction(F fn) {
        PinsBeforeCall();
        ::Common::X64::CallFarFunction(*this, fn);
        PinsAfterCall();
    }

    /// @brief Code emitter: Calls the lambda. Lambda must not have any captures.
    template<typename Lambda>
    void CallLambda(Lambda l) {
        PinsBeforeCall();
        ::Common::X64::CallFarFunction(*this, Common::FptrCast(l));
        PinsAfterCall();
    }

    /// Fritchii (FRITCHII_JIT_STATE_REFS): 0 = per-core addresses are immediates (old code).
    int StateRefsMode() const { return state_refs.mode; }
    const StateRefConfig& StateRefs() const { return state_refs; }
    /// Set before anything that uses it is emitted; the state block gets the same value.
    void SetCoreRef(CoreRef which, u64 value) { state_refs.values[size_t(which)] = value; }
    u32 CoreRefOffset(CoreRef which) const { return state_refs.base_offset + u32(sizeof(u64) * size_t(which)); }
    /// Code emitter: reg = this core's `value` for `which`. With the switch on (and `value` the
    /// one the state block holds) a load through ABI_JIT_PTR, else the immediate.
    void MovCoreRef(const Xbyak::Reg64& reg, CoreRef which, u64 value) {
        if (state_refs.mode != 0 && value != 0 && state_refs.values[size_t(which)] == value) {
            mov(reg, qword[ABI_JIT_PTR + CoreRefOffset(which)]);
        } else {
            mov(reg, value);
        }
    }
    /// ArgCallback: the state block offset of an object argument (callbacks, JIT, config) when
    /// the switch is on and `arg` is one of them, else 0 (the immediate).
    u32 CoreRefArgOffset(u64 arg) const {
        if (state_refs.mode == 0 || arg == 0) {
            return 0;
        }
        for (const CoreRef which : {CoreRef::Callbacks, CoreRef::Jit, CoreRef::Conf}) {
            if (state_refs.values[size_t(which)] == arg) {
                return CoreRefOffset(which);
            }
        }
        return 0;
    }

    /// Fritchii (FRITCHII_JIT_PIN): the pinned guest registers.
    const PinConfig& Pins() const { return pins; }
    /// Code emitter: while the hook is on, calls out write the pinned registers to the state
    /// block before (verify: compare them with it) and load them back after. On only where
    /// ABI_JIT_PTR is the state and the pins are loaded (block code, the shared handlers, the
    /// dispatcher); off for the exception handler's code and the run_code prologue.
    void SetPinCallHook(bool on) { pin_call_hook = on && !pins.regs.empty(); }
    /// Code emitter: around a raw call out (the hook's two halves; nothing while it is off).
    void PinsBeforeCall();
    void PinsAfterCall();
    /// Code emitter: state block <- pinned registers / pinned registers <- state block.
    void StorePinnedRegisters();
    void LoadPinnedRegisters();
    /// Code emitter (FRITCHII_JIT_VERIFY=2): right after the entries' load, shadows <- pinned
    /// registers and the own slots <- poison. Nothing in the other modes.
    void ShadowPinnedRegisters();
    /// Code emitter (FRITCHII_JIT_VERIFY=2): the reload after a call out (see PinConfig::shadow).
    void ReloadPinnedRegistersVerified();
    /// Code emitter (FRITCHII_JIT_VERIFY): each pinned register that differs from its copy (=1
    /// the state block, =2 the shadow) counts one mismatch and takes the copy's value; =2 also
    /// counts an own slot without the poison and poisons it again. Writes the flags.
    void CheckPinnedRegisters();

    void ZeroExtendFrom(size_t bitsize, Xbyak::Reg64 reg) {
        switch (bitsize) {
        case 8:
            movzx(reg.cvt32(), reg.cvt8());
            return;
        case 16:
            movzx(reg.cvt32(), reg.cvt16());
            return;
        case 32:
            mov(reg.cvt32(), reg.cvt32());
            return;
        case 64:
            return;
        default:
            UNREACHABLE();
        }
    }

    Xbyak::Address Const(const Xbyak::AddressFrame& frame, u64 lower, u64 upper = 0);

    template<size_t esize>
    Xbyak::Address BConst(const Xbyak::AddressFrame& frame, u64 value) {
        return Const(frame, mcl::bit::replicate_element<u64>(esize, value),
                     mcl::bit::replicate_element<u64>(esize, value));
    }

    CodePtr GetCodeBegin() const;
    size_t GetTotalCodeSize() const;

    const void* GetReturnFromRunCodeAddress() const {
        return return_from_run_code[0];
    }

    const void* GetForceReturnFromRunCodeAddress() const {
        return return_from_run_code[FORCE_RETURN];
    }

    void int3() { db(0xCC); }

    /// Allocate memory of `size` bytes from the same block of memory the code is in.
    /// This is useful for objects that need to be placed close to or within code.
    /// The lifetime of this memory is the same as the code around it.
    void* AllocateFromCodeSpace(size_t size);

    void SetCodePtr(CodePtr code_ptr);
    void EnsurePatchLocationSize(CodePtr begin, size_t size);

    /// Fritchii (FRITCHII_JIT_FAR_CODE): the last `far_size` bytes of the code space become a
    /// second region for code that rarely runs (slow paths, link stubs), so the blocks' hot
    /// bytes sit next to each other. Both regions are reset together (ClearCache). Call before
    /// PreludeComplete. Without it nothing changes.
    void EnableFarCode(size_t far_size);
    bool HasFarCode() const { return far_code_begin != 0; }
    /// Offset of the far region in the code space (0 without one).
    size_t FarCodeBegin() const { return far_code_begin; }
    /// Bytes used in the near region (blocks, after the prelude) and in the far one.
    size_t NearCodeUsed() const {
        return (in_far_code ? near_code_offset : size_) -
               size_t(static_cast<const u8*>(code_begin) - getCode());
    }
    size_t FarCodeUsed() const {
        return far_code_begin == 0 ? 0 : (in_far_code ? size_ : far_code_offset) - far_code_begin;
    }
    /// Emission continues in the far region (where it last stopped) / back in the near one.
    void SwitchToFarCode();
    void SwitchToNearCode();
    bool InFarCode() const { return in_far_code; }
    /// Where far code would be emitted next (near code current; nullptr without far code).
    CodePtr GetFarCodeCurr() const;
    /// The smaller of the two regions (the whole space without far code).
    size_t SmallestRegionSize() const;

    static const Xbyak::Reg64 ABI_JIT_PTR;
    // ABI registers
#ifdef _WIN32
    static const Xbyak::Reg64 ABI_RETURN;
    static const Xbyak::Reg64 ABI_PARAM1;
    static const Xbyak::Reg64 ABI_PARAM2;
    static const Xbyak::Reg64 ABI_PARAM3;
    static const Xbyak::Reg64 ABI_PARAM4;
    static const std::array<Xbyak::Reg64, ABI_PARAM_COUNT> ABI_PARAMS;
#else
    static const Xbyak::Reg64 ABI_RETURN;
    static const Xbyak::Reg64 ABI_RETURN2;
    static const Xbyak::Reg64 ABI_PARAM1;
    static const Xbyak::Reg64 ABI_PARAM2;
    static const Xbyak::Reg64 ABI_PARAM3;
    static const Xbyak::Reg64 ABI_PARAM4;
    static const Xbyak::Reg64 ABI_PARAM5;
    static const Xbyak::Reg64 ABI_PARAM6;
    static const std::array<Xbyak::Reg64, ABI_PARAM_COUNT> ABI_PARAMS;
#endif

    JitStateInfo GetJitStateInfo() const { return jsi; }

    bool HasHostFeature(HostFeature feature) const noexcept;

private:
    using RunCodeFuncType = HaltReason (*)(void*, CodePtr);
    static constexpr size_t MXCSR_ALREADY_EXITED = 1 << 0;
    static constexpr size_t FORCE_RETURN = 1 << 1;

#ifdef _WIN32
    // Before constant_pool: its constructor commits code space (EnsureMemoryCommitted reads this).
    size_t committed_size = 0;
    size_t far_committed_size = 0; // Fritchii (FRITCHII_JIT_FAR_CODE)
#endif
    // Fritchii (FRITCHII_JIT_FAR_CODE): offsets in the code space. far_code_begin 0 = no far
    // region. The write position of the region not in use is kept in its *_offset. Before
    // constant_pool too: its constructor reaches EnsureMemoryCommitted, which reads them (they
    // were declared after it and read uninitialized: a boot crash with the switch off as well).
    size_t far_code_begin = 0;
    size_t far_code_offset = 0;
    size_t near_code_offset = 0;
    size_t code_space_size = 0; // the whole space (Xbyak's maxSize_ ends at the near region)
    bool in_far_code = false;
    ConstantPool constant_pool;
    JitStateInfo jsi;
    std::array<const void*, 4> return_from_run_code;
    RunCodeFuncType run_code = nullptr;
    RunCodeFuncType step_code = nullptr;
    RunCodeCallbacks cb;
    CodePtr code_begin = nullptr;
    bool prelude_complete = false;
    PinConfig pins;
    bool pin_call_hook = false;
    StateRefConfig state_refs; // Fritchii (FRITCHII_JIT_STATE_REFS); before GenRunCode runs

    void GenRunCode(std::function<void(BlockOfCode&)> rcp);
};

}  // namespace Dynarmic::Backend::X64
