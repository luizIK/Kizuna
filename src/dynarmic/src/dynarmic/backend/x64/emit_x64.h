// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <boost/container/stable_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "dynarmic/backend/x64/xbyak.h"
#include "dynarmic/mcl/bit.hpp"
#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::IR {
class Block;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic {
enum class OptimizationFlag : u32;
}  // namespace Dynarmic

namespace Dynarmic::Backend::X64 {

class A64EmitX64;
class BlockOfCode;
struct SharedCode;

using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
template<typename T>
using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

template<typename T>
using HalfVectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T> / 2>;

/// Fritchii (FRITCHII_JIT_SHARED_FASTMEM_MARKS, boot): the fastmem accesses that faulted (the
/// do_not_fastmem markers), one set for every JIT on the same fastmem arena (one process), so an
/// access that faulted on one core is compiled without fastmem by the others instead of faulting
/// and recompiling there too (port of the idea of zortos293/Lichii #1861, arm64 there). A marker
/// only sends that access to the page table path, so sharing changes no results. The lock is held
/// only around a lookup while a block is emitted and around an insert in the fault handler; no
/// thread faults while it holds it.
struct SharedFastmemMarks {
    std::shared_mutex mutex;
    ::Common::unordered_set<std::tuple<IR::LocationDescriptor, unsigned>> marks;
    std::atomic<std::uint64_t> used_from_other{0};  ///< compiles that took another core's marker
    std::atomic<std::uint64_t> faults{0};           ///< markers added (first fault per access)
};
/// The set for this fastmem arena (null when the switch is off or there is no fastmem).
std::shared_ptr<SharedFastmemMarks> SharedFastmemMarksFor(const void* fastmem_arena);
/// Counts a compile that used another core's marker; logs at powers of two.
void NoteSharedFastmemMarkUsed(SharedFastmemMarks& shared, unsigned core);

using SharedLabel = Xbyak::Label*;
struct EmitContext {
    EmitContext(RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& shared_labels);
    virtual ~EmitContext();
    virtual FP::FPCR FPCR(bool fpcr_controlled = true) const = 0;
    virtual bool HasOptimization(OptimizationFlag flag) const = 0;

    [[nodiscard]] inline Xbyak::Label* GenSharedLabel() noexcept {
        return &shared_labels.emplace_back();
    }

    std::vector<std::function<void()>> deferred_emits;
    RegAlloc& reg_alloc;
    IR::Block& block;
    boost::container::stable_vector<Xbyak::Label>& shared_labels;

    // Fritchii: FRITCHII_JIT_FLAG_BRANCH (A64 blocks only; A64EmitX64::Emit sets track_host_flags
    // and keeps the rest current, see there). What the host flags hold at the current point of
    // the emitted code, so a block's conditional terminal can branch on them directly.
    bool track_host_flags = false;
    /// The NZCV value (a GetNZCVFromOp) whose lahf + seto snapshot is still the host SF, ZF, CF
    /// and OF; nullptr = none.
    IR::Inst* host_flags_nzcv = nullptr;
    /// The guest NZCV is host_flags_nzcv (an A64SetNZCV of it ran since the flags got it).
    bool guest_nzcv_in_host_flags = false;
    /// Host ZF = (check bit == 0): A64SetCheckBit tested the byte it stored.
    bool check_bit_in_host_zf = false;
    /// Set by an emitter whose code left the host flags as recorded above.
    bool host_flags_set_here = false;

    void HostFlagsHoldNZCV(IR::Inst* nzcv) noexcept {
        host_flags_nzcv = nzcv;
        guest_nzcv_in_host_flags = false;
        check_bit_in_host_zf = false;
        host_flags_set_here = true;
    }
    void HostFlagsHoldCheckBit() noexcept {
        host_flags_nzcv = nullptr;
        guest_nzcv_in_host_flags = false;
        check_bit_in_host_zf = true;
        host_flags_set_here = true;
    }
    void HostFlagsUnknown() noexcept {
        host_flags_nzcv = nullptr;
        guest_nzcv_in_host_flags = false;
        check_bit_in_host_zf = false;
    }
};

class EmitX64 {
public:
    struct BlockDescriptor {
        CodePtr entrypoint;  // Entrypoint of emitted code
        size_t size;         // Length in bytes of emitted code
    };
    static_assert(sizeof(BlockDescriptor) == 16);

    explicit EmitX64(BlockOfCode& code);
    virtual ~EmitX64();

    /// Looks up an emitted host block in the cache.
    std::optional<BlockDescriptor> GetBasicBlock(IR::LocationDescriptor descriptor) const;

    /// Empties the entire cache.
    virtual void ClearCache();

    /// Invalidates a selection of basic blocks.
    void InvalidateBasicBlocks(const ::Common::unordered_set<IR::LocationDescriptor>& locations);

//protected:
    // Microinstruction emitters
#define OPCODE(name, type, ...) void Emit##name(EmitContext& ctx, IR::Inst* inst);
#define A32OPC(...)
#define A64OPC(...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC
    void EmitInvalid(EmitContext& ctx, IR::Inst* inst);

    // Helpers
    virtual std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const = 0;
    void EmitAddCycles(size_t cycles);
    Xbyak::Label EmitCond(IR::Cond cond);
    /// Fritchii (FRITCHII_JIT_FLAG_BRANCH): EmitCond's jump without its reload, for when the host
    /// SF, ZF, CF and OF are the ones whose lahf + seto snapshot is the guest NZCV.
    Xbyak::Label EmitCondFromHostFlags(IR::Cond cond);
    /// The jump to `pass` for `cond`, the flags in EmitCond's reloaded form.
    void EmitCondJump(IR::Cond cond, Xbyak::Label& pass);
    BlockDescriptor RegisterBlock(const IR::LocationDescriptor& location_descriptor, CodePtr entrypoint, size_t size);
    void PushRSBHelper(Xbyak::Reg64 loc_desc_reg, Xbyak::Reg64 index_reg, IR::LocationDescriptor target);

#ifndef NDEBUG
    void EmitVerboseDebuggingOutput(RegAlloc& reg_alloc);
#endif
    virtual bool EmitLeafTerminal(IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept = 0;
    virtual bool EmitTerminal(IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept = 0;

    // Patching
    enum class PatchKind : u32 { Jg, Jz, Jmp, MovRcx };
    /// Places in emitted code that jump to a block (or load its entrypoint), patched when it is
    /// compiled or invalidated: each the place's offset in the code space times four plus its
    /// PatchKind, 0 when unused. Most blocks are linked from one or two places.
    struct PatchInformation {
        std::array<u32, 2> first{};
        std::unique_ptr<std::vector<u32>> rest;
    };
    /// Records that the code about to be emitted at the current position is a patch of `kind`.
    void AddPatchLocation(const IR::LocationDescriptor& target_desc, PatchKind kind);
    void Patch(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    virtual void Unpatch(const IR::LocationDescriptor& target_desc);
    virtual void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) = 0;
    /// Fritchii (FRITCHII_JIT_SHARED): where an unlinked compact jump goes (the A64 link stub).
    virtual CodePtr UnlinkedJumpTarget(const IR::LocationDescriptor& target_desc) { return nullptr; }
    /// Fritchii (FRITCHII_JIT_SHARED): pads with nops so the patched field of the patch site about
    /// to be emitted (`field_offset` bytes into it, `field_size` bytes long) is aligned, so one
    /// atomic store rewrites it while other cores may run that code. Nothing per core.
    void AlignPatchField(size_t field_offset, size_t field_size);

    // State
    BlockOfCode& code;
    ExceptionHandler exception_handler;
    ::Common::unordered_map<IR::LocationDescriptor, BlockDescriptor> block_descriptors;
    ::Common::unordered_map<IR::LocationDescriptor, PatchInformation> patch_information;
    /// Fritchii (FRITCHII_JIT_SHARED_FASTMEM_MARKS): set by the A64 emitter only; null = per-core.
    std::shared_ptr<SharedFastmemMarks> shared_fastmem_marks;
    /// Fritchii (FRITCHII_JIT_SHARED): set when this emitter's cache is run by every guest core of
    /// the process (A64 only); null = one cache per core, the old code. With it, blocks go into its
    /// lock-free map, removals into its log, and patches are single aligned atomic stores.
    SharedCode* shared_code = nullptr;

    // We need materialized protected members
    friend class A64EmitX64;
};

}  // namespace Dynarmic::Backend::X64
