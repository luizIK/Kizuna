// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <boost/container/static_vector.hpp>

#include "dynarmic/backend/block_range_information.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::Backend::X64 {

struct A64EmitContext final : public EmitContext {
    A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& shared_labels);

    A64::LocationDescriptor Location() const;
    bool IsSingleStep() const;
    FP::FPCR FPCR(bool fpcr_controlled = true) const override;

    bool HasOptimization(OptimizationFlag flag) const override {
        return conf.HasOptimization(flag);
    }

    const A64::UserConfig& conf;
};

class A64EmitX64 final : public EmitX64 {
public:
    A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface);
    ~A64EmitX64() override;

    /// Emit host machine code for a basic block with intermediate representation `block`.
    /// @note block is modified.
    /// Fritchii: `chained` (FRITCHII_JIT_CHAIN=1) emits it right at the current position,
    /// unaligned, as the block before it links to it there.
    BlockDescriptor Emit(IR::Block& block, bool chained = false) noexcept;

    void ClearCache() override;

    void InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges);

//protected:
    struct FastDispatchEntry {
        u64 location_descriptor = 0xFFFF'FFFF'FFFF'FFFFull;
        const void* code_ptr = nullptr;
    };
    static_assert(sizeof(FastDispatchEntry) == 0x10);
    // Fritchii: the table's size is chosen once per JIT (FRITCHII_JIT_DISPATCH_BITS, entries =
    // 2^bits, 12..20, default 20 = 0x100000 entries). The mask is a byte offset into the table,
    // so its low 4 bits are 0 (16-byte entries): default 0xFFFFF0.
    static constexpr unsigned fast_dispatch_table_default_bits = 20;
    static constexpr unsigned fast_dispatch_table_min_bits = 12;

    void ClearFastDispatchTable();
    void GenMemory128Accessors();
    void GenFastmemFallbacks();
    void GenCntpctFallback();
    void GenTerminalHandlers();

    // Microinstruction emitters
    void EmitPushRSB(EmitContext& ctx, IR::Inst* inst);
#define OPCODE(...)
#define A32OPC(...)
#define A64OPC(name, type, ...) void EmitA64##name(A64EmitContext& ctx, IR::Inst* inst);
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const override;

    // Fastmem information
    using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;
    /// Where a fastmem access that faults goes: code addresses are offsets in the code space.
    struct FastmemPatchInfo {
        u32 rip;
        u32 resume_rip;
        u32 callback;
        u32 inst_name : 31;
        u32 recompile : 1;
        IR::LocationDescriptor location;
    };
    std::optional<DoNotFastmemMarker> ShouldFastmem(A64EmitContext& ctx, IR::Inst* inst) const;
    void AddFastmemPatch(CodePtr location, CodePtr resume_location, void (*callback)(), const DoNotFastmemMarker& marker, bool recompile);
    FakeCall FastmemCallback(u64 rip);

    // Memory access helpers
    void EmitCheckMemoryAbort(A64EmitContext& ctx, IR::Inst* inst, Xbyak::Label* end = nullptr);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryRead(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitMemoryWrite(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemory(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveReadMemoryInline(A64EmitContext& ctx, IR::Inst* inst);
    template<std::size_t bitsize, auto callback>
    void EmitExclusiveWriteMemoryInline(A64EmitContext& ctx, IR::Inst* inst);

    // Terminal instruction emitters
    bool EmitLeafTerminal(IR::Term::LeafTerminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept override;
    bool EmitTerminal(IR::Term::Terminal const& terminal, IR::LocationDescriptor initial_location, bool is_single_step) noexcept override;

    // Patching
    void Unpatch(const IR::LocationDescriptor& target_desc) override;
    void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) override;
    void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) override;

//data
    const A64::UserConfig conf;
    RegAlloc reg_alloc; //reusable reg alloc
    BlockRangeInformation<u64> block_ranges;
    const size_t fast_dispatch_table_size;
    const u32 fast_dispatch_table_mask;
    std::unique_ptr<FastDispatchEntry[]> fast_dispatch_table;
    std::vector<FastmemPatchInfo> fastmem_patch_info;  // sorted by rip
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> read_fallbacks;
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> write_fallbacks;
    ::Common::unordered_map<std::tuple<bool, size_t, int, int>, void (*)()> exclusive_write_fallbacks;
    ::Common::unordered_set<DoNotFastmemMarker> do_not_fastmem;
    boost::container::stable_vector<Xbyak::Label> shared_labels;
    const void* terminal_handler_pop_rsb_hint = nullptr;
    /// FRITCHII_JIT_INLINE_CNTPCT: calls GetCNTPCT, result in rdx, every other register kept.
    const void* cntpct_fallback = nullptr;
    const void* terminal_handler_fast_dispatch_hint = nullptr;
    FastDispatchEntry& (*fast_dispatch_table_lookup)(u64) = nullptr;
    /// Fritchii (FRITCHII_JIT_STATE_REFS): the same lookup, the table passed in (instead of
    /// fast_dispatch_table_lookup, which holds it as an immediate).
    FastDispatchEntry& (*fast_dispatch_table_lookup_in)(u64, FastDispatchEntry*) = nullptr;

    /// Fritchii: FRITCHII_JIT_STATE_REFS (boot, 0..2; helper BO's stage 0 of one code cache for
    /// all cores). 1 = every per-core address generated code needs (BlockOfCode's CoreRef) is read
    /// through the state register instead of being an immediate, so a block's bytes do not depend
    /// on the core that compiled it; caches stay per core. 2 = 1, and each block's bytes are
    /// scanned for any 8 bytes that point into this core's objects (must be 0). 0 = old code.
    static int StateRefsMode();
    /// Built before the emitter: BlockOfCode takes it. `jit` = the lookup's argument.
    static StateRefConfig StateRefSetup(const A64::UserConfig& conf, const void* jit);
    /// Fritchii: FRITCHII_JIT_SHARED (boot, 0..2; stage 1 of one code cache for all cores). 1 =
    /// every guest core of a process runs one code cache (this emitter, one BlockOfCode), compiled
    /// once under a lock; each core keeps its state, dispatch table, return stack buffer and
    /// indirect slots. Needs FRITCHII_JIT_STATE_REFS=1 and compact links (FRITCHII_JIT_CHAIN 4 or 1
    /// with FRITCHII_JIT_FAR_CODE=1). 2 = 1 plus checks (lookups compared with the map, the cores'
    /// state refs checked at entry). 0 = one cache per core, the old code.
    static int SharedMode();
    /// FRITCHII_JIT_SHARED: the link stub an unlinked compact jump goes to.
    CodePtr UnlinkedJumpTarget(const IR::LocationDescriptor& target_desc) override;
    /// FRITCHII_JIT_STATE_REFS=2: an object of this core the scan looks for (from Jit::Impl).
    void AddStateRefScanRange(const void* begin, size_t size, const char* name);
    /// FRITCHII_JIT_STATE_REFS=2: count what in [begin, begin + size) points into this core's
    /// objects; `what` names the part in the first hits' lines.
    size_t ScanStateRefs(const u8* begin, size_t size, u64 pc, const char* what);
    void LogStateRefs(const char* when);
    struct StateRefScanRange {
        u64 begin;
        u64 size;
        const char* name;
    };
    std::vector<StateRefScanRange> state_ref_ranges;
    int state_refs = 0;
    bool state_ref_prelude_scanned = false;
    u64 state_ref_prelude_hits = 0;
    u64 state_ref_blocks = 0;
    u64 state_ref_hit_blocks = 0;
    u64 state_ref_hits = 0;
    u64 state_ref_hits_logged = 0;
    u64 state_ref_next_log = 256;

    // Fritchii: FRITCHII_JIT_INDIRECT_CACHE (boot; fable_a_gameceiling_answer 12.2). 1 = dry: each
    // indirect branch and return site keeps its last target in a slot and the shared handlers
    // count how often the next target equals it; 2 = each indirect branch site compares the
    // target with its slot and jumps straight to the cached block (a miss takes the shared
    // table, which refills the slot); 3 = 2, and each return site pops the return stack buffer
    // itself instead of jumping to the shared handler. 0 elsewhere (or without FastDispatch).
    /// Called from the constructor (the boot switch, read once per process).
    static int IndirectCacheMode();
    /// FRITCHII_JIT_NO_FRAME_POINTER (boot): blocks do not set up rbp (Eden 8e14f07a69's frame).
    static bool OmitFramePointer();
    /// FRITCHII_JIT_FLAG_BRANCH (boot, 0..2): 1 = a block's If terminal branches on the host flags
    /// when the guest NZCV it tests was made in the block and the flags were not touched since
    /// (no reload of NZCV, no sahf); 2 = 1, and the CheckBit terminal (cbz, cbnz, tbz, tbnz,
    /// exclusive stores) branches on a test of the byte A64SetCheckBit stored. 0 = old code.
    static int FlagBranchMode();
    /// FRITCHII_JIT_PIN (boot, Windows): 1 = up to two guest registers (FRITCHII_JIT_PIN_REGS, a
    /// comma list of sp and x0..x30, default "sp,x0") live in rsi and, with
    /// FRITCHII_JIT_NO_FRAME_POINTER=1, rbp while generated code runs; the state block gets them
    /// at every call out and at the exit. FRITCHII_JIT_VERIFY=1 adds the self-check (PinConfig);
    /// FRITCHII_JIT_VERIFY=2 the shadow and poison check (PinConfig::shadow), which also
    /// compares every pinned read with the shadow.
    /// Empty = old code. Built before the emitter: BlockOfCode takes it.
    static PinConfig PinSetup(const A64::UserConfig& conf);
    /// FRITCHII_JIT_VERIFY's line, at most once a second (from Jit::Run, on the owner thread).
    void LogPinMismatches(u64 total);
    /// FRITCHII_JIT_VERIFY=2's line, at most once a second (from Jit::Run, on the owner thread):
    /// the state block's counters and the C++ side's (poison reads by the accessors, poison
    /// found in a slot after the JIT returned).
    void LogPinChecks(const A64JitState& state, u64 poison_reads, u64 exit_poison);
    /// FRITCHII_JIT_PIN_STATS (boot): 1 = every guest register read and write in generated code
    /// counts (inc in the state block), logged every 10 s, to choose FRITCHII_JIT_PIN_REGS. Turns
    /// FRITCHII_JIT_FLAG_BRANCH off (the inc writes the flags).
    static bool PinStatsMode();
    void LogPinStats(const std::array<u64, 64>& counts);
    /// Code emitter (FRITCHII_JIT_PIN_STATS): counts one read or write of guest register `index`.
    void EmitPinStat(size_t index, bool write) {
        if (pin_stats) {
            code.inc(code.qword[code.ABI_JIT_PTR + offsetof(A64JitState, pin_stats) + sizeof(u64) * (index + (write ? 32 : 0))]);
        }
    }
    void EmitPinnedSet(A64EmitContext& ctx, size_t index, Xbyak::Reg64 host, const Xbyak::Address& addr, Argument& arg);
    /// FRITCHII_JIT_VERIFY: where a pinned write's copy goes (=2 its shadow, =1 its own slot).
    Xbyak::Address PinnedCopy(size_t index, const Xbyak::Address& own_slot) const {
        return pin_shadow ? code.qword[code.ABI_JIT_PTR + pin_shadow_offset[index]] : own_slot;
    }
    /// Code emitter (FRITCHII_JIT_VERIFY=2): a pinned read compares its host register with the
    /// shadow (a mismatch counts and the shadow wins). Keeps the host flags.
    void EmitPinnedReadCheck(A64EmitContext& ctx, size_t index, Xbyak::Reg64 host);
    /// The host register guest register `index` (0..30, 31 = SP) is pinned to, if it is.
    std::optional<Xbyak::Reg64> PinnedHost(size_t index) const {
        const int idx = pin_host_index[index];
        return idx < 0 ? std::nullopt : std::optional<Xbyak::Reg64>{Xbyak::Reg64(idx)};
    }
    /// A free slot, or nullptr when all are taken (the site then uses the shared handler).
    FastDispatchEntry* AllocateIndirectSlot();
    /// A slot's byte offset in the slot array (under 4 MiB).
    u32 IndirectSlotOffset(const FastDispatchEntry* slot) const {
        return u32(size_t(slot - indirect_slots.get()) * sizeof(FastDispatchEntry));
    }
    /// Code emitter: reg = the slot's address (FRITCHII_JIT_STATE_REFS: the slot array from the
    /// state block plus its offset, else the immediate).
    void EmitIndirectSlotAddress(const Xbyak::Reg64& reg, const FastDispatchEntry* slot) {
        if (state_refs != 0) {
            code.mov(reg, code.qword[code.ABI_JIT_PTR + code.CoreRefOffset(CoreRef::IndirectSlots)]);
            code.lea(reg, code.ptr[reg + IndirectSlotOffset(slot)]);
        } else {
            code.mov(reg, u64(slot));
        }
    }
    /// Every slot back to the empty entry; `release`: the sites are gone too (ClearCache).
    void ClearIndirectSlots(bool release);
    /// rbx = the location descriptor of the guest pc and fpcr (rdi, rcx clobbered), as the
    /// shared handlers compute it.
    void EmitCalculateLocationDescriptor();
    /// The dry counters' line, at most every 10 s (from Jit::Run, on the owner thread).
    void LogIndirectCounters();
    /// The same for any core's counters (FRITCHII_JIT_SHARED: each core has its own).
    struct IndirectLog;
    void LogIndirectCounters(const u64* counters, IndirectLog& log, unsigned core);
    enum IndirectCounter : size_t {
        IndirectRuns,       ///< =1: indirect branches run through a dry site
        IndirectSameTarget, ///< of them, same target as the site's last
        ReturnRuns,         ///< =1: returns run through a dry site
        ReturnSameTarget,   ///< of them, same target as the site's last
        ReturnRsbHits,      ///< of them, served by the return stack buffer
        SlotMisses,         ///< >=2: slot compares that failed (the shared table then)
        IndirectCounterCount,
    };
    int indirect_cache_mode = 0;
    /// Set while InvalidateCacheRanges removes blocks (it empties the slots once, after).
    bool indirect_bulk_invalidation = false;
    bool omit_frame_pointer = false;
    int flag_branch_mode = 0;
    /// FRITCHII_JIT_PIN: host register index per guest register (31 = SP), -1 = in the state block.
    std::array<int, 32> pin_host_index;
    bool pin_verify = false;  ///< FRITCHII_JIT_VERIFY=1 or 2
    bool pin_shadow = false;  ///< FRITCHII_JIT_VERIFY=2
    std::array<u32, 32> pin_shadow_offset{};  ///< per guest register (31 = SP), =2 only
    u64 pin_checks_logged = 0;
    bool pin_stats = false;
    std::array<u64, 64> pin_stats_logged{};
    s64 pin_stats_log_ns = 0;
    u64 pin_mismatches_logged = 0;
    s64 pin_log_ns = 0;
    /// Set by Emit around its EmitTerminal call (FRITCHII_JIT_FLAG_BRANCH): the If terminal's
    /// guest NZCV / the CheckBit terminal's ZF are in the host flags at the terminal.
    bool terminal_nzcv_in_host_flags = false;
    bool terminal_check_bit_in_host_zf = false;
    static constexpr size_t indirect_slot_capacity = size_t{1} << 18; // 4 MiB per JIT
    std::unique_ptr<FastDispatchEntry[]> indirect_slots;
    /// Atomic for FRITCHII_JIT_SHARED, where other cores read it to empty their own slots.
    std::atomic<size_t> indirect_slots_used{0};
    std::array<u64, IndirectCounterCount> indirect_counters{};  // written by generated code
    struct IndirectLog {
        std::array<u64, IndirectCounterCount> logged{};
        s64 log_ns = 0;
    };
    IndirectLog indirect_log;
    /// Inside the fast dispatch handler, right after the descriptor is computed (rbx).
    const void* terminal_handler_fast_dispatch_after_calc = nullptr;
    const void* terminal_handler_indirect_dry = nullptr;  ///< rax = the site's slot
    const void* terminal_handler_pop_rsb_dry = nullptr;   ///< rax = the site's slot
    const void* terminal_handler_indirect_slot_miss = nullptr; ///< rbx = descriptor, rax = slot

    /// Fritchii: FRITCHII_JIT_FAR_CODE (boot, default 0; Fable A 2.1). 1 = a block's slow paths
    /// (deferred emits) go to a far region at the end of the code space, so the hot bytes of
    /// blocks sit next to each other. Only with a code cache of 64 MiB or more.
    static bool FarCodeEnabled();
    bool far_code = false;
    /// The last emitted block's far part (nullptr / 0 when it had none), for the code dump.
    CodePtr last_far_entry = nullptr;
    size_t last_far_size = 0;

    /// Fritchii: FRITCHII_JIT_CHAIN (boot, bits, default 0; Fable A 2.2). Off with cycle counting.
    /// 4 = compact links: a direct link is one 5-byte jump (to the block, or to a far stub that
    ///     stores the pc and goes to the dispatcher; a nop when the block follows right behind);
    ///     a checked link is `cmp halt, 0; jne <far halt stub>; <slot>`. Needs FAR_CODE.
    /// 2 = halt check only on direct links that go back (target pc <= block start pc), including
    ///     B/BL links that have none today; indirect branches and returns gain one. Between two
    ///     checks only forward links run, so a run without a check is bounded.
    /// 1 = chain: a block's last link target, when not compiled yet, is compiled right behind it
    ///     (FRITCHII_JIT_CHAIN_DEPTH blocks deep, default 4), so the link becomes a nop; If and
    ///     CheckBit put the fall-through leaf last. Needs FAR_CODE, implies 4.
    static int ChainBits();
    static int ChainDepth();
    bool compact_links = false;
    bool backward_halt_checks = false;
    bool chain_successors = false;
    /// Far stubs per link target (cleared with the cache): store the pc, then to the dispatcher /
    /// out of the run (halt).
    ::Common::unordered_map<u64, CodePtr> link_stubs;
    ::Common::unordered_map<u64, CodePtr> halt_stubs;
    CodePtr LinkStub(const IR::LocationDescriptor& target);
    CodePtr HaltStub(const IR::LocationDescriptor& target);
    /// A direct link with the Fritchii shapes (compact and/or backward-only check).
    void EmitFritchiiLink(const IR::LocationDescriptor& target, IR::LocationDescriptor initial_location, bool checked_by_default);
    /// The block being emitted ends here (its fall-through successor).
    IR::LocationDescriptor current_end_location{0};
    /// The last compact slot emitted and where it ends: the block to chain, when it is the last
    /// thing in the block.
    std::optional<IR::LocationDescriptor> chain_target;
    CodePtr chain_slot_end = nullptr;
    /// The block to compile right behind the last one (chain), if any.
    std::optional<IR::LocationDescriptor> ChainCandidate() const;

    A64::Jit* jit_interface = nullptr;
    void (*memory_read_128)() = nullptr;
    void (*memory_write_128)() = nullptr;
    void (*memory_exclusive_write_128)() = nullptr;
};

}  // namespace Dynarmic::Backend::X64
