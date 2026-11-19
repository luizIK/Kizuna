// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#define CONCATENATE_TOKENS(x, y) CONCATENATE_TOKENS_IMPL(x, y)
#define CONCATENATE_TOKENS_IMPL(x, y) x##y

#define AxxEmitX64 CONCATENATE_TOKENS(Axx, EmitX64)
#define AxxEmitContext CONCATENATE_TOKENS(Axx, EmitContext)
#define AxxJitState CONCATENATE_TOKENS(Axx, JitState)
#define AxxUserConfig Axx::UserConfig

namespace {
using Vector = std::array<u64, 2>;
}

std::optional<AxxEmitX64::DoNotFastmemMarker> AxxEmitX64::ShouldFastmem(AxxEmitContext& ctx, IR::Inst* inst) const {
    if (!conf.fastmem_pointer || !exception_handler.SupportsFastmem()) {
        return std::nullopt;
    }

    const auto marker = std::make_tuple(ctx.Location(), inst->GetName());
    if (do_not_fastmem.count(marker) > 0) {
        return std::nullopt;
    }
    // Fritchii (FRITCHII_JIT_SHARED_FASTMEM_MARKS): an access that faulted on another core.
    if (shared_fastmem_marks) {
        bool marked = false;
        {
            std::shared_lock lock{shared_fastmem_marks->mutex};
            marked = shared_fastmem_marks->marks.count(marker) > 0;
        }
        if (marked) {
            NoteSharedFastmemMarkUsed(*shared_fastmem_marks, unsigned(conf.processor_id));
            return std::nullopt;
        }
    }
    return marker;
}

void AxxEmitX64::AddFastmemPatch(CodePtr location, CodePtr resume_location, void (*callback)(), const DoNotFastmemMarker& marker, bool recompile) {
    const u64 base = std::bit_cast<u64>(code.getCode());
    const auto offset = [base](u64 address) {
        ASSERT(address >= base && address - base <= UINT32_MAX);
        return u32(address - base);
    };
    FastmemPatchInfo info{
        .rip = offset(std::bit_cast<u64>(location)),
        .resume_rip = offset(std::bit_cast<u64>(resume_location)),
        .callback = offset(std::bit_cast<u64>(callback)),
        .inst_name = 0,
        .recompile = 0,
        .location = std::get<0>(marker),
    };
    info.inst_name = std::get<1>(marker);
    info.recompile = recompile ? 1 : 0;
    // Code is emitted in address order; only a block's deferred fallbacks can come after a later
    // access of the same block.
    auto it = fastmem_patch_info.end();
    if (!fastmem_patch_info.empty() && fastmem_patch_info.back().rip >= info.rip) {
        it = std::lower_bound(fastmem_patch_info.begin(), fastmem_patch_info.end(), info.rip,
                              [](const FastmemPatchInfo& entry, u32 value) { return entry.rip < value; });
        if (it != fastmem_patch_info.end() && it->rip == info.rip) {
            return;
        }
    }
    fastmem_patch_info.insert(it, info);
}

FakeCall AxxEmitX64::FastmemCallback(u64 rip_) {
    // Fritchii (FRITCHII_JIT_SHARED): the patch table, the markers and the block map are the
    // process's, written by whichever core compiles; the compile lock covers them. No thread holds
    // it while running generated code, so the faulting core can always take it.
    std::unique_lock<std::mutex> shared_lock;
    if (shared_code) {
        shared_lock = std::unique_lock<std::mutex>{shared_code->compile_mutex};
    }
    const u64 base = std::bit_cast<u64>(code.getCode());
    const auto iter = rip_ >= base && rip_ - base <= UINT32_MAX
                        ? std::lower_bound(fastmem_patch_info.begin(), fastmem_patch_info.end(), u32(rip_ - base),
                                           [](const FastmemPatchInfo& entry, u32 value) { return entry.rip < value; })
                        : fastmem_patch_info.end();
    if (iter != fastmem_patch_info.end() && iter->rip == u32(rip_ - base)) {
        FakeCall result{
            .call_rip = base + iter->callback,
            .ret_rip = base + iter->resume_rip,
        };
        if (iter->recompile) {
            const DoNotFastmemMarker marker{iter->location, static_cast<unsigned>(iter->inst_name)};
            do_not_fastmem.insert(marker);
            if (shared_fastmem_marks) {
                // Fritchii (FRITCHII_JIT_SHARED_FASTMEM_MARKS): the other cores compile this
                // access without fastmem from now on (their blocks already compiled still fault
                // once each, as before).
                std::unique_lock lock{shared_fastmem_marks->mutex};
                if (shared_fastmem_marks->marks.insert(marker).second) {
                    shared_fastmem_marks->faults.fetch_add(1, std::memory_order_relaxed);
                }
            }
            bool drop = true;
            if (shared_code) {
                // FRITCHII_JIT_SHARED: another core may still run a copy dropped before (its own
                // dispatch table drops it at its next entry); only the block now in the map, the
                // one holding this access, goes, not the one compiled again without fastmem.
                const auto current = block_descriptors.find(iter->location);
                const u64 entry = current != block_descriptors.end() ? std::bit_cast<u64>(current->second.entrypoint) : 0;
                drop = entry != 0 && rip_ >= entry && rip_ - entry < current->second.size;
            }
            if (drop) {
                InvalidateBasicBlocks({std::get<0>(marker)});
            }
            if (drop && ::Common::Nx2Hitch::SplitEnabled()) {
                // Fritchii (FRITCHII_HITCH_SPLIT): a block dropped to be compiled again without
                // fastmem for this access (once per access and core).
                ::Common::Nx2Hitch::AddCount(::Common::Nx2Hitch::Cause::CodeDropFastmem, 0, 1);
            }
        }
        return result;
    }
    fmt::print("dynarmic: Segfault happened within JITted code at rip = {:016x}\n"
        "Segfault wasn't at a fastmem patch location!\n", rip_);
    UNREACHABLE(); //("iter != fastmem_patch_info.end()");
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitMemoryRead(AxxEmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!conf.page_table && !fastmem_marker) {
        // Neither fastmem nor page table: Use callbacks
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.HostCall(code, nullptr, {}, args[1]);
            if (ordered) code.mfence();
            code.CallFunction(memory_read_128);
            ctx.reg_alloc.DefineValue(code, inst, xmm1);
        } else {
            ctx.reg_alloc.HostCall(code, inst, {}, args[1]);
            code.mov(code.ABI_PARAM3.cvt32(), bitsize / CHAR_BIT);
            if (ordered) code.mfence();
            Devirtualize<callback>(conf.callbacks).EmitCall(code);
            code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
        }
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    if (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const int value_idx = bitsize == 128 ? ctx.reg_alloc.ScratchXmm(code).getIdx() : ctx.reg_alloc.ScratchGpr(code).getIdx();

    const auto wrapped_fn = read_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();

    if (fastmem_marker) {
        // Use fastmem
        bool require_abort_handling = false;
        const auto src_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);

            AddFastmemPatch(location, code.getCurr(), wrapped_fn, *fastmem_marker, conf.recompile_on_fastmem_failure);

            EmitCheckMemoryAbort(ctx, inst, end);
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        // Use page table
        ASSERT(conf.page_table);
        const auto src_ptr = EmitVAddrLookup(code, ctx, bitsize, *abort, vaddr);
        EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);
            EmitCheckMemoryAbort(ctx, inst, end);
            code.jmp(*end, code.T_NEAR);
        });
    }
    code.L(*end);

    if constexpr (bitsize == 128) {
        ctx.reg_alloc.DefineValue(code, inst, Xbyak::Xmm{value_idx});
    } else {
        ctx.reg_alloc.DefineValue(code, inst, Xbyak::Reg64{value_idx});
    }
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitMemoryWrite(AxxEmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    const auto fastmem_marker = ShouldFastmem(ctx, inst);

    if (!conf.page_table && !fastmem_marker) {
        // Neither fastmem nor page table: Use callbacks
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.Use(code, args[1], ABI_PARAM2);
            ctx.reg_alloc.Use(code, args[2], HostLoc::XMM1);
            ctx.reg_alloc.EndOfAllocScope();
            ctx.reg_alloc.HostCall(code, nullptr);
            code.CallFunction(memory_write_128);
        } else {
            // { this, vaddr, value, size }
            ctx.reg_alloc.HostCall(code, nullptr, {}, args[1], args[2]);
            code.mov(code.ABI_PARAM4.cvt32(), bitsize / CHAR_BIT);
            Devirtualize<callback>(conf.callbacks).EmitCall(code);
        }
        if (ordered) code.mfence();
        EmitCheckMemoryAbort(ctx, inst);
        return;
    }

    if (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const int value_idx = bitsize == 128
                            ? ctx.reg_alloc.UseXmm(code, args[2]).getIdx()
                            : (ordered ? ctx.reg_alloc.UseScratchGpr(code, args[2]).getIdx() : ctx.reg_alloc.UseGpr(code, args[2]).getIdx());

    const auto wrapped_fn = write_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();

    if (fastmem_marker) {
        // Use fastmem
        bool require_abort_handling = false;
        const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitWriteMemoryMov<bitsize>(code, dest_ptr, value_idx, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);

            AddFastmemPatch(location, code.getCurr(), wrapped_fn, *fastmem_marker, conf.recompile_on_fastmem_failure);

            EmitCheckMemoryAbort(ctx, inst, end);
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        // Use page table
        ASSERT(conf.page_table);
        const auto dest_ptr = EmitVAddrLookup(code, ctx, bitsize, *abort, vaddr);
        EmitWriteMemoryMov<bitsize>(code, dest_ptr, value_idx, ordered);

        ctx.deferred_emits.emplace_back([=, this, &ctx] {
            code.L(*abort);
            code.call(wrapped_fn);
            EmitCheckMemoryAbort(ctx, inst, end);
            code.jmp(*end, code.T_NEAR);
        });
    }
    code.L(*end);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveReadMemory(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor != nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[2].GetImmediateAccType());

    if constexpr (bitsize != 128) {
        using T = mcl::unsigned_integer_of_size<bitsize>;

        ctx.reg_alloc.HostCall(code, inst, {}, args[1]);

        code.mov(code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)], u8(1));
        code.MovCoreRef(code.ABI_PARAM1, CoreRef::Conf, reinterpret_cast<u64>(&conf));
        if (ordered) {
            code.mfence();
        }
        code.CallLambda([](AxxUserConfig& conf, Axx::VAddr vaddr) -> T {
            return conf.global_monitor->ReadAndMark<T>(conf.processor_id, vaddr, [&]() -> T {
                return (conf.callbacks->*callback)(vaddr, bitsize / CHAR_BIT);
            });
        });
        code.ZeroExtendFrom(bitsize, code.ABI_RETURN);
    } else {
        const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm(code);
        ctx.reg_alloc.Use(code, args[1], ABI_PARAM2);
        ctx.reg_alloc.EndOfAllocScope();
        ctx.reg_alloc.HostCall(code, nullptr);

        code.mov(code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)], u8(1));
        code.MovCoreRef(code.ABI_PARAM1, CoreRef::Conf, reinterpret_cast<u64>(&conf));
        ctx.reg_alloc.AllocStackSpace(code, 16 + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        if (ordered) {
            code.mfence();
        }
        code.CallLambda([](AxxUserConfig& conf, Axx::VAddr vaddr, Vector& ret) {
            ret = conf.global_monitor->ReadAndMark<Vector>(conf.processor_id, vaddr, [&]() -> Vector {
                return (conf.callbacks->*callback)(vaddr);
            });
        });
        code.movups(result, xword[rsp + ABI_SHADOW_SPACE]);
        ctx.reg_alloc.ReleaseStackSpace(code, 16 + ABI_SHADOW_SPACE);

        ctx.reg_alloc.DefineValue(code, inst, result);
    }

    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveWriteMemory(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor != nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());

    if constexpr (bitsize == 128) {
        ctx.reg_alloc.Use(code, args[1], ABI_PARAM2);
        ctx.reg_alloc.Use(code, args[2], HostLoc::XMM1);
        ctx.reg_alloc.EndOfAllocScope();
        ctx.reg_alloc.HostCall(code, inst);
    } else {
        ctx.reg_alloc.HostCall(code, inst, {}, args[1], args[2]);
    }

    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr(code);
    Xbyak::Label end;
    code.mov(code.ABI_RETURN, u32(1));
    code.movzx(tmp.cvt32(), code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)]);
    code.test(tmp.cvt8(), tmp.cvt8());
    // Fritchii (FRITCHII_JIT_PIN): the call's pin stores and reloads may outgrow a short jump.
    code.je(end, code.Pins().regs.empty() ? code.T_AUTO : code.T_NEAR);
    code.xor_(tmp.cvt32(), tmp.cvt32());
    code.xchg(tmp.cvt8(), code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)]);
    code.MovCoreRef(code.ABI_PARAM1, CoreRef::Conf, u64(&conf));
    if constexpr (bitsize != 128) {
        using T = mcl::unsigned_integer_of_size<bitsize>;
        code.CallLambda([](AxxUserConfig& conf, Axx::VAddr vaddr, T value) -> u32 {
            return conf.global_monitor->DoExclusiveOperation<T>(conf.processor_id, vaddr, [&](T expected) -> bool {
                return (conf.callbacks->*callback)(vaddr, value, expected);
            }) ? 0 : 1;
        });
        if (ordered) {
            code.mfence();
        }
    } else {
        ctx.reg_alloc.AllocStackSpace(code, 16 + ABI_SHADOW_SPACE);
        code.lea(code.ABI_PARAM3, ptr[rsp + ABI_SHADOW_SPACE]);
        code.movaps(xword[code.ABI_PARAM3], xmm1);
        code.CallLambda([](AxxUserConfig& conf, Axx::VAddr vaddr, Vector& value) -> u32 {
            return conf.global_monitor->DoExclusiveOperation<Vector>(conf.processor_id, vaddr, [&](Vector expected) -> bool {
                return (conf.callbacks->*callback)(vaddr, value, expected);
            }) ? 0 : 1;
        });
        if (ordered) {
            code.mfence();
        }
        ctx.reg_alloc.ReleaseStackSpace(code, 16 + ABI_SHADOW_SPACE);
    }
    code.L(end);

    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveReadMemoryInline(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor && conf.fastmem_pointer);
    if (!exception_handler.SupportsFastmem()) {
        EmitExclusiveReadMemory<bitsize, callback>(ctx, inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    constexpr bool ordered = true;

    if constexpr (ordered && bitsize == 128) {
        // Required for atomic 128-bit loads/stores
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(code, HostLoc::RDX);
    }

    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const int value_idx = bitsize == 128 ? ctx.reg_alloc.ScratchXmm(code).getIdx() : ctx.reg_alloc.ScratchGpr(code).getIdx();
    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr(code);
    const Xbyak::Reg64 tmp2 = ctx.reg_alloc.ScratchGpr(code);

    const auto wrapped_fn = read_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value_idx)];

    EmitExclusiveLock(code, conf, tmp, tmp2.cvt32());

    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)], u8(1));
    code.MovCoreRef(tmp, CoreRef::MonitorAddress, std::bit_cast<u64>(GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id)));
    code.mov(qword[tmp], vaddr);

    const auto fastmem_marker = ShouldFastmem(ctx, inst);
    if (fastmem_marker) {
        SharedLabel abort = ctx.GenSharedLabel(), end = ctx.GenSharedLabel();
        bool require_abort_handling = false;

        const auto src_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling);

        const auto location = EmitReadMemoryMov<bitsize>(code, value_idx, src_ptr, ordered);

        AddFastmemPatch(location, code.getCurr(), wrapped_fn, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);

        code.L(*end);

        if (require_abort_handling) {
            ctx.deferred_emits.emplace_back([=, this] {
                code.L(*abort);
                code.call(wrapped_fn);
                code.jmp(*end, code.T_NEAR);
            });
        }
    } else {
        code.call(wrapped_fn);
    }

    code.MovCoreRef(tmp, CoreRef::MonitorValue, std::bit_cast<u64>(GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id)));
    EmitWriteMemoryMov<bitsize>(code, tmp, value_idx, false);

    EmitExclusiveUnlock(code, conf, tmp, tmp2.cvt32());

    if constexpr (bitsize == 128) {
        ctx.reg_alloc.DefineValue(code, inst, Xbyak::Xmm{value_idx});
    } else {
        ctx.reg_alloc.DefineValue(code, inst, Xbyak::Reg64{value_idx});
    }

    EmitCheckMemoryAbort(ctx, inst);
}

template<std::size_t bitsize, auto callback>
void AxxEmitX64::EmitExclusiveWriteMemoryInline(AxxEmitContext& ctx, IR::Inst* inst) {
    ASSERT(conf.global_monitor && conf.fastmem_pointer);
    if (!exception_handler.SupportsFastmem()) {
        EmitExclusiveWriteMemory<bitsize, callback>(ctx, inst);
        return;
    }

    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    constexpr bool ordered = true;

    const auto value = [&] {
        if constexpr (bitsize == 128) {
            ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
            ctx.reg_alloc.ScratchGpr(code, HostLoc::RBX);
            ctx.reg_alloc.ScratchGpr(code, HostLoc::RCX);
            ctx.reg_alloc.ScratchGpr(code, HostLoc::RDX);
            return ctx.reg_alloc.UseXmm(code, args[2]);
        } else {
            ctx.reg_alloc.ScratchGpr(code, HostLoc::RAX);
            return ctx.reg_alloc.UseGpr(code, args[2]);
        }
    }();
    const Xbyak::Reg64 vaddr = ctx.reg_alloc.UseGpr(code, args[1]);
    const Xbyak::Reg32 status = ctx.reg_alloc.ScratchGpr(code).cvt32();
    const Xbyak::Reg64 tmp = ctx.reg_alloc.ScratchGpr(code);
    const Xbyak::Reg64 tmp2 = ctx.reg_alloc.ScratchGpr(code);

    const auto wrapped_fn = exclusive_write_fallbacks[std::make_tuple(ordered, bitsize, vaddr.getIdx(), value.getIdx())];

    EmitExclusiveLock(code, conf, tmp, tmp2.cvt32());

    SharedLabel end = ctx.GenSharedLabel();

    code.mov(status, u32(1));
    code.movzx(tmp.cvt32(), code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)]);
    code.test(tmp.cvt8(), tmp.cvt8());
    code.je(*end, code.T_NEAR);
    code.MovCoreRef(tmp, CoreRef::MonitorAddress, std::bit_cast<u64>(GetExclusiveMonitorAddressPointer(conf.global_monitor, conf.processor_id)));
    code.cmp(qword[tmp], vaddr);
    code.jne(*end, code.T_NEAR);

    EmitExclusiveTestAndClear(code, conf, vaddr, tmp, rax);

    code.mov(code.byte[code.ABI_JIT_PTR + offsetof(AxxJitState, exclusive_state)], u8(0));
    code.MovCoreRef(tmp, CoreRef::MonitorValue, std::bit_cast<u64>(GetExclusiveMonitorValuePointer(conf.global_monitor, conf.processor_id)));

    if constexpr (bitsize == 128) {
        code.mov(rax, qword[tmp + 0]);
        code.mov(rdx, qword[tmp + 8]);
        if (code.HasHostFeature(HostFeature::SSE41)) {
            code.movq(rbx, value);
            code.pextrq(rcx, value, 1);
        } else {
            code.movaps(xmm0, value);
            code.movq(rbx, xmm0);
            code.punpckhqdq(xmm0, xmm0);
            code.movq(rcx, xmm0);
        }
    } else {
        EmitReadMemoryMov<bitsize>(code, rax.getIdx(), tmp, false);
    }

    const auto fastmem_marker = ShouldFastmem(ctx, inst);
    if (fastmem_marker) {
        SharedLabel abort = ctx.GenSharedLabel();
        bool require_abort_handling = false;

        const auto dest_ptr = EmitFastmemVAddr(code, ctx, *abort, vaddr, require_abort_handling, tmp);

        const auto location = code.getCurr();

        switch (bitsize) {
        case 8:
            code.lock();
            code.cmpxchg(code.byte[dest_ptr], value.cvt8());
            break;
        case 16:
            code.lock();
            code.cmpxchg(word[dest_ptr], value.cvt16());
            break;
        case 32:
            code.lock();
            code.cmpxchg(dword[dest_ptr], value.cvt32());
            break;
        case 64:
            code.lock();
            code.cmpxchg(qword[dest_ptr], value.cvt64());
            break;
        case 128:
            code.lock();
            code.cmpxchg16b(ptr[dest_ptr]);
            break;
        default:
            UNREACHABLE();
        }

        code.setnz(status.cvt8());
        ctx.deferred_emits.emplace_back([=, this] {
            code.L(*abort);
            code.call(wrapped_fn);

            AddFastmemPatch(location, code.getCurr(), wrapped_fn, *fastmem_marker, conf.recompile_on_exclusive_fastmem_failure);

            code.xor_(status.cvt32(), status.cvt32()); //dep-break
            code.test(code.al, code.al);
            code.setz(status.cvt8());
            code.jmp(*end, code.T_NEAR);
        });
    } else {
        code.call(wrapped_fn);
        code.xor_(status.cvt32(), status.cvt32()); //dep-break
        code.test(code.al, code.al);
        code.setz(status.cvt8());
    }

    code.L(*end);
    EmitExclusiveUnlock(code, conf, tmp, eax);
    ctx.reg_alloc.DefineValue(code, inst, status);
    EmitCheckMemoryAbort(ctx, inst);
}

#undef AxxEmitX64
#undef AxxEmitContext
#undef AxxJitState
#undef AxxUserConfig
