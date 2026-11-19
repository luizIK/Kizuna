// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/breadth_first_search.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/dominance.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/storage_write_extent.h"

namespace Shader::Optimization {
namespace {
/// Address in constant buffers to the storage buffer descriptor
struct StorageBufferAddr {
    auto operator<=>(const StorageBufferAddr&) const noexcept = default;

    u32 index;
    u32 offset;
    /// Offset of the buffer pointer in the memory the constant buffer pointer points to
    u32 indirect{~0U};
};

/// Block iterator to a global memory instruction and the storage buffer it uses
struct StorageInst {
    StorageBufferAddr storage_buffer;
    IR::Inst* inst;
    IR::Block* block;
};

/// Bias towards a certain range of constant buffers when looking for storage buffers
struct Bias {
    u32 index;
    u32 offset_begin;
    u32 offset_end;
    u32 alignment;
};

using boost::container::flat_set;
using boost::container::small_vector;
using StorageBufferSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;
using StorageInstVector = small_vector<StorageInst, 24>;
using StorageWritesSet =
    flat_set<StorageBufferAddr, std::less<StorageBufferAddr>, small_vector<StorageBufferAddr, 16>>;

struct StorageInfo {
    StorageBufferSet set;
    StorageInstVector to_replace;
    StorageWritesSet writes;
    /// Constant buffer words that step strided iterators, see CollectIteratorWords
    StorageBufferSet iterator_words;
    /// Constant buffer pointers a global atomic uses as its address, see CollectAtomicPointers
    StorageBufferSet atomic_pointers;
    /// Track results by the instruction the search starts from, with and without the NVN bias:
    /// accesses through the same address share them, and a search can walk thousands of
    /// instructions (the IR does not change while storage buffers are collected).
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> biased_tracks;
    std::unordered_map<const IR::Inst*, std::optional<StorageBufferAddr>> unbiased_tracks;
    /// Instructions from which a search with (or without) the bias cannot succeed, so later
    /// searches from other addresses skip what failed ones walked.
    std::unordered_set<const IR::Inst*> biased_exhausted;
    std::unordered_set<const IR::Inst*> unbiased_exhausted;
    std::unordered_map<const IR::Inst*, small_vector<StorageBufferAddr, 4>> native_candidates;
    /// Native accesses whose candidates include slots of a dynamically indexed pointer array.
    std::unordered_set<const IR::Inst*> native_dynamic;
    /// A native access with several candidate pointers was left global.
    bool native_runtime_bounds{};
    /// Paired accesses bind the loaded pointers that compete with their pair; cleared when those
    /// leave too few bindings.
    bool bind_competing_loaded{true};
    /// A paired access bound the loaded pointers that compete with its pair.
    bool competing_loaded_bound{};
    /// Pointer table entries reached through a phi or select of table pointers, or through a
    /// table pointer read from a dynamically indexed constant buffer array. They are bound only
    /// in the bindings the program's other storage buffers leave free (table_budget).
    bool tables_enabled{};
    bool tables_seen{};
    size_t table_budget{};
    size_t table_used{};
    /// The storage buffers the pass without table entries bound; bound again, they take nothing
    /// from table_budget, which they were left out of.
    StorageBufferSet counted;
    small_vector<StorageBufferAddr, 8> table_candidates;
    /// Buffers a store or atomic that stayed global may write to (at run time, at any offset)
    StorageBufferSet unbounded_writes;
    /// A store or atomic stayed global without a buffer: it may land in any written buffer
    bool untracked_writes{};
    /// A load stayed global without a buffer: it may read any bound buffer
    bool untracked_reads{};
};

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemory(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicCompareExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Returns true when the backend can pick a global access's buffer among the bound ones at run time
bool IsRuntimeRoutable(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicCompareExchange32:
        return true;
    default:
        return false;
    }
}

/// Returns true when the instruction is a global memory instruction
bool IsGlobalMemoryWrite(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicCompareExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return true;
    default:
        return false;
    }
}

/// Converts a global memory opcode to its storage buffer equivalent
IR::Opcode GlobalToStorage(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
        return IR::Opcode::LoadStorageS8;
    case IR::Opcode::LoadGlobalU8:
        return IR::Opcode::LoadStorageU8;
    case IR::Opcode::LoadGlobalS16:
        return IR::Opcode::LoadStorageS16;
    case IR::Opcode::LoadGlobalU16:
        return IR::Opcode::LoadStorageU16;
    case IR::Opcode::LoadGlobal32:
        return IR::Opcode::LoadStorage32;
    case IR::Opcode::LoadGlobal64:
        return IR::Opcode::LoadStorage64;
    case IR::Opcode::LoadGlobal128:
        return IR::Opcode::LoadStorage128;
    case IR::Opcode::WriteGlobalS8:
        return IR::Opcode::WriteStorageS8;
    case IR::Opcode::WriteGlobalU8:
        return IR::Opcode::WriteStorageU8;
    case IR::Opcode::WriteGlobalS16:
        return IR::Opcode::WriteStorageS16;
    case IR::Opcode::WriteGlobalU16:
        return IR::Opcode::WriteStorageU16;
    case IR::Opcode::WriteGlobal32:
        return IR::Opcode::WriteStorage32;
    case IR::Opcode::WriteGlobal64:
        return IR::Opcode::WriteStorage64;
    case IR::Opcode::WriteGlobal128:
        return IR::Opcode::WriteStorage128;
    case IR::Opcode::GlobalAtomicIAdd32:
        return IR::Opcode::StorageAtomicIAdd32;
    case IR::Opcode::GlobalAtomicSMin32:
        return IR::Opcode::StorageAtomicSMin32;
    case IR::Opcode::GlobalAtomicUMin32:
        return IR::Opcode::StorageAtomicUMin32;
    case IR::Opcode::GlobalAtomicSMax32:
        return IR::Opcode::StorageAtomicSMax32;
    case IR::Opcode::GlobalAtomicUMax32:
        return IR::Opcode::StorageAtomicUMax32;
    case IR::Opcode::GlobalAtomicInc32:
        return IR::Opcode::StorageAtomicInc32;
    case IR::Opcode::GlobalAtomicDec32:
        return IR::Opcode::StorageAtomicDec32;
    case IR::Opcode::GlobalAtomicAnd32:
        return IR::Opcode::StorageAtomicAnd32;
    case IR::Opcode::GlobalAtomicOr32:
        return IR::Opcode::StorageAtomicOr32;
    case IR::Opcode::GlobalAtomicXor32:
        return IR::Opcode::StorageAtomicXor32;
    case IR::Opcode::GlobalAtomicExchange32:
        return IR::Opcode::StorageAtomicExchange32;
    case IR::Opcode::GlobalAtomicCompareExchange32:
        return IR::Opcode::StorageAtomicCompareExchange32;
    case IR::Opcode::GlobalAtomicIAdd64:
        return IR::Opcode::StorageAtomicIAdd64;
    case IR::Opcode::GlobalAtomicSMin64:
        return IR::Opcode::StorageAtomicSMin64;
    case IR::Opcode::GlobalAtomicUMin64:
        return IR::Opcode::StorageAtomicUMin64;
    case IR::Opcode::GlobalAtomicSMax64:
        return IR::Opcode::StorageAtomicSMax64;
    case IR::Opcode::GlobalAtomicUMax64:
        return IR::Opcode::StorageAtomicUMax64;
    case IR::Opcode::GlobalAtomicAnd64:
        return IR::Opcode::StorageAtomicAnd64;
    case IR::Opcode::GlobalAtomicOr64:
        return IR::Opcode::StorageAtomicOr64;
    case IR::Opcode::GlobalAtomicXor64:
        return IR::Opcode::StorageAtomicXor64;
    case IR::Opcode::GlobalAtomicExchange64:
        return IR::Opcode::StorageAtomicExchange64;
    case IR::Opcode::GlobalAtomicIAdd32x2:
        return IR::Opcode::StorageAtomicIAdd32x2;
    case IR::Opcode::GlobalAtomicSMin32x2:
        return IR::Opcode::StorageAtomicSMin32x2;
    case IR::Opcode::GlobalAtomicUMin32x2:
        return IR::Opcode::StorageAtomicUMin32x2;
    case IR::Opcode::GlobalAtomicSMax32x2:
        return IR::Opcode::StorageAtomicSMax32x2;
    case IR::Opcode::GlobalAtomicUMax32x2:
        return IR::Opcode::StorageAtomicUMax32x2;
    case IR::Opcode::GlobalAtomicAnd32x2:
        return IR::Opcode::StorageAtomicAnd32x2;
    case IR::Opcode::GlobalAtomicOr32x2:
        return IR::Opcode::StorageAtomicOr32x2;
    case IR::Opcode::GlobalAtomicXor32x2:
        return IR::Opcode::StorageAtomicXor32x2;
    case IR::Opcode::GlobalAtomicExchange32x2:
        return IR::Opcode::StorageAtomicExchange32x2;
    case IR::Opcode::GlobalAtomicAddF32:
        return IR::Opcode::StorageAtomicAddF32;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF16x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF16x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF16x2;
    case IR::Opcode::GlobalAtomicAddF32x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF32x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        throw InvalidArgument("Invalid global memory opcode {}", opcode);
    }
}

/// Returns true when a storage buffer address satisfies a bias
bool MeetsBias(const StorageBufferAddr& storage_buffer, const Bias& bias) noexcept {
    return storage_buffer.index == bias.index && storage_buffer.offset >= bias.offset_begin &&
           storage_buffer.offset < bias.offset_end;
}

struct LowAddrInfo {
    IR::U32 value;
    IR::U32 high;
    s32 imm_offset;
};

/// Tries to track the first 32-bits of a global memory instruction
std::optional<LowAddrInfo> TrackLowAddress(IR::Inst* inst) {
    // The first argument is the low level GPU pointer to the global memory instruction
    const IR::Value addr{inst->Arg(0)};
    if (addr.IsImmediate()) {
        // Not much we can do if it's an immediate
        return std::nullopt;
    }
    // This address is expected to either be a PackUint2x32, a IAdd64, or a CompositeConstructU32x2
    IR::Inst* addr_inst{addr.InstRecursive()};
    s32 imm_offset{0};
    if (addr_inst->GetOpcode() == IR::Opcode::IAdd64) {
        // If it's an IAdd64, get the immediate offset it is applying and grab the address
        // instruction. This expects for the instruction to be canonicalized having the address on
        // the first argument and the immediate offset on the second one.
        const IR::U64 imm_offset_value{addr_inst->Arg(1)};
        if (!imm_offset_value.IsImmediate()) {
            return std::nullopt;
        }
        imm_offset = static_cast<s32>(static_cast<s64>(imm_offset_value.U64()));
        const IR::U64 iadd_addr{addr_inst->Arg(0)};
        if (iadd_addr.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = iadd_addr.InstRecursive();
    }
    // With IAdd64 handled, now PackUint2x32 is expected
    if (addr_inst->GetOpcode() == IR::Opcode::PackUint2x32) {
        // PackUint2x32 is expected to be generated from a vector
        const IR::Value vector{addr_inst->Arg(0)};
        if (vector.IsImmediate()) {
            return std::nullopt;
        }
        addr_inst = vector.InstRecursive();
    }
    // The vector is expected to be a CompositeConstructU32x2
    if (addr_inst->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return std::nullopt;
    }
    // Grab the first argument from the CompositeConstructU32x2, this is the low address.
    return LowAddrInfo{
        .value{IR::U32{addr_inst->Arg(0)}},
        .high{IR::U32{addr_inst->Arg(1)}},
        .imm_offset = imm_offset,
    };
}

/// Tries to track the storage buffer address used by a global memory instruction
std::optional<StorageBufferAddr> Track(const IR::Value& value, const Bias* bias,
                                       std::unordered_set<const IR::Inst*>& exhausted) {
    const auto pred{[bias](const IR::Inst* inst) -> std::optional<StorageBufferAddr> {
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 &&
            inst->GetOpcode() != IR::Opcode::GetCbufU32x2) {
            return std::nullopt;
        }
        const IR::Value index{inst->Arg(0)};
        const IR::Value offset{inst->Arg(1)};
        if (!index.IsImmediate()) {
            // Definitely not a storage buffer if it's read from a
            // non-immediate index
            return std::nullopt;
        }
        if (!offset.IsImmediate()) {
            // TODO: Support SSBO arrays
            return std::nullopt;
        }
        const StorageBufferAddr storage_buffer{
            .index = index.U32(),
            .offset = offset.U32(),
        };
        const u32 alignment{bias ? bias->alignment : 8U};
        if (!Common::IsAligned(storage_buffer.offset, alignment)) {
            // The SSBO pointer has to be aligned
            return std::nullopt;
        }
        if (bias && !MeetsBias(storage_buffer, *bias)) {
            // We have to blacklist some addresses in case we wrongly
            // point to them
            return std::nullopt;
        }
        return storage_buffer;
    }};
    // A pointer loaded from memory is a new address, not an offset into the buffer that supplied
    // it. Do not follow that load back to its own constant-buffer descriptor.
    const auto stop_at_memory = [](IR::Value operand) {
        if (!operand.IsImmediate() && IsGlobalMemory(*operand.InstRecursive())) {
            return IR::Value{u32{0}};
        }
        return operand;
    };
    return BreadthFirstSearchPruned(value, pred, exhausted, stop_at_memory);
}

/// Finds the one constant buffer word added directly into the low address, which is the base
/// pointer even when other constant buffer reads feed the computed offset
std::optional<StorageBufferAddr> TrackAddend(const IR::Value& value) {
    std::optional<StorageBufferAddr> result;
    bool ambiguous{};
    const auto visit{[&](const auto& self, const IR::Value& term) -> void {
        if (term.IsImmediate() || ambiguous) {
            return;
        }
        const IR::Inst* const inst{term.InstRecursive()};
        if (inst->GetOpcode() == IR::Opcode::IAdd32) {
            self(self, inst->Arg(0));
            self(self, inst->Arg(1));
            return;
        }
        if (inst->GetOpcode() != IR::Opcode::GetCbufU32 || !inst->Arg(0).IsImmediate() ||
            !inst->Arg(1).IsImmediate() || !Common::IsAligned(inst->Arg(1).U32(), 8)) {
            return;
        }
        const StorageBufferAddr found{.index = inst->Arg(0).U32(), .offset = inst->Arg(1).U32()};
        ambiguous = result.has_value();
        result = found;
    }};
    visit(visit, value);
    return ambiguous ? std::nullopt : result;
}

/// Returns the constant buffer word an instruction reads, including one half of a 64-bit read
std::optional<StorageBufferAddr> CbufWord(const IR::Inst* inst) {
    if (inst->GetOpcode() == IR::Opcode::CompositeExtractU32x2 && !inst->Arg(0).IsImmediate()) {
        const IR::Inst* const vector{inst->Arg(0).InstRecursive()};
        if (vector->GetOpcode() != IR::Opcode::GetCbufU32x2 || !vector->Arg(0).IsImmediate() ||
            !vector->Arg(1).IsImmediate()) {
            return std::nullopt;
        }
        return StorageBufferAddr{
            .index = vector->Arg(0).U32(),
            .offset = vector->Arg(1).U32() + inst->Arg(1).U32() * 4,
        };
    }
    if (inst->GetOpcode() != IR::Opcode::GetCbufU32 || !inst->Arg(0).IsImmediate() ||
        !inst->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    return StorageBufferAddr{.index = inst->Arg(0).U32(), .offset = inst->Arg(1).U32()};
}

struct TrackingConditions {
    using Predicate = std::pair<IR::Value, bool>;

    struct Scope {
        Predicate condition;
        IR::Block* header;
        IR::Block* merge;
    };
    std::unordered_map<const IR::Inst*, IR::Block*> owners;
    std::unordered_map<IR::Block*, std::vector<Scope>> scopes;
    std::unordered_map<IR::Block*, std::vector<Scope>> merges;
    std::unordered_map<IR::Block*, IR::Block*> loops;
    IR::Block* current{};
    using BitKey = std::tuple<const IR::Inst*, u32, const IR::Inst*, int>;
    mutable std::map<BitKey, std::optional<Predicate>> bit_cache;

    explicit TrackingConditions(IR::Program& program) {
        for (IR::Block* block : program.post_order_blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (inst.GetOpcode() == IR::Opcode::Phi) {
                    owners.emplace(&inst, block);
                }
            }
        }
        std::vector<Scope> active;
        std::vector<IR::Block*> loop_stack;
        IR::Block* previous{};
        for (const auto& node : program.syntax_list) {
            switch (node.type) {
            case IR::AbstractSyntaxNode::Type::Block:
                previous = node.data.block;
                scopes[previous] = active;
                loops[previous] = loop_stack.empty() ? nullptr : loop_stack.back();
                break;
            case IR::AbstractSyntaxNode::Type::If: {
                bit_cache.clear();
                const Scope scope{Literal(node.data.if_node.cond), previous,
                                  node.data.if_node.merge};
                merges[scope.merge].push_back(scope);
                active.push_back(scope);
                break;
            }
            case IR::AbstractSyntaxNode::Type::EndIf:
                active.pop_back();
                break;
            case IR::AbstractSyntaxNode::Type::Loop:
                loop_stack.push_back(node.data.loop.body);
                break;
            case IR::AbstractSyntaxNode::Type::Repeat:
                loop_stack.pop_back();
                break;
            default:
                break;
            }
        }
    }

    static std::optional<bool> Constant(Predicate value) {
        if (value.first.IsImmediate() && value.first.Type() == IR::Type::U1) {
            return value.first.U1() == value.second;
        }
        return std::nullopt;
    }

    std::optional<Predicate> Bit(IR::Value value, u32 bit, u32 depth,
                                 std::optional<Predicate> assumed = {}) const {
        if (value.IsImmediate())
            return BitImpl(value, bit, depth, assumed);
        const BitKey key{
            value.InstRecursive(), bit, assumed ? assumed->first.TryInstRecursive() : nullptr,
            assumed
                ? (Constant(*assumed).has_value() ? 4 + *Constant(*assumed) : 1 + assumed->second)
                : 0};
        if (const auto found = bit_cache.find(key); found != bit_cache.end())
            return found->second;
        bit_cache[key] = std::nullopt;
        const auto result = BitImpl(value, bit, depth, assumed);
        bit_cache[key] = result;
        return result;
    }

    std::optional<Predicate> BitImpl(IR::Value value, u32 bit, u32 depth,
                                     std::optional<Predicate> assumed) const {
        if (bit >= 32 || depth == 0) {
            return std::nullopt;
        }
        if (value.IsImmediate()) {
            return Predicate{IR::Value{((value.U32() >> bit) & 1) != 0}, true};
        }
        const auto* inst = value.InstRecursive();
        if (inst->GetOpcode() == IR::Opcode::Phi && inst->NumArgs() == 2) {
            const auto owner = owners.find(inst);
            if (owner != owners.end()) {
                const auto merge = merges.find(owner->second);
                if (merge != merges.end() && merge->second.size() == 1) {
                    const auto& branch = merge->second.front();
                    size_t bypass = inst->PhiBlock(0) == branch.header ? 0 : 1;
                    if (inst->PhiBlock(bypass) == branch.header &&
                        inst->PhiBlock(1 - bypass)->GetOrder() < owner->second->GetOrder()) {
                        if (assumed && assumed->first == branch.condition.first) {
                            const bool take_body = assumed->second == branch.condition.second;
                            return Bit(inst->Arg(take_body ? 1 - bypass : bypass), bit, depth - 1,
                                       assumed);
                        }
                        auto false_guard = branch.condition;
                        false_guard.second = !false_guard.second;
                        const auto no = Bit(inst->Arg(bypass), bit, depth - 1, false_guard);
                        const auto yes =
                            Bit(inst->Arg(1 - bypass), bit, depth - 1, branch.condition);
                        if (no && yes) {
                            if (*no == *yes)
                                return no;
                            const auto nc = Constant(*no);
                            const auto yc = Constant(*yes);
                            if (nc && yc) {
                                if (*nc == *yc)
                                    return Predicate{IR::Value{*yc}, true};
                                auto result = branch.condition;
                                result.second = result.second == *yc;
                                return result;
                            }
                        }
                    }
                }
            }
            return std::nullopt;
        }
        if (inst->GetOpcode() == IR::Opcode::SelectU32) {
            const auto yes = Bit(inst->Arg(1), bit, depth - 1, assumed);
            const auto no = Bit(inst->Arg(2), bit, depth - 1, assumed);
            if (yes && no) {
                if (*yes == *no)
                    return yes;
                const auto yc = Constant(*yes);
                const auto nc = Constant(*no);
                if (yc && nc) {
                    if (*nc == *yc)
                        return Predicate{IR::Value{*yc}, true};
                    auto cond = Literal(inst->Arg(0), depth - 1);
                    cond.second = cond.second == *yc;
                    return cond;
                }
            }
        }
        if (inst->GetOpcode() == IR::Opcode::BitwiseOr32 ||
            inst->GetOpcode() == IR::Opcode::BitwiseAnd32) {
            const auto lhs = Bit(inst->Arg(0), bit, depth - 1, assumed);
            const auto rhs = Bit(inst->Arg(1), bit, depth - 1, assumed);
            const bool is_or = inst->GetOpcode() == IR::Opcode::BitwiseOr32;
            if (lhs) {
                if (const auto c = Constant(*lhs))
                    return *c == is_or ? lhs : rhs;
            }
            if (rhs) {
                if (const auto c = Constant(*rhs))
                    return *c == is_or ? rhs : lhs;
            }
        }
        if (inst->GetOpcode() == IR::Opcode::BitFieldUExtract && inst->Arg(1).IsImmediate() &&
            inst->Arg(2).IsImmediate()) {
            if (bit >= inst->Arg(2).U32())
                return Predicate{IR::Value{false}, true};
            return Bit(inst->Arg(0), bit + inst->Arg(1).U32(), depth - 1, assumed);
        }
        return std::nullopt;
    }

    Predicate Literal(IR::Value value, u32 depth = 256) const {
        value = value.Resolve();
        if (value.IsImmediate() || depth == 0) {
            return {value, true};
        }
        const auto* inst = value.InstRecursive();
        switch (inst->GetOpcode()) {
        case IR::Opcode::ConditionRef:
            return Literal(inst->Arg(0), depth - 1);
        case IR::Opcode::LogicalNot: {
            auto result = Literal(inst->Arg(0), depth - 1);
            result.second = !result.second;
            return result;
        }
        case IR::Opcode::SelectU1: {
            if (const auto c = Constant(Literal(inst->Arg(0), depth - 1))) {
                return Literal(inst->Arg(*c ? 1 : 2), depth - 1);
            }
            break;
        }
        case IR::Opcode::IEqual:
        case IR::Opcode::INotEqual:
            if (!inst->Arg(1).IsImmediate() || inst->Arg(1).U32() != 0) {
                break;
            }
            [[fallthrough]];
        case IR::Opcode::GetZeroFromOp: {
            const bool is_zero = inst->GetOpcode() != IR::Opcode::INotEqual;
            const auto input = inst->Arg(0).Resolve();
            if (input.IsImmediate() && input.Type() == IR::Type::U32) {
                return {IR::Value{input.U32() == 0}, is_zero};
            }
            if (!input.IsImmediate()) {
                const auto* extract = input.InstRecursive();
                if (extract->GetOpcode() == IR::Opcode::BitwiseAnd32) {
                    for (size_t arg = 0; arg < 2; ++arg) {
                        const auto mask = extract->Arg(arg).Resolve();
                        if (mask.IsImmediate() && std::has_single_bit(mask.U32())) {
                            if (auto result = Bit(extract->Arg(1 - arg),
                                                  std::countr_zero(mask.U32()), depth - 1)) {
                                result->second = result->second != is_zero;
                                return *result;
                            }
                        }
                    }
                }
                if (extract->GetOpcode() == IR::Opcode::BitFieldUExtract &&
                    extract->Arg(2).IsImmediate() && extract->Arg(2).U32() == 1) {
                    if (auto result = Bit(input, 0, depth - 1)) {
                        result->second = result->second != is_zero;
                        return *result;
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return {IR::Value{const_cast<IR::Inst*>(inst)}, true};
    }

    bool IsBackEdge(const IR::Inst* phi, size_t arg) const {
        const auto owner = owners.find(phi);
        return owner != owners.end() && phi->PhiBlock(arg)->GetOrder() >= owner->second->GetOrder();
    }

    bool AllowsPhi(const IR::Inst* phi, size_t arg) const {
        const auto owner = owners.find(phi);
        if (owner == owners.end() || !loops.contains(owner->second) || !loops.contains(current) ||
            loops.at(owner->second) != loops.at(current)) {
            return true;
        }
        const auto merge = merges.find(owner->second);
        const auto current_scopes = scopes.find(current);
        if (merge == merges.end() || current_scopes == scopes.end()) {
            return true;
        }
        for (const auto& branch : merge->second) {
            const auto [condition, positive] = branch.condition;
            for (const auto& known : current_scopes->second) {
                const auto [known_condition, known_positive] = known.condition;
                if (condition != known_condition) {
                    continue;
                }
                const bool taken = positive == known_positive;
                const bool bypass = phi->PhiBlock(arg) == branch.header;
                if (bypass == taken) {
                    return false;
                }
            }
        }
        return true;
    }
};

/// Visits the constant buffer words summed into a value, nearest first, looking through phis
template <typename Func>
void ForEachSummedWord(const IR::Value& value, Func&& func,
                       const TrackingConditions* conditions = nullptr) {
    std::deque<std::tuple<IR::Value, u32, bool>> queue{{value, 0, true}};
    std::array<std::unordered_set<const IR::Inst*>, 2> visited;
    while (!queue.empty()) {
        const auto [term, depth, same_iteration] = queue.front();
        queue.pop_front();
        if (term.IsImmediate()) {
            continue;
        }
        const IR::Inst* const inst{term.InstRecursive()};
        if (!visited[same_iteration].insert(inst).second) {
            continue;
        }
        if (const auto word{CbufWord(inst)}) {
            func(*word, depth);
            continue;
        }
        if (inst->GetOpcode() == IR::Opcode::IAdd32 || inst->GetOpcode() == IR::Opcode::Phi ||
            inst->GetOpcode() == IR::Opcode::SelectU32) {
            for (size_t arg = inst->GetOpcode() == IR::Opcode::SelectU32 ? 1 : 0;
                 arg < inst->NumArgs(); ++arg) {
                if (inst->GetOpcode() == IR::Opcode::Phi && conditions && same_iteration &&
                    !conditions->AllowsPhi(inst, arg)) {
                    continue;
                }
                const bool next_iteration =
                    same_iteration && !(conditions && inst->GetOpcode() == IR::Opcode::Phi &&
                                        conditions->IsBackEdge(inst, arg));
                queue.emplace_back(inst->Arg(arg), depth + 1, next_iteration);
            }
        }
    }
}

struct WordSum {
    StorageBufferSet words;
    u32 count{};
    u32 budget{64};
    bool reaches_loop{};
};

/// Collects the constant buffer words of an addition tree and returns false when another value
/// feeds it. Reaching the phi given as the loop variable is recorded in the sum.
bool SumWords(const IR::Value& value, const IR::Inst* loop, WordSum& sum) {
    if (value.IsImmediate()) {
        return true;
    }
    if (sum.budget == 0) {
        return false;
    }
    --sum.budget;
    const IR::Inst* const inst{value.InstRecursive()};
    if (inst == loop) {
        sum.reaches_loop = true;
        return false;
    }
    if (const auto word{CbufWord(inst)}) {
        sum.words.insert(*word);
        ++sum.count;
        return true;
    }
    if (inst->GetOpcode() != IR::Opcode::IAdd32) {
        return false;
    }
    const bool lhs{SumWords(inst->Arg(0), loop, sum)};
    if (!lhs && !loop) {
        return false;
    }
    const bool rhs{SumWords(inst->Arg(1), loop, sum)};
    return lhs && rhs;
}

/// Finds the constant buffer words that step strided iterators (CUTLASS-style kernel parameters):
/// words added to a loop-carried value on its back edge, and words summed with two or more other
/// words into an offset table. A base pointer is neither.
StorageBufferSet CollectIteratorWords(IR::Program& program) {
    StorageBufferSet result;
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                for (size_t arg = 0; arg < inst.NumArgs(); ++arg) {
                    WordSum sum;
                    SumWords(inst.Arg(arg), &inst, sum);
                    if (sum.reaches_loop) {
                        result.insert(sum.words.begin(), sum.words.end());
                    }
                }
            } else if (inst.GetOpcode() == IR::Opcode::IAdd32) {
                WordSum sum;
                if (SumWords(IR::Value{&inst}, nullptr, sum) && sum.words.size() >= 3) {
                    result.insert(sum.words.begin(), sum.words.end());
                }
            }
        }
    }
    std::unordered_map<const IR::Inst*, u32> indices;
    std::unordered_map<const IR::Inst*, u32> lowlinks;
    std::unordered_set<const IR::Inst*> on_stack;
    std::vector<const IR::Inst*> stack;
    u32 next_index{};
    const auto visit = [&](auto&& self, const IR::Inst* inst) -> void {
        if (indices.contains(inst)) {
            return;
        }
        const u32 index = next_index++;
        indices[inst] = lowlinks[inst] = index;
        stack.push_back(inst);
        on_stack.insert(inst);
        const auto op = inst->GetOpcode();
        if (op == IR::Opcode::IAdd32 || op == IR::Opcode::Phi || op == IR::Opcode::SelectU32) {
            for (size_t arg = op == IR::Opcode::SelectU32 ? 1 : 0; arg < inst->NumArgs(); ++arg) {
                if (inst->Arg(arg).IsImmediate()) {
                    continue;
                }
                const IR::Inst* next = inst->Arg(arg).InstRecursive();
                if (!indices.contains(next)) {
                    self(self, next);
                    lowlinks[inst] = std::min(lowlinks[inst], lowlinks[next]);
                } else if (on_stack.contains(next)) {
                    lowlinks[inst] = std::min(lowlinks[inst], indices[next]);
                }
            }
        }
        if (lowlinks[inst] != index) {
            return;
        }
        std::vector<const IR::Inst*> component;
        const IR::Inst* member;
        do {
            member = stack.back();
            stack.pop_back();
            on_stack.erase(member);
            component.push_back(member);
        } while (member != inst);
        if (component.size() < 2) {
            return;
        }
        const std::unordered_set<const IR::Inst*> members(component.begin(), component.end());
        for (const IR::Inst* add : component) {
            if (add->GetOpcode() != IR::Opcode::IAdd32) {
                continue;
            }
            for (size_t arg = 0; arg < add->NumArgs(); ++arg) {
                const auto& term = add->Arg(arg);
                if (!term.IsImmediate() && members.contains(term.InstRecursive())) {
                    continue;
                }
                ForEachSummedWord(term, [&](StorageBufferAddr word, u32) { result.insert(word); });
            }
        }
    };
    for (IR::Block* block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                visit(visit, &inst);
            }
        }
    }
    return result;
}

/// Returns true when a condition reads a constant buffer word, as the carry out of an addition to
/// that word does
bool ReadsCbufWord(const IR::Value& condition, StorageBufferAddr word) {
    std::deque<IR::Value> queue{condition};
    std::unordered_set<const IR::Inst*> visited;
    while (!queue.empty() && visited.size() < 64) {
        const IR::Value value{queue.front()};
        queue.pop_front();
        if (value.IsImmediate()) {
            continue;
        }
        const IR::Inst* const inst{value.InstRecursive()};
        if (!visited.insert(inst).second) {
            continue;
        }
        if (CbufWord(inst) == word) {
            return true;
        }
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            queue.push_back(inst->Arg(arg));
        }
    }
    return false;
}

/// Finds the 64-bit constant buffer pointers a global atomic's address is based on: the whole
/// address apart from an immediate offset, or a 64-bit sum whose high half adds the pointer's high
/// word and the carry out of its low word. An atomic has no global fallback, so such a pointer is a
/// buffer even while it is unmapped. A 32-bit scalar added to a pointer carries nothing.
StorageBufferSet CollectAtomicPointers(IR::Program& program) {
    const auto word = [](const IR::U32& value) {
        return value.IsImmediate() ? std::nullopt : CbufWord(value.InstRecursive());
    };
    StorageBufferSet result;
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (!IsGlobalMemoryWrite(inst) || inst.Type() == IR::Type::Void) {
                continue;
            }
            const std::optional<LowAddrInfo> addr{TrackLowAddress(&inst)};
            if (!addr) {
                continue;
            }
            const std::optional<StorageBufferAddr> low{word(addr->value)};
            if (low && Common::IsAligned(low->offset, 8) &&
                word(addr->high) ==
                    StorageBufferAddr{.index = low->index, .offset = low->offset + 4}) {
                result.insert(*low);
                continue;
            }
            StorageBufferSet high_words;
            small_vector<IR::Value, 2> carries;
            std::vector<IR::Value> pending{addr->high};
            std::unordered_set<const IR::Inst*> visited;
            while (!pending.empty()) {
                const IR::Value term{pending.back()};
                pending.pop_back();
                if (term.IsImmediate() || !visited.insert(term.InstRecursive()).second) {
                    continue;
                }
                const IR::Inst* const term_inst{term.InstRecursive()};
                if (const auto high_word{CbufWord(term_inst)}) {
                    high_words.insert(*high_word);
                    continue;
                }
                switch (term_inst->GetOpcode()) {
                case IR::Opcode::IAdd32:
                case IR::Opcode::Phi:
                    for (size_t arg = 0; arg < term_inst->NumArgs(); ++arg) {
                        pending.push_back(term_inst->Arg(arg));
                    }
                    break;
                case IR::Opcode::SelectU32:
                    if (term_inst->Arg(1).IsImmediate() && term_inst->Arg(2).IsImmediate()) {
                        carries.push_back(term_inst->Arg(0));
                    }
                    break;
                default:
                    break;
                }
            }
            if (carries.empty()) {
                continue;
            }
            ForEachSummedWord(addr->value, [&](StorageBufferAddr pointer, u32) {
                if (Common::IsAligned(pointer.offset, 8) &&
                    high_words.contains({.index = pointer.index, .offset = pointer.offset + 4}) &&
                    std::ranges::any_of(carries, [&](const IR::Value& carry) {
                        return ReadsCbufWord(carry, pointer);
                    })) {
                    result.insert(pointer);
                }
            });
        }
    }
    return result;
}

/// Finds the 64-bit constant buffer pointer summed into an address: its low word is summed into
/// the low half and its high word into the high half. Iterator steps and 32-bit scalars such as
/// strides and sizes are skipped. Competing bases require the legacy nearest-word fallback.
std::optional<StorageBufferAddr> TrackPointerAddend(const LowAddrInfo& addr,
                                                    const StorageBufferSet& iterator_words,
                                                    const TrackingConditions& conditions,
                                                    bool legacy_nearest = false) {
    StorageBufferSet high_words;
    ForEachSummedWord(
        addr.high, [&](StorageBufferAddr word, u32) { high_words.insert(word); }, &conditions);
    std::optional<StorageBufferAddr> result;
    u32 result_depth{};
    bool tied{};
    StorageBufferSet candidates;
    ForEachSummedWord(
        addr.value,
        [&](StorageBufferAddr word, u32 depth) {
            if (!Common::IsAligned(word.offset, 8) || iterator_words.contains(word) ||
                !high_words.contains({.index = word.index, .offset = word.offset + 4})) {
                return;
            }
            candidates.insert(word);
            if (!result || depth < result_depth) {
                result = word;
                result_depth = depth;
                tied = false;
            } else if (depth == result_depth && *result != word) {
                tied = true;
            }
        },
        &conditions);
    return tied || (!legacy_nearest && candidates.size() != 1) ? std::nullopt : result;
}

/// The first slot and stride of a pointer array read at a dynamic constant buffer offset:
/// (index << shift) + base or index * stride + base.
std::optional<std::pair<u32, u32>> DynamicPointerSlots(const IR::Value& offset) {
    const auto stride_of = [](const IR::Value& value) -> std::optional<u32> {
        if (value.IsImmediate()) {
            return std::nullopt;
        }
        const IR::Inst* const inst{value.InstRecursive()};
        if (inst->GetOpcode() == IR::Opcode::ShiftLeftLogical32 && inst->Arg(1).IsImmediate() &&
            inst->Arg(1).U32() < 16) {
            return 1U << inst->Arg(1).U32();
        }
        if (inst->GetOpcode() == IR::Opcode::IMul32 && inst->Arg(1).IsImmediate()) {
            return inst->Arg(1).U32();
        }
        return std::nullopt;
    };
    u32 base{};
    std::optional<u32> stride{stride_of(offset)};
    if (!stride && !offset.IsImmediate() &&
        offset.InstRecursive()->GetOpcode() == IR::Opcode::IAdd32) {
        const IR::Inst* const add{offset.InstRecursive()};
        for (size_t arg = 0; arg < 2 && !stride; ++arg) {
            if (add->Arg(arg ^ 1).IsImmediate()) {
                base = add->Arg(arg ^ 1).U32();
                stride = stride_of(add->Arg(arg));
            }
        }
    }
    if (!stride || *stride < 8 || !Common::IsAligned(*stride, 8)) {
        return std::nullopt;
    }
    return std::pair{base, *stride};
}

/// Slots of a pointer array read at a dynamic offset, following phis and selects of the offset.
/// A stride of zero is a single slot.
void CollectPointerSlots(const IR::Value& offset, small_vector<std::pair<u32, u32>, 4>& slots,
                         std::unordered_set<const IR::Inst*>& visited) {
    if (offset.IsImmediate()) {
        if (std::ranges::find(slots, std::pair{offset.U32(), 0U}) == slots.end()) {
            slots.emplace_back(offset.U32(), 0U);
        }
        return;
    }
    const IR::Inst* const inst{offset.InstRecursive()};
    if (!visited.insert(inst).second) {
        return;
    }
    if (const auto pattern{DynamicPointerSlots(offset)}) {
        if (std::ranges::find(slots, *pattern) == slots.end()) {
            slots.push_back(*pattern);
        }
        return;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::Phi:
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            CollectPointerSlots(inst->Arg(arg), slots, visited);
        }
        break;
    case IR::Opcode::SelectU32:
        CollectPointerSlots(inst->Arg(1), slots, visited);
        CollectPointerSlots(inst->Arg(2), slots, visited);
        break;
    default:
        break;
    }
}

/// Native programs: every constant buffer pointer the low address can come from, following only
/// additions, selects and phis. A predicated address assignment becomes a phi whose other input
/// can point into a different buffer, so the search does not stop at the first pointer. A pointer
/// read from an array at a dynamic offset contributes every mapped slot of the array.
small_vector<StorageBufferAddr, 4> NativePointerCandidates(const IR::U32& low_addr,
                                                           Environment& env, bool& dynamic,
                                                           StorageInfo& info) {
    small_vector<StorageBufferAddr, 4> candidates;
    small_vector<const IR::Inst*, 16> pending{low_addr.InstRecursive()};
    std::unordered_set<const IR::Inst*> visited{pending.front()};
    const auto enqueue = [&](const IR::Value& value) {
        if (!value.IsImmediate() && visited.insert(value.InstRecursive()).second) {
            pending.push_back(value.InstRecursive());
        }
    };
    for (size_t next = 0; next < pending.size(); ++next) {
        const IR::Inst* const inst{pending[next]};
        switch (inst->GetOpcode()) {
        case IR::Opcode::GetCbufU32:
        case IR::Opcode::GetCbufU32x2: {
            const IR::Value index{inst->Arg(0)};
            const IR::Value offset{inst->Arg(1)};
            if (index.IsImmediate() && !offset.IsImmediate()) {
                static constexpr u32 max_slots{16};
                small_vector<std::pair<u32, u32>, 4> slots;
                std::unordered_set<const IR::Inst*> slot_visited;
                CollectPointerSlots(offset, slots, slot_visited);
                for (const auto& [first, stride] : slots) {
                    for (u32 slot = 0; slot < (stride ? max_slots : 1U); ++slot) {
                        const StorageBufferAddr pointer{.index = index.U32(),
                                                        .offset = first + slot * stride};
                        if (Common::IsAligned(pointer.offset, 8) &&
                            std::ranges::find(candidates, pointer) == candidates.end() &&
                            env.IsCbufStorageBufferPointer(pointer.index, pointer.offset)) {
                            candidates.push_back(pointer);
                            dynamic = true;
                        }
                    }
                }
                break;
            }
            if (!index.IsImmediate() || !offset.IsImmediate()) {
                break;
            }
            const StorageBufferAddr pointer{.index = index.U32(), .offset = offset.U32()};
            if (Common::IsAligned(pointer.offset, 8) &&
                std::ranges::find(candidates, pointer) == candidates.end() &&
                env.IsCbufStorageBufferPointer(pointer.index, pointer.offset)) {
                candidates.push_back(pointer);
            }
            break;
        }
        case IR::Opcode::IAdd32:
        case IR::Opcode::Phi:
            for (size_t arg = inst->NumArgs(); arg--;) {
                enqueue(inst->Arg(arg));
            }
            break;
        case IR::Opcode::SelectU32:
            enqueue(inst->Arg(1));
            enqueue(inst->Arg(2));
            break;
        case IR::Opcode::ShuffleIndex:
            // A pointer made uniform (R2UR broadcasts the first active lane's value).
            enqueue(inst->Arg(0));
            break;
        case IR::Opcode::CompositeExtractU32x2:
            if (inst->Arg(1).IsImmediate() && inst->Arg(1).U32() == 0) {
                enqueue(inst->Arg(0));
            }
            break;
        case IR::Opcode::LoadGlobal64: {
            // A pointer loaded from the memory a constant buffer pointer points to (the binding
            // reads it again for every dispatch and draw).
            // The table address: a tracked low word, or a constant buffer pointer packed whole.
            const IR::Inst* holder_root{};
            s32 holder_offset{};
            const auto holder_addr{TrackLowAddress(const_cast<IR::Inst*>(inst))};
            if (holder_addr) {
                if (holder_addr->value.IsImmediate()) {
                    break;
                }
                holder_root = holder_addr->value.InstRecursive();
                holder_offset = holder_addr->imm_offset;
            } else {
                IR::Value address{inst->Arg(0)};
                if (!address.IsImmediate() &&
                    address.InstRecursive()->GetOpcode() == IR::Opcode::IAdd64 &&
                    address.InstRecursive()->Arg(1).IsImmediate()) {
                    holder_offset =
                        static_cast<s32>(static_cast<s64>(address.InstRecursive()->Arg(1).U64()));
                    address = address.InstRecursive()->Arg(0);
                }
                if (address.IsImmediate() ||
                    address.InstRecursive()->GetOpcode() != IR::Opcode::PackUint2x32 ||
                    address.InstRecursive()->Arg(0).IsImmediate()) {
                    break;
                }
                holder_root = address.InstRecursive()->Arg(0).InstRecursive();
            }
            if (holder_offset < 0 || !Common::IsAligned(holder_offset, 8)) {
                break;
            }
            const u32 indirect{static_cast<u32>(holder_offset)};
            // The table pointer can also be one of several constant buffer pointers (a phi or
            // select of them), or an entry of a constant buffer array indexed at run time (one
            // table per object type: every slot whose table entry is a pointer). Those are table
            // candidates, bound only in the bindings the other storage buffers leave free.
            const auto add = [&](const StorageBufferAddr& pointer, bool table) {
                if (std::ranges::find(candidates, pointer) != candidates.end()) {
                    return;
                }
                if (table && env.ShaderStage() != Stage::Compute) {
                    // Graphics programs bind every candidate for every draw: table entries
                    // multiply the storage buffers of draws that never read them.
                    return;
                }
                if (table) {
                    info.tables_seen = true;
                    if (!info.tables_enabled) {
                        return;
                    }
                    if (std::ranges::find(info.table_candidates, pointer) ==
                        info.table_candidates.end()) {
                        info.table_candidates.push_back(pointer);
                    }
                }
                candidates.push_back(pointer);
                dynamic = true;
            };
            small_vector<std::pair<const IR::Inst*, bool>, 4> holders{{holder_root, false}};
            std::unordered_set<const IR::Inst*> holder_visited{holder_root};
            for (size_t pending_holder = 0; pending_holder < holders.size(); ++pending_holder) {
                const auto [holder, via_phi] = holders[pending_holder];
                const auto enqueue_holder = [&](const IR::Value& value, bool branch) {
                    if (!value.IsImmediate() && holders.size() < 8 &&
                        holder_visited.insert(value.InstRecursive()).second) {
                        holders.emplace_back(value.InstRecursive(), via_phi || branch);
                    }
                };
                switch (holder->GetOpcode()) {
                case IR::Opcode::CompositeExtractU32x2:
                    if (holder->Arg(1).IsImmediate() && holder->Arg(1).U32() == 0) {
                        enqueue_holder(holder->Arg(0), false);
                    }
                    continue;
                case IR::Opcode::Phi:
                    for (size_t arg = 0; arg < holder->NumArgs(); ++arg) {
                        enqueue_holder(holder->Arg(arg), true);
                    }
                    continue;
                case IR::Opcode::SelectU32:
                    enqueue_holder(holder->Arg(1), true);
                    enqueue_holder(holder->Arg(2), true);
                    continue;
                case IR::Opcode::GetCbufU32:
                case IR::Opcode::GetCbufU32x2:
                    break;
                default:
                    continue;
                }
                if (!holder->Arg(0).IsImmediate()) {
                    continue;
                }
                const u32 index{holder->Arg(0).U32()};
                if (holder->Arg(1).IsImmediate()) {
                    const StorageBufferAddr pointer{
                        .index = index, .offset = holder->Arg(1).U32(), .indirect = indirect};
                    if (Common::IsAligned(pointer.offset, 8) &&
                        env.IsCbufStorageBufferPointer(pointer.index, pointer.offset)) {
                        add(pointer, via_phi || !holder_addr);
                    }
                    continue;
                }
                static constexpr u32 max_slots{16};
                small_vector<std::pair<u32, u32>, 4> slots;
                std::unordered_set<const IR::Inst*> slot_visited;
                CollectPointerSlots(holder->Arg(1), slots, slot_visited);
                for (const auto& [first, stride] : slots) {
                    for (u32 slot = 0; slot < (stride ? max_slots : 1U); ++slot) {
                        const StorageBufferAddr pointer{
                            .index = index, .offset = first + slot * stride, .indirect = indirect};
                        if (Common::IsAligned(pointer.offset, 8) &&
                            env.IsCbufIndirectStorageBufferPointer(pointer.index, pointer.offset,
                                                                   pointer.indirect)) {
                            add(pointer, true);
                        }
                    }
                }
            }
            break;
        }
        default:
            break;
        }
    }
    return candidates;
}

/// Collects the storage buffer used by a global memory instruction and the instruction itself.
/// Returns false when the access stays global without a buffer.
bool CollectStorageBuffers(IR::Block& block, IR::Inst& inst, StorageInfo& info, Environment& env,
                           const TrackingConditions& conditions) {
    // NVN puts storage buffers in a specific range, we have to bias towards these addresses to
    // avoid getting false positives
    static constexpr Bias nvn_bias{
        .index = 0,
        .offset_begin = 0x100,
        .offset_end = 0x700,
        .alignment = 16,
    };
    // Track the low address of the instruction
    const std::optional<LowAddrInfo> low_addr_info{TrackLowAddress(&inst)};
    if (!low_addr_info) {
        // Failed to track the low address, use NVN fallbacks
        return false;
    }
    // First try to find storage buffers in the NVN address
    const IR::U32 low_addr{low_addr_info->value};
    const std::optional<StorageBufferAddr> paired_pointer{
        TrackPointerAddend(*low_addr_info, info.iterator_words, conditions)};
    std::optional<StorageBufferAddr> storage_buffer{paired_pointer};
    const bool native = env.IsNativeProgram();
    if (native && storage_buffer &&
        !env.IsCbufStorageBufferPointer(storage_buffer->index, storage_buffer->offset)) {
        storage_buffer.reset();
    }
    // A paired constant buffer pointer can be only one input of a select or phi whose other input
    // is a pointer loaded from that buffer (Ocarina of Time's foliage unit placement records are
    // read from c0[12] or from *c0[12]). Rebasing on the constant buffer pointer then reads the
    // loaded path's records out of bounds, as zeros, so look for loaded pointers here as well.
    const bool paired_found = storage_buffer.has_value();
    if (native && !low_addr.IsImmediate()) {
        const IR::Inst* const start{low_addr.InstRecursive()};
        auto it{info.native_candidates.find(start)};
        if (it == info.native_candidates.end()) {
            bool dynamic{};
            it = info.native_candidates
                     .emplace(start, NativePointerCandidates(low_addr, env, dynamic, info))
                     .first;
            if (dynamic) {
                info.native_dynamic.insert(start);
            }
        }
        const auto& candidates{it->second};
        const bool has_loaded_pointer{std::ranges::any_of(
            candidates, [](const StorageBufferAddr& c) { return c.indirect != ~0U; })};
        if (paired_found &&
            (!has_loaded_pointer || !info.bind_competing_loaded || !IsRuntimeRoutable(inst))) {
            // No pointer loaded from memory competes with the paired one, or the access cannot
            // stay global: keep the pair.
        } else if (candidates.size() > 1 || paired_found ||
                   (!candidates.empty() && info.native_dynamic.contains(start))) {
            // Bind every candidate and keep the access global; the backend picks the buffer at
            // run time. Rebasing on one of them drops the other path's accesses.
            bool accepted{};
            if (paired_found) {
                // The paired pointer is the other input of the select or phi; bind it too.
                info.competing_loaded_bound = true;
                info.set.insert(*storage_buffer);
                if (IsGlobalMemoryWrite(inst)) {
                    info.writes.insert(*storage_buffer);
                }
            }
            for (const StorageBufferAddr& candidate : candidates) {
                if (std::ranges::find(info.table_candidates, candidate) !=
                    info.table_candidates.end()) {
                    const StorageBufferAddr holder{.index = candidate.index,
                                                   .offset = candidate.offset};
                    const auto takes_binding = [&info](const StorageBufferAddr& buffer) {
                        return info.set.find(buffer) == info.set.end() &&
                               !info.counted.contains(buffer);
                    };
                    const size_t added{
                        (takes_binding(candidate) ? 1U : 0U) +
                        (candidate.indirect != ~0U && takes_binding(holder) ? 1U : 0U)};
                    if (info.table_used + added > info.table_budget) {
                        continue;
                    }
                    info.table_used += added;
                }
                info.set.insert(candidate);
                if (IsGlobalMemoryWrite(inst)) {
                    info.writes.insert(candidate);
                    info.unbounded_writes.insert(candidate);
                }
                if (candidate.indirect != ~0U) {
                    // The backend reads the pointer through the buffer that holds it.
                    info.set.insert(StorageBufferAddr{.index = candidate.index,
                                                      .offset = candidate.offset});
                }
                accepted = true;
            }
            if (accepted) {
                info.native_runtime_bounds = true;
                return true;
            }
            // Only table entries, and no free binding for them: as if none had been found.
        } else if (candidates.size() == 1 && !paired_found) {
            // The one pointer the address is built from. The NVN search below only takes 16-byte
            // aligned slots, so for a pointer at an 8-byte slot it can return a dimension or stride
            // the address arithmetic reads instead, and the access then lands in the wrong buffer.
            storage_buffer = candidates.front();
        }
    }
    if (native && !storage_buffer) {
        // A pointer that is an atomic's address keeps its buffer while unmapped, for the atomic
        // and every access based on it: Bananza's append counters and the records stored after
        // them were unmapped when their pipelines were first built. The binding reads the pointer
        // again for every dispatch.
        if (!paired_pointer || !info.atomic_pointers.contains(*paired_pointer)) {
            return false;
        }
        storage_buffer = paired_pointer;
    }
    const auto cached_track = [&](const Bias* bias, auto& cache, auto& exhausted) {
        if (low_addr.IsImmediate()) {
            return Track(low_addr, bias, exhausted);
        }
        const IR::Inst* const start{low_addr.InstRecursive()};
        if (const auto it = cache.find(start); it != cache.end()) {
            return it->second;
        }
        const std::optional<StorageBufferAddr> result{Track(low_addr, bias, exhausted)};
        cache.emplace(start, result);
        return result;
    };
    if (!storage_buffer) {
        storage_buffer = TrackPointerAddend(*low_addr_info, info.iterator_words, conditions, true);
    }
    if (!storage_buffer) {
        storage_buffer = TrackAddend(low_addr);
    }
    if (!storage_buffer) {
        storage_buffer = cached_track(&nvn_bias, info.biased_tracks, info.biased_exhausted);
    }
    if (!storage_buffer) {
        // If it fails, track without a bias
        storage_buffer = cached_track(nullptr, info.unbiased_tracks, info.unbiased_exhausted);
        if (!storage_buffer) {
            // If that also fails, use NVN fallbacks
            LOG_WARNING(Shader, "Storage buffer failed to track, using global memory fallbacks");
            return false;
        }
        LOG_DEBUG(Shader, "Storage buffer tracked without bias, index {} offset {}",
                    storage_buffer->index, storage_buffer->offset);
    }
    // Collect storage buffer and the instruction
    if (IsGlobalMemoryWrite(inst)) {
        info.writes.insert(*storage_buffer);
    }
    info.set.insert(*storage_buffer);
    info.to_replace.push_back(StorageInst{
        .storage_buffer{*storage_buffer},
        .inst = &inst,
        .block = &block,
    });
    return true;
}

/// Bytes a global load, store or atomic accesses
u32 AccessSize(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
        return 1;
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
        return 2;
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return 8;
    case IR::Opcode::LoadGlobal128:
    case IR::Opcode::WriteGlobal128:
        return 16;
    default:
        return 4;
    }
}

/// Whether a global memory instruction reads memory: a load or an atomic, not a plain store
bool IsGlobalMemoryRead(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return false;
    default:
        return true;
    }
}

/// FRITCHII_EXTENT_SHARED: whether a compute program keeps one private shared memory slot per
/// invocation, and which stores a load of a slot word reads back. A program can park values it
/// computed (a record offset made from its ids) in shared memory and load them again later; the
/// store analyses follow such a load to the values stored when
///
///  - every shared write and atomic of the program has an exact address
///      c.x * tid.x + c.y * tid.y + c.z * tid.z + k
///    (tid = gl_LocalInvocationID; ExactLocal: no 32-bit wrap anywhere for any invocation) with
///    the same coefficients c, all non-negative multiples of 4, and those slots never overlap:
///    with each access widened to whole 32-bit words (the backend may write a byte or a half as
///    its word) and `span` the bytes from the lowest to the highest one, the dimensions of more
///    than one invocation sorted by coefficient each step past everything the smaller ones reach:
///      c1 >= span, c2 >= c1 * (w1 - 1) + span, c3 >= c1 * (w1 - 1) + c2 * (w2 - 1) + span
///    and the last byte stays inside the program's shared memory. Then only the invocation itself
///    writes its slot's bytes (no barrier is needed: an invocation sees its own writes in order);
///  - the load has the same coefficients and reads one whole word that only 32-bit writes of
///    exactly that word write, and one of them runs before the load on every path to it.
///
/// The loaded value is then the newest of those stores' values on the invocation's own path, so
/// a bound or congruence of each of the stored values (every dynamic instance of each) holds for
/// it. Anything else follows nothing.
class SharedSlots {
public:
    explicit SharedSlots(const IR::Program& program_) : program{program_} {}

    struct Forward {
        /// Known when the load reads back `stores`, else why not
        ExtentUnknownWhy why{};
        boost::container::small_vector<const IR::Inst*, 4> stores;
    };

    /// Whether the program loads a 32-bit shared word (the only load followed)
    bool AnyLoad() {
        if (!scanned) {
            Scan();
        }
        return any_load;
    }

    /// The stores a LoadSharedU32 reads back, or why they are not known
    Forward Resolve(const IR::Inst& load) {
        if (!scanned) {
            Scan();
        }
        Forward result;
        if (proof.kind != ExtentUnknown::Known) {
            result.why = proof;
            return result;
        }
        const auto fail = [&](ExtentUnknown kind, const IR::Inst& at) {
            result.why = ExtentUnknownWhy{.kind = kind,
                                          .opcode = static_cast<u16>(at.GetOpcode())};
            result.stores.clear();
            return result;
        };
        const std::optional<Affine> address{ExactLocal(load.Arg(0), 0)};
        if (!address || address->c != coefficients) {
            // Not its own slot (or not exact): another invocation's bytes need a barrier
            return fail(ExtentUnknown::SharedNotPrivate, load);
        }
        const s64 begin{address->k};
        if (begin % 4 != 0) {
            return fail(ExtentUnknown::SharedPartialWrite, load);
        }
        for (const Access& access : writes) {
            if (access.word_end <= begin || begin + 4 <= access.word_begin) {
                continue;
            }
            if (access.inst->GetOpcode() != IR::Opcode::WriteSharedU32 || access.k != begin) {
                return fail(ExtentUnknown::SharedPartialWrite, *access.inst);
            }
            result.stores.push_back(access.inst);
        }
        if (result.stores.empty()) {
            return fail(ExtentUnknown::SharedNotDominated, load);
        }
        if (!dominance) {
            dominance.emplace(program);
        }
        const bool dominated{std::ranges::any_of(result.stores, [&](const IR::Inst* store) {
            return dominance->Dominates(store, &load);
        })};
        if (!dominated) {
            return fail(ExtentUnknown::SharedNotDominated, load);
        }
        result.why = ExtentUnknownWhy{.kind = ExtentUnknown::Known};
        return result;
    }

private:
    static constexpr s64 LIMIT = s64{1} << 40;
    static constexpr s64 U32_END = s64{1} << 32;
    static constexpr u32 MAX_DEPTH = 64;
    static constexpr u32 MAX_WORKGROUP = 0x10000;

    /// c . tid + k, exactly the unsigned 32-bit value for every invocation of the workgroup (a
    /// constant: k is its value as a signed 32-bit number)
    struct Affine {
        std::array<s64, 3> c{};
        s64 k{};

        bool IsConstant() const {
            return c[0] == 0 && c[1] == 0 && c[2] == 0;
        }
    };

    struct Access {
        const IR::Inst* inst{};
        s64 k{};
        s64 word_begin{};
        s64 word_end{};
    };

    static s64 Signed32(s64 value) {
        return static_cast<s64>(static_cast<s32>(static_cast<u32>(static_cast<u64>(value))));
    }

    /// Bytes a shared write or atomic writes (0: not one)
    static u32 WriteSize(IR::Opcode opcode) {
        switch (opcode) {
        case IR::Opcode::WriteSharedU8:
            return 1;
        case IR::Opcode::WriteSharedU16:
            return 2;
        case IR::Opcode::WriteSharedU32:
            return 4;
        case IR::Opcode::WriteSharedU64:
            return 8;
        case IR::Opcode::WriteSharedU128:
            return 16;
        case IR::Opcode::SharedAtomicExchange64:
        case IR::Opcode::SharedAtomicExchange32x2:
            return 8;
        default:
            break;
        }
        const std::string_view name{IR::NameOf(opcode)};
        if (name.starts_with("SharedAtomic")) {
            // A 64-bit or paired atomic not listed above: as wide as the widest
            return name.find("64") != std::string_view::npos ||
                           name.find("32x2") != std::string_view::npos
                       ? 8U
                       : 4U;
        }
        if (name.starts_with("WriteShared")) {
            return 16; // a store width not listed above: the widest
        }
        return 0;
    }

    /// The smallest and largest value over the workgroup's invocations
    std::pair<s64, s64> Range(const Affine& value) const {
        s64 low{value.k};
        s64 high{value.k};
        for (size_t i = 0; i < 3; ++i) {
            // Scan refuses workgroups past MAX_WORKGROUP (and |c| <= LIMIT): no overflow
            const u32 size{(std::min)(program.workgroup_size[i], MAX_WORKGROUP)};
            const s64 span{value.c[i] * static_cast<s64>(size > 0 ? size - 1 : 0)};
            (span < 0 ? low : high) += span;
        }
        return {low, high};
    }

    /// A non-constant result that is exact: in [0, 2^32) for every invocation, small terms
    std::optional<Affine> Exact(Affine value) const {
        if (value.IsConstant()) {
            value.k = Signed32(value.k);
            return value;
        }
        if (value.k > LIMIT || value.k < -LIMIT) {
            return std::nullopt;
        }
        for (const s64 c : value.c) {
            if (c > LIMIT || c < -LIMIT) {
                return std::nullopt;
            }
        }
        const auto [low, high] = Range(value);
        if (low < 0 || high >= U32_END) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<Affine> Scale(const Affine& value, s64 factor) const {
        if (value.IsConstant()) {
            // The 32-bit product (both are 32-bit numbers)
            const u32 product{static_cast<u32>(static_cast<u64>(value.k)) *
                              static_cast<u32>(static_cast<u64>(factor))};
            return Affine{.k = Signed32(static_cast<s64>(product))};
        }
        const auto magnitude{[](s64 x) { return x < 0 ? -x : x; }};
        const s64 scale{magnitude(factor)};
        const auto fits{[&](s64 x) { return scale == 0 || magnitude(x) <= LIMIT / scale; }};
        if (!fits(value.k) || !std::ranges::all_of(value.c, fits)) {
            return std::nullopt;
        }
        Affine result{.k = value.k * factor};
        for (size_t i = 0; i < 3; ++i) {
            result.c[i] = value.c[i] * factor;
        }
        return Exact(result);
    }

    std::optional<Affine> ExactLocal(const IR::Value& value, u32 depth) {
        if (value.IsImmediate()) {
            if (value.Type() != IR::Type::U32) {
                return std::nullopt;
            }
            return Affine{.k = static_cast<s64>(static_cast<s32>(value.U32()))};
        }
        if (depth > MAX_DEPTH) {
            return std::nullopt;
        }
        const IR::Inst* const inst{value.InstRecursive()};
        if (const auto it{memo.find(inst)}; it != memo.end()) {
            return it->second;
        }
        const std::optional<Affine> result{ExactLocalInst(*inst, depth)};
        memo.emplace(inst, result);
        return result;
    }

    std::optional<Affine> ExactLocalInst(const IR::Inst& inst, u32 depth) {
        const auto arg{[&](size_t index) { return ExactLocal(inst.Arg(index), depth + 1); }};
        const auto imm{[&](size_t index) -> std::optional<u32> {
            if (!inst.Arg(index).IsImmediate()) {
                return std::nullopt;
            }
            return inst.Arg(index).U32();
        }};
        switch (inst.GetOpcode()) {
        case IR::Opcode::CompositeExtractU32x3: {
            if (inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() ||
                inst.Arg(1).U32() > 2 ||
                inst.Arg(0).InstRecursive()->GetOpcode() != IR::Opcode::LocalInvocationId) {
                return std::nullopt;
            }
            Affine result;
            const u32 element{inst.Arg(1).U32()};
            // A dimension of one invocation: its id is 0
            result.c[element] = program.workgroup_size[element] > 1 ? 1 : 0;
            return result;
        }
        case IR::Opcode::IAdd32:
        case IR::Opcode::ISub32: {
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            const s64 sign{inst.GetOpcode() == IR::Opcode::IAdd32 ? 1 : -1};
            Affine result{.k = lhs->k + sign * rhs->k};
            for (size_t i = 0; i < 3; ++i) {
                result.c[i] = lhs->c[i] + sign * rhs->c[i];
            }
            return Exact(result);
        }
        case IR::Opcode::IMul32: {
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            if (rhs->IsConstant()) {
                return Scale(*lhs, rhs->k);
            }
            if (lhs->IsConstant()) {
                return Scale(*rhs, lhs->k);
            }
            return std::nullopt;
        }
        case IR::Opcode::ShiftLeftLogical32: {
            const std::optional<u32> shift{imm(1)};
            const auto lhs{shift && *shift < 32 ? arg(0) : std::nullopt};
            return lhs ? Scale(*lhs, s64{1} << *shift) : std::nullopt;
        }
        case IR::Opcode::BitFieldUExtract: {
            const std::optional<u32> offset{imm(1)};
            const std::optional<u32> count{imm(2)};
            const auto source{offset && count && *offset < 32 && *count != 0 && *count <= 32
                                  ? arg(0)
                                  : std::nullopt};
            if (!source) {
                return std::nullopt;
            }
            if (source->IsConstant()) {
                const u64 bits{static_cast<u32>(static_cast<u64>(source->k))};
                const u64 field{(bits >> *offset) & ((u64{1} << *count) - 1)};
                return Exact(Affine{.k = static_cast<s64>(field)});
            }
            const s64 high{Range(*source).second};
            if (*offset == 0 && high < (s64{1} << *count)) {
                return source; // the whole value
            }
            if ((high >> *offset) == 0) {
                return Affine{}; // above every bit it has (a lowered multiplication's high half)
            }
            return std::nullopt;
        }
        case IR::Opcode::BitFieldInsert: {
            const std::optional<u32> offset{imm(2)};
            const std::optional<u32> count{imm(3)};
            if (!offset || !count || *count == 0 || *offset + *count > 32) {
                return std::nullopt;
            }
            const auto base{arg(0)};
            const auto insert{base ? arg(1) : std::nullopt};
            if (!base || !insert) {
                return std::nullopt;
            }
            // A constant as its unsigned value (Exact keeps constants signed)
            const auto as_unsigned{[](Affine v) {
                if (v.IsConstant()) {
                    v.k = static_cast<s64>(static_cast<u32>(static_cast<u64>(v.k)));
                }
                return v;
            }};
            const Affine low{as_unsigned(*base)};
            const Affine field{as_unsigned(*insert)};
            const auto fits{[&](const Affine& v, u32 bits) {
                const auto [lowest, highest] = Range(v);
                return lowest >= 0 && highest < (s64{1} << bits);
            }};
            // The base below the field and the inserted value inside it: a sum
            if (!fits(low, *offset) || !fits(field, *count)) {
                return std::nullopt;
            }
            Affine result{.c = low.c, .k = low.k + (field.k << *offset)};
            for (size_t i = 0; i < 3; ++i) {
                result.c[i] += field.c[i] << *offset;
            }
            return Exact(result);
        }
        case IR::Opcode::BitwiseAnd32: {
            // A mask of low bits over a value inside them
            for (const size_t side : {size_t{1}, size_t{0}}) {
                const std::optional<u32> mask{imm(side)};
                if (!mask || (u64{*mask} & (u64{*mask} + 1)) != 0) {
                    continue;
                }
                const auto other{arg(1 - side)};
                if (other && !other->IsConstant() && Range(*other).second <= s64{*mask}) {
                    return other;
                }
            }
            return std::nullopt;
        }
        default:
            return std::nullopt;
        }
    }

    /// The proof over every shared write and atomic of the program (once)
    void Scan() {
        scanned = true;
        proof = ExtentUnknownWhy{.kind = ExtentUnknown::Known};
        if (std::ranges::any_of(program.workgroup_size,
                                [](u32 size) { return size == 0 || size > MAX_WORKGROUP; })) {
            proof = ExtentUnknownWhy{.kind = ExtentUnknown::SharedNotPrivate};
        }
        bool have_coefficients{};
        s64 lowest{};
        s64 highest{};
        for (const IR::Block* const block : program.post_order_blocks) {
            for (const IR::Inst& inst : block->Instructions()) {
                const IR::Opcode opcode{inst.GetOpcode()};
                any_load |= opcode == IR::Opcode::LoadSharedU32;
                const u32 size{WriteSize(opcode)};
                if (size == 0 || proof.kind != ExtentUnknown::Known) {
                    continue;
                }
                const auto refuse{[&] {
                    proof = ExtentUnknownWhy{.kind = ExtentUnknown::SharedNotPrivate,
                                             .opcode = static_cast<u16>(opcode)};
                }};
                const std::optional<Affine> address{ExactLocal(inst.Arg(0), 0)};
                if (!address || address->k < 0 ||
                    std::ranges::any_of(address->c, [](s64 c) { return c < 0 || c % 4 != 0; })) {
                    refuse();
                    continue;
                }
                if (!have_coefficients) {
                    coefficients = address->c;
                    have_coefficients = true;
                } else if (address->c != coefficients) {
                    refuse();
                    continue;
                }
                const Access access{
                    .inst = &inst,
                    .k = address->k,
                    .word_begin = address->k & ~s64{3},
                    .word_end = (address->k + size + 3) & ~s64{3},
                };
                lowest = writes.empty() ? access.word_begin : (std::min)(lowest, access.word_begin);
                highest = writes.empty() ? access.word_end : (std::max)(highest, access.word_end);
                writes.push_back(access);
            }
        }
        if (proof.kind != ExtentUnknown::Known || writes.empty()) {
            return;
        }
        // The slots one invocation apart in each dimension never overlap (mixed radix), and the
        // last invocation's slot ends inside the shared memory
        const s64 span{highest - lowest};
        std::array<size_t, 3> dims{0, 1, 2};
        std::ranges::sort(dims, [&](size_t a, size_t b) { return coefficients[a] < coefficients[b]; });
        s64 reach{};
        for (const size_t dim : dims) {
            const s64 size{static_cast<s64>(program.workgroup_size[dim])};
            if (size <= 1) {
                continue;
            }
            if (coefficients[dim] < reach + span) {
                proof = ExtentUnknownWhy{.kind = ExtentUnknown::SharedNotPrivate};
                return;
            }
            reach += coefficients[dim] * (size - 1);
        }
        if (highest + reach > static_cast<s64>(program.shared_memory_size)) {
            proof = ExtentUnknownWhy{.kind = ExtentUnknown::SharedNotPrivate};
        }
    }

    const IR::Program& program;
    bool scanned{};
    bool any_load{};
    ExtentUnknownWhy proof{};
    std::array<s64, 3> coefficients{};
    std::vector<Access> writes;
    std::unordered_map<const IR::Inst*, std::optional<Affine>> memo;
    std::optional<Dominance> dominance;
};

/// FRITCHII_EXTENT_GUARD: the branch conditions every path to a store passes, as path facts. An
/// append list's store sits behind `if (index < capacity)` (or an early exit when it is not), so
/// its index is bounded on the store's path although nothing bounds the index itself (an atomic's
/// result). The facts come from the structured control flow's conditional edges (If: header to
/// body when the condition holds, to the merge when not; Break and Repeat alike), each checked
/// against the header's two successors. An edge H -> T gives its condition to a block S when T
/// dominates S and every other predecessor of T is dominated by T (a loop's back edge): then
/// every path to S entered T last from outside through H -> T, so the condition held when H
/// evaluated it. The condition splits into atoms (any boolean instruction, with its value)
/// through ConditionRef, LogicalNot, LogicalAnd (true) and LogicalOr (false).
///
/// Two uses, both exact for an invocation that runs the store:
///  - a bound: `x < K` (unsigned) bounds x by K - 1 when x is defined in a block of no cycle (it
///    has one value in the invocation, the one the branch compared) and K is bounded (an
///    immediate, a constant buffer word, ...). Signed compares and `x + c < K` with c > 0 (it may
///    wrap) refuse; `x - d < K` bounds x by K - 1 + d (a wrap makes the compare fail).
///  - an edge never taken: a phi's incoming edge needs an atom with the other value than a path
///    fact of the store. When that atom is invariant (defined in a block of no cycle, or a compare
///    of such values, immediates and constant buffer words), it has one value in the invocation,
///    so the store's path and that edge never meet in one invocation: the incoming is dropped
///    (predicated moves into the store's address, `@!P mov; @!P st`).
class GuardFacts {
public:
    /// A boolean instruction and the value it has on every path through an edge
    struct Atom {
        const IR::Inst* inst{};
        bool value{};
    };

    explicit GuardFacts(const IR::Program& program) : dominance{program} {
        const IR::AbstractSyntaxList& list{program.syntax_list};
        for (size_t index = 0; index < list.size(); ++index) {
            const IR::AbstractSyntaxNode& node{list[index]};
            const IR::Block* const header{
                index > 0 && list[index - 1].type == IR::AbstractSyntaxNode::Type::Block
                    ? list[index - 1].data.block
                    : nullptr};
            switch (node.type) {
            case IR::AbstractSyntaxNode::Type::If:
                AddBranch(header, node.data.if_node.body, node.data.if_node.merge,
                          node.data.if_node.cond);
                break;
            case IR::AbstractSyntaxNode::Type::Break:
                AddBranch(header, node.data.break_node.merge, node.data.break_node.skip,
                          node.data.break_node.cond);
                break;
            case IR::AbstractSyntaxNode::Type::Repeat:
                AddBranch(header, node.data.repeat.loop_header, node.data.repeat.merge,
                          node.data.repeat.cond);
                break;
            default:
                break;
            }
        }
        // Census: the values some branch compares (a guard that may not dominate a store)
        std::vector<Atom> atoms;
        for (const Edge& edge : edges) {
            atoms.clear();
            Implied(edge.cond, edge.value, atoms, 0);
            for (const Atom& atom : atoms) {
                for (size_t arg = 0; arg < 2 && arg < atom.inst->NumArgs(); ++arg) {
                    const IR::Value operand{atom.inst->Arg(arg).Resolve()};
                    if (operand.IsImmediate()) {
                        continue;
                    }
                    const IR::Inst* const compared_inst{operand.InstRecursive()};
                    compared.insert(compared_inst);
                    const IR::Opcode opcode{compared_inst->GetOpcode()};
                    if (opcode == IR::Opcode::IAdd32 || opcode == IR::Opcode::ISub32) {
                        for (size_t term = 0; term < 2; ++term) {
                            const IR::Value inner{compared_inst->Arg(term).Resolve()};
                            if (!inner.IsImmediate()) {
                                compared.insert(inner.InstRecursive());
                            }
                        }
                    }
                }
            }
        }
    }

    /// The atoms every path to the store passed with their value
    const std::vector<Atom>& FactsAt(const IR::Inst& store) {
        static const std::vector<Atom> none;
        const IR::Block* const block{dominance.BlockOf(&store)};
        if (block == nullptr) {
            return none;
        }
        const auto [it, inserted]{facts.try_emplace(block)};
        if (inserted) {
            for (const Edge& edge : edges) {
                if (EdgeDominates(edge, block)) {
                    Implied(edge.cond, edge.value, it->second, 0);
                }
            }
        }
        return it->second;
    }

    /// Whether the edge from -> to is never taken in an invocation that holds `path` (an atom it
    /// needs is invariant and has the other value there)
    bool Excluded(const IR::Block* from, const IR::Block* to, const std::vector<Atom>& path) {
        if (path.empty()) {
            return false;
        }
        std::vector<Atom> needed;
        for (const Edge& edge : edges) {
            if (edge.from != from || edge.to != to) {
                continue;
            }
            needed.clear();
            Implied(edge.cond, edge.value, needed, 0);
            for (const Atom& atom : needed) {
                for (const Atom& fact : path) {
                    if (fact.inst == atom.inst && fact.value != atom.value &&
                        Invariant(*atom.inst)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /// Whether an instruction may run more than once in an invocation (its block is on a cycle,
    /// or not known)
    bool InLoop(const IR::Inst& inst) {
        const IR::Block* const block{dominance.BlockOf(&inst)};
        return block == nullptr || InCycle(block);
    }

    /// Whether some branch of the program compares the value (FRITCHII_EXTENT_CENSUS)
    bool Compared(const IR::Inst& inst) const {
        return compared.contains(&inst);
    }

    const IR::Block* BlockOf(const IR::Inst& inst) const {
        return dominance.BlockOf(&inst);
    }

private:
    struct Edge {
        const IR::Block* from{};
        const IR::Block* to{};
        IR::U1 cond;
        bool value{};
    };

    /// A conditional branch of `header`: to on_true when cond holds, else to on_false
    void AddBranch(const IR::Block* header, const IR::Block* on_true, const IR::Block* on_false,
                   const IR::U1& cond) {
        if (header == nullptr || on_true == nullptr || on_false == nullptr || on_true == on_false ||
            cond.IsImmediate()) {
            return;
        }
        const std::span<IR::Block* const> successors{header->ImmSuccessors()};
        if (successors.size() != 2 || successors[0] != on_true || successors[1] != on_false) {
            // Not the shape the structured control flow builds: no facts from it
            return;
        }
        edges.push_back(Edge{.from = header, .to = on_true, .cond = cond, .value = true});
        edges.push_back(Edge{.from = header, .to = on_false, .cond = cond, .value = false});
    }

    /// The atoms `cond` == `value` implies
    static void Implied(const IR::Value& cond, bool value, std::vector<Atom>& out, u32 depth) {
        const IR::Value resolved{cond.Resolve()};
        if (resolved.IsImmediate() || depth > 16) {
            return;
        }
        const IR::Inst* const inst{resolved.InstRecursive()};
        switch (inst->GetOpcode()) {
        case IR::Opcode::ConditionRef:
            Implied(inst->Arg(0), value, out, depth + 1);
            return;
        case IR::Opcode::LogicalNot:
            Implied(inst->Arg(0), !value, out, depth + 1);
            return;
        case IR::Opcode::LogicalAnd:
            if (value) {
                Implied(inst->Arg(0), true, out, depth + 1);
                Implied(inst->Arg(1), true, out, depth + 1);
            }
            return;
        case IR::Opcode::LogicalOr:
            if (!value) {
                Implied(inst->Arg(0), false, out, depth + 1);
                Implied(inst->Arg(1), false, out, depth + 1);
            }
            return;
        default:
            out.push_back(Atom{.inst = inst, .value = value});
            return;
        }
    }

    /// Whether every path to `block` goes through the edge last when it enters edge.to
    bool EdgeDominates(const Edge& edge, const IR::Block* block) const {
        if (!dominance.BlockDominates(edge.to, block)) {
            return false;
        }
        for (const IR::Block* const predecessor : edge.to->ImmPredecessors()) {
            if (predecessor != edge.from && !dominance.BlockDominates(edge.to, predecessor)) {
                return false;
            }
        }
        return true;
    }

    /// Whether a path leads from the block back to itself
    bool InCycle(const IR::Block* block) {
        if (const auto it{cyclic.find(block)}; it != cyclic.end()) {
            return it->second;
        }
        bool found{};
        std::unordered_set<const IR::Block*> seen;
        std::vector<const IR::Block*> stack(block->ImmSuccessors().begin(),
                                            block->ImmSuccessors().end());
        while (!stack.empty() && !found) {
            const IR::Block* const current{stack.back()};
            stack.pop_back();
            if (current == block) {
                found = true;
                break;
            }
            if (!seen.insert(current).second) {
                continue;
            }
            for (const IR::Block* const next : current->ImmSuccessors()) {
                stack.push_back(next);
            }
        }
        cyclic.emplace(block, found);
        return found;
    }

    /// Whether a boolean has one value in an invocation
    bool Invariant(const IR::Inst& atom) {
        if (!InLoop(atom)) {
            return true;
        }
        switch (atom.GetOpcode()) {
        case IR::Opcode::SLessThan:
        case IR::Opcode::ULessThan:
        case IR::Opcode::IEqual:
        case IR::Opcode::SLessThanEqual:
        case IR::Opcode::ULessThanEqual:
        case IR::Opcode::SGreaterThan:
        case IR::Opcode::UGreaterThan:
        case IR::Opcode::INotEqual:
        case IR::Opcode::SGreaterThanEqual:
        case IR::Opcode::UGreaterThanEqual:
            break;
        default:
            return false;
        }
        // A compare in a loop of values that each have one value
        for (size_t arg = 0; arg < atom.NumArgs(); ++arg) {
            const IR::Value operand{atom.Arg(arg).Resolve()};
            if (operand.IsImmediate()) {
                continue;
            }
            const IR::Inst* const inst{operand.InstRecursive()};
            if (CbufWord(inst)) {
                continue;
            }
            if (InLoop(*inst)) {
                return false;
            }
        }
        return true;
    }

    Dominance dominance;
    std::vector<Edge> edges;
    std::unordered_set<const IR::Inst*> compared;
    std::unordered_map<const IR::Block*, std::vector<Atom>> facts;
    std::unordered_map<const IR::Block*, bool> cyclic;
};

/// An upper bound of an unsigned 32-bit value for every invocation of a dispatch:
/// c + sum(coef[i] * id[i]) over gl_LocalInvocationID.xyz (0-2) and gl_WorkGroupID.xyz (3-5),
/// plus `base` times the storage buffer's pointer word. All coefficients are non-negative, so the
/// bound is largest at the largest ids. FRITCHII_EXTENT_CBUF's analysis (terms mode) also adds
/// constant buffer words times a non-negative coefficient (their values are read at bind time).
struct IdBound {
    struct Term {
        u32 index{};
        u32 offset{};
        u64 coef{};
    };
    s64 c{};
    std::array<u64, 6> coef{};
    u32 base{};
    u32 num_terms{};
    std::array<Term, MaxExtentTerms> terms{};

    bool IsConstant() const {
        return base == 0 && num_terms == 0 &&
               std::ranges::all_of(coef, [](u64 value) { return value == 0; });
    }
    bool HasGroupTerms() const {
        return coef[3] != 0 || coef[4] != 0 || coef[5] != 0;
    }
    bool HasTerms() const {
        return num_terms != 0;
    }
};

/// Proves where a program's accesses to one storage buffer (its stores, or its loads and atomics)
/// can land relative to its pointer word. Anything not provably affine in the invocation and
/// workgroup ids with constant non-negative strides (a value loaded from memory or read from
/// another constant buffer word, a loop-carried index, a signed or 64-bit computation) leaves the
/// extent unknown, and the first such exit is kept (FRITCHII_EXTENT_CENSUS, Why()). In terms mode
/// (FRITCHII_EXTENT_CBUF) another constant buffer word read at an immediate offset is a term of
/// the bound instead. With `shared` (FRITCHII_EXTENT_SHARED) a 32-bit load from a private shared
/// memory slot is bounded by the values its invocation stored there (SharedSlots).
///
/// With `guards` (FRITCHII_EXTENT_GUARD, terms mode) also, each exact for an invocation that runs
/// the store being analysed:
///  - a value no rule bounds is bounded by the path facts of the store (GuardFacts: the store
///    only runs when `x < K` held), and a phi drops the incoming edges such an invocation never
///    takes;
///  - a subgroup shuffle is bounded by its operand's bound at the workgroup's largest local ids
///    (any lane of the workgroup may be the source; group ids and constant buffer words are the
///    same in all of them). The operand is evaluated without the path facts: the source lane may
///    not run the store. A shuffle from an inactive lane is undefined, as in the guest program
///    (an append list broadcasts from the lane that did the atomic);
///  - a bit count of a 32-bit value is at most 32;
///  - word - v (also v negated plus word) is at most the constant buffer word when the word is at
///    least v's largest value, a floor the dispatch checks at bind time (a clamp
///    min(old, capacity - count)).
class StoreExtentAnalysis {
public:
    explicit StoreExtentAnalysis(StorageBufferAddr buffer_, const std::array<u32, 3>& workgroup_,
                                 bool terms_ = false, SharedSlots* shared_ = nullptr,
                                 GuardFacts* guards_ = nullptr)
        : buffer{buffer_}, workgroup{workgroup_}, terms_mode{terms_}, shared{shared_},
          guards{guards_} {}

    /// FRITCHII_EXTENT_SHARED: whether a load from shared memory was followed to its stores
    bool Forwarded() const {
        return forwarded;
    }

    /// Extent of one access, merged into `extent`
    void AddAccess(IR::Inst& inst, StorageWriteExtent& extent) {
        const std::optional<StorageCbufExtent> store{StoreExtent(inst)};
        if (!store || store->num_terms != 0 || store->num_guard_terms != 0 ||
            store->num_floors != 0) {
            extent = StorageWriteExtent{};
            return;
        }
        MergeStorageWriteExtent(extent, store->ids);
    }

    /// Terms mode: extent of one access with its constant buffer terms, merged into `extent`
    void AddAccess(IR::Inst& inst, StorageCbufExtent& extent) {
        const std::optional<StorageCbufExtent> store{StoreExtent(inst)};
        if (!store) {
            extent = StorageCbufExtent{};
            return;
        }
        MergeStorageCbufExtent(extent, *store);
    }

    /// Why the first access that failed did (Unrecorded while none did)
    const ExtentUnknownWhy& Why() const {
        return why;
    }

private:
    static constexpr s64 CONST_LIMIT = s64{1} << 40;
    static constexpr u64 COEF_LIMIT = u64{1} << 32;
    static constexpr u32 MAX_DEPTH = 256;

    struct Memo {
        std::optional<IdBound> result;
        ExtentUnknownWhy why;
    };

    /// Notes the reason of the current evaluation's first exit
    void Fail(const ExtentUnknownWhy& reason) {
        if (failure.kind == ExtentUnknown::Unrecorded) {
            failure = reason;
        }
    }
    void Fail(ExtentUnknown kind, const IR::Inst* inst = nullptr) {
        Fail(ExtentUnknownWhy{
            .kind = kind,
            .opcode = inst != nullptr ? static_cast<u16>(inst->GetOpcode()) : u16{0},
        });
    }

    std::optional<StorageCbufExtent> StoreExtent(IR::Inst& inst) {
        failure = {};
        std::optional<StorageCbufExtent> result{StoreExtentOf(inst)};
        if (!result && why.kind == ExtentUnknown::Unrecorded) {
            why = failure.kind != ExtentUnknown::Unrecorded
                      ? failure
                      : ExtentUnknownWhy{.kind = ExtentUnknown::OtherOp,
                                         .opcode = static_cast<u16>(inst.GetOpcode())};
        }
        return result;
    }

    std::optional<StorageCbufExtent> StoreExtentOf(IR::Inst& inst) {
        const std::optional<LowAddrInfo> addr{TrackLowAddress(&inst)};
        if (!addr) {
            Fail(ExtentUnknown::AddressNotTracked, &inst);
            return std::nullopt;
        }
        if (guards != nullptr) {
            // FRITCHII_EXTENT_GUARD: this store's path facts; results evaluated under another
            // store's facts may not hold here
            const std::vector<GuardFacts::Atom>* const now{&guards->FactsAt(inst)};
            if (now != path_facts) {
                memo.clear();
                path_facts = now;
            }
        }
        const std::optional<IdBound> low{Evaluate(addr->value, 0)};
        if (!low) {
            return std::nullopt;
        }
        if (low->base != 1) {
            Fail(ExtentUnknown::NoPointerTerm, &inst);
            return std::nullopt;
        }
        if (guard_failed) {
            Fail(ExtentUnknown::GuardFailed, &inst);
            return std::nullopt;
        }
        const s64 end{low->c + addr->imm_offset + AccessSize(inst.GetOpcode())};
        if (end > s64{0xffffffff}) {
            Fail(ExtentUnknown::TooLarge, &inst);
            return std::nullopt;
        }
        StorageCbufExtent result{};
        if (floors.size() > MaxExtentFloors) {
            Fail(ExtentUnknown::TooLarge, &inst);
            return std::nullopt;
        }
        for (const StorageExtentFloor& floor : floors) {
            result.floors[result.num_floors++] = floor;
        }
        result.ids = StorageWriteExtent{
            .bounded = true,
            .const_end = static_cast<u32>((std::max)(end, s64{0})),
        };
        for (size_t i = 0; i < 3; ++i) {
            if (low->coef[i] >= COEF_LIMIT || low->coef[i + 3] >= COEF_LIMIT) {
                Fail(ExtentUnknown::TooLarge, &inst);
                return std::nullopt;
            }
            result.ids.local_stride[i] = static_cast<u32>(low->coef[i]);
            result.ids.group_stride[i] = static_cast<u32>(low->coef[i + 3]);
        }
        if (!CopyTerms(*low, result.terms, result.num_terms)) {
            Fail(ExtentUnknown::TooLarge, &inst);
            return std::nullopt;
        }
        if (guard) {
            if (guard->c > s64{0xffff}) {
                Fail(ExtentUnknown::GuardFailed, &inst);
                return std::nullopt;
            }
            result.ids.guarded = true;
            result.ids.guard_const = static_cast<u32>((std::max)(guard->c, s64{0}));
            for (size_t i = 0; i < 3; ++i) {
                if (guard->coef[i] > 0xffff || guard->coef[i + 3] > 0xffff) {
                    Fail(ExtentUnknown::GuardFailed, &inst);
                    return std::nullopt;
                }
                result.ids.guard_local_stride[i] = static_cast<u32>(guard->coef[i]);
                result.ids.guard_group_stride[i] = static_cast<u32>(guard->coef[i + 3]);
            }
            if (!CopyTerms(*guard, result.guard_terms, result.num_guard_terms)) {
                Fail(ExtentUnknown::GuardFailed, &inst);
                return std::nullopt;
            }
        }
        return result;
    }

    static bool CopyTerms(const IdBound& value, std::array<StorageExtentTerm, MaxExtentTerms>& out,
                          u8& count) {
        count = 0;
        for (u32 i = 0; i < value.num_terms; ++i) {
            if (value.terms[i].coef >= COEF_LIMIT) {
                return false;
            }
            out[count++] = StorageExtentTerm{
                .cbuf_index = value.terms[i].index,
                .cbuf_offset = value.terms[i].offset,
                .coef = static_cast<u32>(value.terms[i].coef),
            };
        }
        return true;
    }

    /// Adds `term` to `into`'s terms: the sum, or with `max` the larger coefficient for a word
    /// already there. False when they do not fit.
    static bool AddTerm(IdBound& into, const IdBound::Term& term, bool max) {
        for (u32 i = 0; i < into.num_terms; ++i) {
            IdBound::Term& existing = into.terms[i];
            if (existing.index == term.index && existing.offset == term.offset) {
                existing.coef =
                    max ? (std::max)(existing.coef, term.coef) : existing.coef + term.coef;
                return true;
            }
        }
        if (into.num_terms >= MaxExtentTerms) {
            return false;
        }
        into.terms[into.num_terms++] = term;
        return true;
    }

    static std::optional<IdBound> Checked(IdBound value) {
        if (value.c > CONST_LIMIT || value.c < -CONST_LIMIT || value.base > 1) {
            return std::nullopt;
        }
        for (const u64 coef : value.coef) {
            if (coef >= COEF_LIMIT) {
                return std::nullopt;
            }
        }
        for (u32 i = 0; i < value.num_terms; ++i) {
            if (value.terms[i].coef >= COEF_LIMIT) {
                return std::nullopt;
            }
        }
        return value;
    }

    static std::optional<IdBound> Constant(s64 value) {
        return IdBound{.c = value};
    }

    static std::optional<IdBound> Sum(const IdBound& lhs, const IdBound& rhs) {
        IdBound result{.c = lhs.c + rhs.c, .base = lhs.base + rhs.base};
        for (size_t i = 0; i < result.coef.size(); ++i) {
            result.coef[i] = lhs.coef[i] + rhs.coef[i];
        }
        result.num_terms = lhs.num_terms;
        result.terms = lhs.terms;
        for (u32 i = 0; i < rhs.num_terms; ++i) {
            if (!AddTerm(result, rhs.terms[i], false)) {
                return std::nullopt;
            }
        }
        return Checked(result);
    }

    /// Component-wise maximum: a bound of either value (ids and words are non-negative)
    static std::optional<IdBound> Max(const IdBound& lhs, const IdBound& rhs) {
        if (lhs.base != rhs.base) {
            return std::nullopt;
        }
        IdBound result{.c = (std::max)(lhs.c, rhs.c), .base = lhs.base};
        for (size_t i = 0; i < result.coef.size(); ++i) {
            result.coef[i] = (std::max)(lhs.coef[i], rhs.coef[i]);
        }
        result.num_terms = lhs.num_terms;
        result.terms = lhs.terms;
        for (u32 i = 0; i < rhs.num_terms; ++i) {
            if (!AddTerm(result, rhs.terms[i], true)) {
                return std::nullopt;
            }
        }
        return result;
    }

    static std::optional<IdBound> Scale(const IdBound& value, u64 factor) {
        if (value.base != 0 || factor >= COEF_LIMIT) {
            return std::nullopt;
        }
        IdBound result{.c = value.c * static_cast<s64>(factor)};
        for (size_t i = 0; i < result.coef.size(); ++i) {
            if (value.coef[i] != 0 && factor > (COEF_LIMIT - 1) / value.coef[i]) {
                return std::nullopt;
            }
            result.coef[i] = value.coef[i] * factor;
        }
        result.num_terms = value.num_terms;
        for (u32 i = 0; i < value.num_terms; ++i) {
            const u64 coef{value.terms[i].coef};
            if (coef != 0 && factor > (COEF_LIMIT - 1) / coef) {
                return std::nullopt;
            }
            result.terms[i] = value.terms[i];
            result.terms[i].coef = coef * factor;
        }
        if (value.c > 0 && factor > static_cast<u64>(CONST_LIMIT / value.c)) {
            return std::nullopt;
        }
        return Checked(result);
    }

    /// value >> shift, rounded up term by term (the sum of the rounded terms bounds the shifted sum)
    static std::optional<IdBound> ShiftDown(const IdBound& value, u32 shift) {
        if (value.base != 0 || value.c < 0 || shift >= 32) {
            return std::nullopt;
        }
        const auto up = [shift](u64 x) { return (x + (u64{1} << shift) - 1) >> shift; };
        IdBound result{.c = static_cast<s64>(up(static_cast<u64>(value.c)))};
        for (size_t i = 0; i < result.coef.size(); ++i) {
            result.coef[i] = up(value.coef[i]);
        }
        result.num_terms = value.num_terms;
        for (u32 i = 0; i < value.num_terms; ++i) {
            result.terms[i] = value.terms[i];
            result.terms[i].coef = up(value.terms[i].coef);
        }
        return result;
    }

    /// The value's bound when only its local-id terms vary (the workgroup size is fixed), or
    /// nullopt when it has workgroup terms, constant buffer words or the pointer
    std::optional<u64> StaticMax(const IdBound& value) const {
        if (value.base != 0 || value.HasGroupTerms() || value.HasTerms() || value.c < 0) {
            return std::nullopt;
        }
        u64 total{static_cast<u64>(value.c)};
        for (size_t i = 0; i < 3; ++i) {
            if (workgroup[i] == 0) {
                return std::nullopt;
            }
            total += value.coef[i] * (workgroup[i] - 1);
        }
        return total;
    }

    /// Requires a value to stay below 2^16 for the bound to hold
    void Guard(const IdBound& value) {
        if (value.base != 0 || value.c < 0) {
            guard_failed = true;
            return;
        }
        if (!guard) {
            guard = value;
            return;
        }
        const std::optional<IdBound> merged{Max(*guard, value)};
        if (!merged) {
            guard_failed = true;
            return;
        }
        guard = *merged;
    }

    std::optional<IdBound> BitFieldExtract(const IR::Inst& inst, u32 depth) {
        if (!inst.Arg(1).IsImmediate() || !inst.Arg(2).IsImmediate()) {
            Fail(ExtentUnknown::OtherOp, &inst);
            return std::nullopt;
        }
        const u32 offset{inst.Arg(1).U32()};
        const u32 count{inst.Arg(2).U32()};
        if (offset >= 32 || count == 0 || count > 32) {
            Fail(ExtentUnknown::OtherOp, &inst);
            return std::nullopt;
        }
        if (inst.Arg(0).IsImmediate()) {
            const u64 field{(u64{inst.Arg(0).U32()} >> offset) & ((u64{1} << count) - 1)};
            return Constant(static_cast<s64>(field));
        }
        const std::optional<IdBound> source{Evaluate(inst.Arg(0), depth + 1)};
        const std::optional<IdBound> field_max{
            count <= 24 ? Constant((s64{1} << count) - 1) : std::nullopt};
        if (!source || source->base != 0 || source->c < 0) {
            return field_max;
        }
        if (const std::optional<u64> static_max{StaticMax(*source)}) {
            if (offset < 64 && (*static_max >> offset) == 0) {
                return Constant(0);
            }
        }
        if (terms_mode && source->HasTerms() && offset < 16 && field_max) {
            // A field of a constant buffer word: its width bounds it without the word
            return field_max;
        }
        if (offset == 0) {
            // A low field is at most the whole value
            return source;
        }
        if (offset >= 16) {
            // The high half of a lowered 32-bit multiplication's operand: zero while the operand
            // stays below 2^16, which the dispatch checks
            Guard(*source);
            return Constant(0);
        }
        return ShiftDown(*source, offset);
    }

    std::optional<IdBound> Evaluate(const IR::Value& value, u32 depth) {
        if (value.IsImmediate()) {
            if (value.Type() != IR::Type::U32) {
                Fail(ExtentUnknown::SignedOr64Bit);
                return std::nullopt;
            }
            return Constant(static_cast<s64>(static_cast<s32>(value.U32())));
        }
        if (depth > MAX_DEPTH) {
            Fail(ExtentUnknown::TooDeep);
            return std::nullopt;
        }
        const IR::Inst* const inst{value.InstRecursive()};
        // FRITCHII_EXTENT_GUARD: results without the path facts hold under any of them
        auto& results{guards != nullptr && facts_off ? memo_no_facts : memo};
        if (const auto it{results.find(inst)}; it != results.end()) {
            if (!it->second.result) {
                Fail(it->second.why);
            }
            return it->second.result;
        }
        if (!active.insert(inst).second) {
            // A cycle: a loop-carried value
            Fail(ExtentUnknown::LoopCarried, inst);
            return std::nullopt;
        }
        // The reason of this instruction's own exit (a failure its operands' evaluation noted
        // but the instruction got past is dropped)
        const ExtentUnknownWhy outer{failure};
        failure = {};
        std::optional<IdBound> result{EvaluateInst(*inst, depth)};
        ExtentUnknownWhy mine{};
        if (!result) {
            mine = failure.kind != ExtentUnknown::Unrecorded ? failure : Classify(*inst);
            if (guards != nullptr && !facts_off) {
                // FRITCHII_EXTENT_GUARD: a branch on the store's path may bound it
                ExtentUnknownWhy refusal{};
                result = PathBound(*inst, depth, refusal);
                if (result) {
                    mine = {};
                } else if (refusal.kind != ExtentUnknown::Unrecorded) {
                    mine = refusal;
                }
            }
        }
        active.erase(inst);
        results.emplace(inst, Memo{result, mine});
        failure = outer;
        if (!result) {
            Fail(mine);
        }
        return result;
    }

    /// The reason an instruction with no other noted reason is not bounded
    static ExtentUnknownWhy Classify(const IR::Inst& inst) {
        const IR::Opcode opcode{inst.GetOpcode()};
        const ExtentUnknownWhy why_of{[&] {
            switch (opcode) {
            case IR::Opcode::IAdd32:
            case IR::Opcode::ShiftLeftLogical32:
            case IR::Opcode::ShiftRightLogical32:
            case IR::Opcode::BitwiseOr32:
            case IR::Opcode::BitwiseXor32:
            case IR::Opcode::UMax32:
            case IR::Opcode::BitFieldInsert:
                // Handled, and failed on its limits (a large constant or stride, or the pointer
                // where it can't be)
                return ExtentUnknownWhy{.kind = ExtentUnknown::TooLarge};
            case IR::Opcode::SelectU32:
            case IR::Opcode::Phi:
                // Max: one side adds the pointer and the other not
                return ExtentUnknownWhy{.kind = ExtentUnknown::NoPointerTerm};
            default:
                break;
            }
            const std::string_view name{IR::NameOf(opcode)};
            const auto has = [&](std::string_view part) {
                return name.find(part) != std::string_view::npos;
            };
            if (name.starts_with("GetCbuf")) {
                return ExtentUnknownWhy{.kind = ExtentUnknown::DynamicCbuf};
            }
            if (has("Atomic")) {
                return ExtentUnknownWhy{.kind = ExtentUnknown::AtomicResult};
            }
            if (name.starts_with("Load") || name.starts_with("Image") || has("Fetch")) {
                return ExtentUnknownWhy{.kind = ExtentUnknown::LoadedFromMemory};
            }
            if (has("64") || has("Arithmetic") || name.starts_with("SConvert") ||
                name.starts_with("SMin") || name.starts_with("SMax") || name.starts_with("SDiv") ||
                name.starts_with("SClamp") || name.starts_with("INeg") ||
                name.starts_with("IAbs") || name.starts_with("BitFieldSExtract") ||
                name.starts_with("ConvertU32") || name.starts_with("ConvertS32") ||
                name.starts_with("UConvert")) {
                return ExtentUnknownWhy{.kind = ExtentUnknown::SignedOr64Bit};
            }
            return ExtentUnknownWhy{.kind = ExtentUnknown::OtherOp};
        }()};
        ExtentUnknownWhy result{why_of};
        result.opcode = static_cast<u16>(opcode);
        return result;
    }

    std::optional<IdBound> EvaluateInst(const IR::Inst& inst, u32 depth) {
        if (const std::optional<StorageBufferAddr> word{CbufWord(&inst)}) {
            if (*word == buffer) {
                return IdBound{.base = 1};
            }
            if (terms_mode) {
                IdBound term{.num_terms = 1};
                term.terms[0] =
                    IdBound::Term{.index = word->index, .offset = word->offset, .coef = 1};
                return term;
            }
            Fail(ExtentUnknownWhy{.kind = ExtentUnknown::OtherCbufWord,
                                  .opcode = static_cast<u16>(inst.GetOpcode()),
                                  .cbuf_index = word->index,
                                  .cbuf_offset = word->offset});
            return std::nullopt;
        }
        const auto arg{[&](size_t index) { return Evaluate(inst.Arg(index), depth + 1); }};
        switch (inst.GetOpcode()) {
        case IR::Opcode::CompositeExtractU32x3: {
            if (inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() || inst.Arg(1).U32() > 2) {
                return std::nullopt;
            }
            const IR::Inst* const vector_inst{inst.Arg(0).InstRecursive()};
            const IR::Opcode vector{vector_inst->GetOpcode()};
            const u32 element{inst.Arg(1).U32()};
            IdBound result;
            if (vector == IR::Opcode::LocalInvocationId) {
                result.coef[element] = 1;
            } else if (vector == IR::Opcode::WorkgroupId) {
                result.coef[3 + element] = 1;
            } else {
                Fail(Classify(*vector_inst));
                return std::nullopt;
            }
            return result;
        }
        case IR::Opcode::IAdd32: {
            if (guards != nullptr) {
                // FRITCHII_EXTENT_GUARD: word + (-v) is word - v (a native IADD with a negated
                // operand)
                for (size_t side = 0; side < 2; ++side) {
                    const IR::Value operand{inst.Arg(side).Resolve()};
                    if (!operand.IsImmediate() &&
                        operand.InstRecursive()->GetOpcode() == IR::Opcode::INeg32) {
                        return Difference(inst, inst.Arg(1 - side),
                                          operand.InstRecursive()->Arg(0), depth);
                    }
                }
            }
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            return lhs && rhs ? Sum(*lhs, *rhs) : std::nullopt;
        }
        case IR::Opcode::ISub32: {
            if (guards != nullptr && !inst.Arg(1).Resolve().IsImmediate()) {
                // FRITCHII_EXTENT_GUARD: a constant bound of the subtrahend (a bit count, a
                // masked value) is not its value; x - v is bounded through the floor instead
                return Difference(inst, inst.Arg(0), inst.Arg(1), depth);
            }
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            // A constant IdBound is an upper bound, not the value: x - (v & 0xff) is at most x, not
            // x - 0xff. Only an immediate subtrahend is its own value (helper DT, 2026-11-19).
            if (!rhs->IsConstant() || !inst.Arg(1).Resolve().IsImmediate()) {
                Fail(ExtentUnknown::VariableProduct, &inst);
                return std::nullopt;
            }
            return Sum(*lhs, IdBound{.c = -rhs->c});
        }
        case IR::Opcode::IMul32: {
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            if (!lhs || !rhs) {
                return std::nullopt;
            }
            if (rhs->IsConstant() && rhs->c >= 0) {
                return Scale(*lhs, static_cast<u64>(rhs->c));
            }
            if (lhs->IsConstant() && lhs->c >= 0) {
                return Scale(*rhs, static_cast<u64>(lhs->c));
            }
            Fail(ExtentUnknown::VariableProduct, &inst);
            return std::nullopt;
        }
        case IR::Opcode::ShiftLeftLogical32: {
            if (!inst.Arg(1).IsImmediate() || inst.Arg(1).U32() >= 32) {
                Fail(ExtentUnknown::VariableProduct, &inst);
                return std::nullopt;
            }
            const auto lhs{arg(0)};
            return lhs ? Scale(*lhs, u64{1} << inst.Arg(1).U32()) : std::nullopt;
        }
        case IR::Opcode::ShiftRightLogical32: {
            if (!inst.Arg(1).IsImmediate()) {
                Fail(ExtentUnknown::VariableProduct, &inst);
                return std::nullopt;
            }
            const auto lhs{arg(0)};
            return lhs ? ShiftDown(*lhs, inst.Arg(1).U32()) : std::nullopt;
        }
        case IR::Opcode::BitwiseAnd32: {
            // x & y <= min(x, y) for unsigned values
            const auto lhs{arg(0)};
            const auto rhs{arg(1)};
            const auto usable{[](const std::optional<IdBound>& bound) {
                return bound && bound->base == 0 && bound->c >= 0;
            }};
            for (const auto* side : {&rhs, &lhs}) {
                if (usable(*side) && (*side)->IsConstant() && (*side)->c <= 0xffffff) {
                    return *side;
                }
            }
            if (terms_mode) {
                // A side without constant buffer words first
                for (const auto* side : {&lhs, &rhs}) {
                    if (usable(*side) && !(*side)->HasTerms()) {
                        return *side;
                    }
                }
            }
            for (const auto* side : {&lhs, &rhs}) {
                if (usable(*side)) {
                    return *side;
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitwiseOr32:
        case IR::Opcode::BitwiseXor32:
        case IR::Opcode::UMax32: {
            // x | y, x ^ y and max(x, y) are at most x + y for unsigned values
            const auto lhs{arg(0)};
            const auto rhs{lhs ? arg(1) : std::nullopt};
            if (!lhs || !rhs || lhs->base != 0 || rhs->base != 0 || lhs->c < 0 || rhs->c < 0) {
                return std::nullopt;
            }
            return Sum(*lhs, *rhs);
        }
        case IR::Opcode::UMin32: {
            const auto lhs{arg(0)};
            const auto rhs{arg(1)};
            if (terms_mode) {
                for (const auto* side : {&rhs, &lhs}) {
                    if (*side && (*side)->base == 0 && (*side)->c >= 0 && !(*side)->HasTerms()) {
                        return *side;
                    }
                }
            }
            for (const auto* side : {&rhs, &lhs}) {
                if (*side && (*side)->base == 0 && (*side)->c >= 0) {
                    return *side;
                }
            }
            return std::nullopt;
        }
        case IR::Opcode::BitFieldUExtract:
            return BitFieldExtract(inst, depth);
        case IR::Opcode::BitFieldInsert: {
            // (base & ~mask) | ((insert << offset) & mask) <= base + (insert << offset)
            if (!inst.Arg(2).IsImmediate() || !inst.Arg(3).IsImmediate() ||
                inst.Arg(2).U32() >= 32) {
                Fail(ExtentUnknown::OtherOp, &inst);
                return std::nullopt;
            }
            const auto base_value{arg(0)};
            const auto insert{base_value ? arg(1) : std::nullopt};
            if (!base_value || !insert || base_value->base != 0 || insert->base != 0 ||
                base_value->c < 0 || insert->c < 0) {
                return std::nullopt;
            }
            const auto shifted{Scale(*insert, u64{1} << inst.Arg(2).U32())};
            return shifted ? Sum(*base_value, *shifted) : std::nullopt;
        }
        case IR::Opcode::SelectU32: {
            const auto lhs{arg(1)};
            const auto rhs{lhs ? arg(2) : std::nullopt};
            return lhs && rhs ? Max(*lhs, *rhs) : std::nullopt;
        }
        case IR::Opcode::Phi: {
            // FRITCHII_EXTENT_GUARD: the incoming edges an invocation running the store never
            // takes are left out
            const IR::Block* const phi_block{
                guards != nullptr && !facts_off && path_facts != nullptr ? guards->BlockOf(inst)
                                                                         : nullptr};
            std::optional<IdBound> result;
            for (size_t index = 0; index < inst.NumArgs(); ++index) {
                if (phi_block != nullptr &&
                    guards->Excluded(inst.PhiBlock(index), phi_block, *path_facts)) {
                    continue;
                }
                const auto incoming{arg(index)};
                if (!incoming) {
                    return std::nullopt;
                }
                result = result ? Max(*result, *incoming) : incoming;
                if (!result) {
                    return std::nullopt;
                }
            }
            return result;
        }
        case IR::Opcode::LoadSharedU32: {
            if (shared == nullptr) {
                return std::nullopt; // loaded from memory
            }
            // The newest of its invocation's stores to the word: at most the largest of them
            const SharedSlots::Forward forward{shared->Resolve(inst)};
            if (forward.why.kind != ExtentUnknown::Known) {
                Fail(forward.why);
                return std::nullopt;
            }
            std::optional<IdBound> result;
            for (const IR::Inst* const store : forward.stores) {
                const auto stored{Evaluate(store->Arg(1), depth + 1)};
                if (!stored) {
                    return std::nullopt;
                }
                result = result ? Max(*result, *stored) : stored;
                if (!result) {
                    return std::nullopt;
                }
            }
            forwarded = true;
            return result;
        }
        case IR::Opcode::ShuffleIndex:
        case IR::Opcode::ShuffleUp:
        case IR::Opcode::ShuffleDown:
        case IR::Opcode::ShuffleButterfly: {
            if (guards == nullptr) {
                return std::nullopt;
            }
            // FRITCHII_EXTENT_GUARD: another lane's value. Its bound at the largest local ids
            // covers every lane of the workgroup; the path facts are this invocation's only.
            const bool was_off{facts_off};
            facts_off = true;
            const auto value{arg(0)};
            facts_off = was_off;
            if (!value || value->base != 0) {
                return std::nullopt;
            }
            IdBound result{*value};
            for (size_t i = 0; i < 3; ++i) {
                if (result.coef[i] == 0) {
                    continue;
                }
                if (workgroup[i] == 0) {
                    return std::nullopt;
                }
                result.c += static_cast<s64>(result.coef[i] * (workgroup[i] - 1));
                result.coef[i] = 0;
            }
            return Checked(result);
        }
        case IR::Opcode::BitCount32:
            if (guards == nullptr) {
                return std::nullopt;
            }
            // FRITCHII_EXTENT_GUARD: the set bits of a 32-bit value
            return Constant(32);
        default:
            return std::nullopt;
        }
    }

    /// FRITCHII_EXTENT_GUARD: word - v is at most word while word >= v. v's largest value is
    /// checked against an immediate here, and against a constant buffer word at bind time (a
    /// floor of the extent).
    std::optional<IdBound> Difference(const IR::Inst& inst, const IR::Value& word,
                                      const IR::Value& subtracted, u32 depth) {
        const std::optional<IdBound> amount{Evaluate(subtracted, depth + 1)};
        if (!amount) {
            return std::nullopt;
        }
        const std::optional<u64> largest{StaticMax(*amount)};
        if (!largest || *largest > 0xffffffff) {
            Fail(ExtentUnknown::VariableProduct, &inst);
            return std::nullopt;
        }
        const IR::Value resolved{word.Resolve()};
        if (resolved.IsImmediate()) {
            if (resolved.Type() != IR::Type::U32 || *largest > resolved.U32()) {
                Fail(ExtentUnknown::VariableProduct, &inst);
                return std::nullopt;
            }
            return Constant(static_cast<s64>(resolved.U32()));
        }
        const std::optional<StorageBufferAddr> address{CbufWord(resolved.InstRecursive())};
        if (!address || *address == buffer) {
            Fail(ExtentUnknown::VariableProduct, &inst);
            return std::nullopt;
        }
        const u32 min{static_cast<u32>(*largest)};
        const auto existing{std::ranges::find_if(floors, [&](const StorageExtentFloor& floor) {
            return floor.cbuf_index == address->index && floor.cbuf_offset == address->offset;
        })};
        if (existing != floors.end()) {
            existing->min = (std::max)(existing->min, min);
        } else {
            // More than MaxExtentFloors fails the store (StoreExtentOf)
            floors.push_back(StorageExtentFloor{
                .cbuf_index = address->index, .cbuf_offset = address->offset, .min = min});
        }
        IdBound term{.num_terms = 1};
        term.terms[0] = IdBound::Term{.index = address->index, .offset = address->offset, .coef = 1};
        return term;
    }

    /// FRITCHII_EXTENT_GUARD: `small <= big - (strict ? 1 : 0)` as unsigned (or signed) values
    struct Relation {
        IR::Value small;
        IR::Value big;
        bool strict{};
        bool is_signed{};
    };

    /// The relations a compare with a known value states
    static void Relations(const GuardFacts::Atom& atom,
                         boost::container::small_vector<Relation, 2>& out) {
        const IR::Inst& compare{*atom.inst};
        if (compare.NumArgs() != 2 || compare.Arg(0).Type() != IR::Type::U32 ||
            compare.Arg(1).Type() != IR::Type::U32) {
            return;
        }
        const IR::Value a{compare.Arg(0)};
        const IR::Value b{compare.Arg(1)};
        const bool holds{atom.value};
        const auto add = [&](const IR::Value& small, const IR::Value& big, bool strict,
                             bool is_signed) {
            out.push_back(Relation{.small = small, .big = big, .strict = strict,
                                   .is_signed = is_signed});
        };
        switch (compare.GetOpcode()) {
        case IR::Opcode::ULessThan:
        case IR::Opcode::SLessThan: {
            const bool is_signed{compare.GetOpcode() == IR::Opcode::SLessThan};
            holds ? add(a, b, true, is_signed) : add(b, a, false, is_signed);
            return;
        }
        case IR::Opcode::ULessThanEqual:
        case IR::Opcode::SLessThanEqual: {
            const bool is_signed{compare.GetOpcode() == IR::Opcode::SLessThanEqual};
            holds ? add(a, b, false, is_signed) : add(b, a, true, is_signed);
            return;
        }
        case IR::Opcode::UGreaterThan:
        case IR::Opcode::SGreaterThan: {
            const bool is_signed{compare.GetOpcode() == IR::Opcode::SGreaterThan};
            holds ? add(b, a, true, is_signed) : add(a, b, false, is_signed);
            return;
        }
        case IR::Opcode::UGreaterThanEqual:
        case IR::Opcode::SGreaterThanEqual: {
            const bool is_signed{compare.GetOpcode() == IR::Opcode::SGreaterThanEqual};
            holds ? add(b, a, false, is_signed) : add(a, b, true, is_signed);
            return;
        }
        case IR::Opcode::IEqual:
        case IR::Opcode::INotEqual:
            if (holds == (compare.GetOpcode() == IR::Opcode::IEqual)) {
                add(a, b, false, false);
                add(b, a, false, false);
            }
            return;
        default:
            return;
        }
    }

    /// FRITCHII_EXTENT_GUARD: a bound of `inst` from the store's path facts, or why none holds
    std::optional<IdBound> PathBound(const IR::Inst& inst, u32 depth, ExtentUnknownWhy& refusal) {
        if (path_facts == nullptr) {
            return std::nullopt;
        }
        const ExtentUnknownWhy saved{failure};
        const auto refuse = [&](ExtentUnknown kind, const IR::Inst& at) {
            if (refusal.kind == ExtentUnknown::Unrecorded) {
                refusal = ExtentUnknownWhy{.kind = kind,
                                           .opcode = static_cast<u16>(at.GetOpcode())};
            }
        };
        std::optional<IdBound> best;
        boost::container::small_vector<Relation, 2> relations;
        for (const GuardFacts::Atom& atom : *path_facts) {
            relations.clear();
            Relations(atom, relations);
            for (const Relation& relation : relations) {
                const IR::Value small{relation.small.Resolve()};
                if (small.IsImmediate()) {
                    continue;
                }
                // The compared value is inst, or inst - d (slack d): a wrap below zero makes it
                // huge and fails the compare, so inst <= limit + d. inst + c (c > 0) or plus a
                // variable may wrap past 2^32 and pass: no bound.
                const IR::Inst* const compared{small.InstRecursive()};
                s64 slack{};
                if (compared != &inst) {
                    const IR::Opcode opcode{compared->GetOpcode()};
                    if (opcode != IR::Opcode::IAdd32 && opcode != IR::Opcode::ISub32) {
                        continue;
                    }
                    const size_t sides{opcode == IR::Opcode::ISub32 ? 1U : 2U};
                    size_t side{sides};
                    for (size_t i = 0; i < sides; ++i) {
                        const IR::Value operand{compared->Arg(i).Resolve()};
                        if (!operand.IsImmediate() && operand.InstRecursive() == &inst) {
                            side = i;
                            break;
                        }
                    }
                    if (side == sides) {
                        continue;
                    }
                    const IR::Value other{compared->Arg(1 - side).Resolve()};
                    if (!other.IsImmediate() || other.Type() != IR::Type::U32) {
                        refuse(ExtentUnknown::GuardWraps, *compared);
                        continue;
                    }
                    const u32 immediate{other.U32()};
                    if (opcode == IR::Opcode::ISub32) {
                        slack = static_cast<s64>(immediate);
                    } else if (static_cast<s32>(immediate) <= 0) {
                        slack = -static_cast<s64>(static_cast<s32>(immediate));
                    } else {
                        refuse(ExtentUnknown::GuardWraps, *compared);
                        continue;
                    }
                }
                if (relation.is_signed) {
                    // A negative value passes a signed `x < K`
                    refuse(ExtentUnknown::GuardSigned, *atom.inst);
                    continue;
                }
                if (guards->InLoop(inst)) {
                    // Another iteration's value may not be the one compared
                    refuse(ExtentUnknown::GuardInLoop, inst);
                    continue;
                }
                std::optional<IdBound> limit;
                const IR::Value big{relation.big.Resolve()};
                if (big.IsImmediate()) {
                    if (big.Type() == IR::Type::U32) {
                        limit = Constant(static_cast<s64>(big.U32()));
                    }
                } else {
                    limit = Evaluate(big, depth + 1);
                }
                failure = saved;
                if (!limit || limit->base != 0) {
                    refuse(ExtentUnknown::GuardLimitUnknown,
                           big.IsImmediate() ? *atom.inst : *big.InstRecursive());
                    continue;
                }
                IdBound bound{*limit};
                bound.c += slack - (relation.strict ? 1 : 0);
                if (bound.IsConstant() && bound.c < 0) {
                    // `x < 0`: the store never runs
                    bound.c = 0;
                }
                const std::optional<IdBound> checked{Checked(bound)};
                if (checked && (!best || (best->HasTerms() && !checked->HasTerms()))) {
                    best = checked;
                }
            }
        }
        failure = saved;
        if (!best && refusal.kind == ExtentUnknown::Unrecorded && guards->Compared(inst)) {
            refuse(ExtentUnknown::GuardNotDominating, inst);
        }
        return best;
    }

    StorageBufferAddr buffer;
    std::array<u32, 3> workgroup;
    bool terms_mode{};
    SharedSlots* shared{};
    bool forwarded{};
    std::unordered_map<const IR::Inst*, Memo> memo;
    std::unordered_set<const IR::Inst*> active;
    std::optional<IdBound> guard;
    bool guard_failed{};
    /// The current access' first exit, and the first failed access' (FRITCHII_EXTENT_CENSUS)
    ExtentUnknownWhy failure{};
    ExtentUnknownWhy why{};
    /// FRITCHII_EXTENT_GUARD: the program's path facts, the current store's, whether they are
    /// set aside (a shuffle's operand: another lane's value), the results evaluated without them
    /// (kept across stores) and the words the bounds need at least a value in
    GuardFacts* guards{};
    const std::vector<GuardFacts::Atom>* path_facts{};
    bool facts_off{};
    std::unordered_map<const IR::Inst*, Memo> memo_no_facts;
    std::vector<StorageExtentFloor> floors;
};

/// FRITCHII_STORE_FOOTPRINT: what is known of a 32-bit value as an exact congruence,
///
///   value = base * pointer word + c + k * m   for some integer k
///
/// with m == 0 meaning exactly base * pointer + c. Unlike IdBound (an upper bound), every step
/// keeps the value's exact residue: an unknown value (loaded from memory, a constant buffer word,
/// a loop counter) is (0, 1), and times a constant stride K it becomes (0, K), so a record index
/// loaded from memory still says which bytes of a K-byte record a store can hit. Moduli that are
/// not powers of two hold while products don't wrap 32 bits (StorageStoreFootprint says which).
/// c is a residue in [0, m), or for m == 0 the value as a signed 32-bit number (an immediate
/// 0xfffffff0 is -16, as a subtraction meant it).
struct ExactCongruence {
    s64 c{};
    u64 m{1};
    u32 base{};
    /// Every unknown multiple came from invocation or workgroup ids (DL's exact index bound)
    bool ids_only{true};
};

class StoreFootprintAnalysis {
public:
    /// With `shared` (FRITCHII_EXTENT_SHARED's SharedSlots) a 32-bit load from a private shared
    /// memory slot is congruent like the values its invocation stored there
    explicit StoreFootprintAnalysis(StorageBufferAddr buffer_, SharedSlots* shared_ = nullptr)
        : buffer{buffer_}, shared{shared_} {}

    /// One store or atomic to the buffer
    void AddStore(IR::Inst& inst) {
        if (why != StoreFootprintWhy::Recorded) {
            return;
        }
        const std::optional<LowAddrInfo> addr{TrackLowAddress(&inst)};
        if (!addr) {
            why = StoreFootprintWhy::AddressNotTracked;
            return;
        }
        const ExactCongruence value{Evaluate(addr->value, 0)};
        if (value.base != 1) {
            why = StoreFootprintWhy::NoPointerTerm;
            return;
        }
        // The immediate offset is added to the low word in 32 bits like the rest
        const ExactCongruence offset{
            Sum(ExactCongruence{.c = value.c, .m = value.m, .ids_only = value.ids_only},
                Constant(static_cast<u32>(addr->imm_offset)))};
        // An exact offset's c is signed (below the pointer when negative)
        stores.push_back(StoreCongruence{
            .c = offset.c,
            .m = offset.m,
            .size = AccessSize(inst.GetOpcode()),
        });
        ids_only &= value.ids_only;
    }

    [[nodiscard]] StorageStoreFootprint Result() const {
        if (why != StoreFootprintWhy::Recorded) {
            StorageStoreFootprint none{};
            none.why = why;
            return none;
        }
        return MakeStoreFootprint(stores.data(), stores.size(), ids_only);
    }

private:
    static constexpr u64 Exact = u64{1} << 32;
    static constexpr u32 MAX_DEPTH = 256;

    static u64 Gcd(u64 a, u64 b) {
        return std::gcd(a, b);
    }

    /// The signed 32-bit number with value's low 32 bits
    static s64 Signed32(s64 value) {
        return static_cast<s64>(static_cast<s32>(static_cast<u32>(static_cast<u64>(value))));
    }

    /// value mod m in [0, m) (m != 0, at most 2^32)
    static u64 Residue(s64 value, u64 m) {
        const s64 r{value % static_cast<s64>(m)};
        return static_cast<u64>(r < 0 ? r + static_cast<s64>(m) : r);
    }

    /// m capped to a divisor of itself no larger than 2^32 (2^32 itself: exact), c reduced
    static ExactCongruence Normalize(ExactCongruence value) {
        if (value.m > Exact) {
            const u32 shift{static_cast<u32>((std::min)(std::countr_zero(value.m), 32))};
            value.m = u64{1} << shift;
        }
        if (value.m == Exact) {
            value.m = 0;
        }
        value.c = value.m == 0 ? Signed32(value.c) : static_cast<s64>(Residue(value.c, value.m));
        return value;
    }

    static ExactCongruence Any(bool ids_only) {
        return ExactCongruence{.c = 0, .m = 1, .base = 0, .ids_only = ids_only};
    }

    static ExactCongruence Constant(u32 value) {
        return ExactCongruence{.c = static_cast<s64>(static_cast<s32>(value)), .m = 0};
    }

    /// |c| (a representative's size, for a product's modulus)
    static u64 Magnitude(s64 c) {
        return c < 0 ? static_cast<u64>(-c) : static_cast<u64>(c);
    }

    /// A product's modulus, or a power-of-two divisor of it when it doesn't fit 64 bits
    static u64 Product(u64 a, u64 b) {
        if (a == 0 || b == 0) {
            return 0;
        }
        if (a > ~u64{0} / b) {
            return u64{1} << (std::min)(std::countr_zero(a) + std::countr_zero(b), 32);
        }
        return a * b;
    }

    static ExactCongruence Sum(const ExactCongruence& a, const ExactCongruence& b) {
        if (a.base + b.base > 1) {
            return Any(false);
        }
        return Normalize(ExactCongruence{.c = a.c + b.c,
                                         .m = Gcd(a.m, b.m),
                                         .base = a.base + b.base,
                                         .ids_only = a.ids_only && b.ids_only});
    }

    static ExactCongruence Difference(const ExactCongruence& a, const ExactCongruence& b) {
        if (b.base > a.base) {
            return Any(false);
        }
        // |c| is below 2^32 on both sides: no overflow
        return Normalize(ExactCongruence{.c = a.c - b.c,
                                         .m = Gcd(a.m, b.m),
                                         .base = a.base - b.base,
                                         .ids_only = a.ids_only && b.ids_only});
    }

    static ExactCongruence Product(const ExactCongruence& a, const ExactCongruence& b) {
        if (a.base != 0 || b.base != 0) {
            return Any(false);
        }
        // (a.c + i a.m)(b.c + j b.m) = a.c b.c + i a.m b.c + j b.m a.c + i j a.m b.m
        const u64 m{Gcd(Gcd(Product(a.m, Magnitude(b.c)), Product(b.m, Magnitude(a.c))),
                        Product(a.m, b.m))};
        const bool ids_only{a.ids_only && b.ids_only};
        if (m == 0) {
            // Both exact: the 32-bit product
            const u32 product{static_cast<u32>(static_cast<u64>(a.c)) *
                              static_cast<u32>(static_cast<u64>(b.c))};
            return Normalize(ExactCongruence{.c = static_cast<s64>(product), .m = 0,
                                             .ids_only = ids_only});
        }
        const u64 capped{Normalize(ExactCongruence{.m = m}).m};
        if (capped == 0) {
            const u32 product{static_cast<u32>(static_cast<u64>(a.c)) *
                              static_cast<u32>(static_cast<u64>(b.c))};
            return Normalize(ExactCongruence{.c = static_cast<s64>(product), .m = 0,
                                             .ids_only = ids_only});
        }
        // Residues below 2^32: the product fits 64 bits
        const u64 product{(Residue(a.c, capped) * Residue(b.c, capped)) % capped};
        return Normalize(ExactCongruence{.c = static_cast<s64>(product), .m = capped,
                                         .ids_only = ids_only});
    }

    /// Either value (a select, a phi, a min or max)
    static ExactCongruence Join(const ExactCongruence& a, const ExactCongruence& b) {
        if (a.base != b.base) {
            return Any(false);
        }
        const u64 diff{Magnitude(a.c - b.c)};
        return Normalize(ExactCongruence{.c = a.c,
                                         .m = Gcd(Gcd(a.m, b.m), diff),
                                         .base = a.base,
                                         .ids_only = a.ids_only && b.ids_only});
    }

    /// The low bits a value's congruence fixes (32: all)
    static u32 KnownBits(const ExactCongruence& value) {
        return value.m == 0 ? 32U
                            : static_cast<u32>((std::min)(std::countr_zero(value.m), 32));
    }

    /// A value from its bits: bit(i) gives {known, value}; the known prefix becomes the modulus
    template <typename BitOf>
    static ExactCongruence FromBits(BitOf&& bit_of, bool ids_only) {
        u64 bits{};
        u32 known{};
        for (; known < 32; ++known) {
            const auto [is_known, value] = bit_of(known);
            if (!is_known) {
                break;
            }
            bits |= u64{value ? 1U : 0U} << known;
        }
        return Normalize(ExactCongruence{.c = static_cast<s64>(bits),
                                         .m = known >= 32 ? 0 : (u64{1} << known),
                                         .ids_only = ids_only});
    }

    /// Bit `bit` of the value, when its congruence fixes it (an exact value's two's complement
    /// bits, a residue's low bits below the modulus' power of two)
    static std::pair<bool, bool> BitOf(const ExactCongruence& value, u32 bit) {
        if (bit >= KnownBits(value)) {
            return {false, false};
        }
        return {true, ((static_cast<u64>(value.c) >> bit) & 1) != 0};
    }

    ExactCongruence Evaluate(const IR::Value& value, u32 depth) {
        if (value.IsImmediate()) {
            if (value.Type() != IR::Type::U32) {
                return Any(false);
            }
            return Constant(value.U32());
        }
        if (depth > MAX_DEPTH) {
            return Any(false);
        }
        const IR::Inst* const inst{value.InstRecursive()};
        if (const auto it{memo.find(inst)}; it != memo.end()) {
            return it->second;
        }
        if (!active.insert(inst).second) {
            // A cycle (a loop-carried value): anything. Values computed from it while it is open
            // are bounded through this "anything", so they stay exact congruences.
            return Any(false);
        }
        const ExactCongruence result{EvaluateInst(*inst, depth)};
        active.erase(inst);
        memo.emplace(inst, result);
        return result;
    }

    ExactCongruence EvaluateInst(const IR::Inst& inst, u32 depth) {
        if (const std::optional<StorageBufferAddr> word{CbufWord(&inst)}) {
            if (*word == buffer) {
                return ExactCongruence{.c = 0, .m = 0, .base = 1};
            }
            return Any(false);
        }
        const auto arg{[&](size_t index) { return Evaluate(inst.Arg(index), depth + 1); }};
        const auto imm{[&](size_t index) -> std::optional<u32> {
            if (!inst.Arg(index).IsImmediate()) {
                return std::nullopt;
            }
            return inst.Arg(index).U32();
        }};
        switch (inst.GetOpcode()) {
        case IR::Opcode::CompositeExtractU32x3: {
            if (inst.Arg(0).IsImmediate()) {
                return Any(false);
            }
            const IR::Opcode vector{inst.Arg(0).InstRecursive()->GetOpcode()};
            return Any(vector == IR::Opcode::LocalInvocationId ||
                       vector == IR::Opcode::WorkgroupId);
        }
        case IR::Opcode::IAdd32:
            return Sum(arg(0), arg(1));
        case IR::Opcode::ISub32:
            return Difference(arg(0), arg(1));
        case IR::Opcode::INeg32:
            return Difference(Constant(0), arg(0));
        case IR::Opcode::IMul32:
            return Product(arg(0), arg(1));
        case IR::Opcode::ShiftLeftLogical32: {
            const std::optional<u32> shift{imm(1)};
            if (!shift) {
                return Any(false);
            }
            if (*shift >= 32) {
                return Constant(0);
            }
            return Product(arg(0), Constant(u32{1} << *shift));
        }
        case IR::Opcode::ShiftRightLogical32: {
            const std::optional<u32> shift{imm(1)};
            const ExactCongruence source{arg(0)};
            if (!shift || source.base != 0) {
                return Any(false);
            }
            const u32 s{*shift};
            return FromBits(
                [&](u32 bit) -> std::pair<bool, bool> {
                    return bit + s < 32 ? BitOf(source, bit + s) : std::pair{true, false};
                },
                source.ids_only);
        }
        case IR::Opcode::BitwiseAnd32:
        case IR::Opcode::BitwiseOr32:
        case IR::Opcode::BitwiseXor32: {
            const ExactCongruence lhs{arg(0)};
            const ExactCongruence rhs{arg(1)};
            if (lhs.base != 0 || rhs.base != 0) {
                return Any(false);
            }
            const IR::Opcode opcode{inst.GetOpcode()};
            return FromBits(
                [&](u32 bit) -> std::pair<bool, bool> {
                    const auto [lk, lv] = BitOf(lhs, bit);
                    const auto [rk, rv] = BitOf(rhs, bit);
                    if (opcode == IR::Opcode::BitwiseAnd32) {
                        if ((lk && !lv) || (rk && !rv)) {
                            return {true, false};
                        }
                        return {lk && rk, lv && rv};
                    }
                    if (opcode == IR::Opcode::BitwiseOr32) {
                        if ((lk && lv) || (rk && rv)) {
                            return {true, true};
                        }
                        return {lk && rk, false};
                    }
                    return {lk && rk, lv != rv};
                },
                lhs.ids_only && rhs.ids_only);
        }
        case IR::Opcode::BitFieldUExtract: {
            const std::optional<u32> offset{imm(1)};
            const std::optional<u32> count{imm(2)};
            const ExactCongruence source{arg(0)};
            if (!offset || !count || *offset >= 32 || *count == 0 || *count > 32 ||
                source.base != 0) {
                return Any(false);
            }
            const u32 off{*offset};
            const u32 bits{*count};
            return FromBits(
                [&](u32 bit) -> std::pair<bool, bool> {
                    if (bit >= bits || off + bit >= 32) {
                        return {true, false};
                    }
                    return BitOf(source, off + bit);
                },
                source.ids_only);
        }
        case IR::Opcode::BitFieldInsert: {
            const std::optional<u32> offset{imm(2)};
            const std::optional<u32> count{imm(3)};
            const ExactCongruence base_value{arg(0)};
            const ExactCongruence insert{arg(1)};
            if (!offset || !count || *offset >= 32 || base_value.base != 0 || insert.base != 0) {
                return Any(false);
            }
            const u32 off{*offset};
            const u32 bits{*count};
            return FromBits(
                [&](u32 bit) -> std::pair<bool, bool> {
                    if (bit >= off && bit - off < bits) {
                        return BitOf(insert, bit - off);
                    }
                    return BitOf(base_value, bit);
                },
                base_value.ids_only && insert.ids_only);
        }
        case IR::Opcode::SelectU32:
            return Join(arg(1), arg(2));
        case IR::Opcode::UMin32:
        case IR::Opcode::UMax32:
        case IR::Opcode::SMin32:
        case IR::Opcode::SMax32:
            return Join(arg(0), arg(1));
        case IR::Opcode::Phi: {
            std::optional<ExactCongruence> result;
            for (size_t index = 0; index < inst.NumArgs(); ++index) {
                const ExactCongruence incoming{arg(index)};
                result = result ? Join(*result, incoming) : incoming;
            }
            return result ? *result : Any(false);
        }
        case IR::Opcode::LoadSharedU32: {
            if (shared == nullptr) {
                return Any(false);
            }
            // The newest of its invocation's stores to the word: congruent like each of them
            const SharedSlots::Forward forward{shared->Resolve(inst)};
            if (forward.why.kind != ExtentUnknown::Known) {
                return Any(false);
            }
            std::optional<ExactCongruence> result;
            for (const IR::Inst* const store : forward.stores) {
                const ExactCongruence stored{Evaluate(store->Arg(1), depth + 1)};
                result = result ? Join(*result, stored) : stored;
            }
            return result ? *result : Any(false);
        }
        default:
            return Any(false);
        }
    }

    StorageBufferAddr buffer;
    SharedSlots* shared{};
    std::unordered_map<const IR::Inst*, ExactCongruence> memo;
    std::unordered_set<const IR::Inst*> active;
    std::vector<StoreCongruence> stores;
    bool ids_only{true};
    StoreFootprintWhy why{StoreFootprintWhy::Recorded};
};

/// Returns the offset in indices (not bytes) for an equivalent storage instruction
IR::U32 StorageOffset(IR::Block& block, IR::Inst& inst, StorageBufferAddr buffer, u32 alignment) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    IR::U32 offset;
    if (const std::optional<LowAddrInfo> low_addr{TrackLowAddress(&inst)}) {
        offset = low_addr->value;
        if (low_addr->imm_offset != 0) {
            offset = ir.IAdd(offset, ir.Imm32(low_addr->imm_offset));
        }
    } else {
        offset = ir.UConvert(32, IR::U64{inst.Arg(0)});
    }
    // Subtract the least significant 32 bits from the guest offset. The result is the storage
    // buffer offset in bytes.
    IR::U32 low_cbuf{ir.GetCbuf(ir.Imm32(buffer.index), ir.Imm32(buffer.offset))};

    // Align the offset base to match the host alignment requirements
    low_cbuf = ir.BitwiseAnd(low_cbuf, ir.Imm32(~(alignment - 1U)));
    return ir.ISub(offset, low_cbuf);
}

/// Replace a global memory load instruction with its storage buffer equivalent
void ReplaceLoad(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                 const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset}, inst.Flags<u32>())};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory write instruction with its storage buffer equivalent
void ReplaceWrite(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)});
    inst.Invalidate();
}

/// Replace an atomic operation on global memory instruction with its storage buffer equivalent
void ReplaceAtomic(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                   const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{
        &*block.PrependNewInst(it, new_opcode, {storage_index, offset, inst.Arg(1)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global compare-exchange with its storage buffer equivalent, keeping both operands
void ReplaceAtomicCompareExchange(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
                                  const IR::U32& offset) {
    const IR::Opcode new_opcode{GlobalToStorage(inst.GetOpcode())};
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Value value{&*block.PrependNewInst(
        it, new_opcode, {storage_index, offset, inst.Arg(1), inst.Arg(2)})};
    inst.ReplaceUsesWith(value);
}

/// Replace a global memory instruction with its storage buffer equivalent
void Replace(IR::Block& block, IR::Inst& inst, const IR::U32& storage_index,
             const IR::U32& offset) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::LoadGlobalS8:
    case IR::Opcode::LoadGlobalU8:
    case IR::Opcode::LoadGlobalS16:
    case IR::Opcode::LoadGlobalU16:
    case IR::Opcode::LoadGlobal32:
    case IR::Opcode::LoadGlobal64:
    case IR::Opcode::LoadGlobal128:
        return ReplaceLoad(block, inst, storage_index, offset);
    case IR::Opcode::WriteGlobalS8:
    case IR::Opcode::WriteGlobalU8:
    case IR::Opcode::WriteGlobalS16:
    case IR::Opcode::WriteGlobalU16:
    case IR::Opcode::WriteGlobal32:
    case IR::Opcode::WriteGlobal64:
    case IR::Opcode::WriteGlobal128:
        return ReplaceWrite(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicIAdd32:
    case IR::Opcode::GlobalAtomicSMin32:
    case IR::Opcode::GlobalAtomicUMin32:
    case IR::Opcode::GlobalAtomicSMax32:
    case IR::Opcode::GlobalAtomicUMax32:
    case IR::Opcode::GlobalAtomicInc32:
    case IR::Opcode::GlobalAtomicDec32:
    case IR::Opcode::GlobalAtomicAnd32:
    case IR::Opcode::GlobalAtomicOr32:
    case IR::Opcode::GlobalAtomicXor32:
    case IR::Opcode::GlobalAtomicExchange32:
    case IR::Opcode::GlobalAtomicIAdd64:
    case IR::Opcode::GlobalAtomicSMin64:
    case IR::Opcode::GlobalAtomicUMin64:
    case IR::Opcode::GlobalAtomicSMax64:
    case IR::Opcode::GlobalAtomicUMax64:
    case IR::Opcode::GlobalAtomicAnd64:
    case IR::Opcode::GlobalAtomicOr64:
    case IR::Opcode::GlobalAtomicXor64:
    case IR::Opcode::GlobalAtomicExchange64:
    case IR::Opcode::GlobalAtomicIAdd32x2:
    case IR::Opcode::GlobalAtomicSMin32x2:
    case IR::Opcode::GlobalAtomicUMin32x2:
    case IR::Opcode::GlobalAtomicSMax32x2:
    case IR::Opcode::GlobalAtomicUMax32x2:
    case IR::Opcode::GlobalAtomicAnd32x2:
    case IR::Opcode::GlobalAtomicOr32x2:
    case IR::Opcode::GlobalAtomicXor32x2:
    case IR::Opcode::GlobalAtomicExchange32x2:
    case IR::Opcode::GlobalAtomicAddF32:
    case IR::Opcode::GlobalAtomicAddF16x2:
    case IR::Opcode::GlobalAtomicAddF32x2:
    case IR::Opcode::GlobalAtomicMinF16x2:
    case IR::Opcode::GlobalAtomicMinF32x2:
    case IR::Opcode::GlobalAtomicMaxF16x2:
    case IR::Opcode::GlobalAtomicMaxF32x2:
        return ReplaceAtomic(block, inst, storage_index, offset);
    case IR::Opcode::GlobalAtomicCompareExchange32:
        return ReplaceAtomicCompareExchange(block, inst, storage_index, offset);
    default:
        throw InvalidArgument("Invalid global memory opcode {}", inst.GetOpcode());
    }
}
} // Anonymous namespace

void GlobalMemoryToStorageBufferPass(Environment& env, IR::Program& program,
                                     const HostTranslateInfo& host_info) {
    const bool has_global_memory =
        std::ranges::any_of(program.post_order_blocks, [](const auto* block) {
            return std::ranges::any_of(block->Instructions(), IsGlobalMemory);
        });
    if (!has_global_memory) {
        return;
    }
    const size_t max_ssbos{StorageBufferLimit(host_info)};
    TrackingConditions conditions{program};
    StorageInfo info;
    // Without 64-bit integers the backend cannot route global accesses among bound buffers.
    info.bind_competing_loaded = host_info.support_int64;
    info.iterator_words = CollectIteratorWords(program);
    info.atomic_pointers = CollectAtomicPointers(program);
    const auto collect = [&] {
        for (IR::Block* const block : program.post_order_blocks) {
            for (IR::Inst& inst : block->Instructions()) {
                if (!IsGlobalMemory(inst)) {
                    continue;
                }
                conditions.current = block;
                const bool bound{CollectStorageBuffers(*block, inst, info, env, conditions)};
                info.untracked_writes |= !bound && IsGlobalMemoryWrite(inst);
                info.untracked_reads |= !bound && !IsGlobalMemoryWrite(inst);
            }
        }
    };
    collect();
    if (info.competing_loaded_bound && info.set.size() > max_ssbos) {
        // Not enough bindings for the loaded pointers that compete with paired ones: keep the
        // pairs, as if no loaded pointer competed.
        StorageInfo paired;
        paired.iterator_words = std::move(info.iterator_words);
        paired.atomic_pointers = std::move(info.atomic_pointers);
        paired.bind_competing_loaded = false;
        info = std::move(paired);
        collect();
    }
    if (info.tables_seen && info.set.size() < max_ssbos) {
        // Again with pointer table entries, in the bindings the first pass left free.
        StorageInfo with_tables;
        with_tables.iterator_words = info.iterator_words;
        with_tables.atomic_pointers = info.atomic_pointers;
        with_tables.bind_competing_loaded = info.bind_competing_loaded;
        with_tables.tables_enabled = true;
        with_tables.table_budget = max_ssbos - info.set.size();
        with_tables.counted = info.set;
        std::swap(info, with_tables);
        collect();
        if (info.set.size() > max_ssbos) {
            // Never more storage buffers than a stage binds: without the table entries, then.
            info = std::move(with_tables);
        }
    }
    if (info.set.size() > max_ssbos) {
        // Binding only some of them would drop the accesses through the others.
        throw NotImplementedException("Binding {} storage buffers to a stage that binds {}",
                                      info.set.size(), max_ssbos);
    }
    program.info.native_global_bounds |= info.native_runtime_bounds;
    // Store extents, before the accesses are rebased (the analysis reads the guest addresses)
    std::map<StorageBufferAddr, StorageWriteExtent> extents;
    // FRITCHII_EXTENT_CENSUS: why an extent stayed unknown; FRITCHII_EXTENT_CBUF: the extents
    // with constant buffer words, for the buffers whose extent stayed unknown
    std::map<StorageBufferAddr, ExtentUnknownWhy> unknown_whys;
    std::map<StorageBufferAddr, StorageCbufExtent> cbuf_extents;
    // FRITCHII_STORE_FOOTPRINT: the residues each written buffer's stores can land on
    std::map<StorageBufferAddr, StorageStoreFootprint> footprints;
    // FRITCHII_EXTENT_SHARED: why following offsets through shared memory slots failed (Known:
    // the cbuf extent is proven that way)
    std::map<StorageBufferAddr, ExtentUnknownWhy> shared_whys;
    // FRITCHII_EXTENT_GUARD: why the path facts did not bound a buffer's stores (Known: the cbuf
    // extent is proven that way)
    std::map<StorageBufferAddr, ExtentUnknownWhy> guard_whys;
    if (!info.untracked_writes) {
        // FRITCHII_EXTENT_SHARED / FRITCHII_STORE_FOOTPRINT: the private shared memory slots,
        // proven on the first load the analyses follow
        SharedSlots shared_slots{program};
        std::map<StorageBufferAddr, StoreFootprintAnalysis> footprint_analyses;
        for (const StorageInst& storage_inst : info.to_replace) {
            if (!IsGlobalMemoryWrite(*storage_inst.inst) ||
                info.unbounded_writes.contains(storage_inst.storage_buffer)) {
                continue;
            }
            footprint_analyses
                .try_emplace(storage_inst.storage_buffer, storage_inst.storage_buffer,
                             &shared_slots)
                .first->second.AddStore(*storage_inst.inst);
        }
        for (const auto& [storage_buffer, analysis] : footprint_analyses) {
            footprints.emplace(storage_buffer, analysis.Result());
        }
        std::map<StorageBufferAddr, StoreExtentAnalysis> analyses;
        for (const StorageInst& storage_inst : info.to_replace) {
            if (!IsGlobalMemoryWrite(*storage_inst.inst) ||
                info.unbounded_writes.contains(storage_inst.storage_buffer)) {
                continue;
            }
            auto analysis{analyses.find(storage_inst.storage_buffer)};
            if (analysis == analyses.end()) {
                analysis = analyses
                               .emplace(storage_inst.storage_buffer,
                                        StoreExtentAnalysis{storage_inst.storage_buffer,
                                                            program.workgroup_size})
                               .first;
                // An empty bounded extent, widened by every store
                extents.emplace(storage_inst.storage_buffer, StorageWriteExtent{.bounded = true});
            }
            analysis->second.AddAccess(*storage_inst.inst, extents[storage_inst.storage_buffer]);
        }
        for (const auto& [storage_buffer, analysis] : analyses) {
            unknown_whys.emplace(storage_buffer, analysis.Why());
        }
        // FRITCHII_EXTENT_CBUF: the buffers left unknown, again with constant buffer words as
        // terms (its own analyses: the plain extents above stay exactly what they were)
        std::map<StorageBufferAddr, StoreExtentAnalysis> term_analyses;
        for (const StorageInst& storage_inst : info.to_replace) {
            if (!IsGlobalMemoryWrite(*storage_inst.inst) ||
                info.unbounded_writes.contains(storage_inst.storage_buffer)) {
                continue;
            }
            const auto plain{extents.find(storage_inst.storage_buffer)};
            if (plain == extents.end() || plain->second.bounded) {
                continue;
            }
            auto analysis{term_analyses.find(storage_inst.storage_buffer)};
            if (analysis == term_analyses.end()) {
                analysis = term_analyses
                               .emplace(storage_inst.storage_buffer,
                                        StoreExtentAnalysis{storage_inst.storage_buffer,
                                                            program.workgroup_size, true})
                               .first;
                cbuf_extents.emplace(storage_inst.storage_buffer,
                                     StorageCbufExtent{.ids = StorageWriteExtent{.bounded = true}});
            }
            analysis->second.AddAccess(*storage_inst.inst,
                                       cbuf_extents[storage_inst.storage_buffer]);
        }
        // FRITCHII_EXTENT_SHARED: the buffers still unknown, again following 32-bit loads from
        // private shared memory slots to the values stored there (its own analyses: the plain
        // and cbuf extents above stay exactly what they were; one proven only this way is marked
        // via_shared, which the switch gates)
        if (shared_slots.AnyLoad()) {
            std::map<StorageBufferAddr, StoreExtentAnalysis> shared_analyses;
            std::map<StorageBufferAddr, StorageCbufExtent> shared_extents;
            for (const StorageInst& storage_inst : info.to_replace) {
                if (!IsGlobalMemoryWrite(*storage_inst.inst) ||
                    info.unbounded_writes.contains(storage_inst.storage_buffer)) {
                    continue;
                }
                const auto cbuf{cbuf_extents.find(storage_inst.storage_buffer)};
                if (cbuf == cbuf_extents.end() || cbuf->second.ids.bounded) {
                    continue;
                }
                auto analysis{shared_analyses.find(storage_inst.storage_buffer)};
                if (analysis == shared_analyses.end()) {
                    analysis = shared_analyses
                                   .emplace(storage_inst.storage_buffer,
                                            StoreExtentAnalysis{storage_inst.storage_buffer,
                                                                program.workgroup_size, true,
                                                                &shared_slots})
                                   .first;
                    shared_extents.emplace(
                        storage_inst.storage_buffer,
                        StorageCbufExtent{.ids = StorageWriteExtent{.bounded = true}});
                }
                analysis->second.AddAccess(*storage_inst.inst,
                                           shared_extents[storage_inst.storage_buffer]);
            }
            for (const auto& [storage_buffer, analysis] : shared_analyses) {
                StorageCbufExtent& extent{shared_extents[storage_buffer]};
                if (extent.ids.bounded && analysis.Forwarded()) {
                    extent.via_shared = true;
                    cbuf_extents[storage_buffer] = extent;
                    shared_whys.emplace(storage_buffer,
                                        ExtentUnknownWhy{.kind = ExtentUnknown::Known});
                } else if (!extent.ids.bounded) {
                    shared_whys.emplace(storage_buffer, analysis.Why());
                }
            }
        }
        // FRITCHII_EXTENT_GUARD: the buffers still unknown, again with the branch conditions on
        // each store's path, subgroup broadcasts, bit counts and clamps (GuardFacts; its own
        // analyses: the plain, cbuf and shared extents above stay exactly what they were; one
        // proven this way is marked via_guard, which the switch gates)
        std::optional<GuardFacts> guard_facts;
        std::map<StorageBufferAddr, StoreExtentAnalysis> guard_analyses;
        std::map<StorageBufferAddr, StorageCbufExtent> guard_extents;
        for (const StorageInst& storage_inst : info.to_replace) {
            if (!IsGlobalMemoryWrite(*storage_inst.inst) ||
                info.unbounded_writes.contains(storage_inst.storage_buffer)) {
                continue;
            }
            const auto cbuf{cbuf_extents.find(storage_inst.storage_buffer)};
            if (cbuf == cbuf_extents.end() || cbuf->second.ids.bounded) {
                continue;
            }
            if (!guard_facts) {
                guard_facts.emplace(program);
            }
            auto analysis{guard_analyses.find(storage_inst.storage_buffer)};
            if (analysis == guard_analyses.end()) {
                analysis = guard_analyses
                               .emplace(storage_inst.storage_buffer,
                                        StoreExtentAnalysis{storage_inst.storage_buffer,
                                                            program.workgroup_size, true, nullptr,
                                                            &*guard_facts})
                               .first;
                guard_extents.emplace(storage_inst.storage_buffer,
                                      StorageCbufExtent{.ids = StorageWriteExtent{.bounded = true}});
            }
            analysis->second.AddAccess(*storage_inst.inst,
                                       guard_extents[storage_inst.storage_buffer]);
        }
        for (const auto& [storage_buffer, analysis] : guard_analyses) {
            StorageCbufExtent& extent{guard_extents[storage_buffer]};
            if (extent.ids.bounded) {
                extent.via_guard = true;
                cbuf_extents[storage_buffer] = extent;
                guard_whys.emplace(storage_buffer, ExtentUnknownWhy{.kind = ExtentUnknown::Known});
            } else {
                guard_whys.emplace(storage_buffer, analysis.Why());
            }
        }
    }
    // Read extents of the loads and atomics. An access left global is routed by address at run
    // time into any bound buffer at any offset, so with one left no read extent is known.
    const bool reads_known{!info.untracked_reads && !info.untracked_writes &&
                           !info.native_runtime_bounds};
    std::map<StorageBufferAddr, StorageWriteExtent> read_extents;
    if (reads_known) {
        // A buffer with no loads or atomics reads nothing: empty and proven
        for (const StorageBufferAddr& storage_buffer : info.set) {
            read_extents.emplace(storage_buffer, StorageWriteExtent{.bounded = true});
        }
        // Their own analyses: the guard state must not mix with the stores'
        std::map<StorageBufferAddr, StoreExtentAnalysis> analyses;
        for (const StorageInst& storage_inst : info.to_replace) {
            if (!IsGlobalMemoryRead(*storage_inst.inst)) {
                continue;
            }
            auto analysis{analyses.find(storage_inst.storage_buffer)};
            if (analysis == analyses.end()) {
                analysis = analyses
                               .emplace(storage_inst.storage_buffer,
                                        StoreExtentAnalysis{storage_inst.storage_buffer,
                                                            program.workgroup_size})
                               .first;
            }
            analysis->second.AddAccess(*storage_inst.inst,
                                       read_extents[storage_inst.storage_buffer]);
        }
    }
    for (const StorageBufferAddr& storage_buffer : info.set) {
        const auto read_extent{read_extents.find(storage_buffer)};
        program.info.storage_read_extents.push_back(
            read_extent != read_extents.end() ? read_extent->second : StorageWriteExtent{});
    }
    std::vector<StorageWriteExtra> extras;
    bool any_unknown{};
    bool any_written{};
    for (const StorageBufferAddr& storage_buffer : info.set) {
        const bool is_written{info.writes.contains(storage_buffer)};
        const auto extent{extents.find(storage_buffer)};
        ExtentUnknownWhy write_unknown{};
        StorageCbufExtent cbuf_extent{};
        StorageStoreFootprint footprint{};
        ExtentUnknownWhy shared_why{};
        ExtentUnknownWhy guard_why{};
        if (is_written) {
            if (const auto why{shared_whys.find(storage_buffer)}; why != shared_whys.end()) {
                shared_why = why->second;
            }
            if (const auto why{guard_whys.find(storage_buffer)}; why != guard_whys.end()) {
                guard_why = why->second;
            }
            any_written = true;
            if (info.untracked_writes) {
                footprint.why = StoreFootprintWhy::UntrackedWrites;
            } else if (info.unbounded_writes.contains(storage_buffer)) {
                footprint.why = StoreFootprintWhy::RuntimeRouted;
            } else if (const auto found{footprints.find(storage_buffer)};
                       found != footprints.end()) {
                footprint = found->second;
            } else {
                footprint.why = StoreFootprintWhy::NoStores;
            }
            if (info.untracked_writes) {
                write_unknown.kind = ExtentUnknown::UntrackedWrites;
            } else if (info.unbounded_writes.contains(storage_buffer)) {
                write_unknown.kind = ExtentUnknown::RuntimeRouted;
            } else if (extent != extents.end() && extent->second.bounded) {
                write_unknown.kind = ExtentUnknown::Known;
            } else if (const auto why{unknown_whys.find(storage_buffer)};
                       why != unknown_whys.end()) {
                write_unknown = why->second;
            } else {
                // Written, but no store of it was rebased on it: they stay global
                write_unknown.kind = ExtentUnknown::UntrackedWrites;
            }
            if (const auto cbuf{cbuf_extents.find(storage_buffer)}; cbuf != cbuf_extents.end()) {
                cbuf_extent = cbuf->second;
            }
        }
        program.info.storage_buffers_descriptors.push_back({
            .cbuf_index = storage_buffer.index,
            .cbuf_offset = storage_buffer.offset,
            .count = 1,
            .is_written = is_written,
            .indirect_offset = storage_buffer.indirect,
            .write_extent = is_written && extent != extents.end() ? extent->second
                                                                  : StorageWriteExtent{},
        });
        extras.push_back({.write_unknown = write_unknown,
                          .cbuf_extent = cbuf_extent,
                          .footprint = footprint,
                          .shared_why = shared_why,
                          .guard_why = guard_why});
        any_unknown |= is_written && write_unknown.kind != ExtentUnknown::Known;
    }
    if (any_unknown || any_written) {
        // Kept only for programs with something to report (Info::storage_write_extras; every
        // written buffer has a FRITCHII_STORE_FOOTPRINT entry)
        auto& out{program.info.storage_write_extras};
        out.resize(program.info.storage_buffers_descriptors.size() - extras.size());
        out.insert(out.end(), extras.begin(), extras.end());
    }
    for (const StorageInst& storage_inst : info.to_replace) {
        const StorageBufferAddr storage_buffer{storage_inst.storage_buffer};
        const auto it{info.set.find(storage_inst.storage_buffer)};
        const IR::U32 index{IR::Value{static_cast<u32>(info.set.index_of(it))}};
        IR::Block* const block{storage_inst.block};
        IR::Inst* const inst{storage_inst.inst};
        const IR::U32 offset{
            StorageOffset(*block, *inst, storage_buffer, u32(host_info.min_ssbo_alignment))};
        Replace(*block, *inst, index, offset);
    }
}

size_t StorageBufferLimit(const HostTranslateInfo& host_info) {
    size_t limit{Info::MAX_SSBOS};
    for (const u32 device_limit : {host_info.max_per_stage_descriptor_storage_buffers,
                                   host_info.max_descriptor_set_storage_buffers}) {
        if (device_limit != 0) {
            limit = (std::min)(limit, size_t{device_limit});
        }
    }
    return limit;
}

std::vector<u32> JoinStorageInfo(Info& base, Info& source, size_t limit) {
    base.requires_coherent_global_memory |= source.requires_coherent_global_memory;
    // An access left global in either half is routed by address into any buffer of the joined
    // program: then no read extent is known.
    const bool reads_known{
        !base.uses_global_memory && !source.uses_global_memory && !base.native_global_bounds &&
        !source.native_global_bounds &&
        base.storage_read_extents.size() == base.storage_buffers_descriptors.size() &&
        source.storage_read_extents.size() == source.storage_buffers_descriptors.size()};
    base.native_global_bounds |= source.native_global_bounds;
    auto& descriptors = base.storage_buffers_descriptors;
    std::vector<u32> bindings;
    auto& read_extents = base.storage_read_extents;
    // FRITCHII_EXTENT_CENSUS / FRITCHII_EXTENT_CBUF: the extras follow the descriptors while
    // joining (dropped again when neither half had any)
    const bool any_extras{!base.storage_write_extras.empty() ||
                          !source.storage_write_extras.empty()};
    auto& extras = base.storage_write_extras;
    if (extras.size() != descriptors.size()) {
        extras.assign(descriptors.size(), StorageWriteExtra{});
    }
    const bool source_extras{source.storage_write_extras.size() ==
                             source.storage_buffers_descriptors.size()};
    for (size_t index = 0; index < source.storage_buffers_descriptors.size(); ++index) {
        auto& desc{source.storage_buffers_descriptors[index]};
        const StorageWriteExtent source_read{reads_known ? source.storage_read_extents[index]
                                                         : StorageWriteExtent{}};
        const StorageWriteExtra source_extra{source_extras ? source.storage_write_extras[index]
                                                           : StorageWriteExtra{}};
        auto it{std::ranges::find_if(descriptors, [&desc](const auto& existing) {
            return desc.cbuf_index == existing.cbuf_index &&
                   desc.cbuf_offset == existing.cbuf_offset && desc.count == existing.count &&
                   desc.indirect_offset == existing.indirect_offset;
        })};
        if (it != descriptors.end()) {
            if (reads_known) {
                const auto existing{static_cast<size_t>(std::distance(descriptors.begin(), it))};
                MergeStorageWriteExtent(read_extents[existing], source_read);
            }
            if (desc.is_written) {
                StorageWriteExtra& extra{
                    extras[static_cast<size_t>(std::distance(descriptors.begin(), it))]};
                if (it->is_written) {
                    // FRITCHII_EXTENT_CENSUS / FRITCHII_EXTENT_CBUF: the half that left it
                    // unknown names why; the cbuf extents widen like the plain ones (a half's
                    // proven plain extent is a cbuf extent without terms)
                    const auto as_cbuf{[](const StorageBufferDescriptor& d,
                                          const StorageWriteExtra& e) {
                        return d.write_extent.bounded ? StorageCbufExtent{.ids = d.write_extent}
                                                      : e.cbuf_extent;
                    }};
                    StorageCbufExtent merged_cbuf{as_cbuf(*it, extra)};
                    MergeStorageCbufExtent(merged_cbuf, as_cbuf(desc, source_extra));
                    const bool was_bounded{it->write_extent.bounded};
                    MergeStorageWriteExtent(it->write_extent, desc.write_extent);
                    if (it->write_extent.bounded) {
                        extra.cbuf_extent = StorageCbufExtent{};
                    } else {
                        if (was_bounded) {
                            extra.write_unknown = source_extra.write_unknown;
                        }
                        extra.cbuf_extent = merged_cbuf;
                    }
                    // FRITCHII_STORE_FOOTPRINT: not merged (graphics halves only)
                    extra.footprint = StorageStoreFootprint{};
                    extra.footprint.why = StoreFootprintWhy::Joined;
                } else {
                    it->write_extent = desc.write_extent;
                    extra = source_extra;
                }
            }
            it->is_written |= desc.is_written;
            bindings.push_back(static_cast<u32>(it - descriptors.begin()));
            continue;
        }
        if (descriptors.size() >= limit) {
            throw NotImplementedException("Binding {} storage buffers to a stage that binds {}",
                                          descriptors.size() + 1, limit);
        }
        bindings.push_back(static_cast<u32>(descriptors.size()));
        descriptors.push_back(desc);
        extras.push_back(source_extra);
        if (reads_known) {
            read_extents.push_back(source_read);
        }
    }
    if (!reads_known) {
        read_extents.assign(descriptors.size(), StorageWriteExtent{});
    }
    if (!any_extras) {
        extras.clear();
    }
    return bindings;
}

void RemapStorageBindings(IR::Program& program, std::span<const u32> bindings) {
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            // Each takes the binding as its first argument (see GlobalToStorage and Replace).
            switch (inst.GetOpcode()) {
            case IR::Opcode::LoadStorageU8:
            case IR::Opcode::LoadStorageS8:
            case IR::Opcode::LoadStorageU16:
            case IR::Opcode::LoadStorageS16:
            case IR::Opcode::LoadStorage32:
            case IR::Opcode::LoadStorage64:
            case IR::Opcode::LoadStorage128:
            case IR::Opcode::WriteStorageU8:
            case IR::Opcode::WriteStorageS8:
            case IR::Opcode::WriteStorageU16:
            case IR::Opcode::WriteStorageS16:
            case IR::Opcode::WriteStorage32:
            case IR::Opcode::WriteStorage64:
            case IR::Opcode::WriteStorage128:
            case IR::Opcode::StorageAtomicIAdd32:
            case IR::Opcode::StorageAtomicSMin32:
            case IR::Opcode::StorageAtomicUMin32:
            case IR::Opcode::StorageAtomicSMax32:
            case IR::Opcode::StorageAtomicUMax32:
            case IR::Opcode::StorageAtomicInc32:
            case IR::Opcode::StorageAtomicDec32:
            case IR::Opcode::StorageAtomicAnd32:
            case IR::Opcode::StorageAtomicOr32:
            case IR::Opcode::StorageAtomicXor32:
            case IR::Opcode::StorageAtomicExchange32:
            case IR::Opcode::StorageAtomicCompareExchange32:
            case IR::Opcode::StorageAtomicIAdd64:
            case IR::Opcode::StorageAtomicSMin64:
            case IR::Opcode::StorageAtomicUMin64:
            case IR::Opcode::StorageAtomicSMax64:
            case IR::Opcode::StorageAtomicUMax64:
            case IR::Opcode::StorageAtomicAnd64:
            case IR::Opcode::StorageAtomicOr64:
            case IR::Opcode::StorageAtomicXor64:
            case IR::Opcode::StorageAtomicExchange64:
            case IR::Opcode::StorageAtomicIAdd32x2:
            case IR::Opcode::StorageAtomicSMin32x2:
            case IR::Opcode::StorageAtomicUMin32x2:
            case IR::Opcode::StorageAtomicSMax32x2:
            case IR::Opcode::StorageAtomicUMax32x2:
            case IR::Opcode::StorageAtomicAnd32x2:
            case IR::Opcode::StorageAtomicOr32x2:
            case IR::Opcode::StorageAtomicXor32x2:
            case IR::Opcode::StorageAtomicExchange32x2:
            case IR::Opcode::StorageAtomicAddF32:
            case IR::Opcode::StorageAtomicAddF16x2:
            case IR::Opcode::StorageAtomicAddF32x2:
            case IR::Opcode::StorageAtomicMinF16x2:
            case IR::Opcode::StorageAtomicMinF32x2:
            case IR::Opcode::StorageAtomicMaxF16x2:
            case IR::Opcode::StorageAtomicMaxF32x2:
                break;
            default:
                continue;
            }
            const IR::Value binding{inst.Arg(0)};
            if (binding.IsImmediate() && binding.U32() < bindings.size()) {
                inst.SetArg(0, IR::Value{bindings[binding.U32()]});
            }
        }
    }
}

} // namespace Shader::Optimization
