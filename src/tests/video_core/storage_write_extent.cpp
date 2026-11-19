// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/storage_write_extent.h"

namespace {
namespace IR = Shader::IR;
using Shader::StorageWriteExtent;

class TestEnvironment final : public Shader::Environment {
public:
    bool native{true};
    TestEnvironment() { stage = Shader::Stage::Compute; }
    bool IsNativeProgram() const override { return native; }
    bool IsCbufStorageBufferPointer(u32 index, u32) override { return index == 0; }
    u64 ReadInstruction(u32) override { return 0; }
    u32 ReadCbufValue(u32, u32) override { return 0; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {64, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }
    void Dump(u64, u64) override {}
};

struct KernelProgram {
    Shader::ObjectPool<IR::Inst> pool{256};
    IR::Block entry{pool};
    IR::Block loop{pool};
    IR::Program program;
    TestEnvironment env;
    bool ran{};

    KernelProgram() {
        program.stage = Shader::Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&entry};
        program.post_order_blocks = {&entry};
    }

    /// Runs the pass once and returns the descriptor of the buffer whose pointer is at c0[offset]
    const Shader::StorageBufferDescriptor& Run(u32 offset) {
        if (!ran) {
            Shader::HostTranslateInfo host{};
            host.min_ssbo_alignment = 16;
            Shader::Optimization::GlobalMemoryToStorageBufferPass(env, program, host);
            ran = true;
        }
        for (const auto& descriptor : program.info.storage_buffers_descriptors) {
            if (descriptor.cbuf_index == 0 && descriptor.cbuf_offset == offset) {
                return descriptor;
            }
        }
        FAIL("No descriptor for the pointer");
        return program.info.storage_buffers_descriptors.front();
    }

    /// Runs the pass once and returns the read extent of the buffer whose pointer is at c0[offset]
    const StorageWriteExtent& Read(u32 offset) {
        const auto& descriptor{Run(offset)};
        const auto& descriptors{program.info.storage_buffers_descriptors};
        REQUIRE(program.info.storage_read_extents.size() == descriptors.size());
        return program.info
            .storage_read_extents[static_cast<size_t>(&descriptor - descriptors.data())];
    }

    /// FRITCHII_EXTENT_CENSUS / FRITCHII_EXTENT_CBUF: the extra of the buffer at c0[offset]
    const Shader::StorageWriteExtra& Extra(u32 offset) {
        const auto& descriptor{Run(offset)};
        const auto& descriptors{program.info.storage_buffers_descriptors};
        REQUIRE(program.info.storage_write_extras.size() == descriptors.size());
        return program.info
            .storage_write_extras[static_cast<size_t>(&descriptor - descriptors.data())];
    }
};

IR::U32 Word(IR::IREmitter& ir, u32 offset) {
    return ir.GetCbuf(ir.Imm32(0), ir.Imm32(offset));
}

/// The 64-bit pointer at c0[offset] plus a 32-bit byte offset
IR::U64 Address(IR::IREmitter& ir, u32 offset, const IR::Value& bytes) {
    return ir.PackUint2x32(ir.CompositeConstruct(
        IR::U32{ir.IAdd(Word(ir, offset), IR::U32{bytes})}, Word(ir, offset + 4)));
}

std::optional<u64> At(const StorageWriteExtent& extent, std::array<u32, 3> block,
                      std::optional<std::array<u32, 3>> grid) {
    return Shader::ResolveStorageWriteExtent(extent, block, grid);
}
} // Anonymous namespace

TEST_CASE("Stores at constant offsets end at the last byte written", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0x40)), ir.Imm32(1));
    ir.WriteGlobal128(Address(ir, 0x170, ir.Imm32(0x10)), ir.CompositeConstruct(
        ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)));
    const auto& descriptor{fixture.Run(0x170)};
    REQUIRE(descriptor.is_written);
    const StorageWriteExtent& extent{descriptor.write_extent};
    CHECK(extent.bounded);
    CHECK_FALSE(extent.guarded);
    CHECK(extent.const_end == 0x44);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{});
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{100, 1, 1}) == 0x44);
    CHECK(At(extent, {64, 1, 1}, std::nullopt) == 0x44);
}

TEST_CASE("Stores indexed by the global invocation id scale with the dispatch",
          "[shader][storage_extent]") {
    // ((ctaid.x << 6) + tid.x) << 8 + 236: one 256-byte record per invocation, as Ocarina of
    // Time's effect and skinning programs write them
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 global{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                                 ir.LocalInvocationIdX())};
    const IR::U32 record{ir.ShiftLeftLogical(global, ir.Imm32(8))};
    ir.WriteGlobal64(Address(ir, 0x150, ir.IAdd(record, ir.Imm32(236))),
                     ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)));
    const StorageWriteExtent& extent{fixture.Run(0x150).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 244);
    CHECK(extent.local_stride == std::array<u32, 3>{256, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{16384, 0, 0});
    // 64 invocations: one group
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 244 + 256 * 63);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{3, 1, 1}) == 244 + 256 * 63 + 16384 * 2);
    // An indirect dispatch's group count is unknown
    CHECK_FALSE(At(extent, {64, 1, 1}, std::nullopt).has_value());
}

TEST_CASE("Constant multiplications and masks keep the bound", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.BitwiseAnd(ir.LocalInvocationIdX(), ir.Imm32(31))};
    const IR::U32 row{ir.IMul(ir.WorkgroupIdY(), ir.Imm32(48))};
    ir.WriteGlobal32(Address(ir, 0x160, ir.IAdd(ir.ShiftLeftLogical(lane, ir.Imm32(2)), row)),
                     ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x160).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.const_end == 31 * 4 + 4);
    CHECK(extent.local_stride == std::array<u32, 3>{});
    CHECK(extent.group_stride == std::array<u32, 3>{0, 48, 0});
}

TEST_CASE("A multiplication split in 16-bit halves is bounded while its operand fits",
          "[shader][storage_extent]") {
    // Native IMAD lowered through XMAD: lo(id) * 48 + ((hi(id) * 48) << 16)
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 id{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                             ir.LocalInvocationIdX())};
    const IR::U32 low_half{ir.BitFieldExtract(id, ir.Imm32(0), ir.Imm32(16), false)};
    const IR::U32 high_half{ir.BitFieldExtract(id, ir.Imm32(16), ir.Imm32(16), false)};
    const IR::U32 product{ir.IAdd(ir.IMul(low_half, ir.Imm32(48)),
                                  ir.ShiftLeftLogical(ir.IMul(high_half, ir.Imm32(48)),
                                                      ir.Imm32(16)))};
    ir.WriteGlobal32(Address(ir, 0x170, product), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(extent.guarded);
    CHECK(extent.local_stride == std::array<u32, 3>{48, 0, 0});
    CHECK(extent.group_stride == std::array<u32, 3>{48 * 64, 0, 0});
    // 1024 groups of 64: the largest id is 0xffff, its high half still zero
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1024, 1, 1}) == 4 + 48 * (1024 * 64 - 1));
    // 1025 groups: the id reaches 2^16 and its high half is no longer zero
    CHECK_FALSE(At(extent, {64, 1, 1}, std::array<u32, 3>{1025, 1, 1}).has_value());
}

TEST_CASE("Offsets read from memory or other parameters are unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const bool loaded = GENERATE(false, true);
    // An index loaded from another buffer, or a count or offset parameter from the constant
    // buffer
    const IR::U32 index{
        loaded ? ir.LoadGlobal32(Address(
                     ir, 0x180, IR::U32{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))}))
               : IR::U32{ir.IAdd(ir.LocalInvocationIdX(), Word(ir, 0x1a0))}};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK(descriptor.is_written);
    CHECK_FALSE(descriptor.write_extent.bounded);
    CHECK_FALSE(At(descriptor.write_extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}));
    if (loaded) {
        // The buffer only read keeps no extent and is not written
        const auto& source{fixture.Run(0x180)};
        CHECK_FALSE(source.is_written);
        CHECK_FALSE(source.write_extent.bounded);
    }
}

TEST_CASE("Loop-carried store offsets are unbounded", "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.entry.AddBranch(&fixture.loop);
    fixture.loop.AddBranch(&fixture.loop);
    fixture.program.blocks = {&fixture.entry, &fixture.loop};
    fixture.program.post_order_blocks = {&fixture.loop, &fixture.entry};
    IR::IREmitter entry{fixture.entry};
    const IR::U32 first{entry.ShiftLeftLogical(entry.LocalInvocationIdX(), entry.Imm32(2))};
    IR::IREmitter body{fixture.loop};
    fixture.loop.AppendNewInst(IR::Opcode::Phi, {});
    IR::Inst* const cursor{&fixture.loop.back()};
    cursor->SetFlags(IR::Type::U32);
    body.WriteGlobal32(Address(body, 0x170, IR::U32{IR::Value{cursor}}), body.Imm32(0));
    const IR::U32 next{body.IAdd(IR::U32{IR::Value{cursor}}, body.Imm32(256))};
    cursor->AddPhiOperand(&fixture.entry, first);
    cursor->AddPhiOperand(&fixture.loop, next);
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Selected offsets take the larger alternative", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U1 condition{ir.IEqual(ir.LocalInvocationIdX(), ir.Imm32(0))};
    const IR::U32 offset{ir.Select(condition, ir.Imm32(0x20),
                                   ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4)))};
    ir.WriteGlobal32(Address(ir, 0x170, offset), ir.Imm32(0));
    const StorageWriteExtent& extent{fixture.Run(0x170).write_extent};
    REQUIRE(extent.bounded);
    CHECK(At(extent, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 0x24 + 16 * 63);
}

TEST_CASE("A global store left untracked makes every written buffer unbounded",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), ir.Imm32(0));
    // An address with no constant-buffer pointer at all stays a global store
    ir.WriteGlobal32(ir.PackUint2x32(ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0))),
                     ir.Imm32(0));
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
}

TEST_CASE("Loads and atomics have read extents, buffers only stored to read nothing",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.env.native = GENERATE(false, true);
    IR::IREmitter ir{fixture.entry};
    const IR::U32 word{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x180, word))};
    const IR::Value vector{ir.LoadGlobal128(Address(
        ir, 0x190, ir.IAdd(ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(4)),
                           ir.Imm32(0x10))))};
    const IR::U32 old{ir.GlobalAtomicIAdd(Address(ir, 0x1a0, word), ir.Imm32(1))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)),
                     ir.IAdd(ir.IAdd(value, old), IR::U32{ir.CompositeExtract(vector, 0)}));
    const StorageWriteExtent& loaded{fixture.Read(0x180)};
    REQUIRE(loaded.bounded);
    CHECK(loaded.const_end == 4);
    CHECK(loaded.local_stride == std::array<u32, 3>{4, 0, 0});
    CHECK(At(loaded, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 4 + 4 * 63);
    const StorageWriteExtent& wide{fixture.Read(0x190)};
    REQUIRE(wide.bounded);
    CHECK(wide.const_end == 0x20);
    CHECK(wide.local_stride == std::array<u32, 3>{16, 0, 0});
    const StorageWriteExtent& atomic{fixture.Read(0x1a0)};
    REQUIRE(atomic.bounded);
    CHECK(atomic.const_end == 4);
    CHECK(atomic.local_stride == std::array<u32, 3>{4, 0, 0});
    // Only stored to: empty and proven
    const StorageWriteExtent& stored{fixture.Read(0x170)};
    CHECK(stored.bounded);
    CHECK(At(stored, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}) == 0);
}

TEST_CASE("A load indexed by a loaded value has no read extent", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 index{ir.LoadGlobal32(
        Address(ir, 0x180, ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))))};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x190, ir.ShiftLeftLogical(index, ir.Imm32(2))))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), value);
    CHECK_FALSE(fixture.Read(0x190).bounded);
}

TEST_CASE("An access left global leaves every read extent unknown", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 value{ir.LoadGlobal32(Address(ir, 0x180, ir.Imm32(0)))};
    // An address with no constant-buffer pointer at all stays a global load
    const IR::U32 other{ir.LoadGlobal32(
        ir.PackUint2x32(ir.CompositeConstruct(ir.LocalInvocationIdX(), ir.Imm32(0))))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0)), ir.IAdd(value, other));
    CHECK_FALSE(fixture.Read(0x180).bounded);
    CHECK_FALSE(fixture.Read(0x170).bounded);
}

TEST_CASE("Store extents merge to cover both", "[shader][storage_extent]") {
    StorageWriteExtent lhs{.bounded = true, .const_end = 16, .local_stride = {4, 0, 0}};
    const StorageWriteExtent rhs{.bounded = true, .const_end = 8, .group_stride = {0, 32, 0}};
    Shader::MergeStorageWriteExtent(lhs, rhs);
    CHECK(lhs.bounded);
    CHECK(lhs.const_end == 16);
    CHECK(lhs.local_stride == std::array<u32, 3>{4, 0, 0});
    CHECK(lhs.group_stride == std::array<u32, 3>{0, 32, 0});
    Shader::MergeStorageWriteExtent(lhs, StorageWriteExtent{});
    CHECK_FALSE(lhs.bounded);
}

TEST_CASE("An index from a constant buffer word is a cbuf extent term",
          "[shader][storage_extent]") {
    // c0[0x1a0] * 0x100 + tid.x * 4: an object's record, picked by a uniform
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 record{ir.IMul(Word(ir, 0x1a0), ir.Imm32(0x100))};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.IAdd(record, lane)), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.write_unknown.kind == Shader::ExtentUnknown::OtherCbufWord);
    CHECK(extra.write_unknown.cbuf_index == 0);
    CHECK(extra.write_unknown.cbuf_offset == 0x1a0);
    const Shader::StorageCbufExtent& cbuf{extra.cbuf_extent};
    REQUIRE(cbuf.ids.bounded);
    REQUIRE(cbuf.num_terms == 1);
    CHECK(cbuf.num_guard_terms == 0);
    CHECK(cbuf.terms[0].cbuf_index == 0);
    CHECK(cbuf.terms[0].cbuf_offset == 0x1a0);
    CHECK(cbuf.terms[0].coef == 0x100);
    CHECK(cbuf.ids.const_end == 4);
    CHECK(cbuf.ids.local_stride == std::array<u32, 3>{4, 0, 0});
    const std::array<u32, Shader::MaxExtentTerms> values{3, 0, 0, 0};
    const std::array<u32, Shader::MaxExtentTerms> no_guard{};
    CHECK(Shader::ResolveStorageCbufExtent(cbuf, {64, 1, 1}, std::array<u32, 3>{1, 1, 1}, values,
                                           no_guard) == 4 + 4 * 63 + 3 * 0x100);
}

TEST_CASE("A constant buffer read at a computed offset leaves the cbuf extent unknown",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    const IR::U32 index{ir.GetCbuf(ir.Imm32(0), lane)};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x170)};
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.write_unknown.kind == Shader::ExtentUnknown::DynamicCbuf);
    CHECK_FALSE(extra.cbuf_extent.ids.bounded);
}

TEST_CASE("Record stores keep a footprint modulo the record stride", "[shader][storage_extent]") {
    // A record picked by a constant buffer word, stride 0x100: stores at +0x20..+0x2f and
    // +0x40..+0x43, never +0x10
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 record{ir.IMul(Word(ir, 0x1a0), ir.Imm32(0x100))};
    ir.WriteGlobal128(Address(ir, 0x170, ir.IAdd(record, ir.Imm32(0x20))),
                      ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0), ir.Imm32(0), ir.Imm32(0)));
    ir.WriteGlobal32(Address(ir, 0x170, ir.IAdd(record, ir.Imm32(0x40))), ir.Imm32(0));
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 0x100);
    CHECK_FALSE(footprint.ids_only);
    CHECK_FALSE(footprint.covers_all);
    REQUIRE(footprint.num_intervals == 2);
    CHECK(footprint.intervals[0].begin == 0x20);
    CHECK(footprint.intervals[0].end == 0x30);
    CHECK(footprint.intervals[1].begin == 0x40);
    CHECK(footprint.intervals[1].end == 0x44);
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x310, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x32c, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x543, 1));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, -0xf0, 4)); // +0x10 of the record before
}

TEST_CASE("Stores a lane apart cover every residue", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 lane{ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2))};
    ir.WriteGlobal32(Address(ir, 0x170, lane), ir.Imm32(0));
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 4);
    CHECK(footprint.ids_only);
    CHECK(footprint.covers_all);
    CHECK(Shader::StoreFootprintHits(footprint, 0x10, 1));
}

TEST_CASE("Constant stores have an exact footprint", "[shader][storage_extent]") {
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0x40)), ir.Imm32(1));
    ir.WriteGlobal32(Address(ir, 0x170, ir.Imm32(0xfffffff0)), ir.Imm32(1)); // 16 bytes below
    const Shader::StorageStoreFootprint& footprint{fixture.Extra(0x170).footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == (u64{1} << 32));
    CHECK(Shader::StoreFootprintHits(footprint, 0x40, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x44, 4));
    CHECK(Shader::StoreFootprintHits(footprint, -0x10, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, -0x20, 4));
}

TEST_CASE("A record stride that is not a power of two keeps its residues",
          "[shader][storage_extent]") {
    // 0x330-byte records picked by a loaded index: stores at +0x40 and (two records on) +0x48
    const std::array<Shader::StoreCongruence, 2> stores{{
        {.c = 0x40, .m = 0x330, .size = 4},
        {.c = -0x330 + 0x48, .m = 0x660, .size = 4},
    }};
    const Shader::StorageStoreFootprint footprint{
        Shader::MakeStoreFootprint(stores.data(), stores.size(), false)};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 0x330);
    CHECK_FALSE(footprint.PowerOfTwo());
    REQUIRE(footprint.num_intervals == 2);
    CHECK(footprint.intervals[0].begin == 0x40);
    CHECK(footprint.intervals[0].end == 0x44);
    CHECK(footprint.intervals[1].begin == 0x48);
    CHECK(footprint.intervals[1].end == 0x4c);
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x330 * 5 + 0x10, 8));
    CHECK(Shader::StoreFootprintHits(footprint, 0x330 * 5 + 0x48, 1));
    CHECK(Shader::StoreFootprintHits(footprint, -0x330 + 0x40, 4));
}

namespace {
/// (tid.y * 64 + tid.x) * 0x14 through a native IMAD lowered to 16-bit halves: each invocation's
/// own 20-byte shared memory slot, as the flower program 989e109982cb409b computes it
IR::U32 SharedSlot(IR::IREmitter& ir) {
    const IR::U32 linear{ir.IAdd(ir.ShiftLeftLogical(ir.LocalInvocationIdY(), ir.Imm32(6)),
                                 ir.LocalInvocationIdX())};
    const IR::U32 low_half{ir.BitFieldExtract(linear, ir.Imm32(0), ir.Imm32(16), false)};
    const IR::U32 high_half{ir.BitFieldExtract(linear, ir.Imm32(16), ir.Imm32(16), false)};
    return ir.IAdd(ir.IMul(low_half, ir.Imm32(0x14)),
                   ir.ShiftLeftLogical(ir.IMul(high_half, ir.Imm32(0x14)), ir.Imm32(16)));
}

/// 0x140 * (64 * ctaid.x + tid.x): the invocation's 320-byte record
IR::U32 FlowerRecord(IR::IREmitter& ir) {
    const IR::U32 global{ir.IAdd(ir.ShiftLeftLogical(ir.WorkgroupIdX(), ir.Imm32(6)),
                                 ir.LocalInvocationIdX())};
    return ir.IMul(global, ir.Imm32(0x140));
}
} // Anonymous namespace

TEST_CASE("Record offsets parked in an invocation's shared slot are followed to their stores",
          "[shader][storage_extent]") {
    // 989e109982cb409b: each invocation spills two offsets into its record to its own shared
    // slot, reloads them later and stores through them
    KernelProgram fixture;
    fixture.program.shared_memory_size = 64 * 0x14;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 slot{SharedSlot(ir)};
    const IR::U32 record{FlowerRecord(ir)};
    ir.WriteShared(32, ir.IAdd(slot, ir.Imm32(8)), ir.IAdd(record, ir.Imm32(0xb0)));
    ir.WriteShared(32, slot, ir.IAdd(record, ir.Imm32(0xa0)));
    ir.WriteShared(32, ir.IAdd(slot, ir.Imm32(4)), ir.Imm32(0x3f800000)); // a spilled float
    ir.WriteGlobal32(Address(ir, 0x160, ir.IAdd(record, ir.Imm32(0x134))), ir.Imm32(0));
    const IR::U32 first{ir.LoadShared(32, false, slot)};
    const IR::U32 second{ir.LoadShared(32, false, ir.IAdd(slot, ir.Imm32(8)))};
    ir.WriteGlobal64(Address(ir, 0x160, first), ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)));
    ir.WriteGlobal32(Address(ir, 0x160, ir.IAdd(second, ir.Imm32(8))), ir.Imm32(0));
    const auto& descriptor{fixture.Run(0x160)};
    REQUIRE(descriptor.is_written);
    // The plain extent stays what it was (default behaviour and the precompiled cache unchanged)
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x160)};
    CHECK(extra.write_unknown.kind == Shader::ExtentUnknown::LoadedFromMemory);
    CHECK(extra.shared_why.kind == Shader::ExtentUnknown::Known);
    const Shader::StorageCbufExtent& cbuf{extra.cbuf_extent};
    REQUIRE(cbuf.ids.bounded);
    CHECK(cbuf.via_shared);
    CHECK(cbuf.num_terms == 0);
    CHECK(cbuf.ids.const_end == 0x138);
    CHECK(cbuf.ids.local_stride == std::array<u32, 3>{0x140, 0, 0});
    CHECK(cbuf.ids.group_stride == std::array<u32, 3>{0x5000, 0, 0});
    const std::array<u32, Shader::MaxExtentTerms> none{};
    // 20 KiB a workgroup: 0x5000 * gx - 8
    CHECK(Shader::ResolveStorageCbufExtent(cbuf, {64, 1, 1}, std::array<u32, 3>{6, 1, 1}, none,
                                           none) == 0x5000 * 6 - 8);
    CHECK_FALSE(Shader::ResolveStorageCbufExtent(cbuf, {64, 1, 1}, std::nullopt, none, none));
    // The footprint sees through the slot too: record residues only
    const Shader::StorageStoreFootprint& footprint{extra.footprint};
    REQUIRE(footprint.Valid());
    CHECK(footprint.modulus == 0x140);
    CHECK(footprint.ids_only);
    CHECK_FALSE(Shader::StoreFootprintHits(footprint, 0x140 * 3 + 0x10, 0x10));
    CHECK(Shader::StoreFootprintHits(footprint, 0x140 * 3 + 0xa4, 4));
}

TEST_CASE("A shared slot that is not the invocation's own is not followed",
          "[shader][storage_extent]") {
    KernelProgram fixture;
    fixture.program.shared_memory_size = 64 * 0x14;
    IR::IREmitter ir{fixture.entry};
    const int variant = GENERATE(0, 1, 2, 3);
    const IR::U32 record{FlowerRecord(ir)};
    IR::U32 slot{SharedSlot(ir)};
    Shader::ExtentUnknown expected{};
    switch (variant) {
    case 0:
        // One slot for the whole workgroup: another invocation's store may be the one loaded
        slot = ir.Imm32(0x40);
        expected = Shader::ExtentUnknown::SharedNotPrivate;
        break;
    case 1:
        // 4-byte steps between slots of 12 bytes: neighbours overlap
        slot = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(2));
        ir.WriteShared(32, ir.IAdd(slot, ir.Imm32(8)), ir.Imm32(0));
        expected = Shader::ExtentUnknown::SharedNotPrivate;
        break;
    case 2:
        // An atomic on the loaded word
        static_cast<void>(ir.SharedAtomicIAdd(slot, ir.Imm32(1)));
        expected = Shader::ExtentUnknown::SharedPartialWrite;
        break;
    default:
        expected = Shader::ExtentUnknown::SharedNotDominated;
        break;
    }
    if (variant == 3) {
        // Loaded before anything is stored to it
        const IR::U32 early{ir.LoadShared(32, false, slot)};
        ir.WriteShared(32, slot, ir.IAdd(record, ir.Imm32(0xa0)));
        ir.WriteGlobal32(Address(ir, 0x160, early), ir.Imm32(0));
    } else {
        ir.WriteShared(32, slot, ir.IAdd(record, ir.Imm32(0xa0)));
        const IR::U32 loaded{ir.LoadShared(32, false, slot)};
        ir.WriteGlobal32(Address(ir, 0x160, loaded), ir.Imm32(0));
    }
    const auto& descriptor{fixture.Run(0x160)};
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x160)};
    CHECK_FALSE(extra.cbuf_extent.ids.bounded);
    CHECK_FALSE(extra.cbuf_extent.via_shared);
    CHECK(extra.shared_why.kind == expected);
}

TEST_CASE("A footprint reduces to its modulus' power-of-two part", "[shader][storage_extent]") {
    // 0x110-byte records, stores at +0x80..+0x8b and +0x108..+0x10b: modulo 0x10 they cover
    // +0x0..+0xb, so a read at +0x10 (residue 0) is inside and +0xc is still outside
    const std::array<Shader::StoreCongruence, 2> stores{{
        {.c = 0x80, .m = 0x110, .size = 12},
        {.c = 0x108, .m = 0x110, .size = 4},
    }};
    const Shader::StorageStoreFootprint footprint{
        Shader::MakeStoreFootprint(stores.data(), stores.size(), true)};
    REQUIRE(footprint.modulus == 0x110);
    const Shader::StorageStoreFootprint reduced{Shader::ReduceStoreFootprint(footprint)};
    REQUIRE(reduced.Valid());
    CHECK(reduced.modulus == 0x10);
    CHECK(reduced.PowerOfTwo());
    REQUIRE(reduced.num_intervals == 1);
    CHECK(reduced.intervals[0].begin == 0);
    CHECK(reduced.intervals[0].end == 0xc);
    CHECK(Shader::StoreFootprintHits(reduced, 0x110 * 3 + 0x10, 4));
    CHECK_FALSE(Shader::StoreFootprintHits(reduced, 0x10c, 4));
}

namespace {
/// FRITCHII_EXTENT_GUARD: a program of a header (entry), `if (cond) { body }` and a merge, as the
/// structured control flow builds it
struct BranchProgram : KernelProgram {
    IR::Block body{pool};
    IR::Block merge{pool};

    /// Links the blocks once the header and the body are filled in
    void Link(const IR::U1& cond) {
        entry.AddBranch(&body);
        entry.AddBranch(&merge);
        body.AddBranch(&merge);
        program.blocks = {&entry, &body, &merge};
        program.post_order_blocks = {&merge, &body, &entry};
        auto& list{program.syntax_list};
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = &entry;
        auto& if_node{list.emplace_back()};
        if_node.type = IR::AbstractSyntaxNode::Type::If;
        if_node.data.if_node.cond = cond;
        if_node.data.if_node.body = &body;
        if_node.data.if_node.merge = &merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = &body;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::EndIf;
        list.back().data.end_if.merge = &merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
        list.back().data.block = &merge;
        list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    }
};

/// A GPU append list's slot: the leader's atomic on the counter at c0[0x180] broadcast to the
/// subgroup, plus the lane's place among the active lanes
IR::U32 AppendIndex(IR::IREmitter& ir) {
    const IR::U32 ballot{ir.SubgroupBallot(ir.Imm1(true))};
    const IR::U32 old{ir.GlobalAtomicIAdd(Address(ir, 0x180, ir.Imm32(0)), ir.BitCount(ballot))};
    const IR::U32 first{ir.ShuffleIndex(old, ir.Imm32(0), ir.Imm32(31), ir.Imm32(0))};
    return ir.IAdd(first, ir.BitCount(ir.BitwiseAnd(ballot, ir.SubgroupLtMask())));
}
} // Anonymous namespace

TEST_CASE("An append index under a capacity guard is bounded by the capacity",
          "[shader][storage_extent]") {
    // 5ea74e193837ae71's lists: `if (!(index >= cap)) store [list + index * 4]`, the capacity
    // an immediate or a constant buffer word
    const bool cbuf_cap = GENERATE(false, true);
    BranchProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 index{AppendIndex(ir)};
    const IR::U32 cap{cbuf_cap ? Word(ir, 0x1a0) : ir.Imm32(0x400)};
    const IR::U1 full{ir.IGreaterThanEqual(index, cap, false)};
    const IR::U1 cond{ir.ConditionRef(ir.LogicalNot(full))};
    IR::IREmitter body{fixture.body};
    body.WriteGlobal32(Address(body, 0x170, body.ShiftLeftLogical(index, body.Imm32(2))),
                       body.Imm32(0));
    fixture.Link(cond);
    const auto& descriptor{fixture.Run(0x170)};
    REQUIRE(descriptor.is_written);
    // The plain extent stays unknown (defaults and the precompiled cache unchanged)
    CHECK_FALSE(descriptor.write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.write_unknown.kind != Shader::ExtentUnknown::Known);
    CHECK(extra.guard_why.kind == Shader::ExtentUnknown::Known);
    const Shader::StorageCbufExtent& extent{extra.cbuf_extent};
    REQUIRE(extent.ids.bounded);
    CHECK(extent.via_guard);
    CHECK_FALSE(extent.via_shared);
    CHECK(extent.num_floors == 0);
    CHECK(extent.ids.local_stride == std::array<u32, 3>{});
    CHECK(extent.ids.group_stride == std::array<u32, 3>{});
    const std::array<u32, Shader::MaxExtentTerms> none{};
    if (cbuf_cap) {
        // index <= c0[0x1a0] - 1: 4 * c0[0x1a0] bytes
        REQUIRE(extent.num_terms == 1);
        CHECK(extent.terms[0].cbuf_offset == 0x1a0);
        CHECK(extent.terms[0].coef == 4);
        CHECK(extent.ids.const_end == 0);
        const std::array<u32, Shader::MaxExtentTerms> values{0x100, 0, 0, 0};
        CHECK(Shader::ResolveStorageCbufExtent(extent, {64, 1, 1}, std::nullopt, values, none) ==
              0x400);
    } else {
        CHECK(extent.num_terms == 0);
        CHECK(extent.ids.const_end == 0x1000);
        CHECK(Shader::ResolveStorageCbufExtent(extent, {64, 1, 1}, std::nullopt, none, none) ==
              0x1000);
    }
}

TEST_CASE("An append index clamped to the room left is bounded by the capacity word",
          "[shader][storage_extent]") {
    // a107e12b6771be48: old = min(atomic, c0[0x78] - count), broadcast, plus the lane's place;
    // the subtraction as ISub or as a negated IADD operand
    const bool negated = GENERATE(false, true);
    KernelProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 ballot{ir.SubgroupBallot(ir.Imm1(true))};
    const IR::U32 count{ir.BitCount(ballot)};
    const IR::U32 old{ir.GlobalAtomicIAdd(Address(ir, 0x180, ir.Imm32(0)), count)};
    const IR::U32 room{negated ? IR::U32{ir.IAdd(IR::U32{ir.INeg(count)}, Word(ir, 0x78))}
                               : IR::U32{ir.ISub(Word(ir, 0x78), count)}};
    const IR::U32 first{
        ir.ShuffleIndex(ir.UMin(old, room), ir.Imm32(0), ir.Imm32(31), ir.Imm32(0))};
    const IR::U32 index{ir.IAdd(first, ir.BitCount(ir.BitwiseAnd(ballot, ir.SubgroupLtMask())))};
    ir.WriteGlobal32(Address(ir, 0x170, ir.ShiftLeftLogical(index, ir.Imm32(2))), ir.Imm32(0));
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.guard_why.kind == Shader::ExtentUnknown::Known);
    const Shader::StorageCbufExtent& extent{extra.cbuf_extent};
    REQUIRE(extent.ids.bounded);
    CHECK(extent.via_guard);
    // index <= c0[0x78] + 32 (a bit count): 4 * c0[0x78] + 0x84 bytes, while c0[0x78] >= 32
    REQUIRE(extent.num_terms == 1);
    CHECK(extent.terms[0].cbuf_offset == 0x78);
    CHECK(extent.terms[0].coef == 4);
    CHECK(extent.ids.const_end == 0x84);
    REQUIRE(extent.num_floors == 1);
    CHECK(extent.floors[0].cbuf_offset == 0x78);
    CHECK(extent.floors[0].min == 32);
    const std::array<u32, Shader::MaxExtentTerms> values{0x100, 0, 0, 0};
    const std::array<u32, Shader::MaxExtentTerms> none{};
    CHECK(Shader::ResolveStorageCbufExtent(extent, {64, 1, 1}, std::nullopt, values, none,
                                           std::array<u32, Shader::MaxExtentFloors>{0x100}) ==
          0x484);
    // A capacity below the largest count: the clamp's subtraction may wrap
    CHECK_FALSE(Shader::ResolveStorageCbufExtent(extent, {64, 1, 1}, std::nullopt, values, none,
                                                 std::array<u32, Shader::MaxExtentFloors>{16}));
}

TEST_CASE("Predicated moves into a guarded store's address follow its guard",
          "[shader][storage_extent]") {
    // `@!P mov off, index * 4 + 0x60; @!P st [list + off]`: the phi before the store also takes
    // off's old value (an unbounded one here) from the edge where P held, which an invocation
    // storing under !P never takes
    BranchProgram fixture;
    IR::Block second_body{fixture.pool};
    IR::Block second_merge{fixture.pool};
    IR::IREmitter ir{fixture.entry};
    const IR::U32 index{AppendIndex(ir)};
    const IR::U32 stale{ir.LoadGlobal32(Address(ir, 0x190, ir.Imm32(0)))};
    const IR::U1 full{ir.IGreaterThanEqual(index, ir.Imm32(0x400), false)};
    const IR::U1 first_cond{ir.ConditionRef(ir.LogicalNot(full))};
    IR::IREmitter body{fixture.body};
    const IR::U32 moved{body.IAdd(body.ShiftLeftLogical(index, body.Imm32(2)), body.Imm32(0x60))};
    // The first merge is the second branch's header
    IR::Inst* const phi{&*fixture.merge.PrependNewInst(fixture.merge.begin(), IR::Opcode::Phi)};
    phi->SetFlags(IR::Type::U32);
    phi->AddPhiOperand(&fixture.entry, stale);
    phi->AddPhiOperand(&fixture.body, moved);
    IR::IREmitter header{fixture.merge};
    const IR::U1 second_cond{header.ConditionRef(header.LogicalNot(full))};
    IR::IREmitter store{second_body};
    store.WriteGlobal32(Address(store, 0x170, IR::U32{IR::Value{phi}}), store.Imm32(0));
    fixture.Link(first_cond);
    fixture.merge.AddBranch(&second_body);
    fixture.merge.AddBranch(&second_merge);
    second_body.AddBranch(&second_merge);
    fixture.program.blocks = {&fixture.entry, &fixture.body, &fixture.merge, &second_body,
                              &second_merge};
    fixture.program.post_order_blocks = {&second_merge, &second_body, &fixture.merge,
                                         &fixture.body, &fixture.entry};
    auto& list{fixture.program.syntax_list};
    list.pop_back(); // the Return: the second branch follows
    auto& if_node{list.emplace_back()};
    if_node.type = IR::AbstractSyntaxNode::Type::If;
    if_node.data.if_node.cond = second_cond;
    if_node.data.if_node.body = &second_body;
    if_node.data.if_node.merge = &second_merge;
    list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    list.back().data.block = &second_body;
    list.emplace_back().type = IR::AbstractSyntaxNode::Type::EndIf;
    list.back().data.end_if.merge = &second_merge;
    list.emplace_back().type = IR::AbstractSyntaxNode::Type::Block;
    list.back().data.block = &second_merge;
    list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK(extra.guard_why.kind == Shader::ExtentUnknown::Known);
    const Shader::StorageCbufExtent& extent{extra.cbuf_extent};
    REQUIRE(extent.ids.bounded);
    CHECK(extent.via_guard);
    CHECK(extent.num_terms == 0);
    CHECK(extent.ids.const_end == 0x3ff * 4 + 0x60 + 4);
}

TEST_CASE("A guard that does not hold on the store's path bounds nothing",
          "[shader][storage_extent]") {
    // 0: the store after the merge, reachable around the guard; 1: a signed compare (a negative
    // index passes it); 2: the guard on another value than the index
    const int variant = GENERATE(0, 1, 2);
    BranchProgram fixture;
    IR::IREmitter ir{fixture.entry};
    const IR::U32 index{AppendIndex(ir)};
    IR::U1 inside{};
    Shader::ExtentUnknown expected{};
    switch (variant) {
    case 0:
        inside = ir.ILessThan(index, ir.Imm32(0x400), false);
        expected = Shader::ExtentUnknown::GuardNotDominating;
        break;
    case 1:
        inside = ir.ILessThan(index, ir.Imm32(0x400), true);
        expected = Shader::ExtentUnknown::GuardSigned;
        break;
    default: {
        const IR::U32 other{ir.GlobalAtomicIAdd(Address(ir, 0x180, ir.Imm32(4)), ir.Imm32(1))};
        inside = ir.ILessThan(other, ir.Imm32(0x400), false);
        expected = Shader::ExtentUnknown::AtomicResult;
        break;
    }
    }
    const IR::U1 cond{ir.ConditionRef(inside)};
    IR::IREmitter store{variant == 0 ? fixture.merge : fixture.body};
    store.WriteGlobal32(Address(store, 0x170, store.ShiftLeftLogical(index, store.Imm32(2))),
                        store.Imm32(0));
    fixture.Link(cond);
    CHECK_FALSE(fixture.Run(0x170).write_extent.bounded);
    const Shader::StorageWriteExtra& extra{fixture.Extra(0x170)};
    CHECK_FALSE(extra.cbuf_extent.ids.bounded);
    CHECK_FALSE(extra.cbuf_extent.via_guard);
    CHECK(extra.guard_why.kind == expected);
}
