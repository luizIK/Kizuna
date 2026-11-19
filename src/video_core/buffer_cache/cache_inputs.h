// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>

#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

/// Memory stage step 5 (first half): everything the buffer cache reads from the 3D and compute
/// engines, the draw manager and the rasterizer by itself, and the game memory words it reads
/// directly, goes through this one accessor. The cache's dirty-flag reads and writes do too.
///
/// Today it is a view on the bound channel's engines, made per use (BufferCache::Inputs()), so it
/// reads the same registers at the same moments as the code it replaced. In step 6 the cache moves
/// to the memory stage's thread, where the back owns the engines: the second half of step 5 feeds
/// these values from the stream instead, and step 7's verify hooks the game memory reads here.
///
/// Not routed here, on purpose: address translation and the other page table queries
/// (GpuToCpuAddress, MaxContinuousRange, MapSerial, ...), which the stage owns with the page
/// table, and the content reads of game memory through device_memory (uploads), which the read log
/// of sub-step 2b covers.
///
/// It holds pointers to the five things it reads (registers, constant buffer bindings, root
/// tables, draw state, dirty flags; piece 5c, fable_memstage_6b_plan.md 5.4): bound to an engine,
/// or to the memory stage's copy of one (OfStage, FRITCHII_MEMSTAGE_SHADOW=3).
class CacheInputs {
public:
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using KeplerCompute = Tegra::Engines::KeplerCompute;
    using Regs = Maxwell3D::Regs;
    using DirtyFlags = Maxwell3D::DirtyState::Flags;
    using DrawManagerState = Maxwell3D::DrawManager::State;

    CacheInputs(Maxwell3D* maxwell3d_, KeplerCompute* kepler_compute_,
                Tegra::MemoryManager* gpu_memory_,
                const Maxwell3D::DrawManager::IndirectParams* draw_indirect_) noexcept
        : maxwell3d{maxwell3d_}, regs{maxwell3d_ ? &maxwell3d_->regs : nullptr},
          state{maxwell3d_ ? &maxwell3d_->state : nullptr},
          root_tables{maxwell3d_ ? &maxwell3d_->root_tables : nullptr},
          draw_state{maxwell3d_ ? &maxwell3d_->draw_manager.draw_state : nullptr},
          flags{maxwell3d_ ? &maxwell3d_->dirty.flags : nullptr}, kepler_compute{kepler_compute_},
          gpu_memory{gpu_memory_}, draw_indirect{draw_indirect_} {
        // FRITCHII_ESPLIT's queued mode, on E2: the buffer flags of E1's engine are E2's view's
        // (fritchii_e_split_thread.h), and E1 writes that engine meanwhile. A mark outside a fill
        // (the scrub of a deleted buffer's bindings, BufferCache::DeleteBuffer through
        // CacheInputs::Of) goes to the view's flags then, never to E1's engine.
        if (ESplit::g_async && maxwell3d_ != nullptr) [[unlikely]] {
            if (void* const view_flags = ESplit::E2FlagsFor(maxwell3d_)) {
                flags = static_cast<DirtyFlags*>(view_flags);
            }
        }
    }

    /// The inputs from the memory stage's copy of `maxwell3d_` (memstage_shadow.cpp). The engine
    /// itself still answers NativeTuring (fixed when its channel is made).
    [[nodiscard]] static CacheInputs OfStage(
        Maxwell3D* maxwell3d_, const Regs& regs_, const Maxwell3D::State& state_,
        const Maxwell3D::RootTables& root_tables_, const DrawManagerState& draw_state_,
        DirtyFlags& flags_, KeplerCompute* kepler_compute_, Tegra::MemoryManager* gpu_memory_,
        const Maxwell3D::DrawManager::IndirectParams* draw_indirect_) noexcept {
        CacheInputs inputs{maxwell3d_, kepler_compute_, gpu_memory_, draw_indirect_};
        inputs.regs = &regs_;
        inputs.state = &state_;
        inputs.root_tables = &root_tables_;
        inputs.draw_state = &draw_state_;
        inputs.flags = &flags_;
        inputs.stage_copy = true;
        return inputs;
    }

    /// Set while FRITCHII_MEMSTAGE_SHADOW is on: the marks the cache makes on an engine's flags
    /// (Mark*, on an engine's inputs) also reach the stage's copy of that engine.
    static inline void (*g_mark_hook)(const Maxwell3D* engine, bool index_buffer,
                                      u32 vertex_mask) = nullptr;

    /// The inputs of another channel than the bound one (the scrub of a deleted buffer's
    /// bindings marks every channel's dirty flags). It has no indirect draw.
    [[nodiscard]] static CacheInputs Of(ChannelInfo& channel) noexcept {
        return CacheInputs{&channel.maxwell3d, &channel.kepler_compute, &channel.gpu_memory,
                           nullptr};
    }

    // --- The draw manager ---

    /// Topology, vertex and index buffer of the draw, and its inline index bytes.
    [[nodiscard]] const Maxwell3D::DrawManager::State& DrawState() const noexcept {
        return *draw_state;
    }

    /// The indirect draw's parameters the rasterizer handed in (SetDrawIndirect), or null.
    [[nodiscard]] const Maxwell3D::DrawManager::IndirectParams* DrawIndirect() const noexcept {
        return draw_indirect;
    }

    // --- 3D registers ---

    [[nodiscard]] const Regs::VertexStream& VertexStream(u32 index) const noexcept {
        return regs->vertex_streams[index];
    }

    [[nodiscard]] const Regs::VertexStreamLimit& VertexStreamLimit(u32 index) const noexcept {
        return regs->vertex_stream_limits[index];
    }

    [[nodiscard]] u32 VertexStreamStride(u32 index) const noexcept {
        return regs->vertex_streams[index].stride;
    }

    [[nodiscard]] u64 NativeVertexStreamSize(u32 index) const {
        return regs->NativeVertexStreamSize(index);
    }

    [[nodiscard]] bool TransformFeedbackEnabled() const noexcept {
        return regs->transform_feedback_enabled != 0;
    }

    [[nodiscard]] const Regs::TransformFeedback::Control& TransformFeedbackControl(
        u32 index) const noexcept {
        return regs->transform_feedback.controls[index];
    }

    [[nodiscard]] const Regs::TransformFeedback::Buffer& TransformFeedbackBuffer(
        u32 index) const noexcept {
        return regs->transform_feedback.buffers[index];
    }

    /// Constant buffer `index` of graphics stage `stage` as the engine has it bound.
    [[nodiscard]] const Tegra::Engines::ConstBufferInfo& StageConstBuffer(
        size_t stage, u32 index) const noexcept {
        return state->shader_stages[stage].const_buffers[index];
    }

    /// The registers the Vulkan vertex repack layout is chosen from (vertex_attrib_format and
    /// vertex_streams strides). Read by the rasterizer for BufferCacheRuntime::SetVertexRepackLayout.
    [[nodiscard]] const Regs& RepackLayoutRegs() const noexcept {
        return *regs;
    }

    // --- The macro engine ---

    /// Native Turing macro mode: native vertex stream sizes and bare graphics storage pointers.
    [[nodiscard]] bool NativeTuring() const {
        return maxwell3d->GetNativeMacroEngine().native_turing;
    }

    /// Root tables (NVN2's push constants in table 0, read as uniform bank 17).
    [[nodiscard]] const Maxwell3D::RootTables& RootTables() const noexcept {
        return *root_tables;
    }

    // --- Dirty flags, read and set ---

    /// The 3D engine's dirty flags. The cache reads and clears IndexBuffer, VertexBuffers and
    /// VertexBuffer0..31, and sets them again when a buffer it bound goes away or is repacked.
    [[nodiscard]] DirtyFlags& Dirty() const noexcept {
        return *flags;
    }

    /// Marks the vertex buffers in `mask` (bit n = VertexBuffer0 + n) and the vertex buffer set
    /// dirty, so the next update rebinds them.
    void MarkVertexBuffersDirty(u32 mask) const noexcept {
        DirtyFlags& dirty = *flags;
        dirty[VideoCommon::Dirty::VertexBuffers] = true;
        for (u32 binding = 0; binding < 32; ++binding) {
            if (mask & (1U << binding)) {
                dirty[VideoCommon::Dirty::VertexBuffer0 + binding] = true;
            }
        }
        if (g_mark_hook != nullptr && !stage_copy) [[unlikely]] {
            g_mark_hook(maxwell3d, false, mask);
        }
    }

    /// Marks the index buffer dirty, so the next update binds it again.
    void MarkIndexBufferDirty() const noexcept {
        (*flags)[VideoCommon::Dirty::IndexBuffer] = true;
        if (g_mark_hook != nullptr && !stage_copy) [[unlikely]] {
            g_mark_hook(maxwell3d, true, 0);
        }
    }

    // --- Compute ---

    [[nodiscard]] const KeplerCompute::LaunchParams& LaunchDescription() const noexcept {
        return kepler_compute->launch_description;
    }

    [[nodiscard]] bool ComputeNativeMode() const noexcept {
        return kepler_compute->IsNativeMode();
    }

    [[nodiscard]] std::optional<GPUVAddr> IndirectComputeAddress() const {
        return kepler_compute->GetIndirectComputeAddress();
    }

    // --- Game memory the cache reads directly ---

    /// A 64-bit word of game memory: storage buffer pointers and size words, table entries.
    [[nodiscard]] u64 ReadGameU64(GPUVAddr address) const {
        return gpu_memory->Read<u64>(address);
    }
    /// A 32-bit word of game memory: FRITCHII_EXTENT_CBUF's constant buffer words.
    [[nodiscard]] u32 ReadGameU32(GPUVAddr address) const {
        return gpu_memory->Read<u32>(address);
    }

private:
    Maxwell3D* maxwell3d;
    const Regs* regs;
    const Maxwell3D::State* state;
    const Maxwell3D::RootTables* root_tables;
    const DrawManagerState* draw_state;
    DirtyFlags* flags;
    KeplerCompute* kepler_compute;
    Tegra::MemoryManager* gpu_memory;
    const Maxwell3D::DrawManager::IndirectParams* draw_indirect;
    bool stage_copy{};
};

} // namespace VideoCommon
