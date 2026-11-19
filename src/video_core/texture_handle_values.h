// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The raw texture handles a draw or dispatch binds, made one way for both sides of the memory
// seam (fable_memstage_step2_spec.md 6.3 and 9.1). The fill calls the Fill* functions and keeps
// the values in the packet; the back's handle loops take them from there (SEAM >= 3) or call
// RawTextureHandle themselves (SEAM < 3, and VERIFY's own read).

#pragma once

#include <cstddef>
#include <cstdlib>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memstage_stats.h"
#include "video_core/shader_handle_program.h"

namespace VideoCommon {

/// A value in the handle list for an element no handle is read for (EXP21's buffer address
/// words): keeps one value per image element.
inline constexpr u32 SKIPPED_HANDLE_VALUE = ~0U;

/// A graphics stage's bound constant buffers (Maxwell3D's const_buffers array) as the handle
/// loops see them. operator() is the address a handle program reads through: 0 when the buffer
/// is not bound.
template <typename Cbufs>
struct GraphicsHandleCbufs {
    const Cbufs& cbufs;

    GPUVAddr operator()(u32 cbuf) const {
        return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address : GPUVAddr{};
    }
    bool Enabled(u32 cbuf) const {
        return cbufs[cbuf].enabled;
    }
    GPUVAddr Base(u32 cbuf) const {
        return cbufs[cbuf].address;
    }
};

/// The QMD's constant buffers (KeplerCompute::launch_description), as above.
template <typename Qmd>
struct ComputeHandleCbufs {
    const Qmd& qmd;

    GPUVAddr operator()(u32 cbuf) const {
        return cbuf < qmd.const_buffer_config.size() &&
                       ((qmd.const_buffer_enable_mask >> cbuf) & 1) != 0
                   ? qmd.const_buffer_config[cbuf].Address()
                   : GPUVAddr{};
    }
    bool Enabled(u32 cbuf) const {
        return ((qmd.const_buffer_enable_mask >> cbuf) & 1) != 0;
    }
    GPUVAddr Base(u32 cbuf) const {
        return qmd.const_buffer_config[cbuf].Address();
    }
};

/// The storage alignment a single handle program reads with. EDEN_NX2_HANDLE_ALIGN=0 reads the
/// words at the unaligned address instead (A/B); handle tables (arrays) always align.
[[nodiscard]] inline u64 HandleProgramAlignment(u64 storage_alignment) {
    static const bool align_storage{[] {
        const char* const value{std::getenv("EDEN_NX2_HANDLE_ALIGN")};
        return value == nullptr || *value != '0';
    }()};
    return align_storage ? storage_alignment : 1;
}

/// What a verify mismatch logs as the address: the handle word RawTextureHandle reads first for
/// element `index`, or for a handle program bit 63 set, the program in bits 32-62 and the element
/// in bits 0-31.
template <typename Descriptor, typename Cbufs>
[[nodiscard]] u64 HandleWordAddress(const Descriptor& desc, u32 index, const Cbufs& cbufs) {
    if (desc.handle_program != ~0U) {
        return (u64{1} << 63) | (u64{desc.handle_program} << 32) | index;
    }
    return cbufs.Base(desc.cbuf_index) + (desc.cbuf_offset + (index << desc.size_shift));
}

/// The raw handle of element `index` of `desc`: a handle program's result, one handle word, or
/// two words shifted and combined. The reads of the back's old read_handle lambdas, in their
/// order. `Compute` picks the MemStageStats kinds.
template <bool Compute, typename Descriptor, typename Memory, typename Cbufs>
[[nodiscard]] u32 RawTextureHandle(const Shader::Info& info, const Descriptor& desc, u32 index,
                                   Memory& gpu_memory, const Cbufs& cbufs,
                                   u64 storage_alignment) {
    namespace Stats = MemStageStats;
    if (desc.handle_program != ~0U) {
        // EDEN_NX2_MEMSTAGE_STATS: a handle program (its storage reads are not sized).
        Stats::Read(Compute ? Stats::Kind::CHandleProgram : Stats::Kind::HandleProgram, 0);
        return EvaluateTextureHandle(info, desc.handle_program, gpu_memory, cbufs,
                                     HandleProgramAlignment(storage_alignment), index);
    }
    ASSERT(cbufs.Enabled(desc.cbuf_index));
    const u32 index_offset{index << desc.size_shift};
    const u32 offset{desc.cbuf_offset + index_offset};
    const GPUVAddr addr{cbufs.Base(desc.cbuf_index) + offset};
    if constexpr (std::is_same_v<Descriptor, Shader::TextureDescriptor> ||
                  std::is_same_v<Descriptor, Shader::TextureBufferDescriptor>) {
        if (desc.has_secondary) {
            ASSERT(cbufs.Enabled(desc.secondary_cbuf_index));
            const u32 second_offset{desc.secondary_cbuf_offset};
            const GPUVAddr separate_addr{cbufs.Base(desc.secondary_cbuf_index) + second_offset};
            const u32 lhs_raw{gpu_memory.ReadHandleWord(addr) << desc.shift_left};
            const u32 rhs_raw{gpu_memory.ReadHandleWord(separate_addr)
                              << desc.secondary_shift_left};
            // EDEN_NX2_MEMSTAGE_STATS: two handle words.
            Stats::Read(Compute ? Stats::Kind::CHandleWord : Stats::Kind::HandleWord,
                        2 * sizeof(u32), 2);
            return lhs_raw | rhs_raw;
        }
    }
    // EDEN_NX2_MEMSTAGE_STATS: one handle word.
    Stats::Read(Compute ? Stats::Kind::CHandleWord : Stats::Kind::HandleWord, sizeof(u32));
    return gpu_memory.ReadHandleWord(addr);
}

/// The raw handles of a handle table (an array descriptor with a handle program), element by
/// element, passed to output(element, raw).
template <typename Descriptor, typename Memory, typename Cbufs, typename Output>
void RawTextureHandleTable(const Shader::Info& info, const Descriptor& desc, Memory& gpu_memory,
                           const Cbufs& cbufs, u64 storage_alignment, Output&& output) {
    EvaluateTextureHandles(info, desc.handle_program, gpu_memory, cbufs, storage_alignment,
                           desc.count, output);
}

/// Every raw handle one stage binds, in the order its handle loop reads them: texture buffers,
/// image buffers, textures (tables expanded), images; one value per image element. Elements EXP21
/// binds no texture for (`null_pointer_handles`) read nothing and get SKIPPED_HANDLE_VALUE.
template <bool Compute, typename Memory, typename Cbufs, typename Push>
void FillStageHandleValues(const Shader::Info& info, Memory& gpu_memory, const Cbufs& cbufs,
                           u64 storage_alignment, bool null_pointer_handles, Push&& push) {
    const auto add = [&](const auto& desc) {
        for (u32 index = 0; index < desc.count; ++index) {
            push(RawTextureHandle<Compute>(info, desc, index, gpu_memory, cbufs,
                                           storage_alignment));
        }
    };
    for (const auto& desc : info.texture_buffer_descriptors) {
        add(desc);
    }
    for (const auto& desc : info.image_buffer_descriptors) {
        add(desc);
    }
    for (const auto& desc : info.texture_descriptors) {
        if (desc.count > 1 && desc.handle_program != ~0U) {
            RawTextureHandleTable(info, desc, gpu_memory, cbufs, storage_alignment,
                                  [&](u32, u32 raw) { push(raw); });
            continue;
        }
        for (u32 index = 0; index < desc.count; ++index) {
            if (null_pointer_handles && HandleIsStoragePointer(info, desc, index)) {
                push(SKIPPED_HANDLE_VALUE);
            } else {
                push(RawTextureHandle<Compute>(info, desc, index, gpu_memory, cbufs,
                                               storage_alignment));
            }
        }
    }
    for (const auto& desc : info.image_descriptors) {
        add(desc);
    }
}

/// Whether a stage's handle loop reads anything: the stages a draw's fill visits.
[[nodiscard]] inline bool StageReadsHandles(const Shader::Info& info) {
    return !info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty() ||
           !info.texture_descriptors.empty() || !info.image_descriptors.empty();
}

/// Every raw handle a graphics draw binds: FillStageHandleValues for each stage in stage order.
/// cbufs_of(stage) gives that stage's GraphicsHandleCbufs.
template <typename Memory, typename CbufsOf, typename Push>
void FillGraphicsHandleValues(std::span<const Shader::Info> stage_infos, Memory& gpu_memory,
                              CbufsOf&& cbufs_of, u64 storage_alignment, bool null_pointer_handles,
                              Push&& push) {
    for (size_t stage = 0; stage < stage_infos.size(); ++stage) {
        const Shader::Info& info{stage_infos[stage]};
        if (StageReadsHandles(info)) {
            FillStageHandleValues<false>(info, gpu_memory, cbufs_of(stage), storage_alignment,
                                         null_pointer_handles, push);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// FRITCHII_HANDLE_PROG_KEEP (fable_b 25.2 cut g): handle program results kept by their reads.

/// One handle program evaluation of a draw's fill (a single element, or a whole table) and what
/// it read: each constant buffer address it read through and each word (GPU address and size,
/// once each) with its value. While all of them read the same again the evaluation gives the
/// same results: the program is a pure function of what it reads, and every address it reads is
/// a constant buffer address or comes from earlier reads, all of which are equal, so each read is
/// the same read (by induction over the program's steps).
struct HandleProgramRecord {
    struct Read {
        GPUVAddr address;
        u64 value;
        u32 size;
    };
    static constexpr size_t MaxBases = 8;
    static constexpr size_t MaxReads = 32;

    const u32* code{}; ///< the program's words (with code_size: which program this record is)
    size_t code_size{};
    u32 element{}; ///< a single evaluation's element (0 for a table)
    u32 count{};   ///< results: 1, or the table's element count
    u32 num_bases{};
    u32 num_reads{};
    bool valid{}; ///< false: never recorded, or more reads than a record holds
    std::array<std::pair<u32, GPUVAddr>, MaxBases> bases{};
    std::array<Read, MaxReads> reads{};
    std::vector<u32> results;

    void Begin(const u32* code_, size_t code_size_, u32 element_, u32 count_) {
        code = code_;
        code_size = code_size_;
        element = element_;
        count = count_;
        num_bases = 0;
        num_reads = 0;
        valid = true;
        results.clear();
    }
    void AddBase(u32 index, GPUVAddr base) {
        for (u32 i = 0; i < num_bases; ++i) {
            if (bases[i].first == index) {
                return;
            }
        }
        if (num_bases == MaxBases) {
            valid = false;
            return;
        }
        bases[num_bases++] = {index, base};
    }
    void AddRead(GPUVAddr address, u32 size, u64 value) {
        for (u32 i = 0; i < num_reads; ++i) {
            if (reads[i].address == address && reads[i].size == size) {
                return;
            }
        }
        if (num_reads == MaxReads) {
            valid = false;
            return;
        }
        reads[num_reads++] = {address, value, size};
    }
};

/// The memory a recording evaluation reads: each Read<T> forwarded and recorded.
template <typename Memory>
struct HandleRecordingMemory {
    Memory& memory;
    HandleProgramRecord& record;

    template <typename T>
    [[nodiscard]] T Read(GPUVAddr address) const {
        const T value = memory.template Read<T>(address);
        record.AddRead(address, static_cast<u32>(sizeof(T)), static_cast<u64>(value));
        return value;
    }
};

/// The constant buffer addresses a recording evaluation reads through, recorded.
template <typename Cbufs>
struct HandleRecordingCbufs {
    const Cbufs& cbufs;
    HandleProgramRecord& record;

    GPUVAddr operator()(u32 cbuf) const {
        const GPUVAddr base = cbufs(cbuf);
        record.AddBase(cbuf, base);
        return base;
    }
};

enum class HandleProgramVerdict : u8 {
    Same,    ///< every constant buffer address and word read as recorded: the results hold
    Moved,   ///< a constant buffer it reads through is bound elsewhere
    Changed, ///< a word it read holds another value
    First,   ///< no usable record
};

/// Reads what `record` read again, through the same memory reads the evaluation makes.
template <typename Memory, typename Cbufs>
[[nodiscard]] HandleProgramVerdict CheckHandleProgramRecord(const HandleProgramRecord& record,
                                                            Memory& gpu_memory,
                                                            const Cbufs& cbufs) {
    if (!record.valid) {
        return HandleProgramVerdict::First;
    }
    for (u32 i = 0; i < record.num_bases; ++i) {
        if (cbufs(record.bases[i].first) != record.bases[i].second) {
            return HandleProgramVerdict::Moved;
        }
    }
    for (u32 i = 0; i < record.num_reads; ++i) {
        const HandleProgramRecord::Read& read = record.reads[i];
        const u64 value = read.size == sizeof(u64)
                              ? gpu_memory.template Read<u64>(read.address)
                              : u64{gpu_memory.template Read<u32>(read.address)};
        if (value != read.value) {
            return HandleProgramVerdict::Changed;
        }
    }
    return HandleProgramVerdict::Same;
}

/// FillGraphicsHandleValues with each handle program evaluation handed to
/// program(info, program_index, cbufs, element, count, alignment), which pushes `count` values:
/// a single element (count 1, HandleProgramAlignment's alignment, as RawTextureHandle; after its
/// stats read) or a table (count elements from 0, the storage alignment itself, as
/// RawTextureHandleTable). Every other value as FillStageHandleValues, in its order.
template <typename Memory, typename CbufsOf, typename Push, typename Program>
void FillGraphicsHandleValuesWith(std::span<const Shader::Info> stage_infos, Memory& gpu_memory,
                                  CbufsOf&& cbufs_of, u64 storage_alignment,
                                  bool null_pointer_handles, Push&& push, Program&& program) {
    namespace Stats = MemStageStats;
    const u64 single_alignment = HandleProgramAlignment(storage_alignment);
    for (size_t stage = 0; stage < stage_infos.size(); ++stage) {
        const Shader::Info& info{stage_infos[stage]};
        if (!StageReadsHandles(info)) {
            continue;
        }
        const auto cbufs = cbufs_of(stage);
        const auto one = [&](const auto& desc, u32 index) {
            if (desc.handle_program != ~0U) {
                Stats::Read(Stats::Kind::HandleProgram, 0);
                program(info, desc.handle_program, cbufs, index, 1U, single_alignment);
            } else {
                push(RawTextureHandle<false>(info, desc, index, gpu_memory, cbufs,
                                             storage_alignment));
            }
        };
        const auto add = [&](const auto& desc) {
            for (u32 index = 0; index < desc.count; ++index) {
                one(desc, index);
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add(desc);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add(desc);
        }
        for (const auto& desc : info.texture_descriptors) {
            if (desc.count > 1 && desc.handle_program != ~0U) {
                program(info, desc.handle_program, cbufs, 0U, desc.count, storage_alignment);
                continue;
            }
            for (u32 index = 0; index < desc.count; ++index) {
                if (null_pointer_handles && HandleIsStoragePointer(info, desc, index)) {
                    push(SKIPPED_HANDLE_VALUE);
                } else {
                    one(desc, index);
                }
            }
        }
        for (const auto& desc : info.image_descriptors) {
            add(desc);
        }
    }
}

} // namespace VideoCommon
