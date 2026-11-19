// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// FRITCHII_ONEF_FRONT_OPS (fable_b 27/28, stages S2 and S3): seven of OoT's hot 3D macros
// (macro_hle_oot.h) sent as one ring entry each, "op + resolved arguments", instead of one entry
// (or Pairs word) per register write. The front still runs the handler on its own state (shadow,
// same-value skip, executable bookkeeping) and records which writes it would have sent; E (and
// the memory stage) rebuild the write list from the arguments with Expand below and apply the
// sent ones exactly as Execute applies the entries they replace.
//
// The arguments carry what the handler read from the data RAM: E has no data RAM of its own (the
// front's native_data_ram is the macros' while they run there), so Encode reads it on the front
// before the handler runs (none of the seven writes a word its own reads use).

#include <array>
#include <span>

#include "common/common_types.h"
#include "video_core/macro_hle_oot.h"

namespace Tegra::OotOps {

enum class Op : u8 {
    None = 0,
    E02 = 1, ///< constant buffer bind: args p1, p2, p3, stage (dram[0x520 + p0] when p0 >= 8)
    E1C = 2, ///< constant buffer select: args dram[0x543], dram[0x53c + p0]
    E36 = 3, ///< constant buffer select at an offset: args dram[0x537], p0
    EA0 = 4, ///< instanced draw with the c[0][0x640] words: args p0..p5
    E08 = 5, ///< a stage's program: args p2, p3, p4 (sent only when the handler wrote)
    E52 = 6, ///< instanced draw: args p0..p5
    E3A = 7, ///< a stage's enable: args bits ((p0 >> 4) & 7) << 4, enabled (dram[0x52c] bit p0 & 15)
    Count = 8,
};

inline constexpr u32 MaxArgs = 6;
/// Writes one op may expand to (the sent and pair masks are one u64 each).
inline constexpr u32 MaxWrites = 64;
/// The instanced draws are ops only up to this many instances (EA0: 13 + 3 * 16 = 61 writes).
inline constexpr u32 MaxOpInstances = 16;

struct Args {
    Op op{};
    u32 count{};
    std::array<u32, MaxArgs> a{};
};

[[nodiscard]] constexpr Op OpOf(OotMacros::Id id) noexcept {
    switch (id) {
    case OotMacros::Id::E02:
        return Op::E02;
    case OotMacros::Id::E1C:
        return Op::E1C;
    case OotMacros::Id::E36:
        return Op::E36;
    case OotMacros::Id::EA0:
        return Op::EA0;
    case OotMacros::Id::E08:
        return Op::E08;
    case OotMacros::Id::E52:
        return Op::E52;
    case OotMacros::Id::E3A:
        return Op::E3A;
    default:
        return Op::None;
    }
}

[[nodiscard]] constexpr const char* OpName(Op op) noexcept {
    constexpr const char* names[]{"none", "e02", "e1c", "e36", "ea0", "e08", "e52", "e3a"};
    const auto index = static_cast<size_t>(op);
    return index < std::size(names) ? names[index] : "?";
}

/// Front, before the handler runs on `dram`: the arguments of the op for this call, or false
/// when the call stays on the old path (the handler would decline it, or it is too long).
/// `max_instances`: the instanced draws' limit (1 while EDEN_NX2_FOLD_INSTANCES counts later
/// instances on the front: those writes keep their own place).
[[nodiscard]] inline bool Encode(OotMacros::Id id, std::span<const u32> p, const u32* dram,
                                 Args& out, u32 max_instances = MaxOpInstances) noexcept {
    using OotMacros::DataRamWords;
    const auto n = p.size();
    out.op = OpOf(id);
    switch (out.op) {
    case Op::E02: {
        if (n < 4) {
            return false;
        }
        u32 stage = p[0];
        if (static_cast<s32>(p[0]) >= 8) {
            if (0x520ULL + p[0] >= DataRamWords) {
                return false;
            }
            stage = dram[0x520 + p[0]];
        }
        out.count = 4;
        out.a = {p[1], p[2], p[3], stage, 0, 0};
        return true;
    }
    case Op::E1C:
        if (n < 1 || 0x53cULL + p[0] >= DataRamWords) {
            return false;
        }
        out.count = 2;
        out.a = {dram[0x543], dram[0x53c + p[0]], 0, 0, 0, 0};
        return true;
    case Op::E36:
        if (n < 1) {
            return false;
        }
        out.count = 2;
        out.a = {dram[0x537], p[0], 0, 0, 0, 0};
        return true;
    case Op::E08:
        if (n < 5 || 0x500ULL + u64{p[0]} * 4 + 3 >= DataRamWords) {
            return false;
        }
        out.count = 3;
        out.a = {p[2], p[3], p[4], 0, 0, 0};
        return true;
    case Op::E3A: {
        if (n < 1) {
            return false;
        }
        const u32 stage = p[0] & 15;
        out.count = 2;
        out.a = {((p[0] >> 4) & 7) << 4, (dram[0x52c] & (1U << stage)) != 0 ? 1U : 0U, 0, 0, 0, 0};
        return true;
    }
    case Op::EA0:
    case Op::E52:
        if (n < 6 || p[2] > max_instances || p[2] > MaxOpInstances) {
            return false;
        }
        out.count = 6;
        out.a = {p[0], p[1], p[2], p[3], p[4], p[5]};
        return true;
    default:
        return false;
    }
}

namespace Detail {
template <typename F>
inline void DrawLoop(F& out, u32 begin, u32 count, u32 instances) {
    out(0x586, begin);
    out(0x5f8, count);
    out(0x585, 0);
    const u32 subsequent = (begin & ~(3U << 26)) | (1U << 26);
    for (u32 i = 1; i < instances; ++i) {
        out(0x586, subsequent);
        out(0x5f8, count);
        out(0x585, 0);
    }
}
} // namespace Detail

/// The op's writes in order, `out(method, argument)` for each, as the handler emits them when it
/// emits any (macro_hle_oot.h). Returns how many. E08's are the writes of a call that wrote.
template <typename F>
inline u32 Expand(Op op, const u32* a, F&& out) {
    u32 written = 0;
    const auto emit = [&](u32 method, u32 value) {
        ++written;
        out(method, value);
    };
    switch (op) {
    case Op::E02: {
        const u32 p1 = a[0], p2 = a[1], p3 = a[2], stage = a[3];
        if (static_cast<s32>(stage) >= 15) {
            break;
        }
        emit(0x8e0, (p3 >> 12) & 0xfffff);
        emit(0x8e1, p3 & 0xff);
        emit(0x8e2, p2);
        emit(0x904 | ((stage & 7) << 3), 1 | ((p1 + 3) << 4));
        break;
    }
    case Op::E1C:
        emit(0x8e0, a[0]);
        emit(0x8e1, a[1] >> 24);
        emit(0x8e2, (a[1] & 0xffffff) << 8);
        break;
    case Op::E36:
        emit(0x8e0, 0x900);
        emit(0x8e1, a[0] >> 24);
        emit(0x8e2, (a[0] & 0xffffff) << 8);
        emit(0x8e3, a[1] << 2);
        break;
    case Op::E08: {
        const u32 p2 = a[0], p3 = a[1], p4 = a[2];
        const u32 stage = ((p4 >> 24) & 7) << 4;
        emit(0x805 | stage, p2 >> 24);
        emit(0x806 | stage, (p2 & 0xffffff) << 8);
        emit(0x807 | stage, (p4 >> 9) & 0x7f);
        emit(0x8e0, 0x10000);
        emit(0x8e1, p3 >> 24);
        emit(0x8e2, (p3 & 0xffffff) << 8);
        emit(0x904 | (((p4 >> 28) & 7) << 3), 0x11);
        emit(0x803 | stage, p4 & 0x1ff);
        emit(0x800 | stage, 1 | stage);
        break;
    }
    case Op::E3A:
        if (a[1] != 0) {
            emit(0x800 | a[0], a[0]);
        }
        break;
    case Op::EA0:
    case Op::E52: {
        const bool ea0 = op == Op::EA0;
        emit(0x5f7, a[3]);
        emit(0x446, a[4]);
        emit(0x50d, a[4]);
        emit(0x50e, a[5]);
        if (ea0) {
            emit(0x8e3, 0x640);
            emit(0x8e4, a[4]);
            emit(0x8e5, a[5]);
        }
        if (a[2] != 0) {
            Detail::DrawLoop(emit, a[0], a[1], a[2]);
        }
        emit(0x446, 0);
        emit(0x50d, 0);
        emit(0x50e, 0);
        if (ea0) {
            emit(0x8e3, 0x640);
            emit(0x8e4, 0);
            emit(0x8e5, 0);
        }
        break;
    }
    default:
        break;
    }
    return written;
}

/// The op's payload after [engine, system]: word 0 the header below, word 1 the sent mask (bit i:
/// write i of Expand was sent), word 2 the pair mask (bit i: sent as a Pairs word, else as a
/// Direct method entry), then the arguments two to a word.
inline constexpr size_t HeaderWords = 3;
[[nodiscard]] constexpr u64 Header(Op op, u32 args, u32 writes, u32 checks, u32 flags) noexcept {
    return u64{static_cast<u8>(op)} | (u64{args & 0xff} << 8) | (u64{writes & 0xff} << 16) |
           (u64{checks & 0xff} << 24) | (u64{flags} << 32);
}
[[nodiscard]] constexpr Op HeaderOp(u64 h) noexcept {
    return static_cast<Op>(h & 0xff);
}
[[nodiscard]] constexpr u32 HeaderArgs(u64 h) noexcept {
    return static_cast<u32>((h >> 8) & 0xff);
}
[[nodiscard]] constexpr u32 HeaderWrites(u64 h) noexcept {
    return static_cast<u32>((h >> 16) & 0xff);
}
/// With FlagCheck: after the arguments, HeaderWrites (method | argument << 32) words, the front's
/// whole write list (sent or not), then HeaderChecks (method | value << 32) words, each written
/// register once with the value the front's copy holds after the call.
[[nodiscard]] constexpr u32 HeaderChecks(u64 h) noexcept {
    return static_cast<u32>((h >> 24) & 0xff);
}
[[nodiscard]] constexpr u32 HeaderFlags(u64 h) noexcept {
    return static_cast<u32>(h >> 32);
}
/// HeaderFlags bits.
inline constexpr u32 FlagCheck = 1;   ///< check words follow (FRITCHII_ONEF_FRONT_OPS=2 or 4)
inline constexpr u32 FlagDirect = 2;  ///< S3: E applies the op directly (=3 or 4)
inline constexpr u32 FlagSkipCheck = 4; ///< E checks the writes the front skipped (sampled)
/// Flags bits 8-15: how many constant buffer word addresses (the front's, one per sent Direct data
/// word in order) end the payload, after the check words when FlagCheck is set (check modes and
/// FRITCHII_CB_WORDS_BLOCK=2).
[[nodiscard]] constexpr u32 CbAddressFlags(u32 count) noexcept {
    return (count & 0xff) << 8;
}
[[nodiscard]] constexpr u32 HeaderCbAddresses(u64 h) noexcept {
    return static_cast<u32>(h >> 40) & 0xff;
}

[[nodiscard]] constexpr size_t ArgWords(u32 args) noexcept {
    return (size_t{args} + 1) / 2;
}
[[nodiscard]] inline u32 ArgAt(const u64* words, u32 index) noexcept {
    return static_cast<u32>(words[index / 2] >> ((index & 1) * 32));
}

} // namespace Tegra::OotOps
