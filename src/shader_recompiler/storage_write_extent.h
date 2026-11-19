// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <compare>
#include <optional>

#include "common/common_types.h"

namespace Shader {

/// How far a program's stores and atomics to one storage buffer can reach, in bytes past the
/// pointer it reads from its constant buffer. Every store ends at or below
///
///   const_end + sum(local_stride[i] * local_id[i]) + sum(group_stride[i] * group_id[i])
///
/// for every invocation, where local_id is gl_LocalInvocationID and group_id gl_WorkGroupID. The
/// bound only holds while each value the stores' offsets split into 16-bit halves (multiplications
/// lowered from native IMAD) stays below 2^16: guard_* is an upper bound of the largest of them in
/// the same form, and the extent is unknown when it can reach 2^16 for a dispatch.
///
/// A native (Switch 2) program's storage pointer carries no size, so the buffer cache binds up to
/// 8 MiB behind it; this is what it can claim the dispatch wrote instead of all of it.
struct StorageWriteExtent {
    /// False: the reach is not known (or the buffer is never written); take the whole binding.
    bool bounded{};
    /// Whether the bound needs the guard below.
    bool guarded{};
    u32 const_end{};
    std::array<u32, 3> local_stride{};
    std::array<u32, 3> group_stride{};
    u32 guard_const{};
    std::array<u32, 3> guard_local_stride{};
    std::array<u32, 3> guard_group_stride{};

    auto operator<=>(const StorageWriteExtent&) const = default;
};

/// Bytes a dispatch's stores can reach past the pointer, or nullopt when unknown. grid is nullopt
/// for an indirect dispatch, whose group count is read by the GPU.
[[nodiscard]] constexpr std::optional<u64> ResolveStorageWriteExtent(
    const StorageWriteExtent& extent, const std::array<u32, 3>& block,
    const std::optional<std::array<u32, 3>>& grid) {
    if (!extent.bounded) {
        return std::nullopt;
    }
    const auto evaluate = [&](u64 constant, const std::array<u32, 3>& local,
                              const std::array<u32, 3>& group) -> std::optional<u64> {
        // Anything past 2^62 bytes is as good as unknown, and stops the sum from overflowing.
        constexpr u64 limit = u64{1} << 62;
        u64 total = constant;
        for (size_t i = 0; i < 3; ++i) {
            if (block[i] == 0) {
                return std::nullopt;
            }
            const u64 local_max = u64{block[i]} - 1;
            if (local[i] != 0 && local_max != 0) {
                if (local_max > limit / local[i]) {
                    return std::nullopt;
                }
                total += u64{local[i]} * local_max;
            }
            if (group[i] == 0) {
                continue;
            }
            if (!grid || (*grid)[i] == 0) {
                return std::nullopt;
            }
            const u64 group_max = u64{(*grid)[i]} - 1;
            if (group_max != 0) {
                if (group_max > limit / group[i]) {
                    return std::nullopt;
                }
                total += u64{group[i]} * group_max;
            }
            if (total > limit) {
                return std::nullopt;
            }
        }
        return total;
    };
    if (extent.guarded) {
        const auto guard =
            evaluate(extent.guard_const, extent.guard_local_stride, extent.guard_group_stride);
        if (!guard || *guard >= 0x10000) {
            return std::nullopt;
        }
    }
    return evaluate(extent.const_end, extent.local_stride, extent.group_stride);
}

/// FRITCHII_EXTENT_CENSUS: why a written storage buffer's store extent is unknown, the first exit
/// of the store analysis (StoreExtentAnalysis). Not stored with the info (info_serialize.cpp):
/// an info from the precompiled cache has Unrecorded.
enum class ExtentUnknown : u8 {
    Unrecorded,        ///< nothing recorded (an info from the precompiled cache)
    Known,             ///< the extent is proven
    OtherCbufWord,     ///< an offset reads another constant buffer word (cbuf_index, cbuf_offset)
    DynamicCbuf,       ///< a constant buffer read at a computed offset (opcode)
    LoadedFromMemory,  ///< an offset loaded from memory (opcode)
    AtomicResult,      ///< an offset is an atomic's result (opcode)
    LoopCarried,       ///< an offset carried around a loop (a cycle through a phi)
    SignedOr64Bit,     ///< a signed or 64-bit computation (opcode)
    VariableProduct,   ///< two non-constant factors multiplied, or a variable subtracted (opcode)
    OtherOp,           ///< an instruction the analysis does not bound (opcode)
    TooDeep,           ///< the expression is deeper than the analysis follows
    GuardFailed,       ///< a lowered multiplication's 16-bit split it cannot bound
    AddressNotTracked, ///< the store's 64-bit address is not split into pointer + low word
    NoPointerTerm,     ///< the low word does not add the pointer word exactly once
    TooLarge,          ///< a constant or stride past the analysis' limits
    UntrackedWrites,   ///< a store of the program stays global (routed by address at run time)
    RuntimeRouted,     ///< the store picks one of several pointers at run time
    GlobalStore,       ///< a driver buffer the program's global stores may land anywhere in
    // FRITCHII_EXTENT_SHARED (StorageWriteExtra::shared_why): why an offset loaded from shared
    // memory was not followed to the value stored there
    SharedNotPrivate,   ///< a shared write or atomic is not in its invocation's own slot (opcode)
    SharedPartialWrite, ///< another width's write or an atomic overlaps the loaded word (opcode)
    SharedNotDominated, ///< no store of the loaded word runs before the load on every path
    // FRITCHII_EXTENT_GUARD (StorageWriteExtra::guard_why): why an index was not bounded by the
    // branch conditions every path to its store passes
    GuardNotDominating, ///< a branch compares the index, but the store is reachable around it
    GuardSigned,        ///< the guard is a signed compare: the index may be negative (opcode)
    GuardWraps,         ///< the guard is on the index plus a value that may wrap 32 bits (opcode)
    GuardLimitUnknown,  ///< the guard's limit is not bounded (loaded from memory...) (opcode)
    GuardInLoop,        ///< the guarded index is redefined in a loop (opcode)
};

struct ExtentUnknownWhy {
    ExtentUnknown kind{};
    u16 opcode{}; ///< Shader::IR::Opcode, for the kinds that name one
    u32 cbuf_index{};
    u32 cbuf_offset{};

    auto operator<=>(const ExtentUnknownWhy&) const = default;
};

[[nodiscard]] constexpr const char* ExtentUnknownName(ExtentUnknown kind) {
    switch (kind) {
    case ExtentUnknown::Unrecorded:
        return "not recorded (no extra: a precompiled-cache info, or a bind path without it)";
    case ExtentUnknown::Known:
        return "proven";
    case ExtentUnknown::OtherCbufWord:
        return "index from a constant buffer word";
    case ExtentUnknown::DynamicCbuf:
        return "constant buffer read at a computed offset";
    case ExtentUnknown::LoadedFromMemory:
        return "index loaded from memory";
    case ExtentUnknown::AtomicResult:
        return "index from an atomic";
    case ExtentUnknown::LoopCarried:
        return "loop-carried index";
    case ExtentUnknown::SignedOr64Bit:
        return "signed or 64-bit math";
    case ExtentUnknown::VariableProduct:
        return "variable times variable (or minus a variable)";
    case ExtentUnknown::OtherOp:
        return "instruction not bounded";
    case ExtentUnknown::TooDeep:
        return "expression too deep";
    case ExtentUnknown::GuardFailed:
        return "16-bit split not bounded";
    case ExtentUnknown::AddressNotTracked:
        return "address not split into pointer + low word";
    case ExtentUnknown::NoPointerTerm:
        return "low word does not add the pointer once";
    case ExtentUnknown::TooLarge:
        return "constant or stride too large";
    case ExtentUnknown::UntrackedWrites:
        return "a store stays global";
    case ExtentUnknown::RuntimeRouted:
        return "pointer picked at run time";
    case ExtentUnknown::GlobalStore:
        return "driver buffer with global stores";
    case ExtentUnknown::SharedNotPrivate:
        return "shared memory not one slot per invocation";
    case ExtentUnknown::SharedPartialWrite:
        return "loaded shared word also written another way";
    case ExtentUnknown::SharedNotDominated:
        return "loaded shared word not stored before the load";
    case ExtentUnknown::GuardNotDominating:
        return "a branch compares the index but the store is reachable around it";
    case ExtentUnknown::GuardSigned:
        return "signed guard on a possibly negative index";
    case ExtentUnknown::GuardWraps:
        return "guard on the index plus a value that may wrap";
    case ExtentUnknown::GuardLimitUnknown:
        return "guard limit not bounded";
    case ExtentUnknown::GuardInLoop:
        return "guarded index redefined in a loop";
    }
    return "?";
}

/// FRITCHII_EXTENT_CBUF: a constant buffer word in a store's offset, coef times its value.
struct StorageExtentTerm {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 coef{};

    auto operator<=>(const StorageExtentTerm&) const = default;
};
inline constexpr size_t MaxExtentTerms = 4;

/// FRITCHII_EXTENT_GUARD: a constant buffer word the bound needs to be at least `min` (an index
/// clamped to word - count, which only holds while the subtraction does not wrap). Read at bind
/// time like the terms; a smaller word leaves the extent unknown.
struct StorageExtentFloor {
    u32 cbuf_index{};
    u32 cbuf_offset{};
    u32 min{};

    auto operator<=>(const StorageExtentFloor&) const = default;
};
inline constexpr size_t MaxExtentFloors = 4;

/// FRITCHII_EXTENT_CBUF: a store extent whose offsets may also add constant buffer words (an
/// object index read from a uniform times a stride). Every store ends at or below
///
///   ids' bound + sum(terms[i].coef * value of terms[i]'s word)
///
/// and the guard (when ids.guarded) is ids' guard bound plus the guard terms the same way. The
/// words are read at bind time, like the pointer word. Not stored with the info: an info from the
/// precompiled cache has an unbounded one.
///
/// FRITCHII_EXTENT_SHARED: via_shared marks an extent proven only by following offsets an
/// invocation parked in its own shared memory slot to the values it stored there (the switch,
/// not FRITCHII_EXTENT_CBUF, decides whether it is used).
///
/// FRITCHII_EXTENT_GUARD: via_guard marks an extent proven only with the branch conditions on the
/// stores' paths (an append index under `if (index < capacity)`), subgroup broadcasts and bit
/// counts, and floors (each word at least its min); the switch, not FRITCHII_EXTENT_CBUF, decides.
struct StorageCbufExtent {
    StorageWriteExtent ids{};
    u8 num_terms{};
    u8 num_guard_terms{};
    bool via_shared{};
    std::array<StorageExtentTerm, MaxExtentTerms> terms{};
    std::array<StorageExtentTerm, MaxExtentTerms> guard_terms{};
    bool via_guard{};
    u8 num_floors{};
    std::array<StorageExtentFloor, MaxExtentFloors> floors{};

    auto operator<=>(const StorageCbufExtent&) const = default;
};

/// Bytes a dispatch's stores can reach past the pointer with the terms' word values (`values`
/// for terms, `guard_values` for guard_terms, `floor_values` for floors, in order), or nullopt
/// when unknown.
[[nodiscard]] constexpr std::optional<u64> ResolveStorageCbufExtent(
    const StorageCbufExtent& extent, const std::array<u32, 3>& block,
    const std::optional<std::array<u32, 3>>& grid, const std::array<u32, MaxExtentTerms>& values,
    const std::array<u32, MaxExtentTerms>& guard_values,
    const std::array<u32, MaxExtentFloors>& floor_values = {}) {
    if (!extent.ids.bounded) {
        return std::nullopt;
    }
    for (size_t i = 0; i < extent.num_floors && i < MaxExtentFloors; ++i) {
        // The clamp's subtraction would wrap: the index is not bounded by the word
        if (floor_values[i] < extent.floors[i].min) {
            return std::nullopt;
        }
    }
    constexpr u64 limit = u64{1} << 62;
    if (extent.ids.guarded) {
        // The guard's id part, resolved as an unguarded extent of its own
        const StorageWriteExtent guard_ids{
            .bounded = true,
            .const_end = extent.ids.guard_const,
            .local_stride = extent.ids.guard_local_stride,
            .group_stride = extent.ids.guard_group_stride,
        };
        const auto guard = ResolveStorageWriteExtent(guard_ids, block, grid);
        if (!guard) {
            return std::nullopt;
        }
        u64 total = *guard;
        if (total >= 0x10000) {
            return std::nullopt;
        }
        for (size_t i = 0; i < extent.num_guard_terms && i < MaxExtentTerms; ++i) {
            // A product of 2^16 or more fails the guard by itself (and keeps the sum small)
            const u64 product = u64{extent.guard_terms[i].coef} * guard_values[i];
            if (product >= 0x10000) {
                return std::nullopt;
            }
            total += product;
        }
        if (total >= 0x10000) {
            return std::nullopt;
        }
    }
    StorageWriteExtent ids = extent.ids;
    ids.guarded = false;
    const auto base = ResolveStorageWriteExtent(ids, block, grid);
    if (!base) {
        return std::nullopt;
    }
    u64 total = *base;
    for (size_t i = 0; i < extent.num_terms && i < MaxExtentTerms; ++i) {
        // coef and value are below 2^32: the product fits in 64 bits
        const u64 product = u64{extent.terms[i].coef} * values[i];
        if (product > limit) {
            return std::nullopt;
        }
        total += product;
        if (total > limit) {
            return std::nullopt;
        }
    }
    return total;
}

/// Widens an extent to also cover another (both bounds hold for each component's maximum).
constexpr void MergeStorageWriteExtent(StorageWriteExtent& base, const StorageWriteExtent& other) {
    if (!base.bounded || !other.bounded) {
        base = StorageWriteExtent{};
        return;
    }
    const auto max3 = [](std::array<u32, 3>& lhs, const std::array<u32, 3>& rhs) {
        for (size_t i = 0; i < 3; ++i) {
            lhs[i] = (std::max)(lhs[i], rhs[i]);
        }
    };
    base.const_end = (std::max)(base.const_end, other.const_end);
    max3(base.local_stride, other.local_stride);
    max3(base.group_stride, other.group_stride);
    if (other.guarded) {
        if (base.guarded) {
            base.guard_const = (std::max)(base.guard_const, other.guard_const);
            max3(base.guard_local_stride, other.guard_local_stride);
            max3(base.guard_group_stride, other.guard_group_stride);
        } else {
            base.guarded = true;
            base.guard_const = other.guard_const;
            base.guard_local_stride = other.guard_local_stride;
            base.guard_group_stride = other.guard_group_stride;
        }
    }
}

/// Adds `term` to a term list (the larger coefficient for a word already on it). False when the
/// list is full.
constexpr bool MergeExtentTerm(std::array<StorageExtentTerm, MaxExtentTerms>& terms, u8& count,
                               const StorageExtentTerm& term) {
    for (size_t i = 0; i < count; ++i) {
        if (terms[i].cbuf_index == term.cbuf_index && terms[i].cbuf_offset == term.cbuf_offset) {
            terms[i].coef = (std::max)(terms[i].coef, term.coef);
            return true;
        }
    }
    if (count >= MaxExtentTerms) {
        return false;
    }
    terms[count++] = term;
    return true;
}

/// FRITCHII_EXTENT_CBUF: widens a cbuf extent to also cover another (unbounded when the terms do
/// not fit).
constexpr void MergeStorageCbufExtent(StorageCbufExtent& base, const StorageCbufExtent& other) {
    MergeStorageWriteExtent(base.ids, other.ids);
    if (!base.ids.bounded) {
        base = StorageCbufExtent{};
        return;
    }
    base.via_shared = base.via_shared || other.via_shared;
    base.via_guard = base.via_guard || other.via_guard;
    for (size_t i = 0; i < other.num_floors && i < MaxExtentFloors; ++i) {
        // Every floor of either stays needed (the larger min for a word on both)
        const StorageExtentFloor& floor = other.floors[i];
        bool merged = false;
        for (size_t j = 0; j < base.num_floors; ++j) {
            if (base.floors[j].cbuf_index == floor.cbuf_index &&
                base.floors[j].cbuf_offset == floor.cbuf_offset) {
                base.floors[j].min = (std::max)(base.floors[j].min, floor.min);
                merged = true;
                break;
            }
        }
        if (!merged) {
            if (base.num_floors >= MaxExtentFloors) {
                base = StorageCbufExtent{};
                return;
            }
            base.floors[base.num_floors++] = floor;
        }
    }
    for (size_t i = 0; i < other.num_terms; ++i) {
        if (!MergeExtentTerm(base.terms, base.num_terms, other.terms[i])) {
            base = StorageCbufExtent{};
            return;
        }
    }
    for (size_t i = 0; i < other.num_guard_terms; ++i) {
        if (!MergeExtentTerm(base.guard_terms, base.num_guard_terms, other.guard_terms[i])) {
            base = StorageCbufExtent{};
            return;
        }
    }
}

/// FRITCHII_STORE_FOOTPRINT: why a written storage buffer has no store footprint.
enum class StoreFootprintWhy : u8 {
    NotRecorded,       ///< nothing recorded (an info from the precompiled cache, or not written)
    Recorded,          ///< it has one
    UntrackedWrites,   ///< a store of the program stays global (routed by address at run time)
    RuntimeRouted,     ///< the store picks one of several pointers at run time
    GlobalStore,       ///< a driver buffer the program's global stores may land anywhere in
    AddressNotTracked, ///< a store's 64-bit address is not split into pointer + low word
    NoPointerTerm,     ///< a store's low word does not add the pointer word exactly once
    NoStores,          ///< written, but no store of it was rebased on it
    Joined,            ///< both halves of a joined program write it
};

[[nodiscard]] constexpr const char* StoreFootprintWhyName(StoreFootprintWhy why) {
    switch (why) {
    case StoreFootprintWhy::NotRecorded:
        return "not recorded (a precompiled-cache info: run with EDEN_NX2_PRECOMPILED_FAST=0)";
    case StoreFootprintWhy::Recorded:
        return "recorded";
    case StoreFootprintWhy::UntrackedWrites:
        return "a store stays global";
    case StoreFootprintWhy::RuntimeRouted:
        return "pointer picked at run time";
    case StoreFootprintWhy::GlobalStore:
        return "the program stores to global memory";
    case StoreFootprintWhy::AddressNotTracked:
        return "address not split into pointer + low word";
    case StoreFootprintWhy::NoPointerTerm:
        return "low word does not add the pointer once";
    case StoreFootprintWhy::NoStores:
        return "no store rebased on it";
    case StoreFootprintWhy::Joined:
        return "written by both halves of a joined program";
    }
    return "?";
}

/// FRITCHII_STORE_FOOTPRINT: the bytes a program's stores and atomics to one storage buffer can
/// land on, relative to the pointer word. Every store's offset is exactly
///
///   c + (a sum of multiples of `modulus`)
///
/// where the multiples are invocation or workgroup ids, values loaded from memory, loop counters
/// or constant buffer words times a stride (ExactCongruence in the pass). So each store byte's
/// offset mod `modulus` falls in one of `intervals` ([begin, end) inside [0, modulus), sorted and
/// apart). A modulus of 2^32 is an exact set of offsets (constant stores only). A modulus that is
/// not a power of two holds only while the 32-bit offset does not wrap (pointer + offset stays
/// inside 4 GiB), which the dry count checks at landing. Over-approximated (gaps filled) past
/// MaxIntervals pieces.
struct StorageStoreFootprint {
    static constexpr size_t MaxIntervals = 8;
    struct Interval {
        u64 begin{};
        u64 end{};
    };
    u64 modulus{}; ///< 0: no footprint (see why)
    StoreFootprintWhy why{StoreFootprintWhy::NotRecorded};
    u8 num_intervals{};
    /// Every multiple came from invocation and workgroup ids (no loaded or constant buffer value)
    bool ids_only{};
    /// The intervals cover every residue: the footprint claims nothing
    bool covers_all{};
    /// Gaps were filled to fit MaxIntervals
    bool widened{};
    u16 stores{}; ///< stores and atomics merged
    std::array<Interval, MaxIntervals> intervals{};

    [[nodiscard]] constexpr bool Valid() const {
        return modulus != 0;
    }
    [[nodiscard]] constexpr bool PowerOfTwo() const {
        return modulus != 0 && (modulus & (modulus - 1)) == 0;
    }
};

/// One store for MakeStoreFootprint: offset == c (mod m), m == 0 for an exact offset (c is then
/// the signed 32-bit offset: negative lands below the pointer).
struct StoreCongruence {
    s64 c{};
    u64 m{};
    u32 size{};
};

/// FRITCHII_STORE_FOOTPRINT: a footprint from every store of a buffer (an empty list: no
/// stores). The modulus is the gcd of the stores' moduli (2^32 when every store is exact).
[[nodiscard]] constexpr StorageStoreFootprint MakeStoreFootprint(const StoreCongruence* stores,
                                                                 size_t count, bool ids_only) {
    StorageStoreFootprint result{};
    if (count == 0) {
        result.why = StoreFootprintWhy::NoStores;
        return result;
    }
    constexpr u64 Exact = u64{1} << 32;
    const auto gcd = [](u64 a, u64 b) {
        while (b != 0) {
            const u64 t = a % b;
            a = b;
            b = t;
        }
        return a;
    };
    u64 modulus = 0;
    for (size_t i = 0; i < count; ++i) {
        modulus = gcd(modulus, stores[i].m);
    }
    if (modulus == 0 || modulus > Exact) {
        modulus = Exact;
    }
    result.modulus = modulus;
    result.why = StoreFootprintWhy::Recorded;
    result.ids_only = ids_only;
    result.stores = static_cast<u16>(count > 0xffff ? 0xffff : count);
    // Pieces, at most two per store (one that wraps past the modulus is split), sorted and merged
    // whenever the array fills up
    constexpr size_t MaxPieces = 64;
    std::array<StorageStoreFootprint::Interval, MaxPieces> pieces{};
    size_t num_pieces = 0;
    // Sort (insertion sort: constexpr and small), merge touching pieces, then fill the smallest
    // gaps down to `limit` pieces (a wider footprint is still a footprint)
    const auto compact = [&](size_t limit) {
        for (size_t i = 1; i < num_pieces; ++i) {
            const auto piece = pieces[i];
            size_t j = i;
            while (j > 0 && pieces[j - 1].begin > piece.begin) {
                pieces[j] = pieces[j - 1];
                --j;
            }
            pieces[j] = piece;
        }
        size_t merged = 0;
        for (size_t i = 0; i < num_pieces; ++i) {
            if (merged != 0 && pieces[i].begin <= pieces[merged - 1].end) {
                pieces[merged - 1].end = (std::max)(pieces[merged - 1].end, pieces[i].end);
            } else {
                pieces[merged++] = pieces[i];
            }
        }
        while (merged > limit) {
            size_t smallest = 1;
            for (size_t i = 2; i < merged; ++i) {
                if (pieces[i].begin - pieces[i - 1].end <
                    pieces[smallest].begin - pieces[smallest - 1].end) {
                    smallest = i;
                }
            }
            pieces[smallest - 1].end = pieces[smallest].end;
            for (size_t i = smallest; i + 1 < merged; ++i) {
                pieces[i] = pieces[i + 1];
            }
            --merged;
            result.widened = true;
        }
        num_pieces = merged;
    };
    for (size_t i = 0; i < count; ++i) {
        const u64 size = stores[i].size;
        if (size >= modulus) {
            result.covers_all = true;
            result.num_intervals = 1;
            result.intervals[0] = {0, modulus};
            return result;
        }
        if (num_pieces + 2 > MaxPieces) {
            compact(MaxPieces / 2);
        }
        const s64 residue = stores[i].c % static_cast<s64>(modulus);
        const u64 begin = static_cast<u64>(residue < 0 ? residue + static_cast<s64>(modulus)
                                                       : residue);
        const u64 end = begin + size;
        if (end <= modulus) {
            pieces[num_pieces++] = {begin, end};
        } else {
            pieces[num_pieces++] = {begin, modulus};
            pieces[num_pieces++] = {0, end - modulus};
        }
    }
    compact(StorageStoreFootprint::MaxIntervals);
    const size_t merged = num_pieces;
    result.num_intervals = static_cast<u8>(merged);
    for (size_t i = 0; i < merged; ++i) {
        result.intervals[i] = pieces[i];
    }
    result.covers_all = merged == 1 && pieces[0].begin == 0 && pieces[0].end == modulus;
    return result;
}

/// FRITCHII_STORE_FOOTPRINT: the footprint modulo the modulus' largest power-of-two divisor (a
/// congruence modulo m holds modulo any divisor of m, and modulo a power of two whatever the 32-bit
/// offset does). The live switch uses it when it can't rule out a wrap at the bind.
[[nodiscard]] constexpr StorageStoreFootprint ReduceStoreFootprint(
    const StorageStoreFootprint& footprint) {
    if (!footprint.Valid() || footprint.PowerOfTwo()) {
        return footprint;
    }
    u64 modulus = 1;
    while ((footprint.modulus & modulus) == 0) {
        modulus <<= 1;
    }
    std::array<StoreCongruence, StorageStoreFootprint::MaxIntervals> stores{};
    size_t count = 0;
    for (size_t i = 0; i < footprint.num_intervals && i < stores.size(); ++i) {
        const u64 length = footprint.intervals[i].end - footprint.intervals[i].begin;
        stores[count++] = StoreCongruence{
            .c = static_cast<s64>(footprint.intervals[i].begin),
            .m = modulus,
            .size = static_cast<u32>(length > 0xffffffffULL ? 0xffffffffULL : length),
        };
    }
    StorageStoreFootprint result = MakeStoreFootprint(stores.data(), count, footprint.ids_only);
    result.stores = footprint.stores;
    result.widened = result.widened || footprint.widened;
    return result;
}

/// FRITCHII_STORE_FOOTPRINT: whether any of the bytes [offset, offset + size) past the pointer
/// (offset may be negative: below the pointer) can be a store byte of the footprint.
[[nodiscard]] constexpr bool StoreFootprintHits(const StorageStoreFootprint& footprint, s64 offset,
                                                u64 size) {
    if (!footprint.Valid() || footprint.covers_all) {
        return true;
    }
    if (size == 0) {
        return false;
    }
    const u64 modulus = footprint.modulus;
    if (size >= modulus) {
        return true;
    }
    u64 begin;
    if (footprint.PowerOfTwo()) {
        // Two's complement: the same residue as the 32-bit offset the shader computes
        begin = static_cast<u64>(offset) & (modulus - 1);
    } else {
        const s64 m = static_cast<s64>(modulus);
        const s64 r = offset % m;
        begin = static_cast<u64>(r < 0 ? r + m : r);
    }
    const u64 end = begin + size;
    const auto overlaps = [&](u64 lower, u64 upper) {
        for (size_t i = 0; i < footprint.num_intervals; ++i) {
            if (footprint.intervals[i].begin < upper && lower < footprint.intervals[i].end) {
                return true;
            }
        }
        return false;
    };
    if (end <= modulus) {
        return overlaps(begin, end);
    }
    return overlaps(begin, modulus) || overlaps(0, end - modulus);
}

} // namespace Shader
