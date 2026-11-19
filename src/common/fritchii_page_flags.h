// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>

#include "common/common_types.h"

/// Page flags of the device address space that writers of game memory set and the native uniform
/// bank check takes (BufferCache::CheckNativeBankPagesTracked). In common so that the writers in
/// common (the write ring's Note, the guest store's after-store point) set them too; core names
/// them Core::g_... (core/gpu_dirty_memory_manager.h).
namespace Common {

/// The price counters' page write generations (video_core/fritchii_price.h, rows 1 and 2 of
/// work\fable_stage_levelpoints.md 18.4): a hash of the device page to 65,536 buckets, raised by
/// every mark of the two flags the native bank check takes (g_collected_write_pages,
/// g_buffer_write_pages) while g_price_gens_on (the GPU profile's dry counters ran). A collision
/// only makes a range look written. A mark over more than 64 pages raises g_price_gens_wide,
/// which every sum includes.
inline constexpr size_t PriceGenBuckets = 65536;
inline std::array<std::atomic<u32>, PriceGenBuckets> g_price_gens{};
inline std::atomic<u32> g_price_gens_wide{0};
inline std::atomic<bool> g_price_gens_on{false};

[[nodiscard]] inline size_t PriceGenBucket(u64 page) noexcept {
    return static_cast<size_t>((page * 0x9E3779B97F4A7C15ULL) >> 48);
}

inline void PriceMarkGens(u64 address, u64 size) noexcept {
    if (size == 0) {
        return;
    }
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    if (last - first >= 64) {
        g_price_gens_wide.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        g_price_gens[PriceGenBucket(page)].fetch_add(1, std::memory_order_relaxed);
    }
}

/// The generations of the pages of [address, address + size) summed (they only grow, so an equal
/// sum over the same range means no mark since). Over 64 pages: ~0, counted as written.
[[nodiscard]] inline u64 PriceGenSum(u64 address, u64 size) noexcept {
    if (size == 0) {
        return 0;
    }
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    if (last - first >= 64) {
        return ~u64{0};
    }
    u64 sum = u64{g_price_gens_wide.load(std::memory_order_relaxed)} << 32;
    for (u64 page = first; page <= last; ++page) {
        sum += g_price_gens[PriceGenBucket(page)].load(std::memory_order_relaxed);
    }
    return sum;
}

/// FRITCHII_STORE_MEMO (core/memory.cpp, WriteToRasterizer): moved by everything that takes away
/// a mark a guest store to a GPU-tracked page leaves (a 64-byte block's collected write in a dirty
/// manager, the page's collected and watch flags). Such a clear runs inside a StoreMemoClear:
/// the low 16 bits count clears running, and each clear's end raises a generation. A core's store
/// memo keeps the value it read before its last full store path, and only when no clear was
/// running then. While the value is unchanged, no clear began or ended since and none was running
/// at the read, so no mark of that store was taken: a later store inside the same 64-byte block
/// would find every mark set (each would be a no-op) and may skip them. Moved always, whatever the
/// switch says, so a memo is never trusted across a switch change. Only equality is compared.
/// Bits 16-39: dirty manager gathers, bits 40-63: flag takes (the dry counter's line); a carry
/// from one field into the next changes the value all the same.
inline std::atomic<u64> g_store_memo_epoch{0};
inline constexpr u64 StoreMemoActiveMask = 0xffff;
inline constexpr u64 StoreMemoGatherGen = u64{1} << 16;
inline constexpr u64 StoreMemoTakeGen = u64{1} << 40;

/// Whether the epoch moves at all: FRITCHII_STORE_MEMO non-zero when the application process is
/// set up (Core::Memory, SetCurrentPageTable, read once, before any guest code ran or any GPU work
/// was made, so no mark exists yet that a clear could take unannounced). Unarmed, a clear costs
/// one relaxed load and the memo stays off for the session even if the switch is set live; armed,
/// every clear moves the epoch from boot on, so no memo is ever kept without them.
inline std::atomic<bool> g_store_memo_armed{false};

[[nodiscard]] inline bool StoreMemoArmed() noexcept {
    return g_store_memo_armed.load(std::memory_order_relaxed);
}

/// Around a clear of store marks while armed (seq_cst both ways, so a store memo's seq_cst read of
/// the epoch that comes after the clear in that order also comes after the begin).
class StoreMemoClear {
public:
    explicit StoreMemoClear(u64 generation_) noexcept
        : generation{generation_}, armed{StoreMemoArmed()} {
        if (armed) {
            g_store_memo_epoch.fetch_add(1, std::memory_order_seq_cst);
        }
    }
    ~StoreMemoClear() {
        if (armed) {
            g_store_memo_epoch.fetch_add(generation - 1, std::memory_order_seq_cst);
        }
    }
    StoreMemoClear(const StoreMemoClear&) = delete;
    StoreMemoClear& operator=(const StoreMemoClear&) = delete;

private:
    u64 generation;
    bool armed;
};

/// One bit per device page of the 36-bit device address space (2 MiB), set by writers and taken
/// by the native uniform bank check (BufferCache, EDEN_NX2_EXP9).
class DevicePageFlags {
public:
    static constexpr u64 AddressBits = 36;
    static constexpr u64 PageBits = 12;

    /// `feeds_price_`: its marks also raise the price counters' generations (above).
    /// `bumps_store_memo_`: a guest store's path marks it (Core::Memory's WriteToRasterizer), so
    /// Take raises g_store_memo_epoch before it clears a set bit.
    constexpr explicit DevicePageFlags(bool feeds_price_ = false,
                                       bool bumps_store_memo_ = false) noexcept
        : feeds_price{feeds_price_}, bumps_store_memo{bumps_store_memo_} {}

    void Mark(u64 address, u64 size) noexcept {
        if (feeds_price && g_price_gens_on.load(std::memory_order_relaxed)) [[unlikely]] {
            PriceMarkGens(address, size);
        }
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return;
        }
        u64 page = address >> PageBits;
        const u64 last = ((std::min)(address + size, limit) - 1) >> PageBits;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            std::atomic<u64>& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_seq_cst);
            }
            page += bits;
        }
    }

    /// Whether any page of [address, address + size) is marked, without clearing anything. A
    /// relaxed look, as Take's first: a mark set after it is seen by the next look.
    [[nodiscard]] bool Any(u64 address, u64 size) const noexcept {
        constexpr u64 limit = u64{1} << AddressBits;
        if (size == 0 || address >= limit) {
            return false;
        }
        u64 page = address >> PageBits;
        const u64 last = ((std::min)(address + size, limit) - 1) >> PageBits;
        while (page <= last) {
            const u64 first_bit = page % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, last - page + 1);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            if ((words[page / 64].load(std::memory_order_relaxed) & mask) != 0) {
                return true;
            }
            page += bits;
        }
        return false;
    }

    /// Clears the bit of the page holding `address`, returning whether it was set.
    bool Take(u64 address) noexcept {
        if (address >= (u64{1} << AddressBits)) {
            return false;
        }
        const u64 page = address >> PageBits;
        const u64 bit = u64{1} << (page % 64);
        std::atomic<u64>& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            return false;
        }
        if (bumps_store_memo && StoreMemoArmed()) {
            // FRITCHII_STORE_MEMO: a guest store's path marks this flag (g_store_memo_epoch).
            const StoreMemoClear clear{StoreMemoTakeGen};
            return (word.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0;
        }
        return (word.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0;
    }

private:
    std::array<std::atomic<u64>, (u64{1} << (AddressBits - PageBits)) / 64> words{};
    bool feeds_price{};
    bool bumps_store_memo{};
};

/// Pages guest CPU writes went to through the dirty memory managers: the buffer cache sees them
/// only at the next InvalidateGPUCache. Set before the store (Core::Memory's
/// HandleRasterizerWrite) and again after it (UniformKeepPages::AfterGuestStore).
inline DevicePageFlags g_collected_write_pages{true, true};
/// Pages whose guest memory the buffer cache saw change (WriteMemory, uploads in place,
/// downloads), and every emulator write the write ring hears of, after its bytes landed
/// (WriteRing::Note, MemoryManager::Write).
inline DevicePageFlags g_buffer_write_pages{true};
/// EDEN_NX2_TEX_BANK_GEN: pages written by anyone the GPU side hears of: guest CPU writes that
/// trapped (Memory::HandleRasterizerWrite, every one, collected or not), the buffer cache's
/// writes (the same places as g_buffer_write_pages) and TextureCache::WriteMemory. Only the
/// texture cache's watched-page harvest takes them (TextureCache::TexBankPageGen).
inline DevicePageFlags g_watch_write_pages{false, true};
/// The bank marks' gate: open while any of its users wants it, one bit each: FRITCHII_BANK_FLAGGED
/// or FRITCHII_BANK_CENSUS non-zero (BankMarksBankCheck, set by the bank check,
/// BufferCache::CheckNativeBankPagesTracked) and FRITCHII_BIND_SKIP non-zero (BankMarksBindSkip,
/// BufferCache::BindSkipPoll). Off, each mark below that the hot rule made unneeded (the write
/// ring's Note, MemoryManager::Write, unsafe and span stores reaching Note, the after-store
/// re-mark) is one relaxed load and a branch. Each user trusts nothing from before the gate
/// opened for it: two frames after.
inline constexpr u32 BankMarksBankCheck = 1;
inline constexpr u32 BankMarksBindSkip = 2;
inline std::atomic<u32> g_bank_marks{0};

[[nodiscard]] inline bool BankMarks() noexcept {
    return g_bank_marks.load(std::memory_order_relaxed) != 0;
}

/// Opens (`want`) or closes the gate for one user's bit.
inline void SetBankMarks(u32 user, bool want) noexcept {
    if (want) {
        g_bank_marks.fetch_or(user, std::memory_order_seq_cst);
    } else {
        g_bank_marks.fetch_and(~user, std::memory_order_seq_cst);
    }
}

/// Guest stores whose ranges the after-store point could not hold (more than a thread records):
/// no page flag was set after them. The bank check with FRITCHII_BANK_FLAGGED drops every
/// snapshot when this moves.
inline std::atomic<u64> g_collected_overflows{0};
/// FRITCHII_BANK_CENSUS=2: pages the write ring recorded a write to (WriteRing::Record), taken by
/// the census at each compare. Only while the ring is on.
inline DevicePageFlags g_ring_write_pages;

} // namespace Common
