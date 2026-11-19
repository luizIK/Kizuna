// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

#include "common/common_types.h"
#include "common/fritchii_page_flags.h"

/// EXP92 (fable52_answer_uniformkeeps 3, tier 2): what the
/// buffer cache's uniform keeps need to know about writes to guest memory that are not the back's
/// cached writes (those are tier 1, by exact range, in BufferCache's ring).
///
///  - `kept`: device pages that ever held a kept range (never cleared). Every writer calls Note
///    after its bytes are in guest memory; Note raises the generation of each such page. A keep
///    remembers the generations it read before reading the bytes, so any Note after that fails it.
///  - `watched`: pages whose page protection this design holds (one UpdatePagesCachedCount +1),
///    so every guest CPU store to them traps (Core::Memory's HandleRasterizerWrite records the
///    range, the after-store point Notes it). Given back when the device page's backing goes
///    away or changes (RasterizerVulkan::UnmapMemory, DeviceMemoryManager::Map and Unmap) and
///    when the page is denied.
///  - `denied`: watched pages the guest stored to more than a limit in a frame: their stores
///    would all take the slow path; never kept again.
///
/// Header only and in common so the GPU write listener (common) can Note too. Device addresses
/// are 36 bits with 4 KiB pages (Tegra::MaxwellDeviceTraits); others are ignored (never kept).
namespace Common::UniformKeepPages {

inline constexpr u64 AddressBits = 36;
inline constexpr u64 PageBits = 12;
inline constexpr u64 PageSize = u64{1} << PageBits;
inline constexpr u64 NumPages = u64{1} << (AddressBits - PageBits);
inline constexpr size_t GenBits = 12;
inline constexpr size_t NumGens = size_t{1} << GenBits;

/// One bit per device page.
class PageBitmap {
public:
    [[nodiscard]] bool Test(u64 page) const noexcept {
        return page < NumPages &&
               (words[page / 64].load(std::memory_order_seq_cst) >> (page % 64) & 1) != 0;
    }
    /// Returns whether the bit was set before.
    bool Set(u64 page) noexcept {
        if (page >= NumPages) {
            return true;
        }
        const u64 bit = u64{1} << (page % 64);
        std::atomic<u64>& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) != 0) {
            return true;
        }
        return (word.fetch_or(bit, std::memory_order_seq_cst) & bit) != 0;
    }
    /// Returns whether the bit was set before.
    bool Clear(u64 page) noexcept {
        if (page >= NumPages) {
            return false;
        }
        const u64 bit = u64{1} << (page % 64);
        std::atomic<u64>& word = words[page / 64];
        if ((word.load(std::memory_order_relaxed) & bit) == 0) {
            return false;
        }
        return (word.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0;
    }
    /// Whether any page of [first, last] is set (relaxed: a hint only).
    [[nodiscard]] bool AnyRelaxed(u64 first, u64 last) const noexcept {
        last = (std::min)(last, NumPages - 1);
        for (u64 page = first; page <= last;) {
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

private:
    std::array<std::atomic<u64>, NumPages / 64> words{};
};

struct State {
    PageBitmap kept;
    PageBitmap watched;
    PageBitmap denied;
    /// Per page, by hash (a collision only fails a keep that need not have failed).
    std::array<std::atomic<u32>, NumGens> page_gen{};
    /// Raised when a writer could not say which pages it wrote (fails every keep).
    std::atomic<u32> global_gen{};
    /// Set at the first kept page, never cleared: until then the writers' checks are one load.
    std::atomic<bool> any_kept{false};
    /// Guest stores to watched pages since the last TickFrame, per page hash (deny counter).
    std::array<std::atomic<u32>, NumGens> stores{};
    std::atomic<u64> guest_stores{};    ///< total guest stores to watched pages
    std::atomic<u64> pages_denied{};    ///< total pages denied
    std::atomic<u64> record_overflows{}; ///< guest stores with more ranges than a thread holds
    std::atomic<u32> watched_count{};
    /// Watch, unwatch and deny; never held while waiting for anything but the device memory's
    /// counter_guard (UpdatePagesCachedCount).
    std::mutex mutex;
    std::vector<u64> watched_list; ///< under `mutex`
};

inline State g_state;

[[nodiscard]] inline size_t GenIndex(u64 page) noexcept {
    return static_cast<size_t>((page * 0x9E3779B97F4A7C15ULL) >> (64 - GenBits));
}

[[nodiscard]] inline bool AnyKept() noexcept {
    return g_state.any_kept.load(std::memory_order_relaxed);
}

[[nodiscard]] inline u32 PageGen(u64 page) noexcept {
    return g_state.page_gen[GenIndex(page)].load(std::memory_order_seq_cst);
}

[[nodiscard]] inline u32 GlobalGen() noexcept {
    return g_state.global_gen.load(std::memory_order_seq_cst);
}

inline void RaisePage(u64 page) noexcept {
    g_state.page_gen[GenIndex(page)].fetch_add(1, std::memory_order_seq_cst);
}

inline void RaiseAll() noexcept {
    g_state.global_gen.fetch_add(1, std::memory_order_seq_cst);
}

/// Every writer, after its bytes are in guest memory: raises the generation of each page of
/// [address, address + size) that ever held a kept range. One relaxed load until the first keep.
/// The fence orders the caller's stores before the kept-bit loads (against the keep's bit set,
/// then generation read, then copy of the bytes).
inline void Note(u64 address, u64 size) noexcept {
    if (!AnyKept() || size == 0 || address >= (u64{1} << AddressBits)) [[likely]] {
        return;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const u64 first = address >> PageBits;
    const u64 last = (std::min)((address + size - 1) >> PageBits, NumPages - 1);
    for (u64 page = first; page <= last; ++page) {
        if (g_state.kept.Test(page)) {
            RaisePage(page);
        }
    }
}

// Guest CPU stores (core/memory.cpp). HandleRasterizerWrite records the device ranges before the
// store; the after-store point (beside WriteTrack::CloseIfArmed) Notes them and counts stores to
// watched pages. Recorded always (a few thread-local stores), so a keep made between the record
// and the store still sees the Note.
struct ThreadStores {
    static constexpr u32 Capacity = 4;
    std::array<u64, Capacity> address{};
    std::array<u64, Capacity> size{};
    u32 count{};
    bool overflow{};
};
inline thread_local ThreadStores t_stores;

inline void RecordGuestStore(u64 address, u64 size) noexcept {
    ThreadStores& stores = t_stores;
    if (stores.count < ThreadStores::Capacity) {
        stores.address[stores.count] = address;
        stores.size[stores.count] = size;
        ++stores.count;
    } else {
        stores.overflow = true;
    }
}

inline void AfterGuestStoreSlow() noexcept {
    ThreadStores& stores = t_stores;
    // FRITCHII_BANK_FLAGGED: the collected write's page flag again, now that the bytes landed
    // (HandleRasterizerWrite set it before the store; a bank check may have taken it, and its one
    // recheck passed, before the bytes arrived). With the bank marks on; a lost range moves the
    // overflow count.
    if (BankMarks()) [[unlikely]] {
        if (stores.overflow) {
            g_collected_overflows.fetch_add(1, std::memory_order_seq_cst);
        }
        for (u32 i = 0; i < stores.count; ++i) {
            g_collected_write_pages.Mark(stores.address[i], stores.size[i]);
        }
    }
    if (AnyKept()) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (stores.overflow) {
            g_state.record_overflows.fetch_add(1, std::memory_order_relaxed);
            RaiseAll();
        }
        for (u32 i = 0; i < stores.count; ++i) {
            const u64 address = stores.address[i];
            const u64 size = stores.size[i];
            if (size == 0 || address >= (u64{1} << AddressBits)) {
                continue;
            }
            const u64 first = address >> PageBits;
            const u64 last = (std::min)((address + size - 1) >> PageBits, NumPages - 1);
            for (u64 page = first; page <= last; ++page) {
                if (!g_state.kept.Test(page)) {
                    continue;
                }
                RaisePage(page);
                if (g_state.watched.Test(page)) {
                    g_state.stores[GenIndex(page)].fetch_add(1, std::memory_order_relaxed);
                    g_state.guest_stores.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
    stores.count = 0;
    stores.overflow = false;
}

inline void AfterGuestStore() noexcept {
    const ThreadStores& stores = t_stores;
    if (stores.count != 0 || stores.overflow) [[unlikely]] {
        AfterGuestStoreSlow();
    }
}

/// Step 1 of making a keep (drawing thread): marks the pages of [address, address + size) kept
/// and watches each once (`update(page_address, delta)` = UpdatePagesCachedCount of that page).
/// False when a page is denied or out of range (the range must not be kept).
template <typename Update>
bool Watch(u64 address, u64 size, Update&& update) {
    if (size == 0 || address + size > (u64{1} << AddressBits)) {
        return false;
    }
    const u64 first = address >> PageBits;
    const u64 last = (address + size - 1) >> PageBits;
    for (u64 page = first; page <= last; ++page) {
        if (g_state.denied.Test(page)) {
            return false;
        }
    }
    if (!g_state.any_kept.load(std::memory_order_relaxed)) {
        g_state.any_kept.store(true, std::memory_order_seq_cst);
    }
    for (u64 page = first; page <= last; ++page) {
        (void)g_state.kept.Set(page);
        if (g_state.watched.Test(page)) {
            continue;
        }
        std::scoped_lock lock{g_state.mutex};
        if (g_state.denied.Test(page)) {
            return false;
        }
        if (!g_state.watched.Set(page)) {
            update(page << PageBits, s32{1});
            g_state.watched_list.push_back(page);
            g_state.watched_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return true;
}

/// Whether every page of the range is still watched (after the generations were read).
[[nodiscard]] inline bool AllWatched(u64 address, u64 size) noexcept {
    const u64 first = address >> PageBits;
    const u64 last = (address + size - 1) >> PageBits;
    for (u64 page = first; page <= last; ++page) {
        if (!g_state.watched.Test(page)) {
            return false;
        }
    }
    return true;
}

namespace Detail {
/// Caller holds g_state.mutex. Gives a watched page's count back and fails its keeps.
template <typename Update>
void UnwatchLocked(u64 page, Update& update) {
    if (!g_state.watched.Clear(page)) {
        return;
    }
    update(page << PageBits, s32{-1});
    auto& list = g_state.watched_list;
    if (const auto it = std::find(list.begin(), list.end(), page); it != list.end()) {
        *it = list.back();
        list.pop_back();
    }
    g_state.watched_count.fetch_sub(1, std::memory_order_relaxed);
    // After the bit is clear: a keep that read the generation before this still sees the page
    // watched when it checks again, and fails at its use; one that checks after sees it unwatched.
    RaisePage(page);
}
} // namespace Detail

/// The device pages of [address, address + size) lose their backing (or get another one): the
/// watches on them are given back (before the backing changes, while the old guest address is
/// still known).
template <typename Update>
void Unwatch(u64 address, u64 size, Update&& update) {
    if (g_state.watched_count.load(std::memory_order_relaxed) == 0 || size == 0 ||
        address >= (u64{1} << AddressBits)) {
        return;
    }
    const u64 first = address >> PageBits;
    const u64 last = (std::min)((address + size - 1) >> PageBits, NumPages - 1);
    if (!g_state.watched.AnyRelaxed(first, last)) {
        return;
    }
    std::scoped_lock lock{g_state.mutex};
    if (last - first + 1 > g_state.watched_list.size() * 4) {
        // A large range: walk the few watched pages instead of the range.
        std::vector<u64> pages;
        for (const u64 page : g_state.watched_list) {
            if (page >= first && page <= last) {
                pages.push_back(page);
            }
        }
        for (const u64 page : pages) {
            Detail::UnwatchLocked(page, update);
        }
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        Detail::UnwatchLocked(page, update);
    }
}

/// Once a frame (no cache lock held): a watched page the guest stored to more than `limit` times
/// since the last call is denied (its watch given back, never kept again).
template <typename Update>
void TickFrame(u32 limit, Update&& update) {
    if (g_state.watched_count.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::scoped_lock lock{g_state.mutex};
    std::vector<u64> deny;
    for (const u64 page : g_state.watched_list) {
        const u32 stores = g_state.stores[GenIndex(page)].exchange(0, std::memory_order_relaxed);
        if (stores > limit) {
            deny.push_back(page);
        }
    }
    for (const u64 page : deny) {
        (void)g_state.denied.Set(page);
        Detail::UnwatchLocked(page, update);
        g_state.pages_denied.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace Common::UniformKeepPages
