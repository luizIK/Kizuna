// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>

#include "common/common_types.h"
#include "common/fritchii_page_flags.h"
#include "common/fritchii_s4_count.h"

/// Who wrote game memory (FRITCHII_MEMSTAGE_READ_LOG, the memory seam's read log): every write
/// emulator code makes to game memory, on any thread, after its bytes landed, by device address.
/// While off, one relaxed load per write (two in Note: the bank marks' gate). The read log and the hook list are explained in
/// vk_mem_stage.cpp ("Who wrote").
namespace Common::WriteRing {

inline std::atomic<bool> g_on{false};

inline constexpr u64 Size = 4096;

struct Slot {
    std::atomic<u64> seq{};
    std::atomic<u64> address{};
    std::atomic<u64> size{};
    std::atomic<const char*> site{};
    std::atomic<const char*> context{};
    std::atomic<const char*> via{};
    std::array<std::atomic<u64>, 2> thread{};
    std::atomic<bool> landing{};
};

struct State {
    std::atomic<u64> next{};
    std::array<Slot, Size> slots{};
};
inline State g_state;

/// A copy of one slot.
struct Entry {
    u64 seq;
    u64 address;
    u64 size;
    const char* site;    ///< Innermost site scope, nullptr: none
    const char* context; ///< Outermost site scope
    const char* via;     ///< The write function
    std::array<char, 17> thread;
    bool landing; ///< A download landing (GPU bytes written back)
};

inline thread_local std::array<u64, 2> t_thread{};
inline thread_local const char* t_site = nullptr;
inline thread_local const char* t_context = nullptr;
inline thread_local bool t_landing = false;
/// Writes this thread makes now are not noted (the memory stage's own constant buffer words with
/// FRITCHII_MEMSTAGE_AHEAD: its own ring names them, with their entry, and they would wrap this one).
inline thread_local bool t_quiet = false;

/// Common::SetCurrentThreadName: the name the ring keeps (16 characters).
inline void NameThread(const char* name) noexcept {
    std::array<char, 16> chars{};
    if (name != nullptr) {
        std::strncpy(chars.data(), name, chars.size());
    }
    std::memcpy(t_thread.data(), chars.data(), chars.size());
}

inline void Record(u64 address, u64 size, const char* via) noexcept {
    // FRITCHII_BANK_CENSUS=2: which pages the ring heard of, for writes the ring lost to wrapping.
    g_ring_write_pages.Mark(address, size);
    const u64 seq = g_state.next.fetch_add(1, std::memory_order_relaxed) + 1;
    Slot& slot = g_state.slots[seq & (Size - 1)];
    slot.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.address.store(address, std::memory_order_relaxed);
    slot.size.store(size, std::memory_order_relaxed);
    slot.site.store(t_site, std::memory_order_relaxed);
    slot.context.store(t_context, std::memory_order_relaxed);
    slot.via.store(via, std::memory_order_relaxed);
    slot.thread[0].store(t_thread[0], std::memory_order_relaxed);
    slot.thread[1].store(t_thread[1], std::memory_order_relaxed);
    slot.landing.store(t_landing, std::memory_order_relaxed);
    slot.seq.store(seq, std::memory_order_release);
}

/// After the bytes of [address, address + size) landed in game memory (device address).
/// Writes kept out while t_quiet (counted: the read log's line says how many).
inline std::atomic<u64> g_quiet{0};
/// With the bank marks on (BankMarks: FRITCHII_BANK_FLAGGED or FRITCHII_BANK_CENSUS), also marks
/// the pages for the native bank check (g_buffer_write_pages, after the bytes landed: a bind that
/// takes the mark reads them). The census found writers that set no other flag (query
/// completions writing reports into the game's banks).
inline void Note(u64 address, u64 size, const char* via) noexcept {
    if (BankMarks()) [[unlikely]] {
        g_buffer_write_pages.Mark(address, size);
    }
    // FRITCHII_S4_COUNT: who wrote the page (stream upload on E, or another thread's write).
    if (S4Count::On()) [[unlikely]] {
        S4Count::NoteWrite(address, size);
    }
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        if (t_quiet) {
            g_quiet.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Record(address, size, via);
    }
}

/// The newest sequence number handed out.
[[nodiscard]] inline u64 Sequence() noexcept {
    return g_state.next.load(std::memory_order_acquire);
}

/// Calls `func(const Entry&)` for each entry newer than `after`, newest first, until it returns
/// true. False when the entries right after `after` were overwritten already (the ring wrapped).
template <typename Func>
bool ForEachSince(u64 after, Func&& func) {
    const u64 now = Sequence();
    const bool complete = now - after <= Size;
    const u64 oldest = complete ? after + 1 : now - Size + 1;
    for (u64 seq = now; seq >= oldest && seq != 0; --seq) {
        const Slot& slot = g_state.slots[seq & (Size - 1)];
        if (slot.seq.load(std::memory_order_acquire) != seq) {
            continue; // being written, or already reused
        }
        Entry entry{};
        entry.seq = seq;
        entry.address = slot.address.load(std::memory_order_relaxed);
        entry.size = slot.size.load(std::memory_order_relaxed);
        entry.site = slot.site.load(std::memory_order_relaxed);
        entry.context = slot.context.load(std::memory_order_relaxed);
        entry.via = slot.via.load(std::memory_order_relaxed);
        const std::array<u64, 2> thread{slot.thread[0].load(std::memory_order_relaxed),
                                        slot.thread[1].load(std::memory_order_relaxed)};
        std::memcpy(entry.thread.data(), thread.data(), 16);
        entry.landing = slot.landing.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot.seq.load(std::memory_order_relaxed) != seq) {
            continue;
        }
        if (func(entry)) {
            return complete;
        }
    }
    return complete;
}

/// The device reads inside a ReadSiteScope (always on, two thread-local stores): the "site" of
/// the unmapped device read lines (Core::ReportUnmappedRead), innermost and outermost.
inline thread_local const char* t_read_site = nullptr;
inline thread_local const char* t_read_context = nullptr;
class ReadSiteScope {
public:
    explicit ReadSiteScope(const char* name) noexcept
        : previous_site{t_read_site}, owns_context{t_read_context == nullptr} {
        t_read_site = name;
        if (owns_context) {
            t_read_context = name;
        }
    }
    ~ReadSiteScope() {
        t_read_site = previous_site;
        if (owns_context) {
            t_read_context = nullptr;
        }
    }
    ReadSiteScope(const ReadSiteScope&) = delete;
    ReadSiteScope& operator=(const ReadSiteScope&) = delete;

private:
    const char* previous_site;
    bool owns_context;
};

/// Names the writes inside it (innermost names the site, outermost the context).
class SiteScope {
public:
    SiteScope(const char* name, bool landing) noexcept {
        if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
            Enter(name, landing);
        }
    }
    ~SiteScope() {
        if (active) [[unlikely]] {
            t_site = previous_site;
            t_landing = previous_landing;
            if (owns_context) {
                t_context = nullptr;
            }
        }
    }
    SiteScope(const SiteScope&) = delete;
    SiteScope& operator=(const SiteScope&) = delete;

private:
    void Enter(const char* name, bool landing) noexcept {
        active = true;
        previous_site = t_site;
        previous_landing = t_landing;
        t_site = name;
        t_landing = landing;
        if (t_context == nullptr) {
            t_context = name;
            owns_context = true;
        }
    }

    const char* previous_site{};
    bool previous_landing{};
    bool active{};
    bool owns_context{};
};

} // namespace Common::WriteRing
