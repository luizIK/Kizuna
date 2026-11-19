// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <memory>

#include "video_core/nx2_write_track.h"

namespace VideoCommon::WriteTrack {

namespace {

/// Verify mode's journal per thread: about a few ms of a busy writer's stores.
constexpr size_t JournalSize = 16384;
/// Threads that write cached guest memory (CPU cores, HLE service threads); more go untracked.
constexpr size_t MaxSlots = 256;
/// A write's range packed in one word: device address << 16 | size (a write never crosses a
/// device page, so it is at most 4 KiB).
constexpr u64 SizeBits = 16;
constexpr u64 SizeMask = (u64{1} << SizeBits) - 1;

struct JournalEntry {
    std::atomic<u64> range{};
    std::atomic<u64> epoch{};
};

struct alignas(64) Slot {
    /// Odd while a write is open; every Open makes a new odd value.
    std::atomic<u64> sequence{0};
    std::atomic<u64> range{0};
    std::atomic<u32> core{0};
    std::atomic<u64> written{0};
    std::atomic<std::array<JournalEntry, JournalSize>*> journal{nullptr};
};

std::array<std::atomic<Slot*>, MaxSlots> slots{};
std::atomic<u32> slot_count{0};
std::atomic<u64> epoch_now{1};
std::atomic<u64> untracked_opens{0};

thread_local Slot* t_slot = nullptr;
thread_local bool t_slot_refused = false;
thread_local bool t_open = false;

u64 Pack(u64 address, u64 size) {
    return (address << SizeBits) | (std::min)(size, SizeMask);
}

/// The calling thread's slot, made on its first write (slots live as long as the process: a
/// thread that ends leaves an even, idle one behind).
Slot* ThisSlot() {
    if (t_slot != nullptr || t_slot_refused) {
        return t_slot;
    }
    const u32 index = slot_count.fetch_add(1, std::memory_order_relaxed);
    if (index >= MaxSlots) {
        t_slot_refused = true;
        return nullptr;
    }
    t_slot = new Slot{};
    slots[index].store(t_slot, std::memory_order_release);
    return t_slot;
}

} // Anonymous namespace

void Open(u64 address, u64 size, u32 core) {
    Slot* const slot = ThisSlot();
    if (slot == nullptr) {
        untracked_opens.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    slot->core.store(core, std::memory_order_relaxed);
    // The range before the sequence: a reader that sees this sequence reads this range or a later
    // write's, and then sees the sequence move on (the next Open stores its range first too).
    slot->range.store(Pack(address, size), std::memory_order_release);
    const u64 sequence = slot->sequence.load(std::memory_order_relaxed);
    slot->sequence.store((sequence | 1) + ((sequence & 1) << 1), std::memory_order_release);
    // Pairs with the gatherer's fence: either its gather left this write's mark in the dirty
    // manager, or its ForEachInFlight sees this write open.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    t_open = true;
    if (!g_journal.load(std::memory_order_relaxed)) {
        return;
    }
    auto* journal = slot->journal.load(std::memory_order_relaxed);
    if (journal == nullptr) {
        journal = new std::array<JournalEntry, JournalSize>{};
        slot->journal.store(journal, std::memory_order_release);
    }
    const u64 index = slot->written.load(std::memory_order_relaxed);
    JournalEntry& entry = (*journal)[index % JournalSize];
    entry.epoch.store(epoch_now.load(std::memory_order_relaxed), std::memory_order_relaxed);
    entry.range.store(Pack(address, size), std::memory_order_relaxed);
    slot->written.store(index + 1, std::memory_order_release);
}

void Close() {
    if (!t_open) {
        return;
    }
    t_open = false;
    Slot* const slot = t_slot;
    if (slot == nullptr) {
        return;
    }
    const u64 sequence = slot->sequence.load(std::memory_order_relaxed);
    if ((sequence & 1) != 0) {
        // Release: a gatherer that sees it even sees the bytes stored before.
        slot->sequence.store(sequence + 1, std::memory_order_release);
    }
}

void ForEachInFlight(void (*callback)(void* argument, u64 address, u64 size, u32 slot, u32 core),
                     void* argument) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const u32 count = (std::min)(slot_count.load(std::memory_order_acquire), u32{MaxSlots});
    for (u32 index = 0; index < count; ++index) {
        Slot* const slot = slots[index].load(std::memory_order_acquire);
        if (slot == nullptr) {
            continue;
        }
        u64 sequence = slot->sequence.load(std::memory_order_acquire);
        // A few tries for a consistent read; a thread writing that fast gets its latest range.
        for (int tries = 0; (sequence & 1) != 0; ++tries) {
            const u64 range = slot->range.load(std::memory_order_acquire);
            const u64 again = slot->sequence.load(std::memory_order_acquire);
            if (again == sequence || tries == 3) {
                callback(argument, range >> SizeBits, range & SizeMask, index,
                         slot->core.load(std::memory_order_relaxed));
                break;
            }
            sequence = again;
        }
    }
}

u64 NewEpoch() {
    const u64 epoch = epoch_now.fetch_add(1, std::memory_order_seq_cst) + 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return epoch;
}

Rewrite FindRewrite(u64 address, u64 size, u64 epoch) {
    Rewrite result{.kind = Rewrite::Kind::None, .slot = 0, .core = 0, .writers = 0};
    bool wrapped = false;
    const u64 end = address + size;
    const u32 count = (std::min)(slot_count.load(std::memory_order_acquire), u32{MaxSlots});
    for (u32 index = 0; index < count; ++index) {
        Slot* const slot = slots[index].load(std::memory_order_acquire);
        if (slot == nullptr) {
            continue;
        }
        const auto* const journal = slot->journal.load(std::memory_order_acquire);
        if (journal == nullptr) {
            continue;
        }
        const u64 written = slot->written.load(std::memory_order_acquire);
        const u64 oldest = written > JournalSize ? written - JournalSize : 0;
        bool found = false;
        bool reached = false;
        for (u64 at = written; at > oldest; --at) {
            const JournalEntry& entry = (*journal)[(at - 1) % JournalSize];
            if (entry.epoch.load(std::memory_order_relaxed) < epoch) {
                reached = true;
                break;
            }
            const u64 range = entry.range.load(std::memory_order_relaxed);
            const u64 begin = range >> SizeBits;
            if (begin < end && address < begin + (range & SizeMask)) {
                found = true;
                break;
            }
        }
        if (found) {
            if (result.writers++ == 0) {
                result.kind = Rewrite::Kind::Found;
                result.slot = index;
                result.core = slot->core.load(std::memory_order_relaxed);
            }
        } else if (!reached && oldest != 0) {
            wrapped = true;
        }
    }
    if (result.kind == Rewrite::Kind::None && wrapped) {
        result.kind = Rewrite::Kind::Unknown;
    }
    return result;
}

u32 SlotCount() {
    return (std::min)(slot_count.load(std::memory_order_relaxed), u32{MaxSlots});
}

u64 TakeUntrackedOpens() {
    return untracked_opens.exchange(0, std::memory_order_relaxed);
}

} // namespace VideoCommon::WriteTrack
