// SPDX-License-Identifier: GPL-3.0-or-later

/* Fritchii: FRITCHII_JIT_SHARED (stage 1 of one JIT code cache for all guest cores). */

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "common/common_types.h"

namespace Dynarmic::Backend::X64 {

/// Fritchii (FRITCHII_JIT_SHARED): what the guest cores of one process share when they run one
/// code cache: the block map read without a lock, the compile lock, the list of removed blocks
/// each core drops from its own dispatch table, and which cores are inside generated code (so a
/// whole-cache clear can wait for all of them to be out).
///
/// Block map: open addressing, location key -> entrypoint. A key is written once, after its
/// value; removal zeroes the value (the key stays as a tombstone, a later compile of the same
/// key reuses it). Readers take no lock; writers hold `compile_mutex`. A grown table replaces the
/// old one, which is freed once every core that was in generated code when it was replaced has
/// left it (a reader only reads the table while its core is marked in code, or under the lock).
/// Each value keeps, in its top 16 bits, which cores looked the block up (the counters' "used by
/// N cores").
struct SharedCode {
    static constexpr u64 empty_key = ~u64(0);  ///< no location descriptor has every bit set
    static constexpr u64 code_mask = (u64(1) << 48) - 1;
    static constexpr size_t max_cores = 16;

    struct Slot {
        std::atomic<u64> key{empty_key};
        std::atomic<u64> value{0};  ///< entrypoint | core bits << 48, 0 = removed
    };
    struct Table {
        explicit Table(unsigned bits_) : bits(bits_), mask((size_t(1) << bits_) - 1), slots(new Slot[size_t(1) << bits_]) {}
        unsigned bits;
        size_t mask;
        std::unique_ptr<Slot[]> slots;
        size_t Index(u64 key) const {
            return size_t((key * 0x9E3779B97F4A7C15ull) >> (64 - bits));
        }
    };
    /// One per guest core (index = processor id): inside generated code or a lock-free lookup.
    struct alignas(64) CoreSlot {
        std::atomic<bool> in_code{false};
        std::atomic<u64> generation{0};  ///< +1 at every exit from generated code
        std::atomic<bool> registered{false};
    };
    struct Retired {
        std::unique_ptr<Table> table;
        std::array<u64, max_cores> generation{};
        u32 waiting = 0;  ///< bit k: core k was in code when the table was replaced
    };

    static constexpr unsigned initial_bits = 16;

    SharedCode() {
        tables.push_back(std::make_unique<Table>(initial_bits));
        current.store(tables.back().get(), std::memory_order_release);
    }

    static u64 CoreBit(unsigned core) {
        return u64(1) << (48 + (core & (max_cores - 1)));
    }

    /// Lock-free (the caller's core is marked in code, or the caller holds compile_mutex).
    const void* Find(u64 key, u64 core_bit) {
        const Table* const t = current.load(std::memory_order_acquire);
        for (size_t i = t->Index(key);; i = (i + 1) & t->mask) {
            Slot& slot = t->slots[i];
            const u64 k = slot.key.load(std::memory_order_acquire);
            if (k == key) {
                const u64 v = slot.value.load(std::memory_order_acquire);
                if ((v & code_mask) == 0) {
                    return nullptr;
                }
                if (core_bit != 0 && (v & core_bit) == 0) [[unlikely]] {
                    NoteFirstLookup(slot.value.fetch_or(core_bit, std::memory_order_relaxed), core_bit);
                }
                return reinterpret_cast<const void*>(uintptr_t(v & code_mask));
            }
            if (k == empty_key) {
                return nullptr;
            }
        }
    }

    /// Under compile_mutex. `compiling_core_bit` is the core that compiled it.
    void Insert(u64 key, const void* code) {
        const u64 value = u64(reinterpret_cast<uintptr_t>(code)) | compiling_core_bit;
        Table* t = current.load(std::memory_order_relaxed);
        if ((used + 1) * 4 > (t->mask + 1) * 3) {
            t = Grow();
        }
        for (size_t i = t->Index(key);; i = (i + 1) & t->mask) {
            Slot& slot = t->slots[i];
            const u64 k = slot.key.load(std::memory_order_relaxed);
            if (k == key) {
                const u64 old = slot.value.exchange(value, std::memory_order_release);
                if ((old & code_mask) != 0) {
                    users[size_t(std::popcount(old >> 48))].fetch_sub(1, std::memory_order_relaxed);
                }
                users[1].fetch_add(1, std::memory_order_relaxed);
                break;
            }
            if (k == empty_key) {
                slot.value.store(value, std::memory_order_relaxed);
                slot.key.store(key, std::memory_order_release);
                ++used;
                users[1].fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        if ((++inserts & 0xfff) == 0 && !retired.empty()) {
            FreeRetired();
        }
    }

    /// Under compile_mutex. Also logged for the cores' own dispatch tables.
    void Remove(u64 key) {
        const Table* const t = current.load(std::memory_order_relaxed);
        for (size_t i = t->Index(key);; i = (i + 1) & t->mask) {
            Slot& slot = t->slots[i];
            const u64 k = slot.key.load(std::memory_order_relaxed);
            if (k == key) {
                const u64 old = slot.value.exchange(0, std::memory_order_release);
                if ((old & code_mask) != 0) {
                    users[size_t(std::popcount(old >> 48))].fetch_sub(1, std::memory_order_relaxed);
                    removals.fetch_add(1, std::memory_order_relaxed);
                }
                break;
            }
            if (k == empty_key) {
                break;
            }
        }
        // A core still reading a replaced table must not find it there either.
        for (const Retired& r : retired) {
            const Table& old_table = *r.table;
            for (size_t i = old_table.Index(key);; i = (i + 1) & old_table.mask) {
                Slot& slot = old_table.slots[i];
                const u64 k = slot.key.load(std::memory_order_relaxed);
                if (k == key) {
                    slot.value.store(0, std::memory_order_release);
                    break;
                }
                if (k == empty_key) {
                    break;
                }
            }
        }
        std::scoped_lock lock{removal_mutex};
        removal_log.push_back(key);
        if (removal_log.size() > removal_log_limit) {
            const size_t drop = removal_log.size() / 2;
            removal_log.erase(removal_log.begin(), removal_log.begin() + std::ptrdiff_t(drop));
            removal_base += drop;
        }
        removal_seq.store(removal_base + removal_log.size(), std::memory_order_release);
    }

    /// Whole-cache clear: under compile_mutex with every core out of generated code.
    void Clear() {
        retired.clear();
        tables.clear();
        tables.push_back(std::make_unique<Table>(initial_bits));
        current.store(tables.back().get(), std::memory_order_release);
        used = 0;
        for (size_t i = 1; i < users.size(); ++i) {
            users[i].store(0, std::memory_order_relaxed);
        }
        std::scoped_lock lock{removal_mutex};
        removal_base += removal_log.size();
        removal_log.clear();
        removal_seq.store(removal_base, std::memory_order_release);
    }

    size_t Entries() const { return used; }

    /// Inside generated code (or a lock-free lookup outside it) / out again.
    std::mutex compile_mutex;
    std::array<CoreSlot, max_cores> cores{};
    std::atomic<bool> clear_pending{false};
    std::atomic<u64> clear_epoch{0};
    u64 compiling_core_bit = 0;  ///< under compile_mutex: the core compiling now

    /// Removed blocks in order (seq = removal_base + index), for the cores' dispatch tables.
    std::mutex removal_mutex;
    std::vector<u64> removal_log;
    u64 removal_base = 0;
    std::atomic<u64> removal_seq{0};
    static constexpr size_t removal_log_limit = size_t(1) << 16;

    // Counters (FRITCHII_JIT_SHARED's line).
    std::atomic<u64> compiled{0};             ///< blocks compiled into the shared cache
    std::atomic<u64> taken_after_wait{0};     ///< compiles another core finished while this one waited
    std::atomic<u64> lock_waits{0};           ///< compiles that found the lock taken
    std::atomic<u64> lock_wait_ns{0};         ///< time waited for it
    std::atomic<u64> other_core_lookups{0};   ///< first lookups of a block another core compiled or ran
    std::atomic<u64> removals{0};
    std::atomic<u64> full_clears{0};
    std::atomic<u64> patch_shape_errors{0};   ///< must be 0: a patch site not in the atomic shape
    std::atomic<u64> verify_checks{0};        ///< FRITCHII_JIT_SHARED=2
    std::atomic<u64> verify_mismatches{0};    ///< FRITCHII_JIT_SHARED=2, must be 0
    /// users[n]: blocks looked up by n cores (the compiling core counts as one).
    std::array<std::atomic<u64>, max_cores + 1> users{};

private:
    void NoteFirstLookup(u64 old, u64 core_bit) {
        if ((old & code_mask) == 0 || (old & core_bit) != 0) {
            return;
        }
        const size_t before = size_t(std::popcount(old >> 48));
        if (before < users.size() - 1) {
            users[before].fetch_sub(1, std::memory_order_relaxed);
            users[before + 1].fetch_add(1, std::memory_order_relaxed);
        }
        if (before != 0) {
            other_core_lookups.fetch_add(1, std::memory_order_relaxed);
        }
    }

    Table* Grow() {
        const Table* const old = current.load(std::memory_order_relaxed);
        auto grown = std::make_unique<Table>(old->bits + 1);
        size_t live = 0;
        for (size_t i = 0; i <= old->mask; ++i) {
            const u64 k = old->slots[i].key.load(std::memory_order_relaxed);
            const u64 v = old->slots[i].value.load(std::memory_order_relaxed);
            if (k == empty_key || (v & code_mask) == 0) {
                continue;  // tombstones are dropped here
            }
            for (size_t j = grown->Index(k);; j = (j + 1) & grown->mask) {
                if (grown->slots[j].key.load(std::memory_order_relaxed) == empty_key) {
                    grown->slots[j].value.store(v, std::memory_order_relaxed);
                    grown->slots[j].key.store(k, std::memory_order_relaxed);
                    break;
                }
            }
            ++live;
        }
        Table* const result = grown.get();
        // seq_cst: a core that marks itself in code after this sees the new table (see FreeRetired).
        current.store(result, std::memory_order_seq_cst);
        Retired r;
        for (size_t c = 0; c < max_cores; ++c) {
            if (cores[c].in_code.load(std::memory_order_seq_cst)) {
                r.waiting |= u32(1) << c;
                r.generation[c] = cores[c].generation.load(std::memory_order_acquire);
            }
        }
        // The old table moves to the retired list (it is in `tables` until freed).
        for (auto it = tables.begin(); it != tables.end(); ++it) {
            if (it->get() == old) {
                r.table = std::move(*it);
                tables.erase(it);
                break;
            }
        }
        tables.push_back(std::move(grown));
        retired.push_back(std::move(r));
        used = live;
        FreeRetired();
        return result;
    }

    void FreeRetired() {
        for (auto it = retired.begin(); it != retired.end();) {
            u32 still = 0;
            for (size_t c = 0; c < max_cores; ++c) {
                if (((it->waiting >> c) & 1) != 0 &&
                    cores[c].generation.load(std::memory_order_acquire) == it->generation[c]) {
                    still |= u32(1) << c;
                }
            }
            it->waiting = still;
            it = still == 0 ? retired.erase(it) : it + 1;
        }
    }

    std::atomic<Table*> current{nullptr};
    std::vector<std::unique_ptr<Table>> tables;  ///< the current table (and nothing else)
    std::vector<Retired> retired;
    size_t used = 0;  ///< keys in the current table, tombstones included
    u64 inserts = 0;
};

}  // namespace Dynarmic::Backend::X64
