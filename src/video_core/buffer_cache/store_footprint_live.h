// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/range_sets.h"
#include "common/range_sets.inc"
#include "shader_recompiler/storage_write_extent.h"
#include "video_core/buffer_cache/gpu_written_mirror.h"
#include "video_core/buffer_cache/read_run_stats.h"
#include "video_core/nx2_debug_control.h"

// FRITCHII_STORE_FOOTPRINT, live part (helper_dl_answer.md item 1, helper_dm_answer.md round 2).
//
// Arm: the switch non-zero at boot (ReadRunStats::g_store_footprint, read at its first use). Armed,
// every native written storage binding's write mark carries its program's store footprint into
// the buffer cache's FootprintShadow, which follows gpu_modified_ranges exactly (every Add,
// Subtract and Clear, MirroredRangeSet::SetFollower): `unclaimed` holds the GPU-modified bytes
// some mark without a footprint made GPU-modified since they were last clean, and each entry the
// bytes marks of one footprint (program binding and pointer phase) made GPU-modified. A byte in
// gpu_modified_ranges is in `unclaimed` or in an entry (or both).
//
// Live value (debug-server switch "FRITCHII_STORE_FOOTPRINT", read at each guest read):
//   0  as before (the shadow is kept while armed, so a switch to 1 is exact at once);
//   1  a guest read whose GPU-modified bytes are all in entries whose footprints exclude them,
//      and none in `unclaimed`, is answered from guest memory without waiting: no mark since
//      those bytes were last clean writes them, so guest memory holds them. Nothing else changes:
//      the cache's sets, downloads and fences stay as they were before the read; the read's pages
//      get their read traps back (the core took them for this read) so the next read of a
//      GPU-written byte there still traps;
//   2  as 1, but every 16th such read waits as before and compares its bytes before the wait with
//      after the landing ("skipped reads that would have differed", must be 0), and every locked
//      download landing compares the bytes it writes in claimed-outside places with guest memory
//      ("landings that would overwrite guest bytes outside a footprint", must be 0);
//   3  the dry count only (read_run_stats.h's "Store footprint" lines; needs 3 at boot: the read
//      ring is on only when the boot value is 3).
namespace VideoCommon::StoreFootprintLive {

/// The live value (0 while unset).
[[nodiscard]] inline int Mode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_store_footprint_live,
                                            "FRITCHII_STORE_FOOTPRINT", 0);
}

/// Every 16th read =2 would answer from guest memory waits and compares instead.
inline constexpr u32 VerifyEvery = 16;
/// =2: bytes compared per landing at most (counted, so the line says how much was looked at).
inline constexpr u64 LandingCheckBytes = 256 * 1024;

struct Stats {
    // Guest reads asked with live 1 or 2 (pages GPU-modified, not preflushable)
    std::atomic<u64> asked{};
    std::atomic<u64> skipped{};       ///< answered from guest memory
    std::atomic<u64> inside{};        ///< a GPU-modified byte inside an entry's footprint
    std::atomic<u64> unclaimed{};     ///< a GPU-modified byte no footprint claims
    std::atomic<u64> not_modified{};  ///< the read's own bytes hold no GPU-modified byte
    std::atomic<u64> rearmed{};       ///< pages whose read traps were armed again
    // Binds (armed)
    std::atomic<u64> bind_full{};     ///< footprint used as recorded
    std::atomic<u64> bind_reduced{};  ///< modulus not a power of two, wrap not ruled out: reduced
    std::atomic<u64> bind_none{};     ///< no footprint (none recorded, multi-range, claims nothing)
    // =2
    std::atomic<u64> verify_sampled{};
    std::atomic<u64> verify_compared{};
    std::atomic<u64> verify_differed{}; ///< must be 0
    std::atomic<u64> landings_checked{};
    std::atomic<u64> landing_bytes_checked{};
    std::atomic<u64> landings_differed{};      ///< must be 0
    std::atomic<u64> landing_bytes_differed{}; ///< must be 0
};

inline Stats& GetStats() {
    static Stats stats;
    return stats;
}

inline void Bump(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

/// How a read stands to the shadow.
enum class Answer : u8 {
    NotModified, ///< none of its bytes is GPU-modified
    Unclaimed,   ///< a GPU-modified byte of it some mark without a footprint made so
    Inside,      ///< a GPU-modified byte of it inside an entry's footprint
    Outside,     ///< every GPU-modified byte of it claimed, and outside every footprint over it
};

/// Who made the GPU-modified bytes so, kept beside gpu_modified_ranges under the buffer cache's
/// mutex (it follows the set's every change).
class FootprintShadow final : public RangeSetFollower {
public:
    struct Entry {
        u32 id{};
        u64 phase{}; ///< the pointer's device address modulo the footprint's modulus
        const Shader::StorageStoreFootprint* footprint{};
        Common::RangeSet<u64> ranges;
    };

    void FollowAdd(u64 begin, u64 size) override {
        unclaimed.Add(begin, size);
    }

    void FollowSubtract(u64 begin, u64 size) override {
        unclaimed.Subtract(begin, size);
        for (auto it = entries.begin(); it != entries.end();) {
            it->ranges.Subtract(begin, size);
            if (it->ranges.Empty()) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }

    void FollowClear() override {
        unclaimed.Clear();
        entries.clear();
    }

    /// A mark of a binding with footprint `id` (ReadRunStats' table) whose offsets count from
    /// `base`. Call it before the set's Add with claimed = true.
    void AddClaimed(u32 id, u64 base, u64 begin, u64 size) {
        const Shader::StorageStoreFootprint* footprint = ReadRunStats::FootprintOf(id);
        if (footprint == nullptr || !footprint->Valid() || footprint->covers_all) {
            unclaimed.Add(begin, size);
            return;
        }
        const u64 phase = base % footprint->modulus;
        for (Entry& entry : entries) {
            if (entry.id == id && entry.phase == phase) {
                entry.ranges.Add(begin, size);
                return;
            }
        }
        Entry& entry = entries.emplace_back();
        entry.id = id;
        entry.phase = phase;
        entry.footprint = footprint;
        entry.ranges.Add(begin, size);
    }

    /// The read [begin, begin + size) against the shadow; `modified` is gpu_modified_ranges.
    template <typename Modified>
    [[nodiscard]] Answer Check(u64 begin, u64 size, const Modified& modified) const {
        Common::RangeSet<u64> rest;
        modified.ForEachInRange(begin, size, [&](u64 lower, u64 upper) {
            rest.Add(lower, upper - lower);
        });
        if (rest.Empty()) {
            return Answer::NotModified;
        }
        if (unclaimed.Intersects(begin, size)) {
            return Answer::Unclaimed;
        }
        bool inside = false;
        for (const Entry& entry : entries) {
            entry.ranges.ForEachInRange(begin, size, [&](u64 lower, u64 upper) {
                rest.Subtract(lower, upper - lower);
                inside = inside ||
                         Shader::StoreFootprintHits(*entry.footprint,
                                                    static_cast<s64>(lower - entry.phase),
                                                    upper - lower);
            });
        }
        if (inside) {
            return Answer::Inside;
        }
        // A GPU-modified byte no entry holds would be a missed Add: never answer it
        return rest.Empty() ? Answer::Outside : Answer::Unclaimed;
    }

    /// =2: whether byte `address` is GPU-modified only by footprint marks that exclude it.
    [[nodiscard]] bool ClaimedOutside(u64 address) const {
        if (unclaimed.Intersects(address, 1)) {
            return false;
        }
        bool claimed = false;
        for (const Entry& entry : entries) {
            if (!entry.ranges.Intersects(address, 1)) {
                continue;
            }
            claimed = true;
            if (Shader::StoreFootprintHits(*entry.footprint,
                                           static_cast<s64>(address - entry.phase), 1)) {
                return false;
            }
        }
        return claimed;
    }

    /// Whether any entry holds a byte of [begin, begin + size).
    [[nodiscard]] bool AnyClaimed(u64 begin, u64 size) const {
        return std::ranges::any_of(entries, [&](const Entry& entry) {
            return entry.ranges.Intersects(begin, size);
        });
    }

    [[nodiscard]] size_t NumEntries() const {
        return entries.size();
    }

    /// Takes the set's ranges as unclaimed (the set had them before the shadow followed it).
    template <typename Modified>
    void Start(const Modified& modified) {
        FollowClear();
        modified.ForEach([&](u64 lower, u64 upper) { unclaimed.Add(lower, upper - lower); });
    }

private:
    Common::RangeSet<u64> unclaimed;
    std::vector<Entry> entries;
};

// ---- =2: the sampled reads' bytes, compared once the read's wait is over (GPU::OnCPURead).

struct PendingVerify {
    u64 address{};
    u32 size{};
    bool used{};
    std::array<u8, 64> before{};
};

struct VerifyQueue {
    std::mutex mutex;
    std::array<PendingVerify, 8> slots{};
    std::atomic<u32> count{};
};

inline VerifyQueue& GetVerifyQueue() {
    static VerifyQueue queue;
    return queue;
}

/// BufferCache::GetFlushArea (=2): a read that would have been answered from guest memory and
/// waits instead; `bytes` are its first `size` (at most 64) bytes in guest memory now.
inline void AddVerify(u64 address, u32 size, const u8* bytes) {
    VerifyQueue& q = GetVerifyQueue();
    std::scoped_lock lock{q.mutex};
    for (PendingVerify& slot : q.slots) {
        if (!slot.used) {
            slot.used = true;
            slot.address = address;
            slot.size = (std::min<u32>)(size, static_cast<u32>(slot.before.size()));
            std::memcpy(slot.before.data(), bytes, slot.size);
            q.count.fetch_add(1, std::memory_order_release);
            Bump(GetStats().verify_sampled);
            return;
        }
    }
}

/// GPU::OnCPURead, once the read of [address, address + size) is answered: compares the sampled
/// reads it covers. `read(address, out, size)` reads guest memory.
template <typename Read>
void FinishVerify(u64 address, u64 size, Read&& read) {
    VerifyQueue& q = GetVerifyQueue();
    if (q.count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::scoped_lock lock{q.mutex};
    for (PendingVerify& slot : q.slots) {
        if (!slot.used || slot.address < address || slot.address + slot.size > address + size) {
            continue;
        }
        std::array<u8, 64> after{};
        read(slot.address, after.data(), slot.size);
        Stats& s = GetStats();
        Bump(s.verify_compared);
        if (std::memcmp(after.data(), slot.before.data(), slot.size) != 0) {
            const u64 differed = s.verify_differed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (differed <= 16) {
                LOG_ERROR(HW_GPU,
                          "Store footprint (FRITCHII_STORE_FOOTPRINT=2): a read at {:#x} +{} that "
                          "=1 would answer from guest memory changed at its landing (must be 0)",
                          slot.address, slot.size);
            }
        }
        slot.used = false;
        q.count.fetch_sub(1, std::memory_order_release);
    }
}

/// BufferCache::TickFrame while armed: the line every 10 s.
inline void MaybeLog(size_t entries) {
    static std::chrono::steady_clock::time_point start{};
    static u64 frames = 0;
    ++frames;
    const auto now = std::chrono::steady_clock::now();
    if (start == std::chrono::steady_clock::time_point{}) {
        start = now;
        frames = 0;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - start).count();
    if (seconds < 10.0) {
        return;
    }
    const double per_frame = static_cast<double>((std::max<u64>)(frames, 1));
    start = now;
    frames = 0;
    Stats& s = GetStats();
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 asked = take(s.asked);
    const u64 skipped = take(s.skipped);
    LOG_INFO(HW_GPU,
             "Store footprint live (FRITCHII_STORE_FOOTPRINT={}, armed) last {:.0f} s: reads asked "
             "{} | answered from guest memory {} ({:.1f} a frame) | waited as before: inside a "
             "footprint {}, a byte no footprint claims {}, own bytes not GPU-modified {} | read "
             "trap pages armed again {} | binds: footprint {}, reduced to its power-of-two part "
             "(wrap not ruled out) {}, none {} | shadow entries {} | =2 verify: sampled {}, "
             "compared {}, would have differed {} (must be 0) | =2 landings: checked {} ({} KiB), "
             "would overwrite guest bytes outside a footprint {} ({} bytes, must be 0)",
             Mode(), seconds, asked, skipped, static_cast<double>(skipped) / per_frame,
             take(s.inside), take(s.unclaimed), take(s.not_modified), take(s.rearmed),
             take(s.bind_full), take(s.bind_reduced), take(s.bind_none), entries,
             take(s.verify_sampled), take(s.verify_compared), take(s.verify_differed),
             take(s.landings_checked), take(s.landing_bytes_checked) >> 10,
             take(s.landings_differed), take(s.landing_bytes_differed));
}

} // namespace VideoCommon::StoreFootprintLive
