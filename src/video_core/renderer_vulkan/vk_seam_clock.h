// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <source_location>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

class MemStage;
class Scheduler;

/// FRITCHII_MEMSTAGE_TICKS >= 1: stamps are packet numbers, freed through the ticks the GPU
/// thread publishes (vk_mem_stage.cpp, "Packet ticks").
class SeamTicks {
public:
    static constexpr u64 Never = (std::numeric_limits<u64>::max)();

    explicit SeamTicks(MasterSemaphore& master, bool verify);
    ~SeamTicks();

    SeamTicks(const SeamTicks&) = delete;
    SeamTicks& operator=(const SeamTicks&) = delete;

    [[nodiscard]] bool Verifying() const noexcept {
        return verify;
    }

    [[nodiscard]] u64 Current() const noexcept {
        return open.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 Now() noexcept {
        const u64 number = Mine();
        if (verify) [[unlikely]] {
            NoteStamp(number);
        }
        return number;
    }

    /// Step 7 (vk_mem_stage.cpp, "Stamps per thread"): while the window is open each thread
    /// stamps with its own packet, not with the newest number anyone opened. GPUMem: the packet
    /// it fills or the operation it runs, under its mark (`stage` true). The back: the packet it
    /// holds or fills, and outside one the last it had, while that is not published yet. Else
    /// (no packet, or the ring off) `open`, as before.
    [[nodiscard]] u64 Mine() const noexcept {
        if (per_thread.load(std::memory_order_relaxed)) [[unlikely]] {
            const u64 mine = t_stamp;
            if (mine != 0 && (t_stamp_stage || mine > published.load(std::memory_order_acquire))) {
                return mine;
            }
        }
        return open.load(std::memory_order_acquire);
    }
    static void SetStamp(u64 number, bool stage) noexcept {
        t_stamp = number;
        t_stamp_stage = stage;
    }
    /// This thread's stamp state (a served sync runs on the back with GPUMem's).
    [[nodiscard]] static u64 CurrentStamp() noexcept {
        return t_stamp;
    }
    [[nodiscard]] static bool CurrentStampStage() noexcept {
        return t_stamp_stage;
    }
    void SetPerThread(bool on) noexcept {
        per_thread.store(on, std::memory_order_relaxed);
    }
    [[nodiscard]] bool PerThread() const noexcept {
        return per_thread.load(std::memory_order_relaxed);
    }

    u64 BeginPacket() noexcept {
        return open.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    [[nodiscard]] bool IsFree(u64 stamp) noexcept {
        if (verify) [[unlikely]] {
            return IsFreeChecked(stamp);
        }
        return IsFreeFast(stamp);
    }

    [[nodiscard]] u64 KnownFree() noexcept {
        const u64 known = master->KnownGpuTick();
        if (verify) [[unlikely]] {
            return KnownFreeChecked(known);
        }
        if (known != advanced_gpu.load(std::memory_order_acquire)) {
            return Advance(known);
        }
        return free_number.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsKnownFree(u64 stamp, u64 known_free) noexcept {
        const bool free = stamp <= known_free;
        if (verify) [[unlikely]] {
            CheckKnown(stamp, free);
        }
        return free;
    }

    [[nodiscard]] u64 Epoch() noexcept {
        const u64 value = epoch.load(std::memory_order_acquire);
        if (verify) [[unlikely]] {
            CheckEpoch(value);
        }
        return value;
    }

    [[nodiscard]] u64 TickOf(u64 stamp);
    [[nodiscard]] u64 WaitTickOf(u64 stamp);
    void CheckHostRead(u64 stamp, u64 tick);
    void Close(u64 tick);

    void SetUnplayed(u64 number) noexcept {
        unplayed_from.store(number, std::memory_order_release);
    }
    void ClearUnplayed() noexcept {
        unplayed_from.store(0, std::memory_order_release);
    }

    /// FRITCHII_MEMSTAGE_THREAD = 2 (vk_mem_stage.cpp, "Packet ticks with the stage ahead"): the
    /// back's mark is SetUnplayed's, GPUMem keeps its own here (its fill, its pending entries), and
    /// Close takes the lowest of the two and ring B's oldest packet, each written by its own side.
    void SetStageUnplayed(u64 number) noexcept {
        stage_unplayed.store(number, std::memory_order_release);
    }
    [[nodiscard]] u64 StageUnplayed() const noexcept {
        return stage_unplayed.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 BackUnplayed() const noexcept {
        return unplayed_from.load(std::memory_order_acquire);
    }
    /// GPUMem: the next packet number, its own mark set to it under the close's lock, so a close
    /// either ends before the number exists or sees the mark.
    u64 BeginStagePacket();
    /// Ring B's oldest packet number, 0 when empty (read by Close on the recording thread).
    using RingOldest = u64 (*)(void* context);
    void SetRingOldest(RingOldest reader, void* context) noexcept {
        std::scoped_lock lock{mutex};
        ring_oldest = reader;
        ring_context = context;
    }

    void NoteHeld(u64 newly_held) noexcept;
    void LogStats();

private:
    struct Published {
        u64 last;
        u64 tick;
    };
    struct Range {
        u64 from;
        u64 to;
    };
    struct StampSlot {
        std::atomic<u64> number{0};
        std::atomic<u64> first{0};
        std::atomic<u64> last{0};
    };
    enum class Check : u8 { IsFree, Known, Wait, HostRead, Count };

    [[nodiscard]] bool IsFreeFast(u64 stamp) noexcept {
        if (stamp <= free_number.load(std::memory_order_acquire)) {
            return true;
        }
        if (stamp > published.load(std::memory_order_acquire)) {
            return false;
        }
        const u64 known = master->KnownGpuTick();
        if (known == advanced_gpu.load(std::memory_order_acquire)) {
            return stamp <= free_number.load(std::memory_order_acquire);
        }
        return stamp <= Advance(known);
    }

    u64 Advance(u64 known);
    u64 AdvanceLocked(u64 known);
    void NoteStamp(u64 number) noexcept;
    bool IsFreeChecked(u64 stamp);
    u64 KnownFreeChecked(u64 known);
    void CheckKnown(u64 stamp, bool free);
    void CheckEpoch(u64 value);
    bool OldTick(u64 stamp, u64& tick);
    void Disagree(Check kind, bool early, u64 stamp, u64 old_tick, u64 detail);

    MasterSemaphore* master;
    const bool verify;

    alignas(64) std::atomic<u64> open{1};
    alignas(64) std::atomic<u64> unplayed_from{0};
    alignas(64) std::atomic<u64> stage_unplayed{0};
    std::atomic<bool> per_thread{false};
    static inline thread_local u64 t_stamp = 0;
    static inline thread_local bool t_stamp_stage = false;
    /// The `open` each recent close saw, by tick (the split check's "raced the tick advance";
    /// with the verify, under the lock).
    struct CloseOpen {
        u64 tick;
        u64 open;
    };
    std::deque<CloseOpen> close_opens;
    alignas(64) std::atomic<u64> free_number{0};
    std::atomic<u64> advanced_gpu{0};
    std::atomic<u64> published{0};
    std::atomic<u64> epoch{0};

    std::mutex mutex;
    std::deque<Published> ring;
    RingOldest ring_oldest{};
    void* ring_context{};

    struct Counters {
        std::atomic<u64> closes{};
        std::atomic<u64> carried{};
        std::atomic<u64> max_depth{};
        std::atomic<u64> held{};
        std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> checks{};
        std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> early{};
        std::array<std::atomic<u64>, static_cast<size_t>(Check::Count)> late{};
        std::atomic<u64> carried_skipped{};
        std::atomic<u64> unchecked{};
        std::atomic<u64> split{};
        std::atomic<u64> epoch_off{};
        std::atomic<u64> epoch_racing{}; ///< GPUMem's reads one apart from a close (step 7)
        std::atomic<u64> stamp_published{}; ///< a stamp on a number already published (must be 0)
        std::atomic<u64> split_raced{};     ///< a split that raced the tick advance (reported)
    };
    Counters counters;
    struct Snap {
        u64 numbers{};
        u64 closes{};
        u64 carried{};
        u64 held{};
        std::array<u64, static_cast<size_t>(Check::Count)> checks{};
        std::array<u64, static_cast<size_t>(Check::Count)> early{};
        std::array<u64, static_cast<size_t>(Check::Count)> late{};
        u64 carried_skipped{};
        u64 unchecked{};
        u64 split{};
        u64 epoch_off{};
        u64 epoch_racing{};
        u64 stamp_published{};
        u64 split_raced{};
    };
    void Take(Snap& out) const noexcept;
    Snap reported{};
    u64 last_log_ns{};

    std::unique_ptr<StampSlot[]> slots;
    std::deque<Range> carried_ranges;
    std::atomic<u64> carried_to{0};
};

// The buffer cache side's only way to the scheduler's clock and syncs (vk_mem_stage.cpp).
class SeamClock {
public:
    explicit SeamClock(Scheduler& scheduler_, MemStage* stage_ = nullptr);

    [[nodiscard]] u64 Now() const noexcept {
        return ticks != nullptr ? ticks->Now() : master->CurrentTick();
    }

    [[nodiscard]] bool IsFree(u64 stamp) const noexcept {
        return ticks != nullptr ? ticks->IsFree(stamp) : master->IsFree(stamp);
    }

    [[nodiscard]] u64 KnownGpu() const noexcept {
        return ticks != nullptr ? ticks->KnownFree() : master->KnownGpuTick();
    }

    [[nodiscard]] bool IsKnownFree(u64 stamp, u64 known) const noexcept {
        return ticks != nullptr ? ticks->IsKnownFree(stamp, known) : stamp <= known;
    }

    [[nodiscard]] u64 Epoch() const noexcept {
        return ticks != nullptr ? ticks->Epoch() : master->CurrentTick();
    }

    [[nodiscard]] u64 TickOf(u64 stamp) const {
        return ticks != nullptr ? ticks->TickOf(stamp) : stamp;
    }

    [[nodiscard]] SeamTicks* Ticks() const noexcept {
        return ticks;
    }

    [[nodiscard]] Scheduler& GetScheduler() const noexcept {
        return *scheduler;
    }

    void Finish(std::source_location location = std::source_location::current());

    void Wait(u64 stamp);

    void WaitTick(u64 tick);

    u64 SubmitForHostRead(std::source_location location = std::source_location::current());

private:
    Scheduler* scheduler;
    MasterSemaphore* master;
    MemStage* stage;
    SeamTicks* ticks;
};

} // namespace Vulkan
