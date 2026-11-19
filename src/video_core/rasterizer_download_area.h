// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <thread>

#include "common/common_types.h"

namespace VideoCore {

struct RasterizerDownloadArea {
    VAddr start_address;
    VAddr end_address;
    bool preemtive;
    /// EDEN_NX2_EXP49 (async read-back): what guest memory lacks there is on its way for an
    /// earlier read. Nothing to flush, only to wait for (RasterizerInterface::FinishRead).
    bool in_flight{};
    /// EDEN_NX2_EXP72: every GPU-written byte of the area is in this fence download (sequence
    /// number, see BufferCache::WaitFenceDownload), already submitted; 0 = none.
    u64 fence_download{};
};

/// EDEN_NX2_EXP49: downloads for guest CPU reads that are submitted but not yet written to guest
/// memory (RasterizerInterface::FlushRegionForRead). While there are any, a write to guest memory
/// from the GPU's side first lands those it overlaps (FinishRead), so none lands over it later.
inline std::atomic<u32> g_read_downloads{};

/// Tegra::Nx2MayBeGpuWritten on Switch 2 titles, true otherwise. Defined in gpu.cpp: nx2.h can't
/// be included here, since the Qt frontend reaches this header and its `slots` macro breaks it.
[[nodiscard]] bool MayHoldGpuWrites(DAddr addr, u64 size);

/// Whether such a download may still land over [addr, addr + size): what a write to guest memory
/// from the GPU's side asks before FinishRead, on any thread, without the buffer cache's lock.
/// The downloads take only GPU-modified bytes, whose pages Switch 2 titles mark
/// (Tegra::Nx2MayBeGpuWritten): writes to other pages skip the lock.
[[nodiscard]] inline bool ReadDownloadsMayCover(DAddr addr, u64 size) {
    return g_read_downloads.load(std::memory_order_acquire) != 0 && MayHoldGpuWrites(addr, size);
}

/// What DownloadLanding publishes.
struct DownloadLandingState {
    /// Odd while a landing is published.
    std::atomic<u64> sequence{};
    std::atomic<DAddr> begin{};
    std::atomic<DAddr> end{};
    std::atomic<u64> waits{};
    /// FRITCHII_FENCE_COPY_UNLOCKED: 1 while the published landing is the fencing thread's,
    /// writing without the buffer cache's lock (DownloadLanding::PublishUnlocked).
    std::atomic<u32> unlocked{};
    /// Lock holders that waited for such a landing (a landing of their own, ClearFenceDownloads),
    /// and how long.
    std::atomic<u64> unlocked_waits{};
    std::atomic<u64> unlocked_wait_ns{};
};

/// The buffer cache writing downloaded GPU bytes to guest memory [begin, end). A download keeps the
/// bytes of the guest CPU writes collected and not yet applied when it looks at them
/// (BufferCache::PendingCpuWriteRanges). A guest core collects a write
/// (Core::GPUDirtyMemoryManager::Collect) and stores its bytes right after, without the buffer
/// cache's lock: a write collected after the download looked had its bytes written over with the
/// GPU's older ones. Downloads now look only inside a landing, and a write collected in the
/// landing's range stores its bytes once the landing has ended (Await, from Core::Memory's
/// HandleRasterizerWrite).
///
/// Each side fences between its store (the landing, the collected write) and its load (the
/// collected writes, the landing): either the download finds the write, or the write finds the
/// landing. Downloads land under the buffer cache's lock, so one landing is published at a time.
/// (Ported.)
class DownloadLanding {
public:
    explicit DownloadLanding(DAddr begin, DAddr end) noexcept {
        // FRITCHII_FENCE_COPY_UNLOCKED: the fencing thread's landing may still be up although
        // this thread holds the lock (it ends without it): one landing at a time.
        if (state.unlocked.load(std::memory_order_acquire) != 0) [[unlikely]] {
            AwaitUnlockedEnd();
        }
        state.begin.store(begin, std::memory_order_relaxed);
        state.end.store(end, std::memory_order_relaxed);
        state.sequence.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    ~DownloadLanding() {
        state.sequence.fetch_add(1, std::memory_order_release);
    }
    DownloadLanding(const DownloadLanding&) = delete;
    DownloadLanding& operator=(const DownloadLanding&) = delete;

    /// A guest CPU write of [addr, addr + size) was collected: returns once its bytes may be
    /// stored.
    static void Await(DAddr addr, u64 size) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u64 sequence = state.sequence.load(std::memory_order_acquire);
        if ((sequence & 1) != 0) [[unlikely]] {
            Wait(sequence, addr, size);
        }
    }

    /// Await, also saying whether no landing was published when it looked. True means every
    /// landing published later finds this write's collected mark while it stays collected (the
    /// fences' either-or above): FRITCHII_STORE_MEMO keeps a memo only after such a write.
    [[nodiscard]] static bool AwaitIdle(DAddr addr, u64 size) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u64 sequence = state.sequence.load(std::memory_order_acquire);
        if ((sequence & 1) != 0) [[unlikely]] {
            Wait(sequence, addr, size);
            return false;
        }
        return true;
    }

    /// FRITCHII_FENCE_COPY_UNLOCKED (BufferCache::PopAsyncBuffersUnlocked): a landing published
    /// under the buffer cache's lock, as the constructor's, that its thread ends after letting
    /// the lock go and writing the bytes. While it is up, every other landing waits for it to
    /// end (they are published under the lock: one at a time still), and so does a lock holder
    /// that takes ranges away from the fence downloads (BufferCache::ClearFenceDownloads) before
    /// it writes them, and a device unmap (AwaitUnlocked).
    static void PublishUnlocked(DAddr begin, DAddr end) noexcept {
        state.begin.store(begin, std::memory_order_relaxed);
        state.end.store(end, std::memory_order_relaxed);
        state.unlocked.store(1, std::memory_order_relaxed);
        state.sequence.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    /// The end of PublishUnlocked's landing, after its writes (sequence first: a landing that
    /// waited for `unlocked` publishes on an even sequence).
    static void EndUnlocked() noexcept {
        state.sequence.fetch_add(1, std::memory_order_release);
        state.unlocked.store(0, std::memory_order_release);
    }

    /// Returns once no landing written without the lock that was up when it looked overlaps
    /// [addr, addr + size) any more. A lock holder or a thread without cache locks (it needs
    /// none: the landing's thread takes no lock until it ended). One acquire load while none is
    /// up.
    static void AwaitUnlocked(DAddr addr, u64 size) noexcept {
        if (state.unlocked.load(std::memory_order_acquire) == 0) [[likely]] {
            return;
        }
        const u64 sequence = state.sequence.load(std::memory_order_acquire);
        const DAddr begin = state.begin.load(std::memory_order_relaxed);
        const DAddr end = state.end.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if ((sequence & 1) == 0 || state.unlocked.load(std::memory_order_acquire) == 0 ||
            state.sequence.load(std::memory_order_acquire) != sequence || addr >= end ||
            addr + size <= begin) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        while (state.sequence.load(std::memory_order_acquire) == sequence) {
            std::this_thread::yield();
        }
        NoteUnlockedWait(start);
    }

    /// AwaitUnlocked for a thread that holds no cache lock (a device unmap): the fence orders its
    /// earlier stores (the invalidation) before the look.
    static void AwaitUnlockedFenced(DAddr addr, u64 size) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        AwaitUnlocked(addr, size);
    }

    /// FRITCHII_FENCE_COPY_UNLOCKED's line: the waits for such landings since the last take.
    static void TakeUnlockedWaits(u64& waits, u64& wait_ns) noexcept {
        if (state.unlocked_waits.load(std::memory_order_relaxed) == 0) {
            return;
        }
        waits += state.unlocked_waits.exchange(0, std::memory_order_relaxed);
        wait_ns += state.unlocked_wait_ns.exchange(0, std::memory_order_relaxed);
    }

    /// Writes that waited for a landing so far.
    [[nodiscard]] static u64 Waits() noexcept {
        return state.waits.load(std::memory_order_relaxed);
    }

private:
    static void Wait(u64 sequence, DAddr addr, u64 size) noexcept {
        const DAddr begin = state.begin.load(std::memory_order_relaxed);
        const DAddr end = state.end.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        // Ended (its writes come first, and the landings after it found this write), or it
        // writes elsewhere.
        if (state.sequence.load(std::memory_order_acquire) != sequence || addr >= end ||
            addr + size <= begin) {
            return;
        }
        state.waits.fetch_add(1, std::memory_order_relaxed);
        while (state.sequence.load(std::memory_order_acquire) == sequence) {
            std::this_thread::yield();
        }
    }

    static void AwaitUnlockedEnd() noexcept {
        const auto start = std::chrono::steady_clock::now();
        while (state.unlocked.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
        NoteUnlockedWait(start);
    }

    static void NoteUnlockedWait(std::chrono::steady_clock::time_point start) noexcept {
        const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
        state.unlocked_waits.fetch_add(1, std::memory_order_relaxed);
        state.unlocked_wait_ns.fetch_add(static_cast<u64>(waited), std::memory_order_relaxed);
    }

    static inline DownloadLandingState state;
};

} // namespace VideoCore
