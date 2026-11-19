// SPDX-License-Identifier: GPL-2.0-or-later

// The back read audit (FRITCHII_BACK_READ_AUDIT=1).
//
// What it hooks: the read functions of Tegra::MemoryManager (Read<T>, GetPointer,
// ReadHandleWord, GetPointerForDescriptor, ReadBlock, ReadBlockUnsafe, ReadBlockGuarded,
// ReadWordsUnsafe, GetSpan, CopyBlock) and of the device memory behind it,
// MaxwellDeviceMemoryManager (Read<T>, GetPointer<T>, GetSpan, ReadBlock, ReadBlockUnsafe).
// Tegra::Memory::GpuGuestMemory and DeviceGuestMemory read through GetSpan and
// ReadBlock(Unsafe), so they are covered too, as is a GpuToCpuAddress followed by a device read.
// The hook sits in the read functions, so a call site is the read function's return address.
//
// Off: each read function loads `enabled` and takes one predictable branch. On: a thread-local
// check (is this the back, is an audited read already counting this one), then a lock-free table
// of 512 slots, open addressing on the return address, with relaxed atomic counters. Only the
// back counts, so the slots have one writer in practice; the atomics keep it right regardless.
//
// MemoryManager reads go through the device memory: those read functions count once at their
// caller and hide what they read inside (OuterScope), so nothing is counted twice. Writes made
// through the same pointer getters inside the two managers (WriteBlock, Write<T>, the constant
// buffer writes, the page table's continuity check) are hidden (HideScope). Pointer getters
// called from outside can still be writes: the Mut kinds say so, the site decides.
//
// A read function inlined into its caller (ThinLTO can do that across files) reports the caller's
// caller, and a GuestMemory helper the compiler did not inline reports itself (the script marks
// those rows "helper").
//
// Every 10 s the back logs the top 40 sites by reads as offsets from the executable's base
// (logged once), with reads/s and bytes/s over the window, and starts a new window.
//
// The memory stage thread (GPUMem, FRITCHII_MEMSTAGE_THREAD=1) counts in a second table, logged
// the same way as "Stage read audit" when it read anything: the draw and dispatch fills it runs
// for the back read game memory there now, so the back's list loses them and this is the stage's
// read list (memory stage step 6 spec 5.7).
// work/back_read_audit.py symbolizes those lines with llvm-symbolizer.

#include "video_core/back_read_audit.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_stage_link.h"

namespace VideoCommon::BackReadAudit {

namespace {

bool ReadSwitch() {
    const char* const value = std::getenv("FRITCHII_BACK_READ_AUDIT");
    return value != nullptr && value[0] == '1';
}

constexpr std::size_t TableSize = 512;
constexpr std::size_t TopSites = 40;
constexpr auto ReportInterval = std::chrono::seconds{10};
/// Counts between two clock reads on the counting thread.
constexpr u32 ClockEvery = 1024;

struct Slot {
    std::atomic<uintptr_t> address{};
    std::atomic<u64> reads{};
    std::atomic<u64> bytes{};
    std::atomic<u8> kind{};
};

/// [0] the back's reads, [1] the memory stage thread's (GPUMem, FRITCHII_MEMSTAGE_THREAD=1): the
/// fills it runs for the back read there now (memory stage spec 5.7: its read list).
std::array<std::array<Slot, TableSize>, 2> tables{};
std::array<std::atomic<u64>, 2> dropped{};
constexpr std::array<const char*, 2> TableNames{"Back read audit", "Stage read audit"};
/// steady_clock nanoseconds of the next report; 0 until the first count.
std::atomic<s64> next_report{};
std::atomic<s64> window_start{};

/// Audited reads open on this thread (OuterScope and HideScope).
thread_local u32 t_depth{};
thread_local u32 t_until_clock{};

constexpr std::array<const char*, static_cast<std::size_t>(Kind::Count)> KindNames{
    "MmRead",          "MmGetPointer",       "MmGetPointerMut", "MmReadHandleWord",
    "MmDescriptorPtr", "MmReadBlock",        "MmReadBlockUnsafe", "MmReadBlockGuarded",
    "MmReadWords",     "MmGetSpan",          "MmGetSpanMut",    "MmCopyBlock",
    "DevRead",         "DevGetPointer",      "DevGetPointerMut", "DevGetSpan",
    "DevGetSpanMut",   "DevReadBlock",       "DevReadBlockUnsafe",
};

const char* KindName(u8 kind) {
    return kind < KindNames.size() ? KindNames[kind] : "?";
}

uintptr_t ImageBase() {
#ifdef _WIN32
    return reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ReadSwitch), &info) != 0) {
        return reinterpret_cast<uintptr_t>(info.dli_fbase);
    }
    return 0;
#endif
}

s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void ReportTable(std::size_t which, double seconds) {
    struct Site {
        uintptr_t address;
        u64 reads;
        u64 bytes;
        u8 kind;
    };
    std::vector<Site> sites;
    sites.reserve(TableSize);
    u64 total_reads = 0;
    u64 total_bytes = 0;
    for (Slot& slot : tables[which]) {
        const uintptr_t address = slot.address.load(std::memory_order_relaxed);
        if (address == 0) {
            continue;
        }
        const u64 reads = slot.reads.exchange(0, std::memory_order_relaxed);
        const u64 bytes = slot.bytes.exchange(0, std::memory_order_relaxed);
        if (reads == 0) {
            continue;
        }
        total_reads += reads;
        total_bytes += bytes;
        sites.push_back({address, reads, bytes, slot.kind.load(std::memory_order_relaxed)});
    }
    const u64 lost = dropped[which].exchange(0, std::memory_order_relaxed);
    if (which != 0 && total_reads == 0 && lost == 0) {
        // GPUMem ran nothing this window (FRITCHII_MEMSTAGE_THREAD=0).
        return;
    }
    std::sort(sites.begin(), sites.end(),
              [](const Site& a, const Site& b) { return a.reads > b.reads; });

    static const uintptr_t base = ImageBase();
    const char* const name = TableNames[which];
    LOG_INFO(HW_GPU,
             "{}: {:.1f} s, {} sites, {:.0f} reads/s, {:.0f} bytes/s, {} reads dropped (table "
             "full)",
             name, seconds, sites.size(), static_cast<double>(total_reads) / seconds,
             static_cast<double>(total_bytes) / seconds, lost);
    const std::size_t shown = (std::min)(sites.size(), TopSites);
    for (std::size_t i = 0; i < shown; ++i) {
        const Site& site = sites[i];
        LOG_INFO(HW_GPU, "{}: #{:02} +{:#x} {} {:.1f} reads/s {:.0f} bytes/s", name, i + 1,
                 site.address - base, KindName(site.kind),
                 static_cast<double>(site.reads) / seconds,
                 static_cast<double>(site.bytes) / seconds);
    }
}

void Report(s64 now) {
    const s64 start = window_start.exchange(now, std::memory_order_relaxed);
    const double seconds = (std::max)(1e-3, static_cast<double>(now - start) * 1e-9);
    ReportTable(0, seconds);
    ReportTable(1, seconds);
}

void MaybeReport() {
    if (++t_until_clock < ClockEvery) [[likely]] {
        return;
    }
    t_until_clock = 0;
    const s64 now = NowNs();
    s64 due = next_report.load(std::memory_order_relaxed);
    const s64 interval =
        std::chrono::duration_cast<std::chrono::nanoseconds>(ReportInterval).count();
    if (due == 0) {
        if (next_report.compare_exchange_strong(due, now + interval, std::memory_order_relaxed)) {
            window_start.store(now, std::memory_order_relaxed);
            LOG_INFO(HW_GPU,
                     "Back read audit: on, image base {:#x} (sites are offsets from it; "
                     "work/back_read_audit.py symbolizes them)",
                     ImageBase());
        }
        return;
    }
    if (now < due ||
        !next_report.compare_exchange_strong(due, now + interval, std::memory_order_relaxed)) {
        return;
    }
    Report(now);
}

/// The table this thread counts in: 0 the back, 1 GPUMem, -1 none.
int TableOfThread() noexcept {
    if (GpuSplit::OnBackThread()) {
        return 0;
    }
    return MemoryStageLink::OnStageThread() ? 1 : -1;
}

void Count(std::size_t which, const void* caller, std::size_t bytes, Kind kind) {
    const auto address = reinterpret_cast<uintptr_t>(caller);
    std::size_t index =
        static_cast<std::size_t>((static_cast<u64>(address) * 0x9E3779B97F4A7C15ULL) >> 55) &
        (TableSize - 1);
    auto& table = tables[which];
    for (std::size_t probe = 0; probe < TableSize; ++probe) {
        Slot& slot = table[index];
        uintptr_t current = slot.address.load(std::memory_order_relaxed);
        if (current == 0 &&
            slot.address.compare_exchange_strong(current, address, std::memory_order_relaxed)) {
            slot.kind.store(static_cast<u8>(kind), std::memory_order_relaxed);
            current = address;
        }
        if (current == address) {
            slot.reads.fetch_add(1, std::memory_order_relaxed);
            slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
            MaybeReport();
            return;
        }
        index = (index + 1) & (TableSize - 1);
    }
    dropped[which].fetch_add(1, std::memory_order_relaxed);
    MaybeReport();
}

} // Anonymous namespace

bool enabled = false;

namespace {
/// FRITCHII_BACK_READ_AUDIT itself: the tables count (`enabled` may be on for g_touch_hook alone).
bool g_counting = false;
} // Anonymous namespace

void Init() {
    // Not at static initialization: main() applies nx2_switches.txt to the environment later.
    [[maybe_unused]] static const bool once = [] {
        g_counting = ReadSwitch();
        enabled = enabled || g_counting;
        if (g_counting) {
            LOG_INFO(HW_GPU, "Back read audit: FRITCHII_BACK_READ_AUDIT=1, counting the GPU "
                             "back thread's reads of game memory by call site");
        }
        return true;
    }();
}

void Note(const void* caller, std::size_t bytes, Kind kind) noexcept {
    if (t_depth != 0) {
        return;
    }
    if (g_touch_hook != nullptr) [[unlikely]] {
        g_touch_hook(caller);
    }
    if (!g_counting) {
        return;
    }
    if (const int which = TableOfThread(); which >= 0) {
        Count(static_cast<std::size_t>(which), caller, bytes, kind);
    }
}

OuterScope::OuterScope(const void* caller, std::size_t bytes, Kind kind) noexcept {
    if (t_depth++ == 0) {
        if (g_touch_hook != nullptr) [[unlikely]] {
            g_touch_hook(caller);
        }
        if (!g_counting) {
            return;
        }
        if (const int which = TableOfThread(); which >= 0) {
            Count(static_cast<std::size_t>(which), caller, bytes, kind);
        }
    }
}

OuterScope::~OuterScope() {
    --t_depth;
}

HideScope::HideScope() noexcept {
    ++t_depth;
}

HideScope::~HideScope() {
    --t_depth;
}

} // namespace VideoCommon::BackReadAudit
