// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <sysinfoapi.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#include <sys/types.h>
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#else
#include <unistd.h>
#endif
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#include "common/logging.h"
#include "common/memory_detect.h"
#ifdef _WIN32
#include "common/string_util.h"
#endif

namespace Common {

// Detects the RAM and Swapfile sizes
static MemoryInfo Detect() {
    MemoryInfo mem_info{};

#ifdef _WIN32
    MEMORYSTATUSEX memorystatus;
    memorystatus.dwLength = sizeof(memorystatus);
    GlobalMemoryStatusEx(&memorystatus);
    mem_info.TotalPhysicalMemory = memorystatus.ullTotalPhys;
    mem_info.TotalSwapMemory = memorystatus.ullTotalPageFile - mem_info.TotalPhysicalMemory;
#elif defined(__APPLE__)
    u64 ramsize;
    struct xsw_usage vmusage;
    std::size_t sizeof_ramsize = sizeof(ramsize);
    std::size_t sizeof_vmusage = sizeof(vmusage);
    // hw and vm are defined in sysctl.h
    // https://github.com/apple/darwin-xnu/blob/master/bsd/sys/sysctl.h#L471
    // sysctlbyname(const char *, void *, size_t *, void *, size_t);
    sysctlbyname("hw.memsize", &ramsize, &sizeof_ramsize, nullptr, 0);
    sysctlbyname("vm.swapusage", &vmusage, &sizeof_vmusage, nullptr, 0);
    mem_info.TotalPhysicalMemory = ramsize;
    mem_info.TotalSwapMemory = vmusage.xsu_total;
#elif defined(__FreeBSD__)
    u_long physmem, swap_total;
    std::size_t sizeof_u_long = sizeof(u_long);
    // sysctlbyname(const char *, void *, size_t *, const void *, size_t);
    sysctlbyname("hw.physmem", &physmem, &sizeof_u_long, nullptr, 0);
    sysctlbyname("vm.swap_total", &swap_total, &sizeof_u_long, nullptr, 0);
    mem_info.TotalPhysicalMemory = physmem;
    mem_info.TotalSwapMemory = swap_total;
#elif defined(__linux__)
    struct sysinfo meminfo;
    sysinfo(&meminfo);
    mem_info.TotalPhysicalMemory = meminfo.totalram;
    mem_info.TotalSwapMemory = meminfo.totalswap;
#else
    mem_info.TotalPhysicalMemory = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    mem_info.TotalSwapMemory = 0;
#endif

    return mem_info;
}

const MemoryInfo& GetMemInfo() {
    static MemoryInfo mem_info = Detect();
    return mem_info;
}

u64 GetProcessPageFaults() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return 0;
    }
    return counters.PageFaultCount;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return static_cast<u64>(usage.ru_minflt) + static_cast<u64>(usage.ru_majflt);
#endif
}

ProcessMemoryUsage GetProcessMemoryUsage() {
    ProcessMemoryUsage usage{};
#ifdef _WIN32
    // PROCESS_MEMORY_COUNTERS_EX2's layout (the SDK declares it only for NTDDI_WIN10_CU and up).
    // Windows builds that know it fill PrivateWorkingSetSize; older ones either refuse the size
    // (then the EX one is asked for) or leave it 0.
    struct CountersEx2 {
        PROCESS_MEMORY_COUNTERS_EX ex;
        SIZE_T PrivateWorkingSetSize;
        ULONG64 SharedCommitUsage;
    };
    CountersEx2 counters{};
    counters.ex.cb = sizeof(counters);
    bool ok = K32GetProcessMemoryInfo(GetCurrentProcess(),
                                      reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                      sizeof(counters));
    if (!ok) {
        counters = {};
        counters.ex.cb = sizeof(counters.ex);
        ok = K32GetProcessMemoryInfo(GetCurrentProcess(),
                                     reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters.ex),
                                     sizeof(counters.ex));
    }
    if (ok) {
        usage.working_set = counters.ex.WorkingSetSize;
        usage.peak_working_set = counters.ex.PeakWorkingSetSize;
        usage.private_bytes = counters.ex.PrivateUsage;
        usage.peak_private_bytes = counters.ex.PeakPagefileUsage;
        usage.private_working_set = counters.PrivateWorkingSetSize;
    }
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        usage.system_commit = status.ullTotalPageFile - status.ullAvailPageFile;
        usage.system_commit_limit = status.ullTotalPageFile;
        usage.available_physical = status.ullAvailPhys;
    }
#endif
    return usage;
}

namespace {
std::mutex g_heap_usage_mutex;
HeapUsage g_heap_usage{};
} // Anonymous namespace

HeapUsage SampleHeapUsage() {
    HeapUsage usage{};
#ifdef _WIN32
    const auto start = std::chrono::steady_clock::now();
#ifdef _MSC_VER
    const HANDLE crt_heap = reinterpret_cast<HANDLE>(_get_heap_handle());
#else
    const HANDLE crt_heap = GetProcessHeap();
#endif
    std::vector<HANDLE> heaps(64);
    for (int attempt = 0; attempt < 4; ++attempt) {
        const DWORD count = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
        if (count <= heaps.size()) {
            heaps.resize(count);
            break;
        }
        heaps.resize(count + 16);
    }
    for (const HANDLE heap : heaps) {
        HEAP_SUMMARY summary{};
        summary.cb = sizeof(summary);
        if (!HeapSummary(heap, 0, &summary)) {
            continue;
        }
        ++usage.heaps;
        if (heap == crt_heap) {
            usage.crt_committed += summary.cbCommitted;
            usage.crt_allocated += summary.cbAllocated;
        } else {
            usage.other_committed += summary.cbCommitted;
            usage.other_allocated += summary.cbAllocated;
        }
    }
    usage.micros = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now() - start)
                                        .count());
#endif
    std::scoped_lock lock{g_heap_usage_mutex};
    g_heap_usage = usage;
    return usage;
}

HeapUsage LastHeapUsage() {
    std::scoped_lock lock{g_heap_usage_mutex};
    return g_heap_usage;
}

namespace {
std::mutex g_top_commit_mutex;
char g_top_commit_text[1024]{};
std::atomic<bool> g_top_commit_running{false};
std::atomic<s64> g_top_commit_last_s{-1000000};
} // Anonymous namespace

std::string TopCommitPrograms(size_t count) {
#ifdef _WIN32
    std::vector<DWORD> pids(8192);
    DWORD bytes = 0;
    if (!EnumProcesses(pids.data(), static_cast<DWORD>(pids.size() * sizeof(DWORD)), &bytes)) {
        return {};
    }
    pids.resize(bytes / sizeof(DWORD));
    struct Program {
        u64 commit{};
        u32 processes{};
    };
    std::map<std::wstring, Program> programs;
    u64 listed{};
    for (const DWORD pid : pids) {
        if (pid == 0) {
            continue;
        }
        HANDLE process =
            OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if (process == nullptr) {
            process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        }
        if (process == nullptr) {
            continue;
        }
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = sizeof(counters);
        wchar_t path[MAX_PATH]{};
        DWORD length = MAX_PATH;
        if (GetProcessMemoryInfo(process, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                    sizeof(counters)) &&
            QueryFullProcessImageNameW(process, 0, path, &length)) {
            const std::wstring_view full{path, length};
            const size_t slash = full.find_last_of(L"\\/");
            Program& program = programs[std::wstring{
                slash == std::wstring_view::npos ? full : full.substr(slash + 1)}];
            program.commit += counters.PrivateUsage;
            ++program.processes;
            listed += counters.PrivateUsage;
        }
        CloseHandle(process);
    }
    std::vector<std::pair<std::wstring, Program>> sorted(programs.begin(), programs.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.commit > b.second.commit; });
    std::string text;
    char line[160];
    for (size_t i = 0; i < (std::min)(count, sorted.size()); ++i) {
        const auto& [name, program] = sorted[i];
        std::snprintf(line, sizeof(line), "%s%s %.1f GiB (%u process%s)", i == 0 ? "" : ", ",
                      UTF16ToUTF8(name).c_str(),
                      static_cast<double>(program.commit) / (1024.0 * 1024.0 * 1024.0),
                      program.processes, program.processes == 1 ? "" : "es");
        text += line;
    }
    PERFORMANCE_INFORMATION performance{};
    performance.cb = sizeof(performance);
    if (GetPerformanceInfo(&performance, sizeof(performance))) {
        // Commit charged to no process: the guest memory section, the drivers' and the kernel's
        // own, virtual machines (WSL, Hyper-V).
        const u64 total = u64{performance.CommitTotal} * performance.PageSize;
        std::snprintf(line, sizeof(line), "; in no program %.1f GiB of %.1f GiB committed",
                      static_cast<double>(total > listed ? total - listed : 0) /
                          (1024.0 * 1024.0 * 1024.0),
                      static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
        text += line;
    }
    {
        std::scoped_lock lock{g_top_commit_mutex};
        std::snprintf(g_top_commit_text, sizeof(g_top_commit_text), "%s", text.c_str());
    }
    return text;
#else
    static_cast<void>(count);
    return {};
#endif
}

const char* LastTopCommitPrograms() {
    return g_top_commit_text;
}

void LogTopCommitProgramsAsync(const char* reason, int min_interval_s) {
#ifdef _WIN32
    const s64 now = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    if (now - g_top_commit_last_s.load(std::memory_order_relaxed) < min_interval_s ||
        g_top_commit_running.exchange(true)) {
        return;
    }
    g_top_commit_last_s.store(now, std::memory_order_relaxed);
    std::thread([reason] {
        const std::string text = TopCommitPrograms(8);
        if (!text.empty()) {
            LOG_INFO(Common_Memory, "Programs by commit ({}): {}", reason, text);
        }
        g_top_commit_running.store(false);
    }).detach();
#else
    static_cast<void>(reason);
    static_cast<void>(min_interval_s);
#endif
}

std::optional<u64> GetSystemCommitRoom() {
#ifdef _WIN32
    PERFORMANCE_INFORMATION performance{};
    performance.cb = sizeof(performance);
    if (GetPerformanceInfo(&performance, sizeof(performance))) {
        const u64 limit = performance.CommitLimit;
        const u64 committed = performance.CommitTotal;
        return (committed < limit ? limit - committed : 0) * performance.PageSize;
    }
#endif
    return std::nullopt;
}

} // namespace Common
