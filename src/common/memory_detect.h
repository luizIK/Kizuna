// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "common/common_types.h"

namespace Common {

struct MemoryInfo {
    u64 TotalPhysicalMemory{};
    u64 TotalSwapMemory{};
};

/**
 * Gets the memory info of the host system
 * @return Reference to a MemoryInfo struct with the physical and swap memory sizes in bytes
 */
[[nodiscard]] const MemoryInfo& GetMemInfo();

/// Page faults of this process so far, soft and hard (0 where unknown).
[[nodiscard]] u64 GetProcessPageFaults();

/// This process's memory and the system's commit, for the memory log line (zeros where unknown;
/// only Windows fills them in).
/// Commit the system can still hand out, in bytes: its commit limit (RAM plus the page files)
/// minus what is committed. No value where the host reports no such limit (anything but Windows).
/// From zortos293/Lichii#1210.
[[nodiscard]] std::optional<u64> GetSystemCommitRoom();

/// Kizuna (Windows): the programs holding the most commit, by executable name (all of a
/// browser's processes together), largest first, e.g. "chrome.exe 6.2 GiB (31 processes),
/// ...; in no program 3.1 GiB" for the `count` largest. Takes ~10 to 50 ms (it opens every
/// process): not for the GPU thread. Empty elsewhere.
[[nodiscard]] std::string TopCommitPrograms(size_t count);
/// The last list TopCommitPrograms made (empty before the first), for the crash report: a
/// static buffer, nothing allocated.
[[nodiscard]] const char* LastTopCommitPrograms();
/// Runs TopCommitPrograms(8) on a thread of its own and logs it with `reason`; at most one at a
/// time, and at most once every `min_interval_s` seconds.
void LogTopCommitProgramsAsync(const char* reason, int min_interval_s);

struct ProcessMemoryUsage {
    u64 working_set{};
    u64 peak_working_set{};
    u64 private_bytes{}; ///< Private commit (mapped sections like the guest memory are not in it)
    u64 peak_private_bytes{};
    /// Resident pages not shareable with other mappings: the working set minus the guest memory
    /// section and mapped files (0 where Windows does not report it: older builds).
    u64 private_working_set{};
    u64 system_commit{}; ///< Commit charge of the whole system
    u64 system_commit_limit{};
    u64 available_physical{};
};
[[nodiscard]] ProcessMemoryUsage GetProcessMemoryUsage();

/// This process's Win32 heaps (the C runtime's malloc heap and every other HeapCreate heap: Qt's,
/// drivers' that use one), summed with HeapSummary. Zeros elsewhere.
struct HeapUsage {
    u64 crt_committed{};   ///< The C runtime's heap (malloc, new)
    u64 crt_allocated{};
    u64 other_committed{}; ///< Every other heap of the process
    u64 other_allocated{};
    u32 heaps{};           ///< Heaps summed, the C runtime's included
    u32 micros{};          ///< What the pass took
};
/// Locks each heap for its summary: call it off the busy threads. Also kept for LastHeapUsage.
HeapUsage SampleHeapUsage();
/// The last SampleHeapUsage result (zeros before the first).
[[nodiscard]] HeapUsage LastHeapUsage();

} // namespace Common
