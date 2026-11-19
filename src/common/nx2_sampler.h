// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

namespace Nx2Sampler {

// EDEN_NX2_SAMPLE=<thread names>: a sampling profiler for named host threads (Windows only).
// Names are comma separated; a name ending in '*' matches by prefix (CPUCore_*). From
// EDEN_NX2_SAMPLE_START seconds (default 60) for EDEN_NX2_SAMPLE_SECONDS (default 30), each
// thread is paused about 1000 times a second and its call stack recorded; the counts per
// function (where the thread is, and everything on its stack) go to EDEN_NX2_SAMPLE_OUT
// (default nx2_samples.txt).
void StartFromEnvironment();

/// Samples the named threads for `seconds` now (blocking), writes the report to `out` (and
/// the folded stacks to `out`.folded) and returns the report.
std::string Sample(const char* names, double seconds, const std::string& out);

/// FRITCHII_LOCK_CENSUS (common/fritchii_lock_census.cpp): a handle to OS thread `os_id` that can
/// be suspended and read, or null (and always null off Windows). Kept by the caller.
void* OpenThreadForCapture(unsigned os_id);

/// Suspends `thread` (OpenThreadForCapture), asks `still(context)` whether the sample is wanted
/// (the caller's test, run while the thread is paused: it must not lock or allocate), walks the
/// paused thread's stack into `frames` (at most `max`) and resumes it. Returns the depth, 0 when
/// nothing was taken. Windows only.
unsigned CaptureSuspended(void* thread, unsigned long long* frames, unsigned max,
                          bool (*still)(const void*), const void* context);

/// The function name at `address` (DbgHelp, the one symbolizer the sampler uses too; the first
/// call loads the symbols, which takes a while). Windows only; elsewhere the address in hex.
std::string SymbolName(unsigned long long address);

} // namespace Nx2Sampler
