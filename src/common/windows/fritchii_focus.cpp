// SPDX-License-Identifier: GPL-3.0-or-later

// Fritchii: the switches behind "the game loses fps when its window is not focused"
// (docs/fritchii/switches.md: FRITCHII_GPU_PRIORITY, FRITCHII_THREAD_QOS, FRITCHII_FOCUS_GRAB).
// Everything here is per process or per thread: no system setting is read or changed.

#include <atomic>
#include <mutex>
#include <vector>
#include <windows.h>

#include "common/windows/fritchii_focus.h"

#ifndef PROCESS_POWER_THROTTLING_EXECUTION_SPEED
#define PROCESS_POWER_THROTTLING_EXECUTION_SPEED 0x1
#endif
#ifndef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
#define PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION 0x4
#endif

namespace Common::Windows::FritchiiFocus {

namespace {

struct HotThread {
    HANDLE handle;
    DWORD id;
};

std::mutex g_hot_mutex;
std::vector<HotThread> g_hot_threads;
std::atomic<bool> g_qos_on{false};
HWND g_previous_foreground = nullptr;
std::mutex g_foreground_mutex;

using SetThreadInformationFn = BOOL(WINAPI*)(HANDLE, THREAD_INFORMATION_CLASS, LPVOID, DWORD);
using SetProcessInformationFn = BOOL(WINAPI*)(HANDLE, PROCESS_INFORMATION_CLASS, LPVOID, DWORD);
// D3DKMT_SCHEDULINGPRIORITYCLASS is an int-sized enum (d3dkmthk.h, WDK).
using SetGpuPriorityFn = LONG(APIENTRY*)(HANDLE, int);
using GetGpuPriorityFn = LONG(APIENTRY*)(HANDLE, int*);

SetThreadInformationFn ThreadInfoFn() {
    static const auto fn = reinterpret_cast<SetThreadInformationFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                                               "SetThreadInformation")));
    return fn;
}

SetProcessInformationFn ProcessInfoFn() {
    static const auto fn = reinterpret_cast<SetProcessInformationFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                                               "SetProcessInformation")));
    return fn;
}

HMODULE Gdi32() {
    static const HMODULE module = LoadLibraryW(L"gdi32.dll");
    return module;
}

bool TagThread(HANDLE thread, bool on) {
    const auto fn = ThreadInfoFn();
    if (fn == nullptr) {
        return false;
    }
    // ControlMask set + StateMask clear = High QoS explicitly; ControlMask clear = the system
    // decides again (focus, visibility, audibility).
    THREAD_POWER_THROTTLING_STATE state{
        .Version = THREAD_POWER_THROTTLING_CURRENT_VERSION,
        .ControlMask = on ? static_cast<ULONG>(THREAD_POWER_THROTTLING_EXECUTION_SPEED) : 0UL,
        .StateMask = 0,
    };
    return fn(thread, ThreadPowerThrottling, &state, sizeof(state)) != FALSE;
}

void AssertProcessHighQoS() {
    const auto fn = ProcessInfoFn();
    if (fn == nullptr) {
        return;
    }
    PROCESS_POWER_THROTTLING_STATE state{
        .Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        .ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED |
                       PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        .StateMask = 0,
    };
    fn(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
}

bool OwnWindow(HWND window) {
    if (window == nullptr) {
        return false;
    }
    DWORD pid = 0;
    GetWindowThreadProcessId(window, &pid);
    return pid == GetCurrentProcessId();
}

struct FindState {
    HWND best = nullptr;
    LONGLONG best_area = -1;
};

BOOL CALLBACK FindLargestOwnWindow(HWND window, LPARAM param) {
    auto* state = reinterpret_cast<FindState*>(param);
    if (!OwnWindow(window) || !IsWindowVisible(window) ||
        GetWindow(window, GW_OWNER) != nullptr) {
        return TRUE;
    }
    RECT rect{};
    if (!GetWindowRect(window, &rect)) {
        return TRUE;
    }
    const LONGLONG area = static_cast<LONGLONG>(rect.right - rect.left) *
                          static_cast<LONGLONG>(rect.bottom - rect.top);
    if (area > state->best_area) {
        state->best_area = area;
        state->best = window;
    }
    return TRUE;
}

/// SetForegroundWindow from a process that isn't the foreground one only works with the input of
/// the current foreground thread attached (the usual way tools hand focus over).
bool ForceForeground(HWND target) {
    const HWND current = GetForegroundWindow();
    if (current == target) {
        return true;
    }
    MSG msg;
    PeekMessageW(&msg, nullptr, 0, 0, PM_NOREMOVE); // AttachThreadInput needs a message queue
    const DWORD me = GetCurrentThreadId();
    const DWORD other = current != nullptr ? GetWindowThreadProcessId(current, nullptr) : 0;
    const bool attached = other != 0 && other != me && AttachThreadInput(me, other, TRUE);
    if (IsIconic(target)) {
        ShowWindow(target, SW_RESTORE);
    }
    BringWindowToTop(target);
    const bool ok = SetForegroundWindow(target) != FALSE;
    if (attached) {
        AttachThreadInput(me, other, FALSE);
    }
    return ok && GetForegroundWindow() == target;
}

} // Anonymous namespace

void RegisterHotThread() {
    HANDLE handle = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                         THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0)) {
        return;
    }
    const DWORD id = GetCurrentThreadId();
    {
        std::scoped_lock lock{g_hot_mutex};
        for (const HotThread& hot : g_hot_threads) {
            if (hot.id == id) {
                CloseHandle(handle);
                handle = nullptr;
                break;
            }
        }
        if (handle != nullptr) {
            // Finished threads are dropped here, so the list stays the live hot threads.
            std::erase_if(g_hot_threads, [](const HotThread& hot) {
                DWORD code = 0;
                if (GetExitCodeThread(hot.handle, &code) && code != STILL_ACTIVE) {
                    CloseHandle(hot.handle);
                    return true;
                }
                return false;
            });
            g_hot_threads.push_back({handle, id});
        }
    }
    if (g_qos_on.load(std::memory_order_relaxed)) {
        TagThread(GetCurrentThread(), true);
    }
}

int SetHotThreadsHighQoS(bool on, int* registered) {
    g_qos_on.store(on, std::memory_order_relaxed);
    if (on) {
        AssertProcessHighQoS();
    }
    int tagged = 0;
    std::scoped_lock lock{g_hot_mutex};
    for (const HotThread& hot : g_hot_threads) {
        tagged += TagThread(hot.handle, on) ? 1 : 0;
    }
    if (registered != nullptr) {
        *registered = static_cast<int>(g_hot_threads.size());
    }
    return tagged;
}

long SetGpuPriorityClass(int priority_class) {
    const HMODULE gdi = Gdi32();
    const auto fn = gdi == nullptr ? nullptr
                                   : reinterpret_cast<SetGpuPriorityFn>(reinterpret_cast<void*>(
                                         GetProcAddress(gdi, "D3DKMTSetProcessSchedulingPriorityClass")));
    if (fn == nullptr) {
        return -1;
    }
    return fn(GetCurrentProcess(), priority_class);
}

int GetGpuPriorityClass() {
    const HMODULE gdi = Gdi32();
    const auto fn = gdi == nullptr ? nullptr
                                   : reinterpret_cast<GetGpuPriorityFn>(reinterpret_cast<void*>(
                                         GetProcAddress(gdi, "D3DKMTGetProcessSchedulingPriorityClass")));
    int value = -1;
    if (fn == nullptr || fn(GetCurrentProcess(), &value) != 0) {
        return -1;
    }
    return value;
}

bool IsForeground() {
    return OwnWindow(GetForegroundWindow());
}

bool GrabForeground() {
    FindState state;
    EnumWindows(FindLargestOwnWindow, reinterpret_cast<LPARAM>(&state));
    if (state.best == nullptr) {
        return false;
    }
    std::scoped_lock lock{g_foreground_mutex};
    const HWND current = GetForegroundWindow();
    if (!OwnWindow(current)) {
        g_previous_foreground = current;
    }
    return ForceForeground(state.best);
}

bool ReleaseForeground() {
    std::scoped_lock lock{g_foreground_mutex};
    const HWND previous = g_previous_foreground;
    g_previous_foreground = nullptr;
    if (!IsForeground() || previous == nullptr || !IsWindow(previous)) {
        return false;
    }
    // This process has the foreground, so it may hand it on directly.
    return SetForegroundWindow(previous) != FALSE;
}

} // namespace Common::Windows::FritchiiFocus
