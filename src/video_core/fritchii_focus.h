// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef _WIN32

#include <chrono>
#include <thread>

#include "common/logging.h"
#include "common/windows/fritchii_focus.h"
#include "video_core/nx2_debug_control.h"

/// The fps the game loses while its window is not focused (helper BN, 2026-11-16;
/// docs/fritchii/switches.md). Every GPU thread, CPU core and timer thread already runs at
/// THREAD_PRIORITY_TIME_CRITICAL (15, the top of the normal range), so Windows' foreground CPU
/// boosts can't raise them further, and EXP111 (process priority class) measured no gain. What is
/// left for a focused window: the GPU scheduler's treatment of the foreground process, the
/// processor power management's QoS level (High when focused, Medium when only visible, Low when
/// covered), and how the present is shown (independent flip vs DWM composition).
/// - FRITCHII_GPU_PRIORITY (live): 0 = Windows' default, 1 = above normal, 2 = high GPU
///   scheduling priority class for the process (D3DKMTSetProcessSchedulingPriorityClass).
/// - FRITCHII_THREAD_QOS (live): 1 = the process opt-out of power throttling is set again and
///   every thread at High priority or above is tagged High QoS on its own.
/// - FRITCHII_FOCUS_GRAB (live): 1 = the game's window takes the foreground, 0 = the window that
///   had it gets it back. For a focused vs unfocused toggle in one session.
/// Focus changes are logged ("Focus: game window focused/unfocused"), checked once a second.
/// Called once a frame from RasterizerVulkan::TickFrame: a few relaxed loads when nothing changes.
namespace VideoCommon::FritchiiFocus {

inline void Tick() {
    namespace Nx2 = VideoCore::Nx2Debug;
    namespace Focus = Common::Windows::FritchiiFocus;
    static int applied_gpu = 0;
    static int applied_qos = 0;
    static int applied_grab = 0;
    static int original_gpu_class = -2; // -2: not read yet
    static int last_foreground = -1;
    static std::chrono::steady_clock::time_point next_check{};

    const int gpu = Nx2::SwitchValue(Nx2::g_gpu_priority, "FRITCHII_GPU_PRIORITY", 0);
    if (gpu != applied_gpu) [[unlikely]] {
        if (original_gpu_class == -2) {
            original_gpu_class = Focus::GetGpuPriorityClass();
        }
        // D3DKMT_SCHEDULINGPRIORITYCLASS: 2 normal, 3 above normal, 4 high. Realtime (5) is left
        // out on purpose: it needs extra privileges and has hung NVIDIA drivers.
        const int wanted_class = gpu >= 2   ? 4
                                 : gpu == 1 ? 3
                                            : (original_gpu_class >= 0 ? original_gpu_class : 2);
        const long status = Focus::SetGpuPriorityClass(wanted_class);
        LOG_INFO(Render_Vulkan,
                 "FRITCHII_GPU_PRIORITY={}: GPU scheduling class {} asked (was {} at start), "
                 "status 0x{:08x}, now {}",
                 gpu, wanted_class, original_gpu_class, static_cast<unsigned long>(status),
                 Focus::GetGpuPriorityClass());
        applied_gpu = gpu;
    }

    const int qos = Nx2::SwitchValue(Nx2::g_thread_qos, "FRITCHII_THREAD_QOS", 0);
    if (qos != applied_qos) [[unlikely]] {
        int registered = 0;
        const int tagged = Focus::SetHotThreadsHighQoS(qos != 0, &registered);
        LOG_INFO(Render_Vulkan, "FRITCHII_THREAD_QOS={}: {} of {} hot threads {}", qos, tagged,
                 registered, qos != 0 ? "tagged High QoS" : "back to system-managed QoS");
        applied_qos = qos;
    }

    const int grab = Nx2::SwitchValue(Nx2::g_focus_grab, "FRITCHII_FOCUS_GRAB", 0);
    if (grab != applied_grab) [[unlikely]] {
        // On its own short thread: the foreground hand-over attaches to another thread's input
        // and may take a few milliseconds; the render thread doesn't wait for it.
        const bool take = grab != 0;
        std::thread([take, grab] {
            const bool ok = take ? Focus::GrabForeground() : Focus::ReleaseForeground();
            LOG_INFO(Render_Vulkan, "FRITCHII_FOCUS_GRAB={}: {} {}", grab,
                     take ? "taking the foreground" : "handing the foreground back",
                     ok ? "done" : "refused by Windows");
        }).detach();
        applied_grab = grab;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= next_check) [[unlikely]] {
        next_check = now + std::chrono::seconds{1};
        const int foreground = Focus::IsForeground() ? 1 : 0;
        if (foreground != last_foreground) {
            // The GPU class too: shows whether Windows itself moves it with the focus.
            LOG_INFO(Render_Vulkan, "Focus: game window {} (GPU scheduling class {})",
                     foreground ? "focused" : "unfocused", Focus::GetGpuPriorityClass());
            last_foreground = foreground;
        }
    }
}

} // namespace VideoCommon::FritchiiFocus

#endif // _WIN32
