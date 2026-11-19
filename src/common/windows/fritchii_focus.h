// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common::Windows::FritchiiFocus {

/// Records the calling thread as a hot thread (Common::SetCurrentThreadPriority at High or above)
/// so FRITCHII_THREAD_QOS can tag it later from another thread.
void RegisterHotThread();

/// FRITCHII_THREAD_QOS: on = the process opt-out of power throttling is asserted again and every
/// hot thread is tagged High QoS explicitly; off = the hot threads go back to system-managed QoS.
/// Returns how many hot threads took the change.
int SetHotThreadsHighQoS(bool on, int* registered);

/// FRITCHII_GPU_PRIORITY: the process's GPU scheduling priority class (D3DKMT_SCHEDULINGPRIORITY-
/// CLASS: 2 normal, 3 above normal, 4 high). Returns the NTSTATUS (0 = success, -1 = no export).
long SetGpuPriorityClass(int priority_class);
/// The current class, or -1 when it can't be read.
int GetGpuPriorityClass();

/// True while a window of this process is the foreground window.
bool IsForeground();
/// FRITCHII_FOCUS_GRAB = 1: brings this process's largest visible top-level window to the
/// foreground and remembers the window that had it.
bool GrabForeground();
/// FRITCHII_FOCUS_GRAB = 0: hands the foreground back to the remembered window (only while this
/// process still has it).
bool ReleaseForeground();

} // namespace Common::Windows::FritchiiFocus
