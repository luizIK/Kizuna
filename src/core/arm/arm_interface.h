// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <span>
#include <atomic>
#include <string>
#include <vector>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/hardware_properties.h"

#include "core/hle/kernel/svc_types.h"

namespace Common {
struct PageTable;
}

namespace Kernel {
enum class DebugWatchpointType : u8;
struct DebugWatchpoint;
class KThread;
class KProcess;
} // namespace Kernel

namespace Core {
using WatchpointArray = std::array<Kernel::DebugWatchpoint, Core::Hardware::NUM_WATCHPOINTS>;

// NOTE: these values match the HaltReason enum in Dynarmic
enum class HaltReason : u64 {
    StepThread = 0x00000001,
    DataAbort = 0x00000004,
    BreakLoop = 0x02000000,
    SupervisorCall = 0x04000000,
    InstructionBreakpoint = 0x08000000,
    PrefetchAbort = 0x20000000,
};
DECLARE_ENUM_FLAG_OPERATORS(HaltReason);

enum class Architecture {
    AArch64,
    AArch32,
};

/// Generic ARMv8 CPU interface
class ArmInterface {
public:
    YUZU_NON_COPYABLE(ArmInterface);
    YUZU_NON_MOVEABLE(ArmInterface);

    explicit ArmInterface(bool uses_wall_clock) : m_uses_wall_clock{uses_wall_clock} {}
    virtual ~ArmInterface() = default;

    // Perform any backend-specific initialization.
    virtual void Initialize() {}

    // Runs the CPU until an event happens.
    virtual HaltReason RunThread(Kernel::KThread* thread) = 0;

    // Runs the CPU for one instruction or until an event happens.
    virtual HaltReason StepThread(Kernel::KThread* thread) = 0;

    // Admits a backend-specific mechanism to lock the thread context.
    virtual void LockThread(Kernel::KThread* thread) {}
    virtual void UnlockThread(Kernel::KThread* thread) {}

    // Clear the entire instruction cache for this CPU.
    virtual void ClearInstructionCache() = 0;

    // Clear a range of the instruction cache for this CPU.
    virtual void InvalidateCacheRange(u64 addr, std::size_t size) = 0;

    // Fritchii (EDEN_NX2_JIT_WARM): compiles blocks other cores ran, without running them, until
    // `stop` is set. Only from this CPU's own thread while it runs no guest code. Returns how many
    // keys it went through.
    virtual std::size_t WarmCompile(std::span<const u64> keys, const std::atomic<bool>& stop) {
        return 0;
    }

    // Fritchii (FRITCHII_JIT_HOT_PACK, a test hook): clears this CPU's code cache and compiles the
    // hot list (FRITCHII_JIT_HOT_LIST) first, in guest pc order, so the hot code sits together.
    // Same thread rules as WarmCompile. Returns how many blocks it compiled.
    virtual std::size_t PackHotCode(const std::atomic<bool>& stop) {
        return 0;
    }

    // Get the current architecture.
    // This returns AArch64 when PSTATE.nRW == 0 and AArch32 when PSTATE.nRW == 1.
    virtual Architecture GetArchitecture() const = 0;

    // Context accessors.
    // These should not be called if the CPU is running.
    virtual void GetContext(Kernel::Svc::ThreadContext& ctx) const = 0;
    virtual void SetContext(const Kernel::Svc::ThreadContext& ctx) = 0;
    virtual void SetTpidrroEl0(u64 value) = 0;

    virtual void GetSvcArguments(std::span<uint64_t, 8> args) const = 0;
    virtual void SetSvcArguments(std::span<const uint64_t, 8> args) = 0;
    virtual u32 GetSvcNumber() const = 0;

    void SetWatchpointArray(const WatchpointArray* watchpoints) {
        m_watchpoints = watchpoints;
    }

    // Signal an interrupt for execution to halt as soon as possible.
    // It is safe to call this if the CPU is not running.
    virtual void SignalInterrupt(Kernel::KThread* thread) = 0;

    // Stack trace generation.
    void LogBacktrace(Kernel::KProcess* process) const;

    // Fritchii: asks for a backtrace of the guest thread this core runs, logged by the core's
    // own thread once the JIT returns (PhysicalCore::RunThread), where the registers are synced.
    // Any thread may ask (the hang report; a burst of unmapped reads from inside the JIT's
    // callbacks). `reason` must outlive the request (a string literal).
    void RequestBacktrace(Kernel::KThread* thread, const char* reason) {
        m_backtrace_reason.store(reason, std::memory_order_release);
        SignalInterrupt(thread);
    }
    /// The pending request's reason (taken), nullptr: none.
    const char* TakeBacktraceRequest() {
        if (m_backtrace_reason.load(std::memory_order_relaxed) == nullptr) [[likely]] {
            return nullptr;
        }
        return m_backtrace_reason.exchange(nullptr, std::memory_order_acquire);
    }

    // Debug functionality.
    virtual const Kernel::DebugWatchpoint* HaltedWatchpoint() const = 0;
    virtual void RewindBreakpointInstruction() = 0;

protected:
    const Kernel::DebugWatchpoint* MatchingWatchpoint(
        u64 addr, u64 size, Kernel::DebugWatchpointType access_type) const;

protected:
    const WatchpointArray* m_watchpoints{};
    bool m_uses_wall_clock{};
    std::atomic<const char*> m_backtrace_reason{};
};

} // namespace Core
