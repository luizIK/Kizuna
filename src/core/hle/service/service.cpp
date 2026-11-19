// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/scope_exit.h"
#include <vector>
#include <tuple>
#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <chrono>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <fmt/ranges.h>
#include "common/assert.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/hle/ipc.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/kernel_helpers.h"
#include "core/hle/service/service.h"
#include "core/hle/service/nx2_stub_interface.h"
#include "core/hle/service/sm/sm.h"
#include "core/reporter.h"

namespace Service {

namespace {

// Calls in flight for the hang report (FRITCHII_HANG_REPORT_S): one slot per thread that handles
// service calls, claimed at its first call. A call stores five words on entry and one on exit,
// relaxed, with the OS's coarse tick (Nx2Hitch::StallTickNs) for its start: nothing a game's
// thousands of calls a second would notice. The report reads them without a lock; a torn read
// only mixes two calls' fields in one diagnostic line.
struct CallSlot {
    std::atomic<const char*> service{};
    std::atomic<const char*> function{}; ///< null until the handler is found
    std::atomic<u32> command{};
    std::atomic<u8> phase{}; ///< 0 no call, 1 waiting for the service's lock, 2 in the handler
    std::atomic<u64> since{};
};
constexpr size_t CallSlotCount = 128;
std::array<CallSlot, CallSlotCount> g_call_slots;
std::atomic<size_t> g_call_slots_claimed{};
thread_local CallSlot* t_call_slot{};
thread_local bool t_call_slot_claimed{};

CallSlot* ThisThreadCallSlot() {
    if (!t_call_slot_claimed) {
        t_call_slot_claimed = true;
        const size_t index = g_call_slots_claimed.fetch_add(1, std::memory_order_relaxed);
        if (index < CallSlotCount) {
            t_call_slot = &g_call_slots[index];
        }
    }
    return t_call_slot;
}

/// The call's slot for its scope; a call made from inside another one (same thread) restores
/// the outer call's fields when it ends.
class CallInFlight {
public:
    CallInFlight(const char* service, u32 command) : slot{ThisThreadCallSlot()} {
        if (slot == nullptr) {
            return;
        }
        outer_service = slot->service.load(std::memory_order_relaxed);
        outer_function = slot->function.load(std::memory_order_relaxed);
        outer_command = slot->command.load(std::memory_order_relaxed);
        outer_phase = slot->phase.load(std::memory_order_relaxed);
        outer_since = slot->since.load(std::memory_order_relaxed);
        slot->service.store(service, std::memory_order_relaxed);
        slot->function.store(nullptr, std::memory_order_relaxed);
        slot->command.store(command, std::memory_order_relaxed);
        slot->since.store(Common::Nx2Hitch::StallTickNs(), std::memory_order_relaxed);
        slot->phase.store(1, std::memory_order_release);
    }
    ~CallInFlight() {
        if (slot == nullptr) {
            return;
        }
        slot->service.store(outer_service, std::memory_order_relaxed);
        slot->function.store(outer_function, std::memory_order_relaxed);
        slot->command.store(outer_command, std::memory_order_relaxed);
        slot->since.store(outer_since, std::memory_order_relaxed);
        slot->phase.store(outer_phase, std::memory_order_release);
    }
    CallInFlight(const CallInFlight&) = delete;
    CallInFlight& operator=(const CallInFlight&) = delete;

    void InHandler() {
        if (slot != nullptr) {
            slot->phase.store(2, std::memory_order_release);
        }
    }

private:
    CallSlot* slot;
    const char* outer_service{};
    const char* outer_function{};
    u32 outer_command{};
    u8 outer_phase{};
    u64 outer_since{};
};

/// The handler's name, once InvokeRequest found it (no-op outside a tracked call).
void NoteCallFunction(const char* name) {
    if (t_call_slot != nullptr) {
        t_call_slot->function.store(name, std::memory_order_relaxed);
    }
}

} // Anonymous namespace

std::string DescribeCallsInFlight() {
    const u64 now = Common::Nx2Hitch::StallTickNs();
    const size_t claimed =
        (std::min)(g_call_slots_claimed.load(std::memory_order_relaxed), CallSlotCount);
    std::string text;
    size_t calls = 0;
    for (size_t i = 0; i < claimed; ++i) {
        const CallSlot& slot = g_call_slots[i];
        const u8 phase = slot.phase.load(std::memory_order_acquire);
        if (phase == 0) {
            continue;
        }
        const char* const service = slot.service.load(std::memory_order_relaxed);
        const char* const function = slot.function.load(std::memory_order_relaxed);
        const u64 since = slot.since.load(std::memory_order_relaxed);
        ++calls;
        text += fmt::format("{}{} command {} ({}) {} for {:.1f} s", text.empty() ? "" : "; ",
                            service != nullptr ? service : "?",
                            slot.command.load(std::memory_order_relaxed),
                            function != nullptr ? function : "no handler name",
                            phase == 1 ? "waiting for the service's lock" : "in its handler",
                            now > since ? static_cast<double>(now - since) / 1e9 : 0.0);
    }
    return fmt::format("{} ({} service threads tracked{})", calls == 0 ? "none" : text, claimed,
                       g_call_slots_claimed.load(std::memory_order_relaxed) > CallSlotCount
                           ? ", more threads than slots: some untracked"
                           : "");
}

/// @brief Creates a function string for logging, complete with the name (or header code, depending
/// on what's passed in) the port name, and all the cmd_buf arguments.
[[maybe_unused]] static std::string MakeFunctionString(std::string_view name, std::string_view port_name, const u32* cmd_buf) {
    // Number of params == bits 0-5 + bits 6-11
    int num_params = (cmd_buf[0] & 0x3F) + ((cmd_buf[0] >> 6) & 0x3F);
    std::string function_string = fmt::format("fn '{}': port={}", name, port_name);
    for (int i = 1; i <= num_params; ++i)
        function_string += fmt::format(", cmd_buf[{}]={:#x}", i, cmd_buf[i]);
    return function_string;
}

ServiceFrameworkBase::ServiceFrameworkBase(Core::System& system_, const char* service_name_, u32 max_sessions_, InvokerFn* handler_invoker_)
    : SessionRequestHandler(system_.Kernel(), service_name_)
    , system{system_}
    , service_name{service_name_}
    , handler_invoker{handler_invoker_}
    , max_sessions{max_sessions_}
{}

ServiceFrameworkBase::~ServiceFrameworkBase() {
    // Wait for other threads to release access before destroying
    const auto guard = ServiceFrameworkBase::LockService();
    if (stub_event) {
        stub_context->CloseEvent(stub_event);
    }
}

namespace {

// Callers read the reply words they expect without checking how many were written, so an empty
// reply hands them the request's leftover words as data (an HDR output format, a refresh rate, a
// device id). Stub replies carry this many zeroed words.
constexpr u32 StubZeroedWords = 32;

void PushStubZeroes(IPC::ResponseBuilder& rb) {
    for (u32 i = 0; i < StubZeroedWords; ++i) {
        rb.Push<u32>(0);
    }
}

} // namespace

bool ServiceFrameworkBase::StubAllowed(HLERequestContext& ctx) {
    static const bool env_auto_stub = std::getenv("EDEN_NX2_AUTO_STUB") != nullptr;
    const auto* const caller = ctx.GetThread().GetOwnerProcess();
    if (Settings::values.use_auto_stub || env_auto_stub ||
        std::string_view{service_name}.starts_with(Nx2StubPrefix) ||
        (caller != nullptr && caller->IsSwitch2())) {
        // Logged once per command: these are still unimplemented.
        static std::mutex mutex;
        static std::set<std::pair<std::string_view, u32>> logged;
        std::scoped_lock lk{mutex};
        if (logged.emplace(service_name, ctx.GetCommand()).second) {
            LOG_WARNING(Service, "(STUBBED) {}::{} (placeholder reply)", service_name, ctx.GetCommand());
        }
        return true;
    }
    const auto& table = ctx.IsTipc() ? handlers_tipc : handlers;
    const auto it = table.find(ctx.GetCommand());
    ReportUnimplementedFunction(ctx, it == table.end() ? nullptr : &it->second);
    return false;
}

void ServiceFrameworkBase::StubReplyZeroed(HLERequestContext& ctx) {
    if (!StubAllowed(ctx)) {
        return;
    }
    IPC::ResponseBuilder rb{ctx, 2 + StubZeroedWords};
    rb.Push(ResultSuccess);
    PushStubZeroes(rb);
}

void ServiceFrameworkBase::StubReplyEvent(HLERequestContext& ctx) {
    if (!StubAllowed(ctx)) {
        return;
    }
    if (!stub_event) {
        stub_context = std::make_unique<KernelHelpers::ServiceContext>(system, service_name);
        stub_event = stub_context->CreateEvent(fmt::format("{}:StubEvent", service_name));
    }
    IPC::ResponseBuilder rb{ctx, 2 + StubZeroedWords, 1, 0};
    rb.Push(ResultSuccess);
    PushStubZeroes(rb);
    rb.PushCopyObjects(ctx, stub_event->GetReadableEvent());
}

void ServiceFrameworkBase::StubReplyInterface(HLERequestContext& ctx) {
    if (!StubAllowed(ctx)) {
        return;
    }
    ReturnNx2StubInterface(system, ctx, service_name);
}

void ServiceFrameworkBase::RegisterHandlersBase(const FunctionInfoBase* functions, std::size_t n) {
    // Usually this array is sorted by id already, so hint to insert at the end
    handlers.reserve(handlers.size() + n);
    for (std::size_t i = 0; i < n; ++i)
        handlers.emplace_hint(handlers.cend(), functions[i].expected_header, functions[i]);
}

void ServiceFrameworkBase::RegisterHandlersBaseTipc(const FunctionInfoBase* functions, std::size_t n) {
    // Usually this array is sorted by id already, so hint to insert at the end
    handlers_tipc.reserve(handlers_tipc.size() + n);
    for (std::size_t i = 0; i < n; ++i)
        handlers_tipc.emplace_hint(handlers_tipc.cend(), functions[i].expected_header, functions[i]);
}

void ServiceFrameworkBase::ReportUnimplementedFunction(HLERequestContext& ctx,
                                                       const FunctionInfoBase* info) {
    // EDEN_NX2_AUTO_STUB=1 auto-stubs every unknown command without touching the config. Switch 2
    // programs are always auto-stubbed: they still call many commands nobody has implemented.
    static const bool env_auto_stub = std::getenv("EDEN_NX2_AUTO_STUB") != nullptr;
    const auto* const caller = ctx.GetThread().GetOwnerProcess();
    const bool auto_stub = Settings::values.use_auto_stub || env_auto_stub ||
                           std::string_view{service_name}.starts_with(Nx2StubPrefix) ||
                           (caller != nullptr && caller->IsSwitch2());
    // Auto-stubbed commands are reported once each: title 1 polls nx2:camera 29 every frame
    // (~47 calls a second), and every report wrote a Critical line, which flushes the log file.
    if (auto_stub) {
        static std::mutex reported_mutex;
        static std::set<std::pair<std::string, u32>> reported;
        std::scoped_lock lk{reported_mutex};
        if (!reported.emplace(service_name, ctx.GetCommand()).second) {
            IPC::ResponseBuilder rb{ctx, 2 + StubZeroedWords};
            rb.Push(ResultSuccess);
            PushStubZeroes(rb);
            return;
        }
    }

    auto cmd_buf = ctx.CommandBuffer();
    std::string function_name = info == nullptr ? "<unknown>" : info->name;

    fmt::memory_buffer buf;
    fmt::format_to(std::back_inserter(buf), "function '{}({})': port='{}' cmd_buf={{[0]={:#x}", ctx.GetCommand(), function_name, service_name, cmd_buf[0]);
    // The Switch 2 video decoder sessions are being reverse engineered: dump whole requests.
    const int last_word = std::string_view{service_name}.starts_with("nx2:vdec") ? 47 : 8;
    for (int i = 1; i <= last_word; ++i)
        fmt::format_to(std::back_inserter(buf), ", [{}]={:#x}", i, cmd_buf[i]);
    buf.push_back('}');

    system.GetReporter().SaveUnimplementedFunctionReport(ctx, ctx.GetCommand(), function_name, service_name);
    if (!auto_stub) {
        UNIMPLEMENTED_MSG("Unknown / unimplemented {}", fmt::to_string(buf));
        return;
    }
    // An auto-stubbed command gets a reply, so it is logged rather than asserted: with debug
    // asserts on, the assert broke before the reply was sent, at the first such command of every
    // Switch 2 program (#1238).
    LOG_CRITICAL(Service, "Unknown / unimplemented {}", fmt::to_string(buf));
    LOG_WARNING(Service, "Using auto stub fallback (repeats of this command are not logged)");
    IPC::ResponseBuilder rb{ctx, 2 + StubZeroedWords};
    rb.Push(ResultSuccess);
    PushStubZeroes(rb);
}

void ServiceFrameworkBase::InvokeRequest(HLERequestContext& ctx) {
    auto it = handlers.find(ctx.GetCommand());
    FunctionInfoBase const* info = it == handlers.end() ? nullptr : &it->second;
    if (info == nullptr || info->handler_callback == nullptr)
        return ReportUnimplementedFunction(ctx, info);

    LOG_TRACE(Service, "{}", MakeFunctionString(info->name, GetServiceName(), ctx.CommandBuffer()));
    NoteCallFunction(info->name);
    // EDEN_NX2_IPC_TRACE=1 logs every dispatched command and the result word it returned.
    static const bool ipc_trace = std::getenv("EDEN_NX2_IPC_TRACE") != nullptr;
    if (ipc_trace) {
        LOG_INFO(Service, "ipc {}::{} ({})", service_name, ctx.GetCommand(), info->name);
    }
    handler_invoker(this, info->handler_callback, ctx);
    if (ipc_trace) {
        // Response layout: [0..1] header, then 'SFCO', version, result.
        const auto* cmd_buf = ctx.CommandBuffer();
        for (int i = 0; i < 12; ++i) {
            if (cmd_buf[i] == 0x4F434653) {  // "SFCO"
                if (cmd_buf[i + 2] != 0) {
                    LOG_INFO(Service, "ipc {}::{} -> result {:#x}", service_name, ctx.GetCommand(),
                             cmd_buf[i + 2]);
                }
                break;
            }
        }
    }
}

void ServiceFrameworkBase::InvokeRequestTipc(HLERequestContext& ctx) {
    auto it = handlers_tipc.find(ctx.GetCommand());
    FunctionInfoBase const* info = it == handlers_tipc.end() ? nullptr : &it->second;
    if (info == nullptr || info->handler_callback == nullptr)
        return ReportUnimplementedFunction(ctx, info);

    LOG_TRACE(Service, "{}", MakeFunctionString(info->name, GetServiceName(), ctx.CommandBuffer()));
    NoteCallFunction(info->name);
    static const bool ipc_trace = std::getenv("EDEN_NX2_IPC_TRACE") != nullptr;
    if (ipc_trace) {
        const auto* w = ctx.CommandBuffer();
        LOG_INFO(Service, "tipc {}::{} ({}) [2]={:#x} [3]={:#x} [4]={:#x}", service_name,
                 ctx.GetCommand(), info->name, w[2], w[3], w[4]);
    }
    handler_invoker(this, info->handler_callback, ctx);
}

namespace {
// EDEN_NX2_IPC_SUMMARY: every 5 s, log how often each service command ran and with which result
// since the last summary (finds a game polling something that never becomes ready).
void CountRequest(const std::string& service, u32 command, u32 type, u32 result) {
    static const bool enabled = std::getenv("EDEN_NX2_IPC_SUMMARY") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::map<std::tuple<std::string, u32, u32, u32>, u32> counts;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++counts[{service, command, type, result}];
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds{5}) {
        return;
    }
    last = now;
    std::vector<std::pair<u32, std::string>> lines;
    for (const auto& [key, count] : counts) {
        lines.emplace_back(count, fmt::format("{}::{}{} -> {:#x}", std::get<0>(key), std::get<1>(key),
                                              std::get<2>(key) >= 16 ? " (tipc)" : "",
                                              std::get<3>(key)));
    }
    std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string text;
    for (size_t i = 0; i < std::min<size_t>(lines.size(), 24); ++i) {
        text += fmt::format(" | {}x {}", lines[i].first, lines[i].second);
    }
    LOG_INFO(Service, "ipc summary ({} kinds){}", lines.size(), text);
    counts.clear();
}
} // Anonymous namespace

Result ServiceFrameworkBase::HandleSyncRequest(Kernel::KServerSession& session,
                                               HLERequestContext& ctx) {
    CallInFlight in_flight{service_name, ctx.GetCommand()};
    const auto guard = LockService();
    in_flight.InHandler();
    SCOPE_EXIT {
        const auto* cmd_buf = ctx.CommandBuffer();
        u32 result_word = 0;
        const u32 type = static_cast<u32>(ctx.GetCommandType());
        if (type >= 16) {
            result_word = cmd_buf[2];
        } else {
            for (int i = 0; i < 12; ++i) {
                if (cmd_buf[i] == 0x4F434653) { // "SFCO"
                    result_word = cmd_buf[i + 2];
                    break;
                }
            }
        }
        CountRequest(service_name, ctx.GetCommand(), type, result_word);
    };

    Result result = ResultSuccess;

    switch (ctx.GetCommandType()) {
    case IPC::CommandType::Close:
    case IPC::CommandType::TIPC_Close: {
        IPC::ResponseBuilder rb{ctx, 2};
        rb.Push(ResultSuccess);
        result = IPC::ResultSessionClosed;
        break;
    }
    case IPC::CommandType::ControlWithContext:
    case IPC::CommandType::Control: {
        system.ServiceManager().InvokeControlRequest(ctx);
        break;
    }
    case IPC::CommandType::RequestWithContext:
    case IPC::CommandType::Request: {
        InvokeRequest(ctx);
        break;
    }
    default:
        if (ctx.IsTipc()) {
            InvokeRequestTipc(ctx);
            break;
        }

        UNIMPLEMENTED_MSG("command_type={}", ctx.GetCommandType());
        break;
    }

    // If emulation was shutdown, we are closing service threads, do not write the response back to
    // memory that may be shutting down as well.
    if (system.IsPoweredOn()) {
        ctx.WriteToOutgoingCommandBuffer();
    }

    return result;
}

} // namespace Service
