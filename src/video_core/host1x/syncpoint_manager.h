// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace Tegra {

namespace Host1x {

class SyncpointManager {
public:
    static constexpr size_t NUM_MAX_SYNCPOINTS = 192;

    u32 GetGuestSyncpointValue(u32 id) const {
        return syncpoints_guest[id].load(std::memory_order_acquire);
    }

    u32 GetHostSyncpointValue(u32 id) const {
        return syncpoints_host[id].load(std::memory_order_acquire);
    }

    struct RegisteredAction {
        explicit RegisteredAction(u32 expected_value_, u64 id_, std::function<void()>&& action_)
            : expected_value{expected_value_}, id{id_}, action{std::move(action_)} {}
        u32 expected_value;
        u64 id;
        std::function<void()> action;
    };
    using ActionHandle = u64;

    template <typename Func>
    ActionHandle RegisterGuestAction(u32 syncpoint_id, u32 expected_value, Func&& action) {
        return RegisterAction(syncpoints_guest[syncpoint_id], guest_action_storage[syncpoint_id],
                              expected_value, std::move(action));
    }

    template <typename Func>
    ActionHandle RegisterHostAction(u32 syncpoint_id, u32 expected_value, Func&& action) {
        return RegisterAction(syncpoints_host[syncpoint_id], host_action_storage[syncpoint_id],
                              expected_value, std::move(action));
    }

    void DeregisterGuestAction(u32 syncpoint_id, const ActionHandle& handle);

    void DeregisterHostAction(u32 syncpoint_id, const ActionHandle& handle);

    void IncrementGuest(u32 syncpoint_id);

    /// Called after every guest increment, on the incrementing thread (nx2:gsv uses it to answer
    /// the driver's armed syncpoint waits at once instead of at its next 500 us poll). The hook
    /// must be cheap and thread-safe; nullptr removes it.
    using GuestIncrementHook = void (*)(void* context);
    void SetGuestIncrementHook(GuestIncrementHook hook, void* context) {
        guest_increment_context.store(context, std::memory_order_relaxed);
        guest_increment_hook.store(hook, std::memory_order_release);
    }

    /// Called with a syncpoint's new guest value on every guest increment, under the manager's
    /// lock, before the value is published and before the actions it meets run. Switch 2 clients
    /// read syncpoint values from a page the nx2:gsv session fills, and an action may wake a
    /// client that reads that page right away: the page must not be behind the value that
    /// answered its wait (nn::movie aborts on purpose with 2162-0001 when it is). The mirror
    /// runs on whichever thread increments: it must be cheap and take no other lock.
    using GuestMirror = std::function<void(u32 syncpoint_id, u32 value)>;
    /// Installs `mirror` for `owner` (replacing any other owner's).
    void SetGuestMirror(const void* owner, GuestMirror mirror);
    /// Removes the mirror if `owner` installed it. No mirror call runs after this returns.
    void ClearGuestMirror(const void* owner);

    void IncrementHost(u32 syncpoint_id);

    void WaitGuest(u32 syncpoint_id, u32 expected_value);

    void WaitHost(u32 syncpoint_id, u32 expected_value);

    bool IsReadyGuest(u32 syncpoint_id, u32 expected_value) const {
        return syncpoints_guest[syncpoint_id].load(std::memory_order_acquire) >= expected_value;
    }

    bool IsReadyHost(u32 syncpoint_id, u32 expected_value) const {
        return syncpoints_host[syncpoint_id].load(std::memory_order_acquire) >= expected_value;
    }

    /// The hang report (FRITCHII_HANG_REPORT_S): every syncpoint something waits on, with its
    /// guest and host values, the values its registered actions wait for (nx2:syncpt's armed
    /// slot waits among them) and the threads blocked in WaitGuest/WaitHost. Read only; never
    /// blocks for long: says so when the lock stays held for 200 ms.
    [[nodiscard]] std::string DescribeWaits();

private:
    /// `timeline_name`: the Nx2Timeline marker of the increment (a literal), with the id.
    /// `guest`: a guest increment, mirrored first (SetGuestMirror) while a mirror is set.
    void Increment(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv,
                   std::list<RegisteredAction>& action_storage, const char* timeline_name,
                   u32 syncpoint_id, bool guest = false);

    ActionHandle RegisterAction(std::atomic<u32>& syncpoint,
                                std::list<RegisteredAction>& action_storage, u32 expected_value,
                                std::function<void()>&& action);

    void DeregisterAction(std::list<RegisteredAction>& action_storage, const ActionHandle& handle);

    void Wait(std::atomic<u32>& syncpoint, std::condition_variable& wait_cv, u32 expected_value,
              u32 syncpoint_id, bool guest);

    /// A thread blocked in Wait (under `guard`), for DescribeWaits.
    struct BlockedWait {
        const void* token;
        u32 syncpoint_id;
        u32 expected_value;
        bool guest;
        std::chrono::steady_clock::time_point since;
    };
    std::vector<BlockedWait> blocked_waits;

    std::array<std::atomic<u32>, NUM_MAX_SYNCPOINTS> syncpoints_guest{};
    std::array<std::atomic<u32>, NUM_MAX_SYNCPOINTS> syncpoints_host{};

    std::array<std::list<RegisteredAction>, NUM_MAX_SYNCPOINTS> guest_action_storage;
    std::array<std::list<RegisteredAction>, NUM_MAX_SYNCPOINTS> host_action_storage;

    std::mutex guard;
    u64 last_action_id{};
    std::atomic<GuestIncrementHook> guest_increment_hook{};
    GuestMirror guest_mirror;               ///< Under `guard`
    const void* guest_mirror_owner{};       ///< Under `guard`
    std::atomic<bool> has_guest_mirror{};   ///< Whether guest_mirror is set (read without `guard`)
    std::atomic<void*> guest_increment_context{};
    std::condition_variable wait_guest_cv;
    std::condition_variable wait_host_cv;
};

} // namespace Host1x

} // namespace Tegra
