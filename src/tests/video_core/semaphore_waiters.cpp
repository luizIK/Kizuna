// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// A ported test, adapted: this tree's waiters have no slot limit (Add
// returns nothing), the wake is off until enabled (EDEN_NX2_GPU_WAKE), and the original's GPU tick wake
// is not ported.

#include <array>
#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2.h"

using Tegra::Nx2PendingWake;
using Tegra::Nx2SemaphoreWaiters;

TEST_CASE("Pending wakes end hardware and OS waits", "[video_core][nx2_waiters]") {
    using namespace std::chrono_literals;
    Nx2PendingWake wake;
    wake.SetEnabled(true);
    wake.Arm();
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now() + 1ms));
    CHECK_FALSE(wake.Sleep(1ms));

    // A wake between Arm and the wait is not lost.
    wake.Notify();
    CHECK(wake.Poll(std::chrono::steady_clock::now()));
    CHECK(wake.Sleep(5s));
    wake.Arm();
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));

    for (int i = 0; i < 20; ++i) {
        const bool sleep = i % 2 != 0;
        wake.Arm();
        std::jthread notifier([&] {
            std::this_thread::sleep_for(1ms);
            wake.Notify();
        });
        const auto start = std::chrono::steady_clock::now();
        CHECK((sleep ? wake.Sleep(5s) : wake.Poll(start + 5s)));
        CHECK(std::chrono::steady_clock::now() - start < 1s);
    }
}

TEST_CASE("Pending wakes do nothing while the switch is off", "[video_core][nx2_waiters]") {
    Nx2PendingWake wake;
    wake.Arm();
    wake.Notify();
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));
}

TEST_CASE("Semaphore waiters see stores to the words they wait on", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    std::array<u32, 2> words{3, 3};
    const int watched{}, unwatched{};
    waiters.Add(&watched, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal, &words[0]);
    waiters.Add(&unwatched, 0x1004, 5, Nx2SemaphoreWaiters::AcquireGequal);
    CHECK_FALSE(waiters.WordStored());

    // Any store that changes the word, met or not, and once per change.
    std::jthread([&] { std::atomic_ref<u32>{words[0]}.store(4); }).join();
    CHECK(waiters.WordStored());
    CHECK_FALSE(waiters.WordStored());
    words[1] = 9;
    CHECK_FALSE(waiters.WordStored());
    words[0] = 5;
    CHECK(waiters.WordStored());

    // A stored word is no longer watched once its waiter leaves or moves to another word.
    waiters.Add(&watched, 0x1004, 9, Nx2SemaphoreWaiters::AcquireGequal, &words[1]);
    words[0] = 6;
    CHECK_FALSE(waiters.WordStored());
    waiters.Remove(&watched);
    words[1] = 10;
    CHECK_FALSE(waiters.WordStored());
    waiters.Remove(&unwatched);
}

TEST_CASE("Latching writes wake the GPU thread", "[video_core][nx2_waiters]") {
    Nx2SemaphoreWaiters waiters;
    auto& wake = Nx2PendingWake::Get();
    wake.SetEnabled(true);
    const int owner{};
    waiters.Add(&owner, 0x1000, 5, Nx2SemaphoreWaiters::AcquireGequal);
    wake.Arm();
    waiters.OnWrite(0x1000, 4);
    waiters.OnWrite(0x2000, 9);
    CHECK_FALSE(wake.Poll(std::chrono::steady_clock::now()));
    waiters.OnWrite(0x1000, 5);
    CHECK(wake.Poll(std::chrono::steady_clock::now()));
    waiters.Remove(&owner);
    wake.SetEnabled(false);
}
