// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon {

/// Container to push objects to be destroyed a few ticks in the future
template <typename T, size_t TICKS_TO_DESTROY>
class DelayedDestructionRing {
public:
    void Tick() {
        index = (index + 1) % TICKS_TO_DESTROY;
        elements[index].clear();
    }

    void Push(T&& object) {
        elements[index].push_back(std::move(object));
    }

private:
    size_t index = 0;
    std::array<std::vector<T>, TICKS_TO_DESTROY> elements;
};

template <typename T, size_t TICKS_TO_DESTROY>
class StampedDestructionRing {
public:
    template <typename IsFree>
    size_t Tick(IsFree&& is_free) {
        index = (index + 1) % TICKS_TO_DESTROY;
        if (!held.empty()) {
            std::vector<std::pair<u64, T>> still;
            for (auto& element : held) {
                if (!is_free(element.first)) {
                    still.push_back(std::move(element));
                }
            }
            held = std::move(still);
        }
        size_t newly_held = 0;
        for (auto& element : elements[index]) {
            if (!is_free(element.first)) {
                held.push_back(std::move(element));
                ++newly_held;
            }
        }
        elements[index].clear();
        return newly_held;
    }

    void Push(T&& object, u64 stamp) {
        elements[index].emplace_back(stamp, std::move(object));
    }

private:
    size_t index = 0;
    std::array<std::vector<std::pair<u64, T>>, TICKS_TO_DESTROY> elements;
    std::vector<std::pair<u64, T>> held;
};

} // namespace VideoCommon
