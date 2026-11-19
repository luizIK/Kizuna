// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>

#include <fmt/format.h>

#include "common/common_types.h"

// Ghost pools (fable52_answer_stall2 section 3b, diagnostic): EDEN_NX2_GHOST_POOLS=1 (boot) runs
// fifteen make-believe copies of EXP42's image pool next to the real one: five eviction rules at
// 256, 512 and 1,024 MiB each (the last two, fable52_answer_stall3 3, appended to the line). They
// hold keys and byte sizes only, no images and no memory, and see exactly the real pool's calls:
// every image the real pool is given (PoolImage) and every lookup it answers (TakePooledImage),
// plus its frame tick (the same aging: 180 frames, EXP82 for proven shapes, EXP65 for never-taken
// ones, emptied when the real pool is emptied for memory).
// The images a pool is given do not depend on which pool answered a lookup (the texture cache
// frees an image the same way either way), so each ghost's hit rate is what that rule and cap
// would have had. Every 600 frames, next to the "Image pool (EXP42)" line, one line with each
// ghost's takes, misses, hit rate and bytes held. Off (the default): one null pointer test per
// pool call.
namespace VideoCommon::Nx2GhostPools {

enum class Policy : u8 {
    Oldest,        ///< Today's: over the cap the oldest entry goes, proven or not
    UnprovenFirst, ///< Never-taken shapes first (oldest first), then least recently taken
    Frequency,     ///< UnprovenFirst, proven ones by fewest takes (then least recently taken)
    /// Frequency with every shape's takes halved each 600 frames (fable52_answer_stall3 3)
    FrequencyHalved,
    /// Frequency where a missed lookup counts as a take for its key (fable52_answer_stall3 3):
    /// a new shape arriving while the pool is full of proven entries can earn its place
    FrequencyDemand,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Policy::Count)> PolicyNames{
    "oldest first", "unproven first", "unproven first + takes",
    "unproven first + takes halved every 600 frames", "unproven first + takes and misses"};
inline constexpr std::array<u64, 3> CapsMiB{256, 512, 1024};

class Ghost {
public:
    Ghost(Policy policy_, u64 cap_) : policy{policy_}, cap{cap_} {}

    void Give(u64 key, u64 bytes, u64 frame) {
        entries.push_back({key, bytes, frame});
        held += bytes;
        while (held > cap && !entries.empty()) {
            EvictOne();
        }
    }

    /// Newest first, as the real pool.
    void Take(u64 key, u64 frame) {
        for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
            if (it->key != key) {
                continue;
            }
            held -= it->bytes;
            entries.erase(std::next(it).base());
            Shape& shape = shapes[key];
            ++shape.takes;
            shape.last_take = frame;
            ++taken;
            return;
        }
        ++missed;
        if (policy == Policy::FrequencyDemand) {
            // The miss is demand: the shape counts as proven, with one more take.
            Shape& shape = shapes[key];
            ++shape.takes;
            shape.last_take = frame;
        }
    }

    /// The real pool's age rules (TextureCacheRuntime::TickFrame), with this ghost's own proof.
    void Age(u64 frame, u64 max_frames, u64 proven_frames, u64 unproven_frames) {
        if (policy == Policy::FrequencyHalved && frame % 600 == 0) {
            // Halved, not dropped: a shape whose takes reach 0 stays proven.
            for (auto& [key, shape] : shapes) {
                shape.takes >>= 1;
            }
        }
        for (auto it = entries.begin(); it != entries.end();) {
            const u64 age = frame - it->frame;
            if (age <= max_frames) {
                break;
            }
            if (Proven(it->key) && age <= proven_frames) {
                ++it;
                continue;
            }
            held -= it->bytes;
            it = entries.erase(it);
        }
        if (unproven_frames == 0) {
            return;
        }
        for (auto it = entries.begin(); it != entries.end();) {
            if (frame - it->frame <= unproven_frames) {
                break;
            }
            if (Proven(it->key)) {
                ++it;
                continue;
            }
            held -= it->bytes;
            it = entries.erase(it);
        }
    }

    void Clear() {
        entries.clear();
        held = 0;
    }

    /// "<cap> MiB <taken>/<missed> <rate>% <held> MiB", then the counts restart.
    [[nodiscard]] std::string Report() {
        const u64 lookups = taken + missed;
        std::string text = fmt::format("{} MiB {}/{} {:.0f}% {} MiB", cap >> 20, taken, missed,
                                       lookups != 0 ? 100.0 * static_cast<double>(taken) /
                                                          static_cast<double>(lookups)
                                                    : 0.0,
                                       held >> 20);
        taken = missed = 0;
        return text;
    }

private:
    struct Entry {
        u64 key;
        u64 bytes;
        u64 frame;
    };
    struct Shape {
        u64 takes{};
        u64 last_take{};
    };

    [[nodiscard]] const Shape* FindShape(u64 key) const {
        const auto it = shapes.find(key);
        return it != shapes.end() ? &it->second : nullptr;
    }

    [[nodiscard]] bool Proven(u64 key) const {
        return FindShape(key) != nullptr;
    }

    void EvictOne() {
        auto victim = entries.begin();
        if (policy != Policy::Oldest) {
            // The oldest never-taken entry; failing that the proven one the rule ranks lowest,
            // oldest first among equals (strict comparisons keep the earlier one).
            auto best = entries.end();
            const Shape* best_shape = nullptr;
            for (auto it = entries.begin(); it != entries.end(); ++it) {
                const Shape* const shape = FindShape(it->key);
                if (shape == nullptr) {
                    best = it;
                    break;
                }
                if (best == entries.end() || Lower(*shape, *best_shape)) {
                    best = it;
                    best_shape = shape;
                }
            }
            victim = best;
        }
        held -= victim->bytes;
        entries.erase(victim);
    }

    /// Whether proven shape `a` goes before `b`.
    [[nodiscard]] bool Lower(const Shape& a, const Shape& b) const {
        if (policy != Policy::UnprovenFirst && a.takes != b.takes) {
            return a.takes < b.takes;
        }
        return a.last_take < b.last_take;
    }

    Policy policy;
    u64 cap;
    std::deque<Entry> entries; ///< Oldest first
    u64 held{};
    /// Shapes this ghost answered a lookup for (proven), by key hash; FrequencyDemand also adds
    /// the ones it missed. Unbounded (a diagnostic).
    std::unordered_map<u64, Shape> shapes;
    u64 taken{};
    u64 missed{};
};

class Ghosts {
public:
    Ghosts() {
        for (size_t p = 0; p < static_cast<size_t>(Policy::Count); ++p) {
            for (size_t c = 0; c < CapsMiB.size(); ++c) {
                ghosts[p * CapsMiB.size() + c] =
                    std::make_unique<Ghost>(static_cast<Policy>(p), CapsMiB[c] << 20);
            }
        }
    }

    void Give(u64 key, u64 bytes, u64 frame) {
        for (auto& ghost : ghosts) {
            ghost->Give(key, bytes, frame);
        }
    }

    void Take(u64 key, u64 frame) {
        for (auto& ghost : ghosts) {
            ghost->Take(key, frame);
        }
    }

    void Age(u64 frame, u64 max_frames, u64 proven_frames, u64 unproven_frames) {
        for (auto& ghost : ghosts) {
            ghost->Age(frame, max_frames, proven_frames, unproven_frames);
        }
    }

    void Clear() {
        for (auto& ghost : ghosts) {
            ghost->Clear();
        }
    }

    /// "Ghost pools (EXP42) last 600 frames, taken/missed hit% held: oldest first: 256 MiB
    /// 1200/1500 44% 250 MiB, 512 MiB ..., 1024 MiB ...; unproven first: ...; ..."
    [[nodiscard]] std::string Report() {
        std::string text;
        for (size_t p = 0; p < static_cast<size_t>(Policy::Count); ++p) {
            text += fmt::format("{}{}: ", p == 0 ? "" : "; ", PolicyNames[p]);
            for (size_t c = 0; c < CapsMiB.size(); ++c) {
                text += fmt::format("{}{}", c == 0 ? "" : ", ",
                                    ghosts[p * CapsMiB.size() + c]->Report());
            }
        }
        return text;
    }

private:
    std::array<std::unique_ptr<Ghost>, static_cast<size_t>(Policy::Count) * CapsMiB.size()> ghosts;
};

/// EDEN_NX2_GHOST_POOLS=1 at the texture cache's creation (boot): the ghosts, else null.
[[nodiscard]] inline std::unique_ptr<Ghosts> CreateIfEnabled() {
    const char* const value = std::getenv("EDEN_NX2_GHOST_POOLS");
    if (value == nullptr || value[0] != '1') {
        return nullptr;
    }
    return std::make_unique<Ghosts>();
}

} // namespace VideoCommon::Nx2GhostPools
