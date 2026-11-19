// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

// EDEN_NX2_DEPTH_CENSUS=1 (boot, off; fable52_answer_potroom 4b): what the game's occlusion
// reports and the 0xe2c macro's reads of them are, logged every 600 frames with the depth census
// (TextureCache::CensusTickFrame). Counters are relaxed atomics: reports are made on the GPU back
// thread, written by fence operations, and the macro reads them on the front.
namespace VideoCommon::Nx2ReportCensus {

[[nodiscard]] inline bool Enabled() {
    // Read on first use, after main() applied nx2_switches.txt.
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_DEPTH_CENSUS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline constexpr size_t NumTypes = 32;

struct Counters {
    std::array<std::atomic<u64>, NumTypes> by_type{};  ///< Reports by the type asked for
    std::atomic<u64> short_reports{};                  ///< 4 bytes (no timestamp)
    std::atomic<u64> long_reports{};                   ///< 16 bytes (value and timestamp)
    std::atomic<u64> constant_one{};    ///< A type without a streamer: payload 1 written
    std::array<std::atomic<u64>, NumTypes> constant_one_by_type{};
    std::atomic<u64> fences{};          ///< Of the reports, fence-flagged
    std::atomic<u64> immediate{};       ///< Payloads written at once (not through an operation)
    std::atomic<u64> sample_zero{};     ///< Type 21 values written by the operation: zero
    std::atomic<u64> sample_nonzero{};
    std::atomic<u64> invalidated{};     ///< Operations that found the report invalidated
    std::atomic<u64> unsynchronized{};  ///< ... the final value not synchronized
    // The 0xe2c macro's reads (MME_DMA_READ_FIFOED, six words: counter, timestamp, counter).
    std::atomic<u64> e2c_reads{};
    std::atomic<u64> e2c_words_zero{};
    std::atomic<u64> e2c_words_nonzero{};
    std::atomic<u64> e2c_equal{};       ///< The two counters equal (no samples between them)
    std::atomic<u64> e2c_differ{};
    std::atomic<u64> e2c_forwarded{};   ///< Served from a queued constant buffer upload
    std::atomic<u64> e2c_snapshot{};    ///< Served from the DMA snapshot without a drain
    std::atomic<u64> e2c_snapshot_miss{}; ///< The snapshot path's drained miss
    std::atomic<u64> e2c_drained{};     ///< The plain drained path (reports listed, or no snapshot)
    std::atomic<u64> e2c_pending{};     ///< Drained reads with a report still pending on the bytes
};

inline Counters g_counters;

inline void Add(std::atomic<u64>& counter, u64 value = 1) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

inline void NoteReport(u32 requested_type, bool has_timestamp, bool is_fence, bool constant) {
    auto& c = g_counters;
    const size_t type = requested_type < NumTypes ? requested_type : NumTypes - 1;
    Add(c.by_type[type]);
    Add(has_timestamp ? c.long_reports : c.short_reports);
    if (is_fence) {
        Add(c.fences);
    }
    if (constant) {
        Add(c.constant_one);
        Add(c.constant_one_by_type[type]);
    }
}

enum class E2cPath : u8 { Forwarded, Snapshot, SnapshotMiss, Drained };

/// One 0xe2c read of `count` words, the path that served it, and whether a report was pending.
inline void NoteE2cRead(const u32* words, size_t count, E2cPath path, bool pending) {
    auto& c = g_counters;
    Add(c.e2c_reads);
    u64 zero = 0;
    for (size_t i = 0; i < count; ++i) {
        zero += words[i] == 0 ? 1 : 0;
    }
    Add(c.e2c_words_zero, zero);
    Add(c.e2c_words_nonzero, count - zero);
    if (count >= 6) {
        Add(words[0] == words[4] && words[1] == words[5] ? c.e2c_equal : c.e2c_differ);
    }
    switch (path) {
    case E2cPath::Forwarded:
        Add(c.e2c_forwarded);
        break;
    case E2cPath::Snapshot:
        Add(c.e2c_snapshot);
        break;
    case E2cPath::SnapshotMiss:
        Add(c.e2c_snapshot_miss);
        break;
    case E2cPath::Drained:
        Add(c.e2c_drained);
        break;
    }
    if (pending) {
        Add(c.e2c_pending);
    }
}

/// The two lines, then the counters start again.
inline void LogAndReset() {
    auto& c = g_counters;
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    std::string types;
    std::string constants;
    for (size_t type = 0; type < NumTypes; ++type) {
        if (const u64 n = take(c.by_type[type]); n != 0) {
            types += fmt::format("{}{}: {}", types.empty() ? "" : ", ", type, n);
        }
        if (const u64 n = take(c.constant_one_by_type[type]); n != 0) {
            constants += fmt::format("{}{}: {}", constants.empty() ? "" : ", ", type, n);
        }
    }
    LOG_INFO(HW_GPU,
             "Report census (EDEN_NX2_DEPTH_CENSUS) last 600 frames: by type {{{}}}; {} of 4 "
             "bytes, {} of 16; {} fence-flagged, {} payloads written at once; answered with the "
             "constant 1 (no streamer): {} {{{}}}; type 21 written zero {}, non-zero {}; "
             "discarded: {} invalidated, {} unsynchronized",
             types, take(c.short_reports), take(c.long_reports), take(c.fences),
             take(c.immediate), take(c.constant_one), constants, take(c.sample_zero),
             take(c.sample_nonzero), take(c.invalidated), take(c.unsynchronized));
    LOG_INFO(HW_GPU,
             "Macro 0xe2c reads (EDEN_NX2_DEPTH_CENSUS) last 600 frames: {} reads, words zero {}, "
             "non-zero {}; counters equal {}, differ {}; served: {} forwarded upload, {} DMA "
             "snapshot (EXP10/EXP73 path, no drain), {} snapshot miss (drained), {} drained; {} "
             "drained reads had a report pending on the bytes",
             take(c.e2c_reads), take(c.e2c_words_zero), take(c.e2c_words_nonzero),
             take(c.e2c_equal), take(c.e2c_differ), take(c.e2c_forwarded), take(c.e2c_snapshot),
             take(c.e2c_snapshot_miss), take(c.e2c_drained), take(c.e2c_pending));
}

} // namespace VideoCommon::Nx2ReportCensus
