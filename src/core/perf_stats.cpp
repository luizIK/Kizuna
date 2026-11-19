// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2017 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <iterator>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/perf_stats.h"

using namespace std::chrono_literals;
using DoubleSecs = std::chrono::duration<double, std::chrono::seconds::period>;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Purposefully ignore the first five frames, as there's a significant amount of overhead in
// booting that we shouldn't account for
constexpr std::size_t IgnoreFrames = 5;

namespace Core {

PerfStats::PerfStats(u64 title_id_) : title_id(title_id_) {}

PerfStats::~PerfStats() {
    LogRunSummary();

    if (!Settings::values.record_frame_times || title_id == 0) {
        return;
    }

    const std::time_t t = std::time(nullptr);
    std::ostringstream stream;
    std::copy(perf_history.begin() + IgnoreFrames, perf_history.begin() + current_index,
              std::ostream_iterator<double>(stream, "\n"));

    const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir);
    // %F Date format expanded is "%Y-%m-%d"
    const auto filename = fmt::format("{}_{:016X}.csv",
        [&] {
            std::ostringstream oss;
            oss << std::put_time(std::localtime(&t), "%F-%H-%M");
            return oss.str();
        }(),
        title_id);

    const auto filepath = path / filename;

    if (Common::FS::CreateParentDir(filepath)) {
        Common::FS::IOFile file(filepath, Common::FS::FileAccessMode::Write,
                                Common::FS::FileType::TextFile);
        void(file.WriteString(stream.str()));
    }
}

void PerfStats::BeginSystemFrame() {
    std::scoped_lock lock{object_mutex};

    frame_begin = Clock::now();
}

void PerfStats::EndSystemFrame() {
    std::scoped_lock lock{object_mutex};

    auto frame_end = Clock::now();
    const auto frame_time = frame_end - frame_begin;
    if (current_index < perf_history.size()) {
        perf_history[current_index++] =
            std::chrono::duration<double, std::milli>(frame_time).count();
    }
    accumulated_frametime += frame_time;
    system_frames += 1;

    // The whole run past the boot frames, for the summary at game stop.
    if (current_index > IgnoreFrames) {
        const auto us = std::chrono::duration_cast<microseconds>(frame_time).count();
        const std::size_t bucket =
            std::min(static_cast<std::size_t>((std::max)(us, decltype(us){0})) / RunBucketUs,
                     RunBuckets - 1);
        ++run_buckets[bucket];
        ++run_frames;
        run_time += frame_time;
        run_longest = (std::max)(run_longest, frame_time);
        run_over_100ms += frame_time > 100ms ? 1 : 0;
    }

    previous_frame_length = frame_end - previous_frame_end;
    previous_frame_end = frame_end;
}

void PerfStats::EndGameFrame() {
    game_frames.fetch_add(1, std::memory_order_relaxed);
}

double PerfStats::GetMeanFrametime() const {
    std::scoped_lock lock{object_mutex};

    if (current_index <= IgnoreFrames) {
        return 0;
    }

    const double sum = std::accumulate(perf_history.begin() + IgnoreFrames,
                                       perf_history.begin() + current_index, 0.0);
    return sum / static_cast<double>(current_index - IgnoreFrames);
}

PerfStatsResults PerfStats::GetAndResetStats(microseconds current_system_time_us) {
    std::scoped_lock lock{object_mutex};

    const auto now = Clock::now();
    // Walltime elapsed since stats were reset
    const auto interval = duration_cast<DoubleSecs>(now - reset_point).count();

    const auto system_us_per_second = (current_system_time_us - reset_point_system_us) / interval;
    const auto current_frames = static_cast<double>(game_frames.load(std::memory_order_relaxed));
    const auto current_fps = current_frames / interval;
    const PerfStatsResults results{
        .system_fps = static_cast<double>(system_frames) / interval,
        .average_game_fps = (current_fps + previous_fps) / 2.0,
        .frametime = duration_cast<DoubleSecs>(accumulated_frametime).count() /
                     static_cast<double>(system_frames),
        .emulation_speed = system_us_per_second.count() / 1'000'000.0,
    };

    // Reset counters
    reset_point = now;
    reset_point_system_us = current_system_time_us;
    accumulated_frametime = Clock::duration::zero();
    system_frames = 0;
    game_frames.store(0, std::memory_order_relaxed);
    previous_fps = current_fps;

    return results;
}

// One line at game stop: the run's average fps and its 1% / 0.1% lows (the fps of the slowest 1% /
// 0.1% of frames, the same way profile_toggle.py reads its histograms), loading screens included.
void PerfStats::LogRunSummary() const {
    std::scoped_lock lock{object_mutex};

    if (run_frames == 0) {
        return;
    }
    const auto low_fps = [this](double share) {
        const u64 need = (std::max)(u64{1}, static_cast<u64>(static_cast<double>(run_frames) * share));
        u64 seen = 0;
        for (std::size_t i = RunBuckets; i-- > 0;) {
            seen += run_buckets[i];
            if (seen >= need) {
                return 1e6 / static_cast<double>((i + 1) * RunBucketUs);
            }
        }
        return 0.0;
    };
    const double seconds = duration_cast<DoubleSecs>(run_time).count();
    LOG_INFO(Frontend,
             "Run summary: {} frames in {:.0f} s, average {:.1f} fps, 1% low {:.1f} fps, 0.1% low "
             "{:.1f} fps, longest frame {:.1f} ms, {} frames over 100 ms",
             run_frames, seconds, seconds > 0.0 ? static_cast<double>(run_frames) / seconds : 0.0,
             low_fps(0.01), low_fps(0.001),
             std::chrono::duration<double, std::milli>(run_longest).count(), run_over_100ms);
}

double PerfStats::GetLastFrameTimeScale() const {
    std::scoped_lock lock{object_mutex};

    constexpr double FRAME_LENGTH = 1.0 / 60;
    return duration_cast<DoubleSecs>(previous_frame_length).count() / FRAME_LENGTH;
}

void SpeedLimiter::DoSpeedLimiting(microseconds current_system_time_us) {
    if (Settings::values.use_multi_core.GetValue() ||
        !Settings::SpeedLimitEnabled()) {
        return;
    }

    auto now = Clock::now();

    const double sleep_scale = Settings::SpeedLimit() / 100.0;

    // Max lag caused by slow frames. Shouldn't be more than the length of a frame at the current
    // speed percent or it will clamp too much and prevent this from properly limiting to that
    // percent. High values means it'll take longer after a slow frame to recover and start
    // limiting
    const microseconds max_lag_time_us = duration_cast<microseconds>(
        std::chrono::duration<double, std::chrono::microseconds::period>(25ms / sleep_scale));
    speed_limiting_delta_err += duration_cast<microseconds>(
        std::chrono::duration<double, std::chrono::microseconds::period>(
            (current_system_time_us - previous_system_time_us) / sleep_scale));
    speed_limiting_delta_err -= duration_cast<microseconds>(now - previous_walltime);
    speed_limiting_delta_err =
        std::clamp(speed_limiting_delta_err, -max_lag_time_us, max_lag_time_us);

    if (speed_limiting_delta_err > microseconds::zero()) {
        std::this_thread::sleep_for(speed_limiting_delta_err);
        auto now_after_sleep = Clock::now();
        speed_limiting_delta_err -= duration_cast<microseconds>(now_after_sleep - now);
        now = now_after_sleep;
    }

    previous_system_time_us = current_system_time_us;
    previous_walltime = now;
}

} // namespace Core
