// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include <QCoreApplication>
#include <QString>

#include "common/frame_budget.h"
#include "common/settings.h"

// The performance overlays' time rows (View > Frame Times), shared by both overlay looks: where
// a frame's time goes (common/frame_budget.h) in milliseconds per frame, and what holds the
// frame rate down.
namespace FrameBudgetView {

enum class Limit { None, Capped, Gpu, Front, Back, Render, Game, Sync };

struct View {
    bool valid{};
    double frame_ms{};
    double gpu_ms = -1;      ///< -1: not measured (yet)
    double gpu_share = -1;   ///< GPU busy share of the wall time
    bool gpu_unsupported{};  ///< "n/a"
    double cpu_share = -1;   ///< Fritchii's share of the whole CPU
    /// Busiest thread of each group, ms per frame (-1: no such thread).
    std::array<double, Common::FrameBudget::Groups> ms{-1, -1, -1, -1};
    Limit limit{};
};

/// A thread or the GPU this busy holds the frame rate (the rest wait on it).
inline constexpr double LimitShare = 0.85;

/// The rows from a Take and the frame rate over the same time.
inline View Make(const Common::FrameBudget::Sample& sample, double fps) {
    using Common::FrameBudget::Group;
    View view;
    view.gpu_unsupported = sample.gpu_unsupported;
    if (!sample.valid || !(fps > 1.0)) {
        return view;
    }
    view.valid = true;
    view.frame_ms = 1000.0 / fps;
    view.cpu_share = sample.cpu;
    if (sample.gpu >= 0) {
        view.gpu_share = sample.gpu;
        view.gpu_ms = sample.gpu * view.frame_ms;
    }
    for (size_t i = 0; i < view.ms.size(); ++i) {
        view.ms[i] = sample.busy[i] >= 0 ? sample.busy[i] * view.frame_ms : -1;
    }

    // At the frame cap (speed limit on, the game's target reached) nothing is the limit.
    const double target = Settings::values.oot_30fps.GetValue() ? 30.0 : 60.0;
    if (Settings::values.use_speed_limit.GetValue() && fps >= target * 0.97) {
        view.limit = Limit::Capped;
        return view;
    }
    const std::array<std::pair<double, Limit>, 5> shares{{
        {sample.gpu, Limit::Gpu},
        {sample.busy[static_cast<size_t>(Group::Front)], Limit::Front},
        {sample.busy[static_cast<size_t>(Group::Back)], Limit::Back},
        {sample.busy[static_cast<size_t>(Group::Render)], Limit::Render},
        {sample.busy[static_cast<size_t>(Group::Game)], Limit::Game},
    }};
    const auto top = std::ranges::max_element(
        shares, [](const auto& a, const auto& b) { return a.first < b.first; });
    // Nobody near full: the threads wait on each other (drains, read-backs, the present).
    view.limit = top->first >= LimitShare ? top->second : Limit::Sync;
    return view;
}

inline QString Tr(const char* text) {
    return QCoreApplication::translate("FrameBudgetView", text);
}

/// "GPU", "front", ... for "Limited by ...", empty for none.
inline QString LimitText(Limit limit) {
    switch (limit) {
    case Limit::Capped:
        return Tr("Capped");
    case Limit::Gpu:
        return Tr("Limited by GPU");
    case Limit::Front:
        return Tr("Limited by front");
    case Limit::Back:
        return Tr("Limited by back");
    case Limit::Render:
        return Tr("Limited by render");
    case Limit::Game:
        return Tr("Limited by game");
    case Limit::Sync:
        return Tr("Limited by sync");
    case Limit::None:
        break;
    }
    return {};
}

/// "9.8" (ms), "-" when unknown.
inline QString Ms(double ms) {
    return ms < 0 ? QStringLiteral("-") : QString::number(ms, 'f', 1);
}

/// "62%", "-" when unknown.
inline QString Percent(double share) {
    return share < 0 ? QStringLiteral("-") : QStringLiteral("%1%").arg(std::lround(share * 100));
}

} // namespace FrameBudgetView
