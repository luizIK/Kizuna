// SPDX-FileCopyrightText: Copyright 2026 Kizuna contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Kizuna is a personal fork of Fritchii (which forks Lichii, which builds on Eden and yuzu; see
// KIZUNA.md and LICENSE.txt). The code still tells builds apart by Common::g_build_name ==
// "Fritchii" (settings defaults, the UI), so that name stays; what a person reads is these.

#pragma once

#include <string>
#include <string_view>

#include "common/scm_rev.h"

namespace Common {

namespace KizunaDetail {
inline constexpr std::string_view TagPrefix{"kizuna-"};

[[nodiscard]] inline bool IsKizunaBuild() {
    return std::string_view{g_build_release_tag}.starts_with(TagPrefix);
}
} // namespace KizunaDetail

/// "Kizuna" for the fork's CI builds (release tag "kizuna-..."), otherwise the upstream name.
[[nodiscard]] inline std::string DisplayBuildName() {
    return KizunaDetail::IsKizunaBuild() ? std::string{"Kizuna"} : std::string{g_build_name};
}

/// "v1.6" for the fork's CI builds (release tag "kizuna-v1.6"; "build N" for the first ones,
/// tagged "kizuna-N"), otherwise the upstream version.
[[nodiscard]] inline std::string DisplayBuildVersion() {
    if (KizunaDetail::IsKizunaBuild()) {
        const std::string_view rest =
            std::string_view{g_build_release_tag}.substr(KizunaDetail::TagPrefix.size());
        return rest.starts_with('v') ? std::string{rest} : "build " + std::string{rest};
    }
    return std::string{g_build_version};
}

} // namespace Common
