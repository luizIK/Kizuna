// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

namespace Common {

/// Package stamp: a 16-byte id written into the binary after the build (per package).
/// Empty for an unstamped build. Shown in the log and the log export.
std::string BuildStamp();

/// Which build this copy is, e.g. "test 8": the first line of fritchii_build.txt next to the exe,
/// which the packaging scripts write (the dev build tree gets "dev <commit> <build time>").
/// Empty when the file is missing, as in a release. Read once.
const std::string& BuildTag();

/// What players see plus the tag: "v6.0 Beta test 8" (just the label without a tag).
const std::string& BuildLabelWithTag();

} // namespace Common
