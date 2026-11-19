// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

class QWidget;

namespace VkDeviceInfo {
class Record;
}

// "Good to know": the common things v6 players will still run into (kill hitches, the slowdown
// after cutting lots of flowers, first-visit pop-in...), shown once per major version after the
// first-launch questions, and from Help > Known Issues. Only items checked against current
// builds; the DLSS line shows on NVIDIA cards only, the hybrid-CPU line on Intel CPUs only.
//
// FRITCHII_KNOWN_ISSUES: 0 never shows it on its own; "ask" shows it again although this version
// already did.
namespace FritchiiKnownIssues {

/// The window, now (Help > Known Issues).
void Show(QWidget* parent, const std::vector<VkDeviceInfo::Record>& records);

/// Main window, on a plain launch: once per major version. Returns true when it recorded that
/// it showed (the caller saves the config).
bool ShowOnce(QWidget* parent, const std::vector<VkDeviceInfo::Record>& records);

} // namespace FritchiiKnownIssues
