// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

class QWidget;

namespace VkDeviceInfo {
class Record;
}

// OoT's "Max Resolution (built-in)" add-on: on, the game draws a fixed 2560x1440 docked (sharper,
// and what the upscalers' 4K choices build on); off, its own dynamic resolution, at most 1080p like
// a Switch 2. v6 turns it off once for everyone and then asks, with this PC's card in the question.
//
// FRITCHII_MAXRES_PROMPT: 0 never switches it off nor asks; "ask" asks again although this
// version already asked (the switch-off is not repeated).
namespace FritchiiMaxResolution {

/// The add-on as saved (on unless disabled in Add-Ons), without the EDEN_NX2_NO_MAX_RES test
/// override Settings::OcarinaMaxResolutionOn reads.
[[nodiscard]] bool Saved();

/// Turns the add-on on or off as the game's Add-Ons page does (Settings::values.disabled_addons,
/// saved with the global config) and drops the game list's cached add-on text for OoT.
void Save(bool on);

/// Once per major version, before a game from the command line boots: turns the add-on off for
/// everyone. Returns true when it recorded that in UISettings (the caller saves the config).
bool SwitchOffOnce();

/// Main window, on a plain launch (one started with a game leaves it to the next one): once per
/// major version, asks whether to turn the add-on on, naming the card `records` would run on.
/// Returns true when it recorded an answer (the caller saves the config).
bool OfferOnce(QWidget* parent, const std::vector<VkDeviceInfo::Record>& records);

} // namespace FritchiiMaxResolution
