// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class QWidget;

// main.cpp: the switches as LoadNx2Switches read them at launch.
/// What this launch runs with, one entry per source, strongest first: source name ("environment",
/// "user nx2_switches.txt", ...), "A=1, B=2".
std::vector<std::pair<std::string, std::string>> Nx2SwitchGroups();
/// FRITCHII_, EDEN_NX2_ or LICHII_.
bool Nx2IsTestSwitch(std::string_view name);
/// The player's nx2_switches.txt in Fritchii's data folder (written by the Switches window).
std::filesystem::path Nx2UserSwitchFile();
/// The same file in the old eden folder, read while the Fritchii one doesn't exist yet.
std::filesystem::path Nx2LegacyUserSwitchFile();

namespace FritchiiUi {

/// Tools > Switches: edits the player's nx2_switches.txt (NAME=VALUE lines) and shows the
/// switches in use since launch. Returns true when the player picked Save and Restart (the file
/// is saved; the caller restarts Fritchii). `game_running` disables the restart while a game runs.
bool ShowSwitchesWindow(QWidget* parent, const std::function<bool()>& game_running);

} // namespace FritchiiUi
