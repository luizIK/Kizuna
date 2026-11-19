// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

class QWidget;

// Once per major version (Common::g_build_version "v6": v6.x asks once, v7 again), before any game
// boots, the player is asked whether to clear the old shader caches, so the new version starts
// from a clean cache. Only the files the emulator rebuilds by itself, and files no part of this
// build reads any more, are removed; the precompiled list, the warm-up files and anything unknown
// stay. Driver caches (NVIDIA DXCache/GLCache, AMD VkCache, Mesa) are never touched.
//
// FRITCHII_CACHE_CLEANUP: 0 never asks; "ask" asks again although this version already asked;
// "dry" logs what would be removed and removes nothing (the answer is not remembered).
namespace FritchiiCacheCleanup {

struct File {
    std::filesystem::path path;
    std::uintmax_t bytes{};
    std::string reason; ///< why it goes or stays (log only)
};

struct Plan {
    std::vector<File> remove;
    std::vector<File> keep;
    std::uintmax_t remove_bytes{};
    std::uintmax_t keep_bytes{};
};

/// Sorts the files of a shader cache folder (<data>/cache/shader) into what goes and what stays.
/// Reads names, sizes and a few small headers; changes nothing.
[[nodiscard]] Plan PlanCleanup(const std::filesystem::path& shader_dir);

/// "v6" of Common::g_build_version "v6", "v6.1" or "v6.0 Beta" (empty for a build without a
/// version): the once-per-major-version questions key their saved answers to it.
[[nodiscard]] std::string MajorVersion();

/// Main window, before a game boots (not when one was started from the command line: that launch
/// leaves the question to the next one). Asks when this major version has not asked yet and there
/// is enough to clear, and clears on "Clear". Returns true when it recorded an answer in
/// UISettings (the caller saves the config).
bool OfferOnce(QWidget* parent);

} // namespace FritchiiCacheCleanup
