// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <optional>
#include <string>
#include <QMainWindow>
#include "yuzu/migration_worker.h"

/// What the player picked in Fritchii's copy dialog (v5.8.2): the old folder ("eden", "lichii",
/// "yuzu"...) and which parts to copy. Mods are off unless ticked.
struct ProfileImportChoice {
    std::string folder;
    bool saves = true;    ///< nand, sdmc, keys, play time, amiibo
    bool settings = true; ///< config, nx2_switches.txt, post shaders, the DLSS/FSR 4 runtimes
    bool mods = false;    ///< load

    [[nodiscard]] std::string Serialize() const;
    [[nodiscard]] static std::optional<ProfileImportChoice> Parse(const std::string& text);
};

// TODO: Quick implementation
class UserDataMigrator {
public:
    UserDataMigrator(QMainWindow* main_window);

    /// Fritchii's copy dialog: which old folder and what to copy. Nullopt when the player
    /// declines (Start fresh on the first launch, Cancel from the Help menu) or none exists.
    /// `from_menu`: Help > Copy Data (the current data is kept as backups, Fritchii restarts).
    [[nodiscard]] static std::optional<ProfileImportChoice> AskImport(QWidget* parent,
                                                                      bool from_menu);

    /// v5.9, at startup once logging runs and before the game list or a game uses the shader
    /// cache: logs what the import did, then, once, copies Eden's and Lichii's shader caches for
    /// players who imported with v5.8 (it left them behind), with a progress dialog and a short
    /// note when done.
    static void CatchUpShaderCache(QWidget* parent);

    bool migrated{false};
    Emulator selected_emu;

private:
    /// Fritchii v5.8: copies the chosen old profile into the Fritchii folder and says what it
    /// did in one box. The old profile is never changed. v5.8.2: asks first (`request` set: the
    /// Help menu already asked).
    void AutoImport(QMainWindow* main_window, std::optional<ProfileImportChoice> request = {});
    void ShowMigrationPrompt(QMainWindow* main_window);
    void ShowMigrationCancelledMessage(QMainWindow* main_window);
    void MigrateUserData(QMainWindow* main_window, const bool clear_shader_cache,
                         const MigrationWorker::MigrationStrategy strategy);
};
