// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace FrontendCommon {

/// v5.9: the shader cache files an import (or the one-time catch-up for players who imported
/// with v5.8) copies from old shader folders (eden, lichii) into Fritchii's. Copied, never moved.
struct ShaderCacheCopy {
    struct File {
        std::filesystem::path source;
        std::filesystem::path destination;
        std::uintmax_t size{};
    };
    std::vector<File> files;
    std::uintmax_t bytes{};
};

/// What `sources` (old shader folders, merged newest-wins per file) hold that `destination`
/// lacks or has an older copy of. This fork's own pipeline caches already in `destination` are
/// never replaced: the old one goes beside it as vulkan-frizti-imported.* and the pipeline cache
/// merges it at the next boot of that game (MergeCompatiblePipelineCaches).
ShaderCacheCopy PlanShaderCacheCopy(const std::vector<std::filesystem::path>& sources,
                                    const std::filesystem::path& destination);

struct ShaderCacheCopyResult {
    std::size_t files{};
    std::uintmax_t bytes{};
    std::size_t failed{};
    std::string first_error; ///< The first failed file and why, for the log
};

/// Copies the planned files (each through a temporary name, then renamed), adding the bytes of
/// every finished file to `copied`. A file that fails is counted and skipped.
ShaderCacheCopyResult CopyShaderCache(const ShaderCacheCopy& plan,
                                      std::atomic<std::uintmax_t>* copied = nullptr);

/// Whether the disk holding `destination` has room for `bytes` and 1 GiB to spare.
bool HasRoomForShaderCache(const std::filesystem::path& destination, std::uintmax_t bytes);

/// The one-time catch-up copy ran (or an import already handled the shader cache).
bool ShaderCacheCatchUpDone();
void MarkShaderCacheCatchUpDone();

std::filesystem::path ProfileImportStateDirectory();
bool NeedsLegacyProfileImport();
/// Fritchii v5.8.1, Help > Copy Data from Eden or Lichii: asks the next launch to copy a profile
/// over Fritchii's own (the current one kept as backups). `request`: the folder ("eden",
/// "lichii"...) on the first line, then what to copy (v5.8.2: saves=, settings=, mods=).
void RequestProfileImport(const std::string& request);
/// The request's text, or empty. Takes it (the request file is removed), puts back an
/// interrupted import and clears the done/fresh state, so the next import runs.
std::string TakeProfileImportRequest();
void CheckLegacyProfileBeforeConfig();
std::filesystem::path RemapImportedPath(const std::filesystem::path& value,
                                        const std::filesystem::path& source,
                                        const std::filesystem::path& destination);
void RemapImportedConfig(const std::filesystem::path& config_directory,
                         const std::filesystem::path& source,
                         const std::filesystem::path& destination);

} // namespace FrontendCommon
