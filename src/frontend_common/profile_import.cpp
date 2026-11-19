// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cwctype>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#ifndef SI_NO_CONVERSION
#define SI_NO_CONVERSION 1
#endif
#include <SimpleIni.h>
#include "common/fs/path_util.h"
#include "common/fs/profile_migration.h"
#include "frontend_common/profile_import.h"

namespace FrontendCommon {
namespace fs = std::filesystem;
namespace Migration = Common::FS::ProfileMigration;

fs::path ProfileImportStateDirectory() {
    const auto& root = Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir);
    return root.parent_path() / (root.filename().string() + ".migration");
}

bool NeedsLegacyProfileImport() {
#ifdef __ANDROID__
    return false;
#else
    using namespace Common::FS;
    if (IsPortableMode()) {
        return false;
    }
    const auto state = Migration::GetState(ProfileImportStateDirectory());
    if (state == Migration::State::Pending) {
        return true;
    }
    if (state != Migration::State::Unresolved) {
        return false;
    }
    constexpr std::array sources{LegacyEdenDir, LichiiDir, CitronDir, SudachiDir, SuyuDir, YuzuDir};
    for (const auto source : sources) {
        if (fs::is_directory(GetLegacyPath(source))) {
            return true;
        }
    }
    return false;
#endif
}

void RequestProfileImport(const std::string& folder) {
    const auto state = ProfileImportStateDirectory();
    fs::create_directories(state);
    std::ofstream file{state / "import-request", std::ios::trunc};
    file << folder;
    file.close();
    if (!file) {
        throw std::runtime_error("Could not write the import request");
    }
}

std::string TakeProfileImportRequest() {
#ifdef __ANDROID__
    return {};
#else
    if (Common::FS::IsPortableMode()) {
        return {};
    }
    const auto state = ProfileImportStateDirectory();
    const auto request = state / "import-request";
    std::error_code ec;
    if (!fs::exists(request, ec)) {
        return {};
    }
    std::string text;
    {
        std::ifstream file{request};
        text.assign(std::istreambuf_iterator<char>{file}, {});
    }
    fs::remove(request, ec);
    // The first line names the folder; v5.8.2 adds what to copy on the next lines.
    const std::string folder = text.substr(0, text.find('\n'));
    constexpr std::array known{"eden", "lichii", "yuzu", "suyu", "sudachi", "citron"};
    if (std::ranges::find(known, folder) == known.end()) {
        return {};
    }
    // The Fritchii that asked is closing: let it finish writing its config first.
    std::this_thread::sleep_for(std::chrono::seconds{2});
    if (Migration::GetState(state) == Migration::State::Pending) {
        Migration::Recover(state);
    }
    // Forget the earlier import's outcome (its lock file stays) so RunImport runs again.
    for (const auto& entry : fs::directory_iterator(state, ec)) {
        if (entry.path().filename() != "lock") {
            fs::remove_all(entry.path(), ec);
        }
    }
    return text;
#endif
}

void CheckLegacyProfileBeforeConfig() {
    if (NeedsLegacyProfileImport()) {
        throw std::runtime_error("Legacy emulator data or an unfinished import was found. "
                                 "Open the Kizuna window to import it or explicitly start fresh "
                                 "before using eden-cli. No configuration was created.");
    }
}

namespace {

/// This fork's own pipeline cache files (vk_pipeline_cache.cpp): never replaced by an old copy.
constexpr std::array<std::string_view, 2> OwnPipelineStems{"vulkan-frizti", "vulkan_used-frizti"};

/// "vulkan-frizti.fork-v54.bin" -> "vulkan-frizti-imported.fork-v54.bin", the name the pipeline
/// cache merges once at boot; empty for any other file.
std::string ImportedPipelineName(const std::string& name) {
    if (!name.ends_with(".bin")) {
        return {};
    }
    for (const auto stem : OwnPipelineStems) {
        if (name.size() > stem.size() && name.starts_with(stem) && name[stem.size()] == '.') {
            return std::string{stem} + "-imported" + name.substr(stem.size());
        }
    }
    return {};
}

/// The pipeline cache's merge log (vulkan-frizti.fork-v54.bin.merged): it belongs to its cache.
bool IsOwnMergeLog(const std::string& name) {
    return name.starts_with("vulkan-frizti.") && name.ends_with(".bin.merged");
}

fs::path ShaderCacheCatchUpMarker() {
    return ProfileImportStateDirectory() / "shader-cache-copied";
}

} // namespace

ShaderCacheCopy PlanShaderCacheCopy(const std::vector<fs::path>& sources,
                                    const fs::path& destination) {
    struct Newest {
        fs::path source;
        fs::file_time_type time;
        std::uintmax_t size{};
    };
    // Newest wins per file across the sources.
    std::map<fs::path, Newest> newest;
    const auto now = fs::file_time_type::clock::now();
    for (const auto& root : sources) {
        std::error_code ec;
        if (!fs::is_directory(root, ec) || fs::is_symlink(root, ec)) {
            continue;
        }
        for (auto it = fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            std::error_code entry_ec;
            if (it->is_symlink(entry_ec)) {
                it.disable_recursion_pending();
                continue;
            }
            if (!it->is_regular_file(entry_ec)) {
                continue;
            }
            const auto name = Common::FS::PathToUTF8String(it->path().filename());
            if (name.ends_with(".fritchii-copy") || name.ends_with(".migrating")) {
                continue; // a copy another program left half done
            }
            const auto time = it->last_write_time(entry_ec);
            const auto size = entry_ec ? 0 : it->file_size(entry_ec);
            if (entry_ec) {
                continue;
            }
            // Written in the last seconds: the old emulator may still be writing it.
            const auto age = now - time;
            if (age >= decltype(age)::zero() && age < std::chrono::seconds{10}) {
                continue;
            }
            const auto [entry, added] = newest.try_emplace(it->path().lexically_relative(root),
                                                           Newest{it->path(), time, size});
            if (!added && time > entry->second.time) {
                entry->second = Newest{it->path(), time, size};
            }
        }
    }

    ShaderCacheCopy plan;
    const auto add = [&](const Newest& from, const fs::path& target) {
        plan.files.push_back({from.source, target, from.size});
        plan.bytes += from.size;
    };
    // True when `target` is missing or older than `from` (and not the same file copied before).
    const auto wanted = [](const Newest& from, const fs::path& target) {
        std::error_code ec;
        if (!fs::exists(target, ec)) {
            return !ec;
        }
        const auto time = fs::last_write_time(target, ec);
        return !ec && time < from.time;
    };
    std::vector<fs::path> own_copied; // this fork's caches copied under their own names
    for (const auto& [relative, from] : newest) {
        const auto name = Common::FS::PathToUTF8String(relative.filename());
        if (IsOwnMergeLog(name)) {
            continue;
        }
        const auto target = destination / relative;
        const auto imported = ImportedPipelineName(name);
        std::error_code ec;
        if (!imported.empty() && fs::exists(target, ec)) {
            // Fritchii has its own cache of this game already (played since the import): keep
            // it, and put the old one beside it for the boot to merge.
            const bool same = fs::file_size(target, ec) == from.size && !ec &&
                              fs::last_write_time(target, ec) == from.time && !ec;
            if (!same && wanted(from, target.parent_path() / imported)) {
                add(from, target.parent_path() / imported);
            }
            continue;
        }
        if (wanted(from, target)) {
            add(from, target);
            if (!imported.empty()) {
                own_copied.push_back(relative);
            }
        }
    }
    // A merge log goes with its cache, and only when that cache was copied under its own name.
    for (const auto& [relative, from] : newest) {
        const auto name = Common::FS::PathToUTF8String(relative.filename());
        if (!IsOwnMergeLog(name)) {
            continue;
        }
        auto cache = relative;
        cache.replace_extension(); // drops ".merged"
        if (std::ranges::find(own_copied, cache) != own_copied.end()) {
            add(from, destination / relative);
        }
    }
    return plan;
}

ShaderCacheCopyResult CopyShaderCache(const ShaderCacheCopy& plan,
                                      std::atomic<std::uintmax_t>* copied) {
    ShaderCacheCopyResult result;
    for (const auto& file : plan.files) {
        std::error_code ec;
        auto temporary = file.destination;
        temporary += ".fritchii-copy";
        fs::create_directories(file.destination.parent_path(), ec);
        if (!ec) {
            fs::copy_file(file.source, temporary, fs::copy_options::overwrite_existing, ec);
        }
        if (!ec) {
            // Keep the old file's time, so a later newest-wins pass compares real ages.
            std::error_code time_ec;
            const auto time = fs::last_write_time(file.source, time_ec);
            if (!time_ec) {
                fs::last_write_time(temporary, time, time_ec);
            }
            fs::rename(temporary, file.destination, ec);
        }
        if (ec) {
            std::error_code ignored;
            fs::remove(temporary, ignored);
            if (result.failed++ == 0) {
                result.first_error =
                    Common::FS::PathToUTF8String(file.source) + ": " + ec.message();
            }
        } else {
            ++result.files;
            result.bytes += file.size;
        }
        if (copied) {
            copied->fetch_add(file.size, std::memory_order_relaxed);
        }
    }
    return result;
}

bool HasRoomForShaderCache(const fs::path& destination, std::uintmax_t bytes) {
    constexpr std::uintmax_t Spare = std::uintmax_t{1} << 30;
    std::error_code ec;
    auto path = destination;
    while (!fs::exists(path, ec)) {
        const auto parent = path.parent_path();
        if (parent.empty() || parent == path) {
            return false;
        }
        path = parent;
    }
    const auto space = fs::space(path, ec);
    return !ec && space.available >= bytes + Spare;
}

bool ShaderCacheCatchUpDone() {
    std::error_code ec;
    return fs::exists(ShaderCacheCatchUpMarker(), ec);
}

void MarkShaderCacheCatchUpDone() {
    std::error_code ec;
    fs::create_directories(ProfileImportStateDirectory(), ec);
    std::ofstream file{ShaderCacheCatchUpMarker(), std::ios::trunc};
    file << "1\n";
}

fs::path RemapImportedPath(const fs::path& value, const fs::path& source,
                           const fs::path& destination) {
    if (value.empty() || !value.is_absolute()) {
        return value;
    }
    const auto normalized = value.lexically_normal();
    auto root = source.lexically_normal();
    if (root.filename().empty()) {
        root = root.parent_path();
    }
    auto current = normalized.begin();
    for (const auto& component : root) {
        if (current == normalized.end()) {
            return value;
        }
#ifdef _WIN32
        auto left = component.wstring();
        auto right = current->wstring();
        std::transform(left.begin(), left.end(), left.begin(), ::towlower);
        std::transform(right.begin(), right.end(), right.begin(), ::towlower);
        if (left != right) {
            return value;
        }
#else
        if (component != *current) {
            return value;
        }
#endif
        ++current;
    }
    auto mapped = destination;
    for (; current != normalized.end(); ++current) {
        mapped /= *current;
    }
    return mapped.lexically_normal();
}

void RemapImportedConfig(const fs::path& config_directory, const fs::path& source,
                         const fs::path& destination) {
    if (!fs::is_directory(config_directory)) {
        return;
    }
    constexpr std::array storage_keys{"nand_directory", "save_directory", "sdmc_directory",
                                      "load_directory", "tas_directory",  "dump_directory"};
    for (const auto& entry : fs::recursive_directory_iterator(config_directory)) {
        if (entry.path().extension() != ".ini" || !entry.is_regular_file()) {
            continue;
        }
        std::ifstream input{entry.path(), std::ios::binary};
        const std::string data{std::istreambuf_iterator<char>{input}, {}};
        if (!input || input.bad()) {
            throw std::runtime_error("Could not read imported configuration");
        }
        CSimpleIniA ini;
        ini.SetUnicode();
        if (ini.LoadData(data) < 0) {
            throw std::runtime_error("Could not parse imported configuration");
        }
        bool changed = false;
        const auto remap = [&](const char* section, const char* key) {
            const auto* value = ini.GetValue(section, key);
            if (!value) {
                return;
            }
            std::string decoded{value};
            std::erase(decoded, '"');
            const auto path = Common::FS::ToU8String(decoded);
            const auto mapped = RemapImportedPath(fs::path{path}, source, destination);
            if (mapped != fs::path{path}) {
                const auto text = Common::FS::PathToUTF8String(mapped);
                if (ini.SetValue(section, key, text.c_str()) < 0) {
                    throw std::runtime_error("Could not remap imported configuration");
                }
                changed = true;
            }
        };
        for (const auto* key : storage_keys) {
            remap("Data%20Storage", key);
        }
        remap("Screenshots", "screenshot_path");
        if (changed) {
            std::string output;
            if (ini.Save(output) < 0) {
                throw std::runtime_error("Could not serialize imported configuration");
            }
            std::ofstream file{entry.path(), std::ios::binary | std::ios::trunc};
            file << output;
            file.close();
            if (!file) {
                throw std::runtime_error("Could not write imported configuration");
            }
        }
    }
}

} // namespace FrontendCommon
