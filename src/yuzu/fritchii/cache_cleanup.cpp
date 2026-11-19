// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/cache_cleanup.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>
#include <utility>

#include <fmt/format.h>

#include <QDialog>
#include <QEventLoop>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include "common/common_types.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "qt_common/config/uisettings.h"
#include "video_core/pipeline_cache_files.h"
#include "yuzu/fritchii/fritchii_ui.h"

namespace FritchiiCacheCleanup {
namespace {

namespace fs = std::filesystem;

/// Less than this to clear is not worth a question (a fresh install has a few KB at most).
constexpr std::uintmax_t MinBytesToAsk = 64ULL << 20;

constexpr std::array<char, 8> RecordsMagic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};

bool EndsWith(std::string_view text, std::string_view end) {
    return text.size() >= end.size() && text.substr(text.size() - end.size()) == end;
}

bool StartsWith(std::string_view text, std::string_view start) {
    return text.substr(0, start.size()) == start;
}

/// <stem>.fork-v<version><suffix>, e.g. vulkan-frizti.fork-v54.bin.merged
struct Versioned {
    std::string stem;
    u32 version{};
    std::string suffix;
};

std::optional<Versioned> ParseVersioned(std::string_view name) {
    constexpr std::string_view Marker = ".fork-v";
    const auto at = name.find(Marker);
    if (at == std::string_view::npos) {
        return std::nullopt;
    }
    size_t i = at + Marker.size();
    const size_t first_digit = i;
    u32 version = 0;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i])) &&
           i - first_digit < 6) {
        version = version * 10 + static_cast<u32>(name[i] - '0');
        ++i;
    }
    if (i == first_digit) {
        return std::nullopt;
    }
    return Versioned{std::string{name.substr(0, at)}, version, std::string{name.substr(i)}};
}

template <size_t N>
bool Contains(const std::array<u32, N>& versions, u32 version) {
    return std::ranges::find(versions, version) != versions.end();
}

struct Decision {
    bool remove{};
    const char* reason{};
};

constexpr Decision Remove(const char* reason) {
    return {true, reason};
}
constexpr Decision Keep(const char* reason) {
    return {false, reason};
}

/// One title's folder, <shader dir>/<title id>/. Its subfolders are left alone.
void PlanTitle(const fs::path& dir, Plan& plan) {
    constexpr u32 Current = VideoCommon::VulkanPipelineCacheVersion;
    std::error_code ec;
    std::vector<std::pair<std::string, std::uintmax_t>> files;
    std::set<std::string> names;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        std::error_code entry_ec;
        if (!entry.is_regular_file(entry_ec)) {
            continue;
        }
        std::string name = Common::FS::PathToUTF8String(entry.path().filename());
        const std::uintmax_t bytes = entry.file_size(entry_ec);
        names.insert(name);
        files.emplace_back(std::move(name), entry_ec ? 0 : bytes);
    }

    // A title whose precompiled list this PC has prepared at least once (its stamp is there): the
    // preparation runs again once the stamp is gone and rebuilds every pipeline of the list, so
    // the pipeline records gathered in play go too. Elsewhere (another game, a driver the
    // preparation skips, a preparation that never finished) those records are the only copy.
    const bool prepared =
        names.contains(fmt::format("precompiled_prepared.fork-v{}.txt", Current)) &&
        names.contains(fmt::format("precompiled.fork-v{}.nx2prep", Current));
    const std::string own_cache = fmt::format("vulkan-frizti.fork-v{}.bin", Current);
    const bool own_exists = names.contains(own_cache);
    const auto merge_log = own_exists ? VideoCommon::ReadPipelineMergeLog(dir / (own_cache + ".merged"))
                                      : VideoCommon::PipelineMergeLog{};
    const auto merged = [&](const std::string& name) {
        const auto logged = merge_log.find(name);
        const auto stamp = VideoCommon::StampPipelineCacheFile(dir / name);
        return logged != merge_log.end() && stamp && *stamp == logged->second;
    };
    // A shared cache (vulkan.fork-v<N>.bin, legacy vulkan.bin / vulkan_pipelines.bin): this build
    // only merges it into its own cache, once per change (the merge log).
    const auto shared_cache = [&](const std::string& name,
                                  std::optional<u32> version) -> Decision {
        if (prepared) {
            return Remove("pipeline records, the preparation builds them again");
        }
        const bool readable =
            version ? Contains(VideoCommon::VulkanCompatibleSharedVersions, *version)
                    : std::ranges::any_of(VideoCommon::VulkanCompatibleSharedVersions,
                                          [&](u32 compatible) {
                                              return VideoCommon::PipelineCacheHeaderMatches(
                                                  dir / name, RecordsMagic, compatible);
                                          });
        if (!readable) {
            return Remove("another version's cache, never read");
        }
        if (merged(name)) {
            return Remove("already merged into Fritchii's cache");
        }
        return Keep("shared cache not merged yet");
    };

    const auto decide = [&](const std::string& name) -> Decision {
        if (EndsWith(name, ".part") || EndsWith(name, ".tmp") || EndsWith(name, ".migrating")) {
            return Remove("unfinished write");
        }
        if (StartsWith(name, "pipeline_binaries.")) {
            return Remove("compiled pipeline store, filled again by the preparation and play");
        }
        if (EndsWith(name, ".nx2recipe")) {
            return Keep("precompile recipe");
        }
        if (StartsWith(name, "warm-")) {
            return Keep("warm-up files, only the Prepare tool makes them");
        }
        if (name == "vulkan.bin" || name == "vulkan_pipelines.bin") {
            return shared_cache(name, std::nullopt);
        }
        if (name == "vulkan_used.bin") {
            return names.contains("vulkan.bin") ? shared_cache("vulkan.bin", std::nullopt)
                                                : Remove("used list without its cache");
        }
        const auto parsed = ParseVersioned(name);
        if (!parsed) {
            return Keep("not a cache file this build knows");
        }
        const auto& [stem, version, suffix] = *parsed;
        if (stem == "opengl") {
            return version < VideoCommon::OpenGLPipelineCacheVersion
                       ? Remove("older OpenGL cache, never read")
                       : Keep("OpenGL shader records, only play makes them");
        }
        if (version > Current) {
            return Keep("a newer build's file");
        }
        if (stem == "precompiled") {
            if (suffix != ".nx2prep" && suffix != ".nx2prep.recipe.txt") {
                return Keep("not a cache file this build knows");
            }
            return version == Current
                       ? Keep("precompiled list, the preparation builds from it")
                       : Remove("an older version's precompiled list, never read");
        }
        if (stem == "precompiled_infos" || stem == "precompiled_prepared" ||
            stem == "precompiled_progress" || stem == "precompiled_order") {
            return Remove("preparation state, written again by the next preparation");
        }
        if (stem == "vulkan_pipelines") {
            return Remove("driver pipeline cache, rebuilt from the records");
        }
        if (stem == "vulkan-frizti" || stem == "vulkan_used-frizti") {
            if (prepared) {
                return Remove("pipeline records, the preparation builds them again");
            }
            if (version == Current) {
                return Keep("Fritchii's pipeline records, the only copy for this game");
            }
            if (Contains(VideoCommon::VulkanOwnPreviousVersions, version) && !own_exists) {
                return Keep("an older version's records, not copied to this version yet");
            }
            return Remove("an older version's records, copied already or never read");
        }
        if (stem == "vulkan-frizti-imported" || stem == "vulkan_used-frizti-imported") {
            if (prepared) {
                return Remove("pipeline records, the preparation builds them again");
            }
            if (version == Current || Contains(VideoCommon::VulkanOwnPreviousVersions, version)) {
                return Keep("imported records, merged at the next boot");
            }
            return Remove("an older version's records, never read");
        }
        if (stem == "vulkan" && suffix == ".bin") {
            return shared_cache(name, version);
        }
        if (stem == "vulkan_used" && suffix == ".bin") {
            const std::string cache = fmt::format("vulkan.fork-v{}.bin", version);
            return names.contains(cache) ? shared_cache(cache, version)
                                         : Remove("used list without its cache");
        }
        return Keep("not a cache file this build knows");
    };

    for (auto& [name, bytes] : files) {
        const Decision decision = decide(name);
        File file{dir / name, bytes, decision.reason};
        if (decision.remove) {
            plan.remove_bytes += bytes;
            plan.remove.push_back(std::move(file));
        } else {
            plan.keep_bytes += bytes;
            plan.keep.push_back(std::move(file));
        }
    }
}

bool IsTitleFolder(const std::string& name) {
    return name.size() == 16 && std::ranges::all_of(name, [](char c) {
               return std::isxdigit(static_cast<unsigned char>(c)) != 0;
           });
}

QString SizeText(std::uintmax_t bytes) {
    constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
    if (bytes >= (1ULL << 30)) {
        return QStringLiteral("%1 GB").arg(static_cast<double>(bytes) / GiB, 0, 'f', 1);
    }
    return QStringLiteral("%1 MB").arg(static_cast<qulonglong>(bytes >> 20));
}

std::string GbText(std::uintmax_t bytes) {
    return fmt::format("{:.2f} GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
}

/// Clear or Keep. Closing the window counts as Keep.
bool AskToClear(QWidget* parent, std::uintmax_t bytes) {
    const QString title = QObject::tr("Clear old shader caches?");
    const QString lead =
        QObject::tr("Frees %1 and can help performance a lot. Shaders are prepared again once.")
            .arg(SizeText(bytes));
    if (!FritchiiUi::Enabled()) {
        QMessageBox box(QMessageBox::Question, QStringLiteral("Fritchii"), title,
                        QMessageBox::NoButton, parent);
        box.setInformativeText(lead);
        QPushButton* const clear = box.addButton(QObject::tr("Clear"), QMessageBox::AcceptRole);
        box.addButton(QObject::tr("Keep"), QMessageBox::RejectRole);
        box.setDefaultButton(clear);
        box.exec();
        return box.clickedButton() == clear;
    }
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("FritchiiCachePrompt"));
    dialog.setWindowTitle(QStringLiteral("Fritchii"));
    auto* layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(40, 34, 40, 30);
    layout->setSpacing(14);
    auto* heading = new QLabel(title);
    heading->setObjectName(QStringLiteral("FritchiiDialogTitle"));
    layout->addWidget(heading);
    auto* sentence = new QLabel(lead);
    sentence->setObjectName(QStringLiteral("FritchiiLead"));
    layout->addWidget(sentence);
    layout->addSpacing(14);
    auto* row = new QHBoxLayout;
    row->setSpacing(10);
    row->addStretch(1);
    auto* keep = new QPushButton(QObject::tr("Keep"));
    keep->setAutoDefault(false);
    auto* clear = new QPushButton(QObject::tr("Clear"));
    clear->setDefault(true);
    QObject::connect(keep, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(clear, &QPushButton::clicked, &dialog, &QDialog::accept);
    row->addWidget(keep);
    row->addWidget(clear);
    layout->addLayout(row);
    return dialog.exec() == QDialog::Accepted;
}

struct Result {
    size_t files{};
    std::uintmax_t bytes{};
    size_t failed{};
};

/// Removes the planned files on a worker while a progress window shows how far it got (only when
/// it takes a moment: deleting a few large files is usually instant).
Result RemoveFiles(QWidget* parent, const Plan& plan) {
    std::atomic<size_t> done{0};
    Result result;
    QProgressDialog progress(parent);
    progress.setWindowTitle(QStringLiteral("Fritchii"));
    progress.setLabelText(QObject::tr("Clearing shader caches..."));
    progress.setRange(0, static_cast<int>(plan.remove.size()));
    progress.setCancelButton(nullptr);
    progress.setAutoClose(false);
    progress.setAutoReset(false);
    progress.setMinimumDuration(400);
    progress.setWindowFlag(Qt::WindowCloseButtonHint, false);
    progress.setWindowModality(Qt::ApplicationModal);
    QEventLoop loop;
    QTimer timer;
    timer.setInterval(100);
    QObject::connect(&timer, &QTimer::timeout, &progress, [&] {
        progress.setValue(static_cast<int>(done.load(std::memory_order_relaxed)));
    });
    progress.setValue(0);
    QThread* const thread = QThread::create([&] {
        for (const File& file : plan.remove) {
            std::error_code ec;
            if (fs::remove(file.path, ec)) {
                ++result.files;
                result.bytes += file.bytes;
            } else if (ec) {
                // In use (another Fritchii running) or read-only: left for now.
                ++result.failed;
                LOG_WARNING(Frontend, "Cache cleanup: could not remove {}: {}",
                            Common::FS::PathToUTF8String(file.path), ec.message());
            }
            done.fetch_add(1, std::memory_order_relaxed);
        }
    });
    QObject::connect(thread, &QThread::finished, &loop, &QEventLoop::quit);
    thread->start();
    timer.start();
    loop.exec();
    thread->wait();
    delete thread;
    timer.stop();
    progress.close();
    return result;
}

void LogPlan(const Plan& plan, std::string_view verb) {
    for (const File& file : plan.remove) {
        LOG_INFO(Frontend, "Cache cleanup: {} {} ({} MB): {}", verb,
                 Common::FS::PathToUTF8String(file.path), file.bytes >> 20, file.reason);
    }
    for (const File& file : plan.keep) {
        LOG_INFO(Frontend, "Cache cleanup: keeps {} ({} MB): {}",
                 Common::FS::PathToUTF8String(file.path), file.bytes >> 20, file.reason);
    }
}

} // Anonymous namespace

std::string MajorVersion() {
    std::string version{Common::g_build_version};
    if (const auto end = version.find_first_of(". "); end != std::string::npos) {
        version.resize(end);
    }
    return version;
}

Plan PlanCleanup(const fs::path& shader_dir) {
    Plan plan;
    std::error_code ec;
    if (!fs::is_directory(shader_dir, ec)) {
        return plan;
    }
    for (const auto& entry : fs::directory_iterator(shader_dir, ec)) {
        std::error_code entry_ec;
        const std::string name = Common::FS::PathToUTF8String(entry.path().filename());
        if (entry.is_directory(entry_ec)) {
            if (IsTitleFolder(name)) {
                PlanTitle(entry.path(), plan);
            }
            continue;
        }
        if (!entry.is_regular_file(entry_ec)) {
            continue;
        }
        const std::uintmax_t bytes = entry.file_size(entry_ec);
        File file{entry.path(), entry_ec ? 0 : bytes, {}};
        // vulkan_static_pipelines.bin: the driver cache of the emulator's own fixed pipelines
        // (blits, present), written again at the next boot (vulkan_device.cpp).
        if (name == "vulkan_static_pipelines.bin" || EndsWith(name, ".part") ||
            EndsWith(name, ".tmp")) {
            file.reason = name == "vulkan_static_pipelines.bin"
                              ? "driver cache of the emulator's own pipelines, written again"
                              : "unfinished write";
            plan.remove_bytes += file.bytes;
            plan.remove.push_back(std::move(file));
        } else {
            file.reason = "not a cache file this build knows";
            plan.keep_bytes += file.bytes;
            plan.keep.push_back(std::move(file));
        }
    }
    return plan;
}

bool OfferOnce(QWidget* parent) {
    const std::string version = MajorVersion();
    if (version.empty()) {
        return false; // a build without a version number: nothing to key the question to
    }
    const QByteArray mode = qgetenv("FRITCHII_CACHE_CLEANUP");
    if (mode == "0") {
        return false;
    }
    const bool dry = mode == "dry";
    const bool ask_again = dry || mode == "ask";
    auto& asked = UISettings::values.fritchii_cache_cleanup_asked;
    if (asked.GetValue() == version && !ask_again) {
        return false;
    }
    const auto remember = [&] {
        if (dry) {
            return false;
        }
        asked.SetValue(version);
        return true;
    };
    const fs::path shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const Plan plan = PlanCleanup(shader_dir);
    if (plan.remove_bytes < MinBytesToAsk) {
        // A fresh install, or a cache this version made: nothing old to offer, now or later.
        LOG_INFO(Frontend, "Cache cleanup ({}): nothing to clear ({} files, {} MB)", version,
                 plan.remove.size(), plan.remove_bytes >> 20);
        return remember();
    }
    if (!AskToClear(parent, plan.remove_bytes)) {
        LOG_INFO(Frontend, "Cache cleanup ({}): kept ({} files, {} offered)", version,
                 plan.remove.size(), GbText(plan.remove_bytes));
        return remember();
    }
    if (dry) {
        LogPlan(plan, "would remove");
        LOG_INFO(Frontend, "Cache cleanup ({}, dry run): {} files, {} would be removed, {} kept",
                 version, plan.remove.size(), GbText(plan.remove_bytes), plan.keep.size());
        return false;
    }
    LogPlan(plan, "removes");
    const Result result = RemoveFiles(parent, plan);
    LOG_INFO(Frontend,
             "Cache cleanup ({}): {} files, {} removed, {} kept{}; driver caches (NVIDIA "
             "DXCache/GLCache, AMD VkCache, Mesa) left alone",
             version, result.files, GbText(result.bytes), plan.keep.size() + result.failed,
             result.failed == 0 ? std::string{}
                                : fmt::format(" ({} could not be removed)", result.failed));
    return remember();
}

} // namespace FritchiiCacheCleanup
