// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/max_resolution.h"

#include <algorithm>
#include <string>

#include <fmt/format.h>

#include <QDialog>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QVBoxLayout>

#include "common/common_types.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/util/vk.h"
#include "yuzu/fritchii/cache_cleanup.h"
#include "yuzu/fritchii/fritchii_ui.h"

namespace FritchiiMaxResolution {
namespace {

constexpr u64 OcarinaTitleId = 0x0400486024DEC000ULL;
constexpr const char* AddonName = "Max Resolution (built-in)";

/// A card with this much device-local memory counts as "8 GB" (8 GB cards report a little less).
constexpr u64 EnoughVram = 7680ULL << 20;

enum class Advice : u8 { Unknown, On, Off };

struct CardLine {
    QString text; ///< Empty when the card is unknown
    Advice advice{Advice::Unknown};
    std::string log; ///< For the log line
};

/// "RTX 4080" of "NVIDIA GeForce RTX 4080", "RX 6700 XT" of "AMD Radeon RX 6700 XT (RADV NAVI22)",
/// "Arc B580" of "Intel(R) Arc(TM) B580 Graphics".
QString ShortName(const std::string& device_name) {
    QString name = QString::fromStdString(device_name);
    name.remove(QStringLiteral("(R)"));
    name.remove(QStringLiteral("(TM)"), Qt::CaseInsensitive);
    if (name.endsWith(QLatin1Char(')'))) {
        if (const qsizetype open = name.lastIndexOf(QStringLiteral(" (")); open > 0) {
            name.truncate(open);
        }
    }
    name = name.simplified();
    for (const char* prefix : {"NVIDIA ", "GeForce ", "AMD ", "Radeon ", "Intel "}) {
        if (name.startsWith(QLatin1String(prefix), Qt::CaseInsensitive)) {
            name.remove(0, static_cast<qsizetype>(std::char_traits<char>::length(prefix)));
        }
    }
    if (name.endsWith(QStringLiteral(" Graphics"))) {
        name.chop(9);
    }
    return name.isEmpty() ? QString::fromStdString(device_name) : name;
}

/// Cards slower than about an RTX 3070 / RX 6800 that can still have 8 GB or more (owner
/// 2026-11-18: the bar a notch above the RTX 3060 Ti / RX 6700 XT first used).
bool EntryLevel(const QString& name) {
    static const QRegularExpression pattern(
        QStringLiteral(R"(\b(GTX|MX)\s*\d|\bRTX\s*([2-5]050|2060|3060)\b|)"
                       R"(\bRTX\s*20[78]0\b(?!\s*Ti)|\bRTX\s*4060\b(?!\s*Ti)|\bRX\s*[45]\d\d\b|)"
                       R"(\bRX\s*(5500|5600|5700|6400|6500|6600|6650|6700|6750|7600)\b|)"
                       R"(\bVega\b|\bR[579]\s|\bArc\s*(A[357]\d\d|B5\d\d))"),
        QRegularExpression::CaseInsensitiveOption);
    return pattern.match(name).hasMatch();
}

/// The card a game would run on, in one line with a plain recommendation.
CardLine DescribeCard(const std::vector<VkDeviceInfo::Record>& records) {
    if (records.empty()) {
        return {{}, Advice::Unknown, "no Vulkan device found"};
    }
    const size_t index = VkDeviceInfo::PickedIndex(
        records, Settings::values.vulkan_device.GetValue(true),
        Settings::values.vulkan_device_name.GetValue(true));
    if (index >= records.size()) {
        return {{}, Advice::Unknown, "no card picked"};
    }
    const VkDeviceInfo::Record& card = records[index];
    if (card.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        return {QObject::tr("Your graphics are built into the processor: keep it off."),
                Advice::Off, fmt::format("{}, integrated", card.device_name)};
    }
    if (card.device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || card.local_memory == 0) {
        return {{}, Advice::Unknown, fmt::format("{}, not a graphics card", card.device_name)};
    }
    const QString name = ShortName(card.device_name);
    const qulonglong gb = (card.local_memory + (512ULL << 20)) >> 30;
    const std::string log = fmt::format("{}, {} GB", card.device_name, gb);
    if (card.local_memory < EnoughVram) {
        return {QObject::tr("Your %1 has %2 GB: keep it off.").arg(name).arg(gb), Advice::Off,
                log};
    }
    if (EntryLevel(name)) {
        return {QObject::tr("Your %1 (%2 GB) may be too slow for it: keep it off.")
                    .arg(name)
                    .arg(gb),
                Advice::Off, log};
    }
    return {QObject::tr("Your %1 (%2 GB) handles it.").arg(name).arg(gb), Advice::On, log};
}

const char* AdviceName(Advice advice) {
    switch (advice) {
    case Advice::On:
        return "on";
    case Advice::Off:
        return "off";
    default:
        return "none";
    }
}

QLabel* Label(const QString& text, const char* object_name) {
    auto* label = new QLabel(text);
    label->setObjectName(QLatin1String(object_name));
    label->setWordWrap(true);
    return label;
}

/// One side of the question: "On" or "Off" over its short lines.
QFrame* SideCard(const QString& side, const QStringList& lines) {
    auto* card = new QFrame;
    card->setObjectName(QStringLiteral("FritchiiMaxResCard"));
    auto* layout = new QVBoxLayout(card);
    layout->setContentsMargins(22, 18, 22, 20);
    layout->setSpacing(8);
    layout->addWidget(Label(side, "FritchiiMaxResSide"));
    for (const QString& line : lines) {
        layout->addWidget(Label(line, "FritchiiLead"));
    }
    layout->addStretch(1);
    return card;
}

/// Turn on or Keep off. Closing the window counts as Keep off.
bool AskToTurnOn(QWidget* parent, const QString& card_line) {
    const QString title = QObject::tr("Max Resolution");
    const QStringList on_lines{
        QObject::tr("Sharper (1440p)"),
        QObject::tr("Needed for 4K and higher with DLSS or FSR"),
        QObject::tr("Needs about an RTX 3070 / RX 6800 or better (8 GB+)"),
    };
    const QStringList off_lines{
        QObject::tr("Like the Switch 2 (up to 1080p)"),
        QObject::tr("More fps on weaker PCs"),
    };
    const QString on_text = QObject::tr("Turn on");
    const QString off_text = QObject::tr("Keep off");

    if (!FritchiiUi::Enabled()) {
        // The same words as the two sides of the Fritchii window, as two sentences.
        QString text = QObject::tr("On: sharper (1440p), needed for 4K and higher with DLSS or FSR. Needs about an "
                                   "RTX 3070 / RX 6800 or better (8 GB+).") +
                       QStringLiteral("\n") +
                       QObject::tr("Off: like the Switch 2 (up to 1080p), more fps on weaker PCs.");
        if (!card_line.isEmpty()) {
            text += QStringLiteral("\n\n") + card_line;
        }
        text += QStringLiteral("\n\n") +
                QObject::tr("Change it later: right-click OoT > Properties > Add-Ons.");
        QMessageBox box(QMessageBox::Question, QStringLiteral("Fritchii"), title,
                        QMessageBox::NoButton, parent);
        box.setInformativeText(text);
        QPushButton* const turn_on = box.addButton(on_text, QMessageBox::AcceptRole);
        QPushButton* const keep_off = box.addButton(off_text, QMessageBox::RejectRole);
        box.setDefaultButton(keep_off);
        box.setEscapeButton(keep_off);
        box.exec();
        return box.clickedButton() == turn_on;
    }

    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("FritchiiMaxResPrompt"));
    dialog.setWindowTitle(QStringLiteral("Fritchii"));
    dialog.setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    dialog.setMinimumWidth(660);
    auto* layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(40, 34, 40, 30);
    layout->setSpacing(16);
    layout->addWidget(Label(title, "FritchiiDialogTitle"));
    auto* sides = new QHBoxLayout;
    sides->setSpacing(14);
    sides->addWidget(SideCard(QObject::tr("On"), on_lines), 1);
    sides->addWidget(SideCard(QObject::tr("Off"), off_lines), 1);
    layout->addLayout(sides);
    if (!card_line.isEmpty()) {
        layout->addWidget(Label(card_line, "FritchiiMaxResYours"));
    }
    layout->addWidget(
        Label(QObject::tr("Change it later in Graphics, or right-click OoT > Properties > Add-Ons."),
              "FritchiiLead"));
    layout->addSpacing(10);
    auto* row = new QHBoxLayout;
    row->setSpacing(10);
    row->addStretch(1);
    auto* turn_on = new QPushButton(on_text);
    turn_on->setAutoDefault(false);
    auto* keep_off = new QPushButton(off_text);
    keep_off->setDefault(true);
    QObject::connect(turn_on, &QPushButton::clicked, &dialog, &QDialog::accept);
    QObject::connect(keep_off, &QPushButton::clicked, &dialog, &QDialog::reject);
    row->addWidget(turn_on);
    row->addWidget(keep_off);
    layout->addLayout(row);
    return dialog.exec() == QDialog::Accepted;
}

} // Anonymous namespace

bool Saved() {
    const auto it = Settings::values.disabled_addons.find(OcarinaTitleId);
    return it == Settings::values.disabled_addons.end() ||
           std::find(it->second.begin(), it->second.end(), AddonName) == it->second.end();
}

void Save(bool on) {
    if (on == Saved()) {
        return;
    }
    auto& disabled = Settings::values.disabled_addons[OcarinaTitleId];
    if (on) {
        std::erase(disabled, std::string{AddonName});
    } else {
        disabled.emplace_back(AddonName);
        std::sort(disabled.begin(), disabled.end());
    }
    // The game list caches the add-ons it shows (ConfigurePerGameAddons::ApplyConfiguration).
    Common::FS::RemoveFile(Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "game_list" /
                           fmt::format("{:016X}.pv.txt", OcarinaTitleId));
}

bool SwitchOffOnce() {
    const std::string version = FritchiiCacheCleanup::MajorVersion();
    if (version.empty() || qgetenv("FRITCHII_MAXRES_PROMPT") == "0") {
        return false;
    }
    auto& done = UISettings::values.fritchii_maxres_off;
    if (done.GetValue() == version) {
        return false;
    }
    const bool was_on = Saved();
    Save(false);
    done.SetValue(version);
    LOG_INFO(Frontend, "Max Resolution ({}): switched off once for this version (was {})", version,
             was_on ? "on" : "off");
    return true;
}

bool OfferOnce(QWidget* parent, const std::vector<VkDeviceInfo::Record>& records) {
    const std::string version = FritchiiCacheCleanup::MajorVersion();
    if (version.empty()) {
        return false; // a build without a version number: nothing to key the question to
    }
    const QByteArray mode = qgetenv("FRITCHII_MAXRES_PROMPT");
    if (mode == "0") {
        return false;
    }
    auto& asked = UISettings::values.fritchii_maxres_asked;
    if (asked.GetValue() == version && mode != "ask") {
        return false;
    }
    const CardLine card = DescribeCard(records);
    const bool on = AskToTurnOn(parent, card.text);
    Save(on);
    asked.SetValue(version);
    LOG_INFO(Frontend, "Max Resolution ({}): {} by the player (card: {}; suggested: {})", version,
             on ? "turned on" : "kept off", card.log, AdviceName(card.advice));
    return true;
}

} // namespace FritchiiMaxResolution
