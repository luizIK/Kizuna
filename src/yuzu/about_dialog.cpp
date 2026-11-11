// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <QIcon>
#include <fmt/ranges.h>
#include "common/kizuna_branding.h"
#include "common/scm_rev.h"
#include "ui_aboutdialog.h"
#include "yuzu/about_dialog.h"
#include "yuzu/fritchii/fritchii_ui.h"

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent), ui{std::make_unique<Ui::AboutDialog>()} {
    static const std::string build_id = std::string{Common::g_build_id};
    static const std::string yuzu_build =
        fmt::format("{} | {} | {}", Common::DisplayBuildName(), Common::DisplayBuildVersion(),
                    std::string{Common::g_compiler_id});

    const auto override_build =
        fmt::format(fmt::runtime(std::string(Common::g_title_bar_format_idle)), build_id);
    const auto yuzu_build_version = override_build.empty() ? yuzu_build : override_build;

    ui->setupUi(this);
    // Try and request the icon from Qt theme (Linux?)
    const QIcon yuzu_logo = QIcon::fromTheme(QStringLiteral("io.github.zortos293.lichii"),
                                           QIcon(QStringLiteral(":/img/lichii.ico")));
    if (!yuzu_logo.isNull()) {
        ui->labelLogo->setPixmap(yuzu_logo.pixmap(200));
    }
    ui->labelBuildInfo->setText(
        ui->labelBuildInfo->text().arg(QString::fromStdString(yuzu_build_version),
                                       QString::fromUtf8(Common::g_build_date).left(10)));
    if (FritchiiUi::Enabled()) {
        const QString name = QString::fromStdString(Common::DisplayBuildName());
        setWindowTitle(tr("About %1").arg(name));
        ui->labelYuzu->setText(QStringLiteral("<span style='font-size:30px'>%1</span>").arg(name));
        if (name == QStringLiteral("Kizuna")) {
            // Kizuna: what it is, the projects it comes from (GPLv3 keeps their notices), and
            // where its source is.
            ui->labelAbout->setText(
                tr("<p style='font-size:12pt'>Kizuna is a personal fork of Fritchii, which builds "
                   "on Lichii, Eden and yuzu. It is free software under the GNU General Public "
                   "License v3.0 or later.</p>"));
            ui->labelLinks->setText(
                tr("<a href=\"https://github.com/luizIK/Kizuna\">Source Code</a> | "
                   "<a href=\"https://github.com/luizIK/Kizuna/issues\">Report an Issue</a> | "
                   "<a href=\"https://github.com/luizIK/Kizuna/blob/kizuna/LICENSE.txt\">"
                   "License</a>"));
        }
    }
}

AboutDialog::~AboutDialog() = default;
