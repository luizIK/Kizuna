// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/switches_window.h"

#include <system_error>

#include <QDesktopServices>
#include <QDialog>
#include <QFile>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSaveFile>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QTextEdit>
#include <QUrl>
#include <QVBoxLayout>

namespace FritchiiUi {
namespace {
namespace fs = std::filesystem;

const QColor BadColor{0xe5, 0x48, 0x4d};
const QColor OddColor{0xe0, 0x9a, 0x1f};
const QColor CommentColor{0x8a, 0x8a, 0x96};

QString PathText(const fs::path& path) {
    return QString::fromStdU16String(path.u16string());
}

/// Scales a font that may be sized in points or, on some Linux desktops, only in pixels
/// (pointSizeF() <= 0 there).
void Scale(QFont& font, double factor) {
    if (font.pointSizeF() > 0) {
        font.setPointSizeF(font.pointSizeF() * factor);
    } else if (font.pixelSize() > 0) {
        font.setPixelSize(static_cast<int>(font.pixelSize() * factor + 0.5));
    }
}

/// Gives `font` the size of `like`, whichever unit `like` uses.
void SameSize(QFont& font, const QFont& like) {
    if (like.pointSizeF() > 0) {
        font.setPointSizeF(like.pointSizeF());
    } else if (like.pixelSize() > 0) {
        font.setPixelSize(like.pixelSize());
    }
}

enum class Problem { None, NotNameValue, NotSwitch };

/// Read the way main.cpp's ParseSwitchFile reads it: blank lines and '#' lines are skipped, the
/// rest needs NAME=VALUE. A name outside FRITCHII_/EDEN_NX2_/LICHII_ is still set (driver
/// variables work too), but is most often a typo.
Problem CheckLine(const QString& raw) {
    const QString line = raw.trimmed();
    if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) {
        return Problem::None;
    }
    const auto eq = line.indexOf(QLatin1Char('='));
    if (eq < 0) {
        return Problem::NotNameValue;
    }
    const QString name = line.left(eq).trimmed();
    if (name.isEmpty()) {
        return Problem::NotNameValue;
    }
    for (const QChar c : name) {
        if (c.isSpace()) {
            return Problem::NotNameValue;
        }
    }
    return Nx2IsTestSwitch(name.toStdString()) ? Problem::None : Problem::NotSwitch;
}

class SwitchHighlighter : public QSyntaxHighlighter {
public:
    using QSyntaxHighlighter::QSyntaxHighlighter;

protected:
    void highlightBlock(const QString& text) override {
        QTextCharFormat format;
        if (text.trimmed().startsWith(QLatin1Char('#'))) {
            format.setForeground(CommentColor);
            setFormat(0, static_cast<int>(text.size()), format);
            return;
        }
        const Problem problem = CheckLine(text);
        if (problem == Problem::None) {
            return;
        }
        const QColor color = problem == Problem::NotNameValue ? BadColor : OddColor;
        format.setForeground(color);
        format.setUnderlineStyle(QTextCharFormat::WaveUnderline);
        format.setUnderlineColor(color);
        setFormat(0, static_cast<int>(text.size()), format);
    }
};

QString GroupName(const std::string& source) {
    if (source == "environment") {
        return QObject::tr("Environment");
    }
    if (source == "user nx2_switches.txt") {
        return QObject::tr("Your file");
    }
    if (source == "nx2_switches.txt next to the exe") {
        return QObject::tr("Next to the exe");
    }
    if (source == "built in") {
        return QObject::tr("Built in");
    }
    if (source == "profiling mode") {
        return QObject::tr("Profiling mode");
    }
    return QString::fromStdString(source);
}

QString InUseHtml() {
    const auto groups = Nx2SwitchGroups();
    if (groups.empty()) {
        return QObject::tr("None");
    }
    QString html;
    for (const auto& [source, list] : groups) {
        html += QStringLiteral("<p style='margin:0 0 10px 0'><b>%1</b>")
                    .arg(GroupName(source).toHtmlEscaped());
        for (const QString& entry :
             QString::fromStdString(list).split(QStringLiteral(", "), Qt::SkipEmptyParts)) {
            html += QStringLiteral("<br><code>%1</code>").arg(entry.toHtmlEscaped());
        }
        html += QStringLiteral("</p>");
    }
    return html;
}

class SwitchesDialog : public QDialog {
public:
    SwitchesDialog(QWidget* parent, std::function<bool()> game_running_)
        : QDialog(parent), game_running{std::move(game_running_)} {
        setObjectName(QStringLiteral("FritchiiSwitches"));
        setWindowTitle(QObject::tr("Switches"));
        setWindowFlag(Qt::WindowContextHelpButtonHint, false);
        QFont big = font();
        Scale(big, 1.15);
        setFont(big);
        resize(680, 720);

        auto* layout = new QVBoxLayout(this);
        layout->setSpacing(10);
        layout->setContentsMargins(22, 18, 22, 18);

        auto* title = new QLabel(QObject::tr("Switches"), this);
        title->setObjectName(QStringLiteral("FritchiiDialogTitle"));
        QFont title_font = big;
        Scale(title_font, 1.6);
        title->setFont(title_font);
        layout->addWidget(title);

        layout->addWidget(Section(QObject::tr("In use now")));
        auto* in_use = new QTextEdit(this);
        in_use->setObjectName(QStringLiteral("FritchiiSwitchesInUse"));
        in_use->setReadOnly(true);
        in_use->setHtml(InUseHtml());
        in_use->setMinimumHeight(150);
        layout->addWidget(in_use, 2);

        auto* file_row = new QHBoxLayout;
        QLabel* file_label = Section(QObject::tr("Your file"));
        file_label->setToolTip(PathText(Nx2UserSwitchFile()));
        file_row->addWidget(file_label);
        file_row->addStretch();
        auto* open_folder = new QPushButton(QObject::tr("Open Folder"), this);
        open_folder->setAutoDefault(false);
        file_row->addWidget(open_folder);
        layout->addLayout(file_row);

        editor = new QPlainTextEdit(this);
        editor->setObjectName(QStringLiteral("FritchiiSwitchesEditor"));
        editor->setLineWrapMode(QPlainTextEdit::NoWrap);
        QFont mono = QFontDatabase::systemFont(QFontDatabase::FixedFont);
        SameSize(mono, big);
        editor->setFont(mono);
        editor->setPlaceholderText(QStringLiteral("FRITCHII_EXAMPLE=1"));
        editor->setMinimumHeight(220);
        new SwitchHighlighter(editor->document());
        layout->addWidget(editor, 3);

        problems = new QLabel(this);
        problems->setObjectName(QStringLiteral("FritchiiSwitchesProblems"));
        QPalette bad = problems->palette();
        bad.setColor(QPalette::WindowText, BadColor);
        problems->setPalette(bad);
        problems->setWordWrap(true);
        problems->hide();
        layout->addWidget(problems);

        auto* buttons = new QHBoxLayout;
        saved = new QLabel(QObject::tr("Saved. Restart to use it."), this);
        saved->setObjectName(QStringLiteral("FritchiiSwitchesSaved"));
        saved->hide();
        buttons->addWidget(saved);
        buttons->addStretch();
        auto* save = new QPushButton(QObject::tr("Save"), this);
        restart = new QPushButton(QObject::tr("Save and Restart"), this);
        buttons->addWidget(save);
        buttons->addWidget(restart);
        layout->addLayout(buttons);

        // While a game runs, a restart would close it: Save only, the restart waits.
        if (game_running && game_running()) {
            restart->setEnabled(false);
            restart->setToolTip(QObject::tr("Close the game first"));
            save->setDefault(true);
        } else {
            restart->setDefault(true);
        }

        Load();
        UpdateProblems();
        connect(editor, &QPlainTextEdit::textChanged, this, [this] {
            saved->hide();
            UpdateProblems();
        });
        connect(save, &QPushButton::clicked, this, [this] { Save(); });
        connect(restart, &QPushButton::clicked, this, [this] {
            if (game_running && game_running()) {
                QMessageBox::information(this, windowTitle(), QObject::tr("Close the game first."));
                return;
            }
            if (Save()) {
                wants_restart = true;
                accept();
            }
        });
        connect(open_folder, &QPushButton::clicked, this, [] {
            const fs::path folder = Nx2UserSwitchFile().parent_path();
            std::error_code ec;
            fs::create_directories(folder, ec);
            QDesktopServices::openUrl(QUrl::fromLocalFile(PathText(folder)));
        });
    }

    bool WantsRestart() const {
        return wants_restart;
    }

protected:
    // Esc and the window's close button: unsaved changes are not dropped silently.
    void reject() override {
        if (editor->document()->isModified()) {
            const auto answer = QMessageBox::question(
                this, windowTitle(), QObject::tr("Save your changes?"),
                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
            if (answer == QMessageBox::Cancel || (answer == QMessageBox::Save && !Save())) {
                return;
            }
        }
        QDialog::reject();
    }

private:
    QLabel* Section(const QString& text) {
        auto* label = new QLabel(text, this);
        label->setObjectName(QStringLiteral("FritchiiSwitchesSection"));
        QFont bold = font();
        bold.setBold(true);
        label->setFont(bold);
        return label;
    }

    /// The Fritchii file; while it doesn't exist yet, the old eden one (main.cpp reads that one
    /// then too), which Save copies into the Fritchii folder. With neither, a new empty file, so
    /// Open Folder shows where it lives.
    void Load() {
        fs::path path = Nx2UserSwitchFile();
        std::error_code ec;
        if (!fs::exists(path, ec)) {
            if (const fs::path legacy = Nx2LegacyUserSwitchFile(); fs::exists(legacy, ec)) {
                path = legacy;
            } else {
                fs::create_directories(path.parent_path(), ec);
                QFile created(PathText(path));
                if (created.open(QIODevice::WriteOnly)) {
                    created.close();
                }
            }
        }
        QString text;
        QFile file(PathText(path));
        if (file.open(QIODevice::ReadOnly)) {
            text = QString::fromUtf8(file.readAll());
            text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
        }
        editor->setPlainText(text);
        editor->document()->setModified(false);
    }

    bool Save() {
        const fs::path path = Nx2UserSwitchFile();
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        QString text = editor->toPlainText();
        if (!text.isEmpty() && !text.endsWith(QLatin1Char('\n'))) {
            text += QLatin1Char('\n');
        }
        const QByteArray bytes = text.toUtf8();
        QSaveFile file(PathText(path));
        if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
            !file.commit()) {
            QMessageBox::warning(this, windowTitle(),
                                 QObject::tr("Could not save:\n%1").arg(PathText(path)));
            return false;
        }
        editor->document()->setModified(false);
        saved->show();
        return true;
    }

    void UpdateProblems() {
        constexpr int MaxShown = 3;
        QStringList lines;
        int count = 0;
        const QStringList text = editor->toPlainText().split(QLatin1Char('\n'));
        for (qsizetype i = 0; i < text.size(); ++i) {
            const Problem problem = CheckLine(text[i]);
            if (problem == Problem::None) {
                continue;
            }
            if (++count > MaxShown) {
                continue;
            }
            const auto number = static_cast<int>(i + 1);
            lines << (problem == Problem::NotNameValue
                          ? QObject::tr("Line %1 isn't NAME=VALUE").arg(number)
                          : QObject::tr("Line %1: not a switch name").arg(number));
        }
        if (count > MaxShown) {
            lines << QObject::tr("and %1 more").arg(count - MaxShown);
        }
        problems->setText(lines.join(QLatin1Char('\n')));
        problems->setVisible(count > 0);
    }

    std::function<bool()> game_running;
    QPlainTextEdit* editor = nullptr;
    QLabel* problems = nullptr;
    QLabel* saved = nullptr;
    QPushButton* restart = nullptr;
    bool wants_restart = false;
};

} // namespace

bool ShowSwitchesWindow(QWidget* parent, const std::function<bool()>& game_running) {
    SwitchesDialog dialog(parent, game_running);
    dialog.exec();
    return dialog.WantsRestart();
}

} // namespace FritchiiUi
