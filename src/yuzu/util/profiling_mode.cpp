// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/util/profiling_mode.h"

#include <algorithm>
#include <fstream>
#include <string_view>

#include <QApplication>
#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTimer>

#include "common/fs/path_util.h"

namespace ProfilingMode {
namespace {

constexpr Qt::WindowFlags BannerFlags = Qt::Tool | Qt::FramelessWindowHint |
                                        Qt::WindowDoesNotAcceptFocus |
                                        Qt::NoDropShadowWindowHint |
                                        Qt::WindowTransparentForInput;
constexpr int FontPx = 22;
constexpr int PadX = 22;
constexpr int PadY = 10;
constexpr int Gap = 18;
constexpr int TopMargin = 12;

const QColor Fill{0xd3, 0x1f, 0x2f, 240};
const QColor Edge{0x7a, 0x0b, 0x16};
const QColor Text{0xff, 0xff, 0xff};

QProcessEnvironment& LaunchEnvironment() {
    static QProcessEnvironment environment;
    return environment;
}

bool& OnFlag() {
    static bool on = false;
    return on;
}

std::string& AppliedText() {
    static std::string text;
    return text;
}

std::string_view Trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() &&
           (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
        text.remove_suffix(1);
    }
    return text;
}

/// [UI] profiling_mode in qt-config.ini, as the settings write it (QtConfig: section "UI", key
/// "profiling_mode", value "true" or "false"). Missing file or key = off.
bool ReadSavedSetting() {
    const auto path =
        Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "qt-config.ini";
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::string line;
    bool in_ui = false;
    while (std::getline(file, line)) {
        const std::string_view text = Trim(line);
        if (text.starts_with('[')) {
            in_ui = text == "[UI]";
            continue;
        }
        if (!in_ui) {
            continue;
        }
        const auto eq = text.find('=');
        if (eq == std::string_view::npos || Trim(text.substr(0, eq)) != "profiling_mode") {
            continue;
        }
        const std::string_view value = Trim(text.substr(eq + 1));
        return value == "true" || value == "1";
    }
    return false;
}

QFont BannerFont(bool bold) {
    QFont font = QApplication::font();
    font.setPixelSize(FontPx);
    font.setWeight(bold ? QFont::Bold : QFont::Medium);
    return font;
}

QString HeadText() {
    return QCoreApplication::translate("ProfilingMode", "PROFILING MODE ON");
}

QString TailText() {
    return QCoreApplication::translate("ProfilingMode", "Turn it off in Help");
}

} // namespace

const std::vector<std::pair<std::string, std::string>>& Switches() {
    // Each one is cheap enough to leave on while playing (docs/fritchii/switches.md, "Profiling
    // mode"): exact counters and timers, a few log lines every 10 s.
    static const std::vector<std::pair<std::string, std::string>> switches{
        // Where the GPU threads spend their time (zones, every 10 s).
        {"EDEN_NX2_GPU_PROFILE", "1"},
        // Who holds the RAM, every 30 s and at new peaks. 3 = committed only: 1 also queries the
        // working set, which holds its lock 150-270 ms (a stutter every 30 s).
        {"FRITCHII_MEM_CENSUS", "3"},
        // Hitch lines with the recompiler, upload and GPU thread causes.
        {"FRITCHII_HITCH_SPLIT", "1"},
        // A Hitch line for every gap over 40 ms (default 50). Not 33: at 30 FPS every frame is
        // 33.3 ms and each one would log.
        {"EDEN_NX2_HITCH_MS", "40"},
        // VRAM use against the card's budget (the 8 GB card cliff).
        {"FRITCHII_VRAM_REPORT", "1"},
        // Image allocation and destroy times (NPC dips, area loads).
        {"EDEN_NX2_IMAGE_ALLOC_TIMING", "1"},
        // Texture cache memory every 600 frames.
        {"EDEN_NX2_TC_LOG", "1"},
    };
    return switches;
}

void ReadAtLaunch() {
    LaunchEnvironment() = QProcessEnvironment::systemEnvironment();
    OnFlag() = ReadSavedSetting();
}

bool OnAtLaunch() {
    return OnFlag();
}

void SetApplied(std::string applied) {
    AppliedText() = std::move(applied);
}

bool Active() {
    return OnFlag();
}

const std::string& Applied() {
    return AppliedText();
}

void RestartFritchii(QWidget& main_window, const std::function<bool()>& closing) {
    const auto connection =
        QObject::connect(qApp, &QCoreApplication::aboutToQuit, qApp, [] {
            QProcess process;
            process.setProgram(QCoreApplication::applicationFilePath());
            if (!LaunchEnvironment().isEmpty()) {
                process.setProcessEnvironment(LaunchEnvironment());
            }
            process.startDetached();
        });
    main_window.close();
    if (closing && !closing()) {
        QObject::disconnect(connection);
    }
}

Banner::Banner(QWidget* owner, std::function<QWidget*()> render_)
    : QWidget(owner, BannerFlags), render{std::move(render_)} {
    setObjectName(QStringLiteral("ProfilingBanner"));
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);
    setAttribute(Qt::WA_TransparentForMouseEvents);
    timer = new QTimer(this);
    timer->setInterval(250);
    connect(timer, &QTimer::timeout, this, [this] { Tick(); });
}

void Banner::Start() {
    timer->start();
}

void Banner::Stop() {
    timer->stop();
    hide();
}

void Banner::Tick() {
    QWidget* game = render ? render() : nullptr;
    const bool usable = game && game->isVisible() && !game->window()->isMinimized() &&
                        QApplication::applicationState() == Qt::ApplicationActive &&
                        QApplication::activeModalWidget() == nullptr;
    if (!usable) {
        hide();
        return;
    }
    // A separate game window (single window mode off) owns the banner, so it stays above it.
    if (parentWidget() != game->window()) {
        setParent(game->window(), BannerFlags);
    }

    const QRect area(game->mapToGlobal(QPoint(0, 0)), game->size());
    const int head = QFontMetrics(BannerFont(true)).horizontalAdvance(HeadText());
    const int tail = QFontMetrics(BannerFont(false)).horizontalAdvance(TailText());
    const int full_width = PadX * 2 + head + Gap + tail;
    const bool fits = full_width <= area.width() - 2 * TopMargin;
    const QSize wanted(fits ? full_width : PadX * 2 + head,
                       QFontMetrics(BannerFont(true)).height() + PadY * 2);
    if (short_text == fits || size() != wanted) {
        short_text = !fits;
        setFixedSize(wanted);
        update();
    }

    // Below the in-game controls (fritchii/game_controls) while they show.
    int top = area.top() + TopMargin;
    if (const auto* strip = game->window()->findChild<QWidget*>(QStringLiteral("FritchiiStrip"),
                                                                Qt::FindDirectChildrenOnly);
        strip && strip->isVisible()) {
        top = std::max(top, strip->frameGeometry().bottom() + 10);
    }
    const QPoint position(area.left() + (area.width() - wanted.width()) / 2, top);
    if (pos() != position) {
        move(position);
    }
    if (!isVisible()) {
        show();
    }
    raise();
}

void Banner::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Edge);
    painter.setBrush(Fill);
    painter.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10);

    painter.setPen(Text);
    const QFont bold = BannerFont(true);
    painter.setFont(bold);
    const QRect inner = rect().adjusted(PadX, 0, -PadX, 0);
    painter.drawText(inner, Qt::AlignLeft | Qt::AlignVCenter, HeadText());
    if (!short_text) {
        const int head = QFontMetrics(bold).horizontalAdvance(HeadText());
        painter.setFont(BannerFont(false));
        painter.drawText(inner.adjusted(head + Gap, 0, 0, 0), Qt::AlignLeft | Qt::AlignVCenter,
                         TailText());
    }
}

} // namespace ProfilingMode
