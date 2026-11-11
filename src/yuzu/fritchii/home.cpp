// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/home.h"

#include <algorithm>
#include <set>
#include <vector>

#include <QAbstractButton>
#include <QApplication>
#include <QBoxLayout>
#include <QDesktopServices>
#include <QDir>
#include <QEvent>
#include <QFileInfo>
#include <QFrame>
#include <QKeyEvent>
#include <QLabel>
#include <QLocale>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QPainter>
#include <QPainterPath>
#include <QPushButton>
#include <QStandardItemModel>
#include <QTimer>
#include <QTime>
#include <QToolButton>
#include <QUrl>

#include "common/fs/path_util.h"
#include "qt_common/game_list/game_list_p.h"
#include "yuzu/fritchii/fritchii_ui.h"
#include "yuzu/fritchii/saves_window.h"

namespace FritchiiUi {
namespace {

constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
constexpr int MaxShelfGames = 10;
constexpr int SelectedTile = 184;
constexpr int OtherTile = 116;
constexpr int TileMargin = 4; ///< room for the selection ring
const QColor Cream{0xf0, 0xeb, 0xdf};
const QColor CardFill{0x26, 0x2e, 0x3b};
const QColor CardLine{0x33, 0x3c, 0x4b};
const QColor RingHover{0x46, 0x50, 0x5f};
const QColor Muted{0xa9, 0xae, 0xb8};

using Game = HomeGame;

QPixmap Rounded(const QPixmap& source, int size, int radius) {
    const qreal ratio = qApp->devicePixelRatio();
    QPixmap out(QSize(size, size) * ratio);
    out.setDevicePixelRatio(ratio);
    out.fill(Qt::transparent);
    {
        QPainter painter(&out);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setRenderHint(QPainter::SmoothPixmapTransform);
        QPainterPath clip;
        clip.addRoundedRect(QRectF(0, 0, size, size), radius, radius);
        painter.setClipPath(clip);
        if (source.isNull()) {
            painter.fillRect(QRect(0, 0, size, size), QColor(0x24, 0x23, 0x2c));
        } else {
            painter.drawPixmap(QRect(0, 0, size, size), source);
        }
    }
    return out;
}

/// True for a missing or fully transparent icon (some dumps have none).
bool IsBlank(const QPixmap& pixmap) {
    if (pixmap.isNull()) {
        return true;
    }
    const QImage image = pixmap.toImage();
    const int step_x = std::max(1, image.width() / 16);
    const int step_y = std::max(1, image.height() / 16);
    for (int y = 0; y < image.height(); y += step_y) {
        for (int x = 0; x < image.width(); x += step_x) {
            if (qAlpha(image.pixel(x, y)) != 0) {
                return false;
            }
        }
    }
    return true;
}

/// A cover the player put next to the game file: "<file name>.png" or "icon.png" (the same files
/// Game Settings > Detect Icon looks at first). Read only; the game file is never changed.
QPixmap LocalCover(const QString& game_path) {
    const QFileInfo info(game_path);
    const QDir dir = info.absoluteDir();
    for (const QString& base : {info.completeBaseName(), QStringLiteral("icon")}) {
        for (const char* ext : {".png", ".jpg", ".jpeg", ".webp"}) {
            const QFileInfo candidate(dir.filePath(base + QLatin1String(ext)));
            if (!candidate.isFile() || candidate.size() > 16 * 1024 * 1024) {
                continue;
            }
            QPixmap pixmap;
            if (pixmap.load(candidate.filePath())) {
                return pixmap;
            }
        }
    }
    return {};
}

std::vector<Game> CollectGames(QStandardItemModel* model, const QPixmap& fallback) {
    std::vector<Game> games;
    std::set<QString> seen;
    if (!model) {
        return games;
    }
    const auto add = [&](const QStandardItem* item) {
        if (item->data(GameListItem::TypeRole).toInt() !=
            static_cast<int>(GameListItemType::Game)) {
            return;
        }
        const QString path = item->data(GameListItemPath::FullPathRole).toString();
        if (path.isEmpty() || !seen.insert(path).second) {
            return;
        }
        Game game;
        game.path = path;
        game.program_id = item->data(GameListItemPath::ProgramIdRole).toULongLong();
        game.title = item->data(GameListItemPath::TitleRole).toString();
        if (game.title.isEmpty()) {
            game.title = QFileInfo(path).completeBaseName();
        }
        const QByteArray icon_data = item->data(GameListItemPath::IconDataRole).toByteArray();
        if (!game.icon.loadFromData(icon_data)) {
            game.icon = item->data(Qt::DecorationRole).value<QPixmap>();
        }
        if (IsBlank(game.icon)) {
            game.icon = LocalCover(path);
        }
        if (IsBlank(game.icon)) {
            // No icon in the dump and none next to it: Fritchii's own icon, not an empty square.
            game.icon = fallback;
        }
        games.push_back(std::move(game));
    };
    for (int row = 0; row < model->rowCount(); ++row) {
        const QStandardItem* top = model->item(row);
        if (!top) {
            continue;
        }
        if (top->hasChildren()) {
            for (int child = 0; child < top->rowCount(); ++child) {
                add(top->child(child));
            }
        } else {
            add(top);
        }
    }
    return games;
}

/// A round dock icon: the glyph on a filled circle, lighter when hovered.
QIcon DockIcon(const QString& name) {
    constexpr int Size = 46;
    const qreal ratio = qApp->devicePixelRatio();
    const QPixmap glyph = Glyph(name, 0xf0ebdf, 20).pixmap(QSize(20, 20) * ratio);
    const auto make = [&](const QColor& fill) {
        QPixmap out(QSize(Size, Size) * ratio);
        out.setDevicePixelRatio(ratio);
        out.fill(Qt::transparent);
        QPainter painter(&out);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(Qt::NoPen);
        painter.setBrush(fill);
        painter.drawEllipse(QRectF(0, 0, Size, Size));
        painter.drawPixmap(QRectF((Size - 20) / 2.0, (Size - 20) / 2.0, 20, 20), glyph,
                           QRectF(QPointF(0, 0), glyph.size()));
        return out;
    };
    QIcon icon;
    icon.addPixmap(make(CardFill), QIcon::Normal);
    icon.addPixmap(make(CardLine), QIcon::Active);
    return icon;
}

QString PlayTimeText(u64 seconds) {
    if (seconds == 0) {
        return QObject::tr("Not played yet");
    }
    const u64 hours = seconds / 3600;
    const u64 minutes = (seconds / 60) % 60;
    if (hours == 0) {
        return QObject::tr("Played %1 min").arg(std::max<u64>(minutes, 1));
    }
    return minutes == 0 ? QObject::tr("Played %1 h").arg(hours)
                        : QObject::tr("Played %1 h %2 min").arg(hours).arg(minutes);
}

} // namespace

/// One tile of the shelf: a game's cover, or a card with a glyph and a label ("All games",
/// "Add folder"). The selected tile is larger and ringed in cream.
class ShelfTile : public QAbstractButton {
public:
    ShelfTile(QWidget* parent, QPixmap art, QString glyph, QString label)
        : QAbstractButton(parent), art{std::move(art)}, glyph{std::move(glyph)},
          label{std::move(label)} {
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::NoFocus);
        setToolTip(this->label);
        SetSelected(false);
    }

    void SetSelected(bool value) {
        selected = value;
        const int side = (selected ? SelectedTile : OtherTile) + TileMargin * 2;
        setFixedSize(side, side);
        update();
    }

    std::function<void()> on_double_click;

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setRenderHint(QPainter::SmoothPixmapTransform);
        const QRectF card = QRectF(rect()).adjusted(TileMargin, TileMargin, -TileMargin,
                                                    -TileMargin);
        const qreal radius = selected ? 24 : 18;
        QPainterPath shape;
        shape.addRoundedRect(card, radius, radius);
        if (!art.isNull()) {
            painter.save();
            painter.setClipPath(shape);
            painter.drawPixmap(card.toRect(), art);
            painter.restore();
        } else {
            painter.setPen(QPen(CardLine, 1));
            painter.setBrush(CardFill);
            painter.drawPath(shape);
            const int glyph_px = selected ? 44 : 30;
            const QPixmap icon = Glyph(glyph, selected ? 0xf0ebdf : 0xa9aeb8, glyph_px)
                                     .pixmap(QSize(glyph_px, glyph_px));
            const QPointF center = card.center();
            painter.drawPixmap(QPointF(center.x() - glyph_px / 2.0,
                                       center.y() - glyph_px / 2.0 - 10),
                               icon);
            QFont font = painter.font();
            font.setPixelSize(selected ? 15 : 12);
            painter.setFont(font);
            painter.setPen(selected ? Cream : Muted);
            painter.drawText(QRectF(card.left() + 6, center.y() + glyph_px / 2.0 - 4,
                                    card.width() - 12, 24),
                             Qt::AlignHCenter | Qt::AlignTop, label);
        }
        if (selected || underMouse()) {
            const qreal width = selected ? 3.0 : 2.0;
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QPen(selected ? Cream : RingHover, width));
            const qreal inset = width / 2.0;
            painter.drawRoundedRect(QRectF(rect()).adjusted(inset, inset, -inset, -inset),
                                    radius + TileMargin, radius + TileMargin);
        }
    }

    void enterEvent(QEnterEvent* event) override {
        update();
        QAbstractButton::enterEvent(event);
    }

    void leaveEvent(QEvent* event) override {
        update();
        QAbstractButton::leaveEvent(event);
    }

    void mouseDoubleClickEvent(QMouseEvent* event) override {
        if (on_double_click) {
            on_double_click();
        }
        QAbstractButton::mouseDoubleClickEvent(event);
    }

private:
    QPixmap art;
    QString glyph;
    QString label;
    bool selected = false;
};

Home::Home(QMainWindow* window_, QBoxLayout* central, QWidget* game_list_, QWidget* placeholder_,
           HomeHooks hooks_)
    : QObject(window_), window{window_}, game_list{game_list_}, placeholder{placeholder_},
      hooks{std::move(hooks_)} {
    // The classic menus live behind the Menu button. A zero-height bar keeps every shortcut
    // working and stays out of the way when fullscreen shows the bar again.
    window->menuBar()->setFixedHeight(0);

    rail = MakeRail();
    page = MakePage();
    central->insertWidget(0, rail);
    central->addWidget(page, 1);
    rail->hide();

    game_list->installEventFilter(this);
    placeholder->installEventFilter(this);
    page->installEventFilter(this);

    refresh_timer = new QTimer(this);
    refresh_timer->setSingleShot(true);
    refresh_timer->setInterval(150);
    connect(refresh_timer, &QTimer::timeout, this, &Home::Refresh);
    if (hooks.model) {
        const auto later = [this] { refresh_timer->start(); };
        connect(hooks.model, &QAbstractItemModel::rowsInserted, this, later);
        connect(hooks.model, &QAbstractItemModel::rowsRemoved, this, later);
        connect(hooks.model, &QAbstractItemModel::modelReset, this, later);
        connect(hooks.model, &QAbstractItemModel::dataChanged, this, later);
    }
    clock_timer = new QTimer(this);
    clock_timer->setInterval(15000);
    connect(clock_timer, &QTimer::timeout, this, &Home::UpdateClock);
    clock_timer->start();
    UpdateClock();
    QTimer::singleShot(0, this, &Home::Restore);
}

QToolButton* Home::RailButton(const QString& icon, const QString& tip) {
    auto* button = new QToolButton(rail);
    button->setObjectName(QStringLiteral("FritchiiRailButton"));
    button->setIcon(Icon(icon, 22));
    button->setIconSize(QSize(22, 22));
    button->setToolTip(tip);
    button->setCursor(Qt::PointingHandCursor);
    button->setFocusPolicy(Qt::NoFocus);
    return button;
}

QWidget* Home::MakeRail() {
    // Only beside the full game list: the way back home, and the settings and menus.
    rail = new QWidget;
    rail->setObjectName(QStringLiteral("FritchiiRail"));
    rail->setAttribute(Qt::WA_StyledBackground);
    rail->setFixedWidth(64);
    auto* layout = new QVBoxLayout(rail);
    layout->setContentsMargins(11, 16, 11, 16);
    layout->setSpacing(8);

    auto* mark = new QLabel(rail);
    mark->setPixmap(Logo(42));
    mark->setAlignment(Qt::AlignHCenter);
    mark->setToolTip(QStringLiteral("Kizuna"));
    layout->addWidget(mark);
    layout->addSpacing(14);

    QToolButton* home = RailButton(QStringLiteral("library"), tr("Home"));
    connect(home, &QToolButton::clicked, this, &Home::ShowHome);
    QToolButton* settings = RailButton(QStringLiteral("settings"), tr("Settings"));
    connect(settings, &QToolButton::clicked, this, [this] { hooks.open_settings(); });
    QToolButton* help = RailButton(QStringLiteral("help"), tr("Help"));
    connect(help, &QToolButton::clicked, this, [this, help] {
        if (hooks.help_menu) {
            hooks.help_menu->exec(help->mapToGlobal(QPoint(help->width() + 8, 0)));
        }
    });
    QToolButton* menu = RailButton(QStringLiteral("menu"), tr("Menu"));
    connect(menu, &QToolButton::clicked, this, [this, menu] {
        ShowClassicMenu(menu->mapToGlobal(QPoint(menu->width() + 8, 0)));
    });
    layout->addWidget(home);
    layout->addWidget(settings);
    layout->addWidget(help);
    layout->addStretch(1);
    layout->addWidget(menu);
    return rail;
}

QToolButton* Home::DockButton(const QString& icon, const QString& text) {
    auto* button = new QToolButton(page);
    button->setObjectName(QStringLiteral("KizunaDock"));
    button->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    button->setIcon(DockIcon(icon));
    button->setIconSize(QSize(46, 46));
    button->setText(text);
    button->setCursor(Qt::PointingHandCursor);
    button->setFocusPolicy(Qt::NoFocus);
    return button;
}

QWidget* Home::MakePage() {
    page = new QWidget;
    page->setObjectName(QStringLiteral("FritchiiHome"));
    page->setAttribute(Qt::WA_StyledBackground);
    page->setFocusPolicy(Qt::StrongFocus);
    page->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(page, &QWidget::customContextMenuRequested, this,
            [this](const QPoint& pos) { ShowGameMenu(page->mapToGlobal(pos)); });

    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(52, 26, 52, 22);
    layout->setSpacing(0);

    // Top bar: logo and name, the clock.
    auto* top = new QHBoxLayout;
    top->setSpacing(12);
    auto* mark = new QLabel(page);
    mark->setPixmap(Logo(46));
    top->addWidget(mark, 0, Qt::AlignVCenter);
    auto* wordmark = new QLabel(QStringLiteral("Kizuna"), page);
    wordmark->setObjectName(QStringLiteral("KizunaWordmark"));
    top->addWidget(wordmark, 0, Qt::AlignVCenter);
    top->addStretch(1);
    clock = new QLabel(page);
    clock->setObjectName(QStringLiteral("KizunaClock"));
    top->addWidget(clock, 0, Qt::AlignVCenter);
    layout->addLayout(top);
    layout->addStretch(1);

    // The shelf.
    shelf_box = new QWidget(page);
    shelf = new QHBoxLayout(shelf_box);
    shelf->setContentsMargins(0, 0, 0, 0);
    shelf->setSpacing(14);
    shelf->setAlignment(Qt::AlignLeft | Qt::AlignBottom);
    shelf_box->setMinimumHeight(SelectedTile + TileMargin * 2);
    layout->addWidget(shelf_box);
    layout->addSpacing(22);

    // The selected game: name, play time, Play and its options.
    title = new QLabel(page);
    title->setObjectName(QStringLiteral("FritchiiHeroTitle"));
    title->setWordWrap(true);
    title->setMaximumWidth(760);
    layout->addWidget(title);
    layout->addSpacing(6);
    subtitle = new QLabel(page);
    subtitle->setObjectName(QStringLiteral("KizunaSubtitle"));
    layout->addWidget(subtitle);
    layout->addSpacing(20);
    auto* actions = new QHBoxLayout;
    actions->setSpacing(10);
    play = new QPushButton(page);
    play->setObjectName(QStringLiteral("FritchiiPlay"));
    play->setCursor(Qt::PointingHandCursor);
    play->setFocusPolicy(Qt::NoFocus);
    play->setIconSize(QSize(18, 18));
    connect(play, &QPushButton::clicked, this, [this] { Activate(focus); });
    actions->addWidget(play);
    options = new QPushButton(page);
    options->setObjectName(QStringLiteral("KizunaRound"));
    options->setIcon(Glyph(QStringLiteral("dots"), 0xf0ebdf, 20));
    options->setIconSize(QSize(20, 20));
    options->setToolTip(tr("Options"));
    options->setCursor(Qt::PointingHandCursor);
    options->setFocusPolicy(Qt::NoFocus);
    connect(options, &QPushButton::clicked, this, [this] {
        ShowGameMenu(options->mapToGlobal(QPoint(0, options->height() + 6)));
    });
    actions->addWidget(options);
    actions->addStretch(1);
    layout->addLayout(actions);
    layout->addStretch(2);

    // The dock and the keyboard hints.
    auto* line = new QFrame(page);
    line->setObjectName(QStringLiteral("KizunaDockLine"));
    line->setFrameShape(QFrame::NoFrame);
    layout->addWidget(line);
    layout->addSpacing(12);
    auto* dock = new QHBoxLayout;
    dock->setSpacing(10);
    QToolButton* library = DockButton(QStringLiteral("library"), tr("Library"));
    connect(library, &QToolButton::clicked, this, &Home::ShowList);
    QToolButton* saves = DockButton(QStringLiteral("save"), tr("Saves"));
    connect(saves, &QToolButton::clicked, this,
            [this] { ShowSavesWindow(window, hooks.emulation_running); });
    QToolButton* shots = DockButton(QStringLiteral("photo"), tr("Screenshots"));
    connect(shots, &QToolButton::clicked, this, [] {
        const QString dir = QString::fromStdString(
            Common::FS::GetEdenPathString(Common::FS::EdenPath::ScreenshotsDir));
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    });
    QToolButton* settings = DockButton(QStringLiteral("settings"), tr("Settings"));
    connect(settings, &QToolButton::clicked, this, [this] { hooks.open_settings(); });
    QToolButton* help = DockButton(QStringLiteral("help"), tr("Help"));
    connect(help, &QToolButton::clicked, this, [this, help] {
        if (hooks.help_menu) {
            const QSize size = hooks.help_menu->sizeHint();
            hooks.help_menu->exec(help->mapToGlobal(QPoint(0, -size.height() - 6)));
        }
    });
    menu_button = DockButton(QStringLiteral("menu"), tr("Menu"));
    connect(menu_button, &QToolButton::clicked, this, [this] {
        ShowClassicMenu(menu_button->mapToGlobal(QPoint(0, 0)));
    });
    for (QToolButton* button : {library, saves, shots, settings, help, menu_button}) {
        dock->addWidget(button);
    }
    dock->addStretch(1);
    auto* hints = new QLabel(page);
    hints->setObjectName(QStringLiteral("KizunaHints"));
    hints->setTextFormat(Qt::RichText);
    const auto hint = [](const QString& key, const QString& what) {
        return QStringLiteral("<span style='color:#f0ebdf;font-weight:500'>%1</span>&nbsp;%2")
            .arg(key, what);
    };
    hints->setText(hint(tr("Enter"), tr("Play")) + QStringLiteral("&nbsp;&nbsp;&nbsp;&nbsp;") +
                   hint(QStringLiteral("← →"), tr("Choose")) +
                   QStringLiteral("&nbsp;&nbsp;&nbsp;&nbsp;") + hint(tr("Esc"), tr("Menu")));
    dock->addWidget(hints, 0, Qt::AlignVCenter);
    layout->addLayout(dock);
    return page;
}

void Home::UpdateClock() {
    if (clock) {
        clock->setText(QLocale().toString(QTime::currentTime(), QLocale::ShortFormat));
    }
}

void Home::Refresh() {
    std::vector<Game> found = CollectGames(hooks.model, window->windowIcon().pixmap(256));
    if (found.size() > MaxShelfGames) {
        // Keep the one picked last on the shelf.
        const auto picked = std::ranges::find_if(found, [this](const Game& game) {
            return !hero_path.isEmpty() && game.path == hero_path;
        });
        if (picked != found.end() && picked - found.begin() >= MaxShelfGames) {
            std::iter_swap(found.begin() + (MaxShelfGames - 1), picked);
        }
        found.resize(MaxShelfGames);
    }
    const size_t game_count = found.size();
    games = std::move(found);
    if (game_count != 0) {
        games.push_back(Game{.title = tr("All games"), .path = {}, .program_id = 0, .icon = {}});
    }
    games.push_back(Game{.title = tr("Add folder"), .path = {}, .program_id = 0, .icon = {}});

    // The game picked last, else Ocarina of Time, else the first one.
    int selected = 0;
    bool matched = false;
    for (size_t i = 0; i < game_count; ++i) {
        if (!hero_path.isEmpty() && games[i].path == hero_path) {
            selected = static_cast<int>(i);
            matched = true;
            break;
        }
    }
    if (!matched) {
        for (size_t i = 0; i < game_count; ++i) {
            if (games[i].program_id == OcarinaOfTime) {
                selected = static_cast<int>(i);
                break;
            }
        }
    }

    // deleteLater: a tile's own click can land here.
    tiles.clear();
    while (QLayoutItem* item = shelf->takeAt(0)) {
        if (QWidget* widget = item->widget()) {
            widget->hide();
            widget->deleteLater();
        }
        delete item;
    }
    for (size_t i = 0; i < games.size(); ++i) {
        const bool is_game = i < game_count;
        const bool is_all = !is_game && game_count != 0 && i == game_count;
        auto* tile = new ShelfTile(
            shelf_box, is_game ? Rounded(games[i].icon, SelectedTile, 24) : QPixmap{},
            is_game ? QString{} : (is_all ? QStringLiteral("list") : QStringLiteral("plus")),
            games[i].title);
        const int index = static_cast<int>(i);
        connect(tile, &QAbstractButton::clicked, this, [this, index] {
            // A click picks the tile; a click on the picked one opens it.
            if (index == focus) {
                Activate(index);
            } else {
                Select(index);
            }
        });
        tile->on_double_click = [this, index] { Activate(index); };
        shelf->addWidget(tile, 0, Qt::AlignBottom);
        tiles.push_back(tile);
    }
    Select(selected);
}

void Home::Select(int index) {
    if (games.empty()) {
        return;
    }
    focus = std::clamp(index, 0, static_cast<int>(games.size()) - 1);
    for (size_t i = 0; i < tiles.size(); ++i) {
        tiles[i]->SetSelected(static_cast<int>(i) == focus);
    }
    const Game& game = games[focus];
    const bool is_game = !game.path.isEmpty();
    options->setVisible(is_game);
    if (is_game) {
        hero_path = game.path;
        hero_id = game.program_id;
        title->setText(game.title);
        subtitle->setText(PlayTimeText(hooks.play_time ? hooks.play_time(game.program_id) : 0));
        play->setText(tr("Play"));
        play->setIcon(Glyph(QStringLiteral("play"), 0x1a1f28, 18));
    } else if (game.title == tr("All games")) {
        title->setText(tr("All games"));
        subtitle->setText(tr("The full list, with search, folders and details"));
        play->setText(tr("Open"));
        play->setIcon(Glyph(QStringLiteral("list"), 0x1a1f28, 18));
    } else {
        title->setText(games.size() == 1 ? tr("Add your games") : tr("Add folder"));
        subtitle->setText(tr("Pick the folder where your games are"));
        play->setText(tr("Add Folder"));
        play->setIcon(Glyph(QStringLiteral("plus"), 0x1a1f28, 18));
    }
}

void Home::Activate(int index) {
    if (index < 0 || index >= static_cast<int>(games.size())) {
        return;
    }
    const Game& game = games[index];
    if (!game.path.isEmpty()) {
        hooks.play(game.path, game.program_id);
    } else if (game.title == tr("All games")) {
        ShowList();
    } else {
        hooks.add_folder();
    }
}

void Home::ShowGameMenu(const QPoint& global_pos) {
    if (hero_path.isEmpty() || focus >= static_cast<int>(games.size()) ||
        games[focus].path.isEmpty()) {
        return;
    }
    // The game list's full menu (modders use its folders and dumps), then Delete Shader Cache.
    QMenu menu;
    if (hooks.game_menu) {
        hooks.game_menu(menu, hero_id, hero_path.toStdString());
    } else {
        menu.addAction(tr("Game Settings..."), &menu,
                       [this] { hooks.game_settings(hero_id, hero_path.toStdString()); });
    }
    menu.addSeparator();
    menu.addAction(tr("Delete Shader Cache..."), &menu,
                   [this] { hooks.delete_shader_cache(hero_id); });
    menu.exec(global_pos);
}

void Home::ShowClassicMenu(const QPoint& global_pos) {
    QMenu classic;
    for (QMenu* menu : hooks.classic_menus) {
        classic.addMenu(menu);
    }
    const QSize size = classic.sizeHint();
    // Above the dock when opened from there.
    QPoint pos = global_pos;
    if (page && page->isVisible() && menu_button &&
        global_pos == menu_button->mapToGlobal(QPoint(0, 0))) {
        pos.ry() -= size.height() + 6;
    }
    classic.exec(pos);
}

void Home::ShowHome() {
    list_mode = false;
    game_list->hide();
    placeholder->hide();
    rail->hide();
    page->show();
    Refresh();
    page->setFocus();
}

void Home::ShowList() {
    list_mode = true;
    page->hide();
    rail->show();
    if (hooks.game_list_empty()) {
        placeholder->show();
    } else {
        game_list->show();
    }
}

void Home::Restore() {
    // A restart shows the list for a moment, and nothing may cover a running game.
    if (hooks.single_window() && hooks.emulation_running()) {
        return;
    }
    if (list_mode) {
        rail->show();
        page->hide();
    } else {
        ShowHome();
    }
}

void Home::OnEmulationStarting() {
    if (!hooks.single_window()) {
        return;
    }
    rail->hide();
    page->hide();
}

bool Home::eventFilter(QObject* watched, QEvent* event) {
    // The main window shows its game list (start up, a game stopped, the list filled): show the
    // home page instead, unless the player asked for the full list.
    if (event->type() == QEvent::Show && (watched == game_list || watched == placeholder)) {
        QTimer::singleShot(0, this, &Home::Restore);
    }
    if (watched == page && event->type() == QEvent::KeyPress) {
        const auto* key = static_cast<QKeyEvent*>(event);
        switch (key->key()) {
        case Qt::Key_Left:
            Select(focus - 1);
            return true;
        case Qt::Key_Right:
            Select(focus + 1);
            return true;
        case Qt::Key_Home:
            Select(0);
            return true;
        case Qt::Key_End:
            Select(static_cast<int>(games.size()) - 1);
            return true;
        case Qt::Key_Return:
        case Qt::Key_Enter:
        case Qt::Key_Space:
            Activate(focus);
            return true;
        case Qt::Key_Escape:
            ShowClassicMenu(menu_button->mapToGlobal(QPoint(0, 0)));
            return true;
        case Qt::Key_Menu:
            ShowGameMenu(options->mapToGlobal(QPoint(0, options->height() + 6)));
            return true;
        default:
            break;
        }
    }
    return QObject::eventFilter(watched, event);
}

} // namespace FritchiiUi
