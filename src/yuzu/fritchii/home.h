// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <vector>
#include <QPixmap>
#include <QWidget>
#include "common/common_types.h"

class QBoxLayout;
class QHBoxLayout;
class QLabel;
class QMainWindow;
class QMenu;
class QPushButton;
class QStandardItemModel;
class QTimer;
class QToolButton;

namespace FritchiiUi {

class ShelfTile;

/// A game on the home shelf.
struct HomeGame {
    QString title;
    QString path;
    u64 program_id = 0;
    QPixmap icon;
};

/// What the home screen asks the main window to do.
struct HomeHooks {
    std::function<void(const QString& path, u64 program_id)> play;
    std::function<void()> add_folder;
    std::function<void()> open_settings;
    std::function<void(u64 program_id, const std::string& path)> game_settings;
    std::function<void(u64 program_id)> delete_shader_cache; ///< after a confirmation
    /// Fills a menu with the game list's own right-click entries for the game (mod folders,
    /// dumps, Remove, Verify Integrity, Configure Game...).
    std::function<void(QMenu& menu, u64 program_id, const std::string& path)> game_menu;
    std::function<bool()> game_list_empty;
    std::function<bool()> single_window;
    std::function<bool()> emulation_running;
    /// Kizuna: seconds played (PlayTimeManager), 0 when never played.
    std::function<u64(u64 program_id)> play_time;
    QStandardItemModel* model = nullptr;
    QMenu* help_menu = nullptr;
    QList<QMenu*> classic_menus;  ///< File, Game, View, Tools, Help behind the Menu button
};

/// Kizuna's main screen, laid out like a console's home: a top bar (logo, name, clock), a shelf
/// of game tiles with the selected one larger and ringed in cream, the selected game's name, play
/// time and Play / options buttons, and a dock of round buttons (Library, Saves, Screenshots,
/// Settings, Help, Menu) with keyboard hints. Left/Right choose, Enter plays, Esc opens the menu.
/// The classic game list stays one click away (Library), with a slim rail to come back; the main
/// window keeps managing it as before.
class Home : public QObject {
    Q_OBJECT
public:
    Home(QMainWindow* window, QBoxLayout* central, QWidget* game_list, QWidget* placeholder,
         HomeHooks hooks);

    /// A game is starting in the main window: the render view takes the space.
    void OnEmulationStarting();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void ShowHome();
    void ShowList();
    void Restore();
    void Refresh();
    void Select(int index);
    void Activate(int index);
    void ShowGameMenu(const QPoint& global_pos);
    void ShowClassicMenu(const QPoint& global_pos);
    void UpdateClock();

    QWidget* MakeRail();
    QWidget* MakePage();
    QToolButton* RailButton(const QString& icon, const QString& tip);
    QToolButton* DockButton(const QString& icon, const QString& text);

    QMainWindow* window;
    QWidget* game_list;
    QWidget* placeholder;
    HomeHooks hooks;

    QWidget* rail = nullptr;
    QWidget* page = nullptr;
    QLabel* clock = nullptr;
    QLabel* title = nullptr;
    QLabel* subtitle = nullptr;
    QPushButton* play = nullptr;
    QPushButton* options = nullptr;
    QHBoxLayout* shelf = nullptr;
    QWidget* shelf_box = nullptr;
    QToolButton* menu_button = nullptr;
    QTimer* refresh_timer = nullptr;
    QTimer* clock_timer = nullptr;

    /// The shelf: the games, then "All games" and "Add folder" (no program id, empty path).
    std::vector<HomeGame> games;
    std::vector<ShelfTile*> tiles;
    int focus = 0;

    bool list_mode = false;
    QString hero_path;
    u64 hero_id = 0;
};

} // namespace FritchiiUi
