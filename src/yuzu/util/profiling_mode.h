// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <QWidget>

class QTimer;

// Profiling mode (Help > Profiling Mode, docs/fritchii/switches.md): for testers who send us a
// log. While it is on, Fritchii starts with the profiling switches and a red banner sits on the
// game so nobody forgets to turn it off. The switches are read once per process, so turning it on
// or off takes a restart of Fritchii.
namespace ProfilingMode {

/// The switches profiling mode adds, name and value. A value the environment or an
/// nx2_switches.txt already sets wins.
[[nodiscard]] const std::vector<std::pair<std::string, std::string>>& Switches();

/// main(), before the switch files are applied: keeps the launch environment for a restart and
/// reads the saved setting straight from qt-config.ini (the settings load much later).
void ReadAtLaunch();
/// The saved setting as ReadAtLaunch found it.
[[nodiscard]] bool OnAtLaunch();

/// main(): the switches profiling mode really set in this process ("A=1, B=2"); empty = none.
void SetApplied(std::string applied);
/// This process runs with profiling mode (the setting was on when it started).
[[nodiscard]] bool Active();
[[nodiscard]] const std::string& Applied();

/// Closes the window and starts Fritchii again once this one has quit, with the environment it
/// was launched with (not the switches this process added). `closing` says, right after the
/// close request, whether the window really closes (false: the player kept it open, no restart).
void RestartFritchii(QWidget& main_window, const std::function<bool()>& closing);

/// The red "PROFILING MODE ON" banner, top centre of the game image, in a window and in
/// fullscreen. A frameless tool window like the in-game controls (fritchii/game_controls), so it
/// shows above the Vulkan surface; it follows the game window and ignores the mouse.
class Banner : public QWidget {
public:
    Banner(QWidget* owner, std::function<QWidget*()> render);

    void Start();
    void Stop();

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    void Tick();

    std::function<QWidget*()> render;
    QTimer* timer = nullptr;
    bool short_text = false;
};

} // namespace ProfilingMode
