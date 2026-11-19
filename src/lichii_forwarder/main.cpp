// SPDX-License-Identifier: GPL-3.0-or-later

// lichii.exe: starts fritchii.exe from the same folder with the same arguments and working folder,
// waits for it, and returns its exit code (launchers such as Steam watch the process they start).
// Fritchii v5.x updaters require lichii.exe in the package and start it after updating; old
// shortcuts, file associations and launchers point to it.

#include <string>

#include <windows.h>

namespace {

/// The full path of this exe.
std::wstring OwnPath() {
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (length == 0) {
            return {};
        }
        if (length < path.size()) {
            path.resize(length);
            return path;
        }
        path.resize(path.size() * 2);
    }
}

/// The command line after the program name (as Windows splits it: a quoted name ends at the next
/// quote, an unquoted one at the first space or tab), without the spaces before the first argument.
const wchar_t* ArgumentsAfterProgram(const wchar_t* line) {
    if (line == nullptr) {
        return L"";
    }
    if (*line == L'"') {
        ++line;
        while (*line != L'\0' && *line != L'"') {
            ++line;
        }
        if (*line == L'"') {
            ++line;
        }
    } else {
        while (*line != L'\0' && *line != L' ' && *line != L'\t') {
            ++line;
        }
    }
    while (*line == L' ' || *line == L'\t') {
        ++line;
    }
    return line;
}

int Fail(const wchar_t* text) {
    MessageBoxW(nullptr, text, L"Fritchii", MB_OK | MB_ICONERROR);
    return 1;
}

} // Anonymous namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    const std::wstring own = OwnPath();
    const size_t slash = own.find_last_of(L"\\/");
    if (own.empty() || slash == std::wstring::npos) {
        return Fail(L"Fritchii could not find its own folder. Start fritchii.exe instead.");
    }
    const std::wstring exe = own.substr(0, slash + 1) + L"fritchii.exe";
    if (GetFileAttributesW(exe.c_str()) == INVALID_FILE_ATTRIBUTES) {
        return Fail(L"fritchii.exe is missing next to lichii.exe.\n\nYour antivirus may have "
                    L"removed it, or the folder is incomplete: unzip the whole Fritchii folder "
                    L"again.");
    }

    std::wstring command = L"\"" + exe + L"\"";
    const wchar_t* const arguments = ArgumentsAfterProgram(GetCommandLineW());
    if (*arguments != L'\0') {
        command += L' ';
        command += arguments;
    }

    // Only the window state a shortcut asked for (minimized, maximized) is passed on: the rest of
    // this process's startup info (standard handles, ...) doesn't apply to the child.
    STARTUPINFOW own_startup{};
    own_startup.cb = sizeof(own_startup);
    GetStartupInfoW(&own_startup);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    if ((own_startup.dwFlags & STARTF_USESHOWWINDOW) != 0) {
        startup.dwFlags = STARTF_USESHOWWINDOW;
        startup.wShowWindow = own_startup.wShowWindow;
    }
    PROCESS_INFORMATION process{};
    // Same environment and working folder as this process (nullptr = inherit).
    if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr,
                        &startup, &process)) {
        return Fail(L"Fritchii could not start fritchii.exe. Try starting fritchii.exe directly.");
    }
    AllowSetForegroundWindow(process.dwProcessId);
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return static_cast<int>(code);
}
