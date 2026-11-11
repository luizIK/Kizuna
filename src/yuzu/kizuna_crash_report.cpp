// SPDX-FileCopyrightText: Copyright 2026 Kizuna contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/kizuna_crash_report.h"

#ifdef _WIN32

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include "common/fs/path_util.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"

namespace KizunaCrash {
namespace {

// Everything the filter touches is static: after a crash the heap may not be trustworthy and a
// stack overflow leaves almost no stack.
wchar_t crash_dir[MAX_PATH * 2]{};
char report[8192];
wchar_t path_buffer[MAX_PATH * 2 + 64];
LONG entered = 0;
// MinGW's own filter (crt0's _gnu_exception_handler): it continues the GCC C++ exceptions nobody
// caught, so that std::terminate runs and the renderer's terminate handler logs the error. v1.6
// replaced it, and a VK_ERROR thrown at a scene change left only "0x20474343" here.
LPTOP_LEVEL_EXCEPTION_FILTER previous_filter = nullptr;

using MiniDumpWriteDumpFn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, MINIDUMP_TYPE,
                                          PMINIDUMP_EXCEPTION_INFORMATION,
                                          PMINIDUMP_USER_STREAM_INFORMATION,
                                          PMINIDUMP_CALLBACK_INFORMATION);

const char* BaseName(const char* path) {
    const char* base = path;
    for (const char* c = path; *c != '\0'; ++c) {
        if (*c == '\\' || *c == '/') {
            base = c + 1;
        }
    }
    return base;
}

// "amdvlk64.dll+0x1a2b3c" for an address, or the bare address outside every module.
int Describe(char* out, size_t size, const void* address) {
    HMODULE module{};
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(address), &module) &&
        module != nullptr) {
        char path[MAX_PATH]{};
        GetModuleFileNameA(module, path, MAX_PATH);
        return std::snprintf(out, size, "%s+0x%llx", BaseName(path),
                             static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) -
                                                             reinterpret_cast<uintptr_t>(module)));
    }
    return std::snprintf(out, size, "0x%llx",
                         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address)));
}

void Append(int& length, int written) {
    if (written > 0) {
        length += written;
        if (length > static_cast<int>(sizeof(report)) - 1) {
            length = static_cast<int>(sizeof(report)) - 1;
        }
    }
}

LONG WINAPI Filter(EXCEPTION_POINTERS* info) {
    if (InterlockedExchange(&entered, 1) != 0 || crash_dir[0] == L'\0' || info == nullptr ||
        info->ExceptionRecord == nullptr) {
        return previous_filter ? previous_filter(info) : EXCEPTION_CONTINUE_SEARCH;
    }
    const EXCEPTION_RECORD& record = *info->ExceptionRecord;
    // Breakpoints and C++ exceptions that something catches never get here; a debugger's
    // single-step and the like do not end the process either.
    SYSTEMTIME now{};
    GetLocalTime(&now);

    int length = 0;
    char where[512]{};
    char* const text = report;
    const size_t size = sizeof(report);
    Append(length, std::snprintf(text + length, size - length,
                                 "Kizuna crash report (%s, %s)\r\n", ::Common::g_build_release_tag,
                                 ::Common::g_scm_desc));
    Describe(where, sizeof(where), record.ExceptionAddress);
    Append(length, std::snprintf(text + length, size - length,
                                 "Exception 0x%08lX at %s, thread %lu, flags 0x%lX\r\n",
                                 static_cast<unsigned long>(record.ExceptionCode), where,
                                 static_cast<unsigned long>(GetCurrentThreadId()),
                                 static_cast<unsigned long>(record.ExceptionFlags)));
    if ((record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION ||
         record.ExceptionCode == EXCEPTION_IN_PAGE_ERROR) &&
        record.NumberParameters >= 2) {
        const ULONG_PTR kind = record.ExceptionInformation[0];
        Append(length, std::snprintf(text + length, size - length, "%s of address 0x%llx",
                                     kind == 0 ? "Read" : (kind == 1 ? "Write" : "Execute"),
                                     static_cast<unsigned long long>(
                                         record.ExceptionInformation[1])));
        if (record.ExceptionCode == EXCEPTION_IN_PAGE_ERROR && record.NumberParameters >= 3) {
            Append(length, std::snprintf(text + length, size - length, ", NTSTATUS 0x%08llX",
                                         static_cast<unsigned long long>(
                                             record.ExceptionInformation[2])));
        }
        Append(length, std::snprintf(text + length, size - length, "\r\n"));
    }
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        // ullTotalPageFile is the system's commit limit, ullAvailPageFile what is left of it.
        Append(length, std::snprintf(
                           text + length, size - length,
                           "Commit: %llu MiB left of %llu MiB; RAM: %llu MiB free of %llu MiB\r\n",
                           static_cast<unsigned long long>(memory.ullAvailPageFile >> 20),
                           static_cast<unsigned long long>(memory.ullTotalPageFile >> 20),
                           static_cast<unsigned long long>(memory.ullAvailPhys >> 20),
                           static_cast<unsigned long long>(memory.ullTotalPhys >> 20)));
    }
    // The programs holding the commit, as last measured while playing (the start, then every
    // minute while it was tight): a static buffer, nothing measured here.
    if (const char* programs = Common::LastTopCommitPrograms(); programs[0] != '\0') {
        Append(length, std::snprintf(text + length, size - length, "Programs by commit: %s\r\n",
                                     programs));
    }
    void* frames[32]{};
    const USHORT count = RtlCaptureStackBackTrace(0, 32, frames, nullptr);
    Append(length, std::snprintf(text + length, size - length, "Stack (%u frames):\r\n",
                                 static_cast<unsigned>(count)));
    for (USHORT i = 0; i < count; ++i) {
        Describe(where, sizeof(where), frames[i]);
        Append(length, std::snprintf(text + length, size - length, "  %s\r\n", where));
    }

    _snwprintf(path_buffer, sizeof(path_buffer) / sizeof(wchar_t),
               L"%ls\\kizuna-crash-%04u%02u%02u-%02u%02u%02u.txt", crash_dir,
               static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth),
               static_cast<unsigned>(now.wDay), static_cast<unsigned>(now.wHour),
               static_cast<unsigned>(now.wMinute), static_cast<unsigned>(now.wSecond));
    HANDLE file = CreateFileW(path_buffer, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file != INVALID_HANDLE_VALUE) {
        DWORD written{};
        WriteFile(file, text, static_cast<DWORD>(length), &written, nullptr);
        CloseHandle(file);
    }

    // The minidump: dbghelp is loaded now, not linked, so a missing one costs only the dump.
    if (const HMODULE dbghelp = LoadLibraryW(L"dbghelp.dll")) {
        const auto write_dump = reinterpret_cast<MiniDumpWriteDumpFn>(
            reinterpret_cast<void*>(GetProcAddress(dbghelp, "MiniDumpWriteDump")));
        if (write_dump != nullptr) {
            _snwprintf(path_buffer, sizeof(path_buffer) / sizeof(wchar_t),
                       L"%ls\\kizuna-crash-%04u%02u%02u-%02u%02u%02u.dmp", crash_dir,
                       static_cast<unsigned>(now.wYear), static_cast<unsigned>(now.wMonth),
                       static_cast<unsigned>(now.wDay), static_cast<unsigned>(now.wHour),
                       static_cast<unsigned>(now.wMinute), static_cast<unsigned>(now.wSecond));
            const HANDLE dump = CreateFileW(path_buffer, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                            FILE_ATTRIBUTE_NORMAL, nullptr);
            if (dump != INVALID_HANDLE_VALUE) {
                MINIDUMP_EXCEPTION_INFORMATION exception{};
                exception.ThreadId = GetCurrentThreadId();
                exception.ExceptionPointers = info;
                exception.ClientPointers = FALSE;
                write_dump(GetCurrentProcess(), GetCurrentProcessId(), dump,
                           static_cast<MINIDUMP_TYPE>(MiniDumpNormal | MiniDumpWithThreadInfo |
                                                      MiniDumpWithUnloadedModules),
                           &exception, nullptr, nullptr);
                CloseHandle(dump);
            }
        }
    }
    // A C++ exception goes on to std::terminate (the log names it); anything else ends here.
    // Only the first exception gets a report (entered stays set): the same second's file name.
    return previous_filter ? previous_filter(info) : EXCEPTION_CONTINUE_SEARCH;
}

} // Anonymous namespace

void Install() {
    try {
        const std::filesystem::path dir =
            Common::FS::GetEdenPath(Common::FS::EdenPath::CrashDumpsDir);
        std::error_code error;
        std::filesystem::create_directories(dir, error);
        const std::wstring wide = dir.wstring();
        if (wide.size() + 1 >= sizeof(crash_dir) / sizeof(wchar_t)) {
            return;
        }
        std::wmemcpy(crash_dir, wide.c_str(), wide.size() + 1);
        previous_filter = SetUnhandledExceptionFilter(Filter);
    } catch (...) {
        // No report is better than no start.
    }
}

} // namespace KizunaCrash

#else

namespace KizunaCrash {
void Install() {}
} // namespace KizunaCrash

#endif
