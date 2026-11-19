// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <system_error>

#include "common/build_stamp.h"
#include "common/scm_rev.h"
#ifdef _WIN32
#include "common/fs/path_util.h"
#endif

namespace Common {

// 16-byte marker followed by a 16-byte slot. Packaging tools find the marker in the binary and
// write the slot; the volatile reads keep the compiler from folding either. A plain build's slot
// holds a value derived from the marker; anything else (a stamp, zeros, an edit) is a stamped copy.
extern "C" {
extern const volatile unsigned char fritchii_build_stamp[32];
const volatile unsigned char fritchii_build_stamp[32] = {
    0xc5, 0xe5, 0x91, 0xf0, 0xf3, 0x0a, 0xfe, 0x2c, 0xe3, 0x2d, 0xf2, 0xb0, 0x60, 0x6e, 0x9d, 0xc1,
    0x70, 0x51, 0x06, 0xa0, 0xb6, 0x58, 0x05, 0x48, 0x4a, 0x51, 0xb5, 0xe8, 0xb9, 0x1c, 0x66, 0x63,
};
}

std::string BuildStamp() {
    unsigned char id[16];
    bool plain = true;
    for (std::size_t i = 0; i < 16; ++i) {
        id[i] = fritchii_build_stamp[16 + i];
        const unsigned char base = fritchii_build_stamp[i];
        plain &= id[i] == static_cast<unsigned char>(base * 7 + 13 + i);
    }
    if (plain) {
        return {};
    }
    char hex[33];
    for (std::size_t i = 0; i < 16; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", id[i]);
    }
    return std::string(hex, 32);
}

namespace {

std::filesystem::path ExeDirectory() {
#ifdef _WIN32
    return FS::GetExeDirectory();
#elif defined(__linux__)
    // The real binary: inside a mounted AppImage that is usr/bin, where the package puts the file.
    std::error_code ec;
    const auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path{} : exe.parent_path();
#else
    return {};
#endif
}

} // namespace

const std::string& BuildTag() {
    static const std::string tag = [] {
        const auto dir = ExeDirectory();
        if (dir.empty()) {
            return std::string{};
        }
        std::ifstream file(dir / "fritchii_build.txt");
        std::string line;
        if (!file || !std::getline(file, line)) {
            return std::string{};
        }
        // Printable ASCII only, at most 40 characters, no edge spaces: it lands in titles and logs.
        std::string out;
        for (const char c : line) {
            if (c >= 0x20 && c <= 0x7e && out.size() < 40) {
                out += c;
            }
        }
        const auto first = out.find_first_not_of(' ');
        if (first == std::string::npos) {
            return std::string{};
        }
        return out.substr(first, out.find_last_not_of(' ') - first + 1);
    }();
    return tag;
}

const std::string& BuildLabelWithTag() {
    static const std::string label =
        BuildTag().empty() ? std::string{g_build_label}
                           : std::string{g_build_label} + " " + BuildTag();
    return label;
}

} // namespace Common
