// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/texture_cache/relocation_memory.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"

namespace VideoCommon::RelocationMemory {
namespace {


/// TextureCacheBase::RelocationShapeKey of a single-level, single-layer, single-sample 2D image.
constexpr u64 Key(u32 format, u32 width, u32 height) {
    return (u64{format & 0xFF} << 56) | (u64{1} << 52) | (u64{1} << 48) | (u64{1} << 36) |
           (u64{width & 0x3FFFF} << 18) | (height & 0x3FFFF);
}

struct Seed {
    u32 format;
    u32 width;
    u32 height;
    u8 guard;
};

/// Colour shapes seen losing their contents to a relocation in our logs (602 logs, 2026-11-18):
/// the black flashes. Only these: v6.0 also seeded every often-guarded shape and every square
/// D32 size, which stopped all relocations (a tester's pot room roam: 33,873 candidates, 0
/// relocated, VRAM 5 -> 11 GB). Format numbers are VideoCore::Surface::PixelFormat.
constexpr Seed Seeds[] = {
    {0, 640, 360, 1},   {0, 960, 540, 1},   {0, 1280, 720, 1},  {16, 960, 540, 1},
    {16, 1280, 720, 1}, {21, 1920, 1080, 1}, {21, 2560, 1440, 1},
};

/// Depth and stencil formats (PixelFormat from D32_FLOAT on) are never remembered: OoT's shadow
/// and per-object depth targets are what relocation exists for (they learn per session, as
/// before v6.0).
constexpr u32 FirstDepthFormat = 105;
constexpr bool Remembered(u64 key) {
    return ((key >> 56) & 0xFF) < FirstDepthFormat;
}

std::mutex g_mutex;
std::unordered_map<u64, u8> g_known; ///< What the file holds (plus the seed)
bool g_loaded{};

std::filesystem::path FilePath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "relocation_lost.txt";
}

} // Anonymous namespace

bool Enabled() {
    static const bool on = [] {
        const char* const value = std::getenv("FRITCHII_RELOCATION_MEMORY");
        return value == nullptr || value[0] == '\0' || std::strtol(value, nullptr, 10) != 0;
    }();
    return on;
}

std::vector<std::pair<u64, u8>> Load() {
    std::scoped_lock lock{g_mutex};
    if (!g_loaded) {
        g_loaded = true;
        for (const Seed& seed : Seeds) {
            g_known[Key(seed.format, seed.width, seed.height)] |= seed.guard;
        }
        const size_t seeded = g_known.size();
        size_t learned = 0;
        std::ifstream file{FilePath()};
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream fields{line};
            std::string key_text;
            unsigned guard = 0;
            if (!(fields >> key_text >> guard) || guard == 0 || guard > 7) {
                continue;
            }
            const u64 key = std::strtoull(key_text.c_str(), nullptr, 16);
            if (key != 0 && Remembered(key)) {
                g_known[key] |= static_cast<u8>(guard);
                ++learned;
            }
        }
        LOG_INFO(HW_GPU,
                 "Render target relocation memory (FRITCHII_RELOCATION_MEMORY=1): {} shapes "
                 "guarded from the start ({} seeded, {} lines learned in earlier sessions)",
                 g_known.size(), seeded, learned);
    }
    return {g_known.begin(), g_known.end()};
}

void Note(u64 shape_key, u8 guard) {
    if (!Remembered(shape_key)) {
        return;
    }
    std::scoped_lock lock{g_mutex};
    u8& known = g_known[shape_key];
    if ((known | guard) == known) {
        return;
    }
    known |= guard;
    std::error_code error;
    std::filesystem::create_directories(FilePath().parent_path(), error);
    std::ofstream file{FilePath(), std::ios::app};
    file << fmt::format("{:016x} {}\n", shape_key, static_cast<unsigned>(known));
}

} // namespace VideoCommon::RelocationMemory
