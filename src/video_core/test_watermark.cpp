// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>

#include "common/build_stamp.h"
#include "common/logging.h"
#include "video_core/test_watermark.h"

namespace VideoCore::TestWatermark {

namespace {

// Second copy of the package stamp, written by the same packaging step as common's. Same layout:
// 16-byte marker, 16-byte slot; a plain build's slot is derived from the marker.
extern "C" {
extern const volatile unsigned char fritchii_frame_stamp[32];
const volatile unsigned char fritchii_frame_stamp[32] = {
    0x8e, 0x4c, 0x76, 0xc4, 0xf5, 0x06, 0x08, 0x79, 0x32, 0x1e, 0x61, 0xef, 0x3c, 0x5e, 0x9c, 0x41,
    0x2b, 0xec, 0xd9, 0x6a, 0x5c, 0xb2, 0xbf, 0xf1, 0xaf, 0xd6, 0xe2, 0x6b, 0xbd, 0x22, 0x63, 0x11,
};
}

/// The id to show: common's stamp, else this slot's when it isn't the plain value. Empty: plain.
std::string StampId() {
    std::string id = Common::BuildStamp();
    if (!id.empty()) {
        return id;
    }
    bool plain = true;
    unsigned char bytes[16];
    for (std::size_t i = 0; i < 16; ++i) {
        bytes[i] = fritchii_frame_stamp[16 + i];
        const unsigned char base = fritchii_frame_stamp[i];
        plain &= bytes[i] == static_cast<unsigned char>((base ^ 0xA5) + 3 * i);
    }
    if (plain) {
        return {};
    }
    static constexpr char Hex[] = "0123456789abcdef";
    for (const unsigned char b : bytes) {
        id += Hex[b >> 4];
        id += Hex[b & 15];
    }
    return id;
}

/// 5x7 glyphs, one row per byte, bit 4 = leftmost column.
using Glyph = std::array<u8, 7>;

Glyph GlyphFor(char c) {
    switch (c) {
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E};
    case '6': return {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C};
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    default: return {};
    }
}

// Spots inside the game picture (fractions of its size, top-left of the text); one a minute.
constexpr std::array<std::array<float, 2>, 6> Spots{{
    {0.03f, 0.04f}, {0.97f, 0.92f}, {0.03f, 0.92f}, {0.97f, 0.04f}, {0.50f, 0.04f}, {0.50f, 0.92f},
}};

} // namespace

int Mode() {
    static const int mode = [] {
        const char* value = std::getenv("FRITCHII_TEST_MARK");
        const int v = value != nullptr ? std::atoi(value) : 1;
        return v == 0 || v == 3 ? v : 1;
    }();
    return mode;
}

void LogMode(const char* how) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        LOG_INFO(Render, "Test mark: {} (FRITCHII_TEST_MARK={})", how, Mode());
    }
}

const Blocks& Build(const Common::Rectangle<u32>& screen, u32 frame_width, u32 frame_height) {
    static const std::string id = StampId();
    static const Blocks none{};
    if (id.empty()) {
        return none;
    }
    static const std::string text = [] {
        // A test package's tag ("test 8") puts its number in: "FRITCHII V6 TEST 8 <id>".
        std::string t = "FRITCHII V6 TEST ";
        if (const auto& tag = Common::BuildTag(); tag.rfind("test ", 0) == 0) {
            for (std::size_t i = 5; i < tag.size() && i < 9; ++i) {
                const char c = tag[i];
                t += (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
            }
            t += ' ';
        }
        for (std::size_t i = 0; i < 8; ++i) {
            const char c = id[i];
            t += (c >= 'a' && c <= 'f') ? static_cast<char>(c - 'a' + 'A') : c;
        }
        return t;
    }();

    const u32 sw = screen.GetWidth();
    const u32 sh = screen.GetHeight();
    const auto minute = std::chrono::duration_cast<std::chrono::minutes>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    const std::size_t spot = static_cast<std::size_t>(minute) % Spots.size();

    // Rebuilt only when the picture or the spot changes (once a minute).
    thread_local Blocks blocks;
    thread_local std::array<u32, 7> key{};
    const std::array<u32, 7> now_key{screen.left, screen.top, sw, sh, frame_width, frame_height,
                                     static_cast<u32>(spot)};
    if (now_key == key && !blocks.text.empty()) {
        return blocks;
    }
    key = now_key;
    static std::atomic<u64> next_version{0};
    blocks.version = next_version.fetch_add(1, std::memory_order_relaxed) + 1;
    blocks.shadow.clear();
    blocks.text.clear();
    if (sw < 64 || sh < 64) {
        return blocks;
    }

    const u32 scale = (std::max)(2u, sh / 360);
    const u32 shadow = (std::max)(1u, scale / 2);
    const u32 text_w = static_cast<u32>(text.size()) * 6 * scale;
    const u32 text_h = 7 * scale;
    const auto place = [](float f, u32 extent, u32 size) -> u32 {
        const float free_space = extent > size ? static_cast<float>(extent - size) : 0.0f;
        return static_cast<u32>(f * free_space);
    };
    const u32 x0 = screen.left + place(Spots[spot][0], sw, text_w + shadow);
    const u32 y0 = screen.top + place(Spots[spot][1], sh, text_h + shadow);

    const auto add = [&](std::vector<Block>& out, u32 x, u32 y, u32 w, u32 h) {
        if (x >= frame_width || y >= frame_height) {
            return;
        }
        w = (std::min)(w, frame_width - x);
        h = (std::min)(h, frame_height - y);
        out.push_back(Block{x, y, w, h});
    };
    for (std::size_t c = 0; c < text.size(); ++c) {
        const Glyph glyph = GlyphFor(text[c]);
        const u32 cx = x0 + static_cast<u32>(c) * 6 * scale;
        for (u32 row = 0; row < 7; ++row) {
            u32 col = 0;
            while (col < 5) {
                if (((glyph[row] >> (4 - col)) & 1) == 0) {
                    ++col;
                    continue;
                }
                u32 run = col;
                while (run < 5 && ((glyph[row] >> (4 - run)) & 1) != 0) {
                    ++run;
                }
                const u32 x = cx + col * scale;
                const u32 y = y0 + row * scale;
                const u32 w = (run - col) * scale;
                add(blocks.shadow, x + shadow, y + shadow, w, scale);
                add(blocks.text, x, y, w, scale);
                col = run;
            }
        }
    }
    return blocks;
}

} // namespace VideoCore::TestWatermark
