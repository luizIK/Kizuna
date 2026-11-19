// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#include "common/jit_symbols.h"

namespace Common::JitSymbols {
namespace {

struct Range {
    std::uintptr_t end;
    std::string name;
};

std::mutex mutex;
std::map<std::uintptr_t, Range> ranges;

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        // Opt-in only (Fritchii): the owner's setup always sets EDEN_NX2_DEBUG_SERVER, and every
        // JIT block compile would otherwise take this lock and store a name nobody reads.
        const char* const value = std::getenv("EDEN_NX2_JIT_SYMBOLS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void Register(const void* start, const void* end, std::string_view name) {
    if (!Enabled() || start == end) {
        return;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(start);
    std::scoped_lock lock{mutex};
    ranges.insert_or_assign(first, Range{reinterpret_cast<std::uintptr_t>(end), std::string{name}});
}

void Clear() {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{mutex};
    ranges.clear();
}

std::optional<std::string> Lookup(const void* address) {
    if (!Enabled()) {
        return std::nullopt;
    }
    const auto value = reinterpret_cast<std::uintptr_t>(address);
    std::scoped_lock lock{mutex};
    auto it = ranges.upper_bound(value);
    if (it == ranges.begin()) {
        return std::nullopt;
    }
    --it;
    if (value >= it->second.end) {
        return std::nullopt;
    }
    return it->second.name;
}

// The sample profile (EDEN_NX2_SAMPLE_PROFILE) records raw addresses; this table at the end of
// a recording lets the report name the guest code JIT samples ran (blocks compiled before the
// last code cache clear only).
bool DumpTo(const std::string& path) {
    if (!Enabled()) {
        return false;
    }
    std::FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return false;
    }
    {
        std::scoped_lock lock{mutex};
        for (const auto& [start, range] : ranges) {
            std::fprintf(file, "%llx %llx %s\n", static_cast<unsigned long long>(start),
                         static_cast<unsigned long long>(range.end), range.name.c_str());
        }
    }
    std::fclose(file);
    return true;
}

} // namespace Common::JitSymbols
