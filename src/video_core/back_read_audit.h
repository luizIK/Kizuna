// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>

#include "common/common_types.h"

#if defined(_MSC_VER)
#include <intrin.h>
#define FRITCHII_RETURN_ADDRESS() _ReturnAddress()
#else
#define FRITCHII_RETURN_ADDRESS() __builtin_return_address(0)
#endif

/// FRITCHII_BACK_READ_AUDIT=1 (boot): counts every read of game memory the GPU back thread
/// makes, by call site (the read function's return address), and logs the top 40 every 10 s.
/// The memory stage's audit (Fable's ideal pipeline design 2.5, step 2 spec 7.3): the list must
/// come down to texture contents, the texture and sampler tables and shader code.
/// See back_read_audit.cpp for the table and the log, docs/fritchii/switches.md for the use.
namespace VideoCommon::BackReadAudit {

/// Which read function was called (the log names it next to the site).
enum class Kind : u8 {
    MmRead,              ///< Tegra::MemoryManager::Read<T>
    MmGetPointer,        ///< MemoryManager::GetPointer (const)
    MmGetPointerMut,     ///< MemoryManager::GetPointer (may be a write)
    MmReadHandleWord,    ///< MemoryManager::ReadHandleWord
    MmDescriptorPointer, ///< MemoryManager::GetPointerForDescriptor
    MmReadBlock,         ///< MemoryManager::ReadBlock
    MmReadBlockUnsafe,   ///< MemoryManager::ReadBlockUnsafe
    MmReadBlockGuarded,  ///< MemoryManager::ReadBlockGuarded
    MmReadWords,         ///< MemoryManager::ReadWordsUnsafe
    MmGetSpan,           ///< MemoryManager::GetSpan (const)
    MmGetSpanMut,        ///< MemoryManager::GetSpan (may be a write)
    MmCopyBlock,         ///< MemoryManager::CopyBlock (reads the source)
    DevRead,             ///< MaxwellDeviceMemoryManager::Read<T>
    DevGetPointer,       ///< device GetPointer<T> (const)
    DevGetPointerMut,    ///< device GetPointer<T> (may be a write)
    DevGetSpan,          ///< device GetSpan (const)
    DevGetSpanMut,       ///< device GetSpan (may be a write)
    DevReadBlock,        ///< device ReadBlock
    DevReadBlockUnsafe,  ///< device ReadBlockUnsafe
    Count,
};

/// Set once at boot from FRITCHII_BACK_READ_AUDIT (Init). The only thing a read checks while off.
extern bool enabled;

/// FRITCHII_ESPLIT=1 (fritchii_e_split_thread.h): every audited read's caller, outside the
/// audit's own nesting, also goes here; set (with `enabled`) before the back runs. The audit's
/// tables still count only with its own switch.
using TouchHook = void (*)(const void* caller);
inline TouchHook g_touch_hook = nullptr;

/// Reads FRITCHII_BACK_READ_AUDIT the first time (MemoryManager's constructor calls it, after
/// main() applied nx2_switches.txt); later calls do nothing.
void Init();

/// Counts one read at `caller`, when this thread is the back and no audited read encloses it.
void Note(const void* caller, std::size_t bytes, Kind kind) noexcept;

/// An audited read that calls other audited reads: counts once (as Note), then hides the
/// reads it makes inside, so a MemoryManager read is not counted again at its device read.
class OuterScope {
public:
    OuterScope(const void* caller, std::size_t bytes, Kind kind) noexcept;
    ~OuterScope();
    OuterScope(const OuterScope&) = delete;
    OuterScope& operator=(const OuterScope&) = delete;
};

/// Hides the audited calls made inside: accesses that are not reads (a write through a
/// pointer, a pointer compare).
class HideScope {
public:
    HideScope() noexcept;
    ~HideScope();
    HideScope(const HideScope&) = delete;
    HideScope& operator=(const HideScope&) = delete;
};

/// Runs `body` hidden from the audit. One branch on `enabled` while off.
template <typename F>
inline decltype(auto) Unaudited(F&& body) {
    if (enabled) [[unlikely]] {
        const HideScope hide;
        return body();
    }
    return body();
}

} // namespace VideoCommon::BackReadAudit

/// First statement of a leaf read function (one that calls no other audited read).
#define FRITCHII_BACK_READ_NOTE(bytes, kind)                                                     \
    do {                                                                                         \
        if (::VideoCommon::BackReadAudit::enabled) [[unlikely]] {                                \
            ::VideoCommon::BackReadAudit::Note(FRITCHII_RETURN_ADDRESS(), (bytes), (kind));      \
        }                                                                                        \
    } while (false)

/// Last statement of a read function whose work is the lambda `body`: counts the call and
/// hides the audited reads `body` makes. The return address is taken here, in the read
/// function itself, and only while the audit is on.
#define FRITCHII_BACK_READ_AUDITED(bytes, kind, body)                                            \
    if (::VideoCommon::BackReadAudit::enabled) [[unlikely]] {                                    \
        const ::VideoCommon::BackReadAudit::OuterScope fritchii_audit_scope{                     \
            FRITCHII_RETURN_ADDRESS(), (bytes), (kind)};                                         \
        return body();                                                                           \
    }                                                                                            \
    return body()
