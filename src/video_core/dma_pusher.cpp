// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <span>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <cstdio>
#include <optional>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/memory.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"
#include "video_core/dma_pusher.h"
#include "video_core/back_read_audit.h"
#include "video_core/command_fetch_order.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/fritchii_front_split.h"
#include "video_core/fritchii_s4_count.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memstage_shadow.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/nx2_list_dispatch.h"
#include "video_core/nx2_list_draw.h"
#include "video_core/nx2_list_verify.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;

namespace {
/// The split's position after the last copy engine method that may write memory (the front's;
/// ported).
u64 g_last_copy_writer{};

/// Whether copy engine work sent so far may still be queued on the split's back. A new session
/// starts the split's positions over: a position past the current one is from an older session.
[[nodiscard]] bool CopyWritersPending(VideoCommon::GpuSplit& split) {
    if (g_last_copy_writer > split.SentPosition()) {
        g_last_copy_writer = 0;
    }
    return !split.Ran(g_last_copy_writer);
}

// FRITCHII_LIST_COPY_SKIP (live, default 0; fable_b 27.9 item 3) and its counters (always on with
// the split): an entry that begins between methods waits for the "list writers" only because a
// copy is queued (the copy engine can copy GPU-written words anywhere). The front notes each
// copy's destination (MaxwellDMA::FrontNoteLaunch), so a queued writer whose noted range misses
// the entry's pages cannot have written it: no wait needed. The deferred check on the back at the
// entry's place stays, for every skipped entry, as the must-be-0.
struct ListWaitStats {
    std::array<u64, 2> waits{};         ///< [0] begins inside a method's data, [1] a copy queued
    std::array<u64, 2> written_after{}; ///< GPU-written by the answer after the wait
    u64 touched{};                      ///< [1]: a queued writer's noted range (or an unbounded
                                        ///< one) touched the entry's pages
    u64 skipped{};                      ///< [1] waits the switch skipped (no range touched)
    std::chrono::steady_clock::time_point since{};
    std::array<u64, 2> waits_logged{};
    std::array<u64, 2> written_logged{};
    u64 touched_logged{};
    u64 skipped_logged{};
};
ListWaitStats g_list_waits{};
/// Must be 0: entries the switch let skip that the back then found GPU-written (stops it).
std::atomic<u64> g_list_skip_missed{};
std::atomic<bool> g_list_skip_stopped{};

/// Whether writers sent but not run yet may write [address, address + size) of `memory`: a
/// noted range over it or an unbounded writer (GpuSplit::WritersMayWriteSince, per page; an
/// unmapped page counts as written). `since` is one before the played position: a range is
/// noted at its entry's start, which has run only once the played position is past it.
[[nodiscard]] bool QueuedWritersMayWrite(VideoCommon::GpuSplit& split,
                                         Tegra::MemoryManager& memory, GPUVAddr address,
                                         u64 size) {
    const u64 played = split.ConsumedNow();
    const u64 since = played != 0 ? played - 1 : 0;
    constexpr GPUVAddr Page = 0x1000;
    for (GPUVAddr page = address & ~(Page - 1); page < address + size; page += Page) {
        const GPUVAddr begin = (std::max)(page, address);
        const GPUVAddr end = (std::min)(page + Page, address + size);
        const std::optional<DAddr> device = memory.GpuToCpuAddress(begin);
        if (!device || split.WritersMayWriteSince(since, *device, end - begin)) {
            return true;
        }
    }
    return false;
}

void MaybeLogListWaits() {
    // FRITCHII_STEP_LEAN=1 (F6): the clock is read on every 256th call only (one read per entry
    // was 13.5 ms/s of the front, sampled). The rates do not change: each is a count over the
    // clock time between two lines; a line comes at most 255 entries (a few ms) later.
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_step_lean,
                                         "FRITCHII_STEP_LEAN") == 1) {
        static thread_local u32 calls{};
        if ((++calls & 255) != 0) {
            return;
        }
    }
    ListWaitStats& s = g_list_waits;
    const auto now = std::chrono::steady_clock::now();
    if (s.since.time_since_epoch().count() == 0) {
        s.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.since).count();
    if (seconds < 10.0) {
        return;
    }
    s.since = now;
    const auto rate = [seconds](u64 count, u64& logged) {
        const double value = static_cast<double>(count - logged) / seconds;
        logged = count;
        return value;
    };
    const double in_data = rate(s.waits[0], s.waits_logged[0]);
    const double in_data_written = rate(s.written_after[0], s.written_logged[0]);
    const double copy = rate(s.waits[1], s.waits_logged[1]);
    const double copy_written = rate(s.written_after[1], s.written_logged[1]);
    const double copy_touched = rate(s.touched, s.touched_logged);
    const double skipped = rate(s.skipped, s.skipped_logged);
    LOG_INFO(HW_GPU,
             "List writer waits (fable_b 27.9; FRITCHII_LIST_COPY_SKIP={}{}) {:.0f} s: inside a "
             "method's data {:.0f}/s (GPU-written after the wait {:.0f}/s); a copy queued "
             "{:.0f}/s (GPU-written after {:.0f}/s, a queued writer's range on the entry's pages "
             "{:.0f}/s); skipped, no range on the pages {:.0f}/s; must be 0: skipped entries "
             "the back found GPU-written {} this session",
             VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_list_copy_skip,
                                              "FRITCHII_LIST_COPY_SKIP"),
             g_list_skip_stopped.load(std::memory_order_relaxed) ? ", STOPPED" : "", seconds,
             in_data, in_data_written, copy, copy_written, copy_touched, skipped,
             g_list_skip_missed.load(std::memory_order_relaxed));
}

// Compute dispatches and DMA copies write the command lists ReadGpuWrittenCommands reads. The
// register writes that set them up only store a value: marking those too made every list fetch
// after one wait for all the work queued before it (draws included).
bool WritesMemory(Engines::EngineInterface* engine, Engines::EngineTypes type, u32 method,
                  u32 count) {
    if (type == Engines::EngineTypes::KeplerCompute) {
        const auto* const compute = static_cast<const Engines::KeplerCompute*>(engine);
        for (u32 i = 0; i < count; ++i) {
            if (compute->MayWriteMemory(method + i)) {
                return true;
            }
        }
        return false;
    }
    if (type == Engines::EngineTypes::MaxwellDMA) {
        for (u32 i = 0; i < count; ++i) {
            const u32 current = method + i;
            if (current >= MacroRegistersStart ||
                (current < engine->execution_mask.size() && engine->execution_mask[current])) {
                return true;
            }
        }
    }
    return false;
}

using DeviceRanges = boost::container::small_vector<std::pair<DAddr, std::size_t>, 32>;

// The device ranges GPU range [begin, begin + size) maps to (MemoryManager::
// GetSubmappedDeviceRange: contiguous pages merged, unmapped ones left out, which nothing
// writes). The last answer is kept until the page tables change (MapSerial): a blit's destination
// surface is the same one frame after frame. Front only.
[[nodiscard]] const DeviceRanges& DeviceRangesOf(const MemoryManager& memory_manager,
                                                 GPUVAddr begin, u64 size) {
    struct Memo {
        const MemoryManager* memory_manager;
        GPUVAddr begin;
        u64 size;
        u64 serial;
        DeviceRanges ranges;
    };
    static Memo memo{};
    const u64 serial = memory_manager.MapSerial();
    if (memo.memory_manager != &memory_manager || memo.begin != begin || memo.size != size ||
        memo.serial != serial) {
        // The serial is taken before the walk: a change during it makes the next call look again.
        memo.memory_manager = &memory_manager;
        memo.begin = begin;
        memo.size = size;
        memo.serial = serial;
        memo.ranges = memory_manager.GetSubmappedDeviceRange(begin, size);
    }
    return memo.ranges;
}

// Why a blit launch stayed an unranged writer with EXP120 = 2 (Fermi2D::FrontBlitRange's codes
// first), for FrontDmaRead's "unranged notes" line (FritchiiBlitMissText). Front only.
enum class BlitMiss : u32 {
    None,
    NotSeen,  ///< 1: a destination register the range needs was not sent yet
    Format,   ///< 2: an undecodable format
    Empty,    ///< 3: an empty surface
    Large,    ///< Over MaxNotedWrite
    Unmapped, ///< No byte of it mapped
    Pieces,   ///< More device ranges than MaxNotedPieces
    Other,    ///< An unknown FrontBlitRange code
    Count,
};
std::array<u64, static_cast<size_t>(BlitMiss::Count)> g_blit_misses{};
/// Blit launches that found destination registers the front never saw sent (the engine was
/// set up before the split started) and took them from the back after a drain.
u64 g_blit_seeds{};

// Fermi2D's front copy of the destination registers (FrontNoteMethod) from what the engine
// holds: its registers and, after them, its method sink. Only after a drain: the back is idle
// and has run every write sent, so this is the state the next blit sees.
void SeedFrontCopy(Engines::Fermi2D& blitter) {
    constexpr u32 first = FERMI2D_REG_INDEX(dst); // 0x80
    constexpr u32 count = 10;
    for (u32 i = 0; i < count; ++i) {
        blitter.FrontNoteMethod(first + i, blitter.regs.reg_array[first + i]);
    }
    for (const auto& [method, value] : blitter.method_sink) {
        const u32 index = method & ~Engines::EngineInterface::SplitPlainTag;
        if (index - first < count) {
            blitter.FrontNoteMethod(index, value);
        }
    }
}
/// The largest write noted by its range, and the most device ranges it may map to.
constexpr u64 MaxNotedWrite = u64{1} << 30;
constexpr size_t MaxNotedPieces = 16;

// The KeplerMemory or Fermi2D method just sent on the split (`words`: its `count` words, `multi`:
// a non-incrementing run, CallMultiMethod). EXP120 = 0 notes every one as an unranged writer,
// which ends every macro DMA read snapshot (fable_answer_long_drains 8.4). EXP120 = 1: the back
// writes memory only on KeplerMemory's data method and Fermi2D's blit launch (their CallMethod),
// so the methods that only store a register are not noted, nor a single data word that is not
// its method's last (Upload::State::ProcessData writes with the last one); a linear upload's data
// notes the bytes it may write (NoteMemoryWrite, one per device range). EXP120 = 2 also notes a
// blit's whole destination surface (Fermi2D::FrontBlitRange). Block linear uploads stay
// unranged, and so does anything over MaxNotedWrite, MaxNotedPieces device ranges or unmapped.
void NoteEngineWriter(VideoCommon::GpuSplit& split, const MemoryManager& memory_manager,
                      Engines::EngineInterface* engine, Engines::EngineTypes type, u32 method,
                      const u32* words, u32 count, bool is_last, bool multi) {
    using Kind = VideoCommon::GpuSplit::EngineWriteKind;
    using Note = VideoCommon::GpuSplit::EngineWriteNote;
    if (count == 0) {
        return split.NoteUnrangedWriter();
    }
    Kind kind = Kind::BlitRegister;
    bool writes = false;
    bool ranged = false;
    GPUVAddr begin{};
    u64 size{};
    BlitMiss miss = BlitMiss::None;
    if (type == Engines::EngineTypes::KeplerMemory) {
        auto* const memory = static_cast<Engines::KeplerMemory*>(engine);
        // Kept with EXP120 off too, so turning it on live finds the registers sent before.
        memory->FrontNoteMethod(method, words[count - 1]);
        if (method == KEPLERMEMORY_REG_INDEX(data)) {
            kind = Kind::I2mData;
            writes = multi || is_last;
            ranged = writes && memory->FrontDataRange(begin, size);
        } else {
            kind = Kind::I2mRegister;
        }
    } else {
        auto* const blitter = static_cast<Engines::Fermi2D*>(engine);
        blitter->FrontNoteMethod(method, words[count - 1]);
        if (method == FERMI2D_REG_INDEX(pixels_from_memory.src_y0) + 1) {
            kind = Kind::BlitLaunch;
            writes = true;
            ranged = blitter->FrontBlitRange(begin, size);
            if (!ranged && (size & 0xff) == static_cast<u64>(BlitMiss::NotSeen) &&
                VideoCore::Nx2Debug::Experiment(120) >= 2) {
                // Registers sent before the split started (or before its last restart) went
                // past the front's copy: take them from the engine once, after a drain. The
                // copy sees every write after that, so each engine drains here at most once.
                static u32 logged{};
                if (logged++ < 8) {
                    LOG_INFO(HW_GPU,
                             "EXP120: blit destination registers not sent since the split "
                             "started: {:#x} (bit n: register {:#x} + n); taken from the "
                             "engine after a drain",
                             size >> 8, static_cast<u32>(FERMI2D_REG_INDEX(dst)));
                }
                split.Drain(VideoCommon::DrainSite::Other);
                SeedFrontCopy(*blitter);
                ++g_blit_seeds;
                ranged = blitter->FrontBlitRange(begin, size);
            }
            if (!ranged) {
                const u64 code = size & 0xff;
                miss = code <= static_cast<u64>(BlitMiss::Empty) ? static_cast<BlitMiss>(code)
                                                                 : BlitMiss::Other;
                static u32 formats_logged{};
                if (miss == BlitMiss::Format && formats_logged++ < 8) {
                    LOG_INFO(HW_GPU,
                             "EXP120: blit destination format {:#x} has no pixel format "
                             "(TryPixelFormatFromRenderTargetFormat): left unranged",
                             size >> 8);
                }
            }
        }
    }
    const auto unranged = [&] {
        split.NoteUnrangedWriter();
        split.CountEngineWrite(multi, kind, Note::Unranged);
    };
    const int mode = VideoCore::Nx2Debug::Experiment(120);
    if (mode == 0) {
        return unranged();
    }
    if (!writes) {
        split.NoteUnrangedWriterSpared();
        split.CountEngineWrite(multi, kind, Note::Nothing);
        return;
    }
    // A huge upload or surface, one with no byte mapped or over too many device ranges, stays
    // unranged.
    const bool blit = kind == Kind::BlitLaunch;
    if (blit && mode < 2) {
        return unranged();
    }
    const auto missed = [&](BlitMiss reason) {
        if (blit) {
            ++g_blit_misses[static_cast<size_t>(reason)];
        }
        unranged();
    };
    if (!ranged) {
        return missed(miss);
    }
    if (size > MaxNotedWrite) {
        return missed(BlitMiss::Large);
    }
    const DeviceRanges& pieces = DeviceRangesOf(memory_manager, begin, size);
    if (pieces.empty()) {
        return missed(BlitMiss::Unmapped);
    }
    if (pieces.size() > MaxNotedPieces) {
        return missed(BlitMiss::Pieces);
    }
    split.NoteMemoryWriter();
    for (const auto& [device, length] : pieces) {
        split.NoteMemoryWrite(device, length, blit ? "Fermi2D blit" : "KeplerMemory upload");
    }
    split.NoteUnrangedWriterSpared();
    split.CountEngineWrite(multi, kind, Note::Ranged);
}

// FRITCHII_STEP_LEAN=1 (dma_pusher.h, CommandListRecycling): the slots the front gives a run
// list's storage back through. State 0 empty, 1 busy (one thread moves vectors in or out), 2
// full; each change of owner is a compare-exchange (acquire), each hand-back a release store.
struct RecycledStorage {
    std::atomic<u32> state{0};
    boost::container::small_vector<CommandListHeader, 512> command_lists;
    std::vector<CommandHeader, DefaultInitAllocator<CommandHeader>> submitted_words;
    std::vector<u32> submitted_offsets;
    std::vector<u32> deferred_entries;
};
constexpr size_t RecycleSlots = 8;
constexpr size_t InlineListEntries = 512; // CommandList::command_lists' inline capacity
constexpr size_t MaxRecycledWords = size_t{1} << 22; // 16 MiB of words: freed, not kept
std::array<RecycledStorage, RecycleSlots> g_recycled;

} // Anonymous namespace

bool CommandListRecycling() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_step_lean,
                                            "FRITCHII_STEP_LEAN") == 1;
}

void GiveCommandListStorage(CommandList& list) noexcept {
    const bool heap_entries = list.command_lists.capacity() > InlineListEntries;
    if ((!heap_entries && list.submitted_words.capacity() == 0 &&
         list.submitted_offsets.capacity() == 0) ||
        list.submitted_words.capacity() > MaxRecycledWords) {
        return; // nothing on the heap worth keeping (or too much): freed with the list
    }
    for (RecycledStorage& slot : g_recycled) {
        u32 expected = 0;
        if (slot.state.load(std::memory_order_relaxed) != 0 ||
            !slot.state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
            continue;
        }
        if (heap_entries) {
            slot.command_lists = std::move(list.command_lists);
        }
        slot.submitted_words = std::move(list.submitted_words);
        slot.submitted_offsets = std::move(list.submitted_offsets);
        slot.deferred_entries = std::move(list.deferred_entries);
        slot.state.store(2, std::memory_order_release);
        return;
    }
    // Every slot full: the submitter is behind on taking them; freed here as before.
}

void TakeCommandListStorage(CommandList& list) noexcept {
    for (RecycledStorage& slot : g_recycled) {
        u32 expected = 2;
        if (slot.state.load(std::memory_order_relaxed) != 2 ||
            !slot.state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
            continue;
        }
        if (slot.command_lists.capacity() > InlineListEntries) {
            list.command_lists = std::move(slot.command_lists);
        }
        list.submitted_words = std::move(slot.submitted_words);
        list.submitted_offsets = std::move(slot.submitted_offsets);
        list.deferred_entries = std::move(slot.deferred_entries);
        slot.state.store(0, std::memory_order_release);
        // Emptied, capacity kept: the submit sizes and fills them as on new vectors.
        list.command_lists.clear();
        list.submitted_words.clear();
        list.submitted_offsets.clear();
        list.deferred_entries.clear();
        return;
    }
}

void DmaPusher::PopFrontList() {
    if (CommandListRecycling()) [[unlikely]] {
        GiveCommandListStorage(dma_pushbuffer.front());
    }
    dma_pushbuffer.pop_front();
}

// Declared in maxwell_3d.cpp for FrontDmaRead's log (not in a header).
std::string FritchiiBlitMissText() {
    const auto at = [](BlitMiss reason) { return g_blit_misses[static_cast<size_t>(reason)]; };
    return fmt::format("{} register not sent, {} format, {} empty, {} over 1 GiB, {} unmapped, "
                       "{} over {} device ranges, {} other; {} took the registers from the "
                       "engine",
                       at(BlitMiss::NotSeen), at(BlitMiss::Format), at(BlitMiss::Empty),
                       at(BlitMiss::Large), at(BlitMiss::Unmapped), at(BlitMiss::Pieces),
                       MaxNotedPieces, at(BlitMiss::Other), g_blit_seeds);
}

namespace {

// EDEN_NX2_LIST_CHANGE_TRACE=1: per GPU-written command list address, how often its words differ
// from the last read of the same address (a wait that only guards unchanged words could go).
// Logged every 2 s with the first words of each list.
class ListChangeTrace {
public:
    static ListChangeTrace* Get() {
        static const bool enabled = std::getenv("EDEN_NX2_LIST_CHANGE_TRACE") != nullptr;
        static ListChangeTrace trace;
        return enabled ? &trace : nullptr;
    }

    void Note(GPUVAddr address, std::span<const CommandHeader> words, u32 method, u32 subchannel,
              u32 method_count, u32 engine) {
        Entry& entry = entries[address];
        ++entry.reads;
        entry.method = method;
        entry.subchannel = subchannel;
        entry.method_count = method_count;
        entry.engine = engine;
        std::vector<u32> now(words.size());
        std::memcpy(now.data(), words.data(), words.size() * sizeof(u32));
        if (!entry.last.empty()) {
            if (entry.last != now) {
                ++entry.changed;
                u32 differing = 0;
                for (size_t i = 0; i < (std::min)(now.size(), entry.last.size()); ++i) {
                    differing += now[i] != entry.last[i] ? 1 : 0;
                }
                entry.differing_words += differing + static_cast<u32>(
                    now.size() > entry.last.size() ? now.size() - entry.last.size()
                                                   : entry.last.size() - now.size());
            }
        }
        entry.last = std::move(now);
        const auto clock = std::chrono::steady_clock::now();
        if (clock - last_log < std::chrono::seconds{2}) {
            return;
        }
        last_log = clock;
        for (auto& [key, value] : entries) {
            if (value.reads == 0) {
                continue;
            }
            std::string head;
            for (size_t i = 0; i < (std::min<size_t>)(value.last.size(), 24); ++i) {
                head += fmt::format(" {:08x}", value.last[i]);
            }
            LOG_INFO(HW_GPU,
                     "List {:#x}: {} words, {} reads, {} changed ({} words differ in all); pending "
                     "method {:#x} x{} on sub {} (engine {:#x});{}",
                     key, value.last.size(), value.reads, value.changed, value.differing_words,
                     value.method, value.method_count, value.subchannel, value.engine, head);
            value.reads = 0;
            value.changed = 0;
            value.differing_words = 0;
        }
    }

private:
    struct Entry {
        u64 reads{};
        u64 changed{};
        u64 differing_words{};
        u32 method{};
        u32 subchannel{};
        u32 method_count{};
        u32 engine{};
        std::vector<u32> last;
    };
    std::unordered_map<GPUVAddr, Entry> entries;
    std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
};
} // Anonymous namespace

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    // EXPERIMENT EDEN_NX2_INVALIDATE_BEFORE_DISPATCH=list: apply the guest's collected CPU writes
    // before the submitted lists run; they were made before the submit.
    static const bool invalidate_at_list = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && std::string_view{value} == "list";
    }();
    if (invalidate_at_list) {
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            // Delta draw count: neutral, the writes break where they land (pools included).
            split.Defer([gpu = &system.GPU()] { gpu->InvalidateGPUCache(); },
                        VideoCommon::GpuSplit::Effect::Unknown,
                        VideoCommon::Nx2FastDrawCount::Neutral());
        } else {
            system.GPU().InvalidateGPUCache();
        }
    }
    // The sub-index is kept: a channel suspended on a semaphore resumes inside its command list.
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        // Submits and releases signaled fences: their writes belong to the entries that queued them.
        split.Defer(
            [gpu = &system.GPU()] {
                gpu->FlushCommands();
                gpu->OnCommandListEnd();
            },
            // Delta draw count: neutral, a submit moves the tick and released fences break.
            // The memory stage passes it (spec 3.3).
            VideoCommon::GpuSplit::Effect::None,
            VideoCommon::Nx2FastDrawCount::Neutral() |
                VideoCommon::StageTagBits(VideoCommon::StageTag::Pass));
        split.Publish();
        return;
    }
    system.GPU().FlushCommands();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::Step() {
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }
    // FRITCHII_FRONT_SPLIT_COUNT (dry, fritchii_front_split.h): counts this call and times 1 in
    // 64 by stage. Off: one relaxed load and a compare.
    namespace FrontSplit = VideoCommon::FrontSplit;
    const FrontSplit::EntryScope front_split_entry{channel_state.bind_id};
    const u64 suspensions_before = suspensions.load(std::memory_order_relaxed);
    if (!puller.ResumeAcquire(*this)) {
        return false;
    }
    // Lists queued while the channel waited were read from memory, not as submitted: the game
    // writes the commands a list runs before it releases the acquire, so the words at submit time
    // could be older. Reading them as late as the parse is too late in Hyrule Warriors: Age of
    // Imprisonment fights: a command ring the game had already refilled for a later frame, read
    // as this frame's commands (methods on unbound subchannels, garbage QMDs, then a channel
    // waiting forever on a release it never got). The acquire is released now, so the commands
    // are written and the memory not yet reused: copy them here. EDEN_NX2_RESNAPSHOT=0: off.
    // EXP39 & 2: the current list's entries before this one
    // ran already and are never parsed again, and an entry the acquire suspended inside reads
    // its remaining words from memory anyway (CanUseSubmittedCommands): neither is copied.
    if (ResnapshotEnabled() && suspensions.load(std::memory_order_relaxed) != suspensions_before) {
        const u64 now = suspensions.load(std::memory_order_acquire);
        size_t first_entry = 0;
        if ((VideoCore::Nx2Debug::Experiment(39) & 2) != 0) {
            first_entry = dma_pushbuffer_subindex + (command_resume_offset != 0 ? 1 : 0);
        }
        // FRITCHII_FRONT_SPLIT_COUNT: entries not parsed yet that an earlier refresh (or Push's)
        // already copied once (the front list's from where its parse is).
        size_t unparsed = dma_pushbuffer_subindex + (command_resume_offset != 0 ? 1 : 0);
        for (CommandList& list : dma_pushbuffer) {
            if (!list.submitted_offsets.empty()) {
                if (FrontSplit::g_state != 0 && list.refreshes != 0 &&
                    list.command_lists.size() > unparsed) [[unlikely]] {
                    FrontSplit::NoteRereadTwice(list.command_lists.size() - unparsed);
                }
                RefreshSnapshot(list, now, first_entry); // counts list.refreshes
            }
            first_entry = 0;
            unparsed = 0;
        }
    }

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        PopFrontList();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset +=
            ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingAcquire()) {
            FrontSplit::NoteSuspension();
            return false;
        }
        command_resume_offset = 0;
        PopFrontList();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced.load(); });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        // FRITCHII_DELAY_FRONT_KIND=4 ("P", FS0): once every 16 list entries, before their
        // checks. Off: one relaxed load and a branch.
        VideoCommon::GpuSplit::DelayFrontIf(VideoCommon::GpuSplit::FrontDelayKind::Entries);
        if (header.sync && g_nx2_ampere.load(std::memory_order_relaxed)) {
            VideoCommon::GpuSplit::Get().NoteSync();
        }
        // FRITCHII_S4_COUNT (dry): where this list entry starts in the split's ring (a page
        // mirror would be brought up to date here), and value 2's samples. Also for
        // FRITCHII_BIND_SKIP's in-entry step (E's list entry index).
        if (Common::S4Count::EntryMarks() && command_resume_offset == 0) [[unlikely]] {
            if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                VideoCommon::S4Count::FrontEntry(split, system.Host1x().MemoryManager());
            }
        }
        // EDEN_NX2_FETCH_DRAIN=1: wait for the back before every entry, as before the split
        // overlapped the two halves again.
        static const bool fetch_drain = std::getenv("EDEN_NX2_FETCH_DRAIN") != nullptr;
        if (fetch_drain && g_nx2_ampere.load(std::memory_order_relaxed)) {
            VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::FetchDrain);
        }
        bool gpu_written = IsGpuWrittenCommands(dma_state.dma_get,
                                                 size_t{header.size} * sizeof(u32));
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            auto& split = VideoCommon::GpuSplit::Get();
            // A 3D macro the split's front runs (the routing of CallMethod/CallMultiMethod below:
            // front macros and the native Turing macro engine) never reaches ProcessMacro, the
            // only reader of the 3D engine's current_dirty: its closure is not sent. It asked
            // the buffer cache (MustFlushRegion) on the back for nothing, and the memory stage
            // has to lend the cache for it (fable_memstage_6b_plan 2.8).
            const bool front_macro =
                macro_payload && split.Active() && split.FrontMacros() &&
                static_cast<Engines::Maxwell3D*>(subchannels[dma_state.subchannel])
                    ->GetNativeMacroEngine()
                    .native_turing;
            if (kepler_payload || (macro_payload && !front_macro)) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    FrontSplit::StepDefer();
                    split.Defer(
                        [engine_state, memory = &memory_manager,
                         address = GPUVAddr{dma_state.dma_get}, bytes = words * sizeof(u32)] {
                            engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                        },
                        VideoCommon::GpuSplit::Effect::None,
                        VideoCommon::Nx2FastDrawCount::Neutral());
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        if (command_resume_offset == 0) {
            // A method's data continues across GP entries, whatever their level: title 4 ends a
            // main entry after a macro's first word and supplies the remaining parameters in a
            // 5-word subroutine entry. EDEN_NX2_PARK_SPLIT_METHODS restores the old guess (an
            // entry elsewhere is a call that starts clean; resume where the split one ended).
            static const bool park_split_methods = std::getenv("EDEN_NX2_PARK_SPLIT_METHODS") != nullptr;
            if (g_nx2_ampere) {
                if (static u32 split_logs{}; dma_state.method_count != 0 && split_logs < 64) {
                    ++split_logs;
                    LOG_DEBUG(HW_GPU, "Split method {:#x} ({} left) on ch {}: next entry {:#x} size {} raw {:#x} (level bit {}, priv bit {}), expected {:#x}, parked {}",
                             dma_state.method, dma_state.method_count, channel_state.bind_id,
                             u64{header.addr}, u64{header.size}, header.raw, u64{header.is_push_buffer},
                             u64{header.allow_flush}, split_method_end, parked_methods.size());
                    std::array<u32, 8> next_words{};
                    std::array<u32, 4> prev_words{};
                    memory_manager.ReadBlockUnsafe(header.addr, next_words.data(),
                                             std::min<size_t>(header.size, 8) * sizeof(u32));
                    memory_manager.ReadBlockUnsafe(split_method_end - sizeof(prev_words), prev_words.data(),
                                             sizeof(prev_words));
                    LOG_DEBUG(HW_GPU, "  previous entry ends {:08x}; next entry starts {:08x}",
                              fmt::join(prev_words, " "), fmt::join(next_words, " "));
                }
                if (park_split_methods) {
                    ResolveSplitMethod(header.addr);
                }
            }
            if (dma_pushbuffer_subindex < command_list.submit_hashes.size()) {
                CheckSubmitHash(header, command_list.submit_hashes[dma_pushbuffer_subindex]);
            }
        }
        // A resumed entry is read again: the words after the acquire may be new by now.
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // Pages no GPU work has written yet may still get the entry from compute or copy work
        // queued before it: the GPU-written page bits learn of a write only when the back records
        // it, so the front waits for that work (the "list writers" site). The new shape
        // (ported):
        // - The GPU fetches an entry after that work only past a host-level wait (NoteSync):
        //   later work is not ordered before the fetch, so the entry waits only for the compute
        //   and copy work sent before the last sync (nor for draws, which write memory too;
        //   draw-written lists wait for the last sync, Nx2ReadListCopy).
        // - EDEN_NX2_LIST_WAIT_SKIP=1 (built in): an entry that begins between methods (a command
        //   buffer the CPU writes) doesn't wait at all. The words GPU work writes for the front
        //   to fetch are the parameters of a method begun in the entry before (an indirect
        //   dispatch's or draw's, as Ocarina of Time's 3D 0xe62 and compute 0xe74 lists), which
        //   begin inside its data and still wait. A resumed entry resumes after a puller method,
        //   which no parameter list holds. The copy engine can copy GPU-written words anywhere:
        //   while its work is queued, every entry waits.
        // EXP41=1: the waits before this port (LIST_WAIT_SKIP=1: none; =0: every writer).
        // The entries that didn't wait for every writer are checked on the back, at their place,
        // and each one GPU work queued before it wrote is logged.
        static const bool skip_list_wait = [] {
            const char* const value = std::getenv("EDEN_NX2_LIST_WAIT_SKIP");
            return value != nullptr && value[0] == '1';
        }();
        if (!gpu_written && VideoCommon::GpuSplit::Get().Active()) {
            auto& split = VideoCommon::GpuSplit::Get();
            FrontSplit::WriterCheck(); // FRITCHII_FRONT_SPLIT_COUNT: 2a's join before them
            const bool old_waits = VideoCore::Nx2Debug::Experiment(41) == 1;
            const bool copy_queued = CopyWritersPending(split);
            bool skip = old_waits ? skip_list_wait
                                  : skip_list_wait && dma_state.method_count == 0 && !copy_queued;
            // fable_b 27.9 item 3: why it waits ([0] inside a method's data, [1] a copy queued),
            // and for a copy whether a queued writer's noted range touches the entry's pages.
            const size_t wait_reason = dma_state.method_count != 0 ? 0 : 1;
            bool range_touched = false;
            bool range_skip = false;
            if (!old_waits && skip_list_wait && !skip && wait_reason == 1) {
                range_touched = QueuedWritersMayWrite(split, memory_manager, dma_state.dma_get,
                                                      u64{header.size} * sizeof(u32));
                if (!range_touched &&
                    VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_list_copy_skip,
                                                     "FRITCHII_LIST_COPY_SKIP") != 0 &&
                    !g_list_skip_stopped.load(std::memory_order_relaxed)) {
                    skip = true;
                    range_skip = true;
                    ++g_list_waits.skipped;
                }
            }
            bool checked = skip;
            if (!skip) {
                if (old_waits ? split.WaitForWriters() : split.WaitForSyncedWriters()) {
                    gpu_written = IsGpuWrittenCommands(dma_state.dma_get,
                                                       size_t{header.size} * sizeof(u32));
                }
                // Writers sent after the last sync may still be queued.
                checked = !gpu_written && !split.Ran(split.LastWriterPosition());
                if (!old_waits && skip_list_wait) {
                    ++g_list_waits.waits[wait_reason];
                    g_list_waits.written_after[wait_reason] += gpu_written ? 1 : 0;
                    g_list_waits.touched += range_touched ? 1 : 0;
                }
            }
            if (!old_waits && skip_list_wait) {
                MaybeLogListWaits();
            }
            // EDEN_NX2_EXP105 (live; 0 = every entry as before, N > 1 = 1 in N): this check only
            // reads (IsGpuWrittenCommands) and logs; it never fired in any OoT run of 2026-11-12,
            // and at ~84,000 entries a second it cost ~8 ms/s of the back (fable answer_backthread 9).
            // A range skip is always checked: it is FRITCHII_LIST_COPY_SKIP's must-be-0.
            if (checked && !range_skip) {
                if (const int every = VideoCore::Nx2Debug::Experiment(105); every > 1) {
                    static thread_local u32 list_check_count{};
                    checked = (++list_check_count % static_cast<u32>(every)) == 0;
                }
            }
            if (checked) {
                FrontSplit::StepDefer();
                split.Defer([this, address = GPUVAddr{dma_state.dma_get},
                             size = size_t{header.size} * sizeof(u32), skip, range_skip] {
                    if (IsGpuWrittenCommands(address, size)) {
                        if (range_skip) {
                            // FRITCHII_LIST_COPY_SKIP's must-be-0: the noted ranges missed a
                            // writer of this entry. The switch stops for the session.
                            g_list_skip_missed.fetch_add(1, std::memory_order_relaxed);
                            if (!g_list_skip_stopped.exchange(true, std::memory_order_relaxed)) {
                                LOG_CRITICAL(HW_GPU,
                                             "FRITCHII_LIST_COPY_SKIP stops for this session: "
                                             "an entry at {:#x} that skipped its wait was "
                                             "GPU-written by queued work",
                                             address);
                            }
                        }
                        static std::atomic<u32> missed{};
                        const u32 count = ++missed;
                        if (count <= 32 || (count & (count - 1)) == 0) {
                            LOG_WARNING(HW_GPU,
                                        "EDEN_NX2_LIST_WAIT_SKIP: command list at {:#x} ({} bytes) "
                                        "was GPU-written by work queued before it ({} so far; "
                                        "{})",
                                        address, size, count,
                                        skip ? "skipped the wait"
                                             : "waited for the writers before the last sync");
                        }
                    }
                    // Only reads and logs: neutral for the delta draw count (fable backthread 9).
                    // The memory stage passes it: the page table and the GPU-written page bits,
                    // no register, no cache object, no game memory (fable_memstage_6b_plan 2.1).
                }, VideoCommon::GpuSplit::Effect::Unknown,
                   VideoCommon::Nx2FastDrawCount::Neutral() |
                       VideoCommon::StageTagBits(VideoCommon::StageTag::Pass));
            }
        }
        // Resume and GPU-written entries need live words, even after a flush cleared dirty state.
        const bool submitted = (command_list.snapshot_suspensions & 1) == 0 &&
                               command_list.snapshot_suspensions ==
                                   suspensions.load(std::memory_order_acquire) &&
                               VideoCommon::CanUseSubmittedCommands(
            command_resume_offset != 0,
            dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                !std::ranges::binary_search(command_list.deferred_entries,
                                            static_cast<u32>(dma_pushbuffer_subindex)),
            gpu_written,
            use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                     header.size * sizeof(u32)));
        // FRITCHII_FRONT_SPLIT_COUNT: a GPU-written list goes down the list draw / dispatch
        // path below, which may ask the macro state (2a's join).
        if (gpu_written && !submitted && !use_safe && FrontSplit::g_state != 0) [[unlikely]] {
            FrontSplit::GpuListCheck();
        }
        if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Submitted);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Safe);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (gpu_written &&
                   (TakeListDraw(dma_state.dma_get, static_cast<u32>(header.size)) ||
                    TakeListDispatch(dma_state.dma_get, static_cast<u32>(header.size)))) {
            // Ported fix: nothing to parse, the words stay on the host GPU and the
            // call is complete. Before the copy read (Nx2ReadListCopy, the "list GPU wait") and
            // before EDEN_NX2_INDIRECT_GRID's placeholders, which then never see these lists.
        } else if (gpu_written && command_resume_offset == 0 && header.size == 3 &&
                   dma_state.method_count == 3 && dma_state.method >= MacroRegistersStart &&
                   subchannel_type[dma_state.subchannel] == Engines::EngineTypes::KeplerCompute &&
                   VideoCommon::GpuSplit::Get().Active() && Nx2IndirectGrid::Enabled()) {
            // A compute macro's (x, y, z) written by GPU work: placeholders the back resolves
            // on the GPU instead of waiting for it here (nx2_indirect_grid.h).
            const u32 slot = Nx2IndirectGrid::Reserve(dma_state.dma_get);
            if (static std::atomic<u32> logged{}; logged.fetch_add(1) < 24) {
                LOG_INFO(HW_GPU, "GPU-sized dispatch: macro {:#x} on ch {} from {:#x} (slot {})",
                         dma_state.method, channel_state.bind_id, dma_state.dma_get, slot);
            }
            command_headers.resize_destructive(3);
            for (u32 k = 0; k < 3; ++k) {
                command_headers[k].argument = Nx2IndirectGrid::Placeholder(slot, k);
            }
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Copy);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (gpu_written && command_resume_offset == 0 && header.size == 2 &&
                   dma_state.method_count == 2 && dma_state.method >= MacroRegistersStart &&
                   subchannel_type[dma_state.subchannel] == Engines::EngineTypes::Maxwell3D &&
                   VideoCommon::GpuSplit::Get().Active() && Nx2IndirectGrid::DrawsEnabled() &&
                   static_cast<Engines::Maxwell3D*>(subchannels[dma_state.subchannel])
                       ->FrontMacroCountsDraw(system, VideoCommon::GpuSplit::Get(),
                                              dma_state.method, 2)) {
            // A 3D macro's (count, unused) written by GPU work, which the macro copies into the
            // draw's vertex count: the back draws with the GPU's count (nx2_indirect_grid.h).
            const u32 slot = Nx2IndirectGrid::Reserve(dma_state.dma_get);
            command_headers.resize_destructive(2);
            for (u32 k = 0; k < 2; ++k) {
                command_headers[k].argument = Nx2IndirectGrid::DrawPlaceholder(slot, k);
            }
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Copy);
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (ReadGpuWrittenCommands(dma_state.dma_get, static_cast<u32>(header.size))) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            if (ListChangeTrace* const trace = ListChangeTrace::Get()) [[unlikely]] {
                const u32 sub = dma_state.subchannel;
                trace->Note(dma_state.dma_get, commands, dma_state.method, sub,
                            dma_state.method_count,
                            sub < subchannel_type.size() ? static_cast<u32>(subchannel_type[sub])
                                                         : ~0U);
            }
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Copy);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            FlushGpuWrittenCommands(dma_state.dma_get, size_t{header.size} * sizeof(u32));
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            RecordParsedEntry(header, commands);
            Nx2CmdPageDiag::CountEntry(Nx2CmdPageDiag::Path::Slow);
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
        if (puller.HasPendingAcquire()) {
            FrontSplit::NoteSuspension();
            return false;
        }
        command_resume_offset = 0;
        if (dma_state.method_count != 0) {
            split_method_end = header.addr + header.size * sizeof(u32);
        }
    }

    if (++dma_pushbuffer_subindex >= command_list_size) {
        // A split method resumes within the submission that split it; ring addresses repeat
        // every frame, so a stale one could otherwise swallow the headers of a later entry.
        for (const auto& parked : parked_methods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 16) {
                LOG_WARNING(HW_GPU,
                            "Submission ended with parked method {:#x} (sub {}, {} words left, "
                            "resume {:#x})",
                            parked.method, parked.subchannel, parked.method_count,
                            parked.resume_address);
            }
        }
        parked_methods.clear();
        PopFrontList();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        FrontSplit::StepRun();
        VideoCommon::GpuSplit::Get().Run([this] {
            rasterizer->SignalFence([this]() {
                std::scoped_lock lk(sync_mutex);
                synced = true;
                sync_cv.notify_all();
            });
        }, VideoCommon::DrainSite::GpSyncFence);
    }

    return true;
}

// EDEN_NX2_SEM_WATCH=<gpu address>: log every method whose argument is the low word of an
// address in the 16 bytes there (finds the engine and method that release a semaphore).
static const u32 g_semaphore_watch = [] {
    const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
    return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
}();

// EDEN_NX2_PB_CHECK also keeps the last parsed entries, for DumpCurrentEntry.
// Read on first use: nx2_switches.txt is applied in main(), after static initialisation.
static bool RecordParsedEntries() {
    static const bool record = std::getenv("EDEN_NX2_PB_CHECK") != nullptr;
    return record;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    // fable_stage_levelpoints 15.7 (profile 2): a GP entry's methods out of "push" (and out of
    // "resume pending" for a resumed channel's); "push" keeps the entry's fetch and checks.
    const VideoCommon::Nx2Profile::Scope methods_zone{
        VideoCommon::Nx2Profile::Zone::FrontMethods, VideoCommon::Nx2Profile::Fine(), false};
    // FRITCHII_FRONT_LIST_LANE bit 1 (live, default 0; fable_b 27.3): a 3D state register word
    // FrontWrite would drop at its same-value test is dropped here, before CallMethod,
    // FrontCallMethod and Front(); bit 4 verifies instead (the word goes the old way and a lane
    // drop that sent a ring entry is counted, must be 0). Decided once per chunk: the switch, the
    // split with front macros, no semaphore watch, the channel's 3D engine and its lane state
    // (Maxwell3D::FrontLaneReady); per word the engine's lane test (FrontLaneDrops).
    Engines::Maxwell3D* lane = nullptr;
    bool lane_verify = false;
    u32 lane_dropped = 0;
    u32 lane_checked = 0;
    u32 lane_differ = 0;
    u32 lane_missed = 0;
    if (const int lane_mode = VideoCore::Nx2Debug::SwitchValue(
            VideoCore::Nx2Debug::g_front_list_lane, "FRITCHII_FRONT_LIST_LANE");
        (lane_mode & 1) != 0 && g_semaphore_watch == 0) [[unlikely]] {
        auto& split = VideoCommon::GpuSplit::Get();
        if (split.Active() && split.FrontMacros()) {
            for (size_t sub = 0; sub < subchannels.size(); ++sub) {
                if (subchannels[sub] == nullptr ||
                    subchannel_type[sub] != Engines::EngineTypes::Maxwell3D) {
                    continue;
                }
                auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannels[sub]);
                if (maxwell3d->GetNativeMacroEngine().native_turing &&
                    maxwell3d->FrontLaneReady()) {
                    lane = maxwell3d;
                    lane_verify = (lane_mode & 4) != 0;
                }
                break;
            }
        }
    }
    SCOPE_EXIT {
        if (lane != nullptr) {
            lane->FrontLaneListDone(lane_dropped, lane_checked, lane_differ, lane_missed);
        }
    };
    // CallMethod, or the lane's drop for the 3D engine's state register words.
    const auto call_method = [&](u32 argument) {
        if (lane != nullptr && subchannels[dma_state.subchannel] == lane &&
            dma_state.method >= non_puller_methods) {
            if (!lane_verify) {
                if (lane->FrontLaneDrops(dma_state.method, argument)) {
                    ++lane_dropped;
                    return;
                }
            } else {
                const bool drops = lane->FrontLaneDrops(dma_state.method, argument);
                const auto& split = VideoCommon::GpuSplit::Get();
                const u64 position = split.WritePositionPeek();
                CallMethod(argument);
                const bool sent = split.WritePositionPeek() != position;
                ++lane_checked;
                if (drops && sent) [[unlikely]] {
                    ++lane_differ;
                } else if (!drops && !sent && dma_state.method < lane->front_lane_methods.size() &&
                           lane->front_lane_methods[dma_state.method] != 0) {
                    ++lane_missed;
                }
                return;
            }
        }
        CallMethod(argument);
    };
    // FRITCHII_FRONT_SPLIT_COUNT: the headers decoded (2a's chunks), passed on at the returns.
    u64 headers = 0;
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (g_nx2_ampere && dma_state.method < non_puller_methods) {
                max_write = 1; // a semaphore acquire may suspend the channel after any word
            }
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            // Fast-path batching for consecutive non-execution methods in increasing mode
            if (!dma_state.non_incrementing && !dma_increment_once &&
                dma_state.method >= non_puller_methods &&
                !VideoCommon::GpuSplit::Get().Active() &&
                g_semaphore_watch == 0) {
                auto subchannel = subchannels[dma_state.subchannel];
                if (subchannel != nullptr) {
                    const size_t available = std::min<size_t>(dma_state.method_count, commands.size() - index);
                    u32 batch_count = 0;
                    while (batch_count < available &&
                           (dma_state.method + batch_count) < subchannel->execution_mask.size() &&
                           !subchannel->execution_mask[dma_state.method + batch_count]) {
                        ++batch_count;
                    }
                    if (batch_count > 1) {
                        subchannel->method_sink.reserve(subchannel->method_sink.size() + batch_count);
                        for (u32 i = 0; i < batch_count; ++i) {
                            subchannel->method_sink.emplace_back(dma_state.method + i, commands[index + i].argument);
                        }
                        dma_state.method += batch_count;
                        dma_state.method_count -= batch_count;
                        index += batch_count;
                        continue;
                    }
                }
            }

            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            call_method(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            ++headers;
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                call_method(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingAcquire()) {
            VideoCommon::FrontSplit::NoteParsed(headers, index);
            return index;
        }
    }
    VideoCommon::FrontSplit::NoteParsed(headers, commands.size());
    return commands.size();
}

// NVN on Switch 2 ends GPFIFO entries in the middle of a method's data and calls another command
// buffer (e.g. a compute dispatch) before the entry that carries the rest, which starts right
// where the first one ended. An entry starting anywhere else is such a call: run it with a clean
// method state and resume the unfinished method when the stream continues at that address.
void DmaPusher::ResolveSplitMethod(GPUVAddr entry_address) {
    const auto continued = std::ranges::find(parked_methods, entry_address,
                                             &ParkedMethod::resume_address);
    std::optional<ParkedMethod> resume;
    if (continued != parked_methods.end()) {
        resume = *continued;
        parked_methods.erase(continued);
    }
    if (dma_state.method_count != 0 && entry_address != split_method_end) {
        if (parked_methods.size() == MaxParkedMethods) {
            static std::atomic<u32> reported{};
            if (reported.fetch_add(1) < 8) {
                const auto& oldest = parked_methods.front();
                LOG_WARNING(HW_GPU,
                            "Dropping parked method {:#x} (sub {}, {} words left, resume {:#x})",
                            oldest.method, oldest.subchannel, oldest.method_count,
                            oldest.resume_address);
            }
            parked_methods.erase(parked_methods.begin());
        }
        parked_methods.push_back(ParkedMethod{
            .method = dma_state.method,
            .subchannel = dma_state.subchannel,
            .method_count = dma_state.method_count,
            .non_incrementing = dma_state.non_incrementing,
            .increment_once = dma_increment_once,
            .resume_address = split_method_end,
        });
        dma_state.method_count = 0;
        dma_increment_once = false;
    }
    if (resume && dma_state.method_count == 0) {
        dma_state.method = resume->method;
        dma_state.subchannel = resume->subchannel;
        dma_state.method_count = resume->method_count;
        dma_state.non_incrementing = resume->non_incrementing;
        dma_increment_once = resume->increment_once;
    }
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}


void DmaPusher::WatchArgument(u32 method, u32 argument) const {
    const u32 watch = g_semaphore_watch;
    if ((argument & ~0xfU) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} subchannel {} method {:#x} argument {:#x}",
                 channel_state.bind_id, dma_state.subchannel, method, argument);
    }
}

bool DmaPusher::WantsSegment() const {
    // Only macro parameters and compute uploads read the engine's current_dma_segment.
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        WatchArgument(dma_state.method, argument);
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = split.Active() &&
                            WritesMemory(subchannel, engine_type, dma_state.method, 1);
        // Copies and blits are GPU work for the front's reads (GpuSplit::Effect).
        const auto effect = engine_type == Engines::EngineTypes::MaxwellDMA ||
                                    engine_type == Engines::EngineTypes::Fermi2D
                                ? VideoCommon::GpuSplit::Effect::Gpu
                                : VideoCommon::GpuSplit::Effect::Unknown;
        const bool memory_writer = split.Active() &&
                                   (engine_type == Engines::EngineTypes::KeplerMemory ||
                                    engine_type == Engines::EngineTypes::Fermi2D);
        // Whether the engine's front noted what the work sent writes (GpuSplit::NoteWriter).
        bool classified = false;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter(classified);
                if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                    g_last_copy_writer = split.SentPosition();
                }
            } else if (memory_writer) {
                NoteEngineWriter(split, memory_manager, subchannel, engine_type,
                                 dma_state.method, &argument, 1, dma_state.is_last_call, false);
            }
        };
        if (split.Active()) {
            namespace FrontSplit = VideoCommon::FrontSplit;
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            // FRITCHII_FRONT_SPLIT_COUNT: in 2a's queue; 2c would send 3D and compute methods
            // raw (a Method entry, one ring word).
            FrontSplit::NoteEngineCall(engine_type == Engines::EngineTypes::Maxwell3D ||
                                               engine_type == Engines::EngineTypes::KeplerCompute
                                           ? 1U
                                           : 0U);
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        const FrontSplit::StageScope stage{FrontSplit::Stage::State3D};
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        const FrontSplit::StageScope stage{FrontSplit::Stage::Compute};
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        classified = true;
                        return;
                    }
                }
            }
            {
                const FrontSplit::StageScope stage{FrontSplit::Stage::RingSend};
                split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                             WantsSegment() ? &segment : nullptr, effect);
            }
            if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                static_cast<Engines::MaxwellDMA*>(subchannel)->FrontMethod(
                    split, dma_state.method, &argument, 1);
                classified = true;
            }
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (g_semaphore_watch != 0) [[unlikely]] {
        for (u32 i = 0; i < num_methods; ++i) {
            WatchArgument(dma_state.method + (dma_state.non_incrementing ? 0 : i), base_start[i]);
        }
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        if (subchannel == nullptr) {
            ReportUnboundSubchannel();
            return;
        }
        const auto engine_type = subchannel_type[dma_state.subchannel];
        const bool writer = split.Active() &&
                            WritesMemory(subchannel, engine_type, dma_state.method,
                                         dma_state.non_incrementing ? 1 : num_methods);
        // Copies and blits are GPU work for the front's reads (GpuSplit::Effect).
        const auto effect = engine_type == Engines::EngineTypes::MaxwellDMA ||
                                    engine_type == Engines::EngineTypes::Fermi2D
                                ? VideoCommon::GpuSplit::Effect::Gpu
                                : VideoCommon::GpuSplit::Effect::Unknown;
        const bool memory_writer = split.Active() &&
                                   (engine_type == Engines::EngineTypes::KeplerMemory ||
                                    engine_type == Engines::EngineTypes::Fermi2D);
        // Whether the engine's front noted what the work sent writes (GpuSplit::NoteWriter).
        bool classified = false;
        SCOPE_EXIT {
            if (writer) {
                split.NoteWriter(classified);
                if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                    g_last_copy_writer = split.SentPosition();
                }
            } else if (memory_writer) {
                NoteEngineWriter(split, memory_manager, subchannel, engine_type,
                                 dma_state.method, base_start, num_methods,
                                 dma_state.is_last_call, true);
            }
        };
        if (split.Active()) {
            namespace FrontSplit = VideoCommon::FrontSplit;
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            // FRITCHII_FRONT_SPLIT_COUNT: in 2a's queue; 2c would send 3D and compute runs raw
            // (a Multi entry: two words, then the run two words to a ring word).
            FrontSplit::NoteEngineCall(engine_type == Engines::EngineTypes::Maxwell3D ||
                                               engine_type == Engines::EngineTypes::KeplerCompute
                                           ? 2U + (num_methods + 1U) / 2U
                                           : 0U);
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->GetNativeMacroEngine().native_turing) {
                        const FrontSplit::StageScope stage{FrontSplit::Stage::State3D};
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        const FrontSplit::StageScope stage{FrontSplit::Stage::Compute};
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        classified = true;
                        return;
                    }
                }
            }
            {
                const FrontSplit::StageScope stage{FrontSplit::Stage::RingSend};
                split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                                  dma_state.method_count, WantsSegment() ? &segment : nullptr,
                                  effect);
            }
            if (engine_type == Engines::EngineTypes::MaxwellDMA) {
                static_cast<Engines::MaxwellDMA*>(subchannel)->FrontMethod(
                    split, dma_state.method, base_start, num_methods);
                classified = true;
            }
            return;
        }
        VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

namespace {

/// FRITCHII_REFRESH_LEAN (live, default 0; helper CG): the words a list refresh copies again
/// (RefreshSnapshot after a resume, RefreshDeferredEntries after a map). Old: ReadBlockUnsafe per
/// entry, which translates the entry's page (GPU page table, then device to host) every time:
/// ~0.9% of the front sampled in GetPointer alone (v4 sample, 2026-11-15), with the entries of a
/// list mostly side by side on the same pages. 1 = each 4 KiB GPU page is translated once per
/// refresh and the entry copied from the host pointer (GetPointer of the page plus the offset,
/// which is what ReadBlockUnsafe's own one-page path reads); a page without a host pointer
/// (unmapped, sparse without memory, outside the address space) sends the entry the old way.
/// 2 = verify: the same copy into a scratch buffer, then the old copy into the list (the words
/// used are the old path's), every byte compared (differ must be 0). Line every 10 s while on.
struct RefreshLeanStats {
    std::chrono::steady_clock::time_point since{};
    u64 refreshes{};
    u64 entries{};
    u64 bytes{};
    u64 translated{}; ///< pages translated
    u64 reused{};     ///< page copies served by the refresh's last translation
    u64 old_way{};    ///< entries a page without a host pointer sent through ReadBlockUnsafe
    u64 compared{};
    u64 differ{};
    u64 differ_session{};
};
thread_local RefreshLeanStats t_refresh_lean;

int RefreshLeanMode() {
    if (VideoCommon::BackReadAudit::enabled) [[unlikely]] {
        return 0; // The audit sees every read the old way
    }
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_refresh_lean,
                                            "FRITCHII_REFRESH_LEAN");
}

/// The refresh's last translated 4 KiB GPU page.
struct RefreshPageCache {
    GPUVAddr page = ~GPUVAddr{0};
    const u8* host = nullptr;
};

/// Copies `bytes` at `address` page by page through `cache`. False: a page has no host pointer
/// (what was copied so far is left for the caller's old-way copy to overwrite).
bool CopyRefreshCached(const Tegra::MemoryManager& memory, RefreshPageCache& cache,
                       GPUVAddr address, u8* dest, size_t bytes, RefreshLeanStats& s) {
    constexpr GPUVAddr PageSize = 0x1000;
    while (bytes != 0) {
        const GPUVAddr page = address & ~(PageSize - 1);
        const size_t offset = static_cast<size_t>(address - page);
        const size_t chunk = (std::min)(bytes, static_cast<size_t>(PageSize) - offset);
        if (page != cache.page) {
            ++s.translated;
            cache.host = memory.GetPointer(page);
            if (cache.host == nullptr) {
                cache.page = ~GPUVAddr{0};
                return false;
            }
            cache.page = page;
        } else {
            ++s.reused;
        }
        std::memcpy(dest, cache.host + offset, chunk);
        dest += chunk;
        address += chunk;
        bytes -= chunk;
    }
    return true;
}

void MaybeLogRefreshLean(int mode) {
    RefreshLeanStats& s = t_refresh_lean;
    const auto now = std::chrono::steady_clock::now();
    if (s.since == std::chrono::steady_clock::time_point{}) {
        s.since = now;
        return;
    }
    const double seconds = std::chrono::duration<double>(now - s.since).count();
    if (seconds < 10.0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Refresh lean (FRITCHII_REFRESH_LEAN={}) {:.0f} s: refreshes {:.0f}/s, entries "
             "{:.0f}/s, {:.0f} KiB/s; pages translated {:.0f}/s, reused {:.0f}/s, entries the old "
             "way {:.0f}/s; verify: {:.0f} entries compared/s, {} differ (must be 0, {} this "
             "session)",
             mode, seconds, static_cast<double>(s.refreshes) / seconds,
             static_cast<double>(s.entries) / seconds,
             static_cast<double>(s.bytes) / 1024.0 / seconds,
             static_cast<double>(s.translated) / seconds, static_cast<double>(s.reused) / seconds,
             static_cast<double>(s.old_way) / seconds, static_cast<double>(s.compared) / seconds,
             s.differ, s.differ_session);
    const u64 differ_session = s.differ_session;
    s = RefreshLeanStats{};
    s.since = now;
    s.differ_session = differ_session;
}

} // namespace

void DmaPusher::RefreshEntryWords(CommandList& list, size_t i, int lean_mode, void* page_cache) {
    if (i >= list.command_lists.size() || i >= list.submitted_offsets.size()) {
        return;
    }
    const CommandListHeader& entry = list.command_lists[i];
    const u32 offset = list.submitted_offsets[i];
    if (offset + entry.size > list.submitted_words.size()) {
        return;
    }
    CommandHeader* const dest = list.submitted_words.data() + offset;
    const size_t bytes = static_cast<size_t>(entry.size) * sizeof(u32);
    if (lean_mode == 0) {
        memory_manager.ReadBlockUnsafe(entry.addr, dest, bytes);
        return;
    }
    RefreshLeanStats& s = t_refresh_lean;
    ++s.entries;
    s.bytes += bytes;
    auto& cache = *static_cast<RefreshPageCache*>(page_cache);
    if (lean_mode == 2) {
        thread_local std::vector<u8> scratch;
        scratch.resize(bytes);
        const bool cached =
            CopyRefreshCached(memory_manager, cache, entry.addr, scratch.data(), bytes, s);
        memory_manager.ReadBlockUnsafe(entry.addr, dest, bytes);
        if (!cached) {
            ++s.old_way;
            return;
        }
        ++s.compared;
        if (bytes != 0 && std::memcmp(scratch.data(), dest, bytes) != 0) {
            ++s.differ;
            if (++s.differ_session <= 8) {
                LOG_WARNING(HW_GPU,
                            "Refresh lean (FRITCHII_REFRESH_LEAN=2, must be 0): the entry of {} "
                            "words at {:#x} differs from ReadBlockUnsafe's copy",
                            static_cast<u32>(entry.size), static_cast<u64>(entry.addr));
            }
        }
        return;
    }
    if (!CopyRefreshCached(memory_manager, cache, entry.addr, reinterpret_cast<u8*>(dest), bytes,
                           s)) {
        ++s.old_way;
        memory_manager.ReadBlockUnsafe(entry.addr, dest, bytes);
    }
}

void DmaPusher::RefreshSnapshot(CommandList& list, u64 now, size_t first_entry) {
    list.snapshot_suspensions = now;
    if (VideoCommon::FrontSplit::g_state != 0) [[unlikely]] {
        ++list.refreshes; // FRITCHII_FRONT_SPLIT_COUNT's "re-read twice" (Step)
    }
    // FRITCHII_REFRESH_LEAN: one relaxed load per refresh while off.
    const int lean_mode = RefreshLeanMode();
    RefreshPageCache cache;
    if (lean_mode != 0) [[unlikely]] {
        ++t_refresh_lean.refreshes;
    }
    for (size_t i = first_entry; i < list.command_lists.size() && i < list.submitted_offsets.size();
         ++i) {
        RefreshEntryWords(list, i, lean_mode, &cache);
    }
    list.deferred_entries.clear();
    if (lean_mode != 0) [[unlikely]] {
        MaybeLogRefreshLean(lean_mode);
    }
}

void DmaPusher::RefreshDeferredEntries(CommandList& list) {
    const int lean_mode = RefreshLeanMode();
    RefreshPageCache cache;
    if (lean_mode != 0) [[unlikely]] {
        ++t_refresh_lean.refreshes;
    }
    for (const u32 i : list.deferred_entries) {
        RefreshEntryWords(list, i, lean_mode, &cache);
    }
    list.deferred_entries.clear();
    if (lean_mode != 0) [[unlikely]] {
        MaybeLogRefreshLean(lean_mode);
    }
}

bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            return true;
        }
    }
    return false;
}

bool DmaPusher::TakeListDraw(GPUVAddr address, u32 count) {
    namespace ListDraw = Tegra::Nx2ListDraw;
    if (ListDraw::Mode() == 0) {
        return false;
    }
    ListDraw::MaybeLog();
    auto& split = VideoCommon::GpuSplit::Get();
    // The two words are all the parameters of a macro call begun in the entry before, on one
    // mapped page.
    if (count != 2 || dma_state.method_count != 2 || command_resume_offset != 0 ||
        !split.Active() || !split.FrontMacros() || dma_state.method < MacroRegistersStart ||
        (dma_state.method & 1) != 0 || (address & 0xfff) > 0x1000 - 2 * sizeof(u32) ||
        subchannels[dma_state.subchannel] == nullptr ||
        subchannel_type[dma_state.subchannel] != Engines::EngineTypes::Maxwell3D ||
        !memory_manager.GpuToCpuAddress(address)) {
        return false;
    }
    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannels[dma_state.subchannel]);
    const Tegra::Nx2ListDispatch::ListReader reader{
        address, Tegra::Nx2ListDispatch::NextKeepId(),
        [](void* self, GPUVAddr list, u64 keep_id, std::span<u32> words) {
            return static_cast<DmaPusher*>(self)->ReadListWords(list, keep_id, words);
        },
        this};
    if (!maxwell3d->FrontTakeListDraw(system, split, dma_state.method, address, reader)) {
        ListDraw::Count(ListDraw::g_stats.refused);
        return false;
    }
    ListDraw::Count(ListDraw::g_stats.taken);
    dma_state.method_count = 0;
    dma_state.is_last_call = true;
    return true;
}

bool DmaPusher::TakeListDispatch(GPUVAddr address, u32 count) {
    namespace ListDispatch = Tegra::Nx2ListDispatch;
    if (ListDispatch::Mode() == 0) {
        return false;
    }
    ListDispatch::MaybeLog();
    auto& split = VideoCommon::GpuSplit::Get();
    // The three words are all the parameters of a compute macro call begun in the entry before,
    // on one mapped page.
    if (count != 3 || dma_state.method_count != 3 || command_resume_offset != 0 ||
        !split.Active() || !split.FrontMacros() || dma_state.method < MacroRegistersStart ||
        (dma_state.method & 1) != 0 || (address & 0xfff) > 0x1000 - 3 * sizeof(u32) ||
        subchannels[dma_state.subchannel] == nullptr ||
        subchannel_type[dma_state.subchannel] != Engines::EngineTypes::KeplerCompute ||
        !memory_manager.GpuToCpuAddress(address)) {
        return false;
    }
    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannels[dma_state.subchannel]);
    const ListDispatch::ListReader reader{
        address, ListDispatch::NextKeepId(),
        [](void* self, GPUVAddr list, u64 keep_id, std::span<u32> words) {
            return static_cast<DmaPusher*>(self)->ReadListWords(list, keep_id, words);
        },
        this};
    if (!compute->FrontRunsMacros() ||
        !compute->FrontTakeListDispatch(split, dma_state.method, address, reader)) {
        ListDispatch::Count(ListDispatch::g_stats.refused);
        return false;
    }
    ListDispatch::Count(ListDispatch::g_stats.taken);
    dma_state.method_count = 0;
    dma_state.is_last_call = true;
    return true;
}

bool DmaPusher::ReadListWords(GPUVAddr address, u64 keep_id, std::span<u32> words) {
    if (words.empty()) {
        return false;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    // The copy the back kept at the list's place in the stream: the words as the list had them,
    // whatever later work wrote there, and whether or not the list is still mapped.
    if (keep_id != 0 && rasterizer != nullptr) {
        bool kept = false;
        split.Run([&] { kept = rasterizer->Nx2ReadKeptListWords(keep_id, words); },
                  VideoCommon::DrainSite::ListFlush);
        if (kept) {
            return true;
        }
    }
    // No copy (the draw was not recorded): the list as it is now.
    if (!memory_manager.GpuToCpuAddress(address)) {
        return false;
    }
    // The work that writes the words is among what the front sent: the back runs it first.
    split.Drain(VideoCommon::DrainSite::ListFlush);
    FlushGpuWrittenCommands(address, words.size_bytes());
    memory_manager.ReadBlockUnsafe(address, words.data(), words.size_bytes());
    return true;
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, u32 count) {
    // The pages are marked as command list pages: from the next write on, the buffer cache copies
    // them to host memory right after the dispatch or copy that writes them and submits. The
    // front then waits only for the back to run that work and for the GPU to finish the copy, not
    // for everything queued and a cache flush (FlushGpuWrittenCommands: ~400 GPU round trips a
    // second in a Mario Kart World race, during which the back had nothing to do). Pages
    // FlushGpuWrittenCommands flushed whole and no GPU work wrote since are read from guest
    // memory; with EXP40 & 1 so are lists on words no GPU work wrote since the page was flushed,
    // or the host wrote since (BufferCache::Nx2ReadListCopy).
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || count == 0 ||
        rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + u64{count} * sizeof(u32);
    bool written = false;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && Tegra::Nx2IsGpuWritten(*device)) {
            Tegra::Nx2MarkCommandListPage(*device);
            written = true;
        }
    }
    if (!written) {
        return false;
    }
    // EXP40 & 2: as for entries on pages no GPU work wrote (Step), the dispatch or copy that
    // rewrites the list is ordered before the fetch only by a host-level wait between them, so
    // the front waits for the work before the last one (the "list copy writers" site), not for
    // compute and copy work sent after it. Else for every dispatch and copy sent so far.
    auto& split = VideoCommon::GpuSplit::Get();
    if (split.Active()) {
        if ((VideoCore::Nx2Debug::Experiment(40) & 2) != 0) {
            split.WaitForSyncedWriters(VideoCommon::DrainSite::ListCopies);
        } else {
            split.WaitForWriters();
        }
    }
    bool past_sync = !split.Active();
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    command_headers.resize_destructive(count);
    u8* const out = reinterpret_cast<u8*>(command_headers.data());
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        GpuWrittenList where = GpuWrittenList::Synced;
        if (Tegra::Nx2IsGpuWritten(*device)) {
            where = rasterizer->ReadGpuWrittenCommandList(*device, part, past_sync);
            if (where == GpuWrittenList::NeedsSync) {
                const auto wait_start = Nx2CmdPageDiag::g_enabled
                                            ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
                const bool waited = split.WaitForSync();
                if (Nx2CmdPageDiag::g_enabled && waited) {
                    Nx2CmdPageDiag::NoteSyncWait(std::chrono::steady_clock::now() - wait_start);
                }
                past_sync = true;
                where = rasterizer->ReadGpuWrittenCommandList(*device, part, true);
            }
            if (Nx2CmdPageDiag::g_enabled) {
                auto& diag = Nx2CmdPageDiag::Get();
                const u64 page = *device & ~u64{0xfff};
                if (where == GpuWrittenList::Copied) {
                    diag.reads_copied.fetch_add(1, std::memory_order_relaxed);
                } else if (where == GpuWrittenList::Synced) {
                    diag.reads_synced.fetch_add(1, std::memory_order_relaxed);
                    if (split.Active() && Nx2CmdPageDiag::IsDrawPage(page)) {
                        // The back, at this list's place: did work sent before it write the page?
                        split.Defer(
                            [page, begin = static_cast<u32>(*device - page),
                             end = static_cast<u32>(*device - page + part.size())] {
                                Nx2CmdPageDiag::CheckMissedWrite(page, begin, end);
                            },
                            VideoCommon::GpuSplit::Effect::Unknown,
                            VideoCommon::Nx2FastDrawCount::Neutral());
                    }
                } else {
                    diag.reads_unknown.fetch_add(1, std::memory_order_relaxed);
                    if (Nx2CmdPageDiag::IsDrawPage(page)) {
                        diag.reads_unknown_draw.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    if (Nx2ListVerify::Enabled()) [[unlikely]] {
        VerifyGpuWrittenCommands(address, count);
    }
    return true;
}

void DmaPusher::VerifyGpuWrittenCommands(GPUVAddr address, u32 count) {
    // The list again as the front read lists before EXP40: after every dispatch and copy sent so
    // far, from the pages' whole copies, or drained and flushed.
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    const size_t bytes = size_t{count} * sizeof(u32);
    std::vector<u32> parsed(count);
    std::memcpy(parsed.data(), command_headers.data(), bytes);
    auto& split = VideoCommon::GpuSplit::Get();
    if (split.Active()) {
        split.WaitForWriters();
    }
    bool past_sync = !split.Active();
    std::vector<u32> reference(count);
    bool reference_drained = false;
    {
        const Nx2ListVerify::ReferenceScope reference_read;
        u8* const out = reinterpret_cast<u8*>(reference.data());
        const GPUVAddr end = address + bytes;
        for (GPUVAddr chunk = address; chunk < end && !reference_drained;) {
            const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
            const std::span<u8> part{out + (chunk - address),
                                     static_cast<size_t>(chunk_end - chunk)};
            const auto device = memory_manager.GpuToCpuAddress(chunk);
            GpuWrittenList where = device ? GpuWrittenList::Synced : GpuWrittenList::Unknown;
            if (device && Tegra::Nx2IsGpuWritten(*device)) {
                where = rasterizer->ReadGpuWrittenCommandList(*device, part, past_sync);
                if (where == GpuWrittenList::NeedsSync) {
                    split.WaitForSync();
                    past_sync = true;
                    where = rasterizer->ReadGpuWrittenCommandList(*device, part, true);
                }
            }
            if (where == GpuWrittenList::Synced) {
                memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
            }
            reference_drained = where == GpuWrittenList::Unknown;
            chunk = chunk_end;
        }
    }
    const auto read_drained = [&](std::vector<u32>& words) {
        FlushGpuWrittenCommands(address, bytes);
        memory_manager.ReadBlockUnsafe(address, words.data(), bytes);
    };
    if (reference_drained) {
        read_drained(reference);
    }
    std::vector<u32> drained;
    if (!reference_drained && reference != parsed) {
        drained.resize(count);
        read_drained(drained);
    }
    Nx2ListVerify::NoteVerify(address, parsed, reference, drained, reference_drained);
    // The front parses the words read last.
    std::memcpy(command_headers.data(), drained.empty() ? reference.data() : drained.data(),
                bytes);
}

bool DmaPusher::FlushGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (!g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return false;
    }
    if (!IsGpuWrittenCommands(address, size)) {
        return false;
    }
    // The caches belong to the split's back: it first runs what the front queued (the dispatch
    // that writes these words among it). The whole pages are flushed: their next lists read
    // guest memory until GPU work writes them again.
    const bool diag = Nx2CmdPageDiag::g_enabled;
    const auto diag_start =
        diag ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<u32> diag_before;
    if (diag) {
        diag_before.resize(size / sizeof(u32));
        memory_manager.ReadBlockUnsafe(address, diag_before.data(), diag_before.size() * sizeof(u32));
    }
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::ListFlush);
    const GPUVAddr first_page = address & ~GPUVAddr{0xfff};
    const GPUVAddr end_page = (address + size + 0xfff) & ~GPUVAddr{0xfff};
    const bool dirty = memory_manager.IsMemoryDirty(first_page, end_page - first_page,
                                                    VideoCommon::CacheType::BufferCache);
    // The other pages draws wrote since their last flush go along in the same GPU round trip:
    // their next lists then read guest memory instead of draining and flushing again.
    // EDEN_NX2_CMDPAGE_BATCH=0 flushes this entry's pages only, as before.
    static const bool batch = [] {
        const char* value = std::getenv("EDEN_NX2_CMDPAGE_BATCH");
        return value == nullptr || std::string_view{value} != "0";
    }();
    if (batch && rasterizer != nullptr) {
        std::vector<DAddr> pages;
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            if (const auto device = memory_manager.GpuToCpuAddress(page)) {
                pages.push_back(*device);
            }
        }
        VideoCommon::GpuSplit::Get().Run(
            [rasterizer = rasterizer, pages = &pages] {
                rasterizer->FlushGpuWrittenCommandListPages(*pages);
            },
            VideoCommon::DrainSite::ListFlush);
    } else if (dirty) {
        VideoCommon::GpuSplit::Get().Run(
            [memory = &memory_manager, first_page, size = end_page - first_page] {
                memory->FlushRegion(first_page, size, VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::ListFlush);
    }
    if (diag) {
        auto kind = Nx2CmdPageDiag::Kind::NoWrite;
        Nx2CmdPageDiag::PageView first_view;
        u64 first_device{};
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            if (!device) {
                continue;
            }
            const u32 begin = static_cast<u32>((std::max)(address, page) - page);
            const u32 end = static_cast<u32>((std::min)(address + size, page + 0x1000) - page);
            const auto view = Nx2CmdPageDiag::TakePage(*device, begin, end);
            if (page == first_page) {
                first_view = view;
                first_device = *device;
            }
            kind = (std::max)(kind, view.kind);
        }
        std::vector<u32> after(diag_before.size());
        memory_manager.ReadBlockUnsafe(address, after.data(), after.size() * sizeof(u32));
        u64 changed{};
        for (size_t i = 0; i < after.size(); ++i) {
            changed += after[i] != diag_before[i] ? 1 : 0;
        }
        const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - diag_start)
                                            .count());
        Nx2CmdPageDiag::RecordSlow(kind, changed, dirty, ns, address, first_device,
                                   static_cast<u32>(address - first_page),
                                   static_cast<u32>((std::min)(address + size, first_page + 0x1000) -
                                                    first_page),
                                   first_view);
    }
    if (rasterizer != nullptr) {
        for (GPUVAddr page = first_page; page < end_page; page += 0x1000) {
            if (const auto device = memory_manager.GpuToCpuAddress(page);
                device && Tegra::Nx2IsCommandListPage(*device)) {
                rasterizer->NoteGpuWrittenCommandListSynced(*device);
            }
        }
    }
    return true;
}

void DmaPusher::ReportUnboundSubchannel() {
    // Methods sent to a subchannel with no engine (no SetObject yet) are dropped.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 8) {
        LOG_WARNING(HW_GPU, "Dropping method {:#x} on unbound subchannel {}", dma_state.method,
                    dma_state.subchannel);
    }
}

void DmaPusher::CheckSubmitHash(const CommandListHeader& header, u64 submit_hash) {
    std::vector<u32> words(header.size);
    memory_manager.ReadBlockUnsafe(header.addr, words.data(), words.size() * sizeof(u32));
    static std::atomic<u64> checked{};
    static std::atomic<u64> changed{};
    ++checked;
    if (HashPushbufferWords(words) != submit_hash) {
        if (changed.fetch_add(1) < 16) {
            LOG_WARNING(HW_GPU, "pushbuffer entry {:#x}+{:#x} changed between submit and execution",
                        header.addr.Value(), header.size.Value());
        }
    }
    if ((checked % 20000) == 0) {
        LOG_WARNING(HW_GPU, "pushbuffer check: {} of {} entries changed after submit",
                    changed.load(), checked.load());
    }
}

void DmaPusher::RecordParsedEntry(const CommandListHeader& header,
                                  std::span<const CommandHeader> commands) {
    if (!RecordParsedEntries()) {
        return;
    }
    u64 hash = 0xcbf29ce484222325ULL;
    for (const CommandHeader& command : commands) {
        hash = (hash ^ command.argument) * 0x100000001b3ULL;
    }
    parsed_entries[parsed_entry_count++ % parsed_entries.size()] = {header.addr, static_cast<u32>(header.size), hash};
}

void DmaPusher::DumpCurrentEntry(const char* reason) {
    // Entries whose memory changed after they were parsed (the CPU rewrote them meanwhile).
    for (size_t i = 0; i < std::min(parsed_entry_count, parsed_entries.size()); ++i) {
        const auto& entry = parsed_entries[(parsed_entry_count - 1 - i) % parsed_entries.size()];
        std::vector<u32> words(entry.size);
        memory_manager.ReadBlockUnsafe(entry.address, words.data(), words.size() * sizeof(u32));
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : words) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        LOG_WARNING(HW_GPU, "  parsed -{}: {:#x}+{:#x} {}", i, entry.address, entry.size,
                    hash == entry.hash ? "unchanged" : "CHANGED since parsed");
    }
    // EDEN_NX2_PB_DUMP=<file>: also write the whole command list (u64 address, u32 size and the
    // current words of every entry, then the current entry index) for offline replay.
    if (static const char* const path = std::getenv("EDEN_NX2_PB_DUMP");
        path != nullptr && !dma_pushbuffer.empty()) {
        if (std::FILE* file = std::fopen(path, "wb")) {
            const auto& lists = dma_pushbuffer.front().command_lists;
            const u64 current = dma_pushbuffer_subindex;
            const u64 count = lists.size();
            std::fwrite(&count, sizeof(count), 1, file);
            std::fwrite(&current, sizeof(current), 1, file);
            for (const CommandListHeader& header : lists) {
                const u64 address = header.addr;
                const u32 size = static_cast<u32>(header.size);
                std::vector<u32> words(size);
                memory_manager.ReadBlockUnsafe(address, words.data(), words.size() * sizeof(u32));
                std::fwrite(&address, sizeof(address), 1, file);
                std::fwrite(&size, sizeof(size), 1, file);
                std::fwrite(words.data(), sizeof(u32), words.size(), file);
            }
            std::fclose(file);
        }
    }
    LOG_WARNING(HW_GPU, "pushbuffer dump ({}): method={:#x} sub={} count={} non_inc={} get={:#x}",
                reason, dma_state.method, dma_state.subchannel, dma_state.method_count,
                dma_state.non_incrementing, dma_state.dma_get);
    if (dma_pushbuffer.empty()) {
        return;
    }
    const auto& lists = dma_pushbuffer.front().command_lists;
    const size_t first = dma_pushbuffer_subindex >= 2 ? dma_pushbuffer_subindex - 2 : 0;
    for (size_t i = first; i <= dma_pushbuffer_subindex && i < lists.size(); ++i) {
        const CommandListHeader& header = lists[i];
        LOG_WARNING(HW_GPU, "  entry {} of {}: addr={:#x} size={:#x}", i, lists.size(),
                    header.addr.Value(), header.size.Value());
        for (u64 word = 0; word < std::min<u64>(header.size, 0x80); word += 8) {
            std::string gpu_line;
            std::string cpu_line;
            for (u64 j = word; j < std::min<u64>(header.size, word + 8); ++j) {
                const GPUVAddr address = header.addr + j * sizeof(u32);
                const u32 gpu_value = memory_manager.Read<u32>(address);
                const u32 cpu_value = system.ApplicationMemory().Read32(address);
                gpu_line += fmt::format(" {:08x}", gpu_value);
                cpu_line += gpu_value == cpu_value ? std::string(" ........")
                                                   : fmt::format(" {:08x}", cpu_value);
            }
            LOG_WARNING(HW_GPU, "    +{:03x}:{}", word * 4, gpu_line);
            LOG_WARNING(HW_GPU, "    cpu :{}", cpu_line);
        }
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

} // namespace Tegra
