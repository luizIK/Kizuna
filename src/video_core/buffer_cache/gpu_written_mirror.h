// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <memory>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/logging.h"
#include "common/range_sets.h"
#include "common/range_sets.inc"
#include "common/spin_recursive_mutex.h"
#include "video_core/mirror_seq.h"

namespace VideoCommon {

/// One bit per 4 KiB device page over the 36-bit device space (PageFilter's shape, 2 MiB), set
/// and cleared under the buffer cache's mutex, read without it.
class PageBitmap {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    static constexpr u64 NumPages = u64{1} << (36 - PageBits);
    static constexpr u64 NumWords = NumPages / 64;

    PageBitmap() : words{std::make_unique<std::atomic<u64>[]>(NumWords)} {}

    /// Sets the bits of pages [first, last].
    void SetPages(u64 first, u64 last, MirrorSeq& seq) {
        if (last >= NumPages) {
            seq.Break();
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                seq.BeforeChange();
                word.fetch_or(mask);
            }
            page += count;
        }
    }

    /// Clears the bits of pages [first, last].
    void ClearPages(u64 first, u64 last, MirrorSeq& seq) {
        if (last >= NumPages) {
            seq.Break();
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != 0) {
                seq.BeforeChange();
                word.fetch_and(~mask);
            }
            page += count;
        }
    }

    /// Sets the bits of the pages [address, address + size) touches.
    void Set(u64 address, u64 size, MirrorSeq& seq) {
        if (size == 0) {
            return;
        }
        SetPages(address >> PageBits, (address + size - 1) >> PageBits, seq);
    }

    /// Clears the bits of the pages [address, address + size) touches.
    void Clear(u64 address, u64 size, MirrorSeq& seq) {
        if (size == 0) {
            return;
        }
        ClearPages(address >> PageBits, (address + size - 1) >> PageBits, seq);
    }

    /// Whether a page [address, address + size) touches has its bit set. The caller checks the
    /// range is inside the bitmap (InRange) first: out of it, this answers false.
    [[nodiscard]] bool Any(u64 address, u64 size) const {
        if (size == 0) {
            return false;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumPages) {
            return false;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            if (words[page / 64].load() & mask) {
                return true;
            }
            page += count;
        }
        return false;
    }

    [[nodiscard]] static bool InRange(u64 address, u64 size) {
        return size != 0 && address + size > address && ((address + size - 1) >> PageBits) < NumPages;
    }

private:
    std::unique_ptr<std::atomic<u64>[]> words;
};

/// EXP81 (fable52_answer_fences2 2): one bit per 4 KiB device page the guest read since the
/// last fold, PageBitmap's shape. Set without any lock by both read paths (the locked
/// GetFlushArea and EXP86's lock-free clean answer), taken under the buffer cache's mutex by
/// TickFrame. No sequence counter: a bit seen one fold late only stamps its page a frame later,
/// and an unmark that comes too early costs one on-demand flush.
class ReadPageBitmap {
public:
    static constexpr u64 PageBits = PageBitmap::PageBits;
    static constexpr u64 NumPages = PageBitmap::NumPages;
    static constexpr u64 NumWords = PageBitmap::NumWords;

    ReadPageBitmap() : words{std::make_unique<std::atomic<u64>[]>(NumWords)} {}

    /// Sets the bits of the pages [address, address + size) touches (out of range: nothing).
    void Set(u64 address, u64 size) {
        if (size == 0 || address + size < address) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (address + size - 1) >> PageBits;
        if (last >= NumPages) {
            return;
        }
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != mask) {
                word.fetch_or(mask, std::memory_order_relaxed);
            }
            page += count;
        }
    }

    /// Clears the set bits of the pages [address, address + size) touches and calls
    /// func(page address) for each of them.
    template <typename Func>
    void Take(u64 address, u64 size, Func&& func) {
        if (size == 0 || address + size < address) {
            return;
        }
        const u64 first = address >> PageBits;
        const u64 last = (std::min<u64>)((address + size - 1) >> PageBits, NumPages - 1);
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min<u64>)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : (u64{1} << count) - 1) << bit;
            auto& word = words[page / 64];
            if ((word.load(std::memory_order_relaxed) & mask) != 0) {
                u64 taken = word.fetch_and(~mask, std::memory_order_relaxed) & mask;
                while (taken != 0) {
                    const u64 index = static_cast<u64>(std::countr_zero(taken));
                    func(((page / 64) * 64 + index) << PageBits);
                    taken &= taken - 1;
                }
            }
            page += count;
        }
    }

private:
    std::unique_ptr<std::atomic<u64>[]> words;
};

/// FRITCHII_STORE_FOOTPRINT: told about every Add, Subtract and Clear of the range set it follows
/// (MirroredRangeSet::SetFollower), so a second record of who made the bytes GPU-modified
/// (BufferCache's FootprintShadow) can never hold bytes the set no longer holds.
class RangeSetFollower {
public:
    /// An Add without a footprint claim (MirroredRangeSet::Add with claimed == false)
    virtual void FollowAdd(u64 begin, u64 size) = 0;
    virtual void FollowSubtract(u64 begin, u64 size) = 0;
    virtual void FollowClear() = 0;

protected:
    ~RangeSetFollower() = default;
};

/// A Common::RangeSet whose Add, Subtract and Clear keep a PageBitmap equal to "the set holds a
/// byte on this page" (fable52_answer_gpuwritten 2.5): Add sets every page it touches; Subtract
/// clears the pages wholly inside it and asks the set about the two edge pages. Without a bitmap
/// attached (tests, other users) it is the plain range set.
template <typename AddressType>
class MirroredRangeSet {
public:
    void Attach(PageBitmap* bitmap_, MirrorSeq* seq_) {
        bitmap = bitmap_;
        seq = seq_;
    }

    /// FRITCHII_STORE_FOOTPRINT: the follower from now on (null: none). Its caller hands it the
    /// set's current ranges first.
    void SetFollower(RangeSetFollower* follower_) {
        follower = follower_;
    }

    /// `claimed`: FRITCHII_STORE_FOOTPRINT, the caller recorded the range's footprint claim with
    /// the follower itself (it is not told about this Add).
    void Add(AddressType base_address, size_t size, bool claimed = false) {
        if (follower != nullptr && !claimed) [[unlikely]] {
            follower->FollowAdd(static_cast<u64>(base_address), size);
        }
        ranges.Add(base_address, size);
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (bitmap == nullptr || end_address <= base_address) {
            return;
        }
        bitmap->Set(base_address, end_address - base_address, *seq);
    }

    void Subtract(AddressType base_address, size_t size) {
        if (follower != nullptr) [[unlikely]] {
            follower->FollowSubtract(static_cast<u64>(base_address), size);
        }
        ranges.Subtract(base_address, size);
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (bitmap == nullptr || end_address <= base_address) {
            return;
        }
        const u64 begin = static_cast<u64>(base_address);
        const u64 end = static_cast<u64>(end_address);
        const u64 first_page = begin >> PageBitmap::PageBits;
        const u64 last_page = (end - 1) >> PageBitmap::PageBits;
        // Pages wholly inside the subtracted range hold nothing of the set now.
        const u64 inner_first = Common::AlignUp(begin, PageBitmap::PageSize) >> PageBitmap::PageBits;
        const u64 inner_end = end >> PageBitmap::PageBits; // first page not wholly inside
        if (inner_first < inner_end) {
            bitmap->ClearPages(inner_first, inner_end - 1, *seq);
        }
        // The edge pages keep their bit only where the set still has a byte on them.
        const auto check_edge = [&](u64 page) {
            if (!ranges.Intersects(static_cast<AddressType>(page << PageBitmap::PageBits),
                                   PageBitmap::PageSize)) {
                bitmap->ClearPages(page, page, *seq);
            }
        };
        if ((begin & (PageBitmap::PageSize - 1)) != 0) {
            check_edge(first_page);
        }
        if ((end & (PageBitmap::PageSize - 1)) != 0 &&
            !(last_page == first_page && (begin & (PageBitmap::PageSize - 1)) != 0)) {
            check_edge(last_page);
        }
    }

    void Clear() {
        if (follower != nullptr) [[unlikely]] {
            follower->FollowClear();
        }
        if (bitmap != nullptr) {
            ranges.ForEach([&](AddressType lower, AddressType upper) {
                bitmap->Clear(lower, upper - lower, *seq);
            });
        }
        ranges.Clear();
    }

    [[nodiscard]] bool Empty() const {
        return ranges.Empty();
    }

    [[nodiscard]] bool Intersects(AddressType base_address, size_t size) const {
        return ranges.Intersects(base_address, size);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        ranges.ForEach(std::forward<Func>(func));
    }

    template <typename Func>
    void ForEachInRange(AddressType device_addr, size_t size, Func&& func) const {
        ranges.ForEachInRange(device_addr, size, std::forward<Func>(func));
    }

private:
    Common::RangeSet<AddressType> ranges;
    PageBitmap* bitmap = nullptr;
    MirrorSeq* seq = nullptr;
    RangeSetFollower* follower = nullptr;
};

/// The three page mirrors of the buffer cache (fable52_answer_gpuwritten 2.1) and their counter.
struct GpuWrittenMirror {
    explicit GpuWrittenMirror(const Common::SpinRecursiveMutex& mutex) : seq{mutex} {}

    MirrorSeq seq;
    PageBitmap gpu_modified_pages; ///< gpu_modified_ranges has a byte on the page
    PageBitmap preflush_pages;     ///< the memory tracker's Preflushable bit
    PageBitmap trap_pages;         ///< read_trapped_ranges has a byte on the page
};

/// What the mirrors answer for a guest read (BufferCache::MirrorCleanForRead).
enum class MirrorAnswer {
    Clean,         ///< the locked path would answer preemptive and change nothing
    Busy,          ///< the counter was odd or moved: take the locked path
    Modified,      ///< a GPU-modified byte on the pages
    ArmedUnmarked, ///< clean, but the locked path would forget an armed trap (no preflush mark)
    Broken,        ///< the mirror can't answer (out of range, or broken for good)
};

} // namespace VideoCommon
