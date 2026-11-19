// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/range_sets.h"
#include "common/range_sets.inc"
#include "video_core/buffer_cache/gpu_written_mirror.h"

namespace VideoCommon {

class ReadTrapTracker {
public:
    /// Returns whether the core must be notified to arm this page-aligned range.
    bool Arm(DAddr address, u64 size) {
        u64 armed_bytes = 0;
        ranges.ForEachInRange(address, size, [&](DAddr lower, DAddr upper) {
            armed_bytes += upper - lower;
        });
        if (armed_bytes == size) {
            return false;
        }
        ranges.Add(address, size);
        return true;
    }

    /// A CPU read consumes the core's trap; unmapping also forgets the range.
    void Subtract(DAddr address, u64 size) {
        ranges.Subtract(address, size);
    }

    /// Whether any armed byte lies in [address, address + size) (EXP86's verify).
    [[nodiscard]] bool Intersects(DAddr address, u64 size) const {
        return ranges.Intersects(address, size);
    }

    /// EXP86 (fable52_answer_gpuwritten 2.1): keep `bitmap` equal to "a byte armed on the page".
    void AttachMirror(PageBitmap* bitmap, MirrorSeq* seq) {
        ranges.Attach(bitmap, seq);
    }

private:
    MirroredRangeSet<DAddr> ranges;
};

} // namespace VideoCommon
