// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "common/alignment.h"
#include "common/common_types.h"

namespace Vulkan {

/// Where the staging ring (FRITCHII_STAGING_RING) puts each upload. Ported from DahBard/Blitchii's
/// StagingRingRegions (r64). Like the stream buffer, each of its regions is reused once the GPU is
/// past the stamp it was last handed out with.
class StagingRingRegions {
public:
    static constexpr size_t NUM_REGIONS = 16;
    static constexpr size_t ALIGNMENT = 256;

    /// `size_` is a multiple of NUM_REGIONS * ALIGNMENT.
    explicit StagingRingRegions(size_t size_) : size{size_}, region_size{size_ / NUM_REGIONS} {}

    /// The offset of `bytes` handed out with `stamp`, or nothing while the GPU may still read a
    /// region they need (`is_free(stamp)` says whether it is past a stamp). Up to half the ring
    /// goes wherever its regions are free; more, up to all of it, only when every region is.
    template <typename IsFree>
    [[nodiscard]] std::optional<size_t> Take(size_t bytes, u64 stamp, IsFree&& is_free) {
        if (bytes == 0 || bytes > size || (bytes > size / 2 && !AllFree(is_free))) {
            return std::nullopt;
        }
        // A region is checked when a lap first enters it: the GPU must be past its last use in
        // the lap before. The regions the lap is already in are this lap's.
        size_t offset = Common::AlignUp(head, ALIGNMENT);
        size_t first_new = head == 0 ? 0 : (head - 1) / region_size + 1;
        if (offset + bytes > size) {
            offset = 0;
            first_new = 0;
        }
        const size_t last = (offset + bytes - 1) / region_size;
        for (size_t region = first_new; region <= last; ++region) {
            if (stamps[region] != 0 && !is_free(stamps[region])) {
                return std::nullopt;
            }
        }
        for (size_t region = offset / region_size; region <= last; ++region) {
            stamps[region] = stamp;
        }
        last_first = offset / region_size;
        last_last = last;
        head = offset + bytes;
        if (bytes > size / 2) {
            large_stamp = stamp;
        }
        return offset;
    }

    /// The regions of the last upload take `stamp` too (a later one): its copy may be recorded
    /// after the stamp it was handed out with, as with the stream buffer's lazy stamps.
    void Restamp(u64 stamp) {
        if (last_last < last_first) {
            return;
        }
        for (size_t region = last_first; region <= last_last; ++region) {
            stamps[region] = (std::max)(stamps[region], stamp);
        }
    }

    /// Whether the GPU may still read an upload over half the ring.
    template <typename IsFree>
    [[nodiscard]] bool LargeUploadInFlight(IsFree&& is_free) const {
        return large_stamp != 0 && !is_free(large_stamp);
    }

    template <typename IsFree>
    [[nodiscard]] bool AllFree(IsFree&& is_free) const {
        for (const u64 stamp : stamps) {
            if (stamp != 0 && !is_free(stamp)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] size_t Size() const {
        return size;
    }

    [[nodiscard]] size_t RegionSize() const {
        return region_size;
    }

private:
    size_t size;
    size_t region_size;
    size_t head{}; ///< The next free byte of this lap
    std::array<u64, NUM_REGIONS> stamps{};
    u64 large_stamp{}; ///< The stamp of the last upload over half the ring
    size_t last_first{1};
    size_t last_last{0};
};

} // namespace Vulkan
