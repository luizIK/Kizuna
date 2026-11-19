// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <cstddef>
#include <utility>
#include "common/slot_vector.h"

namespace VideoCommon {

template <class Image>
struct PendingImageDownload {
    bool is_swizzle;
    size_t async_buffer_id;
    Common::SlotId object_id;
    std::shared_ptr<Image> retired_image{};
    /// The GPU tick current when the retired image was deleted (0 without one, or on OpenGL).
    u64 retired_tick{};

    Image& GetImage(Common::SlotVector<Image>& images) const {
        return retired_image ? *retired_image : images[object_id];
    }
};

// Whole-image copies are recorded together at commit time. Repeated requests
// for the same live image therefore observe the same contents. Keep the first
// occurrence so reverse CPU writeback order remains unchanged for aliases.
template <class Image, class Downloads>
size_t DeduplicatePendingImageDownloads(Downloads& downloads, Common::SlotVector<Image>& images) {
    std::unordered_set<const Image*> seen;
    const size_t previous_size = downloads.size();
    const auto end = std::remove_if(downloads.begin(), downloads.end(), [&](const auto& download) {
        return download.is_swizzle && !seen.insert(&download.GetImage(images)).second;
    });
    downloads.erase(end, downloads.end());
    return previous_size - downloads.size();
}

// Cache slots may be recycled before queued GPU downloads are submitted or
// consumed. Keep the actual image, including its metadata, until those finish.
// Returns the kept image (null when no download names it). A download lands once the GPU passed
// its own copy, which can be before the GPU is past work recorded after that copy (the image
// drawn to or read again, then deleted at `tick`): the caller holds the returned reference until
// the GPU passes `tick` (Fritchii's free guard), so the landing never frees it in use.
template <class Image, class Downloads, class Batches>
std::shared_ptr<Image> RetainPendingImageDownloads(Common::SlotId id, Image& image,
                                                   Downloads& pending, Batches& submitted,
                                                   u64 tick) {
    std::shared_ptr<Image> retained;
    const auto retain = [&](auto& downloads) {
        for (auto& download : downloads) {
            if (!download.is_swizzle || download.object_id != id || download.retired_image) continue;
            if (!retained) retained = std::make_shared<Image>(std::move(image));
            download.retired_image = retained;
            download.retired_tick = tick;
        }
    };
    retain(pending);
    for (auto& downloads : submitted) retain(downloads);
    return retained;
}

} // namespace VideoCommon
