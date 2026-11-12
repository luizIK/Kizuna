// SPDX-FileCopyrightText: Copyright 2026 Kizuna
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// Kizuna: the driver's compiled pipelines kept by the emulator (VK_KHR_pipeline_binary).
///
/// AMD's driver keeps its own cache of compiled pipelines in ~1 GiB, so the pipelines a session
/// built were gone by the next one: every boot compiled all of the pipelines earlier sessions had
/// used again (4,858 in 449 s on a Ryzen 5 5600 and RX 6600, 2026-11-12, more each session), and
/// play compiled pipelines it had compiled before, freezing for each. Here every pipeline created
/// through vk::Device is created with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR, its binaries are
/// read back and appended to one file, keyed by the driver's key for its create info; the next
/// creation with the same key builds the pipeline from those binaries instead of compiling.
///
/// The file starts with the driver's global key: another driver (or driver version) starts it
/// over. A record cut short (the app killed while writing) ends the file there. At most 2 GiB are
/// kept. EDEN_NX2_PIPELINE_BINARIES=0 turns it off, =1 turns it on for any driver (on by default
/// for AMD's own Windows driver only). Pipelines created on a thread with capture off (the shader
/// preparation's) are looked up but not kept.
class PipelineBinaryCache {
public:
    /// Whether the extension is wanted for this driver (EDEN_NX2_PIPELINE_BINARIES).
    [[nodiscard]] static bool Wanted(VkDriverIdKHR driver_id);

    /// The cache for `device`, its file at `path`; nullptr when the driver's global key cannot be
    /// read. Installed for vk::Device's pipeline creation until destroyed.
    [[nodiscard]] static std::unique_ptr<PipelineBinaryCache> Create(
        VkDevice device, const vk::DeviceDispatch& dld, const std::filesystem::path& path);

    /// The installed cache of `device`, or nullptr.
    [[nodiscard]] static PipelineBinaryCache* Get(VkDevice device) noexcept;

    /// Off on the calling thread: pipelines it creates are looked up but not kept (the shader
    /// preparation's threads, whose 120k pipelines would not fit).
    static void SetCaptureOnThisThread(bool capture) noexcept;

    ~PipelineBinaryCache();
    PipelineBinaryCache(const PipelineBinaryCache&) = delete;
    PipelineBinaryCache& operator=(const PipelineBinaryCache&) = delete;

    /// vkCreateGraphicsPipelines / vkCreateComputePipelines for one pipeline, through the cache.
    VkResult CreateGraphics(const VkGraphicsPipelineCreateInfo& ci, VkPipelineCache cache,
                            VkPipeline* pipeline);
    VkResult CreateCompute(const VkComputePipelineCreateInfo& ci, VkPipelineCache cache,
                           VkPipeline* pipeline);

private:
    struct Key {
        u32 size{};
        std::array<u8, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR> bytes{};
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };
    struct Binary {
        Key key;
        std::vector<u8> data;
    };
    /// A record read back: the keys and data point into `payload`.
    struct Loaded {
        std::vector<u8> payload;
        std::vector<VkPipelineBinaryKeyKHR> keys;
        std::vector<VkPipelineBinaryDataKHR> data;
    };

    PipelineBinaryCache(VkDevice device, const vk::DeviceDispatch& dld,
                        std::filesystem::path path, Key global_key);

    bool Open();
    bool ReadRecord(u64 offset, Loaded& loaded);
    void Append(const Key& key, const std::vector<Binary>& binaries);
    bool Capture(VkPipeline pipeline, std::vector<Binary>& binaries);
    void ReportMaybe(bool force);

    template <typename CreateInfo, typename CreateFn>
    VkResult CreateThrough(const CreateInfo& ci, VkPipelineCache cache, VkPipeline* pipeline,
                           CreateFn&& create);

    VkDevice device;
    const vk::DeviceDispatch& dld;
    std::filesystem::path path;
    Key global_key;

    std::shared_mutex index_mutex;
    std::unordered_map<Key, u64, KeyHash> index; ///< record offsets
    std::mutex file_mutex;
    std::fstream file;
    u64 file_end{};
    bool stopped_writing{};

    std::atomic<u64> hits{};
    std::atomic<u64> captured{};
    std::atomic<u64> failed_hits{};
    std::atomic<u64> uncaptured{};
    std::atomic<u64> creations{};
    std::atomic<s64> next_report{};
    std::atomic<u32> failures_logged{};
};

} // namespace Vulkan
