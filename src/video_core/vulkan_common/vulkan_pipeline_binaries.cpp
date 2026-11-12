// SPDX-FileCopyrightText: Copyright 2026 Kizuna
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/vulkan_common/vulkan_pipeline_binaries.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <system_error>

#include "common/cityhash.h"
#include "common/logging.h"

namespace Vulkan {
namespace {

constexpr std::array<char, 8> FILE_MAGIC{'K', 'Z', 'P', 'B', 'I', 'N', '0', '1'};
constexpr u32 RECORD_MAGIC = 0x52505A4B; // "KZPR"
constexpr u64 MAX_FILE_BYTES = 2ULL << 30;
constexpr u32 MAX_BINARIES = 64;
constexpr u64 MAX_PAYLOAD = 256ULL << 20;
constexpr u32 KEY_CAPACITY = VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;

struct FileHeader {
    std::array<char, 8> magic;
    u32 key_size;
    u32 reserved;
    std::array<u8, KEY_CAPACITY> key;
};
static_assert(sizeof(FileHeader) == 48);

struct RecordHeader {
    u32 magic;
    u32 key_size;
    std::array<u8, KEY_CAPACITY> key;
    u32 binary_count;
    u32 reserved;
    u64 payload_size;
    u64 payload_hash;
};
static_assert(sizeof(RecordHeader) == 64);

struct BinaryHeader {
    u32 key_size;
    u32 reserved;
    std::array<u8, KEY_CAPACITY> key;
    u64 data_size;
};
static_assert(sizeof(BinaryHeader) == 48);

std::atomic<PipelineBinaryCache*> g_cache{};
thread_local bool t_no_capture = false;

constexpr VkPipelineCreateFlags INELIGIBLE_FLAGS =
    VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_DERIVATIVE_BIT |
    VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
    VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT |
    VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT |
    VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
    VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

/// Pipelines linked from libraries, or already carrying flags2 or binaries, go to the driver as
/// they are.
template <typename CreateInfo>
bool Eligible(const CreateInfo& ci) {
    if ((ci.flags & INELIGIBLE_FLAGS) != 0) {
        return false;
    }
    for (auto* next = static_cast<const VkBaseInStructure*>(ci.pNext); next; next = next->pNext) {
        switch (next->sType) {
        case VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR:
        case VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT:
        case VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR:
        case VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR:
            return false;
        default:
            break;
        }
    }
    return true;
}

s64 NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // Anonymous namespace

size_t PipelineBinaryCache::KeyHash::operator()(const Key& key) const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(key.bytes.data()), key.size));
}

bool PipelineBinaryCache::Wanted([[maybe_unused]] VkDriverIdKHR driver_id) {
    if (const char* value = std::getenv("EDEN_NX2_PIPELINE_BINARIES"); value && *value) {
        return *value != '0';
    }
#ifdef _WIN32
    return driver_id == VK_DRIVER_ID_AMD_PROPRIETARY;
#else
    return false;
#endif
}

std::unique_ptr<PipelineBinaryCache> PipelineBinaryCache::Create(
    VkDevice device, const vk::DeviceDispatch& dld, const std::filesystem::path& path) {
    if (!dld.vkCreatePipelineBinariesKHR || !dld.vkDestroyPipelineBinaryKHR ||
        !dld.vkGetPipelineKeyKHR || !dld.vkGetPipelineBinaryDataKHR ||
        !dld.vkReleaseCapturedPipelineDataKHR) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: the driver lacks the extension's functions");
        return nullptr;
    }
    if (g_cache.load(std::memory_order_acquire) != nullptr) {
        // Two caches would append to the same file over each other.
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: another device keeps them already");
        return nullptr;
    }
    VkPipelineBinaryKeyKHR global{};
    global.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
    const VkResult result = dld.vkGetPipelineKeyKHR(device, nullptr, &global);
    if (result != VK_SUCCESS || global.keySize == 0 || global.keySize > KEY_CAPACITY) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: no global key from the driver ({})",
                    static_cast<int>(result));
        return nullptr;
    }
    Key key;
    key.size = global.keySize;
    std::memcpy(key.bytes.data(), global.key, global.keySize);
    std::unique_ptr<PipelineBinaryCache> cache{
        new PipelineBinaryCache(device, dld, path, key)};
    if (!cache->Open()) {
        return nullptr;
    }
    g_cache.store(cache.get(), std::memory_order_release);
    return cache;
}

PipelineBinaryCache* PipelineBinaryCache::Get(VkDevice device) noexcept {
    PipelineBinaryCache* const cache = g_cache.load(std::memory_order_acquire);
    return cache && cache->device == device ? cache : nullptr;
}

void PipelineBinaryCache::SetCaptureOnThisThread(bool capture) noexcept {
    t_no_capture = !capture;
}

PipelineBinaryCache::PipelineBinaryCache(VkDevice device_, const vk::DeviceDispatch& dld_,
                                         std::filesystem::path path_, Key global_key_)
    : device{device_}, dld{dld_}, path{std::move(path_)}, global_key{global_key_} {}

PipelineBinaryCache::~PipelineBinaryCache() {
    PipelineBinaryCache* self = this;
    g_cache.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
    ReportMaybe(true);
}

bool PipelineBinaryCache::Open() {
    const auto begin = std::chrono::steady_clock::now();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    u64 size = 0;
    bool fresh = true;
    const char* started_over = nullptr;
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            in.seekg(0, std::ios::end);
            size = static_cast<u64>(in.tellg());
            in.seekg(0);
            FileHeader header{};
            if (size >= sizeof(header) &&
                in.read(reinterpret_cast<char*>(&header), sizeof(header)) &&
                header.magic == FILE_MAGIC && header.key_size == global_key.size &&
                std::memcmp(header.key.data(), global_key.bytes.data(), global_key.size) == 0) {
                fresh = false;
                u64 offset = sizeof(header);
                RecordHeader record{};
                while (offset + sizeof(record) <= size) {
                    in.seekg(static_cast<std::streamoff>(offset));
                    if (!in.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                        break;
                    }
                    if (record.magic != RECORD_MAGIC || record.key_size == 0 ||
                        record.key_size > KEY_CAPACITY || record.binary_count == 0 ||
                        record.binary_count > MAX_BINARIES || record.payload_size > MAX_PAYLOAD ||
                        offset + sizeof(record) + record.payload_size > size) {
                        break;
                    }
                    Key key;
                    key.size = record.key_size;
                    key.bytes = record.key;
                    index[key] = offset; // a later record of the key replaces an unusable one
                    offset += sizeof(record) + record.payload_size;
                }
                file_end = offset;
            } else if (size != 0) {
                started_over = "it was written by another driver or an older version";
            }
        }
    }
    if (fresh) {
        index.clear();
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        FileHeader header{};
        header.magic = FILE_MAGIC;
        header.key_size = global_key.size;
        header.key = global_key.bytes;
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.flush();
        if (!out) {
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: cannot write {}", path.string());
            return false;
        }
        file_end = sizeof(header);
    } else if (file_end < size) {
        // A record cut short: appends start where the last whole one ends.
        std::filesystem::resize_file(path, file_end, ec);
    }
    file.open(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        LOG_WARNING(Render_Vulkan, "Pipeline binaries: cannot open {}", path.string());
        return false;
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
    if (started_over) {
        LOG_INFO(Render_Vulkan, "Pipeline binaries: {} started over, {}", path.filename().string(),
                 started_over);
    }
    LOG_INFO(Render_Vulkan,
             "Pipeline binaries: {} compiled pipelines ({} MiB) kept in {}, indexed in {:.0f} ms",
             index.size(), file_end >> 20, path.filename().string(), ms);
    return true;
}

bool PipelineBinaryCache::ReadRecord(u64 offset, Loaded& loaded) {
    RecordHeader record{};
    {
        std::scoped_lock lock{file_mutex};
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        if (!file.read(reinterpret_cast<char*>(&record), sizeof(record)) ||
            record.magic != RECORD_MAGIC || record.binary_count == 0 ||
            record.binary_count > MAX_BINARIES || record.payload_size > MAX_PAYLOAD) {
            file.clear();
            return false;
        }
        loaded.payload.resize(record.payload_size);
        if (!file.read(reinterpret_cast<char*>(loaded.payload.data()),
                       static_cast<std::streamsize>(loaded.payload.size()))) {
            file.clear();
            return false;
        }
    }
    if (Common::CityHash64(reinterpret_cast<const char*>(loaded.payload.data()),
                           loaded.payload.size()) != record.payload_hash) {
        return false;
    }
    loaded.keys.clear();
    loaded.data.clear();
    size_t pos = 0;
    for (u32 i = 0; i < record.binary_count; ++i) {
        BinaryHeader header;
        if (loaded.payload.size() - pos < sizeof(header)) {
            return false;
        }
        std::memcpy(&header, loaded.payload.data() + pos, sizeof(header));
        pos += sizeof(header);
        if (header.key_size == 0 || header.key_size > KEY_CAPACITY ||
            header.data_size == 0 || header.data_size > loaded.payload.size() - pos) {
            return false;
        }
        VkPipelineBinaryKeyKHR& key = loaded.keys.emplace_back();
        key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
        key.pNext = nullptr;
        key.keySize = header.key_size;
        std::memcpy(key.key, header.key.data(), header.key_size);
        loaded.data.push_back(VkPipelineBinaryDataKHR{
            .dataSize = static_cast<size_t>(header.data_size),
            .pData = loaded.payload.data() + pos,
        });
        pos += static_cast<size_t>(header.data_size);
    }
    return pos == loaded.payload.size();
}

void PipelineBinaryCache::Append(const Key& key, const std::vector<Binary>& binaries) {
    size_t payload_size = 0;
    for (const Binary& binary : binaries) {
        payload_size += sizeof(BinaryHeader) + binary.data.size();
    }
    std::vector<u8> payload(payload_size);
    size_t pos = 0;
    for (const Binary& binary : binaries) {
        BinaryHeader header{};
        header.key_size = binary.key.size;
        header.key = binary.key.bytes;
        header.data_size = binary.data.size();
        std::memcpy(payload.data() + pos, &header, sizeof(header));
        pos += sizeof(header);
        std::memcpy(payload.data() + pos, binary.data.data(), binary.data.size());
        pos += binary.data.size();
    }
    RecordHeader record{};
    record.magic = RECORD_MAGIC;
    record.key_size = key.size;
    record.key = key.bytes;
    record.binary_count = static_cast<u32>(binaries.size());
    record.payload_size = payload.size();
    record.payload_hash =
        Common::CityHash64(reinterpret_cast<const char*>(payload.data()), payload.size());

    u64 offset{};
    {
        std::scoped_lock lock{file_mutex};
        if (stopped_writing) {
            return;
        }
        if (file_end + sizeof(record) + payload.size() > MAX_FILE_BYTES) {
            stopped_writing = true;
            LOG_WARNING(Render_Vulkan,
                        "Pipeline binaries: {} reached {} MiB, newer pipelines are not kept",
                        path.filename().string(), MAX_FILE_BYTES >> 20);
            return;
        }
        file.clear();
        file.seekp(static_cast<std::streamoff>(file_end));
        file.write(reinterpret_cast<const char*>(&record), sizeof(record));
        file.write(reinterpret_cast<const char*>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
        file.flush();
        if (!file) {
            // Disk full or the file gone: what was written past file_end is written over or cut
            // at the next boot.
            file.clear();
            stopped_writing = true;
            LOG_WARNING(Render_Vulkan,
                        "Pipeline binaries: writing {} failed, newer pipelines are not kept",
                        path.filename().string());
            return;
        }
        offset = file_end;
        file_end += sizeof(record) + payload.size();
    }
    std::unique_lock lock{index_mutex};
    index[key] = offset;
}

bool PipelineBinaryCache::Capture(VkPipeline pipeline, std::vector<Binary>& binaries) {
    VkPipelineBinaryCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR;
    create_info.pipeline = pipeline;
    VkPipelineBinaryHandlesInfoKHR handles{};
    handles.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR;
    VkResult result = dld.vkCreatePipelineBinariesKHR(device, &create_info, nullptr, &handles);
    std::vector<VkPipelineBinaryKHR> created;
    if (result == VK_SUCCESS && handles.pipelineBinaryCount > 0 &&
        handles.pipelineBinaryCount <= MAX_BINARIES) {
        created.assign(handles.pipelineBinaryCount, VK_NULL_HANDLE);
        handles.pPipelineBinaries = created.data();
        result = dld.vkCreatePipelineBinariesKHR(device, &create_info, nullptr, &handles);
    } else if (result == VK_SUCCESS) {
        result = VK_ERROR_UNKNOWN;
    }
    VkReleaseCapturedPipelineDataInfoKHR release{};
    release.sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR;
    release.pipeline = pipeline;
    dld.vkReleaseCapturedPipelineDataKHR(device, &release, nullptr);

    bool ok = result == VK_SUCCESS;
    binaries.clear();
    for (const VkPipelineBinaryKHR binary : created) {
        if (binary == VK_NULL_HANDLE) {
            ok = false;
            continue;
        }
        if (ok) {
            VkPipelineBinaryDataInfoKHR info{};
            info.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR;
            info.pipelineBinary = binary;
            VkPipelineBinaryKeyKHR key{};
            key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
            size_t size = 0;
            ok = dld.vkGetPipelineBinaryDataKHR(device, &info, &key, &size, nullptr) ==
                     VK_SUCCESS &&
                 size > 0 && size <= MAX_PAYLOAD && key.keySize > 0 && key.keySize <= KEY_CAPACITY;
            if (ok) {
                Binary& out = binaries.emplace_back();
                out.key.size = key.keySize;
                std::memcpy(out.key.bytes.data(), key.key, key.keySize);
                out.data.resize(size);
                ok = dld.vkGetPipelineBinaryDataKHR(device, &info, &key, &size,
                                                    out.data.data()) == VK_SUCCESS &&
                     size == out.data.size();
            }
        }
        dld.vkDestroyPipelineBinaryKHR(device, binary, nullptr);
    }
    return ok && !binaries.empty();
}

template <typename CreateInfo, typename CreateFn>
VkResult PipelineBinaryCache::CreateThrough(const CreateInfo& ci, VkPipelineCache cache,
                                            VkPipeline* pipeline, CreateFn&& create) {
    creations.fetch_add(1, std::memory_order_relaxed);
    if (!Eligible(ci)) {
        uncaptured.fetch_add(1, std::memory_order_relaxed);
        return create(ci, cache);
    }
    VkPipelineCreateInfoKHR key_info{};
    key_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR;
    key_info.pNext = const_cast<CreateInfo*>(&ci);
    VkPipelineBinaryKeyKHR driver_key{};
    driver_key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
    if (dld.vkGetPipelineKeyKHR(device, &key_info, &driver_key) != VK_SUCCESS ||
        driver_key.keySize == 0 || driver_key.keySize > KEY_CAPACITY) {
        uncaptured.fetch_add(1, std::memory_order_relaxed);
        return create(ci, cache);
    }
    Key key;
    key.size = driver_key.keySize;
    std::memcpy(key.bytes.data(), driver_key.key, driver_key.keySize);

    std::optional<u64> offset;
    {
        std::shared_lock lock{index_mutex};
        if (const auto it = index.find(key); it != index.end()) {
            offset = it->second;
        }
    }
    if (offset) {
        Loaded loaded;
        VkResult result = VK_ERROR_UNKNOWN;
        if (ReadRecord(*offset, loaded)) {
            const u32 count = static_cast<u32>(loaded.keys.size());
            const VkPipelineBinaryKeysAndDataKHR keys_and_data{
                .binaryCount = count,
                .pPipelineBinaryKeys = loaded.keys.data(),
                .pPipelineBinaryData = loaded.data.data(),
            };
            VkPipelineBinaryCreateInfoKHR create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR;
            create_info.pKeysAndDataInfo = &keys_and_data;
            std::vector<VkPipelineBinaryKHR> handles_storage(count, VK_NULL_HANDLE);
            VkPipelineBinaryHandlesInfoKHR handles{};
            handles.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR;
            handles.pipelineBinaryCount = count;
            handles.pPipelineBinaries = handles_storage.data();
            result = dld.vkCreatePipelineBinariesKHR(device, &create_info, nullptr, &handles);
            if (result == VK_SUCCESS) {
                VkPipelineBinaryInfoKHR binary_info{};
                binary_info.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR;
                binary_info.pNext = ci.pNext;
                binary_info.binaryCount = count;
                binary_info.pPipelineBinaries = handles_storage.data();
                CreateInfo from_binaries = ci;
                from_binaries.pNext = &binary_info;
                result = create(from_binaries, VK_NULL_HANDLE);
            }
            for (const VkPipelineBinaryKHR binary : handles_storage) {
                if (binary != VK_NULL_HANDLE) {
                    dld.vkDestroyPipelineBinaryKHR(device, binary, nullptr);
                }
            }
        }
        if (result == VK_SUCCESS) {
            hits.fetch_add(1, std::memory_order_relaxed);
            ReportMaybe(false);
            return VK_SUCCESS;
        }
        if (*pipeline != VK_NULL_HANDLE) {
            dld.vkDestroyPipeline(device, *pipeline, nullptr);
            *pipeline = VK_NULL_HANDLE;
        }
        failed_hits.fetch_add(1, std::memory_order_relaxed);
        if (failures_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan,
                        "Pipeline binaries: a kept pipeline could not be used ({}), compiled again",
                        static_cast<int>(result));
        }
        std::unique_lock lock{index_mutex};
        index.erase(key);
    }
    if (t_no_capture) {
        uncaptured.fetch_add(1, std::memory_order_relaxed);
        return create(ci, cache);
    }
    VkPipelineCreateFlags2CreateInfoKHR flags2{};
    flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR;
    flags2.pNext = ci.pNext;
    flags2.flags = static_cast<VkPipelineCreateFlags2KHR>(ci.flags) |
                   VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
    CreateInfo capturing = ci;
    capturing.pNext = &flags2;
    if (create(capturing, VK_NULL_HANDLE) != VK_SUCCESS) {
        if (*pipeline != VK_NULL_HANDLE) {
            dld.vkDestroyPipeline(device, *pipeline, nullptr);
            *pipeline = VK_NULL_HANDLE;
        }
        uncaptured.fetch_add(1, std::memory_order_relaxed);
        return create(ci, cache);
    }
    std::vector<Binary> binaries;
    if (Capture(*pipeline, binaries)) {
        Append(key, binaries);
        captured.fetch_add(1, std::memory_order_relaxed);
    } else {
        uncaptured.fetch_add(1, std::memory_order_relaxed);
        if (failures_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan, "Pipeline binaries: a compiled pipeline could not be read back");
        }
    }
    ReportMaybe(false);
    return VK_SUCCESS;
}

VkResult PipelineBinaryCache::CreateGraphics(const VkGraphicsPipelineCreateInfo& ci,
                                             VkPipelineCache cache, VkPipeline* pipeline) {
    return CreateThrough(ci, cache, pipeline,
                         [&](const VkGraphicsPipelineCreateInfo& info, VkPipelineCache used) {
                             return dld.vkCreateGraphicsPipelines(device, used, 1, &info, nullptr,
                                                                  pipeline);
                         });
}

VkResult PipelineBinaryCache::CreateCompute(const VkComputePipelineCreateInfo& ci,
                                            VkPipelineCache cache, VkPipeline* pipeline) {
    return CreateThrough(ci, cache, pipeline,
                         [&](const VkComputePipelineCreateInfo& info, VkPipelineCache used) {
                             return dld.vkCreateComputePipelines(device, used, 1, &info, nullptr,
                                                                 pipeline);
                         });
}

void PipelineBinaryCache::ReportMaybe(bool force) {
    const s64 now = NowMs();
    if (!force) {
        s64 due = next_report.load(std::memory_order_relaxed);
        if (now < due || !next_report.compare_exchange_strong(due, now + 30'000,
                                                             std::memory_order_relaxed)) {
            return;
        }
        if (due == 0) {
            return; // the first creation only starts the clock
        }
    }
    u64 kept_bytes{};
    {
        std::scoped_lock lock{file_mutex};
        kept_bytes = file_end;
    }
    LOG_INFO(Render_Vulkan,
             "Pipeline binaries: {} pipelines created, {} from kept binaries, {} compiled and "
             "kept, {} compiled and not kept, {} kept unusable; {} MiB kept",
             creations.load(), hits.load(), captured.load(), uncaptured.load(),
             failed_hits.load(), kept_bytes >> 20);
}

} // namespace Vulkan
