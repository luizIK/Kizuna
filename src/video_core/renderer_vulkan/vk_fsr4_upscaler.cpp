// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FSR 4 through AMD's FidelityFX runtime. The runtime takes plain Vulkan images: the inputs
// in SHADER_READ_ONLY_OPTIMAL (FFX "compute read"), the output in GENERAL ("unordered access"),
// and records into our command buffer. Picking the FSR 4 provider by version, the perceptual
// colour flag, the black-output check and the later tries at a context that couldn't be created
// follow CheezerWheezer's FSR 4 integration for CheezyEmu.

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_vulkan/vk_fsr4_upscaler.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/ffx_runtime.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

// After vulkan.h, which sets the platform macros the wrapper's Win32 entry points need.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "api/include/vk/ffx_api_vk.h"
#include "upscalers/include/ffx_upscale.h"
#ifndef _WIN32
#undef __declspec
#endif

namespace Vulkan {
namespace {
constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageSubresourceRange COLOR_RANGE{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

FfxApiResource Resource(VkImage image, VkFormat format, VkExtent2D size, u32 usage, u32 state) {
    FfxApiResource resource{};
    resource.resource = reinterpret_cast<void*>(image);
    resource.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    resource.description.format = ffxApiGetSurfaceFormatVK(format);
    resource.description.width = size.width;
    resource.description.height = size.height;
    resource.description.depth = 1;
    resource.description.mipCount = 1;
    resource.description.flags = 0;
    resource.description.usage = usage;
    resource.state = state;
    return resource;
}

VkImageMemoryBarrier Barrier(VkImage image, VkImageSubresourceRange range, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = range,
    };
}

VkImageMemoryBarrier Barrier(const Fsr4Upscaler::Image& image, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkImageLayout old_layout,
                             VkImageLayout new_layout) {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return Barrier(image.image, range, src_access, dst_access, old_layout, new_layout);
}

void Message(uint32_t type, const wchar_t* message) {
    // The first 48, then every 500th: a bad input repeats every frame.
    static std::atomic<u32> count{};
    const u32 n = count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > 48 && n % 500 != 0) {
        return;
    }
    std::string text;
    for (const wchar_t* c = message; c && *c; ++c) {
        text.push_back(*c < 128 ? static_cast<char>(*c) : '?');
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        LOG_ERROR(Render_Vulkan, "FSR 4: {} (message {})", text, n);
    } else {
        LOG_INFO(Render_Vulkan, "FSR 4: {} (message {})", text, n);
    }
}

s64 SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The output check: an 8x8 grid of input and output pixels, 16 bytes a pixel (both RGBA16F).
constexpr u32 HEALTH_GRID = 8;
constexpr u32 HEALTH_SAMPLES = HEALTH_GRID * HEALTH_GRID;
constexpr VkDeviceSize HEALTH_STRIDE = 16;
constexpr VkDeviceSize HEALTH_OUTPUT_OFFSET = HEALTH_SAMPLES * HEALTH_STRIDE;
constexpr VkDeviceSize HEALTH_SIZE = 2 * HEALTH_OUTPUT_OFFSET;
constexpr u32 HEALTH_INTERVAL = 30; ///< Frames between samples once one was good.
/// Frames between samples until the first good one: a black FSR 4 is found in about a quarter
/// of a second instead of several seconds of black screen.
constexpr u32 HEALTH_FAST_INTERVAL = 4;
constexpr u32 HEALTH_BLACK_LIMIT = 4; ///< Black samples in a row to give FSR 4 up.
/// A sample counts as black when this many of its 64 pixels are black or NaN: the RX 9070 XT's
/// broken output was black with scattered white dots.
constexpr u32 HEALTH_BLACK_PIXELS = 56;
constexpr u32 HEALTH_GOOD_LIMIT = 4; ///< Good samples to stop checking.

/// FRITCHII_FSR4_CHECK=0 (boot): keep FSR 4 whatever its output looks like.
bool OutputCheckEnabled() {
    static const bool on = VideoCommon::OcarinaSwitch::Read("FRITCHII_FSR4_CHECK", 1) != 0;
    return on;
}

/// Where a black FSR 4 is remembered with the GPU and driver it was seen on: the next start goes
/// straight to FSR 3.1 instead of showing black again, until the driver changes.
std::filesystem::path BlackMarkerPath() {
    return Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "fsr4_black.txt";
}

std::string DriverTag(const Device& device) {
    return fmt::format("{}:{}:{}", static_cast<u32>(device.GetDriverID()),
                       device.GetDriverVersion(), device.GetModelName());
}

/// Whether FSR 4 drew black on this GPU and driver in an earlier session.
bool BlackMarked(const Device& device) {
    if (!OutputCheckEnabled()) {
        return false;
    }
    std::ifstream file{BlackMarkerPath()};
    std::string line;
    return std::getline(file, line) && line == DriverTag(device);
}

VkDeviceContext DeviceContext(const Device& device) {
    return VkDeviceContext{
        .vkDevice = *device.GetLogical(),
        .vkPhysicalDevice = static_cast<VkPhysicalDevice>(device.GetPhysical()),
        .vkDeviceProcAddr = device.GetDispatchLoader().vkGetDeviceProcAddr,
    };
}

struct Provider {
    u64 id{};
    std::string name;
};

/// The newest FSR 4 (or later) upscaler AMD's runtime offers for this device. The runtime also
/// offers FSR 3.1 on GPUs without FSR 4; that one is left to the built-in FSR 3.1.
std::optional<Provider> FindFsr4(const FfxRuntime::Api& api, const Device& device) {
    VkDeviceContext vk_context = DeviceContext(device);
    u64 count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = &vk_context;
    versions.outputCount = &count;
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
        LOG_INFO(Render_Vulkan, "AMD FSR 4: the FidelityFX runtime offers no upscaler for {}",
                 device.GetModelName());
        return std::nullopt;
    }
    std::vector<u64> ids(count);
    std::vector<const char*> names(count);
    versions.versionIds = ids.data();
    versions.versionNames = names.data();
    if (api.query(nullptr, &versions.header) != FFX_API_RETURN_OK) {
        return std::nullopt;
    }
    std::string offered;
    std::optional<Provider> fsr4;
    for (u64 i = 0; i < count; ++i) {
        const std::string name = names[i] ? names[i] : "";
        offered += (offered.empty() ? "" : ", ") + name;
        // Newest first.
        if (!fsr4 && !name.empty() && name[0] >= '4' && name[0] <= '9') {
            fsr4 = Provider{ids[i], name};
        }
    }
    LOG_INFO(Render_Vulkan, "AMD FSR 4: upscalers offered for {}: {}{}", device.GetModelName(),
             offered, fsr4 ? "" : " (no FSR 4; using FSR 3.1)");
    return fsr4;
}
} // Anonymous namespace

bool Fsr4Upscaler::IsWanted(const Device& device) {
    return Settings::values.oot_anti_aliasing.GetValue() == Settings::OotAntiAliasing::Fsr4 &&
           device.GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY;
}

bool Fsr4Upscaler::RunsUnjittered(const Device& device) {
#ifdef _WIN32
    return IsWanted(device) && !VideoCommon::OcarinaFsr4Jitter() && FfxRuntime::FilesPresent() &&
           !BlackMarked(device);
#else
    return false;
#endif
}

Fsr4Upscaler::Fsr4Upscaler(const Device& device_, Scheduler& scheduler_,
                           MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (BlackMarked(device)) {
        LOG_WARNING(Render_Vulkan,
                    "AMD FSR 4 drew a black picture on this GPU and driver in an earlier session: "
                    "using FSR 3.1. A driver update brings FSR 4 back (FRITCHII_FSR4_CHECK=0 "
                    "tries it anyway).");
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    api = FfxRuntime::Acquire();
    const auto provider = api ? FindFsr4(*api, device) : std::nullopt;
    if (!provider) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    provider_id = provider->id;
    provider_name = provider->name;
    CreateExposure();
}

Fsr4Upscaler::~Fsr4Upscaler() {
    DestroyContext();
}

void Fsr4Upscaler::DestroyContext() {
    if (context == nullptr) {
        return;
    }
    scheduler.Finish(); // recorded dispatches may still use it
    ffxContext ctx = context;
    api->destroy_context(&ctx, nullptr);
    context = nullptr;
}

bool Fsr4Upscaler::CreateContext(VkExtent2D render_size, VkExtent2D upscale_size) {
    DestroyContext();
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.header.pNext = nullptr;
    backend.deviceContext = DeviceContext(device);
    ffxOverrideVersion version_override{};
    version_override.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version_override.header.pNext = &backend.header;
    version_override.versionId = provider_id;
    ffxCreateContextDescUpscaleVersion api_version{};
    api_version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    api_version.header.pNext = &version_override.header;
    api_version.version = FFX_UPSCALER_VERSION;

    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &api_version.header;
    // Tonemapped LDR colour, standard (not reversed) finite depth, exposure given (unit). The
    // colour is perceptual (gamma encoded); FRITCHII_FSR4_LINEAR calls it linear as before.
    create.flags =
        VideoCommon::OcarinaFsr4LinearColor() ? 0u : FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE;
    create.maxRenderSize = {render_size.width, render_size.height};
    create.maxUpscaleSize = {upscale_size.width, upscale_size.height};
    create.fpMessage = Message;
    ffxContext ctx{};
    const ffxReturnCode_t result = api->create_context(&ctx, &create.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "FSR 4: ffxCreateContext {}x{} -> {}x{} failed ({}); using FSR 3.1",
                  render_size.width, render_size.height, upscale_size.width, upscale_size.height,
                  result);
        return false;
    }
    context = ctx;
    render_extent = render_size;
    upscale_extent = upscale_size;
    LOG_INFO(Render_Vulkan, "FSR 4: AMD FSR {} context {}x{} -> {}x{} ({} colour, {})",
             provider_name, render_size.width, render_size.height, upscale_size.width,
             upscale_size.height, VideoCommon::OcarinaFsr4LinearColor() ? "linear" : "perceptual",
             VideoCommon::OcarinaFsr4Jitter() ? "camera jitter" : "no camera jitter");
    return true;
}

bool Fsr4Upscaler::Failed() const noexcept {
    if (!failed.load()) {
        return false;
    }
    const s64 retry = retry_at_ns.load();
    return retry == 0 || SteadyNowNs() < retry;
}

bool Fsr4Upscaler::CheckOutputHealth() {
    if (!health_pending || !scheduler.IsFree(health_tick)) {
        return false;
    }
    health_pending = false;
    health_buffer.Invalidate();
    const std::span<const u8> data = health_buffer.Mapped();
    // The input is a real picture when its pixels differ (not a fade, a loading or a flat screen).
    u32 distinct_inputs = 0;
    for (u32 i = 1; i < HEALTH_SAMPLES; ++i) {
        if (std::memcmp(&data[i * HEALTH_STRIDE], &data[0], HEALTH_STRIDE) != 0) {
            ++distinct_inputs;
        }
    }
    // A dark scene (a cave, a fade) is dark in FSR's input too: only a bright input judges it.
    // Below 1/64 in every colour channel (half-float exponent under 9).
    u32 dark_inputs = 0;
    u32 bad_outputs = 0;
    for (u32 i = 0; i < HEALTH_SAMPLES; ++i) {
        std::array<u16, 4> in{};
        std::memcpy(in.data(), &data[i * HEALTH_STRIDE], sizeof(in));
        if ((in[0] & 0x7fff) < 0x2400 && (in[1] & 0x7fff) < 0x2400 && (in[2] & 0x7fff) < 0x2400) {
            ++dark_inputs;
        }
        std::array<u16, 4> out{};
        std::memcpy(out.data(), &data[HEALTH_OUTPUT_OFFSET + i * HEALTH_STRIDE], sizeof(out));
        bool nan = false;
        bool black = true;
        for (u32 channel = 0; channel < 3; ++channel) {
            nan |= (out[channel] & 0x7c00) == 0x7c00;
            black &= (out[channel] & 0x7fff) == 0;
        }
        if (nan || black) {
            ++bad_outputs;
        }
    }
    if (distinct_inputs < HEALTH_SAMPLES / 4 || dark_inputs > HEALTH_SAMPLES / 4) {
        return false; // nothing to judge by
    }
    if (bad_outputs < HEALTH_BLACK_PIXELS) {
        black_checks = 0;
        if (++good_checks >= HEALTH_GOOD_LIMIT) {
            health_done = true;
            health_buffer = vk::Buffer{};
            LOG_INFO(Render_Vulkan, "FSR 4: output check passed");
        }
        return false;
    }
    ++black_checks;
    LOG_WARNING(Render_Vulkan, "FSR 4: black output over a game picture ({} of {})", black_checks,
                HEALTH_BLACK_LIMIT);
    return black_checks >= HEALTH_BLACK_LIMIT;
}

void Fsr4Upscaler::GiveUpBlack() {
    std::error_code ec;
    std::filesystem::create_directories(BlackMarkerPath().parent_path(), ec);
    {
        std::ofstream file{BlackMarkerPath(), std::ios::trunc};
        file << DriverTag(device) << '\n';
    }
    LOG_ERROR(Render_Vulkan, "AMD FSR 4 drew a black picture on this GPU and driver: using FSR 3.1 "
                             "from now on. A driver update may fix FSR 4.");
    health_done = true;
    health_buffer = vk::Buffer{};
    retry_at_ns = 0;
    failed = true;
}

void Fsr4Upscaler::CreateExposure() {
    exposure_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32_SFLOAT,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *exposure_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0,
                               Barrier(image, COLOR_RANGE, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        VkClearColorValue one{};
        one.float32[0] = 1.0f;
        cmdbuf.ClearColorImage(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, one, COLOR_RANGE);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               Barrier(image, COLOR_RANGE, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_SHADER_READ_BIT,
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    });
}

void Fsr4Upscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = OUTPUT_FORMAT,
        .components{},
        .subresourceRange = COLOR_RANGE,
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                               Barrier(image, COLOR_RANGE, 0,
                                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
    });
}

bool Fsr4Upscaler::Upscale(const Inputs& in) {
    if (failed.load()) {
        if (Failed()) {
            return false;
        }
        // Time for another try at a context that couldn't be created.
        failed = false;
        retry_at_ns = 0;
    }
    if (!health_done && OutputCheckEnabled() && CheckOutputHealth()) {
        GiveUpBlack();
        return false;
    }
    const bool new_context = context == nullptr || in.render_size.width != render_extent.width ||
                             in.render_size.height != render_extent.height ||
                             in.upscale_size.width != upscale_extent.width ||
                             in.upscale_size.height != upscale_extent.height;
    if (new_context) {
        if (!CreateContext(in.render_size, in.upscale_size)) {
            // Out of video memory, typically, and a scene load is when memory is shortest: FSR 3.1
            // draws meanwhile and FSR 4 is tried again a few times, later.
            constexpr std::array<s64, 4> waits_s{10, 30, 90, 240};
            const bool again = create_failures < waits_s.size();
            LOG_WARNING(Render_Vulkan, "FSR 4: FSR 3.1 draws{}",
                        again ? fmt::format(", next try in {} s", waits_s[create_failures])
                              : std::string{" for the rest of this session"});
            retry_at_ns =
                again ? SteadyNowNs() + waits_s[create_failures] * s64{1'000'000'000} : s64{0};
            ++create_failures;
            failed = true;
            return false;
        }
        create_failures = 0;
        CreateOutput(in.upscale_size);
    }

    // Check the output now and then until FSR 4 has drawn a picture.
    VkBuffer health = VK_NULL_HANDLE;
    if (!health_done && !health_pending && OutputCheckEnabled() &&
        ++health_frame % (good_checks == 0 ? HEALTH_FAST_INTERVAL : HEALTH_INTERVAL) == 0) {
        if (!health_buffer) {
            health_buffer = memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = HEALTH_SIZE,
                    .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                MemoryUsage::Download);
            std::memset(health_buffer.Mapped().data(), 0, HEALTH_SIZE);
        }
        health = *health_buffer;
        health_pending = true;
        health_tick = scheduler.CurrentTick();
    }

    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.header.pNext = nullptr;
    dispatch.color = Resource(in.color.image, in.color.format, in.render_size, 0,
                              FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.depth = Resource(in.depth.image, in.depth.format, in.render_size,
                              FFX_API_RESOURCE_USAGE_DEPTHTARGET,
                              FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.motionVectors = Resource(in.motion_vectors.image, in.motion_vectors.format,
                                      in.render_size, 0, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    dispatch.exposure = Resource(*exposure_image, VK_FORMAT_R32_SFLOAT, {1, 1}, 0,
                                 FFX_API_RESOURCE_STATE_COMPUTE_READ);
    if (in.reactive) {
        dispatch.reactive = Resource(in.reactive->image, in.reactive->format, in.render_size, 0,
                                     FFX_API_RESOURCE_STATE_COMPUTE_READ);
    }
    dispatch.output = Resource(*output_image, OUTPUT_FORMAT, in.upscale_size,
                               FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch.jitterOffset = {in.jitter[0], in.jitter[1]};
    dispatch.motionVectorScale = {1.0f, 1.0f};
    dispatch.renderSize = {in.render_size.width, in.render_size.height};
    dispatch.upscaleSize = {in.upscale_size.width, in.upscale_size.height};
    dispatch.enableSharpening = in.sharpen;
    dispatch.sharpness = in.sharpness;
    dispatch.frameTimeDelta = in.frame_time_ms;
    dispatch.preExposure = 1.0f;
    dispatch.reset = in.reset || new_context;
    dispatch.cameraNear = in.camera_near;
    dispatch.cameraFar = in.camera_far;
    dispatch.cameraFovAngleVertical = in.vertical_fov;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    dispatch.flags = 0;

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([this, in, dispatch, health,
                      output = *output_image](vk::CommandBuffer cmdbuf) {
        constexpr VkAccessFlags WRITES =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        // The inputs stay in GENERAL otherwise; FSR 4 reads them as sampled images.
        boost::container::static_vector<VkImageMemoryBarrier, 5> to_read{
            Barrier(in.color, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.depth, WRITES, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(in.motion_vectors, WRITES, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            Barrier(output, COLOR_RANGE, VK_ACCESS_SHADER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        if (in.reactive) {
            to_read.push_back(Barrier(*in.reactive, WRITES, VK_ACCESS_SHADER_READ_BIT,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, {}, {}, to_read);

        ffxDispatchDescUpscale recorded = dispatch;
        recorded.commandList = static_cast<void*>(*cmdbuf);
        ffxContext ctx = context;
        const ffxReturnCode_t result = api->dispatch(&ctx, &recorded.header);
        if (result != FFX_API_RETURN_OK && !failed.exchange(true)) {
            LOG_ERROR(Render_Vulkan, "FSR 4: ffxDispatch failed ({}); using FSR 3.1", result);
        }

        boost::container::static_vector<VkImageMemoryBarrier, 5> back{
            Barrier(in.color, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.depth, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(in.motion_vectors, VK_ACCESS_SHADER_READ_BIT, 0,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            Barrier(output, COLOR_RANGE, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL),
        };
        if (in.reactive) {
            back.push_back(Barrier(*in.reactive, VK_ACCESS_SHADER_READ_BIT, 0,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL));
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, back);
        if (health != VK_NULL_HANDLE) {
            // An 8x8 grid of input and output pixels, for CheckOutputHealth.
            const VkMemoryBarrier to_transfer{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = WRITES,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, to_transfer);
            const auto grid = [](u32 mip, u32 layer, VkExtent2D size, VkDeviceSize base) {
                std::array<VkBufferImageCopy, HEALTH_SAMPLES> regions{};
                const VkImageSubresourceLayers layers{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = mip,
                    .baseArrayLayer = layer,
                    .layerCount = 1,
                };
                for (u32 y = 0; y < HEALTH_GRID; ++y) {
                    for (u32 x = 0; x < HEALTH_GRID; ++x) {
                        const u32 i = y * HEALTH_GRID + x;
                        regions[i] = VkBufferImageCopy{
                            .bufferOffset = base + i * HEALTH_STRIDE,
                            .bufferRowLength = 0,
                            .bufferImageHeight = 0,
                            .imageSubresource = layers,
                            .imageOffset{
                                static_cast<s32>((2 * x + 1) * size.width / (2 * HEALTH_GRID)),
                                static_cast<s32>((2 * y + 1) * size.height / (2 * HEALTH_GRID)),
                                0},
                            .imageExtent{1, 1, 1},
                        };
                    }
                }
                return regions;
            };
            const auto input_regions = grid(in.color.range.baseMipLevel,
                                            in.color.range.baseArrayLayer, in.render_size, 0);
            const auto output_regions = grid(0, 0, in.upscale_size, HEALTH_OUTPUT_OFFSET);
            cmdbuf.CopyImageToBuffer(in.color.image, VK_IMAGE_LAYOUT_GENERAL, health,
                                     input_regions);
            cmdbuf.CopyImageToBuffer(output, VK_IMAGE_LAYOUT_GENERAL, health, output_regions);
            const VkMemoryBarrier to_host{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   0, to_host);
        }
    });
    return true;
}

} // namespace Vulkan
