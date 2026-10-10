#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/allreduce-push.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-sm70-d256.cuh"
#include "ggml-cuda/fused-expert.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/ggml-cuda-expert.cuh"
#include "ggml-cuda/ggml-cuda-numa-plan.h"
#include "ggml-cuda/ggml-cuda-routing-trace.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-grouped.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/q4k-skinny.cuh"
#include "ggml-cuda/q8-skinny.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/qsa.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// MPOL_BIND / MPOL_DEFAULT from <linux/mempolicy.h>; defined here to avoid pulling in kernel headers
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif

// on Linux the CUDA mapped host memory is allocated with cudaHostAlloc under a
// set_mempolicy(MPOL_BIND) policy on the allocating thread, which pins the pages to the
// NUMA node selected by a one-time bandwidth probe; HIP and MUSA keep using plain cudaHostAlloc
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__linux__)
#define GGML_CUDA_MAPPED_NUMA
#endif

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

// Host ranges pinned in place with cudaHostRegister through the ggml_backend_register_host_buffer
// proc of this backend. The weights of the model are not registered in place: the mapped
// weights buffer type owns its host memory (cudaHostAlloc mapped), so no model file page is
// ever passed to cudaHostRegister.
//
// The registration is read-only (cudaHostRegisterReadOnly), so the device only reads from such
// a range: the host side is always the source of a copy, and only H2D copies can start inside
// a range and need splitting. If a writable registration is ever added, the D2H direction must
// be re-evaluated.
//
// A cudaMemcpy with a host pointer is classified by its starting address: when the range starts
// inside a registered allocation but ends outside it, the runtime fails the copy with
// "invalid argument". H2D copies that touch a registered host range are therefore split at the
// end of every registered range they enter.

// a range pinned in place with cudaHostRegister; the caller of the register proc owns the
// reference and releases it through the matching unregister call
struct ggml_cuda_registered_host_range {
    uintptr_t begin;
    uintptr_t end;
};

static std::mutex ggml_cuda_registered_host_ranges_mutex;
static std::vector<ggml_cuda_registered_host_range> ggml_cuda_registered_host_ranges; // protected by the mutex

// page size of the host, for aligning a cudaHostRegister range
static size_t ggml_cuda_host_page_size() {
#if defined(__linux__)
    const long page = sysconf(_SC_PAGESIZE);
    if (page > 0) {
        return (size_t) page;
    }
#endif
    return 4096;
}

// register [begin, end) (page-aligned) read-only; fails when the range is already registered
static bool ggml_cuda_registered_host_range_add(uintptr_t begin, uintptr_t end) {
    std::lock_guard<std::mutex> lock(ggml_cuda_registered_host_ranges_mutex);

    for (const ggml_cuda_registered_host_range & range : ggml_cuda_registered_host_ranges) {
        if (range.begin <= begin && end <= range.end) {
            GGML_LOG_ERROR("%s: the host range [%p, %p) is already registered\n",
                           __func__, (void *) begin, (void *) end);
            return false;
        }
    }

    cudaError_t err = cudaHostRegister((void *) begin, end - begin, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: registering %.2f MiB of host memory failed: %s\n",
                       __func__, (end - begin) / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }

    ggml_cuda_registered_host_ranges.push_back({ begin, end });
    return true;
}

// unregister the range registered at begin
static void ggml_cuda_registered_host_range_remove(uintptr_t begin) {
    std::lock_guard<std::mutex> lock(ggml_cuda_registered_host_ranges_mutex);

    for (auto it = ggml_cuda_registered_host_ranges.begin(); it != ggml_cuda_registered_host_ranges.end(); ++it) {
        if (it->begin != begin) {
            continue;
        }
        cudaError_t err = cudaHostUnregister((void *) it->begin);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: unregistering host range [%p, %p) failed: %s\n",
                          __func__, (void *) it->begin, (void *) it->end, cudaGetErrorString(err));
        }
        ggml_cuda_registered_host_ranges.erase(it);
        return;
    }
    GGML_LOG_DEBUG("%s: host range at %p is not registered\n", __func__, (void *) begin);
}

// end of the registered range that contains ptr, or 0 when ptr is in no range
static uintptr_t ggml_cuda_registered_host_range_end(const void * ptr) {
    std::lock_guard<std::mutex> lock(ggml_cuda_registered_host_ranges_mutex);
    const uintptr_t addr = (uintptr_t) ptr;
    for (const ggml_cuda_registered_host_range & range : ggml_cuda_registered_host_ranges) {
        if (addr >= range.begin && addr < range.end) {
            return range.end;
        }
    }
    return 0;
}

// true when [ptr, ptr+size) overlaps any registered range
static bool ggml_cuda_registered_host_range_intersects(const void * ptr, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_registered_host_ranges_mutex);
    const uintptr_t begin = (uintptr_t) ptr;
    const uintptr_t end   = begin + size;
    for (const ggml_cuda_registered_host_range & range : ggml_cuda_registered_host_ranges) {
        if (begin < range.end && range.begin < end) {
            return true;
        }
    }
    return false;
}

// split at the range boundary so that no chunk starts inside a registered range and ends
// outside it; a chunk that starts outside a range may span it (the runtime only looks at
// the starting address)
void ggml_cuda_memcpy_h2d_async(void * dst, const void * src, size_t size, cudaStream_t stream) {
    const char * src_bytes = (const char *) src;
    char * dst_bytes = (char *) dst;

    while (size > 0) {
        const uintptr_t range_end = ggml_cuda_registered_host_range_end(src_bytes);
        size_t chunk = size;
        if (range_end != 0 && (uintptr_t) src_bytes + size > range_end) {
            chunk = (size_t) (range_end - (uintptr_t) src_bytes);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst_bytes, src_bytes, chunk, cudaMemcpyHostToDevice, stream));
        src_bytes += chunk;
        dst_bytes += chunk;
        size      -= chunk;
    }
}

// only the host side of a 2D copy can fall inside a registered range; when the whole
// host-side span stays clear of every range, keep the copy 2D, otherwise split per row
static void ggml_cuda_memcpy_2d_h2d_async(char * dst, size_t dpitch, const char * src, size_t spitch, size_t width, size_t height, cudaStream_t stream) {
    if (height == 0 || width == 0) {
        return;
    }
    if (!ggml_cuda_registered_host_range_intersects(src, (height - 1)*spitch + width)) {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice, stream));
        return;
    }
    for (size_t y = 0; y < height; ++y) {
        ggml_cuda_memcpy_h2d_async(dst + y*dpitch, src + y*spitch, width, stream);
    }
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    // enable peer access between every pair of physical devices that the driver reports as
    // capable; without it cudaMemcpyPeerAsync bounces transfers through host memory.
    // can_access_peer == 0 means the hardware/topology does not support it (e.g. some x86
    // PCIe/IOMMU configs): those pairs keep the host path inside cudaMemcpyPeerAsync.
    // that is a capability branch, not a fallback.
    bool p2p_enabled[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};
    for (int id = 0; id < info.physical_device_count; ++id) {
        CUDA_CHECK(cudaSetDevice(id));
        for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
            if (id == id_other) {
                continue;
            }
            int can_access_peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (!can_access_peer) {
                continue;
            }
            const cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                // already enabled, e.g. by a previous call in this loop
                (void) cudaGetLastError();
            } else if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: cudaDeviceEnablePeerAccess(%d -> %d) failed: %s\n",
                               __func__, id, id_other, cudaGetErrorString(err));
                continue;
            }
            p2p_enabled[id][id_other] = true;
        }
    }

    std::string p2p_summary;
    for (int id = 0; id < info.physical_device_count; ++id) {
        for (int id_other = id + 1; id_other < info.physical_device_count; ++id_other) {
            if (p2p_enabled[id][id_other] && p2p_enabled[id_other][id]) {
                p2p_summary += " " + std::to_string(id) + "<->" + std::to_string(id_other);
            } else if (p2p_enabled[id][id_other]) {
                p2p_summary += " " + std::to_string(id) + "->" + std::to_string(id_other);
            } else if (p2p_enabled[id_other][id]) {
                p2p_summary += " " + std::to_string(id_other) + "->" + std::to_string(id);
            }
        }
    }
    if (!p2p_summary.empty()) {
        GGML_LOG_INFO("%s: P2P enabled:%s\n", __func__, p2p_summary.c_str());
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // cudaDeviceEnablePeerAccess alone does not make VMM pages peer-accessible, so
            // access must be granted here explicitly. Grant it to this physical device and to
            // every other physical device that the driver reports as able to reach it
            // (cudaDeviceCanAccessPeer). Devices without that capability (e.g. some x86
            // PCIe/IOMMU configs) are left out and keep the host path; that is a capability
            // branch, not a fallback. With virtual devices, grant access on the backing
            // *physical* devices (deduplicated, since several virtual devices can map to the
            // same physical GPU).
            std::vector<CUmemAccessDesc> access_descs;
            bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
            const int device_count = ggml_cuda_info().device_count;
            for (int id = 0; id < device_count; ++id) {
                const int id_physical = ggml_cuda_get_physical_device(id);
                if (id_physical != physical_device) {
                    int can_access_peer = 0;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                    if (!can_access_peer) {
                        continue;
                    }
                }
                if (physical_seen[id_physical]) {
                    continue;
                }
                physical_seen[id_physical] = true;
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = id_physical;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                access_descs.push_back(access);
            }
            CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_memcpy_h2d_async((char *) tensor->data + offset, data, size, cudaStreamPerThread);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_memcpy_2d_h2d_async((char *) tensor->data + offset, stride_tensor, (const char *) data, stride_data, size, n_copies, cudaStreamPerThread);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
// (ggml_backend_cuda_buffer_type_context is shared with the expert buffer type in common.cuh)

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    // free memory before/after every large allocation (dense weights, KV cache, compute
    // buffers): the runtime adds device memory after the fit measured it, so these lines
    // show which allocation ate what the fit did not account for
    size_t free_before = 0;
    size_t total       = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) {
        (void) cudaGetLastError();
        free_before = 0;
        total       = 0;
    }

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s (%.2f MiB free of %.2f MiB)\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err),
            free_before / 1024.0 / 1024.0, total / 1024.0 / 1024.0);
        return nullptr;
    }
    GGML_LOG_DEBUG("%s: device %d: cudaMalloc %.2f MiB, free %.2f -> %.2f MiB of %.2f MiB\n",
        __func__, buft_ctx->device, size / 1024.0 / 1024.0,
        free_before / 1024.0 / 1024.0, (free_before - size) / 1024.0 / 1024.0, total / 1024.0 / 1024.0);

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **, bool exact);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  `exact` asks
    // for a lossless reduction (e.g. for tensors that hold ids).  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;
    ggml_cuda_ar_push *         ar_push     = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
        ggml_cuda_ar_push_free(ar_push);
    }
};

#ifdef GGML_USE_NCCL
// FP32 vs BF16 selection for the NCCL reduction, shared by the all-rank and
// the per-rank entry points so both pick the same path for a given tensor.
//
// For small tensors, simply reduce them as FP32.
// The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
// exact skips both the heuristic and the BF16 compression, whose format
// cannot represent every value (e.g. large token ids).
static bool ggml_backend_cuda_comm_allreduce_nccl_use_fp32(
        const ggml_backend_cuda_comm_context * comm_ctx, const struct ggml_tensor * tensor, bool exact) {
    const int64_t ne         = ggml_nelements(tensor);
    const size_t  n_backends = comm_ctx->backends.size();
    return exact || (n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144);
}

// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32. `exact` forces the
// FP32 path regardless of size.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, bool exact) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    if (ggml_backend_cuda_comm_allreduce_nccl_use_fp32(comm_ctx, tensors[0], exact)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

// Per-rank variant of the NCCL reduction: enqueues rank's part on rank's
// stream without ncclGroupStart/End, so every rank can be driven from its own
// thread. All ranks must still call it for the same sequence of collectives.
static bool ggml_backend_cuda_comm_allreduce_nccl_rank(
        ggml_backend_cuda_comm_context * comm_ctx, size_t rank, struct ggml_tensor * tensor, bool exact) {
    GGML_ASSERT(rank < comm_ctx->backends.size());
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguously_allocated(tensor));

    const int64_t ne = ggml_nelements(tensor);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[rank]->context;
    ggml_cuda_set_device(cuda_ctx->device);

    if (ggml_backend_cuda_comm_allreduce_nccl_use_fp32(comm_ctx, tensor, exact)) {
        if ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            CUDA_CHECK(cudaMemsetAsync(tensor->data, 0, ggml_nbytes(tensor), cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclAllReduce(tensor->data, tensor->data, ne, ncclFloat, ncclSum, comm_ctx->comms[rank], cuda_ctx->stream()));
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp;
    tmp.pool = &cuda_ctx->pool();
    tmp.alloc(ne);

    if (tensor->flags & GGML_TENSOR_FLAG_COMPUTE) {
        to_bf16(tensor->data, tmp.get(), ne, cuda_ctx->stream());
    } else {
        CUDA_CHECK(cudaMemsetAsync(tmp.get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
    }
    CUDA_CHECK(cudaGetLastError());

    NCCL_CHECK(ncclAllReduce(tmp.get(), tmp.get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[rank], cuda_ctx->stream()));

    to_fp32(tmp.get(), (float *) tensor->data, ne, cuda_ctx->stream());
    CUDA_CHECK(cudaGetLastError());

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
// exact is forwarded to the pipeline: it disables the BF16 wire round-trip.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, bool exact) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors, exact);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, bool exact) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors, exact);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors, bool exact) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors, exact);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **, bool) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
#if defined(__powerpc64__)
    // IBM AC922 (POWER9): the GPUs of the two sockets reach each other through the
    // CPUs' NVLink ports. NCCL's default topology rules route those hops through
    // host shared memory (SHM/direct), which is much slower for the large prefill
    // all-reduces than direct P2P (4 x V100: 32K prefill TTFT 15.1 -> 13.6 s).
    // Allow P2P across the SMP interconnect unless the user chose a level.
    setenv("NCCL_P2P_LEVEL", "SYS", /*overwrite =*/ 0);
#endif
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    // Opportunistic low-latency path in front of the backend selected above.
    // It handles only small aligned F32 tensors and leaves everything else to
    // that backend.
    if (env == nullptr || std::string(env) != "none") {
        ret->ar_push = ggml_cuda_ar_push_init(ret->dev_ids.data(), ret->dev_ids.size());
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor_impl(
        void * comm_ctx_v, struct ggml_tensor ** tensors, bool exact) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    if (comm_ctx->ar_push != nullptr &&
        ggml_cuda_ar_push_allreduce(comm_ctx->ar_push, comm_ctx->backends.data(), tensors, exact)) {
        return true;
    }
    return comm_ctx->try_allreduce(comm_ctx, tensors, exact);
}

static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_tensor_impl(comm_ctx_v, tensors, false);
}

// Exact variant, for collectives whose payload must survive the reduction
// bitwise (the meta backend's TOP_K/GET_ROWS candidate buffers).
static bool ggml_backend_cuda_comm_allreduce_tensor_exact(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_tensor_impl(comm_ctx_v, tensors, true);
}

// Per-rank dispatch, for callers that drive every rank from its own thread.
// Only the push and the NCCL paths can be driven this way; internal and
// butterfly need cross-rank synchronization. Every rank picks its route from
// its own tensor (see below), so a collective is admitted only if all ranks
// agree on the route.
static bool ggml_backend_cuda_comm_allreduce_rank_can(
        void * comm_ctx_v, struct ggml_tensor ** tensors, bool exact) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    const size_t n = comm_ctx->backends.size();

    size_t n_empty = 0;
    size_t n_push  = 0;
    for (size_t i = 0; i < n; ++i) {
        if (tensors[i] == nullptr) {
            return false;
        }
        if (ggml_nelements(tensors[i]) == 0) {
            // nothing to do, as in the all-rank variants
            n_empty++;
        } else if (comm_ctx->ar_push != nullptr && ggml_cuda_ar_push_can(comm_ctx->ar_push, tensors[i], exact)) {
            n_push++;
        }
    }
    if (n_empty == n || n_push == n) {
        return true;
    }
    if (n_empty != 0 || n_push != 0) {
        return false;
    }
#ifdef GGML_USE_NCCL
    if (comm_ctx->comms.empty()) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (tensors[i]->type != GGML_TYPE_F32 || !ggml_is_contiguously_allocated(tensors[i]) ||
            ggml_nelements(tensors[i]) != ggml_nelements(tensors[0])) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(exact);
    return false;
#endif // GGML_USE_NCCL
}

// the push route is a plain kernel on the rank's stream that keeps its state on the device, so it
// can be recorded into a CUDA graph and replayed; the NCCL route is not recorded
static bool ggml_backend_cuda_comm_allreduce_rank_capturable(
        void * comm_ctx_v, struct ggml_tensor ** tensors, bool exact) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    for (size_t i = 0; i < comm_ctx->backends.size(); ++i) {
        if (tensors[i] == nullptr) {
            return false;
        }
        if (ggml_nelements(tensors[i]) != 0 &&
            (comm_ctx->ar_push == nullptr || !ggml_cuda_ar_push_can(comm_ctx->ar_push, tensors[i], exact))) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_cuda_comm_allreduce_rank(
        void * comm_ctx_v, size_t rank, struct ggml_tensor * tensor, bool exact) {
    GGML_ASSERT(comm_ctx_v != nullptr);
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    GGML_ASSERT(rank < comm_ctx->backends.size());

    if (tensor != nullptr && ggml_nelements(tensor) == 0) {
        // nothing to do, as in the all-rank variants
        return true;
    }
    if (comm_ctx->ar_push != nullptr && ggml_cuda_ar_push_can(comm_ctx->ar_push, tensor, exact)) {
        return ggml_cuda_ar_push_allreduce_rank(comm_ctx->ar_push, comm_ctx->backends[rank], rank, tensor, exact);
    }
#ifdef GGML_USE_NCCL
    if (!comm_ctx->comms.empty() && tensor != nullptr && tensor->type == GGML_TYPE_F32 &&
        ggml_is_contiguously_allocated(tensor)) {
        return ggml_backend_cuda_comm_allreduce_nccl_rank(comm_ctx, rank, tensor, exact);
    }
#endif // GGML_USE_NCCL

    // Calling with a tensor for which the corresponding `can` returned false is a caller error.
    GGML_ABORT("per-rank AllReduce not supported for this tensor");
    return false;
}

// ---------------------------------------------------------------------------
// Meta assist: mount the per-rank assist description on a CUDA context and run
// the idle-rank part of the collective (see ggml-backend.h). The owner part is
// executed inside the SM70 D256 FA launcher.
// ---------------------------------------------------------------------------

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static bool ggml_backend_cuda_set_assist(ggml_backend_t backend, const ggml_backend_meta_assist_rank * desc) {
    if (backend == nullptr || !GGML_CUDA_SM70_D256_COMPILED) {
        // the SM70 D256 object is absent from builds without sm_70
        return false;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    if (desc != nullptr) {
        if (desc->role != 0 && desc->role != 1) {
            return false;
        }
        if (desc->q_len < GGML_BACKEND_META_ASSIST_MIN_Q || desc->kv_view < GGML_BACKEND_META_ASSIST_MIN_KV) {
            return false;
        }
    }
    ggml_backend_cuda_context * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    cuda_ctx->assist = desc;
    return true;
}

static bool ggml_backend_cuda_assist_run(ggml_backend_t backend, const ggml_backend_meta_assist_rank * desc) {
    if (backend == nullptr || desc == nullptr) {
        return false;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    if (desc->role == 0) {
        return true; // the FA launcher of this rank merges the assist partial
    }
    if constexpr (GGML_CUDA_SM70_D256_COMPILED) {
        return ggml_cuda_sm70_d256_assist_run(backend, desc);
    } else {
        return false;
    }
}

static size_t ggml_backend_cuda_assist_workspace_size(int q_pad, int heads_q) {
    if constexpr (GGML_CUDA_SM70_D256_COMPILED) {
        return ggml_cuda_sm70_d256_assist_workspace_size(q_pad, heads_q);
    } else {
        return 0;
    }
}

static bool ggml_backend_cuda_assist_scratch_layout(const ggml_tensor * dst, ggml_backend_meta_assist_scratch * layout) {
    if constexpr (GGML_CUDA_SM70_D256_COMPILED) {
        return ggml_cuda_sm70_d256_assist_scratch_layout(dst, layout);
    } else {
        return false;
    }
}

static int ggml_backend_cuda_get_device_ordinal(ggml_backend_t backend) {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    return static_cast<ggml_backend_cuda_context *>(backend->context)->device;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

// NUMA helpers for the mapped host buffer
//
// The mapped host memory of a device should be allocated on the host NUMA node with the best
// measured read bandwidth for that device. The bandwidth matrix of all (device, node) pairs is
// measured once on the first mapped allocation and the node of each device is planned from it
// under the available memory of the nodes. The memory itself comes from cudaHostAlloc called under a
// set_mempolicy(MPOL_BIND) policy on the allocating thread.

#if !defined(GGML_USE_MUSA) && defined(GGML_CUDA_MAPPED_NUMA)

// list of online NUMA nodes
static const std::vector<int> & ggml_cuda_online_numa_nodes() {
    static const std::vector<int> nodes = []() {
        std::vector<int> ret;

        FILE * f = fopen("/sys/devices/system/node/online", "r");
        if (f == nullptr) {
            return ret;
        }

        // the file holds a CPU list such as "0-1" or "0,8"
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), f) != nullptr) {
            for (char * tok = strtok(buf, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
                int first = -1;
                int last  = -1;
                if (sscanf(tok, "%d-%d", &first, &last) == 2) {
                    for (int n = first; n <= last && n - first < 1024; n++) {
                        ret.push_back(n);
                    }
                } else if (sscanf(tok, "%d", &first) == 1) {
                    ret.push_back(first);
                }
            }
        }
        fclose(f);
        return ret;
    }();
    return nodes;
}

// a host NUMA node has CPUs and host memory; on POWER9 the device memory nodes of the GPUs
// (e.g. 250-255) show up in sysfs as well: their cpulist file exists but is empty, and a
// mapped host buffer must never be placed there. Two independent checks reject them: the
// cpulist must name at least one CPU, and the node directory must hold cpu<n> links. The
// links come from the CPU topology, the cpulist from the node mask, so a device memory node
// fails both even if one of the two files has unexpected content.
static bool ggml_cuda_numa_node_is_host(int node, const char ** reason) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);

    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        *reason = "no cpulist file";
        return false;
    }

    char buf[256] = {};
    const bool has_cpu = fgets(buf, sizeof(buf), f) != nullptr && strspn(buf, " \t\r\n") < strlen(buf);
    fclose(f);
    if (!has_cpu) {
        *reason = "empty cpulist (device memory node)";
        return false;
    }

    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", node);
    DIR * dir = opendir(path);
    if (dir != nullptr) {
        bool has_cpu_links = false;
        while (const struct dirent * entry = readdir(dir)) {
            if (strncmp(entry->d_name, "cpu", 3) == 0 && entry->d_name[3] >= '0' && entry->d_name[3] <= '9') {
                has_cpu_links = true;
                break;
            }
        }
        closedir(dir);
        if (!has_cpu_links) {
            *reason = "no cpu<n> links (device memory node)";
            return false;
        }
    }

    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    if (access(path, R_OK) != 0) {
        *reason = "meminfo not readable";
        return false;
    }

    *reason = nullptr;
    return true;
}

// total host memory of a NUMA node in bytes, -1 if unknown
//
// MemTotal is a fixed quantity of the machine, unlike MemFree plus the page cache, so the
// placement plan does not change from one run to the next. The margin is subtracted by the
// caller.
static int64_t ggml_cuda_numa_mem_total(int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);

    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }

    int64_t total_kb = -1;
    char line[256] = {};
    while (fgets(line, sizeof(line), f) != nullptr) {
        long long kb = 0;
        if (sscanf(line, "Node %*d MemTotal: %lld kB", &kb) == 1) {
            total_kb = (int64_t) kb;
        }
    }
    fclose(f);

    return total_kb >= 0 ? total_kb * 1024 : -1;
}

// highest NUMA node number in /sys/devices/system/node/possible, used to size the node mask.
// The file holds a node list such as "0,8,250-255". A single unsigned long only covers nodes
// 0..63, and this machine has GPU memory nodes up to 255, so the mask must be an array.
static int ggml_cuda_numa_max_node() {
    FILE * f = fopen("/sys/devices/system/node/possible", "r");
    if (f == nullptr) {
        return 1023;
    }

    int max_node = -1;
    char buf[256] = {};
    if (fgets(buf, sizeof(buf), f) != nullptr) {
        for (char * tok = strtok(buf, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
            int first = -1;
            int last  = -1;
            if (sscanf(tok, "%d-%d", &first, &last) == 2) {
                max_node = std::max(max_node, last);
            } else if (sscanf(tok, "%d", &first) == 1) {
                max_node = std::max(max_node, first);
            }
        }
    }
    fclose(f);

    return max_node >= 0 ? max_node : 1023;
}

// build a one-node node mask for set_mempolicy; returns the maxnode the kernel expects
static unsigned long ggml_cuda_nodemask_one(int node, std::vector<unsigned long> & nodemask) {
    const int bits = 8 * (int) sizeof(unsigned long);
    const int max_node = ggml_cuda_numa_max_node();

    // the kernel subtracts one from maxnode before reading the mask, so maxnode is the
    // number of mask bits plus one; the extra word keeps the target bit out of the last word
    nodemask.assign(max_node/bits + 2, 0);
    nodemask[node/bits] |= 1ul << (node % bits);
    return (unsigned long) nodemask.size() * bits + 1;
}

// the memory policy of the calling thread saved by ggml_cuda_mempolicy_save(), restored by
// ggml_cuda_mempolicy_restore()
struct ggml_cuda_mempolicy_saved {
    int mode = MPOL_DEFAULT;
    std::vector<unsigned long> nodemask;
    bool valid = false;
};

// read the memory policy of the calling thread. Restoring the exact policy matters: a thread
// pinned by numactl --membind must not fall back to the default policy, or its later file
// reads fill the page cache of the device memory NUMA nodes once the host nodes run full.
static bool ggml_cuda_mempolicy_save(ggml_cuda_mempolicy_saved & saved) {
    // the mask must cover every node the current policy can name (up to the max node here)
    const int bits = 8 * (int) sizeof(unsigned long);
    saved.nodemask.assign(ggml_cuda_numa_max_node()/bits + 2, 0);
    const unsigned long maxnode = (unsigned long) saved.nodemask.size() * bits + 1;

    if (syscall(SYS_get_mempolicy, &saved.mode, saved.nodemask.data(), maxnode, (void *) nullptr, 0) != 0) {
        GGML_LOG_WARN("%s: failed to read the current memory policy: %s\n", __func__, strerror(errno));
        return false;
    }
    saved.valid = true;
    return true;
}

// restore a policy saved by ggml_cuda_mempolicy_save(); no-op when nothing was saved. The
// caller must restore before any later allocation it does not want under the bound policy.
static void ggml_cuda_mempolicy_restore(const ggml_cuda_mempolicy_saved & saved) {
    if (!saved.valid) {
        return;
    }
    // MPOL_DEFAULT takes an empty node mask, every other mode restores with the saved one
    const bool plain = saved.mode == MPOL_DEFAULT;
    const int bits = 8 * (int) sizeof(unsigned long);
    const unsigned long maxnode = plain ? 0 : (unsigned long) saved.nodemask.size() * bits + 1;
    if (syscall(SYS_set_mempolicy, saved.mode, plain ? (const unsigned long *) nullptr : saved.nodemask.data(), maxnode) != 0) {
        GGML_LOG_WARN("%s: failed to restore the saved memory policy (mode %d): %s\n",
                      __func__, saved.mode, strerror(errno));
    }
}

// bind the memory policy of the calling thread to one NUMA node; cudaHostAlloc allocations made
// by the thread then come from that node. The previous policy is saved into `saved` and must be
// restored with ggml_cuda_mempolicy_restore() after the allocation. When this returns false the
// policy is unchanged and no restore is needed.
static bool ggml_cuda_set_mempolicy_node(int node, ggml_cuda_mempolicy_saved & saved) {
    if (!ggml_cuda_mempolicy_save(saved)) {
        return false;
    }

    std::vector<unsigned long> nodemask;
    const unsigned long maxnode = ggml_cuda_nodemask_one(node, nodemask);

    if (syscall(SYS_set_mempolicy, MPOL_BIND, nodemask.data(), maxnode) != 0) {
        GGML_LOG_WARN("%s: failed to bind the memory policy to NUMA node %d (maxnode %lu): %s\n",
                      __func__, node, maxnode, strerror(errno));
        saved.valid = false; // the policy was not changed, nothing to restore
        return false;
    }
    return true;
}

// verify that the pages of a faulted range are on the requested node; samples at most 16 pages.
// sampled and on_node receive the number of sampled pages and the number of them that are there
static void ggml_cuda_mapped_buffer_verify_numa(void * ptr, size_t size, int node,
        size_t * sampled, size_t * on_node) {
    *sampled = 0;
    *on_node = 0;
#if defined(SYS_move_pages)
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t n_pages = size / page;
    if (n_pages == 0) {
        return;
    }

    // spread the samples over the whole range instead of checking only the first page
    const size_t n_samples = std::min<size_t>(n_pages, 16);
    std::vector<void *> pages(n_samples);
    std::vector<int>    status(n_samples, -1);
    for (size_t i = 0; i < n_samples; i++) {
        pages[i] = (char *) ptr + (n_pages/n_samples)*i*page;
    }

    // a null nodes array makes the call return the node of each page in status
    if (syscall(SYS_move_pages, 0, n_samples, pages.data(), nullptr, status.data(), 0) != 0) {
        GGML_LOG_WARN("%s: failed to query the NUMA node of %.2f MiB of host memory: %s\n",
                      __func__, size / 1024.0 / 1024.0, strerror(errno));
        return;
    }

    size_t hits = 0;
    for (size_t i = 0; i < n_samples; i++) {
        hits += status[i] == node;
    }
    *sampled = n_samples;
    *on_node = hits;
    if (hits == n_samples) {
        GGML_LOG_INFO("%s: verified %.2f MiB of host memory on NUMA node %d (%zu/%zu sampled pages)\n",
                      __func__, size / 1024.0 / 1024.0, node, hits, n_samples);
    } else {
        GGML_LOG_WARN("%s: %.2f MiB of host memory is only partly on NUMA node %d (%zu/%zu sampled pages)\n",
                      __func__, size / 1024.0 / 1024.0, node, hits, n_samples);
    }
#else
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(node);
#endif
}

// CPU side of the host placement: the CPUs that allocate and fill device-readable host memory
//
// A GPU reads the pages of its own node fastest when the CPUs of that node wrote them last: on
// the AC922 44 GB/s, against 17-30 GB/s for pages of its node that the other socket wrote last
// and about 40 GB/s for pages of the other node (tests/bench-pinned-read.cu, 6 x V100, nodes 0
// and 8). The memory policy decides the node of the pages; the thread that allocates them (the
// driver clears the pages on it) and the thread that fills them decide which socket wrote them
// last. So every allocation of such memory runs on a thread bound to the CPUs of its node, and
// every fill of it binds the filling thread to those CPUs for the duration of the copy.

// CPUs of a host NUMA node (sysfs cpulist), false when the node has none
static bool ggml_cuda_numa_node_cpuset(int node, cpu_set_t & set) {
    CPU_ZERO(&set);
    if (node < 0) {
        return false;
    }
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    char buf[1024] = {};
    if (fgets(buf, sizeof(buf), f) != nullptr) {
        for (char * tok = strtok(buf, ",\n"); tok != nullptr; tok = strtok(nullptr, ",\n")) {
            int first = -1;
            int last  = -1;
            if (sscanf(tok, "%d-%d", &first, &last) == 2) {
                for (int c = std::max(first, 0); c <= last && c < CPU_SETSIZE; c++) {
                    CPU_SET(c, &set);
                }
            } else if (sscanf(tok, "%d", &first) == 1 && first >= 0 && first < CPU_SETSIZE) {
                CPU_SET(first, &set);
            }
        }
    }
    fclose(f);
    return CPU_COUNT(&set) > 0;
}

// affinity of the calling thread before ggml_cuda_numa_bind_cpus(), one per thread
static thread_local cpu_set_t ggml_cuda_numa_bind_saved;
static thread_local bool      ggml_cuda_numa_bind_active = false;

bool ggml_cuda_numa_bind_cpus(int node) {
    if (ggml_cuda_numa_bind_active) {
        return false; // an outer call holds the binding and its unbind restores the affinity
    }
    cpu_set_t set;
    if (!ggml_cuda_numa_node_cpuset(node, set)) {
        return false;
    }
    int err = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &ggml_cuda_numa_bind_saved);
    if (err != 0) {
        GGML_LOG_WARN("%s: cannot read the CPU affinity of the thread: %s\n", __func__, strerror(err));
        return false;
    }
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
    if (err != 0) {
        GGML_LOG_WARN("%s: cannot bind the thread to the CPUs of NUMA node %d: %s\n", __func__, node, strerror(err));
        return false;
    }
    ggml_cuda_numa_bind_active = true;
    return true;
}

void ggml_cuda_numa_unbind_cpus(void) {
    if (!ggml_cuda_numa_bind_active) {
        return;
    }
    ggml_cuda_numa_bind_active = false;
    const int err = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &ggml_cuda_numa_bind_saved);
    if (err != 0) {
        GGML_LOG_WARN("%s: cannot restore the CPU affinity of the thread: %s\n", __func__, strerror(err));
    }
}

// cudaHostAlloc of `size` bytes of mapped host memory placed on NUMA node `node`: the
// allocation runs on a thread bound to the CPUs of the node under a memory policy bound to it,
// so the pages belong to the node and the driver's clearing of them was done by its CPUs. The
// pages of a pinned buffer are allocated and pinned inside the cudaHostAlloc call under the
// policy of the calling thread, which decides their node; nothing moves them later (measured:
// the pages of a 1 GiB buffer are all on the policy's node before any mbind, and an mbind of
// the range afterwards changes nothing). node < 0: a plain allocation on the calling thread
// without a policy. Returns nullptr on failure (logged); *bound receives whether the pages were
// allocated under the policy of the node (the caller verifies them with move_pages).
static void * ggml_cuda_numa_host_alloc_on_node(int device, size_t size, int node, bool * bound) {
    *bound = false;
    void *      ptr    = nullptr;
    cudaError_t err    = cudaSuccess;
    bool        policy = false;
    if (node >= 0) {
        std::thread([&]() {
            const bool cpus = ggml_cuda_numa_bind_cpus(node);
            ggml_cuda_set_device(device);
            ggml_cuda_mempolicy_saved saved_policy;
            policy = ggml_cuda_set_mempolicy_node(node, saved_policy);
            err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
            ggml_cuda_mempolicy_restore(saved_policy);
            if (cpus) {
                ggml_cuda_numa_unbind_cpus();
            }
        }).join();
    } else {
        err = cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
    }
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB of mapped host memory on NUMA node %d: cudaHostAlloc failed: %s\n",
                       __func__, size / 1024.0 / 1024.0, node, cudaGetErrorString(err));
        return nullptr;
    }
    *bound = policy;
    return ptr;
}

// write `size` bytes of host memory that lives on NUMA node `node` from the CPUs of that node
static void ggml_cuda_numa_fill_on_node(void * ptr, size_t size, int node, int value) {
    std::thread([&]() {
        const bool cpus = ggml_cuda_numa_bind_cpus(node);
        memset(ptr, value, size);
        if (cpus) {
            ggml_cuda_numa_unbind_cpus();
        }
    }).join();
}

// bandwidth probe and placement of the mapped host buffers
//
// The read bandwidth of a GPU to a host NUMA node cannot be derived from the PCI topology: on
// AC922 some GPUs read the second CPU faster than the first one, and "local" is not always the
// best node. The bandwidth is measured instead. The experts of all devices are read at the same
// time in a prefill step, so the read times are measured under that contention: for one
// candidate assignment every device reads a probe buffer on its own node of the assignment at
// the same time, and the assignment with the lowest step time wins. The bandwidth of a single
// device can not decide this, the devices share the total bandwidth of a node.
//
// The placement is planned once for the whole machine from the bytes every device reads and
// the MemTotal of the nodes, and the node of a device is then fixed for every host allocation
// of that device. The solver is a pure function of those inputs (ggml-cuda-numa-plan.h), so
// fit and the real allocation share one plan and the fit figures are the ones used at load.

#define GGML_CUDA_MAPPED_PROBE_SIZE  (128ull << 20) // bytes of the probe buffer per (device, node)
#define GGML_CUDA_MAPPED_PROBE_REPS  3              // reads per assignment, the median is kept
#define GGML_CUDA_MAPPED_NUMA_MARGIN (4ull << 30)   // host memory kept free on every node

// return values that are not a NUMA node number (GGML_CUDA_NUMA_NODE_ANY / _FAIL) are
// defined in ggml-cuda-expert.cuh and shared with the expert host arena

// read the buffer with 16-byte loads; one xor per block keeps the loads alive
static __global__ void ggml_cuda_mapped_probe_read_kernel(const uint4 * __restrict__ src, size_t n_vec, unsigned int * __restrict__ sink) {
    const size_t tid    = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x  * blockDim.x;

    unsigned int acc = 0;
    for (size_t i = tid; i < n_vec; i += stride) {
        const uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }

    __shared__ unsigned int sdata[256];
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            sdata[threadIdx.x] ^= sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(sink, sdata[0]);
    }
}

// one probe buffer per (physical device, node): pinned host memory placed on that node with the
// same allocation path as the real arenas, plus the device pointer the read kernel uses
struct ggml_cuda_numa_probe_buf {
    void * host = nullptr;
    void * dev  = nullptr;
};

// stream, events and kernel sink of one concurrent read on a physical device
struct ggml_cuda_numa_probe_dev {
    cudaStream_t  stream   = nullptr;
    cudaEvent_t   ev_start = nullptr;
    cudaEvent_t   ev_stop  = nullptr;
    unsigned int * sink     = nullptr;
};

struct ggml_cuda_mapped_numa_state {
    bool probed = false;

    std::vector<int>     nodes;   // host NUMA nodes with memory and CPUs
    std::vector<int64_t> cap;    // [node] bytes the planner may fill (MemTotal minus the margin)
    std::vector<int>     topo;   // [physical device] node of the PCI slot, -1 when unknown
    std::vector<int>     phys_of; // [device] physical device it runs on

    std::vector<std::vector<ggml_cuda_numa_probe_buf>> probe; // [physical device][node]
    std::vector<ggml_cuda_numa_probe_dev> probe_dev;           // [physical device]
    std::vector<std::vector<double>> t_ms;                     // [physical device][node] read time
                                                                   // in ms under contention, -1 when
                                                                   // the pair is unusable
    std::vector<int> t_ms_dev; // devices t_ms was measured for, the read times of a set of
                               // devices do not depend on the bytes they read

    // the placement is decided once for the whole machine and then frozen: fit computes it from
    // the host bytes of every device and commits it, the load reads the committed nodes
    ggml_cuda_numa_plan::committed plan;
    double                       plan_ms        = 0.0; // estimated step time of the plan, in ms
    double                       plan_ms_fast   = 0.0; // every device on its own fastest node, in ms
    double                       plan_ms_local  = 0.0; // every device on the node of its PCI slot, in ms
    bool                         plan_have_local = false; // the PCI slot node is known for every device

    std::vector<size_t> placed;                      // [device] live device-readable host bytes
    // the mapped buffers and the expert host arenas share the nodes, so all live
    // allocations are accounted per node as well; placed_node holds the cross-device total
    // used by the logs, placed_dev_node the per-device bytes
    std::vector<size_t> placed_node;                  // [node index]
    std::vector<std::vector<size_t>> placed_dev_node; // [device][node index]

    // node the first arena of a device got, and the pages move_pages confirmed there: the final
    // placement log compares the node with the plan and reports the samples per device
    std::vector<int>    arena_node;    // [device] node number of the arenas, -1 when none
    std::vector<size_t> verify_pages;  // [device] sampled pages
    std::vector<size_t> verify_on_node;// [device] sampled pages that are on that node
};

static ggml_cuda_mapped_numa_state ggml_cuda_mapped_numa;
static std::mutex ggml_cuda_mapped_numa_mutex;

// index of a node in the state, -1 if absent
static int ggml_cuda_mapped_numa_index(int node) {
    const std::vector<int> & nodes = ggml_cuda_mapped_numa.nodes;
    const auto it = std::find(nodes.begin(), nodes.end(), node);
    return it == nodes.end() ? -1 : (int) (it - nodes.begin());
}

// release the stream, the events and the sink of the concurrent read of one device
static void ggml_cuda_numa_probe_dev_free(ggml_cuda_numa_probe_dev & pd) {
    if (pd.stream   != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(pd.stream));
        pd.stream = nullptr;
    }
    if (pd.ev_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(pd.ev_start));
        pd.ev_start = nullptr;
    }
    if (pd.ev_stop  != nullptr) {
        CUDA_CHECK(cudaEventDestroy(pd.ev_stop));
        pd.ev_stop  = nullptr;
    }
    if (pd.sink     != nullptr) {
        CUDA_CHECK(cudaFree(pd.sink));
        pd.sink = nullptr;
    }
}

// release the probe buffers and the streams of every physical device
static void ggml_cuda_numa_probe_free(ggml_cuda_mapped_numa_state & state) {
    for (std::vector<ggml_cuda_numa_probe_buf> & bufs : state.probe) {
        for (ggml_cuda_numa_probe_buf & buf : bufs) {
            if (buf.host != nullptr) {
                CUDA_CHECK(cudaFreeHost(buf.host));
                buf.host = nullptr;
                buf.dev  = nullptr;
            }
        }
    }
    for (ggml_cuda_numa_probe_dev & pd : state.probe_dev) {
        ggml_cuda_numa_probe_dev_free(pd);
    }
    state.probe.clear();
    state.probe_dev.clear();
}

// allocate the probe buffer of every node of one physical device plus the stream and the
// events of the concurrent reads. A node whose buffer can not be allocated keeps a null entry
// and is left out of the plan: the machine then places that device on one of the other nodes.
static void ggml_cuda_numa_probe_alloc(ggml_cuda_mapped_numa_state & state, int phys) {
    ggml_cuda_set_device(phys);

    state.probe[phys].assign(state.nodes.size(), ggml_cuda_numa_probe_buf());
    for (size_t n = 0; n < state.nodes.size(); n++) {
        ggml_cuda_numa_probe_buf & buf = state.probe[phys][n];

        // allocated like an arena: on a thread of the node, under its policy
        bool bound = false;
        buf.host = ggml_cuda_numa_host_alloc_on_node(phys, GGML_CUDA_MAPPED_PROBE_SIZE, state.nodes[n], &bound);
        if (buf.host == nullptr) {
            GGML_LOG_WARN("%s: physical device %d, node %d: the probe buffer could not be allocated\n",
                __func__, phys, state.nodes[n]);
            continue;
        }
        if (!bound) {
            // the probe measures one node; a buffer of unknown placement measures nothing
            GGML_LOG_WARN("%s: physical device %d, node %d: the probe buffer could not be allocated under the policy of the node\n",
                __func__, phys, state.nodes[n]);
            CUDA_CHECK(cudaFreeHost(buf.host));
            buf.host = nullptr;
            continue;
        }
        // the probe reads what an arena reads: pages the CPUs of their own node wrote last
        // (the loader fills the arenas from those CPUs, see ggml_cuda_numa_bind_cpus)
        ggml_cuda_numa_fill_on_node(buf.host, GGML_CUDA_MAPPED_PROBE_SIZE, state.nodes[n], 0x5a);
        if (cudaHostGetDevicePointer(&buf.dev, buf.host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            CUDA_CHECK(cudaFreeHost(buf.host));
            buf.host = nullptr;
        }
    }

    ggml_cuda_numa_probe_dev & pd = state.probe_dev[phys];
    if (cudaStreamCreateWithFlags(&pd.stream, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: physical device %d: no stream for the probe, the device takes no part in the placement\n",
            __func__, phys);
        return;
    }
    if (cudaEventCreate(&pd.ev_start) != cudaSuccess || cudaEventCreate(&pd.ev_stop) != cudaSuccess ||
        cudaMalloc((void **) &pd.sink, sizeof(unsigned int)) != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: physical device %d: the probe setup failed, the device takes no part in the placement\n",
            __func__, phys);
        ggml_cuda_numa_probe_dev_free(pd);
        return;
    }
    CUDA_CHECK(cudaMemset(pd.sink, 0, sizeof(unsigned int)));
}

// make sure every reading physical device has its probe buffers and its stream
static void ggml_cuda_numa_probe_prepare(ggml_cuda_mapped_numa_state & state, const std::vector<int> & devices) {
    if (state.probe.size() < state.phys_of.size()) {
        state.probe.assign(state.phys_of.size(), std::vector<ggml_cuda_numa_probe_buf>());
        state.probe_dev.assign(state.phys_of.size(), ggml_cuda_numa_probe_dev());
    }

    std::vector<int> done;
    for (const int phys : devices) {
        if (phys < 0 || (size_t) phys >= state.probe.size()) {
            continue;
        }
        if (std::find(done.begin(), done.end(), phys) != done.end()) {
            continue; // one probe covers the virtual devices of that GPU
        }
        done.push_back(phys);

        if (!state.probe[phys].empty() && state.probe_dev[phys].sink != nullptr) {
            continue; // already probed
        }
        ggml_cuda_numa_probe_alloc(state, phys);
    }
}

// one concurrent read: every device reads the probe buffer of the node it gets in the
// assignment at the same time. ms receives the read time of every device in ms, -1 for the
// devices that have no probe buffer or no stream. The caller holds the mutex.
static void ggml_cuda_numa_probe_once(const ggml_cuda_mapped_numa_state & state, const std::vector<int> & assign,
        std::vector<double> & ms) {
    const size_t n_dev = assign.size();
    ms.assign(n_dev, -1.0);

    for (size_t d = 0; d < n_dev; d++) {
        const ggml_cuda_numa_probe_dev & pd = state.probe_dev[d];
        if (pd.sink == nullptr || state.probe[d].empty()) {
            continue;
        }
        const ggml_cuda_numa_probe_buf & buf = state.probe[d][assign[d]];
        if (buf.dev == nullptr) {
            continue;
        }

        ggml_cuda_set_device(d);
        const size_t n_vec = GGML_CUDA_MAPPED_PROBE_SIZE / sizeof(uint4);
        const int    nsm  = std::max(1, ggml_cuda_info().devices[d].nsm);

        CUDA_CHECK(cudaEventRecord(pd.ev_start, pd.stream));
        ggml_cuda_mapped_probe_read_kernel<<<nsm*8, 256, 0, pd.stream>>>((const uint4 *) buf.dev, n_vec, pd.sink);
        CUDA_CHECK(cudaEventRecord(pd.ev_stop, pd.stream));
    }

    for (size_t d = 0; d < n_dev; d++) {
        const ggml_cuda_numa_probe_dev  & pd = state.probe_dev[d];
        if (pd.sink == nullptr || state.probe[d].empty() || state.probe[d][assign[d]].dev == nullptr) {
            continue;
        }

        CUDA_CHECK(cudaStreamSynchronize(pd.stream));

        float t = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t, pd.ev_start, pd.ev_stop));
        ms[d] = (double) t;
    }
}

// the assignments the probe measures: all of them when there are few, otherwise a covering set
// that puts every device on every node a few times, so that every read time is measured under
// a comparable amount of contention. Devices that read nothing do not take part and keep node 0.
static std::vector<std::vector<int>> ggml_cuda_numa_probe_candidates(const ggml_cuda_mapped_numa_state & state,
        const std::vector<int> & part) {
    const size_t n_nodes = state.nodes.size();
    const size_t n_dev   = state.phys_of.size();

    uint64_t count      = 1;
    bool     exhaustive = true;
    for (size_t i = 0; i < part.size(); i++) {
        if (count > ggml_cuda_numa_plan::EXHAUSTIVE) {
            exhaustive = false;
            break;
        }
        count *= n_nodes;
    }
    if (exhaustive && count > ggml_cuda_numa_plan::EXHAUSTIVE) {
        exhaustive = false;
    }

    std::vector<std::vector<int>> cands;
    if (exhaustive) {
        for (uint64_t idx = 0; idx < count; idx++) {
            std::vector<int> a(n_dev, 0);
            uint64_t v = idx;
            for (size_t i = 0; i < part.size(); i++) {
                a[part[i]] = (int) (v % n_nodes);
                v /= n_nodes;
            }
            cands.push_back(std::move(a));
        }
        return cands;
    }

    for (int off = 0; off < 3; off++) {
        for (size_t n = 0; n < n_nodes; n++) {
            for (size_t i = 0; i < part.size(); i++) {
                std::vector<int> a(n_dev, 0);
                a[part[(i + (size_t) off) % part.size()]] = (int) n;
                cands.push_back(std::move(a));
            }
        }
    }
    cands.push_back(std::vector<int>(n_dev, 0));

    return cands;
}

// read an integer from a sysfs file, -1 when the file is missing or has no number
static int ggml_cuda_sysfs_read_int(const char * path) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }
    int value = -1;
    if (fscanf(f, "%d", &value) != 1) {
        value = -1;
    }
    fclose(f);
    return value;
}

// node of the NUMA socket a GPU is plugged into, -1 when the platform does not say. The node is
// not the placement, only the log and the slot local comparison: the placement comes from the
// measured read times.
static int ggml_cuda_numa_topology_node(int device) {
    char bdf[32] = {};
    const int physical = ggml_cuda_get_physical_device(device);
    if (cudaDeviceGetPCIBusId(bdf, sizeof(bdf), physical) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    for (char * p = bdf; *p != '\0'; p++) {
        // sysfs spells the device as 0000:41:00.0 in lower case
        if (*p >= 'A' && *p <= 'Z') {
            *p = (char) (*p - 'A' + 'a');
        }
    }

    char path[160];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bdf);

    const int node = ggml_cuda_sysfs_read_int(path);
    if (node < 0) {
        return -1;
    }

    const char * reason = nullptr;
    return ggml_cuda_numa_node_is_host(node, &reason) ? node : -1;
}

// estimated step time of a plan cost: the solver counts byte*ms of the probe buffer, so a
// device that reads b bytes of experts spends b*t/probe_size ms
static double ggml_cuda_numa_plan_ms(double cost) {
    return cost/(double) GGML_CUDA_MAPPED_PROBE_SIZE;
}

// collect the host NUMA nodes, their capacity, the physical device of every device and the
// PCI slot node. Does not probe and does not plan: both need to know which devices read
// something. The caller holds the mutex and has not probed.
static void ggml_cuda_numa_discover(ggml_cuda_mapped_numa_state & state) {
    // only nodes that are known to hold host memory are candidates; device memory nodes are
    // listed with the reason for their rejection, and the plan only covers the candidates, so
    // no device can ever be placed on a device memory node
    for (int node : ggml_cuda_online_numa_nodes()) {
        const char * reason = nullptr;
        if (ggml_cuda_numa_node_is_host(node, &reason)) {
            GGML_LOG_INFO("%s: NUMA node %d: candidate (CPUs and host memory)\n", __func__, node);
            state.nodes.push_back(node);
        } else {
            GGML_LOG_INFO("%s: NUMA node %d: excluded (%s)\n", __func__, node, reason);
        }
    }

    const int device_count = ggml_backend_cuda_get_device_count();

    state.placed.assign(device_count, 0);
    state.placed_node.assign(state.nodes.size(), 0);
    state.placed_dev_node.assign(device_count, std::vector<size_t>(state.nodes.size(), 0));
    state.arena_node.assign(device_count, -1);
    state.verify_pages.assign(device_count, 0);
    state.verify_on_node.assign(device_count, 0);

    // the capacity of a node is its MemTotal minus the margin: a fixed property of the machine,
    // unlike MemFree plus the page cache, so the plan does not change from one run to the next
    state.cap.assign(state.nodes.size(), 0);
    for (size_t n = 0; n < state.nodes.size(); n++) {
        const int64_t total = ggml_cuda_numa_mem_total(state.nodes[n]);
        if (total < 0) {
            GGML_LOG_ERROR("%s: NUMA node %d: MemTotal unknown, the host arenas get no NUMA policy\n",
                __func__, state.nodes[n]);
            state.nodes.clear();
            break;
        }
        state.cap[n] = total - (int64_t) GGML_CUDA_MAPPED_NUMA_MARGIN;
        GGML_LOG_INFO("%s: NUMA node %d: %.1f GiB of %.1f GiB for the host arenas (MemTotal minus a %.1f GiB margin)\n",
            __func__, state.nodes[n], (double) state.cap[n] / 1024.0/1024.0/1024.0,
            (double) total / 1024.0/1024.0/1024.0,
            (double) GGML_CUDA_MAPPED_NUMA_MARGIN / 1024.0/1024.0/1024.0);
    }

    if (state.nodes.empty()) {
        GGML_LOG_WARN("%s: no NUMA node holds host memory, mapped host buffers get no NUMA policy\n", __func__);
        state.probed = true;
        return;
    }

    // the probe, the read times and the plan are per physical device: the virtual devices that
    // share a GPU read the same node at the same rate
    state.phys_of.resize(device_count);
    state.topo.assign(device_count, -1);
    for (int device = 0; device < device_count; device++) {
        state.phys_of[device] = ggml_cuda_get_physical_device(device);
        state.topo[state.phys_of[device]] = ggml_cuda_numa_topology_node(device);
    }

    state.probe.assign(device_count, std::vector<ggml_cuda_numa_probe_buf>());
    state.probe_dev.assign(device_count, ggml_cuda_numa_probe_dev());

    for (int device = 0; device < device_count; device++) {
        const int phys = state.phys_of[device];
        if (state.topo[phys] >= 0) {
            GGML_LOG_INFO("%s: device %d is plugged into NUMA node %d (for the log only, the placement is measured)\n",
                __func__, phys, state.topo[phys]);
        } else {
            GGML_LOG_INFO("%s: device %d: the platform reports no NUMA node for the PCI slot\n", __func__, phys);
        }
    }

    GGML_LOG_INFO("%s: device-readable host memory is allocated on a thread of its NUMA node and written by the CPUs of that node\n",
        __func__);

    state.probed = true;
}

// host bytes of every physical device: the virtual devices that share a GPU read the same node
// at the same rate, so their bytes are added up
static std::vector<size_t> ggml_cuda_numa_fold_locked(const std::vector<size_t> & dev_bytes) {
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    const size_t n_dev = state.phys_of.size();
    std::vector<size_t> bytes(n_dev, 0);
    for (size_t d = 0; d < dev_bytes.size() && d < n_dev; d++) {
        bytes[state.phys_of[d]] += dev_bytes[d];
    }
    return bytes;
}

// plan the node of every device of the machine and freeze the result; dev_bytes holds the host
// bytes of every device (the expert host arena plus what the mapped buffer type already placed).
// from_fit marks the plan fit reported: the load then keeps its nodes even when its bytes differ.
// A plan that exists is replaced, a new set of bytes is a new placement: the caller either is a
// fit trial or registers the machine for the first time.
// Returns false when the bytes do not fit into the node capacities, after reporting the deficit.
// The caller holds the mutex.
static bool ggml_cuda_numa_plan_locked(const std::vector<size_t> & dev_bytes, bool from_fit) {
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    const size_t n_nodes = state.nodes.size();
    const size_t n_dev   = state.phys_of.size();
    if (n_nodes == 0 || n_dev == 0) {
        return false;
    }

    const std::vector<size_t> bytes = ggml_cuda_numa_fold_locked(dev_bytes);

    // the devices that read something take part in the decision
    std::vector<int> part;
    for (size_t d = 0; d < n_dev; d++) {
        if (bytes[d] > 0) {
            part.push_back((int) d);
        }
    }

    if (part.empty()) {
        ggml_cuda_numa_plan::commit_plan(&state.plan, std::vector<int>(n_dev, 0), bytes, from_fit);
        state.plan_ms = state.plan_ms_fast = state.plan_ms_local = 0.0;
        return true;
    }

    // the read times depend on which devices read at the same time, not on how many bytes they
    // read: a replan of the same set of devices reuses them
    if (state.t_ms_dev != part) {
        if (n_nodes > 1 && part.size() > 1) {
            // measure the read time of every (device, node) pair while all the reading devices read
            ggml_cuda_numa_probe_prepare(state, part);
            const std::vector<std::vector<int>> cands = ggml_cuda_numa_probe_candidates(state, part);
            GGML_LOG_INFO("%s: probing the concurrent read time of %zu device(s) from %zu NUMA node(s), %zu assignment(s) x %d reads\n",
                __func__, part.size(), n_nodes, cands.size(), GGML_CUDA_MAPPED_PROBE_REPS);

            std::vector<std::vector<std::vector<double>>> samples(cands.size());
            std::vector<double> ms;
            for (size_t c = 0; c < cands.size(); c++) {
                samples[c].resize(part.size());
                for (int r = 0; r < GGML_CUDA_MAPPED_PROBE_REPS; r++) {
                    ggml_cuda_numa_probe_once(state, cands[c], ms);
                    for (size_t i = 0; i < part.size(); i++) {
                        samples[c][i].push_back(ms[part[i]]);
                    }
                }
            }

            // the devices that read nothing do not constrain the plan, they keep a positive placeholder
            state.t_ms.assign(n_dev, std::vector<double>(n_nodes, 1.0));
            for (size_t i = 0; i < part.size(); i++) {
                const int device = part[i];
                for (size_t n = 0; n < n_nodes; n++) {
                    std::vector<double> times;
                    for (size_t c = 0; c < cands.size(); c++) {
                        if (cands[c][device] != (int) n) {
                            continue;
                        }
                        for (const double t : samples[c][i]) {
                            if (t >= 0.0) {
                                times.push_back(t);
                            }
                        }
                    }
                    std::sort(times.begin(), times.end());
                    state.t_ms[device][n] = times.empty() ? -1.0 : times[times.size()/2];
                }
            }

            GGML_LOG_INFO("%s: concurrent read time of the %.0f MiB probe buffer in ms, rows: device, columns: node\n",
                __func__, (double) GGML_CUDA_MAPPED_PROBE_SIZE / 1024.0/1024.0);
            for (int device : part) {
                std::string row = "  CUDA" + std::to_string(device) + ":";
                for (size_t n = 0; n < n_nodes; n++) {
                    char cell[32];
                    if (state.t_ms[device][n] >= 0.0) {
                        snprintf(cell, sizeof(cell), " node %d: %.1f", state.nodes[n], state.t_ms[device][n]);
                    } else {
                        snprintf(cell, sizeof(cell), " node %d: n/a", state.nodes[n]);
                    }
                    row += cell;
                }
                GGML_LOG_INFO("%s:%s\n", __func__, row.c_str());
            }

            // the load needs the pinned memory for the arenas, the read times stay
            ggml_cuda_numa_probe_free(state);
        } else {
            // one node, or one reading device: there is no contention to measure, the read time is
            // only needed to compare the assignments, which all cost the same then
            GGML_LOG_INFO("%s: %zu host NUMA node(s) and %zu reading device(s), no contention to measure\n",
                __func__, n_nodes, part.size());
            state.t_ms.assign(n_dev, std::vector<double>(n_nodes, 1.0));
        }
        state.t_ms_dev = part;
    }

    std::vector<double> t_ms(n_dev*n_nodes);
    for (size_t d = 0; d < n_dev; d++) {
        for (size_t n = 0; n < n_nodes; n++) {
            t_ms[d*n_nodes + n] = state.t_ms[d][n];
        }
    }

    ggml_cuda_numa_plan::result res;
    const int ret = ggml_cuda_numa_plan::plan(n_dev, n_nodes, bytes.data(), state.cap.data(),
        t_ms.data(), state.topo.data(), &res);
    if (ret != 1) {
        // the nodes together do not hold the arenas: report what is needed, what is there and
        // the gap, instead of placing them anywhere
        int64_t need_total = 0;
        int64_t cap_total  = 0;
        for (size_t d = 0; d < n_dev; d++) {
            need_total += (int64_t) bytes[d];
        }
        for (size_t n = 0; n < n_nodes; n++) {
            cap_total += state.cap[n];
        }
        GGML_LOG_ERROR("%s: the host arenas need %.1f MiB of NUMA node memory, the nodes hold %.1f MiB (short by %.1f MiB)\n",
            __func__, (double) need_total / 1024.0/1024.0, (double) cap_total / 1024.0/1024.0,
            (double) (need_total - cap_total > 0 ? need_total - cap_total : 0) / 1024.0/1024.0);
        for (size_t n = 0; n < n_nodes; n++) {
            GGML_LOG_ERROR("%s:   NUMA node %d: %.1f GiB available\n", __func__, state.nodes[n],
                (double) state.cap[n] / 1024.0/1024.0/1024.0);
        }
        state.plan.valid = false;
        ggml_cuda_numa_probe_free(state);
        return false;
    }

    ggml_cuda_numa_plan::commit_plan(&state.plan, res.node, bytes, from_fit);
    state.plan_ms          = ggml_cuda_numa_plan_ms(res.cost);
    state.plan_ms_fast     = ggml_cuda_numa_plan_ms(res.cost_fast);
    state.plan_ms_local    = ggml_cuda_numa_plan_ms(res.cost_local);
    state.plan_have_local  = res.have_local;

    size_t total = 0;
    for (size_t d = 0; d < n_dev; d++) {
        total += bytes[d];
    }
    if (from_fit) {
        GGML_LOG_INFO("%s: planned the host arenas of %zu device(s), %.1f MiB in total, the placement is committed for the load\n",
            __func__, part.size(), (double) total / 1024.0/1024.0);
    } else {
        GGML_LOG_INFO("%s: planned the host arenas of %zu device(s), %.1f MiB in total from the bytes the load registered for every device (no plan was committed by fit)\n",
            __func__, part.size(), (double) total / 1024.0/1024.0);
    }
    for (int device : part) {
        const int node = state.nodes[res.node[device]];
        char note[128];
        if (state.topo[device] < 0) {
            snprintf(note, sizeof(note), "planned, the platform reports no NUMA node for the PCI slot");
        } else if (state.topo[device] == node) {
            snprintf(note, sizeof(note), "numa_node of the GPU %d", device);
        } else {
            snprintf(note, sizeof(note), "planned, numa_node of the GPU %d is %d", device, state.topo[device]);
        }
        GGML_LOG_INFO("%s: device %d -> node %d (%.2f GiB, %s)\n", __func__, device, node,
            (double) bytes[device] / 1024.0/1024.0/1024.0, note);
    }
    if (state.plan_have_local) {
        GGML_LOG_INFO("%s: estimated step time %.1f ms (every device on its own fastest node %.1f ms, slot local placement %.1f ms)\n",
            __func__, state.plan_ms, state.plan_ms_fast, state.plan_ms_local);
    } else {
        GGML_LOG_INFO("%s: estimated step time %.1f ms (every device on its own fastest node %.1f ms)\n",
            __func__, state.plan_ms, state.plan_ms_fast);
    }

    return true;
}

// node of a new device-readable host allocation of `device`: the machine has one placement and
// the device keeps the node of that plan for every host arena of its own. GGML_CUDA_NUMA_NODE_ANY
// means the platform exposes no host NUMA node (allocate with no policy) and
// GGML_CUDA_NUMA_NODE_FAIL that there is no placement at all: planning needs the host bytes of
// every device, and a load that only knows the bytes of the device that allocates first must not
// place the machine from them.
// the caller holds the mutex
static int ggml_cuda_numa_plan_node_locked(int device, size_t size) {
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (state.nodes.empty()) {
        return GGML_CUDA_NUMA_NODE_ANY;
    }
    if (!state.plan.valid) {
        GGML_LOG_ERROR("%s: no placement of the machine exists, the %.2f MiB of host memory device %d reads cannot be placed: the placement needs the host bytes of every device, so fit has to commit one or the expert store has to register every device before its first arena\n",
            __func__, (double) size / 1024.0/1024.0, device);
        return GGML_CUDA_NUMA_NODE_FAIL;
    }
    return state.nodes[state.plan.node[state.phys_of[device]]];
}

// choose the node of a new mapped allocation; runs the one-time discovery of the nodes first
static int ggml_cuda_mapped_buffer_numa_node(int device, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    if (!ggml_cuda_mapped_numa.probed) {
        ggml_cuda_numa_discover(ggml_cuda_mapped_numa);
    }

    return ggml_cuda_numa_plan_node_locked(device, size);
}

// account a live device-readable host allocation of `device` on `node`
// the caller holds the mutex
static void ggml_cuda_numa_commit_locked(int device, int node, size_t size) {
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed) {
        return;
    }
    state.placed[device] += size;
    if (state.arena_node[device] < 0 && node >= 0) {
        state.arena_node[device] = node;
    }
    const int n = ggml_cuda_mapped_numa_index(node);
    if (n >= 0) {
        state.placed_node[n] += size;
        state.placed_dev_node[device][n] += size;
    }
}

// release the accounting of a freed allocation of `device` on `node`
// the caller holds the mutex
static void ggml_cuda_numa_unplace_locked(int device, int node, size_t size) {
    if (node == GGML_CUDA_NUMA_NODE_FAIL) {
        return; // a failed reservation was never accounted
    }
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed) {
        return;
    }
    if (state.placed[device] >= size) {
        state.placed[device] -= size;
    }
    const int n = ggml_cuda_mapped_numa_index(node);
    if (n >= 0) {
        if (state.placed_node[n] >= size) {
            state.placed_node[n] -= size;
        }
        if (state.placed_dev_node[device][n] >= size) {
            state.placed_dev_node[device][n] -= size;
        }
    }
}

// account a live mapped buffer of `device`
static void ggml_cuda_mapped_numa_commit(int device, int node, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);
    ggml_cuda_numa_commit_locked(device, node, size);
}

// release the accounting of a freed mapped buffer of `device`
static void ggml_cuda_mapped_numa_unplace(int device, int node, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);
    ggml_cuda_numa_unplace_locked(device, node, size);
}

// account the pages move_pages confirmed on the node of one device, for the final placement log
static void ggml_cuda_numa_verify_record(int device, size_t sampled, size_t on_node) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed || device < 0 || (size_t) device >= state.verify_pages.size()) {
        return;
    }
    state.verify_pages[device]   += sampled;
    state.verify_on_node[device] += on_node;
}

// allocate mapped host memory on the given NUMA node: cudaHostAlloc runs on a thread of the
// node under a memory policy bound to it (ggml_cuda_numa_host_alloc_on_node) and the placement
// is verified with move_pages. The pages of a pinned buffer are allocated and pinned inside the
// cudaHostAlloc call, under the policy of the calling thread, so the policy decides their node;
// there is no later fault that could place them elsewhere. An mbind(MPOL_MF_MOVE) of the range
// afterwards was dropped: pinned pages cannot be migrated, so it never moved anything, and it
// fails with EINVAL on a small buffer, because the driver carves small allocations out of a
// shared 2 MiB mapping with 512 byte alignment and the range then does not start on a page
// boundary (observed with the 1-expert host arenas of the tests). GGML_CUDA_NUMA_NODE_FAIL
// means that the plan did not fit; the allocation then fails (the caller has logged the
// details) instead of landing on an unknown node, which could be the device memory of another
// GPU
static void * ggml_backend_cuda_mapped_buffer_alloc_numa(int device, size_t size, int node) {
    if (node == GGML_CUDA_NUMA_NODE_FAIL) {
        return nullptr;
    }

    bool bound = false;
    void * ptr = ggml_cuda_numa_host_alloc_on_node(device, size, node, &bound);
    if (ptr == nullptr) {
        return nullptr;
    }

    if (bound) {
        size_t sampled = 0;
        size_t on_node = 0;
        ggml_cuda_mapped_buffer_verify_numa(ptr, size, node, &sampled, &on_node);
        ggml_cuda_numa_verify_record(device, sampled, on_node);
    }

    return ptr;
}

// the shared entry points of the host NUMA placement, used by the expert host arena
// (ggml-cuda-expert.cu) so that both halves of the host tier follow one probe and one
// allocation path

// register the host bytes of every device before the first arena of the machine exists, so the
// placement is planned from all of them and not from the device that happens to allocate first.
// The plan fit committed is kept and only checked against these bytes: planning again from the
// bytes a partial load knows would place the machine from partial information, which is the
// placement the single-card plan always degenerates to. Returns 1 when the machine has a
// placement afterwards, 0 when the bytes do not fit into the node capacities.
int ggml_cuda_numa_prepare_host_plan(const size_t * dev_bytes, size_t n_dev) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed) {
        ggml_cuda_numa_discover(state);
    }
    if (state.nodes.empty()) {
        return 1; // capability branch: the platform exposes no host NUMA node, no policy at all
    }

    if (!state.plan.valid) {
        return ggml_cuda_numa_plan_locked(std::vector<size_t>(dev_bytes, dev_bytes + n_dev), false) ? 1 : 0;
    }

    // a plan exists: the load has to use it. Its nodes are those fit reported, a device that
    // reads other bytes than the plan was made for is reported and keeps the planned node.
    const std::vector<size_t> bytes = ggml_cuda_numa_fold_locked(std::vector<size_t>(dev_bytes, dev_bytes + n_dev));
    for (size_t d = 0; d < bytes.size(); d++) {
        bool     match = false;
        const int node  = ggml_cuda_numa_plan::committed_node(state.plan, d, bytes[d], &match);
        if (node < 0 || match) {
            continue;
        }
        const size_t want = d < state.plan.bytes.size() ? state.plan.bytes[d] : 0;
        GGML_LOG_WARN("%s: device %d reads %.2f MiB, the %s placement was made for %.2f MiB; it keeps NUMA node %d\n",
            __func__, (int) d, (double) bytes[d] / 1024.0/1024.0,
            state.plan.from_fit ? "fit" : "load", (double) want / 1024.0/1024.0, state.nodes[node]);
    }

    return 1;
}

// reserve `size` bytes of device-readable host memory for `device`: the machine has one
// placement, the device takes the node of it, and the bytes count as placed while the
// allocation is live. Returns a node number, GGML_CUDA_NUMA_NODE_ANY when there is no node
// information (allocate with no policy) or GGML_CUDA_NUMA_NODE_FAIL when the machine has no
// placement. The caller releases the reservation with ggml_cuda_numa_release() when the
// allocation fails.
int ggml_cuda_numa_reserve(int device, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    if (!ggml_cuda_mapped_numa.probed) {
        ggml_cuda_numa_discover(ggml_cuda_mapped_numa);
    }

    const int node = ggml_cuda_numa_plan_node_locked(device, size);
    if (node != GGML_CUDA_NUMA_NODE_FAIL) {
        ggml_cuda_numa_commit_locked(device, node, size);
    }
    return node;
}

// release a reservation made by ggml_cuda_numa_reserve()
void ggml_cuda_numa_release(int device, int node, size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);
    ggml_cuda_numa_unplace_locked(device, node, size);
}

// the mapped weights tables are read by every device, so they do not belong to one device of the
// plan: they go on the node with the most free capacity and their bytes leave its capacity, so
// the plan of the expert arenas is made with the rest
int ggml_cuda_numa_fixed_alloc(size_t size) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed) {
        ggml_cuda_numa_discover(state);
    }
    if (state.nodes.empty()) {
        return GGML_CUDA_NUMA_NODE_ANY;
    }

    size_t best = 0;
    size_t best_free = 0;
    for (size_t n = 0; n < state.nodes.size(); n++) {
        const size_t cap  = state.cap[n] > 0 ? (size_t) state.cap[n] : 0;
        const size_t used = n < state.placed_node.size() ? state.placed_node[n] : 0;
        const size_t free_bytes = cap > used ? cap - used : 0;
        if (free_bytes > best_free) {
            best_free = free_bytes;
            best = n;
        }
    }

    if (best_free < size) {
        GGML_LOG_ERROR("%s: %.2f GiB of mapped host memory do not fit: node %d has %.2f GiB left of %.2f GiB\n",
                __func__, (double) size / 1024.0/1024.0/1024.0, state.nodes[best],
                (double) best_free / 1024.0/1024.0/1024.0,
                (double) state.cap[best] / 1024.0/1024.0/1024.0);
        return GGML_CUDA_NUMA_NODE_FAIL;
    }

    state.cap[best]         -= (int64_t) size;
    state.placed_node[best] += size;

    std::string capacity;
    for (size_t n = 0; n < state.nodes.size(); n++) {
        const size_t cap  = state.cap[n] > 0 ? (size_t) state.cap[n] : 0;
        const size_t used = n < state.placed_node.size() ? state.placed_node[n] : 0;
        const size_t free_bytes = cap > used ? cap - used : 0;
        capacity += (n == 0 ? "" : ", ") + std::to_string(state.nodes[n]) + ": " +
            std::to_string(free_bytes / (1024*1024)) + " MiB";
    }

    GGML_LOG_INFO("%s: mapped weights tables: %.2f GiB on NUMA node %d (left for the expert arenas: %s)\n",
            __func__, (double) size / 1024.0/1024.0/1024.0, state.nodes[best], capacity.c_str());
    return state.nodes[best];
}

// allocate pinned mapped host memory on the given node, with the same cudaHostAlloc path as
// the CUDA<n>_Mapped buffer type
void * ggml_cuda_numa_host_alloc(int device, size_t size, int node) {
    return ggml_backend_cuda_mapped_buffer_alloc_numa(device, size, node);
}

// startup log of the host placement: one line per device with the node of the plan and the
// bytes it reads there, then one line with the estimated step time and the two placements it
// is compared against
void ggml_cuda_numa_log_placement(const char * what) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    const ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed || state.nodes.empty() || !state.plan.valid) {
        return;
    }

    // the expert store calls this once per device buffer: print the plan of the machine once
    std::string assign;
    size_t total = 0;
    for (size_t d = 0; d < state.plan.node.size(); d++) {
        if (state.plan.bytes[d] == 0) {
            continue;
        }
        total += state.plan.bytes[d];
        assign += (assign.empty() ? "" : ",") + std::to_string(state.nodes[state.plan.node[d]]);
    }
    if (assign.empty()) {
        GGML_LOG_INFO("%s: %s: none (every expert is in a VRAM slot)\n", __func__, what);
        return;
    }

    static std::vector<size_t> logged;
    if (logged == state.plan.bytes) {
        return;
    }
    logged = state.plan.bytes;

    for (size_t d = 0; d < state.plan.node.size(); d++) {
        if (state.plan.bytes[d] == 0) {
            continue;
        }
        const int node = state.nodes[state.plan.node[d]];
        char note[128];
        if (state.topo[d] < 0) {
            snprintf(note, sizeof(note), "the platform reports no NUMA node for the PCI slot");
        } else if (state.topo[d] == node) {
            snprintf(note, sizeof(note), "numa_node of the GPU %d", (int) d);
        } else {
            snprintf(note, sizeof(note), "planned, numa_node of the GPU %d is %d", (int) d, state.topo[d]);
        }
        GGML_LOG_INFO("%s: device %d -> node %d (%.2f GiB, %s)\n", what, (int) d, node,
            (double) state.plan.bytes[d] / 1024.0/1024.0/1024.0, note);
    }

    if (state.plan_have_local) {
        GGML_LOG_INFO("%s: nodes %s, %.1f MiB in total, estimated step time %.1f ms (every device on its own fastest node %.1f ms, slot local placement %.1f ms)\n",
            what, assign.c_str(), (double) total / 1024.0/1024.0, state.plan_ms,
            state.plan_ms_fast, state.plan_ms_local);
    } else {
        GGML_LOG_INFO("%s: nodes %s, %.1f MiB in total, estimated step time %.1f ms (every device on its own fastest node %.1f ms)\n",
            what, assign.c_str(), (double) total / 1024.0/1024.0, state.plan_ms, state.plan_ms_fast);
    }
}

// final placement of the host arenas: printed once after every arena of the model is allocated.
// One line per device with the node its arenas got, the bytes placed there and the pages
// move_pages confirmed on it, then the comparison with the node of the plan device by device.
void ggml_cuda_numa_log_final_placement(void) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    const ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    if (!state.probed || state.nodes.empty() || !state.plan.valid) {
        return;
    }

    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    std::string assign;
    size_t total = 0;
    for (size_t d = 0; d < state.plan.node.size(); d++) {
        if (state.plan.bytes[d] == 0) {
            continue;
        }
        const int planned = state.nodes[state.plan.node[d]];
        const int node    = state.arena_node[d] >= 0 ? state.arena_node[d] : planned;
        const size_t bytes = d < state.placed.size() ? state.placed[d] : 0;
        const size_t pages = state.verify_pages[d];
        const size_t hits  = state.verify_on_node[d];
        total += bytes;
        assign += (assign.empty() ? "" : ",") + std::to_string(node);

        if (node == planned) {
            GGML_LOG_INFO("%s: device %d -> node %d, %.2f GiB placed, %zu/%zu sampled pages on the node (as planned)\n",
                __func__, (int) d, node, (double) bytes / 1024.0/1024.0/1024.0, hits, pages);
        } else {
            GGML_LOG_WARN("%s: device %d -> node %d, %.2f GiB placed, but the plan put it on node %d\n",
                __func__, (int) d, node, (double) bytes / 1024.0/1024.0/1024.0, planned);
        }
    }

    GGML_LOG_INFO("%s: final placement: nodes %s, %.1f MiB in total, estimated step time %.1f ms (every device on its own fastest node %.1f ms, placement %s)\n",
        __func__, assign.c_str(), (double) total / 1024.0/1024.0, state.plan_ms, state.plan_ms_fast,
        state.plan.from_fit ? "committed by fit" : "planned by the load");
}

// fit-side check of the expert host arenas over the host NUMA nodes: given the host complement
// allocations of all devices, plan the machine the same way the real load does (the plan is a
// function of the per device bytes, the node capacities and the measured read times), commit the
// plan for the load and report the bytes it puts on every node. On success the bytes placed on
// every node are written to node_bytes/node_ids (up to the capacity in *n_nodes, which receives
// the actual number of nodes). Returns 1 when the placement fits, -1 when the nodes can not hold
// the arenas (fail_bytes receives the bytes needed, fail_avail the capacity of all nodes
// together), and 0 when there is no NUMA information at all (the caller then falls back to a
// plain total check). The real load reads the plan this call committed, so a fit that passes
// does not fail later and does not place the machine from partial information.
static int ggml_backend_cuda_expert_check_host_nodes(
        const int * devices, const size_t * sizes, size_t n_alloc,
        int64_t * node_bytes, int * node_ids, size_t * n_nodes,
        int * fail_node, size_t * fail_bytes, int64_t * fail_avail) {
    std::lock_guard<std::mutex> lock(ggml_cuda_mapped_numa_mutex);

    if (!ggml_cuda_mapped_numa.probed) {
        ggml_cuda_numa_discover(ggml_cuda_mapped_numa);
    }
    ggml_cuda_mapped_numa_state & state = ggml_cuda_mapped_numa;
    const size_t n_nodes_all = state.nodes.size();
    if (n_nodes_all == 0) {
        if (n_nodes != nullptr) {
            *n_nodes = 0;
        }
        return 0;
    }

    const size_t n_devices = state.placed.size();
    std::vector<size_t> bytes(n_devices, 0);
    for (size_t i = 0; i < n_alloc; i++) {
        const int device = devices[i];
        GGML_ASSERT(device >= 0 && (size_t) device < n_devices);
        bytes[device] += sizes[i];
    }

    if (!ggml_cuda_numa_plan_locked(bytes, /*from_fit =*/ true)) {
        // the plan already logged the deficit per node; report the totals to the caller
        int64_t need_total = 0;
        int64_t cap_total  = 0;
        for (size_t d = 0; d < n_devices; d++) {
            need_total += (int64_t) bytes[d];
        }
        for (size_t n = 0; n < n_nodes_all; n++) {
            cap_total += state.cap[n];
        }
        if (fail_node  != nullptr) {
            *fail_node = -1;
        }
        if (fail_bytes != nullptr) {
            *fail_bytes = (size_t) need_total;
        }
        if (fail_avail != nullptr) {
            *fail_avail = cap_total;
        }
        return -1;
    }

    // bytes of the plan per node
    std::vector<int64_t> used(n_nodes_all, 0);
    for (size_t d = 0; d < state.plan.node.size(); d++) {
        if (state.plan.node[d] >= 0 && (size_t) state.plan.node[d] < n_nodes_all) {
            used[state.plan.node[d]] += (int64_t) state.plan.bytes[d];
        }
    }

    if (n_nodes != nullptr) {
        const size_t cap = *n_nodes;
        const size_t n_out = std::min(cap, n_nodes_all);
        for (size_t n = 0; n < n_out; n++) {
            if (node_bytes != nullptr) {
                node_bytes[n] = used[n];
            }
            if (node_ids != nullptr) {
                node_ids[n] = state.nodes[n];
            }
        }
        *n_nodes = n_out;
    }

    return 1;
}

#endif // !defined(GGML_USE_MUSA) && defined(GGML_CUDA_MAPPED_NUMA)

// stubs when the host NUMA planning is not compiled in (non-Linux, HIP, MUSA): the expert
// host arenas then fall back to plain cudaHostAlloc without a node policy
#if !defined(GGML_CUDA_MAPPED_NUMA)
int ggml_cuda_numa_prepare_host_plan(const size_t * dev_bytes, size_t n_dev) {
    GGML_UNUSED(dev_bytes);
    GGML_UNUSED(n_dev);
    return 1;
}

int ggml_cuda_numa_reserve(int device, size_t size) {
    GGML_UNUSED(device);
    GGML_UNUSED(size);
    return GGML_CUDA_NUMA_NODE_ANY;
}

void ggml_cuda_numa_release(int device, int node, size_t size) {
    GGML_UNUSED(device);
    GGML_UNUSED(node);
    GGML_UNUSED(size);
}

int ggml_cuda_numa_fixed_alloc(size_t size) {
    GGML_UNUSED(size);
    return GGML_CUDA_NUMA_NODE_ANY;
}

bool ggml_cuda_numa_bind_cpus(int node) {
    GGML_UNUSED(node);
    return false;
}

void ggml_cuda_numa_unbind_cpus(void) {
}

void * ggml_cuda_numa_host_alloc(int device, size_t size, int node) {
    GGML_UNUSED(device);
    if (node == GGML_CUDA_NUMA_NODE_FAIL) {
        return nullptr;
    }
#if defined(GGML_USE_MUSA)
    // MUSA has no host-mapped memory APIs; the expert store is not compiled there either
    GGML_UNUSED(size);
    return nullptr;
#else
    void * host_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: cudaHostAlloc of %.2f MiB failed: %s\n",
                       __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    return host_ptr;
#endif
}

void ggml_cuda_numa_log_placement(const char * what) {
    GGML_UNUSED(what);
}

void ggml_cuda_numa_log_final_placement(void) {
}

static int ggml_backend_cuda_expert_check_host_nodes(
        const int * devices, const size_t * sizes, size_t n_alloc,
        int64_t * node_bytes, int * node_ids, size_t * n_nodes,
        int * fail_node, size_t * fail_bytes, int64_t * fail_avail) {
    GGML_UNUSED(devices);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_alloc);
    GGML_UNUSED(node_bytes);
    GGML_UNUSED(node_ids);
    GGML_UNUSED(fail_node);
    GGML_UNUSED(fail_bytes);
    GGML_UNUSED(fail_avail);
    if (n_nodes != nullptr) {
        *n_nodes = 0;
    }
    return 0;
}
#endif // !defined(GGML_CUDA_MAPPED_NUMA)

// mapped host buffer type
//
// Pinned host memory that the device can read directly through its device pointer
// (cudaHostAllocMapped). The scheduler treats it as a device buffer type so that ops
// using it stay on the CUDA backend, and the kernels read the weights in place.
//
// MUSA lacks the host-mapped pinned-memory APIs, so the type is unavailable there.

#if !defined(GGML_USE_MUSA)

struct ggml_backend_cuda_mapped_buffer_context {
    int device;
    int numa_node = GGML_CUDA_NUMA_NODE_ANY; // node the allocation was placed on
    // set when the allocation went through the NUMA planner, which then also accounts it;
    // the mapped weights buffer is allocated without node planning
    bool numa_accounted = false;
    void * host_ptr = nullptr; // host address of the first byte of the buffer
    void * dev_ptr  = nullptr; // device address of the first byte of the buffer
    std::string name;

    ggml_backend_cuda_mapped_buffer_context(int device, int numa_node, void * host_ptr, void * dev_ptr, bool numa_accounted, const char * suffix) :
        device(device), numa_node(numa_node), numa_accounted(numa_accounted), host_ptr(host_ptr), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device) + suffix) {
    }

    ~ggml_backend_cuda_mapped_buffer_context() {
        if (host_ptr != nullptr) {
            CUDA_CHECK(cudaFreeHost(host_ptr));
        }
    }
};

static const char * ggml_backend_cuda_mapped_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static void ggml_backend_cuda_mapped_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *)buffer->context;
#if defined(GGML_CUDA_MAPPED_NUMA)
    if (ctx->numa_accounted) {
        ggml_cuda_mapped_numa_unplace(ctx->device, ctx->numa_node, buffer->size);
    }
#endif
    delete ctx;
}

static void * ggml_backend_cuda_mapped_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

// host pointer of a tensor in a mapped buffer; the offset inside the buffer is recovered
// from the device pointer
static char * ggml_backend_cuda_mapped_buffer_host_ptr(ggml_backend_cuda_mapped_buffer_context * ctx, const ggml_tensor * tensor) {
    return (char *) ctx->host_ptr + ((const char *) tensor->data - (const char *) ctx->dev_ptr);
}

static enum ggml_status ggml_backend_cuda_mapped_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    // quantized weights may be padded by get_alloc_size; zero the padding like the device buffer
    if (ggml_is_quantized(tensor->type) && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            memset(ggml_backend_cuda_mapped_buffer_host_ptr(ctx, tensor) + original_size, 0, padded_size - original_size);
        }
    }
    return GGML_STATUS_SUCCESS;
}

// the host side writes of a mapped buffer run on the CPUs of the node its pages are on: the
// device reads pages the other socket wrote last at 17-30 GB/s instead of 44 (see
// ggml_cuda_numa_bind_cpus); a buffer without a node (GGML_CUDA_NUMA_NODE_ANY) binds nothing
static void ggml_backend_cuda_mapped_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    const bool bound = ggml_cuda_numa_bind_cpus(ctx->numa_node);
    memset(ggml_backend_cuda_mapped_buffer_host_ptr(ctx, tensor) + offset, value, size);
    if (bound) {
        ggml_cuda_numa_unbind_cpus();
    }
}

static void ggml_backend_cuda_mapped_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    const bool bound = ggml_cuda_numa_bind_cpus(ctx->numa_node);
    memcpy(ggml_backend_cuda_mapped_buffer_host_ptr(ctx, tensor) + offset, data, size);
    if (bound) {
        ggml_cuda_numa_unbind_cpus();
    }
}

static void ggml_backend_cuda_mapped_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    memcpy(data, ggml_backend_cuda_mapped_buffer_host_ptr(ctx, tensor) + offset, size);
}

static bool ggml_backend_cuda_mapped_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync(ggml_backend_cuda_mapped_buffer_host_ptr(ctx, dst), src->data, ggml_nbytes(src),
                                   cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;
}

static void ggml_backend_cuda_mapped_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_mapped_buffer_context * ctx = (ggml_backend_cuda_mapped_buffer_context *) buffer->context;

    const bool bound = ggml_cuda_numa_bind_cpus(ctx->numa_node);
    memset(ctx->host_ptr, value, buffer->size);
    if (bound) {
        ggml_cuda_numa_unbind_cpus();
    }
}

static const ggml_backend_buffer_i ggml_backend_cuda_mapped_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_mapped_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_mapped_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_mapped_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_mapped_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_mapped_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_mapped_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cuda_mapped_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_mapped_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_cuda_mapped_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * host_ptr = nullptr;
    cudaError_t err = cudaSuccess;
    int node = GGML_CUDA_NUMA_NODE_ANY;
#if defined(GGML_CUDA_MAPPED_NUMA)
    // allocate on the NUMA node of the machine plan; the plan runs on the first allocation and
    // may leave another device current, hence the set_device below
    node = ggml_cuda_mapped_buffer_numa_node(buft_ctx->device, size);
    ggml_cuda_set_device(buft_ctx->device);
    host_ptr = ggml_backend_cuda_mapped_buffer_alloc_numa(buft_ctx->device, size, node);
#else
    err = cudaHostAlloc((void **) &host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB of mapped host memory for device %d: cudaHostAlloc failed: %s\n",
                       __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }
#endif
    if (host_ptr == nullptr) {
        return nullptr;
    }

    void * dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        CUDA_CHECK(cudaFreeHost(host_ptr));
        GGML_LOG_ERROR("%s: device pointer for mapped host memory on device %d: cudaHostGetDevicePointer failed: %s\n",
                       __func__, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_mapped_buffer_context * ctx = new ggml_backend_cuda_mapped_buffer_context(buft_ctx->device, node, host_ptr, dev_ptr, true, "_Mapped");
#if defined(GGML_CUDA_MAPPED_NUMA)
    ggml_cuda_mapped_numa_commit(buft_ctx->device, node, size);
    if (node >= 0) {
        GGML_LOG_INFO("%s: mapped host buffer for device %d: %.2f MiB on NUMA node %d\n",
                      __func__, buft_ctx->device, size / 1024.0 / 1024.0, node);
    } else {
        GGML_LOG_INFO("%s: mapped host buffer for device %d: %.2f MiB without a NUMA policy\n",
                      __func__, buft_ctx->device, size / 1024.0 / 1024.0);
    }
#endif

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_mapped_buffer_interface, ctx, size);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_mapped_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_mapped_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_mapped_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL, // not a host buffer: the scheduler keeps ops using it on the CUDA backend
};

// mapped weights buffer: a device-readable host buffer for the weights that GPU kernels read
// in place (the PLE table and the token embedding table). Unlike the mapped buffer type above
// its memory is not placed per device: every device reads these tables, so the buffer goes on
// the host node with the most free capacity and its bytes leave that node for the expert
// arenas. The loader copies the tensor data in and drops the source page cache. There is no row
// padding either, the GET_ROWS kernels read the file bytes as they are. Every device of the
// backend may read the buffer, so the scheduler does not have to keep the ops that read it on
// the device the buffer was allocated for.
static ggml_backend_buffer_t ggml_backend_cuda_mapped_weights_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    // the memory-fit pass allocates a zero-sized dummy buffer to learn the placement
    if (size == 0) {
        ggml_backend_cuda_mapped_buffer_context * ctx =
            new ggml_backend_cuda_mapped_buffer_context(buft_ctx->device, GGML_CUDA_NUMA_NODE_ANY, nullptr, nullptr, false, "_Mapped_Weights");
        return ggml_backend_buffer_init(buft, ggml_backend_cuda_mapped_buffer_interface, ctx, 0);
    }

    ggml_cuda_set_device(buft_ctx->device);

    // every device reads the table, so the pages go on the host node with the most free
    // capacity and leave the plan of the expert arenas the rest of it
    const int node = ggml_cuda_numa_fixed_alloc(size);

    // cudaHostAlloc mapped: the device reads the pages in place over UVA
    void * host_ptr = ggml_cuda_numa_host_alloc(buft_ctx->device, size, node);
    if (host_ptr == nullptr) {
        return nullptr;
    }

    void * dev_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaFreeHost(host_ptr));
        GGML_LOG_ERROR("%s: device pointer for mapped host memory on device %d: cudaHostGetDevicePointer failed: %s\n",
                       __func__, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    if (node >= 0) {
        GGML_LOG_INFO("%s: mapped weights host buffer for device %d: %.2f MiB on NUMA node %d\n",
                      __func__, buft_ctx->device, size / 1024.0 / 1024.0, node);
    } else {
        GGML_LOG_INFO("%s: mapped weights host buffer for device %d: %.2f MiB without a NUMA policy\n",
                      __func__, buft_ctx->device, size / 1024.0 / 1024.0);
    }

    // the bytes are accounted per node by ggml_cuda_numa_fixed_alloc(), not per device
    ggml_backend_cuda_mapped_buffer_context * ctx =
        new ggml_backend_cuda_mapped_buffer_context(buft_ctx->device, node, host_ptr, dev_ptr, false, "_Mapped_Weights");
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_mapped_buffer_interface, ctx, size);
}

// the file bytes are copied in as they are, so no row padding is added (the GET_ROWS kernels
// of these weights never read past a row)
static size_t ggml_backend_cuda_mapped_weights_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_mapped_weights_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_mapped_buffer_type_name, // recognized as a mapped buffer by the scheduler
    /* .alloc_buffer        = */ ggml_backend_cuda_mapped_weights_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_mapped_weights_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL, // not a host buffer: the scheduler keeps ops using it on the CUDA backend
};

ggml_backend_buffer_type_t ggml_backend_cuda_mapped_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_mapped_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_mapped_buffer_type_initialized = false;

    if (!ggml_backend_cuda_mapped_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_mapped_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_mapped_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i) + "_Mapped"},
            };
        }
        ggml_backend_cuda_mapped_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_mapped_buffer_types[device];
}

ggml_backend_buffer_type_t ggml_backend_cuda_mapped_weights_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_mapped_weights_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_type_context * ctx = new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i) + "_Mapped_Weights"};
            ctx->mapped_weights = true;
            ggml_backend_cuda_mapped_weights_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_mapped_weights_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ ctx,
            };
        }
        initialized = true;
    }

    return &ggml_backend_cuda_mapped_weights_buffer_types[device];
}

static bool ggml_backend_buft_is_cuda_mapped(ggml_backend_buffer_type_t buft);

// true for the mapped weights buffer type
static bool ggml_backend_buft_is_cuda_mapped_weights(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_cuda_mapped_buffer_type_name &&
        ((ggml_backend_cuda_buffer_type_context *) buft->context)->mapped_weights;
}

// the mapped weights buffer type of a CUDA device, or nullptr when the device is not CUDA
static ggml_backend_buffer_type_t ggml_backend_cuda_dev_mapped_weights_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg != ggml_backend_cuda_reg()) {
        return nullptr;
    }
    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < n_dev; ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            return ggml_backend_cuda_mapped_weights_buffer_type((int) i);
        }
    }
    return nullptr;
}

#else // defined(GGML_USE_MUSA)

// MUSA has no mapped buffer types: the scheduler never sees one
ggml_backend_buffer_type_t ggml_backend_cuda_mapped_buffer_type(int device) {
    GGML_UNUSED(device);
    return nullptr;
}

ggml_backend_buffer_type_t ggml_backend_cuda_mapped_weights_buffer_type(int device) {
    GGML_UNUSED(device);
    return nullptr;
}

static bool ggml_backend_buft_is_cuda_mapped_weights(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_cuda_dev_mapped_weights_buft(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return nullptr;
}

#endif // !defined(GGML_USE_MUSA)

static bool ggml_backend_buft_is_cuda_mapped(ggml_backend_buffer_type_t buft) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(buft);
    return false;
#else
    return buft->iface.get_name == ggml_backend_cuda_mapped_buffer_type_name;
#endif
}

// a tensor whose weights live in device-readable host memory and must be read in place
static bool ggml_cuda_tensor_is_mapped(const ggml_tensor * t) {
    return t != nullptr && t->buffer != nullptr && ggml_backend_buft_is_cuda_mapped(t->buffer->buft);
}

// expert buffer type: the expert store buffer, exposed as an extra buffer type so the fit
// code can place overflowing MoE weights on it. Unlike the mapped type it also owns VRAM
// slots for the hot experts, and its kernels read the weights through a pointer table.
static bool ggml_backend_buft_is_cuda_expert(ggml_backend_buffer_type_t buft) {
    return ggml_cuda_expert_buft_is_expert(buft);
}

// buffer types whose weights sit in device-readable host memory (the mapped type and the
// expert type): the fit code uses this to account for host memory and to drop page cache
static bool ggml_backend_buft_is_cuda_mapped_or_expert(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_cuda_mapped(buft) || ggml_backend_buft_is_cuda_expert(buft);
}

// defined in ggml-cuda-expert.cu. The declaration lives here because ggml-cuda-expert.cuh
// is included by the kernel files and must not depend on ggml-backend-impl.h (internal).
extern const ggml_backend_buffer_type_i ggml_backend_cuda_expert_buffer_type_interface;

// only used by the fit code below in this translation unit
static ggml_backend_buffer_type_t ggml_backend_cuda_expert_buffer_type(int device) {
    if (!ggml_cuda_expert_supported() || device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_expert_buffer_types[GGML_CUDA_MAX_DEVICES];
    static bool ggml_backend_cuda_expert_buffer_type_initialized = false;

    if (!ggml_backend_cuda_expert_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_expert_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_expert_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i) + "_Experts"},
            };
        }
        ggml_backend_cuda_expert_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_expert_buffer_types[device];
}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// True when ggml_cuda_mul_mat_cublas() computes in F16 for F16 weights, so that a shared
// input conversion to F16 is valid.
static bool ggml_cuda_mul_mat_cublas_compute_f16(const ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!fast_fp16_hardware_available(cc)) {
        return false;
    }
    return getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") == nullptr;
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    // the mul_mat_vec fusions read repacked-eligible weights in their original layout, so a
    // repacked weight must not enter them; the MUL_MAT_ID callers have no gated fallback, the
    // plain MUL_MAT pair falls back to the skinny gated kernel (see ggml_cuda_can_fuse)
    if (ggml_cuda_q8_skinny_is_repacked(ffn_up->src[0]) || ggml_cuda_q8_skinny_is_repacked(ffn_gate->src[0]) ||
            ggml_cuda_q4k_skinny_is_repacked(ffn_up->src[0]) || ggml_cuda_q4k_skinny_is_repacked(ffn_gate->src[0])) {
        return false;
    }

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    if (ggml_cuda_q8_skinny_is_repacked(src0) || ggml_cuda_q4k_skinny_is_repacked(src0)) {
        return false;
    }

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    if (ggml_cuda_q8_skinny_is_repacked(src0) || ggml_cuda_q4k_skinny_is_repacked(src0)) {
        return false;
    }

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat_q8_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
static void ggml_cuda_mul_mat_q4k_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    if (ggml_cuda_q8_skinny_is_repacked(src0)) {
        ggml_cuda_mul_mat_q8_skinny(ctx, src0, src1, dst);
        return;
    }

    if (ggml_cuda_q4k_skinny_is_repacked(src0)) {
        ggml_cuda_mul_mat_q4k_skinny(ctx, src0, src1, dst);
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A device-mapped weight must be read in place: cuBLAS would dequantize the whole tensor
    // into device memory. MMQ reads src0->data directly.
    if (ggml_cuda_tensor_is_mapped(src0) && ggml_cuda_mmq_supports_type(src0->type)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

static void ggml_cuda_mul_mat_q8_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_q8_skinny_mul_mat(ctx, src0, src1, dst)) {
        return;
    }

    // Large M or an unsupported input: expand the repacked weights to dense F16 and run the
    // regular path on them.
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    ggml_cuda_pool_alloc<half> src0_f16(ctx.pool(), ggml_nelements(src0));
    ggml_cuda_q8_skinny_to_f16(src0, src0_f16.get(), ctx.stream());

    ggml_tensor src0_tmp = *src0;
    src0_tmp.type = GGML_TYPE_F16;
    src0_tmp.data = src0_f16.get();
    src0_tmp.extra = nullptr;
    src0_tmp.nb[0] = sizeof(half);
    src0_tmp.nb[1] = src0_tmp.nb[0] * k;
    src0_tmp.nb[2] = src0_tmp.nb[1] * n;
    src0_tmp.nb[3] = src0_tmp.nb[2];

    ggml_cuda_mul_mat(ctx, &src0_tmp, src1, dst);
}

static void ggml_cuda_mul_mat_q4k_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (ggml_cuda_q4k_skinny_mul_mat(ctx, src0, src1, dst)) {
        return;
    }

    // M > 64 or an unsupported input: expand the repacked weights to dense F16 and run the
    // regular path on them. Prefill keeps using cuBLAS, as before the repack.
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    ggml_cuda_pool_alloc<half> src0_f16(ctx.pool(), ggml_nelements(src0));
    ggml_cuda_q4k_skinny_to_f16(src0, src0_f16.get(), ctx.stream());

    ggml_tensor src0_tmp = *src0;
    src0_tmp.type = GGML_TYPE_F16;
    src0_tmp.data = src0_f16.get();
    src0_tmp.extra = nullptr;
    src0_tmp.nb[0] = sizeof(half);
    src0_tmp.nb[1] = src0_tmp.nb[0] * k;
    src0_tmp.nb[2] = src0_tmp.nb[1] * n;
    src0_tmp.nb[3] = src0_tmp.nb[2];

    ggml_cuda_mul_mat(ctx, &src0_tmp, src1, dst);
}

// Column count to judge MMQ tiles against on Volta. Volta runs MMQ with dp4a and has no int8
// tensor cores, so a tile padded to J columns is real compute time. Judge by the average number
// of columns per expert instead of the total token count, so prefill (n >= 64) stays on the
// batched MMQ path instead of the per-expert fallback that needs stream synchronization.
static int64_t ggml_cuda_mul_mat_id_ncols_typ(const ggml_tensor * dst, const ggml_tensor * src1, const int64_t n_expert_used, const int cc) {
    const int64_t n_experts = ggml_cuda_mul_mat_id_n_expert_density(dst);
    if (cc != GGML_CUDA_CC_VOLTA || n_experts == 0) {
        return src1->ne[2];
    }
    return (src1->ne[2]*n_expert_used + n_experts - 1) / n_experts;
}

// The MUL_MAT_ID batch boundary, shared by the expert staging, the dispatch, needs_sync and
// supports_op: at most this many tokens per launch go to the MMVQ mmid kernel, larger batches go to
// the kernels above it. The boundary does not look at the epilogue, so a node with the SwiGLU
// epilogue switches at the same token count as one without it.
static bool ggml_cuda_mul_mat_id_use_mmvq(const ggml_tensor * dst, const int cc) {
    return dst->ne[2] <= get_mmvq_mmid_max_batch(dst->src[0]->type, cc);
}

// A MUL_MAT_ID node with the SwiGLU epilogue goes to the MMVQ mmid kernel while these hold: its
// kernel fuses the (gate, up) row pair and writes the F16 H, so the same conditions as the MMVQ
// branch of the dispatch decide it. An F16 activation is converted into an F32 pool buffer on the
// way there (see ggml_cuda_mul_mat_id), so it counts as an F32 one here.
static bool ggml_cuda_mul_mat_id_swiglu_mmvq(const ggml_tensor * dst, const int cc) {
    if (ggml_mul_mat_id_get_epilogue(dst) != GGML_MUL_MAT_ID_EPILOGUE_SWIGLU) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    return ggml_is_quantized(src0->type) &&
           (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) &&
           ggml_cuda_mul_mat_id_use_mmvq(dst, cc);
}

// Prefill batches (above the MMVQ batch) stage the routed host experts into VRAM
// (the grouped kernel and MMQ both read the staged copies). Decode batches (up to the MMVQ
// batch) take the MMVQ mmid branch of ggml_cuda_mul_mat_id, which stages them with
// ggml_cuda_expert_stage_prepare_decode instead.
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
// Pure function of the node and the device: it reads the expert table itself and judges the
// activation the way the dispatch does, so the three places that need to know whether a node
// stages (the graph capture check, ggml_cuda_mul_mat_id_needs_sync and the dispatch itself)
// cannot answer differently. A disagreement is what lets a graph that stages be recorded and
// replayed: the staging (ggml_cuda_expert_stage_prepare) has host side state, so a replay would
// re-run the captured routing instead of the routing of the launch, and the consumer would read
// staging slots this launch never filled.
//
// The F16 activation is judged after the conversion of ggml_cuda_mul_mat_id: unless the
// fused-expert kernel takes the node, that function puts the activation into an F32 pool buffer
// and the F32 paths below run on it. The node handed to the predicates is that converted one,
// because e.g. ggml_cuda_moe_grouped_mul_mat_id_supported rejects a non-F32 activation and would
// otherwise answer for the graph tensor while the dispatch answers for the pool buffer.
//
// A node that never reaches the dispatch block of ggml_cuda_mul_mat_id (a dst that is neither
// F32 nor the SwiGLU node, an activation that is neither F32 nor F16) is answered false here;
// ggml_cuda_mul_mat_id_needs_sync returns true for those before it asks.
static bool ggml_cuda_expert_mmid_stages(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * ids  = dst->src[2];

    ggml_cuda_expert_table ex_table;
    if (!ggml_cuda_expert_table_get(src0, &ex_table) || !ex_table.has_stage) {
        return false;
    }

    ggml_tensor dst_f32  = *dst;
    ggml_tensor src1_f32 = *dst->src[1];
    ggml_tensor * src1   = dst_f32.src[1];

    const bool to_fused = ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc) &&
        !ggml_cuda_mul_mat_id_swiglu_mmvq(dst, cc);
    if (src1->type == GGML_TYPE_F16 && !to_fused) {
        // the conversion writes packed rows, so the strides are the compact F32 ones
        src1_f32.type   = GGML_TYPE_F32;
        src1_f32.nb[0] = sizeof(float);
        src1_f32.nb[1] = src1_f32.nb[0]*src1_f32.ne[0];
        src1_f32.nb[2] = src1_f32.nb[1]*src1_f32.ne[1];
        src1_f32.nb[3] = src1_f32.nb[2]*src1_f32.ne[2];
        src1 = &src1_f32;
    }
    dst_f32.src[1] = src1;

    if (ggml_cuda_mul_mat_id_use_mmvq(&dst_f32, cc)) {
        // decode and verify: the MMVQ mmid branch stages in one kernel without a host side, so
        // it is not the prefill staging this function is about and stays graph recordable
        return false;
    }
    // prefill reads every routed expert row at least once per column group: copy the routed host
    // experts into VRAM once instead of reading them over the host link in place
    if (ggml_cuda_fused_expert_mul_mat_id_supported(&dst_f32, cc)) {
        return true;
    }
    if (ggml_cuda_moe_grouped_mul_mat_id_supported(&dst_f32, cc)) {
        return true;
    }
    return ggml_cuda_should_use_mmq(src0->type, cc,
        ggml_cuda_mul_mat_id_ncols_typ(&dst_f32, src1, ids->ne[0], cc), src0->ne[2]);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    // The SwiGLU epilogue writes an F16 dst, which the F32-only check below
    // rejects; it is a plain launch on either path like the F32 one, so it is decided by the same
    // two predicates further down and has no host-side part of its own.
    const bool mmid_swiglu = ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU;

    // F16 and F32 src1 are treated the same: an F16 activation is either read by the fused-expert
    // kernel (decided by the predicate below) or converted once into an F32 pool buffer before the
    // F32 paths run (see ggml_cuda_mul_mat_id), so neither case adds a host sync of its own.
    if ((src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
        (!mmid_swiglu && dst->type != GGML_TYPE_F32)) {
        return true;
    }

    // the routing trace passes a per-launch ring offset as a kernel argument, so the launch
    // cannot be recorded into a CUDA graph. This is a capability limit, not a fallback: with
    // the trace off the graph path below is unchanged.
    if (ggml_cuda_routing_trace_device_enabled()) {
        return true;
    }

    // expert store: the prefill staging has a host-side part (the stream state), so launches
    // that stage cannot be recorded into a CUDA graph. The decode staging (the MMVQ mmid
    // branch of ggml_cuda_mul_mat_id) is one kernel without a host-side part, so it stays
    // graph recordable and is not covered by this test. Same function the dispatch asks, so
    // the two cannot disagree about which nodes stage.
    if (ggml_cuda_expert_mmid_stages(dst, cc)) {
        return true;
    }

    if (mmid_swiglu) {
        // MMVQ covers the small batches with its own epilogue, the fused-expert kernel the rest.
        // Both are plain launches, so only shapes neither of them covers need a host sync.
        return !ggml_cuda_mul_mat_id_swiglu_mmvq(dst, cc) &&
               !ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc);
    }

    // the fused-expert and grouped kernels are plain launches that build their tables on the
    // device, so they can be recorded into a CUDA graph. Check them before MMVQ so the order
    // matches the dispatch.
    if (ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc)) {
        return false;
    }
    if (ggml_cuda_moe_grouped_mul_mat_id_supported(dst, cc)) {
        return false;
    }

    if (ggml_is_quantized(src0->type)) {
        if (ggml_cuda_mul_mat_id_use_mmvq(dst, cc)) {
            return false;
        }
    } else if (dst->ne[2] <= MMVF_MAX_BATCH_SIZE && GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, ggml_cuda_mul_mat_id_ncols_typ(dst, src1, ids->ne[0], cc), /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    // a device-mapped weight is read in place by the MMQ path (see ggml_cuda_mul_mat_id)
    if (ggml_cuda_tensor_is_mapped(src0) && ggml_cuda_mmq_supports_type(src0->type)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // The SwiGLU epilogue (ggml_mul_mat_id_get_epilogue(dst), op_params[7]) makes this node the
    // F16 H of silu(gate) * up built from the row interleaved weights, with one dst row per weight
    // row pair. Two kernels produce that layout: MMVQ (its MoE kernel fuses the pair, small batches,
    // the same boundary as a node without the epilogue) and the fused-expert kernel (larger batches).
    // With a type or shape neither of them covers there is no path here, so report the shape and
    // stop instead of falling through to a path that would write something else.
    // ggml_cuda_supports_op returns false for the same shapes, so the scheduler normally runs such a
    // node on the CPU.
    const bool mmid_swiglu = ggml_mul_mat_id_get_epilogue(dst) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU;
    if (mmid_swiglu && !ggml_cuda_mul_mat_id_swiglu_mmvq(dst, cc) &&
        !ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc)) {
        GGML_ABORT("%s: MUL_MAT_ID with the SwiGLU epilogue is covered by neither the MMVQ mmid "
                   "kernel nor the fused-expert kernel: src0 %s ne [%lld, %lld, %lld], src1 %s ne "
                   "[%lld, %lld], ids ne [%lld], dst %s ne [%lld, %lld, %lld], cc 0x%x",
                   __func__, ggml_type_name(src0->type),
                   (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                   ggml_type_name(src1->type), (long long) src1->ne[1], (long long) src1->ne[2],
                   (long long) ids->ne[0], ggml_type_name(dst->type),
                   (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], cc);
    }

    // F16 -> F32 activation view: the F16 H written by the epilogue above is read natively by the
    // fused-expert kernel only (its shared-memory X tile is F16). When this launch does not go
    // there - decode/verify batches, where the MMVQ mmid kernel or the grouped dp4a kernel would
    // run instead of its 64 row token tile, and a node with the epilogue goes to MMVQ while its
    // batch fits the mmid limit - convert the activation once per launch into an F32 pool buffer
    // and run the ordinary F32 paths on it, instead of there being no path at all. This is
    // the mirror of the F32 -> F16 X tile fused_expert builds (fused-expert.cu): one pass per
    // launch instead of one per (expert, n-tile) block, which would be ne01/BN passes over the
    // same rows. The conversion is exact, so the q8_1 quantization in MMVQ sees exactly the values
    // an F32 activation would (the F32 -> F16 -> F32 rounding already happened upstream, in the
    // epilogue, as it does on the CPU path).
    // [TAG_MUL_MAT_ID_CUDA_GRAPHS] The pool buffer is a plain device allocation from the same pool
    // as every other per-launch buffer on this path (the fused-expert ids and X16 above), with a
    // size that only depends on the node shape, so a captured launch replays the same addresses.
    ggml_cuda_pool_alloc<float> src1_f32_buf(ctx.pool());
    ggml_tensor src1_f32 = *src1;
    // the kernels that take dst (fused-expert, moe-grouped) read the activation through it, the
    // others through their parameter: point the node at the converted tensor for the launch and
    // put the graph's own pointer back on every exit
    struct src1_f32_scope {
        ggml_tensor * dst;
        ggml_tensor * src1_orig;
        src1_f32_scope(ggml_tensor * dst) : dst(dst), src1_orig(dst->src[1]) {}
        void set(ggml_tensor * src1_f32) { dst->src[1] = src1_f32; }
        ~src1_f32_scope() { dst->src[1] = src1_orig; }
    } src1_f32_scope(dst);

    // The F32 paths need an F32 activation; the fused-expert kernel reads F16 as it is. A node with
    // the SwiGLU epilogue goes to MMVQ for a small batch even when the fused predicate holds (the
    // dispatch below takes MMVQ first), so it is converted there as well.
    const bool mmid_to_fused = ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc) &&
        !ggml_cuda_mul_mat_id_swiglu_mmvq(dst, cc);
    if (src1->type == GGML_TYPE_F16 && !mmid_to_fused) {
        src1_f32_buf.alloc(ggml_nelements(src1));
        if (ggml_is_contiguous(src1)) {
            ggml_get_to_fp32_cuda(GGML_TYPE_F16)(src1->data, src1_f32_buf.get(), ggml_nelements(src1), ctx.stream());
        } else {
            // the strides of the non-contiguous kernel are in elements of the source type
            ggml_get_to_fp32_nc_cuda(GGML_TYPE_F16)(src1->data, src1_f32_buf.get(),
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[1] / (int64_t) sizeof(half), src1->nb[2] / (int64_t) sizeof(half),
                src1->nb[3] / (int64_t) sizeof(half), ctx.stream());
        }
        CUDA_CHECK(cudaGetLastError());
        // the rows are packed by the conversion, so the strides are the compact F32 ones
        src1_f32.type   = GGML_TYPE_F32;
        src1_f32.data   = src1_f32_buf.get();
        src1_f32.nb[0] = sizeof(float);
        src1_f32.nb[1] = src1_f32.nb[0]*src1_f32.ne[0];
        src1_f32.nb[2] = src1_f32.nb[1]*src1_f32.ne[1];
        src1_f32.nb[3] = src1_f32.nb[2]*src1_f32.ne[2];
        src1 = &src1_f32;
        src1_f32_scope.set(&src1_f32);
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    // expert store: the weights are read through a device pointer table, the per-layer
    // placement is slot or host arena. The byte check runs when the loader finished the
    // tensor data (see ggml_backend_cuda_expert_verify_all), not on the first dispatch.
    ggml_cuda_expert_table ex_table;
    const bool is_expert_store = ggml_cuda_expert_table_get(src0, &ex_table);
    if (is_expert_store) {
        ggml_cuda_expert_record_usage(src0, ids, ggml_mul_mat_id_get_expert_offset(dst), ctx.stream());
        ggml_cuda_expert_stats_tick(src0, ctx.stream());
    }

    // routing trace: record the raw ids of every MUL_MAT_ID, expert store or plain CUDA tensor
    ggml_cuda_routing_trace_record(src0, ids, ctx.stream());

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    // An F16 activation is read natively by the fused-expert kernel (its shared-memory X tile is
    // F16, so the kernel converts an F32 src1 itself, once per launch). Either that predicate
    // holds and this node goes there, or the activation was converted into the F32 pool buffer
    // above, so from here on src1 is an F32 tensor and the vector and MMQ paths are reached with
    // the same layout an F32 activation would give them. An F16 dst is the epilogue node itself:
    // MMVQ writes it below and the fused-expert kernel above (checked at the top of this function).
    if ((dst->type == GGML_TYPE_F32 || mmid_swiglu) &&
        (src1->type == GGML_TYPE_F32 || ggml_cuda_fused_expert_mul_mat_id_supported(dst, cc))) {
        if (src1->type == GGML_TYPE_F32 && ggml_is_quantized(src0->type) && ggml_cuda_mul_mat_id_use_mmvq(dst, cc)) {
            // The MMVQ mmid kernel is a separate kernel whose block width is the token count,
            // so on Volta it can go past MMVQ_MAX_BATCH_SIZE. Its MoE kernel also carries the
            // SwiGLU epilogue, so a node with one is written here, not by the fused-expert kernel.
            // A decode batch reads the weight of every expert it routes to once, so a host
            // expert it hits costs the whole UVA read inside the kernel: stage the routed host
            // experts into VRAM first and read them from there. The staging is one kernel with
            // no host side state, so the launch stays graph recordable.
            const char * const * expert_ptrs = nullptr;
            if (is_expert_store && ids != nullptr && ex_table.has_stage) {
                expert_ptrs = ggml_cuda_expert_stage_prepare_decode(src0, ids,
                    ggml_mul_mat_id_get_expert_offset(dst), ctx.stream());
            }
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, nullptr, expert_ptrs);
            return;
        } else if (!mmid_swiglu && src1->type == GGML_TYPE_F32 && ne2 <= MMVF_MAX_BATCH_SIZE && GGML_CUDA_CC_IS_AMD(cc)) {
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
            return;
        }

        // Prefill (T > 8) takes the W4A16 fused-expert kernel on Volta: it dequantizes the weight
        // tile into shared memory and multiplies it on the FP16 tensor cores (mma.m8n8k4), which
        // is the only tensor core path sm_70 has. Types and shapes it does not cover fall
        // through to the grouped dp4a kernel and then to MMQ below. This is a capability branch,
        // not a fallback.
        // prefill re-reads the routed expert rows (per column group or per tile): copy the routed
        // host experts into the staging area once and read them from VRAM (Strata prefill ring)
        const char * const * expert_ptrs = is_expert_store ? ex_table.main : nullptr;
        if (ggml_cuda_expert_mmid_stages(dst, cc)) {
            expert_ptrs = ggml_cuda_expert_stage_prepare(src0, ids, ggml_mul_mat_id_get_expert_offset(dst), ctx.stream());
        }

        if (ggml_cuda_fused_expert_mul_mat_id(ctx, dst, expert_ptrs)) {
            return;
        }

        // The SwiGLU epilogue has no other implementation. The predicate holds here (checked
        // at the top of this function), so fused_expert returned false only if the device
        // changed under us: stop instead of writing a wrong result into the F16 H.
        GGML_ASSERT(!mmid_swiglu);

        // Prefill (T > 8) also takes the grouped kernel: it stages the entries of one expert
        // and reads the weight row once per staged group, instead of the MMQ MUL_MAT_ID path
        // which buckets by (token, slot) and re-reads the weight of a hot expert once per tile.
        // This is a capability branch: the grouped kernel only covers the types and layouts it
        // implements (see ggml_cuda_moe_grouped_mul_mat_id_supported), the other types keep
        // their MMQ path below.
        if (ggml_cuda_moe_grouped_mul_mat_id(ctx, dst, expert_ptrs)) {
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ggml_cuda_mul_mat_id_ncols_typ(dst, src1, ids->ne[0], cc), /*n_experts=*/ne02)) {
            // A fully resident store tensor is a plain strided tensor: MMQ reads it from the store
            // base and pitch (see ggml_cuda_mul_mat_q), no per-block pointer table read. The
            // grouped and fused kernels above keep the table, they index it per expert themselves.
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst,
                    is_expert_store && ex_table.uniform ? nullptr : expert_ptrs);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }

        // A device-mapped weight must be read in place. The fallback path below computes each
        // expert with ggml_cuda_mul_mat(), which for quantized weights can dequantize the whole
        // expert into device memory via cuBLAS. MMQ reads src0->data directly and computes only
        // the experts that have tokens (via expert_bounds), so keep mapped weights on it.
        if (ggml_cuda_tensor_is_mapped(src0) && ggml_cuda_mmq_supports_type(src0->type)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    // expert parallelism appends one zero row that the skipped slots gather
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), (ne2*n_expert_used + 1)* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // expert parallelism: ids outside the local expert range belong to other devices; they are
    // not scheduled on this device and their output slots are overwritten with the zero row.
    // An expert store view reads its weights from the view start (main_host is shifted there),
    // so its ids stay view-local and expert_offset only carries the parallelism offset.
    const int32_t expert_offset = ggml_mul_mat_id_get_expert_offset(dst);

    // non-local slots gather a dedicated zero row appended after the compact rows. The row is
    // also the target of the assert below when the compact rows end at ne_get_rows exactly.
    const int32_t zero_row = (int32_t) ne_get_rows;
    std::fill(ids_from_sorted_host.begin(), ids_from_sorted_host.end(), zero_row);

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0);
                // an expert store view covers a sub range of the tensor, so its ids are view-local
                assert(ggml_mul_mat_id_is_expert_parallel(dst) || is_expert_store ||
                       expert_to_use < ne02);
                if (expert_to_use == i02 + expert_offset) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }

    const int64_t n_matched = (int64_t) ids_to_sorted_host.size();
    GGML_ASSERT(n_matched <= ne_get_rows);
    if (!ggml_mul_mat_id_is_expert_parallel(dst)) {
        GGML_ASSERT(n_matched == ne_get_rows);
    }

    // expert parallelism leaves the maps of the other devices empty, so the compact entries come
    // first and the full from-map for every (token, slot) starts right after them
    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), ids_to_sorted_host.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + n_matched;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        n_matched, 1, 1, sizeof(int32_t), n_matched*sizeof(int32_t), n_matched*sizeof(int32_t),
        ne10*ts_src1_sorted, n_matched*ne10*ts_src1_sorted, n_matched*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    // zero the row that the skipped slots gather
    if (n_matched < ne_get_rows) {
        CUDA_CHECK(cudaMemsetAsync((char *) dst_sorted.ptr + zero_row*ne0*ts_dst_sorted, 0, ne0*ts_dst_sorted, stream));
    }

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        // expert store: the per-expert base comes from the host shadow of the pointer table,
        // the device table cannot be dereferenced on the host
        GGML_ASSERT(!is_expert_store || ex_table.main_host != nullptr);
        // the per-expert base is a plain char * (device address from the host shadow of the
        // pointer table), the view only ever reads it
        src0_slice.data     = is_expert_store ? ex_table.main_host[i02]
                                              : (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    // scatter every dst entry: the matched ones read their compact row, the entries of the other
    // devices (expert parallelism) read the zero row
    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, n_matched*ne0*ts_dst_sorted, n_matched*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    // A repacked Q8_0/Q4_K tensor only carries weights for MUL_MAT src0; anything else reading
    // it would interpret the repacked bytes as the quantized type, so abort loudly instead of
    // computing garbage.
    if (dst->op != GGML_OP_MUL_MAT ||
            (dst->src[1] != nullptr && (ggml_cuda_q8_skinny_is_repacked(dst->src[1]) || ggml_cuda_q4k_skinny_is_repacked(dst->src[1])))) {
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const ggml_tensor * src = dst->src[i];
            if (src == nullptr) {
                continue;
            }
            if (ggml_cuda_q8_skinny_is_repacked(src) || ggml_cuda_q4k_skinny_is_repacked(src) ||
                (src->view_src != nullptr && (ggml_cuda_q8_skinny_is_repacked(src->view_src) || ggml_cuda_q4k_skinny_is_repacked(src->view_src)))) {
                GGML_ABORT("%s: op %s reads a repacked tensor %s", __func__, ggml_op_name(dst->op), src->name);
            }
        }
    }

    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_QSA_SELECT:
            ggml_cuda_op_qsa_select(ctx, dst);
            break;
        case GGML_OP_QSA_ATTN:
            ggml_cuda_op_qsa_attn(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    // drain the routing trace of this device and close the file while the ring is still alive
    ggml_cuda_routing_trace_shutdown_device(cuda_ctx->device);

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_memcpy_h2d_async((char *) tensor->data + offset, data, size, cuda_ctx->stream());
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_memcpy_2d_h2d_async((char *) tensor->data + offset, stride_tensor, (const char *) data, stride_data, size, n_copies, cuda_ctx->stream());
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    // the current device is per host thread; the copy and the copy_event run on the source stream
    ggml_cuda_set_device(cuda_ctx_src->device);

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// index of the first node after idx that is not a view or a no-op, or -1
static int ggml_cuda_next_real_node(const ggml_cgraph * cgraph, int idx) {
    for (int i = idx + 1; i < cgraph->n_nodes; ++i) {
        if (!ggml_cuda_is_view_or_noop(cgraph->nodes[i])) {
            return i;
        }
    }

    return -1;
}

// true if t is already computed when the node at node_idx runs, so a fused kernel placed there
// can read it. Views resolve to their source; leaves (weights and inputs) are always ready.
static bool ggml_cuda_tensor_ready_before(const ggml_cgraph * cgraph, const ggml_tensor * t, int node_idx) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    if (t->op == GGML_OP_NONE) {
        return true;
    }
    for (int i = 0; i < node_idx; ++i) {
        if (cgraph->nodes[i] == t) {
            return true;
        }
    }

    return false;
}

#ifdef USE_CUDA_GRAPH
// [TAG_MUL_MAT_ID_CUDA_GRAPHS] has_staging reports whether one of the nodes copies host experts
// into the VRAM staging area; it is part of the identity of a cached executable (see
// ggml_cuda_graph_get_key). The scan stops at the first node that rules out a capture, so the flag
// is only read for a graph that is capture compatible, where no node stages.
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph, bool * has_staging) {

    bool use_cuda_graph = true;
    *has_staging = false;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            // the staging predicate the dispatch uses, so a node that copies host experts into
            // VRAM is never judged capturable here and staged there
            if (ggml_cuda_expert_mmid_stages(node, cc)) {
                *has_staging = true;
            }
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

// What a cached executable must have been built from, besides the first node pointer that keys the
// map (ggml_cuda_graph::cuda_graph): the number of nodes and whether the graph stages expert
// weights into VRAM. The pointer alone is not an identity - the allocator hands the same address
// to the next graph and the next test case, so an entry can be found for a graph it was not built
// for. The two facts are recorded in the entry (ggml_cuda_graph::num_nodes, which nothing else
// uses) and compared before every reuse; a mismatch re-captures instead of replaying. They are
// recorded when the update is decided and not when the capture ends: an entry that kept the old
// value would report a mismatch on every call and never capture again.
static size_t ggml_cuda_graph_identity(const ggml_cgraph * cgraph, bool has_staging) {
    return ((size_t) cgraph->n_nodes << 1) | (has_staging ? 1 : 0);
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, size_t identity) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    // an executable built for a graph of another size or another staging state must not be
    // replayed, not even through the uid shortcut below
    if (graph->num_nodes != identity) {
        graph->num_nodes = identity;
        res = true;
    }

    // a capture made under another assist descriptor writes to that descriptor's buffers, which the
    // meta backend may already have freed (6-card two-slot servers rebuild them on every switch)
    const ggml_backend_meta_assist_rank * assist = cuda_ctx->assist;
    if ((assist != nullptr) != graph->has_assist ||
            (assist != nullptr && memcmp(assist, &graph->assist, sizeof(*assist)) != 0)) {
        graph->has_assist = assist != nullptr;
        graph->assist = assist != nullptr ? *assist : ggml_backend_meta_assist_rank{};
        res = true;
    }

    if (!res && cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

// match ADD -> RMS_NORM -> MUL, the residual add of a transformer block followed by norm and weight mul
// the ADD result is kept because later nodes use it as a residual
static bool ggml_cuda_should_fuse_add_rms_norm(const ggml_tensor * add,
                                               const ggml_tensor * rms_norm,
                                               const ggml_tensor * mul) {
    if (add->op != GGML_OP_ADD || rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return false;
    }

    if (add->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];

    // no broadcast, the kernel reads contiguous rows of a and b
    if (a == nullptr || b == nullptr || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(a, b) || !ggml_is_contiguous_rows(a) || !ggml_is_contiguous_rows(b)) {
        return false;
    }

    if (rms_norm->src[0] != add) {
        return false;
    }

    // the kernel writes both outputs with the layout of the add result
    if (!ggml_are_same_shape(add, mul) || !ggml_is_contiguous_rows(add) || !ggml_is_contiguous(mul)) {
        return false;
    }

    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    if ((mul->src[0] == rms_norm) == (mul->src[1] == rms_norm)) {
        return false;
    }

    return mul_src->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(mul_src);
}

// Memory check for the ADD -> RMS_NORM -> MUL fusion. The generic check rejects any overlap of an output
// with an input, but ggml-alloc usually computes the residual sum in place of the input that dies there.
// The kernel handles one row per block: the first pass reads a and b and writes the sum at the same
// element, the second pass (after the block reduction) reads the sum and writes the normalized row. So
// the sum may alias a or b exactly, and so may the normalized output (a and b are no longer read by
// then); any other overlap is rejected.
static bool ggml_cuda_add_rms_norm_ranges_ok(const ggml_tensor * add, const ggml_tensor * mul, const ggml_tensor * weight) {
    auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const uintptr_t x0 = (uintptr_t) x->data;
        const uintptr_t y0 = (uintptr_t) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    auto same = [](const ggml_tensor * x, const ggml_tensor * y) {
        return x->data == y->data && ggml_are_same_shape(x, y) && ggml_are_same_stride(x, y);
    };
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    for (const ggml_tensor * out : { add, mul }) {
        for (const ggml_tensor * in : { a, b }) {
            if (overlap(out, in) && !same(out, in)) {
                return false;
            }
        }
    }
    return !overlap(mul, add) && !overlap(weight, add) && !overlap(weight, mul);
}

// Memory check for a fused output: the kernel reads each element before writing it, so the output
// may alias an input exactly; any other overlap is rejected. Same convention as
// ggml_cuda_add_rms_norm_ranges_ok.
static bool ggml_cuda_fused_output_ranges_ok(const ggml_tensor * out, std::initializer_list<const ggml_tensor *> inputs) {
    for (const ggml_tensor * in : inputs) {
        const uintptr_t out0 = (uintptr_t) out->data;
        const uintptr_t in0  = (uintptr_t) in->data;
        if (out0 >= in0 + ggml_nbytes(in) || in0 >= out0 + ggml_nbytes(out)) {
            continue;
        }
        const bool same = out->data == in->data && ggml_are_same_shape(out, in) && ggml_are_same_stride(out, in);
        if (!same) {
            return false;
        }
    }

    return true;
}

struct ggml_cuda_rms_norm_gated_match {
    ggml_tensor * rms_norm = nullptr;
    ggml_tensor * mul      = nullptr;
    ggml_tensor * silu     = nullptr;
    ggml_tensor * mul2     = nullptr;
    int           last     = 0;
};

// match RMS_NORM -> MUL -> UNARY(SILU) -> MUL, the gated norm at the output of the GDN layers.
// Views may separate the nodes of the chain.
static bool ggml_cuda_match_rms_norm_mul_silu_mul(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_rms_norm_gated_match & match) {
    const int i0 = node_idx;
    const int i1 = ggml_cuda_next_real_node(cgraph, i0);
    const int i2 = i1 < 0 ? -1 : ggml_cuda_next_real_node(cgraph, i1);
    const int i3 = i2 < 0 ? -1 : ggml_cuda_next_real_node(cgraph, i2);
    if (i3 < 0) {
        return false;
    }

    const ggml_tensor * rms_norm = cgraph->nodes[i0];
    const ggml_tensor * mul      = cgraph->nodes[i1];
    const ggml_tensor * silu     = cgraph->nodes[i2];
    const ggml_tensor * mul2     = cgraph->nodes[i3];

    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || silu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || mul2->op != GGML_OP_MUL) {
        return false;
    }

    const ggml_tensor * x    = rms_norm->src[0];
    const ggml_tensor * gate = silu->src[0];
    if (x == nullptr || gate == nullptr) {
        return false;
    }

    // one operand of the first mul is the norm output, the other one is the broadcast weight
    if ((mul->src[0] == rms_norm) == (mul->src[1] == rms_norm)) {
        return false;
    }
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    // the kernel reads all operands element-wise
    for (const ggml_tensor * t : { x, rms_norm, mul, weight, gate, silu, mul2 }) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != ggml_type_size(GGML_TYPE_F32)) {
            return false;
        }
    }

    // one operand of the second mul is the silu output, the other one is the first mul output.
    // A view is accepted only when it does not change the element mapping.
    auto resolve_identity_view = [](const ggml_tensor * t) -> const ggml_tensor * {
        while (t->view_src != nullptr) {
            const ggml_tensor * parent = t->view_src;
            if (t->data != parent->data || !ggml_are_same_shape(t, parent) || !ggml_are_same_stride(t, parent)) {
                return nullptr;
            }
            t = parent;
        }
        return t;
    };
    const ggml_tensor * mul2_src0 = resolve_identity_view(mul2->src[0]);
    const ggml_tensor * mul2_src1 = resolve_identity_view(mul2->src[1]);
    if (!((mul2_src0 == silu && mul2_src1 == mul) || (mul2_src0 == mul && mul2_src1 == silu))) {
        return false;
    }

    // the kernel writes the second mul output with the layout of the norm input
    if (!ggml_are_same_shape(rms_norm, mul) || !ggml_are_same_shape(rms_norm, x) ||
        !ggml_are_same_shape(rms_norm, gate) || !ggml_are_same_shape(rms_norm, mul2) ||
        !ggml_is_contiguous(mul2) || !ggml_can_repeat(weight, mul2)) {
        return false;
    }

    // the kernel runs at the position of the rms norm, so the gate and the weight must be ready there
    if (!ggml_cuda_tensor_ready_before(cgraph, gate, i0) || !ggml_cuda_tensor_ready_before(cgraph, weight, i0)) {
        return false;
    }

    const int idxs[] = { i0, i1, i2, i3 };
    const enum ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL };
    const int out = i3;
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 4, ops, &out, 1)) {
        return false;
    }

    // the kernel reads x, gate and weight before overwriting each element
    if (!ggml_cuda_fused_output_ranges_ok(mul2, { x, gate, weight })) {
        return false;
    }

    match.rms_norm = cgraph->nodes[i0];
    match.mul      = cgraph->nodes[i1];
    match.silu     = cgraph->nodes[i2];
    match.mul2     = cgraph->nodes[i3];
    match.last     = i3;
    return true;
}

struct ggml_cuda_cont_sigmoid_mul_match {
    ggml_tensor * cont    = nullptr;
    ggml_tensor * sigmoid = nullptr;
    ggml_tensor * mul     = nullptr;
    int           last    = 0;
};

// match CONT -> UNARY(SIGMOID) -> MUL, the attention gate that reads a strided view.
// The CONT output is only consumed by the sigmoid, so the kernel reads the view directly.
static bool ggml_cuda_match_cont_sigmoid_mul(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_cont_sigmoid_mul_match & match) {
    const int i0 = node_idx;
    const int i1 = ggml_cuda_next_real_node(cgraph, i0);
    const int i2 = i1 < 0 ? -1 : ggml_cuda_next_real_node(cgraph, i1);
    if (i2 < 0) {
        return false;
    }

    const ggml_tensor * cont    = cgraph->nodes[i0];
    const ggml_tensor * sigmoid = cgraph->nodes[i1];
    const ggml_tensor * mul     = cgraph->nodes[i2];

    if (cont->op != GGML_OP_CONT || sigmoid->op != GGML_OP_UNARY ||
        ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID || mul->op != GGML_OP_MUL ||
        sigmoid->src[0] != cont) {
        return false;
    }

    const ggml_tensor * gate = cont->src[0];
    if (gate == nullptr || gate->view_src == nullptr ||
        gate->type != GGML_TYPE_F32 || gate->nb[0] != ggml_type_size(GGML_TYPE_F32)) {
        return false;
    }

    // one operand of the mul is the sigmoid output, the other one is the attention result
    if ((mul->src[0] == sigmoid) == (mul->src[1] == sigmoid)) {
        return false;
    }
    const ggml_tensor * attn = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];

    if (attn->type != GGML_TYPE_F32 || !ggml_is_contiguous(attn) || !ggml_are_same_shape(attn, mul) ||
        cont->type != GGML_TYPE_F32 || !ggml_is_contiguous(cont) || !ggml_are_same_shape(cont, mul) ||
        mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul)) {
        return false;
    }

    // the kernel reads the gate with its own 4D strides: gate dims 0 and 1 span one dst row,
    // gate dims 2 and 3 span the dst rows
    if (ggml_nelements(gate) != ggml_nelements(cont) ||
        cont->ne[0] != gate->ne[0]*gate->ne[1] ||
        cont->ne[1]*cont->ne[2]*cont->ne[3] != gate->ne[2]*gate->ne[3]) {
        return false;
    }

    // the kernel runs at the position of the cont, so the gate and the attention result must be ready there
    if (!ggml_cuda_tensor_ready_before(cgraph, gate, i0) || !ggml_cuda_tensor_ready_before(cgraph, attn, i0)) {
        return false;
    }

    const int idxs[] = { i0, i1, i2 };
    const enum ggml_op ops[] = { GGML_OP_CONT, GGML_OP_UNARY, GGML_OP_MUL };
    const int out = i2;
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 3, ops, &out, 1)) {
        return false;
    }

    if (!ggml_cuda_fused_output_ranges_ok(mul, { gate, attn })) {
        return false;
    }

    match.cont    = cgraph->nodes[i0];
    match.sigmoid = cgraph->nodes[i1];
    match.mul     = cgraph->nodes[i2];
    match.last    = i2;
    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// the prologue kernel reads these tensors, so they must be produced before the first fused node
// (otherwise their producers lie inside the range of skipped nodes)
static bool ggml_cuda_gdn_prologue_input_ready(const ggml_cgraph * cgraph, int node_idx, const ggml_tensor * t) {
    while (t != nullptr && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
        t = t->view_src;
    }
    if (t == nullptr || t->op == GGML_OP_NONE) {
        return true;
    }
    for (int j = 0; j < node_idx; ++j) {
        if (cgraph->nodes[j] == t) {
            return true;
        }
    }
    return false;
}

// match the q/k l2 norm (rms_norm + scale, see build_gdn_l2_norm) directly in front of a
// gated_delta_net, so that the kernel normalizes the raw q/k and skips 4 nodes. The raw q/k are
// views of the conv output, which the gdn also reads as v, so they outlive the gdn node.
// Only the scalar-gate (non-KDA) path is handled.
static bool ggml_cuda_match_gdn_prologue(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gdn_prologue & pro, int & gdn_idx) {
    static const ggml_op ops[5] = {
        GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_GATED_DELTA_NET,
    };

    const ggml_tensor * match[5];
    int                 match_idx[5];
    int                 count = 0;
    for (int j = node_idx; j < cgraph->n_nodes && count < 5; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != ops[count] || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        match[count]     = n;
        match_idx[count] = j;
        count++;
    }
    if (count < 5) {
        return false;
    }

    const ggml_tensor * rms_q   = match[0];
    const ggml_tensor * scale_q = match[1];
    const ggml_tensor * rms_k   = match[2];
    const ggml_tensor * scale_k = match[3];
    const ggml_tensor * gdn     = match[4];

    // the fused nodes feed each other and must have no other users
    for (int k = 0; k < 4; ++k) {
        if (match[k]->type != GGML_TYPE_F32 || (match[k]->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, match_idx[k]) != 1) {
            return false;
        }
    }

    // the rms norm is fed through a unit-bias scale and has no weight
    if (scale_q->src[0] != rms_q || scale_k->src[0] != rms_k ||
        rms_q->src[1] != nullptr || rms_k->src[1] != nullptr ||
        ggml_get_op_params_f32(scale_q, 1) != 0.0f || ggml_get_op_params_f32(scale_k, 1) != 0.0f) {
        return false;
    }

    if (gdn->type != GGML_TYPE_F32 || gdn->src[3]->ne[0] != 1) {
        return false; // a vector gate means KDA, which is not fused
    }

    const int64_t S_v = gdn->src[2]->ne[0];

    const ggml_tensor * q_raw = rms_q->src[0];
    const ggml_tensor * k_raw = rms_k->src[0];
    if (q_raw == nullptr || k_raw == nullptr ||
        q_raw->type != GGML_TYPE_F32 || k_raw->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous_rows(q_raw) || !ggml_is_contiguous_rows(k_raw) ||
        q_raw->ne[0] != S_v || k_raw->ne[0] != S_v ||
        !ggml_are_same_shape(q_raw, gdn->src[0]) || !ggml_are_same_shape(k_raw, gdn->src[1])) {
        return false;
    }

    // q and k reach the gdn directly or through single-use views
    auto trace_view = [&](const ggml_tensor * t) -> const ggml_tensor * {
        while (t != nullptr && (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
            int t_idx = -1;
            for (int j = node_idx; j < match_idx[4]; ++j) {
                if (cgraph->nodes[j] == t) {
                    t_idx = j;
                    break;
                }
            }
            if (t_idx < 0 || (t->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                ggml_node_get_use_count(cgraph, t_idx) != 1) {
                return nullptr;
            }
            t = t->view_src;
        }
        return t;
    };
    if (trace_view(gdn->src[0]) != scale_q || trace_view(gdn->src[1]) != scale_k) {
        return false;
    }

    for (const ggml_tensor * t : { q_raw, k_raw }) {
        if (!ggml_cuda_gdn_prologue_input_ready(cgraph, node_idx, t)) {
            return false;
        }
    }

    pro.q_raw   = (const float *) q_raw->data;
    pro.k_raw   = (const float *) k_raw->data;
    pro.sq1     = q_raw->nb[1] / sizeof(float);
    pro.sq2     = q_raw->nb[2] / sizeof(float);
    pro.sq3     = q_raw->nb[3] / sizeof(float);
    pro.sk1     = k_raw->nb[1] / sizeof(float);
    pro.sk2     = k_raw->nb[2] / sizeof(float);
    pro.sk3     = k_raw->nb[3] / sizeof(float);
    pro.eps_q   = ggml_get_op_params_f32(rms_q, 0);
    pro.scale_q = ggml_get_op_params_f32(scale_q, 0);
    pro.eps_k   = ggml_get_op_params_f32(rms_k, 0);
    pro.scale_k = ggml_get_op_params_f32(scale_k, 0);

    gdn_idx = match_idx[4];
    return true;
}

struct ggml_cuda_gdn_gating_match {
    const ggml_tensor * beta_in  = nullptr;
    ggml_tensor *       beta_out = nullptr;
    const ggml_tensor * alpha_in = nullptr;
    const ggml_tensor * dt       = nullptr;
    const ggml_tensor * a        = nullptr;
    ggml_tensor *       gate_out = nullptr;
    int                 last     = 0;
};

// a per-head bias or weight: 1D f32 [n_heads]
static bool ggml_cuda_gdn_gating_vector(const ggml_tensor * t, int64_t n_heads) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
           t->ne[0] == n_heads && ggml_nelements(t) == n_heads;
}

// match the GDN gating: UNARY(SIGMOID) on beta next to ADD -> UNARY(SOFTPLUS) -> MUL on alpha.
// The chains are independent; either one may come first and views may separate the nodes.
static bool ggml_cuda_match_gdn_gating(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gdn_gating_match & match) {
    int idxs[4];
    int count = 0;
    for (int j = node_idx; j < cgraph->n_nodes && count < 4; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        idxs[count++] = j;
    }
    if (count < 4) {
        return false;
    }

    const ggml_tensor * n0 = cgraph->nodes[idxs[0]];
    const ggml_tensor * n1 = cgraph->nodes[idxs[1]];
    const ggml_tensor * n2 = cgraph->nodes[idxs[2]];
    const ggml_tensor * n3 = cgraph->nodes[idxs[3]];

    const auto is_sigmoid  = [](const ggml_tensor * t) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID;
    };
    const auto is_softplus = [](const ggml_tensor * t) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SOFTPLUS;
    };

    const ggml_tensor * sigmoid  = nullptr;
    const ggml_tensor * add      = nullptr;
    const ggml_tensor * softplus = nullptr;
    const ggml_tensor * mul      = nullptr;

    if (is_sigmoid(n0) && n1->op == GGML_OP_ADD && is_softplus(n2) && n3->op == GGML_OP_MUL) {
        sigmoid  = n0;
        add      = n1;
        softplus = n2;
        mul      = n3;
    } else if (n0->op == GGML_OP_ADD && is_softplus(n1) && n2->op == GGML_OP_MUL && is_sigmoid(n3)) {
        add      = n0;
        softplus = n1;
        mul      = n2;
        sigmoid  = n3;
    } else {
        return false;
    }

    if (softplus->src[0] != add) {
        return false;
    }

    // one mul operand is the softplus output, the other one is the per-head ssm_a weight
    if ((mul->src[0] == softplus) == (mul->src[1] == softplus)) {
        return false;
    }
    const ggml_tensor * a = mul->src[0] == softplus ? mul->src[1] : mul->src[0];

    // the add has the per-head ssm_dt bias and the alpha operand
    const int64_t n_heads = add->ne[0];
    const ggml_tensor * dt    = nullptr;
    const ggml_tensor * alpha = nullptr;
    if (add->src[0] == nullptr || add->src[1] == nullptr) {
        return false;
    }
    if (ggml_cuda_gdn_gating_vector(add->src[1], n_heads)) {
        dt    = add->src[1];
        alpha = add->src[0];
    } else if (ggml_cuda_gdn_gating_vector(add->src[0], n_heads)) {
        dt    = add->src[0];
        alpha = add->src[1];
    } else {
        return false;
    }

    const ggml_tensor * beta_in = sigmoid->src[0];
    if (beta_in == nullptr || alpha == nullptr || a == nullptr) {
        return false;
    }

    // every tensor of the chain is f32 and contiguous
    for (const ggml_tensor * t : { beta_in, sigmoid, alpha, add, softplus, mul, dt, a }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }

    if (!ggml_cuda_gdn_gating_vector(a, n_heads) || !ggml_are_same_shape(add, softplus) ||
        !ggml_are_same_shape(softplus, mul)) {
        return false;
    }

    // the fused kernel runs at the position of the first node of the chain, so its inputs must
    // be computed before it
    if (!ggml_cuda_tensor_ready_before(cgraph, beta_in, node_idx) ||
        !ggml_cuda_tensor_ready_before(cgraph, alpha,   node_idx) ||
        !ggml_cuda_tensor_ready_before(cgraph, dt,      node_idx) ||
        !ggml_cuda_tensor_ready_before(cgraph, a,       node_idx)) {
        return false;
    }

    // alpha_biased and alpha_softplus have no users outside the chain; beta_out and gate_out are kept
    const enum ggml_op ops[] = { n0->op, n1->op, n2->op, n3->op };
    const int out_nodes[] = { sigmoid == n0 ? idxs[0] : idxs[3], mul == n2 ? idxs[2] : idxs[3] };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 4, ops, out_nodes, 2)) {
        return false;
    }

    // the kernel reads each element before writing it. With equal element counts both chains use
    // the same thread index, so either output may exactly alias either input; with different
    // counts the chains run in separate halves of the launch and only the matching input may
    // alias. Outputs never overlap each other, dt or a.
    auto overlap = [](const ggml_tensor * x, const ggml_tensor * y) {
        const uintptr_t x0 = (uintptr_t) x->data;
        const uintptr_t y0 = (uintptr_t) y->data;
        return x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x);
    };
    const auto same_range = [](const ggml_tensor * x, const ggml_tensor * y) {
        // all four tensors are contiguous, so an exact alias shares the pointer and byte range
        return x->data == y->data && ggml_nbytes(x) == ggml_nbytes(y);
    };
    bool ranges_ok;
    if (ggml_nelements(beta_in) == ggml_nelements(alpha)) {
        ranges_ok = (!overlap(sigmoid, beta_in) || same_range(sigmoid, beta_in)) &&
                    (!overlap(sigmoid, alpha)   || same_range(sigmoid, alpha))   &&
                    (!overlap(mul, beta_in)     || same_range(mul, beta_in))     &&
                    (!overlap(mul, alpha)       || same_range(mul, alpha));
    } else {
        ranges_ok = ggml_cuda_fused_output_ranges_ok(sigmoid, { beta_in }) &&
                    ggml_cuda_fused_output_ranges_ok(mul,     { alpha });
    }
    if (!ranges_ok ||
        overlap(sigmoid, mul) || overlap(sigmoid, dt) || overlap(sigmoid, a) ||
        overlap(mul, dt) || overlap(mul, a)) {
        return false;
    }

    // the graph nodes themselves are mutable, the matcher only looks at them through const pointers
    match.beta_in  = beta_in;
    match.beta_out = const_cast<ggml_tensor *>(sigmoid);
    match.alpha_in = alpha;
    match.dt       = dt;
    match.a        = a;
    match.gate_out = const_cast<ggml_tensor *>(mul);
    match.last     = idxs[3];
    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }

        // a repacked Q8_0/Q4_K SwiGLU pair is fused by the skinny gated kernel in the caller
        if (ops.begin()[0] == GGML_OP_MUL_MAT &&
                ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU && !ggml_get_op_params_i32(glu, 1) &&
                ((ggml_cuda_q4k_skinny_is_repacked(ffn_gate->src[0]) && ggml_cuda_q4k_skinny_is_repacked(ffn_up->src[0])) ||
                 (ggml_cuda_q8_skinny_is_repacked(ffn_gate->src[0]) && ggml_cuda_q8_skinny_is_repacked(ffn_up->src[0])))) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// conservative overlap test: ggml_nbytes spans all bytes the tensor may touch
static bool ggml_cuda_cpy_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data;
    const uintptr_t b0 = (uintptr_t) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// batch consecutive same-layout CPY nodes into a single kernel launch, e.g. the per-position
// writes of the recurrent state snapshots. Scans forward from a CPY, skipping view/noop nodes;
// any other op stops the batch. Returns the number of nodes to skip, 0 if nothing was fused.
static int ggml_cuda_try_cpy_batch(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * first = cgraph->nodes[i];
    if (first->op != GGML_OP_CPY || (first->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        first->src[0]->data == nullptr || first->src[1]->data == nullptr) {
        return 0;
    }

    const ggml_tensor * srcs[CUDA_CPY_BATCH_MAX];
    ggml_tensor *       dsts[CUDA_CPY_BATCH_MAX];

    srcs[0] = first->src[0];
    dsts[0] = first->src[1];
    int n    = 1;
    int last = i;

    for (int j = i + 1; j < cgraph->n_nodes && n < CUDA_CPY_BATCH_MAX; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }
        if (node->op != GGML_OP_CPY || (node->flags & (GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_COMPUTE)) != GGML_TENSOR_FLAG_COMPUTE) {
            break;
        }

        const ggml_tensor * src = node->src[0];
        ggml_tensor *       dst = node->src[1];

        // all copies must share the type combination, element count and layout of the first one
        if (src->type != srcs[0]->type || dst->type != dsts[0]->type ||
            ggml_nelements(src) != ggml_nelements(srcs[0]) || ggml_nelements(dst) != ggml_nelements(dsts[0]) ||
            src->data == nullptr || dst->data == nullptr) {
            break;
        }
        bool same_layout = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (src->ne[d] != srcs[0]->ne[d] || src->nb[d] != srcs[0]->nb[d] ||
                dst->ne[d] != dsts[0]->ne[d] || dst->nb[d] != dsts[0]->nb[d]) {
                same_layout = false;
                break;
            }
        }
        if (!same_layout) {
            break;
        }

        // the copies run concurrently, so a new dst must not overlap any src or dst of the batch
        bool conflict = false;
        for (int k = 0; k < n && !conflict; ++k) {
            conflict = ggml_cuda_cpy_ranges_overlap(dst, dsts[k]) ||
                       ggml_cuda_cpy_ranges_overlap(dst, srcs[k]) ||
                       ggml_cuda_cpy_ranges_overlap(src, dsts[k]);
        }
        if (conflict) {
            break;
        }

        srcs[n] = src;
        dsts[n] = dst;
        ++n;
        last = j;
    }

    if (n < 2 || !ggml_cuda_cpy_batch(*cuda_ctx, srcs, dsts, n)) {
        return 0;
    }

    return last - i;
}

// resolve views to the tensor that produced their data
static const ggml_tensor * ggml_cuda_view_base(const ggml_tensor * t) {
    while (t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

// count consumers of `t` among compute nodes at index `from` or later; reads through views count
// as a use of the view source
static int ggml_cuda_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t, int from) {
    int uses = 0;
    for (int j = from; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (node->src[k] != nullptr && ggml_cuda_view_base(node->src[k]) == t) {
                ++uses;
            }
        }
    }
    return uses;
}

// match GET_ROWS -> CONCAT -> CPY x K -> SSM_CONV -> SILU, the conv-state rollback window built by
// build_conv_state() for gated delta net layers, and run it as one fused kernel. The kernel reads
// the state row and the qkv projection directly, so the gather, the concat and all snapshots can
// be skipped. Returns the number of nodes to skip, 0 if the pattern does not match.
static int ggml_cuda_try_ssm_conv_rollback_fusion(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * get_rows = cgraph->nodes[i];
    if (get_rows->op != GGML_OP_GET_ROWS) {
        return 0;
    }

    const ggml_tensor * state_rows = get_rows->src[0];
    const ggml_tensor * row_idx    = get_rows->src[1];

    if (get_rows->type != GGML_TYPE_F32 || (get_rows->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        state_rows == nullptr || state_rows->type != GGML_TYPE_F32 || state_rows->data == nullptr ||
        state_rows->nb[0] != sizeof(float) || state_rows->nb[1] % sizeof(float) != 0 ||
        row_idx == nullptr || row_idx->type != GGML_TYPE_I32 || row_idx->data == nullptr ||
        ggml_nelements(row_idx) != 1) {
        return 0;
    }

    int pos = i + 1;

    // index of the next node that is not a view, empty or a no-op
    auto next_node = [&]() -> const ggml_tensor * {
        while (pos < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[pos])) {
            ++pos;
        }
        return pos < cgraph->n_nodes ? cgraph->nodes[pos] : nullptr;
    };

    const ggml_tensor * concat = next_node();
    if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->type != GGML_TYPE_F32 ||
        (concat->flags & GGML_TENSOR_FLAG_OUTPUT) || concat->data == nullptr ||
        concat->src[0] == nullptr || ggml_get_op_params_i32(concat, 0) != 0 ||
        ggml_cuda_view_base(concat->src[0]) != get_rows) {
        return 0;
    }
    ++pos;

    const int64_t C     = concat->ne[1];
    const int64_t n_tok = concat->ne[0] - 3;

    // the state part of the concat is the gathered row, reshaped to [3, C, 1]
    const ggml_tensor * state_window = concat->src[0];
    if (state_window->ne[0] != 3 || state_window->ne[1] != C || state_window->ne[2] != 1 ||
        state_window->ne[3] != 1 || !ggml_is_contiguous(state_window) ||
        state_window->data != get_rows->data) {
        return 0;
    }

    // src1 is the transposed qkv projection; n_tok is kept small so the fused kernel does not
    // replace the long-token kernel for large batches
    const ggml_tensor * qkv = concat->src[1];
    if (qkv == nullptr || qkv->op != GGML_OP_TRANSPOSE || qkv->type != GGML_TYPE_F32 ||
        qkv->data == nullptr || qkv->ne[0] != n_tok || qkv->ne[1] != C || qkv->ne[2] != 1 ||
        qkv->ne[3] != 1 || n_tok < 1 || n_tok > 32 ||
        qkv->nb[0] != C * sizeof(float) || qkv->nb[1] != sizeof(float)) {
        return 0;
    }

    const ggml_tensor * state_base = ggml_cuda_view_base(state_rows);
    if (state_rows->ne[0] != 3 * C || state_rows->nb[1] < 3 * sizeof(float) ||
        state_rows->nb[1] % (3 * sizeof(float)) != 0 || ggml_cuda_view_base(qkv) == state_base ||
        ((const char *) state_rows->data - (const char *) state_base->data) % state_rows->nb[1] != 0) {
        return 0;
    }

    const float * concat_d = (const float *) concat->data;

    ggml_cuda_ssm_conv_rollback_params params = {};
    params.state        = (const float *) state_rows->data;
    params.state_stride = state_rows->nb[1] / sizeof(float);
    params.row          = (const int32_t *) row_idx->data;
    params.qkv          = (const float *) qkv->data;
    params.qkv_nb0      = qkv->nb[0] / sizeof(float);
    params.qkv_nb1      = qkv->nb[1] / sizeof(float);
    params.n_tok        = n_tok;

    // the snapshot copies between the concat and the conv
    const ggml_tensor * snap_dsts[16];
    while (params.n_snap < (int) (sizeof(params.snap_off) / sizeof(params.snap_off[0]))) {
        const ggml_tensor * cpy = next_node();
        if (cpy == nullptr || cpy->op != GGML_OP_CPY) {
            break;
        }

        const ggml_tensor * src = cpy->src[0];
        const ggml_tensor * dst = cpy->src[1];

        if (cpy->type != GGML_TYPE_F32 || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            src == nullptr || dst == nullptr || src->view_src == nullptr || dst->view_src == nullptr ||
            ggml_cuda_view_base(src) != concat || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src->ne[0] != 3 || src->ne[1] != C || src->ne[2] != 1 || src->ne[3] != 1 ||
            src->nb[0] != sizeof(float) || src->nb[1] != concat->nb[1] ||
            ggml_nelements(dst) != 3 * C || !ggml_is_contiguous(dst) ||
            src->data == nullptr || dst->data == nullptr ||
            ggml_cuda_view_base(dst) != state_base ||
            ((const char *) dst->data - (const char *) state_base->data) % state_rows->nb[1] != 0) {
            return 0;
        }

        // offset of the window inside the concat output
        const int64_t off = (const float *) src->data - concat_d;
        if (off < 0 || off > n_tok) {
            return 0;
        }

        params.snap_off[params.n_snap] = off;
        params.snap_dst[params.n_snap] = (float *) dst->data;
        snap_dsts[params.n_snap]       = dst;
        params.n_snap++;
        ++pos;
    }

    if (params.n_snap == 0) {
        return 0;
    }

    // the kernel writes all snapshots concurrently, overlapping ranges would be unordered
    for (int k = 0; k < params.n_snap; ++k) {
        for (int l = k + 1; l < params.n_snap; ++l) {
            if (ggml_cuda_cpy_ranges_overlap(snap_dsts[k], snap_dsts[l])) {
                return 0;
            }
        }
    }

    const ggml_tensor * conv = next_node();
    if (conv == nullptr || conv->op != GGML_OP_SSM_CONV || (conv->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        conv->src[0] == nullptr || ggml_cuda_view_base(conv->src[0]) != concat || conv->src[1] == nullptr ||
        conv->src[1]->type != GGML_TYPE_F32 || conv->src[1]->data == nullptr ||
        conv->src[1]->ne[0] != 4 || conv->src[1]->ne[1] != C || conv->src[1]->nb[0] != sizeof(float) ||
        conv->src[1]->nb[1] % sizeof(float) != 0) {
        return 0;
    }
    ++pos;

    const ggml_tensor * silu = next_node();
    if (silu == nullptr || silu->op != GGML_OP_UNARY || ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU ||
        silu->type != GGML_TYPE_F32 || silu->src[0] != conv || silu->data == nullptr ||
        silu->ne[0] != C || silu->ne[1] != n_tok || silu->ne[2] != 1 ||
        silu->nb[0] != sizeof(float) || silu->nb[1] % sizeof(float) != 0 || !ggml_is_contiguous(silu)) {
        return 0;
    }

    // the skipped outputs must not be consumed anywhere else
    if (ggml_cuda_use_count(cgraph, get_rows, i + 1) != 1 ||
        ggml_cuda_use_count(cgraph, concat,   i + 1) != params.n_snap + 1 ||
        ggml_cuda_use_count(cgraph, conv,     i + 1) != 1) {
        return 0;
    }

    // the kernel writes the silu output at the position of the get_rows node, so it must not
    // partially overlap the qkv projection it reads; a full overlap is safe because the kernel
    // reads each value before writing it
    if (silu->data != qkv->data && ggml_cuda_cpy_ranges_overlap(silu, qkv)) {
        return 0;
    }

    params.w          = (const float *) conv->src[1]->data;
    params.w_nb1      = conv->src[1]->nb[1] / sizeof(float);
    params.dst        = (float *) silu->data;
    params.dst_nb1    = silu->nb[1] / sizeof(float);
    params.n_channels = C;

    ggml_cuda_op_ssm_conv_rollback_fused(*cuda_ctx, params);

    return pos - i;
}

// Collects up to four MUL_MAT nodes starting at node_idx that share one src1 and read only
// weights one multi-weight kernel can handle. With q4k, every weight must be a repacked
// Q4_K tensor. Otherwise: repacked Q8_0 weights, or narrow row-major Q8_0 weights (N is not
// a multiple of 32) that the kernel dot path reads directly. Above M = 16 the M=32 kernel
// replaces the multi-weight kernel, so every weight must be repacked; groups with a narrow
// weight fall through to the large-M fusion below.
static int ggml_cuda_match_mul_mat_multi(const ggml_cgraph * cgraph, int node_idx,
                                         const ggml_tensor * src0s[4], ggml_tensor * dsts[4],
                                         bool q4k) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_tensor * src1 = first->src[1];
    if (src1 == nullptr || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || src1->ne[1] < 1 || src1->ne[1] > 64) {
        return 0;
    }
    const int64_t k = first->src[0]->ne[0];
    if (k % 32 != 0) {
        return 0;
    }
    int n = 0;
    while (n < 4 && node_idx + n < cgraph->n_nodes) {
        ggml_tensor * node = cgraph->nodes[node_idx + n];
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != src1 ||
                (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            break;
        }
        const ggml_tensor * w = node->src[0];
        if (w->ne[0] != k) {
            break;
        }
        if (q4k) {
            // the skinny multi kernel reads repacked segments only, one codec type per launch
            if (!ggml_cuda_q4k_skinny_is_repacked(w) || w->type != first->src[0]->type) {
                break;
            }
        } else {
            // a repacked skinny weight is not readable by the q8 multi kernel
            if (ggml_cuda_q4k_skinny_is_repacked(w)) {
                break;
            }
            const bool repacked = ggml_cuda_q8_skinny_is_repacked(w);
            const bool narrow = w->type == GGML_TYPE_Q8_0 && w->view_src == nullptr && w->op == GGML_OP_NONE &&
                                w->ne[1] % 32 != 0;
            if (!repacked && (!narrow || src1->ne[1] > 16)) {
                break;
            }
        }
        if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node)) {
            break;
        }
        src0s[n] = w;
        dsts[n]  = node;
        ++n;
    }
    return n >= 2 ? n : 0;
}

// Large-M version of ggml_cuda_match_mul_mat_multi(): same shared src1, but any number of rows.
// Weights can be repacked Q8, plain contiguous Q8_0, or contiguous F16.
static int ggml_cuda_match_mul_mat_multi_large(const ggml_cgraph * cgraph, int node_idx,
                                               const ggml_tensor * src0s[4], ggml_tensor * dsts[4]) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_tensor * src1 = first->src[1];
    if (src1 == nullptr || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || src1->ne[1] <= 16) {
        return 0;
    }
    const int64_t k = first->src[0]->ne[0];
    int n = 0;
    while (n < 4 && node_idx + n < cgraph->n_nodes) {
        ggml_tensor * node = cgraph->nodes[node_idx + n];
        if (node->op != GGML_OP_MUL_MAT || node->src[1] != src1 ||
                (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            break;
        }
        const ggml_tensor * w = node->src[0];
        const bool repacked_q4k = ggml_cuda_q4k_skinny_is_repacked(w);
        const bool repacked = ggml_cuda_q8_skinny_is_repacked(w);
        const bool q8_0 = w->type == GGML_TYPE_Q8_0 && w->view_src == nullptr && w->op == GGML_OP_NONE &&
                          ggml_is_contiguous(w);
        const bool f16 = w->type == GGML_TYPE_F16 && ggml_is_contiguous(w);
        if (w->ne[0] != k || (!repacked && !repacked_q4k && !q8_0 && !f16)) {
            break;
        }
        if (node->type != GGML_TYPE_F32 || !ggml_is_contiguous(node) || node->op_params[0] == GGML_PREC_F32) {
            break;
        }
        src0s[n] = w;
        dsts[n]  = node;
        ++n;
    }
    return n >= 2 ? n : 0;
}

// Runs several MUL_MAT nodes that share one F32 src1: the input is converted to F16 once,
// each weight is densified to F16, and each product uses cuBLAS.
static bool ggml_cuda_mul_mat_multi_shared_src1(ggml_backend_cuda_context & ctx,
                                                 const ggml_tensor * const src0s[4], ggml_tensor * const dsts[4],
                                                 const int n_nodes, const ggml_tensor * src1) {
    if (!ggml_cuda_mul_mat_cublas_compute_f16(ctx)) {
        return false;
    }

    const int64_t k = src1->ne[0];
    const int64_t m = src1->ne[1];

    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(), ggml_nelements(src1));
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    GGML_ASSERT(to_fp16 != nullptr);
    to_fp16(src1->data, src1_f16.get(), ggml_nelements(src1), ctx.stream());

    ggml_tensor src1_tmp = *src1;
    src1_tmp.type = GGML_TYPE_F16;
    src1_tmp.data = src1_f16.get();
    src1_tmp.extra = nullptr;
    src1_tmp.nb[0] = sizeof(half);
    src1_tmp.nb[1] = src1_tmp.nb[0] * k;
    src1_tmp.nb[2] = src1_tmp.nb[1] * m;
    src1_tmp.nb[3] = src1_tmp.nb[2];

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * w = src0s[i];
        if (w->type == GGML_TYPE_F16) {
            ggml_cuda_mul_mat_cublas(ctx, w, &src1_tmp, dsts[i]);
            continue;
        }

        // the weight buffer is released at the end of every iteration
        ggml_cuda_pool_alloc<half> src0_f16(ctx.pool(), ggml_nelements(w));
        if (ggml_cuda_q8_skinny_is_repacked(w)) {
            ggml_cuda_q8_skinny_to_f16(w, src0_f16.get(), ctx.stream());
        } else if (ggml_cuda_q4k_skinny_is_repacked(w)) {
            ggml_cuda_q4k_skinny_to_f16(w, src0_f16.get(), ctx.stream());
        } else {
            const to_fp16_cuda_t to_fp16_w = ggml_get_to_fp16_cuda(w->type);
            GGML_ASSERT(to_fp16_w != nullptr);
            to_fp16_w(w->data, src0_f16.get(), ggml_nelements(w), ctx.stream());
        }

        ggml_tensor src0_tmp = *w;
        src0_tmp.type = GGML_TYPE_F16;
        src0_tmp.data = src0_f16.get();
        src0_tmp.extra = nullptr;
        src0_tmp.nb[0] = sizeof(half);
        src0_tmp.nb[1] = src0_tmp.nb[0] * w->ne[0];
        src0_tmp.nb[2] = src0_tmp.nb[1] * w->ne[1];
        src0_tmp.nb[3] = src0_tmp.nb[2] * w->ne[2];

        ggml_cuda_mul_mat_cublas(ctx, &src0_tmp, &src1_tmp, dsts[i]);
    }

    return true;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    const int n_conv_rollback = ggml_cuda_try_ssm_conv_rollback_fusion(cuda_ctx, cgraph, i);
    if (n_conv_rollback > 0) {
        return n_conv_rollback;
    }

    const int n_cpy_batch = ggml_cuda_try_cpy_batch(cuda_ctx, cgraph, i);
    if (n_cpy_batch > 0) {
        return n_cpy_batch;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // while the routing trace is on, a MUL_MAT_ID must stay its own node: the fused MMVQ
    // kernels take the ids directly and would bypass the trace hook in ggml_cuda_mul_mat_id
    if (ggml_cuda_routing_trace_enabled() && node->op == GGML_OP_MUL_MAT_ID) {
        return 0;
    }

    // Expert parallelism: the fused kernels write the GLU or ADD node, which does not carry the
    // expert state of the MUL_MAT_ID nodes, so it is copied into the fusion arguments here.
    auto fusion_args_ep = [](const ggml_tensor * mm_id) {
        ggml_cuda_mm_fusion_args_host fusion{};
        if (mm_id->op == GGML_OP_MUL_MAT_ID) {
            fusion.expert_parallel = ggml_mul_mat_id_is_expert_parallel(mm_id);
            fusion.expert_offset   = ggml_mul_mat_id_get_expert_offset(mm_id);
            fusion.n_experts       = (int) mm_id->src[0]->ne[2];
        }
        return fusion;
    };

    // gated_delta_net prologue: fold the q/k l2 normalization into the gdn kernel. Tried before
    // the other rms_norm fusions, which would otherwise eat the first rms_norm of the pattern.
    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_gdn_prologue pro{};
        int                     gdn_idx = 0;
        if (ggml_cuda_match_gdn_prologue(cgraph, i, pro, gdn_idx) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, gdn_idx - i + 1, &gdn_idx, 1)) {
            ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
            const int cache_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, gdn_idx, fused_state_cpy);
            ggml_cuda_op_gated_delta_net_fused(*cuda_ctx, cgraph->nodes[gdn_idx], pro,
                cache_skip > 0 ? &fused_state_cpy : nullptr);
            return gdn_idx - i + (cache_skip > 0 ? cache_skip : 0);
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion = fusion_args_ep(up);
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // GDN gating: beta = sigmoid(beta) and gate = softplus(alpha + dt) * a in one kernel.
    // Either chain may come first in the graph.
    if (node->op == GGML_OP_ADD ||
            (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID)) {
        ggml_cuda_gdn_gating_match match;
        if (ggml_cuda_match_gdn_gating(cgraph, i, match)) {
            ggml_cuda_op_gdn_gating(*cuda_ctx, match.beta_in, match.beta_out,
                match.alpha_in, match.dt, match.a, match.gate_out);
            return match.last - i;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up_n);
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up_n);
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up_n);
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up_n);
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            // q8 skinny gated pair: gate and up in one kernel, GLU written directly
            if (op == GGML_OP_MUL_MAT && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    gate->src[1] == up->src[1] &&
                    ggml_cuda_q8_skinny_mul_mat_gated(*cuda_ctx, gate->src[0], up->src[0], src1, glu)) {
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // q4k skinny gated pair: gate and up in one kernel, GLU written directly
            if (op == GGML_OP_MUL_MAT && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    gate->src[1] == up->src[1] &&
                    ggml_cuda_q4k_skinny_mul_mat_gated(*cuda_ctx, gate->src[0], up->src[0], src1, glu)) {
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // the mul_mat_vec fusion kernels cannot read a repacked weight
            if (ggml_cuda_q8_skinny_is_repacked(gate->src[0]) || ggml_cuda_q8_skinny_is_repacked(up->src[0]) ||
                    ggml_cuda_q4k_skinny_is_repacked(gate->src[0]) || ggml_cuda_q4k_skinny_is_repacked(up->src[0])) {
                continue;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up);
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(up);
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // Several MUL_MAT nodes that share one src1: one input conversion and one multi-weight
    // kernel instead of one input conversion and one launch per weight. Runs after the GLU
    // checks above so that gate/up pairs keep their fused SwiGLU path.
    if (node->op == GGML_OP_MUL_MAT) {
        const ggml_tensor * multi_src0[4];
        ggml_tensor * multi_dst[4];
        const bool q4k = ggml_cuda_q4k_skinny_is_repacked(node->src[0]);
        const int n_multi = ggml_cuda_match_mul_mat_multi(cgraph, i, multi_src0, multi_dst, q4k);
        if (n_multi > 0) {
            // every node of the group is a formal output and is only read after the group, so
            // the fusion memory ranges cover all of them
            ggml_op ops[4] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT };
            int out_nodes[4] = { i, i + 1, i + 2, i + 3 };
            if (ggml_can_fuse_subgraph(cgraph, i, n_multi, ops, out_nodes, n_multi) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_multi, out_nodes, n_multi) &&
                    (q4k ? ggml_cuda_q4k_skinny_mul_mat_multi(*cuda_ctx, multi_src0, multi_dst, n_multi, node->src[1])
                         : ggml_cuda_q8_skinny_mul_mat_multi(*cuda_ctx, multi_src0, multi_dst, n_multi, node->src[1]))) {
                return n_multi - 1;
            }
        }
    }

    // Same fusion for a large M, where no multi-weight kernel exists: one shared input
    // conversion, then one cuBLAS call per weight.
    if (node->op == GGML_OP_MUL_MAT) {
        const ggml_tensor * multi_src0[4];
        ggml_tensor * multi_dst[4];
        const int n_multi = ggml_cuda_match_mul_mat_multi_large(cgraph, i, multi_src0, multi_dst);
        if (n_multi > 0) {
            ggml_op ops[4] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_MUL_MAT };
            int out_nodes[4] = { i, i + 1, i + 2, i + 3 };
            if (ggml_can_fuse_subgraph(cgraph, i, n_multi, ops, out_nodes, n_multi) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_multi, out_nodes, n_multi) &&
                    ggml_cuda_mul_mat_multi_shared_src1(*cuda_ctx, multi_src0, multi_dst, n_multi, node->src[1])) {
                return n_multi - 1;
            }
        }
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(mm_node);
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data = fusion_args_ep(mm_node);
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // RMS_NORM -> MUL -> UNARY(SILU) -> MUL: gated norm at the output of the GDN layers
    if (node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_rms_norm_gated_match match;
        if (ggml_cuda_match_rms_norm_mul_silu_mul(cgraph, i, match)) {
            ggml_cuda_op_rms_norm_mul_silu_mul_fused(*cuda_ctx, match.rms_norm, match.mul, match.silu, match.mul2);
            return match.last - i;
        }
    }

    // ADD -> RMS_NORM -> MUL: the residual add feeds the norm, and both results are kept
    if (ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL }, { i, i + 2 }) &&
            ggml_cuda_should_fuse_add_rms_norm(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        const ggml_tensor * mul    = cgraph->nodes[i + 2];
        const ggml_tensor * weight = mul->src[0] == cgraph->nodes[i + 1] ? mul->src[1] : mul->src[0];
        if (ggml_cuda_add_rms_norm_ranges_ok(node, mul, weight)) {
            ggml_cuda_op_add_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            return 2;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    // CONT -> UNARY(SIGMOID) -> MUL: the attention gate reads a strided view, so the CONT is elided
    if (node->op == GGML_OP_CONT) {
        ggml_cuda_cont_sigmoid_mul_match match;
        if (ggml_cuda_match_cont_sigmoid_mul(cgraph, i, match)) {
            ggml_cuda_op_cont_sigmoid_mul(*cuda_ctx, match.cont, match.sigmoid, match.mul);
            return match.last - i;
        }
    }

    // SCALE -> UNARY(SILU): the hyper-connection mixer activation, silu(x*s), as built by
    // qwen4exp. The scale is folded into the silu kernel, which is what the Strata fused_gr.cu
    // down projection does in its epilogue (refs/strata/src/kernels/cuda/fused_gr.cu:102-103).
    if (node->op == GGML_OP_SCALE &&
            ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU }) &&
            ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SILU &&
            cgraph->nodes[i + 1]->src[0] == node &&
            ggml_is_contiguous(node->src[0]) &&
            node->src[0]->type == GGML_TYPE_F32 &&
            node->type == GGML_TYPE_F32) {
        ggml_cuda_op_scale_silu(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // SCALE -> UNARY(SIGMOID) -> SCALE -> DSV4_HC_POST: the hyper-connection combine gate,
    // post = 2*sigmoid(inject/hc), folded into the post kernel. qwen4exp builds the chain in
    // build_hc_combine. Strata folds the same gate into its up projection epilogue:
    // rv = fmaf(bo, 2*sigmoid(inj/HC), rv) (refs/strata/src/kernels/cuda/fused_gr.cu:124).
    if (node->op == GGML_OP_SCALE &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { i + 3 }) &&
            ggml_check_edges(cgraph, i, {{1, 0, 0}, {2, 0, 1}, {3, 2, 2}}) &&
            ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID &&
            cgraph->nodes[i + 3]->src[3] == nullptr &&
            node->src[0]->type == GGML_TYPE_F32 &&
            node->type == GGML_TYPE_F32 &&
            cgraph->nodes[i + 3]->src[0]->type == GGML_TYPE_F32 &&
            cgraph->nodes[i + 3]->src[1]->type == GGML_TYPE_F32) {
        ggml_cuda_op_dsv4_hc_post_gated(*cuda_ctx, cgraph->nodes[i + 3], node, cgraph->nodes[i + 2]);
        return 3;
    }

    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        // mapped and expert host memory is readable by the device, so it is allowed here
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               ggml_backend_buft_is_cuda_mapped(node->src[j]->buffer->buft) ||
                               ggml_backend_buft_is_cuda_expert(node->src[j]->buffer->buft) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    ggml_cuda_q8_skinny_prepass(*cuda_ctx, cgraph);
    ggml_cuda_q4k_skinny_prepass(*cuda_ctx, cgraph);

    if (cuda_ctx->capture_external) {
        // the outer capture records the kernels, see ggml_backend_cuda_capture_begin()
        ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, false, false, nullptr);
        return GGML_STATUS_SUCCESS;
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        bool has_staging = false;
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph, &has_staging);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph,
                ggml_cuda_graph_identity(cgraph, has_staging));

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

// Whole-graph capture for the meta backend: every subgraph and all-reduce one rank enqueues for a
// graph is recorded into one CUDA graph, which later runs with a single launch instead of one launch
// and one property check per subgraph. While the capture is active graph_compute evaluates the
// graphs directly, so their kernels land in the outer capture.
static bool ggml_backend_cuda_capture_begin(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    GGML_ASSERT(!cuda_ctx->capture_external);

    ggml_cuda_set_device(cuda_ctx->device);
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_lock);
        ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
    }
    if (cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        (void) cudaGetLastError();
        std::lock_guard<std::mutex> lock(ggml_cuda_lock);
        if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
            ggml_cuda_lock_cv.notify_all();
        }
        return false;
    }
    cuda_ctx->capture_external = true;
    return true;
}

static void * ggml_backend_cuda_capture_end(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    GGML_ASSERT(cuda_ctx->capture_external);

    ggml_cuda_set_device(cuda_ctx->device);
    cuda_ctx->capture_external = false;

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(cuda_ctx->stream(), &graph);
    {
        std::lock_guard<std::mutex> lock(ggml_cuda_lock);
        if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
            ggml_cuda_lock_cv.notify_all();
        }
    }

    cudaGraphExec_t exec = nullptr;
    if (err == cudaSuccess && graph != nullptr) {
        err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    }
    if (graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    if (err != cudaSuccess) {
        GGML_LOG_WARN("%s: capture failed on device %d: %s\n", __func__, cuda_ctx->device, cudaGetErrorString(err));
        (void) cudaGetLastError();
        return nullptr;
    }
    return exec;
}

static bool ggml_backend_cuda_capture_launch(ggml_backend_t backend, void * exec) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaGraphLaunch((cudaGraphExec_t) exec, cuda_ctx->stream()));
    return true;
}

static void ggml_backend_cuda_capture_free(void * exec) {
    if (exec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy((cudaGraphExec_t) exec));
    }
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    // the current device is per host thread; submission worker threads do not inherit the setting
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    // the current device is per host thread; submission worker threads do not inherit the setting
    ggml_cuda_set_device(cuda_ctx->device);
    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    // the range is registered page-aligned read-only; the caller owns the memory and releases
    // the reference with the matching unregister call
    const size_t page = ggml_cuda_host_page_size();
    const uintptr_t begin = (uintptr_t) buffer & ~(uintptr_t) (page - 1);
    const size_t offset = (size_t) ((char *) buffer - (char *) begin);
    const uintptr_t end = begin + ((offset + size + page - 1) & ~(page - 1));

    return ggml_cuda_registered_host_range_add(begin, end);
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    const size_t page = ggml_cuda_host_page_size();
    const uintptr_t begin = (uintptr_t) buffer & ~(uintptr_t) (page - 1);
    ggml_cuda_registered_host_range_remove(begin);
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

// True if the device has native BF16 support. On devices without it BF16 weights should be
// converted to F16 when loading a model, see llama_model_loader.
static bool ggml_backend_cuda_device_supports_bf16(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    return fast_bf16_hardware_available(ggml_cuda_info().devices[ctx->device].cc);
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer &&
                (ggml_backend_buft_is_cuda(op->src[i]->buffer->buft) || ggml_backend_buft_is_cuda_mapped(op->src[i]->buffer->buft) ||
                 ggml_backend_buft_is_cuda_expert(op->src[i]->buffer->buft))) {
            // a mapped weights source is host memory readable over UVA by every device, so it
            // does not tie the op to the device it was allocated for. this is a capability
            // branch of the mapped weights buffer type, not a fallback
            if (ggml_backend_buft_is_cuda_mapped_weights(op->src[i]->buffer->buft)) {
                continue;
            }
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_mul_mat_id_get_epilogue(op) == GGML_MUL_MAT_ID_EPILOGUE_SWIGLU) {
                    // the SwiGLU epilogue is the F16 H of silu(gate) * up from the row interleaved
                    // weights. MMVQ writes it for a small batch, the fused-expert kernel for the
                    // rest, at the same boundary as a node without the epilogue; with a type or
                    // shape neither covers there is no implementation here, so the scheduler has to
                    // run the node on a backend that has one
                    const int cc_epilogue = ggml_cuda_info().devices[dev_ctx->device].cc;
                    return ggml_cuda_mul_mat_id_swiglu_mmvq(op, cc_epilogue) ||
                           ggml_cuda_fused_expert_mul_mat_id_supported(op, cc_epilogue);
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    // MUL_MAT_ID covers an F16 activation: the fused-expert kernel reads it as it
                    // is (its shared-memory X tile is F16), and every other case converts it once
                    // into an F32 pool buffer and runs the F32 paths (see ggml_cuda_mul_mat_id),
                    // so the F32 result below is the same decision. MUL_MAT has no such
                    // conversion here.
                    if (op->op != GGML_OP_MUL_MAT_ID) {
                        return false;
                    }
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                        // IQ4_NL rows only need to be a multiple of the 32-value sub-block
                        return op->src[0]->ne[0] % QK4_NL == 0;
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_QSA_SELECT:
            return ggml_cuda_qsa_select_supported(dev_ctx->device, op);
        case GGML_OP_QSA_ATTN:
            return ggml_cuda_qsa_attn_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    // a mapped weights buffer holds host memory readable over UVA by every device. this is a
    // capability branch (the buffer type is only offered by CUDA), not a fallback: any device
    // may run an op that reads it
    const bool mapped_weights = ggml_backend_buft_is_cuda_mapped(buft) &&
        ggml_backend_buft_is_cuda_mapped_weights(buft);
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) ||
           (ggml_backend_buft_is_cuda_mapped(buft) && (buft->device == dev || mapped_weights)) ||
           (ggml_backend_buft_is_cuda_expert(buft) && buft->device == dev) ||
           (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

// extra buffer types of a CUDA device: one mapped host buffer type per device
static ggml_backend_buffer_type_t * ggml_backend_cuda_device_get_extra_bufts(ggml_backend_dev_t dev) {    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type_t> extra_bufts[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    const int device = ctx->device;

    if (!initialized[device]) {
        // the expert buffer type comes first: it is the preferred home of the MoE weights
        // that do not fit in device memory (slots in VRAM + complement in mapped host memory)
        ggml_backend_buffer_type_t expert_buft = ggml_backend_cuda_expert_buffer_type(device);
        if (expert_buft != nullptr) {
            extra_bufts[device].push_back(expert_buft);
        }
        ggml_backend_buffer_type_t mapped_buft = ggml_backend_cuda_mapped_buffer_type(device);
        if (mapped_buft != nullptr) {
            extra_bufts[device].push_back(mapped_buft);
        }
        extra_bufts[device].push_back(nullptr);
        initialized[device] = true;
    }

    return extra_bufts[device].data();
}

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    // while the routing trace is on, the whole-graph capture of the meta backend must not be
    // offered: the trace hook runs per MUL_MAT_ID dispatch, which a captured graph replays
    // without calling. This is a capability branch for a debug feature, not a fallback path.
    if (ggml_cuda_routing_trace_enabled()) {
        if (strcmp(name, "ggml_backend_capture_begin")  == 0 ||
            strcmp(name, "ggml_backend_capture_end")    == 0 ||
            strcmp(name, "ggml_backend_capture_launch") == 0 ||
            strcmp(name, "ggml_backend_capture_free")   == 0 ||
            strcmp(name, "ggml_backend_comm_allreduce_rank_capturable") == 0) {
            return nullptr;
        }
    }

    if (strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *)ggml_backend_cuda_device_get_extra_bufts;
    }
    if (strcmp(name, "ggml_backend_buft_is_device_mapped") == 0) {
        return (void *)ggml_backend_buft_is_cuda_mapped_or_expert;
    }
    if (strcmp(name, "ggml_backend_buft_is_device_expert") == 0) {
        return (void *)ggml_backend_buft_is_cuda_expert;
    }
    if (strcmp(name, "ggml_backend_buft_is_device_mapped_weights") == 0) {
        return (void *)ggml_backend_buft_is_cuda_mapped_weights;
    }
    if (strcmp(name, "ggml_backend_cuda_dev_mapped_weights_buft") == 0) {
        return (void *)ggml_backend_cuda_dev_mapped_weights_buft;
    }
    // the backend independent name of the same buffer type, the meta backend builds its own
    // buffer type for the devices that offer it
    if (strcmp(name, "ggml_backend_dev_mapped_weights_buft") == 0) {
        return (void *)ggml_backend_cuda_dev_mapped_weights_buft;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_set_slot_limit") == 0) {
        return (void *)ggml_cuda_expert_set_slot_limit;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_reset_streams") == 0) {
        return (void *)ggml_cuda_expert_reset_streams;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_last_stage_hits") == 0) {
        return (void *)ggml_cuda_expert_last_stage_hits;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_set_budget") == 0) {
        return (void *)ggml_cuda_expert_set_budget;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_set_tensor_base") == 0) {
        return (void *)ggml_cuda_expert_set_tensor_base;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_record_plan") == 0) {
        return (void *)ggml_cuda_expert_record_plan;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_get_allocation") == 0) {
        return (void *)ggml_backend_cuda_expert_get_allocation;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_get_host_peak") == 0) {
        return (void *)ggml_backend_cuda_expert_get_host_peak;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_reset_allocation") == 0) {
        return (void *)ggml_cuda_expert_reset_allocation;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_device_ready") == 0) {
        return (void *)ggml_backend_cuda_expert_device_ready;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_plan") == 0) {
        return (void *)ggml_cuda_expert_plan;
    }
    if (strcmp(name, "ggml_backend_cuda_expert_check_host_nodes") == 0) {
        return (void *)ggml_backend_cuda_expert_check_host_nodes;
    }
    if (strcmp(name, "ggml_backend_expert_store_model_loaded") == 0) {
        return (void *)ggml_backend_cuda_expert_verify_all;
    }
    if (strcmp(name, "ggml_backend_expert_store_mark_tensor") == 0) {
        return (void *)ggml_cuda_expert_mark_tensor;
    }
    if (strcmp(name, "ggml_backend_dev_supports_bf16") == 0) {
        return (void *)ggml_backend_cuda_device_supports_bf16;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor_exact") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor_exact;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_rank_can") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_rank_can;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_rank") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_rank;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_rank_capturable") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_rank_capturable;
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (strcmp(name, "ggml_backend_cuda_set_assist") == 0) {
        return (void *)ggml_backend_cuda_set_assist;
    }
    if (strcmp(name, "ggml_backend_cuda_assist_run") == 0) {
        return (void *)ggml_backend_cuda_assist_run;
    }
    if (strcmp(name, "ggml_backend_cuda_assist_workspace_size") == 0) {
        return (void *)ggml_backend_cuda_assist_workspace_size;
    }
    if (strcmp(name, "ggml_backend_cuda_assist_scratch_layout") == 0) {
        return (void *)ggml_backend_cuda_assist_scratch_layout;
    }
    if (strcmp(name, "ggml_backend_cuda_get_device_ordinal") == 0) {
        return (void *)ggml_backend_cuda_get_device_ordinal;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (strcmp(name, "ggml_backend_capture_begin") == 0) {
        return (void *)ggml_backend_cuda_capture_begin;
    }
    if (strcmp(name, "ggml_backend_capture_end") == 0) {
        return (void *)ggml_backend_cuda_capture_end;
    }
    if (strcmp(name, "ggml_backend_capture_launch") == 0) {
        return (void *)ggml_backend_cuda_capture_launch;
    }
    if (strcmp(name, "ggml_backend_capture_free") == 0) {
        return (void *)ggml_backend_cuda_capture_free;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
