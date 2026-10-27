/* TASK-05 CUDA probe: prints one JSON object describing GPU 0, including
 * whether the hardware Decompression Engine path is exposed. Driver API only,
 * so it builds with a plain C compiler:
 *   cc -std=c11 -I/usr/local/cuda/include tools/probe/cuda_probe.c -lcuda */
#include <stdio.h>

#include <cuda.h>

#define CK(call)                                                   \
    do {                                                           \
        CUresult rc_ = (call);                                     \
        if (rc_ != CUDA_SUCCESS) {                                 \
            const char *msg_ = "unknown";                          \
            cuGetErrorString(rc_, &msg_);                          \
            fprintf(stderr, "%s failed: %s\n", #call, msg_);       \
            return 1;                                              \
        }                                                          \
    } while (0)

static int attr(CUdevice dev, CUdevice_attribute a)
{
    int v = -1;
    if (cuDeviceGetAttribute(&v, a, dev) != CUDA_SUCCESS)
        return -1;
    return v;
}

/* Allocates one granule with the HW decompress usage flag and asks the driver
 * whether the pointer is decompress-capable. Returns -1 if the allocation is
 * refused, which is itself an answer. */
static int hw_decompress_pointer(CUdevice dev)
{
    CUmemAllocationProp prop = {0};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.allocFlags.usage = CU_MEM_CREATE_USAGE_HW_DECOMPRESS;

    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return -1;
    CUmemGenericAllocationHandle h;
    if (cuMemCreate(&h, gran, &prop, 0) != CUDA_SUCCESS)
        return -1;

    int capable = -1;
    CUdeviceptr ptr = 0;
    if (cuMemAddressReserve(&ptr, gran, 0, 0, 0) == CUDA_SUCCESS) {
        if (cuMemMap(ptr, gran, 0, h, 0) == CUDA_SUCCESS) {
            CUmemAccessDesc access = {{CU_MEM_LOCATION_TYPE_DEVICE, dev}, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
            if (cuMemSetAccess(ptr, gran, &access, 1) == CUDA_SUCCESS) {
                int v = 0;
                if (cuPointerGetAttribute(&v, CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE, ptr) == CUDA_SUCCESS)
                    capable = v;
            }
            cuMemUnmap(ptr, gran);
        }
        cuMemAddressFree(ptr, gran);
    }
    cuMemRelease(h);
    return capable;
}

int main(void)
{
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    char name[128];
    CK(cuDeviceGetName(name, sizeof name, dev));
    size_t total = 0;
    CK(cuDeviceTotalMem(&total, dev));
    int driver = 0;
    CK(cuDriverGetVersion(&driver));

    CUcontext ctx;
    CUctxCreateParams params = {0};
    CK(cuCtxCreate(&ctx, &params, 0, dev));
    size_t free_b = 0, total_b = 0;
    CK(cuMemGetInfo(&free_b, &total_b));
    int hw_ptr = hw_decompress_pointer(dev);
    cuCtxDestroy(ctx);

    printf("{\n");
    printf("  \"name\": \"%s\",\n", name);
    printf("  \"driver_api_version\": %d,\n", driver);
    printf("  \"compute_capability\": \"%d.%d\",\n",
           attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
           attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR));
    printf("  \"sm_count\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
    printf("  \"total_mem_bytes\": %zu,\n", total);
    printf("  \"free_mem_bytes\": %zu,\n", free_b);
    printf("  \"l2_cache_bytes\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE));
    printf("  \"max_shared_mem_per_block_optin\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN));
    printf("  \"integrated\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_INTEGRATED));
    printf("  \"pageable_memory_access\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS));
    printf("  \"gpu_direct_rdma_supported\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED));
    printf("  \"mem_decompress_algorithm_mask\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK));
    printf("  \"mem_decompress_max_length\": %d,\n", attr(dev, CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH));
    printf("  \"hw_decompress_capable_pointer\": %d\n", hw_ptr);
    printf("}\n");
    return 0;
}
