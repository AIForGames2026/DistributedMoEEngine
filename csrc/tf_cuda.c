#include <stdio.h>

#include <cuda_runtime_api.h>

#include "tf_gate.h"

int tf_cuda_device_count(void)
{
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(err));
        return -(int)err;
    }
    return n;
}

int tf_cuda_device_info(int dev, tf_cuda_info *out)
{
    struct cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess)
        return -(int)err;
    snprintf(out->name, sizeof out->name, "%s", prop.name);
    out->major = prop.major;
    out->minor = prop.minor;
    out->sm_count = prop.multiProcessorCount;
    out->total_mem = prop.totalGlobalMem;

    /* Touch the device so the gate proves a context and an allocation work,
     * not only that the driver answers queries. */
    void *p = NULL;
    if ((err = cudaSetDevice(dev)) != cudaSuccess || (err = cudaMalloc(&p, 1 << 20)) != cudaSuccess)
        return -(int)err;
    err = cudaMemset(p, 0, 1 << 20);
    cudaFree(p);
    return err == cudaSuccess ? 0 : -(int)err;
}
