/* GPU matmuls (TASK-10) on the CUDA driver API. See include/tf_gpu.h. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tf_gpu.h"

#define MAX_MAPS 8
#define TOKENS_PER_PASS 8 /* must match kernels.cu */

struct tf_gpu {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction q8_0_matmul;
    tf_gpu_info info;
    /* Scratch for inputs and outputs, grown on demand. */
    CUdeviceptr x;
    size_t x_bytes;
    CUdeviceptr y;
    size_t y_bytes;
    /* Regions registered in place (integrated) or copied (discrete). */
    void *registered[MAX_MAPS];
    CUdeviceptr copied[MAX_MAPS];
    int n_registered, n_copied;
};

static int fail(CUresult r, const char *what, char *err, size_t len)
{
    const char *s = NULL;
    cuGetErrorString(r, &s);
    if (err && len)
        snprintf(err, len, "%s: %s", what, s ? s : "unknown error");
    return -(int)r;
}

int tf_gpu_open(tf_gpu **out, const void *fatbin, char *err, size_t err_len)
{
    *out = NULL;
    tf_gpu *g = calloc(1, sizeof *g);
    if (!g)
        return -1;
    CUresult r;
    if ((r = cuInit(0)) != CUDA_SUCCESS || (r = cuDeviceGet(&g->dev, 0)) != CUDA_SUCCESS) {
        free(g);
        return fail(r, "cuInit/cuDeviceGet", err, err_len);
    }
    if ((r = cuDevicePrimaryCtxRetain(&g->ctx, g->dev)) != CUDA_SUCCESS || (r = cuCtxSetCurrent(g->ctx)) != CUDA_SUCCESS) {
        free(g);
        return fail(r, "primary context", err, err_len);
    }
    cuDeviceGetName(g->info.name, sizeof g->info.name, g->dev);
    cuDeviceGetAttribute(&g->info.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g->dev);
    cuDeviceGetAttribute(&g->info.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g->dev);
    cuDeviceGetAttribute(&g->info.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, g->dev);
    if ((r = cuModuleLoadData(&g->mod, fatbin)) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->q8_0_matmul, g->mod, "tf_q8_0_matmul")) != CUDA_SUCCESS) {
        tf_gpu_close(g);
        return fail(r, "loading kernels", err, err_len);
    }
    *out = g;
    return 0;
}

void tf_gpu_get_info(const tf_gpu *g, tf_gpu_info *info)
{
    *info = g->info;
}

int tf_gpu_map(tf_gpu *g, void *host, size_t len, uint64_t *dev)
{
    CUresult r;
    if (g->info.integrated) {
        if (g->n_registered == MAX_MAPS)
            return -1;
        if ((r = cuMemHostRegister(host, len, CU_MEMHOSTREGISTER_DEVICEMAP)) != CUDA_SUCCESS)
            return -(int)r;
        CUdeviceptr p;
        if ((r = cuMemHostGetDevicePointer(&p, host, 0)) != CUDA_SUCCESS)
            return -(int)r;
        g->registered[g->n_registered++] = host;
        *dev = (uint64_t)p;
        return 0;
    }
    if (g->n_copied == MAX_MAPS)
        return -1;
    CUdeviceptr p;
    if ((r = cuMemAlloc(&p, len)) != CUDA_SUCCESS)
        return -(int)r;
    if ((r = cuMemcpyHtoD(p, host, len)) != CUDA_SUCCESS) {
        cuMemFree(p);
        return -(int)r;
    }
    g->copied[g->n_copied++] = p;
    *dev = (uint64_t)p;
    return 0;
}

static CUresult grow(CUdeviceptr *p, size_t *have, size_t want)
{
    if (*have >= want)
        return CUDA_SUCCESS;
    if (*p)
        cuMemFree(*p);
    *p = 0;
    *have = 0;
    CUresult r = cuMemAlloc(p, want);
    if (r == CUDA_SUCCESS)
        *have = want;
    return r;
}

int tf_gpu_q8_0_matmul(tf_gpu *g, uint64_t w, const float *x, int n, int rows, int cols, float *y)
{
    if (n <= 0 || rows <= 0 || cols <= 0 || cols % 32)
        return -1;
    const size_t xb = (size_t)n * cols * sizeof(float), yb = (size_t)n * rows * sizeof(float);
    CUresult r;
    if ((r = grow(&g->x, &g->x_bytes, xb)) != CUDA_SUCCESS || (r = grow(&g->y, &g->y_bytes, yb)) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->x, x, xb)) != CUDA_SUCCESS)
        return -(int)r;
    CUdeviceptr wp = (CUdeviceptr)w;
    void *args[] = {&wp, &g->x, &g->y, &rows, &cols, &n};
    const unsigned threads = 256; /* 8 rows per block */
    const unsigned grid_x = (unsigned)((rows * 32 + threads - 1) / threads);
    const unsigned grid_y = (unsigned)((n + TOKENS_PER_PASS - 1) / TOKENS_PER_PASS);
    if ((r = cuLaunchKernel(g->q8_0_matmul, grid_x, grid_y, 1, threads, 1, 1, 0, 0, args, NULL)) != CUDA_SUCCESS ||
        (r = cuMemcpyDtoH(y, g->y, yb)) != CUDA_SUCCESS)
        return -(int)r;
    return 0;
}

void tf_gpu_close(tf_gpu *g)
{
    if (!g)
        return;
    cuCtxSetCurrent(g->ctx);
    for (int i = 0; i < g->n_registered; i++)
        cuMemHostUnregister(g->registered[i]);
    for (int i = 0; i < g->n_copied; i++)
        cuMemFree(g->copied[i]);
    if (g->x)
        cuMemFree(g->x);
    if (g->y)
        cuMemFree(g->y);
    if (g->mod)
        cuModuleUnload(g->mod);
    cuDevicePrimaryCtxRelease(g->dev);
    free(g);
}
