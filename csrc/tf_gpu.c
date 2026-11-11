/* GPU matmuls (TASK-10) and attention (TASK-12) on the CUDA driver API. See include/tf_gpu.h. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tf_gpu.h"

#define MAX_MAPS 8
#define TOKENS_PER_PASS 8 /* must match kernels.cu */
#define ATT_HD 128 /* must match kernels.cu */
#define ATT_QT 32
#define ATT_W_Q 64 /* tf_attention_wide */
#define ATT_W_SMEM ((2 * 64 * 132 + 64 * ATT_HD) * 4)

struct tf_gpu {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction q8_0_matmul;
    CUfunction attention;
    CUfunction attention_merge;
    CUfunction attention_wide;
    CUfunction silu_mul;
    /* 1: tf_attention, 0 or 2: tf_attention_wide when the device allows it (TF_ATT_KERNEL). */
    int att_kernel;
    int wide_ok;
    tf_gpu_info info;
    /* Scratch for inputs and outputs, grown on demand. */
    CUdeviceptr x;
    size_t x_bytes;
    CUdeviceptr y;
    size_t y_bytes;
    /* Attention: queries, outputs, tile tables, and the splits' partial results. */
    CUdeviceptr aq, aout, tiles, pm, pl, pacc;
    size_t aq_bytes, aout_bytes, tiles_bytes, pm_bytes, pl_bytes, pacc_bytes;
    /* Experts: inputs, gate and up outputs, and results for a whole layer. */
    CUdeviceptr ex, eg, eu, ey;
    size_t ex_bytes, eg_bytes, eu_bytes, ey_bytes;
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
        (r = cuModuleGetFunction(&g->q8_0_matmul, g->mod, "tf_q8_0_matmul")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention, g->mod, "tf_attention")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_merge, g->mod, "tf_attention_merge")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_wide, g->mod, "tf_attention_wide")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->silu_mul, g->mod, "tf_silu_mul")) != CUDA_SUCCESS) {
        tf_gpu_close(g);
        return fail(r, "loading kernels", err, err_len);
    }
    /* The wide kernel needs more shared memory than the default 48 KB. */
    int optin = 0;
    cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, g->dev);
    g->wide_ok = optin >= ATT_W_SMEM && cuFuncSetAttribute(g->attention_wide, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                          ATT_W_SMEM) == CUDA_SUCCESS;
    const char *k = getenv("TF_ATT_KERNEL");
    g->att_kernel = k ? atoi(k) : 0;
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

static CUresult launch_q8(tf_gpu *g, CUdeviceptr w, CUdeviceptr x, CUdeviceptr y, int rows, int cols, int n)
{
    void *args[] = {&w, &x, &y, &rows, &cols, &n};
    const unsigned threads = 256; /* 8 rows per block */
    const unsigned grid_x = (unsigned)((rows * 32 + threads - 1) / threads);
    const unsigned grid_y = (unsigned)((n + TOKENS_PER_PASS - 1) / TOKENS_PER_PASS);
    return cuLaunchKernel(g->q8_0_matmul, grid_x, grid_y, 1, threads, 1, 1, 0, 0, args, NULL);
}

int tf_gpu_experts(tf_gpu *g, int experts, const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                   const int *first, const float *x, int d, int f, float *y)
{
    if (experts <= 0 || d <= 0 || f <= 0 || d % 32 || f % 32)
        return -1;
    const int rows = first[experts];
    if (rows <= 0)
        return 0;
    const size_t xb = (size_t)rows * d * sizeof(float), fb = (size_t)rows * f * sizeof(float);
    CUresult r;
    if ((r = grow(&g->ex, &g->ex_bytes, xb)) != CUDA_SUCCESS || (r = grow(&g->eg, &g->eg_bytes, fb)) != CUDA_SUCCESS ||
        (r = grow(&g->eu, &g->eu_bytes, fb)) != CUDA_SUCCESS || (r = grow(&g->ey, &g->ey_bytes, xb)) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->ex, x, xb)) != CUDA_SUCCESS)
        return -(int)r;
    for (int e = 0; e < experts; e++) {
        const int n = first[e + 1] - first[e];
        if (n <= 0)
            continue;
        const CUdeviceptr xe = g->ex + (size_t)first[e] * d * sizeof(float);
        const CUdeviceptr ge = g->eg + (size_t)first[e] * f * sizeof(float);
        const CUdeviceptr ue = g->eu + (size_t)first[e] * f * sizeof(float);
        const CUdeviceptr ye = g->ey + (size_t)first[e] * d * sizeof(float);
        int count = n * f;
        void *sargs[] = {(void *)&ge, (void *)&ue, &count};
        if ((r = launch_q8(g, (CUdeviceptr)gate[e], xe, ge, f, d, n)) != CUDA_SUCCESS ||
            (r = launch_q8(g, (CUdeviceptr)up[e], xe, ue, f, d, n)) != CUDA_SUCCESS ||
            (r = cuLaunchKernel(g->silu_mul, (unsigned)((count + 255) / 256), 1, 1, 256, 1, 1, 0, 0, sargs, NULL)) !=
                CUDA_SUCCESS ||
            (r = launch_q8(g, (CUdeviceptr)down[e], ge, ye, d, f, n)) != CUDA_SUCCESS)
            return -(int)r;
    }
    if ((r = cuMemcpyDtoH(y, g->ey, xb)) != CUDA_SUCCESS)
        return -(int)r;
    return 0;
}

int tf_gpu_register(tf_gpu *g, void *host, size_t len, uint64_t *dev)
{
    CUresult r;
    if ((r = cuMemHostRegister(host, len, CU_MEMHOSTREGISTER_DEVICEMAP)) != CUDA_SUCCESS)
        return -(int)r;
    CUdeviceptr p;
    if ((r = cuMemHostGetDevicePointer(&p, host, 0)) != CUDA_SUCCESS) {
        cuMemHostUnregister(host);
        return -(int)r;
    }
    (void)g;
    *dev = (uint64_t)p;
    return 0;
}

void tf_gpu_unregister(tf_gpu *g, void *host)
{
    (void)g;
    cuMemHostUnregister(host);
}

int tf_gpu_attention(tf_gpu *g, const float *q, int n, int pos0, int heads, int kv_heads, int kvd, float scale,
                     const uint64_t *tiles_k, const uint64_t *tiles_v, int tiles, int splits, float *out)
{
    if (n <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || tiles <= 0 || splits <= 0)
        return -1;
    const int rows = n * (heads / kv_heads);
    const size_t qb = (size_t)n * heads * ATT_HD * sizeof(float);
    const size_t tb = (size_t)tiles * sizeof(uint64_t);
    const size_t parts = (size_t)splits * kv_heads * rows;
    CUresult r;
    if ((r = grow(&g->aq, &g->aq_bytes, qb)) != CUDA_SUCCESS || (r = grow(&g->aout, &g->aout_bytes, qb)) != CUDA_SUCCESS ||
        (r = grow(&g->tiles, &g->tiles_bytes, 2 * tb)) != CUDA_SUCCESS ||
        (r = grow(&g->pm, &g->pm_bytes, parts * sizeof(float))) != CUDA_SUCCESS ||
        (r = grow(&g->pl, &g->pl_bytes, parts * sizeof(float))) != CUDA_SUCCESS ||
        (r = grow(&g->pacc, &g->pacc_bytes, parts * ATT_HD * sizeof(float))) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->aq, q, qb)) != CUDA_SUCCESS || (r = cuMemcpyHtoD(g->tiles, tiles_k, tb)) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->tiles + tb, tiles_v, tb)) != CUDA_SUCCESS)
        return -(int)r;
    CUdeviceptr tk = g->tiles, tv = g->tiles + tb;
    int per_split = (tiles + splits - 1) / splits;
    /* The wide kernel is faster for every shape measured; it takes tiles in pairs. */
    int wide = g->wide_ok && g->att_kernel != 1;
    if (wide)
        per_split += per_split & 1;
    void *args[] = {&g->aq, &tk, &tv, &g->pm, &g->pl, &g->pacc, &n, &pos0, &heads, &kv_heads, &kvd, &per_split, &scale};
    const int qt = wide ? ATT_W_Q : ATT_QT;
    const unsigned grid_x = (unsigned)((rows + qt - 1) / qt);
    if ((r = cuLaunchKernel(wide ? g->attention_wide : g->attention, grid_x, (unsigned)kv_heads, (unsigned)splits, 256, 1,
                            1, wide ? ATT_W_SMEM : 0, 0, args, NULL)) != CUDA_SUCCESS)
        return -(int)r;
    void *margs[] = {&g->pm, &g->pl, &g->pacc, &g->aout, &n, &heads, &kv_heads, &splits};
    if ((r = cuLaunchKernel(g->attention_merge, (unsigned)rows, (unsigned)kv_heads, 1, ATT_HD, 1, 1, 0, 0, margs, NULL)) !=
            CUDA_SUCCESS ||
        (r = cuMemcpyDtoH(out, g->aout, qb)) != CUDA_SUCCESS)
        return -(int)r;
    return 0;
}

int tf_gpu_set_attention_kernel(tf_gpu *g, int which)
{
    const int old = g->att_kernel;
    g->att_kernel = which;
    return old;
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
    CUdeviceptr att[] = {g->aq, g->aout, g->tiles, g->pm, g->pl, g->pacc, g->ex, g->eg, g->eu, g->ey};
    for (size_t i = 0; i < sizeof att / sizeof att[0]; i++)
        if (att[i])
            cuMemFree(att[i]);
    if (g->mod)
        cuModuleUnload(g->mod);
    cuDevicePrimaryCtxRelease(g->dev);
    free(g);
}
