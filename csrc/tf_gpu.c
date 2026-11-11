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
#define ATT_TC_Q 64 /* tf_attention_tc, with 2 threads per row */
#define ATT_DEC_ROWS 16 /* tf_attention_decode16 */
#define MM_BM 128 /* tf_mm_q8 and tf_mm_f32 */
#define MM_BN 64
#define MV_TOK 8 /* tf_mv_q8 */

struct tf_gpu {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction q8_0_matmul;
    CUfunction attention;
    CUfunction attention_merge;
    CUfunction attention_wide;
    CUfunction silu_mul;
    CUfunction mm_q8, mm_f32, mv_q8, mv_gateup, gather, combine, attention_tc, attention_tc128, attention_dec8, attention_dec16;
    /* 1: the first Q8_0 kernel for every matmul (TF_MM_KERNEL); 0: tiled and decode kernels. */
    int mm_kernel;
    int tc_ok;
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
    CUdeviceptr ex, eg, eu, ey, exg, eo, meta;
    size_t ex_bytes, eg_bytes, eu_bytes, ey_bytes, exg_bytes, eo_bytes, meta_bytes;
    /* Host side of `meta`: weight tables and row maps, copied in one go. */
    unsigned char *host_meta;
    size_t host_meta_bytes;
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
        (r = cuModuleGetFunction(&g->silu_mul, g->mod, "tf_silu_mul")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mm_q8, g->mod, "tf_mm_q8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mm_f32, g->mod, "tf_mm_f32")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mv_q8, g->mod, "tf_mv_q8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mv_gateup, g->mod, "tf_mv_q8_gateup")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->gather, g->mod, "tf_gather")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->combine, g->mod, "tf_combine")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_tc, g->mod, "tf_attention_tc")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_tc128, g->mod, "tf_attention_tc128")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_dec8, g->mod, "tf_attention_decode8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->attention_dec16, g->mod, "tf_attention_decode16")) != CUDA_SUCCESS) {
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
    const char *mk = getenv("TF_MM_KERNEL");
    g->mm_kernel = mk ? atoi(mk) : 0;
    /* f16 mma needs compute capability 8.0. */
    g->tc_ok = g->info.major >= 8;
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

static CUresult launch_q8(tf_gpu *g, CUdeviceptr w, CUdeviceptr x, CUdeviceptr y, int rows, int cols, int n)
{
    void *args[] = {&w, &x, &y, &rows, &cols, &n};
    const unsigned threads = 256; /* 8 rows per block */
    const unsigned grid_x = (unsigned)((rows * 32 + threads - 1) / threads);
    const unsigned grid_y = (unsigned)((n + TOKENS_PER_PASS - 1) / TOKENS_PER_PASS);
    return cuLaunchKernel(g->q8_0_matmul, grid_x, grid_y, 1, threads, 1, 1, 0, 0, args, NULL);
}

/* y = W x for one matrix, or for a group where matrix z takes rows first[z] to
 * first[z + 1] - 1 of x. The kernel depends only on the format and the shape. */
static CUresult launch_mm(tf_gpu *g, int format, CUdeviceptr w0, CUdeviceptr wt, CUdeviceptr first, int groups,
                          int most, CUdeviceptr x, CUdeviceptr y, int rows, int cols)
{
    if (most <= 0)
    {
        return CUDA_SUCCESS;
    }
    if (format == TF_GPU_Q8_0 && g->mm_kernel == 1 && !wt)
    {
        return launch_q8(g, w0, x, y, rows, cols, most);
    }
    void *args[] = {&w0, &wt, &first, &x, &y, &rows, &cols, &most};
    if (format == TF_GPU_Q8_0 && most <= MV_TOK)
    {
        return cuLaunchKernel(g->mv_q8, (unsigned)((rows + 7) / 8), 1, (unsigned)groups, 256, 1, 1, 0, 0, args, NULL);
    }
    return cuLaunchKernel(format == TF_GPU_Q8_0 ? g->mm_q8 : g->mm_f32, (unsigned)((rows + MM_BM - 1) / MM_BM),
                          (unsigned)((most + MM_BN - 1) / MM_BN), (unsigned)groups, 256, 1, 1, 0, 0, args, NULL);
}

static int format_ok(int format, int cols)
{
    return (format == TF_GPU_Q8_0 || format == TF_GPU_F32) && cols > 0 && cols % 32 == 0;
}

int tf_gpu_matmul(tf_gpu *g, uint64_t w, int format, const float *x, int n, int rows, int cols, float *y)
{
    float *ys[1] = {y};
    return tf_gpu_matmuls(g, 1, &w, format, &rows, x, n, cols, ys);
}

int tf_gpu_q8_0_matmul(tf_gpu *g, uint64_t w, const float *x, int n, int rows, int cols, float *y)
{
    return tf_gpu_matmul(g, w, TF_GPU_Q8_0, x, n, rows, cols, y);
}

int tf_gpu_matmuls(tf_gpu *g, int count, const uint64_t *w, int format, const int *rows, const float *x, int n, int cols,
                   float *const *y)
{
    if (count <= 0 || n <= 0 || !format_ok(format, cols))
    {
        return -1;
    }
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        if (rows[i] <= 0)
        {
            return -1;
        }
        total += (size_t)rows[i];
    }
    const size_t xb = (size_t)n * cols * sizeof(float);
    CUresult r;
    if ((r = grow(&g->x, &g->x_bytes, xb)) != CUDA_SUCCESS ||
        (r = grow(&g->y, &g->y_bytes, (size_t)n * total * sizeof(float))) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->x, x, xb)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr out = g->y;
    for (int i = 0; i < count; i++)
    {
        if ((r = launch_mm(g, format, (CUdeviceptr)w[i], 0, 0, 1, n, g->x, out, rows[i], cols)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
        out += (size_t)n * rows[i] * sizeof(float);
    }
    out = g->y;
    for (int i = 0; i < count; i++)
    {
        const size_t yb = (size_t)n * rows[i] * sizeof(float);
        if ((r = cuMemcpyDtoH(y[i], out, yb)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
        out += yb;
    }
    return 0;
}

int tf_gpu_experts(tf_gpu *g, int groups, const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                   const int *first, const int *src, const int *inv, const float *w, const float *x, int n, int k,
                   int d, int f, float *y)
{
    if (groups <= 0 || n <= 0 || k <= 0 || d <= 0 || f <= 0 || d % 32 || f % 32)
    {
        return -1;
    }
    const int rows = first[groups];
    if (rows <= 0)
    {
        return -1;
    }
    int most = 0;
    for (int e = 0; e < groups; e++)
    {
        if (first[e + 1] - first[e] > most)
        {
            most = first[e + 1] - first[e];
        }
    }
    const size_t calls = (size_t)n * k;
    const size_t tables = 3 * (size_t)groups * sizeof(uint64_t);
    const size_t meta = tables + ((size_t)groups + 1 + (size_t)rows + calls) * sizeof(int) + calls * sizeof(float);
    const size_t xb = (size_t)n * d * sizeof(float), rb = (size_t)rows * d * sizeof(float);
    const size_t fb = (size_t)rows * f * sizeof(float);
    if (g->host_meta_bytes < meta)
    {
        free(g->host_meta);
        g->host_meta = malloc(meta);
        g->host_meta_bytes = g->host_meta ? meta : 0;
        if (!g->host_meta)
        {
            return -1;
        }
    }
    unsigned char *hm = g->host_meta;
    memcpy(hm, gate, groups * sizeof(uint64_t));
    memcpy(hm + groups * sizeof(uint64_t), up, groups * sizeof(uint64_t));
    memcpy(hm + 2 * groups * sizeof(uint64_t), down, groups * sizeof(uint64_t));
    size_t at = tables;
    memcpy(hm + at, first, (groups + 1) * sizeof(int));
    const size_t first_at = at;
    at += (groups + 1) * sizeof(int);
    memcpy(hm + at, src, rows * sizeof(int));
    const size_t src_at = at;
    at += rows * sizeof(int);
    memcpy(hm + at, inv, calls * sizeof(int));
    const size_t inv_at = at;
    at += calls * sizeof(int);
    memcpy(hm + at, w, calls * sizeof(float));
    const size_t w_at = at;
    CUresult r;
    if ((r = grow(&g->ex, &g->ex_bytes, xb)) != CUDA_SUCCESS || (r = grow(&g->exg, &g->exg_bytes, rb)) != CUDA_SUCCESS ||
        (r = grow(&g->eg, &g->eg_bytes, fb)) != CUDA_SUCCESS || (r = grow(&g->eu, &g->eu_bytes, fb)) != CUDA_SUCCESS ||
        (r = grow(&g->ey, &g->ey_bytes, rb)) != CUDA_SUCCESS || (r = grow(&g->eo, &g->eo_bytes, xb)) != CUDA_SUCCESS ||
        (r = grow(&g->meta, &g->meta_bytes, meta)) != CUDA_SUCCESS || (r = cuMemcpyHtoD(g->ex, x, xb)) != CUDA_SUCCESS ||
        (r = cuMemcpyHtoD(g->meta, hm, meta)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr gt = g->meta, ut = g->meta + groups * sizeof(uint64_t), dt = g->meta + 2 * groups * sizeof(uint64_t);
    CUdeviceptr dfirst = g->meta + first_at, dsrc = g->meta + src_at, dinv = g->meta + inv_at, dw = g->meta + w_at;
    void *gargs[] = {&g->ex, &dsrc, &g->exg, &d};
    if ((r = cuLaunchKernel(g->gather, (unsigned)rows, 1, 1, 128, 1, 1, 0, 0, gargs, NULL)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    if (g->mm_kernel == 1)
    {
        for (int e = 0; e < groups; e++)
        {
            const int cnt = first[e + 1] - first[e];
            if (cnt <= 0)
            {
                continue;
            }
            const CUdeviceptr xe = g->exg + (size_t)first[e] * d * sizeof(float);
            const CUdeviceptr ge = g->eg + (size_t)first[e] * f * sizeof(float);
            const CUdeviceptr ue = g->eu + (size_t)first[e] * f * sizeof(float);
            const CUdeviceptr ye = g->ey + (size_t)first[e] * d * sizeof(float);
            int count = cnt * f;
            void *sargs[] = {(void *)&ge, (void *)&ue, &count};
            if ((r = launch_q8(g, (CUdeviceptr)gate[e], xe, ge, f, d, cnt)) != CUDA_SUCCESS ||
                (r = launch_q8(g, (CUdeviceptr)up[e], xe, ue, f, d, cnt)) != CUDA_SUCCESS ||
                (r = cuLaunchKernel(g->silu_mul, (unsigned)((count + 255) / 256), 1, 1, 256, 1, 1, 0, 0, sargs, NULL)) !=
                    CUDA_SUCCESS ||
                (r = launch_q8(g, (CUdeviceptr)down[e], ge, ye, d, f, cnt)) != CUDA_SUCCESS)
            {
                return -(int)r;
            }
        }
    }
    else if (most <= MV_TOK)
    {
        void *args[] = {&gt, &ut, &dfirst, &g->exg, &g->eg, &f, &d};
        if ((r = cuLaunchKernel(g->mv_gateup, (unsigned)((f + 7) / 8), 1, (unsigned)groups, 256, 1, 1, 0, 0, args,
                                NULL)) != CUDA_SUCCESS ||
            (r = launch_mm(g, TF_GPU_Q8_0, 0, dt, dfirst, groups, most, g->eg, g->ey, d, f)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
    }
    else
    {
        int count = rows * f;
        void *sargs[] = {&g->eg, &g->eu, &count};
        if ((r = launch_mm(g, TF_GPU_Q8_0, 0, gt, dfirst, groups, most, g->exg, g->eg, f, d)) != CUDA_SUCCESS ||
            (r = launch_mm(g, TF_GPU_Q8_0, 0, ut, dfirst, groups, most, g->exg, g->eu, f, d)) != CUDA_SUCCESS ||
            (r = cuLaunchKernel(g->silu_mul, (unsigned)((count + 255) / 256), 1, 1, 256, 1, 1, 0, 0, sargs, NULL)) !=
                CUDA_SUCCESS ||
            (r = launch_mm(g, TF_GPU_Q8_0, 0, dt, dfirst, groups, most, g->eg, g->ey, d, f)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
    }
    void *cargs[] = {&g->ey, &dinv, &dw, &g->eo, &k, &d};
    if ((r = cuLaunchKernel(g->combine, (unsigned)n, 1, 1, 256, 1, 1, 0, 0, cargs, NULL)) != CUDA_SUCCESS ||
        (r = cuMemcpyDtoH(y, g->eo, xb)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
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
    /* Tensor cores for prefill, the decode kernel for a few rows, else the wide
     * kernel. The choice depends only on the shape. */
    int which = g->att_kernel;
    if (!which)
    {
        which = rows <= ATT_DEC_ROWS ? 5 : !g->tc_ok ? 2 : rows >= 2 * ATT_TC_Q ? 4 : rows >= ATT_TC_Q ? 3 : 2;
    }
    const int dec = which == 5 && rows <= ATT_DEC_ROWS;
    const int tc = (which == 3 || which == 4) && g->tc_ok;
    const int tc_rows = which == 4 ? 2 * ATT_TC_Q : ATT_TC_Q;
    const int wide = !tc && !dec && which != 1 && g->wide_ok;
    if (wide)
    {
        per_split += per_split & 1;
    }
    void *args[] = {&g->aq, &tk, &tv, &g->pm, &g->pl, &g->pacc, &n, &pos0, &heads, &kv_heads, &kvd, &per_split, &scale};
    if (dec)
    {
        r = cuLaunchKernel(rows <= 8 ? g->attention_dec8 : g->attention_dec16, 1, (unsigned)kv_heads, (unsigned)splits, 128,
                           1, 1, 0, 0, args, NULL);
    }
    else if (tc)
    {
        r = cuLaunchKernel(which == 4 ? g->attention_tc128 : g->attention_tc, (unsigned)((rows + tc_rows - 1) / tc_rows),
                           (unsigned)kv_heads, (unsigned)splits, 2 * tc_rows, 1, 1, 0, 0, args, NULL);
    }
    else
    {
        const int qt = wide ? ATT_W_Q : ATT_QT;
        r = cuLaunchKernel(wide ? g->attention_wide : g->attention, (unsigned)((rows + qt - 1) / qt), (unsigned)kv_heads,
                           (unsigned)splits, 256, 1, 1, wide ? ATT_W_SMEM : 0, 0, args, NULL);
    }
    if (r != CUDA_SUCCESS)
    {
        return -(int)r;
    }
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

int tf_gpu_set_matmul_kernel(tf_gpu *g, int which)
{
    const int old = g->mm_kernel;
    g->mm_kernel = which;
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
    CUdeviceptr att[] = {g->aq, g->aout, g->tiles, g->pm, g->pl, g->pacc, g->ex, g->eg, g->eu, g->ey, g->exg, g->eo, g->meta};
    for (size_t i = 0; i < sizeof att / sizeof att[0]; i++)
        if (att[i])
            cuMemFree(att[i]);
    free(g->host_meta);
    if (g->mod)
        cuModuleUnload(g->mod);
    cuDevicePrimaryCtxRelease(g->dev);
    free(g);
}
