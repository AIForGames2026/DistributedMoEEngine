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
#define A2_HD 256 /* tf_attention256 (TASK-29) */
#define A2_QT 16
#define A2_SMEM ((A2_QT + 2 * 32) * A2_HD * 4)
#define DR_DK 128 /* tf_delta_rule */
#define MM_BM 128 /* tf_mm_q8 and tf_mm_f32 */
#define MM_BN 64
#define MV_TOK 8 /* tf_mv_q8 */
#define TM_BM 64 /* tf_tm_q8 */
#define TM_BN 64
/* Calls that move at most this much go through pinned memory and wait once. */
#define XFER_SMALL (1 << 20)
#define XFER_OUTS 8

struct tf_gpu {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction q8_0_matmul;
    CUfunction attention_merge;
    CUfunction silu_mul;
    CUfunction mm_q8, mm_f32, mv_q8, mv_gateup, gather, combine;
    /* Attention kernels, one of each per KV cache format (TASK-27): f32, bf16, fp8, bf16 keys with fp8 values. */
    CUfunction attention[4], attention_wide[4], attention_tc[4], attention_tc128[4], attention_dec8[4], attention_dec16[4];
    /* 256-dim heads (TASK-29): one kernel per format and its merge; a256_ok once their shared memory is granted. */
    CUfunction attention256[4], attention_merge256, delta_rule;
    int a256_ok;
    CUfunction tm_q8, rowscale, silu_rows;
    /* TF_MM_KERNEL: 1 the first Q8_0 kernel, 2 tiled f32 kernels, 0 tensor cores where they apply. */
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
    /* Gated delta rule: per-token inputs and outputs. */
    CUdeviceptr dr_in, dr_out;
    size_t dr_in_bytes, dr_out_bytes;
    /* Experts: inputs, gate and up outputs, and results for a whole layer. */
    CUdeviceptr ex, eg, eu, ey, exg, eo, meta;
    size_t ex_bytes, eg_bytes, eu_bytes, ey_bytes, exg_bytes, eo_bytes, meta_bytes;
    /* Per-row input scales for the tensor-core matmul: inputs, and expert gate outputs. */
    CUdeviceptr up, up2;
    size_t up_bytes, up2_bytes;
    /* Host side of `meta`: weight tables and row maps, copied in one go. */
    unsigned char *host_meta;
    size_t host_meta_bytes;
    /* Pinned staging for small calls, and the outputs to hand back after the wait. */
    unsigned char *pin;
    int xfer_on, xfer_outs;
    size_t xfer_used;
    void *xfer_dst[XFER_OUTS];
    size_t xfer_off[XFER_OUTS], xfer_len[XFER_OUTS];
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
        (r = cuModuleGetFunction(&g->attention_merge, g->mod, "tf_attention_merge")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->silu_mul, g->mod, "tf_silu_mul")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mm_q8, g->mod, "tf_mm_q8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mm_f32, g->mod, "tf_mm_f32")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mv_q8, g->mod, "tf_mv_q8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->mv_gateup, g->mod, "tf_mv_q8_gateup")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->gather, g->mod, "tf_gather")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->combine, g->mod, "tf_combine")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->tm_q8, g->mod, "tf_tm_q8")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->rowscale, g->mod, "tf_rowscale")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->silu_rows, g->mod, "tf_silu_rows")) != CUDA_SUCCESS) {
        tf_gpu_close(g);
        return fail(r, "loading kernels", err, err_len);
    }
    static const char *const suffix[4] = {"", "_bf16", "_fp8", "_bf16_fp8"};
    static const char *const base[6] = {"tf_attention", "tf_attention_wide", "tf_attention_tc", "tf_attention_tc128",
                                        "tf_attention_decode8", "tf_attention_decode16"};
    for (int f = 0; f < 4; f++)
    {
        CUfunction *dst[6] = {&g->attention[f], &g->attention_wide[f], &g->attention_tc[f], &g->attention_tc128[f],
                              &g->attention_dec8[f], &g->attention_dec16[f]};
        for (int i = 0; i < 6; i++)
        {
            char name[64];
            snprintf(name, sizeof name, "%s%s", base[i], suffix[f]);
            if ((r = cuModuleGetFunction(dst[i], g->mod, name)) != CUDA_SUCCESS)
            {
                tf_gpu_close(g);
                return fail(r, "loading attention kernels", err, err_len);
            }
        }
    }
    for (int f = 0; f < 4; f++)
    {
        char name[64];
        snprintf(name, sizeof name, "tf_attention256%s", suffix[f]);
        if ((r = cuModuleGetFunction(&g->attention256[f], g->mod, name)) != CUDA_SUCCESS)
        {
            tf_gpu_close(g);
            return fail(r, "loading attention kernels", err, err_len);
        }
    }
    if ((r = cuModuleGetFunction(&g->attention_merge256, g->mod, "tf_attention_merge256")) != CUDA_SUCCESS ||
        (r = cuModuleGetFunction(&g->delta_rule, g->mod, "tf_delta_rule")) != CUDA_SUCCESS)
    {
        tf_gpu_close(g);
        return fail(r, "loading kernels", err, err_len);
    }
    /* The wide and 256-dim kernels need more shared memory than the default 48 KB. */
    int optin = 0;
    cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, g->dev);
    g->a256_ok = optin >= A2_SMEM;
    for (int f = 0; f < 4; f++)
    {
        g->a256_ok = g->a256_ok && cuFuncSetAttribute(g->attention256[f], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                      A2_SMEM) == CUDA_SUCCESS;
    }
    g->wide_ok = optin >= ATT_W_SMEM;
    for (int f = 0; f < 4; f++)
    {
        g->wide_ok = g->wide_ok && cuFuncSetAttribute(g->attention_wide[f], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                      ATT_W_SMEM) == CUDA_SUCCESS;
    }
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

static size_t device_bytes;

size_t tf_gpu_device_bytes(void)
{
    return device_bytes;
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
    device_bytes += len;
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
    device_bytes -= *have;
    *p = 0;
    *have = 0;
    CUresult r = cuMemAlloc(p, want);
    if (r == CUDA_SUCCESS)
    {
        *have = want;
        device_bytes += want;
    }
    return r;
}

static size_t align16(size_t n)
{
    return (n + 15) & ~(size_t)15;
}

/* Starts a call that moves `bytes` in all: small calls stage through pinned
 * memory with asynchronous copies, so the call waits only in xfer_end. */
static void xfer_begin(tf_gpu *g, size_t bytes)
{
    g->xfer_used = 0;
    g->xfer_outs = 0;
    g->xfer_on = 0;
    if (bytes > XFER_SMALL)
    {
        return;
    }
    if (!g->pin && cuMemAllocHost((void **)&g->pin, XFER_SMALL) != CUDA_SUCCESS)
    {
        g->pin = NULL;
        return;
    }
    g->xfer_on = 1;
}

static CUresult xfer_in(tf_gpu *g, CUdeviceptr dev, const void *src, size_t len)
{
    if (!g->xfer_on)
    {
        return cuMemcpyHtoD(dev, src, len);
    }
    unsigned char *at = g->pin + g->xfer_used;
    memcpy(at, src, len);
    g->xfer_used += align16(len);
    return cuMemcpyHtoDAsync(dev, at, len, 0);
}

static CUresult xfer_out(tf_gpu *g, void *dst, CUdeviceptr dev, size_t len)
{
    if (!g->xfer_on || g->xfer_outs == XFER_OUTS)
    {
        return cuMemcpyDtoH(dst, dev, len);
    }
    const int i = g->xfer_outs++;
    g->xfer_dst[i] = dst;
    g->xfer_off[i] = g->xfer_used;
    g->xfer_len[i] = len;
    g->xfer_used += align16(len);
    return cuMemcpyDtoHAsync(g->pin + g->xfer_off[i], dev, len, 0);
}

/* Waits for the call's copies and kernels, then hands back its outputs. */
static CUresult xfer_end(tf_gpu *g)
{
    if (!g->xfer_on)
    {
        return CUDA_SUCCESS;
    }
    CUresult r = cuStreamSynchronize(0);
    if (r != CUDA_SUCCESS)
    {
        return r;
    }
    for (int i = 0; i < g->xfer_outs; i++)
    {
        memcpy(g->xfer_dst[i], g->pin + g->xfer_off[i], g->xfer_len[i]);
    }
    g->xfer_on = 0;
    return CUDA_SUCCESS;
}

static CUresult launch_q8(tf_gpu *g, CUdeviceptr w, CUdeviceptr x, CUdeviceptr y, int rows, int cols, int n)
{
    void *args[] = {&w, &x, &y, &rows, &cols, &n};
    const unsigned threads = 256; /* 8 rows per block */
    const unsigned grid_x = (unsigned)((rows * 32 + threads - 1) / threads);
    const unsigned grid_y = (unsigned)((n + TOKENS_PER_PASS - 1) / TOKENS_PER_PASS);
    return cuLaunchKernel(g->q8_0_matmul, grid_x, grid_y, 1, threads, 1, 1, 0, 0, args, NULL);
}

/* Whether launch_mm uses the tensor-core kernel, which needs each input row's scale. */
static int use_tm(const tf_gpu *g, int format, int most)
{
    return format == TF_GPU_Q8_0 && most > MV_TOK && g->mm_kernel == 0 && g->tc_ok;
}

/* y = W x for one matrix, or for a group where matrix z takes rows first[z] to first[z + 1] - 1
 * of x, with row scales `up` when use_tm. The kernel depends only on the format and the shape. */
static CUresult launch_mm(tf_gpu *g, int format, CUdeviceptr w0, CUdeviceptr wt, CUdeviceptr first, int groups,
                          int most, CUdeviceptr up, CUdeviceptr x, CUdeviceptr y, int rows, int cols)
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
    if (use_tm(g, format, most))
    {
        void *targs[] = {&w0, &wt, &first, &x, &y, &rows, &cols, &most, &up};
        return cuLaunchKernel(g->tm_q8, (unsigned)((rows + TM_BM - 1) / TM_BM), (unsigned)((most + TM_BN - 1) / TM_BN),
                              (unsigned)groups, 128, 1, 1, 0, 0, targs, NULL);
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
        (r = grow(&g->y, &g->y_bytes, (size_t)n * total * sizeof(float))) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    xfer_begin(g, align16(xb) + count * 16 + (size_t)n * total * sizeof(float));
    if ((r = xfer_in(g, g->x, x, xb)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    if (use_tm(g, format, n))
    {
        void *sargs[] = {&g->x, &g->up, &n, &cols};
        if ((r = grow(&g->up, &g->up_bytes, (size_t)n * sizeof(float))) != CUDA_SUCCESS ||
            (r = cuLaunchKernel(g->rowscale, (unsigned)((n + 7) / 8), 1, 1, 256, 1, 1, 0, 0, sargs, NULL)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
    }
    CUdeviceptr out = g->y;
    for (int i = 0; i < count; i++)
    {
        if ((r = launch_mm(g, format, (CUdeviceptr)w[i], 0, 0, 1, n, g->up, g->x, out, rows[i], cols)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
        out += (size_t)n * rows[i] * sizeof(float);
    }
    out = g->y;
    for (int i = 0; i < count; i++)
    {
        const size_t yb = (size_t)n * rows[i] * sizeof(float);
        if ((r = xfer_out(g, y[i], out, yb)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
        out += yb;
    }
    if ((r = xfer_end(g)) != CUDA_SUCCESS)
    {
        return -(int)r;
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
        (r = grow(&g->meta, &g->meta_bytes, meta)) != CUDA_SUCCESS ||
        (r = grow(&g->up, &g->up_bytes, (size_t)rows * sizeof(float))) != CUDA_SUCCESS ||
        (r = grow(&g->up2, &g->up2_bytes, (size_t)rows * sizeof(float))) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    xfer_begin(g, 2 * align16(xb) + align16(meta));
    if ((r = xfer_in(g, g->ex, x, xb)) != CUDA_SUCCESS || (r = xfer_in(g, g->meta, hm, meta)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr gt = g->meta, ut = g->meta + groups * sizeof(uint64_t), dt = g->meta + 2 * groups * sizeof(uint64_t);
    CUdeviceptr dfirst = g->meta + first_at, dsrc = g->meta + src_at, dinv = g->meta + inv_at, dw = g->meta + w_at;
    void *gargs[] = {&g->ex, &dsrc, &g->exg, &d, &g->up};
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
            (r = launch_mm(g, TF_GPU_Q8_0, 0, dt, dfirst, groups, most, 0, g->eg, g->ey, d, f)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
    }
    else
    {
        void *sargs[] = {&g->eg, &g->eu, &f, &g->up2};
        if ((r = launch_mm(g, TF_GPU_Q8_0, 0, gt, dfirst, groups, most, g->up, g->exg, g->eg, f, d)) != CUDA_SUCCESS ||
            (r = launch_mm(g, TF_GPU_Q8_0, 0, ut, dfirst, groups, most, g->up, g->exg, g->eu, f, d)) != CUDA_SUCCESS ||
            (r = cuLaunchKernel(g->silu_rows, (unsigned)rows, 1, 1, 256, 1, 1, 0, 0, sargs, NULL)) != CUDA_SUCCESS ||
            (r = launch_mm(g, TF_GPU_Q8_0, 0, dt, dfirst, groups, most, g->up2, g->eg, g->ey, d, f)) != CUDA_SUCCESS)
        {
            return -(int)r;
        }
    }
    void *cargs[] = {&g->ey, &dinv, &dw, &g->eo, &k, &d};
    if ((r = cuLaunchKernel(g->combine, (unsigned)n, 1, 1, 256, 1, 1, 0, 0, cargs, NULL)) != CUDA_SUCCESS ||
        (r = xfer_out(g, y, g->eo, xb)) != CUDA_SUCCESS || (r = xfer_end(g)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    return 0;
}

int tf_gpu_alloc_host(tf_gpu *g, size_t len, void **host, uint64_t *dev)
{
    (void)g;
    CUresult r;
    if ((r = cuMemAllocHost(host, len)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr p;
    if ((r = cuMemHostGetDevicePointer(&p, *host, 0)) != CUDA_SUCCESS)
    {
        cuMemFreeHost(*host);
        return -(int)r;
    }
    *dev = (uint64_t)p;
    return 0;
}

void tf_gpu_free_host(tf_gpu *g, void *host)
{
    (void)g;
    cuMemFreeHost(host);
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
                     const uint64_t *tiles_k, const uint64_t *tiles_v, int tiles, int splits, float *out, int fmt, int hd)
{
    if (n <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || tiles <= 0 || splits <= 0 || fmt < 0 || fmt > 3 ||
        (hd != ATT_HD && !(hd == A2_HD && g->a256_ok)))
        return -1;
    const int rows = n * (heads / kv_heads);
    const size_t qb = (size_t)n * heads * hd * sizeof(float);
    const size_t tb = (size_t)tiles * sizeof(uint64_t);
    const size_t parts = (size_t)splits * kv_heads * rows;
    CUresult r;
    if ((r = grow(&g->aq, &g->aq_bytes, qb)) != CUDA_SUCCESS || (r = grow(&g->aout, &g->aout_bytes, qb)) != CUDA_SUCCESS ||
        (r = grow(&g->tiles, &g->tiles_bytes, 2 * tb)) != CUDA_SUCCESS ||
        (r = grow(&g->pm, &g->pm_bytes, parts * sizeof(float))) != CUDA_SUCCESS ||
        (r = grow(&g->pl, &g->pl_bytes, parts * sizeof(float))) != CUDA_SUCCESS ||
        (r = grow(&g->pacc, &g->pacc_bytes, parts * hd * sizeof(float))) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    xfer_begin(g, 2 * align16(qb) + 2 * align16(tb));
    if ((r = xfer_in(g, g->aq, q, qb)) != CUDA_SUCCESS || (r = xfer_in(g, g->tiles, tiles_k, tb)) != CUDA_SUCCESS ||
        (r = xfer_in(g, g->tiles + tb, tiles_v, tb)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
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
    if (hd == A2_HD)
    {
        r = cuLaunchKernel(g->attention256[fmt], (unsigned)((rows + A2_QT - 1) / A2_QT), (unsigned)kv_heads, (unsigned)splits,
                           256, 1, 1, A2_SMEM, 0, args, NULL);
    }
    else if (dec)
    {
        r = cuLaunchKernel(rows <= 8 ? g->attention_dec8[fmt] : g->attention_dec16[fmt], 1, (unsigned)kv_heads, (unsigned)splits, 128,
                           1, 1, 0, 0, args, NULL);
    }
    else if (tc)
    {
        r = cuLaunchKernel(which == 4 ? g->attention_tc128[fmt] : g->attention_tc[fmt], (unsigned)((rows + tc_rows - 1) / tc_rows),
                           (unsigned)kv_heads, (unsigned)splits, 2 * tc_rows, 1, 1, 0, 0, args, NULL);
    }
    else
    {
        const int qt = wide ? ATT_W_Q : ATT_QT;
        r = cuLaunchKernel(wide ? g->attention_wide[fmt] : g->attention[fmt], (unsigned)((rows + qt - 1) / qt), (unsigned)kv_heads,
                           (unsigned)splits, 256, 1, 1, wide ? ATT_W_SMEM : 0, 0, args, NULL);
    }
    if (r != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    void *margs[] = {&g->pm, &g->pl, &g->pacc, &g->aout, &n, &heads, &kv_heads, &splits};
    if ((r = cuLaunchKernel(hd == A2_HD ? g->attention_merge256 : g->attention_merge, (unsigned)rows, (unsigned)kv_heads, 1,
                            (unsigned)hd, 1, 1, 0, 0, margs, NULL)) != CUDA_SUCCESS ||
        (r = xfer_out(g, out, g->aout, qb)) != CUDA_SUCCESS || (r = xfer_end(g)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    return 0;
}

int tf_gpu_delta_rule(tf_gpu *g, const float *mixed, const float *beta, const float *decay, int n, int cd, int kh, int dk,
                      int vh, int dv, uint64_t state, float *core)
{
    if (n <= 0 || dk != DR_DK || kh <= 0 || vh % kh || dv <= 0 || dv > 256 || !state)
        return -1;
    const size_t mb = (size_t)n * cd * sizeof(float), hb = (size_t)n * vh * sizeof(float);
    const size_t ob = (size_t)n * vh * dv * sizeof(float);
    CUresult r;
    if ((r = grow(&g->dr_in, &g->dr_in_bytes, align16(mb) + 2 * align16(hb))) != CUDA_SUCCESS ||
        (r = grow(&g->dr_out, &g->dr_out_bytes, ob)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr dm = g->dr_in, db = g->dr_in + align16(mb), dd = db + align16(hb);
    xfer_begin(g, align16(mb) + 2 * align16(hb) + align16(ob));
    if ((r = xfer_in(g, dm, mixed, mb)) != CUDA_SUCCESS || (r = xfer_in(g, db, beta, hb)) != CUDA_SUCCESS ||
        (r = xfer_in(g, dd, decay, hb)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
    CUdeviceptr st = (CUdeviceptr)state;
    void *args[] = {&dm, &db, &dd, &st, &g->dr_out, &n, &cd, &kh, &vh, &dv};
    if ((r = cuLaunchKernel(g->delta_rule, (unsigned)vh, 1, 1, (unsigned)dv, 1, 1, 0, 0, args, NULL)) != CUDA_SUCCESS ||
        (r = xfer_out(g, core, g->dr_out, ob)) != CUDA_SUCCESS || (r = xfer_end(g)) != CUDA_SUCCESS)
    {
        return -(int)r;
    }
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
    CUdeviceptr att[] = {g->aq, g->aout, g->tiles, g->pm, g->pl, g->pacc, g->ex, g->eg, g->eu, g->ey, g->exg, g->eo, g->meta, g->up, g->up2};
    for (size_t i = 0; i < sizeof att / sizeof att[0]; i++)
        if (att[i])
            cuMemFree(att[i]);
    free(g->host_meta);
    if (g->pin)
    {
        cuMemFreeHost(g->pin);
    }
    if (g->mod)
        cuModuleUnload(g->mod);
    cuDevicePrimaryCtxRelease(g->dev);
    free(g);
}
