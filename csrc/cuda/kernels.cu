// TensorFold-Swarm GPU kernels (TASK-10). Compiled by nvcc to a fatbin that
// the engine embeds and loads through the CUDA driver API (csrc/tf_gpu.c), so
// no nvcc-built host code is linked. Weights stay in their GGUF block format.

#include <cuda_fp16.h>

// Inputs handled per pass; a second grid dimension covers more.
#define TOKENS_PER_PASS 8

// Y[t][r] = sum over c of W[r][c] * X[t][c] for t < n, where W is `rows` rows
// of `cols` values in ggml Q8_0 (34-byte blocks: f16 scale, 32 int8). One warp
// per row: lane i reads byte i of each block, so a warp's loads are one
// contiguous 32-byte run per block, then a warp sum gives the dot product.
extern "C" __global__ void tf_q8_0_matmul(const unsigned char *__restrict__ w, const float *__restrict__ x,
                                          float *__restrict__ y, int rows, int cols, int n)
{
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int t0 = blockIdx.y * TOKENS_PER_PASS;
    if (row >= rows)
        return;
    const int nb = cols >> 5;
    const unsigned char *wr = w + (size_t)row * nb * 34;
    const int nt = min(TOKENS_PER_PASS, n - t0);

    float acc[TOKENS_PER_PASS];
#pragma unroll
    for (int t = 0; t < TOKENS_PER_PASS; t++)
        acc[t] = 0.0f;
    for (int b = 0; b < nb; b++) {
        const unsigned char *blk = wr + b * 34;
        const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
        const float v = d * (float)(signed char)blk[2 + lane];
        const int c = (b << 5) + lane;
#pragma unroll
        for (int t = 0; t < TOKENS_PER_PASS; t++)
            if (t < nt)
                acc[t] += v * x[(size_t)(t0 + t) * cols + c];
    }
#pragma unroll
    for (int t = 0; t < TOKENS_PER_PASS; t++) {
        float s = acc[t];
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0 && t < nt)
            y[(size_t)(t0 + t) * rows + row] = s;
    }
}

// g[i] = silu(g[i]) * u[i], between an expert's gate/up and down matmuls (TASK-12).
extern "C" __global__ void tf_silu_mul(float *__restrict__ g, const float *__restrict__ u, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float a = g[i];
        g[i] = a / (1.0f + expf(-a)) * u[i];
    }
}

// GLM-5.3-Flash's SwiGLU (TASK-32): the gate clamped from above, the up projection on both sides.
extern "C" __global__ void tf_silu_mul_clamp(float *__restrict__ g, const float *__restrict__ u, int count, float limit)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        const float a = fminf(g[i], limit);
        g[i] = a / (1.0f + expf(-a)) * fminf(fmaxf(u[i], -limit), limit);
    }
}

// Q4_K's j-th 6-bit scale and min from its 12 packed bytes (ggml's get_scale_min_k4).
__device__ __forceinline__ void q4k_scale_min(const unsigned char *q, int j, int *sc, int *m)
{
    if (j < 4)
    {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    }
    else
    {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// One layer's Q4_K expert rows (TASK-32): group z has weight wt[z] and batch rows first[z] to
// first[z + 1] - 1 of x (cols wide), writing those rows of y (rows wide). Each warp takes RW
// weight rows; each lane decodes 8 of a block's 256 values per row and applies them to TT inputs.
template <int TT, int RW>
__device__ __forceinline__ void q4k_grouped(const unsigned long long *__restrict__ wt, const int *__restrict__ first,
                                            const float *__restrict__ x, float *__restrict__ y, int rows, int cols)
{
    const int g = blockIdx.z;
    const int base = first[g], cnt = first[g + 1] - base;
    if (cnt <= 0)
    {
        return;
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 8 + warp) * RW;
    const int nb = cols >> 8;
    __shared__ __align__(16) float xs[TT][256];
    const int c = lane >> 3, sub = lane & 7;
    const int o_lo = c * 64 + sub * 4, o_hi = o_lo + 32;
    for (int t0 = 0; t0 < cnt; t0 += TT)
    {
        const int nt = min(TT, cnt - t0);
        float acc[RW][TT];
#pragma unroll
        for (int r = 0; r < RW; r++)
        {
#pragma unroll
            for (int t = 0; t < TT; t++)
            {
                acc[r][t] = 0.0f;
            }
        }
        for (int b = 0; b < nb; b++)
        {
            __syncthreads();
            for (int i = threadIdx.x; i < TT * 256; i += 256)
            {
                const int t = i >> 8, col = i & 255;
                xs[t][col] = t < nt ? x[(size_t)(base + t0 + t) * cols + (b << 8) + col] : 0.0f;
            }
            __syncthreads();
            float w[RW][8];
#pragma unroll
            for (int r = 0; r < RW; r++)
            {
                const unsigned char *blk = (const unsigned char *)wt[g] + ((size_t)min(row0 + r, rows - 1) * nb + b) * 144;
                const float d = __half2float(__ushort_as_half((unsigned short)(blk[0] | (blk[1] << 8))));
                const float dmin = __half2float(__ushort_as_half((unsigned short)(blk[2] | (blk[3] << 8))));
                int s1, m1, s2, m2;
                q4k_scale_min(blk + 4, 2 * c, &s1, &m1);
                q4k_scale_min(blk + 4, 2 * c + 1, &s2, &m2);
                const float d1 = d * (float)s1, mm1 = dmin * (float)m1, d2 = d * (float)s2, mm2 = dmin * (float)m2;
                const unsigned int q = *(const unsigned int *)(blk + 16 + c * 32 + sub * 4);
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    const unsigned int byte = (q >> (8 * j)) & 0xFF;
                    w[r][j] = d1 * (float)(byte & 0xF) - mm1;
                    w[r][4 + j] = d2 * (float)(byte >> 4) - mm2;
                }
            }
#pragma unroll
            for (int t = 0; t < TT; t++)
            {
                const float4 lo = *(const float4 *)&xs[t][o_lo];
                const float4 hi = *(const float4 *)&xs[t][o_hi];
#pragma unroll
                for (int r = 0; r < RW; r++)
                {
                    acc[r][t] += w[r][0] * lo.x + w[r][1] * lo.y + w[r][2] * lo.z + w[r][3] * lo.w +
                                 w[r][4] * hi.x + w[r][5] * hi.y + w[r][6] * hi.z + w[r][7] * hi.w;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RW; r++)
        {
#pragma unroll
            for (int t = 0; t < TT; t++)
            {
                float s = acc[r][t];
                for (int off = 16; off > 0; off >>= 1)
                {
                    s += __shfl_xor_sync(0xffffffffu, s, off);
                }
                if (lane == 0 && t < nt && row0 + r < rows)
                {
                    y[(size_t)(base + t0 + t) * rows + row0 + r] = s;
                }
            }
        }
    }
}

// Few inputs per expert (decode): a row per warp, up to 8 inputs per pass.
extern "C" __global__ void __launch_bounds__(256) tf_q4k_grouped(const unsigned long long *__restrict__ wt,
                                                                 const int *__restrict__ first,
                                                                 const float *__restrict__ x, float *__restrict__ y,
                                                                 int rows, int cols)
{
    q4k_grouped<8, 1>(wt, first, x, y, rows, cols);
}

// Many inputs per expert (prefill): 4 rows per warp share each staged input, 16 inputs per pass.
extern "C" __global__ void __launch_bounds__(256) tf_q4k_grouped_wide(const unsigned long long *__restrict__ wt,
                                                                      const int *__restrict__ first,
                                                                      const float *__restrict__ x, float *__restrict__ y,
                                                                      int rows, int cols)
{
    q4k_grouped<16, 4>(wt, first, x, y, rows, cols);
}

// The power of 2 that brings a row whose largest magnitude is `mx` near the top of f16's range.
__device__ __forceinline__ float row_up(float mx)
{
    const int k = max(-100, min(100, 140 - ((__float_as_int(mx) >> 23) & 0xff)));
    return __int_as_float((127 + k) << 23);
}

// The largest of every thread's `mx` in a block of up to 1024 threads.
__device__ __forceinline__ float block_max(float mx)
{
    __shared__ float part[32];
    for (int off = 16; off > 0; off >>= 1)
    {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    if ((threadIdx.x & 31) == 0)
    {
        part[threadIdx.x >> 5] = mx;
    }
    __syncthreads();
    mx = 0.0f;
    for (int w = 0; w < (int)(blockDim.x + 31) / 32; w++)
    {
        mx = fmaxf(mx, part[w]);
    }
    return mx;
}

// tf_silu_mul by rows of f values, one block per row, also writing each row's tf_rowscale.
extern "C" __global__ void tf_silu_rows(float *__restrict__ g, const float *__restrict__ u, int f, float *__restrict__ up)
{
    float *gr = g + (size_t)blockIdx.x * f;
    const float *ur = u + (size_t)blockIdx.x * f;
    float mx = 0.0f;
    for (int c = threadIdx.x; c < f; c += blockDim.x)
    {
        const float a = gr[c];
        const float v = a / (1.0f + expf(-a)) * ur[c];
        gr[c] = v;
        mx = fmaxf(mx, fabsf(v));
    }
    mx = block_max(mx);
    if (threadIdx.x == 0)
    {
        up[blockIdx.x] = row_up(mx);
    }
}

// Many inputs at once: a block covers 128 weight rows by 64 inputs, so each
// weight is read once per 64 inputs. Sums still run in column order.
#define MM_BM 128
#define MM_BN 64
#define MM_LDW (MM_BM + 4)
#define MM_LDX (MM_BN + 4)

template <bool Q8>
__device__ __forceinline__ void mm_block(const unsigned char *__restrict__ w, const float *__restrict__ x,
                                         float *__restrict__ y, int rows, int cols, int n, int r0, int t0)
{
    __shared__ __align__(16) float ws[32][MM_LDW];
    __shared__ __align__(16) float xs[32][MM_LDX];
    const int tid = threadIdx.x;
    const int ty = tid >> 4; // rows 8 ty to 8 ty + 7
    const int tx = tid & 15; // inputs 4 tx to 4 tx + 3
    const int nb = cols >> 5;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            acc[i][j] = 0.0f;
        }
    }
    for (int b = 0; b < nb; b++)
    {
        if (Q8)
        {
            const int r = tid >> 1, half = tid & 1;
            const int gr = r0 + r;
            float v[16];
            if (gr < rows)
            {
                const unsigned char *blk = w + ((size_t)gr * nb + b) * 34;
                const float d = __half2float(__ushort_as_half(*(const unsigned short *)blk));
                const unsigned short *qp = (const unsigned short *)(blk + 2 + half * 16);
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    const unsigned short u = qp[j];
                    v[2 * j] = d * (float)(signed char)(u & 0xff);
                    v[2 * j + 1] = d * (float)(signed char)(u >> 8);
                }
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 16; j++)
                {
                    v[j] = 0.0f;
                }
            }
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                ws[half * 16 + j][r] = v[j];
            }
        }
        else
        {
            const float *wf = (const float *)w;
            for (int i = tid; i < MM_BM * 8; i += 256)
            {
                const int r = i >> 3, c4 = i & 7;
                const int gr = r0 + r;
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (gr < rows)
                {
                    v = ((const float4 *)(wf + (size_t)gr * cols + b * 32))[c4];
                }
                ws[c4 * 4][r] = v.x;
                ws[c4 * 4 + 1][r] = v.y;
                ws[c4 * 4 + 2][r] = v.z;
                ws[c4 * 4 + 3][r] = v.w;
            }
        }
        for (int i = tid; i < MM_BN * 8; i += 256)
        {
            const int t = i >> 3, c4 = i & 7;
            const int gt = t0 + t;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gt < n)
            {
                v = ((const float4 *)(x + (size_t)gt * cols + b * 32))[c4];
            }
            xs[c4 * 4][t] = v.x;
            xs[c4 * 4 + 1][t] = v.y;
            xs[c4 * 4 + 2][t] = v.z;
            xs[c4 * 4 + 3][t] = v.w;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 32; k++)
        {
            const float4 a0 = *(const float4 *)&ws[k][ty * 8];
            const float4 a1 = *(const float4 *)&ws[k][ty * 8 + 4];
            const float4 bv = *(const float4 *)&xs[k][tx * 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bb[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    acc[i][j] += a[i] * bb[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        const int gt = t0 + tx * 4 + j;
        if (gt >= n)
        {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            const int gr = r0 + ty * 8 + i;
            if (gr < rows)
            {
                y[(size_t)gt * rows + gr] = acc[i][j];
            }
        }
    }
}

// Picks one matrix of a group: with a table, matrix blockIdx.z takes input
// rows first[z] to first[z + 1] - 1; without one, w0 takes n0 rows.
struct mm_pick
{
    const unsigned char *w;
    int f0;
    int n;
};

__device__ __forceinline__ mm_pick mm_select(unsigned long long w0, const unsigned long long *wt, const int *first, int n0)
{
    mm_pick p;
    if (wt)
    {
        const int z = blockIdx.z;
        p.w = (const unsigned char *)wt[z];
        p.f0 = first[z];
        p.n = first[z + 1] - p.f0;
    }
    else
    {
        p.w = (const unsigned char *)w0;
        p.f0 = 0;
        p.n = n0;
    }
    return p;
}

// y = W x for many inputs, W in Q8_0 (tf_mm_q8) or f32 (tf_mm_f32). Grid:
// rows / 128, inputs / 64, and one z per matrix of a group.
extern "C" __global__ void __launch_bounds__(256) tf_mm_q8(unsigned long long w0, const unsigned long long *wt,
                                                           const int *first, const float *x, float *y, int rows,
                                                           int cols, int n0)
{
    const mm_pick p = mm_select(w0, wt, first, n0);
    if (blockIdx.y * MM_BN >= p.n)
    {
        return;
    }
    mm_block<true>(p.w, x + (size_t)p.f0 * cols, y + (size_t)p.f0 * rows, rows, cols, p.n, blockIdx.x * MM_BM,
                   blockIdx.y * MM_BN);
}

extern "C" __global__ void __launch_bounds__(256) tf_mm_f32(unsigned long long w0, const unsigned long long *wt,
                                                            const int *first, const float *x, float *y, int rows,
                                                            int cols, int n0)
{
    const mm_pick p = mm_select(w0, wt, first, n0);
    if (blockIdx.y * MM_BN >= p.n)
    {
        return;
    }
    mm_block<false>(p.w, x + (size_t)p.f0 * cols, y + (size_t)p.f0 * rows, rows, cols, p.n, blockIdx.x * MM_BM,
                    blockIdx.y * MM_BN);
}

// A few inputs (decode): one warp per row. Each lane takes half of a Q8_0
// block at a time, so a warp reads 16 blocks per step. Up to MV_TOK inputs.
#define MV_TOK 8

template <bool GATEUP>
__device__ __forceinline__ void mv_row(const unsigned char *__restrict__ w, const unsigned char *__restrict__ w2,
                                       const float *__restrict__ x, float *__restrict__ y, int rows, int cols, int n,
                                       int row)
{
    const int lane = threadIdx.x & 31;
    const int nb = cols >> 5;
    const int half = lane & 1;
    const size_t row_off = (size_t)row * nb * 34;
    float acc[MV_TOK], acc2[MV_TOK];
#pragma unroll
    for (int t = 0; t < MV_TOK; t++)
    {
        acc[t] = 0.0f;
        acc2[t] = 0.0f;
    }
#pragma unroll 2
    for (int b = lane >> 1; b < nb; b += 16)
    {
        const int c = b * 32 + half * 16;
        float v[16], v2[16];
        float d, d2 = 0.0f;
        {
            const unsigned char *blk = w + row_off + (size_t)b * 34;
            d = __half2float(__ushort_as_half(*(const unsigned short *)blk));
            const unsigned short *qp = (const unsigned short *)(blk + 2 + half * 16);
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const unsigned short u = qp[j];
                v[2 * j] = (float)(signed char)(u & 0xff);
                v[2 * j + 1] = (float)(signed char)(u >> 8);
            }
        }
        if (GATEUP)
        {
            const unsigned char *blk = w2 + row_off + (size_t)b * 34;
            d2 = __half2float(__ushort_as_half(*(const unsigned short *)blk));
            const unsigned short *qp = (const unsigned short *)(blk + 2 + half * 16);
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                const unsigned short u = qp[j];
                v2[2 * j] = (float)(signed char)(u & 0xff);
                v2[2 * j + 1] = (float)(signed char)(u >> 8);
            }
        }
#pragma unroll
        for (int t = 0; t < MV_TOK; t++)
        {
            if (t >= n)
            {
                break;
            }
            const float4 *xp = (const float4 *)(x + (size_t)t * cols + c);
            float s = 0.0f, s2 = 0.0f;
#pragma unroll
            for (int q4 = 0; q4 < 4; q4++)
            {
                const float4 xv = xp[q4];
                s += v[4 * q4] * xv.x + v[4 * q4 + 1] * xv.y + v[4 * q4 + 2] * xv.z + v[4 * q4 + 3] * xv.w;
                if (GATEUP)
                {
                    s2 += v2[4 * q4] * xv.x + v2[4 * q4 + 1] * xv.y + v2[4 * q4 + 2] * xv.z + v2[4 * q4 + 3] * xv.w;
                }
            }
            acc[t] += d * s;
            if (GATEUP)
            {
                acc2[t] += d2 * s2;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MV_TOK; t++)
    {
        if (t >= n)
        {
            break;
        }
        float s = acc[t], s2 = acc2[t];
        for (int off = 16; off > 0; off >>= 1)
        {
            s += __shfl_xor_sync(0xffffffffu, s, off);
            if (GATEUP)
            {
                s2 += __shfl_xor_sync(0xffffffffu, s2, off);
            }
        }
        if (lane == 0)
        {
            y[(size_t)t * rows + row] = GATEUP ? s / (1.0f + expf(-s)) * s2 : s;
        }
    }
}

// y = W x for up to MV_TOK inputs per matrix, W in Q8_0. Grid: rows / 8, 1,
// one z per matrix of a group.
extern "C" __global__ void __launch_bounds__(256) tf_mv_q8(unsigned long long w0, const unsigned long long *wt,
                                                           const int *first, const float *x, float *y, int rows,
                                                           int cols, int n0)
{
    const mm_pick p = mm_select(w0, wt, first, n0);
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows || p.n <= 0)
    {
        return;
    }
    mv_row<false>(p.w, nullptr, x + (size_t)p.f0 * cols, y + (size_t)p.f0 * rows, rows, cols, p.n, row);
}

// An expert's silu(gate x) * up x in one pass over both weights, for up to
// MV_TOK inputs per expert of the group.
extern "C" __global__ void __launch_bounds__(256) tf_mv_q8_gateup(const unsigned long long *gt,
                                                                  const unsigned long long *ut, const int *first,
                                                                  const float *x, float *y, int rows, int cols)
{
    const int z = blockIdx.z;
    const int f0 = first[z], n = first[z + 1] - f0;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows || n <= 0)
    {
        return;
    }
    mv_row<true>((const unsigned char *)gt[z], (const unsigned char *)ut[z], x + (size_t)f0 * cols,
                 y + (size_t)f0 * rows, rows, cols, n, row);
}

// out[i] = x[src[i]], rows of d floats: an expert batch's inputs, with each row's tf_rowscale.
extern "C" __global__ void tf_gather(const float *__restrict__ x, const int *__restrict__ src, float *__restrict__ out,
                                     int d, float *__restrict__ up)
{
    const int i = blockIdx.x;
    const float4 *from = (const float4 *)(x + (size_t)src[i] * d);
    float4 *to = (float4 *)(out + (size_t)i * d);
    float mx = 0.0f;
    for (int c = threadIdx.x; c < d / 4; c += blockDim.x)
    {
        const float4 v = from[c];
        to[c] = v;
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    mx = block_max(mx);
    if (threadIdx.x == 0)
    {
        up[i] = row_up(mx);
    }
}

// out[t] = sum over j < k of w[t k + j] * yb[inv[t k + j]], in order of j:
// each token's expert outputs, weighted by the router.
extern "C" __global__ void tf_combine(const float *__restrict__ yb, const int *__restrict__ inv,
                                      const float *__restrict__ w, float *__restrict__ out, int k, int d)
{
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
    {
        float s = 0.0f;
        for (int j = 0; j < k; j++)
        {
            s += w[t * k + j] * yb[(size_t)inv[t * k + j] * d + c];
        }
        out[(size_t)t * d + c] = s;
    }
}

// Causal attention over a KV cache (TASK-12), head_dim 128. Must match tf_gpu.c.
#define ATT_HD 128
#define ATT_QT 32 // query rows per block: (token, query head) pairs sharing one KV head
#define ATT_PT 32 // cache positions per tile

// How the cache stores each position (TASK-27); must match src/kv_format.zig.
// FP8 rows end with one scale byte per KV head, padded to 16.
#define KV_F32 0
#define KV_BF16 1
#define KV_FP8 2
#define KV_BF16_FP8 3 // keys as bf16, values as fp8

// The format of the keys and of the values in a cache of format fmt.
__device__ __forceinline__ constexpr int kv_kf(int fmt)
{
    return fmt == KV_BF16_FP8 ? KV_BF16 : fmt;
}

__device__ __forceinline__ constexpr int kv_vf(int fmt)
{
    return fmt == KV_BF16_FP8 ? KV_FP8 : fmt;
}

__device__ __forceinline__ int kv_row_bytes(int fmt, int kvd, int hd = ATT_HD)
{
    return fmt == KV_F32 ? kvd * 4 : fmt == KV_BF16 ? kvd * 2 : kvd + ((kvd / hd + 15) & ~15);
}

// KV head h of position p in a tile: where its values start, and its scale.
struct KvHead
{
    const unsigned char *p;
    float scale;
};

__device__ __forceinline__ KvHead kv_head(unsigned long long tile, int p, int h, int fmt, int kvd, int hd = ATT_HD)
{
    const unsigned char *row = (const unsigned char *)tile + (size_t)p * kv_row_bytes(fmt, kvd, hd);
    KvHead r;
    r.p = row + (size_t)h * hd * (fmt == KV_F32 ? 4 : fmt == KV_BF16 ? 2 : 1);
    // Scale byte 0 is never written, so a row past the end (zeroed memory) decodes to zeros, not NaN.
    const unsigned sb = fmt == KV_FP8 ? row[kvd + h] : 0;
    r.scale = fmt != KV_FP8 ? 1.0f : sb ? __int_as_float((int)(sb - 1) << 23) : 0.0f;
    return r;
}

__device__ __forceinline__ float kv_e4m3(unsigned b)
{
    const unsigned e = (b >> 3) & 15, m = b & 7;
    const float mag = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;
    return (b & 0x80) ? -mag : mag;
}

__device__ __forceinline__ float4 kv_get4(const KvHead &r, int d4, int fmt)
{
    if (fmt == KV_F32)
    {
        return ((const float4 *)r.p)[d4];
    }
    if (fmt == KV_BF16)
    {
        const uint2 u = ((const uint2 *)r.p)[d4];
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                           __uint_as_float(u.y & 0xffff0000u));
    }
    const unsigned u = ((const unsigned *)r.p)[d4];
    return make_float4(kv_e4m3(u & 255) * r.scale, kv_e4m3((u >> 8) & 255) * r.scale, kv_e4m3((u >> 16) & 255) * r.scale,
                       kv_e4m3(u >> 24) * r.scale);
}

__device__ __forceinline__ float kv_get1(const KvHead &r, int d, int fmt)
{
    if (fmt == KV_F32)
    {
        return ((const float *)r.p)[d];
    }
    if (fmt == KV_BF16)
    {
        return __uint_as_float((unsigned)((const unsigned short *)r.p)[d] << 16);
    }
    return kv_e4m3(r.p[d]) * r.scale;
}

// One block takes ATT_QT query rows of KV head blockIdx.y and the tiles of
// split blockIdx.z, with an online softmax, and writes a partial result
// (running max, sum, unnormalized output) that tf_attention_merge combines.
// Tile i's keys start at tiles_k[i]: position p of the tile, head h, is at
// tiles_k[i] + p * kvd + h * 128 floats (the same for values). Rows are
// t * group + g for token t and query head h * group + g; token t sits at
// position pos0 + t and sees positions 0 to pos0 + t.
template <int fmt>
__device__ __forceinline__ void att_first(const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,
                                        const unsigned long long *__restrict__ tiles_v, float *__restrict__ part_m,
                                        float *__restrict__ part_l, float *__restrict__ part_acc, int n, int pos0,
                                        int heads, int kv_heads, int kvd, int tiles_per_split, float scale)
{
    __shared__ float qs[ATT_QT][ATT_HD];
    __shared__ float ks[ATT_PT][ATT_HD]; // also holds the tile's probabilities once scores are done
    __shared__ float vs[ATT_PT][ATT_HD];
    float(*ps)[ATT_PT] = (float(*)[ATT_PT])ks;

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int j0 = blockIdx.x * ATT_QT;
    const int tid = threadIdx.x;
    const int r = tid >> 3; // this thread's query row in the block
    const int sub = tid & 7; // 8 threads per row: 4 positions and 16 output values each
    const int lane = tid & 31;
    const int j = j0 + r;
    const int limit = j < rows ? pos0 + j / group : -1;

    for (int i = tid; i < ATT_QT * ATT_HD; i += blockDim.x) {
        const int jr = j0 + i / ATT_HD, d = i % ATT_HD;
        float v = 0.0f;
        if (jr < rows)
            v = q[((size_t)(jr / group) * heads + h * group + jr % group) * ATT_HD + d];
        qs[i / ATT_HD][d] = v;
    }

    const int last_row = min(j0 + ATT_QT, rows) - 1;
    const int block_limit = pos0 + last_row / group;
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int k = 0; k < 16; k++)
        acc[k] = 0.0f;

    for (int tile = first_tile; tile < end_tile; tile++) {
        __syncthreads();
        for (int i = tid; i < ATT_PT * ATT_HD; i += blockDim.x) {
            const int p = i / ATT_HD, d = i % ATT_HD;
            const bool ok = tile * ATT_PT + p <= block_limit;
            ks[p][d] = ok ? kv_get1(kv_head(tiles_k[tile], p, h, kv_kf(fmt), kvd), d, kv_kf(fmt)) : 0.0f;
            vs[p][d] = ok ? kv_get1(kv_head(tiles_v[tile], p, h, kv_vf(fmt), kvd), d, kv_vf(fmt)) : 0.0f;
        }
        __syncthreads();
        float s[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int p = sub * 4 + i;
            float dot = 0.0f;
            for (int d0 = 0; d0 < ATT_HD; d0++) {
                const int d = (d0 + lane) & (ATT_HD - 1);
                dot += qs[r][d] * ks[p][d];
            }
            s[i] = tile * ATT_PT + p <= limit ? dot * scale : -INFINITY;
        }
        float mx = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
        for (int off = 4; off > 0; off >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        const float m_new = fmaxf(m, mx);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            s[i] = m_new == -INFINITY ? 0.0f : expf(s[i] - m_new);
            sum += s[i];
        }
        for (int off = 4; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        const float alpha = m == -INFINITY ? 0.0f : expf(m - m_new);
        l = l * alpha + sum;
        m = m_new;
#pragma unroll
        for (int k = 0; k < 16; k++)
            acc[k] *= alpha;
        __syncthreads(); // every row is done reading keys
#pragma unroll
        for (int i = 0; i < 4; i++)
            ps[r][sub * 4 + i] = s[i];
        __syncthreads();
        for (int p = 0; p < ATT_PT; p++) {
            const float w = ps[r][p];
#pragma unroll
            for (int k = 0; k < 16; k++)
                acc[k] += w * vs[p][sub * 16 + k];
        }
    }
    if (j >= rows)
        return;
    const size_t at = ((size_t)split * kv_heads + h) * rows + j;
    if (sub == 0) {
        part_m[at] = m;
        part_l[at] = l;
    }
#pragma unroll
    for (int k = 0; k < 16; k++)
        part_acc[at * ATT_HD + sub * 16 + k] = acc[k];
}

// The same attention for many query rows (prefill), with 64-row by
// 64-position tiles and register blocking: each of 256 threads keeps 4 rows x
// 4 positions of scores and 4 rows x 8 values of output. Thread tx takes
// positions tx + 16 j and values 4 tx + 64 j, so 128-bit shared loads from
// neighbouring lanes hit different banks (rows padded to 132 floats).
// Dynamic shared memory: ATT_W_SMEM bytes.
#define ATT_W_Q 64
#define ATT_W_P 64
#define ATT_W_LD 132
#define ATT_W_SMEM ((2 * ATT_W_Q * ATT_W_LD + ATT_W_P * ATT_HD) * 4)

template <int fmt>
__device__ __forceinline__ void att_wide(
    const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,
    const unsigned long long *__restrict__ tiles_v, float *__restrict__ part_m, float *__restrict__ part_l,
    float *__restrict__ part_acc, int n, int pos0, int heads, int kv_heads, int kvd, int tiles_per_split, float scale)
{
    extern __shared__ float4 smem4[];
    float *smem = (float *)smem4;
    float(*qs)[ATT_W_LD] = (float(*)[ATT_W_LD])smem;
    float(*ks)[ATT_W_LD] = (float(*)[ATT_W_LD])(smem + ATT_W_Q * ATT_W_LD);
    float(*vs)[ATT_HD] = (float(*)[ATT_HD])(smem + 2 * ATT_W_Q * ATT_W_LD);
    float(*ps)[ATT_W_LD] = ks; // probabilities replace keys once scores are done

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int j0 = blockIdx.x * ATT_W_Q;
    const int tid = threadIdx.x;
    const int ty = tid >> 4; // rows 4 ty to 4 ty + 3
    const int tx = tid & 15;

    for (int i = tid; i < ATT_W_Q * (ATT_HD / 4); i += 256) {
        const int r = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4);
        const int jr = j0 + r;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (jr < rows)
            v = ((const float4 *)(q + ((size_t)(jr / group) * heads + h * group + jr % group) * ATT_HD))[d4];
        *(float4 *)&qs[r][d4 * 4] = v;
    }

    int limit[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int jr = j0 + ty * 4 + i;
        limit[i] = jr < rows ? pos0 + jr / group : -1;
    }
    const int last_row = min(j0 + ATT_W_Q, rows) - 1;
    const int block_limit = pos0 + last_row / group;
    // Tiles are 32 positions; this kernel takes them two at a time.
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m[4], l[4], acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        m[i] = -INFINITY;
        l[i] = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++)
            acc[i][k] = 0.f;
    }

    for (int tile = first_tile; tile < end_tile; tile += 2) {
        __syncthreads();
        for (int i = tid; i < ATT_W_P * (ATT_HD / 4); i += 256) {
            const int p = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4);
            const int t = tile + p / ATT_PT;
            const int pos = t * ATT_PT + p % ATT_PT;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (t < end_tile && pos <= block_limit) {
                kv = kv_get4(kv_head(tiles_k[t], p % ATT_PT, h, kv_kf(fmt), kvd), d4, kv_kf(fmt));
                vv = kv_get4(kv_head(tiles_v[t], p % ATT_PT, h, kv_vf(fmt), kvd), d4, kv_vf(fmt));
            }
            *(float4 *)&ks[p][d4 * 4] = kv;
            *(float4 *)&vs[p][d4 * 4] = vv;
        }
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                s[i][j] = 0.f;
#pragma unroll 4
        for (int d = 0; d < ATT_HD; d += 4) {
            float4 qa[4], kb[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
                qa[i] = *(const float4 *)&qs[ty * 4 + i][d];
#pragma unroll
            for (int j = 0; j < 4; j++)
                kb[j] = *(const float4 *)&ks[tx + 16 * j][d];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    s[i][j] += qa[i].x * kb[j].x + qa[i].y * kb[j].y + qa[i].z * kb[j].z + qa[i].w * kb[j].w;
        }

#pragma unroll
        for (int i = 0; i < 4; i++) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int pos = tile * ATT_PT + tx + 16 * j;
                s[i][j] = pos <= limit[i] && tile + (tx + 16 * j) / ATT_PT < end_tile ? s[i][j] * scale : -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
            for (int off = 8; off > 0; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float m_new = fmaxf(m[i], mx);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s[i][j] = m_new == -INFINITY ? 0.f : expf(s[i][j] - m_new);
                sum += s[i][j];
            }
            for (int off = 8; off > 0; off >>= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            const float alpha = m[i] == -INFINITY ? 0.f : expf(m[i] - m_new);
            l[i] = l[i] * alpha + sum;
            m[i] = m_new;
#pragma unroll
            for (int k = 0; k < 8; k++)
                acc[i][k] *= alpha;
        }
        __syncthreads(); // every row is done reading keys
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                ps[ty * 4 + i][tx + 16 * j] = s[i][j];
        __syncthreads();
#pragma unroll 4
        for (int p = 0; p < ATT_W_P; p++) {
            const float4 va = *(const float4 *)&vs[p][tx * 4];
            const float4 vb = *(const float4 *)&vs[p][64 + tx * 4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float w = ps[ty * 4 + i][p];
                acc[i][0] += w * va.x;
                acc[i][1] += w * va.y;
                acc[i][2] += w * va.z;
                acc[i][3] += w * va.w;
                acc[i][4] += w * vb.x;
                acc[i][5] += w * vb.y;
                acc[i][6] += w * vb.z;
                acc[i][7] += w * vb.w;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int jr = j0 + ty * 4 + i;
        if (jr >= rows)
            continue;
        const size_t at = ((size_t)split * kv_heads + h) * rows + jr;
        if (tx == 0) {
            part_m[at] = m[i];
            part_l[at] = l[i];
        }
        float *dst = part_acc + at * ATT_HD;
        *(float4 *)&dst[tx * 4] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *(float4 *)&dst[64 + tx * 4] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

// Combines the splits' partial results into each query head's output, in
// split order: out[t][head][d], one block of HD threads per (row, KV head).
template <int HD>
__device__ __forceinline__ void att_merge(const float *__restrict__ part_m, const float *__restrict__ part_l,
                                          const float *__restrict__ part_acc, float *__restrict__ out, int n, int heads,
                                          int kv_heads, int splits)
{
    const int j = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int group = heads / kv_heads;
    const int rows = n * group;
    float m = -INFINITY;
    for (int s = 0; s < splits; s++)
        m = fmaxf(m, part_m[((size_t)s * kv_heads + h) * rows + j]);
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < splits; s++) {
        const size_t at = ((size_t)s * kv_heads + h) * rows + j;
        const float ms = part_m[at];
        if (ms == -INFINITY)
            continue;
        const float f = expf(ms - m);
        l += part_l[at] * f;
        acc += part_acc[at * HD + d] * f;
    }
    out[((size_t)(j / group) * heads + h * group + j % group) * HD + d] = acc / l;
}

extern "C" __global__ void tf_attention_merge(const float *__restrict__ part_m, const float *__restrict__ part_l,
                                              const float *__restrict__ part_acc, float *__restrict__ out, int n,
                                              int heads, int kv_heads, int splits)
{
    att_merge<ATT_HD>(part_m, part_l, part_acc, out, n, heads, kv_heads, splits);
}

extern "C" __global__ void tf_attention_merge256(const float *__restrict__ part_m, const float *__restrict__ part_l,
                                                 const float *__restrict__ part_acc, float *__restrict__ out, int n,
                                                 int heads, int kv_heads, int splits)
{
    att_merge<256>(part_m, part_l, part_acc, out, n, heads, kv_heads, splits);
}

// Attention on tensor cores for prefill. Each f32 input is split into high and
// low f16 parts, and three mma products per step keep close to f32 accuracy.
#define TC_P 32
#define TC_LD (ATT_HD + 8)

__device__ __forceinline__ void tc_mma(float c[4], const unsigned a[4], const unsigned b[2])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The high and low f16 parts of two floats, packed as two half2 words.
__device__ __forceinline__ void tc_split(float x, float y, unsigned &hi, unsigned &lo)
{
    const __half2 h = __floats2half2_rn(x, y);
    const float2 back = __half22float2(h);
    const __half2 l = __floats2half2_rn(x - back.x, y - back.y);
    hi = *(const unsigned *)&h;
    lo = *(const unsigned *)&l;
}

__device__ __forceinline__ void tc_ldm_trans(unsigned b[2], const __half *p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[0]), "=r"(b[1]) : "r"(a));
}

template <int TC_Q, int fmt>
__device__ __forceinline__ void tc_attention(const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,
                                             const unsigned long long *__restrict__ tiles_v, float *__restrict__ part_m,
                                             float *__restrict__ part_l, float *__restrict__ part_acc, int n, int pos0,
                                             int heads, int kv_heads, int kvd, int tiles_per_split, float scale)
{
    constexpr int TC_THREADS = TC_Q * 2;
    __shared__ __align__(16) __half kh[TC_P][TC_LD];
    __shared__ __align__(16) __half kl[TC_P][TC_LD];
    __shared__ __align__(16) __half vh[TC_P][TC_LD];
    __shared__ __align__(16) __half vl[TC_P][TC_LD];

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int j0 = blockIdx.x * TC_Q;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int r_lo = j0 + warp * 16 + g, r_hi = r_lo + 8;

    unsigned qh[8][4], ql[8][4];
    {
        const float *q_lo = r_lo < rows ? q + ((size_t)(r_lo / group) * heads + h * group + r_lo % group) * ATT_HD : nullptr;
        const float *q_hi = r_hi < rows ? q + ((size_t)(r_hi / group) * heads + h * group + r_hi % group) * ATT_HD : nullptr;
#pragma unroll
        for (int kk = 0; kk < 8; kk++)
        {
            const int d = kk * 16 + c * 2;
            float2 a = make_float2(0.0f, 0.0f), b = a, e = a, f = a;
            if (q_lo)
            {
                a = *(const float2 *)(q_lo + d);
                e = *(const float2 *)(q_lo + d + 8);
            }
            if (q_hi)
            {
                b = *(const float2 *)(q_hi + d);
                f = *(const float2 *)(q_hi + d + 8);
            }
            tc_split(a.x, a.y, qh[kk][0], ql[kk][0]);
            tc_split(b.x, b.y, qh[kk][1], ql[kk][1]);
            tc_split(e.x, e.y, qh[kk][2], ql[kk][2]);
            tc_split(f.x, f.y, qh[kk][3], ql[kk][3]);
        }
    }
    const int limit_lo = r_lo < rows ? pos0 + r_lo / group : -1;
    const int limit_hi = r_hi < rows ? pos0 + r_hi / group : -1;
    const int last_row = min(j0 + TC_Q, rows) - 1;
    const int block_limit = pos0 + last_row / group;
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
#pragma unroll
        for (int e = 0; e < 4; e++)
        {
            o[i][e] = 0.0f;
        }
    }

    for (int tile = first_tile; tile < end_tile; tile++)
    {
        __syncthreads();
        for (int i = tid; i < TC_P * (ATT_HD / 4); i += TC_THREADS)
        {
            const int p = i / (ATT_HD / 4), d4 = i % (ATT_HD / 4);
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vv = kv;
            if (tile * ATT_PT + p <= block_limit)
            {
                kv = kv_get4(kv_head(tiles_k[tile], p, h, kv_kf(fmt), kvd), d4, kv_kf(fmt));
                vv = kv_get4(kv_head(tiles_v[tile], p, h, kv_vf(fmt), kvd), d4, kv_vf(fmt));
            }
            unsigned a0, a1, b0, b1;
            tc_split(kv.x, kv.y, a0, b0);
            tc_split(kv.z, kv.w, a1, b1);
            *(uint2 *)&kh[p][d4 * 4] = make_uint2(a0, a1);
            *(uint2 *)&kl[p][d4 * 4] = make_uint2(b0, b1);
            tc_split(vv.x, vv.y, a0, b0);
            tc_split(vv.z, vv.w, a1, b1);
            *(uint2 *)&vh[p][d4 * 4] = make_uint2(a0, a1);
            *(uint2 *)&vl[p][d4 * 4] = make_uint2(b0, b1);
        }
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            s[nt][0] = 0.0f;
            s[nt][1] = 0.0f;
            s[nt][2] = 0.0f;
            s[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 8; kk++)
            {
                unsigned bh[2], bl[2];
                bh[0] = *(const unsigned *)&kh[nt * 8 + g][kk * 16 + c * 2];
                bh[1] = *(const unsigned *)&kh[nt * 8 + g][kk * 16 + c * 2 + 8];
                bl[0] = *(const unsigned *)&kl[nt * 8 + g][kk * 16 + c * 2];
                bl[1] = *(const unsigned *)&kl[nt * 8 + g][kk * 16 + c * 2 + 8];
                tc_mma(s[nt], ql[kk], bh);
                tc_mma(s[nt], qh[kk], bl);
                tc_mma(s[nt], qh[kk], bh);
            }
        }

        float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                const int pos = tile * ATT_PT + nt * 8 + c * 2 + e;
                s[nt][e] = pos <= limit_lo ? s[nt][e] * scale : -INFINITY;
                s[nt][2 + e] = pos <= limit_hi ? s[nt][2 + e] * scale : -INFINITY;
                mx_lo = fmaxf(mx_lo, s[nt][e]);
                mx_hi = fmaxf(mx_hi, s[nt][2 + e]);
            }
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
        }
        const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                s[nt][e] = mn_lo == -INFINITY ? 0.0f : expf(s[nt][e] - mn_lo);
                s[nt][2 + e] = mn_hi == -INFINITY ? 0.0f : expf(s[nt][2 + e] - mn_hi);
                sum_lo += s[nt][e];
                sum_hi += s[nt][2 + e];
            }
        }
        const float al_lo = m_lo == -INFINITY ? 0.0f : expf(m_lo - mn_lo);
        const float al_hi = m_hi == -INFINITY ? 0.0f : expf(m_hi - mn_hi);
        l_lo = l_lo * al_lo + sum_lo;
        l_hi = l_hi * al_hi + sum_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            o[i][0] *= al_lo;
            o[i][1] *= al_lo;
            o[i][2] *= al_hi;
            o[i][3] *= al_hi;
        }

#pragma unroll
        for (int kk = 0; kk < 2; kk++)
        {
            unsigned ph[4], pl[4];
            tc_split(s[2 * kk][0], s[2 * kk][1], ph[0], pl[0]);
            tc_split(s[2 * kk][2], s[2 * kk][3], ph[1], pl[1]);
            tc_split(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[2], pl[2]);
            tc_split(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[3], pl[3]);
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                unsigned bh[2], bl[2];
                tc_ldm_trans(bh, &vh[kk * 16 + (lane & 15)][nt * 8]);
                tc_ldm_trans(bl, &vl[kk * 16 + (lane & 15)][nt * 8]);
                tc_mma(o[nt], pl, bh);
                tc_mma(o[nt], ph, bl);
                tc_mma(o[nt], ph, bh);
            }
        }
    }

    // Each lane summed a quarter of its rows: add up the 4 lanes of a row.
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l_lo += __shfl_xor_sync(0xffffffffu, l_lo, off);
        l_hi += __shfl_xor_sync(0xffffffffu, l_hi, off);
    }
    if (r_lo < rows)
    {
        const size_t at = ((size_t)split * kv_heads + h) * rows + r_lo;
        if (c == 0)
        {
            part_m[at] = m_lo;
            part_l[at] = l_lo;
        }
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            *(float2 *)&part_acc[at * ATT_HD + nt * 8 + c * 2] = make_float2(o[nt][0], o[nt][1]);
        }
    }
    if (r_hi < rows)
    {
        const size_t at = ((size_t)split * kv_heads + h) * rows + r_hi;
        if (c == 0)
        {
            part_m[at] = m_hi;
            part_l[at] = l_hi;
        }
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            *(float2 *)&part_acc[at * ATT_HD + nt * 8 + c * 2] = make_float2(o[nt][2], o[nt][3]);
        }
    }
}



// Attention for at most R query rows per KV head (decode): each lane scores one
// cached position, and the 4 warps merge in order at the end.
#define DEC_WARPS 4

template <int R, int fmt>
__device__ __forceinline__ void dec_attention(const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,
                                              const unsigned long long *__restrict__ tiles_v, float *__restrict__ part_m,
                                              float *__restrict__ part_l, float *__restrict__ part_acc, int n, int pos0,
                                              int heads, int kv_heads, int kvd, int tiles_per_split, float scale)
{
    __shared__ __align__(16) float qs[R][ATT_HD];
    __shared__ float wm[DEC_WARPS][R];
    __shared__ float wl[DEC_WARPS][R];
    __shared__ __align__(16) float wacc[DEC_WARPS][R][ATT_HD];

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    for (int i = tid; i < R * ATT_HD; i += 32 * DEC_WARPS)
    {
        const int r = i / ATT_HD, d = i % ATT_HD;
        qs[r][d] = r < rows ? q[((size_t)(r / group) * heads + h * group + r % group) * ATT_HD + d] : 0.0f;
    }
    __syncthreads();

    const int block_limit = pos0 + (rows - 1) / group;
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m[R], l[R], acc[R][4];
#pragma unroll
    for (int r = 0; r < R; r++)
    {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    for (int tile = first_tile + warp; tile < end_tile; tile += DEC_WARPS)
    {
        const int pos = tile * ATT_PT + lane;
        float s[R];
#pragma unroll
        for (int r = 0; r < R; r++)
        {
            s[r] = 0.0f;
        }
        const KvHead kr = kv_head(tiles_k[tile], lane, h, kv_kf(fmt), kvd);
#pragma unroll 4
        for (int d4 = 0; d4 < ATT_HD / 4; d4++)
        {
            const float4 kv = kv_get4(kr, d4, kv_kf(fmt));
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                const float4 qv = *(const float4 *)&qs[r][d4 * 4];
                s[r] += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++)
        {
            const bool ok = r < rows && pos <= pos0 + r / group;
            s[r] = ok ? s[r] * scale : -INFINITY;
            float mx = s[r];
            for (int off = 16; off > 0; off >>= 1)
            {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            }
            const float mn = fmaxf(m[r], mx);
            s[r] = mn == -INFINITY ? 0.0f : expf(s[r] - mn);
            float sum = s[r];
            for (int off = 16; off > 0; off >>= 1)
            {
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            }
            const float alpha = m[r] == -INFINITY ? 0.0f : expf(m[r] - mn);
            l[r] = l[r] * alpha + sum;
            m[r] = mn;
            acc[r][0] *= alpha;
            acc[r][1] *= alpha;
            acc[r][2] *= alpha;
            acc[r][3] *= alpha;
        }
        // Rows past the last position may hold anything; a zero weight times an infinity there would be NaN.
        const int valid = min(ATT_PT, block_limit - tile * ATT_PT + 1);
#pragma unroll 4
        for (int p = 0; p < valid; p++)
        {
            const float4 vv = kv_get4(kv_head(tiles_v[tile], p, h, kv_vf(fmt), kvd), lane, kv_vf(fmt));
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                const float w = __shfl_sync(0xffffffffu, s[r], p);
                acc[r][0] += w * vv.x;
                acc[r][1] += w * vv.y;
                acc[r][2] += w * vv.z;
                acc[r][3] += w * vv.w;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++)
    {
        if (lane == 0)
        {
            wm[warp][r] = m[r];
            wl[warp][r] = l[r];
        }
        *(float4 *)&wacc[warp][r][lane * 4] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    }
    __syncthreads();
    for (int r = 0; r < rows && r < R; r++)
    {
        float mm = -INFINITY;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; w++)
        {
            mm = fmaxf(mm, wm[w][r]);
        }
        float ll = 0.0f, aa = 0.0f;
#pragma unroll
        for (int w = 0; w < DEC_WARPS; w++)
        {
            if (wm[w][r] == -INFINITY)
            {
                continue;
            }
            const float f = expf(wm[w][r] - mm);
            ll += wl[w][r] * f;
            aa += wacc[w][r][tid] * f;
        }
        const size_t at = ((size_t)split * kv_heads + h) * rows + r;
        if (tid == 0)
        {
            part_m[at] = mm;
            part_l[at] = ll;
        }
        part_acc[at * ATT_HD + tid] = aa;
    }
}



// Entry points, one set per KV cache format, so each is compiled with its loads fixed
// (the f32 ones keep their names). Must match tf_gpu.c.
#define ATT_ARGS                                                                                                        \
    const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,                                         \
        const unsigned long long *__restrict__ tiles_v, float *__restrict__ part_m, float *__restrict__ part_l,          \
        float *__restrict__ part_acc, int n, int pos0, int heads, int kv_heads, int kvd, int tiles_per_split, float scale
#define ATT_PASS q, tiles_k, tiles_v, part_m, part_l, part_acc, n, pos0, heads, kv_heads, kvd, tiles_per_split, scale
#define ATT_KERNELS(F, SUFFIX)                                                                                          \
    extern "C" __global__ void tf_attention##SUFFIX(ATT_ARGS)                                                           \
    {                                                                                                                   \
        att_first<F>(ATT_PASS);                                                                                         \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) tf_attention_wide##SUFFIX(ATT_ARGS)                               \
    {                                                                                                                   \
        att_wide<F>(ATT_PASS);                                                                                          \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_attention_tc##SUFFIX(ATT_ARGS)                                 \
    {                                                                                                                   \
        tc_attention<64, F>(ATT_PASS);                                                                                  \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) tf_attention_tc128##SUFFIX(ATT_ARGS)                              \
    {                                                                                                                   \
        tc_attention<128, F>(ATT_PASS);                                                                                 \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_attention_decode8##SUFFIX(ATT_ARGS)                            \
    {                                                                                                                   \
        dec_attention<8, F>(ATT_PASS);                                                                                  \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) tf_attention_decode16##SUFFIX(ATT_ARGS)                           \
    {                                                                                                                   \
        dec_attention<16, F>(ATT_PASS);                                                                                 \
    }

ATT_KERNELS(KV_F32, )
ATT_KERNELS(KV_BF16, _bf16)
ATT_KERNELS(KV_FP8, _fp8)
ATT_KERNELS(KV_BF16_FP8, _bf16_fp8)

// Attention with 256-dim heads (TASK-29, Qwen3.5), for any number of rows:
// att_first's scheme with 16 query rows per block, 16 threads per row (2
// positions and 16 output values each, values strided so a row's threads hit
// different banks). Same partial results as the 128-dim kernels, merged by
// tf_attention_merge256. Dynamic shared memory: A2_SMEM bytes.
#define A2_HD 256
#define A2_QT 16
#define A2_SMEM ((A2_QT + 2 * ATT_PT) * A2_HD * 4)

template <int fmt>
__device__ __forceinline__ void att_256(ATT_ARGS)
{
    extern __shared__ __align__(16) float a2_smem[];
    float(*qs)[A2_HD] = (float(*)[A2_HD])a2_smem;
    float(*ks)[A2_HD] = (float(*)[A2_HD])(a2_smem + A2_QT * A2_HD);
    float(*vs)[A2_HD] = (float(*)[A2_HD])(a2_smem + (A2_QT + ATT_PT) * A2_HD);
    float(*ps)[ATT_PT] = (float(*)[ATT_PT])ks;

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int j0 = blockIdx.x * A2_QT;
    const int tid = threadIdx.x;
    const int r = tid >> 4;
    const int sub = tid & 15;
    const int lane = tid & 31;
    const int j = j0 + r;
    const int limit = j < rows ? pos0 + j / group : -1;

    for (int i = tid; i < A2_QT * (A2_HD / 4); i += blockDim.x)
    {
        const int jr = j0 + i / (A2_HD / 4), d4 = i % (A2_HD / 4);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (jr < rows)
        {
            v = ((const float4 *)(q + ((size_t)(jr / group) * heads + h * group + jr % group) * A2_HD))[d4];
        }
        *(float4 *)&qs[i / (A2_HD / 4)][d4 * 4] = v;
    }

    const int last_row = min(j0 + A2_QT, rows) - 1;
    const int block_limit = pos0 + last_row / group;
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m = -INFINITY, l = 0.0f, acc[16];
#pragma unroll
    for (int k = 0; k < 16; k++)
    {
        acc[k] = 0.0f;
    }
    for (int tile = first_tile; tile < end_tile; tile++)
    {
        __syncthreads();
        for (int i = tid; i < ATT_PT * (A2_HD / 4); i += blockDim.x)
        {
            const int p = i / (A2_HD / 4), d4 = i % (A2_HD / 4);
            const bool ok = tile * ATT_PT + p <= block_limit;
            const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            *(float4 *)&ks[p][d4 * 4] = ok ? kv_get4(kv_head(tiles_k[tile], p, h, kv_kf(fmt), kvd, A2_HD), d4, kv_kf(fmt)) : zero;
            *(float4 *)&vs[p][d4 * 4] = ok ? kv_get4(kv_head(tiles_v[tile], p, h, kv_vf(fmt), kvd, A2_HD), d4, kv_vf(fmt)) : zero;
        }
        __syncthreads();
        float s[2];
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            const int p = sub * 2 + i;
            float dot = 0.0f;
            for (int d0 = 0; d0 < A2_HD; d0++)
            {
                const int d = (d0 + lane) & (A2_HD - 1);
                dot += qs[r][d] * ks[p][d];
            }
            s[i] = tile * ATT_PT + p <= limit ? dot * scale : -INFINITY;
        }
        float mx = fmaxf(s[0], s[1]);
        for (int off = 8; off > 0; off >>= 1)
        {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        }
        const float m_new = fmaxf(m, mx);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            s[i] = m_new == -INFINITY ? 0.0f : expf(s[i] - m_new);
            sum += s[i];
        }
        for (int off = 8; off > 0; off >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        const float alpha = m == -INFINITY ? 0.0f : expf(m - m_new);
        l = l * alpha + sum;
        m = m_new;
#pragma unroll
        for (int k = 0; k < 16; k++)
        {
            acc[k] *= alpha;
        }
        __syncthreads();
        ps[r][sub * 2] = s[0];
        ps[r][sub * 2 + 1] = s[1];
        __syncthreads();
        for (int p = 0; p < ATT_PT; p++)
        {
            const float w = ps[r][p];
#pragma unroll
            for (int k = 0; k < 16; k++)
            {
                acc[k] += w * vs[p][sub + 16 * k];
            }
        }
    }
    if (j >= rows)
    {
        return;
    }
    const size_t at = ((size_t)split * kv_heads + h) * rows + j;
    if (sub == 0)
    {
        part_m[at] = m;
        part_l[at] = l;
    }
#pragma unroll
    for (int k = 0; k < 16; k++)
    {
        part_acc[at * A2_HD + sub + 16 * k] = acc[k];
    }
}

// tc_attention for 256-dim heads: 64 rows per block, two warps per 16 rows,
// each holding half the dimensions. The pair adds their partial scores through
// shared memory, so both take the same softmax and each keeps half the output.
// Dynamic shared memory: T2_SMEM bytes.
#define T2_Q 64
#define T2_LD (A2_HD + 8)
#define T2_SMEM (4 * TC_P * T2_LD * 2)

template <int fmt>
__device__ __forceinline__ void tc_attention256(ATT_ARGS)
{
    extern __shared__ __align__(16) __half t2_smem[];
    __half(*kh)[T2_LD] = (__half(*)[T2_LD])t2_smem;
    __half(*kl)[T2_LD] = (__half(*)[T2_LD])(t2_smem + TC_P * T2_LD);
    __half(*vh)[T2_LD] = (__half(*)[T2_LD])(t2_smem + 2 * TC_P * T2_LD);
    __half(*vl)[T2_LD] = (__half(*)[T2_LD])(t2_smem + 3 * TC_P * T2_LD);
    float4 *xs = (float4 *)t2_smem; // partial scores, over the keys once they are used

    const int h = blockIdx.y;
    const int split = blockIdx.z;
    const int group = heads / kv_heads;
    const int rows = n * group;
    const int j0 = blockIdx.x * T2_Q;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int half_d = (warp & 1) * 128;
    const int r_lo = j0 + (warp >> 1) * 16 + g, r_hi = r_lo + 8;

    unsigned qh[8][4], ql[8][4];
    {
        const float *q_lo = r_lo < rows ? q + ((size_t)(r_lo / group) * heads + h * group + r_lo % group) * A2_HD + half_d : nullptr;
        const float *q_hi = r_hi < rows ? q + ((size_t)(r_hi / group) * heads + h * group + r_hi % group) * A2_HD + half_d : nullptr;
#pragma unroll
        for (int kk = 0; kk < 8; kk++)
        {
            const int d = kk * 16 + c * 2;
            float2 a = make_float2(0.0f, 0.0f), b = a, e = a, f = a;
            if (q_lo)
            {
                a = *(const float2 *)(q_lo + d);
                e = *(const float2 *)(q_lo + d + 8);
            }
            if (q_hi)
            {
                b = *(const float2 *)(q_hi + d);
                f = *(const float2 *)(q_hi + d + 8);
            }
            tc_split(a.x, a.y, qh[kk][0], ql[kk][0]);
            tc_split(b.x, b.y, qh[kk][1], ql[kk][1]);
            tc_split(e.x, e.y, qh[kk][2], ql[kk][2]);
            tc_split(f.x, f.y, qh[kk][3], ql[kk][3]);
        }
    }
    const int limit_lo = r_lo < rows ? pos0 + r_lo / group : -1;
    const int limit_hi = r_hi < rows ? pos0 + r_hi / group : -1;
    const int last_row = min(j0 + T2_Q, rows) - 1;
    const int block_limit = pos0 + last_row / group;
    const int needed = block_limit / ATT_PT + 1;
    const int first_tile = split * tiles_per_split;
    const int end_tile = min(first_tile + tiles_per_split, needed);

    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++)
    {
#pragma unroll
        for (int e = 0; e < 4; e++)
        {
            o[i][e] = 0.0f;
        }
    }

    for (int tile = first_tile; tile < end_tile; tile++)
    {
        __syncthreads();
        for (int i = tid; i < TC_P * (A2_HD / 4); i += 256)
        {
            const int p = i / (A2_HD / 4), d4 = i % (A2_HD / 4);
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vv = kv;
            if (tile * ATT_PT + p <= block_limit)
            {
                kv = kv_get4(kv_head(tiles_k[tile], p, h, kv_kf(fmt), kvd, A2_HD), d4, kv_kf(fmt));
                vv = kv_get4(kv_head(tiles_v[tile], p, h, kv_vf(fmt), kvd, A2_HD), d4, kv_vf(fmt));
            }
            unsigned a0, a1, b0, b1;
            tc_split(kv.x, kv.y, a0, b0);
            tc_split(kv.z, kv.w, a1, b1);
            *(uint2 *)&kh[p][d4 * 4] = make_uint2(a0, a1);
            *(uint2 *)&kl[p][d4 * 4] = make_uint2(b0, b1);
            tc_split(vv.x, vv.y, a0, b0);
            tc_split(vv.z, vv.w, a1, b1);
            *(uint2 *)&vh[p][d4 * 4] = make_uint2(a0, a1);
            *(uint2 *)&vl[p][d4 * 4] = make_uint2(b0, b1);
        }
        __syncthreads();

        float s[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            s[nt][0] = 0.0f;
            s[nt][1] = 0.0f;
            s[nt][2] = 0.0f;
            s[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 8; kk++)
            {
                const int d = half_d + kk * 16 + c * 2;
                unsigned bh[2], bl[2];
                bh[0] = *(const unsigned *)&kh[nt * 8 + g][d];
                bh[1] = *(const unsigned *)&kh[nt * 8 + g][d + 8];
                bl[0] = *(const unsigned *)&kl[nt * 8 + g][d];
                bl[1] = *(const unsigned *)&kl[nt * 8 + g][d + 8];
                tc_mma(s[nt], ql[kk], bh);
                tc_mma(s[nt], qh[kk], bl);
                tc_mma(s[nt], qh[kk], bh);
            }
        }

        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            xs[(warp * 4 + nt) * 32 + lane] = make_float4(s[nt][0], s[nt][1], s[nt][2], s[nt][3]);
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
            const float4 other = xs[((warp ^ 1) * 4 + nt) * 32 + lane];
            s[nt][0] += other.x;
            s[nt][1] += other.y;
            s[nt][2] += other.z;
            s[nt][3] += other.w;
        }

        float mx_lo = -INFINITY, mx_hi = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                const int pos = tile * ATT_PT + nt * 8 + c * 2 + e;
                s[nt][e] = pos <= limit_lo ? s[nt][e] * scale : -INFINITY;
                s[nt][2 + e] = pos <= limit_hi ? s[nt][2 + e] * scale : -INFINITY;
                mx_lo = fmaxf(mx_lo, s[nt][e]);
                mx_hi = fmaxf(mx_hi, s[nt][2 + e]);
            }
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffffu, mx_lo, off));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffffu, mx_hi, off));
        }
        const float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
        {
#pragma unroll
            for (int e = 0; e < 2; e++)
            {
                s[nt][e] = mn_lo == -INFINITY ? 0.0f : expf(s[nt][e] - mn_lo);
                s[nt][2 + e] = mn_hi == -INFINITY ? 0.0f : expf(s[nt][2 + e] - mn_hi);
                sum_lo += s[nt][e];
                sum_hi += s[nt][2 + e];
            }
        }
        const float al_lo = m_lo == -INFINITY ? 0.0f : expf(m_lo - mn_lo);
        const float al_hi = m_hi == -INFINITY ? 0.0f : expf(m_hi - mn_hi);
        l_lo = l_lo * al_lo + sum_lo;
        l_hi = l_hi * al_hi + sum_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            o[i][0] *= al_lo;
            o[i][1] *= al_lo;
            o[i][2] *= al_hi;
            o[i][3] *= al_hi;
        }

#pragma unroll
        for (int kk = 0; kk < 2; kk++)
        {
            unsigned ph[4], pl[4];
            tc_split(s[2 * kk][0], s[2 * kk][1], ph[0], pl[0]);
            tc_split(s[2 * kk][2], s[2 * kk][3], ph[1], pl[1]);
            tc_split(s[2 * kk + 1][0], s[2 * kk + 1][1], ph[2], pl[2]);
            tc_split(s[2 * kk + 1][2], s[2 * kk + 1][3], ph[3], pl[3]);
#pragma unroll
            for (int nt = 0; nt < 16; nt++)
            {
                unsigned bh[2], bl[2];
                tc_ldm_trans(bh, &vh[kk * 16 + (lane & 15)][half_d + nt * 8]);
                tc_ldm_trans(bl, &vl[kk * 16 + (lane & 15)][half_d + nt * 8]);
                tc_mma(o[nt], pl, bh);
                tc_mma(o[nt], ph, bl);
                tc_mma(o[nt], ph, bh);
            }
        }
    }

    // Each lane summed a quarter of its rows: add up the 4 lanes of a row.
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l_lo += __shfl_xor_sync(0xffffffffu, l_lo, off);
        l_hi += __shfl_xor_sync(0xffffffffu, l_hi, off);
    }
    const bool first = c == 0 && half_d == 0;
    if (r_lo < rows)
    {
        const size_t at = ((size_t)split * kv_heads + h) * rows + r_lo;
        if (first)
        {
            part_m[at] = m_lo;
            part_l[at] = l_lo;
        }
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            *(float2 *)&part_acc[at * A2_HD + half_d + nt * 8 + c * 2] = make_float2(o[nt][0], o[nt][1]);
        }
    }
    if (r_hi < rows)
    {
        const size_t at = ((size_t)split * kv_heads + h) * rows + r_hi;
        if (first)
        {
            part_m[at] = m_hi;
            part_l[at] = l_hi;
        }
#pragma unroll
        for (int nt = 0; nt < 16; nt++)
        {
            *(float2 *)&part_acc[at * A2_HD + half_d + nt * 8 + c * 2] = make_float2(o[nt][2], o[nt][3]);
        }
    }
}

#define ATT_KERNELS_256(F, SUFFIX)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) tf_attention256##SUFFIX(ATT_ARGS)                                 \
    {                                                                                                                   \
        att_256<F>(ATT_PASS);                                                                                           \
    }                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) tf_attention_tc256##SUFFIX(ATT_ARGS)                              \
    {                                                                                                                   \
        tc_attention256<F>(ATT_PASS);                                                                                   \
    }

ATT_KERNELS_256(KV_F32, )
ATT_KERNELS_256(KV_BF16, _bf16)
ATT_KERNELS_256(KV_FP8, _fp8)
ATT_KERNELS_256(KV_BF16_FP8, _bf16_fp8)

// The gated delta rule of a Gated DeltaNet layer (TASK-29) over n tokens:
// one block per value head h, thread j holding column j of the head's DR_DK
// x dv state in registers, so each token's update needs no reduction across
// threads. Rows of `mixed` (cd floats per token) hold the L2-normalized,
// scaled queries, the keys, then the values; value head h reads key head
// h mod kh. `state` is read at the start and written back at the end.
#define DR_DK 128

extern "C" __global__ void __launch_bounds__(256) tf_delta_rule(const float *__restrict__ mixed,
                                                                const float *__restrict__ beta,
                                                                const float *__restrict__ decay, float *__restrict__ state,
                                                                float *__restrict__ core, int n, int cd, int kh, int vh,
                                                                int dv)
{
    __shared__ float qs[DR_DK];
    __shared__ float ks[DR_DK];
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    const int kq = h % kh;
    float *m = state + (size_t)h * DR_DK * dv;
    float st[DR_DK];
#pragma unroll
    for (int i = 0; i < DR_DK; i++)
    {
        st[i] = m[(size_t)i * dv + j];
    }
    for (int t = 0; t < n; t++)
    {
        const float *row = mixed + (size_t)t * cd;
        __syncthreads();
        for (int i = j; i < DR_DK; i += blockDim.x)
        {
            qs[i] = row[kq * DR_DK + i];
            ks[i] = row[(kh + kq) * DR_DK + i];
        }
        __syncthreads();
        const float v = row[2 * kh * DR_DK + h * dv + j];
        const float g = expf(decay[(size_t)t * vh + h]);
        const float b = beta[(size_t)t * vh + h];
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < DR_DK; i++)
        {
            st[i] *= g;
            kv += st[i] * ks[i];
        }
        const float delta = (v - kv) * b;
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < DR_DK; i++)
        {
            st[i] += ks[i] * delta;
            o += st[i] * qs[i];
        }
        core[(size_t)t * vh * dv + h * dv + j] = o;
    }
#pragma unroll
    for (int i = 0; i < DR_DK; i++)
    {
        m[(size_t)i * dv + j] = st[i];
    }
}

// Q8_0 matmul on tensor cores for many inputs. Weights go in as exact f16 integers and
// inputs, scaled by a power of 2 per row (tf_rowscale), as high and low f16 parts.
#define TM_BM 64
#define TM_BN 64
#define TM_BLOCKS 2
#define TM_KS (32 * TM_BLOCKS)
#define TM_LD (TM_KS + 8)

__device__ __forceinline__ void tm_block(const unsigned char *__restrict__ w, const float *__restrict__ x,
                                         const float *__restrict__ up, float *__restrict__ y, int rows, int cols,
                                         int n, int r0, int t0)
{
    __shared__ __align__(16) __half ws[TM_BM][TM_LD];
    __shared__ __align__(16) __half xh[TM_BN][TM_LD];
    __shared__ __align__(16) __half xl[TM_BN][TM_LD];
    __shared__ float sc[TM_BLOCKS][TM_BM];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int nb = cols >> 5;
    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
    {
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
        {
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                acc[mi][ni][e] = 0.0f;
            }
        }
    }
    for (int b0 = 0; b0 < nb; b0 += TM_BLOCKS)
    {
        for (int i = tid; i < TM_BM * TM_BLOCKS * 2; i += 128)
        {
            const int r = i / (TM_BLOCKS * 2), rem = i % (TM_BLOCKS * 2);
            const int bb = rem >> 1, half = rem & 1;
            const int gr = r0 + r, b = b0 + bb;
            unsigned packed[8];
            float d = 0.0f;
            if (gr < rows && b < nb)
            {
                const unsigned char *blk = w + ((size_t)gr * nb + b) * 34;
                d = __half2float(__ushort_as_half(*(const unsigned short *)blk));
                const unsigned short *qp = (const unsigned short *)(blk + 2 + half * 16);
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    const unsigned short u = qp[j];
                    __half2 h = __floats2half2_rn((float)(signed char)(u & 0xff), (float)(signed char)(u >> 8));
                    packed[j] = *(unsigned *)&h;
                }
            }
            else
            {
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    packed[j] = 0;
                }
            }
            if (half == 0)
            {
                sc[bb][r] = d;
            }
            uint4 *dst = (uint4 *)&ws[r][bb * 32 + half * 16];
            dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
            dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        }
        for (int i = tid; i < TM_BN * (TM_KS / 4); i += 128)
        {
            const int t = i / (TM_KS / 4), c4 = i % (TM_KS / 4);
            const int gt = t0 + t, col = b0 * 32 + c4 * 4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gt < n && col < cols)
            {
                v = *(const float4 *)(x + (size_t)gt * cols + col);
            }
            if (gt < n)
            {
                const float u = up[gt];
                v.x *= u;
                v.y *= u;
                v.z *= u;
                v.w *= u;
            }
            unsigned h0, l0, h1, l1;
            tc_split(v.x, v.y, h0, l0);
            tc_split(v.z, v.w, h1, l1);
            *(uint2 *)&xh[t][c4 * 4] = make_uint2(h0, h1);
            *(uint2 *)&xl[t][c4 * 4] = make_uint2(l0, l1);
        }
        __syncthreads();
#pragma unroll
        for (int bb = 0; bb < TM_BLOCKS; bb++)
        {
            float p[2][4][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                {
#pragma unroll
                    for (int e = 0; e < 4; e++)
                    {
                        p[mi][ni][e] = 0.0f;
                    }
                }
            }
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
            {
                const int k0 = bb * 32 + kk * 16 + c * 2;
                unsigned a[2][4];
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
                {
                    const int row = wm * 32 + mi * 16 + g;
                    a[mi][0] = *(const unsigned *)&ws[row][k0];
                    a[mi][1] = *(const unsigned *)&ws[row + 8][k0];
                    a[mi][2] = *(const unsigned *)&ws[row][k0 + 8];
                    a[mi][3] = *(const unsigned *)&ws[row + 8][k0 + 8];
                }
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                {
                    const int tok = wn * 32 + ni * 8 + g;
                    const unsigned bh[2] = {*(const unsigned *)&xh[tok][k0], *(const unsigned *)&xh[tok][k0 + 8]};
                    const unsigned bl[2] = {*(const unsigned *)&xl[tok][k0], *(const unsigned *)&xl[tok][k0 + 8]};
#pragma unroll
                    for (int mi = 0; mi < 2; mi++)
                    {
                        tc_mma(p[mi][ni], a[mi], bl);
                        tc_mma(p[mi][ni], a[mi], bh);
                    }
                }
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
            {
                const float s_lo = sc[bb][wm * 32 + mi * 16 + g];
                const float s_hi = sc[bb][wm * 32 + mi * 16 + g + 8];
#pragma unroll
                for (int ni = 0; ni < 4; ni++)
                {
                    acc[mi][ni][0] += s_lo * p[mi][ni][0];
                    acc[mi][ni][1] += s_lo * p[mi][ni][1];
                    acc[mi][ni][2] += s_hi * p[mi][ni][2];
                    acc[mi][ni][3] += s_hi * p[mi][ni][3];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
    {
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
        {
            const int row = r0 + wm * 32 + mi * 16 + g;
            const int tok = t0 + wn * 32 + ni * 8 + c * 2;
#pragma unroll
            for (int e = 0; e < 4; e++)
            {
                const int rr = row + (e >> 1) * 8, tt = tok + (e & 1);
                if (rr < rows && tt < n)
                {
                    y[(size_t)tt * rows + rr] = acc[mi][ni][e] / up[tt];
                }
            }
        }
    }
}

// y = W x for many inputs, W in Q8_0, on tensor cores. Grid: rows / 64,
// inputs / 64, and one z per matrix of a group.
extern "C" __global__ void __launch_bounds__(128) tf_tm_q8(unsigned long long w0, const unsigned long long *wt,
                                                           const int *first, const float *x, float *y, int rows,
                                                           int cols, int n0, const float *up)
{
    const mm_pick p = mm_select(w0, wt, first, n0);
    if (blockIdx.y * TM_BN >= p.n)
    {
        return;
    }
    tm_block(p.w, x + (size_t)p.f0 * cols, up + p.f0, y + (size_t)p.f0 * rows, rows, cols, p.n, blockIdx.x * TM_BM,
             blockIdx.y * TM_BN);
}

// up[i] = row_up of row i's largest magnitude, one warp per row.
extern "C" __global__ void tf_rowscale(const float *__restrict__ x, float *__restrict__ up, int n, int cols)
{
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= n)
    {
        return;
    }
    const float *xr = x + (size_t)row * cols;
    float mx = 0.0f;
    for (int c = lane; c < cols; c += 32)
    {
        mx = fmaxf(mx, fabsf(xr[c]));
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    if (lane == 0)
    {
        up[row] = row_up(mx);
    }
}

/* QSA (TASK-31): each query's score against every complete block of keys, the sum over
 * the indexer's heads of ReLU(q . k) times 1 / sqrt(dim); -1 for blocks it cannot see yet.
 * One block of threads covers QS_Q queries and QS_B key blocks. */
#define QS_Q 8
#define QS_B 32
#define QS_MAXHD 512
#define QS_MAXD 128

extern "C" __global__ void __launch_bounds__(256) tf_qsa_scores(const float *__restrict__ iq, const float *__restrict__ blocks,
                                                                float *__restrict__ scores, int nq, int q0, int pos0, int ih,
                                                                int dim, int bsize, int nb, float iscale)
{
    __shared__ float qs[QS_Q][QS_MAXHD];
    __shared__ float ks[QS_B][QS_MAXD + 1];
    const int tid = threadIdx.x;
    const int qa = blockIdx.y * QS_Q, ba = blockIdx.x * QS_B;
    const int hd = ih * dim;
    for (int i = tid; i < QS_Q * hd; i += 256)
    {
        const int r = i / hd, c = i % hd;
        qs[r][c] = qa + r < nq ? iq[(size_t)(q0 + qa + r) * hd + c] : 0.f;
    }
    for (int i = tid; i < QS_B * dim; i += 256)
    {
        const int r = i / dim, c = i % dim;
        ks[r][c] = ba + r < nb ? blocks[(size_t)(ba + r) * dim + c] : 0.f;
    }
    __syncthreads();
    const int r = tid / QS_B, b = tid % QS_B;
    const int qi = qa + r, bi = ba + b;
    if (qi >= nq || bi >= nb)
    {
        return;
    }
    const int full = (pos0 + q0 + qi + 1) / bsize;
    float s = 0.f;
    for (int h = 0; h < ih; h++)
    {
        float d = 0.f;
        for (int i = 0; i < dim; i++)
        {
            d += qs[r][h * dim + i] * ks[b][i];
        }
        s += fmaxf(d, 0.f);
    }
    scores[(size_t)qi * nb + bi] = bi < full ? s * iscale : -1.f;
}

/* Exclusive prefix sum of `v` over the 256 threads; `*total` gets the sum. */
__device__ __forceinline__ int qsa_scan(int v, int *total, int *ws)
{
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1)
    {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o)
        {
            x += y;
        }
    }
    if (lane == 31)
    {
        ws[w] = x;
    }
    __syncthreads();
    if (w == 0)
    {
        int y = lane < 8 ? ws[lane] : 0;
        for (int o = 1; o < 8; o <<= 1)
        {
            const int z = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o)
            {
                y += z;
            }
        }
        if (lane < 8)
        {
            ws[lane] = y;
        }
    }
    __syncthreads();
    const int ex = (w ? ws[w - 1] : 0) + x - v;
    *total = ws[7];
    __syncthreads();
    return ex;
}

/* The `top` blocks each query keeps, in ascending order: every block scoring above the cut,
 * then the lowest-numbered of those scoring exactly at it, as the CPU path does. Scores are
 * never negative, so their bits sort as unsigned integers and a radix select finds the cut.
 * Queries with no more complete blocks than `top` keep them all and are skipped. */
extern "C" __global__ void __launch_bounds__(256) tf_qsa_topk(const float *__restrict__ scores, int *__restrict__ sel, int q0,
                                                              int pos0, int bsize, int top, int nb)
{
    __shared__ unsigned hist[256];
    __shared__ int ws[8];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    const int qi = blockIdx.x, tid = threadIdx.x;
    const int full = (pos0 + q0 + qi + 1) / bsize;
    if (full <= top)
    {
        return;
    }
    const unsigned *row = (const unsigned *)(scores + (size_t)qi * nb);
    unsigned prefix = 0, mask = 0;
    int k = top;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        hist[tid] = 0;
        __syncthreads();
        for (int b = tid; b < full; b += 256)
        {
            const unsigned u = row[b];
            if ((u & mask) == prefix)
            {
                atomicAdd(&hist[(u >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            int cum = 0;
            int j = 255;
            for (; j > 0; j--)
            {
                if (cum + (int)hist[j] >= k)
                {
                    break;
                }
                cum += (int)hist[j];
            }
            s_prefix = prefix | ((unsigned)j << shift);
            s_need = k - cum;
        }
        __syncthreads();
        prefix = s_prefix;
        k = s_need;
        mask |= 255u << shift;
        __syncthreads();
    }
    int eq_carry = 0, sel_carry = 0;
    for (int b0 = 0; b0 < full; b0 += 256)
    {
        const int b = b0 + tid;
        const unsigned u = b < full ? row[b] : 0u;
        const int gt = b < full && u > prefix;
        const int eq = b < full && u == prefix;
        int total;
        const int eq_rank = qsa_scan(eq, &total, ws) + eq_carry;
        eq_carry += total;
        const int take = gt || (eq && eq_rank < k);
        const int at = qsa_scan(take, &total, ws) + sel_carry;
        sel_carry += total;
        if (take)
        {
            sel[(size_t)qi * top + at] = b;
        }
    }
}

/* QSA attention for one query and one KV head over an f32 cache read through its tile
 * table: the positions of the kept blocks and the unfinished last one (every position while
 * the query keeps all blocks), in ascending order, with an online softmax. 256-dim heads,
 * each thread owning one dimension of every query head in the group. */
#define QA_HD 256
#define QA_P 8
#define QA_MAXG 16

extern "C" __global__ void __launch_bounds__(256) tf_qsa_attention(const float *__restrict__ q,
                                                                   const unsigned long long *__restrict__ tiles_k,
                                                                   const unsigned long long *__restrict__ tiles_v,
                                                                   const int *__restrict__ sel, float *__restrict__ out,
                                                                   int pos0, int heads, int kv_heads, int kvd, int bsize, int top,
                                                                   float scale)
{
    __shared__ float qs[QA_MAXG][QA_HD];
    __shared__ float ks[QA_P][QA_HD];
    __shared__ float vs[QA_P][QA_HD];
    __shared__ float ps[QA_MAXG][QA_P];
    __shared__ float corr[QA_MAXG];
    __shared__ float lsum[QA_MAXG];
    __shared__ int pos_s[QA_P];
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, w = tid >> 5;
    const int group = heads / kv_heads;
    const int p = pos0 + t, full = (p + 1) / bsize;
    const int dense = full <= top;
    const int count = dense ? p + 1 : top * bsize + (p + 1 - full * bsize);
    for (int i = tid; i < group * QA_HD; i += 256)
    {
        qs[i / QA_HD][i % QA_HD] = q[((size_t)t * heads + g * group + i / QA_HD) * QA_HD + i % QA_HD];
    }
    float acc[QA_MAXG];
#pragma unroll
    for (int h = 0; h < QA_MAXG; h++)
    {
        acc[h] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int c0 = 0; c0 < count; c0 += QA_P)
    {
        const int nv = min(QA_P, count - c0);
        if (tid < QA_P)
        {
            const int i = c0 + tid;
            int s = 0;
            if (i < count)
            {
                s = dense ? i
                          : (i < top * bsize ? sel[(size_t)t * top + i / bsize] * bsize + i % bsize : full * bsize + (i - top * bsize));
            }
            pos_s[tid] = s;
        }
        __syncthreads();
        for (int i = tid; i < QA_P * QA_HD; i += 256)
        {
            const int j = i / QA_HD, d = i % QA_HD;
            float kv = 0.f, vv = 0.f;
            if (j < nv)
            {
                const int s = pos_s[j];
                const size_t off = (size_t)(s % 32) * kvd + (size_t)g * QA_HD + d;
                kv = ((const float *)tiles_k[s / 32])[off];
                vv = ((const float *)tiles_v[s / 32])[off];
            }
            ks[j][d] = kv;
            vs[j][d] = vv;
        }
        __syncthreads();
        for (int idx = w; idx < group * QA_P; idx += 8)
        {
            const int h = idx / QA_P, j = idx % QA_P;
            float s = 0.f;
            for (int d = lane; d < QA_HD; d += 32)
            {
                s += qs[h][d] * ks[j][d];
            }
            for (int o = 16; o; o >>= 1)
            {
                s += __shfl_xor_sync(0xffffffffu, s, o);
            }
            if (lane == 0)
            {
                ps[h][j] = j < nv ? s * scale : -INFINITY;
            }
        }
        __syncthreads();
        if (tid < group)
        {
            float mx = m;
            for (int j = 0; j < nv; j++)
            {
                mx = fmaxf(mx, ps[tid][j]);
            }
            const float cr = expf(m - mx);
            float sum = 0.f;
            for (int j = 0; j < QA_P; j++)
            {
                const float e = j < nv ? expf(ps[tid][j] - mx) : 0.f;
                ps[tid][j] = e;
                sum += e;
            }
            l = l * cr + sum;
            m = mx;
            corr[tid] = cr;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < QA_MAXG; h++)
        {
            if (h < group)
            {
                float a = acc[h] * corr[h];
                for (int j = 0; j < nv; j++)
                {
                    a += ps[h][j] * vs[j][tid];
                }
                acc[h] = a;
            }
        }
        __syncthreads();
    }
    if (tid < group)
    {
        lsum[tid] = l;
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < QA_MAXG; h++)
    {
        if (h < group)
        {
            out[((size_t)t * heads + g * group + h) * QA_HD + tid] = acc[h] / lsum[h];
        }
    }
}
