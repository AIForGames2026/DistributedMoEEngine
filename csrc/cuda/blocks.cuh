// Every other ggml block format on the GPU (TASK-36). Each format has one small decoder that
// turns 8 consecutive values of a row into exact small codes plus a shared scale and offset.
// Two kernels per format use it: a matrix-vector kernel for few inputs, tensor cores for many.
#pragma once

#include "iq_tables.cuh"

// ggml type ids.
#define QF_F16 1
#define QF_Q4_0 2
#define QF_Q4_1 3
#define QF_Q5_0 6
#define QF_Q5_1 7
#define QF_Q8_0 8
#define QF_Q2_K 10
#define QF_Q3_K 11
#define QF_Q4_K 12
#define QF_Q5_K 13
#define QF_Q6_K 14
#define QF_IQ2_XXS 16
#define QF_IQ2_XS 17
#define QF_IQ3_XXS 18
#define QF_IQ1_S 19
#define QF_IQ4_NL 20
#define QF_IQ3_S 21
#define QF_IQ2_S 22
#define QF_IQ4_XS 23
#define QF_IQ1_M 29
#define QF_BF16 30
#define QF_MXFP4 39
#define QF_NVFP4 40

// Values and bytes per block, and whether the format has an offset.
template <int F> struct qf;
#define QF_TRAITS(F, QK, BYTES, MIN)                                                                                   \
    template <> struct qf<F>                                                                                           \
    {                                                                                                                  \
        static constexpr int qk = QK, bytes = BYTES;                                                                   \
        static constexpr bool min = MIN;                                                                               \
    };
QF_TRAITS(QF_F16, 1, 2, false)
QF_TRAITS(QF_BF16, 1, 2, false)
QF_TRAITS(QF_Q4_0, 32, 18, false)
QF_TRAITS(QF_Q4_1, 32, 20, true)
QF_TRAITS(QF_Q5_0, 32, 22, false)
QF_TRAITS(QF_Q5_1, 32, 24, true)
QF_TRAITS(QF_Q8_0, 32, 34, false)
QF_TRAITS(QF_Q2_K, 256, 84, true)
QF_TRAITS(QF_Q3_K, 256, 110, false)
QF_TRAITS(QF_Q4_K, 256, 144, true)
QF_TRAITS(QF_Q5_K, 256, 176, true)
QF_TRAITS(QF_Q6_K, 256, 210, false)
QF_TRAITS(QF_IQ2_XXS, 256, 66, false)
QF_TRAITS(QF_IQ2_XS, 256, 74, false)
QF_TRAITS(QF_IQ2_S, 256, 82, false)
QF_TRAITS(QF_IQ3_XXS, 256, 98, false)
QF_TRAITS(QF_IQ3_S, 256, 110, false)
QF_TRAITS(QF_IQ1_S, 256, 50, false)
QF_TRAITS(QF_IQ1_M, 256, 56, false)
QF_TRAITS(QF_IQ4_NL, 32, 18, false)
QF_TRAITS(QF_IQ4_XS, 256, 136, false)
QF_TRAITS(QF_MXFP4, 32, 17, false)
QF_TRAITS(QF_NVFP4, 64, 36, false)

__device__ __forceinline__ unsigned qf_u16(const unsigned char *p)
{
    return p[0] | (p[1] << 8);
}

__device__ __forceinline__ unsigned qf_u32(const unsigned char *p)
{
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned)p[3] << 24);
}

__device__ __forceinline__ float qf_half(const unsigned char *p)
{
    return __half2float(__ushort_as_half((unsigned short)qf_u16(p)));
}

// Byte j of a grid entry.
template <typename T> __device__ __forceinline__ unsigned qf_byte(T g, int j)
{
    return (unsigned)(g >> (8 * j)) & 0xFF;
}

__device__ __forceinline__ float qf_sign(unsigned signs, int j)
{
    return (signs >> j) & 1 ? -1.0f : 1.0f;
}

// MXFP4's E8M0 scale and NVFP4's unsigned E4M3 scale, both halved as in quant.zig.
__device__ __forceinline__ float qf_e8m0_half(unsigned e)
{
    return __uint_as_float(e < 2 ? 0x00200000u << e : (e - 1) << 23);
}

__device__ __forceinline__ float qf_ue4m3_half(unsigned x)
{
    if (x == 0 || x == 0x7F)
    {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const float man = (float)(x & 7);
    const float v = exp == 0 ? man * 0x1p-9f : (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
    return v * 0.5f;
}

__device__ __forceinline__ float qf_fp4(unsigned code)
{
    const float t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return t[code & 15];
}

// Values 8 c8 to 8 c8 + 7 of a row starting at `row`: code[j], with one scale and offset for all 8.
template <int F> __device__ __forceinline__ void qf_decode8(const unsigned char *row, int c8, float code[8], float &scale,
                                                           float &off)
{
    const int v0 = c8 * 8;
    const unsigned char *blk = row + (size_t)(v0 / qf<F>::qk) * qf<F>::bytes;
    const int o = v0 % qf<F>::qk;
    off = 0.0f;
    if (F == QF_F16 || F == QF_BF16)
    {
        scale = 1.0f;
        const unsigned char *p = row + (size_t)v0 * 2;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = F == QF_F16 ? qf_half(p + 2 * j) : __uint_as_float(qf_u16(p + 2 * j) << 16);
        }
    }
    else if (F == QF_Q4_0 || F == QF_Q4_1 || F == QF_IQ4_NL || F == QF_MXFP4)
    {
        const int at = F == QF_Q4_1 ? 4 : F == QF_MXFP4 ? 1 : 2;
        scale = F == QF_MXFP4 ? qf_e8m0_half(blk[0]) : qf_half(blk);
        off = F == QF_Q4_1 ? qf_half(blk + 2) : 0.0f;
        const unsigned char *qs = blk + at + (o & 15);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const unsigned q = o >= 16 ? qs[j] >> 4 : qs[j] & 15;
            code[j] = F == QF_Q4_0 ? (float)((int)q - 8) : F == QF_Q4_1 ? (float)q : F == QF_MXFP4 ? qf_fp4(q) : (float)kvalues_iq4nl[q];
        }
    }
    else if (F == QF_Q5_0 || F == QF_Q5_1)
    {
        const int at = F == QF_Q5_1 ? 4 : 2;
        scale = qf_half(blk);
        off = F == QF_Q5_1 ? qf_half(blk + 2) : 0.0f;
        const unsigned qh = qf_u32(blk + at);
        const unsigned char *qs = blk + at + 4 + (o & 15);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const unsigned q = (o >= 16 ? qs[j] >> 4 : qs[j] & 15) | (((qh >> (o + j)) & 1) << 4);
            code[j] = F == QF_Q5_0 ? (float)((int)q - 16) : (float)q;
        }
    }
    else if (F == QF_Q8_0)
    {
        scale = qf_half(blk);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = (float)(signed char)blk[2 + o + j];
        }
    }
    else if (F == QF_Q2_K || F == QF_Q3_K)
    {
        const int n = o >> 7, r = o & 127, j2 = r >> 5, half = (r >> 4) & 1, l0 = r & 15;
        const int k = n * 8 + j2 * 2 + half;
        const int qat = (F == QF_Q2_K ? 16 : 32) + n * 32 + half * 16 + l0;
        if (F == QF_Q2_K)
        {
            const unsigned sc = blk[k];
            scale = qf_half(blk + 80) * (float)(sc & 15);
            off = -(qf_half(blk + 82) * (float)(sc >> 4));
        }
        else
        {
            const unsigned char *s = blk + 96;
            const int low = k < 8 ? s[k] & 15 : s[k - 8] >> 4;
            const int high = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
            scale = qf_half(blk + 108) * (float)((low | (high << 4)) - 32);
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const int q = (blk[qat + j] >> (2 * j2)) & 3;
            if (F == QF_Q2_K)
            {
                code[j] = (float)q;
            }
            else
            {
                const int hbit = (blk[half * 16 + l0 + j] >> (n * 4 + j2)) & 1;
                code[j] = (float)(q - (hbit ? 0 : 4));
            }
        }
    }
    else if (F == QF_Q4_K || F == QF_Q5_K)
    {
        const int c = o >> 6, hi = (o >> 5) & 1, l0 = o & 31;
        const unsigned char *s = blk + 4;
        const int jj = 2 * c + hi;
        int sc, m;
        if (jj < 4)
        {
            sc = s[jj] & 63;
            m = s[jj + 4] & 63;
        }
        else
        {
            sc = (s[jj + 4] & 0xF) | ((s[jj - 4] >> 6) << 4);
            m = (s[jj + 4] >> 4) | ((s[jj] >> 6) << 4);
        }
        scale = qf_half(blk) * (float)sc;
        off = -(qf_half(blk + 2) * (float)m);
        const unsigned char *ql = blk + (F == QF_Q4_K ? 16 : 48) + c * 32 + l0;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            unsigned q = hi ? ql[j] >> 4 : ql[j] & 15;
            if (F == QF_Q5_K)
            {
                q += ((blk[16 + l0 + j] >> jj) & 1) * 16;
            }
            code[j] = (float)q;
        }
    }
    else if (F == QF_Q6_K)
    {
        const int n = o >> 7, r = o & 127, quarter = r >> 5, l0 = r & 31;
        scale = qf_half(blk + 208) * (float)(signed char)blk[192 + n * 8 + (l0 >> 4) + 2 * quarter];
        const unsigned char *ql = blk + n * 64 + (quarter & 1) * 32 + l0;
        const unsigned char *qh = blk + 128 + n * 32 + l0;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            const int low = quarter < 2 ? ql[j] & 15 : ql[j] >> 4;
            code[j] = (float)((low | (((qh[j] >> (2 * quarter)) & 3) << 4)) - 32);
        }
    }
    else if (F == QF_IQ2_XXS || F == QF_IQ2_XS || F == QF_IQ2_S)
    {
        const int ib = o >> 5, l = (o >> 3) & 3;
        const float d = qf_half(blk);
        unsigned long long g;
        unsigned signs;
        if (F == QF_IQ2_XXS)
        {
            const unsigned aux1 = qf_u32(blk + 2 + 8 * ib + 4);
            scale = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            g = iq2xxs_grid[blk[2 + 8 * ib + l]];
            signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
        }
        else
        {
            const unsigned sc = blk[(F == QF_IQ2_XS ? 66 : 74) + ib];
            scale = d * (0.5f + (float)(l < 2 ? sc & 15 : sc >> 4)) * 0.25f;
            if (F == QF_IQ2_XS)
            {
                const unsigned q = qf_u16(blk + 2 + 2 * (4 * ib + l));
                g = iq2xs_grid[q & 511];
                signs = ksigns_iq2xs[q >> 9];
            }
            else
            {
                const unsigned qh = blk[66 + ib];
                g = iq2s_grid[blk[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300)];
                signs = blk[34 + 4 * ib + l];
            }
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = (float)qf_byte(g, j) * qf_sign(signs, j);
        }
    }
    else if (F == QF_IQ3_XXS || F == QF_IQ3_S)
    {
        const int ib = o >> 5, l = (o >> 3) & 3;
        const float d = qf_half(blk);
        unsigned g1, g2, signs;
        if (F == QF_IQ3_XXS)
        {
            const unsigned aux = qf_u32(blk + 66 + 4 * ib);
            scale = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            g1 = iq3xxs_grid[blk[2 + 8 * ib + 2 * l]];
            g2 = iq3xxs_grid[blk[2 + 8 * ib + 2 * l + 1]];
        }
        else
        {
            const unsigned sc = blk[106 + (ib >> 1)];
            scale = d * (float)(1 + 2 * (ib & 1 ? sc >> 4 : sc & 15));
            const unsigned qh = blk[66 + ib];
            g1 = iq3s_grid[blk[2 + 8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256)];
            g2 = iq3s_grid[blk[2 + 8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256)];
            signs = blk[74 + 4 * ib + l];
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            code[j] = (float)qf_byte(g1, j) * qf_sign(signs, j);
            code[j + 4] = (float)qf_byte(g2, j) * qf_sign(signs, j + 4);
        }
    }
    else if (F == QF_IQ1_S || F == QF_IQ1_M)
    {
        const int ib = o >> 5, l = (o >> 3) & 3;
        unsigned idx;
        float delta;
        if (F == QF_IQ1_S)
        {
            const unsigned qh = qf_u16(blk + 34 + 2 * ib);
            scale = qf_half(blk) * (float)(2 * ((qh >> 12) & 7) + 1);
            delta = qh & 0x8000 ? -0.125f : 0.125f;
            idx = blk[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
        }
        else
        {
            const unsigned s0 = qf_u16(blk + 48), s1 = qf_u16(blk + 50), s2 = qf_u16(blk + 52), s3 = qf_u16(blk + 54);
            const unsigned bits = (s0 >> 12) | ((s1 >> 8) & 0x00f0) | ((s2 >> 4) & 0x0f00) | (s3 & 0xf000);
            const float d = __half2float(__ushort_as_half((unsigned short)bits));
            const unsigned s = qf_u16(blk + 48 + 2 * (ib >> 1));
            const int shift = 6 * (ib & 1) + (l >= 2 ? 3 : 0);
            scale = d * (float)(2 * ((s >> shift) & 7) + 1);
            const unsigned qh = blk[32 + 2 * ib + (l >> 1)];
            idx = blk[4 * ib + l] | ((qh << (l & 1 ? 4 : 8)) & 0x700);
            delta = qh & (l & 1 ? 0x80 : 0x08) ? -0.125f : 0.125f;
        }
        const unsigned long long g = iq1s_grid[idx];
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = (float)(signed char)qf_byte(g, j) + delta;
        }
    }
    else if (F == QF_IQ4_XS)
    {
        const int ib = o >> 5, w = o & 31;
        const int ls = ((blk[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((qf_u16(blk + 2) >> (2 * ib)) & 3) << 4);
        scale = qf_half(blk) * (float)(ls - 32);
        const unsigned char *qs = blk + 8 + 16 * ib + (w & 15);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = (float)kvalues_iq4nl[w >= 16 ? qs[j] >> 4 : qs[j] & 15];
        }
    }
    else if (F == QF_NVFP4)
    {
        const int g = o >> 4, w = o & 15;
        scale = qf_ue4m3_half(blk[g]);
        const unsigned char *qs = blk + 4 + g * 8;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            code[j] = qf_fp4(w >= 8 ? qs[j] >> 4 : qs[j] & 15);
        }
    }
}

// Few inputs: one warp per row, each lane 8 values at a time, MV_TOK inputs per pass.
template <int F>
__device__ __forceinline__ void qf_mv_row(const unsigned char *__restrict__ w, const float *__restrict__ x,
                                          float *__restrict__ y, int rows, int cols, int n, int row)
{
    const int lane = threadIdx.x & 31;
    const int chunks = cols >> 3;
    const unsigned char *wr = w + (size_t)row * (cols / qf<F>::qk) * qf<F>::bytes;
    for (int t0 = 0; t0 < n; t0 += MV_TOK)
    {
        const int nt = min(MV_TOK, n - t0);
        float acc[MV_TOK];
#pragma unroll
        for (int t = 0; t < MV_TOK; t++)
        {
            acc[t] = 0.0f;
        }
        for (int c8 = lane; c8 < chunks; c8 += 32)
        {
            float code[8], scale, off;
            qf_decode8<F>(wr, c8, code, scale, off);
#pragma unroll
            for (int t = 0; t < MV_TOK; t++)
            {
                if (t >= nt)
                {
                    break;
                }
                const float4 *xp = (const float4 *)(x + (size_t)(t0 + t) * cols + c8 * 8);
                const float4 a = xp[0], b = xp[1];
                const float s = code[0] * a.x + code[1] * a.y + code[2] * a.z + code[3] * a.w + code[4] * b.x +
                                code[5] * b.y + code[6] * b.z + code[7] * b.w;
                acc[t] += scale * s;
                if (qf<F>::min)
                {
                    acc[t] += off * (a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MV_TOK; t++)
        {
            if (t >= nt)
            {
                break;
            }
            float s = acc[t];
            for (int off = 16; off > 0; off >>= 1)
            {
                s += __shfl_xor_sync(0xffffffffu, s, off);
            }
            if (lane == 0)
            {
                y[(size_t)(t0 + t) * rows + row] = s;
            }
        }
    }
}

// Many inputs on tensor cores: codes go in as exact f16 values, inputs (scaled by a power of 2
// per row, tf_rowscale) as high and low f16 parts; each group of 16 values then takes its scale
// and offset in f32. Same tiles and warp layout as tm_block.
#define QT_GROUPS (TM_KS / 16)

template <int F>
__device__ __forceinline__ void qf_tm_block(const unsigned char *__restrict__ w, const float *__restrict__ x,
                                            const float *__restrict__ up, float *__restrict__ y, int rows, int cols,
                                            int n, int r0, int t0)
{
    __shared__ __align__(16) __half ws[TM_BM][TM_LD];
    __shared__ __align__(16) __half xh[TM_BN][TM_LD];
    __shared__ __align__(16) __half xl[TM_BN][TM_LD];
    __shared__ float sc[QT_GROUPS][TM_BM];
    __shared__ float of[QT_GROUPS][TM_BM];
    __shared__ float xs[TM_BN][QT_GROUPS];
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, c = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const size_t row_bytes = (size_t)(cols / qf<F>::qk) * qf<F>::bytes;
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
    for (int k0 = 0; k0 < cols; k0 += TM_KS)
    {
        for (int i = tid; i < TM_BM * (TM_KS / 8); i += 128)
        {
            const int r = i / (TM_KS / 8), ch = i % (TM_KS / 8);
            const int gr = r0 + r, col = k0 + ch * 8;
            float code[8], scale = 0.0f, off = 0.0f;
            unsigned packed[4] = {0, 0, 0, 0};
            if (gr < rows && col < cols)
            {
                qf_decode8<F>(w + (size_t)gr * row_bytes, col >> 3, code, scale, off);
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    __half2 h = __floats2half2_rn(code[2 * j], code[2 * j + 1]);
                    packed[j] = *(unsigned *)&h;
                }
            }
            if ((ch & 1) == 0)
            {
                sc[ch >> 1][r] = scale;
                of[ch >> 1][r] = off;
            }
            *(uint4 *)&ws[r][ch * 8] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        }
        for (int i = tid; i < TM_BN * (TM_KS / 4); i += 128)
        {
            const int t = i / (TM_KS / 4), c4 = i % (TM_KS / 4);
            const int gt = t0 + t, col = k0 + c4 * 4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (gt < n && col < cols)
            {
                v = *(const float4 *)(x + (size_t)gt * cols + col);
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
        if (qf<F>::min)
        {
            for (int i = tid; i < TM_BN * QT_GROUPS; i += 128)
            {
                const int t = i / QT_GROUPS, gg = i % QT_GROUPS;
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; j++)
                {
                    s += __half2float(xh[t][gg * 16 + j]) + __half2float(xl[t][gg * 16 + j]);
                }
                xs[t][gg] = s;
            }
            __syncthreads();
        }
#pragma unroll
        for (int kk = 0; kk < QT_GROUPS; kk++)
        {
            const int kc = kk * 16 + c * 2;
            unsigned a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++)
            {
                const int row = wm * 32 + mi * 16 + g;
                a[mi][0] = *(const unsigned *)&ws[row][kc];
                a[mi][1] = *(const unsigned *)&ws[row + 8][kc];
                a[mi][2] = *(const unsigned *)&ws[row][kc + 8];
                a[mi][3] = *(const unsigned *)&ws[row + 8][kc + 8];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++)
            {
                const int tok = wn * 32 + ni * 8 + g;
                const unsigned bh[2] = {*(const unsigned *)&xh[tok][kc], *(const unsigned *)&xh[tok][kc + 8]};
                const unsigned bl[2] = {*(const unsigned *)&xl[tok][kc], *(const unsigned *)&xl[tok][kc + 8]};
                const int tc0 = wn * 32 + ni * 8 + c * 2;
#pragma unroll
                for (int mi = 0; mi < 2; mi++)
                {
                    float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    tc_mma(p, a[mi], bl);
                    tc_mma(p, a[mi], bh);
                    const int row = wm * 32 + mi * 16 + g;
                    const float s_lo = sc[kk][row], s_hi = sc[kk][row + 8];
                    acc[mi][ni][0] += s_lo * p[0];
                    acc[mi][ni][1] += s_lo * p[1];
                    acc[mi][ni][2] += s_hi * p[2];
                    acc[mi][ni][3] += s_hi * p[3];
                    if (qf<F>::min)
                    {
                        const float o_lo = of[kk][row], o_hi = of[kk][row + 8];
                        const float x0 = xs[tc0][kk], x1 = xs[tc0 + 1][kk];
                        acc[mi][ni][0] += o_lo * x0;
                        acc[mi][ni][1] += o_lo * x1;
                        acc[mi][ni][2] += o_hi * x0;
                        acc[mi][ni][3] += o_hi * x1;
                    }
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

// Grids as tf_mv_q8 and tf_tm_q8: one z per matrix of a group, or w0 alone with n0 inputs.
#define QF_KERNELS(F, NAME)                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) tf_qmv_##NAME(unsigned long long w0, const unsigned long long *wt, \
                                                                    const int *first, const float *x, float *y,        \
                                                                    int rows, int cols, int n0)                        \
    {                                                                                                                  \
        const mm_pick p = mm_select(w0, wt, first, n0);                                                                \
        const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;                                                  \
        if (row >= rows || p.n <= 0)                                                                                   \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        qf_mv_row<F>(p.w, x + (size_t)p.f0 * cols, y + (size_t)p.f0 * rows, rows, cols, p.n, row);                     \
    }
#define QF_TM_KERNEL(F, NAME)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) tf_qtm_##NAME(unsigned long long w0, const unsigned long long *wt, \
                                                                    const int *first, const float *x, float *y,        \
                                                                    int rows, int cols, int n0, const float *up)       \
    {                                                                                                                  \
        const mm_pick p = mm_select(w0, wt, first, n0);                                                                \
        if (blockIdx.y * TM_BN >= p.n)                                                                                 \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        qf_tm_block<F>(p.w, x + (size_t)p.f0 * cols, up + p.f0, y + (size_t)p.f0 * rows, rows, cols, p.n,               \
                       blockIdx.x * TM_BM, blockIdx.y * TM_BN);                                                         \
    }
#define QF_BOTH(F, NAME) QF_KERNELS(F, NAME) QF_TM_KERNEL(F, NAME)

QF_BOTH(QF_F16, f16)
QF_KERNELS(QF_BF16, bf16)
QF_BOTH(QF_Q4_0, q4_0)
QF_BOTH(QF_Q4_1, q4_1)
QF_BOTH(QF_Q5_0, q5_0)
QF_BOTH(QF_Q5_1, q5_1)
QF_BOTH(QF_Q8_0, q8_0)
QF_BOTH(QF_Q2_K, q2_k)
QF_BOTH(QF_Q3_K, q3_k)
QF_BOTH(QF_Q4_K, q4_k)
QF_BOTH(QF_Q5_K, q5_k)
QF_BOTH(QF_Q6_K, q6_k)
QF_BOTH(QF_IQ2_XXS, iq2_xxs)
QF_BOTH(QF_IQ2_XS, iq2_xs)
QF_BOTH(QF_IQ2_S, iq2_s)
QF_BOTH(QF_IQ3_XXS, iq3_xxs)
QF_BOTH(QF_IQ3_S, iq3_s)
QF_BOTH(QF_IQ1_S, iq1_s)
QF_BOTH(QF_IQ1_M, iq1_m)
QF_BOTH(QF_IQ4_NL, iq4_nl)
QF_BOTH(QF_IQ4_XS, iq4_xs)
QF_BOTH(QF_MXFP4, mxfp4)
QF_BOTH(QF_NVFP4, nvfp4)
