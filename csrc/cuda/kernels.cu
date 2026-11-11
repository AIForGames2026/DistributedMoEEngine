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

// Causal attention over a KV cache (TASK-12), head_dim 128. Must match tf_gpu.c.
#define ATT_HD 128
#define ATT_QT 32 // query rows per block: (token, query head) pairs sharing one KV head
#define ATT_PT 32 // cache positions per tile

// One block takes ATT_QT query rows of KV head blockIdx.y and the tiles of
// split blockIdx.z, with an online softmax, and writes a partial result
// (running max, sum, unnormalized output) that tf_attention_merge combines.
// Tile i's keys start at tiles_k[i]: position p of the tile, head h, is at
// tiles_k[i] + p * kvd + h * 128 floats (the same for values). Rows are
// t * group + g for token t and query head h * group + g; token t sits at
// position pos0 + t and sees positions 0 to pos0 + t.
extern "C" __global__ void tf_attention(const float *__restrict__ q, const unsigned long long *__restrict__ tiles_k,
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
        const float *kt = (const float *)tiles_k[tile];
        const float *vt = (const float *)tiles_v[tile];
        for (int i = tid; i < ATT_PT * ATT_HD; i += blockDim.x) {
            const int p = i / ATT_HD, d = i % ATT_HD;
            const bool ok = tile * ATT_PT + p <= block_limit;
            ks[p][d] = ok ? kt[(size_t)p * kvd + h * ATT_HD + d] : 0.0f;
            vs[p][d] = ok ? vt[(size_t)p * kvd + h * ATT_HD + d] : 0.0f;
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

extern "C" __global__ void __launch_bounds__(256) tf_attention_wide(
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
                const size_t at = (size_t)(p % ATT_PT) * kvd + h * ATT_HD;
                kv = ((const float4 *)((const float *)tiles_k[t] + at))[d4];
                vv = ((const float4 *)((const float *)tiles_v[t] + at))[d4];
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
// split order: out[t][head][d], one block of 128 threads per (row, KV head).
extern "C" __global__ void tf_attention_merge(const float *__restrict__ part_m, const float *__restrict__ part_l,
                                              const float *__restrict__ part_acc, float *__restrict__ out, int n,
                                              int heads, int kv_heads, int splits)
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
        acc += part_acc[at * ATT_HD + d] * f;
    }
    out[((size_t)(j / group) * heads + h * group + j % group) * ATT_HD + d] = acc / l;
}
