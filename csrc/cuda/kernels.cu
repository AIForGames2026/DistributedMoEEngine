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
