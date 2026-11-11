/* GPU matmuls (TASK-10) and attention (TASK-12) for the engine, on the CUDA
 * driver API. Kernels come from csrc/cuda/kernels.cu as an embedded fatbin. */
#ifndef TF_GPU_H
#define TF_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tf_gpu tf_gpu;

typedef struct tf_gpu_info {
    char name[96];
    int major;
    int minor;
    int integrated; /* 1: shares memory with the CPU (GB10) */
} tf_gpu_info;

/* Opens device 0 and loads the kernels from `fatbin`. Returns 0 or a negative
 * CUresult; on failure a message is in `err`. */
int tf_gpu_open(tf_gpu **out, const void *fatbin, char *err, size_t err_len);

void tf_gpu_get_info(const tf_gpu *g, tf_gpu_info *info);

/* Makes host memory [host, host + len) readable by kernels and returns its
 * device address. An integrated GPU reads it in place (registered, no copy);
 * a discrete GPU gets a copy in device memory. Call once per region. */
int tf_gpu_map(tf_gpu *g, void *host, size_t len, uint64_t *dev);

/* Weight formats for tf_gpu_matmul. */
#define TF_GPU_Q8_0 0
#define TF_GPU_F32 1

/* y (n x rows) = W x for x (n x cols), W in `format` at device address `w`.
 * Copies x in and y out; returns when y is ready. */
int tf_gpu_matmul(tf_gpu *g, uint64_t w, int format, const float *x, int n, int rows, int cols, float *y);
int tf_gpu_q8_0_matmul(tf_gpu *g, uint64_t w, const float *x, int n, int rows, int cols, float *y);

/* tf_gpu_matmul for `count` weights on the same inputs, with one copy in. */
int tf_gpu_matmuls(tf_gpu *g, int count, const uint64_t *w, int format, const int *rows, const float *x, int n, int cols,
                   float *const *y);

/* Pins host memory [host, host + len) and maps it for kernels to read in
 * place, on any GPU (TASK-12: the KV cache, which keeps changing). Undo with
 * tf_gpu_unregister. */
int tf_gpu_register(tf_gpu *g, void *host, size_t len, uint64_t *dev);
void tf_gpu_unregister(tf_gpu *g, void *host);

/* Causal attention for n new tokens at positions pos0 to pos0 + n - 1, head
 * dim 128 (TASK-12). q and out are n x heads x 128 floats in host memory. The
 * cache is read in place through `tiles` tiles of 32 positions: tile i's keys
 * start at device address tiles_k[i], position p of the tile at p * kvd
 * floats after it, KV head h at h * 128 floats; values likewise. `splits`
 * divides the tiles among blocks (1 for prefill, more for long decode). */
int tf_gpu_attention(tf_gpu *g, const float *q, int n, int pos0, int heads, int kv_heads, int kvd, float scale,
                     const uint64_t *tiles_k, const uint64_t *tiles_v, int tiles, int splits, float *out);

/* One layer's experts for n tokens, k calls each: expert e takes batch rows first[e]
 * on, row i is token src[i], and y[t] adds call j's row inv[t k + j] times w[t k + j]. */
int tf_gpu_experts(tf_gpu *g, int groups, const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                   const int *first, const int *src, const int *inv, const float *w, const float *x, int n, int k,
                   int d, int f, float *y);

/* Attention kernel: 1 first, 2 wide, 3 and 4 tensor cores (64 or 128 rows), 5 decode,
 * 0 chosen by shape (default; TF_ATT_KERNEL sets it). Returns the previous choice. */
int tf_gpu_set_attention_kernel(tf_gpu *g, int which);

/* 1: the first Q8_0 kernel for every matmul; 0: tiled and decode kernels (default;
 * TF_MM_KERNEL sets it). Returns the previous choice. */
int tf_gpu_set_matmul_kernel(tf_gpu *g, int which);

void tf_gpu_close(tf_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* TF_GPU_H */
