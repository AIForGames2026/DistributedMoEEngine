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

/* y (n x rows) = W x for the n inputs in x (n x cols), W in Q8_0 at device
 * address `w`. Copies x in and y out; returns when y is ready. */
int tf_gpu_q8_0_matmul(tf_gpu *g, uint64_t w, const float *x, int n, int rows, int cols, float *y);

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

/* SwiGLU experts for one layer (TASK-12): expert e takes input rows first[e]
 * to first[e + 1] - 1 of x (rows x d floats, host memory) and writes the
 * same rows of y: down(silu(gate x) * up x), with Q8_0 weights at device
 * addresses gate[e] (f x d), up[e] (f x d) and down[e] (d x f). One copy
 * in, every expert queued, one copy out. */
int tf_gpu_experts(tf_gpu *g, int experts, const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                   const int *first, const float *x, int d, int f, float *y);

/* Which attention kernel to use: 1 tf_attention, 0 or 2 tf_attention_wide
 * when the device has the shared memory for it (the default; TF_ATT_KERNEL
 * sets it at open). Returns the previous choice. */
int tf_gpu_set_attention_kernel(tf_gpu *g, int which);

void tf_gpu_close(tf_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* TF_GPU_H */
