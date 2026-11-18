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

/* Bytes of device memory this process holds for weight copies and working
 * buffers. On an integrated GPU they come out of system memory. */
size_t tf_gpu_device_bytes(void);

/* Makes host memory [host, host + len) readable by kernels and returns its
 * device address. An integrated GPU reads it in place (registered, no copy);
 * a discrete GPU gets a copy in device memory. Call once per region. */
int tf_gpu_map(tf_gpu *g, void *host, size_t len, uint64_t *dev);

/* Weight formats for tf_gpu_matmul. */
#define TF_GPU_Q8_0 0
#define TF_GPU_F32 1
#define TF_GPU_Q4_K 2

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

/* Pinned host memory that kernels read in place; free with tf_gpu_free_host. */
int tf_gpu_alloc_host(tf_gpu *g, size_t len, void **host, uint64_t *dev);
void tf_gpu_free_host(tf_gpu *g, void *host);

/* Causal attention for n new tokens at positions pos0 to pos0 + n - 1, head
 * dim 128 (TASK-12). q and out are n x heads x 128 floats in host memory. The
 * cache is read in place through `tiles` tiles of 32 positions: tile i's keys
 * start at device address tiles_k[i], one row per position, KV head h at
 * h * 128 values into the row; values likewise. `fmt` says how a row stores
 * kvd values (TASK-27, src/kv_format.zig): 0 f32 (kvd * 4 bytes), 1 bfloat16
 * (kvd * 2), 2 E4M3 then a power-of-two scale byte per KV head, padded to 16,
 * 3 keys as in 1 and values as in 2.
 * `splits` divides the tiles among blocks (1 for prefill, more for long decode).
 * `hd` is the head size: 128, or 256 (TASK-29) where the device allows its kernel. */
int tf_gpu_attention(tf_gpu *g, const float *q, int n, int pos0, int heads, int kv_heads, int kvd, float scale,
                     const uint64_t *tiles_k, const uint64_t *tiles_v, int tiles, int splits, float *out, int fmt, int hd);

/* The gated delta rule of a Gated DeltaNet layer (TASK-29) over n tokens, in
 * order. mixed holds each token's cd values: kh key heads of queries
 * (L2-normalized and scaled), kh of keys (L2-normalized), then vh value heads
 * of dv values; value head h reads key head h mod kh. beta and decay (its
 * log) are n x vh. state is the device address of the vh dk x dv matrices,
 * updated in place; core gets n x vh x dv outputs. dk must be 128. */
int tf_gpu_delta_rule(tf_gpu *g, const float *mixed, const float *beta, const float *decay, int n, int cd, int kh, int dk,
                      int vh, int dv, uint64_t state, float *core);

/* QSA attention (TASK-31) for n queries from pos0 over an f32 cache's tiles, 256-dim heads:
 * q holds their n x heads x 256 queries and iq their n x ih x dim indexer queries; blocks is
 * the device address of every complete block's pooled key (dim each). A query with more than
 * `top` complete blocks keeps the `top` the indexer scores highest (ties to the lower block),
 * written ascending to kept (n x top) when kept is not NULL; out gets n x heads x 256. */
int tf_gpu_qsa(tf_gpu *g, const float *q, const float *iq, int n, int pos0, int heads, int kv_heads, int kvd, int ih, int dim,
               int bsize, int top, float scale, uint64_t blocks, const uint64_t *tiles_k, const uint64_t *tiles_v, int tiles,
               float *out, int *kept);

/* One layer's experts for n tokens, k calls each (see olmoe.ExpertJob). `most`, if larger
 * than any expert's rows here, picks the kernels, so a layer split into calls rounds as one. */
int tf_gpu_experts(tf_gpu *g, int groups, const uint64_t *gate, const uint64_t *up, const uint64_t *down,
                   const int *first, const int *src, const int *inv, const float *w, const float *x, int n, int k,
                   int d, int f, float *y, int fmt, float limit, int most);

/* Attention kernel: 1 first, 2 wide, 3 and 4 tensor cores (64 or 128 rows), 5 decode,
 * 0 chosen by shape (default; TF_ATT_KERNEL sets it). Returns the previous choice. */
int tf_gpu_set_attention_kernel(tf_gpu *g, int which);

/* Matmul kernels: 1 the first Q8_0 kernel, 2 tiled f32, 0 tensor cores for Q8_0 prefill
 * (default; TF_MM_KERNEL sets it). Returns the previous choice. */
int tf_gpu_set_matmul_kernel(tf_gpu *g, int which);

void tf_gpu_close(tf_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* TF_GPU_H */
