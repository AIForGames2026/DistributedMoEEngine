/* GPU matmuls for the engine (TASK-10), on the CUDA driver API. Kernels come
 * from csrc/cuda/kernels.cu as an embedded fatbin. */
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

void tf_gpu_close(tf_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* TF_GPU_H */
