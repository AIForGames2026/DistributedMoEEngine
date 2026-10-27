/* Toolchain gate helpers (TASK-01): the smallest calls that prove liburing
 * and the CUDA runtime link and work on this machine. */
#ifndef TF_GATE_H
#define TF_GATE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tf_cuda_info {
    char name[96];
    int major;
    int minor;
    int sm_count;
    size_t total_mem;
} tf_cuda_info;

/* Submits one NOP through a fresh io_uring and reaps it. 0 on success,
 * negative errno otherwise. */
int tf_uring_selftest(void);

/* Number of CUDA devices, or a negative cudaError_t. */
int tf_cuda_device_count(void);

/* Fills `out` for device `dev` and runs a 1 MiB cudaMalloc + cudaMemset.
 * 0 on success, negative cudaError_t otherwise. */
int tf_cuda_device_info(int dev, tf_cuda_info *out);

#ifdef __cplusplus
}
#endif

#endif /* TF_GATE_H */
