/* TensorFold-Swarm public C ABI. Every subsystem (io_uring, verbs, CUDA)
 * sits behind this boundary so it can be called from Zig or C. */
#ifndef TENSORFOLD_H
#define TENSORFOLD_H

#ifdef __cplusplus
extern "C" {
#endif

#define TF_VERSION_MAJOR 0
#define TF_VERSION_MINOR 1
#define TF_VERSION_PATCH 0

/* Packed as (major << 16) | (minor << 8) | patch. */
unsigned tf_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TENSORFOLD_H */
