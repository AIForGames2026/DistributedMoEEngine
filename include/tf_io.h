/* Expert weight reader (TASK-03): io_uring + O_DIRECT reads straight into a
 * page-aligned destination region, registered as a fixed buffer when the
 * locked-memory limit allows. */
#ifndef TF_IO_H
#define TF_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tf_reader tf_reader;

typedef struct tf_reader_opts {
    unsigned queue_depth; /* requests in flight, 1 to 256 */
    size_t block_bytes;   /* bytes per request; multiple of the DIO alignment */
    int sqpoll;           /* nonzero: kernel submission thread (IORING_SETUP_SQPOLL) */
} tf_reader_opts;

typedef struct tf_reader_info {
    uint64_t file_bytes;
    uint32_t dio_mem_align;    /* from statx(STATX_DIOALIGN) */
    uint32_t dio_offset_align;
    int fixed_buffer;          /* 1 if the region is registered with the ring */
    int fixed_file;
    int sqpoll;                /* 1 if the kernel granted SQPOLL */
} tf_reader_info;

/* Opens `path` with O_DIRECT and sets up a ring. `region` (page aligned) is
 * where reads may land; it is registered as one fixed buffer if possible.
 * Returns 0 or a negative errno. */
int tf_reader_open(tf_reader **out, const char *path, const tf_reader_opts *opts,
                   void *region, size_t region_len);

void tf_reader_get_info(const tf_reader *r, tf_reader_info *info);

/* Reads file bytes [offset, offset + len) into `dst`, which must lie inside
 * the region. `offset` and `len` must be multiples of dio_offset_align, and
 * `dst` of dio_mem_align. Splits into block_bytes requests with up to
 * queue_depth in flight and returns when all have completed.
 * If `lat_ns` is not NULL, each request's submit-to-reap time is appended at
 * lat_ns[*lat_count] while *lat_count < lat_cap.
 * Returns 0 or a negative errno (-EIO on a short read). */
int tf_reader_read(tf_reader *r, uint64_t offset, size_t len, void *dst,
                   uint32_t *lat_ns, size_t *lat_count, size_t lat_cap);

void tf_reader_close(tf_reader *r);

#ifdef __cplusplus
}
#endif

#endif /* TF_IO_H */
