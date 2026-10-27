/* io_uring O_DIRECT expert reader (TASK-03). Based on the fixed-buffer loop
 * in the platform verification report (Section 4.3), turned into a library:
 * the caller owns the destination region (normally an Arena mapping), so
 * weights land where the GPU and NIC will use them, with no bounce copy. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "tf_io.h"

#define MAX_QD 256

struct tf_reader {
    struct io_uring ring;
    int fd;
    tf_reader_opts opts;
    tf_reader_info info;
    char *region;
    size_t region_len;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

int tf_reader_open(tf_reader **out, const char *path, const tf_reader_opts *opts,
                   void *region, size_t region_len)
{
    if (!opts || opts->queue_depth == 0 || opts->queue_depth > MAX_QD || opts->block_bytes == 0)
        return -EINVAL;

    tf_reader *r = calloc(1, sizeof *r);
    if (!r)
        return -ENOMEM;
    r->opts = *opts;
    r->region = region;
    r->region_len = region_len;

    int rc;
    r->fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (r->fd < 0) {
        rc = -errno;
        free(r);
        return rc;
    }

    struct statx sx = {0};
    if (statx(r->fd, "", AT_EMPTY_PATH, STATX_SIZE | STATX_DIOALIGN, &sx) != 0) {
        rc = -errno;
        goto fail_fd;
    }
    if (!(sx.stx_mask & STATX_DIOALIGN) || !sx.stx_dio_mem_align || !sx.stx_dio_offset_align) {
        rc = -EOPNOTSUPP; /* filesystem does not support direct I/O */
        goto fail_fd;
    }
    r->info.file_bytes = sx.stx_size;
    r->info.dio_mem_align = sx.stx_dio_mem_align;
    r->info.dio_offset_align = sx.stx_dio_offset_align;
    if (opts->block_bytes % sx.stx_dio_offset_align || (uintptr_t)region % sx.stx_dio_mem_align) {
        rc = -EINVAL;
        goto fail_fd;
    }

    struct io_uring_params p = {0};
    if (opts->sqpoll) {
        p.flags = IORING_SETUP_SQPOLL;
        p.sq_thread_idle = 2000;
    }
    rc = io_uring_queue_init_params(opts->queue_depth, &r->ring, &p);
    if (rc < 0 && opts->sqpoll) {
        /* SQPOLL can be refused (older kernels, restricted users); fall back. */
        memset(&p, 0, sizeof p);
        rc = io_uring_queue_init_params(opts->queue_depth, &r->ring, &p);
    }
    if (rc < 0)
        goto fail_fd;
    r->info.sqpoll = (p.flags & IORING_SETUP_SQPOLL) != 0;

    /* Registration pins the region and counts against RLIMIT_MEMLOCK. If it
     * is refused, plain (unregistered) reads still work, just with a per-I/O
     * page lookup. */
    struct iovec iov = {.iov_base = region, .iov_len = region_len};
    r->info.fixed_buffer = io_uring_register_buffers(&r->ring, &iov, 1) == 0;
    r->info.fixed_file = io_uring_register_files(&r->ring, &r->fd, 1) == 0;

    *out = r;
    return 0;

fail_fd:
    close(r->fd);
    free(r);
    return rc;
}

void tf_reader_get_info(const tf_reader *r, tf_reader_info *info)
{
    *info = r->info;
}

int tf_reader_read(tf_reader *r, uint64_t offset, size_t len, void *dst,
                   uint32_t *lat_ns, size_t *lat_count, size_t lat_cap)
{
    char *d = dst;
    if (d < r->region || len > r->region_len || (size_t)(d - r->region) > r->region_len - len)
        return -EFAULT;
    if (offset % r->info.dio_offset_align || len % r->info.dio_offset_align ||
        (uintptr_t)d % r->info.dio_mem_align)
        return -EINVAL;
    if (offset > r->info.file_bytes || len > r->info.file_bytes - offset)
        return -ERANGE;

    const unsigned qd = r->opts.queue_depth;
    uint64_t started[MAX_QD];
    unsigned requested[MAX_QD];
    unsigned char busy[MAX_QD] = {0};
    size_t next = 0, done = 0;
    unsigned inflight = 0;
    int rc = 0;

    while (done < len) {
        unsigned prepared = 0;
        for (unsigned i = 0; i < qd && next < len && rc == 0; i++) {
            if (busy[i])
                continue;
            struct io_uring_sqe *sqe = io_uring_get_sqe(&r->ring);
            if (!sqe)
                break;
            unsigned bytes = len - next < r->opts.block_bytes ? (unsigned)(len - next) : (unsigned)r->opts.block_bytes;
            int file = r->info.fixed_file ? 0 : r->fd;
            if (r->info.fixed_buffer)
                io_uring_prep_read_fixed(sqe, file, d + next, bytes, offset + next, 0);
            else
                io_uring_prep_read(sqe, file, d + next, bytes, offset + next);
            if (r->info.fixed_file)
                sqe->flags |= IOSQE_FIXED_FILE;
            io_uring_sqe_set_data64(sqe, i);
            started[i] = now_ns();
            requested[i] = bytes;
            busy[i] = 1;
            next += bytes;
            inflight++;
            prepared++;
        }
        if (prepared) {
            int s = io_uring_submit(&r->ring);
            if (s < 0)
                return s; /* nothing was submitted, so nothing is in flight to drain */
        }
        if (!inflight)
            break;

        struct io_uring_cqe *cqe;
        int w = io_uring_wait_cqe(&r->ring, &cqe);
        if (w < 0)
            return w;
        do {
            uint64_t id = io_uring_cqe_get_data64(cqe);
            if (id >= qd || !busy[id]) {
                rc = -EPROTO;
            } else {
                if (cqe->res < 0 && rc == 0)
                    rc = cqe->res;
                else if ((unsigned)cqe->res != requested[id] && rc == 0)
                    rc = -EIO;
                uint64_t elapsed = now_ns() - started[id];
                if (lat_ns && lat_count && *lat_count < lat_cap)
                    lat_ns[(*lat_count)++] = elapsed > UINT32_MAX ? UINT32_MAX : (uint32_t)elapsed;
                done += requested[id];
                busy[id] = 0;
                inflight--;
            }
            io_uring_cqe_seen(&r->ring, cqe);
        } while (io_uring_peek_cqe(&r->ring, &cqe) == 0);

        /* After an error, stop issuing and drain what is in flight. */
        if (rc != 0 && inflight == 0)
            return rc;
        if (rc != 0)
            next = len;
    }
    return rc;
}

void tf_reader_close(tf_reader *r)
{
    if (!r)
        return;
    if (r->info.fixed_file)
        io_uring_unregister_files(&r->ring);
    if (r->info.fixed_buffer)
        io_uring_unregister_buffers(&r->ring);
    io_uring_queue_exit(&r->ring);
    close(r->fd);
    free(r);
}
