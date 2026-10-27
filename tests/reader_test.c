/* Unit test for the io_uring expert reader (TASK-03). Writes a small pattern
 * file under $TMPDIR (default /tmp), reads it back through the reader in
 * several configurations, and checks the error contract. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "tf_io.h"

#define FILE_BYTES (4u << 20)
#define REGION_BYTES (1u << 20)

static int failures;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                \
        }                                                              \
    } while (0)

static uint64_t word(uint64_t i)
{
    return i * 0x9E3779B97F4A7C15u ^ (i >> 7);
}

static int read_all(const char *path, unsigned qd, size_t bs, int sqpoll, char *region)
{
    tf_reader *r;
    tf_reader_opts o = {.queue_depth = qd, .block_bytes = bs, .sqpoll = sqpoll};
    int rc = tf_reader_open(&r, path, &o, region, REGION_BYTES);
    if (rc < 0)
        return rc;
    uint32_t lat[64];
    size_t n = 0;
    for (uint64_t off = 0; off < FILE_BYTES && rc == 0; off += REGION_BYTES) {
        memset(region, 0, REGION_BYTES);
        rc = tf_reader_read(r, off, REGION_BYTES, region, lat, &n, 64);
        const uint64_t *w = (const uint64_t *)region;
        for (size_t i = 0; i < REGION_BYTES / 8 && rc == 0; i++)
            if (w[i] != word(off / 8 + i))
                rc = -EBADMSG;
    }
    size_t requests = FILE_BYTES / bs;
    CHECK(rc != 0 || n == (requests < 64 ? requests : 64)); /* one sample per request, up to the cap */
    tf_reader_close(r);
    return rc;
}

int main(void)
{
    const char *tmp = getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp";
    char path[512];
    snprintf(path, sizeof path, "%s/tf_reader_test.%d", tmp, (int)getpid());

    uint64_t *buf = malloc(FILE_BYTES);
    for (size_t i = 0; i < FILE_BYTES / 8; i++)
        buf[i] = word(i);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(fd >= 0 && write(fd, buf, FILE_BYTES) == FILE_BYTES);
    fsync(fd);
    close(fd);
    free(buf);

    char *region = mmap(NULL, REGION_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(region != MAP_FAILED);

    CHECK(read_all(path, 8, 64u << 10, 0, region) == 0);
    CHECK(read_all(path, 32, 256u << 10, 1, region) == 0);
    CHECK(read_all(path, 1, 1u << 20, 0, region) == 0);

    tf_reader *r;
    tf_reader_opts o = {.queue_depth = 4, .block_bytes = 64u << 10, .sqpoll = 0};
    CHECK(tf_reader_open(&r, path, &o, region, REGION_BYTES) == 0);
    tf_reader_info info;
    tf_reader_get_info(r, &info);
    CHECK(info.file_bytes == FILE_BYTES);
    CHECK(info.dio_offset_align > 0 && info.dio_mem_align > 0);
    CHECK(tf_reader_read(r, 1, 4096, region, NULL, NULL, 0) == -EINVAL);              /* misaligned offset */
    CHECK(tf_reader_read(r, 0, 4096, region + REGION_BYTES, NULL, NULL, 0) == -EFAULT); /* outside region */
    CHECK(tf_reader_read(r, FILE_BYTES, 4096, region, NULL, NULL, 0) == -ERANGE);     /* past end of file */
    CHECK(tf_reader_read(r, 4096, 8192, region, NULL, NULL, 0) == 0);
    CHECK(((uint64_t *)region)[0] == word(4096 / 8));
    tf_reader_close(r);

    o.block_bytes = 1000; /* not a multiple of the DIO alignment */
    CHECK(tf_reader_open(&r, path, &o, region, REGION_BYTES) == -EINVAL);
    CHECK(tf_reader_open(&r, "/nonexistent/tf", &o, region, REGION_BYTES) == -ENOENT);

    munmap(region, REGION_BYTES);
    unlink(path);
    if (failures == 0)
        printf("reader_test: ok (fixed_buffer=%d, fixed_file=%d)\n", info.fixed_buffer, info.fixed_file);
    return failures != 0;
}
