/* TASK-03 bench for the io_uring expert reader.
 *   uring_bench mk <path> <MiB>        write a deterministic test file
 *   uring_bench read <path> [--bs KiB] [--qd N] [--window MiB] [--sqpoll 0|1] [--verify]
 * `read` streams the whole file through a reusable window (registered as a
 * fixed buffer when RLIMIT_MEMLOCK allows) and prints throughput, request
 * latency percentiles, and how much the page cache grew (O_DIRECT should
 * leave it flat). --verify checks every 8-byte word against the pattern. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "tf_io.h"

static uint64_t pattern(uint64_t word_index)
{
    uint64_t z = word_index + 0x9E3779B97F4A7C15u;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBu;
    return z ^ (z >> 31);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static long meminfo_kib(const char *key)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f)
        return -1;
    char line[256];
    long v = -1;
    size_t klen = strlen(key);
    while (fgets(line, sizeof line, f))
        if (strncmp(line, key, klen) == 0 && line[klen] == ':') {
            v = atol(line + klen + 1);
            break;
        }
    fclose(f);
    return v;
}

static int make_file(const char *path, long mib)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    const size_t chunk = 1u << 20;
    uint64_t *buf = malloc(chunk);
    if (!buf)
        return 1;
    for (long m = 0; m < mib; m++) {
        uint64_t base = (uint64_t)m * (chunk / 8);
        for (size_t i = 0; i < chunk / 8; i++)
            buf[i] = pattern(base + i);
        if (write(fd, buf, chunk) != (ssize_t)chunk) {
            perror("write");
            return 1;
        }
    }
    /* Flush and drop the file from the page cache so reads hit the drive. */
    fsync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    free(buf);
    printf("wrote %ld MiB to %s\n", mib, path);
    return 0;
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static double pct_us(const uint32_t *sorted, size_t n, double p)
{
    size_t i = (size_t)(p * (n - 1));
    return sorted[i] / 1000.0;
}

static int read_file(const char *path, int argc, char **argv)
{
    tf_reader_opts o = {.queue_depth = 64, .block_bytes = 256u << 10, .sqpoll = 1};
    size_t window = 32u << 20;
    int verify = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--verify")) {
            verify = 1;
            continue;
        }
        if (i + 1 >= argc)
            goto usage;
        long v = atol(argv[++i]);
        if (!strcmp(argv[i - 1], "--bs"))
            o.block_bytes = (size_t)v << 10;
        else if (!strcmp(argv[i - 1], "--qd"))
            o.queue_depth = (unsigned)v;
        else if (!strcmp(argv[i - 1], "--window"))
            window = (size_t)v << 20;
        else if (!strcmp(argv[i - 1], "--sqpoll"))
            o.sqpoll = (int)v;
        else
            goto usage;
    }

    char *region = mmap(NULL, window, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (region == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    tf_reader *r;
    int rc = tf_reader_open(&r, path, &o, region, window);
    if (rc < 0) {
        fprintf(stderr, "tf_reader_open: %s\n", strerror(-rc));
        return 1;
    }
    tf_reader_info info;
    tf_reader_get_info(r, &info);
    if (info.file_bytes % window) {
        fprintf(stderr, "file size must be a multiple of the window\n");
        return 1;
    }

    size_t cap = info.file_bytes / o.block_bytes + 1;
    uint32_t *lat = malloc(cap * sizeof *lat);
    size_t nlat = 0;
    uint64_t bad = 0;
    long cached_before = meminfo_kib("Cached");
    double t0 = now_s(), io_s = 0;

    for (uint64_t off = 0; off < info.file_bytes; off += window) {
        double s = now_s();
        rc = tf_reader_read(r, off, window, region, lat, &nlat, cap);
        io_s += now_s() - s;
        if (rc < 0) {
            fprintf(stderr, "read at %llu: %s\n", (unsigned long long)off, strerror(-rc));
            return 1;
        }
        if (verify) {
            const uint64_t *w = (const uint64_t *)region;
            for (size_t i = 0; i < window / 8; i++)
                bad += w[i] != pattern(off / 8 + i);
        }
    }
    double wall = now_s() - t0;
    long cached_after = meminfo_kib("Cached");

    qsort(lat, nlat, sizeof *lat, cmp_u32);
    double gb = info.file_bytes / 1e9;
    printf("bs=%zuKiB qd=%u window=%zuMiB sqpoll=%d fixed_buf=%d fixed_file=%d dio_align=%u/%u\n",
           o.block_bytes >> 10, o.queue_depth, window >> 20, info.sqpoll, info.fixed_buffer,
           info.fixed_file, info.dio_mem_align, info.dio_offset_align);
    printf("read %.2f GB: %.2f GB/s (I/O time), %.2f GB/s (wall%s)\n", gb, gb / io_s, gb / wall,
           verify ? ", includes verify" : "");
    printf("latency us: p50 %.0f  p95 %.0f  p99 %.0f  max %.0f  (%zu requests)\n", pct_us(lat, nlat, 0.50),
           pct_us(lat, nlat, 0.95), pct_us(lat, nlat, 0.99), lat[nlat - 1] / 1000.0, nlat);
    printf("page cache delta: %+ld MiB\n", (cached_after - cached_before) / 1024);
    if (verify)
        printf("verify: %s (%llu bad words)\n", bad ? "FAILED" : "ok", (unsigned long long)bad);

    tf_reader_close(r);
    munmap(region, window);
    free(lat);
    return bad ? 1 : 0;

usage:
    fprintf(stderr, "usage: uring_bench read <path> [--bs KiB] [--qd N] [--window MiB] [--sqpoll 0|1] [--verify]\n");
    return 2;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "mk"))
        return make_file(argv[2], atol(argv[3]));
    if (argc >= 3 && !strcmp(argv[1], "read"))
        return read_file(argv[2], argc - 3, argv + 3);
    fprintf(stderr, "usage: uring_bench mk <path> <MiB> | read <path> [options]\n");
    return 2;
}
