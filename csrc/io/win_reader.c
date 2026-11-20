/* Windows expert reader (TASK-40): the tf_io.h interface of the io_uring one,
 * with unbuffered ReadFile and up to queue_depth overlapped reads in flight. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "tf_io.h"

#define MAX_QD 64

struct tf_reader {
    HANDLE file;
    tf_reader_opts opts;
    tf_reader_info info;
    char *region;
    size_t region_len;
    OVERLAPPED ov[MAX_QD];
};

static int win_errno(DWORD e)
{
    switch (e)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
        return ENOENT;
    case ERROR_ACCESS_DENIED:
        return EACCES;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
        return ENOMEM;
    case ERROR_INVALID_PARAMETER:
        return EINVAL;
    case ERROR_HANDLE_EOF:
        return EIO;
    default:
        return EIO;
    }
}

static uint64_t now_ns(void)
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (uint64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart);
}

/* The volume's physical sector size, which unbuffered reads must align to. */
static DWORD sector_bytes(const char *path)
{
    char full[MAX_PATH];
    char root[MAX_PATH];
    if (!GetFullPathNameA(path, MAX_PATH, full, NULL) || !GetVolumePathNameA(full, root, MAX_PATH))
    {
        return 4096;
    }
    DWORD spc, bps, free_c, total_c;
    if (!GetDiskFreeSpaceA(root, &spc, &bps, &free_c, &total_c) || bps == 0)
    {
        return 4096;
    }
    return bps < 4096 ? 4096 : bps; /* 4096 also covers 4Kn drives behind 512e */
}

int tf_reader_open(tf_reader **out, const char *path, const tf_reader_opts *opts,
                   void *region, size_t region_len)
{
    if (!opts || opts->queue_depth == 0 || opts->block_bytes == 0)
    {
        return -EINVAL;
    }
    tf_reader *r = calloc(1, sizeof *r);
    if (!r)
    {
        return -ENOMEM;
    }
    r->opts = *opts;
    if (r->opts.queue_depth > MAX_QD)
    {
        r->opts.queue_depth = MAX_QD;
    }
    r->region = region;
    r->region_len = region_len;
    r->file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                          FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
    if (r->file == INVALID_HANDLE_VALUE)
    {
        int rc = -win_errno(GetLastError());
        free(r);
        return rc;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(r->file, &size))
    {
        int rc = -win_errno(GetLastError());
        CloseHandle(r->file);
        free(r);
        return rc;
    }
    DWORD align = sector_bytes(path);
    r->info.file_bytes = (uint64_t)size.QuadPart;
    r->info.dio_mem_align = align;
    r->info.dio_offset_align = align;
    if (opts->block_bytes % align || (uintptr_t)region % align)
    {
        CloseHandle(r->file);
        free(r);
        return -EINVAL;
    }
    for (unsigned i = 0; i < r->opts.queue_depth; i++)
    {
        r->ov[i].hEvent = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!r->ov[i].hEvent)
        {
            tf_reader_close(r);
            return -ENOMEM;
        }
    }
    *out = r;
    return 0;
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
    {
        return -EFAULT;
    }
    if (offset % r->info.dio_offset_align || len % r->info.dio_offset_align ||
        (uintptr_t)d % r->info.dio_mem_align)
    {
        return -EINVAL;
    }
    if (offset > r->info.file_bytes || len > r->info.file_bytes - offset)
    {
        return -ERANGE;
    }

    const unsigned qd = r->opts.queue_depth;
    uint64_t started[MAX_QD];
    DWORD requested[MAX_QD];
    unsigned char busy[MAX_QD] = {0};
    size_t next = 0;
    unsigned inflight = 0;
    unsigned oldest = 0;
    int rc = 0;

    while (next < len || inflight)
    {
        for (unsigned k = 0; k < qd && next < len && rc == 0; k++)
        {
            unsigned i = (oldest + inflight) % qd;
            if (busy[i])
            {
                break;
            }
            DWORD bytes = len - next < r->opts.block_bytes ? (DWORD)(len - next) : (DWORD)r->opts.block_bytes;
            OVERLAPPED *o = &r->ov[i];
            uint64_t at = offset + next;
            o->Offset = (DWORD)at;
            o->OffsetHigh = (DWORD)(at >> 32);
            o->Internal = 0;
            o->InternalHigh = 0;
            ResetEvent(o->hEvent);
            started[i] = now_ns();
            requested[i] = bytes;
            if (!ReadFile(r->file, d + next, bytes, NULL, o) && GetLastError() != ERROR_IO_PENDING)
            {
                rc = -win_errno(GetLastError());
                break;
            }
            busy[i] = 1;
            next += bytes;
            inflight++;
        }
        if (!inflight)
        {
            break;
        }
        /* Reap in submission order, so the ring of slots stays contiguous. */
        unsigned i = oldest;
        DWORD got = 0;
        if (!GetOverlappedResult(r->file, &r->ov[i], &got, TRUE))
        {
            if (rc == 0)
            {
                rc = -win_errno(GetLastError());
            }
        }
        else if (got != requested[i] && rc == 0)
        {
            rc = -EIO;
        }
        uint64_t elapsed = now_ns() - started[i];
        if (lat_ns && lat_count && *lat_count < lat_cap)
        {
            lat_ns[(*lat_count)++] = elapsed > UINT32_MAX ? UINT32_MAX : (uint32_t)elapsed;
        }
        busy[i] = 0;
        inflight--;
        oldest = (oldest + 1) % qd;
        if (rc != 0)
        {
            next = len; /* stop issuing and drain what is in flight */
        }
    }
    return rc;
}

void tf_reader_close(tf_reader *r)
{
    if (!r)
    {
        return;
    }
    for (unsigned i = 0; i < MAX_QD; i++)
    {
        if (r->ov[i].hEvent)
        {
            CloseHandle(r->ov[i].hEvent);
        }
    }
    if (r->file && r->file != INVALID_HANDLE_VALUE)
    {
        CloseHandle(r->file);
    }
    free(r);
}
