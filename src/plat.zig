//! The few process and file calls `olmoe` makes outside the model reader, on
//! Linux through libc and on Windows through Win32 (TASK-40).
const std = @import("std");
const c = std.c;
const is_windows = @import("builtin").os.tag == .windows;
const win32 = @import("win32.zig");

extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// An environment variable, or null when it is not set.
pub fn env(name: [*:0]const u8) ?[]const u8
{
    const v = if (is_windows) getenv(name) else c.getenv(name);
    return if (v) |p| std.mem.sliceTo(p, 0) else null;
}

pub fn pid() u32
{
    if (is_windows)
    {
        return win32.GetCurrentProcessId();
    }
    return @intCast(c.getpid());
}

/// Seconds since 1970.
pub fn unixNow() i64
{
    if (is_windows)
    {
        var ft: u64 = 0;
        win32.GetSystemTimeAsFileTime(&ft);
        return @intCast(ft / 10_000_000 - 11_644_473_600); // FILETIME counts 100 ns from 1601
    }
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @intCast(ts.sec);
}

/// This machine's name, cut to fit `buf`; null if it cannot be read.
pub fn hostName(buf: []u8) ?[]const u8
{
    if (is_windows)
    {
        var n: win32.DWORD = @intCast(buf.len);
        if (win32.GetComputerNameA(buf.ptr, &n) == 0)
        {
            return null;
        }
        return buf[0..n];
    }
    if (c.gethostname(buf.ptr, buf.len - 1) != 0)
    {
        return null;
    }
    buf[buf.len - 1] = 0;
    return std.mem.sliceTo(buf, 0);
}

/// A file's size in bytes, or null if it cannot be opened.
pub fn fileBytes(path: [:0]const u8) ?u64
{
    if (is_windows)
    {
        const h = win32.openFile(path, win32.GENERIC_READ, win32.OPEN_EXISTING) orelse return null;
        defer _ = win32.CloseHandle(h);
        var size: i64 = 0;
        return if (win32.GetFileSizeEx(h, &size) != 0) @intCast(size) else null;
    }
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0)
    {
        return null;
    }
    defer _ = c.close(fd);
    const size = c.lseek(fd, 0, c.SEEK.END);
    return if (size < 0) null else @intCast(size);
}

/// Creates or replaces `path` with `bytes`.
pub fn writeFile(path: [:0]const u8, bytes: []const u8) !void
{
    if (is_windows)
    {
        const h = win32.openFile(path, win32.GENERIC_WRITE, win32.CREATE_ALWAYS) orelse return error.WriteFailed;
        defer _ = win32.CloseHandle(h);
        var off: usize = 0;
        while (off < bytes.len)
        {
            var n: win32.DWORD = 0;
            const len: win32.DWORD = @intCast(@min(bytes.len - off, 1 << 30));
            if (win32.WriteFile(h, bytes[off..].ptr, len, &n, null) == 0 or n == 0)
            {
                return error.WriteFailed;
            }
            off += n;
        }
        return;
    }
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o644));
    if (fd < 0)
    {
        return error.WriteFailed;
    }
    defer _ = c.close(fd);
    var off: usize = 0;
    while (off < bytes.len)
    {
        const n = c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n <= 0)
        {
            return error.WriteFailed;
        }
        off += @intCast(n);
    }
}

/// Resident memory of this process in bytes, or 0 if it cannot be read.
pub fn rssBytes() u64
{
    if (is_windows)
    {
        var pmc: win32.PROCESS_MEMORY_COUNTERS = .{};
        pmc.cb = @sizeOf(win32.PROCESS_MEMORY_COUNTERS);
        if (win32.K32GetProcessMemoryInfo(win32.GetCurrentProcess(), &pmc, pmc.cb) == 0)
        {
            return 0;
        }
        return pmc.working_set;
    }
    const fd = c.open("/proc/self/statm", .{ .ACCMODE = .RDONLY });
    if (fd < 0)
    {
        return 0;
    }
    defer _ = c.close(fd);
    var buf: [128]u8 = undefined;
    const n = c.read(fd, &buf, buf.len);
    if (n <= 0)
    {
        return 0;
    }
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(n)], ' ');
    _ = it.next();
    const pages = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return pages * std.heap.pageSize();
}
