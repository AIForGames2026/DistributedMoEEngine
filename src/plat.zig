//! The few process and file calls `tfswarm` makes outside the model reader, on
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

/// A lock between threads: a pthread mutex, or a Win32 slim reader/writer lock.
pub const Mutex = if (is_windows) struct {
    lock: ?*anyopaque = null,

    pub fn acquire(self: *Mutex) void
    {
        win32.AcquireSRWLockExclusive(&self.lock);
    }
    pub fn release(self: *Mutex) void
    {
        win32.ReleaseSRWLockExclusive(&self.lock);
    }
} else struct {
    m: c.pthread_mutex_t = .{},

    pub fn acquire(self: *Mutex) void
    {
        _ = c.pthread_mutex_lock(&self.m);
    }
    pub fn release(self: *Mutex) void
    {
        _ = c.pthread_mutex_unlock(&self.m);
    }
};

/// Fills `buf` from the system's cryptographic random source.
pub fn randomBytes(buf: []u8) void
{
    if (is_windows)
    {
        var off: usize = 0;
        while (off < buf.len)
        {
            const n: u32 = @intCast(@min(buf.len - off, 1 << 20));
            if (win32.SystemFunction036(buf[off..].ptr, n) != 0)
            {
                off += n;
            }
        }
        return;
    }
    var off: usize = 0;
    while (off < buf.len)
    {
        const n = std.os.linux.getrandom(buf[off..].ptr, buf.len - off, 0);
        if (@as(isize, @bitCast(n)) <= 0)
        {
            continue;
        }
        off += n;
    }
}

/// Removes a file if it is there.
pub fn deleteFile(path: [:0]const u8) void
{
    if (is_windows)
    {
        var wide: [1024:0]u16 = undefined;
        const w = win32.widePath(path, &wide) orelse return;
        _ = win32.DeleteFileW(w);
        return;
    }
    _ = c.unlink(path);
}

/// A monotonic clock in nanoseconds.
pub fn monoNs() u64
{
    if (is_windows)
    {
        return win32.nowNs();
    }
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Milliseconds since 1970.
pub fn unixMs() u64
{
    if (is_windows)
    {
        var ft: u64 = 0;
        win32.GetSystemTimeAsFileTime(&ft);
        return ft / 10_000 - 11_644_473_600_000;
    }
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / std.time.ns_per_ms;
}

/// The absolute path of an existing file or folder, in `buf`; null if it does not exist.
pub fn realPath(path: [:0]const u8, buf: *[4096]u8) ?[]const u8
{
    if (is_windows)
    {
        var wide: [1024:0]u16 = undefined;
        const w = win32.widePath(path, &wide) orelse return null;
        var full: [1024]u16 = undefined;
        const n = win32.GetFullPathNameW(w, full.len, &full, null);
        if (n == 0 or n >= full.len)
        {
            return null;
        }
        full[n] = 0;
        // Expands 8.3 short names (C:\Users\LUKEHI~1), so one folder has one spelling.
        var long: [1024]u16 = undefined;
        const m = win32.GetLongPathNameW(full[0..n :0], &long, long.len);
        if (m == 0 or m >= long.len)
        {
            return null;
        }
        const len = std.unicode.utf16LeToUtf8(buf, long[0..m]) catch return null;
        return buf[0..len];
    }
    const r = c.realpath(path, buf) orelse return null;
    return std.mem.sliceTo(r, 0);
}

/// The path of this program: as given on Linux (`/proc/self/exe` resolves it), else this executable's.
pub fn selfExe(given: [:0]const u8, buf: *[4096]u8) ?[]const u8
{
    if (is_windows and std.mem.eql(u8, given, "/proc/self/exe"))
    {
        var wide: [1024]u16 = undefined;
        const n = win32.GetModuleFileNameW(null, &wide, wide.len);
        if (n == 0 or n >= wide.len)
        {
            return null;
        }
        const len = std.unicode.utf16LeToUtf8(buf, wide[0..n]) catch return null;
        return buf[0..len];
    }
    return realPath(given, buf);
}

/// Makes the folder if it is not there yet.
pub fn makeDir(path: [:0]const u8) void
{
    if (is_windows)
    {
        var wide: [1024:0]u16 = undefined;
        const w = win32.widePath(path, &wide) orelse return;
        _ = win32.CreateDirectoryW(w, null);
        return;
    }
    _ = c.mkdir(path, 0o755);
}

/// Writes `bytes` to `tmp` (owner-only on Linux), then renames it over `dst`.
pub fn replaceFile(tmp: [:0]const u8, dst: [:0]const u8, bytes: []const u8) bool
{
    if (is_windows)
    {
        writeFile(tmp, bytes) catch return false;
        var wt: [1024:0]u16 = undefined;
        var wd: [1024:0]u16 = undefined;
        const a = win32.widePath(tmp, &wt) orelse return false;
        const b = win32.widePath(dst, &wd) orelse return false;
        return win32.MoveFileExW(a, b, win32.MOVEFILE_REPLACE_EXISTING) != 0;
    }
    const fd = c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    if (fd < 0)
    {
        return false;
    }
    const ok = c.write(fd, bytes.ptr, bytes.len) == @as(isize, @intCast(bytes.len));
    _ = c.close(fd);
    if (ok)
    {
        _ = c.rename(tmp, dst);
    }
    return ok;
}

/// Size of a regular file; null for a folder or a missing file. Windows only (Linux uses `proc.statAt`).
pub fn regularFileBytes(path: []const u8) ?u64
{
    var wide: [1024:0]u16 = undefined;
    const w = win32.widePath(path, &wide) orelse return null;
    var attr: win32.FILE_ATTRIBUTE_DATA = undefined;
    if (win32.GetFileAttributesExW(w, 0, &attr) == 0 or attr.attributes & win32.FILE_ATTRIBUTE_DIRECTORY != 0)
    {
        return null;
    }
    return (@as(u64, attr.size_high) << 32) | attr.size_low;
}

/// Windows: true if the existing file or folder `path` is the user's profile folder or
/// inside it; its default access list admits only the user, SYSTEM and Administrators.
pub fn inUserProfile(path: [:0]const u8) bool
{
    var buf: [4096]u8 = undefined;
    const full = realPath(path, &buf) orelse return false;
    var profile_z: [1024:0]u8 = undefined;
    const profile_env = env("USERPROFILE") orelse return false;
    if (profile_env.len >= profile_z.len)
    {
        return false;
    }
    @memcpy(profile_z[0..profile_env.len], profile_env);
    profile_z[profile_env.len] = 0;
    var profile_buf: [4096]u8 = undefined;
    const profile = realPath(profile_z[0..profile_env.len :0], &profile_buf) orelse return false;
    if (full.len < profile.len or !std.ascii.eqlIgnoreCase(full[0..profile.len], profile))
    {
        return false;
    }
    return full.len == profile.len or full[profile.len] == '\\' or full[profile.len] == '/';
}

pub fn sleepMs(ms: u32) void
{
    if (is_windows)
    {
        win32.Sleep(ms);
        return;
    }
    const ts: c.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * 1_000_000) };
    _ = c.nanosleep(&ts, null);
}

/// A file created or replaced for writing, kept open (a recording).
pub const OutFile = struct {
    h: if (is_windows) win32.HANDLE else c.fd_t,

    pub fn create(path: [:0]const u8) ?OutFile
    {
        if (is_windows)
        {
            return .{ .h = win32.openFile(path, win32.GENERIC_WRITE, win32.CREATE_ALWAYS) orelse return null };
        }
        const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o644));
        return if (fd < 0) null else .{ .h = fd };
    }

    /// Writes all of `bytes`; false if the file refused some.
    pub fn write(self: OutFile, bytes: []const u8) bool
    {
        var off: usize = 0;
        while (off < bytes.len)
        {
            if (is_windows)
            {
                var n: win32.DWORD = 0;
                const len: win32.DWORD = @intCast(@min(bytes.len - off, 1 << 30));
                if (win32.WriteFile(self.h, bytes[off..].ptr, len, &n, null) == 0 or n == 0)
                {
                    return false;
                }
                off += n;
                continue;
            }
            const n = c.write(self.h, bytes[off..].ptr, bytes.len - off);
            if (n <= 0)
            {
                return false;
            }
            off += @intCast(n);
        }
        return true;
    }

    pub fn close(self: OutFile) void
    {
        if (is_windows)
        {
            _ = win32.CloseHandle(self.h);
            return;
        }
        _ = c.close(self.h);
    }
};

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
