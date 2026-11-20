//! Child processes and machine facts for the operator dashboard and its node
//! agents (TASK-25). Linux, and Windows (TASK-43) through `win_child.zig` and Win32.
const std = @import("std");
const c = std.c;
const linux = std.os.linux;
const openai = @import("openai.zig");
const plat = @import("plat.zig");
const win32 = @import("win32.zig");
const is_windows = @import("builtin").os.tag == .windows;

pub const FileStat = struct {
    size: u64,
    mode: u32,
};

/// Size and mode of an open file (`path` empty) or of `path`.
pub fn statAt(fd: c.fd_t, path: [*:0]const u8) ?FileStat
{
    var sx: linux.Statx = undefined;
    const empty = path[0] == 0;
    const r = linux.statx(if (empty) fd else linux.AT.FDCWD, path, if (empty) 0x1000 else 0, .{ .TYPE = true, .MODE = true, .SIZE = true }, &sx);
    if (linux.errno(r) != .SUCCESS)
    {
        return null;
    }
    return .{ .size = sx.size, .mode = sx.mode };
}

/// An engine process started by the dashboard or an agent, with its output in a log file.
pub const Child = if (is_windows) @import("win_child.zig").Child else PosixChild;

const PosixChild = struct {
    pid: c.pid_t,
    log_path: []u8,
    started_ns: u64,
    /// Set once the process has exited: its exit code, or 128 plus the signal that ended it.
    exit: ?u8 = null,

    /// Starts `argv` in its own process group with output going to `log_path`. The
    /// child is sent SIGTERM if the thread that started it ends, so call this
    /// from a thread that lives as long as the child should.
    pub fn spawn(gpa: std.mem.Allocator, argv: []const []const u8, log_path: []const u8) !Child
    {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const args = try a.allocSentinel(?[*:0]const u8, argv.len, null);
        for (argv, 0..) |arg, i|
        {
            args[i] = try a.dupeSentinel(u8, arg, 0);
        }
        const log_z = try a.dupeSentinel(u8, log_path, 0);
        const owned_log = try gpa.dupe(u8, log_path);
        errdefer gpa.free(owned_log);
        const pid = c.fork();
        if (pid < 0)
        {
            return error.ForkFailed;
        }
        if (pid == 0)
        {
            // Only async-signal-safe calls between fork and exec.
            _ = c.setpgid(0, 0);
            _ = c.prctl(1, @as(c_ulong, 15));
            const fd = c.open(log_z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o644));
            const nul = c.open("/dev/null", .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
            if (fd < 0 or nul < 0)
            {
                c._exit(126);
            }
            _ = c.dup2(nul, 0);
            _ = c.dup2(fd, 1);
            _ = c.dup2(fd, 2);
            var other: c.fd_t = 3;
            while (other < 4096) : (other += 1)
            {
                _ = c.close(other);
            }
            _ = c.execve(args[0].?, args.ptr, @ptrCast(c.environ));
            c._exit(127);
        }
        // Also from this side, so a stop right away reaches the whole group.
        _ = c.setpgid(pid, pid);
        return .{ .pid = pid, .log_path = owned_log, .started_ns = monoNs() };
    }

    pub fn deinit(self: *Child, gpa: std.mem.Allocator) void
    {
        gpa.free(self.log_path);
    }

    /// Collects the exit status if the process has ended. True while it runs.
    pub fn alive(self: *Child) bool
    {
        if (self.exit != null)
        {
            return false;
        }
        var status: c_int = 0;
        const r = c.waitpid(self.pid, &status, 1);
        if (r == self.pid)
        {
            const s: u32 = @bitCast(status);
            self.exit = if (s & 0x7f == 0) @intCast((s >> 8) & 0xff) else @intCast(128 + (s & 0x7f));
            return false;
        }
        return true;
    }

    /// SIGTERM to the whole group, then SIGKILL after `grace_ms`; waits for the end.
    pub fn stop(self: *Child, grace_ms: u32) void
    {
        if (!self.alive())
        {
            return;
        }
        _ = c.kill(-self.pid, .TERM);
        var waited: u32 = 0;
        while (waited < grace_ms and self.alive())
        {
            sleepMs(50);
            waited += 50;
        }
        if (self.alive())
        {
            _ = c.kill(-self.pid, .KILL);
            while (self.alive())
            {
                sleepMs(20);
            }
        }
    }

    /// True once the log contains `needle`.
    pub fn logHas(self: *const Child, gpa: std.mem.Allocator, needle: []const u8) bool
    {
        const t = tail(gpa, self.log_path, 64 << 10) catch return false;
        defer gpa.free(t);
        return std.mem.indexOf(u8, t, needle) != null;
    }
};

pub fn monoNs() u64
{
    return plat.monoNs();
}

pub fn sleepMs(ms: u32) void
{
    plat.sleepMs(ms);
}

/// The last `max` bytes of a file, starting at a line boundary when cut.
pub fn tail(gpa: std.mem.Allocator, path: []const u8, max: usize) ![]u8
{
    if (is_windows)
    {
        const h = win32.openFile(path, win32.GENERIC_READ, win32.OPEN_EXISTING) orelse return gpa.dupe(u8, "");
        defer _ = win32.CloseHandle(h);
        var bytes: i64 = 0;
        if (win32.GetFileSizeEx(h, &bytes) == 0)
        {
            return error.ReadFailed;
        }
        const size: usize = @intCast(bytes);
        const start = if (size > max) size - max else 0;
        const buf = try gpa.alloc(u8, size - start);
        defer gpa.free(buf);
        const n = win32.readAt(h, start, buf) orelse return error.ReadFailed;
        return tailFrom(gpa, buf[0..n], start);
    }
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0)
    {
        return gpa.dupe(u8, "");
    }
    defer _ = c.close(fd);
    const st = statAt(fd, "") orelse return error.ReadFailed;
    const size: usize = @intCast(st.size);
    const start = if (size > max) size - max else 0;
    if (size == start)
    {
        return gpa.dupe(u8, "");
    }
    const buf = try gpa.alloc(u8, size - start);
    defer gpa.free(buf);
    const n = c.pread(fd, buf.ptr, buf.len, @intCast(start));
    if (n < 0)
    {
        return error.ReadFailed;
    }
    return tailFrom(gpa, buf[0..@intCast(n)], start);
}

/// A copy of `got`, read from offset `start`, without the partial first line when cut.
fn tailFrom(gpa: std.mem.Allocator, bytes: []const u8, start: usize) ![]u8
{
    var got = bytes;
    if (start > 0)
    {
        if (std.mem.indexOfScalar(u8, got, '\n')) |i|
        {
            got = got[i + 1 ..];
        }
    }
    return gpa.dupe(u8, got);
}

/// A whole small file (up to 1 MB), or null if it cannot be read. Reads to the
/// end instead of trusting the size, which is 0 for /proc and /sys files.
pub fn readSmall(gpa: std.mem.Allocator, path: []const u8) ?[]u8
{
    if (is_windows)
    {
        const h = win32.openFile(path, win32.GENERIC_READ, win32.OPEN_EXISTING) orelse return null;
        defer _ = win32.CloseHandle(h);
        var wout: std.ArrayList(u8) = .empty;
        while (wout.items.len < 1 << 20)
        {
            wout.ensureUnusedCapacity(gpa, 16 << 10) catch break;
            const n = win32.readAt(h, wout.items.len, wout.unusedCapacitySlice()) orelse break;
            if (n == 0)
            {
                break;
            }
            wout.items.len += n;
        }
        return wout.toOwnedSlice(gpa) catch null;
    }
    const z = gpa.dupeSentinel(u8, path, 0) catch return null;
    defer gpa.free(z);
    const fd = c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0)
    {
        return null;
    }
    defer _ = c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    while (out.items.len < 1 << 20)
    {
        out.ensureUnusedCapacity(gpa, 16 << 10) catch break;
        const n = c.read(fd, out.unusedCapacitySlice().ptr, out.unusedCapacitySlice().len);
        if (n <= 0)
        {
            break;
        }
        out.items.len += @intCast(n);
    }
    return out.toOwnedSlice(gpa) catch null;
}

pub const ModelFile = struct {
    name: []const u8,
    bytes: u64,
};

/// A model name the agents accept: a plain file name ending in .gguf.
pub fn validModelName(name: []const u8) bool
{
    if (name.len < 6 or name.len > 200 or name[0] == '.' or !std.mem.endsWith(u8, name, ".gguf"))
    {
        return false;
    }
    for (name) |ch|
    {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-' or ch == '_'))
        {
            return false;
        }
    }
    return true;
}

fn fileSize(gpa: std.mem.Allocator, path: []const u8) ?u64
{
    if (is_windows)
    {
        return plat.regularFileBytes(path);
    }
    const z = gpa.dupeSentinel(u8, path, 0) catch return null;
    defer gpa.free(z);
    const st = statAt(-1, z) orelse return null;
    if (st.mode & 0o170000 != 0o100000)
    {
        return null;
    }
    return st.size;
}

/// Size of `dir/name` if it is a model the engine can load (with its .tfsum index).
pub fn modelSize(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) ?u64
{
    if (!validModelName(name))
    {
        return null;
    }
    const path = std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, name }) catch return null;
    defer gpa.free(path);
    const sum = std.fmt.allocPrint(gpa, "{s}.tfsum", .{path}) catch return null;
    defer gpa.free(sum);
    const size = fileSize(gpa, path) orelse return null;
    _ = fileSize(gpa, sum) orelse return null;
    return size;
}

/// The loadable models in `dir`, sorted by name.
pub fn listModels(a: std.mem.Allocator, dir: []const u8) ![]ModelFile
{
    var out: std.ArrayList(ModelFile) = .empty;
    const names = try fileNames(a, dir);
    for (names) |name|
    {
        if (modelSize(a, dir, name)) |size|
        {
            try out.append(a, .{ .name = name, .bytes = size });
        }
    }
    std.mem.sort(ModelFile, out.items, {}, struct {
        fn lt(_: void, x: ModelFile, y: ModelFile) bool
        {
            return std.mem.order(u8, x.name, y.name) == .lt;
        }
    }.lt);
    return out.items;
}

/// The names in folder `dir`, without "." and "..".
fn fileNames(a: std.mem.Allocator, dir: []const u8) ![][]const u8
{
    var out: std.ArrayList([]const u8) = .empty;
    if (is_windows)
    {
        var wide: [1024:0]u16 = undefined;
        const pattern = try std.fmt.allocPrint(a, "{s}\\*", .{dir});
        const w = win32.widePath(pattern, &wide) orelse return out.items;
        var data: win32.WIN32_FIND_DATAW = undefined;
        const find = win32.FindFirstFileW(w, &data);
        if (find == win32.INVALID_HANDLE_VALUE)
        {
            return out.items;
        }
        defer _ = win32.FindClose(find);
        while (true)
        {
            const len = std.mem.indexOfScalar(u16, &data.name, 0) orelse data.name.len;
            const name = try std.unicode.utf16LeToUtf8Alloc(a, data.name[0..len]);
            if (!std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, ".."))
            {
                try out.append(a, name);
            }
            if (win32.FindNextFileW(find, &data) == 0)
            {
                break;
            }
        }
        return out.items;
    }
    const z = try a.dupeSentinel(u8, dir, 0);
    const d = c.opendir(z) orelse return out.items;
    defer _ = c.closedir(d);
    while (c.readdir(d)) |ent|
    {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (!std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, ".."))
        {
            try out.append(a, try a.dupe(u8, name));
        }
    }
    return out.items;
}

pub const Rdma = struct {
    dev: []const u8,
    /// IPv4 addresses (network order) from the device's RoCE GID table.
    ips: []u32,
};

/// The machine's RDMA devices and their IPv4 addresses, read from sysfs.
pub fn rdmaDevices(a: std.mem.Allocator) ![]Rdma
{
    var out: std.ArrayList(Rdma) = .empty;
    if (is_windows)
    {
        return out.items; // RDMA is Linux only
    }
    const d = c.opendir("/sys/class/infiniband") orelse return out.items;
    defer _ = c.closedir(d);
    while (c.readdir(d)) |ent|
    {
        const dev = std.mem.sliceTo(&ent.name, 0);
        if (dev[0] == '.' or !validDevice(dev))
        {
            continue;
        }
        var ips: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < 16) : (i += 1)
        {
            const path = try std.fmt.allocPrint(a, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ dev, i });
            const text = readSmall(a, path) orelse break;
            if (ipFromGid(std.mem.trim(u8, text, " \n"))) |ip|
            {
                if (std.mem.indexOfScalar(u32, ips.items, ip) == null)
                {
                    try ips.append(a, ip);
                }
            }
        }
        try out.append(a, .{ .dev = try a.dupe(u8, dev), .ips = ips.items });
    }
    return out.items;
}

/// An RDMA device name as the agents accept it.
pub fn validDevice(dev: []const u8) bool
{
    if (dev.len == 0 or dev.len > 32)
    {
        return false;
    }
    for (dev) |ch|
    {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '-'))
        {
            return false;
        }
    }
    return true;
}

/// The IPv4 address in an IPv4-mapped GID ("0000:...:ffff:c0a8:c50d").
pub fn ipFromGid(gid: []const u8) ?u32
{
    const prefix = "0000:0000:0000:0000:0000:ffff:";
    if (gid.len != prefix.len + 9 or !std.mem.startsWith(u8, gid, prefix) or gid[prefix.len + 4] != ':')
    {
        return null;
    }
    const hi = std.fmt.parseInt(u16, gid[prefix.len..][0..4], 16) catch return null;
    const lo = std.fmt.parseInt(u16, gid[prefix.len + 5 ..][0..4], 16) catch return null;
    const b: [4]u8 = .{ @intCast(hi >> 8), @intCast(hi & 0xff), @intCast(lo >> 8), @intCast(lo & 0xff) };
    return @bitCast(b);
}

/// Bits two IPv4 addresses (network order) share from the top.
pub fn commonPrefix(x: u32, y: u32) u6
{
    const a = std.mem.bigToNative(u32, x);
    const b = std.mem.bigToNative(u32, y);
    return @intCast(@clz(a ^ b));
}

pub const Memory = struct {
    total_mb: u64 = 0,
    available_mb: u64 = 0,
};

pub fn memory(gpa: std.mem.Allocator) Memory
{
    var m: Memory = .{};
    if (is_windows)
    {
        var st: win32.MEMORYSTATUSEX = .{};
        if (win32.GlobalMemoryStatusEx(&st) != 0)
        {
            m.total_mb = st.total_phys >> 20;
            m.available_mb = st.avail_phys >> 20;
        }
        return m;
    }
    const text = readSmall(gpa, "/proc/meminfo") orelse return m;
    defer gpa.free(text);
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line|
    {
        const field = if (std.mem.startsWith(u8, line, "MemTotal:")) &m.total_mb else if (std.mem.startsWith(u8, line, "MemAvailable:")) &m.available_mb else continue;
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        _ = words.next();
        const kb = std.fmt.parseInt(u64, words.next() orelse "0", 10) catch 0;
        field.* = kb / 1024;
    }
    return m;
}

/// The first NVIDIA GPU's model name, from the driver's proc files.
pub fn gpuName(a: std.mem.Allocator) ?[]const u8
{
    if (is_windows)
    {
        // The display adapters Windows knows; the first NVIDIA one.
        var i: u32 = 0;
        while (i < 16) : (i += 1)
        {
            var dd: win32.DISPLAY_DEVICEA = .{};
            if (win32.EnumDisplayDevicesA(null, i, &dd, 0) == 0)
            {
                break;
            }
            const name = std.mem.sliceTo(&dd.string, 0);
            if (std.mem.startsWith(u8, name, "NVIDIA "))
            {
                return std.mem.trim(u8, a.dupe(u8, name["NVIDIA ".len..]) catch return null, " ");
            }
        }
        return null;
    }
    const d = c.opendir("/proc/driver/nvidia/gpus") orelse return null;
    defer _ = c.closedir(d);
    while (c.readdir(d)) |ent|
    {
        const bus = std.mem.sliceTo(&ent.name, 0);
        if (bus[0] == '.')
        {
            continue;
        }
        const path = std.fmt.allocPrint(a, "/proc/driver/nvidia/gpus/{s}/information", .{bus}) catch return null;
        const text = readSmall(a, path) orelse continue;
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line|
        {
            if (std.mem.startsWith(u8, line, "Model:"))
            {
                return std.mem.trim(u8, line[6..], " \t");
            }
        }
    }
    return null;
}

pub fn hostName(buf: *[64]u8) []const u8
{
    @memset(buf, 0);
    return plat.hostName(buf) orelse "";
}

pub fn formatIp4(addr: u32, buf: *[16]u8) []const u8
{
    const b: [4]u8 = @bitCast(addr);
    return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ b[0], b[1], b[2], b[3] }) catch unreachable;
}

/// Appends `s` as a JSON string.
pub fn jsonString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void
{
    try openai.writeString(a, out, s);
}

test "reads IPv4 addresses from RoCE GIDs" {
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("192.168.197.13", formatIp4(ipFromGid("0000:0000:0000:0000:0000:ffff:c0a8:c50d").?, &buf));
    try std.testing.expect(ipFromGid("fe80:0000:0000:0000:4a2c:d6ff:fe00:0001") == null);
    const a: u32 = @bitCast([4]u8{ 192, 168, 197, 13 });
    const b: u32 = @bitCast([4]u8{ 192, 168, 197, 12 });
    const o: u32 = @bitCast([4]u8{ 192, 168, 177, 11 });
    try std.testing.expect(commonPrefix(a, b) > commonPrefix(a, o));
}

test "accepts only plain model file names" {
    try std.testing.expect(validModelName("olmoe-1b-7b-q8_0.gguf"));
    try std.testing.expect(!validModelName("../etc/passwd.gguf"));
    try std.testing.expect(!validModelName("a/b.gguf"));
    try std.testing.expect(!validModelName(".hidden.gguf"));
    try std.testing.expect(!validModelName("model.bin"));
    try std.testing.expect(!validModelName("x y.gguf"));
}

test "a child's exit code and log are collected" {
    const gpa = std.testing.allocator;
    if (is_windows)
    {
        const temp = plat.env("TEMP") orelse return error.SkipZigTest;
        const log1 = try std.fmt.allocPrint(gpa, "{s}\\tf_proc_test.log", .{temp});
        defer gpa.free(log1);
        const log2 = try std.fmt.allocPrint(gpa, "{s}\\tf_proc_test2.log", .{temp});
        defer gpa.free(log2);
        var ch = try Child.spawn(gpa, &.{ "cmd.exe", "/c", "echo listening on port 5 & exit 3" }, log1);
        defer ch.deinit(gpa);
        var n: usize = 0;
        while (ch.alive() and n < 500) : (n += 1)
        {
            sleepMs(10);
        }
        try std.testing.expectEqual(@as(?u8, 3), ch.exit);
        try std.testing.expect(ch.logHas(gpa, "listening on port"));
        var long = try Child.spawn(gpa, &.{ "cmd.exe", "/c", "ping -n 30 127.0.0.1" }, log2);
        defer long.deinit(gpa);
        long.stop(2000);
        try std.testing.expectEqual(@as(?u8, 128 + 15), long.exit);
        return;
    }
    var ch = try Child.spawn(gpa, &.{ "/bin/sh", "-c", "echo listening on port 5; exit 3" }, "/tmp/tf_proc_test.log");
    defer ch.deinit(gpa);
    var n: usize = 0;
    while (ch.alive() and n < 200) : (n += 1)
    {
        sleepMs(10);
    }
    try std.testing.expectEqual(@as(?u8, 3), ch.exit);
    try std.testing.expect(ch.logHas(gpa, "listening on port"));
    var long = try Child.spawn(gpa, &.{ "/bin/sh", "-c", "sleep 30" }, "/tmp/tf_proc_test2.log");
    defer long.deinit(gpa);
    long.stop(2000);
    try std.testing.expectEqual(@as(?u8, 128 + 15), long.exit);
}
