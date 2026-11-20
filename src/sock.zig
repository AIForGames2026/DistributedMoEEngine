//! Sockets for the server (`serve.zig`) and its live feed (`telemetry.zig`):
//! BSD sockets on Linux, Winsock on Windows (TASK-40). On Windows a loopback
//! socket pair stands in for the pipe between threads, since WSAPoll waits on sockets only.
const std = @import("std");
const c = std.c;
const is_windows = @import("builtin").os.tag == .windows;
const win32 = @import("win32.zig");
const ws = std.os.windows.ws2_32;

pub const Fd = if (is_windows) win32.SOCKET else c.fd_t;
pub const invalid: Fd = if (is_windows) win32.INVALID_SOCKET else -1;

pub const PollFd = if (is_windows) win32.WSAPOLLFD else c.pollfd;
pub const POLL_IN: i16 = if (is_windows) win32.POLLIN else c.POLL.IN;
pub const POLL_OUT: i16 = if (is_windows) 0x0010 else c.POLL.OUT;
pub const POLL_HUP: i16 = if (is_windows) 0x0002 else c.POLL.HUP;
pub const POLL_ERR: i16 = if (is_windows) 0x0001 else c.POLL.ERR;

extern "ws2_32" fn ioctlsocket(s: win32.SOCKET, cmd: c_long, arg: *c_ulong) callconv(.winapi) c_int;
const FIONBIO: c_long = @bitCast(@as(c_ulong, 0x8004667E));
const WSAEWOULDBLOCK = 10035;

var started = std.atomic.Value(bool).init(false);

fn startup() !void
{
    if (is_windows and !started.swap(true, .acq_rel))
    {
        try win32.startWinsock();
    }
}

pub fn close(fd: Fd) void
{
    if (is_windows)
    {
        _ = win32.closesocket(fd);
        return;
    }
    _ = c.close(fd);
}

/// A listening TCP socket on `addr` (network byte order) and `port`.
pub fn listenTcp(addr: u32, port: u16, backlog: c_int) !Fd
{
    try startup();
    if (is_windows)
    {
        const fd = win32.socket(ws.AF.INET, ws.SOCK.STREAM, ws.IPPROTO.TCP);
        if (fd == win32.INVALID_SOCKET)
        {
            return error.SystemResources;
        }
        errdefer close(fd);
        // No SO_REUSEADDR: on Windows it lets another socket take over a bound port.
        const sa: ws.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
        if (win32.bind(fd, &sa, @sizeOf(ws.sockaddr.in)) != 0)
        {
            return error.AddressInUse;
        }
        if (win32.listen(fd, backlog) != 0)
        {
            return error.SystemResources;
        }
        return fd;
    }
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM | c.SOCK.CLOEXEC, 0);
    if (fd < 0)
    {
        return error.SystemResources;
    }
    errdefer _ = c.close(fd);
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, &one, @sizeOf(c_int));
    var sa: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
    if (c.bind(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0)
    {
        return error.AddressInUse;
    }
    if (c.listen(fd, @intCast(backlog)) != 0)
    {
        return error.SystemResources;
    }
    return fd;
}

pub fn accept(lfd: Fd) ?Fd
{
    if (is_windows)
    {
        const fd = win32.accept(lfd, null, null);
        return if (fd == win32.INVALID_SOCKET) null else fd;
    }
    const fd = c.accept(lfd, null, null);
    return if (fd < 0) null else fd;
}

/// Receive and send timeouts, in seconds.
pub fn setTimeouts(fd: Fd, recv_s: u32, send_s: u32) void
{
    if (is_windows)
    {
        const rcv: u32 = recv_s * 1000;
        const snd: u32 = send_s * 1000;
        _ = win32.setsockopt(fd, ws.SOL.SOCKET, ws.SO.RCVTIMEO, &rcv, @sizeOf(u32));
        _ = win32.setsockopt(fd, ws.SOL.SOCKET, ws.SO.SNDTIMEO, &snd, @sizeOf(u32));
        return;
    }
    const rcv: c.timeval = .{ .sec = @intCast(recv_s), .usec = 0 };
    const snd: c.timeval = .{ .sec = @intCast(send_s), .usec = 0 };
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, std.mem.asBytes(&rcv), @sizeOf(c.timeval));
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, std.mem.asBytes(&snd), @sizeOf(c.timeval));
}

pub fn setSendBuffer(fd: Fd, bytes: c_int) void
{
    if (is_windows)
    {
        _ = win32.setsockopt(fd, ws.SOL.SOCKET, ws.SO.SNDBUF, &bytes, @sizeOf(c_int));
        return;
    }
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDBUF, std.mem.asBytes(&bytes), @sizeOf(c_int));
}

/// Makes `sendNow` and `recvNow` return at once on Windows; Linux asks per call.
pub fn setNonBlocking(fd: Fd) void
{
    if (is_windows)
    {
        var one: c_ulong = 1;
        _ = ioctlsocket(fd, FIONBIO, &one);
    }
}

/// Sends every byte, blocking; error.ClientGone if the peer went away.
pub fn sendAll(fd: Fd, bytes: []const u8) !void
{
    var off: usize = 0;
    while (off < bytes.len)
    {
        const n: isize = if (is_windows)
            win32.send(fd, bytes[off..].ptr, @intCast(@min(bytes.len - off, 1 << 30)), 0)
        else
            c.send(fd, bytes[off..].ptr, bytes.len - off, c.MSG.NOSIGNAL);
        if (n <= 0)
        {
            return error.ClientGone;
        }
        off += @intCast(n);
    }
}

/// A blocking receive: the byte count, 0 when the peer closed, negative on an error or timeout.
pub fn recv(fd: Fd, buf: []u8) isize
{
    if (is_windows)
    {
        return win32.recv(fd, buf.ptr, @intCast(@min(buf.len, 1 << 30)), 0);
    }
    return c.recv(fd, buf.ptr, buf.len, 0);
}

pub const Now = union(enum) {
    done: usize,
    would_block,
    failed,
};

/// Sends what fits without waiting.
pub fn sendNow(fd: Fd, bytes: []const u8) Now
{
    if (is_windows)
    {
        const n = win32.send(fd, bytes.ptr, @intCast(@min(bytes.len, 1 << 30)), 0);
        if (n > 0)
        {
            return .{ .done = @intCast(n) };
        }
        return if (n < 0 and win32.WSAGetLastError() == WSAEWOULDBLOCK) .would_block else .failed;
    }
    const n = c.send(fd, bytes.ptr, bytes.len, c.MSG.DONTWAIT | c.MSG.NOSIGNAL);
    if (n > 0)
    {
        return .{ .done = @intCast(n) };
    }
    return if (n < 0 and c._errno().* == @intFromEnum(c.E.AGAIN)) .would_block else .failed;
}

/// Receives what is there without waiting; `.done = 0` means the peer closed.
pub fn recvNow(fd: Fd, buf: []u8) Now
{
    if (is_windows)
    {
        const n = win32.recv(fd, buf.ptr, @intCast(@min(buf.len, 1 << 30)), 0);
        if (n >= 0)
        {
            return .{ .done = @intCast(n) };
        }
        return if (win32.WSAGetLastError() == WSAEWOULDBLOCK) .would_block else .failed;
    }
    const n = c.recv(fd, buf.ptr, buf.len, c.MSG.DONTWAIT);
    if (n >= 0)
    {
        return .{ .done = @intCast(n) };
    }
    return if (c._errno().* == @intFromEnum(c.E.AGAIN)) .would_block else .failed;
}

/// Waits up to `timeout_ms` for any of `fds`; returns how many are ready.
pub fn poll(fds: []PollFd, timeout_ms: c_int) usize
{
    const n = if (is_windows) win32.WSAPoll(fds.ptr, @intCast(fds.len), timeout_ms) else c.poll(fds.ptr, @intCast(fds.len), timeout_ms);
    return if (n > 0) @intCast(n) else 0;
}

/// One-way channel between threads of this process: a pipe, or on Windows a loopback socket pair.
pub const Pipe = struct {
    read_end: Fd,
    write_end: Fd,

    pub fn open() !Pipe
    {
        if (!is_windows)
        {
            var fds: [2]c.fd_t = undefined;
            if (c.pipe(&fds) != 0)
            {
                return error.SystemResources;
            }
            return .{ .read_end = fds[0], .write_end = fds[1] };
        }
        const loopback: u32 = @bitCast([4]u8{ 127, 0, 0, 1 });
        const l = try listenTcp(loopback, 0, 1);
        defer close(l);
        var sa: ws.sockaddr.in = undefined;
        var len: c_int = @sizeOf(ws.sockaddr.in);
        if (win32.getsockname(l, &sa, &len) != 0)
        {
            return error.SystemResources;
        }
        const w = win32.socket(ws.AF.INET, ws.SOCK.STREAM, ws.IPPROTO.TCP);
        if (w == win32.INVALID_SOCKET)
        {
            return error.SystemResources;
        }
        errdefer close(w);
        if (win32.connect(w, &sa, @sizeOf(ws.sockaddr.in)) != 0)
        {
            return error.SystemResources;
        }
        const r = accept(l) orelse return error.SystemResources;
        const one: c_int = 1;
        _ = win32.setsockopt(w, ws.IPPROTO.TCP, ws.TCP.NODELAY, &one, @sizeOf(c_int));
        return .{ .read_end = r, .write_end = w };
    }

    /// Writes all of `bytes` (small values, a pointer or a socket); false if it could not.
    pub fn put(self: Pipe, bytes: []const u8) bool
    {
        if (is_windows)
        {
            sendAll(self.write_end, bytes) catch return false;
            return true;
        }
        return c.write(self.write_end, bytes.ptr, bytes.len) == @as(isize, @intCast(bytes.len));
    }

    /// Reads exactly `out.len` bytes, blocking; false if the writer is gone.
    pub fn take(self: Pipe, out: []u8) bool
    {
        if (is_windows)
        {
            var off: usize = 0;
            while (off < out.len)
            {
                const n = recv(self.read_end, out[off..]);
                if (n <= 0)
                {
                    return false;
                }
                off += @intCast(n);
            }
            return true;
        }
        return c.read(self.read_end, out.ptr, out.len) == @as(isize, @intCast(out.len));
    }
};

test "the pipe carries values between threads, and non-blocking calls do not wait" {
    const p = try Pipe.open();
    defer close(p.read_end);
    defer close(p.write_end);
    const sent: usize = 0x1234_5678;
    try std.testing.expect(p.put(std.mem.asBytes(&sent)));
    var got: usize = 0;
    try std.testing.expect(p.take(std.mem.asBytes(&got)));
    try std.testing.expectEqual(sent, got);

    // Nothing waiting: a non-blocking read returns at once.
    setNonBlocking(p.read_end);
    var buf: [8]u8 = undefined;
    if (!is_windows)
    {
        // A pipe is not a socket on Linux; recvNow is for the viewers' sockets.
        return;
    }
    try std.testing.expect(recvNow(p.read_end, &buf) == .would_block);
    try std.testing.expect(sendNow(p.write_end, "hi") == .done);
    var fds = [1]PollFd{.{ .fd = p.read_end, .events = POLL_IN, .revents = 0 }};
    try std.testing.expectEqual(@as(usize, 1), poll(&fds, 1000));
    try std.testing.expectEqual(Now{ .done = 2 }, recvNow(p.read_end, &buf));
}
