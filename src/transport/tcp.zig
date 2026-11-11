//! TCP backend (TASK-02). Links the desktop (WSL2) and gx10-03 over Tailscale,
//! and is the fallback if RDMA is unavailable (risk R2). Each message is a
//! frame header carrying the immediate, followed by the packet for writes.
//! The receiver reads the packet straight into the slot named by the
//! immediate before surfacing the immediate, matching RDMA's ordering.
//! Runs on Linux (BSD sockets) and Windows (Winsock); `sock` below hides the
//! difference.
const std = @import("std");
const c = std.c;
const wire = @import("../wire.zig");
const transport = @import("../transport.zig");
const win32 = @import("../win32.zig");

const is_windows = @import("builtin").os.tag == .windows;

/// A connected or listening socket.
pub const Socket = if (is_windows) win32.SOCKET else c.fd_t;

const Frame = extern struct {
    imm: u32,
    /// 0 for an immediate alone, `wire.packet_bytes` for a write.
    len: u32,
};

pub const Link = struct {
    fd: Socket,
    ring: []wire.Packet,

    /// `ring` is the local receive ring, normally carved from an `Arena`.
    pub fn init(fd: Socket, ring: []wire.Packet) Link {
        return .{ .fd = fd, .ring = ring };
    }

    /// Graceful close: stop sending, then discard incoming bytes until the peer
    /// closes too (or 5 s pass). Closing with unread bytes would make the
    /// kernel send a reset, which destroys data the peer has not read yet.
    pub fn close(self: *Link) void {
        self.shutdownSend();
        self.drainAndClose();
    }

    /// Close for the side that answered the goodbye (`Endpoint.finish`, rank
    /// 1): wait for the peer to close first (or 5 s), so our answer is never
    /// the last thing in flight before our FIN.
    pub fn closeAfterPeer(self: *Link) void {
        self.drainAndClose();
    }

    fn drainAndClose(self: *Link) void {
        drainAndCloseSocket(self.fd);
        self.fd = undefined;
    }

    /// Tells the peer we will send nothing more.
    pub fn shutdownSend(self: *Link) void {
        sock.shutdownSend(self.fd);
    }

    pub fn backend(self: *Link) transport.Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: transport.Backend.VTable = .{
        .post_write_imm = postWriteImm,
        .post_imm = postImm,
        .poll_completions = pollCompletions,
        .slot = slot,
    };

    fn sendAll(self: *Link, bytes: []const u8, more: bool) transport.Error!void {
        return sendAllSocket(self.fd, bytes, more);
    }

    fn recvAll(self: *Link, bytes: []u8) transport.Error!void {
        return recvAllSocket(self.fd, bytes);
    }

    fn postWriteImm(ptr: *anyopaque, i: u16, pkt: *const wire.Packet, imm: u32) transport.Error!void {
        const self: *Link = @ptrCast(@alignCast(ptr));
        _ = i; // the receiver takes the slot from the immediate
        const frame: Frame = .{ .imm = imm, .len = wire.packet_bytes };
        try self.sendAll(std.mem.asBytes(&frame), true);
        try self.sendAll(std.mem.asBytes(pkt), false);
    }

    fn postImm(ptr: *anyopaque, imm: u32) transport.Error!void {
        const self: *Link = @ptrCast(@alignCast(ptr));
        const frame: Frame = .{ .imm = imm, .len = 0 };
        try self.sendAll(std.mem.asBytes(&frame), false);
    }

    fn readable(self: *Link) transport.Error!bool {
        return sock.waitReadable(self.fd, 0) catch error.Unexpected;
    }

    fn pollCompletions(ptr: *anyopaque, out: []u32) transport.Error!usize {
        const self: *Link = @ptrCast(@alignCast(ptr));
        var n: usize = 0;
        while (n < out.len and try self.readable()) {
            var frame: Frame = undefined;
            try self.recvAll(std.mem.asBytes(&frame));
            if (frame.len != 0) {
                const i = wire.Imm.decode(frame.imm).slot;
                if (frame.len != wire.packet_bytes or i >= self.ring.len) return error.ProtocolError;
                try self.recvAll(std.mem.asBytes(&self.ring[i]));
            }
            out[n] = frame.imm;
            n += 1;
        }
        return n;
    }

    fn slot(ptr: *anyopaque, i: u16) *wire.Packet {
        const self: *Link = @ptrCast(@alignCast(ptr));
        return &self.ring[i];
    }
};

fn sendAllSocket(fd: Socket, bytes: []const u8, more: bool) transport.Error!void {
    var off: usize = 0;
    while (off < bytes.len) {
        off += sock.send(fd, bytes[off..], more) catch |err| {
            std.log.err("tcp send: {t}", .{err});
            return error.ConnectionClosed;
        };
    }
}

fn recvAllSocket(fd: Socket, bytes: []u8) transport.Error!void {
    var off: usize = 0;
    while (off < bytes.len) {
        const n = sock.recv(fd, bytes[off..]) catch |err| {
            std.log.err("tcp recv: {t}", .{err});
            return error.ConnectionClosed;
        };
        if (n == 0) {
            std.log.err("tcp recv: peer closed with {d} of {d} bytes read", .{ off, bytes.len });
            return error.ConnectionClosed;
        }
        off += n;
    }
}

/// Discards incoming bytes until the peer closes (or 5 s of silence), then
/// closes the socket. Closing with unread bytes would make the kernel send a
/// reset, which destroys data the peer has not read yet.
fn drainAndCloseSocket(fd: Socket) void {
    var scratch: [4096]u8 = undefined;
    while (sock.waitReadable(fd, 5000) catch false) {
        const n = sock.recv(fd, &scratch) catch break;
        if (n == 0) break;
    }
    sock.close(fd);
}

/// Sends `mine` and receives exactly `theirs.len` bytes from the peer. Both
/// sides call it at once; the messages are small enough to sit in the socket
/// buffers, so neither side blocks the other. Used to bootstrap RDMA.
pub fn exchange(fd: Socket, mine: []const u8, theirs: []u8) transport.Error!void {
    try sendAllSocket(fd, mine, false);
    try recvAllSocket(fd, theirs);
}

/// Ordered close of a bare socket, as `Link.close` (`after_peer` false) or
/// `Link.closeAfterPeer` (true) do for a TCP link.
pub fn closeAfterPeerSocket(fd: Socket, after_peer: bool) void {
    if (!after_peer) sock.shutdownSend(fd);
    drainAndCloseSocket(fd);
}

pub fn parseIp4(text: []const u8) !u32 {
    var parts = std.mem.splitScalar(u8, text, '.');
    var addr: [4]u8 = undefined;
    for (&addr) |*octet| octet.* = try std.fmt.parseInt(u8, parts.next() orelse return error.InvalidAddress, 10);
    if (parts.next() != null) return error.InvalidAddress;
    return @bitCast(addr); // already network byte order in memory
}

/// Listens on `port` on all interfaces; port 0 picks a free port. Returns the
/// listening socket and the bound port. Accept a peer with `acceptOn`.
pub const Listener = struct { fd: Socket, port: u16 };

pub fn listen(port: u16) !Listener {
    return listenMany(port, 1);
}

/// Like `listen`, with room for `backlog` connections waiting to be accepted.
pub fn listenMany(port: u16, backlog: c_int) !Listener {
    const lfd = try sock.open();
    errdefer sock.close(lfd);
    try sock.bindListen(lfd, 0, port, backlog);
    return .{ .fd = lfd, .port = try sock.boundPort(lfd) };
}

/// Waits up to `timeout_ms` for `fd` to have bytes (or a close) to read.
pub fn waitReadable(fd: Socket, timeout_ms: c_int) !bool {
    return sock.waitReadable(fd, timeout_ms);
}

pub fn acceptOn(lfd: Socket) !Socket {
    const fd = try sock.accept(lfd);
    sock.noDelay(fd);
    return fd;
}

/// Closes a socket that never became a `Link` (a listener).
pub fn closeSocket(fd: Socket) void {
    sock.close(fd);
}

/// Connects to `host:port`, retrying for up to `retries` attempts 100 ms apart
/// so the client can start before the server.
pub fn connect(host: []const u8, port: u16, retries: u32) !Socket {
    const addr = try parseIp4(host);
    var attempt: u32 = 0;
    while (true) : (attempt += 1) {
        const fd = try sock.open();
        if (sock.connect(fd, addr, port)) {
            sock.noDelay(fd);
            return fd;
        } else |_| {
            sock.close(fd);
        }
        if (attempt + 1 >= retries) return error.ConnectionRefused;
        sock.sleepMs(100);
    }
}

/// The socket calls `Link` needs, on BSD sockets or Winsock.
const sock = if (is_windows) struct {
    const ws = win32.ws2;
    var started = std.atomic.Value(bool).init(false);

    fn open() !Socket {
        if (!started.swap(true, .acq_rel)) try win32.startWinsock();
        const fd = win32.socket(ws.AF.INET, ws.SOCK.STREAM, ws.IPPROTO.TCP);
        if (fd == win32.INVALID_SOCKET) return error.SystemResources;
        return fd;
    }

    fn close(fd: Socket) void {
        _ = win32.closesocket(fd);
    }

    fn lastError() error{ ConnectionResetByPeer, ConnectionAborted, SocketNotConnected, Unexpected } {
        return switch (win32.WSAGetLastError()) {
            10054 => error.ConnectionResetByPeer, // WSAECONNRESET
            10053 => error.ConnectionAborted, // WSAECONNABORTED
            10057 => error.SocketNotConnected, // WSAENOTCONN
            else => |code| {
                std.log.err("winsock error {d}", .{code});
                return error.Unexpected;
            },
        };
    }

    fn send(fd: Socket, bytes: []const u8, more: bool) !usize {
        _ = more; // Winsock has no MSG_MORE; frames are small and NODELAY is set
        const n = win32.send(fd, bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))), 0);
        if (n < 0) return lastError();
        return @intCast(n);
    }

    fn recv(fd: Socket, bytes: []u8) !usize {
        while (true) {
            const n = win32.recv(fd, bytes.ptr, @intCast(@min(bytes.len, std.math.maxInt(c_int))), 0);
            if (n >= 0) return @intCast(n);
            if (win32.WSAGetLastError() != win32.WSAEINTR) return lastError();
        }
    }

    fn waitReadable(fd: Socket, timeout_ms: c_int) !bool {
        var pfd = [1]win32.WSAPOLLFD{.{ .fd = fd, .events = win32.POLLIN, .revents = 0 }};
        const rc = win32.WSAPoll(&pfd, 1, timeout_ms);
        if (rc < 0) return lastError();
        return rc > 0;
    }

    fn shutdownSend(fd: Socket) void {
        _ = win32.shutdown(fd, win32.SD_SEND);
    }

    fn noDelay(fd: Socket) void {
        const one: c_int = 1;
        _ = win32.setsockopt(fd, ws.IPPROTO.TCP, ws.TCP.NODELAY, &one, @sizeOf(c_int));
    }

    fn bindListen(fd: Socket, addr: u32, port: u16, backlog: c_int) !void {
        // No SO_REUSEADDR: on Windows it lets another socket take over a bound port.
        const sa: ws.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
        if (win32.bind(fd, &sa, @sizeOf(ws.sockaddr.in)) != 0) return error.AddressInUse;
        if (win32.listen(fd, backlog) != 0) return error.SystemResources;
    }

    fn boundPort(fd: Socket) !u16 {
        var sa: ws.sockaddr.in = undefined;
        var len: c_int = @sizeOf(ws.sockaddr.in);
        if (win32.getsockname(fd, &sa, &len) != 0) return error.Unexpected;
        return std.mem.bigToNative(u16, sa.port);
    }

    fn accept(lfd: Socket) !Socket {
        const fd = win32.accept(lfd, null, null);
        if (fd == win32.INVALID_SOCKET) return error.ConnectionRefused;
        return fd;
    }

    fn connect(fd: Socket, addr: u32, port: u16) !void {
        const sa: ws.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
        if (win32.connect(fd, &sa, @sizeOf(ws.sockaddr.in)) != 0) return error.ConnectionRefused;
    }

    fn sleepMs(ms: u32) void {
        win32.Sleep(ms);
    }
} else struct {
    fn open() !Socket {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SystemResources;
        return fd;
    }

    fn close(fd: Socket) void {
        _ = c.close(fd);
    }

    fn send(fd: Socket, bytes: []const u8, more: bool) !usize {
        const flags: u32 = c.MSG.NOSIGNAL | if (more) @as(u32, c.MSG.MORE) else 0;
        while (true) {
            const n = c.send(fd, bytes.ptr, bytes.len, flags);
            if (n >= 0) return @intCast(n);
            const e = c.errno(n);
            if (e != .INTR) {
                std.log.err("errno {t}", .{e});
                return error.Unexpected;
            }
        }
    }

    fn recv(fd: Socket, bytes: []u8) !usize {
        while (true) {
            const n = c.recv(fd, bytes.ptr, bytes.len, 0);
            if (n >= 0) return @intCast(n);
            const e = c.errno(n);
            if (e != .INTR) {
                std.log.err("errno {t}", .{e});
                return error.Unexpected;
            }
        }
    }

    fn waitReadable(fd: Socket, timeout_ms: c_int) !bool {
        var pfd = [1]c.pollfd{.{ .fd = fd, .events = c.POLL.IN, .revents = 0 }};
        const rc = c.poll(&pfd, 1, timeout_ms);
        if (rc < 0) return if (c.errno(rc) == .INTR) false else error.Unexpected;
        return rc > 0;
    }

    fn shutdownSend(fd: Socket) void {
        _ = c.shutdown(fd, 1); // SHUT_WR
    }

    fn noDelay(fd: Socket) void {
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, c.TCP.NODELAY, &one, @sizeOf(c_int));
    }

    fn bindListen(fd: Socket, addr: u32, port: u16, backlog: c_int) !void {
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, &one, @sizeOf(c_int));
        var sa: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
        if (c.bind(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) return error.AddressInUse;
        if (c.listen(fd, @intCast(backlog)) != 0) return error.SystemResources;
    }

    fn boundPort(fd: Socket) !u16 {
        var sa: c.sockaddr.in = undefined;
        var len: c.socklen_t = @sizeOf(c.sockaddr.in);
        if (c.getsockname(fd, @ptrCast(&sa), &len) != 0) return error.Unexpected;
        return std.mem.bigToNative(u16, sa.port);
    }

    fn accept(lfd: Socket) !Socket {
        const fd = c.accept(lfd, null, null);
        if (fd < 0) return error.ConnectionRefused;
        return fd;
    }

    fn connect(fd: Socket, addr: u32, port: u16) !void {
        var sa: c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
        if (c.connect(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) return error.ConnectionRefused;
    }

    fn sleepMs(ms: u32) void {
        const ts: c.timespec = .{ .sec = 0, .nsec = @as(c_long, ms) * std.time.ns_per_ms };
        _ = c.nanosleep(&ts, null);
    }
};

test "parseIp4" {
    try std.testing.expectEqual(@as(u32, @bitCast([4]u8{ 100, 119, 15, 36 })), try parseIp4("100.119.15.36"));
    try std.testing.expectError(error.InvalidAddress, parseIp4("1.2.3"));
}

test "TCP loopback carries writes and credits like shared memory" {
    const l = try listen(0);
    defer closeSocket(l.fd);
    const client_fd = try connect("127.0.0.1", l.port, 1);
    const server_fd = try acceptOn(l.fd);

    var rings: [2][4]wire.Packet = undefined;
    var a_link = Link.init(client_fd, &rings[0]);
    defer a_link.close();
    var b_link = Link.init(server_fd, &rings[1]);
    defer b_link.close();
    var a = transport.Endpoint.init(a_link.backend(), 0, 4);
    var b = transport.Endpoint.init(b_link.backend(), 1, 4);

    for (0..10) |i| {
        var pkt = std.mem.zeroes(wire.Packet);
        pkt.header.token_id = i;
        transport.fillPayload(&pkt, i, 0);
        try a.send(&pkt, .dispatch);
        const r = try b.recv();
        try std.testing.expectEqual(@as(u64, i), r.packet.header.token_id);
        try std.testing.expect(transport.payloadMatches(r.packet, i, 0));
        try b.release();
        try b.flushCredits();
    }
    // Both sides in one thread: announce the end first so neither close waits.
    a_link.shutdownSend();
    b_link.shutdownSend();
}
