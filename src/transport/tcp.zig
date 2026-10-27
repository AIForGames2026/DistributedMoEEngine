//! TCP backend (TASK-02). Links the desktop (WSL2) and gx10-03 over Tailscale,
//! and is the fallback if RDMA is unavailable (risk R2). Each message is a
//! frame header carrying the immediate, followed by the packet for writes.
//! The receiver reads the packet straight into the slot named by the
//! immediate before surfacing the immediate, matching RDMA's ordering.
const std = @import("std");
const c = std.c;
const wire = @import("../wire.zig");
const transport = @import("../transport.zig");

const Frame = extern struct {
    imm: u32,
    /// 0 for an immediate alone, `wire.packet_bytes` for a write.
    len: u32,
};

pub const Link = struct {
    fd: c.fd_t,
    ring: []wire.Packet,

    /// `ring` is the local receive ring, normally carved from an `Arena`.
    pub fn init(fd: c.fd_t, ring: []wire.Packet) Link {
        return .{ .fd = fd, .ring = ring };
    }

    /// Graceful close: stop sending, then discard incoming bytes until the peer
    /// closes too (or 5 s pass). Closing with unread bytes would make the
    /// kernel send a reset, which destroys data the peer has not read yet.
    pub fn close(self: *Link) void {
        self.shutdownSend();
        var scratch: [4096]u8 = undefined;
        var pfd = [1]c.pollfd{.{ .fd = self.fd, .events = c.POLL.IN, .revents = 0 }};
        while (c.poll(&pfd, 1, 5000) > 0) {
            if (c.recv(self.fd, &scratch, scratch.len, 0) <= 0) break;
        }
        _ = c.close(self.fd);
        self.fd = -1;
    }

    /// Tells the peer we will send nothing more.
    pub fn shutdownSend(self: *Link) void {
        _ = c.shutdown(self.fd, 1); // SHUT_WR
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
        var off: usize = 0;
        const flags: u32 = c.MSG.NOSIGNAL | if (more) @as(u32, c.MSG.MORE) else 0;
        while (off < bytes.len) {
            const n = c.send(self.fd, bytes[off..].ptr, bytes.len - off, flags);
            if (n < 0) {
                const e = c.errno(n);
                if (e == .INTR) continue;
                std.log.err("tcp send: {t}", .{e});
                return error.ConnectionClosed;
            }
            off += @intCast(n);
        }
    }

    fn recvAll(self: *Link, bytes: []u8) transport.Error!void {
        var off: usize = 0;
        while (off < bytes.len) {
            const n = c.recv(self.fd, bytes[off..].ptr, bytes.len - off, 0);
            if (n == 0) {
                std.log.err("tcp recv: peer closed with {d} of {d} bytes read", .{ off, bytes.len });
                return error.ConnectionClosed;
            }
            if (n < 0) {
                const e = c.errno(n);
                if (e == .INTR) continue;
                std.log.err("tcp recv: {t}", .{e});
                return error.ConnectionClosed;
            }
            off += @intCast(n);
        }
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
        var pfd = [1]c.pollfd{.{ .fd = self.fd, .events = c.POLL.IN, .revents = 0 }};
        const rc = c.poll(&pfd, 1, 0);
        if (rc < 0) return if (c.errno(rc) == .INTR) false else error.Unexpected;
        return rc > 0;
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

pub fn parseIp4(text: []const u8) !u32 {
    var parts = std.mem.splitScalar(u8, text, '.');
    var addr: [4]u8 = undefined;
    for (&addr) |*octet| octet.* = try std.fmt.parseInt(u8, parts.next() orelse return error.InvalidAddress, 10);
    if (parts.next() != null) return error.InvalidAddress;
    return @bitCast(addr); // already network byte order in memory
}

fn sockaddr(addr: u32, port: u16) c.sockaddr.in {
    return .{ .port = std.mem.nativeToBig(u16, port), .addr = addr };
}

fn tune(fd: c.fd_t) void {
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.IPPROTO.TCP, c.TCP.NODELAY, &one, @sizeOf(c_int));
}

/// Listens on `port` on all interfaces; port 0 picks a free port. Returns the
/// listening socket and the bound port. Accept a peer with `acceptOn`.
pub fn listen(port: u16) !struct { fd: c.fd_t, port: u16 } {
    const lfd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
    if (lfd < 0) return error.SystemResources;
    errdefer _ = c.close(lfd);
    const one: c_int = 1;
    _ = c.setsockopt(lfd, c.SOL.SOCKET, c.SO.REUSEADDR, &one, @sizeOf(c_int));
    var sa = sockaddr(0, port);
    if (c.bind(lfd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) != 0) return error.AddressInUse;
    if (c.listen(lfd, 1) != 0) return error.SystemResources;
    return .{ .fd = lfd, .port = try boundPort(lfd) };
}

pub fn acceptOn(lfd: c.fd_t) !c.fd_t {
    const fd = c.accept(lfd, null, null);
    if (fd < 0) return error.ConnectionRefused;
    tune(fd);
    return fd;
}

fn boundPort(fd: c.fd_t) !u16 {
    var sa: c.sockaddr.in = undefined;
    var len: c.socklen_t = @sizeOf(c.sockaddr.in);
    if (c.getsockname(fd, @ptrCast(&sa), &len) != 0) return error.Unexpected;
    return std.mem.bigToNative(u16, sa.port);
}

/// Connects to `host:port`, retrying for up to `retries` attempts 100 ms apart
/// so the client can start before the server.
pub fn connect(host: []const u8, port: u16, retries: u32) !c.fd_t {
    var sa = sockaddr(try parseIp4(host), port);
    var attempt: u32 = 0;
    while (true) : (attempt += 1) {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SystemResources;
        if (c.connect(fd, @ptrCast(&sa), @sizeOf(c.sockaddr.in)) == 0) {
            tune(fd);
            return fd;
        }
        _ = c.close(fd);
        if (attempt + 1 >= retries) return error.ConnectionRefused;
        const ts: c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
        _ = c.nanosleep(&ts, null);
    }
}

test "parseIp4" {
    try std.testing.expectEqual(@as(u32, @bitCast([4]u8{ 100, 119, 15, 36 })), try parseIp4("100.119.15.36"));
    try std.testing.expectError(error.InvalidAddress, parseIp4("1.2.3"));
}

test "TCP loopback carries writes and credits like shared memory" {
    const l = try listen(0);
    defer _ = c.close(l.fd);
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
