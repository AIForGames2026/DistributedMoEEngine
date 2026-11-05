//! RoCE v2 RDMA backend (TASK-07). The transport's model is RDMA's own:
//! `post_write_imm` is one IBV_WR_RDMA_WRITE_WITH_IMM into the peer's
//! registered ring, and the peer sees the immediate in a receive completion
//! after the data has landed. The verbs work is in csrc/tf_rdma.c (most verbs
//! calls are inline C functions); this file adapts it to `Backend`.
//!
//! Linux only, and built only with `zig build -Drdma=true` (links libibverbs).
//! Queue-pair details are swapped over a TCP socket before connecting, and the
//! same socket orders teardown (see `Link.close`).
const std = @import("std");
const wire = @import("../wire.zig");
const transport = @import("../transport.zig");
const tcp = @import("tcp.zig");

const Handle = opaque {};

/// include/tf_rdma.h `tf_rdma_conn_info`.
pub const ConnInfo = extern struct {
    qpn: u32,
    psn: u32,
    gid: [16]u8,
    ring_addr: u64,
    ring_rkey: u32,
    slots: u32,
};

pub const Stats = extern struct {
    writes: u64,
    imms: u64,
    staging_waits: u64,
};

extern fn tf_rdma_open(out: *?*Handle, device: ?[*:0]const u8, gid_index: c_int, ring: *anyopaque, slots: u32, packet_bytes: usize, err: [*]u8, err_len: usize) c_int;
extern fn tf_rdma_local_info(r: *const Handle, info: *ConnInfo) void;
extern fn tf_rdma_connect(r: *Handle, peer: *const ConnInfo, err: [*]u8, err_len: usize) c_int;
extern fn tf_rdma_write_imm(r: *Handle, slot: u32, pkt: *const anyopaque, imm: u32) c_int;
extern fn tf_rdma_imm(r: *Handle, imm: u32) c_int;
extern fn tf_rdma_poll(r: *Handle, out: [*]u32, max: c_int) c_int;
extern fn tf_rdma_drain(r: *Handle, timeout_ms: c_int) c_int;
extern fn tf_rdma_get_stats(r: *const Handle, stats: *Stats) void;
extern fn tf_rdma_close(r: *Handle) void;

pub const Options = struct {
    /// RDMA device name; null takes the first one.
    device: ?[:0]const u8 = null,
    /// GID table index; negative picks the first RoCE v2 IPv4 GID.
    gid_index: c_int = -1,
};

pub const Link = struct {
    handle: *Handle,
    ring: []wire.Packet,
    /// The bootstrap socket, kept to order teardown.
    sock: tcp.Socket,
    rank: u6,

    /// Opens the device, registers `ring` (the local receive ring, page
    /// aligned, normally from an `Arena`), swaps queue-pair details with the
    /// peer over the connected socket `sock`, and connects. Both sides must
    /// use the same ring size.
    pub fn connect(sock: tcp.Socket, rank: u6, ring: []wire.Packet, opts: Options) !Link {
        var err_buf: [256]u8 = undefined;
        @memset(&err_buf, 0);
        var h: ?*Handle = null;
        if (tf_rdma_open(&h, if (opts.device) |d| d.ptr else null, opts.gid_index, ring.ptr, @intCast(ring.len), @sizeOf(wire.Packet), &err_buf, err_buf.len) != 0) {
            std.log.err("rdma: {s}", .{std.mem.sliceTo(&err_buf, 0)});
            return error.RdmaOpenFailed;
        }
        errdefer tf_rdma_close(h.?);

        var mine: ConnInfo = undefined;
        tf_rdma_local_info(h.?, &mine);
        var peer: ConnInfo = undefined;
        try tcp.exchange(sock, std.mem.asBytes(&mine), std.mem.asBytes(&peer));
        if (tf_rdma_connect(h.?, &peer, &err_buf, err_buf.len) != 0) {
            std.log.err("rdma: {s}", .{std.mem.sliceTo(&err_buf, 0)});
            return error.RdmaConnectFailed;
        }
        // Neither side may write until the other has reached ready-to-receive.
        var ready: [1]u8 = .{1};
        var peer_ready: [1]u8 = undefined;
        try tcp.exchange(sock, &ready, &peer_ready);
        return .{ .handle = h.?, .ring = ring, .sock = sock, .rank = rank };
    }

    /// Call after `Endpoint.finish`. Waits for our sends to complete, then
    /// tears down in the goodbye's order: rank 0 closes the socket first, and
    /// rank 1 destroys its queue pair only after seeing that close, so no
    /// write can arrive at a queue pair that no longer exists.
    pub fn close(self: *Link) void {
        if (tf_rdma_drain(self.handle, 5000) != 0) std.log.err("rdma: sends did not complete before close", .{});
        tcp.closeAfterPeerSocket(self.sock, self.rank != 0);
        tf_rdma_close(self.handle);
        self.* = undefined;
    }

    pub fn stats(self: *const Link) Stats {
        var s: Stats = undefined;
        tf_rdma_get_stats(self.handle, &s);
        return s;
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

    fn check(rc: c_int, what: []const u8) transport.Error!void {
        if (rc == 0) return;
        std.log.err("rdma {s}: error {d}", .{ what, -rc });
        return error.ConnectionClosed;
    }

    fn postWriteImm(ptr: *anyopaque, i: u16, pkt: *const wire.Packet, imm: u32) transport.Error!void {
        const self: *Link = @ptrCast(@alignCast(ptr));
        if (i >= self.ring.len) return error.ProtocolError;
        try check(tf_rdma_write_imm(self.handle, i, pkt, imm), "write");
    }

    fn postImm(ptr: *anyopaque, imm: u32) transport.Error!void {
        const self: *Link = @ptrCast(@alignCast(ptr));
        try check(tf_rdma_imm(self.handle, imm), "imm");
    }

    fn pollCompletions(ptr: *anyopaque, out: []u32) transport.Error!usize {
        const self: *Link = @ptrCast(@alignCast(ptr));
        const n = tf_rdma_poll(self.handle, out.ptr, @intCast(@min(out.len, std.math.maxInt(c_int))));
        if (n < 0) {
            std.log.err("rdma poll: error {d}", .{-n});
            return error.ConnectionClosed;
        }
        return @intCast(n);
    }

    fn slot(ptr: *anyopaque, i: u16) *wire.Packet {
        const self: *Link = @ptrCast(@alignCast(ptr));
        return &self.ring[i];
    }
};
