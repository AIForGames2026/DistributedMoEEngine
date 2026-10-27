//! Shared-memory backend (TASK-02). Both ranks map one memfd region holding a
//! receive ring per rank and a single-producer, single-consumer immediate
//! queue per rank. Create the `Pair` before `fork()` and each process uses its
//! own `side`; tests use both sides in one process.
const std = @import("std");
const c = std.c;
const wire = @import("../wire.zig");
const transport = @import("../transport.zig");

const queue_len = 4096;
const cache_line = 64;

/// Immediates destined for one rank. `tail` is written only by the peer,
/// `head` only by the owner; each sits on its own cache line.
const ImmQueue = extern struct {
    head: u64 align(cache_line),
    tail: u64 align(cache_line),
    buf: [queue_len]u32 align(cache_line),
};

const Header = extern struct {
    queues: [2]ImmQueue,
};

const rings_offset = std.mem.alignForward(usize, @sizeOf(Header), 4096);

pub const Side = struct {
    region: [*]align(4096) u8,
    rank: u1,
    slots: u16,

    fn header(self: *Side) *Header {
        return @ptrCast(self.region);
    }

    fn ring(self: *Side, rank: u1) [*]wire.Packet {
        const offset = rings_offset + @as(usize, rank) * self.slots * @sizeOf(wire.Packet);
        return @ptrCast(@alignCast(self.region + offset));
    }

    pub fn backend(self: *Side) transport.Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: transport.Backend.VTable = .{
        .post_write_imm = postWriteImm,
        .post_imm = postImm,
        .poll_completions = pollCompletions,
        .slot = slot,
    };

    fn push(self: *Side, imm: u32) void {
        const q = &self.header().queues[1 - @as(usize, self.rank)];
        const tail = @atomicLoad(u64, &q.tail, .monotonic);
        while (tail - @atomicLoad(u64, &q.head, .acquire) >= queue_len) std.atomic.spinLoopHint();
        q.buf[tail % queue_len] = imm;
        @atomicStore(u64, &q.tail, tail + 1, .release);
    }

    fn postWriteImm(ptr: *anyopaque, i: u16, pkt: *const wire.Packet, imm: u32) transport.Error!void {
        const self: *Side = @ptrCast(@alignCast(ptr));
        if (i >= self.slots) return error.ProtocolError;
        self.ring(1 - self.rank)[i] = pkt.*;
        // The release store in push orders the packet copy before the immediate.
        self.push(imm);
    }

    fn postImm(ptr: *anyopaque, imm: u32) transport.Error!void {
        const self: *Side = @ptrCast(@alignCast(ptr));
        self.push(imm);
    }

    fn pollCompletions(ptr: *anyopaque, out: []u32) transport.Error!usize {
        const self: *Side = @ptrCast(@alignCast(ptr));
        const q = &self.header().queues[self.rank];
        const head = @atomicLoad(u64, &q.head, .monotonic);
        const tail = @atomicLoad(u64, &q.tail, .acquire);
        const n: usize = @intCast(@min(tail - head, out.len));
        for (out[0..n], 0..) |*o, k| o.* = q.buf[(head + k) % queue_len];
        @atomicStore(u64, &q.head, head + n, .release);
        return n;
    }

    fn slot(ptr: *anyopaque, i: u16) *wire.Packet {
        const self: *Side = @ptrCast(@alignCast(ptr));
        return &self.ring(self.rank)[i];
    }
};

pub const Pair = struct {
    region: []align(4096) u8,
    sides: [2]Side,

    /// Maps a zeroed shared region for two ranks with `slots` slots each.
    pub fn create(slots: u16) !Pair {
        const len = rings_offset + 2 * @as(usize, slots) * @sizeOf(wire.Packet);
        const fd = c.memfd_create("tensorfold-shm", 0);
        if (fd < 0) return error.SystemResources;
        defer _ = c.close(fd);
        if (c.ftruncate(fd, @intCast(len)) != 0) return error.SystemResources;
        const ptr = c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        if (ptr == c.MAP_FAILED) return error.SystemResources;
        const region: [*]align(4096) u8 = @ptrCast(@alignCast(ptr));
        return .{
            .region = region[0..len],
            .sides = .{
                .{ .region = region, .rank = 0, .slots = slots },
                .{ .region = region, .rank = 1, .slots = slots },
            },
        };
    }

    pub fn side(self: *Pair, rank: u1) *Side {
        return &self.sides[rank];
    }

    pub fn deinit(self: *Pair) void {
        _ = c.munmap(@ptrCast(self.region.ptr), self.region.len);
        self.* = undefined;
    }
};

test "immediates flow both ways and keep order" {
    var pair = try Pair.create(2);
    defer pair.deinit();
    const a = pair.side(0).backend();
    const b = pair.side(1).backend();
    try a.vtable.post_imm(a.ptr, 1);
    try a.vtable.post_imm(a.ptr, 2);
    try b.vtable.post_imm(b.ptr, 3);
    var out: [4]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, out[0..try b.vtable.poll_completions(b.ptr, &out)]);
    try std.testing.expectEqualSlices(u32, &.{3}, out[0..try a.vtable.poll_completions(a.ptr, &out)]);
    try std.testing.expectEqual(@as(usize, 0), try a.vtable.poll_completions(a.ptr, &out));
}
