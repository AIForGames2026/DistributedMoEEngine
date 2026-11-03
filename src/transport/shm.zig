//! Shared-memory backend (TASK-02). Both ranks map one shared region holding a
//! receive ring per rank and a single-producer, single-consumer immediate
//! queue per rank. On Linux the region is a memfd: create the `Pair` before
//! `fork()` and each process uses its own `side`. Windows has no fork, so the
//! region is a named mapping: the second process maps it with `Pair.open`.
//! Tests use both sides in one process.
const std = @import("std");
const c = std.c;
const wire = @import("../wire.zig");
const transport = @import("../transport.zig");
const win32 = @import("../win32.zig");

const is_windows = @import("builtin").os.tag == .windows;

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
    /// Windows: the named mapping, open until `deinit`.
    mapping: ?win32.HANDLE = null,
    /// Windows: the mapping's name, for the other process's `open`.
    name_buf: [64]u8 = undefined,
    name_len: usize = 0,

    fn regionLen(slots: u16) usize {
        return rings_offset + 2 * @as(usize, slots) * @sizeOf(wire.Packet);
    }

    /// Maps a zeroed shared region for two ranks with `slots` slots each.
    pub fn create(slots: u16) !Pair {
        const len = regionLen(slots);
        if (is_windows) {
            const State = struct {
                var count: u32 = 0;
            };
            var buf: [64]u8 = undefined;
            const name_text = try std.fmt.bufPrint(&buf, "Local\\tensorfold-shm-{d}-{d}", .{
                win32.GetCurrentProcessId(),
                @atomicRmw(u32, &State.count, .Add, 1, .monotonic),
            });
            var wide: [64:0]u16 = undefined;
            const h = win32.CreateFileMappingW(win32.INVALID_HANDLE_VALUE, null, win32.PAGE_READWRITE, @intCast(len >> 32), @truncate(len), widen(name_text, &wide)) orelse
                return error.SystemResources;
            return mapWindows(h, len, slots, name_text);
        }
        const fd = c.memfd_create("tensorfold-shm", 0);
        if (fd < 0) return error.SystemResources;
        defer _ = c.close(fd);
        if (c.ftruncate(fd, @intCast(len)) != 0) return error.SystemResources;
        const ptr = c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        if (ptr == c.MAP_FAILED) return error.SystemResources;
        return fromRegion(@ptrCast(@alignCast(ptr)), len, slots);
    }

    /// Windows only: maps the region another process made with `create`.
    pub fn open(name: []const u8, slots: u16) !Pair {
        if (!is_windows) @compileError("Pair.open is for Windows; Linux shares the region by fork");
        var wide: [64:0]u16 = undefined;
        const h = win32.OpenFileMappingW(win32.FILE_MAP_ALL_ACCESS, 0, widen(name, &wide)) orelse
            return error.FileNotFound;
        return mapWindows(h, regionLen(slots), slots, name);
    }

    /// The name to pass to `open` (Windows).
    pub fn mappingName(self: *const Pair) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    fn mapWindows(h: win32.HANDLE, len: usize, slots: u16, name_text: []const u8) !Pair {
        errdefer _ = win32.CloseHandle(h);
        const ptr = win32.MapViewOfFile(h, win32.FILE_MAP_ALL_ACCESS, 0, 0, len) orelse return error.SystemResources;
        var pair = fromRegion(@ptrCast(@alignCast(ptr)), len, slots);
        pair.mapping = h;
        @memcpy(pair.name_buf[0..name_text.len], name_text);
        pair.name_len = name_text.len;
        return pair;
    }

    fn widen(text: []const u8, out: *[64:0]u16) [*:0]const u16 {
        for (text, 0..) |ch, i| out[i] = ch;
        out[text.len] = 0;
        return out;
    }

    fn fromRegion(region: [*]align(4096) u8, len: usize, slots: u16) Pair {
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
        if (is_windows) {
            _ = win32.UnmapViewOfFile(self.region.ptr);
            if (self.mapping) |h| _ = win32.CloseHandle(h);
        } else {
            _ = c.munmap(@ptrCast(self.region.ptr), self.region.len);
        }
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
