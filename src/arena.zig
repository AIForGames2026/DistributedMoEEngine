//! Page-aligned bump arena (TASK-01). Each arena is one anonymous mapping
//! aligned to 2 MiB so the kernel can back it with transparent huge pages.
//! RDMA and io_uring buffers are carved from arenas and registered once.
const std = @import("std");
const posix = std.posix;

pub const huge_page = 2 * 1024 * 1024;

/// Allocation tracing hook. Called on every allocation and reset.
pub const TraceFn = *const fn (ctx: ?*anyopaque, event: Event) void;

pub const Event = union(enum) {
    alloc: struct { offset: usize, len: usize, alignment: usize },
    reset: struct { used: usize },
};

pub const Arena = struct {
    /// The whole mapping, including the slack used to reach 2 MiB alignment.
    mapping: []align(std.heap.page_size_min) u8,
    /// The usable, 2 MiB aligned region.
    buf: []align(huge_page) u8,
    used: usize = 0,
    high_water: usize = 0,
    trace: ?TraceFn = null,
    trace_ctx: ?*anyopaque = null,

    /// Maps at least `min_bytes`, rounded up to a whole huge page.
    pub fn init(min_bytes: usize) !Arena {
        const cap = std.mem.alignForward(usize, @max(min_bytes, 1), huge_page);
        // Over-map by one huge page so an aligned start always fits. Zig's
        // munmap cannot trim a mapping, so the slack stays mapped but untouched.
        const mapping = try posix.mmap(
            null,
            cap + huge_page,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
            -1,
            0,
        );
        const start = std.mem.alignForward(usize, @intFromPtr(mapping.ptr), huge_page);
        const base: [*]align(huge_page) u8 = @ptrFromInt(start);
        // Huge pages are a hint; kernels without THP reject it and the arena still works.
        posix.madvise(base, cap, posix.MADV.HUGEPAGE) catch {};
        return .{ .mapping = mapping, .buf = base[0..cap] };
    }

    pub fn deinit(self: *Arena) void {
        posix.munmap(self.mapping);
        self.* = undefined;
    }

    pub fn capacity(self: *const Arena) usize {
        return self.buf.len;
    }

    pub fn allocBytes(self: *Arena, len: usize, alignment: std.mem.Alignment) error{OutOfMemory}![]u8 {
        const base = @intFromPtr(self.buf.ptr);
        const offset = alignment.forward(base + self.used) - base;
        if (offset > self.buf.len or len > self.buf.len - offset) return error.OutOfMemory;
        self.used = offset + len;
        self.high_water = @max(self.high_water, self.used);
        if (self.trace) |t| t(self.trace_ctx, .{ .alloc = .{ .offset = offset, .len = len, .alignment = alignment.toByteUnits() } });
        return self.buf[offset..][0..len];
    }

    pub fn alloc(self: *Arena, comptime T: type, n: usize) error{OutOfMemory}![]T {
        const len = std.math.mul(usize, n, @sizeOf(T)) catch return error.OutOfMemory;
        const bytes = try self.allocBytes(len, .fromByteUnits(@alignOf(T)));
        const ptr: [*]T = @ptrCast(@alignCast(bytes.ptr));
        return ptr[0..n];
    }

    /// Frees everything at once. Memory stays mapped for reuse.
    pub fn reset(self: *Arena) void {
        if (self.trace) |t| t(self.trace_ctx, .{ .reset = .{ .used = self.used } });
        self.used = 0;
    }

    /// Adapter for code that takes a `std.mem.Allocator`. Only the most recent
    /// allocation can grow or be freed; everything else is freed by `reset`.
    pub fn allocator(self: *Arena) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = vAlloc,
        .resize = vResize,
        .remap = vRemap,
        .free = vFree,
    };

    fn isLast(self: *Arena, memory: []u8) bool {
        return @intFromPtr(memory.ptr) + memory.len == @intFromPtr(self.buf.ptr) + self.used;
    }

    fn vAlloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *Arena = @ptrCast(@alignCast(ctx));
        const bytes = self.allocBytes(len, alignment) catch return null;
        return bytes.ptr;
    }

    fn vResize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = alignment;
        _ = ret_addr;
        const self: *Arena = @ptrCast(@alignCast(ctx));
        if (new_len <= memory.len) {
            if (self.isLast(memory)) self.used -= memory.len - new_len;
            return true;
        }
        if (!self.isLast(memory)) return false;
        const end = @intFromPtr(memory.ptr) - @intFromPtr(self.buf.ptr) + new_len;
        if (end > self.buf.len) return false;
        self.used = end;
        self.high_water = @max(self.high_water, self.used);
        return true;
    }

    fn vRemap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        return if (vResize(ctx, memory, alignment, new_len, ret_addr)) memory.ptr else null;
    }

    fn vFree(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        _ = alignment;
        _ = ret_addr;
        const self: *Arena = @ptrCast(@alignCast(ctx));
        if (self.isLast(memory)) self.used -= memory.len;
    }
};

test "arena is 2 MiB aligned and rounds capacity up" {
    var arena = try Arena.init(100);
    defer arena.deinit();
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(arena.buf.ptr) % huge_page);
    try std.testing.expectEqual(@as(usize, huge_page), arena.capacity());
}

test "allocations respect alignment and fail cleanly when full" {
    var arena = try Arena.init(huge_page);
    defer arena.deinit();
    _ = try arena.allocBytes(3, .@"1");
    const page = try arena.allocBytes(4096, .fromByteUnits(4096));
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(page.ptr) % 4096);
    const words = try arena.alloc(u64, 10);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(words.ptr) % @alignOf(u64));
    try std.testing.expectError(error.OutOfMemory, arena.allocBytes(huge_page, .@"1"));
    try std.testing.expectError(error.OutOfMemory, arena.alloc(u64, std.math.maxInt(usize) / 4));
}

test "reset reuses memory and tracing sees every event" {
    const Counter = struct {
        allocs: usize = 0,
        resets: usize = 0,
        fn hook(ctx: ?*anyopaque, event: Event) void {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            switch (event) {
                .alloc => c.allocs += 1,
                .reset => c.resets += 1,
            }
        }
    };
    var counter: Counter = .{};
    var arena = try Arena.init(huge_page);
    defer arena.deinit();
    arena.trace = Counter.hook;
    arena.trace_ctx = &counter;

    const first = try arena.alloc(u32, 256);
    arena.reset();
    const second = try arena.alloc(u32, 256);
    try std.testing.expectEqual(first.ptr, second.ptr);
    try std.testing.expectEqual(@as(usize, 2), counter.allocs);
    try std.testing.expectEqual(@as(usize, 1), counter.resets);
    try std.testing.expectEqual(@as(usize, 1024), arena.high_water);
}

test "std.mem.Allocator adapter grows and frees the last allocation in place" {
    var arena = try Arena.init(huge_page);
    defer arena.deinit();
    const a = arena.allocator();
    var buf = try a.alloc(u8, 16);
    buf = try a.realloc(buf, 64);
    try std.testing.expectEqual(@as(usize, 64), arena.used);
    a.free(buf);
    try std.testing.expectEqual(@as(usize, 0), arena.used);
}
