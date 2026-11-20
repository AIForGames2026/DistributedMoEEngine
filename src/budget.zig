//! A hard memory budget for one engine process (TASK-17): `Heap` counts live allocations and refuses
//! any past its limit, and `Memory` sets that limit from the budget and checks resident memory.
const std = @import("std");
const builtin = @import("builtin");

pub const Heap = struct {
    child: std.mem.Allocator,
    limit: std.atomic.Value(usize) = .init(std.math.maxInt(usize)),
    live: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    allocs: std.atomic.Value(u64) = .init(0),
    frees: std.atomic.Value(u64) = .init(0),
    refused: std.atomic.Value(u64) = .init(0),

    pub fn init(child: std.mem.Allocator) Heap
    {
        return .{ .child = child };
    }

    pub fn allocator(self: *Heap) std.mem.Allocator
    {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    /// Claims `n` more bytes, or returns false if that would pass the limit.
    fn claim(self: *Heap, n: usize) bool
    {
        var cur = self.live.load(.monotonic);
        while (true)
        {
            if (n > self.limit.load(.monotonic) -| cur)
            {
                _ = self.refused.fetchAdd(1, .monotonic);
                return false;
            }
            cur = self.live.cmpxchgWeak(cur, cur + n, .monotonic, .monotonic) orelse break;
        }
        _ = self.peak.fetchMax(cur + n, .monotonic);
        return true;
    }

    fn unclaim(self: *Heap, n: usize) void
    {
        _ = self.live.fetchSub(n, .monotonic);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8
    {
        const self: *Heap = @ptrCast(@alignCast(ctx));
        if (!self.claim(len))
        {
            return null;
        }
        const p = self.child.rawAlloc(len, alignment, ret_addr) orelse
        {
            self.unclaim(len);
            return null;
        };
        _ = self.allocs.fetchAdd(1, .monotonic);
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool
    {
        const self: *Heap = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len and !self.claim(new_len - memory.len))
        {
            return false;
        }
        if (!self.child.rawResize(memory, alignment, new_len, ret_addr))
        {
            if (new_len > memory.len)
            {
                self.unclaim(new_len - memory.len);
            }
            return false;
        }
        if (new_len < memory.len)
        {
            self.unclaim(memory.len - new_len);
        }
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8
    {
        const self: *Heap = @ptrCast(@alignCast(ctx));
        if (new_len > memory.len and !self.claim(new_len - memory.len))
        {
            return null;
        }
        const p = self.child.rawRemap(memory, alignment, new_len, ret_addr) orelse
        {
            if (new_len > memory.len)
            {
                self.unclaim(new_len - memory.len);
            }
            return null;
        };
        if (new_len < memory.len)
        {
            self.unclaim(memory.len - new_len);
        }
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void
    {
        const self: *Heap = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, alignment, ret_addr);
        self.unclaim(memory.len);
        _ = self.frees.fetchAdd(1, .monotonic);
    }
};

/// This process's resident memory, in bytes (Linux and Windows; 0 elsewhere).
pub fn residentBytes() u64
{
    if (builtin.os.tag == .windows)
    {
        return @import("plat.zig").rssBytes();
    }
    return statmField(1);
}

/// The most this process has had resident so far, in bytes (Linux; 0 elsewhere).
pub fn peakResidentBytes() u64
{
    if (builtin.os.tag != .linux)
    {
        return 0;
    }
    var buf: [4096]u8 = undefined;
    const text = readProc("/proc/self/status", &buf);
    const at = std.mem.indexOf(u8, text, "VmHWM:") orelse return 0;
    var it = std.mem.tokenizeAny(u8, text[at + 6 ..], " \t\n");
    const kb = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return kb * 1024;
}

fn statmField(i: usize) u64
{
    if (builtin.os.tag != .linux)
    {
        return 0;
    }
    var buf: [256]u8 = undefined;
    var it = std.mem.tokenizeScalar(u8, readProc("/proc/self/statm", &buf), ' ');
    var field: []const u8 = "";
    for (0..i + 1) |_|
    {
        field = it.next() orelse return 0;
    }
    const pages = std.fmt.parseInt(u64, std.mem.trimEnd(u8, field, "\n"), 10) catch return 0;
    return pages * std.heap.pageSize();
}

fn readProc(path: [*:0]const u8, buf: []u8) []const u8
{
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0)
    {
        return "";
    }
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n <= 0)
    {
        return "";
    }
    return buf[0..@intCast(n)];
}

/// The budget itself: the heap may grow only by what the budget leaves after
/// the memory already resident and the GPU's allocations when it is armed.
pub const Memory = struct {
    heap: *Heap,
    budget: u64,
    /// GPU memory outside the process's resident set, if any (integrated GPUs share system memory).
    device: ?*const fn () u64 = null,
    /// Resident memory not on the heap when armed: weights, the CUDA runtime, code.
    fixed: u64 = 0,
    /// The highest resident plus GPU memory seen at a check, and checks over budget.
    max_seen: u64 = 0,
    over: u64 = 0,
    checks: u64 = 0,

    /// Sets the heap's limit. Fails if the fixed memory alone leaves nothing.
    pub fn arm(self: *Memory) !void
    {
        const rss = residentBytes();
        const live = self.heap.live.load(.monotonic);
        const dev = if (self.device) |f| f() else 0;
        self.fixed = (rss -| live) + dev;
        if (self.fixed >= self.budget)
        {
            return error.OverMemoryBudget;
        }
        self.heap.limit.store(@intCast(self.budget - self.fixed), .monotonic);
        self.check();
    }

    /// Samples resident and GPU memory; cheap enough to run every engine step.
    pub fn check(self: *Memory) void
    {
        const now = residentBytes() + if (self.device) |f| f() else 0;
        self.checks += 1;
        self.max_seen = @max(self.max_seen, now);
        if (now > self.budget)
        {
            self.over += 1;
        }
    }

    /// What the heap may still take.
    pub fn headroom(self: *const Memory) u64
    {
        return self.heap.limit.load(.monotonic) -| self.heap.live.load(.monotonic);
    }
};

test "the heap counts live bytes and ends balanced" {
    var heap: Heap = .init(std.testing.allocator);
    const a = heap.allocator();
    const x = try a.alloc(u8, 1000);
    var list: std.ArrayList(u32) = .empty;
    for (0..500) |i|
    {
        try list.append(a, @intCast(i));
    }
    try std.testing.expect(heap.live.load(.monotonic) >= 1000 + 500 * 4);
    a.free(x);
    list.deinit(a);
    try std.testing.expectEqual(@as(usize, 0), heap.live.load(.monotonic));
    try std.testing.expectEqual(heap.allocs.load(.monotonic), heap.frees.load(.monotonic));
    try std.testing.expect(heap.peak.load(.monotonic) >= 3000);
}

test "the heap refuses to pass its limit and recovers after frees" {
    var heap: Heap = .init(std.testing.allocator);
    heap.limit.store(4096, .monotonic);
    const a = heap.allocator();
    const x = try a.alloc(u8, 3000);
    try std.testing.expectError(error.OutOfMemory, a.alloc(u8, 2000));
    try std.testing.expectEqual(@as(u64, 1), heap.refused.load(.monotonic));
    try std.testing.expect(!a.resize(x, 5000));
    const y = try a.alloc(u8, 1000);
    a.free(x);
    const z = try a.alloc(u8, 3000);
    a.free(y);
    a.free(z);
    try std.testing.expectEqual(@as(usize, 0), heap.live.load(.monotonic));
    try std.testing.expect(heap.peak.load(.monotonic) <= 4096);
}

test "an arena on the heap is refused as a whole once the limit is reached" {
    var heap: Heap = .init(std.testing.allocator);
    heap.limit.store(64 << 10, .monotonic);
    var arena: std.heap.ArenaAllocator = .init(heap.allocator());
    const a = arena.allocator();
    var got: usize = 0;
    while (a.alloc(u8, 4096)) |_|
    {
        got += 1;
    }
    else |err|
    {
        try std.testing.expectEqual(error.OutOfMemory, err);
    }
    try std.testing.expect(got > 0 and got < 16);
    arena.deinit();
    try std.testing.expectEqual(@as(usize, 0), heap.live.load(.monotonic));
}
