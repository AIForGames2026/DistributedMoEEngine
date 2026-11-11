//! Expert cache (TASK-11): a fixed number of expert slots, filled from the
//! model file by one worker thread. Lookahead hints queue loads ahead of use;
//! an expert that is not in a slot when it is needed is loaded on demand,
//! ahead of any queued hints. The first layers can be pinned, since the
//! predictor guesses them worst. Which slot an expert sits in never changes
//! its weights, so results are the same with any cache size.
const std = @import("std");
const builtin = @import("builtin");
const olmoe = @import("olmoe.zig");
const loader = @import("loader.zig");

/// Where expert bytes come from.
pub const Source = struct {
    ctx: *anyopaque,
    /// Reads `r` into `dst` (`r.len + pad` bytes) and returns the offset in
    /// `dst` where `r` starts; direct I/O reads whole aligned blocks.
    read: *const fn (ctx: *anyopaque, r: loader.Range, dst: []u8) anyerror!usize,
    pad: usize,
    /// Byte alignment the slots need (direct I/O's memory alignment).
    alignment: usize,
};

/// How each expert call found its expert.
pub const Stats = struct {
    /// Already in a slot from earlier use, or in a pinned layer.
    resident: u64 = 0,
    /// Loaded because of a hint and ready before it was needed.
    hinted: u64 = 0,
    /// Loaded because of a hint but still loading when needed.
    late: u64 = 0,
    /// Not in a slot and not loading: loaded on demand.
    missed: u64 = 0,
    wait_ns: u64 = 0,
    /// Worker side.
    loads: u64 = 0,
    hint_loads: u64 = 0,
    load_ns: u64 = 0,
    bytes: u64 = 0,
    /// Hints dropped because the queue was full.
    dropped: u64 = 0,
    /// Hint-loaded experts evicted before any use.
    wasted: u64 = 0,

    pub fn calls(self: Stats) u64
    {
        return self.resident + self.hinted + self.late + self.missed;
    }
};

/// Single-producer, single-consumer queue of expert keys.
const Ring = struct {
    buf: []u32,
    head: std.atomic.Value(usize) = .init(0),
    tail: std.atomic.Value(usize) = .init(0),

    fn push(self: *Ring, v: u32) bool
    {
        const t = self.tail.raw;
        if (t - self.head.load(.acquire) == self.buf.len)
        {
            return false;
        }
        self.buf[t % self.buf.len] = v;
        self.tail.store(t + 1, .release);
        return true;
    }

    fn pop(self: *Ring) ?u32
    {
        const h = self.head.raw;
        if (h == self.tail.load(.acquire))
        {
            return null;
        }
        const v = self.buf[h % self.buf.len];
        self.head.store(h + 1, .release);
        return v;
    }
};

const State = enum(u8) { empty, loading, ready };

const Slot = struct {
    key: std.atomic.Value(u32) = .init(none),
    state: std.atomic.Value(State) = .init(.empty),
    /// Calls using this slot right now; the worker never evicts it meanwhile.
    pins: std.atomic.Value(u32) = .init(0),
    last_use: std.atomic.Value(u64) = .init(0),
    /// Loaded for a hint and not used since.
    from_hint: std.atomic.Value(bool) = .init(false),
    pinned: bool = false,
    bytes: []u8,
    expert: olmoe.Expert = undefined,
};

const none = std.math.maxInt(u32);

pub const ExpertCache = struct {
    cfg: olmoe.Config,
    model: loader.Model,
    owned: []const bool,
    source: Source,
    index: ?*const loader.Index,
    /// Per (layer, expert): its slot, or `none`. Written only by the worker.
    where: []std.atomic.Value(u32),
    /// Per (layer, expert): checksum already verified once.
    checked: []bool,
    slots: []Slot,
    region: []u8,
    part_cap: [3]usize,
    hints: Ring,
    demand: Ring,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// Updated by the calling thread and the worker respectively.
    stats: Stats = .{},
    clock: std.atomic.Value(u64) = .init(1),

    /// Room each slot needs for one expert's gate, up, and down weights.
    pub fn partSizes(model: loader.Model, pad: usize, alignment: usize) ![3]usize
    {
        var cap: [3]usize = .{ 0, 0, 0 };
        for (model.weights) |w|
        {
            if (w.layout != .gguf or w.parts[1].len != 0)
            {
                return error.UnsupportedLayout;
            }
            const p = @backingInt(w.proj);
            cap[p] = @max(cap[p], std.mem.alignForward(usize, @intCast(w.parts[0].len + pad), alignment));
        }
        return cap;
    }

    pub fn slotBytes(model: loader.Model, pad: usize, alignment: usize) !usize
    {
        const cap = try partSizes(model, pad, alignment);
        return cap[0] + cap[1] + cap[2];
    }

    /// Slots in `region` (a multiple of `slotBytes`) for this process's
    /// `owned` experts. Layers below `pin_layers` are loaded now and stay
    /// loaded. Call `start` next.
    pub fn init(a: std.mem.Allocator, cfg: olmoe.Config, model: loader.Model, owned: []const bool, source: Source, index: ?*const loader.Index, region: []u8, pin_layers: usize) !*ExpertCache
    {
        const self = try a.create(ExpertCache);
        const cap = try partSizes(model, source.pad, source.alignment);
        const slot_bytes = cap[0] + cap[1] + cap[2];
        const slot_count = region.len / slot_bytes;
        if (@intFromPtr(region.ptr) % source.alignment != 0)
        {
            return error.Misaligned;
        }
        const where = try a.alloc(std.atomic.Value(u32), cfg.layers * cfg.experts);
        for (where) |*v|
        {
            v.* = .init(none);
        }
        const slots = try a.alloc(Slot, slot_count);
        for (slots, 0..) |*s, i|
        {
            s.* = .{ .bytes = region[i * slot_bytes ..][0..slot_bytes] };
        }
        self.* = .{
            .cfg = cfg,
            .model = model,
            .owned = owned,
            .source = source,
            .index = index,
            .where = where,
            .checked = try a.alloc(bool, where.len),
            .slots = slots,
            .region = region,
            .part_cap = cap,
            .hints = .{ .buf = try a.alloc(u32, 4096) },
            .demand = .{ .buf = try a.alloc(u32, 256) },
        };
        @memset(self.checked, false);
        var pinned: usize = 0;
        for (0..@min(pin_layers, cfg.layers)) |l|
        {
            for (0..cfg.experts) |e|
            {
                if (!owned[e])
                {
                    continue;
                }
                if (pinned + 1 >= slot_count)
                {
                    return error.CacheTooSmall;
                }
                slots[pinned].pinned = true;
                try self.load(@intCast(l * cfg.experts + e), pinned, false);
                pinned += 1;
            }
        }
        if (pinned >= slot_count)
        {
            return error.CacheTooSmall;
        }
        return self;
    }

    pub fn start(self: *ExpertCache) !void
    {
        self.thread = try std.Thread.spawn(.{}, work, .{self});
    }

    pub fn deinit(self: *ExpertCache) void
    {
        self.stop.store(true, .release);
        if (self.thread) |t|
        {
            t.join();
        }
        self.thread = null;
    }

    /// Queues a load of `layer`'s `expert` if it is owned here and not
    /// already in a slot. Never waits.
    pub fn hint(self: *ExpertCache, layer: usize, expert: usize) void
    {
        if (layer >= self.cfg.layers or expert >= self.cfg.experts or !self.owned[expert])
        {
            return;
        }
        const key: u32 = @intCast(layer * self.cfg.experts + expert);
        if (self.where[key].load(.acquire) != none)
        {
            return;
        }
        if (!self.hints.push(key))
        {
            self.stats.dropped += 1;
        }
    }

    pub const Held = struct {
        expert: olmoe.Expert,
        slot: u32,
    };

    /// The expert's weights, loading them first if needed. Keeps its slot
    /// until `release`.
    pub fn acquire(self: *ExpertCache, layer: usize, expert: usize) !Held
    {
        if (layer >= self.cfg.layers or expert >= self.cfg.experts or !self.owned[expert])
        {
            return error.ExpertNotLocal;
        }
        const key: u32 = @intCast(layer * self.cfg.experts + expert);
        var waited = false;
        var asked = false;
        var was_hint = false;
        var t0: u64 = 0;
        while (true)
        {
            if (self.failed.load(.acquire))
            {
                return error.ExpertLoadFailed;
            }
            const s = self.where[key].load(.acquire);
            if (s != none)
            {
                const slot = &self.slots[s];
                _ = slot.pins.fetchAdd(1, .seq_cst);
                if (slot.key.load(.seq_cst) == key)
                {
                    const st = slot.state.load(.seq_cst);
                    if (st == .ready)
                    {
                        const fresh = slot.from_hint.swap(false, .acq_rel);
                        if (asked and !was_hint)
                        {
                            self.stats.missed += 1;
                        }
                        else if (waited)
                        {
                            self.stats.late += 1;
                        }
                        else if (fresh)
                        {
                            self.stats.hinted += 1;
                        }
                        else
                        {
                            self.stats.resident += 1;
                        }
                        if (waited or asked)
                        {
                            self.stats.wait_ns += nowNs() - t0;
                        }
                        slot.last_use.store(self.clock.fetchAdd(1, .monotonic), .release);
                        return .{ .expert = slot.expert, .slot = s };
                    }
                    if (st == .loading and !waited)
                    {
                        waited = true;
                        was_hint = was_hint or slot.from_hint.load(.acquire);
                        t0 = nowNs();
                    }
                }
                _ = slot.pins.fetchSub(1, .seq_cst);
            }
            else if (!asked)
            {
                if (self.demand.push(key))
                {
                    asked = true;
                    t0 = nowNs();
                }
            }
            std.atomic.spinLoopHint();
        }
    }

    pub fn release(self: *ExpertCache, h: Held) void
    {
        _ = self.slots[h.slot].pins.fetchSub(1, .release);
    }

    /// Slot bytes, for mapping into the GPU.
    pub fn memory(self: *ExpertCache) []u8
    {
        return self.region;
    }

    fn work(self: *ExpertCache) void
    {
        while (!self.stop.load(.acquire))
        {
            const from_hint, const key = if (self.demand.pop()) |k| .{ false, k } else if (self.hints.pop()) |k| .{ true, k } else
            {
                idle();
                continue;
            };
            if (self.where[key].load(.acquire) != none)
            {
                continue;
            }
            const victim = self.evict() orelse
            {
                // Every slot is in use; try again once a call finishes.
                if (!from_hint)
                {
                    _ = self.demand.push(key);
                }
                idle();
                continue;
            };
            self.load(key, victim, from_hint) catch |err|
            {
                std.log.err("expert cache: loading layer {d} expert {d}: {t}", .{ key / self.cfg.experts, key % self.cfg.experts, err });
                self.failed.store(true, .release);
                return;
            };
        }
    }

    /// Frees the least recently used unpinned slot and returns it, marked
    /// loading, or null if every slot is pinned or in use.
    fn evict(self: *ExpertCache) ?usize
    {
        while (true)
        {
            var best: ?usize = null;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.slots, 0..) |*s, i|
            {
                if (s.pinned or s.pins.load(.acquire) != 0)
                {
                    continue;
                }
                const st = s.state.load(.acquire);
                if (st == .empty)
                {
                    s.state.store(.loading, .seq_cst);
                    return i;
                }
                if (st == .ready and s.last_use.load(.acquire) < oldest)
                {
                    oldest = s.last_use.load(.acquire);
                    best = i;
                }
            }
            const i = best orelse return null;
            const s = &self.slots[i];
            s.state.store(.loading, .seq_cst);
            if (s.pins.load(.seq_cst) != 0)
            {
                // A call took it between the scan and now.
                s.state.store(.ready, .seq_cst);
                continue;
            }
            const old = s.key.load(.acquire);
            if (old != none)
            {
                self.where[old].store(none, .release);
                if (s.from_hint.swap(false, .acq_rel))
                {
                    self.stats.wasted += 1;
                }
            }
            return i;
        }
    }

    /// Reads one expert's three weights into slot `si` (already marked
    /// loading, or not yet visible) and publishes it.
    fn load(self: *ExpertCache, key: u32, si: usize, from_hint: bool) !void
    {
        const t0 = nowNs();
        const s = &self.slots[si];
        const layer: u32 = key / @as(u32, @intCast(self.cfg.experts));
        const e: u32 = key % @as(u32, @intCast(self.cfg.experts));
        s.state.store(.loading, .seq_cst);
        s.from_hint.store(from_hint, .release);
        s.key.store(key, .seq_cst);
        self.where[key].store(@intCast(si), .release);
        var mats: [3]olmoe.Mat = undefined;
        var off: usize = 0;
        inline for (.{ loader.Projection.gate, loader.Projection.up, loader.Projection.down }, 0..) |proj, p|
        {
            const w = self.model.weight(layer, proj, e);
            const r = w.parts[0];
            const dst = s.bytes[off..][0..self.part_cap[p]];
            const at = try self.source.read(self.source.ctx, r, dst);
            const bytes = dst[at..][0..@intCast(r.len)];
            if (self.index) |idx|
            {
                if (!self.checked[key])
                {
                    const entry = idx.map.get(r.offset) orelse return error.MissingChecksum;
                    if (entry.len != r.len or loader.crc32c(bytes) != entry.crc)
                    {
                        return error.BadChecksum;
                    }
                }
            }
            mats[p] = .{ .format = w.format, .rows = @intCast(w.rows), .cols = @intCast(w.cols), .bytes = bytes };
            off += self.part_cap[p];
            self.stats.bytes += r.len;
        }
        self.checked[key] = true;
        s.expert = .{ .gate = mats[0], .up = mats[1], .down = mats[2] };
        s.last_use.store(self.clock.fetchAdd(1, .monotonic), .release);
        s.state.store(.ready, .seq_cst);
        self.stats.loads += 1;
        self.stats.hint_loads += @intFromBool(from_hint);
        self.stats.load_ns += nowNs() - t0;
    }
};

fn idle() void
{
    if (builtin.os.tag == .windows)
    {
        std.Thread.yield() catch {};
        return;
    }
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
    _ = std.c.nanosleep(&ts, null);
}

fn nowNs() u64
{
    if (builtin.os.tag == .windows)
    {
        return @import("win32.zig").nowNs();
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Reads from a model file held in memory (tests).
pub const MemorySource = struct {
    file: []const u8,

    pub fn source(self: *MemorySource) Source
    {
        return .{ .ctx = self, .read = read, .pad = 0, .alignment = 64 };
    }

    fn read(ctx: *anyopaque, r: loader.Range, dst: []u8) anyerror!usize
    {
        const self: *MemorySource = @ptrCast(@alignCast(ctx));
        @memcpy(dst[0..@intCast(r.len)], self.file[@intCast(r.offset)..][0..@intCast(r.len)]);
        return 0;
    }
};

test "a cache of two slots gives the same experts as loading them all" {
    const gguf = @import("gguf.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("olmoe_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try olmoe.Config.fromGguf(h);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var owned: [8]bool = @splat(true);
    var all = try olmoe.LocalExperts.init(a, cfg, m, owned[0..cfg.experts], plan, file);

    var mem: MemorySource = .{ .file = file };
    // Layer 0 pinned (8 experts) plus two slots that every layer-1 expert shares.
    const slot_bytes = try ExpertCache.slotBytes(m, mem.source().pad, mem.source().alignment);
    const region = try a.alignedAlloc(u8, .fromByteUnits(64), (cfg.experts + 2) * slot_bytes);
    const cache = try ExpertCache.init(a, cfg, m, owned[0..cfg.experts], mem.source(), null, region, 1);
    try cache.start();
    defer cache.deinit();

    const x = try a.alloc(f32, cfg.d_model);
    for (x, 0..) |*v, i|
    {
        v.* = @sin(@as(f32, @floatFromInt(i)));
    }
    const want = try a.alloc(f32, cfg.d_model);
    const got = try a.alloc(f32, cfg.d_model);
    for (0..3) |_|
    {
        for (0..cfg.layers) |l|
        {
            for (0..cfg.experts) |e|
            {
                if (l == 1 and e % 2 == 0)
                {
                    cache.hint(l, e + 1);
                }
                all.get(l, e).?.forward(x, want, all.scratch);
                const held = try cache.acquire(l, e);
                held.expert.forward(x, got, all.scratch);
                cache.release(held);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
            }
        }
    }
    const st = cache.stats;
    try std.testing.expectEqual(@as(u64, 3 * cfg.layers * cfg.experts), st.calls());
    // Layer 0 stays pinned: 3 rounds x 8 experts found in place, at least.
    try std.testing.expect(st.resident >= 3 * cfg.experts);
    try std.testing.expect(st.missed + st.late + st.hinted > 0);
}
