//! Expert cache (TASK-11, TASK-14): a fixed number of expert slots, filled
//! from the model file by one or more loader threads. Lookahead hints queue
//! loads ahead of use, and a hint whose layer has already started is dropped;
//! an expert that is not in a slot when it is needed is loaded on demand,
//! ahead of any queued hints. Callers can poll, so they run the experts that
//! are ready while the others load. The first layers can be pinned, since the
//! predictor guesses them worst. Which slot an expert sits in never changes
//! its weights, so results are the same with any cache size.
const std = @import("std");
const builtin = @import("builtin");
const olmoe = @import("olmoe.zig");
const loader = @import("loader.zig");

/// Where expert bytes come from. Each loader thread has its own.
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
    /// Time the calling thread spent with nothing to run but loading experts.
    wait_ns: u64 = 0,
    /// Calls set aside while other experts ran, because theirs was loading.
    deferred: u64 = 0,
    /// Loader side.
    loads: u64 = 0,
    hint_loads: u64 = 0,
    load_ns: u64 = 0,
    bytes: u64 = 0,
    /// Hints dropped because the queue was full.
    dropped: u64 = 0,
    /// Hints dropped because their layer had already started.
    cancelled: u64 = 0,
    /// Hint-loaded experts evicted before any use.
    wasted: u64 = 0,

    pub fn calls(self: Stats) u64
    {
        return self.resident + self.hinted + self.late + self.missed;
    }
};

/// A load ahead of use, for the layer at `pos` (see `ExpertCache.enter`).
const Hint = struct {
    key: u32,
    pos: u64,
};

/// Single-producer queue. Loader threads take turns consuming under the lock.
fn Ring(comptime T: type) type
{
    return struct {
        buf: []T,
        head: std.atomic.Value(usize) = .init(0),
        tail: std.atomic.Value(usize) = .init(0),

        /// Returns a ticket that `taken` reports once the entry is consumed.
        fn push(self: *@This(), v: T) ?usize
        {
            const t = self.tail.raw;
            if (t - self.head.load(.acquire) == self.buf.len)
            {
                return null;
            }
            self.buf[t % self.buf.len] = v;
            self.tail.store(t + 1, .release);
            return t + 1;
        }

        fn taken(self: *@This(), ticket: usize) bool
        {
            return self.head.load(.acquire) >= ticket;
        }

        fn pop(self: *@This()) ?T
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
}

const State = enum(u8) { empty, loading, ready };

const Slot = struct {
    key: std.atomic.Value(u32) = .init(none),
    state: std.atomic.Value(State) = .init(.empty),
    /// Calls using this slot right now; no loader evicts it meanwhile.
    pins: std.atomic.Value(u32) = .init(0),
    last_use: std.atomic.Value(u64) = .init(0),
    /// Loaded for a hint and not used since.
    from_hint: std.atomic.Value(bool) = .init(false),
    /// Loaded on demand and not used since; evicted only as a last resort.
    demanded: std.atomic.Value(bool) = .init(false),
    pinned: bool = false,
    bytes: []u8,
    expert: olmoe.Expert = undefined,
};

const none = std.math.maxInt(u32);

/// One expert call in progress: what it has asked for so far.
pub const Want = struct {
    layer: u32,
    expert: u32,
    asked: bool = false,
    /// The demand queue entry, and whether a slot has held it since.
    ticket: usize = 0,
    seen: bool = false,
    loading: bool = false,
    was_hint: bool = false,
};

pub const ExpertCache = struct {
    cfg: olmoe.Config,
    model: loader.Model,
    owned: []const bool,
    sources: []const Source,
    index: ?*const loader.Index,
    /// Per (layer, expert): its slot, or `none`. Written only under the lock.
    where: []std.atomic.Value(u32),
    /// Per (layer, expert): checksum already verified once.
    checked: []bool,
    slots: []Slot,
    region: []u8,
    part_cap: [3]usize,
    hints: Ring(Hint),
    demand: Ring(u32),
    /// A demand load that found every slot busy, to try again first.
    retry: u32 = none,
    /// Held by a loader while it picks its next job.
    locked: std.atomic.Value(bool) = .init(false),
    /// Where the calling side is: step times layers plus layer.
    pos: std.atomic.Value(u64) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    threads: []std.Thread = &.{},
    /// Updated by the calling thread and the loaders respectively.
    stats: Stats = .{},
    clock: std.atomic.Value(u64) = .init(1),
    allocator: std.mem.Allocator,

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

    /// Slots in `region` for the `owned` experts, one loader thread per source.
    /// Layers below `pin_layers` are loaded now and stay loaded. Call `start` next.
    pub fn init(a: std.mem.Allocator, cfg: olmoe.Config, model: loader.Model, owned: []const bool, sources: []const Source, index: ?*const loader.Index, region: []u8, pin_layers: usize) !*ExpertCache
    {
        if (sources.len == 0)
        {
            return error.NoSource;
        }
        const self = try a.create(ExpertCache);
        const src = sources[0];
        const cap = try partSizes(model, src.pad, src.alignment);
        const slot_bytes = cap[0] + cap[1] + cap[2];
        const slot_count = region.len / slot_bytes;
        if (@intFromPtr(region.ptr) % src.alignment != 0)
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
            .sources = sources,
            .index = index,
            .where = where,
            .checked = try a.alloc(bool, where.len),
            .slots = slots,
            .region = region,
            .part_cap = cap,
            .hints = .{ .buf = try a.alloc(Hint, 4096) },
            .demand = .{ .buf = try a.alloc(u32, 256) },
            .allocator = a,
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
                const key: u32 = @intCast(l * cfg.experts + e);
                slots[pinned].pinned = true;
                self.claim(key, pinned, false);
                try self.fill(src, key, pinned, false);
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
        self.threads = try self.allocator.alloc(std.Thread, self.sources.len);
        for (self.threads, 0..) |*t, i|
        {
            t.* = try std.Thread.spawn(.{}, work, .{ self, i });
        }
    }

    pub fn deinit(self: *ExpertCache) void
    {
        self.stop.store(true, .release);
        for (self.threads) |t|
        {
            t.join();
        }
        self.threads = &.{};
    }

    /// Tells the loaders which layer the calling side has reached, as
    /// `step * layers + layer`; hints for it or earlier are dropped.
    pub fn enter(self: *ExpertCache, pos: u64) void
    {
        self.pos.store(pos, .release);
    }

    /// Queues a load of `layer`'s `expert` for the layer at `pos` if it is
    /// owned here and not already in a slot. Never waits.
    pub fn hint(self: *ExpertCache, pos: u64, layer: usize, expert: usize) void
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
        if (self.hints.push(.{ .key = key, .pos = pos }) == null)
        {
            self.stats.dropped += 1;
        }
    }

    pub const Held = struct {
        expert: olmoe.Expert,
        slot: u32,
    };

    pub fn want(self: *ExpertCache, layer: usize, expert: usize) !Want
    {
        if (layer >= self.cfg.layers or expert >= self.cfg.experts or !self.owned[expert])
        {
            return error.ExpertNotLocal;
        }
        return .{ .layer = @intCast(layer), .expert = @intCast(expert) };
    }

    /// The expert's weights if they are ready, or null after making sure
    /// they are loading. Keeps the slot until `release`. Never waits.
    pub fn poll(self: *ExpertCache, w: *Want) !?Held
    {
        if (self.failed.load(.acquire))
        {
            return error.ExpertLoadFailed;
        }
        const key: u32 = w.layer * @as(u32, @intCast(self.cfg.experts)) + w.expert;
        const s = self.where[key].load(.acquire);
        if (s == none)
        {
            // Ask again if the loaders took the request but the expert is gone.
            const gone = w.asked and (w.seen or self.demand.taken(w.ticket));
            if (!w.asked or gone)
            {
                if (self.demand.push(key)) |ticket|
                {
                    w.asked = true;
                    w.ticket = ticket;
                    w.seen = false;
                }
            }
            return null;
        }
        w.seen = true;
        const slot = &self.slots[s];
        _ = slot.pins.fetchAdd(1, .seq_cst);
        if (slot.key.load(.seq_cst) == key)
        {
            const st = slot.state.load(.seq_cst);
            if (st == .ready)
            {
                const fresh = slot.from_hint.swap(false, .acq_rel);
                slot.demanded.store(false, .release);
                if (w.asked and !w.was_hint)
                {
                    self.stats.missed += 1;
                }
                else if (w.loading)
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
                slot.last_use.store(self.clock.fetchAdd(1, .monotonic), .release);
                return .{ .expert = slot.expert, .slot = s };
            }
            if (st == .loading and !w.loading)
            {
                w.loading = true;
                w.was_hint = w.was_hint or slot.from_hint.load(.acquire);
            }
        }
        _ = slot.pins.fetchSub(1, .seq_cst);
        return null;
    }

    /// Polls until the expert is ready, counting the time as waiting.
    pub fn wait(self: *ExpertCache, w: *Want) !Held
    {
        var t0: u64 = 0;
        while (true)
        {
            if (try self.poll(w)) |h|
            {
                if (t0 != 0)
                {
                    self.stats.wait_ns += nowNs() - t0;
                }
                return h;
            }
            if (t0 == 0)
            {
                t0 = nowNs();
            }
            std.atomic.spinLoopHint();
        }
    }

    /// The expert's weights, loading them first if needed. Keeps its slot
    /// until `release`.
    pub fn acquire(self: *ExpertCache, layer: usize, expert: usize) !Held
    {
        var w = try self.want(layer, expert);
        return self.wait(&w);
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

    const Job = struct {
        key: u32,
        slot: usize,
        from_hint: bool,
    };

    fn work(self: *ExpertCache, i: usize) void
    {
        const src = self.sources[i];
        while (!self.stop.load(.acquire))
        {
            self.lock();
            const job = self.next();
            self.unlock();
            const j = job orelse
            {
                idle();
                continue;
            };
            self.fill(src, j.key, j.slot, j.from_hint) catch |err|
            {
                std.log.err("expert cache: loading layer {d} expert {d}: {t}", .{ j.key / self.cfg.experts, j.key % self.cfg.experts, err });
                self.failed.store(true, .release);
                return;
            };
        }
    }

    /// Picks the next load and claims a slot for it, or returns null if there
    /// is nothing to do now. Called under the lock.
    fn next(self: *ExpertCache) ?Job
    {
        while (true)
        {
            var from_hint = false;
            var key: u32 = undefined;
            if (self.retry != none)
            {
                key = self.retry;
                self.retry = none;
            }
            else if (self.demand.pop()) |k|
            {
                key = k;
            }
            else if (self.hints.pop()) |h|
            {
                if (h.pos <= self.pos.load(.acquire))
                {
                    bump(&self.stats.cancelled, 1);
                    continue;
                }
                key = h.key;
                from_hint = true;
            }
            else
            {
                return null;
            }
            if (self.where[key].load(.acquire) != none)
            {
                continue;
            }
            const victim = self.evict() orelse
            {
                // Every slot is in use; try again once a call finishes.
                if (!from_hint)
                {
                    self.retry = key;
                }
                return null;
            };
            self.claim(key, victim, from_hint);
            return .{ .key = key, .slot = victim, .from_hint = from_hint };
        }
    }

    /// Frees the least recently used unpinned slot and returns it, marked
    /// loading, or null if every slot is pinned, in use, or loading.
    fn evict(self: *ExpertCache) ?usize
    {
        while (true)
        {
            // Index 0 skips experts loaded on demand and not yet used; 1 takes any.
            var best: [2]?usize = .{ null, null };
            var oldest: [2]u64 = @splat(std.math.maxInt(u64));
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
                if (st != .ready)
                {
                    continue;
                }
                const used = s.last_use.load(.acquire);
                for (0..2) |tier|
                {
                    if ((tier == 1 or !s.demanded.load(.acquire)) and used < oldest[tier])
                    {
                        oldest[tier] = used;
                        best[tier] = i;
                    }
                }
            }
            const i = best[0] orelse best[1] orelse return null;
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
                    bump(&self.stats.wasted, 1);
                }
            }
            return i;
        }
    }

    /// Points `key` at slot `si`, marked loading, so no other loader takes it.
    fn claim(self: *ExpertCache, key: u32, si: usize, from_hint: bool) void
    {
        const s = &self.slots[si];
        s.state.store(.loading, .seq_cst);
        s.from_hint.store(from_hint, .release);
        s.demanded.store(!from_hint, .release);
        s.key.store(key, .seq_cst);
        self.where[key].store(@intCast(si), .release);
    }

    /// Reads one expert's three weights into its claimed slot and publishes it.
    fn fill(self: *ExpertCache, src: Source, key: u32, si: usize, from_hint: bool) !void
    {
        const t0 = nowNs();
        const s = &self.slots[si];
        const layer: u32 = key / @as(u32, @intCast(self.cfg.experts));
        const e: u32 = key % @as(u32, @intCast(self.cfg.experts));
        var mats: [3]olmoe.Mat = undefined;
        var off: usize = 0;
        var bytes_read: u64 = 0;
        inline for (.{ loader.Projection.gate, loader.Projection.up, loader.Projection.down }, 0..) |proj, p|
        {
            const w = self.model.weight(layer, proj, e);
            const r = w.parts[0];
            const dst = s.bytes[off..][0..self.part_cap[p]];
            const at = try src.read(src.ctx, r, dst);
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
            bytes_read += r.len;
        }
        self.checked[key] = true;
        s.expert = .{ .gate = mats[0], .up = mats[1], .down = mats[2] };
        s.last_use.store(self.clock.fetchAdd(1, .monotonic), .release);
        s.state.store(.ready, .seq_cst);
        bump(&self.stats.loads, 1);
        bump(&self.stats.hint_loads, @intFromBool(from_hint));
        bump(&self.stats.bytes, bytes_read);
        bump(&self.stats.load_ns, nowNs() - t0);
    }

    fn lock(self: *ExpertCache) void
    {
        while (self.locked.swap(true, .acquire))
        {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *ExpertCache) void
    {
        self.locked.store(false, .release);
    }
};

/// Loader threads share their counters.
fn bump(p: *u64, v: u64) void
{
    _ = @atomicRmw(u64, p, .Add, v, .monotonic);
}

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

const Tiny = struct {
    arena: std.heap.ArenaAllocator,
    h: @import("gguf.zig").Header,
    m: loader.Model,
    cfg: olmoe.Config,
    all: olmoe.LocalExperts,
    mem: MemorySource,
    owned: [8]bool = @splat(true),

    const file = @embedFile("olmoe_tiny.gguf");

    fn init(self: *Tiny, gpa: std.mem.Allocator) !void
    {
        const gguf = @import("gguf.zig");
        self.h = try gguf.parse(gpa, file, file.len);
        self.cfg = try olmoe.Config.fromGguf(self.h);
        self.m = try loader.Model.fromGguf(gpa, self.h);
        var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
        var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
        defer plan.deinit(gpa);
        self.arena = .init(gpa);
        self.owned = @splat(true);
        self.all = try olmoe.LocalExperts.init(self.arena.allocator(), self.cfg, self.m, self.owned[0..self.cfg.experts], plan, file);
        self.mem = .{ .file = file };
    }

    fn deinit(self: *Tiny, gpa: std.mem.Allocator) void
    {
        self.arena.deinit();
        self.m.deinit(gpa);
        self.h.deinit(gpa);
    }

    /// A cache with layer 0 pinned plus `free` shared slots, `loaders` threads.
    fn cache(self: *Tiny, free: usize, loaders: usize) !*ExpertCache
    {
        const a = self.arena.allocator();
        const src = self.mem.source();
        const sources = try a.alloc(Source, loaders);
        @memset(sources, src);
        const slot_bytes = try ExpertCache.slotBytes(self.m, src.pad, src.alignment);
        const region = try a.alignedAlloc(u8, .fromByteUnits(64), (self.cfg.experts + free) * slot_bytes);
        const c = try ExpertCache.init(a, self.cfg, self.m, self.owned[0..self.cfg.experts], sources, null, region, 1);
        try c.start();
        return c;
    }

    fn input(self: *Tiny) ![]f32
    {
        const x = try self.arena.allocator().alloc(f32, self.cfg.d_model);
        for (x, 0..) |*v, i|
        {
            v.* = @sin(@as(f32, @floatFromInt(i)));
        }
        return x;
    }
};

test "a cache of two slots gives the same experts as loading them all" {
    const gpa = std.testing.allocator;
    var t: Tiny = undefined;
    try t.init(gpa);
    defer t.deinit(gpa);
    const a = t.arena.allocator();
    const cfg = t.cfg;
    // Layer 0 pinned (8 experts) plus two slots that every layer-1 expert shares.
    const cache = try t.cache(2, 1);
    defer cache.deinit();
    const x = try t.input();
    const want = try a.alloc(f32, cfg.d_model);
    const got = try a.alloc(f32, cfg.d_model);
    for (0..3) |round|
    {
        for (0..cfg.layers) |l|
        {
            const pos = (round + 1) * cfg.layers + l;
            cache.enter(pos);
            for (0..cfg.experts) |e|
            {
                if (l == 1 and e % 2 == 0)
                {
                    cache.hint(pos + 1, l, e + 1);
                }
                t.all.get(l, e).?.forward(x, want, t.all.scratch);
                const held = try cache.acquire(l, e);
                held.expert.forward(x, got, t.all.scratch);
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

test "several loaders, polling, and stale hints give the same experts" {
    const gpa = std.testing.allocator;
    var t: Tiny = undefined;
    try t.init(gpa);
    defer t.deinit(gpa);
    const a = t.arena.allocator();
    const cfg = t.cfg;
    const cache = try t.cache(3, 3);
    defer cache.deinit();
    const x = try t.input();
    const want = try a.alloc(f32, cfg.d_model);
    const got = try a.alloc(f32, cfg.d_model);
    var wants: [8]Want = undefined;
    var done: [8]bool = undefined;
    for (0..20) |round|
    {
        for (0..cfg.layers) |l|
        {
            const pos = (round + 1) * cfg.layers + l;
            // Hints for this layer and the one before are already stale.
            cache.hint(pos - 1, l, (round + 3) % cfg.experts);
            cache.enter(pos);
            cache.hint(pos, l, (round + 5) % cfg.experts);
            // Ask for every expert at once, then run each as it becomes ready.
            for (&wants, 0..) |*w, e|
            {
                w.* = try cache.want(l, e);
            }
            @memset(&done, false);
            var left: usize = cfg.experts;
            while (left > 0)
            {
                for (&wants, 0..) |*w, e|
                {
                    if (done[e])
                    {
                        continue;
                    }
                    const held = (try cache.poll(w)) orelse continue;
                    t.all.get(l, e).?.forward(x, want, t.all.scratch);
                    held.expert.forward(x, got, t.all.scratch);
                    cache.release(held);
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
                    done[e] = true;
                    left -= 1;
                }
            }
        }
    }
    cache.deinit();
    const st = cache.stats;
    try std.testing.expectEqual(@as(u64, 20 * cfg.layers * cfg.experts), st.calls());
    try std.testing.expectEqual(@as(u64, 0), st.hint_loads);
    try std.testing.expect(st.cancelled > 0);
}
