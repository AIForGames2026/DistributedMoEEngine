//! Telemetry (TASK-16b, TASK-16): which experts each engine step used, on
//! which rank, streamed to the built-in page over a WebSocket and, with
//! `--record`, written to a file the page replays.
//!
//! `Tap` wraps the engine's `Moe`. Only while a viewer is connected does it
//! count each layer's expert choices and time the layer; at the end of every
//! step it copies the counts into a slot of `Feed`, a single-producer ring
//! the engine thread never waits on (a full ring drops the step, and the
//! viewer sees a gap in `seq`). `Broadcaster` owns the viewer sockets on its
//! own thread, turns slots into JSON frames, and drops frames for a viewer
//! that falls behind instead of slowing anyone down. A recording is the same
//! frames, one per line, with the chat's notes placed among the steps.
//!
//! Frames (text, one JSON object each):
//!   {"t":"hello", model, layers, experts, top_k, transport, ranks, owner, chunks, cache_ranks, ms}
//!   {"t":"step", seq, ms, n, step_us, layer_us, x: [layer, expert, count, ...],
//!    engine: [rank 0 calls, rank 1 calls, ...] (one per rank, at least two; null where the engine keeps no count),
//!    rtt_us, rtt_layers, cache: {resident, hinted, late, missed, loads} or null,
//!    c: [layer, expert, status, ...] (rank 0's cache), h: [for layer, expert, sent during layer, ...]}
//! Recordings add `recorded` (Unix seconds) to the hello, notes from the
//! server (`chat`, `text`, `done`), each with `after`, the seq of the step
//! before it, and a closing {"t":"end", after, steps, dropped, ms}.
const std = @import("std");
const olmoe = @import("olmoe.zig");

/// The most processes a run has (TASK-33), as the transport's 6-bit rank allows.
pub const max_ranks = 64;

/// What the engine itself counted, copied into each step's frame.
pub const Counters = struct {
    /// Expert calls by rank, counted where they ran or were sent; null where no count is kept.
    calls: [max_ranks]?u64 = @splat(null),
    /// Time from the first dispatch to the last combine, summed over the layers that had remote calls.
    rtt_ns: u64 = 0,
    rtt_layers: u64 = 0,
    cache: ?CacheCounts = null,
};

pub const CacheCounts = struct {
    resident: u64,
    hinted: u64,
    late: u64,
    missed: u64,
    loads: u64,
};

/// How an expert call found its expert in rank 0's cache, as step frames send it (`c`).
pub const CacheStatus = enum(u8) { none = 0, resident = 1, hinted = 2, late = 3, missed = 4 };

/// Each (layer, expert)'s status this step, kept by the cache while the tap is counting. Engine thread only.
pub const CacheTrace = struct {
    on: bool = false,
    status: []u8,

    /// Keeps the first call's status for each (layer, expert) in a step.
    pub fn note(self: *CacheTrace, key: usize, how: CacheStatus) void
    {
        if (self.on and self.status[key] == 0)
        {
            self.status[key] = @intFromEnum(how);
        }
    }
};

/// Reads the engine's own counters on the engine thread.
pub const Probe = struct {
    ctx: ?*anyopaque = null,
    sample: ?*const fn (ctx: *anyopaque, out: *Counters) void = null,
};

/// What the page draws: the processes and which of them holds each expert.
pub const Topology = struct {
    model: []const u8,
    transport: []const u8,
    ranks: []const []const u8,
    /// For each expert, the rank holding it (the same in every layer).
    owner: []const u8,
    /// Packets per remote expert call, each way; 0 when nothing is remote.
    chunks: usize = 0,
    /// Ranks whose experts come through an expert cache (TASK-41).
    cache_ranks: []const u8 = &.{},
};

const Slot = struct {
    seq: u64 = 0,
    t_ns: u64 = 0,
    tokens: u32 = 0,
    step_ns: u64 = 0,
    counts: []u32,
    layer_ns: []u64,
    counters: Counters = .{},
    /// Per (layer, expert): a `CacheStatus`.
    status: []u8,
    /// Prefetch hints as (layer it is for, expert, layer it was sent during).
    hints: []u16,
    hint_count: usize = 0,
    hints_dropped: usize = 0,
};

/// The ring between the engine thread (producer) and the broadcaster (consumer).
pub const Feed = struct {
    slots: []Slot,
    layers: usize,
    experts: usize,
    head: std.atomic.Value(u64) align(64) = .init(0),
    tail: std.atomic.Value(u64) align(64) = .init(0),
    viewers: std.atomic.Value(u32) align(64) = .init(0),
    /// Steps lost to a full ring.
    dropped: std.atomic.Value(u64) = .init(0),

    pub fn init(gpa: std.mem.Allocator, layers: usize, experts: usize, slots: usize) !Feed
    {
        const le = layers * experts;
        const all = try gpa.alloc(Slot, slots);
        errdefer gpa.free(all);
        const counts = try gpa.alloc(u32, slots * le);
        errdefer gpa.free(counts);
        const times = try gpa.alloc(u64, slots * layers);
        errdefer gpa.free(times);
        const status = try gpa.alloc(u8, slots * le);
        errdefer gpa.free(status);
        // Each layer's experts are guessed at most once per step, so a step has at most one hint per (layer, expert).
        const hints = try gpa.alloc(u16, slots * le * 3);
        for (all, 0..) |*s, i|
        {
            s.* = .{
                .counts = counts[i * le ..][0..le],
                .layer_ns = times[i * layers ..][0..layers],
                .status = status[i * le ..][0..le],
                .hints = hints[i * le * 3 ..][0 .. le * 3],
            };
        }
        return .{ .slots = all, .layers = layers, .experts = experts };
    }

    pub fn deinit(self: *Feed, gpa: std.mem.Allocator) void
    {
        const le = self.layers * self.experts;
        gpa.free(self.slots[0].counts.ptr[0 .. self.slots.len * le]);
        gpa.free(self.slots[0].layer_ns.ptr[0 .. self.slots.len * self.layers]);
        gpa.free(self.slots[0].status.ptr[0 .. self.slots.len * le]);
        gpa.free(self.slots[0].hints.ptr[0 .. self.slots.len * le * 3]);
        gpa.free(self.slots);
    }

    /// The oldest unread slot, or null.
    pub fn peek(self: *Feed) ?*const Slot
    {
        const t = self.tail.load(.monotonic);
        if (t == self.head.load(.acquire))
        {
            return null;
        }
        return &self.slots[t % self.slots.len];
    }

    pub fn pop(self: *Feed) void
    {
        _ = self.tail.fetchAdd(1, .release);
    }
};

/// Wraps the engine's `Moe` and records each step for `Feed`.
pub const Tap = struct {
    inner: olmoe.Moe,
    top_k: usize,
    feed: *Feed,
    probe: Probe,
    clock: *const fn () u64,
    counts: []u32,
    layer_ns: []u64,
    /// Filled by an expert cache pointed at it (TASK-41).
    trace: CacheTrace,
    hints: []u16,
    hint_count: usize = 0,
    hints_dropped: usize = 0,
    /// The layer whose experts run next this step, which hints are sent during.
    layer_now: usize = 0,
    /// Counting this step: a viewer was connected when it began.
    active: bool = false,
    step_t0: u64 = 0,
    next_seq: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, inner: olmoe.Moe, top_k: usize, feed: *Feed, probe: Probe, clock: *const fn () u64) !Tap
    {
        const le = feed.layers * feed.experts;
        const counts = try gpa.alloc(u32, le);
        errdefer gpa.free(counts);
        const layer_ns = try gpa.alloc(u64, feed.layers);
        errdefer gpa.free(layer_ns);
        const status = try gpa.alloc(u8, le);
        errdefer gpa.free(status);
        @memset(status, 0);
        const hints = try gpa.alloc(u16, le * 3);
        return .{
            .inner = inner, .top_k = top_k, .feed = feed, .probe = probe, .clock = clock, .counts = counts, .layer_ns = layer_ns,
            .trace = .{ .status = status }, .hints = hints,
        };
    }

    pub fn deinit(self: *Tap, gpa: std.mem.Allocator) void
    {
        gpa.free(self.counts);
        gpa.free(self.layer_ns);
        gpa.free(self.trace.status);
        gpa.free(self.hints);
    }

    pub fn moe(self: *Tap) olmoe.Moe
    {
        return .{ .ctx = self, .run = run, .ahead = self.inner.ahead, .prefetch = if (self.inner.prefetch != null) prefetch else null, .mark = mark, .capacity = self.inner.capacity };
    }

    fn mark(ctx: *anyopaque, begin: bool, tokens: usize) void
    {
        const self: *Tap = @ptrCast(@alignCast(ctx));
        if (begin)
        {
            self.beginStep();
        }
        else
        {
            self.endStep(tokens);
        }
        if (self.inner.mark) |m|
        {
            m(self.inner.ctx, begin, tokens);
        }
    }

    fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void
    {
        const self: *Tap = @ptrCast(@alignCast(ctx));
        if (!self.active)
        {
            return self.inner.run(self.inner.ctx, layer, x, n, idx, w, out);
        }
        const t0 = self.clock();
        try self.inner.run(self.inner.ctx, layer, x, n, idx, w, out);
        self.layer_ns[layer] += self.clock() - t0;
        self.layer_now = layer + 1;
        const row = self.counts[layer * self.feed.experts ..][0..self.feed.experts];
        for (idx[0 .. n * self.top_k]) |e|
        {
            if (e != olmoe.Capacity.skip)
            {
                row[e] += 1;
            }
        }
    }

    fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
    {
        const self: *Tap = @ptrCast(@alignCast(ctx));
        if (self.active)
        {
            for (experts) |e|
            {
                if (self.hint_count * 3 >= self.hints.len)
                {
                    self.hints_dropped += 1;
                    continue;
                }
                self.hints[self.hint_count * 3 ..][0..3].* = .{ @intCast(layer), @intCast(e), @intCast(@min(self.layer_now, self.feed.layers - 1)) };
                self.hint_count += 1;
            }
        }
        return self.inner.prefetch.?(self.inner.ctx, layer, experts);
    }

    /// Called as each engine step begins.
    pub fn beginStep(self: *Tap) void
    {
        self.active = self.feed.viewers.load(.monotonic) > 0;
        self.trace.on = self.active;
        if (!self.active)
        {
            return;
        }
        @memset(self.counts, 0);
        @memset(self.layer_ns, 0);
        @memset(self.trace.status, 0);
        self.hint_count = 0;
        self.hints_dropped = 0;
        self.layer_now = 0;
        self.step_t0 = self.clock();
    }

    /// Called after each engine step of `tokens` tokens; publishes it if it was counted.
    pub fn endStep(self: *Tap, tokens: usize) void
    {
        if (!self.active)
        {
            return;
        }
        self.active = false;
        self.trace.on = false;
        const seq = self.next_seq;
        self.next_seq += 1;
        const f = self.feed;
        const h = f.head.load(.monotonic);
        if (h - f.tail.load(.acquire) >= f.slots.len)
        {
            _ = f.dropped.fetchAdd(1, .monotonic);
            return;
        }
        const s = &f.slots[h % f.slots.len];
        const now = self.clock();
        s.seq = seq;
        s.t_ns = now;
        s.tokens = @intCast(tokens);
        s.step_ns = now - self.step_t0;
        @memcpy(s.counts, self.counts);
        @memcpy(s.layer_ns, self.layer_ns);
        @memcpy(s.status, self.trace.status);
        @memcpy(s.hints[0 .. self.hint_count * 3], self.hints[0 .. self.hint_count * 3]);
        s.hint_count = self.hint_count;
        s.hints_dropped = self.hints_dropped;
        s.counters = .{};
        if (self.probe.sample) |sample|
        {
            sample(self.probe.ctx.?, &s.counters);
        }
        f.head.store(h + 1, .release);
    }
};

/// The `Sec-WebSocket-Accept` value for a client's key.
pub fn acceptKey(key: []const u8, out: *[28]u8) []const u8
{
    var sha = std.crypto.hash.Sha1.init(.{});
    sha.update(key);
    sha.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    var digest: [20]u8 = undefined;
    sha.final(&digest);
    return std.base64.standard.Encoder.encode(out, &digest);
}

/// Appends a server-to-client frame (unmasked, final).
pub fn appendFrame(gpa: std.mem.Allocator, out: *std.ArrayList(u8), opcode: u8, payload: []const u8) !void
{
    try out.append(gpa, 0x80 | opcode);
    if (payload.len < 126)
    {
        try out.append(gpa, @intCast(payload.len));
    }
    else if (payload.len <= 0xFFFF)
    {
        try out.append(gpa, 126);
        try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(payload.len))));
    }
    else
    {
        try out.append(gpa, 127);
        try out.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u64, payload.len)));
    }
    try out.appendSlice(gpa, payload);
}

/// One client-to-server frame, unmasked in place.
pub const InFrame = struct {
    opcode: u8,
    payload: []const u8,
    len: usize,
};

/// Parses the frame at the start of `buf`: null until it has all arrived.
pub fn parseFrame(buf: []u8) error{BadFrame}!?InFrame
{
    if (buf.len < 2)
    {
        return null;
    }
    // Clients must mask what they send.
    if (buf[1] & 0x80 == 0)
    {
        return error.BadFrame;
    }
    var n: u64 = buf[1] & 0x7F;
    var at: usize = 2;
    if (n == 126)
    {
        if (buf.len < 4)
        {
            return null;
        }
        n = std.mem.readInt(u16, buf[2..4], .big);
        at = 4;
    }
    else if (n == 127)
    {
        if (buf.len < 10)
        {
            return null;
        }
        n = std.mem.readInt(u64, buf[2..10], .big);
        at = 10;
    }
    if (n > 1 << 16)
    {
        return error.BadFrame;
    }
    if (buf.len < at + 4 + n)
    {
        return null;
    }
    const mask = buf[at..][0..4].*;
    const payload = buf[at + 4 ..][0..@intCast(n)];
    for (payload, 0..) |*b, i|
    {
        b.* ^= mask[i % 4];
    }
    return .{ .opcode = buf[0] & 0x0F, .payload = payload, .len = at + 4 + @as(usize, @intCast(n)) };
}

pub fn writeHello(gpa: std.mem.Allocator, out: *std.ArrayList(u8), topo: Topology, layers: usize, experts: usize, top_k: usize, now_ns: u64, recorded: ?i64) !void
{
    const openai = @import("openai.zig");
    try out.appendSlice(gpa, "{\"t\":\"hello\",\"schema\":\"tensorfold.live\",\"version\":0,\"model\":");
    try openai.writeString(gpa, out, topo.model);
    try out.print(gpa, ",\"layers\":{d},\"experts\":{d},\"top_k\":{d},\"transport\":", .{ layers, experts, top_k });
    try openai.writeString(gpa, out, topo.transport);
    try out.appendSlice(gpa, ",\"ranks\":[");
    for (topo.ranks, 0..) |r, i|
    {
        if (i > 0)
        {
            try out.append(gpa, ',');
        }
        try openai.writeString(gpa, out, r);
    }
    try out.appendSlice(gpa, "],\"owner\":[");
    for (topo.owner, 0..) |o, i|
    {
        try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", o });
    }
    try out.print(gpa, "],\"chunks\":{d},\"cache_ranks\":[", .{topo.chunks});
    for (topo.cache_ranks, 0..) |r, i|
    {
        try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", r });
    }
    try out.print(gpa, "],\"ms\":{d:.3}", .{ms(now_ns)});
    if (recorded) |t|
    {
        try out.print(gpa, ",\"recorded\":{d}", .{t});
    }
    try out.append(gpa, '}');
}

pub fn writeStep(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: *const Slot, experts: usize, ranks: usize) !void
{
    try out.print(gpa, "{{\"t\":\"step\",\"seq\":{d},\"ms\":{d:.3},\"n\":{d},\"step_us\":{d:.1},\"layer_us\":[", .{ s.seq, ms(s.t_ns), s.tokens, us(s.step_ns) });
    for (s.layer_ns, 0..) |t, i|
    {
        try out.print(gpa, "{s}{d:.1}", .{ if (i == 0) "" else ",", us(t) });
    }
    try out.appendSlice(gpa, "],\"x\":[");
    var first = true;
    for (s.counts, 0..) |count, i|
    {
        if (count == 0)
        {
            continue;
        }
        try out.print(gpa, "{s}{d},{d},{d}", .{ if (first) "" else ",", i / experts, i % experts, count });
        first = false;
    }
    const c = s.counters;
    try out.appendSlice(gpa, "],\"engine\":[");
    for (c.calls[0..@max(ranks, 2)], 0..) |v, i|
    {
        try out.appendSlice(gpa, if (i == 0) "" else ",");
        if (v) |n|
        {
            try out.print(gpa, "{d}", .{n});
        }
        else
        {
            try out.appendSlice(gpa, "null");
        }
    }
    try out.print(gpa, "],\"rtt_us\":{d:.1},\"rtt_layers\":{d},\"cache\":", .{ us(c.rtt_ns), c.rtt_layers });
    if (c.cache) |k|
    {
        try out.print(gpa, "{{\"resident\":{d},\"hinted\":{d},\"late\":{d},\"missed\":{d},\"loads\":{d}}}", .{ k.resident, k.hinted, k.late, k.missed, k.loads });
    }
    else
    {
        try out.appendSlice(gpa, "null");
    }
    try out.appendSlice(gpa, ",\"c\":[");
    first = true;
    for (s.status, 0..) |st, i|
    {
        if (st == 0)
        {
            continue;
        }
        try out.print(gpa, "{s}{d},{d},{d}", .{ if (first) "" else ",", i / experts, i % experts, st });
        first = false;
    }
    try out.appendSlice(gpa, "],\"h\":[");
    for (s.hints[0 .. s.hint_count * 3], 0..) |v, i|
    {
        try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", v });
    }
    try out.append(gpa, ']');
    if (s.hints_dropped > 0)
    {
        try out.print(gpa, ",\"h_dropped\":{d}", .{s.hints_dropped});
    }
    try out.append(gpa, '}');
}

fn ms(ns: u64) f64
{
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn us(ns: u64) f64
{
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

/// Sends the feed to every connected viewer, and to the recording if there
/// is one, from its own thread. Linux.
pub const Broadcaster = struct {
    gpa: std.mem.Allocator,
    feed: *Feed,
    topo: Topology,
    top_k: usize,
    clock: *const fn () u64,
    /// The front thread writes each new viewer's socket here.
    pipe: [2]std.c.fd_t = .{ -1, -1 },
    viewers: std.ArrayList(Viewer) = .empty,
    json: std.ArrayList(u8) = .empty,
    frame: std.ArrayList(u8) = .empty,
    /// Set by `record`; the recording counts as a viewer that never falls behind.
    recording: bool = false,
    rec_fd: std.c.fd_t = -1,
    rec: std.ArrayList(u8) = .empty,
    /// Notes from the engine thread, then held until the steps before them are written.
    notes: std.ArrayList(Note) = .empty,
    notes_lock: std.atomic.Value(bool) = .init(false),
    held: std.ArrayList(Note) = .empty,
    last_seq: ?u64 = null,
    steps_written: u64 = 0,
    stopping: std.atomic.Value(bool) = .init(false),
    stopped: std.atomic.Value(bool) = .init(false),

    const Note = struct {
        /// How many steps had been published when it was noted.
        head: u64,
        /// A JSON object.
        bytes: []u8,
    };

    /// A viewer may fall this far behind before its frames are dropped.
    const max_pending = 256 << 10;
    /// Kept small so a slow viewer gets dropped frames rather than stale ones.
    const kernel_buffer: c_int = 128 << 10;

    const Viewer = struct {
        fd: std.c.fd_t,
        out: std.ArrayList(u8) = .empty,
        sent: usize = 0,
        in: std.ArrayList(u8) = .empty,
        closed: bool = false,
        /// Close once `out` has drained.
        closing: bool = false,
    };

    pub fn start(self: *Broadcaster) !void
    {
        if (self.pipe[0] >= 0)
        {
            return;
        }
        if (std.c.pipe(&self.pipe) != 0)
        {
            return error.SystemResources;
        }
        const t = try std.Thread.spawn(.{}, loop, .{self});
        t.detach();
    }

    /// Writes every step to `path` from now on, starting with the hello. Call before `start`.
    pub fn record(self: *Broadcaster, path: [:0]const u8, unix: i64) !void
    {
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0)
        {
            return error.CannotRecord;
        }
        self.rec_fd = fd;
        try writeHello(self.gpa, &self.rec, self.topo, self.feed.layers, self.feed.experts, self.top_k, self.clock(), unix);
        try self.rec.append(self.gpa, '\n');
        self.recording = true;
        _ = self.feed.viewers.fetchAdd(1, .monotonic);
    }

    /// Adds a JSON object to the recording after the steps published so far. Engine thread only.
    pub fn note(self: *Broadcaster, bytes: []const u8) void
    {
        if (!self.recording)
        {
            return;
        }
        const copy = self.gpa.dupe(u8, bytes) catch return;
        const head = self.feed.head.load(.monotonic);
        self.lock();
        defer self.unlock();
        self.notes.append(self.gpa, .{ .head = head, .bytes = copy }) catch self.gpa.free(copy);
    }

    /// Writes out what is left and closes the recording; call once the engine has stopped.
    pub fn finish(self: *Broadcaster) void
    {
        if (!self.recording)
        {
            return;
        }
        self.stopping.store(true, .release);
        while (!self.stopped.load(.acquire))
        {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    fn lock(self: *Broadcaster) void
    {
        while (self.notes_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null)
        {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *Broadcaster) void
    {
        self.notes_lock.store(false, .release);
    }

    /// Hands over a socket that has finished the WebSocket handshake.
    pub fn add(self: *Broadcaster, fd: std.c.fd_t) bool
    {
        return std.c.write(self.pipe[1], std.mem.asBytes(&fd), @sizeOf(std.c.fd_t)) == @sizeOf(std.c.fd_t);
    }

    fn loop(self: *Broadcaster) void
    {
        var fds: std.ArrayList(std.c.pollfd) = .empty;
        while (true)
        {
            fds.clearRetainingCapacity();
            fds.append(self.gpa, .{ .fd = self.pipe[0], .events = std.c.POLL.IN, .revents = 0 }) catch return;
            for (self.viewers.items) |*v|
            {
                const want_out: i16 = if (v.out.items.len > v.sent) std.c.POLL.OUT else 0;
                fds.append(self.gpa, .{ .fd = v.fd, .events = std.c.POLL.IN | want_out, .revents = 0 }) catch return;
            }
            _ = std.c.poll(fds.items.ptr, @intCast(fds.items.len), 10);
            // Read before draining, so every step the engine published before stopping is written.
            const stop = self.recording and self.stopping.load(.acquire);
            for (fds.items[1..], self.viewers.items) |p, *v|
            {
                if (p.revents & (std.c.POLL.IN | std.c.POLL.HUP | std.c.POLL.ERR) != 0)
                {
                    self.readFrom(v);
                }
            }
            if (fds.items[0].revents & std.c.POLL.IN != 0)
            {
                var fd: std.c.fd_t = -1;
                if (std.c.read(self.pipe[0], std.mem.asBytes(&fd), @sizeOf(std.c.fd_t)) == @sizeOf(std.c.fd_t))
                {
                    self.welcome(fd);
                }
            }
            self.drain();
            if (self.recording)
            {
                self.writeRecording();
                if (stop)
                {
                    self.closeRecording();
                    self.stopped.store(true, .release);
                }
            }
            var i: usize = 0;
            while (i < self.viewers.items.len)
            {
                const v = &self.viewers.items[i];
                self.flush(v);
                if (v.closing and v.sent == v.out.items.len)
                {
                    v.closed = true;
                }
                if (!v.closed)
                {
                    i += 1;
                    continue;
                }
                _ = std.c.close(v.fd);
                v.out.deinit(self.gpa);
                v.in.deinit(self.gpa);
                _ = self.viewers.swapRemove(i);
                _ = self.feed.viewers.fetchSub(1, .monotonic);
            }
        }
    }

    fn welcome(self: *Broadcaster, fd: std.c.fd_t) void
    {
        self.json.clearRetainingCapacity();
        var v: Viewer = .{ .fd = fd };
        _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.SNDBUF, std.mem.asBytes(&kernel_buffer), @sizeOf(c_int));
        writeHello(self.gpa, &self.json, self.topo, self.feed.layers, self.feed.experts, self.top_k, self.clock(), null) catch return;
        appendFrame(self.gpa, &v.out, 0x1, self.json.items) catch return;
        self.viewers.append(self.gpa, v) catch
        {
            _ = std.c.close(fd);
            return;
        };
        _ = self.feed.viewers.fetchAdd(1, .monotonic);
    }

    /// Turns every waiting slot into a frame for each viewer with room for it, and a line of the recording.
    fn drain(self: *Broadcaster) void
    {
        // Notes taken after this point may belong after steps published since, so those wait for the next round.
        const head = self.feed.head.load(.acquire);
        if (self.recording)
        {
            self.lock();
            self.held.appendSlice(self.gpa, self.notes.items) catch {};
            self.notes.clearRetainingCapacity();
            self.unlock();
        }
        while (self.feed.tail.load(.monotonic) < head)
        {
            const at = self.feed.tail.load(.monotonic);
            const s = self.feed.peek().?;
            if (self.recording)
            {
                self.writeNotes(at);
            }
            self.json.clearRetainingCapacity();
            self.frame.clearRetainingCapacity();
            const ok = blk: {
                writeStep(self.gpa, &self.json, s, self.feed.experts, self.topo.ranks.len) catch break :blk false;
                if (self.recording)
                {
                    self.rec.appendSlice(self.gpa, self.json.items) catch break :blk false;
                    self.rec.append(self.gpa, '\n') catch break :blk false;
                    self.last_seq = s.seq;
                    self.steps_written += 1;
                }
                appendFrame(self.gpa, &self.frame, 0x1, self.json.items) catch break :blk false;
                break :blk true;
            };
            self.feed.pop();
            if (!ok)
            {
                continue;
            }
            for (self.viewers.items) |*v|
            {
                if (v.closing or v.out.items.len - v.sent > max_pending)
                {
                    continue;
                }
                v.out.appendSlice(self.gpa, self.frame.items) catch continue;
            }
        }
        if (self.recording)
        {
            self.writeNotes(head);
        }
    }

    /// Adds the held notes noted after at most `published` steps to the recording.
    fn writeNotes(self: *Broadcaster, published: u64) void
    {
        var done: usize = 0;
        for (self.held.items) |n|
        {
            if (n.head > published)
            {
                break;
            }
            if (self.last_seq) |s|
            {
                self.rec.print(self.gpa, "{{\"after\":{d},", .{s}) catch {};
            }
            else
            {
                self.rec.appendSlice(self.gpa, "{\"after\":null,") catch {};
            }
            self.rec.appendSlice(self.gpa, n.bytes[1..]) catch {};
            self.rec.append(self.gpa, '\n') catch {};
            self.gpa.free(n.bytes);
            done += 1;
        }
        std.mem.copyForwards(Note, self.held.items, self.held.items[done..]);
        self.held.items.len -= done;
    }

    fn writeRecording(self: *Broadcaster) void
    {
        var off: usize = 0;
        while (off < self.rec.items.len and self.rec_fd >= 0)
        {
            const n = std.c.write(self.rec_fd, self.rec.items[off..].ptr, self.rec.items.len - off);
            if (n <= 0)
            {
                std.debug.print("telemetry: writing the recording failed; it stops here\n", .{});
                _ = std.c.close(self.rec_fd);
                self.rec_fd = -1;
                break;
            }
            off += @intCast(n);
        }
        self.rec.clearRetainingCapacity();
    }

    fn closeRecording(self: *Broadcaster) void
    {
        self.writeNotes(std.math.maxInt(u64));
        self.rec.print(self.gpa, "{{\"t\":\"end\",\"after\":{?d},\"steps\":{d},\"dropped\":{d},\"ms\":{d:.3}}}\n", .{
            self.last_seq, self.steps_written, self.feed.dropped.load(.monotonic), ms(self.clock()),
        }) catch {};
        self.writeRecording();
        if (self.rec_fd >= 0)
        {
            _ = std.c.close(self.rec_fd);
            self.rec_fd = -1;
        }
        self.recording = false;
    }

    fn flush(self: *Broadcaster, v: *Viewer) void
    {
        _ = self;
        while (v.sent < v.out.items.len)
        {
            const rest = v.out.items[v.sent..];
            const n = std.c.send(v.fd, rest.ptr, rest.len, std.c.MSG.DONTWAIT | std.c.MSG.NOSIGNAL);
            if (n > 0)
            {
                v.sent += @intCast(n);
                continue;
            }
            if (n < 0 and std.c._errno().* == @intFromEnum(std.c.E.AGAIN))
            {
                break;
            }
            v.closed = true;
            return;
        }
        if (v.sent == v.out.items.len)
        {
            v.out.clearRetainingCapacity();
            v.sent = 0;
        }
    }

    /// Reads what the viewer sent: answers pings and close frames, ignores the rest.
    fn readFrom(self: *Broadcaster, v: *Viewer) void
    {
        var buf: [4096]u8 = undefined;
        const n = std.c.recv(v.fd, &buf, buf.len, std.c.MSG.DONTWAIT);
        if (n == 0 or (n < 0 and std.c._errno().* != @intFromEnum(std.c.E.AGAIN)))
        {
            v.closed = true;
            return;
        }
        if (n < 0)
        {
            return;
        }
        v.in.appendSlice(self.gpa, buf[0..@intCast(n)]) catch
        {
            v.closed = true;
            return;
        };
        while (true)
        {
            const f = (parseFrame(v.in.items) catch
            {
                v.closed = true;
                return;
            }) orelse break;
            switch (f.opcode)
            {
                0x8 => {
                    appendFrame(self.gpa, &v.out, 0x8, f.payload[0..@min(f.payload.len, 2)]) catch {};
                    v.closing = true;
                },
                0x9 => appendFrame(self.gpa, &v.out, 0xA, f.payload) catch {},
                else => {},
            }
            std.mem.copyForwards(u8, v.in.items, v.in.items[f.len..]);
            v.in.items.len -= f.len;
        }
    }
};

test "computes the WebSocket accept key" {
    var buf: [28]u8 = undefined;
    // The example from RFC 6455.
    try std.testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", acceptKey("dGhlIHNhbXBsZSBub25jZQ==", &buf));
}

test "frames and parses messages" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try appendFrame(gpa, &out, 0x1, "hi");
    try std.testing.expectEqualSlices(u8, &.{ 0x81, 2, 'h', 'i' }, out.items);
    out.clearRetainingCapacity();
    const long: [300]u8 = @splat('a');
    try appendFrame(gpa, &out, 0x1, &long);
    try std.testing.expectEqualSlices(u8, &.{ 0x81, 126, 1, 44 }, out.items[0..4]);

    // A masked "Hello" from RFC 6455, then a partial second frame.
    var in = [_]u8{ 0x81, 0x85, 0x37, 0xfa, 0x21, 0x3d, 0x7f, 0x9f, 0x4d, 0x51, 0x58, 0x88, 0x80 };
    const f = (try parseFrame(&in)).?;
    try std.testing.expectEqual(@as(u8, 1), f.opcode);
    try std.testing.expectEqualStrings("Hello", f.payload);
    try std.testing.expectEqual(@as(?InFrame, null), try parseFrame(in[f.len..]));
    var unmasked = [_]u8{ 0x81, 0x01, 'x' };
    try std.testing.expectError(error.BadFrame, parseFrame(&unmasked));
}

test "the tap counts a step only while someone watches" {
    const gpa = std.testing.allocator;
    const Fake = struct {
        calls: usize = 0,
        fn run(ctx: *anyopaque, _: usize, _: []const f32, _: usize, _: []const u32, _: []const f32, _: []f32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
        }
        fn clock() u64
        {
            return 1000;
        }
    };
    var fake: Fake = .{};
    var feed = try Feed.init(gpa, 2, 4, 2);
    defer feed.deinit(gpa);
    var tap = try Tap.init(gpa, .{ .ctx = &fake, .run = Fake.run }, 2, &feed, .{}, Fake.clock);
    defer tap.deinit(gpa);
    const m = tap.moe();
    try std.testing.expect(m.prefetch == null);

    tap.beginStep();
    try m.run(m.ctx, 0, &.{}, 1, &.{ 1, 3 }, &.{}, &.{});
    tap.endStep(1);
    try std.testing.expect(feed.peek() == null);

    feed.viewers.store(1, .monotonic);
    for (0..3) |_|
    {
        tap.beginStep();
        try m.run(m.ctx, 0, &.{}, 2, &.{ 1, 3, 1, 2 }, &.{}, &.{});
        try m.run(m.ctx, 1, &.{}, 2, &.{ 0, 1, 2, 3 }, &.{}, &.{});
        tap.endStep(2);
    }
    // The third step found the ring full and was dropped.
    try std.testing.expectEqual(@as(usize, 7), fake.calls);
    const s = feed.peek().?;
    try std.testing.expectEqual(@as(u64, 0), s.seq);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 1, 1, 1, 1, 1, 1 }, s.counts);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try writeStep(gpa, &json, s, 4, 2);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"x\":[0,1,2,0,2,1,0,3,1,1,0,1,1,1,1,1,2,1,1,3,1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"engine\":[null,null]") != null);
    feed.pop();
    try std.testing.expectEqual(@as(u64, 1), feed.peek().?.seq);
    feed.pop();
    try std.testing.expect(feed.peek() == null);
}

test "the tap records cache statuses and hints only while someone watches" {
    const gpa = std.testing.allocator;
    const Fake = struct {
        hints: usize = 0,
        trace: ?*CacheTrace = null,
        fn run(ctx: *anyopaque, layer: usize, _: []const f32, _: usize, idx: []const u32, _: []const f32, _: []f32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            // Stands in for the cache: layer 0's experts are resident, layer 1's first was hinted, the second missed.
            for (idx, 0..) |e, i|
            {
                self.trace.?.note(layer * 4 + e, if (layer == 0) .resident else if (i == 0) .hinted else .missed);
            }
        }
        fn prefetch(ctx: *anyopaque, _: usize, experts: []const u32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.hints += experts.len;
        }
        fn clock() u64
        {
            return 1000;
        }
    };
    var fake: Fake = .{};
    var feed = try Feed.init(gpa, 2, 4, 4);
    defer feed.deinit(gpa);
    var tap = try Tap.init(gpa, .{ .ctx = &fake, .run = Fake.run, .prefetch = Fake.prefetch }, 2, &feed, .{}, Fake.clock);
    defer tap.deinit(gpa);
    fake.trace = &tap.trace;
    const m = tap.moe();

    const step = struct {
        fn f(mo: olmoe.Moe, t: *Tap) !void
        {
            t.beginStep();
            // Embedding guesses for layer 0, then layer 0's guess for layer 1, as the forward pass sends them.
            try mo.prefetch.?(mo.ctx, 0, &.{ 1, 3 });
            try mo.prefetch.?(mo.ctx, 1, &.{2});
            try mo.run(mo.ctx, 0, &.{}, 1, &.{ 1, 3 }, &.{}, &.{});
            try mo.prefetch.?(mo.ctx, 1, &.{0});
            try mo.run(mo.ctx, 1, &.{}, 1, &.{ 2, 0 }, &.{}, &.{});
            t.endStep(1);
        }
    }.f;
    // No viewer: hints still reach the engine, but nothing is noted or published.
    try step(m, &tap);
    try std.testing.expectEqual(@as(usize, 4), fake.hints);
    try std.testing.expect(feed.peek() == null);
    for (tap.trace.status) |s|
    {
        try std.testing.expectEqual(@as(u8, 0), s);
    }

    feed.viewers.store(1, .monotonic);
    try step(m, &tap);
    const s = feed.peek().?;
    try std.testing.expectEqual(@as(usize, 4), s.hint_count);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try writeStep(gpa, &json, s, 4, 1);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"c\":[0,1,1,0,3,1,1,0,4,1,2,2]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"h\":[0,1,0,0,3,0,1,2,0,1,0,1]}") != null);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "h_dropped") == null);
    const step_json = try std.json.parseFromSlice(std.json.Value, gpa, json.items, .{});
    step_json.deinit();
    json.clearRetainingCapacity();
    try writeHello(gpa, &json, .{ .model = "m", .transport = "local", .ranks = &.{"a"}, .owner = &.{ 0, 0, 0, 0 }, .cache_ranks = &.{0} }, 2, 4, 2, 0, null);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"cache_ranks\":[0]") != null);
    const hello_json = try std.json.parseFromSlice(std.json.Value, gpa, json.items, .{});
    hello_json.deinit();
}

test "a recording places each note after the steps published before it" {
    // The broadcaster is Linux only.
    if (comptime @import("builtin").os.tag == .windows)
    {
        return error.SkipZigTest;
    }
    const gpa = std.testing.allocator;
    const Fake = struct {
        fn run(_: *anyopaque, _: usize, _: []const f32, _: usize, _: []const u32, _: []const f32, _: []f32) anyerror!void
        {}
        fn clock() u64
        {
            return 2_000_000;
        }
    };
    var feed = try Feed.init(gpa, 1, 2, 8);
    defer feed.deinit(gpa);
    var tap = try Tap.init(gpa, .{ .ctx = &feed, .run = Fake.run }, 1, &feed, .{}, Fake.clock);
    defer tap.deinit(gpa);
    var bc: Broadcaster = .{
        .gpa = gpa,
        .feed = &feed,
        .topo = .{ .model = "m", .transport = "local", .ranks = &.{"a"}, .owner = &.{ 0, 0 } },
        .top_k = 1,
        .clock = Fake.clock,
    };
    defer
    {
        bc.rec.deinit(gpa);
        bc.json.deinit(gpa);
        bc.frame.deinit(gpa);
        bc.notes.deinit(gpa);
        for (bc.held.items) |n|
        {
            gpa.free(n.bytes);
        }
        bc.held.deinit(gpa);
    }
    bc.recording = true;
    feed.viewers.store(1, .monotonic);
    const m = tap.moe();

    bc.note("{\"t\":\"chat\"}");
    for (0..2) |i|
    {
        m.mark.?(m.ctx, true, 1);
        try m.run(m.ctx, 0, &.{}, 1, &.{@intCast(i)}, &.{}, &.{});
        m.mark.?(m.ctx, false, 1);
        bc.note("{\"t\":\"text\"}");
    }
    bc.drain();
    var lines = std.mem.splitScalar(u8, bc.rec.items, '\n');
    try std.testing.expectEqualStrings("{\"after\":null,\"t\":\"chat\"}", lines.next().?);
    try std.testing.expect(std.mem.startsWith(u8, lines.next().?, "{\"t\":\"step\",\"seq\":0,"));
    try std.testing.expectEqualStrings("{\"after\":0,\"t\":\"text\"}", lines.next().?);
    try std.testing.expect(std.mem.startsWith(u8, lines.next().?, "{\"t\":\"step\",\"seq\":1,"));
    try std.testing.expectEqualStrings("{\"after\":1,\"t\":\"text\"}", lines.next().?);
    try std.testing.expectEqualStrings("", lines.next().?);

    // A note made after a step the drain has not seen yet waits for that step.
    bc.rec.clearRetainingCapacity();
    try bc.held.append(gpa, .{ .head = 3, .bytes = try gpa.dupe(u8, "{\"t\":\"late\"}") });
    bc.drain();
    try std.testing.expectEqual(@as(usize, 1), bc.held.items.len);
    try std.testing.expectEqual(@as(usize, 0), bc.rec.items.len);
}
