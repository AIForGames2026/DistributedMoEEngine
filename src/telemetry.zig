//! Live telemetry for `--serve` (TASK-16b): which experts each engine step
//! used, on which rank, streamed to the built-in page over a WebSocket.
//!
//! `Tap` wraps the engine's `Moe`. Only while a viewer is connected does it
//! count each layer's expert choices and time the layer; at the end of every
//! step it copies the counts into a slot of `Feed`, a single-producer ring
//! the engine thread never waits on (a full ring drops the step, and the
//! viewer sees a gap in `seq`). `Broadcaster` owns the viewer sockets on its
//! own thread, turns slots into JSON frames, and drops frames for a viewer
//! that falls behind instead of slowing anyone down.
//!
//! Frames (text, one JSON object each):
//!   {"t":"hello", model, layers, experts, top_k, transport, ranks, owner, chunks, ms}
//!   {"t":"step", seq, ms, n, step_us, layer_us, x: [layer, expert, count, ...],
//!    engine: [rank 0 calls, rank 1 calls] (null where the engine keeps no count),
//!    rtt_us, rtt_layers, cache: {resident, hinted, late, missed, loads} or null}
const std = @import("std");
const olmoe = @import("olmoe.zig");

/// What the engine itself counted, copied into each step's frame.
pub const Counters = struct {
    /// Expert calls by rank, counted where they ran or were sent; null where no count is kept.
    calls: [2]?u64 = .{ null, null },
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
};

const Slot = struct {
    seq: u64 = 0,
    t_ns: u64 = 0,
    tokens: u32 = 0,
    step_ns: u64 = 0,
    counts: []u32,
    layer_ns: []u64,
    counters: Counters = .{},
};

/// The ring between the engine thread (producer) and the broadcaster (consumer).
pub const Feed = struct {
    slots: []Slot,
    layers: usize,
    experts: usize,
    head: std.atomic.Value(u64) align(64) = .init(0),
    tail: std.atomic.Value(u64) align(64) = .init(0),
    viewers: std.atomic.Value(u32) align(64) = .init(0),

    pub fn init(gpa: std.mem.Allocator, layers: usize, experts: usize, slots: usize) !Feed
    {
        const all = try gpa.alloc(Slot, slots);
        errdefer gpa.free(all);
        const counts = try gpa.alloc(u32, slots * layers * experts);
        errdefer gpa.free(counts);
        const times = try gpa.alloc(u64, slots * layers);
        for (all, 0..) |*s, i|
        {
            s.* = .{ .counts = counts[i * layers * experts ..][0 .. layers * experts], .layer_ns = times[i * layers ..][0..layers] };
        }
        return .{ .slots = all, .layers = layers, .experts = experts };
    }

    pub fn deinit(self: *Feed, gpa: std.mem.Allocator) void
    {
        gpa.free(self.slots[0].counts.ptr[0 .. self.slots.len * self.layers * self.experts]);
        gpa.free(self.slots[0].layer_ns.ptr[0 .. self.slots.len * self.layers]);
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
    /// Counting this step: a viewer was connected when it began.
    active: bool = false,
    step_t0: u64 = 0,
    next_seq: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, inner: olmoe.Moe, top_k: usize, feed: *Feed, probe: Probe, clock: *const fn () u64) !Tap
    {
        const counts = try gpa.alloc(u32, feed.layers * feed.experts);
        errdefer gpa.free(counts);
        const layer_ns = try gpa.alloc(u64, feed.layers);
        return .{ .inner = inner, .top_k = top_k, .feed = feed, .probe = probe, .clock = clock, .counts = counts, .layer_ns = layer_ns };
    }

    pub fn deinit(self: *Tap, gpa: std.mem.Allocator) void
    {
        gpa.free(self.counts);
        gpa.free(self.layer_ns);
    }

    pub fn moe(self: *Tap) olmoe.Moe
    {
        return .{ .ctx = self, .run = run, .ahead = self.inner.ahead, .prefetch = if (self.inner.prefetch != null) prefetch else null };
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
        const row = self.counts[layer * self.feed.experts ..][0..self.feed.experts];
        for (idx[0 .. n * self.top_k]) |e|
        {
            row[e] += 1;
        }
    }

    fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
    {
        const self: *Tap = @ptrCast(@alignCast(ctx));
        return self.inner.prefetch.?(self.inner.ctx, layer, experts);
    }

    /// Call before each engine step.
    pub fn beginStep(self: *Tap) void
    {
        self.active = self.feed.viewers.load(.monotonic) > 0;
        if (!self.active)
        {
            return;
        }
        @memset(self.counts, 0);
        @memset(self.layer_ns, 0);
        self.step_t0 = self.clock();
    }

    /// Call after each engine step of `tokens` tokens; publishes it if it was counted.
    pub fn endStep(self: *Tap, tokens: usize) void
    {
        if (!self.active)
        {
            return;
        }
        self.active = false;
        const seq = self.next_seq;
        self.next_seq += 1;
        const f = self.feed;
        const h = f.head.load(.monotonic);
        if (h - f.tail.load(.acquire) >= f.slots.len)
        {
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

pub fn writeHello(gpa: std.mem.Allocator, out: *std.ArrayList(u8), topo: Topology, layers: usize, experts: usize, top_k: usize, now_ns: u64) !void
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
    try out.print(gpa, "],\"chunks\":{d},\"ms\":{d:.3}}}", .{ topo.chunks, ms(now_ns) });
}

pub fn writeStep(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: *const Slot, experts: usize) !void
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
    for (c.calls, 0..) |v, i|
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
        try out.print(gpa, "{{\"resident\":{d},\"hinted\":{d},\"late\":{d},\"missed\":{d},\"loads\":{d}}}}}", .{ k.resident, k.hinted, k.late, k.missed, k.loads });
    }
    else
    {
        try out.appendSlice(gpa, "null}");
    }
}

fn ms(ns: u64) f64
{
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn us(ns: u64) f64
{
    return @as(f64, @floatFromInt(ns)) / 1e3;
}

/// Sends the feed to every connected viewer from its own thread. Linux.
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
        if (std.c.pipe(&self.pipe) != 0)
        {
            return error.SystemResources;
        }
        const t = try std.Thread.spawn(.{}, loop, .{self});
        t.detach();
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
        writeHello(self.gpa, &self.json, self.topo, self.feed.layers, self.feed.experts, self.top_k, self.clock()) catch return;
        appendFrame(self.gpa, &v.out, 0x1, self.json.items) catch return;
        self.viewers.append(self.gpa, v) catch
        {
            _ = std.c.close(fd);
            return;
        };
        _ = self.feed.viewers.fetchAdd(1, .monotonic);
    }

    /// Turns every waiting slot into a frame for each viewer with room for it.
    fn drain(self: *Broadcaster) void
    {
        while (self.feed.peek()) |s|
        {
            self.json.clearRetainingCapacity();
            self.frame.clearRetainingCapacity();
            const ok = blk: {
                writeStep(self.gpa, &self.json, s, self.feed.experts) catch break :blk false;
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
    try writeStep(gpa, &json, s, 4);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"x\":[0,1,2,0,2,1,0,3,1,1,0,1,1,1,1,1,2,1,1,3,1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json.items, "\"engine\":[null,null]") != null);
    feed.pop();
    try std.testing.expectEqual(@as(u64, 1), feed.peek().?.seq);
    feed.pop();
    try std.testing.expect(feed.peek() == null);
}
