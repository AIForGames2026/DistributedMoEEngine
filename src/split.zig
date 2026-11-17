//! The forward pass split across processes (TASK-10, TASK-33). Rank 0 holds every
//! non-expert weight and its own experts; ranks 1 to N-1 hold the rest, as the
//! placement says. For each layer, rank 0 sends every (token, expert) call to the
//! rank that owns the expert, runs its own calls meanwhile, and adds the results
//! back in a fixed order, so the output does not depend on which rank ran what or
//! in which order the results arrived.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const wire = @import("wire.zig");
const transport = @import("transport.zig");
const mf = @import("model_file.zig");
const budget = @import("budget.zig");
const telemetry = @import("telemetry.zig");
const tracker = @import("tracker.zig");
const ExpertCache = @import("expert_cache.zig").ExpertCache;
const Want = @import("expert_cache.zig").Want;

/// Processes in one split, rank 0 included (the transport's 6-bit rank).
pub const max_ranks = olmoe.Capacity.max_ranks;

/// Owners for `ranks` processes with no manifest: equal contiguous ranges, rank 0 first.
pub fn evenPlacement(owner: []u8, ranks: usize) void
{
    for (owner, 0..) |*o, e|
    {
        o.* = @intCast(e * ranks / owner.len);
    }
}

/// Reads a placement manifest: one `<rank> <first>-<last>` range per line, `#` comments.
/// Every expert needs an owner below `ranks`.
pub fn parsePlacement(text: []const u8, owner: []u8, ranks: usize) !void
{
    @memset(owner, 0xFF);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line|
    {
        const l = std.mem.trim(u8, line, " \r\t");
        if (l.len == 0 or l[0] == '#')
        {
            continue;
        }
        var f = std.mem.tokenizeAny(u8, l, " \t");
        const rank = try std.fmt.parseInt(u8, f.next() orelse return error.BadPlacement, 10);
        var r = std.mem.splitScalar(u8, f.next() orelse return error.BadPlacement, '-');
        const first = try std.fmt.parseInt(usize, r.next().?, 10);
        const last = try std.fmt.parseInt(usize, r.next() orelse return error.BadPlacement, 10);
        if (rank >= ranks or last < first or last >= owner.len)
        {
            return error.BadPlacement;
        }
        @memset(owner[first .. last + 1], rank);
    }
    for (owner) |o|
    {
        if (o == 0xFF)
        {
            return error.BadPlacement;
        }
    }
}

/// The highest rank a manifest names, plus one.
pub fn placementRanks(text: []const u8) !usize
{
    var most: usize = 0;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line|
    {
        const l = std.mem.trim(u8, line, " \r\t");
        if (l.len == 0 or l[0] == '#')
        {
            continue;
        }
        var f = std.mem.tokenizeAny(u8, l, " \t");
        most = @max(most, @as(usize, try std.fmt.parseInt(u8, f.next() orelse return error.BadPlacement, 10)) + 1);
    }
    return most;
}

pub fn cacheCounts(cache: *const ExpertCache) telemetry.CacheCounts
{
    const st = cache.stats;
    return .{ .resident = st.resident, .hinted = st.hinted, .late = st.late, .missed = st.missed, .loads = st.loads };
}

/// This process's experts: all loaded at startup, or held in an expert cache.
pub const Experts = struct {
    local: *olmoe.LocalExperts,
    cache: ?*ExpertCache = null,
    /// Forward passes so far; hints and layers are placed by step and layer.
    step: u64 = 0,
    /// One expert output per (token, choice), n x top_k x d_model.
    contrib: []f32 = &.{},
    /// Calls whose expert is still loading, by (token, choice) index.
    pending: []u32 = &.{},
    pending_want: []Want = &.{},
    /// The most tokens one step may bring (`--max-step`).
    max_tokens: usize = default_max_tokens,

    pub const default_max_tokens = 64;

    pub fn alloc(self: *Experts, gpa: std.mem.Allocator) !void
    {
        const cfg = self.local.cfg;
        self.contrib = try gpa.alloc(f32, self.max_tokens * cfg.top_k * cfg.d_model);
        self.pending = try gpa.alloc(u32, self.max_tokens * cfg.top_k);
        self.pending_want = try gpa.alloc(Want, self.max_tokens * cfg.top_k);
    }

    pub fn free(self: *Experts, gpa: std.mem.Allocator) void
    {
        gpa.free(self.contrib);
        gpa.free(self.pending);
        gpa.free(self.pending_want);
    }

    fn pos(self: *Experts, layer: usize) u64
    {
        return self.step * self.local.cfg.layers + layer;
    }

    fn want(self: *Experts, layer: usize, expert: usize) !Want
    {
        if (self.cache) |cache|
        {
            return cache.want(layer, expert);
        }
        return .{ .layer = @intCast(layer), .expert = @intCast(expert) };
    }

    /// The expert if it is ready now, or null while it loads.
    fn poll(self: *Experts, w: *Want) !?ExpertCache.Held
    {
        if (self.cache) |cache|
        {
            return cache.poll(w);
        }
        return .{ .expert = self.local.get(w.layer, w.expert) orelse return error.ExpertNotLocal, .slot = 0 };
    }

    fn wait(self: *Experts, w: *Want) !ExpertCache.Held
    {
        if (self.cache) |cache|
        {
            return cache.wait(w);
        }
        return (try self.poll(w)).?;
    }

    fn put(self: *Experts, h: ExpertCache.Held) void
    {
        if (self.cache) |cache|
        {
            cache.release(h);
        }
    }

    /// Hints at an expert of this pass's `layer`.
    fn hint(self: *Experts, layer: usize, expert: usize) void
    {
        self.hintAt(self.pos(layer), layer, expert);
    }

    fn hintAt(self: *Experts, at: u64, layer: usize, expert: usize) void
    {
        if (self.cache) |cache|
        {
            cache.hint(at, layer, expert);
        }
    }

    /// Marks this pass's `layer` as started, so its queued hints are dropped.
    fn enter(self: *Experts, layer: usize) void
    {
        self.enterAt(self.pos(layer));
    }

    fn enterAt(self: *Experts, at: u64) void
    {
        if (self.cache) |cache|
        {
            cache.enter(at);
        }
    }

    /// Runs the calls of `idx` that rank 0 owns (all without `owner`) into `contrib`,
    /// ready experts first and the rest as they load. Returns how many ran.
    fn runCalls(self: *Experts, layer: usize, x: []const f32, n: usize, idx: []const u32, contrib: []f32, owner: ?[]const u8) !usize
    {
        const d = self.local.cfg.d_model;
        const k = self.local.cfg.top_k;
        var ran: usize = 0;
        var waiting: usize = 0;
        for (0..n * k) |s|
        {
            if (idx[s] == olmoe.Capacity.skip)
            {
                continue;
            }
            if (owner) |o|
            {
                if (o[idx[s]] != 0)
                {
                    continue;
                }
            }
            var w = try self.want(layer, idx[s]);
            if (try self.poll(&w)) |h|
            {
                h.expert.forward(x[(s / k) * d ..][0..d], contrib[s * d ..][0..d], self.local.scratch);
                self.put(h);
                ran += 1;
                continue;
            }
            self.pending[waiting] = @intCast(s);
            self.pending_want[waiting] = w;
            waiting += 1;
        }
        const cache = self.cache orelse return ran;
        cache.stats.deferred += waiting;
        var t0: u64 = 0;
        while (waiting > 0)
        {
            var progressed = false;
            var i: usize = 0;
            while (i < waiting)
            {
                const h = (try cache.poll(&self.pending_want[i])) orelse
                {
                    i += 1;
                    continue;
                };
                const s = self.pending[i];
                h.expert.forward(x[(s / k) * d ..][0..d], contrib[s * d ..][0..d], self.local.scratch);
                cache.release(h);
                ran += 1;
                waiting -= 1;
                self.pending[i] = self.pending[waiting];
                self.pending_want[i] = self.pending_want[waiting];
                progressed = true;
            }
            if (progressed and t0 != 0)
            {
                cache.stats.wait_ns += mf.nowNs() - t0;
                t0 = 0;
            }
            else if (!progressed)
            {
                if (t0 == 0)
                {
                    t0 = mf.nowNs();
                }
                std.atomic.spinLoopHint();
            }
        }
        return ran;
    }

    /// One process with every expert, through the cache.
    pub fn moe(self: *Experts) olmoe.Moe
    {
        return .{ .ctx = self, .run = run, .prefetch = prefetch };
    }

    fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void
    {
        const self: *Experts = @ptrCast(@alignCast(ctx));
        if (n > self.max_tokens)
        {
            return error.TooManyTokens;
        }
        self.enter(layer);
        _ = try self.runCalls(layer, x, n, idx, self.contrib, null);
        addInOrder(self.local.cfg, n, idx, w, self.contrib, out);
    }

    fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
    {
        const self: *Experts = @ptrCast(@alignCast(ctx));
        // Layer 0's hint comes first in every pass.
        if (layer == 0)
        {
            self.step += 1;
        }
        for (experts) |e|
        {
            self.hint(layer, e);
        }
    }
};

/// Adds each call's output, weighted, in (token, choice) order.
fn addInOrder(cfg: olmoe.Config, n: usize, idx: []const u32, w: []const f32, contrib: []const f32, out: []f32) void
{
    const d = cfg.d_model;
    const k = cfg.top_k;
    for (0..n) |t|
    {
        for (0..k) |j|
        {
            const s = t * k + j;
            if (idx[s] == olmoe.Capacity.skip)
            {
                continue;
            }
            for (out[t * d ..][0..d], contrib[s * d ..][0..d]) |*o, y|
            {
                o.* += w[s] * y;
            }
        }
    }
}

/// One expert process rank 0 routes to: rank `i + 1` for `peers[i]`.
pub const Peer = struct {
    ep: *transport.Endpoint,
    /// Expert calls sent to it, and prefetch hints.
    calls: u64 = 0,
    hints: u64 = 0,
    /// Packets sent and answered in the current layer.
    sent: usize = 0,
    got: usize = 0,
};

/// Rank 0's side of the split: a `Moe` that runs its own experts and sends
/// the rest to the peer owning each.
pub const SplitMoe = struct {
    peers: []Peer,
    local: *olmoe.LocalExperts,
    experts: *Experts,
    owner: []const u8,
    enc: wire.Encoding,
    /// One expert output per (token, choice), n x top_k x d_model.
    contrib: []f32,
    pkt: *wire.Packet,
    local_calls: u64 = 0,
    /// Layers with remote calls, and their time from the first dispatch to the last combine.
    rtt_layers: u64 = 0,
    rtt_ns: u64 = 0,
    /// The tracker link watching every peer (TASK-09), if there is one.
    health: ?*tracker.Client = null,
    /// The longest the router waited on its peers with nothing sent or received
    /// (TASK-17), and the sends that waited for a peer to grant ring slots.
    stall_max_ns: u64 = 0,
    credit_waits: u64 = 0,
    /// The longest such wait since the spike test last cleared it.
    stall_recent_ns: u64 = 0,

    pub fn moe(self: *SplitMoe) olmoe.Moe
    {
        return .{ .ctx = self, .run = run };
    }

    /// Expert calls sent to all peers.
    pub fn remote(self: *const SplitMoe) u64
    {
        var n: u64 = 0;
        for (self.peers) |p|
        {
            n += p.calls;
        }
        return n;
    }

    pub fn hints(self: *const SplitMoe) u64
    {
        var n: u64 = 0;
        for (self.peers) |p|
        {
            n += p.hints;
        }
        return n;
    }

    /// The live diagram's view of this split's own counters.
    pub fn sample(ctx: *anyopaque, out: *telemetry.Counters) void
    {
        const self: *SplitMoe = @ptrCast(@alignCast(ctx));
        out.calls[0] = self.local_calls;
        for (self.peers, 1..) |p, r|
        {
            out.calls[r] = p.calls;
        }
        out.rtt_ns = self.rtt_ns;
        out.rtt_layers = self.rtt_layers;
        if (self.experts.cache) |cache|
        {
            out.cache = cacheCounts(cache);
        }
    }

    /// Refuses to route once the tracker marks any peer unhealthy.
    fn checkPeers(self: *SplitMoe) !void
    {
        if (self.health) |h|
        {
            if (!h.peerHealthy())
            {
                return error.PeerUnhealthy;
            }
        }
    }

    /// `ep.send`, but gives up if a peer turns unhealthy while it waits.
    fn sendChecked(self: *SplitMoe, ep: *transport.Endpoint, kind: wire.Kind) !void
    {
        while (true)
        {
            ep.trySend(self.pkt, kind) catch |err| switch (err)
            {
                error.WouldBlock =>
                {
                    try ep.flushCredits();
                    try self.checkPeers();
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => |e| return e,
            };
            return;
        }
    }

    /// Sends each peer the guessed experts it owns for `layer` (TASK-11), even
    /// when there are none, so it can time every layer's hint.
    pub fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
    {
        const self: *SplitMoe = @ptrCast(@alignCast(ctx));
        // Layer 0's hint comes first in every pass; the step tells a peer's passes apart.
        if (layer == 0)
        {
            self.experts.step += 1;
        }
        for (experts) |e|
        {
            if (self.owner[e] == 0)
            {
                self.experts.hint(layer, e);
            }
        }
        try self.checkPeers();
        for (self.peers, 1..) |*p, r|
        {
            wire.beginPrefetch(self.pkt, self.experts.step, @intCast(layer));
            for (experts) |e|
            {
                if (self.owner[e] == r)
                {
                    try wire.addPrefetch(self.pkt, @intCast(e));
                }
            }
            try self.sendChecked(p.ep, .dispatch);
            p.hints += 1;
        }
    }

    /// Takes one combine from rank `rank`; it must answer a call that rank owns.
    fn take(self: *SplitMoe, p: *Peer, rank: usize, r: transport.Received, layer: usize, idx: []const u32, n: usize) !void
    {
        const k = self.local.cfg.top_k;
        const d = self.local.cfg.d_model;
        const hdr = r.packet.header;
        const slot = hdr.token_id;
        if (r.kind != .combine or slot >= n * k or hdr.layer != layer or hdr.expert != idx[@intCast(slot)] or self.owner[hdr.expert] != rank)
        {
            return error.BadCombine;
        }
        _ = try wire.decodeChunk(r.packet, self.contrib[@intCast(slot * d)..][0..d]);
        try p.ep.release();
        p.got += 1;
    }

    /// Takes every combine that has arrived from any peer. Returns whether there was one.
    fn drain(self: *SplitMoe, layer: usize, idx: []const u32, n: usize) !bool
    {
        var any = false;
        for (self.peers, 1..) |*p, rank|
        {
            while (try p.ep.tryRecv()) |r|
            {
                try self.take(p, rank, r, layer, idx, n);
                any = true;
            }
        }
        return any;
    }

    fn outstanding(self: *const SplitMoe) bool
    {
        for (self.peers) |p|
        {
            if (p.got < p.sent)
            {
                return true;
            }
        }
        return false;
    }

    fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void
    {
        const self: *SplitMoe = @ptrCast(@alignCast(ctx));
        self.runLayer(layer, x, n, idx, w, out) catch |err|
        {
            return self.explain(err);
        };
    }

    /// A lost link (a killed peer's TCP reset) can come before the tracker's
    /// verdict: with a tracker, waits up to 2 s for it, and reports the peer as unhealthy.
    fn explain(self: *SplitMoe, err: anyerror) anyerror
    {
        const h = self.health orelse return err;
        if (err != error.ConnectionClosed and err != error.Unexpected and err != error.ProtocolError)
        {
            return err;
        }
        const end = mf.nowNs() + 2 * std.time.ns_per_s;
        while (mf.nowNs() < end)
        {
            if (!h.peerHealthy())
            {
                return error.PeerUnhealthy;
            }
            std.Thread.yield() catch {};
        }
        return err;
    }

    fn runLayer(self: *SplitMoe, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) !void
    {
        const k = self.local.cfg.top_k;
        const d = self.local.cfg.d_model;
        if (n * k * d > self.contrib.len)
        {
            return error.TooManyTokens;
        }
        try self.checkPeers();
        self.experts.enter(layer);
        const t0 = mf.nowNs();
        const chunks = wire.chunkCount(self.enc, d);
        for (self.peers) |*p|
        {
            p.sent = 0;
            p.got = 0;
        }
        var any_sent = false;
        // When the current wait on the peers began, or 0 when not waiting.
        var waiting: u64 = 0;
        // Dispatch every remote (token, choice) to its owner; drain combines from
        // every peer whenever one has no free slot, so no side waits forever.
        for (0..n * k) |s|
        {
            const e = idx[s];
            if (e == olmoe.Capacity.skip or self.owner[e] == 0)
            {
                continue;
            }
            const p = &self.peers[self.owner[e] - 1];
            for (0..chunks) |ch|
            {
                self.pkt.header = .{ .token_id = s, .generation = 0, .src_rank = 0, .layer = @intCast(layer), .expert = @intCast(e), .flags = 0, .gate_weight = w[s], .payload_len = 0 };
                try wire.encodeChunk(self.pkt, self.enc, x[(s / k) * d ..][0..d], ch);
                var blocked = false;
                while (true)
                {
                    p.ep.trySend(self.pkt, .dispatch) catch |err| switch (err)
                    {
                        error.WouldBlock =>
                        {
                            if (!blocked)
                            {
                                blocked = true;
                                self.credit_waits += 1;
                            }
                            const any = try self.drain(layer, idx, n);
                            waiting = self.waited(waiting, any);
                            try p.ep.flushCredits();
                            try self.checkPeers();
                            continue;
                        },
                        else => |e2| return e2,
                    };
                    break;
                }
                waiting = self.waited(waiting, true);
                p.sent += 1;
                any_sent = true;
            }
            p.calls += 1;
        }
        // Local experts run while the peers work on the remote ones.
        self.local_calls += try self.experts.runCalls(layer, x, n, idx, self.contrib, self.owner);
        while (self.outstanding())
        {
            if (try self.drain(layer, idx, n))
            {
                waiting = self.waited(waiting, true);
                continue;
            }
            waiting = self.waited(waiting, false);
            try self.checkPeers();
            std.atomic.spinLoopHint();
        }
        for (self.peers) |*p|
        {
            try p.ep.flushCredits();
        }
        if (any_sent)
        {
            self.rtt_ns += mf.nowNs() - t0;
            self.rtt_layers += 1;
        }
        // Fixed order (token, then choice), whatever order combines arrived in.
        addInOrder(self.local.cfg, n, idx, w, self.contrib, out);
    }

    /// Tracks waits on the peers: starts one when nothing moved, and on progress
    /// records how long it lasted. Returns the new start (0 when not waiting).
    fn waited(self: *SplitMoe, since: u64, progressed: bool) u64
    {
        if (!progressed)
        {
            return if (since == 0) mf.nowNs() else since;
        }
        if (since != 0)
        {
            const gap = mf.nowNs() - since;
            self.stall_max_ns = @max(self.stall_max_ns, gap);
            self.stall_recent_ns = @max(self.stall_recent_ns, gap);
        }
        return 0;
    }
};

/// An expert process's record of rank 0's prefetch hints (TASK-11): how long
/// before each layer's first expert call its hint arrived. Rank 0 measures how
/// often the hints were right.
const HintLog = struct {
    experts: usize,
    /// Per layer: the pass of the latest hint, when its first hint arrived,
    /// and the pass whose first call has been seen.
    latest: []u64,
    first_ns: []u64,
    started: []u64,
    hints: u64 = 0,
    starts: u64 = 0,
    lead_ns: u64 = 0,
    lead_min_ns: u64 = std.math.maxInt(u64),

    const none = std.math.maxInt(u64);

    fn init(a: std.mem.Allocator, layers: usize, experts: usize) !HintLog
    {
        const self: HintLog = .{
            .experts = experts,
            .latest = try a.alloc(u64, layers),
            .first_ns = try a.alloc(u64, layers),
            .started = try a.alloc(u64, layers),
        };
        @memset(self.latest, none);
        @memset(self.started, none);
        return self;
    }

    fn hint(self: *HintLog, pkt: *const wire.Packet, now: u64) !void
    {
        const layer = pkt.header.layer;
        const step = pkt.header.token_id;
        if (layer >= self.latest.len)
        {
            return error.BadDispatch;
        }
        if (self.latest[layer] != step)
        {
            self.latest[layer] = step;
            self.first_ns[layer] = now;
        }
        for (0..try wire.prefetchCount(pkt)) |i|
        {
            if (wire.prefetchExpert(pkt, i) >= self.experts)
            {
                return error.BadDispatch;
            }
        }
        self.hints += 1;
    }

    /// Where rank 0's pass is when `layer` starts, as the cache counts it.
    fn position(self: *HintLog, layer: usize) ?u64
    {
        const step = self.latest[layer];
        if (step == none)
        {
            return null;
        }
        return step * self.latest.len + layer;
    }

    fn call(self: *HintLog, layer: usize, now: u64) void
    {
        const step = self.latest[layer];
        if (step == none)
        {
            return;
        }
        if (self.started[layer] != step)
        {
            self.started[layer] = step;
            self.starts += 1;
            const lead = now - self.first_ns[layer];
            self.lead_ns += lead;
            self.lead_min_ns = @min(self.lead_min_ns, lead);
        }
    }
};

/// A call whose expert was still loading when it arrived (TASK-14).
const Parked = struct {
    hdr: wire.Header,
    want: Want,
};

/// An expert process (rank 1 or above): answers rank 0's dispatches with its experts.
pub const Server = struct {
    ep: *transport.Endpoint,
    experts: *Experts,
    owned: []const bool,
    /// Called every 256 answers to check memory (TASK-17).
    check_memory: ?*const fn () void = null,
    /// Time spent inside expert FFNs (TASK-22), the slowest one, and the calls answered.
    ffn_ns: u64 = 0,
    ffn_max_ns: u64 = 0,
    answered: u64 = 0,

    /// Runs one call and sends its result back in as many chunks as it came in.
    fn answer(self: *Server, held: ExpertCache.Held, hdr: wire.Header, x: []const f32, y: []f32, out: *wire.Packet) !void
    {
        const t0 = mf.nowNs();
        held.expert.forward(x, y, self.experts.local.scratch);
        const took = mf.nowNs() - t0;
        self.ffn_ns += took;
        self.ffn_max_ns = @max(self.ffn_max_ns, took);
        self.experts.put(held);
        self.answered += 1;
        if (self.answered % 256 == 0)
        {
            if (self.check_memory) |check|
            {
                check();
            }
        }
        const enc = wire.encodingOf(hdr);
        for (0..wire.chunkCount(enc, self.experts.local.cfg.d_model)) |ch|
        {
            out.header = .{ .token_id = hdr.token_id, .generation = 0, .src_rank = self.ep.rank, .layer = hdr.layer, .expert = hdr.expert, .flags = 0, .gate_weight = hdr.gate_weight, .payload_len = 0 };
            try wire.encodeChunk(out, enc, y, ch);
            try self.ep.send(out, .combine);
        }
    }

    /// Answers dispatches until rank 0 says goodbye. A call whose expert is
    /// still loading waits while later calls run.
    pub fn serve(self: *Server) !void
    {
        const ep = self.ep;
        const experts = self.experts;
        const cfg = experts.local.cfg;
        const d = cfg.d_model;
        const big = olmoe.big_allocator;
        const park_cap = Experts.default_max_tokens * cfg.top_k;
        const parked = try big.alloc(Parked, park_cap);
        defer big.free(parked);
        const parked_x = try big.alloc(f32, park_cap * d);
        defer big.free(parked_x);
        var n_parked: usize = 0;
        var set_aside: u64 = 0;
        var hint_arena: std.heap.ArenaAllocator = .init(big);
        defer hint_arena.deinit();
        var hints = try HintLog.init(hint_arena.allocator(), cfg.layers, cfg.experts);
        const x = try big.alloc(f32, 2 * d);
        defer big.free(x);
        const y = x[d..];
        const out = try big.create(wire.Packet);
        defer big.destroy(out);
        out.* = std.mem.zeroes(wire.Packet);
        var served: u64 = 0;
        std.debug.print("expert: rank {d} serving, resident memory {d:.0} MB\n", .{ ep.rank, mb(budget.residentBytes()) });
        // A call's chunks arrive back to back (one sender, in-order transport).
        var have: usize = 0;
        var first: wire.Header = undefined;
        while (true)
        {
            var i: usize = 0;
            while (i < n_parked)
            {
                const held = (try experts.poll(&parked[i].want)) orelse
                {
                    i += 1;
                    continue;
                };
                try self.answer(held, parked[i].hdr, parked_x[i * d ..][0..d], y, out);
                served += 1;
                n_parked -= 1;
                parked[i] = parked[n_parked];
                @memcpy(parked_x[i * d ..][0..d], parked_x[n_parked * d ..][0..d]);
            }
            const r = (try ep.tryRecv()) orelse
            {
                if (ep.peer_done and n_parked == 0)
                {
                    break;
                }
                std.atomic.spinLoopHint();
                continue;
            };
            const hdr = r.packet.header;
            if (r.kind == .dispatch and wire.isPrefetch(hdr) and have == 0)
            {
                try hints.hint(r.packet, mf.nowNs());
                for (0..try wire.prefetchCount(r.packet)) |j|
                {
                    experts.hintAt(hdr.token_id * cfg.layers + hdr.layer, hdr.layer, wire.prefetchExpert(r.packet, j));
                }
                try ep.release();
                continue;
            }
            if (r.kind != .dispatch or hdr.layer >= cfg.layers or hdr.expert >= cfg.experts or !self.owned[hdr.expert])
            {
                return error.BadDispatch;
            }
            const enc = wire.encodingOf(hdr);
            const chunks = wire.chunkCount(enc, d);
            if (have == 0)
            {
                first = hdr;
                hints.call(hdr.layer, mf.nowNs());
                if (hints.position(hdr.layer)) |at|
                {
                    experts.enterAt(at);
                }
            }
            if (hdr.token_id != first.token_id or hdr.layer != first.layer or hdr.expert != first.expert or
                try wire.decodeChunk(r.packet, x[0..d]) != have)
            {
                return error.BadDispatch;
            }
            try ep.release();
            have += 1;
            if (have < chunks)
            {
                continue;
            }
            have = 0;
            var w = try experts.want(hdr.layer, hdr.expert);
            if (try experts.poll(&w)) |held|
            {
                try self.answer(held, hdr, x[0..d], y, out);
                served += 1;
                continue;
            }
            if (n_parked == park_cap)
            {
                // Full: finish the oldest first.
                const held = try experts.wait(&parked[0].want);
                try self.answer(held, parked[0].hdr, parked_x[0..d], y, out);
                served += 1;
                n_parked -= 1;
                parked[0] = parked[n_parked];
                @memcpy(parked_x[0..d], parked_x[n_parked * d ..][0..d]);
            }
            parked[n_parked] = .{ .hdr = hdr, .want = w };
            @memcpy(parked_x[n_parked * d ..][0..d], x[0..d]);
            n_parked += 1;
            set_aside += 1;
        }
        try ep.finish();
        if (experts.cache) |cache|
        {
            cache.stats.deferred += set_aside;
        }
        std.debug.print("expert: rank {d} ran {d} expert FFNs for rank 0, {d:.0} us each on average (max {d:.0} us), {d:.2} s in all; resident memory {d:.0} MB\n", .{
            ep.rank,
            served,
            @as(f64, @floatFromInt(self.ffn_ns)) / @as(f64, @floatFromInt(@max(served, 1))) / 1e3,
            @as(f64, @floatFromInt(self.ffn_max_ns)) / 1e3,
            @as(f64, @floatFromInt(self.ffn_ns)) / 1e9,
            mb(budget.residentBytes()),
        });
        if (hints.hints > 0)
        {
            std.debug.print("expert: lookahead: {d} prefetch hints; {d} layer starts here, each after its hint, by {d:.0} us on average (min {d:.0} us)\n", .{
                hints.hints,
                hints.starts,
                @as(f64, @floatFromInt(hints.lead_ns)) / @as(f64, @floatFromInt(@max(hints.starts, 1))) / 1e3,
                @as(f64, @floatFromInt(if (hints.starts > 0) hints.lead_min_ns else 0)) / 1e3,
            });
        }
    }
};

fn mb(bytes: u64) f64
{
    return @as(f64, @floatFromInt(bytes)) / 1e6;
}

test "placement: even split, manifest, and its rank count" {
    var owner: [8]u8 = undefined;
    evenPlacement(&owner, 2);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 1, 1, 1, 1 }, &owner);
    evenPlacement(&owner, 3);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 1, 1, 1, 2, 2 }, &owner);
    const text = "# three ranks\n0 0-1\n2 2-5\r\n1 6-7\n";
    try std.testing.expectEqual(@as(usize, 3), try placementRanks(text));
    try parsePlacement(text, &owner, 3);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 2, 2, 2, 2, 1, 1 }, &owner);
    try std.testing.expectError(error.BadPlacement, parsePlacement(text, &owner, 2));
    try std.testing.expectError(error.BadPlacement, parsePlacement("0 0-6\n", &owner, 2));
}

/// An expert rank of the 3-rank test, on its own thread.
const TestRank = struct {
    local: olmoe.LocalExperts,
    experts: Experts,
    side: *transport.shm.Side,
    ep: transport.Endpoint,
    result: anyerror!void = {},

    fn run(self: *TestRank, owned: []const bool) void
    {
        self.result = self.serve(owned);
    }

    fn serve(self: *TestRank, owned: []const bool) !void
    {
        var server: Server = .{ .ep = &self.ep, .experts = &self.experts, .owned = owned };
        try self.ep.grantCapacity(0);
        try server.serve();
    }
};

test "a split over 3 processes gives the same bits as one process (tiny OLMoE)" {
    const gguf = @import("gguf.zig");
    const loader = @import("loader.zig");
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
    var w = try olmoe.Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ranks = 3;
    var owner: [8]u8 = undefined;
    evenPlacement(owner[0..cfg.experts], ranks);
    var owned: [ranks][8]bool = undefined;
    for (&owned, 0..) |*o, r|
    {
        for (o[0..cfg.experts], owner[0..cfg.experts]) |*x, own|
        {
            x.* = own == r;
        }
    }
    var all: [8]bool = @splat(true);
    var whole_local = try olmoe.LocalExperts.init(a, cfg, m, all[0..cfg.experts], plan, file);
    defer whole_local.deinit();

    var pairs: [ranks - 1]transport.shm.Pair = undefined;
    for (&pairs) |*p|
    {
        p.* = try transport.shm.Pair.create(64);
    }
    defer for (&pairs) |*p|
    {
        p.deinit();
    };
    var rank_side: [ranks - 1]TestRank = undefined;
    var threads: [ranks - 1]std.Thread = undefined;
    for (&rank_side, &threads, 1..) |*t, *th, r|
    {
        t.* = .{ .local = try olmoe.LocalExperts.init(a, cfg, m, owned[r][0..cfg.experts], plan, file), .experts = undefined, .side = pairs[r - 1].side(1), .ep = undefined };
        t.experts = .{ .local = &t.local };
        try t.experts.alloc(a);
        t.ep = transport.Endpoint.init(t.side.backend(), @intCast(r), 64);
        th.* = try std.Thread.spawn(.{}, TestRank.run, .{ t, owned[r][0..cfg.experts] });
    }

    var local0 = try olmoe.LocalExperts.init(a, cfg, m, owned[0][0..cfg.experts], plan, file);
    var experts0: Experts = .{ .local = &local0 };
    try experts0.alloc(a);
    var eps: [ranks - 1]transport.Endpoint = undefined;
    var peers: [ranks - 1]Peer = undefined;
    for (&eps, &peers, &pairs) |*ep, *p, *pair|
    {
        ep.* = transport.Endpoint.init(pair.side(0).backend(), 0, 64);
        p.* = .{ .ep = ep };
    }
    const pkt = try a.create(wire.Packet);
    pkt.* = std.mem.zeroes(wire.Packet);
    var sm: SplitMoe = .{
        .peers = &peers,
        .local = &local0,
        .experts = &experts0,
        .owner = owner[0..cfg.experts],
        .enc = .f32,
        .contrib = try a.alloc(f32, Experts.default_max_tokens * cfg.top_k * cfg.d_model),
        .pkt = pkt,
    };
    for (&eps) |*ep|
    {
        _ = try ep.awaitCapacity();
    }
    // 6 tokens run calls one at a time in one process; 24 run them batched.
    for ([_]usize{ 6, 24 }) |n|
    {
        const tokens = try a.alloc(u32, n);
        for (tokens, 0..) |*t, i|
        {
            t.* = @intCast((i * 37 + 5) % cfg.vocab);
        }
        const want = try a.alloc(f32, n * cfg.vocab);
        const got = try a.alloc(f32, n * cfg.vocab);
        try olmoe.forward(gpa, &w, whole_local.moe(), tokens, want, null);
        try olmoe.forward(gpa, &w, sm.moe(), tokens, got, null);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    }
    for (&eps) |*ep|
    {
        try ep.finish();
    }
    for (&threads, &rank_side) |th, *t|
    {
        th.join();
        try t.result;
    }
    for (peers) |p|
    {
        try std.testing.expect(p.calls > 0);
    }
    try std.testing.expect(sm.local_calls > 0);
}
