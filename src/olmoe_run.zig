//! OLMoE-1B-7B (TASK-10) or Qwen3-MoE (TASK-12) forward pass split across
//! two processes. Linux.
//!
//!   olmoe local MODEL --golden G.tfref                 one process, every expert
//!   olmoe shm   MODEL --golden G.tfref [options]       two processes, shared memory
//!   olmoe rdma  MODEL --golden G.tfref [options]       two processes, RDMA (-Drdma=true)
//!   olmoe expert MODEL --port P [options]              TCP rank 1
//!   olmoe router MODEL --host IP --port P --golden G.tfref [options]
//!   olmoe rdma-expert / rdma-router                    the same over RDMA
//!
//! options: --payload fp8|bf16|f32 (default fp8), --placement FILE, --slots S,
//!          --dev D, --gid G (RDMA), --gpu (Q8_0 matmuls on CUDA; -Dgpu=true),
//!          --generate N (TASK-24: N new tokens per prompt, one at a time),
//!          --temperature T, --top-k K, --seed S (seeded sampling; default greedy),
//!          --minutes M (repeat generation for M minutes, checking memory),
//!          --lookahead H (TASK-11: guess experts H layers ahead, 1 or 2, and
//!          send rank 1 a prefetch hint for its share), --extra E (guesses
//!          beyond top_k, default 2), --cache N (hold this process's
//!          experts in N slots loaded from the file on hints and on demand,
//!          instead of all at startup), --pin L (keep layers below L loaded;
//!          default 1), --loaders N (TASK-14: cache loader threads, each
//!          with its own reader; default 2), --mispredict P (replace a share
//!          P of the guesses with wrong experts, to test miss handling),
//!          --tracker IP[:PORT] (TASK-09: join the swarm tracker; an expert
//!          announces its address and experts there, and a router given no
//!          --host connects to a healthy expert it lists and refuses to route
//!          to it once the tracker marks it unhealthy), --advertise IP (the
//!          address an expert announces; default the one the tracker sees),
//!          --yarn F (TASK-12: YaRN rope scaling by F over 32,768 positions,
//!          for Qwen3-MoE past its native context; Qwen uses 4 for 128k)
//!
//! Rank 0 holds every non-expert weight (embedding, attention, norms, router,
//! LM head) and its own experts; rank 1 holds the rest of the experts. The
//! placement manifest says which rank owns which experts, one line per range:
//! `<rank> <first>-<last>`; without one, rank 0 owns the lower half. For each
//! layer, rank 0 sends every (token, expert) pair owned by rank 1 as a dispatch
//! packet carrying the token's normalized hidden state (as several packets
//! when it does not fit one, as with f32), runs its own experts
//! meanwhile, and adds the combines back in a fixed order, so the result does
//! not depend on arrival order.
//!
//! MODEL is the GGUF file with its `.tfsum` index (`tfload index`); G.tfref
//! comes from tools/ref/ref_olmoe.py. Rank 0 runs each golden prompt and
//! compares every position's logits and top-1 token, each layer's hidden
//! state, and the chosen experts with the reference. With --generate, it then
//! generates after each prompt using a KV cache and, when greedy, compares the
//! tokens and the logits each was picked from with transformers' generate().
const std = @import("std");
const c = std.c;
const build_options = @import("build_options");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const olmoe = @import("olmoe.zig");
const tfref = @import("tfref.zig");
const wire = @import("wire.zig");
const transport = @import("transport.zig");
const mf = @import("model_file.zig");
const Arena = @import("arena.zig").Arena;
const ExpertCache = @import("expert_cache.zig").ExpertCache;
const Want = @import("expert_cache.zig").Want;
const CacheSource = @import("expert_cache.zig").Source;
const tracker = @import("tracker.zig");
const kv_tier = @import("kv_tier.zig");
const gpu = if (build_options.gpu) @import("gpu.zig") else struct {
    fn open() !void {
        std.debug.print("this build has no GPU support; rebuild with -Dgpu=true\n", .{});
        return error.NoGpuSupport;
    }
    fn map(_: []u8) !void {}
    fn mapLive(_: []u8) !bool
    {
        return false;
    }
    fn stop() void {}
};

const max_experts = 1024;

/// `--yarn F` (TASK-12): YaRN rope scaling by F over Qwen3's 32,768 native positions.
var yarn_factor: f32 = 0;
const yarn_original = 32768;

const Options = struct {
    mode: []const u8,
    model: [:0]const u8,
    golden: ?[]const u8 = null,
    placement: ?[]const u8 = null,
    payload: wire.Encoding = .fp8_e4m3,
    host: []const u8 = "127.0.0.1",
    port: u16 = 7480,
    slots: ?u16 = null,
    device: ?[:0]const u8 = null,
    gid: i32 = -1,
    gpu: bool = false,
    generate: usize = 0,
    temperature: f32 = 0,
    top_k: usize = 0,
    seed: u64 = 0,
    minutes: f64 = 0,
    lookahead: usize = 0,
    extra: usize = 2,
    cache: usize = 0,
    pin: usize = 1,
    loaders: usize = 2,
    mispredict: f32 = 0,
    tracker: ?[]const u8 = null,
    advertise: ?[]const u8 = null,
    context: usize = 0,
    chunk: usize = 2048,
    tokens: ?[]const u8 = null,
    kv_offload: ?[]const u8 = null,
    kv_window: usize = 8192,
    save: ?[]const u8 = null,
    compare: ?[]const u8 = null,
    swarm: ?*const Swarm = null,
};

/// A router's view of the swarm (TASK-09): its tracker link, watching the
/// expert process it connected to, and the ownership that process announced.
const Swarm = struct {
    client: *tracker.Client,
    peer: tracker.Entry,
    owner: [max_experts]u8,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const opts = parseArgs(init) catch {
        std.debug.print(
            \\usage: olmoe local|shm|rdma MODEL --golden G.tfref [options]
            \\       olmoe expert|rdma-expert MODEL --port P [options]
            \\       olmoe router|rdma-router MODEL --host IP --port P --golden G.tfref [options]
            \\options: --payload fp8|bf16|f32  --placement FILE  --slots S  --dev D  --gid G  --gpu  --yarn F
            \\         --generate N  --temperature T  --top-k K  --seed S  --minutes M
            \\         --lookahead H  --extra E  --cache N  --pin L  --loaders N  --mispredict P
            \\         --tracker IP[:PORT]  --advertise IP  (a router with --tracker needs no --host)
            \\
        , .{});
        std.process.exit(2);
    };
    const mode = opts.mode;
    if (std.mem.eql(u8, mode, "local")) {
        var all: [max_experts]bool = @splat(true);
        if (opts.gpu) try gpu.open();
        defer if (opts.gpu) gpu.stop();
        var l = try Loaded.init(gpa, opts.model, &all, true, opts.cache == 0);
        defer l.deinit(gpa);
        if (opts.gpu) try gpu.map(l.weightRegion());
        var experts: Experts = .{ .local = &l.local };
        try experts.alloc(gpa);
        defer experts.free(gpa);
        const box = if (opts.cache > 0) try CacheBox.open(gpa, opts, &l, &all, "local") else null;
        defer if (box) |b| b.close(gpa);
        if (box) |b|
        {
            experts.cache = b.cache;
            phase_cache = b.cache;
        }
        var la: olmoe.Lookahead = .{ .horizon = opts.lookahead, .extra = opts.extra, .mispredict = opts.mispredict };
        var moe = if (box != null) experts.moe() else l.local.moe();
        if (opts.lookahead > 0)
        {
            moe.ahead = &la;
        }
        const result = if (opts.context > 0) runLong(gpa, &l.weights.?, moe, opts, "local") else runGolden(gpa, &l.weights.?, moe, opts, "local");
        if (box) |b|
        {
            reportCache("local", b.cache);
        }
        return result;
    }
    const is_rdma = std.mem.startsWith(u8, mode, "rdma");
    if (is_rdma and !build_options.rdma) {
        std.debug.print("this build has no RDMA support; rebuild with -Drdma=true\n", .{});
        std.process.exit(2);
    }
    const slots = opts.slots orelse if (is_rdma) @as(u16, 256) else 64;
    if (std.mem.eql(u8, mode, "shm") or std.mem.eql(u8, mode, "rdma")) return runForked(gpa, opts, slots, is_rdma);
    if (std.mem.eql(u8, mode, "expert") or std.mem.eql(u8, mode, "rdma-expert")) {
        const l = try transport.tcp.listen(opts.port);
        std.debug.print("expert: listening on port {d}\n", .{l.port});
        const client = if (opts.tracker) |t| try joinAsExpert(gpa, opts, t, l.port) else null;
        defer if (client) |cl| cl.close(gpa, false);
        const fd = try transport.tcp.acceptOn(l.fd);
        transport.tcp.closeSocket(l.fd);
        return runLinked(gpa, opts, slots, is_rdma, fd, 1);
    }
    if (std.mem.eql(u8, mode, "router") or std.mem.eql(u8, mode, "rdma-router")) {
        if (opts.tracker) |t|
        {
            return runTrackedRouter(gpa, opts, slots, is_rdma, t);
        }
        return runLinked(gpa, opts, slots, is_rdma, try transport.tcp.connect(opts.host, opts.port, 300), 0);
    }
    std.debug.print("unknown mode '{s}'\n", .{mode});
    std.process.exit(2);
}

fn parseArgs(init: std.process.Init) !Options {
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    var opts: Options = .{ .mode = it.next() orelse return error.Usage, .model = it.next() orelse return error.Usage };
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            opts.gpu = true;
            continue;
        }
        const value = it.next() orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--golden")) {
            opts.golden = value;
        } else if (std.mem.eql(u8, arg, "--placement")) {
            opts.placement = value;
        } else if (std.mem.eql(u8, arg, "--payload")) {
            opts.payload = if (std.mem.eql(u8, value, "fp8")) .fp8_e4m3 else if (std.mem.eql(u8, value, "bf16")) .bf16 else if (std.mem.eql(u8, value, "f32")) .f32 else return error.Usage;
        } else if (std.mem.eql(u8, arg, "--host")) {
            opts.host = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            opts.port = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--slots")) {
            opts.slots = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--dev")) {
            opts.device = value;
        } else if (std.mem.eql(u8, arg, "--gid")) {
            opts.gid = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, arg, "--generate")) {
            opts.generate = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--temperature")) {
            opts.temperature = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            opts.top_k = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--seed")) {
            opts.seed = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, arg, "--minutes")) {
            opts.minutes = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, arg, "--lookahead")) {
            opts.lookahead = try std.fmt.parseInt(usize, value, 10);
            if (opts.lookahead > olmoe.Lookahead.max_horizon) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--extra")) {
            opts.extra = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--cache")) {
            opts.cache = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--pin")) {
            opts.pin = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--loaders")) {
            opts.loaders = try std.fmt.parseInt(usize, value, 10);
            if (opts.loaders == 0 or opts.loaders > 16) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--context")) {
            opts.context = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--chunk")) {
            opts.chunk = try std.fmt.parseInt(usize, value, 10);
            if (opts.chunk == 0) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--tokens")) {
            opts.tokens = value;
        } else if (std.mem.eql(u8, arg, "--kv-offload")) {
            opts.kv_offload = value;
        } else if (std.mem.eql(u8, arg, "--kv-window")) {
            opts.kv_window = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, arg, "--save")) {
            opts.save = value;
        } else if (std.mem.eql(u8, arg, "--compare")) {
            opts.compare = value;
        } else if (std.mem.eql(u8, arg, "--yarn")) {
            yarn_factor = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, arg, "--tracker")) {
            opts.tracker = value;
        } else if (std.mem.eql(u8, arg, "--advertise")) {
            opts.advertise = value;
        } else if (std.mem.eql(u8, arg, "--mispredict")) {
            opts.mispredict = try std.fmt.parseFloat(f32, value);
            if (opts.mispredict < 0 or opts.mispredict > 1) return error.Usage;
        } else return error.Usage;
    }
    return opts;
}

/// Which rank owns each expert (`owner[e]`).
fn readPlacement(gpa: std.mem.Allocator, path: ?[]const u8, experts: usize, owner: []u8) !void {
    if (path == null) {
        for (owner[0..experts], 0..) |*o, e| o.* = if (e < experts / 2) 0 else 1;
        return;
    }
    @memset(owner[0..experts], 0xFF);
    const text = try mf.readWhole(gpa, path.?);
    defer gpa.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const l = std.mem.trim(u8, line, " \r\t");
        if (l.len == 0 or l[0] == '#') continue;
        var f = std.mem.tokenizeAny(u8, l, " \t");
        const rank = try std.fmt.parseInt(u8, f.next() orelse return error.BadPlacement, 10);
        var r = std.mem.splitScalar(u8, f.next() orelse return error.BadPlacement, '-');
        const first = try std.fmt.parseInt(usize, r.next().?, 10);
        const last = try std.fmt.parseInt(usize, r.next() orelse return error.BadPlacement, 10);
        if (rank > 1 or last < first or last >= experts) return error.BadPlacement;
        for (owner[first .. last + 1]) |*o| o.* = rank;
    }
    for (owner[0..experts]) |o| if (o == 0xFF) return error.BadPlacement; // every expert needs an owner
}

/// The weights one rank needs, read in one plan through io_uring and checked
/// against the file's `.tfsum` index: the owned experts, plus every
/// non-expert weight if `base` (rank 0).
const Loaded = struct {
    file: mf.File,
    state: mf.ModelState,
    cfg: olmoe.Config,
    plan: loader.Plan,
    arena: Arena,
    weights: ?olmoe.Weights,
    local: olmoe.LocalExperts,
    heap: std.heap.ArenaAllocator,

    /// With `resident` false, the owned experts are left to an expert cache.
    fn init(gpa: std.mem.Allocator, path: [:0]const u8, owned: []const bool, base: bool, resident: bool) !Loaded {
        const file = try mf.File.open(path);
        errdefer file.close();
        var state = try mf.openModel(gpa, file, null);
        errdefer state.deinit(gpa);
        const h = state.gguf_header orelse return error.NotGguf;
        var cfg = try olmoe.Config.fromGguf(h);
        if (yarn_factor > 0)
        {
            cfg = cfg.withYarn(yarn_factor, yarn_original);
        }
        if (cfg.experts > max_experts) return error.TooManyExperts;
        const m = state.model;

        var selected: std.ArrayList(loader.ExpertWeight) = .empty;
        defer selected.deinit(gpa);
        for (m.weights) |w| if (resident and owned[w.expert]) try selected.append(gpa, w);
        var ranges: std.ArrayList(loader.Range) = .empty;
        defer ranges.deinit(gpa);
        if (base) {
            const r = try olmoe.Weights.ranges(gpa, h);
            defer gpa.free(r);
            try ranges.appendSlice(gpa, r);
        }
        for (selected.items) |w| try ranges.append(gpa, w.parts[0]);
        var plan = try loader.Plan.fromRanges(gpa, ranges.items, mf.alignment, mf.max_gap, file.bytes);
        errdefer plan.deinit(gpa);

        var arena = try Arena.init(@intCast(plan.region_bytes));
        errdefer arena.deinit();
        const region = arena.buf[0..@intCast(plan.region_bytes)];
        @memset(region, 0);
        const t0 = mf.nowNs();
        _ = try mf.readPlan(path, file, plan, region);
        const t1 = mf.nowNs();

        const sum_path = try std.fmt.allocPrint(gpa, "{s}.tfsum", .{path});
        defer gpa.free(sum_path);
        const sum_text = try mf.readWhole(gpa, sum_path);
        defer gpa.free(sum_text);
        var idx = try loader.Index.parse(gpa, sum_text);
        defer idx.deinit(gpa);
        if (idx.file_bytes != file.bytes) return error.BadIndex;
        try idx.verify(plan, region, selected.items);

        var owned_n: usize = 0;
        for (owned[0..cfg.experts]) |o| owned_n += @intFromBool(o);
        std.debug.print("load: {s}{d} of {d} experts{s}, {d:.2} GB in {d:.2} s, expert CRC32C OK\n", .{
            if (base) "non-expert weights and " else "", owned_n, cfg.experts, if (resident) "" else " (through the expert cache)", @as(f64, @floatFromInt(plan.region_bytes)) / 1e9, @as(f64, @floatFromInt(t1 - t0)) / 1e9,
        });

        var heap: std.heap.ArenaAllocator = .init(gpa);
        errdefer heap.deinit();
        var weights: ?olmoe.Weights = if (base) try olmoe.Weights.init(gpa, h, cfg, plan, region) else null;
        errdefer if (weights) |*w| w.deinit();
        const none_owned: [max_experts]bool = @splat(false);
        const local = try olmoe.LocalExperts.init(heap.allocator(), cfg, m, if (resident) owned else &none_owned, plan, region);
        return .{ .file = file, .state = state, .cfg = cfg, .plan = plan, .arena = arena, .weights = weights, .local = local, .heap = heap };
    }

    /// Where every loaded weight lives.
    fn weightRegion(self: *Loaded) []u8 {
        return self.arena.buf[0..@intCast(self.plan.region_bytes)];
    }

    fn deinit(self: *Loaded, gpa: std.mem.Allocator) void {
        self.local.deinit();
        self.heap.deinit();
        if (self.weights) |*w| w.deinit();
        self.arena.deinit();
        self.plan.deinit(gpa);
        self.state.deinit(gpa);
        self.file.close();
    }
};

/// Experts this rank owns, as a mask, after reading the header for the count.
/// A router that found its expert through the tracker uses what it announced.
fn ownedMask(gpa: std.mem.Allocator, opts: Options, rank: u8, owner: *[max_experts]u8, owned: *[max_experts]bool) !usize {
    const file = try mf.File.open(opts.model);
    defer file.close();
    var state = try mf.openModel(gpa, file, null);
    defer state.deinit(gpa);
    const experts = state.model.experts;
    if (opts.swarm) |sw|
    {
        if (sw.peer.info.experts != experts)
        {
            std.debug.print("{s}: the expert process announced {d} experts, but this model has {d}\n", .{ opts.mode, sw.peer.info.experts, experts });
            return error.PlacementMismatch;
        }
        owner.* = sw.owner;
    }
    else
    {
        try readPlacement(gpa, opts.placement, experts, owner);
    }
    for (owned, 0..) |*o, e| o.* = e < experts and owner[e] == rank;
    return experts;
}

/// Announces this expert process to the tracker: where to reach it and which
/// experts it owns. The tracker link sends heartbeats until it is closed.
fn joinAsExpert(gpa: std.mem.Allocator, opts: Options, where: []const u8, port: u16) !*tracker.Client
{
    var owner: [max_experts]u8 = undefined;
    var owned: [max_experts]bool = undefined;
    const experts = try ownedMask(gpa, opts, 1, &owner, &owned);
    var info: tracker.Info = .{ .role = .expert, .rank = 1, .port = port, .experts = @intCast(experts) };
    info.setLocal();
    if (opts.advertise) |a|
    {
        info.addr = try transport.tcp.parseIp4(a);
    }
    for (0..experts) |e|
    {
        if (owned[e])
        {
            info.setOwned(e);
        }
    }
    const t = try tracker.parseAddress(where);
    const client = try tracker.Client.start(gpa, t.host, t.port, info);
    var ts: [16]u8 = undefined;
    std.debug.print("{s} expert: joined the tracker at {s}:{d} as id {d} with {d} of {d} experts (heartbeat {d} ms)\n", .{ tracker.stamp(&ts), t.host, t.port, client.id, info.ownedCount(), experts, client.beat_ms });
    return client;
}

/// A router with no --host: joins the tracker, connects to a healthy expert
/// process it lists, takes that process's experts as the placement, and
/// watches its health for the rest of the run.
fn runTrackedRouter(gpa: std.mem.Allocator, opts: Options, slots: u16, is_rdma: bool, where: []const u8) !void
{
    const t = try tracker.parseAddress(where);
    var info: tracker.Info = .{ .role = .router, .rank = 0 };
    info.setLocal();
    const client = try tracker.Client.start(gpa, t.host, t.port, info);
    defer client.close(gpa, false);
    var tj: [16]u8 = undefined;
    std.debug.print("{s} router: joined the tracker at {s}:{d} as id {d}; waiting for a healthy expert process\n", .{ tracker.stamp(&tj), t.host, t.port, client.id });
    const peer = try client.waitFor(.expert, 120_000);
    var swarm: Swarm = .{ .client = client, .peer = peer, .owner = @splat(0) };
    info.experts = peer.info.experts;
    for (0..peer.info.experts) |e|
    {
        if (peer.info.owns(e))
        {
            swarm.owner[e] = 1;
        }
        else
        {
            info.setOwned(e);
        }
    }
    try client.update(info);
    client.watch(peer.id);
    var ip: [16]u8 = undefined;
    const host = tracker.formatIp4(peer.info.addr, &ip);
    var ts: [16]u8 = undefined;
    std.debug.print("{s} router: tracker lists expert id {d} (pid {d} on {s}) at {s}:{d} with {d} of {d} experts\n", .{ tracker.stamp(&ts), peer.id, peer.info.pid, peer.info.nodeName(), host, peer.info.port, peer.info.ownedCount(), peer.info.experts });
    var tracked = opts;
    tracked.swarm = &swarm;
    return runLinked(gpa, tracked, slots, is_rdma, try transport.tcp.connect(host, peer.info.port, 300), 0);
}

/// This process's experts: all loaded at startup, or held in an expert cache.
const Experts = struct {
    local: *olmoe.LocalExperts,
    cache: ?*ExpertCache = null,
    /// Forward passes so far; hints and layers are placed by step and layer.
    step: u64 = 0,
    /// One expert output per (token, choice), n x top_k x d_model.
    contrib: []f32 = &.{},
    /// Calls whose expert is still loading, by (token, choice) index.
    pending: []u32 = &.{},
    pending_want: []Want = &.{},

    const max_tokens = 64;

    fn alloc(self: *Experts, gpa: std.mem.Allocator) !void
    {
        const cfg = self.local.cfg;
        self.contrib = try gpa.alloc(f32, max_tokens * cfg.top_k * cfg.d_model);
        self.pending = try gpa.alloc(u32, max_tokens * cfg.top_k);
        self.pending_want = try gpa.alloc(Want, max_tokens * cfg.top_k);
    }

    fn free(self: *Experts, gpa: std.mem.Allocator) void
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
    fn moe(self: *Experts) olmoe.Moe
    {
        return .{ .ctx = self, .run = run, .prefetch = prefetch };
    }

    fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void
    {
        const self: *Experts = @ptrCast(@alignCast(ctx));
        const d = self.local.cfg.d_model;
        const k = self.local.cfg.top_k;
        if (n > max_tokens)
        {
            return error.TooManyTokens;
        }
        self.enter(layer);
        _ = try self.runCalls(layer, x, n, idx, self.contrib, null);
        // Fixed order (token, then choice), whatever order the experts ran in.
        for (0..n) |t|
        {
            for (0..k) |j|
            {
                const s = t * k + j;
                for (out[t * d ..][0..d], self.contrib[s * d ..][0..d]) |*o, y|
                {
                    o.* += w[s] * y;
                }
            }
        }
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

/// The expert cache's memory, file reader, and checksums (TASK-11).
const CacheBox = struct {
    arena: Arena,
    /// One reader per loader thread.
    readers: []mf.DirectSource,
    sources: []CacheSource,
    index: loader.Index,
    heap: std.heap.ArenaAllocator,
    cache: *ExpertCache,

    fn open(gpa: std.mem.Allocator, opts: Options, l: *Loaded, owned: []const bool, name: []const u8) !*CacheBox
    {
        const self = try gpa.create(CacheBox);
        errdefer gpa.destroy(self);
        const m = l.state.model;
        const slot_bytes = try ExpertCache.slotBytes(m, mf.DirectSource.pad, mf.alignment);
        self.arena = try Arena.init(slot_bytes * opts.cache);
        errdefer self.arena.deinit();
        const region = self.arena.buf[0 .. slot_bytes * opts.cache];
        self.readers = try gpa.alloc(mf.DirectSource, opts.loaders);
        errdefer gpa.free(self.readers);
        self.sources = try gpa.alloc(CacheSource, opts.loaders);
        errdefer gpa.free(self.sources);
        var opened: usize = 0;
        errdefer for (self.readers[0..opened]) |*r|
        {
            r.close();
        };
        for (self.readers, self.sources) |*r, *src|
        {
            r.* = try mf.DirectSource.open(opts.model, l.file, region);
            opened += 1;
            src.* = .{ .ctx = r, .read = mf.DirectSource.read, .pad = mf.DirectSource.pad, .alignment = mf.alignment };
        }
        const sum_path = try std.fmt.allocPrint(gpa, "{s}.tfsum", .{opts.model});
        defer gpa.free(sum_path);
        const sum_text = try mf.readWhole(gpa, sum_path);
        defer gpa.free(sum_text);
        self.index = try loader.Index.parse(gpa, sum_text);
        errdefer self.index.deinit(gpa);
        self.heap = .init(gpa);
        errdefer self.heap.deinit();
        const t0 = mf.nowNs();
        self.cache = try ExpertCache.init(self.heap.allocator(), l.cfg, m, owned[0..l.cfg.experts], self.sources, &self.index, region, opts.pin);
        const pinned = self.cache.stats.loads;
        const on_gpu = opts.gpu and try gpu.mapLive(region);
        try self.cache.start();
        std.debug.print("{s}: expert cache: {d} slots ({d:.2} GB), {d} loader threads, {d} experts of layers below {d} loaded and kept in {d:.2} s; cached experts run on the {s}\n", .{
            name, opts.cache, @as(f64, @floatFromInt(region.len)) / 1e9, opts.loaders, pinned, opts.pin, @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e9, if (on_gpu) "GPU" else "CPU",
        });
        return self;
    }

    fn close(self: *CacheBox, gpa: std.mem.Allocator) void
    {
        self.cache.deinit();
        self.heap.deinit();
        self.index.deinit(gpa);
        for (self.readers) |*r|
        {
            r.close();
        }
        gpa.free(self.readers);
        gpa.free(self.sources);
        self.arena.deinit();
        gpa.destroy(self);
    }
};

/// Rank 0's expert cache, if any, so each phase can report and reset it.
var phase_cache: ?*ExpertCache = null;

fn reportCache(name: []const u8, cache: *ExpertCache) void
{
    const st = cache.stats;
    if (st.calls() == 0)
    {
        // Already reported phase by phase.
        return;
    }
    const calls = st.calls();
    const p = struct {
        fn f(a: u64, b: u64) f64
        {
            return @as(f64, @floatFromInt(a)) * 100 / @as(f64, @floatFromInt(b));
        }
    }.f;
    std.debug.print("{s}: expert cache: {d} expert calls: {d:.1}% already loaded, {d:.1}% loaded by a hint in time, {d:.1}% hinted but still loading, {d:.1}% missed; {d} calls set aside while others ran; {d:.1} ms with nothing to run but loads; {d} loads ({d} for hints, {d:.2} GB, {d:.2} ms each); {d} hinted experts evicted unused; {d} hints dropped as stale, {d} for a full queue\n", .{
        name,
        st.calls(),
        p(st.resident, calls),
        p(st.hinted, calls),
        p(st.late, calls),
        p(st.missed, calls),
        st.deferred,
        @as(f64, @floatFromInt(st.wait_ns)) / 1e6,
        st.loads,
        st.hint_loads,
        @as(f64, @floatFromInt(st.bytes)) / 1e9,
        @as(f64, @floatFromInt(st.load_ns)) / @as(f64, @floatFromInt(@max(st.loads, 1))) / 1e6,
        st.wasted,
        st.cancelled,
        st.dropped,
    });
}

/// Rank 0's `Moe`: local experts in this process, the rest over `ep`.
const SplitMoe = struct {
    ep: *transport.Endpoint,
    local: *olmoe.LocalExperts,
    experts: *Experts,
    owner: []const u8,
    enc: wire.Encoding,
    /// One expert output per (token, choice), n x top_k x d_model.
    contrib: []f32,
    pkt: *wire.Packet,
    remote: u64 = 0,
    local_calls: u64 = 0,
    hints: u64 = 0,
    /// The tracker link watching rank 1 (TASK-09), if there is one.
    health: ?*tracker.Client = null,

    fn moe(self: *SplitMoe) olmoe.Moe {
        return .{ .ctx = self, .run = run };
    }

    /// Refuses to route to rank 1 once the tracker marks it unhealthy.
    fn checkPeer(self: *SplitMoe) !void
    {
        if (self.health) |h|
        {
            if (!h.peerHealthy())
            {
                return error.PeerUnhealthy;
            }
        }
    }

    /// `ep.send`, but gives up if rank 1 turns unhealthy while it waits.
    fn sendChecked(self: *SplitMoe, kind: wire.Kind) !void
    {
        while (true)
        {
            self.ep.trySend(self.pkt, kind) catch |err| switch (err) {
                error.WouldBlock => {
                    try self.ep.flushCredits();
                    try self.checkPeer();
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => |e| return e,
            };
            return;
        }
    }

    /// Sends rank 1 the guessed experts it owns for `layer` (TASK-11), even
    /// when there are none, so it can time every layer's hint.
    fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
    {
        const self: *SplitMoe = @ptrCast(@alignCast(ctx));
        // Layer 0's hint comes first in every pass; the step tells rank 1's passes apart.
        if (layer == 0)
        {
            self.experts.step += 1;
        }
        wire.beginPrefetch(self.pkt, self.experts.step, @intCast(layer));
        for (experts) |e|
        {
            if (self.owner[e] != 0)
            {
                try wire.addPrefetch(self.pkt, @intCast(e));
            }
            else
            {
                self.experts.hint(layer, e);
            }
        }
        try self.checkPeer();
        try self.sendChecked(.dispatch);
        self.hints += 1;
    }

    fn take(self: *SplitMoe, r: transport.Received, layer: usize, idx: []const u32, n: usize) !void {
        const k = self.local.cfg.top_k;
        const d = self.local.cfg.d_model;
        const hdr = r.packet.header;
        const slot = hdr.token_id;
        if (r.kind != .combine or slot >= n * k or hdr.layer != layer or hdr.expert != idx[@intCast(slot)]) return error.BadCombine;
        _ = try wire.decodeChunk(r.packet, self.contrib[@intCast(slot * d)..][0..d]);
        try self.ep.release();
    }

    fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void {
        const self: *SplitMoe = @ptrCast(@alignCast(ctx));
        const k = self.local.cfg.top_k;
        const d = self.local.cfg.d_model;
        if (n * k * d > self.contrib.len) return error.TooManyTokens;
        try self.checkPeer();
        self.experts.enter(layer);
        const chunks = wire.chunkCount(self.enc, d);
        // Packets, not expert calls: each call is `chunks` packets each way.
        var sent: usize = 0;
        var got: usize = 0;
        // Dispatch every remote (token, choice); drain combines whenever the
        // peer has no free slot, so neither side waits on the other forever.
        for (0..n * k) |s| {
            const e = idx[s];
            if (self.owner[e] == 0) continue;
            for (0..chunks) |ch| {
                self.pkt.header = .{ .token_id = s, .generation = 0, .src_rank = 0, .layer = @intCast(layer), .expert = @intCast(e), .flags = 0, .gate_weight = w[s], .payload_len = 0 };
                try wire.encodeChunk(self.pkt, self.enc, x[(s / k) * d ..][0..d], ch);
                while (true) {
                    self.ep.trySend(self.pkt, .dispatch) catch |err| switch (err) {
                        error.WouldBlock => {
                            while (try self.ep.tryRecv()) |r| {
                                try self.take(r, layer, idx, n);
                                got += 1;
                            }
                            try self.ep.flushCredits();
                            try self.checkPeer();
                            continue;
                        },
                        else => |e2| return e2,
                    };
                    break;
                }
                sent += 1;
            }
            self.remote += 1;
        }
        // Local experts run while the peer works on the remote ones.
        self.local_calls += try self.experts.runCalls(layer, x, n, idx, self.contrib, self.owner);
        while (got < sent) {
            const r = (try self.ep.tryRecv()) orelse
            {
                try self.checkPeer();
                std.atomic.spinLoopHint();
                continue;
            };
            try self.take(r, layer, idx, n);
            got += 1;
        }
        try self.ep.flushCredits();
        // Fixed order (token, then choice), whatever order combines arrived in.
        for (0..n) |t| {
            for (0..k) |j| {
                const s = t * k + j;
                for (out[t * d ..][0..d], self.contrib[s * d ..][0..d]) |*o, y| o.* += w[s] * y;
            }
        }
    }
};

/// Rank 1's record of rank 0's prefetch hints (TASK-11): how long before each
/// layer's first expert call its hint arrived. Rank 0 measures how often the
/// hints were right.
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

/// Time rank 1 spent inside expert FFNs (TASK-22), and the slowest one.
var ffn_ns: u64 = 0;
var ffn_max_ns: u64 = 0;

/// Runs one call and sends its result back in as many chunks as it came in.
fn answer(ep: *transport.Endpoint, experts: *Experts, held: ExpertCache.Held, hdr: wire.Header, x: []const f32, y: []f32, out: *wire.Packet) !void
{
    const t0 = mf.nowNs();
    held.expert.forward(x, y, experts.local.scratch);
    const took = mf.nowNs() - t0;
    ffn_ns += took;
    ffn_max_ns = @max(ffn_max_ns, took);
    experts.put(held);
    const enc = wire.encodingOf(hdr);
    for (0..wire.chunkCount(enc, experts.local.cfg.d_model)) |ch|
    {
        out.header = .{ .token_id = hdr.token_id, .generation = 0, .src_rank = 1, .layer = hdr.layer, .expert = hdr.expert, .flags = 0, .gate_weight = hdr.gate_weight, .payload_len = 0 };
        try wire.encodeChunk(out, enc, y, ch);
        try ep.send(out, .combine);
    }
}

/// Rank 1: answers dispatches with its experts until rank 0 says goodbye.
/// A call whose expert is still loading waits while later calls run.
fn serveExperts(ep: *transport.Endpoint, experts: *Experts, owned: []const bool) !void {
    const local = experts.local;
    const cfg = local.cfg;
    const d = cfg.d_model;
    const park_cap = Experts.max_tokens * cfg.top_k;
    const parked = try std.heap.page_allocator.alloc(Parked, park_cap);
    defer std.heap.page_allocator.free(parked);
    const parked_x = try std.heap.page_allocator.alloc(f32, park_cap * d);
    defer std.heap.page_allocator.free(parked_x);
    var n_parked: usize = 0;
    var set_aside: u64 = 0;
    var hint_arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer hint_arena.deinit();
    var hints = try HintLog.init(hint_arena.allocator(), cfg.layers, cfg.experts);
    const x = try std.heap.page_allocator.alloc(f32, 2 * d);
    defer std.heap.page_allocator.free(x);
    const y = x[d..];
    const out = try std.heap.page_allocator.create(wire.Packet);
    defer std.heap.page_allocator.destroy(out);
    out.* = std.mem.zeroes(wire.Packet);
    var served: u64 = 0;
    std.debug.print("expert: serving, resident memory {d:.0} MB\n", .{mb(rssBytes())});
    // A call's chunks arrive back to back (one sender, in-order transport).
    var have: usize = 0;
    var first: wire.Header = undefined;
    while (true) {
        var i: usize = 0;
        while (i < n_parked)
        {
            const held = (try experts.poll(&parked[i].want)) orelse
            {
                i += 1;
                continue;
            };
            try answer(ep, experts, held, parked[i].hdr, parked_x[i * d ..][0..d], y, out);
            served += 1;
            n_parked -= 1;
            parked[i] = parked[n_parked];
            @memcpy(parked_x[i * d ..][0..d], parked_x[n_parked * d ..][0..d]);
        }
        const r = (try ep.tryRecv()) orelse {
            if (ep.peer_done and n_parked == 0) break;
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
        if (r.kind != .dispatch or hdr.layer >= cfg.layers or hdr.expert >= cfg.experts or !owned[hdr.expert]) return error.BadDispatch;
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
            try wire.decodeChunk(r.packet, x[0..d]) != have) return error.BadDispatch;
        try ep.release();
        have += 1;
        if (have < chunks) continue;
        have = 0;
        var w = try experts.want(hdr.layer, hdr.expert);
        if (try experts.poll(&w)) |held|
        {
            try answer(ep, experts, held, hdr, x[0..d], y, out);
            served += 1;
            continue;
        }
        if (n_parked == park_cap)
        {
            // Full: finish the oldest first.
            const held = try experts.wait(&parked[0].want);
            try answer(ep, experts, held, parked[0].hdr, parked_x[0..d], y, out);
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
    std.debug.print("expert: ran {d} expert FFNs for rank 0, {d:.0} us each on average (max {d:.0} us), {d:.2} s in all; resident memory {d:.0} MB\n", .{
        served,
        @as(f64, @floatFromInt(ffn_ns)) / @as(f64, @floatFromInt(@max(served, 1))) / 1e3,
        @as(f64, @floatFromInt(ffn_max_ns)) / 1e3,
        @as(f64, @floatFromInt(ffn_ns)) / 1e9,
        mb(rssBytes()),
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


/// Loads this rank's weights and runs its side over `ep`.
fn runRank(gpa: std.mem.Allocator, opts: Options, ep: *transport.Endpoint, rank: u8, name: []const u8) !void {
    var owner: [max_experts]u8 = undefined;
    var owned: [max_experts]bool = undefined;
    _ = try ownedMask(gpa, opts, rank, &owner, &owned);
    // After any fork (CUDA state does not survive one), before the weights.
    if (opts.gpu) try gpu.open();
    defer if (opts.gpu) gpu.stop();
    var l = try Loaded.init(gpa, opts.model, &owned, rank == 0, opts.cache == 0);
    defer l.deinit(gpa);
    // With an expert cache, rank 1 loads nothing at startup.
    if (opts.gpu and l.weightRegion().len > 0)
    {
        try gpu.map(l.weightRegion());
    }
    const weights_hash = if (opts.minutes > 0) std.hash.Wyhash.hash(0, l.weightRegion()) else 0;
    var experts: Experts = .{ .local = &l.local };
    try experts.alloc(gpa);
    defer experts.free(gpa);
    const label = if (rank == 1) "expert" else name;
    defer if (opts.minutes > 0) reportWeights(label, weights_hash, l.weightRegion());
    const box = if (opts.cache > 0) try CacheBox.open(gpa, opts, &l, &owned, label) else null;
    defer if (box) |b| b.close(gpa);
    if (box) |b|
    {
        experts.cache = b.cache;
    }
    if (rank == 1)
    {
        const served = serveExperts(ep, &experts, &owned);
        if (box) |b|
        {
            reportCache(label, b.cache);
        }
        return served;
    }

    const cfg = l.cfg;
    phase_cache = experts.cache;
    const pkt = try gpa.create(wire.Packet);
    defer gpa.destroy(pkt);
    pkt.* = std.mem.zeroes(wire.Packet);
    var split: SplitMoe = .{
        .ep = ep,
        .local = &l.local,
        .experts = &experts,
        .owner = owner[0..cfg.experts],
        .enc = opts.payload,
        .contrib = try gpa.alloc(f32, 64 * cfg.top_k * cfg.d_model),
        .pkt = pkt,
        .health = if (opts.swarm) |sw| sw.client else null,
    };
    defer gpa.free(split.contrib);
    var la: olmoe.Lookahead = .{ .horizon = opts.lookahead, .extra = opts.extra, .mispredict = opts.mispredict };
    var moe = split.moe();
    if (opts.lookahead > 0)
    {
        moe.ahead = &la;
        moe.prefetch = SplitMoe.prefetch;
    }
    const result = runGolden(gpa, &l.weights.?, moe, opts, name);
    if (result) |_| {} else |err|
    {
        if (err == error.PeerUnhealthy)
        {
            reportRefusal(name, opts.swarm.?, split.remote);
            return err;
        }
    }
    try ep.finish();
    std.debug.print("{s}: {d} expert calls sent to rank 1, {d} run locally; payload {t}\n", .{ name, split.remote, split.local_calls, opts.payload });
    if (opts.lookahead > 0)
    {
        std.debug.print("{s}: lookahead: {d} prefetch hints sent to rank 1\n", .{ name, split.hints });
    }
    if (box) |b|
    {
        reportCache(name, b.cache);
    }
    return result;
}

/// Says why rank 0 stopped routing: what the tracker reports for rank 1.
fn reportRefusal(name: []const u8, sw: *const Swarm, sent: u64) void
{
    var v: tracker.View = .{};
    sw.client.snapshot(&v);
    const why = if (sw.client.lost.load(.acquire)) "the tracker connection was lost" else if (v.find(sw.peer.id)) |e| @tagName(e.health) else "missing from the map";
    var t: [16]u8 = undefined;
    std.debug.print("{s} {s}: refusing to route to rank 1 (tracker id {d}, pid {d} on {s}): {s} in map v{d}, received {d:.0} ms ago; {d} expert calls had been sent to it\n", .{
        tracker.stamp(&t), name, sw.peer.id, sw.peer.info.pid, sw.peer.info.nodeName(), why, v.version, @as(f64, @floatFromInt(tracker.monoNs() - v.at_ns)) / 1e6, sent,
    });
}

/// Both ranks on this machine: rank 1 is a fork. Each loads only its own
/// weights after the fork.
fn runForked(gpa: std.mem.Allocator, opts: Options, slots: u16, is_rdma: bool) !void {
    const name: []const u8 = if (is_rdma) "rdma" else "shm";
    if (is_rdma) {
        if (!build_options.rdma) unreachable;
        const l = try transport.tcp.listen(0);
        const pid = c.fork();
        if (pid < 0) return error.ForkFailed;
        if (pid == 0) {
            transport.tcp.closeSocket(l.fd);
            const fd = transport.tcp.connect("127.0.0.1", l.port, 50) catch c._exit(1);
            runRdmaRank(gpa, opts, slots, fd, 1, name) catch |err| {
                std.debug.print("expert: {t}\n", .{err});
                c._exit(1);
            };
            c._exit(0);
        }
        const fd = try transport.tcp.acceptOn(l.fd);
        transport.tcp.closeSocket(l.fd);
        const result = runRdmaRank(gpa, opts, slots, fd, 0, name);
        return waitChild(pid, result);
    }
    var pair = try transport.shm.Pair.create(slots);
    defer pair.deinit();
    const pid = c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        var ep = transport.Endpoint.init(pair.side(1).backend(), 1, slots);
        runRank(gpa, opts, &ep, 1, name) catch |err| {
            std.debug.print("expert: {t}\n", .{err});
            c._exit(1);
        };
        c._exit(0);
    }
    var ep = transport.Endpoint.init(pair.side(0).backend(), 0, slots);
    return waitChild(pid, runRank(gpa, opts, &ep, 0, name));
}

fn waitChild(pid: c.pid_t, result: anyerror!void) !void {
    if (result) |_| {} else |_| _ = c.kill(pid, c.SIG.TERM);
    var status: c_int = 0;
    _ = c.waitpid(pid, &status, 0);
    try result;
    if (status != 0) return error.ExpertFailed;
}

/// One rank over an already-connected socket: TCP, or RDMA bootstrapped on it.
fn runLinked(gpa: std.mem.Allocator, opts: Options, slots: u16, is_rdma: bool, fd: transport.tcp.Socket, rank: u8) !void {
    if (is_rdma) {
        if (!build_options.rdma) unreachable;
        return runRdmaRank(gpa, opts, slots, fd, rank, "rdma");
    }
    var ring_arena = try Arena.init(@as(usize, slots) * @sizeOf(wire.Packet));
    defer ring_arena.deinit();
    const ring = try ring_arena.alloc(wire.Packet, slots);
    var link = transport.tcp.Link.init(fd, ring);
    var ep = transport.Endpoint.init(link.backend(), @intCast(rank), slots);
    const result = runRank(gpa, opts, &ep, rank, "tcp");
    if (result) |_| {} else |err|
    {
        // An unhealthy peer may never close its side, so do not wait for it.
        if (err == error.PeerUnhealthy)
        {
            transport.tcp.closeSocket(link.fd);
            return err;
        }
    }
    if (rank == 0)
    {
        link.close();
    }
    else
    {
        link.closeAfterPeer();
    }
    return result;
}

fn runRdmaRank(gpa: std.mem.Allocator, opts: Options, slots: u16, fd: transport.tcp.Socket, rank: u8, name: []const u8) !void {
    var ring_arena = try Arena.init(@as(usize, slots) * @sizeOf(wire.Packet));
    defer ring_arena.deinit();
    const ring = try ring_arena.alloc(wire.Packet, slots);
    var link = try transport.rdma.Link.connect(fd, @intCast(rank), ring, .{ .device = opts.device, .gid_index = opts.gid });
    var ep = transport.Endpoint.init(link.backend(), @intCast(rank), slots);
    const result = runRank(gpa, opts, &ep, rank, name);
    if (result) |_| {} else |err|
    {
        if (err == error.PeerUnhealthy)
        {
            link.abort();
            return err;
        }
    }
    link.close();
    return result;
}

/// Long context (TASK-12): `--context N` tokens from `--tokens FILE` (u32
/// ids, repeated if short) are prefilled `--chunk` at a time, then
/// `--generate M` tokens are decoded greedily. Reports time to first token,
/// decode speed, and the memory the KV cache holds; `--kv-offload DIR` keeps
/// the cold middle on NVMe. `--save F` writes the generated tokens and their
/// logits, and `--compare F` checks a run against them bit for bit.
fn runLong(gpa: std.mem.Allocator, w: *const olmoe.Weights, moe: olmoe.Moe, opts: Options, name: []const u8) !void
{
    const cfg = w.cfg;
    const n = opts.context;
    const tokens = try gpa.alloc(u32, n);
    defer gpa.free(tokens);
    const raw = try mf.readWhole(gpa, opts.tokens orelse return error.MissingTokens);
    defer gpa.free(raw);
    if (raw.len < 4)
    {
        return error.MissingTokens;
    }
    const count = raw.len / 4;
    for (tokens, 0..) |*t, i|
    {
        t.* = std.mem.readInt(u32, raw[(i % count) * 4 ..][0..4], .little);
        if (t.* >= cfg.vocab)
        {
            return error.BadToken;
        }
    }
    const cap = n + opts.generate;
    var tier: ?*kv_tier.TierCache = null;
    defer if (tier) |t| t.close(gpa);
    var cache: olmoe.KvCache = undefined;
    var path_buf: [512]u8 = undefined;
    if (opts.kv_offload) |dir|
    {
        const path = try zPrint(&path_buf, "{s}/tf_kv_{d}.bin", .{ dir, c.getpid() });
        tier = try kv_tier.TierCache.open(gpa, cfg, cap, .{ .max_chunk = opts.chunk, .window = opts.kv_window, .path = path });
        cache = olmoe.KvCache.tiered(cfg, cap, tier.?.tier());
    }
    else
    {
        cache = try olmoe.KvCache.init(gpa, cfg, cap);
    }
    defer cache.deinit(gpa);
    const kv_bytes = if (tier) |t| t.memoryBytes() else (cache.k.len + cache.v.len) * @sizeOf(f32);
    const full_bytes = cfg.layers * 2 * cap * cfg.kvDim() * @sizeOf(f32);
    std.debug.print("{s}: context {d} tokens + {d} generated, chunks of {d}; KV cache {s}: {d:.2} GB in memory (all of it would be {d:.2} GB), GPU attention {s}\n", .{
        name, n, opts.generate, opts.chunk, if (tier != null) "offloaded to NVMe" else "all in memory", @as(f64, @floatFromInt(kv_bytes)) / 1e9, @as(f64, @floatFromInt(full_bytes)) / 1e9, if (cache.dev_k != 0 or tier != null) "on" else "off",
    });

    const logits = try gpa.alloc(f32, cfg.vocab);
    defer gpa.free(logits);
    const t0 = mf.nowNs();
    var pos: usize = 0;
    var next_report: usize = n / 10;
    while (pos < n)
    {
        const e = @min(pos + opts.chunk, n);
        try olmoe.step(gpa, w, moe, &cache, tokens[pos..e], logits, null);
        pos = e;
        if (pos >= next_report and pos < n)
        {
            std.debug.print("{s}: prefill {d} of {d} positions, {d:.1} s\n", .{ name, pos, n, @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e9 });
            next_report += @max(n / 10, 1);
        }
    }
    const ttft = mf.nowNs() - t0;
    reportProfile(name, "prefill");
    olmoe.profile = .{};
    const out_tokens = try gpa.alloc(u32, opts.generate);
    defer gpa.free(out_tokens);
    const out_logits = try gpa.alloc(f32, opts.generate * cfg.vocab);
    defer gpa.free(out_logits);
    const read0 = if (tier) |t| t.stats else kv_tier.Stats{};
    const t1 = mf.nowNs();
    for (0..opts.generate) |i|
    {
        @memcpy(out_logits[i * cfg.vocab ..][0..cfg.vocab], logits);
        const tok: u32 = @intCast(olmoe.argmax(logits));
        out_tokens[i] = tok;
        if (i + 1 < opts.generate)
        {
            try olmoe.step(gpa, w, moe, &cache, &.{tok}, logits, null);
        }
    }
    const decode_ns = mf.nowNs() - t1;
    reportProfile(name, "decode");
    const steps = @max(@max(opts.generate, 1) - 1, 1);
    const per_step = @as(f64, @floatFromInt(decode_ns)) / 1e9 / @as(f64, @floatFromInt(steps));
    std.debug.print("{s}: time to first token {d:.1} s ({d:.0} tokens/s prefill); decode {d:.3} s per token ({d:.2} tokens/s) over {d} steps from position {d}; resident memory {d:.0} MB\n", .{
        name,
        @as(f64, @floatFromInt(ttft)) / 1e9,
        @as(f64, @floatFromInt(n)) / (@as(f64, @floatFromInt(ttft)) / 1e9),
        per_step,
        1 / per_step,
        steps,
        n,
        mb(rssBytes()),
    });
    if (tier) |t|
    {
        const st = t.stats;
        const rb = st.read_bytes - read0.read_bytes;
        const rn = st.read_ns - read0.read_ns;
        const wn = st.wait_ns - read0.wait_ns;
        std.debug.print("{s}: kv offload during decode: read {d:.2} GB from NVMe at {d:.2} GB/s ({d:.3} s per token), attention waited {d:.3} s per token; over the whole run, read {d:.1} GB and wrote {d:.1} GB ({d:.2} GB/s)\n", .{
            name,
            @as(f64, @floatFromInt(rb)) / 1e9,
            @as(f64, @floatFromInt(rb)) / @max(@as(f64, @floatFromInt(rn)), 1),
            @as(f64, @floatFromInt(rn)) / 1e9 / @as(f64, @floatFromInt(steps)),
            @as(f64, @floatFromInt(wn)) / 1e9 / @as(f64, @floatFromInt(steps)),
            @as(f64, @floatFromInt(st.read_bytes)) / 1e9,
            @as(f64, @floatFromInt(st.write_bytes)) / 1e9,
            @as(f64, @floatFromInt(st.write_bytes)) / @max(@as(f64, @floatFromInt(st.write_ns)), 1),
        });
    }
    const record = try gpa.alloc(u8, 16 + out_tokens.len * 4 + out_logits.len * 4);
    defer gpa.free(record);
    std.mem.writeInt(u64, record[0..8], n, .little);
    std.mem.writeInt(u64, record[8..16], opts.generate, .little);
    @memcpy(record[16..][0 .. out_tokens.len * 4], std.mem.sliceAsBytes(out_tokens));
    @memcpy(record[16 + out_tokens.len * 4 ..], std.mem.sliceAsBytes(out_logits));
    if (opts.save) |path|
    {
        try writeFile(path, record);
        std.debug.print("{s}: saved {d} tokens and their logits to {s}\n", .{ name, opts.generate, path });
    }
    if (opts.compare) |path|
    {
        const want = try mf.readWhole(gpa, path);
        defer gpa.free(want);
        if (want.len != record.len or !std.mem.eql(u8, want[0..16], record[0..16]))
        {
            std.debug.print("{s}: FAIL: {s} is from a run of another shape\n", .{ name, path });
            return error.CompareFailed;
        }
        const tb = out_tokens.len * 4;
        const same_tokens = std.mem.eql(u8, want[16..][0..tb], record[16..][0..tb]);
        const same_logits = std.mem.eql(u8, want[16 + tb ..], record[16 + tb ..]);
        if (!same_tokens or !same_logits)
        {
            std.debug.print("{s}: FAIL: generated tokens {s}, logits {s} compared with {s}\n", .{ name, if (same_tokens) "match" else "differ", if (same_logits) "match bit for bit" else "differ", path });
            return error.CompareFailed;
        }
        std.debug.print("{s}: PASS: all {d} generated tokens and their logits match {s} bit for bit\n", .{ name, opts.generate, path });
    }
}

/// Where `olmoe.step`'s time went, in seconds.
fn reportProfile(name: []const u8, what: []const u8) void
{
    const p = olmoe.profile;
    const s = struct {
        fn f(ns: u64) f64
        {
            return @as(f64, @floatFromInt(ns)) / 1e9;
        }
    }.f;
    std.debug.print("{s}: {s} time: projections {d:.1} s, QK norm and rope {d:.1} s, attention {d:.1} s, output {d:.1} s, router {d:.1} s, experts {d:.1} s, LM head {d:.1} s\n", .{
        name, what, s(p.projections), s(p.qk_rope), s(p.attention), s(p.output), s(p.router), s(p.experts), s(p.head),
    });
}

/// Formats into `buf` with a terminating zero.
fn zPrint(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]const u8
{
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

fn writeFile(path: []const u8, bytes: []const u8) !void
{
    var buf: [512]u8 = undefined;
    const z = try zPrint(&buf, "{s}", .{path});
    const fd = c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o644));
    if (fd < 0)
    {
        return error.WriteFailed;
    }
    defer _ = c.close(fd);
    var off: usize = 0;
    while (off < bytes.len)
    {
        const n = c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (n <= 0)
        {
            return error.WriteFailed;
        }
        off += @intCast(n);
    }
}

/// The model, for messages.
fn modelName(cfg: olmoe.Config) []const u8
{
    return switch (cfg.arch) {
        .olmoe => "OLMoE",
        .qwen3moe => if (cfg.yarn > 0) "Qwen3-MoE (YaRN)" else "Qwen3-MoE",
    };
}

/// Largest absolute difference; a NaN anywhere counts as infinitely wrong
/// (@max would otherwise drop it).
fn maxErr(a: []const f32, b: []align(1) const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| {
        const e = @abs(x - y);
        if (std.math.isNan(e)) return std.math.inf(f32);
        m = @max(m, e);
    }
    return m;
}

/// Runs every golden prompt and compares with the reference. Fails if any
/// position's top-1 token differs or the logits differ by more than 1e-2.
pub fn runGolden(gpa: std.mem.Allocator, w: *const olmoe.Weights, moe: olmoe.Moe, opts: Options, name: []const u8) !void {
    const gbytes = try mf.readWhole(gpa, opts.golden orelse return error.MissingGolden);
    defer gpa.free(gbytes);
    const golden = try tfref.File.parse(gbytes);
    const gcfg = (try golden.get("config")).i32s();
    const cfg = w.cfg;
    if (gcfg[1] != cfg.layers or gcfg[2] != cfg.experts or gcfg[3] != cfg.top_k or gcfg[4] != cfg.d_model or gcfg[5] != cfg.vocab)
        return error.GoldenConfigMismatch;

    var ok = true;
    var worst_logit: f32 = 0;
    var same_bits = true;
    var compared: usize = 0;
    for (0..@intCast(gcfg[0])) |p| {
        var buf: [64]u8 = undefined;
        const toks_g = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const n = toks_g.len;
        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        for (tokens, toks_g) |*t, g| t.* = @intCast(g);
        const logits = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(logits);
        const trace: olmoe.Trace = .{
            .hidden = try gpa.alloc(f32, cfg.layers * n * cfg.d_model),
            .topk = try gpa.alloc(u32, cfg.layers * n * cfg.top_k),
        };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);

        const t0 = mf.nowNs();
        try olmoe.forward(gpa, w, moe, tokens, logits, trace);
        const ms = @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e6;
        if (moe.ahead != null)
        {
            const plain = try gpa.alloc(f32, n * cfg.vocab);
            defer gpa.free(plain);
            try olmoe.forward(gpa, w, withoutLookahead(moe), tokens, plain, null);
            same_bits = same_bits and std.mem.eql(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(logits));
            compared += plain.len;
        }

        var hidden_err: f32 = 0;
        var routing_same: usize = 0;
        var routing_total: usize = 0;
        for (0..cfg.layers) |l| {
            if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))) |gh| {
                hidden_err = @max(hidden_err, maxErr(trace.hidden[l * n * cfg.d_model ..][0 .. n * cfg.d_model], gh.f32s()));
            } else |_| {}
            const gi = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l }))).i32s();
            // Compare chosen sets per token (order within ties may differ).
            for (0..n) |t| {
                for (trace.topk[(l * n + t) * cfg.top_k ..][0..cfg.top_k]) |e| {
                    var found = false;
                    for (gi[t * cfg.top_k ..][0..cfg.top_k]) |ge| found = found or ge == e;
                    routing_same += @intFromBool(found);
                    routing_total += 1;
                }
            }
        }
        const gl = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        const lerr = maxErr(logits, gl);
        var top1: usize = 0;
        for (0..n) |t| top1 += @intFromBool(olmoe.argmax(logits[t * cfg.vocab ..][0..cfg.vocab]) == olmoe.argmax(gl[t * cfg.vocab ..][0..cfg.vocab]));
        worst_logit = @max(worst_logit, lerr);
        const pass = top1 == n and lerr <= 1e-2;
        ok = ok and pass;
        std.debug.print("{s}: prompt {d} ({d} tokens, {d:.0} ms): logits max err {e:.2}, top-1 {d}/{d}, hidden max err {e:.2}, experts chosen {d}/{d} same {s}\n", .{
            name, p, n, ms, lerr, top1, n, hidden_err, routing_same, routing_total, if (pass) "ok" else "FAIL",
        });
    }
    if (!ok) {
        std.debug.print("{s}: FAIL: logits differ by up to {e:.2} (target 1e-2) or a top-1 token differs\n", .{ name, worst_logit });
        return error.ForwardMismatch;
    }
    std.debug.print("{s}: PASS: {s} logits within 1e-2 of the PyTorch reference (worst {e:.2}), every top-1 token identical\n", .{ name, modelName(cfg), worst_logit });
    if (moe.ahead) |la|
    {
        try reportLookahead(la, name, "prompts", same_bits, compared);
    }
    if (opts.generate > 0)
    {
        try runGenerate(gpa, w, moe, golden, opts, name);
    }
    if (opts.minutes > 0)
    {
        if (moe.ahead == null)
        {
            if (phase_cache) |cache|
            {
                cache.stats = .{};
            }
        }
        try runSoak(gpa, w, moe, golden, opts, name);
        if (moe.ahead) |la|
        {
            try reportLookahead(la, name, "soak", true, 0);
        }
        else if (phase_cache) |cache|
        {
            std.debug.print("{s}: (soak) ", .{name});
            reportCache(name, cache);
        }
    }
}

/// The same `Moe` with the predictor off, for the on-versus-off check.
fn withoutLookahead(moe: olmoe.Moe) olmoe.Moe
{
    var m = moe;
    m.ahead = null;
    m.prefetch = null;
    return m;
}

/// Prints the predictor's recall and cost since the last report, then starts
/// a new count. Fails if turning it on changed any output bit.
fn reportLookahead(la: *olmoe.Lookahead, name: []const u8, phase: []const u8, same_bits: bool, compared: usize) !void
{
    const h = la.horizon;
    std.debug.print("{s}: lookahead ({s}, top_k + {d} guesses): recall at H={d} {d:.1}% ({d} expert choices)", .{ name, phase, la.extra, h, la.recall(h) * 100, la.total[h - 1] });
    for (1..h) |near|
    {
        std.debug.print(", layer {d} from the embeddings {d:.1}%", .{ near - 1, la.recall(near) * 100 });
    }
    std.debug.print("; guessing took {d:.2}% of layer time ({d:.1} of {d:.1} ms)", .{
        la.overhead() * 100, @as(f64, @floatFromInt(la.predict_ns)) / 1e6, @as(f64, @floatFromInt(la.layer_ns)) / 1e6,
    });
    if (compared > 0)
    {
        std.debug.print("; predictor on vs off: {s} ({d} values)", .{ if (same_bits) "bit-identical" else "DIFFERENT", compared });
    }
    if (la.mispredict > 0)
    {
        std.debug.print("; {d} guesses forced wrong ({d:.0}% asked)", .{ la.replaced, la.mispredict * 100 });
    }
    std.debug.print("\n", .{});
    la.restart();
    if (phase_cache) |cache|
    {
        std.debug.print("{s}: ({s}{s}) ", .{ name, phase, if (compared > 0) ", including the predictor-off passes" else "" });
        reportCache(name, cache);
        cache.stats = .{};
    }
    if (!same_bits)
    {
        return error.LookaheadChangedOutput;
    }
}

fn goldenTokens(gpa: std.mem.Allocator, golden: tfref.File, p: usize) ![]u32
{
    var buf: [64]u8 = undefined;
    const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
    const tokens = try gpa.alloc(u32, ids.len);
    for (tokens, ids) |*t, id|
    {
        t.* = @intCast(id);
    }
    return tokens;
}

fn mb(bytes: u64) f64
{
    return @as(f64, @floatFromInt(bytes)) / 1e6;
}

/// Resident memory of this process in bytes, or 0 if /proc is unreadable.
fn rssBytes() u64
{
    const fd = c.open("/proc/self/statm", .{ .ACCMODE = .RDONLY });
    if (fd < 0)
    {
        return 0;
    }
    defer _ = c.close(fd);
    var buf: [128]u8 = undefined;
    const n = c.read(fd, &buf, buf.len);
    if (n <= 0)
    {
        return 0;
    }
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(n)], ' ');
    _ = it.next();
    const pages = std.fmt.parseInt(u64, it.next() orelse return 0, 10) catch return 0;
    return pages * std.heap.pageSize();
}

/// Generates `opts.generate` tokens after each golden prompt. When greedy and
/// the golden file has transformers' generate() output, every token must
/// match and the logits each was picked from must be within 1e-2.
fn runGenerate(gpa: std.mem.Allocator, w: *const olmoe.Weights, moe: olmoe.Moe, golden: tfref.File, opts: Options, name: []const u8) !void
{
    const cfg = w.cfg;
    const greedy = opts.temperature <= 0;
    var sampler = olmoe.Sampler.seeded(opts.temperature, opts.top_k, opts.seed);
    const out = try gpa.alloc(u32, opts.generate);
    defer gpa.free(out);
    const step_logits = try gpa.alloc(f32, opts.generate * cfg.vocab);
    defer gpa.free(step_logits);
    var ok = true;
    var compared = false;
    var same_bits = true;
    var compared_bits: usize = 0;
    for (0..@intCast((try golden.get("config")).i32s()[0])) |p|
    {
        const prompt = try goldenTokens(gpa, golden, p);
        defer gpa.free(prompt);
        var cache = try olmoe.KvCache.init(gpa, cfg, prompt.len + opts.generate);
        defer cache.deinit(gpa);
        const sampler_start = sampler;
        const t0 = mf.nowNs();
        try olmoe.generate(gpa, w, moe, &cache, prompt, &sampler, out, step_logits);
        const s = @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e9;
        if (moe.ahead != null)
        {
            var again = sampler_start;
            const out2 = try gpa.alloc(u32, out.len);
            defer gpa.free(out2);
            const logits2 = try gpa.alloc(f32, step_logits.len);
            defer gpa.free(logits2);
            try olmoe.generate(gpa, w, withoutLookahead(moe), &cache, prompt, &again, out2, logits2);
            same_bits = same_bits and std.mem.eql(u32, out, out2) and std.mem.eql(u8, std.mem.sliceAsBytes(step_logits), std.mem.sliceAsBytes(logits2));
            compared_bits += logits2.len;
        }
        std.debug.print("{s}: generate prompt {d}: {d} tokens in {d:.2} s ({d:.1} tokens/s, prompt included)\n", .{ name, p, out.len, s, @as(f64, @floatFromInt(out.len)) / s });
        std.debug.print("{s}:   tokens:", .{name});
        for (out) |t|
        {
            std.debug.print(" {d}", .{t});
        }
        std.debug.print("\n", .{});

        var buf: [64]u8 = undefined;
        const want = golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p})) catch continue;
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        if (!greedy)
        {
            continue;
        }
        compared = true;
        const g = @min(out.len, want.len());
        var same: usize = 0;
        while (same < g and out[same] == @as(u32, @intCast(want.i32s()[same])))
        {
            same += 1;
        }
        // Logits stay comparable up to and including the first differing token.
        const steps = @min(g, same + 1);
        const lerr = maxErr(step_logits[0 .. steps * cfg.vocab], want_logits[0 .. steps * cfg.vocab]);
        const pass = same == g and lerr <= 1e-2;
        ok = ok and pass;
        std.debug.print("{s}:   same as transformers' generate(): {d}/{d} tokens, logits max err {e:.2} {s}\n", .{ name, same, g, lerr, if (pass) "ok" else "FAIL" });
    }
    if (moe.ahead) |la|
    {
        try reportLookahead(la, name, "generation", same_bits, compared_bits);
    }
    if (!compared)
    {
        std.debug.print("{s}: generated without a check (sampling, or no generate() output in the golden file)\n", .{name});
        return;
    }
    if (!ok)
    {
        std.debug.print("{s}: FAIL: generated tokens or their logits differ from transformers' generate()\n", .{name});
        return error.GenerateMismatch;
    }
    std.debug.print("{s}: PASS: every generated token matches transformers' generate(), logits within 1e-2\n", .{name});
}

/// Repeats generation for `opts.minutes`; fails if tokens or logits leave the reference
/// (a predictor-off round, or else round 1) or memory grows 1% after round 1.
/// Says how a soak round's generation for one prompt differs from the reference.
fn reportDiff(name: []const u8, round: usize, prompt: usize, ref: []const u32, out: []const u32, ref_l: []const f32, logits: []const f32, vocab: usize) void
{
    var tokens_differ: usize = 0;
    var first_token: ?usize = null;
    for (ref, out, 0..) |a, b, i|
    {
        if (a != b)
        {
            tokens_differ += 1;
            if (first_token == null)
            {
                first_token = i;
            }
        }
    }
    var first_row: ?usize = null;
    var rows_differ: usize = 0;
    var worst: f32 = 0;
    for (0..ref.len) |t|
    {
        const ra = ref_l[t * vocab ..][0..vocab];
        const rb = logits[t * vocab ..][0..vocab];
        if (std.mem.eql(u8, std.mem.sliceAsBytes(ra), std.mem.sliceAsBytes(rb)))
        {
            continue;
        }
        rows_differ += 1;
        if (first_row == null)
        {
            first_row = t;
        }
        for (ra, rb) |a, b|
        {
            const e = @abs(a - b);
            if (std.math.isNan(e) or e > worst)
            {
                worst = if (std.math.isNan(e)) std.math.inf(f32) else e;
            }
        }
    }
    std.debug.print("{s}: soak round {d}, prompt {d}: {d} of {d} tokens differ (first at {?d}); logits of {d} tokens differ (first at {?d}), by up to {e}\n", .{
        name, round, prompt, tokens_differ, ref.len, first_token, rows_differ, first_row, worst,
    });
}

/// With `--minutes`, says whether this process's loaded weights still hash as they did at startup.
fn reportWeights(name: []const u8, before: u64, region: []const u8) void
{
    const after = std.hash.Wyhash.hash(0, region);
    std.debug.print("{s}: weights in memory ({d:.2} GB) {s} since startup\n", .{
        name, @as(f64, @floatFromInt(region.len)) / 1e9, if (after == before) "unchanged" else "CHANGED",
    });
}

fn runSoak(gpa: std.mem.Allocator, w: *const olmoe.Weights, moe: olmoe.Moe, golden: tfref.File, opts: Options, name: []const u8) !void
{
    const cfg = w.cfg;
    const n_new = if (opts.generate > 0) opts.generate else 64;
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    var longest: usize = 0;
    for (0..prompts) |p|
    {
        const t = try goldenTokens(gpa, golden, p);
        defer gpa.free(t);
        longest = @max(longest, t.len);
    }
    var cache = try olmoe.KvCache.init(gpa, cfg, longest + n_new);
    defer cache.deinit(gpa);
    const first = try gpa.alloc(u32, prompts * n_new);
    defer gpa.free(first);
    const out = try gpa.alloc(u32, n_new);
    defer gpa.free(out);
    const logits = try gpa.alloc(f32, n_new * cfg.vocab);
    defer gpa.free(logits);
    const per = n_new * cfg.vocab;
    const ref_logits = try gpa.alloc(f32, prompts * per);
    defer gpa.free(ref_logits);
    const plain = moe.ahead != null;
    if (plain)
    {
        var sampler = olmoe.Sampler.seeded(opts.temperature, opts.top_k, opts.seed);
        for (0..prompts) |p|
        {
            const prompt = try goldenTokens(gpa, golden, p);
            defer gpa.free(prompt);
            try olmoe.generate(gpa, w, withoutLookahead(moe), &cache, prompt, &sampler, out, logits);
            @memcpy(first[p * n_new ..][0..n_new], out);
            @memcpy(ref_logits[p * per ..][0..per], logits);
        }
        std.debug.print("{s}: soak reference round run with the predictor off\n", .{name});
    }
    const t0 = mf.nowNs();
    const end = t0 + @as(u64, @intFromFloat(opts.minutes * 60e9));
    var rounds: usize = 0;
    var tokens: u64 = 0;
    var rss_base: u64 = 0;
    var rss_max: u64 = 0;
    var same = true;
    var bad_rounds: usize = 0;
    while (rounds == 0 or mf.nowNs() < end)
    {
        const r0 = mf.nowNs();
        var differ = false;
        // The same seed every round, so a healthy run repeats its tokens exactly.
        var sampler = olmoe.Sampler.seeded(opts.temperature, opts.top_k, opts.seed);
        for (0..prompts) |p|
        {
            const prompt = try goldenTokens(gpa, golden, p);
            defer gpa.free(prompt);
            try olmoe.generate(gpa, w, moe, &cache, prompt, &sampler, out, logits);
            const ref = first[p * n_new ..][0..n_new];
            const ref_l = ref_logits[p * per ..][0..per];
            if (rounds == 0 and !plain)
            {
                @memcpy(ref, out);
                @memcpy(ref_l, logits);
            }
            else if (!std.mem.eql(u32, ref, out) or !std.mem.eql(u8, std.mem.sliceAsBytes(ref_l), std.mem.sliceAsBytes(logits)))
            {
                same = false;
                differ = true;
                reportDiff(name, rounds + 1, p, ref, out, ref_l, logits, cfg.vocab);
            }
            tokens += n_new;
        }
        rounds += 1;
        if (differ)
        {
            bad_rounds += 1;
        }
        const rss = rssBytes();
        if (rounds == 1)
        {
            rss_base = rss;
        }
        rss_max = @max(rss_max, rss);
        const rs = @as(f64, @floatFromInt(mf.nowNs() - r0)) / 1e9;
        std.debug.print("{s}: soak round {d}: {d:.1} tokens/s, resident memory {d:.0} MB{s}\n", .{
            name, rounds, @as(f64, @floatFromInt(prompts * n_new)) / rs, mb(rss), if (differ) ", tokens or logits differ from the reference" else "",
        });
    }
    const secs = @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e9;
    const drift = if (rss_base > 0) (mb(rss_max) - mb(rss_base)) / mb(rss_base) * 100 else 0;
    const pass = same and drift < 1;
    std.debug.print("{s}: soak: {d} tokens in {d:.1} min ({d:.1} tokens/s), {d} rounds, tokens and logits identical to the {s} in every round: {s} ({d} rounds differ), resident memory {d:.0} MB after round 1, max {d:.0} MB (+{d:.2}%) {s}\n", .{
        name, tokens, secs / 60, @as(f64, @floatFromInt(tokens)) / secs, rounds, if (plain) "predictor-off round" else "first round", if (same) "yes" else "no", bad_rounds, mb(rss_base), mb(rss_max), drift, if (pass) "PASS" else "FAIL",
    });
    if (!pass)
    {
        return error.SoakFailed;
    }
}
