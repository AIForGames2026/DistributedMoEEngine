//! OLMoE-1B-7B (TASK-10), Qwen3-MoE (TASK-12), Qwen3.5-MoE (TASK-29) or Qwen3.8-Flash-Next (TASK-31) forward pass split across
//! any number of processes (TASK-33; two by default). Linux.
//!
//!   olmoe local MODEL --golden G.tfref                 one process, every expert
//!   olmoe shm   MODEL --golden G.tfref [options]       --ranks N processes (default 2), shared memory
//!   olmoe rdma  MODEL --golden G.tfref [options]       the same over RDMA (-Drdma=true)
//!   olmoe expert MODEL --port P [--rank R --ranks N]   an expert rank over TCP (rank 1 by default)
//!   olmoe router MODEL --host IP --port P --golden G.tfref [options]
//!   olmoe router MODEL --peer [tcp:|rdma:]IP:PORT[@DEV] ...  one --peer per expert rank, in rank order
//!   olmoe rdma-expert / rdma-router                    the same over RDMA
//!   olmoe dashboard --models DIR                       the operator dashboard (TASK-25, dashboard.zig)
//!   olmoe node --dashboard IP --models DIR             its node agent (node_agent.zig)
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
//!          --host waits for --ranks N - 1 healthy experts it lists, takes each
//!          expert from the first that announced it, holds the rest itself, and
//!          refuses to route once any of them turns unhealthy), --advertise IP (the
//!          address an expert announces; default the one the tracker sees),
//!          --yarn F (TASK-12: YaRN rope scaling by F over 32,768 positions,
//!          for Qwen3-MoE past its native context; Qwen uses 4 for 128k),
//!          --serve [IP:]PORT (TASK-30: instead of the golden checks, rank 0
//!          answers OpenAI chat completions; 127.0.0.1 unless an IP is given),
//!          --ctx N (the server's context, default 4096), --no-reuse (every
//!          request runs cold instead of reusing the previous one's prefix),
//!          --record FILE (TASK-16: rank 0 writes each step's experts, and the
//!          server's chat, to FILE for the built-in page to replay),
//!          --capacity C (TASK-17: each process takes at most C times its even
//!          share of a --chunk step's expert calls per layer, rank 1 granting
//!          its limit to rank 0; calls past it are dropped and counted),
//!          --expert-capacity C (the same for each expert), --max-step N (the
//!          most tokens one step may bring to the split, default 64),
//!          --memory-budget MB (a hard limit on this process's memory: the
//!          heap refuses to grow past what the weights and GPU leave),
//!          --spike F (with --context: the middle third of the prompt comes in
//!          steps F times as big, reported phase by phase),
//!          --kv-format f32|bf16|fp8|bf16_fp8 (TASK-27: how rank 0's KV cache
//!          stores keys and values; bf16 halves it, fp8 cuts it about 3.9 times,
//!          bf16_fp8 keeps keys in bf16 and values in fp8, about 2.6 times)
//!
//! Rank 0 holds every non-expert weight (embedding, attention, norms, router,
//! LM head) and its own experts; ranks 1 to N-1 hold the rest of the experts
//! (`split.zig`). The placement manifest says which rank owns which experts, one
//! line per range: `<rank> <first>-<last>`; without one, the experts are split
//! into N equal ranges, rank 0 first. For each layer, rank 0 sends every (token,
//! expert) pair to the rank owning it as a dispatch packet carrying the token's
//! normalized hidden state (as several packets when it does not fit one, as
//! with f32), runs its own experts meanwhile, and adds the combines back in a
//! fixed order, so the result does not depend on which rank ran what.
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
const split = @import("split.zig");
const dashboard = @import("dashboard.zig");
const node_agent = @import("node_agent.zig");
const Experts = split.Experts;
const SplitMoe = split.SplitMoe;
const CacheSource = @import("expert_cache.zig").Source;
const tracker = @import("tracker.zig");
const kv_tier = @import("kv_tier.zig");
const prefix_cache = @import("prefix_cache.zig");
const tokenizer = @import("tokenizer.zig");
const chat = @import("chat.zig");
const completion = @import("completion.zig");
const serve = @import("serve.zig");
const telemetry = @import("telemetry.zig");
const budget = @import("budget.zig");
const gpu =if (build_options.gpu) @import("gpu.zig") else struct {
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
    fn systemBytes() u64
    {
        return 0;
    }
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
    prefix_cache: ?[]const u8 = null,
    save: ?[]const u8 = null,
    compare: ?[]const u8 = null,
    swarm: ?*const Swarm = null,
    serve: ?[]const u8 = null,
    ctx: usize = 4096,
    reuse: bool = true,
    record: ?[]const u8 = null,
    capacity: f32 = 0,
    expert_capacity: f32 = 0,
    max_step: usize = Experts.default_max_tokens,
    memory_budget: u64 = 0,
    spike: usize = 0,
    kv_format: olmoe.KvFormat = .f32,
    /// TASK-33: processes in the run, this expert's rank, and the peers a router dials.
    ranks: ?usize = null,
    rank: u8 = 1,
    peers: []const PeerAddr = &.{},
};

/// Every allocation of this process goes through `proc_heap` (TASK-17), which
/// counts what is live and enforces `--memory-budget` once it is armed.
var proc_heap: budget.Heap = undefined;
var heap_label: []const u8 = "olmoe";
var memory: ?budget.Memory = null;

/// A router's view of the swarm (TASK-09, TASK-33): its tracker link, watching
/// the expert processes it connected to (rank `i + 1` for `peers[i]`), and the
/// ownership taken from what they announced.
const Swarm = struct {
    client: *tracker.Client,
    peers: []const tracker.Entry,
    owner: [max_experts]u8,
};

/// A peer rank 0 dials (`--peer [tcp:|rdma:]HOST:PORT[@DEV]`), with the RDMA device
/// that reaches it when not `--dev`.
const PeerAddr = struct {
    host: []const u8,
    port: u16,
    rdma: bool,
    device: ?[:0]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    proc_heap = .init(init.gpa);
    defer reportHeap();
    const gpa = proc_heap.allocator();
    olmoe.big_allocator = gpa;
    {
        // The operator dashboard and its node agents (TASK-25) take their own options.
        var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
        var all: std.ArrayList([]const u8) = .empty;
        while (it.next()) |arg|
        {
            try all.append(init.arena.allocator(), arg);
        }
        if (all.items.len >= 2 and std.mem.eql(u8, all.items[1], "dashboard"))
        {
            return dashboard.main(init.gpa, all.items[2..]);
        }
        if (all.items.len >= 2 and std.mem.eql(u8, all.items[1], "node"))
        {
            return node_agent.main(init.gpa, all.items[2..]);
        }
    }
    const opts = parseArgs(init) catch {
        std.debug.print(
            \\usage: olmoe local|shm|rdma MODEL --golden G.tfref [--ranks N] [options]
            \\       olmoe dashboard --models DIR [options]   (TASK-25: the operator dashboard)
            \\       olmoe node --dashboard IP[:PORT] --models DIR [options]   (its node agent)
            \\       olmoe expert|rdma-expert MODEL --port P [--rank R --ranks N] [options]
            \\       olmoe router|rdma-router MODEL --host IP --port P --golden G.tfref [options]
            \\       olmoe router|rdma-router MODEL --peer [tcp:|rdma:]IP:PORT[@DEV] ... --golden G.tfref [options]
            \\options: --payload fp8|bf16|f32  --placement FILE  --slots S  --dev D  --gid G  --gpu  --yarn F
            \\         --generate N  --temperature T  --top-k K  --seed S  --minutes M
            \\         --lookahead H  --extra E  --cache N  --pin L  --loaders N  --mispredict P
            \\         --tracker IP[:PORT]  --advertise IP  (a router with --tracker needs no --host; --ranks N waits for N-1 experts)
            \\         --serve [IP:]PORT  --ctx N  --no-reuse  (an OpenAI-compatible server instead of --golden)
            \\         --record FILE  (rank 0 records every step for the page to replay)
            \\         --capacity C  --expert-capacity C  --max-step N  --memory-budget MB  --spike F  (TASK-17)
            \\         --kv-format f32|bf16|fp8|bf16_fp8  (TASK-27; f32 by default)
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
        heap_label = "local";
        var experts: Experts = .{ .local = &l.local, .max_tokens = opts.max_step };
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
        var cap = olmoe.Capacity.init(l.cfg, opts.chunk, opts.capacity, opts.expert_capacity, null);
        if (opts.capacity > 0 or opts.expert_capacity > 0)
        {
            moe.capacity = &cap;
        }
        try armBudget(opts, "local");
        var host_buf: [64]u8 = undefined;
        const host = hostName(&host_buf);
        const nobody: [max_experts]u8 = @splat(0);
        const live: Live = .{
            .owner = nobody[0..l.cfg.experts],
            .ranks = &.{host},
            .probe = if (box) |b| .{ .ctx = b.cache, .sample = sampleCache } else .{ .ctx = &l.local, .sample = sampleLocal },
        };
        const tel = try openTelemetry(gpa, l.cfg, moe, opts, "local", live);
        const run_moe = if (tel) |t| t.tap.moe() else moe;
        const result = if (opts.serve != null)
            runServe(gpa, &l, run_moe, opts, "local", tel.?.bc, if (box != null) opts.max_step else null)
        else if (opts.context > 0)
            runLong(gpa, &l.weights.?, run_moe, opts, "local")
        else
            runGolden(gpa, &l.weights.?, run_moe, opts, "local");
        if (tel) |t|
        {
            t.close("local");
        }
        if (box) |b|
        {
            reportCache("local", b.cache);
        }
        reportGuards(opts, "local", moe.capacity, null);
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
        var link: Link = undefined;
        try link.initSocket(opts, fd, is_rdma, @intCast(opts.rank), slots);
        return runLinks(gpa, opts, (&link)[0..1], opts.rank, if (is_rdma) "rdma" else "tcp", &.{});
    }
    if (std.mem.eql(u8, mode, "router") or std.mem.eql(u8, mode, "rdma-router")) {
        if (opts.tracker) |t|
        {
            if (opts.peers.len > 0)
            {
                std.debug.print("router: give the tracker or --peer addresses, not both\n", .{});
                std.process.exit(2);
            }
            return runTrackedRouter(gpa, opts, is_rdma, t);
        }
        const one = [_]PeerAddr{.{ .host = opts.host, .port = opts.port, .rdma = is_rdma }};
        const addrs = if (opts.peers.len > 0) opts.peers else &one;
        var names: [split.max_ranks][]const u8 = undefined;
        for (addrs, 0..) |a, i|
        {
            names[i] = a.host;
        }
        return runRouter(gpa, opts, addrs, names[0..addrs.len]);
    }
    std.debug.print("unknown mode '{s}'\n", .{mode});
    std.process.exit(2);
}

fn parseArgs(init: std.process.Init) !Options {
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    var opts: Options = .{ .mode = it.next() orelse return error.Usage, .model = it.next() orelse return error.Usage };
    var peers: std.ArrayList(PeerAddr) = .empty;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--gpu")) {
            opts.gpu = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--no-reuse"))
        {
            opts.reuse = false;
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
        } else if (std.mem.eql(u8, arg, "--prefix-cache")) {
            opts.prefix_cache = value;
        } else if (std.mem.eql(u8, arg, "--save")) {
            opts.save = value;
        } else if (std.mem.eql(u8, arg, "--compare")) {
            opts.compare = value;
        } else if (std.mem.eql(u8, arg, "--yarn")) {
            yarn_factor = try std.fmt.parseFloat(f32, value);
        } else if (std.mem.eql(u8, arg, "--kv-format")) {
            opts.kv_format = olmoe.KvFormat.parse(value) orelse return error.Usage;
        } else if (std.mem.eql(u8, arg, "--serve")) {
            opts.serve = value;
        } else if (std.mem.eql(u8, arg, "--record")) {
            opts.record = value;
        } else if (std.mem.eql(u8, arg, "--capacity")) {
            opts.capacity = try std.fmt.parseFloat(f32, value);
            if (!(opts.capacity > 0))
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--expert-capacity")) {
            opts.expert_capacity = try std.fmt.parseFloat(f32, value);
            if (!(opts.expert_capacity > 0))
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--max-step")) {
            opts.max_step = try std.fmt.parseInt(usize, value, 10);
            if (opts.max_step == 0)
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--memory-budget")) {
            opts.memory_budget = (try std.fmt.parseInt(u64, value, 10)) * 1_000_000;
            if (opts.memory_budget == 0)
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--spike")) {
            opts.spike = try std.fmt.parseInt(usize, value, 10);
            if (opts.spike < 2)
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--ctx")) {
            opts.ctx = try std.fmt.parseInt(usize, value, 10);
            if (opts.ctx < 2) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--tracker")) {
            opts.tracker = value;
        } else if (std.mem.eql(u8, arg, "--advertise")) {
            opts.advertise = value;
        } else if (std.mem.eql(u8, arg, "--mispredict")) {
            opts.mispredict = try std.fmt.parseFloat(f32, value);
            if (opts.mispredict < 0 or opts.mispredict > 1) return error.Usage;
        } else if (std.mem.eql(u8, arg, "--ranks")) {
            const n = try std.fmt.parseInt(usize, value, 10);
            if (n < 2 or n > split.max_ranks)
            {
                return error.Usage;
            }
            opts.ranks = n;
        } else if (std.mem.eql(u8, arg, "--rank")) {
            opts.rank = try std.fmt.parseInt(u8, value, 10);
            if (opts.rank == 0 or opts.rank >= split.max_ranks)
            {
                return error.Usage;
            }
        } else if (std.mem.eql(u8, arg, "--peer")) {
            const peer = try parsePeer(value, std.mem.startsWith(u8, opts.mode, "rdma"));
            if (peer.rdma and !build_options.rdma)
            {
                return error.Usage;
            }
            try peers.append(init.arena.allocator(), peer);
        } else return error.Usage;
    }
    opts.peers = peers.items;
    if (opts.ranks != null and opts.peers.len > 0 and opts.ranks.? != opts.peers.len + 1)
    {
        return error.Usage;
    }
    return opts;
}

/// `[tcp:|rdma:]HOST:PORT[@DEV]`; without a prefix, the mode's transport.
fn parsePeer(text: [:0]const u8, rdma_default: bool) !PeerAddr
{
    var device: ?[:0]const u8 = null;
    var rest: []const u8 = text;
    if (std.mem.lastIndexOfScalar(u8, text, '@')) |at|
    {
        device = text[at + 1 ..];
        rest = text[0..at];
    }
    var rdma = rdma_default;
    if (std.mem.startsWith(u8, rest, "tcp:"))
    {
        rdma = false;
        rest = rest[4..];
    }
    else if (std.mem.startsWith(u8, rest, "rdma:"))
    {
        rdma = true;
        rest = rest[5..];
    }
    const colon = std.mem.lastIndexOfScalar(u8, rest, ':') orelse return error.Usage;
    if (device != null and !rdma)
    {
        return error.Usage;
    }
    return .{ .host = rest[0..colon], .port = try std.fmt.parseInt(u16, rest[colon + 1 ..], 10), .rdma = rdma, .device = device };
}

/// Processes in this run: `--ranks`, else rank 0 plus each `--peer`, else the
/// ranks the manifest names, else 2.
fn rankCount(opts: Options, manifest: ?[]const u8) !usize
{
    const n = if (opts.ranks) |r| r else if (opts.peers.len > 0) opts.peers.len + 1 else if (manifest) |m| @max(2, try split.placementRanks(m)) else 2;
    if (n > split.max_ranks)
    {
        return error.TooManyRanks;
    }
    return n;
}

fn ranksOf(gpa: std.mem.Allocator, opts: Options) !usize
{
    const path = opts.placement orelse return rankCount(opts, null);
    const text = try mf.readWhole(gpa, path);
    defer gpa.free(text);
    return rankCount(opts, text);
}

/// Which rank owns each expert (`owner[e]`), from `--placement` or split evenly.
/// Returns the number of ranks.
fn readPlacement(gpa: std.mem.Allocator, opts: Options, experts: usize, owner: []u8) !usize {
    if (opts.placement) |path|
    {
        const text = try mf.readWhole(gpa, path);
        defer gpa.free(text);
        const ranks = try rankCount(opts, text);
        try split.parsePlacement(text, owner[0..experts], ranks);
        return ranks;
    }
    const ranks = try rankCount(opts, null);
    split.evenPlacement(owner[0..experts], ranks);
    return ranks;
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
        var state = try mf.openModel(gpa, path, file, null);
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
/// A router that found its experts through the tracker uses what they announced.
fn ownedMask(gpa: std.mem.Allocator, opts: Options, rank: u8, owner: *[max_experts]u8, owned: *[max_experts]bool) !struct { experts: usize, ranks: usize } {
    const file = try mf.File.open(opts.model);
    defer file.close();
    var state = try mf.openModel(gpa, opts.model, file, null);
    defer state.deinit(gpa);
    const experts = state.model.experts;
    var ranks: usize = 0;
    if (opts.swarm) |sw|
    {
        if (sw.peers[0].info.experts != experts)
        {
            std.debug.print("{s}: the expert processes announced {d} experts, but this model has {d}\n", .{ opts.mode, sw.peers[0].info.experts, experts });
            return error.PlacementMismatch;
        }
        owner.* = sw.owner;
        ranks = sw.peers.len + 1;
    }
    else
    {
        ranks = try readPlacement(gpa, opts, experts, owner);
    }
    if (rank >= ranks)
    {
        std.debug.print("{s}: this is rank {d}, but the run has {d} ranks (--ranks, --placement)\n", .{ opts.mode, rank, ranks });
        return error.PlacementMismatch;
    }
    for (owned, 0..) |*o, e|
    {
        o.* = e < experts and owner[e] == rank;
    }
    return .{ .experts = experts, .ranks = ranks };
}

/// Announces this expert process to the tracker: where to reach it and which
/// experts it owns. The tracker link sends heartbeats until it is closed.
fn joinAsExpert(gpa: std.mem.Allocator, opts: Options, where: []const u8, port: u16) !*tracker.Client
{
    var owner: [max_experts]u8 = undefined;
    var owned: [max_experts]bool = undefined;
    const experts = (try ownedMask(gpa, opts, opts.rank, &owner, &owned)).experts;
    var info: tracker.Info = .{ .role = .expert, .rank = opts.rank, .port = port, .experts = @intCast(experts) };
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

/// A router with no --host or --peer: joins the tracker, waits for `--ranks` - 1
/// healthy expert processes (one by default), takes each expert from the lowest
/// tracker id that announced it, holds the rest itself, and refuses to route
/// once any of them turns unhealthy.
fn runTrackedRouter(gpa: std.mem.Allocator, opts: Options, is_rdma: bool, where: []const u8) !void
{
    const t = try tracker.parseAddress(where);
    var info: tracker.Info = .{ .role = .router, .rank = 0 };
    info.setLocal();
    const client = try tracker.Client.start(gpa, t.host, t.port, info);
    defer client.close(gpa, false);
    const want = (opts.ranks orelse 2) - 1;
    var tj: [16]u8 = undefined;
    std.debug.print("{s} router: joined the tracker at {s}:{d} as id {d}; waiting for {d} healthy expert process{s}\n", .{ tracker.stamp(&tj), t.host, t.port, client.id, want, if (want == 1) "" else "es" });
    var entries: [split.max_ranks]tracker.Entry = undefined;
    const peers = entries[0..want];
    try client.waitForMany(.expert, peers, 120_000);
    // Ranks in the order the experts announced them (their --rank), then by tracker id.
    std.mem.sort(tracker.Entry, peers, {}, struct {
        fn lt(_: void, a: tracker.Entry, b: tracker.Entry) bool
        {
            return a.info.rank < b.info.rank or (a.info.rank == b.info.rank and a.id < b.id);
        }
    }.lt);
    var swarm: Swarm = .{ .client = client, .peers = peers, .owner = @splat(0) };
    const experts = peers[0].info.experts;
    for (peers) |p|
    {
        if (p.info.experts != experts)
        {
            std.debug.print("router: expert processes announce {d} and {d} experts\n", .{ experts, p.info.experts });
            return error.PlacementMismatch;
        }
    }
    info.experts = experts;
    for (0..experts) |e|
    {
        for (peers, 1..) |p, r|
        {
            if (p.info.owns(e))
            {
                swarm.owner[e] = @intCast(r);
                break;
            }
        }
        if (swarm.owner[e] == 0)
        {
            info.setOwned(e);
        }
    }
    try client.update(info);
    var ids: [split.max_ranks]u32 = undefined;
    var ips: [split.max_ranks][16]u8 = undefined;
    var addrs: [split.max_ranks]PeerAddr = undefined;
    var names: [split.max_ranks][]const u8 = undefined;
    for (peers, 0..) |*p, i|
    {
        ids[i] = p.id;
        const host = tracker.formatIp4(p.info.addr, &ips[i]);
        addrs[i] = .{ .host = host, .port = p.info.port, .rdma = is_rdma };
        names[i] = p.info.nodeName();
        var used: usize = 0;
        for (swarm.owner[0..experts]) |o|
        {
            used += @intFromBool(o == i + 1);
        }
        var ts: [16]u8 = undefined;
        std.debug.print("{s} router: rank {d} is expert id {d} (pid {d} on {s}) at {s}:{d}, announcing {d} of {d} experts, {d} used\n", .{ tracker.stamp(&ts), i + 1, p.id, p.info.pid, p.info.nodeName(), host, p.info.port, p.info.ownedCount(), experts, used });
    }
    client.watch(ids[0..want]);
    var tracked = opts;
    tracked.swarm = &swarm;
    return runRouter(gpa, tracked, addrs[0..want], names[0..want]);
}

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

/// Loads this rank's weights and runs its side: rank 0 over one endpoint per peer
/// (`names` are the peers' machines), an expert rank over its link to rank 0.
fn runRank(gpa: std.mem.Allocator, opts: Options, eps: []const *transport.Endpoint, rank: u8, name: []const u8, names: []const []const u8) !void {
    var owner: [max_experts]u8 = undefined;
    var owned: [max_experts]bool = undefined;
    const layout = try ownedMask(gpa, opts, rank, &owner, &owned);
    if (rank == 0 and layout.ranks != eps.len + 1)
    {
        std.debug.print("{s}: the placement has {d} ranks, but rank 0 has {d} peers\n", .{ name, layout.ranks, eps.len });
        return error.PlacementMismatch;
    }
    // After any fork (CUDA state does not survive one), before the weights.
    if (opts.gpu) try gpu.open();
    defer if (opts.gpu) gpu.stop();
    var l = try Loaded.init(gpa, opts.model, &owned, rank == 0, opts.cache == 0);
    defer l.deinit(gpa);
    // With an expert cache, an expert rank loads nothing at startup.
    if (opts.gpu and l.weightRegion().len > 0)
    {
        try gpu.map(l.weightRegion());
    }
    const weights_hash = if (opts.minutes > 0) std.hash.Wyhash.hash(0, l.weightRegion()) else 0;
    var experts: Experts = .{ .local = &l.local, .max_tokens = opts.max_step };
    try experts.alloc(gpa);
    defer experts.free(gpa);
    const label = if (rank != 0) "expert" else name;
    heap_label = label;
    defer if (opts.minutes > 0) reportWeights(label, weights_hash, l.weightRegion());
    const box = if (opts.cache > 0) try CacheBox.open(gpa, opts, &l, &owned, label) else null;
    defer if (box) |b| b.close(gpa);
    if (box) |b|
    {
        experts.cache = b.cache;
    }
    if (rank != 0)
    {
        try armBudget(opts, label);
        var owned_count: usize = 0;
        for (owned[0..l.cfg.experts]) |o|
        {
            owned_count += @intFromBool(o);
        }
        // Rank 0 waits for this grant when it limits calls (TASK-17); 0 means no limit.
        const grant: u16 = if (opts.capacity > 0) @intCast(@min(olmoe.Capacity.limitFor(opts.capacity, opts.chunk, l.cfg, owned_count), std.math.maxInt(u16))) else 0;
        try eps[0].grantCapacity(grant);
        if (grant > 0)
        {
            std.debug.print("expert: rank {d} granting rank 0 {d} expert calls per layer\n", .{ rank, grant });
        }
        var server: split.Server = .{ .ep = eps[0], .experts = &experts, .owned = owned[0..l.cfg.experts], .check_memory = &checkMemory };
        const served = server.serve();
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
    const peers = try gpa.alloc(split.Peer, eps.len);
    defer gpa.free(peers);
    for (peers, eps) |*p, ep|
    {
        p.* = .{ .ep = ep };
    }
    var sm: SplitMoe = .{
        .peers = peers,
        .local = &l.local,
        .experts = &experts,
        .owner = owner[0..cfg.experts],
        .enc = opts.payload,
        .contrib = try gpa.alloc(f32, opts.max_step * cfg.top_k * cfg.d_model),
        .pkt = pkt,
        .health = if (opts.swarm) |sw| sw.client else null,
    };
    defer gpa.free(sm.contrib);
    var la: olmoe.Lookahead = .{ .horizon = opts.lookahead, .extra = opts.extra, .mispredict = opts.mispredict };
    var moe = sm.moe();
    if (opts.lookahead > 0)
    {
        moe.ahead = &la;
        moe.prefetch = SplitMoe.prefetch;
    }
    var cap = olmoe.Capacity.init(cfg, opts.chunk, opts.capacity, opts.expert_capacity, owner[0..cfg.experts]);
    if (opts.capacity > 0)
    {
        // Each peer's limit is whatever it grants.
        for (eps, 1..) |ep, r|
        {
            const grant = try ep.awaitCapacity();
            cap.rank_limit[r] = if (grant == 0) olmoe.Capacity.none else grant;
            std.debug.print("{s}: capacity: rank {d} grants {d}{s}\n", .{ name, r, grant, if (grant == 0) " (no limit)" else "" });
        }
        std.debug.print("{s}: capacity: {d} expert calls per layer here\n", .{ name, cap.rank_limit[0] });
    }
    if (opts.capacity > 0 or opts.expert_capacity > 0)
    {
        moe.capacity = &cap;
    }
    split_probe = &sm;
    defer split_probe = null;
    try armBudget(opts, name);
    var host_buf: [64]u8 = undefined;
    var rank_names: [split.max_ranks][]const u8 = undefined;
    rank_names[0] = hostName(&host_buf);
    for (names, 1..) |n, r|
    {
        rank_names[r] = n;
    }
    const live: Live = .{
        .owner = owner[0..cfg.experts],
        .ranks = rank_names[0 .. eps.len + 1],
        .chunks = wire.chunkCount(opts.payload, cfg.d_model),
        .probe = .{ .ctx = &sm, .sample = SplitMoe.sample },
    };
    const tel = try openTelemetry(gpa, cfg, moe, opts, name, live);
    const run_moe = if (tel) |t| t.tap.moe() else moe;
    const result = if (opts.serve != null)
        runServe(gpa, &l, run_moe, opts, name, tel.?.bc, opts.max_step)
    else if (opts.context > 0)
        runLong(gpa, &l.weights.?, run_moe, opts, name)
    else
        runGolden(gpa, &l.weights.?, run_moe, opts, name);
    if (tel) |t|
    {
        t.close(name);
    }
    if (result) |_| {} else |err|
    {
        if (err == error.PeerUnhealthy)
        {
            reportRefusal(name, opts.swarm.?, &sm);
            return err;
        }
    }
    for (eps) |ep|
    {
        try ep.finish();
    }
    std.debug.print("{s}: {d} expert calls sent to {d} peer{s}, {d} run locally; payload {t}\n", .{ name, sm.remote(), peers.len, if (peers.len == 1) "" else "s", sm.local_calls, opts.payload });
    if (peers.len > 1)
    {
        for (peers, 1..) |p, r|
        {
            std.debug.print("{s}: rank {d} ({s}): {d} expert calls\n", .{ name, r, rank_names[r], p.calls });
        }
    }
    if (opts.lookahead > 0)
    {
        std.debug.print("{s}: lookahead: {d} prefetch hints sent to the peers\n", .{ name, sm.hints() });
    }
    if (box) |b|
    {
        reportCache(name, b.cache);
    }
    reportGuards(opts, name, moe.capacity, &sm);
    return result;
}

/// Rank 0's split, for the spike test's per-phase stall figures.
var split_probe: ?*SplitMoe = null;

/// `--memory-budget MB` (TASK-17): sets the heap's limit from what this
/// process already holds, and checks memory after every engine step.
fn armBudget(opts: Options, label: []const u8) !void
{
    if (opts.memory_budget == 0)
    {
        return;
    }
    memory = .{ .heap = &proc_heap, .budget = opts.memory_budget, .device = &gpu.systemBytes };
    memory.?.arm() catch |err|
    {
        std.debug.print("{s}: FAIL: the memory budget is {d:.0} MB, but {d:.0} MB is already held (weights, GPU buffers, runtime)\n", .{ label, mb(opts.memory_budget), mb(memory.?.fixed) });
        return err;
    };
    olmoe.after_step = checkMemory;
    std.debug.print("{s}: memory budget {d:.0} MB: {d:.0} MB held at startup (weights, GPU buffers, runtime), so the heap may grow by {d:.0} MB\n", .{ label, mb(opts.memory_budget), mb(memory.?.fixed), mb(memory.?.headroom()) });
}

fn checkMemory() void
{
    if (memory) |*m|
    {
        m.check();
    }
}

/// Prints the heap's balance at exit, and how close memory came to the budget.
fn reportHeap() void
{
    const live = proc_heap.live.load(.monotonic);
    std.debug.print("{s}: heap: {d} allocations, {d} frees, peak {d:.1} MB, {d} bytes still allocated at exit{s}\n", .{
        heap_label,
        proc_heap.allocs.load(.monotonic),
        proc_heap.frees.load(.monotonic),
        mb(proc_heap.peak.load(.monotonic)),
        live,
        if (live == 0) " (balanced)" else "",
    });
    if (memory) |*m|
    {
        m.check();
        std.debug.print("{s}: memory budget {d:.0} MB: at most {d:.0} MB resident plus GPU in {d} checks, {d} over budget; {d} allocations refused; peak resident {d:.0} MB\n", .{
            heap_label, mb(m.budget), mb(m.max_seen), m.checks, m.over, proc_heap.refused.load(.monotonic), mb(budget.peakResidentBytes()),
        });
    }
}

/// Capacity drops and, for a split, rank 0's longest wait on its peers.
fn reportGuards(opts: Options, name: []const u8, cap: ?*olmoe.Capacity, sm: ?*SplitMoe) void
{
    if (opts.capacity == 0 and opts.expert_capacity == 0 and opts.memory_budget == 0 and opts.spike == 0)
    {
        return;
    }
    if (cap) |cp|
    {
        var bufs: [2][32]u8 = undefined;
        std.debug.print("{s}: capacity over {d}-token steps: calls per layer at most {s} here, per expert {s}; dropped {d} of {d} calls ({d:.3}%) in {d} layer passes\n", .{
            name,
            opts.chunk,
            limitText(&bufs[0], cp.rank_limit[0]),
            limitText(&bufs[1], cp.expert_limit),
            cp.dropped,
            cp.routed,
            pct(cp.dropped, cp.routed),
            cp.layers_over,
        });
        if (sm) |s|
        {
            for (1..s.peers.len + 1) |r|
            {
                var buf: [32]u8 = undefined;
                std.debug.print("{s}: capacity: at most {s} calls per layer on rank {d}\n", .{ name, limitText(&buf, cp.rank_limit[r]), r });
            }
        }
    }
    if (sm) |s|
    {
        std.debug.print("{s}: longest wait on the peers with nothing moving {d:.2} ms; {d} sends waited for a peer to grant ring slots\n", .{
            name, @as(f64, @floatFromInt(s.stall_max_ns)) / 1e6, s.credit_waits,
        });
    }
}

fn limitText(buf: *[32]u8, limit: u32) []const u8
{
    if (limit == olmoe.Capacity.none)
    {
        return "no limit";
    }
    return std.fmt.bufPrint(buf, "{d}", .{limit}) catch "?";
}

fn pct(a: u64, b: u64) f64
{
    return 100 * @as(f64, @floatFromInt(a)) / @max(@as(f64, @floatFromInt(b)), 1);
}

/// Says why rank 0 stopped routing: what the tracker reports for each peer.
fn reportRefusal(name: []const u8, sw: *const Swarm, sm: *const SplitMoe) void
{
    var v: tracker.View = .{};
    sw.client.snapshot(&v);
    var t: [16]u8 = undefined;
    for (sw.peers, sm.peers, 1..) |peer, p, r|
    {
        const e = v.find(peer.id);
        if (!sw.client.lost.load(.acquire) and e != null and e.?.health == .healthy)
        {
            continue;
        }
        const why = if (sw.client.lost.load(.acquire)) "the tracker connection was lost" else if (e) |x| @tagName(x.health) else "missing from the map";
        std.debug.print("{s} {s}: refusing to route to rank {d} (tracker id {d}, pid {d} on {s}): {s} in map v{d}, received {d:.0} ms ago; {d} expert calls had been sent to it\n", .{
            tracker.stamp(&t), name, r, peer.id, peer.info.pid, peer.info.nodeName(), why, v.version, @as(f64, @floatFromInt(tracker.monoNs() - v.at_ns)) / 1e6, p.calls,
        });
    }
}

/// One process-to-process link (TASK-33): shared memory, TCP, or RDMA bootstrapped
/// over TCP, with its receive ring and endpoint. It must not move once set up.
const Link = struct {
    kind: Kind = .shm,
    ring_arena: ?Arena = null,
    tcp: transport.tcp.Link = undefined,
    rdma: RdmaLink = undefined,
    ep: transport.Endpoint = undefined,

    const Kind = enum { shm, tcp, rdma };
    const RdmaLink = if (build_options.rdma) transport.rdma.Link else void;

    fn initShm(self: *Link, side: *transport.shm.Side, rank: u6, slots: u16) void
    {
        self.* = .{ .kind = .shm };
        self.ep = transport.Endpoint.init(side.backend(), rank, slots);
    }

    fn initSocket(self: *Link, opts: Options, fd: transport.tcp.Socket, rdma: bool, rank: u6, slots: u16) !void
    {
        self.* = .{ .kind = if (rdma) .rdma else .tcp };
        self.ring_arena = try Arena.init(@as(usize, slots) * @sizeOf(wire.Packet));
        errdefer
        {
            self.ring_arena.?.deinit();
            self.ring_arena = null;
        }
        const ring = try self.ring_arena.?.alloc(wire.Packet, slots);
        if (rdma)
        {
            if (!build_options.rdma)
            {
                unreachable;
            }
            self.rdma = try transport.rdma.Link.connect(fd, rank, ring, .{ .device = opts.device, .gid_index = opts.gid });
            self.ep = transport.Endpoint.init(self.rdma.backend(), rank, slots);
            return;
        }
        self.tcp = transport.tcp.Link.init(fd, ring);
        self.ep = transport.Endpoint.init(self.tcp.backend(), rank, slots);
    }

    /// Tears the link down after the goodbye (`ep.finish`), or at once when its
    /// peer turned unhealthy and may never close its side.
    fn close(self: *Link, unhealthy: bool) void
    {
        switch (self.kind)
        {
            .shm => {},
            .tcp =>
            {
                if (unhealthy)
                {
                    transport.tcp.closeSocket(self.tcp.fd);
                }
                else if (self.ep.rank == 0)
                {
                    self.tcp.close();
                }
                else
                {
                    self.tcp.closeAfterPeer();
                }
            },
            .rdma =>
            {
                if (!build_options.rdma)
                {
                    unreachable;
                }
                if (unhealthy)
                {
                    self.rdma.abort();
                }
                else
                {
                    self.rdma.close();
                }
            },
        }
        if (self.ring_arena) |*a|
        {
            a.deinit();
        }
        self.ring_arena = null;
    }
};

/// Runs `rank` over `links` (one per peer for rank 0, the link to rank 0 for an
/// expert), then closes them.
fn runLinks(gpa: std.mem.Allocator, opts: Options, links: []Link, rank: u8, name: []const u8, names: []const []const u8) !void
{
    var eps: [split.max_ranks]*transport.Endpoint = undefined;
    for (links, 0..) |*l, i|
    {
        eps[i] = &l.ep;
    }
    const result = runRank(gpa, opts, eps[0..links.len], rank, name, names);
    const unhealthy = if (result) |_| false else |err| err == error.PeerUnhealthy;
    for (links) |*l|
    {
        l.close(unhealthy);
    }
    return result;
}

/// An expert rank forked by `runForked`: runs, then exits without returning.
fn runChild(gpa: std.mem.Allocator, opts: Options, link: *Link, rank: u8, name: []const u8) noreturn
{
    runLinks(gpa, opts, link[0..1], rank, name, &.{}) catch |err|
    {
        std.debug.print("expert: rank {d}: {t}\n", .{ rank, err });
        c._exit(1);
    };
    reportHeap();
    c._exit(0);
}

/// Every rank on this machine (`--ranks N`, two by default): ranks 1 to N-1 are
/// forks, linked to rank 0 by shared memory or RDMA. Each loads only its own
/// weights after the fork.
fn runForked(gpa: std.mem.Allocator, opts: Options, slots: u16, is_rdma: bool) !void {
    const name: []const u8 = if (is_rdma) "rdma" else "shm";
    const peers = (try ranksOf(gpa, opts)) - 1;
    // On the stack, so the forks inherit no heap allocations they would report as leaks.
    var link_space: [split.max_ranks - 1]Link = undefined;
    const links = link_space[0..peers];
    var pids: [split.max_ranks]c.pid_t = undefined;
    var forked: usize = 0;
    var host_buf: [64]u8 = undefined;
    const host = hostName(&host_buf);
    var names: [split.max_ranks][]const u8 = undefined;
    @memset(names[0..peers], host);
    if (is_rdma)
    {
        if (!build_options.rdma)
        {
            unreachable;
        }
        var listeners: [split.max_ranks]transport.tcp.Listener = undefined;
        for (listeners[0..peers]) |*l|
        {
            l.* = try transport.tcp.listen(0);
        }
        for (0..peers) |i|
        {
            const pid = c.fork();
            if (pid < 0)
            {
                return waitChildren(pids[0..forked], error.ForkFailed);
            }
            if (pid == 0)
            {
                for (listeners[0..peers]) |l|
                {
                    transport.tcp.closeSocket(l.fd);
                }
                const fd = transport.tcp.connect("127.0.0.1", listeners[i].port, 50) catch c._exit(1);
                var link: Link = undefined;
                link.initSocket(opts, fd, true, @intCast(i + 1), slots) catch c._exit(1);
                runChild(gpa, opts, &link, @intCast(i + 1), name);
            }
            pids[i] = pid;
            forked += 1;
        }
        const linked = for (0..peers) |i|
        {
            const fd = transport.tcp.acceptOn(listeners[i].fd) catch |err| break err;
            transport.tcp.closeSocket(listeners[i].fd);
            links[i].initSocket(opts, fd, true, 0, slots) catch |err| break err;
        } else {};
        linked catch |err|
        {
            return waitChildren(pids[0..forked], err);
        };
        return waitChildren(pids[0..forked], runLinks(gpa, opts, links, 0, name, names[0..peers]));
    }
    var pair_space: [split.max_ranks - 1]transport.shm.Pair = undefined;
    const pairs = pair_space[0..peers];
    var made: usize = 0;
    defer for (pairs[0..made]) |*p|
    {
        p.deinit();
    };
    for (pairs) |*p|
    {
        p.* = try transport.shm.Pair.create(slots);
        made += 1;
    }
    for (0..peers) |i|
    {
        const pid = c.fork();
        if (pid < 0)
        {
            return waitChildren(pids[0..forked], error.ForkFailed);
        }
        if (pid == 0)
        {
            var link: Link = undefined;
            link.initShm(pairs[i].side(1), @intCast(i + 1), slots);
            runChild(gpa, opts, &link, @intCast(i + 1), name);
        }
        pids[i] = pid;
        forked += 1;
    }
    for (links, pairs) |*l, *p|
    {
        l.initShm(p.side(0), 0, slots);
    }
    return waitChildren(pids[0..forked], runLinks(gpa, opts, links, 0, name, names[0..peers]));
}

/// Waits for every forked rank, stopping them first if rank 0 failed.
fn waitChildren(pids: []const c.pid_t, result: anyerror!void) !void {
    if (result) |_| {} else |_|
    {
        for (pids) |pid|
        {
            _ = c.kill(pid, c.SIG.TERM);
        }
    }
    var failed = false;
    for (pids) |pid|
    {
        var status: c_int = 0;
        _ = c.waitpid(pid, &status, 0);
        failed = failed or status != 0;
    }
    try result;
    if (failed)
    {
        return error.ExpertFailed;
    }
}

/// Rank 0 dialing each peer: TCP, or RDMA bootstrapped on the socket. Each link
/// has its transport's ring size unless `--slots` sets one, as its expert does.
fn runRouter(gpa: std.mem.Allocator, opts: Options, addrs: []const PeerAddr, names: []const []const u8) !void
{
    const links = try gpa.alloc(Link, addrs.len);
    defer gpa.free(links);
    var rdma_links: usize = 0;
    for (addrs, links, 0..) |a, *l, i|
    {
        const fd = transport.tcp.connect(a.host, a.port, 300) catch |err|
        {
            for (links[0..i]) |*o|
            {
                o.close(true);
            }
            return err;
        };
        var own = opts;
        if (a.device) |d|
        {
            own.device = d;
        }
        l.initSocket(own, fd, a.rdma, 0, opts.slots orelse if (a.rdma) @as(u16, 256) else 64) catch |err|
        {
            transport.tcp.closeSocket(fd);
            for (links[0..i]) |*o|
            {
                o.close(true);
            }
            return err;
        };
        rdma_links += @intFromBool(a.rdma);
    }
    const name: []const u8 = if (rdma_links == addrs.len) "rdma" else if (rdma_links == 0) "tcp" else "tcp+rdma";
    return runLinks(gpa, opts, links, 0, name, names);
}

/// `--serve [IP:]PORT` (TASK-30): answers OpenAI chat completions with this
/// rank's weights and `moe` (local experts or a split over any transport,
/// wrapped by the telemetry tap), keeping one `--ctx` position KV cache whose
/// shared prefix later requests reuse unless `--no-reuse`. Prompts go in steps of
/// `--chunk` tokens, or of `step_limit` when `moe` takes no more at once (a split).
fn runServe(gpa: std.mem.Allocator, l: *Loaded, moe: olmoe.Moe, opts: Options, name: []const u8, bc: *telemetry.Broadcaster, step_limit: ?usize) !void
{
    const spec = opts.serve.?;
    const colon = std.mem.lastIndexOfScalar(u8, spec, ':');
    const addr = try transport.tcp.parseIp4(if (colon) |i| spec[0..i] else "127.0.0.1");
    const port = try std.fmt.parseInt(u16, if (colon) |i| spec[i + 1 ..] else spec, 10);

    var tok = try tokenizer.Tokenizer.fromGguf(gpa, l.state.gguf_header.?);
    defer tok.deinit();
    const format = chat.detect(&tok) orelse {
        std.debug.print("{s}: this model's chat template is not one the server knows (OLMoE's or Qwen3's)\n", .{name});
        return error.UnknownChatTemplate;
    };
    const w = &l.weights.?;
    var cache = try olmoe.KvCache.initAs(gpa, w.cfg, opts.ctx, opts.kv_format);
    defer cache.deinit(gpa);
    const model_id = modelId(opts.model);
    var eng: ServeEngine = .{ .gpa = gpa, .w = w, .moe = moe, .cache = &cache };
    return serve.run(gpa, eng.engine(opts.ctx), &tok, format, .{
        .addr = addr,
        .port = port,
        .model_id = model_id,
        .chunk = @min(opts.chunk, step_limit orelse opts.chunk),
        .reuse = opts.reuse,
        .clock = mf.nowNs,
        .name = name,
        .page = live_page,
        .live = bc,
    });
}

/// The model's file name without its directory and `.gguf`.
fn modelId(path: []const u8) []const u8
{
    const base = if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| path[i + 1 ..] else path;
    return if (std.mem.endsWith(u8, base, ".gguf")) base[0 .. base.len - 5] else base;
}

/// The live feed (TASK-16b) and the recording (TASK-16) around rank 0's experts.
const Telemetry = struct {
    tap: *telemetry.Tap,
    bc: *telemetry.Broadcaster,

    /// Finishes the recording, if any, and says where it went.
    fn close(self: Telemetry, name: []const u8) void
    {
        const bc = self.bc;
        if (!bc.recording)
        {
            return;
        }
        bc.finish();
        std.debug.print("{s}: recorded {d} steps ({d} dropped) to the trace\n", .{ name, bc.steps_written, bc.feed.dropped.load(.monotonic) });
    }
};

/// Sets up the telemetry tap when `--serve` or `--record` asks for it, else returns null.
fn openTelemetry(gpa: std.mem.Allocator, cfg: olmoe.Config, moe: olmoe.Moe, opts: Options, name: []const u8, live: Live) !?Telemetry
{
    if (opts.serve == null and opts.record == null)
    {
        return null;
    }
    // Never freed: the broadcaster's thread reads them until the process exits.
    const feed = try gpa.create(telemetry.Feed);
    feed.* = try telemetry.Feed.init(gpa, cfg.layers, cfg.experts, 256);
    const tap = try gpa.create(telemetry.Tap);
    tap.* = try telemetry.Tap.init(gpa, moe, cfg.top_k, feed, live.probe, mf.nowNs);
    const bc = try gpa.create(telemetry.Broadcaster);
    bc.* = .{
        .gpa = std.heap.c_allocator,
        .feed = feed,
        .topo = .{ .model = modelId(opts.model), .transport = name, .ranks = live.ranks, .owner = live.owner, .chunks = live.chunks },
        .top_k = cfg.top_k,
        .clock = mf.nowNs,
    };
    if (opts.record) |path|
    {
        var buf: [512]u8 = undefined;
        bc.record(try zPrint(&buf, "{s}", .{path}), unixNow()) catch |err|
        {
            std.debug.print("{s}: cannot write the recording {s}\n", .{ name, path });
            return err;
        };
        std.debug.print("{s}: recording every step to {s}\n", .{ name, path });
    }
    try bc.start();
    return .{ .tap = tap, .bc = bc };
}

fn unixNow() i64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @intCast(ts.sec);
}

/// The built-in chat page (TASK-30), embedded from `web/live.html`.
const live_page = @embedFile("live.html");

/// What the live diagram shows about this run.
const Live = struct {
    owner: []const u8,
    ranks: []const []const u8,
    chunks: usize = 0,
    probe: telemetry.Probe = .{},
};

fn hostName(buf: *[64]u8) []const u8
{
    @memset(buf, 0);
    if (c.gethostname(buf, buf.len - 1) != 0)
    {
        return "this machine";
    }
    return std.mem.sliceTo(buf, 0);
}

/// A single process's experts, all loaded, as the live diagram's engine counter.
fn sampleLocal(ctx: *anyopaque, out: *telemetry.Counters) void
{
    const local: *olmoe.LocalExperts = @ptrCast(@alignCast(ctx));
    out.calls[0] = local.calls;
}

/// A single process's expert cache, as the live diagram's engine counter.
fn sampleCache(ctx: *anyopaque, out: *telemetry.Counters) void
{
    const cache: *ExpertCache = @ptrCast(@alignCast(ctx));
    out.calls[0] = cache.stats.calls();
    out.cache = split.cacheCounts(cache);
}

/// One phase of the spike test (TASK-17): before, during, or after the spike.
const SpikePhase = struct {
    steps: usize = 0,
    tokens: usize = 0,
    ns: u64 = 0,
    max_step_ns: u64 = 0,
    routed: u64 = 0,
    dropped: u64 = 0,
    stall_ns: u64 = 0,
    /// Which of the phase's steps (from 1) had the longest wait.
    stall_step: usize = 0,
    credit_waits: u64 = 0,
    max_memory: u64 = 0,

    const Counts = struct {
        routed: u64 = 0,
        dropped: u64 = 0,
        credit_waits: u64 = 0,
    };

    /// Reads the running counts.
    fn sample(moe: olmoe.Moe) Counts
    {
        var out: Counts = .{};
        if (moe.capacity) |cp|
        {
            out.routed = cp.routed;
            out.dropped = cp.dropped;
        }
        if (split_probe) |s|
        {
            out.credit_waits = s.credit_waits;
        }
        return out;
    }

    fn add(self: *SpikePhase, before: Counts, after: Counts, tokens: usize, ns: u64) void
    {
        self.steps += 1;
        self.tokens += tokens;
        self.ns += ns;
        self.max_step_ns = @max(self.max_step_ns, ns);
        self.routed += after.routed - before.routed;
        self.dropped += after.dropped - before.dropped;
        self.credit_waits += after.credit_waits - before.credit_waits;
        if (split_probe) |s|
        {
            if (s.stall_recent_ns > self.stall_ns)
            {
                self.stall_ns = s.stall_recent_ns;
                self.stall_step = self.steps;
            }
            s.stall_recent_ns = 0;
        }
        self.max_memory = @max(self.max_memory, budget.residentBytes() + gpu.systemBytes());
    }

    fn report(self: SpikePhase, name: []const u8, i: usize, size: usize) void
    {
        const names = [_][]const u8{ "before the spike", "during the spike", "after the spike" };
        const secs = @as(f64, @floatFromInt(self.ns)) / 1e9;
        std.debug.print("{s}: {s}: {d} steps of up to {d} tokens, {d} tokens in {d:.2} s ({d:.0} tokens/s), longest step {d:.1} ms; capacity dropped {d} of {d} calls ({d:.3}%); longest wait on the peers {d:.2} ms (in step {d}), {d} sends waited for ring slots; memory at most {d:.0} MB\n", .{
            name,
            names[i],
            self.steps,
            size,
            self.tokens,
            secs,
            @as(f64, @floatFromInt(self.tokens)) / @max(secs, 1e-9),
            @as(f64, @floatFromInt(self.max_step_ns)) / 1e6,
            self.dropped,
            self.routed,
            pct(self.dropped, self.routed),
            @as(f64, @floatFromInt(self.stall_ns)) / 1e6,
            self.stall_step,
            self.credit_waits,
            mb(self.max_memory),
        });
    }
};

/// The model behind the server's engine interface.
const ServeEngine = struct {
    gpa: std.mem.Allocator,
    w: *const olmoe.Weights,
    moe: olmoe.Moe,
    cache: *olmoe.KvCache,

    fn engine(self: *ServeEngine, cap: usize) completion.Engine
    {
        return .{ .ctx = self, .vocab = self.w.cfg.vocab, .cap = cap, .truncate = truncate, .step = step };
    }

    fn truncate(ctx: *anyopaque, len: usize) usize
    {
        const self: *ServeEngine = @ptrCast(@alignCast(ctx));
        return self.cache.truncate(len);
    }

    fn step(ctx: *anyopaque, tokens: []const u32, logits: []f32) anyerror!void
    {
        const self: *ServeEngine = @ptrCast(@alignCast(ctx));
        try olmoe.step(self.gpa, self.w, self.moe, self.cache, tokens, logits, null);
    }
};

/// Long context (TASK-12): `--context N` tokens from `--tokens FILE` (u32
/// ids, repeated if short) are prefilled `--chunk` at a time, then
/// `--generate M` tokens are decoded greedily. Reports time to first token,
/// decode speed, and the memory the KV cache holds; `--kv-offload DIR` keeps
/// the cold middle on NVMe. `--save F` writes the generated tokens and their
/// logits, and `--compare F` checks a run against them bit for bit.
/// `--prefix-cache DIR` (TASK-28) restores the longest saved prefix of the
/// prompt from DIR and saves the prompt's cache there after prefill.
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
    if (cfg.recurrentLayers() > 0 and opts.prefix_cache != null)
    {
        // Saved prefixes do not carry a Gated DeltaNet layer's state yet; the NVMe tier keeps it in memory.
        std.debug.print("{s}: --prefix-cache does not support {s} yet\n", .{ name, modelName(cfg) });
        return error.Unsupported;
    }
    if (opts.kv_offload) |dir|
    {
        const path = try zPrint(&path_buf, "{s}/tf_kv_{d}.bin", .{ dir, c.getpid() });
        tier = try kv_tier.TierCache.open(gpa, cfg, cap, .{ .max_chunk = opts.chunk, .window = opts.kv_window, .path = path, .format = opts.kv_format });
        cache = try olmoe.KvCache.tiered(cfg, cap, opts.kv_format, tier.?.tier());
    }
    else
    {
        cache = try olmoe.KvCache.initAs(gpa, cfg, cap, opts.kv_format);
    }
    defer cache.deinit(gpa);
    const kv_bytes = if (tier) |t| t.memoryBytes() else cache.k.len + cache.v.len;
    const full_bytes = cfg.attnLayers() * cap * cache.positionBytes();
    std.debug.print("{s}: context {d} tokens + {d} generated, chunks of {d}; KV cache {s}, {t}, {d} bytes per token: {d:.2} GB in memory (all of it would be {d:.2} GB), GPU attention {s}\n", .{
        name, n, opts.generate, opts.chunk, if (tier != null) "offloaded to NVMe" else "all in memory", opts.kv_format, cfg.attnLayers() * cache.positionBytes(), @as(f64, @floatFromInt(kv_bytes)) / 1e9, @as(f64, @floatFromInt(full_bytes)) / 1e9, if (cache.dev_k != 0 or tier != null) "on" else "off",
    });

    var key: ?prefix_cache.Key = null;
    var fp_buf: [1024]u8 = undefined;
    if (opts.prefix_cache) |dir|
    {
        if (tier != null)
        {
            std.debug.print("{s}: --prefix-cache needs the in-memory KV cache, not --kv-offload\n", .{name});
            return error.Usage;
        }
        key = .{ .dir = dir, .fingerprint = try fingerprint(&fp_buf, opts), .cfg = cfg, .chunk = opts.chunk };
    }

    const logits = try gpa.alloc(f32, cfg.vocab);
    defer gpa.free(logits);
    const t0 = mf.nowNs();
    var pos: usize = 0;
    var full = false;
    if (key) |k|
    {
        if (try prefix_cache.find(gpa, k, tokens)) |m|
        {
            const tr = try prefix_cache.restore(k, m, &cache, logits);
            pos = m.positions;
            full = m.full;
            std.debug.print("{s}: prefix cache: restored {d} of {d} positions{s} from {s} ({d} saved): {d:.2} GB in {d:.2} s ({d:.2} GB/s)\n", .{
                name,
                m.positions,
                n,
                if (m.full) " and the last logits" else "",
                m.fileName(),
                m.saved,
                @as(f64, @floatFromInt(tr.bytes)) / 1e9,
                @as(f64, @floatFromInt(tr.ns)) / 1e9,
                @as(f64, @floatFromInt(tr.bytes)) / @max(@as(f64, @floatFromInt(tr.ns)), 1),
            });
        }
        else
        {
            std.debug.print("{s}: prefix cache: no saved prefix shares a whole chunk with this prompt\n", .{name});
        }
    }
    var next_report: usize = n / 10;
    var phases: [3]SpikePhase = @splat(.{});
    while (pos < n)
    {
        // `--spike F` (TASK-17): the middle third of the prompt arrives in steps F times as big.
        const phase: usize = if (opts.spike == 0 or pos < n / 3) 0 else if (pos < 2 * n / 3) 1 else 2;
        const e = @min(pos + if (phase == 1) opts.chunk * opts.spike else opts.chunk, n);
        const before = SpikePhase.sample(moe);
        const s0 = mf.nowNs();
        try olmoe.step(gpa, w, moe, &cache, tokens[pos..e], logits, null);
        if (opts.spike > 0)
        {
            phases[phase].add(before, SpikePhase.sample(moe), e - pos, mf.nowNs() - s0);
        }
        pos = e;
        if (pos >= next_report and pos < n)
        {
            std.debug.print("{s}: prefill {d} of {d} positions, {d:.1} s\n", .{ name, pos, n, @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e9 });
            next_report += @max(n / 10, 1);
        }
    }
    const ttft = mf.nowNs() - t0;
    if (opts.spike > 0)
    {
        for (phases, 0..) |p, i|
        {
            p.report(name, i, if (i == 1) opts.chunk * opts.spike else opts.chunk);
        }
    }
    reportProfile(name, "prefill");
    olmoe.profile = .{};
    if (key) |k|
    {
        if (!full)
        {
            const s = try prefix_cache.save(k, tokens, &cache, logits);
            std.debug.print("{s}: prefix cache: saved {d} positions to {s}: {d:.2} GB in {d:.2} s ({d:.2} GB/s)\n", .{
                name,
                n,
                s.fileName(),
                @as(f64, @floatFromInt(s.transfer.bytes)) / 1e9,
                @as(f64, @floatFromInt(s.transfer.ns)) / 1e9,
                @as(f64, @floatFromInt(s.transfer.bytes)) / @max(@as(f64, @floatFromInt(s.transfer.ns)), 1),
            });
        }
    }
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
    std.debug.print("{s}: time to first token {d:.3} s ({d:.0} tokens/s prefill); decode {d:.3} s per token ({d:.2} tokens/s) over {d} steps from position {d}; resident memory {d:.0} MB\n", .{
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
            var worst: f32 = 0;
            const a: []align(1) const f32 = std.mem.bytesAsSlice(f32, want[16 + tb ..]);
            const b: []align(1) const f32 = std.mem.bytesAsSlice(f32, record[16 + tb ..]);
            const ta: []align(1) const u32 = std.mem.bytesAsSlice(u32, want[16..][0..tb]);
            var same: usize = 0;
            while (same < out_tokens.len and ta[same] == out_tokens[same])
            {
                same += 1;
            }
            // Logits stay comparable up to and including the first differing token.
            const vocab = a.len / out_tokens.len;
            for (a[0 .. @min(same + 1, out_tokens.len) * vocab], b[0 .. @min(same + 1, out_tokens.len) * vocab]) |x, y|
            {
                worst = @max(worst, @abs(x - y));
            }
            std.debug.print("{s}: FAIL: generated tokens {s} ({d} of {d} the same before the first difference), logits {s} (by up to {e:.2} through it) compared with {s}\n", .{ name, if (same_tokens) "match" else "differ", same, out_tokens.len, if (same_logits) "match bit for bit" else "differ", worst, path });
            return error.CompareFailed;
        }
        std.debug.print("{s}: PASS: all {d} generated tokens and their logits match {s} bit for bit\n", .{ name, opts.generate, path });
    }
}

/// Everything besides the tokens that decides a saved prefix's bits (TASK-28).
fn fingerprint(buf: []u8, opts: Options) ![]const u8
{
    var path_buf: [512]u8 = undefined;
    const path = try zPrint(&path_buf, "{s}", .{opts.model});
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0)
    {
        return error.ModelNotFound;
    }
    const size = c.lseek(fd, 0, c.SEEK.END);
    _ = c.close(fd);
    const att = if (c.getenv("TF_ATT_KERNEL")) |v| std.mem.sliceTo(v, 0) else "0";
    const mm = if (c.getenv("TF_MM_KERNEL")) |v| std.mem.sliceTo(v, 0) else "0";
    const base = if (std.mem.lastIndexOfScalar(u8, opts.model, '/')) |i| opts.model[i + 1 ..] else opts.model;
    return std.fmt.bufPrint(buf, "engine {s}; model {s} ({d} bytes); yarn {d}; gpu {}; TF_ATT_KERNEL {s}; TF_MM_KERNEL {s}; kv {t}", .{
        build_options.engine_build, base, size, yarn_factor, opts.gpu, att, mm, opts.kv_format,
    });
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
    std.debug.print("{s}: {s} time: projections {d:.1} s, QK norm and rope {d:.1} s, attention {d:.1} s, Gated DeltaNet {d:.1} s, output {d:.1} s, router {d:.1} s, experts {d:.1} s, LM head {d:.1} s\n", .{
        name, what, s(p.projections), s(p.qk_rope), s(p.attention), s(p.recurrent), s(p.output), s(p.router), s(p.experts), s(p.head),
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
        .qwen35moe => "Qwen3.5-MoE",
        .qwen4exp => "Qwen3.8-Flash-Next",
        .glm5next => "GLM-5.3-Flash",
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
        // A long prompt's golden keeps no hidden states (TASK-31), so none are traced for it.
        const has_hidden = if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer0.hidden", .{p}))) |_| true else |_| false;
        const R = cfg.residual();
        // QSA's blocks (TASK-31) or DSA's pools (TASK-32): kept per query, compared the same way.
        const top = if (cfg.qsa_heads > 0) cfg.qsaTop() else if (cfg.dsa_heads > 0) cfg.dsaTop() else 0;
        const dsa = cfg.dsa_heads > 0;
        const trace: olmoe.Trace = .{
            .hidden = try gpa.alloc(f32, if (has_hidden) cfg.layers * n * R else 0),
            .topk = try gpa.alloc(u32, cfg.layers * n * cfg.top_k),
            .qsa = try gpa.alloc(i32, cfg.attnLayers() * n * top),
        };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        defer gpa.free(trace.qsa);

        const t0 = mf.nowNs();
        try olmoe.forwardAs(gpa, w, moe, tokens, logits, trace, opts.kv_format);
        const ms = @as(f64, @floatFromInt(mf.nowNs() - t0)) / 1e6;
        if (moe.ahead != null)
        {
            const plain = try gpa.alloc(f32, n * cfg.vocab);
            defer gpa.free(plain);
            try olmoe.forwardAs(gpa, w, withoutLookahead(moe), tokens, plain, null, opts.kv_format);
            same_bits = same_bits and std.mem.eql(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(logits));
            compared += plain.len;
        }

        var hidden_err: f32 = 0;
        var routing_same: usize = 0;
        var routing_total: usize = 0;
        var blocks_same: usize = 0;
        var blocks_total: usize = 0;
        // Whether the first QSA layer that chose blocks chose exactly transformers' blocks.
        var first_exact: ?bool = null;
        for (0..cfg.layers) |l| {
            if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))) |gh| {
                hidden_err = @max(hidden_err, maxErr(trace.hidden[l * n * R ..][0 .. n * R], gh.f32s()));
            } else |_| {}
            // QSA's kept blocks, for the queries with more complete blocks than its budget (TASK-31).
            if (top > 0 and !cfg.recurrent(l))
            {
                if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.{s}", .{ p, l, if (dsa) "dsa_from" else "qsa_from" }))) |gf|
                {
                    const from: usize = @intCast(gf.i32s()[0]);
                    const gb = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.{s}", .{ p, l, if (dsa) "dsa_pools" else "qsa_blocks" }))).i32s();
                    const kept = trace.qsa[cfg.attnIndex(l) * n * top ..][0 .. n * top];
                    // Blocks both kept, per query; the lists are ascending, so one swap would shift every later position.
                    var same: usize = 0;
                    var row: usize = 0;
                    while (row * top < gb.len) : (row += 1)
                    {
                        const mine = kept[(from + row) * top ..][0..top];
                        const ref = gb[row * top ..][0..top];
                        var i: usize = 0;
                        var j: usize = 0;
                        while (i < top and j < top)
                        {
                            if (mine[i] == ref[j])
                            {
                                same += 1;
                                i += 1;
                                j += 1;
                            }
                            else if (mine[i] < ref[j])
                            {
                                i += 1;
                            }
                            else
                            {
                                j += 1;
                            }
                        }
                    }
                    blocks_same += same;
                    blocks_total += gb.len;
                    if (first_exact == null)
                    {
                        first_exact = same == gb.len;
                    }
                    if (same != gb.len)
                    {
                        std.debug.print("{s}: prompt {d}, layer {d}: {s} {d}/{d} same\n", .{ name, p, l, if (dsa) "DSA pools" else "QSA blocks", same, gb.len });
                    }
                } else |_| {}
            }
            // Dense leading layers (TASK-32) route nothing.
            if (l < cfg.dense_layers)
            {
                continue;
            }
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
        // A long prompt's golden keeps only its last positions' logits.
        const lfrom: usize = if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits_from", .{p}))) |lf| @intCast(lf.i32s()[0]) else |_| 0;
        const ln = n - lfrom;
        const mine = logits[lfrom * cfg.vocab ..][0 .. ln * cfg.vocab];
        const lerr = maxErr(mine, gl);
        var top1: usize = 0;
        for (0..ln) |t| top1 += @intFromBool(olmoe.argmax(mine[t * cfg.vocab ..][0..cfg.vocab]) == olmoe.argmax(gl[t * cfg.vocab ..][0..cfg.vocab]));
        // A prompt past QSA's budget (its golden keeps only the last logits) amplifies rounding through
        // the layers, so it is held to shares, as transformers is against itself (TASK-31).
        const long = lfrom > 0;
        const share = struct {
            fn f(same: usize, total: usize) f64
            {
                return if (total == 0) 1 else @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt(total));
            }
        }.f;
        const pass = if (long)
            top1 == ln and (first_exact orelse true) and share(blocks_same, blocks_total) >= 0.999 and share(routing_same, routing_total) >= 0.995
        else
            top1 == ln and lerr <= 1e-2 and blocks_same == blocks_total;
        if (!long)
        {
            worst_logit = @max(worst_logit, lerr);
        }
        ok = ok and pass;
        const unit = if (dsa) "DSA pools" else "QSA blocks";
        std.debug.print("{s}: prompt {d} ({d} tokens, {d:.0} ms): logits max err {e:.2} (last {d} positions{s}), top-1 {d}/{d}, hidden max err {e:.2}, experts chosen {d}/{d} same ({d:.3}%), {s} {d}/{d} same ({d:.3}%{s}) {s}\n", .{
            name,                                 p,                               n,                  ms,          lerr, ln,
            if (long) ", not held to 1e-2" else "", top1,                          ln,                 hidden_err,  routing_same,
            routing_total,                        100 * share(routing_same, routing_total), unit, blocks_same, blocks_total, 100 * share(blocks_same, blocks_total),
            if (first_exact orelse true) "" else ", first selecting layer differs", if (pass) "ok" else "FAIL",
        });
    }
    if (cfg.qsa_heads > 0)
    {
        std.debug.print("{s}: QSA layer passes that chose blocks: {d} on the GPU, {d} on the CPU\n", .{ name, olmoe.profile.qsa_gpu, olmoe.profile.qsa_cpu });
    }
    if (!ok and opts.kv_format != .f32)
    {
        std.debug.print("{s}: {t} KV cache: logits differ by up to {e:.2}; reported, not held to 1e-2\n", .{ name, opts.kv_format, worst_logit });
    }
    else if (!ok) {
        std.debug.print("{s}: FAIL: a short prompt's logits differ by up to {e:.2} (target 1e-2) or a QSA block differs, a top-1 token differs, or a long prompt shares too few choices\n", .{ name, worst_logit });
        return error.ForwardMismatch;
    }
    else
    {
        std.debug.print("{s}: PASS: {s} logits within 1e-2 of the PyTorch reference (worst {e:.2}), every top-1 token identical\n", .{ name, modelName(cfg), worst_logit });
    }
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
        var cache = try olmoe.KvCache.initAs(gpa, cfg, prompt.len + opts.generate, opts.kv_format);
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
        // Each step's smallest router gap in transformers (TASK-31): where its k-th and (k+1)-th experts
        // are closer than rounding, either choice is right. A step there that misses 1e-2 took the other
        // expert, which carries into later steps' state, so as after a differing token, logits are held
        // only up to it.
        const gaps: []align(1) const f32 = if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_gaps", .{p}))) |gg| gg.f32s() else |_| &.{};
        const near_tie = 1e-6;
        var lerr: f32 = 0;
        var worst: usize = 0;
        var excused: usize = 0;
        for (0..steps) |st|
        {
            const e = maxErr(step_logits[st * cfg.vocab ..][0..cfg.vocab], want_logits[st * cfg.vocab ..][0..cfg.vocab]);
            if (e > 1e-2 and st < gaps.len and gaps[st] < near_tie)
            {
                std.debug.print("{s}:   step {d}: logits max err {e:.2} at a router near-tie (gap {e:.2} in transformers); logits held up to here\n", .{ name, st, e, gaps[st] });
                excused += 1;
                break;
            }
            if (e > lerr)
            {
                lerr = e;
                worst = st;
            }
        }
        // A prompt past QSA's budget is held to its tokens only, as in runGolden (TASK-31).
        const long = if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits_from", .{p}))) |_| true else |_| false;
        const pass = same == g and (long or lerr <= 1e-2);
        ok = ok and pass;
        std.debug.print("{s}:   same as transformers' generate(): {d}/{d} tokens, logits max err {e:.2} at step {d}{s}{s} {s}\n", .{
            name,                                                  same, g, lerr, worst, if (long) ", not held to 1e-2" else "",
            if (excused > 0) " (before the near-tie)" else "", if (pass) "ok" else "FAIL",
        });
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
    if (!ok and opts.kv_format != .f32)
    {
        std.debug.print("{s}: {t} KV cache: generated tokens or their logits differ from transformers' generate(); reported, not held to it\n", .{ name, opts.kv_format });
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
    var cache = try olmoe.KvCache.initAs(gpa, cfg, longest + n_new, opts.kv_format);
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
