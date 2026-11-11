//! OLMoE forward pass on CPU (TASK-10), matching Hugging Face's
//! OlmoeForCausalLM: per layer, pre-norm attention with RMSNorm on the full
//! query and key projections (QK norm), rotary embeddings in the rotate-half
//! (NeoX) layout, causal softmax attention; then a pre-norm sparse MoE block
//! that takes the softmax over all experts, keeps the top k without
//! renormalizing, and sums the chosen SwiGLU experts' outputs weighted by
//! those probabilities. Final RMSNorm, then the LM head.
//!
//! Qwen3-MoE (TASK-12, matching Qwen3MoeForCausalLM) differs in three places:
//! keys and values have fewer heads than queries (each serves a group of
//! query heads), QK norm applies to each head on its own, and the top-k
//! probabilities are renormalized to sum to 1. It can also use YaRN rope
//! scaling (`Rope`) to run past its native context.
//!
//! Weights stay in their GGUF block format and are dequantized one row at a
//! time. The MoE block calls experts through `Moe`, so a process can run the
//! experts it owns and send the rest to a peer (`olmoe_run.zig`).
const std = @import("std");
const quant = @import("quant.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");

/// The two model families the engine runs.
pub const Arch = enum {
    /// OLMoE: QK norm over the whole projection, top-k probabilities used as is.
    olmoe,
    /// Qwen3-MoE (TASK-12): grouped KV heads, QK norm per head, top-k renormalized.
    qwen3moe,
};

pub const Config = struct {
    arch: Arch = .olmoe,
    layers: usize,
    d_model: usize,
    heads: usize,
    /// Key and value heads; each serves `heads / kv_heads` query heads.
    kv_heads: usize,
    head_dim: usize,
    d_ff: usize,
    experts: usize,
    top_k: usize,
    vocab: usize,
    eps: f32,
    rope_theta: f32,
    /// YaRN rope scaling (TASK-12): 0 for none, else the factor over `rope_original` positions.
    yarn: f32 = 0,
    rope_original: usize = 0,

    pub fn qDim(self: Config) usize
    {
        return self.heads * self.head_dim;
    }

    pub fn kvDim(self: Config) usize
    {
        return self.kv_heads * self.head_dim;
    }

    pub fn fromGguf(h: gguf.Header) !Config {
        const name = h.getString("general.architecture") orelse return error.NotOlmoe;
        const arch = std.meta.stringToEnum(Arch, name) orelse return error.NotOlmoe;
        const Keys = struct {
            hh: gguf.Header,
            prefix: []const u8,
            buf: [96]u8 = undefined,

            fn key(s: *@This(), k: []const u8) []const u8
            {
                return std.fmt.bufPrint(&s.buf, "{s}.{s}", .{ s.prefix, k }) catch unreachable;
            }

            fn int(s: *@This(), k: []const u8) !usize
            {
                return @intCast(s.hh.getInt(s.key(k)) orelse
                {
                    std.log.err("missing GGUF key {s}", .{s.key(k)});
                    return error.MissingKey;
                });
            }

            fn opt(s: *@This(), k: []const u8) ?usize
            {
                return if (s.hh.getInt(s.key(k))) |v| @intCast(v) else null;
            }

            fn float(s: *@This(), k: []const u8, default: f32) f32
            {
                const v = s.hh.get(s.key(k)) orelse return default;
                return switch (v) {
                    .float => |x| @floatCast(x),
                    else => default,
                };
            }
        };
        var g: Keys = .{ .hh = h, .prefix = name };
        const d = try g.int("embedding_length");
        const heads = try g.int("attention.head_count");
        const kv_heads = g.opt("attention.head_count_kv") orelse heads;
        if (kv_heads == 0 or heads % kv_heads != 0)
        {
            return error.Unsupported;
        }
        const embd = h.tensor("token_embd.weight") orelse return error.MissingTensor;
        return .{
            .arch = arch,
            .layers = try g.int("block_count"),
            .d_model = d,
            .heads = heads,
            .kv_heads = kv_heads,
            .head_dim = g.opt("attention.key_length") orelse d / heads,
            .d_ff = g.opt("expert_feed_forward_length") orelse try g.int("feed_forward_length"),
            .experts = try g.int("expert_count"),
            .top_k = try g.int("expert_used_count"),
            .vocab = @intCast(embd.dims[1]),
            .eps = g.float("attention.layer_norm_rms_epsilon", 1e-5),
            .rope_theta = g.float("rope.freq_base", 10000),
        };
    }

    /// Turns on YaRN with `factor` over `original` positions (Qwen uses 4 over 32,768 for 128k).
    pub fn withYarn(self: Config, factor: f32, original: usize) Config
    {
        var c = self;
        c.yarn = factor;
        c.rope_original = original;
        return c;
    }
};

/// Set by a caller with a GPU (`gpu.zig`): runs a matmul there and returns
/// true, or returns false (unsupported format, say) to use the CPU path.
pub var gpu_matmul: ?*const fn (m: Mat, x: []const f32, n: usize, y: []f32) bool = null;

/// Set with a GPU (TASK-12): pins `buf` for kernels to read in place and
/// returns its device address, or null to leave attention on the CPU.
pub var gpu_register: ?*const fn (buf: []u8) ?u64 = null;
pub var gpu_unregister: ?*const fn (buf: []u8) void = null;

/// Set with a GPU (TASK-12): runs a group of experts there in one call, each
/// on its own rows of `x`, and returns true, or false to use the CPU path.
pub var gpu_experts: ?*const fn (job: ExpertJob) bool = null;

/// Expert `experts[e]` takes rows `first[e]` to `first[e + 1] - 1` of `x`
/// (rows x d_model) and writes the same rows of `y`.
pub const ExpertJob = struct {
    experts: []const Expert,
    first: []const u32,
    x: []const f32,
    y: []f32,
};

/// Set with a GPU (TASK-12): runs one layer's attention there and returns
/// true, or false to use the CPU path.
pub var gpu_attention: ?*const fn (job: AttentionJob) bool = null;

/// Time spent in each part of `step`, summed over every call (TASK-12).
pub const Profile = struct {
    projections: u64 = 0,
    qk_rope: u64 = 0,
    attention: u64 = 0,
    output: u64 = 0,
    router: u64 = 0,
    experts: u64 = 0,
    head: u64 = 0,

    /// Adds the time since `t` to `field` and moves `t` on.
    fn lap(field: *u64, t: *u64) void
    {
        const now = nowNs();
        field.* += now - t.*;
        t.* = now;
    }
};

pub var profile: Profile = .{};

/// One layer's causal attention for `n` new tokens at positions `pos0` on,
/// reading keys and values through tiles of `tile_positions` positions.
pub const AttentionJob = struct {
    q: []const f32,
    n: usize,
    pos0: usize,
    heads: usize,
    kv_heads: usize,
    kvd: usize,
    scale: f32,
    /// Device addresses of each tile's first key and first value.
    tiles_k: []const u64,
    tiles_v: []const u64,
    splits: usize,
    out: []f32,

    pub const tile_positions = 32;
    pub const head_dim = 128;

    /// How many blocks share the tiles: enough to fill the GPU when there
    /// are few query rows (decode), and a function of the shape only, so
    /// any two runs of the same shape add up in the same order.
    pub fn splitsFor(n: usize, heads: usize, kv_heads: usize, tiles: usize) usize
    {
        const rows = n * (heads / kv_heads);
        const blocks = (rows + 31) / 32 * kv_heads;
        const want = (128 + blocks - 1) / blocks;
        return std.math.clamp(@min(want, tiles / 8), 1, 256);
    }
};

/// A row-major weight in a ggml block format: `rows` rows of `cols` values.
pub const Mat = struct {
    format: quant.Format,
    rows: usize,
    cols: usize,
    bytes: []const u8,

    pub fn rowBytes(self: Mat) usize {
        return self.cols / self.format.blockValues() * self.format.blockBytes();
    }

    /// One row, dequantized into `out` (`cols` values).
    pub fn row(self: Mat, r: usize, out: []f32) void {
        const rb = self.rowBytes();
        quant.dequantize(self.format, self.bytes[r * rb ..][0..rb], out[0..self.cols]) catch unreachable;
    }

    /// y[t][r] = sum over c of W[r][c] * x[t][c], for `n` inputs (x is n x cols,
    /// y is n x rows). Each row is dequantized once and used for every input.
    pub fn matmul(self: Mat, x: []const f32, n: usize, y: []f32, scratch: []f32) void {
        std.debug.assert(x.len >= n * self.cols and y.len >= n * self.rows and scratch.len >= self.cols);
        if (gpu_matmul) |f| if (f(self, x[0 .. n * self.cols], n, y[0 .. n * self.rows])) return;
        const threads = cpuThreads();
        if (threads > 1 and self.rows >= 2 * threads and self.rows * n * self.cols >= parallel_min and self.cols <= max_parallel_cols)
        {
            return self.matmulParallel(x, n, y, threads);
        }
        self.matmulRows(x, n, y, scratch, 0, self.rows);
    }

    /// Below this many multiply-adds a matmul stays on one thread.
    pub var parallel_min: usize = 1 << 21;
    const max_parallel_cols = 16384;

    /// Splits the rows among threads. Each row is computed exactly as on one
    /// thread, so the result does not change.
    fn matmulParallel(self: Mat, x: []const f32, n: usize, y: []f32, threads: usize) void
    {
        const Worker = struct {
            fn run(m: Mat, xx: []const f32, nn: usize, yy: []f32, first: usize, end: usize) void
            {
                var buf: [max_parallel_cols]f32 = undefined;
                m.matmulRows(xx, nn, yy, &buf, first, end);
            }
        };
        var pool: [max_threads]?std.Thread = @splat(null);
        const per = (self.rows + threads - 1) / threads;
        var first: usize = per;
        for (pool[1..threads]) |*t|
        {
            if (first >= self.rows)
            {
                break;
            }
            const end = @min(first + per, self.rows);
            t.* = std.Thread.spawn(.{}, Worker.run, .{ self, x, n, y, first, end }) catch null;
            if (t.* == null)
            {
                Worker.run(self, x, n, y, first, end);
            }
            first = end;
        }
        Worker.run(self, x, n, y, 0, @min(per, self.rows));
        for (pool[1..threads]) |t|
        {
            if (t) |th|
            {
                th.join();
            }
        }
    }

    /// Rows `first` to `end` - 1 of y = W x.
    fn matmulRows(self: Mat, x: []const f32, n: usize, y: []f32, scratch: []f32, first: usize, end: usize) void
    {
        if (self.format == .f32 and @intFromPtr(self.bytes.ptr) % @alignOf(f32) == 0)
        {
            // Rows are already f32 (the router, say): read them in place, same sums.
            const wv: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, self.bytes[0 .. self.rows * self.cols * 4]));
            // Inputs in blocks that stay in cache while every row passes over them.
            const block = @max(1, (256 * 1024) / (self.cols * @sizeOf(f32)));
            var t0: usize = 0;
            while (t0 < n) : (t0 += block)
            {
                const t1 = @min(t0 + block, n);
                for (first..end) |r|
                {
                    for (t0..t1) |t|
                    {
                        y[t * self.rows + r] = dot(wv[r * self.cols ..][0..self.cols], x[t * self.cols ..][0..self.cols]);
                    }
                }
            }
            return;
        }
        for (first..end) |r|
        {
            self.row(r, scratch);
            for (0..n) |t|
            {
                y[t * self.rows + r] = dot(scratch[0..self.cols], x[t * self.cols ..][0..self.cols]);
            }
        }
    }
};

const max_threads = 32;
var thread_count = std.atomic.Value(usize).init(0);

/// Threads for CPU matmuls: TF_THREADS, else the CPU count, at most 32.
pub fn cpuThreads() usize
{
    const cached = thread_count.load(.monotonic);
    if (cached != 0)
    {
        return cached;
    }
    var count: usize = std.Thread.getCpuCount() catch 1;
    if (@import("builtin").os.tag != .windows)
    {
        if (std.c.getenv("TF_THREADS")) |v|
        {
            count = std.fmt.parseInt(usize, std.mem.sliceTo(v, 0), 10) catch count;
        }
    }
    count = std.math.clamp(count, 1, max_threads);
    thread_count.store(count, .monotonic);
    return count;
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    const V = @Vector(16, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= a.len) : (i += 16) {
        const va: V = a[i..][0..16].*;
        const vb: V = b[i..][0..16].*;
        acc += va * vb;
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

pub fn rmsnorm(x: []const f32, w: []const f32, eps: f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out[0..x.len], x, w) |*o, v, g| o.* = v * inv * g;
}

/// QK norm in place on one token's projection `v`: over all of it (OLMoE)
/// or over each head (Qwen3). `tmp` holds at least `v.len` values.
fn qkNorm(cfg: Config, v: []f32, w: []const f32, tmp: []f32) void
{
    @memcpy(tmp[0..v.len], v);
    if (cfg.arch == .olmoe)
    {
        rmsnorm(tmp[0..v.len], w, cfg.eps, v);
        return;
    }
    var off: usize = 0;
    while (off < v.len) : (off += cfg.head_dim)
    {
        rmsnorm(tmp[off..][0..cfg.head_dim], w, cfg.eps, v[off..][0..cfg.head_dim]);
    }
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Rotary frequencies for one head (`head_dim / 2` of them) and the factor
/// cos and sin are scaled by, as transformers computes them: plain, or YaRN.
pub const Rope = struct {
    inv_freq: [max_half]f32,
    half: usize,
    scale: f32,

    pub const max_half = 256;

    pub fn init(cfg: Config) Rope
    {
        const dim = cfg.head_dim;
        var r: Rope = .{ .inv_freq = undefined, .half = dim / 2, .scale = 1 };
        const fdim: f32 = @floatFromInt(dim);
        for (0..r.half) |i|
        {
            r.inv_freq[i] = 1.0 / std.math.pow(f32, cfg.rope_theta, @as(f32, @floatFromInt(2 * i)) / fdim);
        }
        if (cfg.yarn <= 1)
        {
            return r;
        }
        // Low frequencies are interpolated by the factor, high ones kept, with a linear ramp between.
        const f = cfg.yarn;
        const correction = struct {
            fn dimFor(rotations: f64, d: f64, base: f64, positions: f64) f64
            {
                return d * @log(positions / (rotations * 2 * std.math.pi)) / (2 * @log(base));
            }
        }.dimFor;
        const orig: f64 = @floatFromInt(cfg.rope_original);
        var low = @floor(correction(32, fdim, cfg.rope_theta, orig));
        var high = @ceil(correction(1, fdim, cfg.rope_theta, orig));
        low = @max(low, 0);
        high = @min(high, fdim - 1);
        if (low == high)
        {
            high += 0.001;
        }
        for (0..r.half) |i|
        {
            const ramp: f32 = @floatCast(std.math.clamp((@as(f64, @floatFromInt(i)) - low) / (high - low), 0, 1));
            const extrapolate = 1 - ramp;
            const interp = r.inv_freq[i] / f;
            r.inv_freq[i] = interp * (1 - extrapolate) + r.inv_freq[i] * extrapolate;
        }
        r.scale = 0.1 * @log(f) + 1.0;
        return r;
    }

    /// Rotates each `head_dim` slice of `x` (one token, all heads) in place:
    /// pairs are (i, i + head_dim / 2), as in rotate_half.
    pub fn apply(self: *const Rope, x: []f32, heads: usize, head_dim: usize, pos: usize) void
    {
        for (0..self.half) |i|
        {
            const angle = @as(f32, @floatFromInt(pos)) * self.inv_freq[i];
            const cs = @cos(angle) * self.scale;
            const sn = @sin(angle) * self.scale;
            for (0..heads) |h|
            {
                const v = x[h * head_dim ..];
                const a = v[i];
                const b = v[i + self.half];
                v[i] = a * cs - b * sn;
                v[i + self.half] = b * cs + a * sn;
            }
        }
    }
};

/// One expert's SwiGLU: down(silu(gate x) * up x), for one token.
pub const Expert = struct {
    gate: Mat,
    up: Mat,
    down: Mat,

    /// `scratch` needs 3 * d_ff + max(d_model, d_ff) values.
    pub fn forward(self: Expert, x: []const f32, out: []f32, scratch: []f32) void {
        if (gpu_experts) |run|
        {
            const first = [_]u32{ 0, 1 };
            if (run(.{ .experts = &.{self}, .first = &first, .x = x[0..self.gate.cols], .y = out[0..self.down.rows] }))
            {
                return;
            }
        }
        const f = self.gate.rows;
        const g = scratch[0..f];
        const u = scratch[f..][0..f];
        const row = scratch[2 * f ..][0..@max(self.gate.cols, f)];
        self.gate.matmul(x, 1, g, row);
        self.up.matmul(x, 1, u, row);
        for (g, u) |*a, b| a.* = silu(a.*) * b;
        self.down.matmul(g, 1, out, row);
    }
};

/// How the MoE block reaches its experts.
pub const Moe = struct {
    ctx: *anyopaque,
    /// For each of the `n` rows of `x` (normalized hidden states, n x d_model)
    /// adds sum over j of w[t][j] * expert idx[t][j] (x[t]) to out[t] (zeroed
    /// by the caller). idx and w are n x top_k.
    run: *const fn (ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void,
    /// If set, guesses the experts of later layers while the forward pass runs.
    ahead: ?*Lookahead = null,
    /// Called with each guess (TASK-11), before `layer` starts. It may start
    /// loading those experts but must not change any result.
    prefetch: ?*const fn (ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void = null,
};

/// The lookahead predictor (TASK-11). After layer S's attention, it applies
/// each later layer T's own MoE norm and router to S's hidden state and keeps
/// the top_k + extra experts per token as T's guess. It needs no training,
/// and the real router still picks the experts that run.
pub const Lookahead = struct {
    /// How many layers ahead to guess: 1 or 2. Layers before that are
    /// guessed from the token embeddings, as early as they can be.
    horizon: usize = 2,
    /// Candidates beyond top_k in each guess.
    extra: usize = 2,
    /// By how far ahead each guess was made: true experts found among the
    /// guesses, and true experts.
    hits: [max_horizon]u64 = @splat(0),
    total: [max_horizon]u64 = @splat(0),
    prefetches: u64 = 0,
    /// Time spent guessing, and in whole layers, for the overhead figure.
    predict_ns: u64 = 0,
    layer_ns: u64 = 0,
    /// Share of guesses replaced with wrong experts, to test miss handling
    /// (TASK-14), and how many were replaced.
    mispredict: f32 = 0,
    replaced: u64 = 0,
    rng: u64 = 0x9E3779B97F4A7C15,

    pub const max_horizon = 2;

    /// A fresh count with the same settings.
    pub fn restart(self: *Lookahead) void
    {
        self.* = .{ .horizon = self.horizon, .extra = self.extra, .mispredict = self.mispredict, .rng = self.rng };
    }

    /// Replaces about a `mispredict` share of one token's guesses with
    /// experts it does not hold.
    fn corrupt(self: *Lookahead, row: []u32, experts: usize) void
    {
        for (row) |*e|
        {
            if (self.uniform() >= self.mispredict)
            {
                continue;
            }
            while (true)
            {
                const r: u32 = @intCast(self.random() % experts);
                if (std.mem.indexOfScalar(u32, row, r) == null)
                {
                    e.* = r;
                    break;
                }
            }
            self.replaced += 1;
        }
    }

    fn random(self: *Lookahead) u64
    {
        self.rng +%= 0x9E3779B97F4A7C15;
        var z = self.rng;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    fn uniform(self: *Lookahead) f32
    {
        return @as(f32, @floatFromInt(self.random() >> 40)) / 16777216.0;
    }

    pub fn recall(self: Lookahead, h: usize) f64
    {
        if (self.total[h - 1] == 0)
        {
            return 0;
        }
        return @as(f64, @floatFromInt(self.hits[h - 1])) / @as(f64, @floatFromInt(self.total[h - 1]));
    }

    pub fn overhead(self: Lookahead) f64
    {
        if (self.layer_ns == 0)
        {
            return 0;
        }
        return @as(f64, @floatFromInt(self.predict_ns)) / @as(f64, @floatFromInt(self.layer_ns));
    }
};

pub const Layer = struct {
    attn_norm: []f32,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: []f32,
    k_norm: []f32,
    ffn_norm: []f32,
    router: Mat,
};

/// Every weight except the experts, pointing into a region loaded by a plan
/// (norm vectors are copied out as aligned f32).
pub const Weights = struct {
    cfg: Config,
    embed: Mat,
    layers: []Layer,
    out_norm: []f32,
    head: Mat,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Weights) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// The non-expert tensors' byte ranges, to add to a load plan.
    pub fn ranges(gpa: std.mem.Allocator, h: gguf.Header) ![]loader.Range {
        var list: std.ArrayList(loader.Range) = .empty;
        for (h.tensors) |t| {
            if (std.mem.indexOf(u8, t.name, "_exps.") != null) continue;
            try list.append(gpa, .{ .offset = t.offset, .len = t.bytes });
        }
        return list.toOwnedSlice(gpa);
    }

    pub fn init(gpa: std.mem.Allocator, h: gguf.Header, cfg: Config, plan: loader.Plan, region: []const u8) !Weights {
        var self: Weights = .{ .cfg = cfg, .embed = undefined, .layers = undefined, .out_norm = undefined, .head = undefined, .arena = .init(gpa) };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        const Get = struct {
            hh: gguf.Header,
            p: loader.Plan,
            reg: []const u8,
            al: std.mem.Allocator,
            fn info(s: @This(), name: []const u8) !gguf.TensorInfo {
                return s.hh.tensor(name) orelse {
                    std.log.err("missing tensor {s}", .{name});
                    return error.MissingTensor;
                };
            }
            fn mat(s: @This(), name: []const u8, rows: usize, cols: usize) !Mat {
                const t = try s.info(name);
                if (t.dims[0] != cols or t.dims[1] != rows) {
                    std.log.err("{s}: dims {any}, expected [{d}, {d}]", .{ name, t.dims[0..t.ndim], cols, rows });
                    return error.BadShape;
                }
                return .{ .format = t.format, .rows = rows, .cols = cols, .bytes = s.p.locate(s.reg, .{ .offset = t.offset, .len = t.bytes }) };
            }
            fn vec(s: @This(), name: []const u8, n: usize) ![]f32 {
                const m = try s.mat(name, 1, n);
                const out = try s.al.alloc(f32, n);
                try quant.dequantize(m.format, m.bytes, out);
                return out;
            }
        };
        const g: Get = .{ .hh = h, .p = plan, .reg = region, .al = a };
        const d = cfg.d_model;
        self.embed = try g.mat("token_embd.weight", cfg.vocab, d);
        self.head = try g.mat("output.weight", cfg.vocab, d);
        self.out_norm = try g.vec("output_norm.weight", d);
        self.layers = try a.alloc(Layer, cfg.layers);
        for (self.layers, 0..) |*l, i| {
            var buf: [64]u8 = undefined;
            const n = struct {
                fn f(b: []u8, layer: usize, s: []const u8) []const u8 {
                    return std.fmt.bufPrint(b, "blk.{d}.{s}.weight", .{ layer, s }) catch unreachable;
                }
            }.f;
            // OLMoE normalizes the whole projection; Qwen3 each head.
            const qn = if (cfg.arch == .qwen3moe) cfg.head_dim else cfg.qDim();
            const kn = if (cfg.arch == .qwen3moe) cfg.head_dim else cfg.kvDim();
            l.* = .{
                .attn_norm = try g.vec(n(&buf, i, "attn_norm"), d),
                .q = try g.mat(n(&buf, i, "attn_q"), cfg.qDim(), d),
                .k = try g.mat(n(&buf, i, "attn_k"), cfg.kvDim(), d),
                .v = try g.mat(n(&buf, i, "attn_v"), cfg.kvDim(), d),
                .o = try g.mat(n(&buf, i, "attn_output"), d, cfg.qDim()),
                .q_norm = try g.vec(n(&buf, i, "attn_q_norm"), qn),
                .k_norm = try g.vec(n(&buf, i, "attn_k_norm"), kn),
                .ffn_norm = try g.vec(n(&buf, i, "ffn_norm"), d),
                .router = try g.mat(n(&buf, i, "ffn_gate_inp"), cfg.experts, d),
            };
        }
        return self;
    }
};

/// Per-layer values recorded during `forward`, for comparison with the
/// reference: hidden state after each layer (layers x n x d_model) and the
/// chosen experts (layers x n x top_k).
pub const Trace = struct {
    hidden: []f32,
    topk: []u32,
};

/// Top-k of `p` by value, largest first; ties go to the lower index (as
/// torch.topk does on CPU in practice).
pub fn topK(p: []const f32, k: usize, idx: []u32, val: []f32) void {
    for (0..k) |j| {
        var best: ?usize = null;
        for (p, 0..) |v, e| {
            var taken = false;
            for (idx[0..j]) |x| taken = taken or x == e;
            if (taken) continue;
            if (best == null or v > p[best.?]) best = e;
        }
        idx[j] = @intCast(best.?);
        val[j] = p[best.?];
    }
}

/// Keys (after QK norm and rotary) and values for every position seen so
/// far, per layer, so each new token attends to them without recomputing.
pub const KvCache = struct {
    cap: usize,
    d: usize,
    len: usize = 0,
    k: []f32,
    v: []f32,
    /// Device addresses of `k` and `v` when a GPU reads them in place, else 0.
    dev_k: u64 = 0,
    dev_v: u64 = 0,
    /// Set for a cache whose cold middle lives on NVMe (TASK-12); `k` and `v` are then empty.
    tier: ?Tier = null,

    /// Where a tiered cache keeps its positions (`kv_tier.zig`). Attention
    /// on it runs on the GPU only.
    pub const Tier = struct {
        ctx: *anyopaque,
        /// Takes `n` new positions' keys and values (n x kvd each) from `pos0` on.
        store: *const fn (ctx: *anyopaque, layer: usize, pos0: usize, n: usize, k: []const f32, v: []const f32) void,
        /// Device addresses of every tile of `layer` up to `len` positions, once they are readable.
        tiles: *const fn (ctx: *anyopaque, layer: usize, len: usize, out_k: []u64, out_v: []u64) anyerror!void,
        /// Called before a forward pass over positions `len` on, and after it with the new length.
        begin: *const fn (ctx: *anyopaque, len: usize) anyerror!void,
        end: *const fn (ctx: *anyopaque, len: usize) anyerror!void,
    };

    /// A cache that keeps its positions through `tier`, with room for `cap`.
    pub fn tiered(cfg: Config, cap: usize, tier: Tier) KvCache
    {
        return .{ .cap = cap, .d = cfg.kvDim(), .k = &.{}, .v = &.{}, .tier = tier };
    }

    /// Room for `cap` positions in every layer. Page-aligned, so a GPU can
    /// map it; `a` is unused and kept for callers.
    pub fn init(a: std.mem.Allocator, cfg: Config, cap: usize) !KvCache
    {
        _ = a;
        const pages = std.heap.page_allocator;
        const d = cfg.kvDim();
        // Positions rounded up to whole tiles, so the GPU never reads past the end.
        const room = (cap + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions * AttentionJob.tile_positions;
        const k = try pages.alloc(f32, cfg.layers * room * d);
        errdefer pages.free(k);
        const v = try pages.alloc(f32, cfg.layers * room * d);
        var self: KvCache = .{ .cap = room, .d = d, .k = k, .v = v };
        if (gpu_register) |reg|
        {
            if (cfg.head_dim == AttentionJob.head_dim)
            {
                if (reg(std.mem.sliceAsBytes(k))) |dk|
                {
                    if (reg(std.mem.sliceAsBytes(v))) |dv|
                    {
                        self.dev_k = dk;
                        self.dev_v = dv;
                    }
                    else
                    {
                        gpu_unregister.?(std.mem.sliceAsBytes(k));
                    }
                }
            }
        }
        return self;
    }

    pub fn deinit(self: *KvCache, a: std.mem.Allocator) void
    {
        _ = a;
        if (self.tier != null)
        {
            self.* = undefined;
            return;
        }
        if (self.dev_k != 0)
        {
            if (gpu_unregister) |un|
            {
                un(std.mem.sliceAsBytes(self.k));
                un(std.mem.sliceAsBytes(self.v));
            }
        }
        std.heap.page_allocator.free(self.k);
        std.heap.page_allocator.free(self.v);
        self.* = undefined;
    }

    /// Device addresses of each tile of `layer` covering positions 0 to `len` - 1.
    pub fn tiles(self: KvCache, layer: usize, len: usize, out_k: []u64, out_v: []u64) void
    {
        const tile_bytes = AttentionJob.tile_positions * self.d * @sizeOf(f32);
        const layer_bytes = self.cap * self.d * @sizeOf(f32);
        for (0..(len + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions) |i|
        {
            out_k[i] = self.dev_k + layer * layer_bytes + i * tile_bytes;
            out_v[i] = self.dev_v + layer * layer_bytes + i * tile_bytes;
        }
    }

    /// Adds `n` positions' keys and values (n x kvd each) from `pos0` on.
    pub fn store(self: *KvCache, layer: usize, pos0: usize, n: usize, k: []const f32, v: []const f32) void
    {
        if (self.tier) |t|
        {
            return t.store(t.ctx, layer, pos0, n, k, v);
        }
        @memcpy(self.keys(layer)[pos0 * self.d ..][0 .. n * self.d], k[0 .. n * self.d]);
        @memcpy(self.values(layer)[pos0 * self.d ..][0 .. n * self.d], v[0 .. n * self.d]);
    }

    pub fn keys(self: KvCache, layer: usize) []f32
    {
        return self.k[layer * self.cap * self.d ..][0 .. self.cap * self.d];
    }

    pub fn values(self: KvCache, layer: usize) []f32
    {
        return self.v[layer * self.cap * self.d ..][0 .. self.cap * self.d];
    }
};

/// Runs `tokens` (positions 0 to n-1) through the model and writes every
/// position's logits (n x vocab).
pub fn forward(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, tokens: []const u32, logits: []f32, trace: ?Trace) !void
{
    var cache = try KvCache.init(gpa, w.cfg, tokens.len);
    defer cache.deinit(gpa);
    try step(gpa, w, moe, &cache, tokens, logits, trace);
}

/// Runs `tokens` at the positions after those already in `cache`, adding
/// them to it. `logits` takes every new position's logits (n x vocab), or
/// only the last one's if it holds fewer rows than that.
pub fn step(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, cache: *KvCache, tokens: []const u32, logits: []f32, trace: ?Trace) !void
{
    const cfg = w.cfg;
    const n = tokens.len;
    const d = cfg.d_model;
    const k = cfg.top_k;
    const pos0 = cache.len;
    if (pos0 + n > cache.cap)
    {
        return error.ContextFull;
    }
    const all_logits = logits.len >= n * cfg.vocab;
    if (logits.len < cfg.vocab)
    {
        return error.LogitsTooSmall;
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const qd = cfg.qDim();
    const kvd = cfg.kvDim();
    const group = cfg.heads / cfg.kv_heads;
    const rope: Rope = .init(cfg);
    const h = try a.alloc(f32, n * d);
    const x = try a.alloc(f32, n * d);
    const q = try a.alloc(f32, n * qd);
    const att = try a.alloc(f32, n * qd);
    const tmp = try a.alloc(f32, n * @max(d, qd));
    const scores = try a.alloc(f32, pos0 + n);
    const kn = try a.alloc(f32, n * kvd);
    const vn = try a.alloc(f32, n * kvd);
    const tile_count = (pos0 + n + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions;
    const tiles_k = try a.alloc(u64, tile_count);
    const tiles_v = try a.alloc(u64, tile_count);
    const scratch = try a.alloc(f32, @max(@max(d, qd), cfg.d_ff));
    const probs = try a.alloc(f32, n * cfg.experts);
    const idx = try a.alloc(u32, n * k);
    const wts = try a.alloc(f32, n * k);
    // Lookahead guesses: per horizon and target layer, top_k + extra experts per token.
    const la = moe.ahead;
    const kx = if (la) |p| k + p.extra else 0;
    const horizon = if (la) |p| @min(p.horizon, Lookahead.max_horizon) else 0;
    const guess = try a.alloc(u32, horizon * cfg.layers * n * kx);
    const guess_val = try a.alloc(f32, kx);
    const votes = try a.alloc(u32, if (la != null) cfg.experts else 0);
    const wanted_list = try a.alloc(u32, votes.len);

    for (tokens, 0..) |tok, t|
    {
        if (tok >= cfg.vocab)
        {
            return error.BadToken;
        }
        w.embed.row(tok, h[t * d ..][0..d]);
    }
    const ahead: Guesser = .{ .w = w, .moe = moe, .n = n, .kx = kx, .out = guess, .val = guess_val, .votes = votes, .wanted_list = wanted_list };
    if (cache.tier) |tr|
    {
        try tr.begin(tr.ctx, pos0);
    }
    if (la) |p|
    {
        // The first layers' guesses come from the token embeddings, before any attention.
        const t0 = nowNs();
        for (1..@min(horizon, cfg.layers) + 1) |hz|
        {
            try ahead.guess(hz - 1, hz, h, x, probs, scratch);
        }
        p.predict_ns += nowNs() - t0;
    }
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    for (w.layers, 0..) |*l, li|
    {
        const layer_t0 = if (la != null) nowNs() else 0;
        var pt = nowNs();
        for (0..n) |t|
        {
            rmsnorm(h[t * d ..][0..d], l.attn_norm, cfg.eps, x[t * d ..][0..d]);
        }
        l.q.matmul(x, n, q, scratch);
        l.k.matmul(x, n, kn, scratch);
        l.v.matmul(x, n, vn, scratch);
        Profile.lap(&profile.projections, &pt);
        for (0..n) |t|
        {
            const qt = q[t * qd ..][0..qd];
            const kt = kn[t * kvd ..][0..kvd];
            qkNorm(cfg, qt, l.q_norm, tmp);
            qkNorm(cfg, kt, l.k_norm, tmp);
            rope.apply(qt, cfg.heads, cfg.head_dim, pos0 + t);
            rope.apply(kt, cfg.kv_heads, cfg.head_dim, pos0 + t);
        }
        cache.store(li, pos0, n, kn, vn);
        Profile.lap(&profile.qk_rope, &pt);
        const kc = cache.k;
        const vc = cache.v;
        var on_gpu = false;
        if (gpu_attention != null and (cache.dev_k != 0 or cache.tier != null))
        {
            if (cache.tier) |tr|
            {
                try tr.tiles(tr.ctx, li, pos0 + n, tiles_k, tiles_v);
            }
            else
            {
                cache.tiles(li, pos0 + n, tiles_k, tiles_v);
            }
            const count = (pos0 + n + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions;
            on_gpu = gpu_attention.?(.{
                .q = q,
                .n = n,
                .pos0 = pos0,
                .heads = cfg.heads,
                .kv_heads = cfg.kv_heads,
                .kvd = kvd,
                .scale = scale,
                .tiles_k = tiles_k[0..count],
                .tiles_v = tiles_v[0..count],
                .splits = AttentionJob.splitsFor(n, cfg.heads, cfg.kv_heads, count),
                .out = att,
            });
        }
        if (!on_gpu and cache.tier != null)
        {
            return error.OffloadNeedsGpu;
        }
        const layer_k = if (on_gpu) kc else cache.keys(li);
        const layer_v = if (on_gpu) vc else cache.values(li);
        for (0..n) |t|
        {
            if (on_gpu)
            {
                break;
            }
            const last = pos0 + t;
            for (0..cfg.heads) |hd|
            {
                const off = hd * cfg.head_dim;
                const kv_off = hd / group * cfg.head_dim;
                const qh = q[t * qd + off ..][0..cfg.head_dim];
                var mx: f32 = -std.math.inf(f32);
                for (0..last + 1) |s|
                {
                    scores[s] = dot(qh, layer_k[s * kvd + kv_off ..][0..cfg.head_dim]) * scale;
                    mx = @max(mx, scores[s]);
                }
                var sum: f32 = 0;
                for (scores[0 .. last + 1]) |*sc|
                {
                    sc.* = @exp(sc.* - mx);
                    sum += sc.*;
                }
                const out = att[t * qd + off ..][0..cfg.head_dim];
                @memset(out, 0);
                for (0..last + 1) |s|
                {
                    const p = scores[s] / sum;
                    for (out, layer_v[s * kvd + kv_off ..][0..cfg.head_dim]) |*o, vv|
                    {
                        o.* += p * vv;
                    }
                }
            }
        }
        Profile.lap(&profile.attention, &pt);
        l.o.matmul(att, n, tmp, scratch);
        for (h, tmp[0 .. n * d]) |*hh, o|
        {
            hh.* += o;
        }

        if (la) |p|
        {
            const t0 = nowNs();
            if (li + horizon < cfg.layers)
            {
                try ahead.guess(li + horizon, horizon, h, x, probs, scratch);
            }
            p.predict_ns += nowNs() - t0;
        }

        for (0..n) |t|
        {
            rmsnorm(h[t * d ..][0..d], l.ffn_norm, cfg.eps, x[t * d ..][0..d]);
        }
        Profile.lap(&profile.output, &pt);
        l.router.matmul(x, n, probs, scratch);
        for (0..n) |t|
        {
            const p = probs[t * cfg.experts ..][0..cfg.experts];
            var mx: f32 = -std.math.inf(f32);
            for (p) |pv|
            {
                mx = @max(mx, pv);
            }
            var sum: f32 = 0;
            for (p) |*pv|
            {
                pv.* = @exp(pv.* - mx);
                sum += pv.*;
            }
            for (p) |*pv|
            {
                pv.* /= sum;
            }
            topK(p, k, idx[t * k ..][0..k], wts[t * k ..][0..k]);
            if (cfg.arch == .qwen3moe)
            {
                // Qwen3 renormalizes the chosen probabilities to sum to 1.
                const wt = wts[t * k ..][0..k];
                var s: f32 = 0;
                for (wt) |v|
                {
                    s += v;
                }
                for (wt) |*v|
                {
                    v.* /= s;
                }
            }
        }
        if (la) |p|
        {
            scoreGuesses(p, guess, idx, li, n, k, kx, horizon, cfg.layers);
        }
        Profile.lap(&profile.router, &pt);
        @memset(tmp, 0);
        try moe.run(moe.ctx, li, x, n, idx, wts, tmp[0 .. n * d]);
        Profile.lap(&profile.experts, &pt);
        for (h, tmp[0 .. n * d]) |*hh, o|
        {
            hh.* += o;
        }
        if (la) |p|
        {
            p.layer_ns += nowNs() - layer_t0;
        }

        if (trace) |tr|
        {
            @memcpy(tr.hidden[li * n * d ..][0 .. n * d], h);
            @memcpy(tr.topk[li * n * k ..][0 .. n * k], idx);
        }
    }
    cache.len = pos0 + n;
    if (cache.tier) |tr|
    {
        try tr.end(tr.ctx, cache.len);
    }
    var head_t = nowNs();
    defer Profile.lap(&profile.head, &head_t);
    if (all_logits)
    {
        for (0..n) |t|
        {
            rmsnorm(h[t * d ..][0..d], w.out_norm, cfg.eps, x[t * d ..][0..d]);
        }
        w.head.matmul(x, n, logits, scratch);
    }
    else
    {
        rmsnorm(h[(n - 1) * d ..][0..d], w.out_norm, cfg.eps, x[0..d]);
        w.head.matmul(x, 1, logits, scratch);
    }
}

/// One forward pass's lookahead guesses (n x kx experts per horizon and target).
const Guesser = struct {
    w: *const Weights,
    moe: Moe,
    n: usize,
    kx: usize,
    out: []u32,
    val: []f32,
    /// Per expert: how many tokens guessed it.
    votes: []u32,
    wanted_list: []u32,

    /// Guesses layer `target`'s experts from the hidden states `h`, `hz` layers
    /// early, and passes them to `moe.prefetch`. `x` and `logits` are scratch
    /// the caller overwrites later.
    fn guess(self: Guesser, target: usize, hz: usize, h: []const f32, x: []f32, logits: []f32, scratch: []f32) !void
    {
        const cfg = self.w.cfg;
        const d = cfg.d_model;
        const n = self.n;
        const tl = &self.w.layers[target];
        const out = self.out[((hz - 1) * cfg.layers + target) * n * self.kx ..][0 .. n * self.kx];
        for (0..n) |t|
        {
            rmsnorm(h[t * d ..][0..d], tl.ffn_norm, cfg.eps, x[t * d ..][0..d]);
        }
        tl.router.matmul(x, n, logits, scratch);
        for (0..n) |t|
        {
            // Softmax keeps the order, so the raw router outputs rank the same.
            const row = out[t * self.kx ..][0..self.kx];
            topK(logits[t * cfg.experts ..][0..cfg.experts], self.kx, row, self.val);
            const p = self.moe.ahead.?;
            if (p.mispredict > 0)
            {
                p.corrupt(row, cfg.experts);
            }
        }
        const f = self.moe.prefetch orelse return;
        @memset(self.votes, 0);
        var count: usize = 0;
        for (out) |e|
        {
            if (self.votes[e] == 0)
            {
                self.wanted_list[count] = e;
                count += 1;
            }
            self.votes[e] += 1;
        }
        // Experts that more tokens guessed are hinted, and so loaded, first.
        std.sort.insertion(u32, self.wanted_list[0..count], self.votes, struct {
            fn more(v: []u32, a: u32, b: u32) bool
            {
                return v[a] > v[b];
            }
        }.more);
        try f(self.moe.ctx, target, self.wanted_list[0..count]);
        self.moe.ahead.?.prefetches += 1;
    }
};

/// Counts how many of layer `li`'s chosen experts its earlier guesses held.
fn scoreGuesses(p: *Lookahead, guess: []const u32, idx: []const u32, li: usize, n: usize, k: usize, kx: usize, horizon: usize, layers: usize) void
{
    // The first layers were guessed from the embeddings, fewer layers ahead.
    const hz = @min(horizon, li + 1);
    const g = guess[((hz - 1) * layers + li) * n * kx ..][0 .. n * kx];
    for (0..n) |t|
    {
        for (idx[t * k ..][0..k]) |e|
        {
            p.hits[hz - 1] += @intFromBool(std.mem.indexOfScalar(u32, g[t * kx ..][0..kx], e) != null);
        }
    }
    p.total[hz - 1] += n * k;
}

fn nowNs() u64
{
    if (@import("builtin").os.tag == .windows)
    {
        return @import("win32.zig").nowNs();
    }
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Index of the largest value; ties go to the lowest index, as torch.argmax.
pub fn argmax(x: []align(1) const f32) usize
{
    var best: usize = 0;
    for (x, 0..) |v, i|
    {
        if (v > x[best])
        {
            best = i;
        }
    }
    return best;
}

/// Picks the next token from a row of logits: the highest one when
/// `temperature` is 0, otherwise a seeded draw, so a run can be repeated.
pub const Sampler = struct {
    temperature: f32 = 0,
    /// Draw only from this many of the most likely tokens; 0 keeps them all.
    top_k: usize = 0,
    rng: std.Random.DefaultPrng = .init(0),

    pub fn seeded(temperature: f32, top_k: usize, seed: u64) Sampler
    {
        return .{ .temperature = temperature, .top_k = top_k, .rng = .init(seed) };
    }

    /// `scratch` needs as many values as `logits`.
    pub fn pick(self: *Sampler, logits: []const f32, scratch: []f32) u32
    {
        const best = argmax(logits);
        if (self.temperature <= 0)
        {
            return @intCast(best);
        }
        const p = scratch[0..logits.len];
        var floor: f32 = -std.math.inf(f32);
        if (self.top_k > 0 and self.top_k < logits.len)
        {
            @memcpy(p, logits);
            std.sort.pdq(f32, p, {}, std.sort.desc(f32));
            floor = p[self.top_k - 1];
        }
        var sum: f32 = 0;
        for (logits, p) |v, *pv|
        {
            pv.* = if (v >= floor) @exp((v - logits[best]) / self.temperature) else 0;
            sum += pv.*;
        }
        var r = self.rng.random().float(f32) * sum;
        for (p, 0..) |pv, i|
        {
            r -= pv;
            if (pv > 0 and r < 0)
            {
                return @intCast(i);
            }
        }
        return @intCast(best);
    }
};

/// Feeds `prompt` from an empty cache, then picks `out.len` tokens one at a
/// time. If set, `step_logits` gets the logits each token was picked from
/// (out.len x vocab). The cache needs room for prompt.len + out.len - 1.
pub fn generate(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, cache: *KvCache, prompt: []const u32, sampler: *Sampler, out: []u32, step_logits: ?[]f32) !void
{
    const vocab = w.cfg.vocab;
    const row = try gpa.alloc(f32, 2 * vocab);
    defer gpa.free(row);
    cache.len = 0;
    try step(gpa, w, moe, cache, prompt, row[0..vocab], null);
    for (out, 0..) |*tok, i|
    {
        if (step_logits) |sl|
        {
            @memcpy(sl[i * vocab ..][0..vocab], row[0..vocab]);
        }
        tok.* = sampler.pick(row[0..vocab], row[vocab..]);
        if (i + 1 < out.len)
        {
            try step(gpa, w, moe, cache, out[i..][0..1], row[0..vocab], null);
        }
    }
}

// Experts held in this process, by (layer, expert); null if not owned.
pub const LocalExperts = struct {
    cfg: Config,
    experts: []?Expert,
    scratch: []f32,
    out: []f32,
    /// Working space for batched calls (TASK-12), grown as needed.
    batch: []f32 = &.{},
    /// Below this many tokens, calls run one at a time.
    batch_min: usize = 16,

    const max_group = 1024;

    pub fn init(a: std.mem.Allocator, cfg: Config, m: loader.Model, owned: []const bool, plan: loader.Plan, region: []const u8) !LocalExperts {
        const ex = try a.alloc(?Expert, cfg.layers * cfg.experts);
        for (ex, 0..) |*e, i| {
            const layer: u32 = @intCast(i / cfg.experts);
            const id: u32 = @intCast(i % cfg.experts);
            if (!owned[id]) {
                e.* = null;
                continue;
            }
            const mk = struct {
                fn f(w: loader.ExpertWeight, p: loader.Plan, r: []const u8) Mat {
                    return .{ .format = w.format, .rows = @intCast(w.rows), .cols = @intCast(w.cols), .bytes = p.locate(r, w.parts[0]) };
                }
            }.f;
            e.* = .{
                .gate = mk(m.weight(layer, .gate, id), plan, region),
                .up = mk(m.weight(layer, .up, id), plan, region),
                .down = mk(m.weight(layer, .down, id), plan, region),
            };
        }
        return .{
            .cfg = cfg,
            .experts = ex,
            .scratch = try a.alloc(f32, 3 * cfg.d_ff + @max(cfg.d_model, cfg.d_ff)),
            .out = try a.alloc(f32, cfg.d_model),
        };
    }

    pub fn get(self: *const LocalExperts, layer: usize, expert: usize) ?Expert {
        return self.experts[layer * self.cfg.experts + expert];
    }

    /// `Moe` for a process that owns every expert.
    pub fn moe(self: *LocalExperts) Moe {
        return .{ .ctx = self, .run = runAll };
    }

    /// Frees the batch space.
    pub fn deinit(self: *LocalExperts) void
    {
        if (self.batch.len > 0)
        {
            std.heap.page_allocator.free(self.batch);
        }
        self.batch = &.{};
    }

    fn runAll(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void {
        const self: *LocalExperts = @ptrCast(@alignCast(ctx));
        const d = self.cfg.d_model;
        const k = self.cfg.top_k;
        if (n >= self.batch_min)
        {
            return self.runBatched(layer, x, n, idx, w, out);
        }
        for (0..n) |t| {
            for (0..k) |j| {
                const e = self.get(layer, idx[t * k + j]) orelse return error.ExpertNotLocal;
                e.forward(x[t * d ..][0..d], self.out, self.scratch);
                for (out[t * d ..][0..d], self.out) |*o, y| o.* += w[t * k + j] * y;
            }
        }
    }

    /// Runs each expert once on all the tokens that chose it, which a long
    /// prefill needs. Each row comes out the same as a call on its own, and
    /// the outputs are summed in the same order, so the result is unchanged.
    fn runBatched(self: *LocalExperts, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) !void
    {
        const d = self.cfg.d_model;
        const f = self.cfg.d_ff;
        const k = self.cfg.top_k;
        const calls = n * k;
        // Per call: its output; per expert's batch: inputs, gate, up, and outputs.
        const need = 3 * calls * d + 2 * calls * f + calls;
        if (self.batch.len < need)
        {
            if (self.batch.len > 0)
            {
                std.heap.page_allocator.free(self.batch);
            }
            self.batch = try std.heap.page_allocator.alloc(f32, need);
        }
        const contrib = self.batch[0 .. calls * d];
        const xb = self.batch[calls * d ..][0 .. calls * d];
        const gb = self.batch[2 * calls * d ..][0 .. calls * f];
        const ub = self.batch[2 * calls * d + calls * f ..][0 .. calls * f];
        const yb = self.batch[2 * calls * d + 2 * calls * f ..][0 .. calls * d];
        const order: []u32 = @ptrCast(self.batch[3 * calls * d + 2 * calls * f ..][0..calls]);
        // Calls grouped by expert, each group in call order, inputs gathered to match.
        var used: [max_group]Expert = undefined;
        var starts: [max_group + 1]u32 = undefined;
        var groups: usize = 0;
        var filled: usize = 0;
        for (0..self.cfg.experts) |e|
        {
            const first = filled;
            for (idx[0..calls], 0..) |ce, c|
            {
                if (ce == e)
                {
                    order[filled] = @intCast(c);
                    @memcpy(xb[filled * d ..][0..d], x[(c / k) * d ..][0..d]);
                    filled += 1;
                }
            }
            if (filled == first)
            {
                continue;
            }
            used[groups] = self.get(layer, e) orelse return error.ExpertNotLocal;
            starts[groups] = @intCast(first);
            groups += 1;
        }
        starts[groups] = @intCast(filled);
        const on_gpu = if (gpu_experts) |run| run(.{ .experts = used[0..groups], .first = starts[0 .. groups + 1], .x = xb[0 .. filled * d], .y = yb[0 .. filled * d] }) else false;
        if (!on_gpu)
        {
            for (used[0..groups], 0..) |ex, gi|
            {
                const first = starts[gi];
                const count = starts[gi + 1] - first;
                ex.gate.matmul(xb[first * d ..], count, gb, self.scratch);
                ex.up.matmul(xb[first * d ..], count, ub, self.scratch);
                for (gb[0 .. count * f], ub[0 .. count * f]) |*a, b|
                {
                    a.* = silu(a.*) * b;
                }
                ex.down.matmul(gb, count, yb[first * d ..], self.scratch);
            }
        }
        for (order[0..filled], 0..) |c, i|
        {
            @memcpy(contrib[c * d ..][0..d], yb[i * d ..][0..d]);
        }
        for (0..n) |t|
        {
            for (0..k) |j|
            {
                const c = t * k + j;
                for (out[t * d ..][0..d], contrib[c * d ..][0..d]) |*o, y|
                {
                    o.* += w[c] * y;
                }
            }
        }
    }
};

test "tiny OLMoE matches transformers (tools/ref/ref_olmoe.py --tiny)" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("olmoe_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    // The whole file is the loaded region: one span at offset 0.
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [8]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);

    const golden = try tfref.File.parse(@embedFile("olmoe_tiny.tfref"));
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    for (0..prompts) |p| {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const tokens = try gpa.alloc(u32, ids.len);
        defer gpa.free(tokens);
        for (tokens, ids) |*t, id| t.* = @intCast(id);
        const logits = try gpa.alloc(f32, ids.len * cfg.vocab);
        defer gpa.free(logits);
        const trace: Trace = .{ .hidden = try gpa.alloc(f32, cfg.layers * ids.len * cfg.d_model), .topk = try gpa.alloc(u32, cfg.layers * ids.len * cfg.top_k) };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        try forward(gpa, &w, local.moe(), tokens, logits, trace);

        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        for (logits, want) |a, b| try std.testing.expectApproxEqAbs(b, a, 1e-4);
        for (0..cfg.layers) |l| {
            const gi = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l }))).i32s();
            for (trace.topk[l * ids.len * cfg.top_k ..][0 .. ids.len * cfg.top_k], gi) |got, exp| try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
    }
}

test "cached generation matches a full recompute and transformers' generate() on the tiny OLMoE" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("olmoe_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [8]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);

    const golden = try tfref.File.parse(@embedFile("olmoe_tiny.tfref"));
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    for (0..prompts) |p|
    {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p}))).i32s();
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        const g = want.len;
        const all = try gpa.alloc(u32, ids.len + g);
        defer gpa.free(all);
        for (all[0..ids.len], ids) |*t, id|
        {
            t.* = @intCast(id);
        }
        var cache = try KvCache.init(gpa, cfg, ids.len + g);
        defer cache.deinit(gpa);
        var sampler: Sampler = .{};
        const step_logits = try gpa.alloc(f32, g * cfg.vocab);
        defer gpa.free(step_logits);
        try generate(gpa, &w, local.moe(), &cache, all[0..ids.len], &sampler, all[ids.len..], step_logits);

        for (all[ids.len..], want) |got, exp|
        {
            try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
        for (step_logits, want_logits) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-4);
        }
        // The same logits with no cache: one pass over the prompt and every generated token but the last.
        const full = try gpa.alloc(f32, (ids.len + g - 1) * cfg.vocab);
        defer gpa.free(full);
        try forward(gpa, &w, local.moe(), all[0 .. ids.len + g - 1], full, null);
        for (step_logits, full[(ids.len - 1) * cfg.vocab ..]) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-5);
        }
    }
}

test "tiny Qwen3-MoE with YaRN matches transformers (tools/ref/ref_qwen3.py --tiny)" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("qwen3_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = (try Config.fromGguf(h)).withYarn(4, 16);
    try std.testing.expectEqual(Arch.qwen3moe, cfg.arch);
    try std.testing.expectEqual(@as(usize, 2), cfg.kv_heads);
    try std.testing.expectEqual(@as(usize, 32), cfg.head_dim);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [8]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);
    defer local.deinit();

    const golden = try tfref.File.parse(@embedFile("qwen3_tiny.tfref"));
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    for (0..prompts) |p|
    {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const tokens = try gpa.alloc(u32, ids.len);
        defer gpa.free(tokens);
        for (tokens, ids) |*t, id|
        {
            t.* = @intCast(id);
        }
        const logits = try gpa.alloc(f32, ids.len * cfg.vocab);
        defer gpa.free(logits);
        const trace: Trace = .{ .hidden = try gpa.alloc(f32, cfg.layers * ids.len * cfg.d_model), .topk = try gpa.alloc(u32, cfg.layers * ids.len * cfg.top_k) };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        try forward(gpa, &w, local.moe(), tokens, logits, trace);
        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        for (logits, want) |a, b|
        {
            try std.testing.expectApproxEqAbs(b, a, 1e-4);
        }
        for (0..cfg.layers) |l|
        {
            const gi = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l }))).i32s();
            for (trace.topk[l * ids.len * cfg.top_k ..][0 .. ids.len * cfg.top_k], gi) |got, exp|
            {
                try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
            }
        }

        const want_tokens = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p}))).i32s();
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        const g = want_tokens.len;
        const all = try gpa.alloc(u32, ids.len + g);
        defer gpa.free(all);
        @memcpy(all[0..ids.len], tokens);
        var cache = try KvCache.init(gpa, cfg, ids.len + g);
        defer cache.deinit(gpa);
        var sampler: Sampler = .{};
        const step_logits = try gpa.alloc(f32, g * cfg.vocab);
        defer gpa.free(step_logits);
        try generate(gpa, &w, local.moe(), &cache, all[0..ids.len], &sampler, all[ids.len..], step_logits);
        for (all[ids.len..], want_tokens) |got, exp|
        {
            try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
        for (step_logits, want_logits) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-4);
        }
    }
}

test "lookahead guesses leave every logit bit-identical and arrive before their layer" {
    const gpa = std.testing.allocator;
    const file = @embedFile("olmoe_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [8]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);

    // Records each prefetch's target and the layers that had run by then.
    const Log = struct {
        local: *LocalExperts,
        layers_run: usize = 0,
        ok: bool = true,
        calls: usize = 0,
        fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, wt: []const f32, out: []f32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.layers_run = layer + 1;
            return LocalExperts.runAll(self.local, layer, x, n, idx, wt, out);
        }
        fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
            self.ok = self.ok and layer >= self.layers_run and experts.len > 0;
        }
    };
    var log: Log = .{ .local = &local };
    var la: Lookahead = .{ .horizon = 2, .extra = 1 };
    const with: Moe = .{ .ctx = &log, .run = Log.run, .ahead = &la, .prefetch = Log.prefetch };

    const tokens = [_]u32{ 3, 17, 200, 45, 9, 1, 2, 3 };
    const plain = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(plain);
    const guessed = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(guessed);
    try forward(gpa, &w, local.moe(), &tokens, plain, null);
    try forward(gpa, &w, with, &tokens, guessed, null);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(guessed));
    try std.testing.expect(log.ok);
    // Tiny OLMoE has 2 layers, both guessed from the embeddings: layer 0 one layer ahead, layer 1 two.
    try std.testing.expectEqual(@as(usize, 2), log.calls);
    try std.testing.expectEqual(@as(u64, tokens.len * cfg.top_k), la.total[0]);
    try std.testing.expectEqual(@as(u64, tokens.len * cfg.top_k), la.total[1]);
    try std.testing.expect(la.recall(1) > 0 and la.recall(1) <= 1);
    try std.testing.expect(la.layer_ns > 0);
}

test "threaded CPU matmuls give the same output as one thread" {
    const gpa = std.testing.allocator;
    const rows = 96;
    const cols = 64;
    const n = 5;
    var rng: std.Random.DefaultPrng = .init(3);
    const r = rng.random();
    const wf = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wf);
    const x = try gpa.alloc(f32, n * cols);
    defer gpa.free(x);
    for ([_][]f32{ wf, x }) |buf|
    {
        for (buf) |*v|
        {
            v.* = r.float(f32) * 2 - 1;
        }
    }
    // Q8_0 blocks: an f16 scale, then 32 signed bytes.
    const q8 = try gpa.alloc(u8, rows * cols / 32 * 34);
    defer gpa.free(q8);
    r.bytes(q8);
    var b: usize = 0;
    while (b < q8.len) : (b += 34)
    {
        std.mem.writeInt(u16, q8[b..][0..2], @bitCast(@as(f16, 0.01)), .little);
    }
    const scratch = try gpa.alloc(f32, cols);
    defer gpa.free(scratch);
    for ([_]Mat{
        .{ .format = .f32, .rows = rows, .cols = cols, .bytes = std.mem.sliceAsBytes(wf) },
        .{ .format = .q8_0, .rows = rows, .cols = cols, .bytes = q8 },
    }) |m|
    {
        const one = try gpa.alloc(f32, n * rows);
        defer gpa.free(one);
        const many = try gpa.alloc(f32, n * rows);
        defer gpa.free(many);
        const saved = Mat.parallel_min;
        defer Mat.parallel_min = saved;
        Mat.parallel_min = std.math.maxInt(usize);
        m.matmul(x, n, one, scratch);
        Mat.parallel_min = 1;
        m.matmulParallel(x, n, many, 4);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(many));
    }
}

test "batched expert calls give the same output as one call at a time" {
    const gpa = std.testing.allocator;
    const file = @embedFile("olmoe_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [8]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);
    defer local.deinit();
    var tokens: [40]u32 = undefined;
    for (&tokens, 0..) |*t, i|
    {
        t.* = @intCast((i * 37 + 11) % cfg.vocab);
    }
    const one = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(one);
    const batched = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(batched);
    local.batch_min = std.math.maxInt(usize);
    try forward(gpa, &w, local.moe(), &tokens, one, null);
    local.batch_min = 1;
    try forward(gpa, &w, local.moe(), &tokens, batched, null);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(batched));
}

test "seeded sampling repeats with the same seed and stays inside top_k" {
    const logits = [_]f32{ 0.1, 2.0, 1.9, -1.0, 1.8, 0.0 };
    var scratch: [logits.len]f32 = undefined;
    var a = Sampler.seeded(1.0, 3, 42);
    var b = Sampler.seeded(1.0, 3, 42);
    for (0..200) |_|
    {
        const x = a.pick(&logits, &scratch);
        try std.testing.expectEqual(x, b.pick(&logits, &scratch));
        try std.testing.expect(x == 1 or x == 2 or x == 4);
    }
    var greedy: Sampler = .{};
    try std.testing.expectEqual(@as(u32, 1), greedy.pick(&logits, &scratch));
}

test "rope matches rotate_half on one pair" {
    var x = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0 }; // head_dim 4: pairs (0, 2), (1, 3)
    const cfg: Config = .{ .layers = 1, .d_model = 8, .heads = 2, .kv_heads = 2, .head_dim = 4, .d_ff = 4, .experts = 1, .top_k = 1, .vocab = 1, .eps = 1e-5, .rope_theta = 10000 };
    const r: Rope = .init(cfg);
    r.apply(&x, 2, 4, 1);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)), x[2], 1e-6);
}

test "topK keeps order and breaks ties toward the lower index" {
    var idx: [3]u32 = undefined;
    var val: [3]f32 = undefined;
    topK(&.{ 0.1, 0.3, 0.3, 0.05, 0.25 }, 3, &idx, &val);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 4 }, &idx);
}
