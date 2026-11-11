//! OLMoE forward pass on CPU (TASK-10), matching Hugging Face's
//! OlmoeForCausalLM: per layer, pre-norm attention with RMSNorm on the full
//! query and key projections (QK norm), rotary embeddings in the rotate-half
//! (NeoX) layout, causal softmax attention; then a pre-norm sparse MoE block
//! that takes the softmax over all experts, keeps the top k without
//! renormalizing, and sums the chosen SwiGLU experts' outputs weighted by
//! those probabilities. Final RMSNorm, then the LM head.
//!
//! Weights stay in their GGUF block format and are dequantized one row at a
//! time. The MoE block calls experts through `Moe`, so a process can run the
//! experts it owns and send the rest to a peer (`olmoe_run.zig`).
const std = @import("std");
const quant = @import("quant.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");

pub const Config = struct {
    layers: usize,
    d_model: usize,
    heads: usize,
    head_dim: usize,
    d_ff: usize,
    experts: usize,
    top_k: usize,
    vocab: usize,
    eps: f32,
    rope_theta: f32,

    pub fn fromGguf(h: gguf.Header) !Config {
        const arch = h.getString("general.architecture") orelse return error.NotOlmoe;
        if (!std.mem.eql(u8, arch, "olmoe")) return error.NotOlmoe;
        const int = struct {
            fn f(hh: gguf.Header, key: []const u8) !usize {
                return @intCast(hh.getInt(key) orelse return error.MissingKey);
            }
        }.f;
        const float = struct {
            fn f(hh: gguf.Header, key: []const u8, default: f32) f32 {
                const v = hh.get(key) orelse return default;
                return switch (v) {
                    .float => |x| @floatCast(x),
                    else => default,
                };
            }
        }.f;
        const d = try int(h, "olmoe.embedding_length");
        const heads = try int(h, "olmoe.attention.head_count");
        if (try int(h, "olmoe.attention.head_count_kv") != heads) return error.Unsupported; // OLMoE-1B-7B has no GQA
        const embd = h.tensor("token_embd.weight") orelse return error.MissingTensor;
        return .{
            .layers = try int(h, "olmoe.block_count"),
            .d_model = d,
            .heads = heads,
            .head_dim = d / heads,
            .d_ff = @intCast(h.getInt("olmoe.expert_feed_forward_length") orelse h.getInt("olmoe.feed_forward_length") orelse return error.MissingKey),
            .experts = try int(h, "olmoe.expert_count"),
            .top_k = try int(h, "olmoe.expert_used_count"),
            .vocab = @intCast(embd.dims[1]),
            .eps = float(h, "olmoe.attention.layer_norm_rms_epsilon", 1e-5),
            .rope_theta = float(h, "olmoe.rope.freq_base", 10000),
        };
    }
};

/// Set by a caller with a GPU (`gpu.zig`): runs a matmul there and returns
/// true, or returns false (unsupported format, say) to use the CPU path.
pub var gpu_matmul: ?*const fn (m: Mat, x: []const f32, n: usize, y: []f32) bool = null;

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
        if (self.format == .f32 and @intFromPtr(self.bytes.ptr) % @alignOf(f32) == 0)
        {
            // Rows are already f32 (the router, say): read them in place, same sums.
            const wv: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, self.bytes[0 .. self.rows * self.cols * 4]));
            for (0..self.rows) |r|
            {
                for (0..n) |t|
                {
                    y[t * self.rows + r] = dot(wv[r * self.cols ..][0..self.cols], x[t * self.cols ..][0..self.cols]);
                }
            }
            return;
        }
        for (0..self.rows) |r| {
            self.row(r, scratch);
            for (0..n) |t| y[t * self.rows + r] = dot(scratch[0..self.cols], x[t * self.cols ..][0..self.cols]);
        }
    }
};

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

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Rotary embedding in place on each `head_dim` slice of `x` (one token,
/// all heads): pairs are (i, i + head_dim / 2), as in rotate_half.
pub fn rope(x: []f32, heads: usize, head_dim: usize, pos: usize, theta: f32) void {
    const half = head_dim / 2;
    for (0..half) |i| {
        const inv_freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
        const angle = @as(f32, @floatFromInt(pos)) * inv_freq;
        const cs = @cos(angle);
        const sn = @sin(angle);
        for (0..heads) |h| {
            const v = x[h * head_dim ..];
            const a = v[i];
            const b = v[i + half];
            v[i] = a * cs - b * sn;
            v[i + half] = b * cs + a * sn;
        }
    }
}

/// One expert's SwiGLU: down(silu(gate x) * up x), for one token.
pub const Expert = struct {
    gate: Mat,
    up: Mat,
    down: Mat,

    /// `scratch` needs 3 * d_ff + max(d_model, d_ff) values.
    pub fn forward(self: Expert, x: []const f32, out: []f32, scratch: []f32) void {
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
            l.* = .{
                .attn_norm = try g.vec(n(&buf, i, "attn_norm"), d),
                .q = try g.mat(n(&buf, i, "attn_q"), d, d),
                .k = try g.mat(n(&buf, i, "attn_k"), d, d),
                .v = try g.mat(n(&buf, i, "attn_v"), d, d),
                .o = try g.mat(n(&buf, i, "attn_output"), d, d),
                .q_norm = try g.vec(n(&buf, i, "attn_q_norm"), d),
                .k_norm = try g.vec(n(&buf, i, "attn_k_norm"), d),
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

    /// Room for `cap` positions in every layer.
    pub fn init(a: std.mem.Allocator, cfg: Config, cap: usize) !KvCache
    {
        const k = try a.alloc(f32, cfg.layers * cap * cfg.d_model);
        errdefer a.free(k);
        return .{ .cap = cap, .d = cfg.d_model, .k = k, .v = try a.alloc(f32, cfg.layers * cap * cfg.d_model) };
    }

    pub fn deinit(self: *KvCache, a: std.mem.Allocator) void
    {
        a.free(self.k);
        a.free(self.v);
        self.* = undefined;
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
    const h = try a.alloc(f32, n * d);
    const x = try a.alloc(f32, n * d);
    const q = try a.alloc(f32, n * d);
    const att = try a.alloc(f32, n * d);
    const tmp = try a.alloc(f32, n * d);
    const scores = try a.alloc(f32, pos0 + n);
    const scratch = try a.alloc(f32, @max(d, cfg.d_ff));
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
        // New keys and values go straight into the cache, after the old ones.
        const kc = cache.keys(li);
        const vc = cache.values(li);
        const kn = kc[pos0 * d ..][0 .. n * d];
        for (0..n) |t|
        {
            rmsnorm(h[t * d ..][0..d], l.attn_norm, cfg.eps, x[t * d ..][0..d]);
        }
        l.q.matmul(x, n, q, scratch);
        l.k.matmul(x, n, kn, scratch);
        l.v.matmul(x, n, vc[pos0 * d ..][0 .. n * d], scratch);
        for (0..n) |t|
        {
            const qt = q[t * d ..][0..d];
            const kt = kn[t * d ..][0..d];
            @memcpy(tmp[0..d], qt);
            rmsnorm(tmp[0..d], l.q_norm, cfg.eps, qt);
            @memcpy(tmp[0..d], kt);
            rmsnorm(tmp[0..d], l.k_norm, cfg.eps, kt);
            rope(qt, cfg.heads, cfg.head_dim, pos0 + t, cfg.rope_theta);
            rope(kt, cfg.heads, cfg.head_dim, pos0 + t, cfg.rope_theta);
        }
        for (0..n) |t|
        {
            const last = pos0 + t;
            for (0..cfg.heads) |hd|
            {
                const off = hd * cfg.head_dim;
                const qh = q[t * d + off ..][0..cfg.head_dim];
                var mx: f32 = -std.math.inf(f32);
                for (0..last + 1) |s|
                {
                    scores[s] = dot(qh, kc[s * d + off ..][0..cfg.head_dim]) * scale;
                    mx = @max(mx, scores[s]);
                }
                var sum: f32 = 0;
                for (scores[0 .. last + 1]) |*sc|
                {
                    sc.* = @exp(sc.* - mx);
                    sum += sc.*;
                }
                const out = att[t * d + off ..][0..cfg.head_dim];
                @memset(out, 0);
                for (0..last + 1) |s|
                {
                    const p = scores[s] / sum;
                    for (out, vc[s * d + off ..][0..cfg.head_dim]) |*o, vv|
                    {
                        o.* += p * vv;
                    }
                }
            }
        }
        l.o.matmul(att, n, tmp, scratch);
        for (h, tmp) |*hh, o|
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
        }
        if (la) |p|
        {
            scoreGuesses(p, guess, idx, li, n, k, kx, horizon, cfg.layers);
        }
        @memset(tmp, 0);
        try moe.run(moe.ctx, li, x, n, idx, wts, tmp);
        for (h, tmp) |*hh, o|
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

    fn runAll(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void {
        const self: *LocalExperts = @ptrCast(@alignCast(ctx));
        const d = self.cfg.d_model;
        const k = self.cfg.top_k;
        for (0..n) |t| {
            for (0..k) |j| {
                const e = self.get(layer, idx[t * k + j]) orelse return error.ExpertNotLocal;
                e.forward(x[t * d ..][0..d], self.out, self.scratch);
                for (out[t * d ..][0..d], self.out) |*o, y| o.* += w[t * k + j] * y;
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
    rope(&x, 2, 4, 1, 10000);
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)), x[0], 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)), x[2], 1e-6);
}

test "topK keeps order and breaks ties toward the lower index" {
    var idx: [3]u32 = undefined;
    var val: [3]f32 = undefined;
    topK(&.{ 0.1, 0.3, 0.3, 0.05, 0.25 }, 3, &idx, &val);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 4 }, &idx);
}
