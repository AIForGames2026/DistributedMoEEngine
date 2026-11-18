//! DeepSeek V3 and Kimi K2 (TASK-39, transformers' deepseek_v3, GGUF "deepseek2"),
//! run by `olmoe.step` through `step` here. Each layer's attention is MLA: keys and
//! values come from one latent per position, plus a small key part shared by every
//! head that rotates (decoupled rope). The cache keeps each position's normalized
//! latent then its rotated key part, and attention runs on them with each head's
//! key and value projections folded in, as llama.cpp does. Routing scores experts
//! by a sigmoid, chooses by score plus a bias among the best groups of experts, and
//! scales the chosen scores; the leading layers are dense and the others add shared
//! experts.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const glm53 = @import("glm53.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const pool = @import("pool.zig");
const quant = @import("quant.zig");

const Config = olmoe.Config;
const Mat = olmoe.Mat;
const Expert = olmoe.Expert;

/// Widest latent plus rotated part, and most experts and groups, this file handles.
pub const max_row = 2048;
pub const max_experts = 1024;
pub const max_groups = 64;

pub fn config(h: gguf.Header) !Config
{
    const K = struct {
        hh: gguf.Header,
        buf: [128]u8 = undefined,

        fn key(s: *@This(), k: []const u8) []const u8
        {
            return std.fmt.bufPrint(&s.buf, "deepseek2.{s}", .{k}) catch unreachable;
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

        fn float(s: *@This(), k: []const u8, default: ?f32) !f32
        {
            const v = s.hh.get(s.key(k)) orelse return default orelse
            {
                std.log.err("missing GGUF key {s}", .{s.key(k)});
                return error.MissingKey;
            };
            return switch (v) {
                .float => |x| @floatCast(x),
                else => error.MissingKey,
            };
        }
    };
    var g: K = .{ .hh = h };
    const embd = h.tensor("token_embd.weight") orelse return error.MissingTensor;
    const kvr = try g.int("attention.kv_lora_rank");
    const rope = try g.int("rope.dimension_count");
    const d_ff = try g.int("expert_feed_forward_length");
    var cfg: Config = .{
        .arch = .deepseek2,
        .layers = try g.int("block_count"),
        .d_model = try g.int("embedding_length"),
        .heads = try g.int("attention.head_count"),
        // The cache keeps one row per position: the latent, then the rotated key part.
        .kv_heads = 1,
        .head_dim = kvr + rope,
        .d_ff = d_ff,
        .experts = try g.int("expert_count"),
        .top_k = try g.int("expert_used_count"),
        .vocab = @intCast(embd.dims[1]),
        .eps = try g.float("attention.layer_norm_rms_epsilon", null),
        .rope_theta = try g.float("rope.freq_base", null),
        .rope_dim = rope,
        .rope_adjacent = true,
        .dense_layers = g.opt("leading_dense_block_count") orelse 0,
        .dense_ff = try g.int("feed_forward_length"),
        .mla_q_rank = g.opt("attention.q_lora_rank") orelse 0,
        .mla_kv_rank = kvr,
        .mla_k = try g.int("attention.key_length_mla"),
        .mla_v = try g.int("attention.value_length_mla"),
        .shared_ff = d_ff * try g.int("expert_shared_count"),
        .route_scale = try g.float("expert_weights_scale", 1),
        .route_groups = g.opt("expert_group_count") orelse 1,
        .route_groups_used = g.opt("expert_group_used_count") orelse 1,
        .route_norm = h.getBool("deepseek2.expert_weights_norm") orelse false,
    };
    cfg.attn_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.mla_k)));
    if (h.getString("deepseek2.rope.scaling.type")) |kind|
    {
        // YaRN as DeepSeek V3 sets it (mscale equal to mscale_all_dim), so cos and sin keep their size.
        if (!std.mem.eql(u8, kind, "yarn"))
        {
            return error.Unsupported;
        }
        const factor = try g.float("rope.scaling.factor", null);
        cfg.yarn = factor;
        cfg.rope_original = try g.int("rope.scaling.original_context_length");
        // The converter stores 0.1 times mscale_all_dim.
        const m = try g.float("rope.scaling.yarn_log_multiplier", 0) * @log(factor) + 1;
        cfg.attn_scale *= m * m;
    }
    const ok = (g.opt("expert_gating_func") orelse 0) == 2 and
        cfg.dense_layers < cfg.layers and cfg.shared_ff > 0 and
        cfg.experts <= max_experts and cfg.route_groups <= max_groups and
        cfg.route_groups > 0 and cfg.experts % cfg.route_groups == 0 and cfg.route_groups_used <= cfg.route_groups and
        cfg.experts / cfg.route_groups >= 2 and cfg.top_k <= 64 and
        cfg.head_dim <= max_row and rope % 2 == 0 and rope <= cfg.mla_k and
        cfg.mla_k - rope <= max_row and cfg.mla_v <= max_row and
        h.tensor("blk.0.attn_k_b.weight") != null;
    if (!ok)
    {
        // DeepSeek V2's softmax router and GGUFs with an unsplit attn_kv_b are not handled.
        return error.Unsupported;
    }
    return cfg;
}

/// A DeepSeek V3 layer beyond `olmoe.Layer`: without a query rank `q_b` is the whole query projection.
/// `k_b` holds each head's key projection transposed, `v_b` each head's value projection.
pub const Parts = struct {
    q_a: Mat = Mat.empty,
    q_a_norm: []f32 = &.{},
    q_b: Mat,
    kv_a: Mat,
    kv_a_norm: []f32,
    k_b: Mat,
    v_b: Mat,
    out: Mat,
    /// The dense feed-forward block of a leading layer, or the shared experts of an MoE layer.
    mlp: Expert,
    /// The router's choice bias, in an MoE layer; empty for none.
    bias: []f32 = &.{},
};

const Tensors = struct {
    h: gguf.Header,
    plan: loader.Plan,
    region: []const u8,
    a: std.mem.Allocator,

    fn info(s: *Tensors, n: []const u8) !gguf.TensorInfo
    {
        return s.h.tensor(n) orelse
        {
            std.log.err("missing tensor {s}", .{n});
            return error.MissingTensor;
        };
    }

    /// `rows` rows of `cols`: the innermost dimension is `cols`, the others multiply to `rows`.
    fn mat(s: *Tensors, n: []const u8, rows: usize, cols: usize) !Mat
    {
        const t = try s.info(n);
        var outer: u64 = 1;
        for (t.dims[1..t.ndim]) |dim|
        {
            outer *= dim;
        }
        if (t.dims[0] != cols or outer != rows)
        {
            std.log.err("{s}: dims {any}, expected {d} rows of {d}", .{ n, t.dims[0..t.ndim], rows, cols });
            return error.BadShape;
        }
        return .{ .format = t.format, .rows = rows, .cols = cols, .bytes = s.plan.locate(s.region, .{ .offset = t.offset, .len = t.bytes }) };
    }

    fn flat(s: *Tensors, n: []const u8, count: usize) ![]f32
    {
        const t = try s.info(n);
        if (t.elements() != count)
        {
            std.log.err("{s}: dims {any}, expected {d} values", .{ n, t.dims[0..t.ndim], count });
            return error.BadShape;
        }
        const out = try s.a.alloc(f32, count);
        try quant.dequantize(t.format, s.plan.locate(s.region, .{ .offset = t.offset, .len = t.bytes }), out);
        return out;
    }

    fn lmat(s: *Tensors, layer: usize, n: []const u8, rows: usize, cols: usize) !Mat
    {
        var b: [96]u8 = undefined;
        return s.mat(std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ layer, n }) catch unreachable, rows, cols);
    }

    fn lflat(s: *Tensors, layer: usize, n: []const u8, count: usize) ![]f32
    {
        var b: [96]u8 = undefined;
        return s.flat(std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ layer, n }) catch unreachable, count);
    }

    fn has(s: *Tensors, layer: usize, n: []const u8) bool
    {
        var b: [96]u8 = undefined;
        return s.h.tensor(std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ layer, n }) catch unreachable) != null;
    }
};

/// The non-expert weights, as `olmoe.Weights` (whose `init` calls this for the arch).
pub fn weights(gpa: std.mem.Allocator, h: gguf.Header, cfg: Config, plan: loader.Plan, region: []const u8) !olmoe.Weights
{
    var self: olmoe.Weights = .{ .cfg = cfg, .embed = undefined, .layers = undefined, .out_norm = undefined, .head = undefined, .arena = .init(gpa) };
    errdefer self.arena.deinit();
    const a = self.arena.allocator();
    var t: Tensors = .{ .h = h, .plan = plan, .region = region, .a = a };
    const d = cfg.d_model;
    const heads = cfg.heads;
    const kvr = cfg.mla_kv_rank;
    const nope = cfg.mla_k - cfg.rope_dim;
    self.embed = try t.mat("token_embd.weight", cfg.vocab, d);
    self.head = try t.mat("output.weight", cfg.vocab, d);
    self.out_norm = try t.flat("output_norm.weight", d);
    self.layers = try a.alloc(olmoe.Layer, cfg.layers);
    for (self.layers, 0..) |*l, i|
    {
        const dense = i < cfg.dense_layers;
        var parts: Parts = .{
            .q_b = undefined,
            .kv_a = try t.lmat(i, "attn_kv_a_mqa.weight", cfg.head_dim, d),
            .kv_a_norm = try t.lflat(i, "attn_kv_a_norm.weight", kvr),
            .k_b = try t.lmat(i, "attn_k_b.weight", heads * kvr, nope),
            .v_b = try t.lmat(i, "attn_v_b.weight", heads * cfg.mla_v, kvr),
            .out = try t.lmat(i, "attn_output.weight", d, heads * cfg.mla_v),
            .mlp = if (dense) .{
                .gate = try t.lmat(i, "ffn_gate.weight", cfg.dense_ff, d),
                .up = try t.lmat(i, "ffn_up.weight", cfg.dense_ff, d),
                .down = try t.lmat(i, "ffn_down.weight", d, cfg.dense_ff),
            } else .{
                .gate = try t.lmat(i, "ffn_gate_shexp.weight", cfg.shared_ff, d),
                .up = try t.lmat(i, "ffn_up_shexp.weight", cfg.shared_ff, d),
                .down = try t.lmat(i, "ffn_down_shexp.weight", d, cfg.shared_ff),
            },
        };
        if (cfg.mla_q_rank > 0)
        {
            parts.q_a = try t.lmat(i, "attn_q_a.weight", cfg.mla_q_rank, d);
            parts.q_a_norm = try t.lflat(i, "attn_q_a_norm.weight", cfg.mla_q_rank);
            parts.q_b = try t.lmat(i, "attn_q_b.weight", heads * cfg.mla_k, cfg.mla_q_rank);
        }
        else
        {
            parts.q_b = try t.lmat(i, "attn_q.weight", heads * cfg.mla_k, d);
        }
        if (!dense and t.has(i, "exp_probs_b.bias"))
        {
            parts.bias = try t.lflat(i, "exp_probs_b.bias", cfg.experts);
        }
        l.* = .{
            .attn_norm = try t.lflat(i, "attn_norm.weight", d),
            .ffn_norm = try t.lflat(i, "ffn_norm.weight", d),
            .router = if (dense) Mat.empty else try t.lmat(i, "ffn_gate_inp.weight", cfg.experts, d),
            .ds = parts,
        };
    }
    return self;
}

/// Buffers for one step, sized for `n` tokens.
const Buffers = struct {
    qa: []f32,
    q: []f32,
    nope: []f32,
    absorbed: []f32,
    qfull: []f32,
    rows: []f32,
    lsum: []f32,
    att: []f32,
    gather: []f32,
    gather_out: []f32,

    fn init(a: std.mem.Allocator, cfg: Config, n: usize) !Buffers
    {
        const heads = cfg.heads;
        const kvr = cfg.mla_kv_rank;
        const nope = cfg.mla_k - cfg.rope_dim;
        return .{
            .qa = try a.alloc(f32, n * cfg.mla_q_rank),
            .q = try a.alloc(f32, n * heads * cfg.mla_k),
            .nope = try a.alloc(f32, n * heads * nope),
            .absorbed = try a.alloc(f32, n * heads * kvr),
            .qfull = try a.alloc(f32, n * heads * cfg.head_dim),
            .rows = try a.alloc(f32, n * cfg.head_dim),
            .lsum = try a.alloc(f32, n * heads * kvr),
            .att = try a.alloc(f32, n * heads * cfg.mla_v),
            .gather = try a.alloc(f32, n * @max(nope, kvr)),
            .gather_out = try a.alloc(f32, n * @max(kvr, cfg.mla_v)),
        };
    }
};

/// One MLA layer on `n` normalized inputs `x` from position `pos0`; each position's
/// latent and rotated key part go into the cache's layer `li`.
fn mla(b: *const Buffers, cfg: Config, w: Parts, rope: *const olmoe.Rope, cs: []const f32, sn: []const f32, cache: *olmoe.KvCache, li: usize, x: []const f32, n: usize, pos0: usize, out: []f32, scratch: []f32) void
{
    const heads = cfg.heads;
    const kvr = cfg.mla_kv_rank;
    const r = cfg.rope_dim;
    const nope = cfg.mla_k - r;
    const row = cfg.head_dim;
    const half = rope.half;
    const q = b.q[0 .. n * heads * cfg.mla_k];
    if (cfg.mla_q_rank > 0)
    {
        const qa = b.qa[0 .. n * cfg.mla_q_rank];
        w.q_a.matmul(x, n, qa, scratch);
        for (0..n) |t|
        {
            const v = qa[t * cfg.mla_q_rank ..][0..cfg.mla_q_rank];
            olmoe.rmsnorm(v, w.q_a_norm, cfg.eps, v);
        }
        w.q_b.matmul(qa, n, q, scratch);
    }
    else
    {
        w.q_b.matmul(x, n, q, scratch);
    }
    const rows = b.rows[0 .. n * row];
    w.kv_a.matmul(x, n, rows, scratch);
    for (0..n) |t|
    {
        const v = rows[t * row ..][0..row];
        olmoe.rmsnorm(v[0..kvr], w.kv_a_norm, cfg.eps, v[0..kvr]);
        rope.applyAt(v[kvr..][0..r], 1, r, cs[t * half ..][0..half], sn[t * half ..][0..half]);
        for (0..heads) |hh|
        {
            const qh = q[(t * heads + hh) * cfg.mla_k ..][0..cfg.mla_k];
            @memcpy(b.nope[(t * heads + hh) * nope ..][0..nope], qh[0..nope]);
            rope.applyAt(qh[nope..][0..r], 1, r, cs[t * half ..][0..half], sn[t * half ..][0..half]);
        }
    }
    cache.store(li, pos0, n, rows, &.{});
    // Each head's query against the latent: its key projection folded into the query.
    glm53.perHead(w.k_b, heads, kvr, b.nope, n, b.absorbed, b.gather, b.gather_out, scratch);
    for (0..n * heads) |i|
    {
        const dst = b.qfull[i * row ..][0..row];
        @memcpy(dst[0..kvr], b.absorbed[i * kvr ..][0..kvr]);
        @memcpy(dst[kvr..][0..r], q[i * cfg.mla_k + nope ..][0..r]);
    }
    const groups = heads / headGroup(heads);
    pool.parallelFor(olmoe.cpuThreads(), n * groups, Attend{ .cfg = cfg, .q = b.qfull, .rows = cache.keysF32(li), .pos0 = pos0, .lsum = b.lsum }, Attend.run);
    // Each head's weighted latent through its value projection.
    glm53.perHead(w.v_b, heads, cfg.mla_v, b.lsum, n, b.att, b.gather, b.gather_out, scratch);
    w.out.matmul(b.att[0 .. n * heads * cfg.mla_v], n, out, scratch);
}

/// Heads that attend together, reading each cached row once: up to 8.
fn headGroup(heads: usize) usize
{
    var g: usize = 8;
    while (heads % g != 0)
    {
        g /= 2;
    }
    return g;
}

/// Causal attention for work items `first` to `end`, each a token and a group of heads:
/// a softmax over every earlier position per head, then the latents weighted by it into `lsum`.
const Attend = struct {
    cfg: Config,
    q: []const f32,
    rows: []const f32,
    pos0: usize,
    lsum: []f32,

    fn run(self: @This(), first: usize, end: usize) void
    {
        const c = self.cfg;
        const heads = c.heads;
        const kvr = c.mla_kv_rank;
        const row = c.head_dim;
        const gs = headGroup(heads);
        const groups = heads / gs;
        const most = self.pos0 + (end - 1) / groups + 1;
        const scores = std.heap.page_allocator.alloc(f32, gs * most) catch @panic("out of memory in MLA attention");
        defer std.heap.page_allocator.free(scores);
        for (first..end) |item|
        {
            const t = item / groups;
            const h0 = (item % groups) * gs;
            const len = self.pos0 + t + 1;
            const q = self.q[(t * heads + h0) * row ..][0 .. gs * row];
            var mx: [8]f32 = @splat(-std.math.inf(f32));
            for (0..len) |s|
            {
                const r = self.rows[s * row ..][0..row];
                for (0..gs) |g|
                {
                    const sc = olmoe.dot(q[g * row ..][0..row], r) * c.attn_scale;
                    scores[g * most + s] = sc;
                    mx[g] = @max(mx[g], sc);
                }
            }
            var sum: [8]f32 = @splat(0);
            for (0..gs) |g|
            {
                for (scores[g * most ..][0..len]) |*sc|
                {
                    sc.* = @exp(sc.* - mx[g]);
                    sum[g] += sc.*;
                }
            }
            const out = self.lsum[(t * heads + h0) * kvr ..][0 .. gs * kvr];
            @memset(out, 0);
            for (0..len) |s|
            {
                const r = self.rows[s * row ..][0..kvr];
                for (0..gs) |g|
                {
                    glm53.axpy(out[g * kvr ..][0..kvr], scores[g * most + s] / sum[g], r);
                }
            }
        }
    }
};

/// Sigmoid scores; the top k by score plus bias among the best groups (each scored by its
/// two best), as transformers does it; their scores renormalized if set, then scaled.
fn route(cfg: Config, bias: []const f32, logits: []f32, n: usize, idx: []u32, wts: []f32, gaps: []f32) void
{
    const k = cfg.top_k;
    const e = cfg.experts;
    const per = e / cfg.route_groups;
    var choice_buf: [max_experts]f32 = undefined;
    var gscore: [max_groups]f32 = undefined;
    var gidx: [max_groups]u32 = undefined;
    var val: [64]f32 = undefined;
    for (0..n) |t|
    {
        const s = logits[t * e ..][0..e];
        const choice = choice_buf[0..e];
        for (s, choice, 0..) |*sv, *cv, ex|
        {
            sv.* = 1.0 / (1.0 + @exp(-sv.*));
            cv.* = sv.* + if (bias.len > 0) bias[ex] else 0;
        }
        if (cfg.route_groups_used < cfg.route_groups)
        {
            for (0..cfg.route_groups) |gi|
            {
                var top2: [2]u32 = undefined;
                var top2v: [2]f32 = undefined;
                olmoe.topK(choice[gi * per ..][0..per], 2, &top2, &top2v);
                gscore[gi] = top2v[0] + top2v[1];
            }
            olmoe.topK(gscore[0..cfg.route_groups], cfg.route_groups_used, gidx[0..cfg.route_groups_used], val[0..cfg.route_groups_used]);
            // Experts outside the chosen groups score 0, not minus infinity, as in transformers.
            for (0..cfg.route_groups) |gi|
            {
                if (std.mem.indexOfScalar(u32, gidx[0..cfg.route_groups_used], @intCast(gi)) == null)
                {
                    @memset(choice[gi * per ..][0..per], 0);
                }
            }
        }
        const it = idx[t * k ..][0..k];
        olmoe.topK(choice, k, it, val[0..k]);
        if (gaps.len > 0)
        {
            var next: f32 = -std.math.inf(f32);
            for (choice, 0..) |cv, ex|
            {
                if (cv > next and std.mem.indexOfScalar(u32, it, @intCast(ex)) == null)
                {
                    next = cv;
                }
            }
            gaps[t] = val[k - 1] - next;
        }
        const wt = wts[t * k ..][0..k];
        var sum: f32 = 0;
        for (it, wt) |ex, *wv|
        {
            wv.* = s[ex];
            sum += wv.*;
        }
        for (wt) |*wv|
        {
            if (cfg.route_norm)
            {
                wv.* /= sum + 1e-20;
            }
            wv.* *= cfg.route_scale;
        }
    }
}

/// `olmoe.step` for DeepSeek V3.
pub fn step(gpa: std.mem.Allocator, w: *const olmoe.Weights, moe: olmoe.Moe, cache: *olmoe.KvCache, tokens: []const u32, logits: []f32, trace: ?olmoe.Trace) !void
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
    if (logits.len < cfg.vocab)
    {
        return error.LogitsTooSmall;
    }
    if (cache.tier != null or cache.format != .f32)
    {
        return error.Unsupported;
    }
    const all_logits = logits.len >= n * cfg.vocab;
    if (moe.mark) |m|
    {
        m(moe.ctx, true, n);
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var rope: olmoe.Rope = .init(cfg);
    // DeepSeek's YaRN leaves cos and sin unscaled; its size goes into `attn_scale`.
    rope.scale = 1;
    const cs = try a.alloc(f32, n * rope.half);
    const sn = try a.alloc(f32, n * rope.half);
    for (0..n) |t|
    {
        rope.angles(pos0 + t, cs[t * rope.half ..][0..rope.half], sn[t * rope.half ..][0..rope.half]);
    }
    const h = try a.alloc(f32, n * d);
    const x = try a.alloc(f32, n * d);
    const y = try a.alloc(f32, n * d);
    const tmp = try a.alloc(f32, n * d);
    const big = @max(@max(cfg.dense_ff, cfg.shared_ff), cfg.d_ff);
    const g = try a.alloc(f32, n * big);
    const u = try a.alloc(f32, n * big);
    const scratch = try a.alloc(f32, @max(@max(d, cfg.heads * cfg.mla_v), @max(big, @max(cfg.mla_q_rank, cfg.heads * cfg.mla_k))));
    const probs = try a.alloc(f32, n * cfg.experts);
    const idx = try a.alloc(u32, n * k);
    const wts = try a.alloc(f32, n * k);
    const cap_count = try a.alloc(u32, if (moe.capacity != null) cfg.experts else 0);
    const cap_list = try a.alloc(u32, if (moe.capacity != null) n * k else 0);
    const bufs = try Buffers.init(a, cfg, n);

    for (tokens, 0..) |tok, t|
    {
        if (tok >= cfg.vocab)
        {
            return error.BadToken;
        }
        w.embed.row(tok, h[t * d ..][0..d]);
    }
    const threads = if (n >= olmoe.parallel_tokens) olmoe.cpuThreads() else 1;
    const prof = &olmoe.profile;
    for (w.layers, 0..) |*l, li|
    {
        const parts = l.ds.?;
        var pt = olmoe.nowNs();
        @memcpy(x, h);
        glm53.normRows(threads, x, l.attn_norm, cfg.eps, d, n);
        mla(&bufs, cfg, parts, &rope, cs, sn, cache, li, x, n, pos0, y, scratch);
        olmoe.Profile.lap(&prof.attention, &pt);
        for (h, y) |*hv, yv|
        {
            hv.* += yv;
        }
        @memcpy(x, h);
        glm53.normRows(threads, x, l.ffn_norm, cfg.eps, d, n);
        @memset(y, 0);
        if (li < cfg.dense_layers)
        {
            glm53.mlp(parts.mlp, 0, x, n, g, u, tmp, y, scratch);
            olmoe.Profile.lap(&prof.experts, &pt);
        }
        else
        {
            l.router.matmul(x, n, probs, scratch);
            const gaps: []f32 = if (trace) |tr| (if (tr.gaps.len > 0) tr.gaps[li * n ..][0..n] else &.{}) else &.{};
            route(cfg, parts.bias, probs, n, idx, wts, gaps);
            if (moe.capacity) |cp|
            {
                cp.apply(idx, wts, n * k, cap_count, cap_list);
            }
            olmoe.Profile.lap(&prof.router, &pt);
            try moe.run(moe.ctx, li, x, n, idx, wts, y);
            glm53.mlp(parts.mlp, 0, x, n, g, u, tmp, y, scratch);
            olmoe.Profile.lap(&prof.experts, &pt);
            if (trace) |tr|
            {
                @memcpy(tr.topk[li * n * k ..][0 .. n * k], idx);
            }
        }
        for (h, y) |*hv, yv|
        {
            hv.* += yv;
        }
        if (!olmoe.allFinite(h))
        {
            std.log.err("step: a non-finite value in the residual after layer {d} (positions {d} to {d})", .{ li, pos0, pos0 + n - 1 });
            return error.NonFinite;
        }
        if (trace) |tr|
        {
            if (tr.hidden.len > 0)
            {
                @memcpy(tr.hidden[li * n * d ..][0 .. n * d], h);
            }
        }
    }
    cache.len = pos0 + n;
    var head_t = olmoe.nowNs();
    defer olmoe.Profile.lap(&prof.head, &head_t);
    const first: usize = if (all_logits) 0 else n - 1;
    const rows = n - first;
    for (first..n) |t|
    {
        olmoe.rmsnorm(h[t * d ..][0..d], w.out_norm, cfg.eps, x[(t - first) * d ..][0..d]);
    }
    w.head.matmul(x, rows, logits, scratch);
    if (olmoe.after_step) |f|
    {
        f();
    }
    if (moe.mark) |m|
    {
        m(moe.ctx, false, n);
    }
}
