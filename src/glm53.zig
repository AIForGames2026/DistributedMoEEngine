//! GLM-5.3-Flash (TASK-32, transformers' glm5_next, GGUF "glm5-next"), whose layers
//! `olmoe.step` runs with the parts here. Its blocks differ from the Qwen models':
//!
//! - mHC: the residual is several streams. Before each block a projection of
//!   all of them gives the block's input mix, its output weights, and a stream
//!   mix made doubly stochastic by Sinkhorn iterations (`hcPre`, `hcPost`).
//! - KDA layers: Gated DeltaNet with a decay per key channel from a low-rank
//!   forget gate, and an output norm gated by a low-rank projection (`kda`).
//! - MLA layers: queries through a low rank, keys and values from one latent
//!   per token, no rope. The cache keeps the latent, and attention runs on it
//!   with each head's key and value projections folded in (`mla`).
//! - DSA: an indexer scores pools of tokens (each pooled by a learned softmax)
//!   and each query keeps the best pools plus the unfinished last one.
//! - A sigmoid router with a bias used only for choosing (`deepseek.route`), dense
//!   leading layers, and SwiGLU inputs clamped.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const quant = @import("quant.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const pool = @import("pool.zig");

const Config = olmoe.Config;
const Mat = olmoe.Mat;
const Expert = olmoe.Expert;

/// Most residual streams, MLA latent width, and DSA indexer heads this file handles.
pub const max_streams = 8;
pub const max_latent = 1024;
pub const max_dsa_heads = 64;

fn silu(x: f32) f32
{
    return x / (1.0 + @exp(-x));
}

fn sigmoid(x: f32) f32
{
    return 1.0 / (1.0 + @exp(-x));
}

fn tokenThreads(n: usize) usize
{
    return if (n >= olmoe.parallel_tokens) olmoe.cpuThreads() else 1;
}

pub fn config(h: gguf.Header) !Config
{
    const K = struct {
        hh: gguf.Header,
        buf: [128]u8 = undefined,

        fn key(s: *@This(), k: []const u8) []const u8
        {
            return std.fmt.bufPrint(&s.buf, "glm5-next.{s}", .{k}) catch unreachable;
        }

        fn int(s: *@This(), k: []const u8) !usize
        {
            return @intCast(s.hh.getInt(s.key(k)) orelse
            {
                std.log.err("missing GGUF key {s}", .{s.key(k)});
                return error.MissingKey;
            });
        }

        fn float(s: *@This(), k: []const u8) !f32
        {
            const v = s.hh.get(s.key(k)) orelse
            {
                std.log.err("missing GGUF key {s}", .{s.key(k)});
                return error.MissingKey;
            };
            return switch (v) {
                .float => |x| @floatCast(x),
                else => error.MissingKey,
            };
        }

        fn flag(s: *@This(), k: []const u8) bool
        {
            const v = s.hh.get(s.key(k)) orelse return false;
            return switch (v) {
                .boolean => |b| b,
                else => false,
            };
        }

        fn floatAt(s: *@This(), k: []const u8, i: usize) ?f32
        {
            const v = s.hh.get(s.key(k)) orelse return null;
            const arr = switch (v) {
                .array => |a| a,
                else => return null,
            };
            if (arr.elem != .f32 or i >= arr.len)
            {
                return null;
            }
            return @bitCast(std.mem.readInt(u32, arr.data[i * 4 ..][0..4], .little));
        }
    };
    var g: K = .{ .hh = h };
    const embd = h.tensor("token_embd.weight") orelse return error.MissingTensor;
    const heads = try g.int("attention.head_count");
    const kda_dim = try g.int("kda.head_dim");
    var cfg: Config = .{
        .layers = try g.int("block_count"),
        .d_model = try g.int("embedding_length"),
        .heads = heads,
        // The cache keeps one MLA latent per position in place of keys.
        .kv_heads = 1,
        .head_dim = try g.int("attention.kv_lora_rank"),
        .d_ff = try g.int("expert_feed_forward_length"),
        .experts = try g.int("expert_count"),
        .top_k = try g.int("expert_used_count"),
        .vocab = @intCast(embd.dims[1]),
        .eps = try g.float("attention.layer_norm_rms_epsilon"),
        .rope_theta = 0,
        .attn_interval = 4,
        .ssm_conv = try g.int("ssm.conv_kernel"),
        .ssm_k_heads = heads,
        .ssm_dk = kda_dim,
        .ssm_v_heads = heads,
        .ssm_dv = kda_dim,
        .streams = try g.int("hyper_connection.count"),
        .dense_layers = try g.int("leading_dense_block_count"),
        .dense_ff = try g.int("feed_forward_length"),
        .kda_lower = try g.float("kda.gate_lower_bound"),
        .mla_q_rank = try g.int("attention.q_lora_rank"),
        .mla_kv_rank = try g.int("attention.kv_lora_rank"),
        .mla_k = try g.int("attention.key_length_mla"),
        .mla_v = try g.int("attention.value_length_mla"),
        .dsa_heads = try g.int("attention.indexer.head_count"),
        .dsa_dim = try g.int("attention.indexer.key_length"),
        .dsa_budget = try g.int("attention.indexer.top_k"),
        .dsa_pool = try g.int("attention.indexer.kpool"),
        .hc_iters = try g.int("hyper_connection.sinkhorn_iterations"),
        .hc_eps = try g.float("hyper_connection.epsilon"),
        .route_scale = try g.float("expert_weights_scale"),
    };
    cfg.shared_ff = cfg.d_ff * try g.int("expert_shared_count");
    // MLA on every fourth layer (the cache's attention layers), KDA elsewhere, as released.
    for (0..cfg.layers) |l|
    {
        const kv = h.intAt(g.key("attention.head_count_kv"), l) orelse return error.MissingKey;
        if ((kv != 0) == cfg.recurrent(l))
        {
            return error.Unsupported;
        }
        if (h.intAt(g.key("attention.indexer.types"), l) != 1)
        {
            return error.Unsupported;
        }
    }
    cfg.swiglu_limit = g.floatAt("swiglu_clamp_exp", 0) orelse 0;
    for (0..cfg.layers) |l|
    {
        if ((g.floatAt("swiglu_clamp_exp", l) orelse 0) != cfg.swiglu_limit or (g.floatAt("swiglu_clamp_shexp", l) orelse 0) != cfg.swiglu_limit)
        {
            return error.Unsupported;
        }
    }
    const ok = (g.hh.getInt(g.key("rope.dimension_count")) orelse 0) == 0 and
        try g.int("expert_gating_func") == 2 and
        try g.int("expert_group_count") == 1 and
        g.flag("expert_weights_norm") and
        g.flag("attention.indexer.kpool_select_tail") and
        cfg.streams >= 2 and cfg.streams <= max_streams and
        cfg.dense_layers < cfg.layers and
        cfg.dsa_pool > 0 and cfg.dsa_budget % cfg.dsa_pool == 0 and cfg.dsa_heads <= max_dsa_heads and
        cfg.mla_kv_rank <= max_latent and cfg.ssm_dk <= 256 and cfg.shared_ff > 0;
    if (!ok)
    {
        return error.Unsupported;
    }
    return cfg;
}

/// One mHC mix: a projection (`(2 + streams) * streams` rows over every stream), its bias, and three scales.
pub const Hc = struct {
    proj: Mat,
    base: []f32,
    scale: []f32,
};

/// A KDA layer's weights. `a` holds -exp(A_log) per head; `conv` holds q, k, then v channels, oldest tap first.
pub const Kda = struct {
    q: Mat,
    k: Mat,
    v: Mat,
    conv: []f32,
    f_a: Mat,
    f_b: Mat,
    g_a: Mat,
    g_b: Mat,
    beta: Mat,
    a: []f32,
    dt_bias: []f32,
    norm: []f32,
    out: Mat,
};

/// An MLA layer's weights and its DSA indexer. `k_b` is each head's key projection
/// transposed (heads x latent rows of key-size), `v_b` each head's value projection.
pub const Mla = struct {
    q_a: Mat,
    q_a_norm: []f32,
    q_b: Mat,
    kv_a: Mat,
    kv_a_norm: []f32,
    k_b: Mat,
    v_b: Mat,
    out: Mat,
    iq: Mat,
    ik: Mat,
    ik_norm: []f32,
    ik_bias: []f32,
    iw: Mat,
    ape: []f32,
    gate: Mat,
};

const Tensors = struct {
    h: gguf.Header,
    plan: loader.Plan,
    region: []const u8,
    a: std.mem.Allocator,
    buf: [96]u8 = undefined,

    fn name(s: *Tensors, layer: usize, n: []const u8) []const u8
    {
        return std.fmt.bufPrint(&s.buf, "blk.{d}.{s}", .{ layer, n }) catch unreachable;
    }

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

    /// A tensor of `count` values as floats.
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
        const full = std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ layer, n }) catch unreachable;
        return s.mat(full, rows, cols);
    }

    fn lflat(s: *Tensors, layer: usize, n: []const u8, count: usize) ![]f32
    {
        var b: [96]u8 = undefined;
        const full = std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ layer, n }) catch unreachable;
        return s.flat(full, count);
    }

    fn hc(s: *Tensors, cfg: Config, layer: usize, prefix: []const u8) !Hc
    {
        var b: [64]u8 = undefined;
        const st = cfg.streams;
        return .{
            .proj = try s.lmat(layer, std.fmt.bufPrint(&b, "{s}fn.weight", .{prefix}) catch unreachable, (2 + st) * st, cfg.residual()),
            .base = try s.lflat(layer, std.fmt.bufPrint(&b, "{s}base.weight", .{prefix}) catch unreachable, (2 + st) * st),
            .scale = try s.lflat(layer, std.fmt.bufPrint(&b, "{s}scale.weight", .{prefix}) catch unreachable, 3),
        };
    }
};

/// The non-expert weights, as `olmoe.Weights` (`families.zig` names this for the architecture).
pub fn weights(gpa: std.mem.Allocator, h: gguf.Header, cfg: Config, plan: loader.Plan, region: []const u8) anyerror!olmoe.Weights
{
    var self: olmoe.Weights = .{ .cfg = cfg, .embed = undefined, .layers = undefined, .out_norm = undefined, .head = undefined, .arena = .init(gpa) };
    errdefer self.arena.deinit();
    const a = self.arena.allocator();
    var t: Tensors = .{ .h = h, .plan = plan, .region = region, .a = a };
    const d = cfg.d_model;
    const inner = cfg.ssm_v_heads * cfg.ssm_dv;
    self.embed = try t.mat("token_embd.weight", cfg.vocab, d);
    self.head = try t.mat("output.weight", cfg.vocab, d);
    self.out_norm = try t.flat("output_norm.weight", d);
    self.layers = try a.alloc(olmoe.Layer, cfg.layers);
    for (self.layers, 0..) |*l, i|
    {
        const dense = i < cfg.dense_layers;
        l.* = .{
            .attn_norm = try t.lflat(i, "attn_norm.weight", d),
            .ffn_norm = try t.lflat(i, "ffn_norm.weight", d),
            .router = if (dense) Mat.empty else try t.lmat(i, "ffn_gate_inp.weight", cfg.experts, d),
            .mhc_attn = try t.hc(cfg, i, "hc_attn_"),
            .mhc_ffn = try t.hc(cfg, i, "hc_ffn_"),
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
        if (!dense)
        {
            l.route_bias = try t.lflat(i, "exp_probs_b.bias", cfg.experts);
        }
        if (cfg.recurrent(i))
        {
            const taps = cfg.ssm_conv;
            const conv = try a.alloc(f32, 3 * inner * taps);
            for ([_][]const u8{ "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight" }, 0..) |cn, j|
            {
                const part = try t.lflat(i, cn, inner * taps);
                @memcpy(conv[j * inner * taps ..][0 .. inner * taps], part);
            }
            const rank = (try t.info(t.name(i, "ssm_f_a.weight"))).dims[1];
            l.kda = .{
                .q = try t.lmat(i, "attn_q.weight", inner, d),
                .k = try t.lmat(i, "attn_k.weight", inner, d),
                .v = try t.lmat(i, "attn_v.weight", inner, d),
                .conv = conv,
                .f_a = try t.lmat(i, "ssm_f_a.weight", @intCast(rank), d),
                .f_b = try t.lmat(i, "ssm_f_b.weight", inner, @intCast(rank)),
                .g_a = try t.lmat(i, "ssm_g_a.weight", @intCast(rank), d),
                .g_b = try t.lmat(i, "ssm_g_b.weight", inner, @intCast(rank)),
                .beta = try t.lmat(i, "ssm_beta.weight", cfg.ssm_v_heads, d),
                .a = try t.lflat(i, "ssm_a", cfg.ssm_v_heads),
                .dt_bias = try t.lflat(i, "ssm_dt.bias", inner),
                .norm = try t.lflat(i, "ssm_norm.weight", cfg.ssm_dv),
                .out = try t.lmat(i, "attn_output.weight", d, inner),
            };
        }
        else
        {
            const heads = cfg.heads;
            const kvr = cfg.mla_kv_rank;
            l.mla_dsa = .{
                .q_a = try t.lmat(i, "attn_q_a.weight", cfg.mla_q_rank, d),
                .q_a_norm = try t.lflat(i, "attn_q_a_norm.weight", cfg.mla_q_rank),
                .q_b = try t.lmat(i, "attn_q_b.weight", heads * cfg.mla_k, cfg.mla_q_rank),
                .kv_a = try t.lmat(i, "attn_kv_a_mqa.weight", kvr, d),
                .kv_a_norm = try t.lflat(i, "attn_kv_a_norm.weight", kvr),
                .k_b = try t.lmat(i, "attn_k_b.weight", heads * kvr, cfg.mla_k),
                .v_b = try t.lmat(i, "attn_v_b.weight", heads * cfg.mla_v, kvr),
                .out = try t.lmat(i, "attn_output.weight", d, heads * cfg.mla_v),
                .iq = try t.lmat(i, "indexer.attn_q_b.weight", cfg.dsa_heads * cfg.dsa_dim, cfg.mla_q_rank),
                .ik = try t.lmat(i, "indexer.attn_k.weight", cfg.dsa_dim, d),
                .ik_norm = try t.lflat(i, "indexer.k_norm.weight", cfg.dsa_dim),
                .ik_bias = try t.lflat(i, "indexer.k_norm.bias", cfg.dsa_dim),
                .iw = try t.lmat(i, "indexer.proj.weight", cfg.dsa_heads, d),
                .ape = try t.lflat(i, "indexer_compressor_ape.weight", cfg.dsa_pool * cfg.dsa_dim),
                .gate = try t.lmat(i, "indexer_compressor_gate.weight", cfg.dsa_dim, d),
            };
        }
    }
    try widen(a, self.layers);
    return self;
}

/// The BF16 matrices as f32 in one buffer (an exact conversion): the GPU's matmuls read
/// f32 and Q8_0 only, and the CPU skips decoding each row. The buffer lives with the weights.
fn widen(a: std.mem.Allocator, layers: []olmoe.Layer) !void
{
    var total: usize = 0;
    for (0..2) |pass|
    {
        var buf: []u8 = &.{};
        if (pass == 1)
        {
            if (total == 0)
            {
                return;
            }
            buf = try a.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_max), std.mem.alignForward(usize, total, std.heap.page_size_max));
        }
        var at: usize = 0;
        for (layers) |*l|
        {
            var mats: [5]*Mat = undefined;
            var count: usize = 0;
            if (l.kda) |*k|
            {
                mats = .{ &k.f_a, &k.f_b, &k.g_a, &k.g_b, &k.beta };
                count = 5;
            }
            if (l.mla_dsa) |*m|
            {
                mats = .{ &m.kv_a, &m.k_b, &m.v_b, &m.iq, &m.ik };
                count = 5;
            }
            for (mats[0..count]) |m|
            {
                if (m.format != .bf16)
                {
                    continue;
                }
                const bytes = m.rows * m.cols * @sizeOf(f32);
                if (pass == 1)
                {
                    const out: []f32 = @alignCast(std.mem.bytesAsSlice(f32, buf[at..][0..bytes]));
                    try quant.dequantize(.bf16, m.bytes, out);
                    m.* = .{ .format = .f32, .rows = m.rows, .cols = m.cols, .bytes = buf[at..][0..bytes] };
                }
                at += std.mem.alignForward(usize, bytes, 64);
            }
        }
        total = at;
        if (pass == 1)
        {
            if (olmoe.gpu_adopt) |f|
            {
                _ = f(buf);
            }
        }
    }
}

/// One block's mHC weights per token: the output weights and the stream mix (rows are input streams).
const Mix = struct {
    post: []f32,
    comb: []f32,
};

/// mHC's working space for one step of `n` tokens: the mix, its projection, and the next streams.
pub const Mhc = struct {
    mix: Mix,
    hy: []f32,
    next: []f32,

    pub fn init(a: std.mem.Allocator, cfg: Config, n: usize) !Mhc
    {
        const st = cfg.streams;
        return .{
            .mix = .{ .post = try a.alloc(f32, n * st), .comb = try a.alloc(f32, n * st * st) },
            .hy = try a.alloc(f32, n * (2 + st) * st),
            .next = try a.alloc(f32, n * cfg.residual()),
        };
    }

    /// A block's input `x` from the streams `h`, then its norm.
    pub fn pre(self: *const Mhc, cfg: Config, hc: Hc, norm: []const f32, h: []const f32, n: usize, x: []f32, scratch: []f32) void
    {
        hcPre(cfg, hc, h, n, x, self.mix, self.hy, scratch);
        normRows(tokenThreads(n), x, norm, cfg.eps, cfg.d_model, n);
    }

    /// The streams after the block's output `y`, which then replace `h.*`.
    pub fn post(self: *Mhc, cfg: Config, h: *[]f32, y: []const f32, n: usize) void
    {
        hcPost(cfg, h.*, y, self.mix, n, self.next);
        std.mem.swap([]f32, h, &self.next);
    }
};

/// The model's output: each token's streams averaged, then normalized, for rows `first` to n.
pub fn streamMean(cfg: Config, norm: []const f32, h: []const f32, first: usize, n: usize, x: []f32) void
{
    const d = cfg.d_model;
    const st = cfg.streams;
    const R = cfg.residual();
    for (first..n) |t|
    {
        const xt = x[(t - first) * d ..][0..d];
        @memset(xt, 0);
        for (0..st) |s|
        {
            for (xt, h[t * R + s * d ..][0..d]) |*o, v|
            {
                o.* += v;
            }
        }
        for (xt) |*o|
        {
            o.* /= @floatFromInt(st);
        }
        olmoe.rmsnorm(xt, norm, cfg.eps, xt);
    }
}

/// mHC before a block: from each token's streams `h` (n x residual), its input `x` (n x d_model),
/// and the weights `hcPost` uses. `y` is scratch of n x (2 + streams) * streams.
fn hcPre(cfg: Config, hc: Hc, h: []const f32, n: usize, x: []f32, mix: Mix, y: []f32, scratch: []f32) void
{
    const st = cfg.streams;
    const width = (2 + st) * st;
    hc.proj.matmul(h, n, y[0 .. n * width], scratch);
    const Pre = struct {
        cfg: Config,
        hc: Hc,
        h: []const f32,
        x: []f32,
        mix: Mix,
        y: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const c = self.cfg;
            const s_n = c.streams;
            const w = (2 + s_n) * s_n;
            const r = c.residual();
            for (first..end) |t|
            {
                const ht = self.h[t * r ..][0..r];
                var ss: f32 = 0;
                for (ht) |v|
                {
                    ss += v * v;
                }
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(r)) + c.eps);
                const yt = self.y[t * w ..][0..w];
                var pre: [max_streams]f32 = undefined;
                const post = self.mix.post[t * s_n ..][0..s_n];
                for (0..s_n) |s|
                {
                    pre[s] = sigmoid(yt[s] * inv * self.hc.scale[0] + self.hc.base[s]) + c.hc_eps;
                    post[s] = 2 * sigmoid(yt[s_n + s] * inv * self.hc.scale[1] + self.hc.base[s_n + s]);
                }
                const comb = self.mix.comb[t * s_n * s_n ..][0 .. s_n * s_n];
                for (0..s_n) |i|
                {
                    const row = comb[i * s_n ..][0..s_n];
                    var mx: f32 = -std.math.inf(f32);
                    for (row, 0..) |*e, j|
                    {
                        e.* = yt[2 * s_n + i * s_n + j] * inv * self.hc.scale[2] + self.hc.base[2 * s_n + i * s_n + j];
                        mx = @max(mx, e.*);
                    }
                    var sum: f32 = 0;
                    for (row) |*e|
                    {
                        e.* = @exp(e.* - mx);
                        sum += e.*;
                    }
                    for (row) |*e|
                    {
                        e.* = e.* / sum + c.hc_eps;
                    }
                }
                sinkhorn(comb, s_n, c.hc_iters, c.hc_eps);
                const xt = self.x[t * c.d_model ..][0..c.d_model];
                @memset(xt, 0);
                for (0..s_n) |s|
                {
                    for (xt, ht[s * c.d_model ..][0..c.d_model]) |*o, v|
                    {
                        o.* += pre[s] * v;
                    }
                }
            }
        }
    };
    pool.parallelFor(tokenThreads(n), n, Pre{ .cfg = cfg, .hc = hc, .h = h, .x = x, .mix = mix, .y = y }, Pre.run);
}

/// Columns, then rows and columns `iters` - 1 more times, each divided by its sum plus `eps`.
fn sinkhorn(m: []f32, s: usize, iters: usize, eps: f32) void
{
    const cols = struct {
        fn f(mm: []f32, ss: usize, e: f32) void
        {
            for (0..ss) |j|
            {
                var sum: f32 = 0;
                for (0..ss) |i|
                {
                    sum += mm[i * ss + j];
                }
                for (0..ss) |i|
                {
                    mm[i * ss + j] /= sum + e;
                }
            }
        }
    }.f;
    cols(m, s, eps);
    for (1..iters) |_|
    {
        for (0..s) |i|
        {
            const row = m[i * s ..][0..s];
            var sum: f32 = 0;
            for (row) |v|
            {
                sum += v;
            }
            for (row) |*v|
            {
                v.* /= sum + eps;
            }
        }
        cols(m, s, eps);
    }
}

/// mHC after a block: each stream becomes its output weight times the block's output `y`
/// plus the mix of the old streams. `out` (n x residual) takes the new streams.
fn hcPost(cfg: Config, h: []const f32, y: []const f32, mix: Mix, n: usize, out: []f32) void
{
    const Post = struct {
        cfg: Config,
        h: []const f32,
        y: []const f32,
        mix: Mix,
        out: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const c = self.cfg;
            const s_n = c.streams;
            const d = c.d_model;
            const r = c.residual();
            for (first..end) |t|
            {
                const ht = self.h[t * r ..][0..r];
                const yt = self.y[t * d ..][0..d];
                const comb = self.mix.comb[t * s_n * s_n ..][0 .. s_n * s_n];
                const post = self.mix.post[t * s_n ..][0..s_n];
                for (0..s_n) |o|
                {
                    const dst = self.out[t * r + o * d ..][0..d];
                    for (dst, yt) |*e, v|
                    {
                        e.* = post[o] * v;
                    }
                    for (0..s_n) |i|
                    {
                        const wgt = comb[i * s_n + o];
                        for (dst, ht[i * d ..][0..d]) |*e, v|
                        {
                            e.* += wgt * v;
                        }
                    }
                }
            }
        }
    };
    pool.parallelFor(tokenThreads(n), n, Post{ .cfg = cfg, .h = h, .y = y, .mix = mix, .out = out }, Post.run);
}

/// GLM's SwiGLU: the gate clamped from above and the up projection on both sides.
pub fn swiglu(g: f32, u: f32, limit: f32) f32
{
    if (limit <= 0)
    {
        return silu(g) * u;
    }
    return silu(@min(g, limit)) * std.math.clamp(u, -limit, limit);
}

/// A dense block or shared expert on `n` inputs, added into `out`. `g` and `u` take n x hidden each.
pub fn mlp(e: Expert, limit: f32, x: []const f32, n: usize, g: []f32, u: []f32, y: []f32, out: []f32, scratch: []f32) void
{
    const f = e.gate.rows;
    const d = e.down.rows;
    const fused = if (olmoe.gpu_matmuls) |fm| fm(&.{ e.gate, e.up }, x[0 .. n * e.gate.cols], n, &.{ g[0 .. n * f], u[0 .. n * f] }) else false;
    if (!fused)
    {
        e.gate.matmul(x, n, g, scratch);
        e.up.matmul(x, n, u, scratch);
    }
    for (g[0 .. n * f], u[0 .. n * f]) |*a, b|
    {
        a.* = swiglu(a.*, b, limit);
    }
    e.down.matmul(g, n, y, scratch);
    for (out[0 .. n * d], y[0 .. n * d]) |*o, v|
    {
        o.* += v;
    }
}

/// Buffers for one step, sized for `n` tokens.
pub const Buffers = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    mixed: []f32,
    low: []f32,
    decay: []f32,
    beta: []f32,
    gate: []f32,
    core: []f32,
    qa: []f32,
    qh: []f32,
    lat: []f32,
    qp: []f32,
    lsum: []f32,
    att: []f32,
    gather: []f32,
    gather_out: []f32,
    iq: []f32,
    ik: []f32,
    igate: []f32,
    iw: []f32,
    pos: []u32,
    count: []u32,

    pub fn init(a: std.mem.Allocator, cfg: Config, n: usize) !Buffers
    {
        const inner = cfg.ssm_v_heads * cfg.ssm_dv;
        const heads = cfg.heads;
        return .{
            .q = try a.alloc(f32, n * inner),
            .k = try a.alloc(f32, n * inner),
            .v = try a.alloc(f32, n * inner),
            .mixed = try a.alloc(f32, n * 3 * inner),
            .low = try a.alloc(f32, n * @max(cfg.ssm_dk, 256)),
            .decay = try a.alloc(f32, n * inner),
            .beta = try a.alloc(f32, n * cfg.ssm_v_heads),
            .gate = try a.alloc(f32, n * inner),
            .core = try a.alloc(f32, n * inner),
            .qa = try a.alloc(f32, n * cfg.mla_q_rank),
            .qh = try a.alloc(f32, n * heads * cfg.mla_k),
            .lat = try a.alloc(f32, n * cfg.mla_kv_rank),
            .qp = try a.alloc(f32, n * heads * cfg.mla_kv_rank),
            .lsum = try a.alloc(f32, n * heads * cfg.mla_kv_rank),
            .att = try a.alloc(f32, n * heads * cfg.mla_v),
            .gather = try a.alloc(f32, n * @max(cfg.mla_k, cfg.mla_kv_rank)),
            .gather_out = try a.alloc(f32, n * @max(cfg.mla_kv_rank, cfg.mla_v)),
            .iq = try a.alloc(f32, n * cfg.dsa_heads * cfg.dsa_dim),
            .ik = try a.alloc(f32, n * cfg.dsa_dim),
            .igate = try a.alloc(f32, n * cfg.dsa_dim),
            .iw = try a.alloc(f32, n * cfg.dsa_heads),
            .pos = try a.alloc(u32, n * choiceStride(cfg)),
            .count = try a.alloc(u32, n),
        };
    }
};

/// A KDA layer on `n` normalized inputs `x`: `out` takes its output and `state` moves on.
pub fn kda(b: *const Buffers, cfg: Config, w: Kda, state: []f32, x: []const f32, n: usize, out: []f32, scratch: []f32) void
{
    const inner = cfg.ssm_v_heads * cfg.ssm_dv;
    const d = cfg.d_model;
    const q = b.q[0 .. n * inner];
    const k = b.k[0 .. n * inner];
    const v = b.v[0 .. n * inner];
    const fused = if (olmoe.gpu_matmuls) |f| f(&.{ w.q, w.k, w.v }, x[0 .. n * d], n, &.{ q, k, v }) else false;
    if (!fused)
    {
        w.q.matmul(x, n, q, scratch);
        w.k.matmul(x, n, k, scratch);
        w.v.matmul(x, n, v, scratch);
    }
    const rank = w.f_a.rows;
    const low = b.low[0 .. n * rank];
    w.f_a.matmul(x, n, low, scratch);
    w.f_b.matmul(low, n, b.decay[0 .. n * inner], scratch);
    w.g_a.matmul(x, n, low, scratch);
    w.g_b.matmul(low, n, b.gate[0 .. n * inner], scratch);
    w.beta.matmul(x, n, b.beta[0 .. n * cfg.ssm_v_heads], scratch);
    const cd = 3 * inner;
    const conv_state = state[0 .. (cfg.ssm_conv - 1) * cd];
    const mats = state[conv_state.len..];
    const threads = tokenThreads(n);
    pool.parallelFor(threads, n, Conv{ .cfg = cfg, .w = w.conv, .state = conv_state, .q = q, .k = k, .v = v, .out = b.mixed }, Conv.run);
    Conv.shift(cfg, conv_state, q, k, v, n);
    pool.parallelFor(threads, n, Prepare{ .cfg = cfg, .w = w, .mixed = b.mixed, .decay = b.decay, .beta = b.beta }, Prepare.run);
    pool.parallelFor(olmoe.cpuThreads(), cfg.ssm_v_heads, Rule{ .cfg = cfg, .mats = mats, .mixed = b.mixed, .decay = b.decay, .beta = b.beta, .core = b.core, .n = n }, Rule.run);
    pool.parallelFor(threads, n, Norm{ .cfg = cfg, .w = w.norm, .core = b.core, .gate = b.gate }, Norm.run);
    w.out.matmul(b.core[0 .. n * inner], n, out, scratch);
}

/// The causal convolution over q, k and v with SiLU; the state holds the last inputs as [q | k | v] rows.
const Conv = struct {
    cfg: Config,
    w: []const f32,
    state: []const f32,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32,

    fn input(self: @This(), i: usize, c: usize) []const f32
    {
        const inner = self.cfg.ssm_v_heads * self.cfg.ssm_dv;
        const held = self.cfg.ssm_conv - 1;
        if (i < held)
        {
            return self.state[i * 3 * inner + c * inner ..][0..inner];
        }
        const src = switch (c) {
            0 => self.q,
            1 => self.k,
            else => self.v,
        };
        return src[(i - held) * inner ..][0..inner];
    }

    fn run(self: @This(), first: usize, end: usize) void
    {
        const kw = self.cfg.ssm_conv;
        const inner = self.cfg.ssm_v_heads * self.cfg.ssm_dv;
        for (first..end) |t|
        {
            for (0..3) |c|
            {
                const o = self.out[t * 3 * inner + c * inner ..][0..inner];
                @memset(o, 0);
                const wc = self.w[c * inner * kw ..][0 .. inner * kw];
                for (0..kw) |j|
                {
                    for (o, self.input(t + j, c), 0..) |*e, x, ch|
                    {
                        e.* += wc[ch * kw + j] * x;
                    }
                }
                for (o) |*e|
                {
                    e.* = silu(e.*);
                }
            }
        }
    }

    fn shift(cfg: Config, state: []f32, q: []const f32, k: []const f32, v: []const f32, n: usize) void
    {
        const inner = cfg.ssm_v_heads * cfg.ssm_dv;
        const held = cfg.ssm_conv - 1;
        for (0..held) |r|
        {
            const i = n + r;
            for ([_][]const f32{ q, k, v }, 0..) |src, c|
            {
                const dst = state[r * 3 * inner + c * inner ..][0..inner];
                if (i < held)
                {
                    @memcpy(dst, state[i * 3 * inner + c * inner ..][0..inner]);
                }
                else
                {
                    @memcpy(dst, src[(i - held) * inner ..][0..inner]);
                }
            }
        }
    }
};

/// Per token: L2-normalized queries (scaled) and keys, beta as a sigmoid, and each channel's log decay.
const Prepare = struct {
    cfg: Config,
    w: Kda,
    mixed: []f32,
    decay: []f32,
    beta: []f32,

    fn run(self: @This(), first: usize, end: usize) void
    {
        const c = self.cfg;
        const dk = c.ssm_dk;
        const heads = c.ssm_v_heads;
        const inner = heads * dk;
        const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
        for (first..end) |t|
        {
            const row = self.mixed[t * 3 * inner ..];
            for (0..2 * heads) |hh|
            {
                const vv = row[hh * dk ..][0..dk];
                var ss: f32 = 0;
                for (vv) |e|
                {
                    ss += e * e;
                }
                const norm = @sqrt(ss + 1e-6);
                for (vv) |*e|
                {
                    e.* /= norm;
                    if (hh < heads)
                    {
                        e.* *= qscale;
                    }
                }
            }
            for (0..heads) |hh|
            {
                const i = t * heads + hh;
                self.beta[i] = sigmoid(self.beta[i]);
                const rate = -self.w.a[hh];
                for (0..dk) |j|
                {
                    const ch = hh * dk + j;
                    const g = &self.decay[t * inner + ch];
                    g.* = c.kda_lower * sigmoid(rate * (g.* + self.w.dt_bias[ch]));
                }
            }
        }
    }
};

/// Heads `first` to `end`, token by token: decay each key row of the matrix, correct it toward the value, read it out.
const Rule = struct {
    cfg: Config,
    mats: []f32,
    mixed: []const f32,
    decay: []const f32,
    beta: []const f32,
    core: []f32,
    n: usize,

    fn run(self: @This(), first: usize, end: usize) void
    {
        const c = self.cfg;
        const dk = c.ssm_dk;
        const dv = c.ssm_dv;
        const heads = c.ssm_v_heads;
        const inner = heads * dk;
        var mem: [256]f32 = undefined;
        for (first..end) |hh|
        {
            const m = self.mats[hh * dk * dv ..][0 .. dk * dv];
            for (0..self.n) |t|
            {
                const row = self.mixed[t * 3 * inner ..];
                const q = row[hh * dk ..][0..dk];
                const k = row[inner + hh * dk ..][0..dk];
                const v = row[2 * inner + hh * dv ..][0..dv];
                const g = self.decay[t * inner + hh * dk ..][0..dk];
                const b = self.beta[t * heads + hh];
                const kv = mem[0..dv];
                @memset(kv, 0);
                for (0..dk) |i|
                {
                    const mi = m[i * dv ..][0..dv];
                    const e_g = @exp(g[i]);
                    for (mi, kv) |*e, *acc|
                    {
                        e.* *= e_g;
                        acc.* += e.* * k[i];
                    }
                }
                for (kv, v) |*e, vv|
                {
                    e.* = (vv - e.*) * b;
                }
                const o = self.core[t * inner + hh * dv ..][0..dv];
                @memset(o, 0);
                for (0..dk) |i|
                {
                    const mi = m[i * dv ..][0..dv];
                    for (mi, kv, o) |*e, delta, *acc|
                    {
                        e.* += k[i] * delta;
                        acc.* += e.* * q[i];
                    }
                }
            }
        }
    }
};

/// Per token and head: RMSNorm with its weight, times a sigmoid of the gate.
const Norm = struct {
    cfg: Config,
    w: []const f32,
    core: []f32,
    gate: []const f32,

    fn run(self: @This(), first: usize, end: usize) void
    {
        const dv = self.cfg.ssm_dv;
        const inner = self.cfg.ssm_v_heads * dv;
        for (first..end) |t|
        {
            for (0..self.cfg.ssm_v_heads) |hh|
            {
                const o = self.core[t * inner + hh * dv ..][0..dv];
                const g = self.gate[t * inner + hh * dv ..][0..dv];
                var ss: f32 = 0;
                for (o) |e|
                {
                    ss += e * e;
                }
                const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dv)) + self.cfg.eps);
                for (o, g, self.w) |*e, gg, wt|
                {
                    e.* = wt * (e.* * inv) * sigmoid(gg);
                }
            }
        }
    }
};

/// LayerNorm with weight and bias, as torch's (biased variance), in place.
fn layerNorm(x: []f32, w: []const f32, bias: []const f32, eps: f32) void
{
    var mean: f32 = 0;
    for (x) |v|
    {
        mean += v;
    }
    mean /= @floatFromInt(x.len);
    var variance: f32 = 0;
    for (x) |v|
    {
        variance += (v - mean) * (v - mean);
    }
    variance /= @floatFromInt(x.len);
    const inv = 1.0 / @sqrt(variance + eps);
    for (x, w, bias) |*v, g, bb|
    {
        v.* = (v.* - mean) * inv * g + bb;
    }
}

/// `m` (heads blocks of `rows` rows) applied per head to the head slices of `src`
/// (n x heads x m.cols), into `dst` (n x heads x rows), through contiguous buffers.
pub fn perHead(m: Mat, heads: usize, rows: usize, src: []const f32, n: usize, dst: []f32, gather: []f32, gather_out: []f32, scratch: []f32) void
{
    const cols = m.cols;
    for (0..heads) |hh|
    {
        for (0..n) |t|
        {
            @memcpy(gather[t * cols ..][0..cols], src[(t * heads + hh) * cols ..][0..cols]);
        }
        m.rowRange(hh * rows, rows).matmul(gather, n, gather_out, scratch);
        for (0..n) |t|
        {
            @memcpy(dst[(t * heads + hh) * rows ..][0..rows], gather_out[t * rows ..][0..rows]);
        }
    }
}

/// An MLA layer with its DSA indexer on `n` normalized inputs `x` from position `pos0`;
/// the latents and indexer state go into the cache's attention layer `ai`.
pub fn mla(b: *const Buffers, cfg: Config, w: Mla, cache: *olmoe.KvCache, ai: usize, x: []const f32, n: usize, pos0: usize, out: []f32, kept: []i32, scratch: []f32) void
{
    const heads = cfg.heads;
    const kvr = cfg.mla_kv_rank;
    const qa = b.qa[0 .. n * cfg.mla_q_rank];
    w.q_a.matmul(x, n, qa, scratch);
    for (0..n) |t|
    {
        const r = qa[t * cfg.mla_q_rank ..][0..cfg.mla_q_rank];
        olmoe.rmsnorm(r, w.q_a_norm, cfg.eps, r);
    }
    w.q_b.matmul(qa, n, b.qh[0 .. n * heads * cfg.mla_k], scratch);
    const lat = b.lat[0 .. n * kvr];
    w.kv_a.matmul(x, n, lat, scratch);
    for (0..n) |t|
    {
        const r = lat[t * kvr ..][0..kvr];
        olmoe.rmsnorm(r, w.kv_a_norm, cfg.eps, r);
    }
    cache.store(ai, pos0, n, lat, &.{});
    // Each head's query against the latent: its key projection folded into the query.
    perHead(w.k_b, heads, kvr, b.qh, n, b.qp, b.gather, b.gather_out, scratch);

    const dim = cfg.dsa_dim;
    w.iq.matmul(qa, n, b.iq[0 .. n * cfg.dsa_heads * dim], scratch);
    const ik = b.ik[0 .. n * dim];
    w.ik.matmul(x, n, ik, scratch);
    for (0..n) |t|
    {
        layerNorm(ik[t * dim ..][0..dim], w.ik_norm, w.ik_bias, 1e-6);
    }
    w.gate.matmul(x, n, b.igate[0 .. n * dim], scratch);
    w.iw.matmul(x, n, b.iw[0 .. n * cfg.dsa_heads], scratch);
    const keys = cache.dsaKeys(ai);
    const gates = cache.dsaGates(ai);
    @memcpy(keys[pos0 * dim ..][0 .. n * dim], ik);
    @memcpy(gates[pos0 * dim ..][0 .. n * dim], b.igate[0 .. n * dim]);
    // Pools completed by these tokens: a softmax over each channel's tokens weighs their keys.
    const p = cfg.dsa_pool;
    const pools = cache.dsaPools(ai);
    for (pos0 / p..(pos0 + n) / p) |pl|
    {
        const pk = pools[pl * dim ..][0..dim];
        for (0..dim) |ch|
        {
            var mx: f32 = -std.math.inf(f32);
            for (0..p) |j|
            {
                mx = @max(mx, gates[(pl * p + j) * dim + ch] + w.ape[j * dim + ch]);
            }
            var sum: f32 = 0;
            var acc: f32 = 0;
            for (0..p) |j|
            {
                const e = @exp(gates[(pl * p + j) * dim + ch] + w.ape[j * dim + ch] - mx);
                sum += e;
                acc += e * keys[(pl * p + j) * dim + ch];
            }
            pk[ch] = acc / sum;
        }
    }
    pool.parallelFor(tokenThreads(n), n, Choose{ .cfg = cfg, .iq = b.iq, .iw = b.iw, .pools = pools, .pos0 = pos0, .pos = b.pos, .count = b.count, .kept = kept }, Choose.run);
    const groups = cfg.heads / headGroup(cfg.heads);
    pool.parallelFor(olmoe.cpuThreads(), n * groups, Attend{ .cfg = cfg, .qp = b.qp, .lat = cache.keysF32(ai), .pos = b.pos, .count = b.count, .lsum = b.lsum }, Attend.run);
    // Each head's weighted latent through its value projection.
    perHead(w.v_b, heads, cfg.mla_v, b.lsum, n, b.att, b.gather, b.gather_out, scratch);
    w.out.matmul(b.att[0 .. n * heads * cfg.mla_v], n, out, scratch);
}

/// Room for one token's chosen positions: the best pools plus an unfinished one.
fn choiceStride(cfg: Config) usize
{
    return (cfg.dsaTop() + 1) * cfg.dsa_pool;
}

/// Heads that attend together, reading each cached latent once: up to 8.
fn headGroup(heads: usize) usize
{
    var g: usize = 8;
    while (heads % g != 0)
    {
        g /= 2;
    }
    return g;
}

/// DSA's choice of positions for tokens `first` to `end`: every position while the prompt is
/// within the budget, else the best pools plus the unfinished tail.
const Choose = struct {
    cfg: Config,
    iq: []const f32,
    iw: []const f32,
    pools: []const f32,
    pos0: usize,
    pos: []u32,
    count: []u32,
    kept: []i32,

    fn run(self: @This(), first: usize, end: usize) void
    {
        self.runOrFail(first, end) catch @panic("out of memory in DSA selection");
    }

    fn runOrFail(self: @This(), first: usize, end: usize) !void
    {
        const c = self.cfg;
        const p = c.dsa_pool;
        const top = c.dsaTop();
        const dim = c.dsa_dim;
        const ih = c.dsa_heads;
        const stride = choiceStride(c);
        const last = self.pos0 + end - 1;
        const a = std.heap.page_allocator;
        const pscore = try a.alloc(f32, (last + 1) / p + 1);
        defer a.free(pscore);
        const order = try a.alloc(u32, top);
        defer a.free(order);
        const iscale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
        const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ih)));
        for (first..end) |t|
        {
            const pos = self.pos0 + t;
            const full = (pos + 1) / p;
            const positions = self.pos[t * stride ..][0..stride];
            const kept_row: []i32 = if (self.kept.len > 0) self.kept[t * top ..][0..top] else &.{};
            var count: usize = 0;
            if (full <= top)
            {
                for (0..pos + 1) |s|
                {
                    positions[s] = @intCast(s);
                }
                count = pos + 1;
                @memset(kept_row, -1);
            }
            else
            {
                const iqt = self.iq[t * ih * dim ..][0 .. ih * dim];
                const iwt = self.iw[t * ih ..][0..ih];
                for (0..full) |pl|
                {
                    const pk = self.pools[pl * dim ..][0..dim];
                    var s: f32 = 0;
                    for (0..ih) |hh|
                    {
                        s += iwt[hh] * wscale * @max(0, olmoe.dot(iqt[hh * dim ..][0..dim], pk) * iscale);
                    }
                    pscore[pl] = s;
                }
                selectTop(pscore[0..full], order);
                for (order, 0..) |pl, i|
                {
                    if (kept_row.len > 0)
                    {
                        kept_row[i] = @intCast(pl);
                    }
                    for (0..p) |j|
                    {
                        positions[count] = @intCast(pl * p + j);
                        count += 1;
                    }
                }
                for (full * p..pos + 1) |s|
                {
                    positions[count] = @intCast(s);
                    count += 1;
                }
            }
            self.count[t] = @intCast(count);
        }
    }
};

/// Attention on the chosen positions for work items `first` to `end`, each a token and a group
/// of heads: a softmax over the latents per head, then the latents weighted by it into `lsum`.
const Attend = struct {
    cfg: Config,
    qp: []const f32,
    lat: []const f32,
    pos: []const u32,
    count: []const u32,
    lsum: []f32,

    fn run(self: @This(), first: usize, end: usize) void
    {
        const c = self.cfg;
        const heads = c.heads;
        const kvr = c.mla_kv_rank;
        const gs = headGroup(heads);
        const groups = heads / gs;
        const stride = choiceStride(c);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.mla_k)));
        const scores = std.heap.page_allocator.alloc(f32, gs * stride) catch @panic("out of memory in MLA attention");
        defer std.heap.page_allocator.free(scores);
        for (first..end) |item|
        {
            const t = item / groups;
            const h0 = (item % groups) * gs;
            const positions = self.pos[t * stride ..][0..self.count[t]];
            const q = self.qp[(t * heads + h0) * kvr ..][0 .. gs * kvr];
            var mx: [8]f32 = @splat(-std.math.inf(f32));
            for (positions, 0..) |s, i|
            {
                const row = self.lat[s * kvr ..][0..kvr];
                for (0..gs) |g|
                {
                    const sc = olmoe.dot(q[g * kvr ..][0..kvr], row) * scale;
                    scores[g * stride + i] = sc;
                    mx[g] = @max(mx[g], sc);
                }
            }
            var sum: [8]f32 = @splat(0);
            for (0..gs) |g|
            {
                for (scores[g * stride ..][0..positions.len]) |*sc|
                {
                    sc.* = @exp(sc.* - mx[g]);
                    sum[g] += sc.*;
                }
            }
            const out = self.lsum[(t * heads + h0) * kvr ..][0 .. gs * kvr];
            @memset(out, 0);
            for (positions, 0..) |s, i|
            {
                const row = self.lat[s * kvr ..][0..kvr];
                for (0..gs) |g|
                {
                    axpy(out[g * kvr ..][0..kvr], scores[g * stride + i] / sum[g], row);
                }
            }
        }
    }
};

/// y += a x.
pub fn axpy(y: []f32, a: f32, x: []const f32) void
{
    const V = @Vector(16, f32);
    const av: V = @splat(a);
    var i: usize = 0;
    while (i + 16 <= y.len) : (i += 16)
    {
        const xv: V = x[i..][0..16].*;
        const yv: V = y[i..][0..16].*;
        y[i..][0..16].* = yv + av * xv;
    }
    while (i < y.len) : (i += 1)
    {
        y[i] += a * x[i];
    }
}

/// The `out.len` highest of `score` (ties to the lower index), ascending by index: a heap
/// whose root is the worst kept, so each of n scores costs log k rather than a full sort.
pub fn selectTop(score: []const f32, out: []u32) void
{
    const k = out.len;
    const worse = struct {
        fn f(sc: []const f32, x: u32, y: u32) bool
        {
            return sc[x] < sc[y] or (sc[x] == sc[y] and x > y);
        }
    }.f;
    const sift = struct {
        fn down(sc: []const f32, h: []u32, start: usize) void
        {
            var i = start;
            while (true)
            {
                var w = i;
                const l = 2 * i + 1;
                if (l < h.len and worse(sc, h[l], h[w]))
                {
                    w = l;
                }
                if (l + 1 < h.len and worse(sc, h[l + 1], h[w]))
                {
                    w = l + 1;
                }
                if (w == i)
                {
                    return;
                }
                std.mem.swap(u32, &h[i], &h[w]);
                i = w;
            }
        }
    }.down;
    for (0..k) |i|
    {
        out[i] = @intCast(i);
    }
    var i = k / 2;
    while (i > 0)
    {
        i -= 1;
        sift(score, out, i);
    }
    for (k..score.len) |j|
    {
        if (worse(score, out[0], @intCast(j)))
        {
            out[0] = @intCast(j);
            sift(score, out, 0);
        }
    }
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
}

test "selectTop keeps the highest scores, ties to the lower index, ascending" {
    const score = [_]f32{ 0.5, 2, 1, 2, 0, 1, 3 };
    var out: [4]u32 = undefined;
    selectTop(&score, &out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 6 }, &out);
}


pub fn normRows(threads: usize, x: []f32, w: []const f32, eps: f32, d: usize, n: usize) void
{
    const Rows = struct {
        x: []f32,
        w: []const f32,
        eps: f32,
        d: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            for (first..end) |t|
            {
                const r = self.x[t * self.d ..][0..self.d];
                olmoe.rmsnorm(r, self.w, self.eps, r);
            }
        }
    };
    pool.parallelFor(threads, n, Rows{ .x = x, .w = w, .eps = eps, .d = d }, Rows.run);
}
