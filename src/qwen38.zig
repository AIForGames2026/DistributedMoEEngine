//! The parts of Qwen3.8-Flash-Next (TASK-31, transformers' qwen4_exp) that
//! earlier models lack, called from `olmoe.step`:
//!
//! - Hyper-connections: the residual is several streams. Before each block a
//!   low-rank gate mixes them into the block's input (`mix`), and the block's
//!   output is added back to each stream with its own weight (`inject`).
//! - The n-gram embedding (`ple`): each token's bigram and trigram are hashed
//!   into stacked tables, and the rows found are gated per stream and passed
//!   through a short dilated convolution, then added to the streams.
//! - QSA: attention that keeps, per query, the blocks of tokens an indexer
//!   scores highest plus the unfinished last block (`index`, `attend`). With
//!   no more complete blocks than its budget, it is plain attention.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const pool = @import("pool.zig");
const Config = olmoe.Config;

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

/// Each `d_model` stream of `x` normalized on its own, times its part of `w`.
fn streamNorm(cfg: Config, x: []const f32, w: []const f32, out: []f32) void
{
    const d = cfg.d_model;
    for (0..cfg.streams) |s|
    {
        olmoe.rmsnorm(x[s * d ..][0..d], w[s * d ..][0..d], cfg.eps, out[s * d ..][0..d]);
    }
}

/// Buffers for `mix`, allocated once per step.
pub const Buffers = struct {
    hn: []f32,
    low: []f32,
    a: []f32,
    u: []f32,
    scratch: []f32,

    pub fn init(a: std.mem.Allocator, cfg: Config, n: usize) !Buffers
    {
        if (cfg.streams < 2)
        {
            return .{ .hn = &.{}, .low = &.{}, .a = &.{}, .u = &.{}, .scratch = &.{} };
        }
        const r = cfg.residual();
        const e = if (cfg.ple_layer != null) cfg.pleHeads() * cfg.ple_row else 0;
        return .{
            .hn = try a.alloc(f32, n * r),
            .low = try a.alloc(f32, n * (cfg.hc_rank + cfg.streams)),
            .a = try a.alloc(f32, n * cfg.hc_rank),
            .u = try a.alloc(f32, n * r),
            .scratch = try a.alloc(f32, @max(r, @max(cfg.hc_rank, e))),
        };
    }
};

/// One hyper-connection on `n` tokens' streams `h` (n x residual): the block's input
/// `x` (n x d_model) and, if `hy` injects, each stream's weight for the block's output
/// (`inj`, n x streams). Matches transformers' Qwen4ExpTextGatedResidual.
pub fn mix(cfg: Config, hy: olmoe.Hyper, bufs: *const Buffers, h: []const f32, n: usize, x: []f32, inj: []f32) void
{
    const r = cfg.residual();
    const d = cfg.d_model;
    const s_count = cfg.streams;
    const rank = cfg.hc_rank;
    const rows = hy.down.rows;
    const hn = bufs.hn[0 .. n * r];
    const low = bufs.low[0 .. n * rows];
    const a = bufs.a[0 .. n * rank];
    const u = bufs.u[0 .. n * r];
    const Norms = struct {
        cfg: Config,
        w: []const f32,
        h: []const f32,
        hn: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const rr = self.cfg.residual();
            for (first..end) |t|
            {
                streamNorm(self.cfg, self.h[t * rr ..][0..rr], self.w, self.hn[t * rr ..][0..rr]);
            }
        }
    };
    pool.parallelFor(tokenThreads(n), n, Norms{ .cfg = cfg, .w = hy.norm, .h = h, .hn = hn }, Norms.run);
    hy.down.matmul(hn, n, low, bufs.scratch);
    const sf: f32 = @floatFromInt(s_count);
    for (0..n) |t|
    {
        for (0..rank) |j|
        {
            a[t * rank + j] = silu(low[t * rows + j] / sf);
        }
        if (hy.inject)
        {
            for (0..s_count) |s|
            {
                inj[t * s_count + s] = 2 * sigmoid(low[t * rows + rank + s] / sf);
            }
        }
    }
    hy.up.matmul(a, n, u, bufs.scratch);
    for (0..n) |t|
    {
        const ut = u[t * r ..][0..r];
        const ht = hn[t * r ..][0..r];
        for (x[t * d ..][0..d], 0..) |*o, i|
        {
            var sum: f32 = 0;
            for (0..s_count) |s|
            {
                sum += sigmoid(ut[s * d + i]) * ht[s * d + i];
            }
            o.* = sum / sf;
        }
    }
}

/// Adds the block's output `y` (n x d_model) to every stream of `h`, weighted per token and stream by `inj`.
pub fn inject(cfg: Config, h: []f32, y: []const f32, inj: []const f32, n: usize) void
{
    const r = cfg.residual();
    const d = cfg.d_model;
    for (0..n) |t|
    {
        for (0..cfg.streams) |s|
        {
            const wt = inj[t * cfg.streams + s];
            for (h[t * r + s * d ..][0..d], y[t * d ..][0..d]) |*hh, yy|
            {
                hh.* += yy * wt;
            }
        }
    }
}

/// Token `i` of the history: a new token for `i` >= 0, else one of the `held` tokens before them.
fn historyAt(ctx: []const u32, tokens: []const u32, i: isize) u32
{
    if (i >= 0)
    {
        return tokens[@intCast(i)];
    }
    return ctx[@intCast(@as(isize, @intCast(ctx.len)) + i)];
}

/// The table rows each new token's n-grams hash to (n x hash heads), as transformers'
/// Qwen4ExpTextNGramEmbedding computes them. An n-gram never reaches back past an
/// end-of-text token: a position that would is read as end-of-text itself.
pub fn ngramRows(cfg: Config, p: olmoe.Ple, ctx: []const u32, tokens: []const u32, out: []u64) void
{
    const heads = cfg.pleHeads();
    for (tokens, 0..) |_, t|
    {
        var shifted: [Config.max_ngram]u64 = undefined;
        for (0..cfg.ple_ngram) |s|
        {
            var tok = historyAt(ctx, tokens, @as(isize, @intCast(t)) - @as(isize, @intCast(s)));
            for (1..s + 1) |j|
            {
                if (historyAt(ctx, tokens, @as(isize, @intCast(t)) - @as(isize, @intCast(j))) == cfg.ple_eos)
                {
                    tok = cfg.ple_eos;
                }
            }
            shifted[s] = tok;
        }
        for (2..cfg.ple_ngram + 1) |g|
        {
            var mixed: u64 = shifted[0] *% p.mult[0];
            for (1..g) |j|
            {
                mixed ^= shifted[j] *% p.mult[j];
            }
            for (0..cfg.ple_heads_per) |hh|
            {
                const head = (g - 2) * cfg.ple_heads_per + hh;
                out[t * heads + head] = mixed % p.sizes[head] + p.offsets[head];
            }
        }
    }
}

/// Adds the n-gram embedding (transformers' Qwen4ExpTextPLELayer) to `n` tokens' streams `h`,
/// and moves on the cache's last tokens `ctx` and convolution inputs `conv_state`.
pub fn ple(a: std.mem.Allocator, cfg: Config, p: olmoe.Ple, l: olmoe.PleLayer, bufs: *const Buffers, ctx: *[Config.max_ngram]u32, conv_state: []f32, tokens: []const u32, h: []f32) !void
{
    const n = tokens.len;
    const r = cfg.residual();
    const d = cfg.d_model;
    const heads = cfg.pleHeads();
    const row = cfg.ple_row;
    const e = heads * row;
    const held_tokens = ctx[0 .. cfg.ple_ngram - 1];
    const ids = try a.alloc(u64, n * heads);
    ngramRows(cfg, p, held_tokens, tokens, ids);
    const emb = try a.alloc(f32, n * e);
    for (ids, 0..) |id, i|
    {
        p.table.row(@intCast(id), emb[i * row ..][0..row]);
    }
    const key = try a.alloc(f32, n * r);
    const value = try a.alloc(f32, n * d);
    l.key.matmul(emb, n, key, bufs.scratch);
    l.value.matmul(emb, n, value, bufs.scratch);
    const gv = try a.alloc(f32, n * r);
    // The convolution reads the past inputs first, then the new ones.
    const held = cfg.pleHeld();
    const seq = try a.alloc(f32, (held + n) * r);
    @memcpy(seq[0 .. held * r], conv_state);
    const qn = try a.alloc(f32, r);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
    for (0..n) |t|
    {
        const kt = key[t * r ..][0..r];
        streamNorm(cfg, kt, l.norm_key, kt);
        streamNorm(cfg, h[t * r ..][0..r], l.norm_query, qn);
        const gt = gv[t * r ..][0..r];
        for (0..cfg.streams) |s|
        {
            var g = olmoe.dot(kt[s * d ..][0..d], qn[s * d ..][0..d]) * scale;
            g = if (g > 0) @sqrt(@max(g, 1e-6)) else if (g < 0) -@sqrt(@max(-g, 1e-6)) else 0;
            const sg = sigmoid(g);
            for (gt[s * d ..][0..d], value[t * d ..][0..d]) |*o, v|
            {
                o.* = sg * v;
            }
        }
        streamNorm(cfg, gt, l.norm_conv, seq[(held + t) * r ..][0..r]);
    }
    const taps = cfg.ple_conv;
    const dil = cfg.ple_ngram;
    for (0..n) |t|
    {
        const ht = h[t * r ..][0..r];
        for (0..r) |c|
        {
            var acc: f32 = 0;
            for (0..taps) |j|
            {
                acc += l.conv[c * taps + j] * seq[(t + j * dil) * r + c];
            }
            ht[c] += gv[t * r + c] + silu(acc);
        }
    }
    @memcpy(conv_state, seq[n * r ..][0 .. held * r]);
    // The last tokens, for the next step's n-grams.
    const keep = cfg.ple_ngram - 1;
    var hist: [2 * Config.max_ngram]u32 = undefined;
    const take = @min(n, keep);
    @memcpy(hist[0..keep], held_tokens);
    @memcpy(hist[keep..][0..take], tokens[n - take ..]);
    @memcpy(ctx[0..keep], hist[take..][0..keep]);
}

/// QSA's indexer on `n` new tokens from `pos0`: their queries (`iq`, n x heads x dim,
/// normalized and rotated) and raw keys (into `keys`), and the pooled key of every
/// block they complete (into `blocks`), as transformers' Qwen4ExpTextQSAIndexer.
pub fn index(cfg: Config, ix: olmoe.Indexer, rope: *const olmoe.Rope, cs: []const f32, sn: []const f32, x: []const f32, n: usize, pos0: usize, keys: []f32, blocks: []f32, iq: []f32, scratch: []f32) void
{
    const dim = cfg.qsa_dim;
    const hq = cfg.qsa_heads;
    const half = rope.half;
    ix.q.matmul(x, n, iq, scratch);
    ix.k.matmul(x, n, keys[pos0 * dim ..][0 .. n * dim], scratch);
    for (0..n) |t|
    {
        const qt = iq[t * hq * dim ..][0 .. hq * dim];
        for (0..hq) |hh|
        {
            const v = qt[hh * dim ..][0..dim];
            olmoe.rmsnorm(v, ix.q_norm, cfg.eps, v);
        }
        rope.applyAt(qt, hq, dim, cs[t * half ..][0..half], sn[t * half ..][0..half]);
    }
    const b = cfg.qsa_block;
    const bf: f32 = @floatFromInt(b);
    for (pos0 / b..(pos0 + n) / b) |blk|
    {
        const out = blocks[blk * dim ..][0..dim];
        @memset(out, 0);
        for (0..b) |j|
        {
            for (out, keys[(blk * b + j) * dim ..][0..dim]) |*o, kk|
            {
                o.* += kk;
            }
        }
        for (out) |*o|
        {
            o.* /= bf;
        }
        olmoe.rmsnorm(out, ix.k_norm, cfg.eps, out);
        rope.apply(out, 1, dim, blk * b);
    }
}

/// Whether every query of `n` tokens from `pos0` keeps every block, so attention is dense.
pub fn dense(cfg: Config, pos0: usize, n: usize) bool
{
    return (pos0 + n) / cfg.qsa_block <= cfg.qsaTop();
}

/// QSA attention on the CPU for tokens `first` to `end`, over an f32 cache: each query attends
/// to the blocks the indexer scores highest (all of them while there are few) and the unfinished
/// last block. `kept`, if not empty, takes each query's blocks (n x qsaTop(), -1 when it kept all).
pub const Attend = struct {
    cfg: Config,
    q: []const f32,
    iq: []const f32,
    keys: []const f32,
    values: []const f32,
    blocks: []const f32,
    pos0: usize,
    scale: f32,
    out: []f32,
    kept: []i32,

    pub fn run(self: @This(), first: usize, end: usize) void
    {
        self.runOrFail(first, end) catch @panic("out of memory in QSA attention");
    }

    fn runOrFail(self: @This(), first: usize, end: usize) !void
    {
        const cfg = self.cfg;
        const hd = cfg.head_dim;
        const qd = cfg.qDim();
        const kvd = cfg.kvDim();
        const group = cfg.heads / cfg.kv_heads;
        const b = cfg.qsa_block;
        const top = cfg.qsaTop();
        const dim = cfg.qsa_dim;
        const last_pos = self.pos0 + end - 1;
        const a = std.heap.page_allocator;
        const positions = try a.alloc(u32, last_pos + 1);
        defer a.free(positions);
        const scores = try a.alloc(f32, last_pos + 1);
        defer a.free(scores);
        const bscore = try a.alloc(f32, (last_pos + 1) / b + 1);
        defer a.free(bscore);
        const sel = try a.alloc(u32, top);
        defer a.free(sel);
        const val = try a.alloc(f32, top);
        defer a.free(val);
        const iscale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
        for (first..end) |t|
        {
            const p = self.pos0 + t;
            const full = (p + 1) / b;
            var count: usize = 0;
            const kept_row: []i32 = if (self.kept.len > 0) self.kept[t * top ..][0..top] else &.{};
            if (full <= top)
            {
                for (0..p + 1) |s|
                {
                    positions[s] = @intCast(s);
                }
                count = p + 1;
                @memset(kept_row, -1);
            }
            else
            {
                const iqt = self.iq[t * cfg.qsa_heads * dim ..][0 .. cfg.qsa_heads * dim];
                for (0..full) |blk|
                {
                    const bk = self.blocks[blk * dim ..][0..dim];
                    var s: f32 = 0;
                    for (0..cfg.qsa_heads) |hh|
                    {
                        s += @max(0, olmoe.dot(iqt[hh * dim ..][0..dim], bk));
                    }
                    bscore[blk] = s * iscale;
                }
                olmoe.topK(bscore[0..full], top, sel, val);
                std.mem.sort(u32, sel, {}, std.sort.asc(u32));
                for (sel, 0..) |blk, i|
                {
                    if (kept_row.len > 0)
                    {
                        kept_row[i] = @intCast(blk);
                    }
                    for (0..b) |j|
                    {
                        positions[count] = @intCast(blk * b + j);
                        count += 1;
                    }
                }
                for (full * b..p + 1) |s|
                {
                    positions[count] = @intCast(s);
                    count += 1;
                }
            }
            for (0..cfg.heads) |hh|
            {
                const off = hh * hd;
                const kv_off = hh / group * hd;
                const qh = self.q[t * qd + off ..][0..hd];
                var mx: f32 = -std.math.inf(f32);
                for (positions[0..count], 0..) |s, i|
                {
                    scores[i] = olmoe.dot(qh, self.keys[s * kvd + kv_off ..][0..hd]) * self.scale;
                    mx = @max(mx, scores[i]);
                }
                var sum: f32 = 0;
                for (scores[0..count]) |*sc|
                {
                    sc.* = @exp(sc.* - mx);
                    sum += sc.*;
                }
                const o = self.out[t * qd + off ..][0..hd];
                @memset(o, 0);
                for (positions[0..count], 0..) |s, i|
                {
                    const pr = scores[i] / sum;
                    for (o, self.values[s * kvd + kv_off ..][0..hd]) |*e, vv|
                    {
                        e.* += pr * vv;
                    }
                }
            }
        }
    }
};
