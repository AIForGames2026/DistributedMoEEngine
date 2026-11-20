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
//! Qwen3.5-MoE (TASK-29, matching Qwen3_5MoeForCausalLM, as in Qwen3.6-35B-A3B)
//! is a hybrid. Three of every four layers mix tokens with a Gated DeltaNet
//! (a linear attention whose fixed-size state is updated by the delta rule,
//! `recurrentMix`) instead of attention. The others use attention with an
//! output gate and rotary embeddings on part of each head. Every MoE block
//! adds a gated shared expert.
//!
//! Qwen3.8-Flash-Next (TASK-31, matching Qwen4ExpForCausalLM) adds three
//! things on top (`qwen38.zig`): a residual of several streams mixed by
//! hyper-connections in place of the norms, QSA attention that keeps each
//! query's highest-scoring blocks of tokens, and a hashed n-gram embedding
//! at one layer.
//!
//! Weights stay in their GGUF block format and are dequantized one row at a
//! time. The MoE block calls experts through `Moe`, so a process can run the
//! experts it owns and send the rest to a peer (`olmoe_run.zig`).
const std = @import("std");
const builtin = @import("builtin");
const quant = @import("quant.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const pool = @import("pool.zig");
const kv_format = @import("kv_format.zig");
const qwen38 = @import("qwen38.zig");
const glm53 = @import("glm53.zig");
const deepseek = @import("deepseek.zig");
const gptoss = @import("gptoss.zig");
pub const KvFormat = kv_format.KvFormat;

/// The model families the engine runs.
pub const Arch = enum {
    /// OLMoE: QK norm over the whole projection, top-k probabilities used as is.
    olmoe,
    /// Qwen3-MoE (TASK-12): grouped KV heads, QK norm per head, top-k renormalized.
    qwen3moe,
    /// Qwen3.5-MoE (TASK-29): Qwen3-MoE plus Gated DeltaNet layers, gated attention, and a shared expert.
    qwen35moe,
    /// Qwen3.8-Flash-Next (TASK-31): Qwen3.5-MoE with hyper-connections, QSA, and an n-gram embedding.
    qwen4exp,
    /// GLM-5.3-Flash (TASK-32, GGUF "glm5-next"): KDA and MLA layers with a DSA indexer, mHC, dense leading layers.
    glm5next,
    /// Mixtral (TASK-39, GGUF "llama" with experts): grouped KV heads, no QK norm, rope on adjacent pairs.
    mixtral,
    /// DeepSeek V3 and Kimi K2 (TASK-39, GGUF "deepseek2"): MLA with decoupled rope, grouped sigmoid routing.
    deepseek2,
    /// gpt-oss (TASK-39, GGUF "gpt-oss"): attention sinks, sliding windows, biases, and its own SwiGLU.
    gptoss,
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
    /// YaRN's ramp starts and ends on whole dimensions (transformers' `truncate`, false for gpt-oss).
    yarn_truncate: bool = true,
    /// Dimensions of each head that rotate (the first ones); 0 for all of them.
    rope_dim: usize = 0,
    /// Rope rotates adjacent pairs (llama.cpp's "norm" rope) rather than each half against the other.
    rope_adjacent: bool = false,
    /// Every this-many-th layer uses attention, the others a Gated DeltaNet; 0 for attention everywhere.
    attn_interval: usize = 0,
    /// Attention multiplies its output by a sigmoid gate that comes with the query projection.
    attn_gate: bool = false,
    /// The shared expert's hidden size, 0 for none.
    shared_ff: usize = 0,
    /// Gated DeltaNet sizes: conv width, key heads and their size, value heads and their size.
    ssm_conv: usize = 0,
    ssm_k_heads: usize = 0,
    ssm_dk: usize = 0,
    ssm_v_heads: usize = 0,
    ssm_dv: usize = 0,
    /// The Gated DeltaNet's output gate is a sigmoid rather than SiLU (TASK-31).
    ssm_sigmoid_gate: bool = false,
    /// Residual streams mixed by hyper-connections (TASK-31), 1 for a plain residual, and the mix's rank.
    streams: usize = 1,
    hc_rank: usize = 0,
    /// QSA (TASK-31): indexer heads (0 for dense attention), their size, tokens kept, and tokens per block.
    qsa_heads: usize = 0,
    qsa_dim: usize = 0,
    qsa_budget: usize = 0,
    qsa_block: usize = 0,
    /// The n-gram embedding (TASK-31): its layer, longest n-gram, hash heads per n-gram size,
    /// values per table row, convolution taps, and the token that starts n-grams over.
    ple_layer: ?usize = null,
    ple_ngram: usize = 0,
    ple_heads_per: usize = 0,
    ple_row: usize = 0,
    ple_conv: usize = 0,
    ple_eos: u32 = 0,
    /// GLM-5.3-Flash (TASK-32): leading layers with a dense feed-forward block, and its size.
    dense_layers: usize = 0,
    dense_ff: usize = 0,
    /// KDA's forget gate lower bound (its heads and sizes are the ssm_ fields).
    kda_lower: f32 = 0,
    /// MLA: query rank, key and value latent width, and each head's key and value sizes.
    mla_q_rank: usize = 0,
    mla_kv_rank: usize = 0,
    mla_k: usize = 0,
    mla_v: usize = 0,
    /// DSA: indexer heads, their size, tokens kept, and tokens per pool.
    dsa_heads: usize = 0,
    dsa_dim: usize = 0,
    dsa_budget: usize = 0,
    dsa_pool: usize = 0,
    /// mHC: Sinkhorn iterations and its epsilon (the streams are `streams`).
    hc_iters: usize = 0,
    hc_eps: f32 = 0,
    /// Router: sigmoid scores with a choice bias, renormalized and scaled by this.
    route_scale: f32 = 0,
    /// SwiGLU inputs clamped at this, 0 for none.
    swiglu_limit: f32 = 0,
    /// DeepSeek V3 (TASK-39): experts in groups, the best groups whose experts may be chosen,
    /// and whether the chosen scores are renormalized before `route_scale`.
    route_groups: usize = 1,
    route_groups_used: usize = 1,
    route_norm: bool = true,
    /// Attention's score scale, 0 for one over the square root of the key size.
    attn_scale: f32 = 0,
    /// gpt-oss (TASK-39): every other layer, from the first, attends over this many positions; 0 for none.
    sliding_window: usize = 0,

    pub fn qDim(self: Config) usize
    {
        return self.heads * self.head_dim;
    }

    pub fn rotaryDim(self: Config) usize
    {
        return if (self.rope_dim == 0) self.head_dim else self.rope_dim;
    }

    /// Whether `layer` is a Gated DeltaNet layer rather than attention.
    pub fn recurrent(self: Config, layer: usize) bool
    {
        return self.attn_interval != 0 and (layer + 1) % self.attn_interval != 0;
    }

    /// Layers with attention, which are the ones the KV cache holds.
    pub fn attnLayers(self: Config) usize
    {
        return if (self.attn_interval == 0) self.layers else self.layers / self.attn_interval;
    }

    /// Where attention layer `layer` sits among the attention layers.
    pub fn attnIndex(self: Config, layer: usize) usize
    {
        return if (self.attn_interval == 0) layer else layer / self.attn_interval;
    }

    /// Where Gated DeltaNet layer `layer` sits among those layers.
    pub fn recurrentIndex(self: Config, layer: usize) usize
    {
        return layer - layer / self.attn_interval;
    }

    pub fn recurrentLayers(self: Config) usize
    {
        return self.layers - self.attnLayers();
    }

    /// Channels the Gated DeltaNet's short convolution runs over: queries, keys, values.
    pub fn convDim(self: Config) usize
    {
        return 2 * self.ssm_k_heads * self.ssm_dk + self.ssm_v_heads * self.ssm_dv;
    }

    /// Floats of one Gated DeltaNet layer's state: the last inputs of the convolution, then each value head's matrix.
    pub fn stateFloats(self: Config) usize
    {
        return (self.ssm_conv - 1) * self.convDim() + self.ssm_v_heads * self.ssm_dk * self.ssm_dv;
    }

    pub fn kvDim(self: Config) usize
    {
        return self.kv_heads * self.head_dim;
    }

    /// Floats of one token's residual: every stream.
    pub fn residual(self: Config) usize
    {
        return self.streams * self.d_model;
    }

    /// Complete blocks QSA keeps per query.
    pub fn qsaTop(self: Config) usize
    {
        return self.qsa_budget / self.qsa_block;
    }

    /// The n-gram embedding's hash heads (each with its own table), across n-gram sizes 2 and up.
    pub fn pleHeads(self: Config) usize
    {
        return (self.ple_ngram - 1) * self.ple_heads_per;
    }

    /// Past inputs its dilated convolution reads: taps apart by the n-gram size.
    pub fn pleHeld(self: Config) usize
    {
        return (self.ple_conv - 1) * self.ple_ngram;
    }

    /// DSA's complete pools kept per query.
    pub fn dsaTop(self: Config) usize
    {
        return self.dsa_budget / self.dsa_pool;
    }

    pub fn fromGguf(h: gguf.Header) !Config {
        const name = h.getString("general.architecture") orelse return error.NotOlmoe;
        const arch: Arch = if (std.mem.eql(u8, name, "glm5-next"))
            .glm5next
        else if (std.mem.eql(u8, name, "llama"))
            .mixtral
        else if (std.mem.eql(u8, name, "gpt-oss"))
            .gptoss
        else
            std.meta.stringToEnum(Arch, name) orelse return error.NotOlmoe;
        if (arch == .glm5next)
        {
            return glm53.config(h);
        }
        if (arch == .deepseek2)
        {
            return deepseek.config(h);
        }
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
        var cfg: Config = .{
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
        if (arch == .mixtral)
        {
            // A dense Llama has no experts; llama.cpp converts Mixtral as "llama", rotating adjacent pairs.
            if (g.opt("expert_count") == null or (g.opt("rope.dimension_count") orelse cfg.head_dim) != cfg.head_dim or h.get("llama.rope.scaling.type") != null)
            {
                return error.Unsupported;
            }
            cfg.rope_adjacent = true;
        }
        if (arch == .gptoss)
        {
            try gptoss.configure(h, &cfg);
        }
        if (arch == .qwen35moe or arch == .qwen4exp)
        {
            cfg.rope_dim = g.opt("rope.dimension_count") orelse 0;
            cfg.attn_interval = g.opt("full_attention_interval") orelse 4;
            cfg.attn_gate = true;
            cfg.shared_ff = try g.int("expert_shared_feed_forward_length");
            cfg.ssm_conv = try g.int("ssm.conv_kernel");
            cfg.ssm_dk = try g.int("ssm.state_size");
            cfg.ssm_k_heads = try g.int("ssm.group_count");
            cfg.ssm_v_heads = try g.int("ssm.time_step_rank");
            const inner = try g.int("ssm.inner_size");
            if (cfg.ssm_v_heads == 0 or inner % cfg.ssm_v_heads != 0 or cfg.ssm_v_heads % cfg.ssm_k_heads != 0 or cfg.layers % cfg.attn_interval != 0)
            {
                return error.Unsupported;
            }
            cfg.ssm_dv = inner / cfg.ssm_v_heads;
        }
        if (arch == .qwen4exp)
        {
            cfg.ssm_sigmoid_gate = true;
            cfg.streams = try g.int("hyper_connection.count");
            cfg.hc_rank = try g.int("hyper_connection.low_rank");
            cfg.qsa_heads = try g.int("attention.indexer.head_count");
            cfg.qsa_dim = try g.int("attention.indexer.key_length");
            cfg.qsa_budget = try g.int("attention.indexer.top_k");
            // Every attention layer compresses by the same ratio, and no other layer does.
            for (0..cfg.layers) |l|
            {
                const r: usize = @intCast(h.intAt(g.key("attention.compress_ratios"), l) orelse return error.MissingKey);
                if (cfg.recurrent(l) != (r == 0) or (r != 0 and cfg.qsa_block != 0 and r != cfg.qsa_block))
                {
                    return error.Unsupported;
                }
                if (r != 0)
                {
                    cfg.qsa_block = r;
                }
            }
            if (cfg.streams < 2 or cfg.qsa_block == 0 or cfg.qsa_budget % cfg.qsa_block != 0 or cfg.qsa_dim > max_head or cfg.qsa_dim > cfg.head_dim)
            {
                return error.Unsupported;
            }
            // One layer with the n-gram embedding, as in the released model.
            if (h.intAt(g.key("ple.layers"), 1) != null)
            {
                return error.Unsupported;
            }
            if (h.intAt(g.key("ple.layers"), 0)) |pl|
            {
                cfg.ple_layer = @intCast(pl);
                cfg.ple_ngram = try g.int("ple.ngram_size");
                cfg.ple_heads_per = try g.int("ple.heads_per_ngram");
                cfg.ple_conv = try g.int("ple.conv_kernel");
                cfg.ple_eos = @intCast(try g.int("ple.eos_token_id"));
                cfg.ple_row = try g.int("embedding_length_per_layer_input");
                if (cfg.ple_ngram < 2 or cfg.ple_ngram > max_ngram or cfg.pleHeads() > max_ple_heads or cfg.ple_layer.? >= cfg.layers)
                {
                    return error.Unsupported;
                }
            }
        }
        return cfg;
    }

    /// Longest n-gram and most hash heads the n-gram embedding handles.
    pub const max_ngram = 4;
    pub const max_ple_heads = 64;

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

/// Set with a GPU: several matmuls on the same inputs in one call, ys[i] =
/// mats[i] x, or false to run them one at a time.
pub var gpu_matmuls: ?*const fn (mats: []const Mat, x: []const f32, n: usize, ys: []const []f32) bool = null;

/// Set with a GPU (TASK-12): pins `buf` for kernels to read in place and
/// returns its device address, or null to leave attention on the CPU.
pub var gpu_register: ?*const fn (buf: []u8) ?u64 = null;
pub var gpu_unregister: ?*const fn (buf: []u8) void = null;

/// Set with a GPU (TASK-32): maps a buffer of weights for the rest of the run, so matmuls
/// on weights in it run there; false leaves them on the CPU.
pub var gpu_adopt: ?*const fn (buf: []u8) bool = null;

/// Set with a GPU (TASK-12): runs a group of experts there in one call, each
/// on its own rows of `x`, and returns true, or false to use the CPU path.
pub var gpu_experts: ?*const fn (job: ExpertJob) bool = null;
/// Diagnosis only: called with a layer's intermediate values by name (TF_DUMP_LAYER in olmoe_run).
pub var dump: ?*const fn (name: []const u8, layer: usize, data: []const f32) void = null;

/// Set with a GPU (TASK-29): the gated delta rule of one Gated DeltaNet layer
/// there, updating the state in place, or false to run it on the CPU.
pub var gpu_delta_rule: ?*const fn (job: DeltaJob) bool = null;

/// One Gated DeltaNet layer's delta rule over `n` tokens (`Recurrent.Rule`):
/// `mixed` rows of convDim prepared queries, keys and values, `beta` and the
/// decay's log per token and value head, the state's device address, and the
/// outputs, n x value heads x dv.
pub const DeltaJob = struct {
    cfg: Config,
    mixed: []const f32,
    beta: []const f32,
    decay: []const f32,
    state_dev: u64,
    core: []f32,
    n: usize,
};

/// An expert with at most this many rows in a GPU job runs on the small-batch kernels (SMALL_ROWS in tf_gpu.c).
/// A job lists those experts first.
pub const gpu_small_rows = 8;

/// One layer's experts for `n` tokens with `k` calls each: batch row i is token src[i],
/// and y[t] adds call j's batch row inv[t k + j] times w[t k + j], in order.
pub const ExpertJob = struct {
    experts: []const Expert,
    first: []const u32,
    src: []const u32,
    inv: []const u32,
    w: []const f32,
    x: []const f32,
    n: usize,
    k: usize,
    y: []f32,
};

/// Set with a GPU (TASK-12): runs one layer's attention there and returns
/// true, or false to use the CPU path.
pub var gpu_attention: ?*const fn (job: AttentionJob) bool = null;

/// Set with a GPU (TASK-31): QSA attention for one layer there (block scores, the kept
/// blocks, and attention over them), or false to use the CPU path.
pub var gpu_qsa: ?*const fn (job: QsaJob) bool = null;

/// One QSA layer's attention for `n` new tokens from `pos0`: their queries (normalized and
/// rotated), indexer queries, the device address of every complete block's pooled key, the
/// cache's tiles, and the outputs; `kept` (n x qsaTop(), or empty) takes each selecting
/// query's blocks, ascending.
pub const QsaJob = struct {
    cfg: Config,
    q: []const f32,
    iq: []const f32,
    n: usize,
    pos0: usize,
    scale: f32,
    blocks_dev: u64,
    tiles_k: []const u64,
    tiles_v: []const u64,
    out: []f32,
    kept: []i32,
};

/// Time spent in each part of `step`, summed over every call (TASK-12).
pub const Profile = struct {
    projections: u64 = 0,
    qk_rope: u64 = 0,
    attention: u64 = 0,
    /// Gated DeltaNet layers (TASK-29), projections included.
    recurrent: u64 = 0,
    output: u64 = 0,
    router: u64 = 0,
    experts: u64 = 0,
    head: u64 = 0,
    /// QSA layer passes that chose blocks (TASK-31), on the GPU and on the CPU.
    qsa_gpu: u64 = 0,
    qsa_cpu: u64 = 0,

    /// Adds the time since `t` to `field` and moves `t` on.
    pub fn lap(field: *u64, t: *u64) void
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
    /// How the tiles store each position (TASK-27).
    format: KvFormat = .f32,
    /// Values per head: `head_dim`, or 256 for Qwen3.5 (TASK-29).
    hd: usize = head_dim,

    pub const tile_positions = 32;
    pub const head_dim = 128;

    /// Head sizes the GPU kernels handle.
    pub fn gpuHeadDim(hd: usize) bool
    {
        return hd == head_dim or hd == 256;
    }

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

    pub const empty: Mat = .{ .format = .f32, .rows = 0, .cols = 0, .bytes = &.{} };

    pub fn rowBytes(self: Mat) usize {
        return self.cols / self.format.blockValues() * self.format.blockBytes();
    }

    /// Rows `first` to `first + count` - 1 as a matrix of their own (one part of a merged weight).
    pub fn rowRange(self: Mat, first: usize, count: usize) Mat
    {
        const rb = self.rowBytes();
        return .{ .format = self.format, .rows = count, .cols = self.cols, .bytes = self.bytes[first * rb ..][0 .. count * rb] };
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
        const Work = struct {
            m: Mat,
            x: []const f32,
            n: usize,
            y: []f32,
            fn run(w: @This(), first: usize, end: usize) void
            {
                var buf: [max_parallel_cols]f32 = undefined;
                w.m.matmulRows(w.x, w.n, w.y, &buf, first, end);
            }
        };
        pool.parallelFor(threads, self.rows, Work{ .m = self, .x = x, .n = n, .y = y }, Work.run);
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

const max_threads = pool.max_threads;
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
    if (@import("plat.zig").env("TF_THREADS")) |v|
    {
        count = std.fmt.parseInt(usize, v, 10) catch count;
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

/// Whether every value is finite: x times 0 is 0 for those, NaN for infinities and NaNs.
pub fn allFinite(x: []const f32) bool
{
    const V = @Vector(16, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= x.len) : (i += 16)
    {
        const v: V = x[i..][0..16].*;
        acc += v * @as(V, @splat(0));
    }
    var s = @reduce(.Add, acc);
    while (i < x.len) : (i += 1)
    {
        s += x[i] * 0;
    }
    return s == 0;
}

/// How many rows of `width` values hold a non-finite value.
fn badRows(x: []const f32, width: usize) usize
{
    var bad: usize = 0;
    for (0..x.len / width) |r|
    {
        bad += @intFromBool(!allFinite(x[r * width ..][0..width]));
    }
    return bad;
}

fn firstBadRow(x: []const f32, width: usize) usize
{
    for (0..x.len / width) |r|
    {
        if (!allFinite(x[r * width ..][0..width]))
        {
            return r;
        }
    }
    return x.len / width;
}

/// The largest magnitude in `x`, or NaN if any value is NaN.
pub fn maxAbs(x: []const f32) f32
{
    var m: f32 = 0;
    for (x) |v|
    {
        if (std.math.isNan(v))
        {
            return v;
        }
        m = @max(m, @abs(v));
    }
    return m;
}

/// TF_NAN_TRACE=1 logs each layer's largest values and any expert call that comes out huge (TASK-31's rare NaN).
pub fn nanTrace() bool
{
    const S = struct {
        var on: ?bool = null;
    };
    if (S.on == null)
    {
        S.on = builtin.os.tag != .windows and if (std.c.getenv("TF_NAN_TRACE")) |v| v[0] == '1' else false;
    }
    return S.on.?;
}

test "allFinite catches NaN and infinities" {
    var x: [37]f32 = @splat(1.5);
    try std.testing.expect(allFinite(&x));
    x[20] = std.math.nan(f32);
    try std.testing.expect(!allFinite(&x));
    x[20] = 1;
    x[36] = -std.math.inf(f32);
    try std.testing.expect(!allFinite(&x));
}

pub fn rmsnorm(x: []const f32, w: []const f32, eps: f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out[0..x.len], x, w) |*o, v, g| o.* = v * inv * g;
}

/// QK norm in place on one token's projection `v`: over all of it (OLMoE)
/// or over each head (Qwen3).
fn qkNorm(cfg: Config, v: []f32, w: []const f32) void
{
    if (w.len == 0)
    {
        return;
    }
    if (cfg.arch == .olmoe)
    {
        rmsnorm(v, w, cfg.eps, v);
        return;
    }
    var off: usize = 0;
    while (off < v.len) : (off += cfg.head_dim)
    {
        rmsnorm(v[off..][0..cfg.head_dim], w, cfg.eps, v[off..][0..cfg.head_dim]);
    }
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn sigmoid(x: f32) f32
{
    return 1.0 / (1.0 + @exp(-x));
}

/// As torch's softplus: linear past 20, where exp would add nothing.
fn softplus(x: f32) f32
{
    if (x > 20)
    {
        return x;
    }
    return std.math.log1p(@exp(x));
}

/// Largest value-head size `recurrentMix` handles.
const max_head = 256;

/// A Gated DeltaNet layer (TASK-29) on `n` new tokens: their normalized
/// inputs `x` (n x d_model) become `out` (n x d_model), and `state` moves on
/// from the cache's last position to the last new one. Matches transformers'
/// Qwen3_5MoeGatedDeltaNet: a causal depthwise convolution with SiLU over the
/// query, key and value projections, L2-normalized queries and keys, and per
/// value head a matrix updated token by token by the gated delta rule, read
/// out by the query and passed through an RMSNorm gated by SiLU(z).
/// Buffers for `recurrentMix` and `sharedExpert`, allocated once per step and
/// reused by every layer: GPU results copied into fresh pages each layer made
/// those copies many times slower.
const MixBuffers = struct {
    qkv: []f32,
    mixed: []f32,
    z: []f32,
    beta: []f32,
    decay: []f32,
    core: []f32,
    g: []f32,
    u: []f32,
    y: []f32,

    fn init(a: std.mem.Allocator, cfg: Config, n: usize) !MixBuffers
    {
        const rec = cfg.recurrentLayers() > 0;
        const cd = if (rec) cfg.convDim() else 0;
        const vh = if (rec) cfg.ssm_v_heads else 0;
        const inner = vh * cfg.ssm_dv;
        return .{
            .qkv = try a.alloc(f32, n * cd),
            .mixed = try a.alloc(f32, n * cd),
            .z = try a.alloc(f32, n * inner),
            .beta = try a.alloc(f32, n * vh),
            .decay = try a.alloc(f32, n * vh),
            .core = try a.alloc(f32, n * inner),
            .g = try a.alloc(f32, n * cfg.shared_ff),
            .u = try a.alloc(f32, n * cfg.shared_ff),
            .y = try a.alloc(f32, if (cfg.shared_ff != 0) n * cfg.d_model else 0),
        };
    }
};

fn recurrentMix(bufs: *const MixBuffers, cfg: Config, lin: Linear, state: []f32, state_dev: u64, x: []const f32, n: usize, out: []f32, scratch: []f32) !void
{
    const cd = cfg.convDim();
    const vh = cfg.ssm_v_heads;
    const inner = vh * cfg.ssm_dv;
    std.debug.assert(cfg.ssm_dv <= max_head);
    const qkv = bufs.qkv[0 .. n * cd];
    const mixed = bufs.mixed[0 .. n * cd];
    const z = bufs.z[0 .. n * inner];
    const beta = bufs.beta[0 .. n * vh];
    const decay = bufs.decay[0 .. n * vh];
    const core = bufs.core[0 .. n * inner];
    const d = cfg.d_model;
    const fused = if (gpu_matmuls) |f| f(&.{ lin.qkv, lin.z, lin.beta, lin.alpha }, x[0 .. n * d], n, &.{ qkv, z, beta, decay }) else false;
    if (!fused)
    {
        lin.qkv.matmul(x, n, qkv, scratch);
        lin.z.matmul(x, n, z, scratch);
        lin.beta.matmul(x, n, beta, scratch);
        lin.alpha.matmul(x, n, decay, scratch);
    }
    const conv_state = state[0 .. (cfg.ssm_conv - 1) * cd];
    const mats = state[conv_state.len..];
    const threads = cpuThreads();
    const token_threads = if (n >= parallel_tokens) threads else 1;
    pool.parallelFor(token_threads, n, Recurrent.Conv{ .cfg = cfg, .w = lin.conv, .state = conv_state, .qkv = qkv, .out = mixed }, Recurrent.Conv.run);
    Recurrent.Conv.shift(cfg, conv_state, qkv, n);
    pool.parallelFor(token_threads, n, Recurrent.Prepare{ .cfg = cfg, .lin = lin, .mixed = mixed, .beta = beta, .decay = decay }, Recurrent.Prepare.run);
    // The matrices follow the convolution's inputs in the state, on the device as on the host.
    const on_gpu = if (gpu_delta_rule) |f| state_dev != 0 and f(.{ .cfg = cfg, .mixed = mixed, .beta = beta, .decay = decay, .state_dev = state_dev + conv_state.len * @sizeOf(f32), .core = core, .n = n }) else false;
    if (!on_gpu)
    {
        pool.parallelFor(threads, vh, Recurrent.Rule{ .cfg = cfg, .mats = mats, .mixed = mixed, .beta = beta, .decay = decay, .core = core, .n = n }, Recurrent.Rule.run);
    }
    pool.parallelFor(token_threads, n, Recurrent.Norm{ .cfg = cfg, .w = lin.norm, .core = core, .z = z }, Recurrent.Norm.run);
    lin.out.matmul(core, n, out, scratch);
}

/// The pieces of `recurrentMix`, for `pool.parallelFor`.
const Recurrent = struct {
    /// Tokens `first` to `end` of the causal convolution, channels innermost so rows are read in order;
    /// the state holds the inputs before the new ones and is moved on afterwards by `shift`.
    const Conv = struct {
        cfg: Config,
        w: []const f32,
        state: []const f32,
        qkv: []const f32,
        out: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const kw = self.cfg.ssm_conv;
            const cd = self.cfg.convDim();
            const held = kw - 1;
            for (first..end) |t|
            {
                const o = self.out[t * cd ..][0..cd];
                @memset(o, 0);
                for (0..kw) |j|
                {
                    const i = t + j;
                    const in = if (i < held) self.state[i * cd ..][0..cd] else self.qkv[(i - held) * cd ..][0..cd];
                    for (o, in, 0..) |*e, x, c|
                    {
                        e.* += self.w[c * kw + j] * x;
                    }
                }
                for (o) |*e|
                {
                    e.* = silu(e.*);
                }
            }
        }

        /// The new state: the last inputs of the old state followed by the `n` new tokens.
        fn shift(cfg: Config, state: []f32, qkv: []const f32, n: usize) void
        {
            const cd = cfg.convDim();
            const held = cfg.ssm_conv - 1;
            for (0..held) |r|
            {
                const i = n + r;
                const src = if (i < held) state[i * cd ..][0..cd] else qkv[(i - held) * cd ..][0..cd];
                @memcpy(state[r * cd ..][0..cd], src);
            }
        }
    };

    /// Per token: L2-normalized queries (scaled by 1 / sqrt(dk)) and keys, beta as a sigmoid, and the decay's log.
    const Prepare = struct {
        cfg: Config,
        lin: Linear,
        mixed: []f32,
        beta: []f32,
        decay: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const cfg = self.cfg;
            const cd = cfg.convDim();
            const dk = cfg.ssm_dk;
            const vh = cfg.ssm_v_heads;
            const qscale = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
            for (first..end) |t|
            {
                const row = self.mixed[t * cd ..];
                for (0..2 * cfg.ssm_k_heads) |hh|
                {
                    const v = row[hh * dk ..][0..dk];
                    var ss: f32 = 0;
                    for (v) |e|
                    {
                        ss += e * e;
                    }
                    var inv = 1.0 / @sqrt(ss + 1e-6);
                    if (hh < cfg.ssm_k_heads)
                    {
                        inv *= qscale;
                    }
                    for (v) |*e|
                    {
                        e.* *= inv;
                    }
                }
                for (0..vh) |hh|
                {
                    const i = t * vh + hh;
                    self.beta[i] = sigmoid(self.beta[i]);
                    self.decay[i] = self.lin.a[hh] * softplus(self.decay[i] + self.lin.dt_bias[hh]);
                }
            }
        }
    };

    /// Value heads `first` to `end`, token by token: decay the matrix, correct it toward the value, read it out.
    const Rule = struct {
        cfg: Config,
        mats: []f32,
        mixed: []const f32,
        beta: []const f32,
        decay: []const f32,
        core: []f32,
        n: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const cfg = self.cfg;
            const cd = cfg.convDim();
            const dk = cfg.ssm_dk;
            const dv = cfg.ssm_dv;
            const kh = cfg.ssm_k_heads;
            const vh = cfg.ssm_v_heads;
            var mem: [max_head]f32 = undefined;
            for (first..end) |hh|
            {
                // Value heads are tiled: head hh reads key head hh mod kh.
                const kq = hh % kh;
                const m = self.mats[hh * dk * dv ..][0 .. dk * dv];
                for (0..self.n) |t|
                {
                    const row = self.mixed[t * cd ..];
                    const q = row[kq * dk ..][0..dk];
                    const k = row[(kh + kq) * dk ..][0..dk];
                    const v = row[2 * kh * dk + hh * dv ..][0..dv];
                    const g = @exp(self.decay[t * vh + hh]);
                    const b = self.beta[t * vh + hh];
                    const kv = mem[0..dv];
                    @memset(kv, 0);
                    for (0..dk) |i|
                    {
                        const mi = m[i * dv ..][0..dv];
                        for (mi, kv) |*e, *acc|
                        {
                            e.* *= g;
                            acc.* += e.* * k[i];
                        }
                    }
                    for (kv, v) |*e, vv|
                    {
                        e.* = (vv - e.*) * b;
                    }
                    const o = self.core[t * vh * dv + hh * dv ..][0..dv];
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

    /// Per token and value head: RMSNorm with its weight, times SiLU of the gate z.
    const Norm = struct {
        cfg: Config,
        w: []const f32,
        core: []f32,
        z: []const f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const dv = self.cfg.ssm_dv;
            const inner = self.cfg.ssm_v_heads * dv;
            for (first..end) |t|
            {
                for (0..self.cfg.ssm_v_heads) |hh|
                {
                    const o = self.core[t * inner + hh * dv ..][0..dv];
                    const zz = self.z[t * inner + hh * dv ..][0..dv];
                    var ss: f32 = 0;
                    for (o) |e|
                    {
                        ss += e * e;
                    }
                    const inv = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(dv)) + self.cfg.eps);
                    for (o, zz, self.w) |*e, g, wt|
                    {
                        e.* = wt * (e.* * inv) * (if (self.cfg.ssm_sigmoid_gate) sigmoid(g) else silu(g));
                    }
                }
            }
        }
    };
};

/// The shared expert (TASK-29) on `n` normalized inputs, scaled per token by
/// a sigmoid of `gate_w` dotted with the input, added into `out`.
fn sharedExpert(bufs: *const MixBuffers, se: Expert, gate_w: []const f32, x: []const f32, n: usize, out: []f32, scratch: []f32) !void
{
    const f = se.gate.rows;
    const d = se.down.rows;
    const g = bufs.g[0 .. n * f];
    const u = bufs.u[0 .. n * f];
    const y = bufs.y[0 .. n * d];
    const fused = if (gpu_matmuls) |fm| fm(&.{ se.gate, se.up }, x[0 .. n * d], n, &.{ g, u }) else false;
    if (!fused)
    {
        se.gate.matmul(x, n, g, scratch);
        se.up.matmul(x, n, u, scratch);
    }
    for (g, u) |*e, uu|
    {
        e.* = silu(e.*) * uu;
    }
    se.down.matmul(g, n, y, scratch);
    for (0..n) |t|
    {
        const s = sigmoid(dot(x[t * d ..][0..d], gate_w));
        for (out[t * d ..][0..d], y[t * d ..][0..d]) |*o, yy|
        {
            o.* += s * yy;
        }
    }
}

/// Rotary frequencies for one head (`head_dim / 2` of them) and the factor
/// cos and sin are scaled by, as transformers computes them: plain, or YaRN.
pub const Rope = struct {
    inv_freq: [max_half]f32,
    half: usize,
    scale: f32,
    /// Pairs are (2i, 2i + 1) rather than (i, i + half).
    adjacent: bool = false,

    pub const max_half = 256;

    pub fn init(cfg: Config) Rope
    {
        const dim = cfg.rotaryDim();
        var r: Rope = .{ .inv_freq = undefined, .half = dim / 2, .scale = 1, .adjacent = cfg.rope_adjacent };
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
        var low = correction(32, fdim, cfg.rope_theta, orig);
        var high = correction(1, fdim, cfg.rope_theta, orig);
        if (cfg.yarn_truncate)
        {
            low = @floor(low);
            high = @ceil(high);
        }
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
    /// pairs are (i, i + half), as in rotate_half, over the first 2 half dimensions.
    pub fn apply(self: *const Rope, x: []f32, heads: usize, head_dim: usize, pos: usize) void
    {
        var cs: [max_half]f32 = undefined;
        var sn: [max_half]f32 = undefined;
        self.angles(pos, &cs, &sn);
        self.applyAt(x, heads, head_dim, &cs, &sn);
    }

    /// The scaled cos and sin for position `pos`, `half` of each.
    pub fn angles(self: *const Rope, pos: usize, cs: []f32, sn: []f32) void
    {
        for (0..self.half) |i|
        {
            const angle = @as(f32, @floatFromInt(pos)) * self.inv_freq[i];
            cs[i] = @cos(angle) * self.scale;
            sn[i] = @sin(angle) * self.scale;
        }
    }

    /// `apply` with the cos and sin from `angles`.
    pub fn applyAt(self: *const Rope, x: []f32, heads: usize, head_dim: usize, cs_all: []const f32, sn_all: []const f32) void
    {
        if (self.adjacent)
        {
            for (0..heads) |h|
            {
                const v = x[h * head_dim ..];
                for (0..self.half) |i|
                {
                    const a = v[2 * i];
                    const b = v[2 * i + 1];
                    v[2 * i] = a * cs_all[i] - b * sn_all[i];
                    v[2 * i + 1] = b * cs_all[i] + a * sn_all[i];
                }
            }
            return;
        }
        for (0..self.half) |i|
        {
            const cs = cs_all[i];
            const sn = sn_all[i];
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
    /// Biases of each projection (gpt-oss, TASK-39); empty for none.
    gate_b: []const f32 = &.{},
    up_b: []const f32 = &.{},
    down_b: []const f32 = &.{},

    /// After the gate and up projections of `rows` calls: their biases, then the activation, into `g`.
    pub fn activate(self: Expert, g: []f32, u: []f32, rows: usize) void
    {
        const f = self.gate.rows;
        for (0..rows) |r|
        {
            const gr = g[r * f ..][0..f];
            const ur = u[r * f ..][0..f];
            if (self.gate_b.len > 0)
            {
                for (gr, self.gate_b, ur, self.up_b) |*a, ab, *b, bb|
                {
                    a.* += ab;
                    b.* += bb;
                }
            }
            if (swiglu_oai)
            {
                for (gr, ur) |*a, b|
                {
                    a.* = gptoss.swiglu(a.*, b, swiglu_limit);
                }
            }
            else
            {
                for (gr, ur) |*a, b|
                {
                    a.* = glm53.swiglu(a.*, b, swiglu_limit);
                }
            }
        }
    }

    /// After the down projection of `rows` calls: its bias.
    pub fn finish(self: Expert, y: []f32, rows: usize) void
    {
        if (self.down_b.len == 0)
        {
            return;
        }
        const d = self.down.rows;
        for (0..rows) |r|
        {
            for (y[r * d ..][0..d], self.down_b) |*o, b|
            {
                o.* += b;
            }
        }
    }

    /// Whether the GPU's expert kernels can run it: no biases or gpt-oss activation.
    pub fn plain(self: Expert) bool
    {
        return self.gate_b.len == 0 and self.down_b.len == 0 and !swiglu_oai;
    }

    /// `scratch` needs 3 * d_ff + max(d_model, d_ff) values.
    pub fn forward(self: Expert, x: []const f32, out: []f32, scratch: []f32) void {
        if (gpu_experts) |run|
        {
            const first = [_]u32{ 0, 1 };
            const zero = [_]u32{0};
            const one = [_]f32{1};
            if (run(.{ .experts = &.{self}, .first = &first, .src = &zero, .inv = &zero, .w = &one, .x = x[0..self.gate.cols], .n = 1, .k = 1, .y = out[0..self.down.rows] }))
            {
                return;
            }
        }
        self.forwardCpu(x, out, scratch);
    }

    /// `forward` without the GPU.
    pub fn forwardCpu(self: Expert, x: []const f32, out: []f32, scratch: []f32) void {
        const f = self.gate.rows;
        const g = scratch[0..f];
        const u = scratch[f..][0..f];
        const row = scratch[2 * f ..][0..@max(self.gate.cols, f)];
        self.gate.matmul(x, 1, g, row);
        self.up.matmul(x, 1, u, row);
        self.activate(g, u, 1);
        self.down.matmul(g, 1, out, row);
        self.finish(out, 1);
    }
};

/// SwiGLU input clamp for routed experts (TASK-32's GLM-5.3-Flash), 0 for none. Every process
/// that runs experts sets it from its model's config (`LocalExperts.init`, the expert cache).
pub var swiglu_limit: f32 = 0;

/// Routed experts use gpt-oss's SwiGLU (TASK-39), set with `swiglu_limit`.
pub var swiglu_oai: bool = false;

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
    /// If set, called as each step of `tokens` tokens begins and after it ends, for telemetry.
    mark: ?*const fn (ctx: *anyopaque, begin: bool, tokens: usize) void = null,
    /// If set, caps each expert's calls per layer (TASK-17); `run` skips calls marked `Capacity.skip`.
    capacity: ?*Capacity = null,
};

/// Capacity (TASK-17): the most calls each process, and optionally each expert, takes per layer in one step.
/// Calls past a limit with the lowest gate weights are dropped, so their tokens keep only the residual path.
pub const Capacity = struct {
    /// Calls per layer each rank takes (`rank_limit[r]`), and each expert.
    rank_limit: [max_ranks]u32 = @splat(none),
    expert_limit: u32 = none,
    /// Which rank owns each expert; without it every call is rank 0's.
    owner: ?[]const u8 = null,
    /// (token, expert) calls routed and dropped, and layer passes that dropped any.
    routed: u64 = 0,
    dropped: u64 = 0,
    layers_over: u64 = 0,

    /// Processes in one split (TASK-33), as the transport's 6-bit rank allows.
    pub const max_ranks = 64;
    pub const none = std.math.maxInt(u32);
    /// Marks a dropped call in `idx`.
    pub const skip = std.math.maxInt(u32);

    /// `factor` times an even share of `nominal` tokens' calls over `share`
    /// of the experts, rounded up; never below 1.
    pub fn limitFor(factor: f32, nominal: usize, cfg: Config, share: usize) u32
    {
        const even = @as(f32, @floatFromInt(nominal * cfg.top_k * share)) / @as(f32, @floatFromInt(cfg.experts));
        return @max(1, @as(u32, @intFromFloat(@min(@ceil(factor * even), 4e9))));
    }

    /// Limits for each rank (`rank_factor`) and each expert (`expert_factor`), 0 for none.
    pub fn init(cfg: Config, nominal: usize, rank_factor: f32, expert_factor: f32, owner: ?[]const u8) Capacity
    {
        var self: Capacity = .{ .owner = owner };
        if (expert_factor > 0)
        {
            self.expert_limit = limitFor(expert_factor, nominal, cfg, 1);
        }
        if (rank_factor > 0)
        {
            for (&self.rank_limit, 0..) |*l, r|
            {
                var owned: usize = 0;
                for (0..cfg.experts) |e|
                {
                    owned += @intFromBool(if (owner) |o| o[e] == r else r == 0);
                }
                if (owned > 0)
                {
                    l.* = limitFor(rank_factor, nominal, cfg, owned);
                }
            }
        }
        return self;
    }

    fn rankOf(self: *const Capacity, e: u32) usize
    {
        return if (self.owner) |o| o[e] else 0;
    }

    /// Drops calls past each expert's limit, then each rank's, keeping the highest gate weights.
    /// `count` holds one entry per expert, `list` one per call.
    pub fn apply(self: *Capacity, idx: []u32, wts: []f32, calls: usize, count: []u32, list: []u32) void
    {
        @memset(count, 0);
        var per_rank: [max_ranks]u32 = @splat(0);
        var over = false;
        for (idx[0..calls]) |e|
        {
            if (e == skip)
            {
                continue;
            }
            count[e] += 1;
            self.routed += 1;
            over = over or count[e] > self.expert_limit;
            const r = self.rankOf(e);
            per_rank[r] += 1;
            over = over or per_rank[r] > self.rank_limit[r];
        }
        if (!over)
        {
            return;
        }
        self.layers_over += 1;
        for (count, 0..) |cnt, e|
        {
            if (cnt > self.expert_limit)
            {
                const ctx: Pick = .{ .cap = self, .expert = @intCast(e) };
                per_rank[self.rankOf(@intCast(e))] -= self.dropOver(idx, wts, calls, list, ctx, self.expert_limit);
            }
        }
        for (per_rank, 0..) |cnt, r|
        {
            if (cnt > self.rank_limit[r])
            {
                _ = self.dropOver(idx, wts, calls, list, .{ .cap = self, .rank = r }, self.rank_limit[r]);
            }
        }
    }

    /// Which calls a limit covers: one expert's, or one rank's.
    const Pick = struct {
        cap: *const Capacity,
        expert: ?u32 = null,
        rank: usize = 0,

        fn has(self: Pick, e: u32) bool
        {
            if (e == skip)
            {
                return false;
            }
            return if (self.expert) |x| e == x else self.cap.rankOf(e) == self.rank;
        }
    };

    /// Keeps the `limit` highest-weight calls `pick` covers; returns how many it dropped.
    fn dropOver(self: *Capacity, idx: []u32, wts: []f32, calls: usize, list: []u32, pick: Pick, limit: u32) u32
    {
        var m: usize = 0;
        for (idx[0..calls], 0..) |x, s|
        {
            if (pick.has(x))
            {
                list[m] = @intCast(s);
                m += 1;
            }
        }
        if (m <= limit)
        {
            return 0;
        }
        const byWeight = struct {
            fn lt(w: []const f32, a: u32, b: u32) bool
            {
                return w[a] > w[b] or (w[a] == w[b] and a < b);
            }
        }.lt;
        std.mem.sort(u32, list[0..m], wts, byWeight);
        for (list[limit..m]) |s|
        {
            idx[s] = skip;
            wts[s] = 0;
        }
        self.dropped += m - limit;
        return @intCast(m - limit);
    }
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
    /// Attention weights; empty in a Gated DeltaNet layer. With a gate, `q` gives each head's query then its gate.
    q: Mat = Mat.empty,
    k: Mat = Mat.empty,
    v: Mat = Mat.empty,
    o: Mat = Mat.empty,
    q_norm: []f32 = &.{},
    k_norm: []f32 = &.{},
    ffn_norm: []f32,
    router: Mat,
    /// Set in a Gated DeltaNet layer (TASK-29).
    lin: ?Linear = null,
    /// The shared expert every token goes through, scaled by a sigmoid of `shared_gate` dotted with the input.
    shared: ?Expert = null,
    shared_gate: []f32 = &.{},
    /// With hyper-connections (TASK-31), the mixes in place of `attn_norm` and `ffn_norm`.
    hc_attn: ?Hyper = null,
    hc_ffn: ?Hyper = null,
    /// QSA's indexer, in an attention layer of a model with QSA.
    indexer: ?Indexer = null,
    /// The n-gram embedding, in its one layer.
    ple: ?PleLayer = null,
    /// GLM-5.3-Flash's blocks (TASK-32); `router` is empty in its dense layers.
    glm: ?glm53.Parts = null,
    /// DeepSeek V3's blocks (TASK-39); `router` is empty in its dense layers.
    ds: ?deepseek.Parts = null,
    /// gpt-oss's biases and sinks (TASK-39).
    oss: ?gptoss.Parts = null,
};

/// A hyper-connection (TASK-31): an RMSNorm over each stream (`norm`, streams x d_model),
/// a low-rank gate (`down`, then `up`) that mixes the streams into one input, and with
/// `inject` rows after `down`'s, each stream's weight for the block's output.
pub const Hyper = struct {
    norm: []f32,
    /// hc_rank rows, then `streams` inject rows unless this is the final mix.
    down: Mat,
    up: Mat,
    inject: bool,
};

/// QSA's indexer (TASK-31): query heads and one key, each with an RMSNorm.
pub const Indexer = struct {
    q: Mat,
    k: Mat,
    q_norm: []f32,
    k_norm: []f32,
};

/// The n-gram embedding's layer weights (TASK-31); the table itself is in `Weights.ple`.
pub const PleLayer = struct {
    key: Mat,
    value: Mat,
    norm_key: []f32,
    norm_query: []f32,
    norm_conv: []f32,
    /// residual() rows of ple_conv taps, oldest input first.
    conv: []f32,
};

/// The n-gram embedding's hashed tables (TASK-31): one table per hash head, stacked
/// as `offsets` and `sizes` say, and each n-gram position's multiplier.
pub const Ple = struct {
    table: Mat,
    mult: [Config.max_ngram]u64,
    offsets: [Config.max_ple_heads]u64,
    sizes: [Config.max_ple_heads]u64,
};

/// A Gated DeltaNet layer's weights, in llama.cpp's layout: value heads tiled,
/// so value head h reads key head h mod key heads, and `a` already -exp(A_log).
pub const Linear = struct {
    /// Queries, keys, values (convDim rows), then the output gate, beta, and alpha projections.
    qkv: Mat,
    z: Mat,
    beta: Mat,
    alpha: Mat,
    a: []f32,
    dt_bias: []f32,
    /// convDim rows of ssm_conv taps, oldest input first.
    conv: []f32,
    norm: []f32,
    out: Mat,
};

/// Every weight except the experts, pointing into a region loaded by a plan
/// (norm vectors are copied out as aligned f32).
pub const Weights = struct {
    cfg: Config,
    embed: Mat,
    layers: []Layer,
    /// Empty with hyper-connections, which end in `out_hc` instead.
    out_norm: []f32,
    head: Mat,
    out_hc: ?Hyper = null,
    ple: ?Ple = null,
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
        if (cfg.arch == .glm5next)
        {
            return glm53.weights(gpa, h, cfg, plan, region);
        }
        if (cfg.arch == .deepseek2)
        {
            return deepseek.weights(gpa, h, cfg, plan, region);
        }
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
            /// A tensor of any shape holding `n` values, as floats.
            fn flat(s: @This(), name: []const u8, n: usize) ![]f32 {
                const t = try s.info(name);
                var count: usize = 1;
                for (t.dims[0..t.ndim]) |dim|
                {
                    count *= @intCast(dim);
                }
                if (count != n)
                {
                    std.log.err("{s}: dims {any}, expected {d} values", .{ name, t.dims[0..t.ndim], n });
                    return error.BadShape;
                }
                const out = try s.al.alloc(f32, n);
                try quant.dequantize(t.format, s.p.locate(s.reg, .{ .offset = t.offset, .len = t.bytes }), out);
                return out;
            }
        };
        const g: Get = .{ .hh = h, .p = plan, .reg = region, .al = a };
        const d = cfg.d_model;
        const hyper = cfg.streams > 1;
        const R = cfg.residual();
        const Hc = struct {
            fn get(gg: Get, c: Config, prefix: []const u8, inject: bool) !Hyper
            {
                var b: [64]u8 = undefined;
                const fmt = std.fmt.bufPrint;
                const r = c.residual();
                return .{
                    .norm = try gg.vec(fmt(&b, "{s}_norm.weight", .{prefix}) catch unreachable, r),
                    .down = if (inject) try gg.mat(fmt(&b, "{s}_down_inject.weight", .{prefix}) catch unreachable, c.hc_rank + c.streams, r) else try gg.mat(fmt(&b, "{s}_down.weight", .{prefix}) catch unreachable, c.hc_rank, r),
                    .up = try gg.mat(fmt(&b, "{s}_up.weight", .{prefix}) catch unreachable, r, c.hc_rank),
                    .inject = inject,
                };
            }
        };
        self.embed = try g.mat("token_embd.weight", cfg.vocab, d);
        self.head = try g.mat("output.weight", cfg.vocab, d);
        if (hyper)
        {
            self.out_norm = &.{};
            self.out_hc = try Hc.get(g, cfg, "output_hc", false);
        }
        else
        {
            self.out_norm = try g.vec("output_norm.weight", d);
        }
        if (cfg.ple_layer != null)
        {
            const t = try g.info("per_layer_token_embd.weight");
            var p: Ple = .{ .table = undefined, .mult = @splat(0), .offsets = @splat(0), .sizes = @splat(0) };
            p.table = try g.mat("per_layer_token_embd.weight", @intCast(t.dims[1]), cfg.ple_row);
            for (0..cfg.ple_ngram) |i|
            {
                p.mult[i] = @bitCast(h.intAt("qwen4exp.ple.layer_multipliers", i) orelse return error.MissingKey);
            }
            for (0..cfg.pleHeads()) |i|
            {
                p.offsets[i] = @bitCast(h.intAt("qwen4exp.ple.head_offsets", i) orelse return error.MissingKey);
                p.sizes[i] = @bitCast(h.intAt("qwen4exp.ple.head_vocab_sizes", i) orelse return error.MissingKey);
                if (p.sizes[i] == 0 or p.offsets[i] + p.sizes[i] > p.table.rows)
                {
                    return error.BadShape;
                }
            }
            self.ple = p;
        }
        self.layers = try a.alloc(Layer, cfg.layers);
        for (self.layers, 0..) |*l, i| {
            var buf: [64]u8 = undefined;
            const n = struct {
                fn f(b: []u8, layer: usize, s: []const u8) []const u8 {
                    return std.fmt.bufPrint(b, "blk.{d}.{s}.weight", .{ layer, s }) catch unreachable;
                }
            }.f;
            // OLMoE normalizes the whole projection; Qwen3 each head.
            const qn = if (cfg.arch == .olmoe) cfg.qDim() else cfg.head_dim;
            const kn = if (cfg.arch == .olmoe) cfg.kvDim() else cfg.head_dim;
            l.* = .{
                .attn_norm = if (hyper) &.{} else try g.vec(n(&buf, i, "attn_norm"), d),
                .ffn_norm = if (hyper) &.{} else try g.vec(n(&buf, i, if (cfg.arch == .qwen35moe or cfg.arch == .gptoss) "post_attention_norm" else "ffn_norm"), d),
                .router = try g.mat(n(&buf, i, "ffn_gate_inp"), cfg.experts, d),
            };
            if (hyper)
            {
                var b2: [64]u8 = undefined;
                l.hc_attn = try Hc.get(g, cfg, std.fmt.bufPrint(&b2, "blk.{d}.hc_attn", .{i}) catch unreachable, true);
                l.hc_ffn = try Hc.get(g, cfg, std.fmt.bufPrint(&b2, "blk.{d}.hc_ffn", .{i}) catch unreachable, true);
            }
            if (cfg.ple_layer == i)
            {
                const e = cfg.pleHeads() * cfg.ple_row;
                l.ple = .{
                    .key = try g.mat(n(&buf, i, "ple_key"), R, e),
                    .value = try g.mat(n(&buf, i, "ple_value"), d, e),
                    .norm_key = try g.vec(n(&buf, i, "ple_norm_key"), R),
                    .norm_query = try g.vec(n(&buf, i, "ple_norm_query"), R),
                    .norm_conv = try g.vec(n(&buf, i, "ple_norm_conv"), R),
                    .conv = try g.flat(n(&buf, i, "ple_conv1d"), R * cfg.ple_conv),
                };
            }
            if (cfg.recurrent(i) and cfg.arch == .qwen4exp)
            {
                // llama.cpp merges weights that read the same input: [qkv; z] and [beta; alpha].
                var b2: [64]u8 = undefined;
                const inner = cfg.ssm_v_heads * cfg.ssm_dv;
                const qkvz = try g.mat(n(&buf, i, "attn_qkv_gate"), cfg.convDim() + inner, d);
                const ba = try g.mat(n(&buf, i, "ssm_beta_alpha"), 2 * cfg.ssm_v_heads, d);
                l.lin = .{
                    .qkv = qkvz.rowRange(0, cfg.convDim()),
                    .z = qkvz.rowRange(cfg.convDim(), inner),
                    .beta = ba.rowRange(0, cfg.ssm_v_heads),
                    .alpha = ba.rowRange(cfg.ssm_v_heads, cfg.ssm_v_heads),
                    .a = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_a", .{i}) catch unreachable, cfg.ssm_v_heads),
                    .dt_bias = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_dt.bias", .{i}) catch unreachable, cfg.ssm_v_heads),
                    .conv = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_conv1d.weight", .{i}) catch unreachable, cfg.convDim() * cfg.ssm_conv),
                    .norm = try g.vec(n(&buf, i, "ssm_norm"), cfg.ssm_dv),
                    .out = try g.mat(n(&buf, i, "ssm_out"), d, inner),
                };
            }
            else if (cfg.recurrent(i))
            {
                var b2: [64]u8 = undefined;
                const inner = cfg.ssm_v_heads * cfg.ssm_dv;
                const conv = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_conv1d.weight", .{i}) catch unreachable, cfg.convDim() * cfg.ssm_conv);
                l.lin = .{
                    .qkv = try g.mat(n(&buf, i, "attn_qkv"), cfg.convDim(), d),
                    .z = try g.mat(n(&buf, i, "attn_gate"), inner, d),
                    .beta = try g.mat(n(&buf, i, "ssm_beta"), cfg.ssm_v_heads, d),
                    .alpha = try g.mat(n(&buf, i, "ssm_alpha"), cfg.ssm_v_heads, d),
                    .a = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_a", .{i}) catch unreachable, cfg.ssm_v_heads),
                    .dt_bias = try g.flat(std.fmt.bufPrint(&b2, "blk.{d}.ssm_dt.bias", .{i}) catch unreachable, cfg.ssm_v_heads),
                    .conv = conv,
                    .norm = try g.vec(n(&buf, i, "ssm_norm"), cfg.ssm_dv),
                    .out = try g.mat(n(&buf, i, "ssm_out"), d, inner),
                };
            }
            else
            {
                const gate: usize = if (cfg.attn_gate) 2 else 1;
                l.q = try g.mat(n(&buf, i, "attn_q"), gate * cfg.qDim(), d);
                l.k = try g.mat(n(&buf, i, "attn_k"), cfg.kvDim(), d);
                l.v = try g.mat(n(&buf, i, "attn_v"), cfg.kvDim(), d);
                l.o = try g.mat(n(&buf, i, "attn_output"), d, cfg.qDim());
                // Mixtral and gpt-oss have no QK norm.
                if (cfg.arch != .mixtral and cfg.arch != .gptoss)
                {
                    l.q_norm = try g.vec(n(&buf, i, "attn_q_norm"), qn);
                    l.k_norm = try g.vec(n(&buf, i, "attn_k_norm"), kn);
                }
                if (cfg.arch == .gptoss)
                {
                    l.oss = try gptoss.parts(h, plan, region, a, cfg, i);
                }
                if (cfg.qsa_heads > 0)
                {
                    l.indexer = .{
                        .q = try g.mat(n(&buf, i, "indexer.q_proj"), cfg.qsa_heads * cfg.qsa_dim, d),
                        .k = try g.mat(n(&buf, i, "indexer.k_proj"), cfg.qsa_dim, d),
                        .q_norm = try g.vec(n(&buf, i, "indexer.q_norm"), cfg.qsa_dim),
                        .k_norm = try g.vec(n(&buf, i, "indexer.k_norm"), cfg.qsa_dim),
                    };
                }
            }
            if (cfg.shared_ff != 0)
            {
                l.shared = .{
                    .gate = try g.mat(n(&buf, i, "ffn_gate_shexp"), cfg.shared_ff, d),
                    .up = try g.mat(n(&buf, i, "ffn_up_shexp"), cfg.shared_ff, d),
                    .down = try g.mat(n(&buf, i, "ffn_down_shexp"), d, cfg.shared_ff),
                };
                l.shared_gate = try g.flat(n(&buf, i, "ffn_gate_inp_shexp"), d);
            }
        }
        return self;
    }
};

/// Per-layer values recorded during `forward`, for comparison with the
/// reference: hidden state after each layer (layers x n x d_model) and the
/// chosen experts (layers x n x top_k).
pub const Trace = struct {
    /// layers x n x residual(): with hyper-connections, every stream.
    hidden: []f32,
    topk: []u32,
    /// With QSA (TASK-31): per attention layer and token, the blocks it kept in
    /// ascending order (attention layers x n x qsaTop()), or -1s where it kept every block.
    qsa: []i32 = &.{},
    /// Optional, layers x n: each token's gap between its k-th and (k+1)-th router choice score.
    gaps: []f32 = &.{},
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
/// Each position is one row of keys and one of values, `row` bytes each, in `format` (TASK-27).
pub const KvCache = struct {
    cap: usize,
    d: usize,
    kv_heads: usize,
    format: KvFormat = .f32,
    /// Bytes per position of the keys (index 0) and the values (1).
    row: [2]usize,
    /// Bytes from one layer's first position to the next layer's: whole pages, for O_DIRECT.
    layer_bytes: [2]usize = .{ 0, 0 },
    len: usize = 0,
    k: []u8,
    v: []u8,
    /// Device addresses of `k` and `v` when a GPU reads them in place, else 0.
    dev_k: u64 = 0,
    dev_v: u64 = 0,
    /// Set for a cache whose cold middle lives on NVMe (TASK-12); `k` and `v` are then empty.
    tier: ?Tier = null,
    /// Each Gated DeltaNet layer's state after `len` positions (TASK-29), `state_floats` apiece.
    state: []f32 = &.{},
    state_floats: usize = 0,
    /// Device address of `state` when a GPU updates it in place, else 0.
    dev_state: u64 = 0,
    /// QSA (TASK-31), per attention layer: each position's indexer key, then each complete block's
    /// pooled key, `qsa_floats` apiece.
    qsa: []f32 = &.{},
    qsa_floats: usize = 0,
    qsa_dim: usize = 0,
    /// Device address of `qsa` when a GPU reads it in place, else 0.
    dev_qsa: u64 = 0,
    /// DSA (TASK-32), per MLA layer: each position's indexer key and pooling gate, then each
    /// complete pool's key, `dsa_floats` apiece.
    dsa: []f32 = &.{},
    dsa_floats: usize = 0,
    dsa_dim: usize = 0,
    dsa_pool: usize = 1,
    /// The n-gram embedding's state (TASK-31): the last tokens before `len`, and its convolution's past inputs.
    ple_ctx: [Config.max_ngram]u32 = @splat(0),
    ple_eos: u32 = 0,
    ple_conv: []f32 = &.{},

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

    /// A cache that keeps its positions through `tier`, in `format`, with room for `cap`.
    /// What is not keys and values (Gated DeltaNet state, QSA keys, n-gram state) stays in memory.
    pub fn tiered(cfg: Config, cap_in: usize, format: KvFormat, tier: Tier) !KvCache
    {
        // Rounded up to whole tiles, as `initAs` does, so QSA's block keys line up with it.
        const cap = (cap_in + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions * AttentionJob.tile_positions;
        var self: KvCache = .{ .cap = cap, .d = cfg.kvDim(), .kv_heads = cfg.kv_heads, .format = format, .row = format.rows(cfg.kvDim(), cfg.kv_heads), .k = &.{}, .v = &.{}, .tier = tier };
        try self.addExtras(cfg);
        return self;
    }

    /// Everything but the keys and values, for `cap` positions: allocated, reset, and mapped for a GPU.
    fn addExtras(self: *KvCache, cfg: Config) !void
    {
        if (cfg.recurrentLayers() > 0)
        {
            self.state_floats = cfg.stateFloats();
            self.state = try big_allocator.alloc(f32, cfg.recurrentLayers() * self.state_floats);
            @memset(self.state, 0);
        }
        errdefer big_allocator.free(self.state);
        if (cfg.qsa_heads > 0)
        {
            self.qsa_dim = cfg.qsa_dim;
            self.qsa_floats = self.cap * cfg.qsa_dim + self.cap / cfg.qsa_block * cfg.qsa_dim;
            self.qsa = try big_allocator.alloc(f32, cfg.attnLayers() * self.qsa_floats);
        }
        errdefer big_allocator.free(self.qsa);
        if (cfg.dsa_heads > 0)
        {
            self.dsa_dim = cfg.dsa_dim;
            self.dsa_pool = cfg.dsa_pool;
            self.dsa_floats = 2 * self.cap * cfg.dsa_dim + self.cap / cfg.dsa_pool * cfg.dsa_dim;
            self.dsa = try big_allocator.alloc(f32, cfg.attnLayers() * self.dsa_floats);
        }
        errdefer big_allocator.free(self.dsa);
        if (cfg.ple_layer != null)
        {
            self.ple_eos = cfg.ple_eos;
            self.ple_conv = try big_allocator.alloc(f32, cfg.pleHeld() * cfg.residual());
        }
        self.resetPle();
        if (gpu_register) |reg|
        {
            if (self.state.len > 0 and gpu_delta_rule != null)
            {
                self.dev_state = reg(std.mem.sliceAsBytes(self.state)) orelse 0;
            }
            if (self.qsa.len > 0 and gpu_qsa != null)
            {
                self.dev_qsa = reg(std.mem.sliceAsBytes(self.qsa)) orelse 0;
            }
        }
    }

    fn freeExtras(self: *KvCache) void
    {
        if (gpu_unregister) |un|
        {
            if (self.dev_state != 0)
            {
                un(std.mem.sliceAsBytes(self.state));
            }
            if (self.dev_qsa != 0)
            {
                un(std.mem.sliceAsBytes(self.qsa));
            }
        }
        big_allocator.free(self.state);
        big_allocator.free(self.qsa);
        big_allocator.free(self.dsa);
        big_allocator.free(self.ple_conv);
    }

    /// Device address of an attention layer's complete blocks' pooled keys, or 0 without a GPU.
    pub fn qsaBlocksDev(self: KvCache, layer: usize) u64
    {
        return if (self.dev_qsa == 0) 0 else self.dev_qsa + (layer * self.qsa_floats + self.cap * self.qsa_dim) * @sizeOf(f32);
    }

    /// Room for `cap` positions in every layer, in f32.
    pub fn init(a: std.mem.Allocator, cfg: Config, cap: usize) !KvCache
    {
        return initAs(a, cfg, cap, .f32);
    }

    /// Room for `cap` positions in every layer, in `format`, from `big_allocator`.
    /// Page-aligned, so a GPU can map it; `a` is unused and kept for callers.
    pub fn initAs(a: std.mem.Allocator, cfg: Config, cap: usize, format: KvFormat) !KvCache
    {
        _ = a;
        const d = cfg.kvDim();
        var row = format.rows(d, cfg.kv_heads);
        // MLA models keep one latent per position (with DeepSeek's rotated key part) and no values.
        if (cfg.mla_kv_rank > 0)
        {
            row[1] = 0;
        }
        // Positions rounded up to whole tiles, so the GPU never reads past the end.
        const room = (cap + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions * AttentionJob.tile_positions;
        const layer_bytes: [2]usize = .{
            std.mem.alignForward(usize, room * row[0], std.heap.page_size_max),
            std.mem.alignForward(usize, room * row[1], std.heap.page_size_max),
        };
        // Only attention layers keep keys and values; a Gated DeltaNet layer keeps a fixed-size state.
        const k = try allocPages(cfg.attnLayers() * layer_bytes[0]);
        errdefer freePages(k);
        const v: []u8 = if (layer_bytes[1] == 0) @constCast(&[_]u8{}) else try allocPages(cfg.attnLayers() * layer_bytes[1]);
        errdefer freePages(v);
        var self: KvCache = .{ .cap = room, .d = d, .kv_heads = cfg.kv_heads, .format = format, .row = row, .layer_bytes = layer_bytes, .k = k, .v = v };
        try self.addExtras(cfg);
        errdefer self.freeExtras();
        if (gpu_register) |reg|
        {
            if (AttentionJob.gpuHeadDim(cfg.head_dim) and v.len > 0)
            {
                if (reg(k)) |dk|
                {
                    if (reg(v)) |dv|
                    {
                        self.dev_k = dk;
                        self.dev_v = dv;
                    }
                    else
                    {
                        gpu_unregister.?(k);
                    }
                }
            }
        }
        return self;
    }

    pub fn deinit(self: *KvCache, a: std.mem.Allocator) void
    {
        _ = a;
        self.freeExtras();
        if (self.tier != null)
        {
            self.* = undefined;
            return;
        }
        if (self.dev_k != 0)
        {
            if (gpu_unregister) |un|
            {
                un(self.k);
                un(self.v);
            }
        }
        freePages(self.k);
        freePages(self.v);
        self.* = undefined;
    }

    /// Empties the cache, so the next step starts at position 0.
    pub fn reset(self: *KvCache) void
    {
        self.len = 0;
        @memset(self.state, 0);
        self.resetPle();
    }

    /// Before the first token, n-grams look back on end-of-text tokens and the convolution on zeros.
    fn resetPle(self: *KvCache) void
    {
        self.ple_ctx = @splat(self.ple_eos);
        @memset(self.ple_conv, 0);
    }

    /// One attention layer's QSA indexer keys, `qsa_dim` per position.
    pub fn qsaKeys(self: KvCache, layer: usize) []f32
    {
        return self.qsa[layer * self.qsa_floats ..][0 .. self.cap * self.qsa_dim];
    }

    /// Its complete blocks' pooled keys, normalized and rotated, `qsa_dim` per block.
    pub fn qsaBlocks(self: KvCache, layer: usize) []f32
    {
        return self.qsa[layer * self.qsa_floats + self.cap * self.qsa_dim ..][0 .. self.qsa_floats - self.cap * self.qsa_dim];
    }

    /// Keeps the first `len` positions and returns how many it kept. A Gated
    /// DeltaNet state cannot step back, so a hybrid model keeps all or nothing.
    pub fn truncate(self: *KvCache, len: usize) usize
    {
        if (len >= self.len)
        {
            return self.len;
        }
        if (self.state.len > 0)
        {
            self.reset();
            return 0;
        }
        self.len = len;
        return len;
    }

    /// One MLA layer's DSA indexer keys, `dsa_dim` per position.
    pub fn dsaKeys(self: KvCache, layer: usize) []f32
    {
        return self.dsa[layer * self.dsa_floats ..][0 .. self.cap * self.dsa_dim];
    }

    /// Its pooling gate scores, `dsa_dim` per position.
    pub fn dsaGates(self: KvCache, layer: usize) []f32
    {
        return self.dsa[layer * self.dsa_floats + self.cap * self.dsa_dim ..][0 .. self.cap * self.dsa_dim];
    }

    /// Its complete pools' keys, `dsa_dim` per pool.
    pub fn dsaPools(self: KvCache, layer: usize) []f32
    {
        return self.dsa[layer * self.dsa_floats + 2 * self.cap * self.dsa_dim ..][0 .. self.cap / self.dsa_pool * self.dsa_dim];
    }

    /// The state of the `index`-th Gated DeltaNet layer.
    pub fn recurrentState(self: KvCache, index: usize) []f32
    {
        return self.state[index * self.state_floats ..][0..self.state_floats];
    }

    /// Its device address, or 0 without a GPU.
    pub fn recurrentStateDev(self: KvCache, index: usize) u64
    {
        return if (self.dev_state == 0) 0 else self.dev_state + index * self.state_floats * @sizeOf(f32);
    }

    const page_align = std.mem.Alignment.fromByteUnits(std.heap.page_size_max);

    fn allocPages(n: usize) ![]u8
    {
        const bytes = big_allocator.rawAlloc(n, page_align, @returnAddress()) orelse return error.OutOfMemory;
        return bytes[0..n];
    }

    fn freePages(s: []u8) void
    {
        if (s.len == 0)
        {
            return;
        }
        big_allocator.rawFree(s, page_align, @returnAddress());
    }

    /// Device addresses of each tile of `layer` covering positions 0 to `len` - 1.
    pub fn tiles(self: KvCache, layer: usize, len: usize, out_k: []u64, out_v: []u64) void
    {
        const tp = AttentionJob.tile_positions;
        for (0..(len + tp - 1) / tp) |i|
        {
            out_k[i] = self.dev_k + layer * self.layer_bytes[0] + i * tp * self.row[0];
            out_v[i] = self.dev_v + layer * self.layer_bytes[1] + i * tp * self.row[1];
        }
    }

    /// Adds `n` positions' keys and values (n x kvd each) from `pos0` on.
    pub fn store(self: *KvCache, layer: usize, pos0: usize, n: usize, k: []const f32, v: []const f32) void
    {
        if (self.tier) |t|
        {
            return t.store(t.ctx, layer, pos0, n, k, v);
        }
        const dk = self.keys(layer)[pos0 * self.row[0] ..];
        const dv = self.values(layer)[pos0 * self.row[1] ..];
        if (self.format == .f32)
        {
            @memcpy(dk[0 .. n * self.row[0]], std.mem.sliceAsBytes(k[0 .. n * self.d]));
            if (self.row[1] > 0)
            {
                @memcpy(dv[0 .. n * self.row[1]], std.mem.sliceAsBytes(v[0 .. n * self.d]));
            }
            return;
        }
        const Rows = struct {
            c: *const KvCache,
            k: []const f32,
            v: []const f32,
            dk: []u8,
            dv: []u8,
            fn run(w: @This(), first: usize, end: usize) void
            {
                const c = w.c;
                for (first..end) |i|
                {
                    c.format.part(0).encodeRow(w.k[i * c.d ..][0..c.d], w.dk[i * c.row[0] ..][0..c.row[0]], c.kv_heads);
                    c.format.part(1).encodeRow(w.v[i * c.d ..][0..c.d], w.dv[i * c.row[1] ..][0..c.row[1]], c.kv_heads);
                }
            }
        };
        // Rows are encoded independently, so a long prefill chunk spreads them over threads.
        const threads = if (n >= parallel_tokens) cpuThreads() else 1;
        pool.parallelFor(threads, n, Rows{ .c = self, .k = k, .v = v, .dk = dk, .dv = dv }, Rows.run);
    }

    /// One layer's keys, `row[0]` bytes per position.
    pub fn keys(self: KvCache, layer: usize) []u8
    {
        return self.k[layer * self.layer_bytes[0] ..][0 .. self.cap * self.row[0]];
    }

    pub fn values(self: KvCache, layer: usize) []u8
    {
        return self.v[layer * self.layer_bytes[1] ..][0 .. self.cap * self.row[1]];
    }

    /// One layer's keys (kind 0) or values (1) with the padding after them: page-aligned and whole pages.
    pub fn pages(self: KvCache, layer: usize, kind: usize) []u8
    {
        const lb = self.layer_bytes[kind];
        return (if (kind == 0) self.k else self.v)[layer * lb ..][0..lb];
    }

    /// Bytes one position takes in one layer, keys and values together.
    pub fn positionBytes(self: KvCache) usize
    {
        return self.row[0] + self.row[1];
    }

    /// One layer's keys as floats, for an f32 cache.
    pub fn keysF32(self: KvCache, layer: usize) []f32
    {
        std.debug.assert(self.format == .f32);
        return @alignCast(std.mem.bytesAsSlice(f32, self.keys(layer)));
    }

    pub fn valuesF32(self: KvCache, layer: usize) []f32
    {
        std.debug.assert(self.format == .f32);
        return @alignCast(std.mem.bytesAsSlice(f32, self.values(layer)));
    }
};

/// Runs `tokens` (positions 0 to n-1) through the model and writes every
/// position's logits (n x vocab).
pub fn forward(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, tokens: []const u32, logits: []f32, trace: ?Trace) !void
{
    return forwardAs(gpa, w, moe, tokens, logits, trace, .f32);
}

/// `forward` with the KV cache in `format` (TASK-27).
pub fn forwardAs(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, tokens: []const u32, logits: []f32, trace: ?Trace, format: KvFormat) !void
{
    var cache = try KvCache.initAs(gpa, w.cfg, tokens.len, format);
    defer cache.deinit(gpa);
    try step(gpa, w, moe, &cache, tokens, logits, trace);
}

/// Runs `tokens` at the positions after those already in `cache`, adding
/// them to it. `logits` takes every new position's logits (n x vocab), or
/// only the last one's if it holds fewer rows than that.
pub fn step(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, cache: *KvCache, tokens: []const u32, logits: []f32, trace: ?Trace) !void
{
    if (w.cfg.arch == .glm5next)
    {
        return glm53.step(gpa, w, moe, cache, tokens, logits, trace);
    }
    if (w.cfg.arch == .deepseek2)
    {
        return deepseek.step(gpa, w, moe, cache, tokens, logits, trace);
    }
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
    if (moe.mark) |m|
    {
        m(moe.ctx, true, n);
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const qd = cfg.qDim();
    const kvd = cfg.kvDim();
    const group = cfg.heads / cfg.kv_heads;
    const rope: Rope = .init(cfg);
    // With hyper-connections, each token's residual is several streams, and `inj` weighs a block's output per stream.
    const R = cfg.residual();
    const h = try a.alloc(f32, n * R);
    const inj = try a.alloc(f32, n * cfg.streams);
    const hc_bufs = try qwen38.Buffers.init(a, cfg, n);
    const iq = try a.alloc(f32, n * cfg.qsa_heads * cfg.qsa_dim);
    const x = try a.alloc(f32, n * d);
    const q = try a.alloc(f32, n * qd);
    // With an output gate, the query projection gives each head's query then its gate.
    const qg = try a.alloc(f32, if (cfg.attn_gate) 2 * n * qd else 0);
    const gate = try a.alloc(f32, if (cfg.attn_gate) n * qd else 0);
    const att = try a.alloc(f32, n * qd);
    const tmp = try a.alloc(f32, n * @max(d, qd));
    // Cos and sin for each new position, shared by every layer.
    const rope_cs = try a.alloc(f32, n * rope.half);
    const rope_sn = try a.alloc(f32, n * rope.half);
    const scores = try a.alloc(f32, pos0 + n);
    const kn = try a.alloc(f32, n * kvd);
    const vn = try a.alloc(f32, n * kvd);
    const tile_count = (pos0 + n + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions;
    const tiles_k = try a.alloc(u64, tile_count);
    const tiles_v = try a.alloc(u64, tile_count);
    const scratch = try a.alloc(f32, @max(@max(d, qd), @max(cfg.d_ff, @max(cfg.ssm_v_heads * cfg.ssm_dv, cfg.shared_ff))));
    const kh = try a.alloc(f32, cfg.head_dim);
    const vh = try a.alloc(f32, cfg.head_dim);
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
    const cap_count = try a.alloc(u32, if (moe.capacity != null) cfg.experts else 0);
    const cap_list = try a.alloc(u32, if (moe.capacity != null) n * k else 0);
    const mix_bufs = try MixBuffers.init(a, cfg, n);

    for (tokens, 0..) |tok, t|
    {
        if (tok >= cfg.vocab)
        {
            return error.BadToken;
        }
        const ht = h[t * R ..][0..R];
        w.embed.row(tok, ht[0..d]);
        for (1..cfg.streams) |s|
        {
            @memcpy(ht[s * d ..][0..d], ht[0..d]);
        }
    }
    const guess_inj = try a.alloc(f32, if (la != null) n * cfg.streams else 0);
    const ahead: Guesser = .{ .w = w, .moe = moe, .n = n, .kx = kx, .out = guess, .val = guess_val, .votes = votes, .wanted_list = wanted_list, .hc = &hc_bufs, .inj = guess_inj };
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
    // Loops over tokens split across threads in long passes; each token comes out the same.
    const threads = if (n >= parallel_tokens) cpuThreads() else 1;
    pool.parallelFor(threads, n, TokenLoops.Angles{ .rope = &rope, .pos0 = pos0, .cs = rope_cs, .sn = rope_sn }, TokenLoops.Angles.run);
    for (w.layers, 0..) |*l, li|
    {
        const layer_t0 = if (la != null) nowNs() else 0;
        var pt = nowNs();
        if (l.ple) |pl|
        {
            try qwen38.ple(a, cfg, w.ple.?, pl, &hc_bufs, &cache.ple_ctx, cache.ple_conv, tokens, h);
        }
        if (l.hc_attn) |hy|
        {
            qwen38.mix(cfg, hy, &hc_bufs, h, n, x, inj);
        }
        else
        {
            pool.parallelFor(threads, n, TokenLoops.Norm{ .h = h, .x = x, .w = l.attn_norm, .eps = cfg.eps, .d = d }, TokenLoops.Norm.run);
        }
        if (l.lin) |lin|
        {
            const ri = cfg.recurrentIndex(li);
            try recurrentMix(&mix_bufs, cfg, lin, cache.recurrentState(ri), cache.recurrentStateDev(ri), x, n, tmp, scratch);
            Profile.lap(&profile.recurrent, &pt);
        }
        else
        {
            const ai = cfg.attnIndex(li);
            const qp = if (cfg.attn_gate) qg else q;
            const fused = if (gpu_matmuls) |f| f(&.{ l.q, l.k, l.v }, x[0 .. n * d], n, &.{ qp[0 .. n * l.q.rows], kn[0 .. n * kvd], vn[0 .. n * kvd] }) else false;
            if (!fused)
            {
                l.q.matmul(x, n, qp, scratch);
                l.k.matmul(x, n, kn, scratch);
                l.v.matmul(x, n, vn, scratch);
            }
            if (l.oss) |os|
            {
                gptoss.addRows(qp[0 .. n * qd], os.q_b);
                gptoss.addRows(kn[0 .. n * kvd], os.k_b);
                gptoss.addRows(vn[0 .. n * kvd], os.v_b);
            }
            if (cfg.attn_gate)
            {
                pool.parallelFor(threads, n, TokenLoops.SplitGate{ .qg = qg, .q = q, .gate = gate, .heads = cfg.heads, .hd = cfg.head_dim }, TokenLoops.SplitGate.run);
            }
            Profile.lap(&profile.projections, &pt);
            pool.parallelFor(threads, n, TokenLoops.QkRope{ .cfg = cfg, .rope = &rope, .q = q, .k = kn, .q_norm = l.q_norm, .k_norm = l.k_norm, .cs = rope_cs, .sn = rope_sn }, TokenLoops.QkRope.run);
            cache.store(ai, pos0, n, kn, vn);
            // QSA keeps only some blocks once there are more than its budget; until then attention is dense.
            const sparse = if (l.indexer) |ix| blk: {
                qwen38.index(cfg, ix, &rope, rope_cs, rope_sn, x, n, pos0, cache.qsaKeys(ai), cache.qsaBlocks(ai), iq, scratch);
                break :blk !qwen38.dense(cfg, pos0, n);
            } else false;
            const kept: []i32 = if (trace) |tr| (if (tr.qsa.len > 0) tr.qsa[ai * n * cfg.qsaTop() ..][0 .. n * cfg.qsaTop()] else &.{}) else &.{};
            if (l.indexer != null and !sparse)
            {
                @memset(kept, -1);
            }
            Profile.lap(&profile.qk_rope, &pt);
            var on_gpu = false;
            if (sparse)
            {
                var done = false;
                if (gpu_qsa) |f|
                {
                    if (cache.dev_qsa != 0 and cache.format == .f32 and (cache.dev_k != 0 or cache.tier != null))
                    {
                        if (cache.tier) |tr|
                        {
                            try tr.tiles(tr.ctx, ai, pos0 + n, tiles_k, tiles_v);
                        }
                        else
                        {
                            cache.tiles(ai, pos0 + n, tiles_k, tiles_v);
                        }
                        const count = (pos0 + n + AttentionJob.tile_positions - 1) / AttentionJob.tile_positions;
                        done = f(.{ .cfg = cfg, .q = q, .iq = iq, .n = n, .pos0 = pos0, .scale = scale, .blocks_dev = cache.qsaBlocksDev(ai), .tiles_k = tiles_k[0..count], .tiles_v = tiles_v[0..count], .out = att, .kept = kept });
                        profile.qsa_gpu += @intFromBool(done);
                        // The GPU writes only the queries that choose; the others keep every block.
                        for (0..if (done and kept.len > 0) n else 0) |t|
                        {
                            if ((pos0 + t + 1) / cfg.qsa_block <= cfg.qsaTop())
                            {
                                @memset(kept[t * cfg.qsaTop() ..][0..cfg.qsaTop()], -1);
                            }
                        }
                    }
                }
                if (!done)
                {
                    if (cache.tier != null)
                    {
                        return error.OffloadNeedsGpu;
                    }
                    if (cache.format != .f32)
                    {
                        return error.Unsupported;
                    }
                    profile.qsa_cpu += 1;
                    pool.parallelFor(threads, n, qwen38.Attend{ .cfg = cfg, .q = q, .iq = iq, .keys = cache.keysF32(ai), .values = cache.valuesF32(ai), .blocks = cache.qsaBlocks(ai), .pos0 = pos0, .scale = scale, .out = att, .kept = kept }, qwen38.Attend.run);
                }
                on_gpu = true; // `att` is done, so the dense loops below are skipped

            }
            else if (gpu_attention != null and l.oss == null and (cache.dev_k != 0 or cache.tier != null))
            {
                if (cache.tier) |tr|
                {
                    try tr.tiles(tr.ctx, ai, pos0 + n, tiles_k, tiles_v);
                }
                else
                {
                    cache.tiles(ai, pos0 + n, tiles_k, tiles_v);
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
                    .format = cache.format,
                    .hd = cfg.head_dim,
                });
            }
            if (!on_gpu and cache.tier != null)
            {
                return error.OffloadNeedsGpu;
            }
            // A compact cache decodes each position's head into `kh` and `vh` as it is read.
            const plain = cache.format == .f32;
            const layer_k = if (on_gpu or !plain) &[_]f32{} else cache.keysF32(ai);
            const layer_v = if (on_gpu or !plain) &[_]f32{} else cache.valuesF32(ai);
            for (0..n) |t|
            {
                if (on_gpu)
                {
                    break;
                }
                const last = pos0 + t;
                // gpt-oss: a sliding layer sees the last `sliding_window` positions, and a sink per head joins each softmax.
                const window = if (l.oss) |os| os.window else 0;
                const from = if (window > 0 and last + 1 > window) last + 1 - window else 0;
                for (0..cfg.heads) |hd|
                {
                    const off = hd * cfg.head_dim;
                    const kv_off = hd / group * cfg.head_dim;
                    const qh = q[t * qd + off ..][0..cfg.head_dim];
                    const sink: ?f32 = if (l.oss) |os| os.sinks[hd] else null;
                    var mx: f32 = if (sink) |sv| sv else -std.math.inf(f32);
                    for (0..last + 1) |s|
                    {
                        if (s < from)
                        {
                            scores[s] = -std.math.inf(f32);
                            continue;
                        }
                        const ks = if (plain) layer_k[s * kvd + kv_off ..][0..cfg.head_dim] else blk: {
                            cache.format.part(0).decodeHead(cache.keys(ai)[s * cache.row[0] ..][0..cache.row[0]], kvd, hd / group, kh);
                            break :blk kh;
                        };
                        scores[s] = dot(qh, ks) * scale;
                        mx = @max(mx, scores[s]);
                    }
                    var sum: f32 = if (sink) |sv| @exp(sv - mx) else 0;
                    for (scores[from .. last + 1]) |*sc|
                    {
                        sc.* = @exp(sc.* - mx);
                        sum += sc.*;
                    }
                    const out = att[t * qd + off ..][0..cfg.head_dim];
                    @memset(out, 0);
                    for (from..last + 1) |s|
                    {
                        const p = scores[s] / sum;
                        const vs = if (plain) layer_v[s * kvd + kv_off ..][0..cfg.head_dim] else blk: {
                            cache.format.part(1).decodeHead(cache.values(ai)[s * cache.row[1] ..][0..cache.row[1]], kvd, hd / group, vh);
                            break :blk vh;
                        };
                        for (out, vs) |*o, vv|
                        {
                            o.* += p * vv;
                        }
                    }
                }
            }
            if (cfg.attn_gate)
            {
                pool.parallelFor(threads, n, TokenLoops.ApplyGate{ .att = att, .gate = gate, .qd = qd }, TokenLoops.ApplyGate.run);
            }
            Profile.lap(&profile.attention, &pt);
            l.o.matmul(att, n, tmp, scratch);
            if (l.oss) |os|
            {
                gptoss.addRows(tmp[0 .. n * d], os.o_b);
            }
        }
        const attn_max = if (nanTrace()) maxAbs(tmp[0 .. n * d]) else 0;
        if (l.hc_attn != null)
        {
            qwen38.inject(cfg, h, tmp, inj, n);
        }
        else
        {
            pool.parallelFor(threads, n, TokenLoops.Add{ .h = h, .y = tmp, .d = d }, TokenLoops.Add.run);
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

        if (l.hc_ffn) |hy|
        {
            qwen38.mix(cfg, hy, &hc_bufs, h, n, x, inj);
        }
        else
        {
            pool.parallelFor(threads, n, TokenLoops.Norm{ .h = h, .x = x, .w = l.ffn_norm, .eps = cfg.eps, .d = d }, TokenLoops.Norm.run);
        }
        Profile.lap(&profile.output, &pt);
        l.router.matmul(x, n, probs, scratch);
        if (l.oss) |os|
        {
            gptoss.addRows(probs[0 .. n * cfg.experts], os.router_b);
        }
        pool.parallelFor(threads, n, TokenLoops.Route{ .cfg = cfg, .probs = probs, .idx = idx, .wts = wts }, TokenLoops.Route.run);
        if (la) |p|
        {
            scoreGuesses(p, guess, idx, li, n, k, kx, horizon, cfg.layers);
        }
        if (moe.capacity) |cap|
        {
            cap.apply(idx, wts, n * k, cap_count, cap_list);
        }
        Profile.lap(&profile.router, &pt);
        @memset(tmp, 0);
        try moe.run(moe.ctx, li, x, n, idx, wts, tmp[0 .. n * d]);
        if (l.shared) |se|
        {
            try sharedExpert(&mix_bufs, se, l.shared_gate, x, n, tmp[0 .. n * d], scratch);
        }
        Profile.lap(&profile.experts, &pt);
        if (nanTrace())
        {
            std.debug.print("trace: layer {d} at {d}+{d}: input {e:.3}, attention out {e:.3}, experts out {e:.3}, residual {e:.3}\n", .{ li, pos0, n, maxAbs(x[0 .. n * d]), attn_max, maxAbs(tmp[0 .. n * d]), maxAbs(h) });
        }
        if (l.hc_ffn != null)
        {
            qwen38.inject(cfg, h, tmp, inj, n);
        }
        else
        {
            pool.parallelFor(threads, n, TokenLoops.Add{ .h = h, .y = tmp, .d = d }, TokenLoops.Add.run);
        }
        if (la) |p|
        {
            p.layer_ns += nowNs() - layer_t0;
        }

        // A non-finite value would spread to every later token and layer; stop where it first shows (TASK-31).
        if (!allFinite(h))
        {
            std.log.err("step: a non-finite value in the residual after layer {d} (positions {d} to {d}); non-finite rows: expert input {d}, expert output {d}, residual {d}, first at token {d}", .{
                li, pos0, pos0 + n - 1, badRows(x[0 .. n * d], d), badRows(tmp[0 .. n * d], d), badRows(h, R), firstBadRow(h, R),
            });
            return error.NonFinite;
        }
        if (trace) |tr|
        {
            if (tr.hidden.len > 0)
            {
                @memcpy(tr.hidden[li * n * R ..][0 .. n * R], h);
            }
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
    if (w.out_hc) |hy|
    {
        // The streams' final mix takes the place of the output norm.
        if (all_logits)
        {
            qwen38.mix(cfg, hy, &hc_bufs, h, n, x, inj);
            w.head.matmul(x, n, logits, scratch);
        }
        else
        {
            qwen38.mix(cfg, hy, &hc_bufs, h[(n - 1) * R ..][0..R], 1, x, inj);
            w.head.matmul(x, 1, logits, scratch);
        }
    }
    else if (all_logits)
    {
        pool.parallelFor(threads, n, TokenLoops.Norm{ .h = h, .x = x, .w = w.out_norm, .eps = cfg.eps, .d = d }, TokenLoops.Norm.run);
        w.head.matmul(x, n, logits, scratch);
    }
    else
    {
        rmsnorm(h[(n - 1) * d ..][0..d], w.out_norm, cfg.eps, x[0..d]);
        w.head.matmul(x, 1, logits, scratch);
    }
    if (after_step) |f|
    {
        f();
    }
    if (moe.mark) |m|
    {
        m(moe.ctx, false, n);
    }
}

/// Called at the end of every step when set (TASK-17: the memory budget's check).
pub var after_step: ?*const fn () void = null;

/// Where the KV cache and batched expert space come from; the engine points
/// it at its budgeted heap (TASK-17).
pub var big_allocator: std.mem.Allocator = std.heap.page_allocator;

/// From this many tokens on, `step` splits its loops over tokens across threads.
pub var parallel_tokens: usize = 64;

/// The per-token loops of `step`, as pieces for `pool.parallelFor`.
const TokenLoops = struct {
    const Angles = struct {
        rope: *const Rope,
        pos0: usize,
        cs: []f32,
        sn: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const half = self.rope.half;
            for (first..end) |t|
            {
                self.rope.angles(self.pos0 + t, self.cs[t * half ..][0..half], self.sn[t * half ..][0..half]);
            }
        }
    };

    const Norm = struct {
        h: []const f32,
        x: []f32,
        w: []const f32,
        eps: f32,
        d: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            for (first..end) |t|
            {
                rmsnorm(self.h[t * self.d ..][0..self.d], self.w, self.eps, self.x[t * self.d ..][0..self.d]);
            }
        }
    };

    const QkRope = struct {
        cfg: Config,
        rope: *const Rope,
        q: []f32,
        k: []f32,
        q_norm: []const f32,
        k_norm: []const f32,
        cs: []const f32,
        sn: []const f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const cfg = self.cfg;
            const qd = cfg.qDim();
            const kvd = cfg.kvDim();
            const half = self.rope.half;
            for (first..end) |t|
            {
                const qt = self.q[t * qd ..][0..qd];
                const kt = self.k[t * kvd ..][0..kvd];
                const cs = self.cs[t * half ..][0..half];
                const sn = self.sn[t * half ..][0..half];
                qkNorm(cfg, qt, self.q_norm);
                qkNorm(cfg, kt, self.k_norm);
                self.rope.applyAt(qt, cfg.heads, cfg.head_dim, cs, sn);
                self.rope.applyAt(kt, cfg.kv_heads, cfg.head_dim, cs, sn);
            }
        }
    };

    /// Splits each head's query and gate out of the gated query projection.
    const SplitGate = struct {
        qg: []const f32,
        q: []f32,
        gate: []f32,
        heads: usize,
        hd: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const qd = self.heads * self.hd;
            for (first..end) |t|
            {
                for (0..self.heads) |hh|
                {
                    const src = self.qg[t * 2 * qd + hh * 2 * self.hd ..];
                    @memcpy(self.q[t * qd + hh * self.hd ..][0..self.hd], src[0..self.hd]);
                    @memcpy(self.gate[t * qd + hh * self.hd ..][0..self.hd], src[self.hd..][0..self.hd]);
                }
            }
        }
    };

    const ApplyGate = struct {
        att: []f32,
        gate: []const f32,
        qd: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            for (self.att[first * self.qd .. end * self.qd], self.gate[first * self.qd .. end * self.qd]) |*o, g|
            {
                o.* *= sigmoid(g);
            }
        }
    };

    const Add = struct {
        h: []f32,
        y: []const f32,
        d: usize,
        fn run(self: @This(), first: usize, end: usize) void
        {
            for (self.h[first * self.d .. end * self.d], self.y[first * self.d .. end * self.d]) |*hh, o|
            {
                hh.* += o;
            }
        }
    };

    const Route = struct {
        cfg: Config,
        probs: []f32,
        idx: []u32,
        wts: []f32,
        fn run(self: @This(), first: usize, end: usize) void
        {
            const cfg = self.cfg;
            const k = cfg.top_k;
            for (first..end) |t|
            {
                const p = self.probs[t * cfg.experts ..][0..cfg.experts];
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
                topK(p, k, self.idx[t * k ..][0..k], self.wts[t * k ..][0..k]);
                if (cfg.arch != .olmoe)
                {
                    // Qwen3 renormalizes the chosen probabilities to sum to 1.
                    const wt = self.wts[t * k ..][0..k];
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
        }
    };
};

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
    /// With hyper-connections (TASK-31): the mix's buffers and stream weights, free between a block's inject and the next mix.
    hc: *const qwen38.Buffers,
    inj: []f32,

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
        if (tl.hc_ffn) |hy|
        {
            // The target layer's own mix of the streams as they are now, in place of its norm.
            qwen38.mix(cfg, hy, self.hc, h, n, x, self.inj);
        }
        else
        {
            for (0..n) |t|
            {
                rmsnorm(h[t * d ..][0..d], tl.ffn_norm, cfg.eps, x[t * d ..][0..d]);
            }
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

pub fn nowNs() u64
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
    /// Draw only from the most likely tokens that together hold this much probability.
    top_p: f32 = 1,
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
        if (self.top_p > 0 and self.top_p < 1)
        {
            sum = nucleus(p, sum * self.top_p);
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

    /// Zeroes the weights below the highest cutoff whose kept weights still
    /// reach `target`, found by bisection; returns the kept total.
    fn nucleus(p: []f32, target: f32) f32
    {
        var lo: f32 = 0;
        var hi: f32 = 1; // the most likely token's weight is exactly 1
        for (0..40) |_|
        {
            const mid = (lo + hi) / 2;
            var mass: f32 = 0;
            for (p) |pv|
            {
                if (pv >= mid)
                {
                    mass += pv;
                }
            }
            if (mass >= target)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }
        var kept: f32 = 0;
        for (p) |*pv|
        {
            if (pv.* < lo)
            {
                pv.* = 0;
            }
            kept += pv.*;
        }
        return kept;
    }
};

test "top-p keeps only the most likely tokens" {
    // Weights relative to the top token: 1, 0.5, 0.25, 0.125.
    const logits = [_]f32{ 0, -std.math.ln2, -2 * std.math.ln2, -3 * std.math.ln2 };
    var scratch: [4]f32 = undefined;
    var seen: [4]bool = @splat(false);
    var s = Sampler.seeded(1, 0, 7);
    s.top_p = 0.7; // 1 and 0.5 hold 1.5 of 1.875, which is 0.8
    for (0..200) |_|
    {
        seen[s.pick(&logits, &scratch)] = true;
    }
    try std.testing.expect(seen[0] and seen[1] and !seen[2] and !seen[3]);
}

/// Feeds `prompt` from an empty cache, then picks `out.len` tokens one at a
/// time. If set, `step_logits` gets the logits each token was picked from
/// (out.len x vocab). The cache needs room for prompt.len + out.len - 1.
pub fn generate(gpa: std.mem.Allocator, w: *const Weights, moe: Moe, cache: *KvCache, prompt: []const u32, sampler: *Sampler, out: []u32, step_logits: ?[]f32) !void
{
    const vocab = w.cfg.vocab;
    const row = try gpa.alloc(f32, 2 * vocab);
    defer gpa.free(row);
    cache.reset();
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
    /// Expert calls run so far, for the live diagram's check (TASK-16b).
    calls: u64 = 0,

    const max_group = 1024;

    pub fn init(a: std.mem.Allocator, cfg: Config, m: loader.Model, owned: []const bool, plan: loader.Plan, region: []const u8) !LocalExperts {
        swiglu_limit = cfg.swiglu_limit;
        swiglu_oai = cfg.arch == .gptoss;
        const ex = try a.alloc(?Expert, cfg.layers * cfg.experts);
        for (ex, 0..) |*e, i| {
            const layer: u32 = @intCast(i / cfg.experts);
            const id: u32 = @intCast(i % cfg.experts);
            if (!owned[id] or layer < cfg.dense_layers) {
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
                .gate_b = loader.biasOf(m.weight(layer, .gate, id), plan.partsOf(region, m.weight(layer, .gate, id))),
                .up_b = loader.biasOf(m.weight(layer, .up, id), plan.partsOf(region, m.weight(layer, .up, id))),
                .down_b = loader.biasOf(m.weight(layer, .down, id), plan.partsOf(region, m.weight(layer, .down, id))),
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
            big_allocator.free(self.batch);
        }
        self.batch = &.{};
    }

    fn runAll(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, w: []const f32, out: []f32) anyerror!void {
        const self: *LocalExperts = @ptrCast(@alignCast(ctx));
        const d = self.cfg.d_model;
        const k = self.cfg.top_k;
        if (n >= self.batch_min or gpu_experts != null)
        {
            return self.runBatched(layer, x, n, idx, w, out);
        }
        for (0..n) |t| {
            for (0..k) |j| {
                if (idx[t * k + j] == Capacity.skip)
                {
                    continue;
                }
                const e = self.get(layer, idx[t * k + j]) orelse return error.ExpertNotLocal;
                e.forward(x[t * d ..][0..d], self.out, self.scratch);
                self.calls += 1;
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
        const ne = self.cfg.experts;
        const calls = n * k;
        if (ne > max_group)
        {
            return error.TooManyExperts;
        }
        // Call order, batch row of each call, then the CPU path's inputs, gate, up, and outputs.
        const small = 2 * calls + n * d;
        const need = small + if (gpu_experts != null) 0 else 3 * calls * d + 2 * calls * f;
        if (self.batch.len < need)
        {
            if (self.batch.len > 0)
            {
                big_allocator.free(self.batch);
            }
            self.batch = &.{};
            self.batch = try big_allocator.alloc(f32, need);
        }
        const order: []u32 = @ptrCast(self.batch[0..calls]);
        const inv: []u32 = @ptrCast(self.batch[calls..][0..calls]);
        const sum = self.batch[2 * calls ..][0 .. n * d];
        // Calls grouped by expert, each group in call order; dropped calls get no row.
        var count: [max_group]u32 = @splat(0);
        var rows: usize = 0;
        for (idx[0..calls]) |e|
        {
            if (e == Capacity.skip)
            {
                continue;
            }
            if (e >= ne)
            {
                return error.BadExpert;
            }
            count[e] += 1;
            rows += 1;
        }
        // Experts with few calls come first, as the GPU's kernels expect (gpu_small_rows).
        var start: [max_group]u32 = undefined;
        var at: u32 = 0;
        for ([_]bool{ true, false }) |few|
        {
            for (0..ne) |e|
            {
                if (count[e] > 0 and (count[e] <= gpu_small_rows) == few)
                {
                    start[e] = at;
                    at += count[e];
                }
            }
        }
        var fill: [max_group]u32 = undefined;
        @memcpy(fill[0..ne], start[0..ne]);
        for (idx[0..calls], 0..) |e, c|
        {
            if (e == Capacity.skip)
            {
                // Weight 0 on row 0 adds nothing on the GPU path.
                inv[c] = 0;
                continue;
            }
            order[fill[e]] = @intCast(c);
            inv[c] = fill[e];
            fill[e] += 1;
        }
        const kept = order[0..rows];
        var used: [max_group]Expert = undefined;
        var starts: [max_group + 1]u32 = undefined;
        var groups: usize = 0;
        for ([_]bool{ true, false }) |few|
        {
            for (0..ne) |e|
            {
                if (count[e] == 0 or (count[e] <= gpu_small_rows) != few)
                {
                    continue;
                }
                used[groups] = self.get(layer, e) orelse return error.ExpertNotLocal;
                starts[groups] = start[e];
                groups += 1;
            }
        }
        starts[groups] = @intCast(rows);
        if (rows == 0)
        {
            return;
        }
        for (0..groups) |g|
        {
            self.calls += starts[g + 1] - starts[g];
        }
        if (gpu_experts) |run|
        {
            // Batch row i is token order[i] / k: reuse `order` once it holds tokens.
            for (kept) |*c|
            {
                c.* /= @intCast(k);
            }
            if (run(.{ .experts = used[0..groups], .first = starts[0 .. groups + 1], .src = kept, .inv = inv, .w = w[0..calls], .x = x[0 .. n * d], .n = n, .k = k, .y = sum }))
            {
                for (out[0 .. n * d], sum) |*o, v|
                {
                    o.* += v;
                }
                return;
            }
            for (inv, 0..) |r, c|
            {
                if (idx[c] != Capacity.skip)
                {
                    order[r] = @intCast(c);
                }
            }
        }
        // Without a GPU the batch has room for the CPU path; after a GPU failure it is borrowed.
        const cpu_need = 3 * calls * d + 2 * calls * f;
        const borrowed = if (self.batch.len < small + cpu_need) try big_allocator.alloc(f32, cpu_need) else @as([]f32, &.{});
        defer
        {
            if (borrowed.len > 0)
            {
                big_allocator.free(borrowed);
            }
        }
        const rest = if (borrowed.len > 0) borrowed else self.batch[small..];
        const xb = rest[0 .. calls * d];
        const gb = rest[calls * d ..][0 .. calls * f];
        const ub = rest[calls * d + calls * f ..][0 .. calls * f];
        const yb = rest[calls * d + 2 * calls * f ..][0 .. calls * d];
        const contrib = rest[2 * calls * d + 2 * calls * f ..][0 .. calls * d];
        for (kept, 0..) |c, i|
        {
            @memcpy(xb[i * d ..][0..d], x[(c / k) * d ..][0..d]);
        }
        for (used[0..groups], 0..) |ex, gi|
        {
            const first = starts[gi];
            const cnt = starts[gi + 1] - first;
            ex.gate.matmul(xb[first * d ..], cnt, gb, self.scratch);
            ex.up.matmul(xb[first * d ..], cnt, ub, self.scratch);
            ex.activate(gb, ub, cnt);
            ex.down.matmul(gb, cnt, yb[first * d ..], self.scratch);
            ex.finish(yb[first * d ..], cnt);
        }
        for (kept, 0..) |c, i|
        {
            @memcpy(contrib[c * d ..][0..d], yb[i * d ..][0..d]);
        }
        for (0..n) |t|
        {
            for (0..k) |j|
            {
                const c = t * k + j;
                if (idx[c] == Capacity.skip)
                {
                    continue;
                }
                for (out[t * d ..][0..d], contrib[c * d ..][0..d]) |*o, y|
                {
                    o.* += w[c] * y;
                }
            }
        }
    }
};

test {
    _ = pool;
    _ = glm53;
}

test "capacity keeps each expert's highest-weight calls and drops the rest" {
    // Two choices per token over 4 experts; expert 1 is chosen by all five tokens.
    var idx = [_]u32{ 1, 0, 1, 2, 1, 3, 1, 0, 1, 2 };
    var wts = [_]f32{ 0.5, 0.1, 0.9, 0.1, 0.2, 0.1, 0.9, 0.1, 0.7, 0.3 };
    var count: [4]u32 = undefined;
    var list: [10]u32 = undefined;
    var cap: Capacity = .{ .expert_limit = 2 };
    cap.apply(&idx, &wts, idx.len, &count, &list);
    // Kept: 0.9 at call 2 and the tied 0.9 at call 6 (the earlier token first).
    const want = [_]u32{ Capacity.skip, 0, 1, 2, Capacity.skip, 3, 1, 0, Capacity.skip, 2 };
    try std.testing.expectEqualSlices(u32, &want, &idx);
    try std.testing.expectEqual(@as(f32, 0), wts[0]);
    try std.testing.expectEqual(@as(u64, 3), cap.dropped);
    try std.testing.expectEqual(@as(u64, 10), cap.routed);
    try std.testing.expectEqual(@as(u64, 1), cap.layers_over);
    // Applying again changes nothing: the kept calls already fit.
    cap.apply(&idx, &wts, idx.len, &count, &list);
    try std.testing.expectEqual(@as(u64, 3), cap.dropped);
    try std.testing.expectEqualSlices(u32, &want, &idx);
    const cfg: Config = .{ .arch = .olmoe, .layers = 1, .d_model = 1, .d_ff = 1, .heads = 1, .kv_heads = 1, .head_dim = 1, .experts = 64, .top_k = 8, .vocab = 1, .eps = 0, .rope_theta = 0 };
    try std.testing.expectEqual(@as(u32, 10), Capacity.limitFor(1.25, 64, cfg, 1));
    try std.testing.expectEqual(@as(u32, 1), Capacity.limitFor(1.25, 1, cfg, 1));
    // Half the experts each: 1.25 x 64 tokens x 8 choices / 2.
    var owner: [64]u8 = undefined;
    for (&owner, 0..) |*o, e|
    {
        o.* = @intFromBool(e >= 32);
    }
    const split = Capacity.init(cfg, 64, 1.25, 0, &owner);
    try std.testing.expectEqual(@as(u32, 320), split.rank_limit[0]);
    try std.testing.expectEqual(@as(u32, 320), split.rank_limit[1]);
    try std.testing.expectEqual(Capacity.none, split.expert_limit);
    try std.testing.expectEqual(@as(u32, 640), Capacity.init(cfg, 64, 1.25, 0, null).rank_limit[0]);
}

test "capacity per rank keeps each rank's highest-weight calls" {
    // Four experts, 0 and 1 on rank 0, 2 and 3 on rank 1; two choices per token.
    const owner = [_]u8{ 0, 0, 1, 1 };
    var idx = [_]u32{ 2, 0, 3, 1, 2, 3, 2, 0 };
    var wts = [_]f32{ 0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.9, 0.1 };
    var count: [4]u32 = undefined;
    var list: [8]u32 = undefined;
    var limits: [Capacity.max_ranks]u32 = @splat(Capacity.none);
    limits[1] = 3;
    var cap: Capacity = .{ .owner = &owner, .rank_limit = limits };
    cap.apply(&idx, &wts, idx.len, &count, &list);
    // Rank 1 had 5 calls (weights 0.6, 0.7, 0.5, 0.5, 0.9): it keeps 0.9, 0.7, 0.6.
    const want = [_]u32{ 2, 0, 3, 1, Capacity.skip, Capacity.skip, 2, 0 };
    try std.testing.expectEqualSlices(u32, &want, &idx);
    try std.testing.expectEqual(@as(u64, 2), cap.dropped);
    // An expert limit first, then the rank's: expert 2 keeps one call, so rank 1 fits.
    var idx2 = [_]u32{ 2, 0, 3, 1, 2, 3, 2, 0 };
    var wts2 = [_]f32{ 0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.9, 0.1 };
    var both: Capacity = .{ .owner = &owner, .rank_limit = limits, .expert_limit = 1 };
    both.apply(&idx2, &wts2, idx2.len, &count, &list);
    const want2 = [_]u32{ Capacity.skip, 0, 3, 1, Capacity.skip, Capacity.skip, 2, Capacity.skip };
    try std.testing.expectEqualSlices(u32, &want2, &idx2);
    try std.testing.expectEqual(@as(u64, 4), both.dropped);
}

test "capacity: unchanged output when nothing is dropped, finite output when calls are" {
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
    // 6 tokens run calls one at a time; 24 run them batched.
    for ([_]usize{ 6, 24 }) |n|
    {
        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        for (tokens, 0..) |*t, i|
        {
            t.* = @intCast((i * 37 + 5) % cfg.vocab);
        }
        const plain = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(plain);
        const roomy = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(roomy);
        const tight = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(tight);
        try forward(gpa, &w, local.moe(), tokens, plain, null);
        var big: Capacity = .{ .expert_limit = @intCast(n) };
        var with = local.moe();
        with.capacity = &big;
        try forward(gpa, &w, with, tokens, roomy, null);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(roomy));
        try std.testing.expectEqual(@as(u64, 0), big.dropped);
        const calls_before = local.calls;
        var small: Capacity = .{ .expert_limit = 1 };
        with.capacity = &small;
        try forward(gpa, &w, with, tokens, tight, null);
        try std.testing.expect(small.dropped > 0);
        try std.testing.expectEqual(small.routed - small.dropped, local.calls - calls_before);
        for (tight) |v|
        {
            try std.testing.expect(std.math.isFinite(v));
        }
        try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(tight)));
    }
}

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

test "compact KV caches stay close to f32, and cached steps match a full pass in the same format" {
    const gpa = std.testing.allocator;
    const Case = struct { file: []const u8, yarn: bool };
    const cases = [_]Case{ .{ .file = @embedFile("olmoe_tiny.gguf"), .yarn = false }, .{ .file = @embedFile("qwen3_tiny.gguf"), .yarn = true } };
    for (cases) |cs|
    {
        var h = try gguf.parse(gpa, cs.file, cs.file.len);
        defer h.deinit(gpa);
        const base = try Config.fromGguf(h);
        const cfg = if (cs.yarn) base.withYarn(4, 16) else base;
        var m = try loader.Model.fromGguf(gpa, h);
        defer m.deinit(gpa);
        var whole = [_]loader.Range{.{ .offset = 0, .len = cs.file.len }};
        var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, cs.file.len);
        defer plan.deinit(gpa);
        var w = try Weights.init(gpa, h, cfg, plan, cs.file);
        defer w.deinit();
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var owned: [8]bool = @splat(true);
        var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, cs.file);
        defer local.deinit();

        const n = 40;
        var tokens: [n]u32 = undefined;
        for (&tokens, 0..) |*t, i|
        {
            t.* = @intCast((i * 7919 + 13) % cfg.vocab);
        }
        const ref = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(ref);
        try forward(gpa, &w, local.moe(), &tokens, ref, null);
        var peak: f32 = 0;
        for (ref) |x|
        {
            peak = @max(peak, @abs(x));
        }
        for ([_]KvFormat{ .bf16, .fp8, .bf16_fp8 }) |fmt|
        {
            const got = try gpa.alloc(f32, n * cfg.vocab);
            defer gpa.free(got);
            try forwardAs(gpa, &w, local.moe(), &tokens, got, null, fmt);
            var worst: f32 = 0;
            for (got, ref) |a, b|
            {
                worst = @max(worst, @abs(a - b));
            }
            // Close to f32, and not equal to it: the cache really holds the compact values.
            const bound: f32 = if (fmt == .bf16) 0.02 else 0.1;
            try std.testing.expect(worst > 0 and worst <= bound * peak);

            // One token at a time through a cache of the same format gives the full pass's logits.
            var cache = try KvCache.initAs(gpa, cfg, n, fmt);
            defer cache.deinit(gpa);
            const one = try gpa.alloc(f32, cfg.vocab);
            defer gpa.free(one);
            for (0..n) |t|
            {
                try step(gpa, &w, local.moe(), &cache, tokens[t .. t + 1], one, null);
                for (one, got[t * cfg.vocab ..][0..cfg.vocab]) |a, b|
                {
                    try std.testing.expectApproxEqAbs(b, a, 1e-4);
                }
            }
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

test "tiny Qwen3.5-MoE (Gated DeltaNet, gated attention, shared expert) matches transformers (tools/ref/ref_qwen35.py --tiny)" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("qwen35_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    try std.testing.expectEqual(Arch.qwen35moe, cfg.arch);
    try std.testing.expect(cfg.recurrent(0) and cfg.recurrent(2) and !cfg.recurrent(3));
    try std.testing.expectEqual(@as(usize, 1), cfg.attnLayers());
    try std.testing.expectEqual(@as(usize, 8), cfg.rotaryDim());
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

    const golden = try tfref.File.parse(@embedFile("qwen35_tiny.tfref"));
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
        // Each layer's output (all but the last, which transformers reports after the final norm).
        for (0..cfg.layers - 1) |l|
        {
            const hw = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))).f32s();
            for (trace.hidden[l * ids.len * cfg.d_model ..][0 .. ids.len * cfg.d_model], hw, 0..) |a, b, i|
            {
                if (@abs(a - b) > 1e-4)
                {
                    std.debug.print("layer {d}, token {d}, value {d}: {d} against {d}\n", .{ l, i / cfg.d_model, i % cfg.d_model, a, b });
                    return error.TestHiddenMismatch;
                }
            }
        }
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

        // Generation steps one token at a time, so the Gated DeltaNet state carries across steps.
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

        // A hybrid cache cannot step back: truncating empties it, and a rerun gives the same logits.
        try std.testing.expectEqual(@as(usize, 0), cache.truncate(ids.len));
        const again = try gpa.alloc(f32, cfg.vocab);
        defer gpa.free(again);
        try step(gpa, &w, local.moe(), &cache, tokens, again, null);
        for (again, logits[(ids.len - 1) * cfg.vocab ..][0..cfg.vocab]) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-5);
        }
    }
}

test "tiny Qwen3.8-Flash-Next (hyper-connections, QSA, n-gram embedding) matches transformers (tools/ref/ref_qwen38.py --tiny)" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("qwen38_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    try std.testing.expectEqual(Arch.qwen4exp, cfg.arch);
    try std.testing.expectEqual(@as(usize, 4), cfg.streams);
    try std.testing.expectEqual(@as(usize, 2), cfg.qsaTop());
    try std.testing.expectEqual(@as(?usize, 1), cfg.ple_layer);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [16]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);
    defer local.deinit();

    const golden = try tfref.File.parse(@embedFile("qwen38_tiny.tfref"));
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    const R = cfg.residual();
    const top = cfg.qsaTop();
    var checked_blocks: usize = 0;
    for (0..prompts) |p|
    {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const n = ids.len;
        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        for (tokens, ids) |*t, id|
        {
            t.* = @intCast(id);
        }
        const logits = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(logits);
        const trace: Trace = .{
            .hidden = try gpa.alloc(f32, cfg.layers * n * R),
            .topk = try gpa.alloc(u32, cfg.layers * n * cfg.top_k),
            .qsa = try gpa.alloc(i32, cfg.attnLayers() * n * top),
        };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        defer gpa.free(trace.qsa);
        try forward(gpa, &w, local.moe(), tokens, logits, trace);
        // Each layer's output, every stream (all but the last, which transformers reports after the final mix).
        for (0..cfg.layers - 1) |l|
        {
            const hw = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))).f32s();
            for (trace.hidden[l * n * R ..][0 .. n * R], hw, 0..) |a, b, i|
            {
                if (@abs(a - b) > 1e-4)
                {
                    std.debug.print("layer {d}, token {d}, value {d}: {d} against {d}\n", .{ l, i / R, i % R, a, b });
                    return error.TestHiddenMismatch;
                }
            }
        }
        for (0..cfg.layers) |l|
        {
            const gi = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l }))).i32s();
            for (trace.topk[l * n * cfg.top_k ..][0 .. n * cfg.top_k], gi) |got, exp|
            {
                try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
            }
            if (cfg.recurrent(l))
            {
                continue;
            }
            // Every block QSA kept, for each query with more complete blocks than its budget.
            const kept = trace.qsa[cfg.attnIndex(l) * n * top ..][0 .. n * top];
            const from_t = golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.qsa_from", .{ p, l })) catch
            {
                for (kept) |b|
                {
                    try std.testing.expectEqual(@as(i32, -1), b);
                }
                continue;
            };
            const from: usize = @intCast(from_t.i32s()[0]);
            const gb = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.qsa_blocks", .{ p, l }))).i32s();
            for (kept[0 .. from * top]) |b|
            {
                try std.testing.expectEqual(@as(i32, -1), b);
            }
            for (kept[from * top ..], gb, 0..) |got, exp, i|
            {
                if (got != exp)
                {
                    std.debug.print("layer {d}, token {d}: kept block {d}, transformers {d}\n", .{ l, from + i / top, got, exp });
                    return error.TestQsaMismatch;
                }
            }
            checked_blocks += gb.len;
        }
        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        for (logits, want) |a, b|
        {
            try std.testing.expectApproxEqAbs(b, a, 1e-4);
        }

        // Generation steps one token at a time, so the recurrent, n-gram and QSA state carries across steps.
        const want_tokens = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p}))).i32s();
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        const g = want_tokens.len;
        const all = try gpa.alloc(u32, n + g);
        defer gpa.free(all);
        @memcpy(all[0..n], tokens);
        var cache = try KvCache.init(gpa, cfg, n + g);
        defer cache.deinit(gpa);
        var sampler: Sampler = .{};
        const step_logits = try gpa.alloc(f32, g * cfg.vocab);
        defer gpa.free(step_logits);
        try generate(gpa, &w, local.moe(), &cache, all[0..n], &sampler, all[n..], step_logits);
        for (all[n..], want_tokens) |got, exp|
        {
            try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
        for (step_logits, want_logits) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-4);
        }

        // A prompt fed in pieces gives the same last logits as in one pass.
        cache.reset();
        const half = n / 2;
        const piece = try gpa.alloc(f32, cfg.vocab);
        defer gpa.free(piece);
        try step(gpa, &w, local.moe(), &cache, tokens[0..half], piece, null);
        try step(gpa, &w, local.moe(), &cache, tokens[half..], piece, null);
        for (piece, logits[(n - 1) * cfg.vocab ..][0..cfg.vocab]) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-5);
        }
    }
    try std.testing.expect(checked_blocks > 0);
}

test "tiny GLM-5.3-Flash (mHC, KDA, MLA with DSA, dense layers) matches transformers (tools/ref/ref_glm53.py --tiny)" {
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const file = @embedFile("glm53_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    try std.testing.expectEqual(Arch.glm5next, cfg.arch);
    try std.testing.expectEqual(@as(usize, 4), cfg.streams);
    try std.testing.expectEqual(@as(usize, 1), cfg.dense_layers);
    try std.testing.expectEqual(@as(usize, 2), cfg.dsaTop());
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
    const golden = try tfref.File.parse(@embedFile("glm53_tiny.tfref"));

    // Q4_K, which the tiny experts do not use, against gguf-py on a separate tensor.
    const qt = h.tensor("test.q4_k").?;
    const want_q = (try golden.get("test.q4_k")).f32s();
    const got_q = try gpa.alloc(f32, want_q.len);
    defer gpa.free(got_q);
    try quant.dequantize(qt.format, file[qt.offset..][0..qt.bytes], got_q);
    for (got_q, want_q) |got, exp|
    {
        try std.testing.expectApproxEqAbs(exp, got, 1e-6);
    }

    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    const R = cfg.residual();
    const k = cfg.top_k;
    const top = cfg.dsaTop();
    var checked_pools: usize = 0;
    for (0..prompts) |p|
    {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const n = ids.len;
        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        for (tokens, ids) |*t, id|
        {
            t.* = @intCast(id);
        }
        const logits = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(logits);
        const trace: Trace = .{
            .hidden = try gpa.alloc(f32, cfg.layers * n * R),
            .topk = try gpa.alloc(u32, cfg.layers * n * k),
            .qsa = try gpa.alloc(i32, cfg.attnLayers() * n * top),
        };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        defer gpa.free(trace.qsa);
        try forward(gpa, &w, local.moe(), tokens, logits, trace);
        for (0..cfg.layers) |l|
        {
            const hw = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))).f32s();
            for (trace.hidden[l * n * R ..][0 .. n * R], hw, 0..) |a, b, i|
            {
                if (@abs(a - b) > 1e-4)
                {
                    std.debug.print("layer {d}, token {d}, value {d}: {d} against {d}\n", .{ l, i / R, i % R, a, b });
                    return error.TestHiddenMismatch;
                }
            }
            if (l >= cfg.dense_layers)
            {
                // transformers' choice order is unspecified, so the sets are compared, ascending.
                const gi = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l }))).i32s();
                for (0..n) |t|
                {
                    const row = trace.topk[(l * n + t) * k ..][0..k];
                    std.mem.sort(u32, row, {}, std.sort.asc(u32));
                    for (row, gi[t * k ..][0..k]) |got, exp|
                    {
                        try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
                    }
                }
            }
            if (cfg.recurrent(l))
            {
                continue;
            }
            const kept = trace.qsa[cfg.attnIndex(l) * n * top ..][0 .. n * top];
            const from_t = golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.dsa_from", .{ p, l })) catch
            {
                for (kept) |b|
                {
                    try std.testing.expectEqual(@as(i32, -1), b);
                }
                continue;
            };
            const from: usize = @intCast(from_t.i32s()[0]);
            const gp = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.dsa_pools", .{ p, l }))).i32s();
            for (kept[0 .. from * top]) |b|
            {
                try std.testing.expectEqual(@as(i32, -1), b);
            }
            for (kept[from * top ..], gp, 0..) |got, exp, i|
            {
                if (got != exp)
                {
                    std.debug.print("layer {d}, token {d}: kept pool {d}, transformers {d}\n", .{ l, from + i / top, got, exp });
                    return error.TestDsaMismatch;
                }
            }
            checked_pools += gp.len;
        }
        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        for (logits, want) |a, b|
        {
            try std.testing.expectApproxEqAbs(b, a, 1e-4);
        }

        // Generation steps one token at a time, so the KDA, MLA and DSA state carries across steps.
        const want_tokens = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p}))).i32s();
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        const g = want_tokens.len;
        const all = try gpa.alloc(u32, n + g);
        defer gpa.free(all);
        @memcpy(all[0..n], tokens);
        var cache = try KvCache.init(gpa, cfg, n + g);
        defer cache.deinit(gpa);
        var sampler: Sampler = .{};
        const step_logits = try gpa.alloc(f32, g * cfg.vocab);
        defer gpa.free(step_logits);
        try generate(gpa, &w, local.moe(), &cache, all[0..n], &sampler, all[n..], step_logits);
        for (all[n..], want_tokens) |got, exp|
        {
            try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
        for (step_logits, want_logits) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-4);
        }

        // A prompt fed in pieces gives the same last logits as in one pass.
        cache.reset();
        const half = n / 2;
        const piece = try gpa.alloc(f32, cfg.vocab);
        defer gpa.free(piece);
        try step(gpa, &w, local.moe(), &cache, tokens[0..half], piece, null);
        try step(gpa, &w, local.moe(), &cache, tokens[half..], piece, null);
        for (piece, logits[(n - 1) * cfg.vocab ..][0..cfg.vocab]) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-5);
        }
    }
    try std.testing.expect(checked_pools > 0);
}

test "lookahead on hyper-connections leaves every logit bit-identical" {
    const gpa = std.testing.allocator;
    const file = @embedFile("qwen38_tiny.gguf");
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
    var owned: [16]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);
    const Log = struct {
        calls: usize = 0,
        fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
        {
            _ = layer;
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += @intFromBool(experts.len > 0);
        }
    };
    var log: Log = .{};
    var la: Lookahead = .{ .horizon = 2, .extra = 1 };
    // Experts run through `local`; prefetch hints go to the log.
    const Moe2 = struct {
        local: *LocalExperts,
        log: *Log,
        fn run(ctx: *anyopaque, layer: usize, x: []const f32, n: usize, idx: []const u32, wt: []const f32, out: []f32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return LocalExperts.runAll(self.local, layer, x, n, idx, wt, out);
        }
        fn prefetch(ctx: *anyopaque, layer: usize, experts: []const u32) anyerror!void
        {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return Log.prefetch(self.log, layer, experts);
        }
    };
    var both: Moe2 = .{ .local = &local, .log = &log };
    const with: Moe = .{ .ctx = &both, .run = Moe2.run, .ahead = &la, .prefetch = Moe2.prefetch };

    const tokens = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 2, 19, 20, 21, 22, 23, 24, 25 };
    const plain = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(plain);
    const guessed = try gpa.alloc(f32, tokens.len * cfg.vocab);
    defer gpa.free(guessed);
    try forward(gpa, &w, local.moe(), &tokens, plain, null);
    try forward(gpa, &w, with, &tokens, guessed, null);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(guessed));
    try std.testing.expectEqual(@as(usize, cfg.layers), log.calls);
    try std.testing.expect(la.recall(1) > 0 and la.recall(1) <= 1);
}

test "the n-gram hash starts over after an end-of-text token" {
    var cfg: Config = .{ .layers = 1, .d_model = 1, .heads = 1, .kv_heads = 1, .head_dim = 1, .d_ff = 1, .experts = 1, .top_k = 1, .vocab = 16, .eps = 1e-6, .rope_theta = 1 };
    cfg.ple_ngram = 3;
    cfg.ple_heads_per = 1;
    cfg.ple_eos = 2;
    var p: Ple = .{ .table = Mat.empty, .mult = .{ 3, 5, 7, 0 }, .offsets = @splat(0), .sizes = @splat(1 << 40) };
    p.offsets[1] = 1000;
    var out: [8]u64 = undefined;
    // Tokens 9, 2 (end of text), 4: the bigram and trigram at 4 see end-of-text before it,
    // and the trigram at 9 reads the two held end-of-text tokens.
    qwen38.ngramRows(cfg, p, &.{ 2, 2 }, &.{ 9, 2, 4, 6 }, &out);
    try std.testing.expectEqual(@as(u64, (9 * 3) ^ (2 * 5)), out[0]);
    try std.testing.expectEqual(@as(u64, ((9 * 3) ^ (2 * 5) ^ (2 * 7)) + 1000), out[1]);
    try std.testing.expectEqual(@as(u64, (4 * 3) ^ (2 * 5)), out[4]);
    try std.testing.expectEqual(@as(u64, ((4 * 3) ^ (2 * 5) ^ (2 * 7)) + 1000), out[5]);
    try std.testing.expectEqual(@as(u64, ((6 * 3) ^ (4 * 5) ^ (2 * 7)) + 1000), out[7]);
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

/// Runs a TASK-39 tiny model against transformers' outputs from its tools/ref script: logits,
/// each recorded hidden state, each MoE layer's chosen experts as a set, and greedy generation.
fn checkTiny(comptime stem: []const u8, arch: Arch) !void
{
    const tfref = @import("tfref.zig");
    const gpa = std.testing.allocator;
    const embedded = @embedFile(stem ++ ".gguf");
    // An embedded file is only byte aligned, and expert biases are read in place as floats.
    const file = try gpa.alignedAlloc(u8, .fromByteUnits(64), embedded.len);
    defer gpa.free(file);
    @memcpy(file, embedded);
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    const cfg = try Config.fromGguf(h);
    try std.testing.expectEqual(arch, cfg.arch);
    var m = try loader.Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var whole = [_]loader.Range{.{ .offset = 0, .len = file.len }};
    var plan = try loader.Plan.fromRanges(gpa, &whole, 1, 0, file.len);
    defer plan.deinit(gpa);
    var w = try Weights.init(gpa, h, cfg, plan, file);
    defer w.deinit();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var owned: [64]bool = @splat(true);
    var local = try LocalExperts.init(arena.allocator(), cfg, m, owned[0..cfg.experts], plan, file);
    defer local.deinit();
    const golden = try tfref.File.parse(@embedFile(stem ++ ".tfref"));
    const prompts: usize = @intCast((try golden.get("config")).i32s()[0]);
    const d = cfg.d_model;
    const k = cfg.top_k;
    for (0..prompts) |p|
    {
        var buf: [64]u8 = undefined;
        const ids = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.tokens", .{p}))).i32s();
        const n = ids.len;
        const tokens = try gpa.alloc(u32, n);
        defer gpa.free(tokens);
        for (tokens, ids) |*t, id|
        {
            t.* = @intCast(id);
        }
        const logits = try gpa.alloc(f32, n * cfg.vocab);
        defer gpa.free(logits);
        const trace: Trace = .{ .hidden = try gpa.alloc(f32, cfg.layers * n * d), .topk = try gpa.alloc(u32, cfg.layers * n * k) };
        defer gpa.free(trace.hidden);
        defer gpa.free(trace.topk);
        try forward(gpa, &w, local.moe(), tokens, logits, trace);
        for (0..cfg.layers) |l|
        {
            if (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.hidden", .{ p, l }))) |hw|
            {
                for (trace.hidden[l * n * d ..][0 .. n * d], hw.f32s(), 0..) |a, b, i|
                {
                    if (@abs(a - b) > 1e-4)
                    {
                        std.debug.print("{s}: prompt {d}, layer {d}, token {d}, value {d}: {d} against {d}\n", .{ stem, p, l, i / d, i % d, a, b });
                        return error.TestHiddenMismatch;
                    }
                }
            }
            else |_|
            {}
            const gi = (golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.layer{d}.topk_idx", .{ p, l })) catch continue).i32s();
            var want: [64]u32 = undefined;
            for (0..n) |t|
            {
                const row = trace.topk[(l * n + t) * k ..][0..k];
                for (want[0..k], gi[t * k ..][0..k]) |*o, v|
                {
                    o.* = @intCast(v);
                }
                std.mem.sort(u32, row, {}, std.sort.asc(u32));
                std.mem.sort(u32, want[0..k], {}, std.sort.asc(u32));
                try std.testing.expectEqualSlices(u32, want[0..k], row);
            }
        }
        const want = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.logits", .{p}))).f32s();
        for (logits, want) |a, b|
        {
            try std.testing.expectApproxEqAbs(b, a, 1e-4);
        }

        const want_tokens = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_tokens", .{p}))).i32s();
        const want_logits = (try golden.get(try std.fmt.bufPrint(&buf, "prompt{d}.gen_logits", .{p}))).f32s();
        const g = want_tokens.len;
        const all = try gpa.alloc(u32, n + g);
        defer gpa.free(all);
        @memcpy(all[0..n], tokens);
        var cache = try KvCache.init(gpa, cfg, n + g);
        defer cache.deinit(gpa);
        var sampler: Sampler = .{};
        const step_logits = try gpa.alloc(f32, g * cfg.vocab);
        defer gpa.free(step_logits);
        try generate(gpa, &w, local.moe(), &cache, all[0..n], &sampler, all[n..], step_logits);
        for (all[n..], want_tokens) |got, exp|
        {
            try std.testing.expectEqual(@as(u32, @intCast(exp)), got);
        }
        for (step_logits, want_logits) |got, exp|
        {
            try std.testing.expectApproxEqAbs(exp, got, 1e-4);
        }
    }
}

test "tiny Mixtral (GGUF llama with experts, rope on adjacent pairs) matches transformers (tools/ref/ref_mixtral.py --tiny)"
{
    try checkTiny("mixtral_tiny", .mixtral);
}

test "tiny DeepSeek V3 (MLA with decoupled rope, grouped routing, YaRN) matches transformers (tools/ref/ref_deepseek.py --tiny)"
{
    try checkTiny("deepseek_tiny", .deepseek2);
}

test "tiny gpt-oss (sinks, sliding window, biases, MXFP4 experts) matches transformers (tools/ref/ref_gptoss.py --tiny)"
{
    try checkTiny("gptoss_tiny", .gptoss);
}
