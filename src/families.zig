//! The model families (TASK-42): one entry per GGUF architecture, the only place an
//! architecture name is checked. Each entry reads its GGUF keys into a `Config` that
//! describes the model by its parts (attention, rope, norms, router, experts), fills in
//! what the file does not record, and names the code that loads its non-expert weights.
const std = @import("std");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const olmoe = @import("olmoe.zig");
const glm53 = @import("glm53.zig");
const deepseek = @import("deepseek.zig");
const gptoss = @import("gptoss.zig");

const Config = olmoe.Config;

pub const Family = struct {
    /// The GGUF's `general.architecture`.
    gguf: []const u8,
    config: *const fn (h: gguf.Header) anyerror!Config,
    weights: *const fn (gpa: std.mem.Allocator, h: gguf.Header, cfg: Config, plan: loader.Plan, region: []const u8) anyerror!olmoe.Weights = olmoe.Weights.standard,
};

pub const all = [_]Family{
    .{ .gguf = "olmoe", .config = olmoeConfig },
    .{ .gguf = "qwen3moe", .config = qwen3Config },
    .{ .gguf = "qwen35moe", .config = qwen35Config },
    .{ .gguf = "qwen4exp", .config = qwen38Config },
    .{ .gguf = "glm5-next", .config = glm53Config, .weights = glm53.weights },
    .{ .gguf = "llama", .config = mixtralConfig },
    .{ .gguf = "deepseek2", .config = deepseekConfig, .weights = deepseek.weights },
    .{ .gguf = "gpt-oss", .config = gptossConfig },
    .{ .gguf = "qwen2moe", .config = qwen2Config },
};

/// The entry for the GGUF's architecture.
pub fn find(h: gguf.Header) !*const Family
{
    const name = h.getString("general.architecture") orelse return error.NotOlmoe;
    for (&all) |*f|
    {
        if (std.mem.eql(u8, f.gguf, name))
        {
            return f;
        }
    }
    return error.NotOlmoe;
}

/// Reads `ARCH.KEY` values, with ARCH the GGUF's architecture.
const Keys = struct {
    hh: gguf.Header,
    prefix: []const u8,
    buf: [96]u8 = undefined,

    fn init(h: gguf.Header) Keys
    {
        return .{ .hh = h, .prefix = h.getString("general.architecture").? };
    }

    fn key(s: *Keys, k: []const u8) []const u8
    {
        return std.fmt.bufPrint(&s.buf, "{s}.{s}", .{ s.prefix, k }) catch unreachable;
    }

    fn int(s: *Keys, k: []const u8) !usize
    {
        return @intCast(s.hh.getInt(s.key(k)) orelse
        {
            std.log.err("missing GGUF key {s}", .{s.key(k)});
            return error.MissingKey;
        });
    }

    fn opt(s: *Keys, k: []const u8) ?usize
    {
        return if (s.hh.getInt(s.key(k))) |v| @intCast(v) else null;
    }

    fn float(s: *Keys, k: []const u8, default: f32) f32
    {
        const v = s.hh.get(s.key(k)) orelse return default;
        return switch (v) {
            .float => |x| @floatCast(x),
            else => default,
        };
    }
};

/// The keys every family with grouped-query attention shares.
fn common(h: gguf.Header, name: []const u8) !Config
{
    var g: Keys = .init(h);
    const d = try g.int("embedding_length");
    const heads = try g.int("attention.head_count");
    const kv_heads = g.opt("attention.head_count_kv") orelse heads;
    if (kv_heads == 0 or heads % kv_heads != 0)
    {
        return error.Unsupported;
    }
    const embd = h.tensor("token_embd.weight") orelse return error.MissingTensor;
    return .{
        .name = name,
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

/// OLMoE: QK norm over the whole projection, top-k probabilities used as they are.
fn olmoeConfig(h: gguf.Header) anyerror!Config
{
    var cfg = try common(h, "OLMoE");
    cfg.qk_norm = .whole;
    cfg.route_norm = false;
    return cfg;
}

/// Qwen3-MoE (TASK-12): QK norm per head, top-k renormalized.
fn qwen3Config(h: gguf.Header) anyerror!Config
{
    var cfg = try common(h, "Qwen3-MoE");
    cfg.name_yarn = "Qwen3-MoE (YaRN)";
    return cfg;
}

/// Qwen3.5-MoE (TASK-29): Qwen3-MoE plus Gated DeltaNet layers, gated attention, and a gated shared expert.
fn qwen35Config(h: gguf.Header) anyerror!Config
{
    var cfg = try common(h, "Qwen3.5-MoE");
    try qwen35Parts(h, &cfg);
    cfg.ffn_norm_name = "post_attention_norm";
    return cfg;
}

fn qwen35Parts(h: gguf.Header, cfg: *Config) !void
{
    var g: Keys = .init(h);
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

/// Qwen3.8-Flash-Next (TASK-31): Qwen3.5-MoE with hyper-connections, QSA, and an n-gram embedding.
fn qwen38Config(h: gguf.Header) anyerror!Config
{
    var g: Keys = .init(h);
    var cfg = try common(h, "Qwen3.8-Flash-Next");
    try qwen35Parts(h, &cfg);
    // llama.cpp merges the Gated DeltaNet's projections that read the same input.
    cfg.ssm_merged = true;
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
    if (cfg.streams < 2 or cfg.qsa_block == 0 or cfg.qsa_budget % cfg.qsa_block != 0 or cfg.qsa_dim > olmoe.max_head or cfg.qsa_dim > cfg.head_dim)
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
        if (cfg.ple_ngram < 2 or cfg.ple_ngram > Config.max_ngram or cfg.pleHeads() > Config.max_ple_heads or cfg.ple_layer.? >= cfg.layers)
        {
            return error.Unsupported;
        }
    }
    return cfg;
}

/// GLM-5.3-Flash (TASK-32, GGUF "glm5-next"): KDA and MLA layers with a DSA indexer, mHC, dense leading layers.
fn glm53Config(h: gguf.Header) anyerror!Config
{
    var cfg = try glm53.config(h);
    cfg.name = "GLM-5.3-Flash";
    cfg.router = .sigmoid;
    cfg.route_norm = true;
    return cfg;
}

/// Mixtral (TASK-39, GGUF "llama" with experts): no QK norm, rope on adjacent pairs.
fn mixtralConfig(h: gguf.Header) anyerror!Config
{
    var g: Keys = .init(h);
    var cfg = try common(h, "Mixtral");
    // A dense Llama has no experts; llama.cpp converts Mixtral as "llama", rotating adjacent pairs.
    if (g.opt("expert_count") == null or (g.opt("rope.dimension_count") orelse cfg.head_dim) != cfg.head_dim or h.get("llama.rope.scaling.type") != null)
    {
        return error.Unsupported;
    }
    cfg.qk_norm = .none;
    cfg.rope_adjacent = true;
    return cfg;
}

/// DeepSeek V3 and Kimi K2 (TASK-39, GGUF "deepseek2"): MLA with decoupled rope, grouped sigmoid routing.
fn deepseekConfig(h: gguf.Header) anyerror!Config
{
    var cfg = try deepseek.config(h);
    cfg.name = "DeepSeek V3";
    cfg.router = .sigmoid;
    // DeepSeek's YaRN leaves cos and sin unscaled; its size is in `attn_scale`.
    cfg.yarn_mscale = false;
    return cfg;
}

/// gpt-oss (TASK-39): attention sinks, sliding windows, biases, and its own SwiGLU.
fn gptossConfig(h: gguf.Header) anyerror!Config
{
    var cfg = try common(h, "gpt-oss");
    cfg.qk_norm = .none;
    cfg.ffn_norm_name = "post_attention_norm";
    cfg.attn_bias = true;
    cfg.attn_out_bias = true;
    cfg.attn_sinks = true;
    cfg.router_bias = true;
    cfg.expert_bias = true;
    cfg.swiglu_oai = true;
    try gptoss.configure(h, &cfg);
    return cfg;
}

/// Qwen2-MoE and Qwen1.5-MoE (TASK-42): biases on the query, key and value projections,
/// no QK norm, a gated shared expert, and top-k probabilities used as they are.
fn qwen2Config(h: gguf.Header) anyerror!Config
{
    var g: Keys = .init(h);
    var cfg = try common(h, "Qwen2-MoE");
    cfg.qk_norm = .none;
    cfg.attn_bias = true;
    cfg.shared_ff = try g.int("expert_shared_feed_forward_length");
    // transformers' norm_topk_prob, which the GGUF does not record, is false in every released model.
    cfg.route_norm = false;
    return cfg;
}
