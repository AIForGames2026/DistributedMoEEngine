//! gpt-oss (TASK-39, transformers' gpt_oss, GGUF "gpt-oss"), run by `olmoe.step`'s
//! own layer loop with these additions: biases on the attention projections and the
//! router, a learned sink per head that joins each attention softmax, sliding-window
//! attention on every other layer from the first, YaRN without whole-dimension ramp
//! ends, and experts with biases and gpt-oss's clamped SwiGLU (`swiglu`).
const std = @import("std");
const olmoe = @import("olmoe.zig");
const gguf = @import("gguf.zig");
const loader = @import("loader.zig");
const quant = @import("quant.zig");

const Config = olmoe.Config;

/// gpt-oss's SwiGLU constants (transformers' GptOssExperts, llama.cpp's swiglu_oai).
pub const alpha: f32 = 1.702;
pub const limit: f32 = 7.0;

/// Fills in what `Config.fromGguf` does not read for gpt-oss.
pub fn configure(h: gguf.Header, cfg: *Config) !void
{
    const float = struct {
        fn f(hh: gguf.Header, key: []const u8) ?f32
        {
            const v = hh.get(key) orelse return null;
            return switch (v) {
                .float => |x| @floatCast(x),
                else => null,
            };
        }
    }.f;
    cfg.sliding_window = @intCast(h.getInt("gpt-oss.attention.sliding_window") orelse return error.MissingKey);
    cfg.swiglu_limit = limit;
    if (h.getString("gpt-oss.rope.scaling.type")) |kind|
    {
        const fast = float(h, "gpt-oss.rope.scaling.yarn_beta_fast") orelse 32;
        const slow = float(h, "gpt-oss.rope.scaling.yarn_beta_slow") orelse 1;
        if (!std.mem.eql(u8, kind, "yarn") or fast != 32 or slow != 1)
        {
            return error.Unsupported;
        }
        cfg.yarn = float(h, "gpt-oss.rope.scaling.factor") orelse return error.MissingKey;
        cfg.rope_original = @intCast(h.getInt("gpt-oss.rope.scaling.original_context_length") orelse return error.MissingKey);
        // gpt-oss's config sets truncate false, which the GGUF does not record.
        cfg.yarn_truncate = false;
    }
    if (cfg.sliding_window == 0 or cfg.head_dim > olmoe.Rope.max_half * 2)
    {
        return error.Unsupported;
    }
}

/// A gpt-oss layer's biases, sinks, and window (0 in a full-attention layer).
pub const Parts = struct {
    q_b: []f32,
    k_b: []f32,
    v_b: []f32,
    o_b: []f32,
    sinks: []f32,
    router_b: []f32,
    window: usize,
};

pub fn parts(h: gguf.Header, plan: loader.Plan, region: []const u8, a: std.mem.Allocator, cfg: Config, layer: usize) !Parts
{
    const Get = struct {
        hh: gguf.Header,
        p: loader.Plan,
        reg: []const u8,
        al: std.mem.Allocator,
        l: usize,

        fn vec(s: @This(), name: []const u8, n: usize) ![]f32
        {
            var b: [96]u8 = undefined;
            const full = std.fmt.bufPrint(&b, "blk.{d}.{s}", .{ s.l, name }) catch unreachable;
            const t = s.hh.tensor(full) orelse
            {
                std.log.err("missing tensor {s}", .{full});
                return error.MissingTensor;
            };
            if (t.elements() != n)
            {
                std.log.err("{s}: dims {any}, expected {d} values", .{ full, t.dims[0..t.ndim], n });
                return error.BadShape;
            }
            const out = try s.al.alloc(f32, n);
            try quant.dequantize(t.format, s.p.locate(s.reg, .{ .offset = t.offset, .len = t.bytes }), out);
            return out;
        }
    };
    const g: Get = .{ .hh = h, .p = plan, .reg = region, .al = a, .l = layer };
    return .{
        .q_b = try g.vec("attn_q.bias", cfg.qDim()),
        .k_b = try g.vec("attn_k.bias", cfg.kvDim()),
        .v_b = try g.vec("attn_v.bias", cfg.kvDim()),
        .o_b = try g.vec("attn_output.bias", cfg.d_model),
        .sinks = try g.vec("attn_sinks.weight", cfg.heads),
        .router_b = try g.vec("ffn_gate_inp.bias", cfg.experts),
        // Sliding layers alternate with full ones, starting with a sliding one, as released.
        .window = if (layer % 2 == 0) cfg.sliding_window else 0,
    };
}

/// Adds `b` to every row of `x` (rows of b.len).
pub fn addRows(x: []f32, b: []const f32) void
{
    var off: usize = 0;
    while (off < x.len) : (off += b.len)
    {
        for (x[off..][0..b.len], b) |*v, bv|
        {
            v.* += bv;
        }
    }
}

/// gpt-oss's SwiGLU: the gate clamped from above, the up projection on both sides and
/// shifted by one, times the gate and a sigmoid of alpha times the gate, in transformers' order.
pub fn swiglu(g: f32, u: f32, lim: f32) f32
{
    const gc = @min(g, lim);
    const uc = std.math.clamp(u, -lim, lim);
    const glu = gc * (1.0 / (1.0 + @exp(-gc * alpha)));
    return (uc + 1) * glu;
}
