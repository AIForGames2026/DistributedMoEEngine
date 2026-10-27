//! Plain-Zig CPU reference for the tiny synthetic MoE in tools/ref/ref_moe.py
//! (TASK-06). Same math, float32, one token at a time; checked against the
//! PyTorch golden file in tests/golden. TASK-10's GPU kernels are checked
//! against the same goldens.
const std = @import("std");
const tfref = @import("tfref.zig");

pub const Config = struct {
    layers: usize,
    experts: usize,
    top_k: usize,
    d_model: usize,
    d_ff: usize,
    vocab: usize,
    tokens: usize,
    eps: f32 = 1e-5,

    pub fn fromFile(f: tfref.File) !Config {
        const c = (try f.get("config")).i32s();
        return .{
            .layers = @intCast(c[0]),
            .experts = @intCast(c[1]),
            .top_k = @intCast(c[2]),
            .d_model = @intCast(c[3]),
            .d_ff = @intCast(c[4]),
            .vocab = @intCast(c[5]),
            .tokens = @intCast(c[6]),
        };
    }
};

const max_dim = 4096;
const max_experts = 256;

fn rmsnorm(x: []const f32, w: []align(1) const f32, eps: f32, out: []f32) void {
    var ss: f32 = 0;
    for (x) |v| ss += v * v;
    const scale = 1.0 / @sqrt(ss / @as(f32, @floatFromInt(x.len)) + eps);
    for (out, x, w) |*o, v, g| o.* = v * scale * g;
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Per-layer outputs, laid out like the goldens ([tokens, ...] row-major).
pub const LayerOut = struct {
    router_logits: []f32,
    topk_idx: []i32,
    topk_w: []f32,
    moe_out: []f32,
};

/// Runs MoE block `l` on `h` ([tokens, d_model]) in place.
pub fn layer(cfg: Config, f: tfref.File, l: usize, h: []f32, out: LayerOut) !void {
    var name_buf: [64]u8 = undefined;
    const w = struct {
        fn get(file: tfref.File, buf: []u8, li: usize, field: []const u8) ![]align(1) const f32 {
            return (try file.get(try std.fmt.bufPrint(buf, "layer{d}.{s}", .{ li, field }))).f32s();
        }
    };
    const norm_w = try w.get(f, &name_buf, l, "norm_w");
    const router_w = try w.get(f, &name_buf, l, "router_w");
    const gate = try w.get(f, &name_buf, l, "gate");
    const up = try w.get(f, &name_buf, l, "up");
    const down = try w.get(f, &name_buf, l, "down");

    const d = cfg.d_model;
    const ff = cfg.d_ff;
    const e_count = cfg.experts;
    std.debug.assert(d <= max_dim and ff <= max_dim and e_count <= max_experts);

    var n: [max_dim]f32 = undefined;
    var a: [max_dim]f32 = undefined;
    var b: [max_dim]f32 = undefined;
    var probs: [max_experts]f32 = undefined;

    for (0..cfg.tokens) |t| {
        const row = h[t * d ..][0..d];
        rmsnorm(row, norm_w, cfg.eps, n[0..d]);

        const logits = out.router_logits[t * e_count ..][0..e_count];
        for (logits, 0..) |*lg, e| {
            var acc: f32 = 0;
            for (0..d) |i| acc += n[i] * router_w[i * e_count + e];
            lg.* = acc;
        }
        var max: f32 = -std.math.inf(f32);
        for (logits) |lg| max = @max(max, lg);
        var sum: f32 = 0;
        for (probs[0..e_count], logits) |*p, lg| {
            p.* = @exp(lg - max);
            sum += p.*;
        }
        for (probs[0..e_count]) |*p| p.* /= sum;

        // Top-k by repeated selection, descending; ties go to the lower index.
        var taken: [max_experts]bool = @splat(false);
        const moe = out.moe_out[t * d ..][0..d];
        @memset(moe, 0);
        for (0..cfg.top_k) |j| {
            var best: usize = 0;
            var best_p: f32 = -1;
            for (probs[0..e_count], 0..) |p, e| {
                if (!taken[e] and p > best_p) {
                    best = e;
                    best_p = p;
                }
            }
            taken[best] = true;
            out.topk_idx[t * cfg.top_k + j] = @intCast(best);
            out.topk_w[t * cfg.top_k + j] = best_p;

            // expert(n) = (silu(n @ gate) * (n @ up)) @ down
            const g_e = gate[best * d * ff ..][0 .. d * ff];
            const u_e = up[best * d * ff ..][0 .. d * ff];
            const dn_e = down[best * ff * d ..][0 .. ff * d];
            for (0..ff) |k| {
                var ga: f32 = 0;
                var ub: f32 = 0;
                for (0..d) |i| {
                    ga += n[i] * g_e[i * ff + k];
                    ub += n[i] * u_e[i * ff + k];
                }
                a[k] = silu(ga) * ub;
            }
            for (0..d) |i| {
                var acc: f32 = 0;
                for (0..ff) |k| acc += a[k] * dn_e[k * d + i];
                b[i] = acc;
            }
            for (moe, b[0..d]) |*m, v| m.* += best_p * v;
        }
        for (row, moe) |*x, m| x.* += m;
    }
}

/// Final norm and LM head: logits [tokens, vocab].
pub fn head(cfg: Config, f: tfref.File, h: []const f32, logits: []f32) !void {
    const norm_w = (try f.get("final_norm_w")).f32s();
    const lm = (try f.get("lm_head")).f32s();
    var n: [max_dim]f32 = undefined;
    for (0..cfg.tokens) |t| {
        rmsnorm(h[t * cfg.d_model ..][0..cfg.d_model], norm_w, cfg.eps, n[0..cfg.d_model]);
        for (0..cfg.vocab) |v| {
            var acc: f32 = 0;
            for (0..cfg.d_model) |i| acc += n[i] * lm[i * cfg.vocab + v];
            logits[t * cfg.vocab + v] = acc;
        }
    }
}

/// torch.allclose-style check: |x - y| <= atol + rtol * |y| for every element.
/// The absolute term covers near-zero values, where float32 summation-order
/// differences are large relative to the value itself.
fn allClose(x: []const f32, y: []align(1) const f32) bool {
    const atol = 1e-5;
    const rtol = 1e-4;
    if (x.len != y.len) return false;
    for (x, y) |u, v| {
        if (!(@abs(u - v) <= atol + rtol * @abs(v))) return false;
    }
    return true;
}

test "CPU reference matches the PyTorch goldens layer by layer" {
    const f = try tfref.File.parse(@embedFile("moe_tiny.tfref"));
    const cfg = try Config.fromFile(f);
    const gpa = std.testing.allocator;

    const h = try gpa.alloc(f32, cfg.tokens * cfg.d_model);
    defer gpa.free(h);
    for (h, (try f.get("input")).f32s()) |*x, v| x.* = v;

    const out: LayerOut = .{
        .router_logits = try gpa.alloc(f32, cfg.tokens * cfg.experts),
        .topk_idx = try gpa.alloc(i32, cfg.tokens * cfg.top_k),
        .topk_w = try gpa.alloc(f32, cfg.tokens * cfg.top_k),
        .moe_out = try gpa.alloc(f32, cfg.tokens * cfg.d_model),
    };
    defer {
        gpa.free(out.router_logits);
        gpa.free(out.topk_idx);
        gpa.free(out.topk_w);
        gpa.free(out.moe_out);
    }

    var name: [64]u8 = undefined;
    for (0..cfg.layers) |l| {
        try layer(cfg, f, l, h, out);
        const g = struct {
            fn get(file: tfref.File, buf: []u8, li: usize, field: []const u8) !tfref.Tensor {
                return file.get(try std.fmt.bufPrint(buf, "golden.layer{d}.{s}", .{ li, field }));
            }
        };
        // Routing must match exactly; values within float32 reordering error.
        for (out.topk_idx, (try g.get(f, &name, l, "topk_idx")).i32s()) |got, want| try std.testing.expectEqual(want, got);
        try std.testing.expect(allClose(out.router_logits, (try g.get(f, &name, l, "router_logits")).f32s()));
        try std.testing.expect(allClose(out.topk_w, (try g.get(f, &name, l, "topk_w")).f32s()));
        try std.testing.expect(allClose(out.moe_out, (try g.get(f, &name, l, "moe_out")).f32s()));
        try std.testing.expect(allClose(h, (try g.get(f, &name, l, "hidden")).f32s()));
    }

    const logits = try gpa.alloc(f32, cfg.tokens * cfg.vocab);
    defer gpa.free(logits);
    try head(cfg, f, h, logits);
    try std.testing.expect(allClose(logits, (try f.get("golden.logits")).f32s()));
}
