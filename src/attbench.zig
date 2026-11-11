//! Attention kernel benchmark (TASK-12). Linux, `zig build attbench -Dgpu=true`.
//!
//!   attbench [--heads H] [--kv-heads K] [--n N] [--pos P[,P...]] [--reps R]
//!
//! Times tf_attention and tf_attention_wide on random queries and a random
//! KV cache read in place, for N new tokens after P cached positions
//! (defaults: Qwen3-30B-A3B's 32 query and 4 KV heads, a 2,048-token chunk),
//! and reports how far apart their outputs are. The cache is read in place on
//! a GPU that shares memory (GB10) and copied to device memory otherwise.
const std = @import("std");
const c = std.c;

const Handle = opaque {};
extern fn tf_gpu_open(out: *?*Handle, fatbin: *const anyopaque, err: [*]u8, err_len: usize) c_int;
extern fn tf_gpu_map(g: *Handle, host: *anyopaque, len: usize, dev: *u64) c_int;
extern fn tf_gpu_attention(g: *Handle, q: [*]const f32, n: c_int, pos0: c_int, heads: c_int, kv_heads: c_int, kvd: c_int, scale: f32, tiles_k: [*]const u64, tiles_v: [*]const u64, tiles: c_int, splits: c_int, out: [*]f32) c_int;
extern fn tf_gpu_set_attention_kernel(g: *Handle, which: c_int) c_int;
extern fn tf_gpu_close(g: *Handle) void;

const fatbin align(64) = @embedFile("kernels.fatbin").*;
const hd = 128;
const tile = 32;

pub fn main(init: std.process.Init) !void
{
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    var heads: usize = 32;
    var kv_heads: usize = 4;
    var n: usize = 2048;
    var reps: usize = 3;
    var pos_text: []const u8 = "0,16384,65536,129024";
    while (it.next()) |arg|
    {
        const v = it.next() orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--heads"))
        {
            heads = try std.fmt.parseInt(usize, v, 10);
        }
        else if (std.mem.eql(u8, arg, "--kv-heads"))
        {
            kv_heads = try std.fmt.parseInt(usize, v, 10);
        }
        else if (std.mem.eql(u8, arg, "--n"))
        {
            n = try std.fmt.parseInt(usize, v, 10);
        }
        else if (std.mem.eql(u8, arg, "--pos"))
        {
            pos_text = v;
        }
        else if (std.mem.eql(u8, arg, "--reps"))
        {
            reps = try std.fmt.parseInt(usize, v, 10);
        }
        else
        {
            return error.Usage;
        }
    }
    var positions: [16]usize = undefined;
    var count: usize = 0;
    var split = std.mem.splitScalar(u8, pos_text, ',');
    var max_pos: usize = 0;
    while (split.next()) |p|
    {
        positions[count] = try std.fmt.parseInt(usize, p, 10);
        max_pos = @max(max_pos, positions[count]);
        count += 1;
    }

    var err: [256]u8 = @splat(0);
    var g: ?*Handle = null;
    if (tf_gpu_open(&g, &fatbin, &err, err.len) != 0)
    {
        std.debug.print("gpu: {s}\n", .{std.mem.sliceTo(&err, 0)});
        return error.GpuOpenFailed;
    }
    defer tf_gpu_close(g.?);
    const pages = std.heap.page_allocator;
    const kvd = kv_heads * hd;
    const cap = (max_pos + n + tile - 1) / tile * tile;
    const k = try pages.alloc(f32, cap * kvd);
    defer pages.free(k);
    const v = try pages.alloc(f32, cap * kvd);
    defer pages.free(v);
    const q = try pages.alloc(f32, n * heads * hd);
    defer pages.free(q);
    const out = [2][]f32{ try pages.alloc(f32, q.len), try pages.alloc(f32, q.len) };
    defer pages.free(out[0]);
    defer pages.free(out[1]);
    var rng: std.Random.DefaultPrng = .init(7);
    const r = rng.random();
    for ([_][]f32{ k, v, q }) |buf|
    {
        for (buf) |*x|
        {
            x.* = r.float(f32) * 2 - 1;
        }
    }
    var dk: u64 = 0;
    var dv: u64 = 0;
    if (tf_gpu_map(g.?, k.ptr, k.len * 4, &dk) != 0 or tf_gpu_map(g.?, v.ptr, v.len * 4, &dv) != 0)
    {
        return error.GpuMapFailed;
    }
    const tiles_k = try pages.alloc(u64, cap / tile);
    defer pages.free(tiles_k);
    const tiles_v = try pages.alloc(u64, cap / tile);
    defer pages.free(tiles_v);
    for (tiles_k, tiles_v, 0..) |*a, *b, i|
    {
        a.* = dk + i * tile * kvd * 4;
        b.* = dv + i * tile * kvd * 4;
    }
    const scale = 1.0 / @sqrt(@as(f32, hd));
    std.debug.print("attbench: {d} query heads, {d} KV heads, {d} new tokens, {d} reps\n", .{ heads, kv_heads, n, reps });
    for (positions[0..count]) |pos0|
    {
        const ntiles = (pos0 + n + tile - 1) / tile;
        const rows = n * (heads / kv_heads);
        const blocks = (rows + 31) / 32 * kv_heads;
        const splits = std.math.clamp(@min((128 + blocks - 1) / blocks, ntiles / 8), 1, 256);
        var ms: [2]f64 = undefined;
        for (0..2) |which|
        {
            _ = tf_gpu_set_attention_kernel(g.?, @intCast(which + 1));
            var best: u64 = std.math.maxInt(u64);
            for (0..reps + 1) |_|
            {
                const t0 = nowNs();
                if (tf_gpu_attention(g.?, q.ptr, @intCast(n), @intCast(pos0), @intCast(heads), @intCast(kv_heads), @intCast(kvd), scale, tiles_k.ptr, tiles_v.ptr, @intCast(ntiles), @intCast(splits), out[which].ptr) != 0)
                {
                    return error.AttentionFailed;
                }
                best = @min(best, nowNs() - t0);
            }
            ms[which] = @as(f64, @floatFromInt(best)) / 1e6;
        }
        var worst: f32 = 0;
        for (out[0], out[1]) |a, b|
        {
            worst = @max(worst, @abs(a - b));
        }
        const avg_seen = @as(f64, @floatFromInt(pos0)) + @as(f64, @floatFromInt(n + 1)) / 2;
        const flops = 4 * @as(f64, @floatFromInt(n * heads * hd)) * avg_seen;
        std.debug.print("  {d} cached positions: tf_attention {d:.1} ms ({d:.2} TFLOPS), tf_attention_wide {d:.1} ms ({d:.2} TFLOPS), {d:.2}x; outputs differ by up to {e:.2}\n", .{
            pos0, ms[0], flops / ms[0] / 1e9, ms[1], flops / ms[1] / 1e9, ms[0] / ms[1], worst,
        });
    }
}

fn nowNs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
