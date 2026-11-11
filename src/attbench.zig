//! Attention kernel benchmark (TASK-12). Linux, `zig build attbench -Dgpu=true`.
//!
//!   attbench [--heads H] [--kv-heads K] [--n N] [--pos P[,P...]] [--reps R] [--alloc map|host]
//!   attbench --mm [--alloc map|host] [--reps R]
//!
//! Times each attention kernel on random data for N new tokens after P cached
//! positions, and reports how far each output is from the wide kernel's. `--mm`
//! times the matmul kernels instead. `--alloc host` puts data in pinned memory.
const std = @import("std");
const c = std.c;

const Handle = opaque {};
extern fn tf_gpu_open(out: *?*Handle, fatbin: *const anyopaque, err: [*]u8, err_len: usize) c_int;
extern fn tf_gpu_map(g: *Handle, host: *anyopaque, len: usize, dev: *u64) c_int;
extern fn tf_gpu_attention(g: *Handle, q: [*]const f32, n: c_int, pos0: c_int, heads: c_int, kv_heads: c_int, kvd: c_int, scale: f32, tiles_k: [*]const u64, tiles_v: [*]const u64, tiles: c_int, splits: c_int, out: [*]f32) c_int;
extern fn tf_gpu_set_attention_kernel(g: *Handle, which: c_int) c_int;
extern fn tf_gpu_close(g: *Handle) void;
extern fn tf_gpu_matmul(g: *Handle, w: u64, format: c_int, x: [*]const f32, n: c_int, rows: c_int, cols: c_int, y: [*]f32) c_int;
extern fn tf_gpu_set_matmul_kernel(g: *Handle, which: c_int) c_int;
extern fn tf_gpu_alloc_host(g: *Handle, len: usize, host: *?*anyopaque, dev: *u64) c_int;

const fatbin align(64) = @embedFile("kernels.fatbin").*;
const hd = 128;
const tile = 32;
/// tf_gpu_set_attention_kernel 1 to 5 (decode only up to 16 rows), compared with the wide kernel (2).
const names = [_][]const u8{ "first", "wide", "tensor cores 64", "tensor cores 128", "decode" };
const kernels = names.len;
const max_decode_rows = 16;

pub fn main(init: std.process.Init) !void
{
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    var heads: usize = 32;
    var kv_heads: usize = 4;
    var n: usize = 2048;
    var reps: usize = 3;
    var pos_text: []const u8 = "0,16384,65536,129024";
    var pinned = false;
    var mm = false;
    while (it.next()) |arg|
    {
        if (std.mem.eql(u8, arg, "--mm"))
        {
            mm = true;
            continue;
        }
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
        else if (std.mem.eql(u8, arg, "--alloc"))
        {
            pinned = std.mem.eql(u8, v, "host");
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
    if (mm)
    {
        return matmuls(g.?, pinned, reps);
    }
    const pages = std.heap.page_allocator;
    const kvd = kv_heads * hd;
    const cap = (max_pos + n + tile - 1) / tile * tile;
    var dk: u64 = 0;
    var dv: u64 = 0;
    const k = try deviceBuffer(g.?, pinned, cap * kvd, &dk);
    const v = try deviceBuffer(g.?, pinned, cap * kvd, &dv);
    const q = try pages.alloc(f32, n * heads * hd);
    defer pages.free(q);
    var out: [kernels][]f32 = undefined;
    for (&out) |*o|
    {
        o.* = try pages.alloc(f32, q.len);
    }
    defer for (out) |o|
    {
        pages.free(o);
    };
    var rng: std.Random.DefaultPrng = .init(7);
    const r = rng.random();
    for ([_][]f32{ k, v, q }) |buf|
    {
        for (buf) |*x|
        {
            x.* = r.float(f32) * 2 - 1;
        }
    }
    if (!pinned and (tf_gpu_map(g.?, k.ptr, k.len * 4, &dk) != 0 or tf_gpu_map(g.?, v.ptr, v.len * 4, &dv) != 0))
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
        var ms: [kernels]f64 = @splat(0);
        const used: usize = if (rows <= max_decode_rows) kernels else kernels - 1;
        for (0..used) |which|
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
        const avg_seen = @as(f64, @floatFromInt(pos0)) + @as(f64, @floatFromInt(n + 1)) / 2;
        const flops = 4 * @as(f64, @floatFromInt(n * heads * hd)) * avg_seen;
        const bytes = @as(f64, @floatFromInt((pos0 + n) * kvd * 4 * 2));
        std.debug.print("  {d} cached:", .{pos0});
        for (names[0..used], ms[0..used], out[0..used]) |name, t, o|
        {
            var worst: f32 = 0;
            for (o, out[1]) |a, b|
            {
                worst = @max(worst, @abs(a - b));
            }
            std.debug.print(" {s} {d:.2} ms ({d:.2} TFLOPS, {d:.0} GB/s, {e:.1} off wide);", .{ name, t, flops / t / 1e9, bytes / t / 1e6, worst });
        }
        std.debug.print("\n", .{});
    }
}

/// `len` floats in pinned memory (device address in `dev`) or from the page allocator
/// (mapped after they are filled). Freed at exit.
fn deviceBuffer(g: *Handle, pinned: bool, len: usize, dev: *u64) ![]f32
{
    if (!pinned)
    {
        return std.heap.page_allocator.alloc(f32, len);
    }
    var host: ?*anyopaque = null;
    if (tf_gpu_alloc_host(g, len * 4, &host, dev) != 0)
    {
        return error.GpuAllocFailed;
    }
    const p: [*]f32 = @ptrCast(@alignCast(host.?));
    return p[0..len];
}

/// Decode matmul bandwidth on a large Q8_0 weight, then prefill speed of the
/// tiled f32 kernel and the tensor-core kernel on an expert-sized weight.
fn matmuls(g: *Handle, pinned: bool, reps: usize) !void
{
    const cols = 2048;
    const big_rows = 65536;
    const nb = cols / 32;
    const bytes = big_rows * nb * 34;
    var dev: u64 = 0;
    const wf = try deviceBuffer(g, pinned, (bytes + 3) / 4, &dev);
    const w = std.mem.sliceAsBytes(wf)[0..bytes];
    var rng: std.Random.DefaultPrng = .init(11);
    const r = rng.random();
    for (0..big_rows * nb) |b|
    {
        const blk = w[b * 34 ..][0..34];
        std.mem.writeInt(u16, blk[0..2], 0x211f, .little); // f16 0.01
        r.bytes(blk[2..]);
    }
    if (!pinned and tf_gpu_map(g, w.ptr, w.len, &dev) != 0)
    {
        return error.GpuMapFailed;
    }
    const n_big = 2048;
    const x = try std.heap.page_allocator.alloc(f32, n_big * cols);
    for (x) |*v|
    {
        v.* = r.float(f32) * 2 - 1;
    }
    const y = try std.heap.page_allocator.alloc(f32, 8 * big_rows);
    std.debug.print("attbench --mm: weights in {s} memory\n", .{if (pinned) "pinned" else "mapped"});
    for ([_]usize{ 1, 8 }) |n|
    {
        var best: u64 = std.math.maxInt(u64);
        for (0..reps + 1) |_|
        {
            const t0 = nowNs();
            if (tf_gpu_matmul(g, dev, 0, x.ptr, @intCast(n), big_rows, cols, y.ptr) != 0)
            {
                return error.MatmulFailed;
            }
            best = @min(best, nowNs() - t0);
        }
        std.debug.print("  {d} x {d} Q8_0, {d} inputs: {d:.2} ms, {d:.0} GB/s\n", .{ big_rows, cols, n, @as(f64, @floatFromInt(best)) / 1e6, @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(best)) });
    }
    const rows = 2048;
    const outs = [2][]f32{ try std.heap.page_allocator.alloc(f32, n_big * rows), try std.heap.page_allocator.alloc(f32, n_big * rows) };
    const kernel_names = [_][]const u8{ "tiled f32", "tensor cores" };
    const which = [_]c_int{ 2, 0 };
    var ms: [2]f64 = undefined;
    for (which, 0..) |kernel, i|
    {
        _ = tf_gpu_set_matmul_kernel(g, kernel);
        var best: u64 = std.math.maxInt(u64);
        for (0..reps + 1) |_|
        {
            const t0 = nowNs();
            if (tf_gpu_matmul(g, dev, 0, x.ptr, n_big, rows, cols, outs[i].ptr) != 0)
            {
                return error.MatmulFailed;
            }
            best = @min(best, nowNs() - t0);
        }
        ms[i] = @as(f64, @floatFromInt(best)) / 1e6;
    }
    _ = tf_gpu_set_matmul_kernel(g, 0);
    var worst: f32 = 0;
    var largest: f32 = 0;
    for (outs[0], outs[1]) |a, b|
    {
        worst = @max(worst, @abs(a - b));
        largest = @max(largest, @abs(a));
    }
    const flops = 2.0 * @as(f64, rows * cols * n_big);
    for (kernel_names, ms) |name, t|
    {
        std.debug.print("  {d} x {d} Q8_0, {d} inputs, {s}: {d:.2} ms ({d:.2} TFLOPS)\n", .{ rows, cols, n_big, name, t, flops / t / 1e9 });
    }
    std.debug.print("  outputs differ by up to {e:.2} (largest output {d:.1})\n", .{ worst, largest });
    // Each kernel against a double-precision sum on the CPU, for the first inputs.
    const checked = 32;
    var err: [2]f64 = @splat(0);
    for (0..rows) |row|
    {
        for (0..checked) |t|
        {
            var sum: f64 = 0;
            for (0..nb) |b|
            {
                const blk = w[(row * nb + b) * 34 ..][0..34];
                const d: f64 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, blk[0..2], .little))));
                var part: f64 = 0;
                for (0..32) |j|
                {
                    part += @as(f64, @floatFromInt(@as(i8, @bitCast(blk[2 + j])))) * x[t * cols + b * 32 + j];
                }
                sum += d * part;
            }
            for (outs, 0..) |o, i|
            {
                err[i] = @max(err[i], @abs(o[t * rows + row] - sum));
            }
        }
    }
    std.debug.print("  against a double-precision sum ({d} inputs): tiled f32 off by up to {e:.2}, tensor cores by {e:.2}\n", .{ checked, err[0], err[1] });
}

fn nowNs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
