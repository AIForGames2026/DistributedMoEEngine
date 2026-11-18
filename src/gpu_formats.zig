//! `olmoe gpu-formats` (TASK-36): every weight format's GPU matmul and grouped expert path
//! against the CPU, on random blocks with realistic scales. Exits with status 1 if any output
//! is more than 1e-3 from the CPU's, the bound TF_GPU_CHECK applies.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const quant = @import("quant.zig");
const gpu = @import("gpu.zig");

const rows = 512;
const cols = 2048;
const d_model = 1024;
const d_ff = 512;
const experts = 8;
const tolerance = 1e-3;

/// Every format but f32, whose kernels predate this check.
const formats = blk: {
    const all = std.enums.values(quant.Format);
    var out: [all.len - 1]quant.Format = undefined;
    var i = 0;
    for (all) |f|
    {
        if (f != .f32)
        {
            out[i] = f;
            i += 1;
        }
    }
    break :blk out;
};

/// Gate and up in one format, down in another, as Q4_K_M and IQ4_XS files mix them.
const mixes = [_][2]quant.Format{ .{ .q4_k, .q6_k }, .{ .q4_k, .q5_k }, .{ .iq4_xs, .q6_k }, .{ .q3_k, .q5_k }, .{ .iq2_xs, .q4_k }, .{ .q8_0, .q6_k } };

fn bytesOf(f: quant.Format, r: usize, c: usize) usize
{
    return r * (c / f.blockValues()) * f.blockBytes();
}

fn putHalf(b: []u8, at: usize, v: f32) void
{
    std.mem.writeInt(u16, b[at..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
}

/// Random blocks of `f`, with every scale set to a size real weights have.
fn fill(rng: std.Random, f: quant.Format, buf: []u8) void
{
    rng.bytes(buf);
    if (f == .f16 or f == .bf16)
    {
        var i: usize = 0;
        while (i < buf.len) : (i += 2)
        {
            const v = rng.float(f32) * 0.2 - 0.1;
            if (f == .f16)
            {
                putHalf(buf, i, v);
            }
            else
            {
                std.mem.writeInt(u16, buf[i..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
            }
        }
        return;
    }
    const bb = f.blockBytes();
    var at: usize = 0;
    while (at < buf.len) : (at += bb)
    {
        const blk = buf[at..][0..bb];
        // Block scales that keep weights within about 0.2 of zero.
        const d = 0.1 * (0.5 + rng.float(f32)) / largestCode(f);
        switch (f)
        {
            .q4_1, .q5_1 => {
                putHalf(blk, 0, d);
                putHalf(blk, 2, (rng.float(f32) - 0.5) * 0.05);
            },
            .q2_k => {
                putHalf(blk, 80, d);
                putHalf(blk, 82, d * 0.2);
            },
            .q3_k => putHalf(blk, 108, d),
            .q4_k, .q5_k => {
                putHalf(blk, 0, d);
                putHalf(blk, 2, d * 0.2);
            },
            .q6_k => putHalf(blk, 208, d),
            .iq1_m => {
                const bits: u16 = @bitCast(@as(f16, @floatCast(d)));
                for (0..4) |i|
                {
                    const s = std.mem.readInt(u16, blk[48 + 2 * i ..][0..2], .little);
                    const top: u16 = (bits >> @intCast(4 * i)) & 0xF;
                    std.mem.writeInt(u16, blk[48 + 2 * i ..][0..2], (s & 0x0FFF) | (top << 12), .little);
                }
            },
            .mxfp4 => blk[0] = 121 + rng.uintLessThan(u8, 4),
            .nvfp4 => {
                for (blk[0..4]) |*s|
                {
                    s.* = 0x08 + rng.uintLessThan(u8, 8);
                }
            },
            else => putHalf(blk, 0, d),
        }
    }
}

/// The largest block scale times code a format can reach, before its f16 scale.
fn largestCode(f: quant.Format) f32
{
    return switch (f)
    {
        .q8_0, .iq4_nl => 127,
        .q4_0 => 8,
        .q4_1 => 15,
        .q5_0 => 16,
        .q5_1 => 31,
        .q2_k => 15 * 3,
        .q3_k => 32 * 4,
        .q4_k => 63 * 15,
        .q5_k => 63 * 31,
        .q6_k => 128 * 32,
        .iq2_xxs, .iq2_xs, .iq2_s => 15.5 * 0.25 * 43,
        .iq3_xxs => 15.5 * 0.5 * 62,
        .iq3_s => 31 * 15,
        .iq1_s, .iq1_m => 15 * 1.125,
        .iq4_xs => 32 * 127,
        else => 1,
    };
}

/// Largest gap between a and b, and the largest magnitude in b.
fn compare(a: []const f32, b: []const f32) [2]f32
{
    var worst: f32 = 0;
    var big: f32 = 0;
    for (a, b) |x, y|
    {
        const e = @abs(x - y);
        worst = if (std.math.isNan(e)) std.math.inf(f32) else @max(worst, e);
        big = @max(big, @abs(y));
    }
    return .{ worst, big };
}

pub fn main(gpa: std.mem.Allocator, args: []const []const u8) !void
{
    _ = args;
    try gpu.open();
    defer gpu.stop();
    var prng: std.Random.DefaultPrng = .init(20261118);
    const rng = prng.random();

    var total: usize = 0;
    for (formats) |f|
    {
        total += bytesOf(f, rows, cols) + experts * (2 * bytesOf(f, d_ff, d_model) + bytesOf(f, d_model, d_ff)) + 4 * 64;
    }
    const page = std.heap.page_size_max;
    const region = try gpa.alignedAlloc(u8, .fromByteUnits(page), std.mem.alignForward(usize, total, page));
    defer gpa.free(region);
    var mats: [formats.len]olmoe.Mat = undefined;
    var sets: [formats.len][experts]olmoe.Expert = undefined;
    var used: usize = 0;
    const take = struct
    {
        fn f(r: std.Random, reg: []u8, at: *usize, fmt: quant.Format, nr: usize, nc: usize) olmoe.Mat
        {
            const len = bytesOf(fmt, nr, nc);
            const b = reg[at.*..][0..len];
            at.* = std.mem.alignForward(usize, at.* + len, 64);
            fill(r, fmt, b);
            return .{ .format = fmt, .rows = nr, .cols = nc, .bytes = b };
        }
    }.f;
    for (formats, 0..) |f, i|
    {
        mats[i] = take(rng, region, &used, f, rows, cols);
        for (&sets[i]) |*e|
        {
            e.gate = take(rng, region, &used, f, d_ff, d_model);
            e.up = take(rng, region, &used, f, d_ff, d_model);
            e.down = take(rng, region, &used, f, d_model, d_ff);
        }
    }
    try gpu.map(region);

    var failures: usize = 0;
    const max_n = 130;
    const x = try gpa.alloc(f32, max_n * @max(cols, d_model));
    defer gpa.free(x);
    for (x) |*v|
    {
        v.* = rng.float(f32) * 2 - 1;
    }
    const got = try gpa.alloc(f32, max_n * @max(rows, d_model));
    defer gpa.free(got);
    const want = try gpa.alloc(f32, max_n * @max(rows, d_model));
    defer gpa.free(want);
    const scratch = try gpa.alloc(f32, 4 * @max(cols, d_model));
    defer gpa.free(scratch);

    std.debug.print("matmuls, {d}x{d}: largest gap from the CPU at 1, 3, 8, 9, 40 and 130 inputs\n", .{ rows, cols });
    for (mats) |m|
    {
        std.debug.print("  {s:<8}", .{@tagName(m.format)});
        var ok = true;
        for ([_]usize{ 1, 3, 8, 9, 40, 130 }) |n|
        {
            const saved = olmoe.gpu_matmul;
            olmoe.gpu_matmul = null;
            m.matmul(x[0 .. n * cols], n, want[0 .. n * rows], scratch);
            olmoe.gpu_matmul = saved;
            if (!saved.?(m, x[0 .. n * cols], n, got[0 .. n * rows]))
            {
                std.debug.print(" not on the GPU", .{});
                ok = false;
                break;
            }
            const c = compare(got[0 .. n * rows], want[0 .. n * rows]);
            std.debug.print(" {e:9.2}", .{c[0]});
            ok = ok and c[0] <= tolerance;
        }
        std.debug.print("{s}\n", .{if (ok) "" else "  FAIL"});
        failures += @intFromBool(!ok);
    }

    std.debug.print("experts, {d} of {d}x{d}, 4 calls per token: largest gap at 2, 5 and 40 tokens (largest output)\n", .{ experts, d_ff, d_model });
    var cases: [formats.len + mixes.len + 1][2]usize = undefined;
    for (0..formats.len) |i|
    {
        cases[i] = .{ i, i };
    }
    for (mixes, 0..) |mx, j|
    {
        cases[formats.len + j] = .{ std.mem.indexOfScalar(quant.Format, &formats, mx[0]).?, std.mem.indexOfScalar(quant.Format, &formats, mx[1]).? };
    }
    // Q8_0 with GLM's SwiGLU clamp, which now runs on the GPU too.
    const q8 = std.mem.indexOfScalar(quant.Format, &formats, .q8_0).?;
    cases[cases.len - 1] = .{ q8, q8 };
    for (cases, 0..) |cs, ci|
    {
        const clamp = ci == cases.len - 1;
        olmoe.swiglu_limit = if (clamp) 0.05 else 0;
        defer olmoe.swiglu_limit = 0;
        var set: [experts]olmoe.Expert = undefined;
        for (&set, sets[cs[0]], sets[cs[1]]) |*e, a, b|
        {
            e.* = .{ .gate = a.gate, .up = a.up, .down = b.down };
        }
        std.debug.print("  {s:<8}{s:<8}{s:<6}", .{ @tagName(formats[cs[0]]), @tagName(formats[cs[1]]), if (clamp) "clamp" else "" });
        var ok = true;
        for ([_]usize{ 2, 5, 40 }) |n|
        {
            const c = try checkExperts(gpa, rng, &set, x[0 .. n * d_model], n, got[0 .. n * d_model], want[0 .. n * d_model], scratch);
            if (c == null)
            {
                std.debug.print(" not on the GPU", .{});
                ok = false;
                break;
            }
            std.debug.print(" {e:9.2} ({d:.1})", .{ c.?[0], c.?[1] });
            ok = ok and c.?[0] <= tolerance;
        }
        std.debug.print("{s}\n", .{if (ok) "" else "  FAIL"});
        failures += @intFromBool(!ok);
    }
    if (failures > 0)
    {
        std.debug.print("FAIL: {d} formats or mixes differ from the CPU by more than {e}\n", .{ failures, tolerance });
        std.process.exit(1);
    }
    std.debug.print("PASS: every format agrees with the CPU within {e}\n", .{tolerance});
}

/// One grouped expert call of `n` tokens, 4 experts each, against the CPU; null if it stayed on the CPU.
fn checkExperts(gpa: std.mem.Allocator, rng: std.Random, set: []const olmoe.Expert, x: []const f32, n: usize, got: []f32, want: []f32, scratch: []f32) !?[2]f32
{
    const k = 4;
    const idx = try gpa.alloc(u32, n * k);
    defer gpa.free(idx);
    const w = try gpa.alloc(f32, n * k);
    defer gpa.free(w);
    for (0..n) |t|
    {
        var pick: [experts]u32 = undefined;
        for (&pick, 0..) |*p, i|
        {
            p.* = @intCast(i);
        }
        rng.shuffle(u32, &pick);
        for (0..k) |j|
        {
            idx[t * k + j] = pick[j];
            w[t * k + j] = rng.float(f32);
        }
    }
    // Batch rows grouped by expert, experts with few rows first, as the engine orders them.
    var count: [experts]u32 = @splat(0);
    for (idx) |e|
    {
        count[e] += 1;
    }
    var order: [experts]u32 = undefined;
    var used: usize = 0;
    for ([_]bool{ true, false }) |small|
    {
        for (0..experts) |e|
        {
            if (count[e] > 0 and (count[e] <= olmoe.gpu_small_rows) == small)
            {
                order[used] = @intCast(e);
                used += 1;
            }
        }
    }
    var job_experts: [experts]olmoe.Expert = undefined;
    var first: [experts + 1]u32 = undefined;
    var start: [experts]u32 = undefined;
    first[0] = 0;
    for (order[0..used], 0..) |e, g|
    {
        job_experts[g] = set[e];
        start[e] = first[g];
        first[g + 1] = first[g] + count[e];
    }
    const src = try gpa.alloc(u32, n * k);
    defer gpa.free(src);
    const inv = try gpa.alloc(u32, n * k);
    defer gpa.free(inv);
    var fill_at = start;
    for (idx, 0..) |e, call|
    {
        const row = fill_at[e];
        fill_at[e] += 1;
        src[row] = @intCast(call / k);
        inv[call] = row;
    }
    const on_gpu = olmoe.gpu_experts.?(.{ .experts = job_experts[0..used], .first = first[0 .. used + 1], .src = src, .inv = inv, .w = w, .x = x, .n = n, .k = k, .y = got });
    if (!on_gpu)
    {
        return null;
    }
    const d = set[0].down.rows;
    const out = try gpa.alloc(f32, d);
    defer gpa.free(out);
    const big = try gpa.alloc(f32, 3 * set[0].gate.rows + @max(set[0].gate.cols, set[0].gate.rows));
    defer gpa.free(big);
    _ = scratch;
    const saved = olmoe.gpu_matmul;
    olmoe.gpu_matmul = null;
    defer olmoe.gpu_matmul = saved;
    @memset(want, 0);
    for (0..n) |t|
    {
        for (0..k) |j|
        {
            set[idx[t * k + j]].forwardCpu(x[t * d ..][0..d], out, big);
            for (want[t * d ..][0..d], out) |*o, v|
            {
                o.* += w[t * k + j] * v;
            }
        }
    }
    return compare(got, want);
}
