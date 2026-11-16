//! How a KV cache stores each position's keys and values (TASK-27). One row
//! holds every KV head of one position:
//! - `f32`: kvd floats, as computed.
//! - `bf16`: kvd bfloat16 values, rounded to nearest even.
//! - `fp8`: kvd E4M3 bytes, then one byte per KV head (padded to 16) giving
//!   that head's power-of-two scale, so a value decodes as its E4M3 value
//!   times the scale, exactly, on the CPU and the GPU alike.
const std = @import("std");

pub const KvFormat = enum(u8) {
    f32 = 0,
    bf16 = 1,
    fp8 = 2,

    pub fn parse(s: []const u8) ?KvFormat
    {
        return std.meta.stringToEnum(KvFormat, s);
    }

    /// Bytes one position takes, keys or values alone.
    pub fn rowBytes(self: KvFormat, kvd: usize, kv_heads: usize) usize
    {
        return switch (self)
        {
            .f32 => kvd * 4,
            .bf16 => kvd * 2,
            .fp8 => kvd + scaleBytes(kv_heads),
        };
    }

    /// Writes one position's `kvd` values into `row`.
    pub fn encodeRow(self: KvFormat, src: []const f32, row: []u8, kv_heads: usize) void
    {
        switch (self)
        {
            .f32 => @memcpy(row[0 .. src.len * 4], std.mem.sliceAsBytes(src)),
            .bf16 =>
            {
                for (src, 0..) |x, i|
                {
                    std.mem.writeInt(u16, row[2 * i ..][0..2], toBf16(x), .little);
                }
            },
            .fp8 =>
            {
                const hd = src.len / kv_heads;
                const scales = row[src.len..][0..scaleBytes(kv_heads)];
                @memset(scales, 0);
                for (0..kv_heads) |h|
                {
                    const head = src[h * hd ..][0..hd];
                    var amax: f32 = 0;
                    for (head) |x|
                    {
                        amax = @max(amax, @abs(x));
                    }
                    const e = scaleExponent(amax);
                    scales[h] = @intCast(e + scale_bias);
                    const inv = std.math.ldexp(@as(f32, 1), -e);
                    for (head, 0..) |x, i|
                    {
                        row[h * hd + i] = toE4m3(x * inv);
                    }
                }
            },
        }
    }

    /// Decodes head `h` (`out.len` values) of a row holding `kvd` values.
    pub fn decodeHead(self: KvFormat, row: []const u8, kvd: usize, h: usize, out: []f32) void
    {
        const hd = out.len;
        switch (self)
        {
            .f32 =>
            {
                for (out, 0..) |*o, i|
                {
                    o.* = @bitCast(std.mem.readInt(u32, row[(h * hd + i) * 4 ..][0..4], .little));
                }
            },
            .bf16 =>
            {
                for (out, 0..) |*o, i|
                {
                    o.* = fromBf16(std.mem.readInt(u16, row[(h * hd + i) * 2 ..][0..2], .little));
                }
            },
            .fp8 =>
            {
                const scale = scaleOf(row[kvd + h]);
                for (out, row[h * hd ..][0..hd]) |*o, b|
                {
                    o.* = e4m3_table[b] * scale;
                }
            },
        }
    }
};

/// The scale bytes after an FP8 row, padded so rows stay 16-byte aligned.
pub fn scaleBytes(kv_heads: usize) usize
{
    return (kv_heads + 15) / 16 * 16;
}

const scale_bias = 128;
const e4m3_max: f32 = 448;

/// The smallest power of two that brings `amax` within E4M3's range; 0 for an all-zero head.
fn scaleExponent(amax: f32) i32
{
    if (!(amax >= 1e-30) or !std.math.isFinite(amax))
    {
        return 0;
    }
    var e: i32 = std.math.ilogb(amax / e4m3_max) + 1;
    // log2 may round either way near a power of two; settle it exactly.
    while (amax * std.math.ldexp(@as(f32, 1), -e) > e4m3_max)
    {
        e += 1;
    }
    while (amax * std.math.ldexp(@as(f32, 1), -(e - 1)) <= e4m3_max)
    {
        e -= 1;
    }
    return std.math.clamp(e, -100, 100);
}

/// Byte 0 is never written: rows not yet stored decode to zeros, on the GPU as here.
fn scaleOf(b: u8) f32
{
    if (b == 0)
    {
        return 0;
    }
    return std.math.ldexp(@as(f32, 1), @as(i32, b) - scale_bias);
}

pub fn toBf16(x: f32) u16
{
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x))
    {
        return @intCast((u >> 16) | 0x40);
    }
    const round = 0x7fff + ((u >> 16) & 1);
    return @intCast((u +% round) >> 16);
}

pub fn fromBf16(b: u16) f32
{
    return @bitCast(@as(u32, b) << 16);
}

/// E4M3 (1 sign, 4 exponent with bias 7, 3 mantissa bits; no infinities), rounded to
/// nearest even and saturated at 448. Inputs are already scaled into range.
pub fn toE4m3(x: f32) u8
{
    const u: u32 = @bitCast(x);
    const sign: u8 = @intCast((u >> 24) & 0x80);
    const a = u & 0x7fffffff;
    if (a >= 0x43e00000)
    {
        // 448 and up, infinities and NaN.
        return sign | 0x7e;
    }
    const exp = a >> 23;
    if (exp < 121)
    {
        // Below 2^-6 the step is 2^-9 (subnormals).
        const f: f32 = @bitCast(a);
        return sign | @as(u8, @intFromFloat(roundEven(f * 512)));
    }
    // Keep 3 mantissa bits, rounding the rest to nearest even; a carry moves into the exponent.
    const mant = a & 0x7fffff;
    var code = ((exp - 120) << 3) | (mant >> 20);
    const rest = mant & 0xfffff;
    if (rest > 0x80000 or (rest == 0x80000 and code & 1 == 1))
    {
        code += 1;
    }
    return sign | @as(u8, @intCast(@min(code, 0x7e)));
}

fn roundEven(x: f32) f32
{
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5 or (d == 0.5 and @mod(f, 2) == 1))
    {
        return f + 1;
    }
    return f;
}

pub fn fromE4m3(b: u8) f32
{
    const e: i32 = (b >> 3) & 15;
    const m: f32 = @floatFromInt(b & 7);
    const mag = if (e == 0) m * std.math.ldexp(@as(f32, 1), -9) else (1 + m / 8) * std.math.ldexp(@as(f32, 1), e - 7);
    return if (b & 0x80 != 0) -mag else mag;
}

const e4m3_table: [256]f32 = blk: {
    @setEvalBranchQuota(10000);
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, i|
    {
        v.* = fromE4m3(@intCast(i));
    }
    break :blk t;
};

test "bf16 rounds to nearest even and round-trips its own values" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    // 1 + 2^-8 is halfway between 1 and the next bf16; even is 1.
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0 + std.math.ldexp(@as(f32, 1), -8)));
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(1.0 + 3 * std.math.ldexp(@as(f32, 1), -8)));
    for (0..65536) |i|
    {
        const b: u16 = @intCast(i);
        const x = fromBf16(b);
        if (!std.math.isNan(x))
        {
            try std.testing.expectEqual(b, toBf16(x));
        }
    }
}

test "e4m3 decodes the standard values and every code round-trips" {
    try std.testing.expectEqual(@as(f32, 448), fromE4m3(0x7e));
    try std.testing.expectEqual(@as(f32, 1), fromE4m3(0x38));
    try std.testing.expectEqual(std.math.ldexp(@as(f32, 1), -9), fromE4m3(0x01));
    try std.testing.expectEqual(std.math.ldexp(@as(f32, 1), -6), fromE4m3(0x08));
    for (0..256) |i|
    {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f)
        {
            continue;
        }
        try std.testing.expectEqual(if (b == 0x80) @as(u8, 0x80) else b, toE4m3(fromE4m3(b)));
    }
    // Halfway between 1 and 1.125 rounds to even (1); just above goes up.
    try std.testing.expectEqual(@as(u8, 0x38), toE4m3(1.0625));
    try std.testing.expectEqual(@as(u8, 0x39), toE4m3(1.0626));
    try std.testing.expectEqual(@as(u8, 0x7e), toE4m3(1e6));
    try std.testing.expectEqual(@as(u8, 0xfe), toE4m3(-500));
}

test "e4m3 picks the nearest value for random inputs" {
    var rng = std.Random.DefaultPrng.init(27);
    const r = rng.random();
    for (0..20000) |_|
    {
        const x = (r.float(f32) * 2 - 1) * 448 * std.math.ldexp(@as(f32, 1), -@as(i32, r.intRangeAtMost(u5, 0, 20)));
        const got = fromE4m3(toE4m3(x));
        var best: f32 = std.math.inf(f32);
        for (0..256) |i|
        {
            if (i & 0x7f != 0x7f)
            {
                best = @min(best, @abs(fromE4m3(@intCast(i)) - x));
            }
        }
        try std.testing.expectEqual(best, @abs(got - x));
    }
}

test "rows of each format decode within their precision, scales exactly" {
    const kvd = 2 * 128;
    var src: [kvd]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(5);
    for (&src, 0..) |*x, i|
    {
        x.* = (rng.random().floatNorm(f32)) * if (i < 128) @as(f32, 0.01) else @as(f32, 30);
    }
    src[7] = 0;
    for ([_]KvFormat{ .f32, .bf16, .fp8 }) |fmt|
    {
        var row: [kvd * 4]u8 = undefined;
        const rb = fmt.rowBytes(kvd, 2);
        try std.testing.expectEqual(@as(usize, 0), rb % 16);
        fmt.encodeRow(&src, row[0..rb], 2);
        for (0..2) |h|
        {
            var out: [128]f32 = undefined;
            fmt.decodeHead(row[0..rb], kvd, h, &out);
            var amax: f32 = 0;
            for (src[h * 128 ..][0..128]) |x|
            {
                amax = @max(amax, @abs(x));
            }
            for (out, src[h * 128 ..][0..128]) |o, x|
            {
                const tol: f32 = switch (fmt)
                {
                    .f32 => 0,
                    .bf16 => @abs(x) / 256,
                    .fp8 => @max(@abs(x) / 16, amax / 448 * std.math.ldexp(@as(f32, 1), -9)),
                };
                try std.testing.expect(@abs(o - x) <= tol);
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 2048 + 16), KvFormat.fp8.rowBytes(2048, 16));
    try std.testing.expectEqual(@as(usize, 512 + 16), KvFormat.fp8.rowBytes(512, 4));
}
