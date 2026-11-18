//! Quantized weight formats (TASK-08) and their dequantization to f32.
//! Block layouts match llama.cpp's ggml (and gguf-py, which the reference
//! harness uses to check these): values per block, bytes per block, and the
//! order of scales and packed values inside a block.
//!
//! The 4-bit formats store E2M1 values (sign, 2 exponent bits, 1 mantissa
//! bit). Like ggml, the code table holds them doubled, and the decoded block
//! scale is halved to match.
//!
//! The formats added in TASK-36 (Q4_1 to IQ4_XS) are ported from ggml's reference
//! dequantization (MIT, see THIRD_PARTY.md), with the same order of float operations.
const std = @import("std");
const iq = @import("iq_tables.zig");

pub const Format = enum {
    f32,
    f16,
    bf16,
    /// 32 values: f16 scale, then 32 int8.
    q8_0,
    /// 32 values: f16 scale, then 16 bytes of 4-bit values offset by 8; low nibbles
    /// are values 0 to 15, high nibbles 16 to 31.
    q4_0,
    /// 32 values (OCP MX): E8M0 power-of-two scale, then 16 bytes of E2M1;
    /// low nibbles are values 0 to 15, high nibbles 16 to 31.
    mxfp4,
    /// 64 values (ggml NVFP4): 4 unsigned E4M3 scales, one per 16 values,
    /// then 4 groups of 8 bytes; in each group, low nibbles are its first 8
    /// values and high nibbles its last 8. A tensor-wide f32 scale, if any,
    /// is applied separately.
    nvfp4,
    /// 256 values (ggml Q4_K, TASK-32): f16 scale and f16 min, 12 bytes packing eight 6-bit
    /// scales and mins (one pair per 32 values), then 128 bytes of 4-bit values; each 32 bytes
    /// hold 64 values, low nibbles first.
    q4_k,
    /// 32 values: f16 scale, f16 min, then 16 bytes of 4-bit values.
    q4_1,
    /// 32 values: f16 scale, 4 bytes of fifth bits, then 16 bytes of low nibbles; offset by 16.
    q5_0,
    /// 32 values: f16 scale, f16 min, 4 bytes of fifth bits, then 16 bytes of low nibbles.
    q5_1,
    /// 256 values: 16 bytes of 4-bit scale and min pairs, 64 bytes of 2-bit values, f16 scale and min.
    q2_k,
    /// 256 values: 32 bytes of high bits, 64 bytes of 2-bit values, 12 bytes of 6-bit scales, f16 scale.
    q3_k,
    /// 256 values: like Q4_K plus 32 bytes of fifth bits before the nibbles.
    q5_k,
    /// 256 values: 128 bytes of low nibbles, 64 bytes of 2-bit highs, 16 int8 scales, f16 scale.
    q6_k,
    /// 256 values: f16 scale, then per 32 values 8 bytes of grid indices, signs and a 4-bit scale.
    iq2_xxs,
    /// 256 values: f16 scale, 32 u16 grid indices with signs, then 8 bytes of 4-bit scales.
    iq2_xs,
    /// 256 values: f16 scale, 64 bytes of grid indices, 8 of their high bits, 32 of signs, 8 of scales.
    iq2_s,
    /// 256 values: f16 scale, 64 bytes of grid indices, then 32 bytes of signs and scales.
    iq3_xxs,
    /// 256 values: f16 scale, 64 bytes of indices, 8 of high bits, 32 of signs, 4 of scales.
    iq3_s,
    /// 256 values: f16 scale, 32 bytes of grid indices, 8 u16 of high index bits, scales and delta signs.
    iq1_s,
    /// 256 values: 32 bytes of grid indices, 16 of high bits and delta signs, 8 of scales holding the f16 scale.
    iq1_m,
    /// 32 values: f16 scale, then 16 bytes of 4-bit indices into a non-linear table.
    iq4_nl,
    /// 256 values: f16 scale, u16 of high scale bits, 4 bytes of low scale bits, then 128 bytes as IQ4_NL.
    iq4_xs,

    pub fn blockValues(self: Format) usize {
        return switch (self) {
            .f32, .f16, .bf16 => 1,
            .q8_0, .q4_0, .mxfp4, .q4_1, .q5_0, .q5_1, .iq4_nl => 32,
            .nvfp4 => 64,
            .q4_k, .q2_k, .q3_k, .q5_k, .q6_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m, .iq4_xs => 256,
        };
    }

    pub fn blockBytes(self: Format) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q8_0 => 34,
            .q4_0 => 18,
            .mxfp4 => 17,
            .nvfp4 => 36,
            .q4_k => 144,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q2_k => 84,
            .q3_k => 110,
            .q5_k => 176,
            .q6_k => 210,
            .iq2_xxs => 66,
            .iq2_xs => 74,
            .iq2_s => 82,
            .iq3_xxs => 98,
            .iq3_s => 110,
            .iq1_s => 50,
            .iq1_m => 56,
            .iq4_nl => 18,
            .iq4_xs => 136,
        };
    }

    /// The format's ggml type id, as GGUF stores it.
    pub fn ggmlType(self: Format) u32
    {
        return switch (self)
        {
            .f32 => 0,
            .f16 => 1,
            .q4_0 => 2,
            .q4_1 => 3,
            .q5_0 => 6,
            .q5_1 => 7,
            .q8_0 => 8,
            .q2_k => 10,
            .q3_k => 11,
            .q4_k => 12,
            .q5_k => 13,
            .q6_k => 14,
            .iq2_xxs => 16,
            .iq2_xs => 17,
            .iq3_xxs => 18,
            .iq1_s => 19,
            .iq4_nl => 20,
            .iq3_s => 21,
            .iq2_s => 22,
            .iq4_xs => 23,
            .iq1_m => 29,
            .bf16 => 30,
            .mxfp4 => 39,
            .nvfp4 => 40,
        };
    }

    /// Bytes for `n` values, or an error if `n` is not a whole number of blocks.
    pub fn bytesFor(self: Format, n: u64) error{PartialBlock}!u64 {
        if (n % self.blockValues() != 0) return error.PartialBlock;
        return n / self.blockValues() * self.blockBytes();
    }
};

/// E2M1 code to value, doubled (ggml's kvalues_mxfp4).
const e2m1_doubled = [16]f32{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

pub fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// E8M0 scale 2^(e - 127), halved (ggml_e8m0_to_fp32_half). e = 255 is NaN in
/// the MX spec; it decodes to a huge value here, as in ggml.
pub fn e8m0Half(e: u8) f32 {
    const bits: u32 = if (e < 2) @as(u32, 0x00200000) << @intCast(e) else @as(u32, e - 1) << 23;
    return @bitCast(bits);
}

/// Unsigned E4M3 (bias 7) scale, halved. 0x7F (NaN) decodes to 0, as in ggml.
pub fn ue4m3Half(x: u8) f32 {
    if (x == 0 or x == 0x7F) return 0;
    const exp: i32 = (x >> 3) & 0xF;
    const man: f32 = @floatFromInt(x & 0x7);
    const v = if (exp == 0) man * 0x1p-9 else (1.0 + man / 8.0) * std.math.pow(f32, 2.0, @floatFromInt(exp - 7));
    return v * 0.5;
}

/// Signed OCP E4M3 ("e4m3fn": bias 7, no infinities; 0x7F and 0xFF are NaN).
pub fn e4m3ToF32(x: u8) f32 {
    if (x & 0x7F == 0x7F) return std.math.nan(f32);
    const mag = ue4m3Half(x & 0x7F) * 2.0;
    return if (x & 0x80 != 0) -mag else mag;
}

/// NVIDIA ModelOpt NVFP4 weight (the safetensors export): `values` holds
/// rows x cols/2 bytes, two E2M1 values per byte with the low nibble first;
/// `scales` holds rows x cols/16 E4M3 block scales; `global` is the tensor's
/// f32 `weight_scale_2`. Value = e2m1 x block scale x global.
pub fn dequantizeModelOptNvfp4(values: []const u8, scales: []const u8, global: f32, rows: usize, cols: usize, out: []f32) error{SizeMismatch}!void {
    if (cols % 16 != 0 or values.len != rows * cols / 2 or scales.len != rows * cols / 16 or out.len != rows * cols)
        return error.SizeMismatch;
    for (0..rows * cols / 16) |g| {
        const d = e4m3ToF32(scales[g]) * global * 0.5; // halved: the code table is doubled
        const o = out[g * 16 ..][0..16];
        for (values[g * 8 ..][0..8], 0..) |q, j| {
            o[2 * j] = d * e2m1_doubled[q & 0xF];
            o[2 * j + 1] = d * e2m1_doubled[q >> 4];
        }
    }
}

/// Dequantizes `out.len` values from `src`, which must hold exactly
/// `fmt.bytesFor(out.len)` bytes. `src` need not be aligned.
pub fn dequantize(fmt: Format, src: []const u8, out: []f32) error{ PartialBlock, SizeMismatch }!void {
    if (src.len != try fmt.bytesFor(out.len)) return error.SizeMismatch;
    switch (fmt) {
        .f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, src[i * 4 ..][0..4], .little));
        },
        .f16 => for (out, 0..) |*o, i| {
            o.* = f16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
        },
        .bf16 => for (out, 0..) |*o, i| {
            o.* = bf16ToF32(std.mem.readInt(u16, src[i * 2 ..][0..2], .little));
        },
        .q8_0 => {
            var b: usize = 0;
            while (b < out.len / 32) : (b += 1) {
                const blk = src[b * 34 ..][0..34];
                const d = f16ToF32(std.mem.readInt(u16, blk[0..2], .little));
                for (out[b * 32 ..][0..32], blk[2..34]) |*o, q| o.* = d * @as(f32, @floatFromInt(@as(i8, @bitCast(q))));
            }
        },
        .q4_0 => {
            var b: usize = 0;
            while (b < out.len / 32) : (b += 1)
            {
                const blk = src[b * 18 ..][0..18];
                const d = f16ToF32(std.mem.readInt(u16, blk[0..2], .little));
                const o = out[b * 32 ..][0..32];
                for (blk[2..18], 0..) |q, j|
                {
                    o[j] = d * @as(f32, @floatFromInt(@as(i32, q & 0xF) - 8));
                    o[j + 16] = d * @as(f32, @floatFromInt(@as(i32, q >> 4) - 8));
                }
            }
        },
        .mxfp4 => {
            var b: usize = 0;
            while (b < out.len / 32) : (b += 1) {
                const blk = src[b * 17 ..][0..17];
                const d = e8m0Half(blk[0]);
                const o = out[b * 32 ..][0..32];
                for (blk[1..17], 0..) |q, j| {
                    o[j] = d * e2m1_doubled[q & 0xF];
                    o[j + 16] = d * e2m1_doubled[q >> 4];
                }
            }
        },
        .nvfp4 => {
            var b: usize = 0;
            while (b < out.len / 64) : (b += 1) {
                const blk = src[b * 36 ..][0..36];
                for (0..4) |g| {
                    const d = ue4m3Half(blk[g]);
                    const o = out[b * 64 + g * 16 ..][0..16];
                    for (blk[4 + g * 8 ..][0..8], 0..) |q, j| {
                        o[j] = d * e2m1_doubled[q & 0xF];
                        o[j + 8] = d * e2m1_doubled[q >> 4];
                    }
                }
            }
        },
        .q4_k => {
            var b: usize = 0;
            while (b < out.len / 256) : (b += 1)
            {
                const blk = src[b * 144 ..][0..144];
                const d = f16ToF32(std.mem.readInt(u16, blk[0..2], .little));
                const dmin = f16ToF32(std.mem.readInt(u16, blk[2..4], .little));
                const scales = blk[4..16];
                for (0..4) |c|
                {
                    const lo = scaleMinK4(scales, 2 * c);
                    const hi = scaleMinK4(scales, 2 * c + 1);
                    const d1 = d * @as(f32, @floatFromInt(lo[0]));
                    const m1 = dmin * @as(f32, @floatFromInt(lo[1]));
                    const d2 = d * @as(f32, @floatFromInt(hi[0]));
                    const m2 = dmin * @as(f32, @floatFromInt(hi[1]));
                    const o = out[b * 256 + c * 64 ..][0..64];
                    for (blk[16 + c * 32 ..][0..32], 0..) |q, l|
                    {
                        o[l] = d1 * @as(f32, @floatFromInt(q & 0xF)) - m1;
                        o[l + 32] = d2 * @as(f32, @floatFromInt(q >> 4)) - m2;
                    }
                }
            }
        },
        else => {
            const bv = fmt.blockValues();
            const bb = fmt.blockBytes();
            for (0..out.len / bv) |b|
            {
                decodeBlock(fmt, src[b * bb ..][0..bb], out[b * bv ..][0..bv]);
            }
        },
    }
}

fn half(b: []const u8, at: usize) f32
{
    return f16ToF32(std.mem.readInt(u16, b[at..][0..2], .little));
}

fn word(b: []const u8, at: usize) u32
{
    return std.mem.readInt(u32, b[at..][0..4], .little);
}

fn short(b: []const u8, at: usize) u16
{
    return std.mem.readInt(u16, b[at..][0..2], .little);
}

fn fl(x: anytype) f32
{
    return @floatFromInt(x);
}

/// Byte `j` of a grid entry, as ggml reads it through a byte pointer.
fn gridByte(g: anytype, j: usize) u8
{
    return @truncate(g >> @intCast(8 * j));
}

/// -1 where bit `j` of `signs` is set, else 1.
fn sign(signs: u8, j: usize) f32
{
    return if (signs & (@as(u8, 1) << @intCast(j)) != 0) -1.0 else 1.0;
}

const iq1s_delta: f32 = 0.125;

/// One block of a TASK-36 format, as ggml's dequantize_row_* writes it.
fn decodeBlock(fmt: Format, blk: []const u8, y: []f32) void
{
    switch (fmt)
    {
        .q4_1 => {
            const d = half(blk, 0);
            const m = half(blk, 2);
            for (blk[4..20], 0..) |q, j|
            {
                y[j] = fl(q & 0xF) * d + m;
                y[j + 16] = fl(q >> 4) * d + m;
            }
        },
        .q5_0, .q5_1 => {
            const with_min = fmt == .q5_1;
            const d = half(blk, 0);
            const m = if (with_min) half(blk, 2) else 0;
            const at: usize = if (with_min) 4 else 2;
            const qh = word(blk, at);
            for (blk[at + 4 ..][0..16], 0..) |q, j|
            {
                const h0: u32 = ((qh >> @intCast(j)) << 4) & 0x10;
                const h1: u32 = (qh >> @intCast(j + 12)) & 0x10;
                const x0: i32 = @intCast((q & 0xF) | h0);
                const x1: i32 = @intCast((q >> 4) | h1);
                if (with_min)
                {
                    y[j] = fl(x0) * d + m;
                    y[j + 16] = fl(x1) * d + m;
                }
                else
                {
                    y[j] = fl(x0 - 16) * d;
                    y[j + 16] = fl(x1 - 16) * d;
                }
            }
        },
        .q2_k => {
            const scales = blk[0..16];
            const d = half(blk, 80);
            const min = half(blk, 82);
            var o: usize = 0;
            var is: usize = 0;
            for (0..2) |n|
            {
                const q = blk[16 + n * 32 ..][0..32];
                for (0..4) |j|
                {
                    const shift: u3 = @intCast(2 * j);
                    for (0..2) |half_|
                    {
                        const sc = scales[is];
                        is += 1;
                        const dl = d * fl(sc & 0xF);
                        const ml = min * fl(sc >> 4);
                        for (0..16) |l|
                        {
                            y[o] = dl * fl((q[l + 16 * half_] >> shift) & 3) - ml;
                            o += 1;
                        }
                    }
                }
            }
        },
        .q3_k => {
            const hm = blk[0..32];
            const d_all = half(blk, 108);
            var aux = [4]u32{ word(blk, 96), word(blk, 100), word(blk, 104), 0 };
            const tmp = aux[2];
            const kmask1: u32 = 0x03030303;
            const kmask2: u32 = 0x0f0f0f0f;
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            var scales: [16]u8 = undefined;
            for (0..4) |w|
            {
                std.mem.writeInt(u32, scales[w * 4 ..][0..4], aux[w], .little);
            }
            var o: usize = 0;
            var is: usize = 0;
            var m: u8 = 1;
            for (0..2) |n|
            {
                const q = blk[32 + n * 32 ..][0..32];
                for (0..4) |j|
                {
                    const shift: u3 = @intCast(2 * j);
                    for (0..2) |half_|
                    {
                        const dl = d_all * fl(@as(i32, @as(i8, @bitCast(scales[is]))) - 32);
                        is += 1;
                        for (0..16) |l|
                        {
                            const i = l + 16 * half_;
                            const low: i32 = (q[i] >> shift) & 3;
                            const v: i32 = low - @as(i32, if (hm[i] & m != 0) 0 else 4);
                            y[o] = dl * fl(v);
                            o += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        },
        .q5_k => {
            const d = half(blk, 0);
            const min = half(blk, 2);
            const scales = blk[4..16];
            const qh = blk[16..48];
            var bit_lo: u8 = 1;
            var bit_hi: u8 = 2;
            for (0..4) |c|
            {
                const ql = blk[48 + c * 32 ..][0..32];
                const lo = scaleMinK4(scales, 2 * c);
                const hi = scaleMinK4(scales, 2 * c + 1);
                const d1 = d * fl(lo[0]);
                const m1 = min * fl(lo[1]);
                const d2 = d * fl(hi[0]);
                const m2 = min * fl(hi[1]);
                const o = y[c * 64 ..][0..64];
                for (0..32) |l|
                {
                    const a: u32 = @as(u32, ql[l] & 0xF) + @as(u32, if (qh[l] & bit_lo != 0) 16 else 0);
                    const b: u32 = @as(u32, ql[l] >> 4) + @as(u32, if (qh[l] & bit_hi != 0) 16 else 0);
                    o[l] = d1 * fl(a) - m1;
                    o[l + 32] = d2 * fl(b) - m2;
                }
                bit_lo <<= 2;
                bit_hi <<= 2;
            }
        },
        .q6_k => {
            const d = half(blk, 208);
            for (0..2) |n|
            {
                const ql = blk[n * 64 ..][0..64];
                const qh = blk[128 + n * 32 ..][0..32];
                const sc = blk[192 + n * 8 ..][0..8];
                const o = y[n * 128 ..][0..128];
                for (0..32) |l|
                {
                    const is = l / 16;
                    const q1: i32 = @as(i32, (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const q2: i32 = @as(i32, (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const q3: i32 = @as(i32, (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const q4: i32 = @as(i32, (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[l] = d * fl(@as(i8, @bitCast(sc[is]))) * fl(q1);
                    o[l + 32] = d * fl(@as(i8, @bitCast(sc[is + 2]))) * fl(q2);
                    o[l + 64] = d * fl(@as(i8, @bitCast(sc[is + 4]))) * fl(q3);
                    o[l + 96] = d * fl(@as(i8, @bitCast(sc[is + 6]))) * fl(q4);
                }
            }
        },
        .iq2_xxs => {
            const d = half(blk, 0);
            for (0..8) |ib|
            {
                const at = 2 + 8 * ib;
                const aux1 = word(blk, at + 4);
                const db = d * (0.5 + fl(aux1 >> 28)) * 0.25;
                for (0..4) |l|
                {
                    const g = iq.iq2xxs_grid[blk[at + l]];
                    const signs = iq.ksigns_iq2xs[(aux1 >> @intCast(7 * l)) & 127];
                    for (0..8) |j|
                    {
                        y[ib * 32 + l * 8 + j] = db * fl(gridByte(g, j)) * sign(signs, j);
                    }
                }
            }
        },
        .iq2_xs => {
            const d = half(blk, 0);
            for (0..8) |ib|
            {
                const sc = blk[66 + ib];
                const db = [2]f32{ d * (0.5 + fl(sc & 0xF)) * 0.25, d * (0.5 + fl(sc >> 4)) * 0.25 };
                for (0..4) |l|
                {
                    const q = short(blk, 2 + 2 * (4 * ib + l));
                    const g = iq.iq2xs_grid[q & 511];
                    const signs = iq.ksigns_iq2xs[q >> 9];
                    for (0..8) |j|
                    {
                        y[ib * 32 + l * 8 + j] = db[l / 2] * fl(gridByte(g, j)) * sign(signs, j);
                    }
                }
            }
        },
        .iq2_s => {
            const d = half(blk, 0);
            for (0..8) |ib|
            {
                const sc = blk[74 + ib];
                const qh: u32 = blk[66 + ib];
                const db = [2]f32{ d * (0.5 + fl(sc & 0xF)) * 0.25, d * (0.5 + fl(sc >> 4)) * 0.25 };
                for (0..4) |l|
                {
                    const idx = @as(u32, blk[2 + 4 * ib + l]) | ((qh << @intCast(8 - 2 * l)) & 0x300);
                    const g = iq.iq2s_grid[idx];
                    const signs = blk[34 + 4 * ib + l];
                    for (0..8) |j|
                    {
                        y[ib * 32 + l * 8 + j] = db[l / 2] * fl(gridByte(g, j)) * sign(signs, j);
                    }
                }
            }
        },
        .iq3_xxs => {
            const d = half(blk, 0);
            for (0..8) |ib|
            {
                const aux = word(blk, 66 + 4 * ib);
                const db = d * (0.5 + fl(aux >> 28)) * 0.5;
                for (0..4) |l|
                {
                    const signs = iq.ksigns_iq2xs[(aux >> @intCast(7 * l)) & 127];
                    const g1 = iq.iq3xxs_grid[blk[2 + 8 * ib + 2 * l]];
                    const g2 = iq.iq3xxs_grid[blk[2 + 8 * ib + 2 * l + 1]];
                    const o = y[ib * 32 + l * 8 ..][0..8];
                    for (0..4) |j|
                    {
                        o[j] = db * fl(gridByte(g1, j)) * sign(signs, j);
                        o[j + 4] = db * fl(gridByte(g2, j)) * sign(signs, j + 4);
                    }
                }
            }
        },
        .iq3_s => {
            const d = half(blk, 0);
            for (0..4) |p|
            {
                const sc = blk[106 + p];
                const dbs = [2]f32{ d * fl(1 + 2 * @as(u32, sc & 0xF)), d * fl(1 + 2 * @as(u32, sc >> 4)) };
                for (0..2) |s|
                {
                    const ib = 2 * p + s;
                    const qh: u32 = blk[66 + ib];
                    for (0..4) |l|
                    {
                        const q1 = @as(u32, blk[2 + 8 * ib + 2 * l]) | ((qh << @intCast(8 - 2 * l)) & 256);
                        const q2 = @as(u32, blk[2 + 8 * ib + 2 * l + 1]) | ((qh << @intCast(7 - 2 * l)) & 256);
                        const g1 = iq.iq3s_grid[q1];
                        const g2 = iq.iq3s_grid[q2];
                        const signs = blk[74 + 4 * ib + l];
                        const o = y[ib * 32 + l * 8 ..][0..8];
                        for (0..4) |j|
                        {
                            o[j] = dbs[s] * fl(gridByte(g1, j)) * sign(signs, j);
                            o[j + 4] = dbs[s] * fl(gridByte(g2, j)) * sign(signs, j + 4);
                        }
                    }
                }
            }
        },
        .iq1_s => {
            const d = half(blk, 0);
            for (0..8) |ib|
            {
                const qh: u32 = short(blk, 34 + 2 * ib);
                const dl = d * fl(2 * ((qh >> 12) & 7) + 1);
                const delta: f32 = if (qh & 0x8000 != 0) -iq1s_delta else iq1s_delta;
                for (0..4) |l|
                {
                    const idx = @as(u32, blk[2 + 4 * ib + l]) | (((qh >> @intCast(3 * l)) & 7) << 8);
                    const g = iq.iq1s_grid[idx];
                    for (0..8) |j|
                    {
                        y[ib * 32 + l * 8 + j] = dl * (fl(@as(i8, @bitCast(gridByte(g, j)))) + delta);
                    }
                }
            }
        },
        .iq1_m => {
            var sc: [4]u16 = undefined;
            for (&sc, 0..) |*s, i|
            {
                s.* = short(blk, 48 + 2 * i);
            }
            const scale: u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const d = f16ToF32(scale);
            for (0..8) |ib|
            {
                const s: u32 = sc[ib / 2];
                const shift: u5 = @intCast(6 * (ib % 2));
                const dl = [2]f32{ d * fl(2 * ((s >> shift) & 7) + 1), d * fl(2 * ((s >> (shift + 3)) & 7) + 1) };
                const qh = [2]u32{ blk[32 + 2 * ib], blk[32 + 2 * ib + 1] };
                const qs = blk[4 * ib ..][0..4];
                const idx = [4]u32{
                    qs[0] | ((qh[0] << 8) & 0x700),
                    qs[1] | ((qh[0] << 4) & 0x700),
                    qs[2] | ((qh[1] << 8) & 0x700),
                    qs[3] | ((qh[1] << 4) & 0x700),
                };
                const delta = [4]f32{
                    if (qh[0] & 0x08 != 0) -iq1s_delta else iq1s_delta,
                    if (qh[0] & 0x80 != 0) -iq1s_delta else iq1s_delta,
                    if (qh[1] & 0x08 != 0) -iq1s_delta else iq1s_delta,
                    if (qh[1] & 0x80 != 0) -iq1s_delta else iq1s_delta,
                };
                for (0..4) |l|
                {
                    const g = iq.iq1s_grid[idx[l]];
                    for (0..8) |j|
                    {
                        y[ib * 32 + l * 8 + j] = dl[l / 2] * (fl(@as(i8, @bitCast(gridByte(g, j)))) + delta[l]);
                    }
                }
            }
        },
        .iq4_nl => {
            const d = half(blk, 0);
            for (blk[2..18], 0..) |q, j|
            {
                y[j] = d * fl(iq.kvalues_iq4nl[q & 0xF]);
                y[j + 16] = d * fl(iq.kvalues_iq4nl[q >> 4]);
            }
        },
        .iq4_xs => {
            const d = half(blk, 0);
            const sh: u32 = short(blk, 2);
            for (0..8) |ib|
            {
                const low: u32 = (blk[4 + ib / 2] >> @intCast(4 * (ib % 2))) & 0xF;
                const ls: i32 = @intCast(low | (((sh >> @intCast(2 * ib)) & 3) << 4));
                const dl = d * fl(ls - 32);
                for (blk[8 + 16 * ib ..][0..16], 0..) |q, j|
                {
                    y[ib * 32 + j] = dl * fl(iq.kvalues_iq4nl[q & 0xF]);
                    y[ib * 32 + j + 16] = dl * fl(iq.kvalues_iq4nl[q >> 4]);
                }
            }
        },
        else => unreachable,
    }
}

/// Q4_K's j-th 6-bit scale and min from its 12 packed bytes (ggml's get_scale_min_k4).
pub fn scaleMinK4(q: []const u8, j: usize) [2]u8
{
    if (j < 4)
    {
        return .{ q[j] & 63, q[j + 4] & 63 };
    }
    return .{ (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4) };
}

test "q4_k block decodes scales, mins and nibbles in ggml order" {
    var blk: [144]u8 = @splat(0);
    std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, 0.5)), .little);
    std.mem.writeInt(u16, blk[2..4], @bitCast(@as(f16, 0.25)), .little);
    blk[4] = 2; // sub-block 0: scale 2
    blk[8] = 4; // sub-block 0: min 4
    blk[12] = 0x3_1; // sub-block 4: scale 1, min 3
    blk[16] = 0x0_7; // value 0: 7, value 32: 0
    blk[16 + 64] = 0x0_5; // value 128 (sub-block 4): 5
    var out: [256]f32 = undefined;
    try dequantize(.q4_k, &blk, &out);
    try std.testing.expectEqual(@as(f32, 0.5 * 2 * 7 - 0.25 * 4), out[0]);
    try std.testing.expectEqual(@as(f32, 0), out[32]);
    try std.testing.expectEqual(@as(f32, 0.5 * 1 * 5 - 0.25 * 3), out[128]);
    try std.testing.expectEqual(@as(u64, 144 * 2), try Format.q4_k.bytesFor(512));
}

test "format sizes match ggml" {
    try std.testing.expectEqual(@as(u64, 34 * 4), try Format.q8_0.bytesFor(128));
    try std.testing.expectEqual(@as(u64, 17 * 4), try Format.mxfp4.bytesFor(128));
    try std.testing.expectEqual(@as(u64, 36 * 2), try Format.nvfp4.bytesFor(128));
    try std.testing.expectError(error.PartialBlock, Format.nvfp4.bytesFor(96));
}

test "scale decoders" {
    try std.testing.expectEqual(@as(f32, 0.5), e8m0Half(127)); // 2^0, halved
    try std.testing.expectEqual(@as(f32, 4.0), e8m0Half(130));
    try std.testing.expectEqual(@as(f32, 0x1p-128), e8m0Half(0));
    try std.testing.expectEqual(@as(f32, 0.5), ue4m3Half(0x38)); // exp 7, man 0: 1.0
    try std.testing.expectEqual(@as(f32, 224.0), ue4m3Half(0x7E)); // 448, the largest
    try std.testing.expectEqual(@as(f32, 0x1p-10), ue4m3Half(0x01)); // smallest subnormal 2^-9
    try std.testing.expectEqual(@as(f32, 0), ue4m3Half(0x7F));
}

test "q4_0 block decodes nibbles in ggml order" {
    var blk: [18]u8 = @splat(0x88); // every value 0
    std.mem.writeInt(u16, blk[0..2], @bitCast(@as(f16, 0.5)), .little);
    blk[2] = 0x0_F; // value 0: 15 - 8 = 7, value 16: 0 - 8 = -8
    var out: [32]f32 = undefined;
    try dequantize(.q4_0, &blk, &out);
    try std.testing.expectEqual(@as(f32, 3.5), out[0]);
    try std.testing.expectEqual(@as(f32, -4), out[16]);
    try std.testing.expectEqual(@as(f32, 0), out[1]);
    try std.testing.expectEqual(@as(u64, 18 * 4), try Format.q4_0.bytesFor(128));
}

test "mxfp4 block decodes nibbles in ggml order" {
    var blk: [17]u8 = @splat(0);
    blk[0] = 128; // scale 2, halved to 1, so values are the doubled codes
    blk[1] = 0x9_2; // value 0: code 2 (2.0 doubled), value 16: code 9 (-1 doubled)
    var out: [32]f32 = undefined;
    try dequantize(.mxfp4, &blk, &out);
    try std.testing.expectEqual(@as(f32, 2), out[0]);
    try std.testing.expectEqual(@as(f32, -1), out[16]);
    try std.testing.expectEqual(@as(f32, 0), out[1]);
}
