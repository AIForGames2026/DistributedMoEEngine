//! Quantized weight formats (TASK-08) and their dequantization to f32.
//! Block layouts match llama.cpp's ggml (and gguf-py, which the reference
//! harness uses to check these): values per block, bytes per block, and the
//! order of scales and packed values inside a block.
//!
//! The 4-bit formats store E2M1 values (sign, 2 exponent bits, 1 mantissa
//! bit). Like ggml, the code table holds them doubled, and the decoded block
//! scale is halved to match.
const std = @import("std");

pub const Format = enum {
    f32,
    f16,
    bf16,
    /// 32 values: f16 scale, then 32 int8.
    q8_0,
    /// 32 values (OCP MX): E8M0 power-of-two scale, then 16 bytes of E2M1;
    /// low nibbles are values 0 to 15, high nibbles 16 to 31.
    mxfp4,
    /// 64 values (ggml NVFP4): 4 unsigned E4M3 scales, one per 16 values,
    /// then 4 groups of 8 bytes; in each group, low nibbles are its first 8
    /// values and high nibbles its last 8. A tensor-wide f32 scale, if any,
    /// is applied separately.
    nvfp4,

    pub fn blockValues(self: Format) usize {
        return switch (self) {
            .f32, .f16, .bf16 => 1,
            .q8_0, .mxfp4 => 32,
            .nvfp4 => 64,
        };
    }

    pub fn blockBytes(self: Format) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q8_0 => 34,
            .mxfp4 => 17,
            .nvfp4 => 36,
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
    }
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
