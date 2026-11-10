//! Wire format v0 (TASK-06): the token packet and the 32-bit RDMA immediate.
//! docs/wire-format.md is the spec; include/tf_wire.h is the C mirror. Both
//! sides assert the same sizes and offsets at compile time.
const std = @import("std");

pub const version = 0;
pub const hidden_bytes = 7168;
/// One e8m0 scale per 32-element block.
pub const scale_bytes = hidden_bytes / 32;
pub const packet_bytes = 7552;

/// CRC32C (Castagnoli), the same CRC used by iSCSI and ext4 metadata.
/// Implemented in csrc/tf_crc32c.c with the CPU's CRC instructions.
extern fn tf_crc32c(data: [*]const u8, len: usize) u32;

pub fn crc32c(bytes: []const u8) u32 {
    return tf_crc32c(bytes.ptr, bytes.len);
}

pub const Kind = enum(u2) { dispatch = 0, combine = 1, credit = 2, control = 3 };

pub const Header = extern struct {
    token_id: u64,
    /// Full slot generation. Starts at 1 and increments every time the slot is reused.
    generation: u64,
    src_rank: u16,
    layer: u16,
    expert: u16,
    flags: u16,
    gate_weight: f32,
    /// Valid bytes in `payload`; models smaller than 7168 leave the rest zero.
    payload_len: u32,
};

pub const Tail = extern struct {
    /// CRC32C over every byte before the tail.
    crc32c: u32,
    /// Low 32 bits of `Header.generation`, written last so a poller can tell a
    /// fully placed packet from a stale one.
    generation_lo: u32,
    reserved: [8]u8,
};

pub const Packet = extern struct {
    header: Header,
    payload: [hidden_bytes]u8,
    scales: [scale_bytes]u8,
    pad: [112]u8,
    tail: Tail,

    pub fn computeCrc(self: *const Packet) u32 {
        const bytes: *const [packet_bytes]u8 = @ptrCast(self);
        return crc32c(bytes[0..@offsetOf(Packet, "tail")]);
    }

    /// Fills the tail. Call after the header, payload, and scales are final.
    pub fn seal(self: *Packet) void {
        self.tail = .{
            .crc32c = self.computeCrc(),
            .generation_lo = @truncate(self.header.generation),
            .reserved = @splat(0),
        };
    }

    pub const CheckError = error{ BadCrc, StaleGeneration };

    pub fn check(self: *const Packet, expected_generation: u64) CheckError!void {
        if (self.header.generation != expected_generation or
            self.tail.generation_lo != @as(u32, @truncate(expected_generation))) return error.StaleGeneration;
        if (self.tail.crc32c != self.computeCrc()) return error.BadCrc;
    }
};

/// The 32-bit immediate sent with RDMA WRITE_WITH_IMM (and carried in TCP framing).
/// Fields are listed from bit 0 upward.
pub const Imm = packed struct(u32) {
    /// Slot index in the receiver's ring.
    slot: u16,
    kind: Kind,
    /// Low 8 bits of the slot generation, for a check before touching the slot.
    gen8: u8,
    src_rank: u6,

    pub fn encode(self: Imm) u32 {
        return @bitCast(self);
    }

    pub fn decode(raw: u32) Imm {
        return @bitCast(raw);
    }
};

/// How `payload` holds the hidden state: bits 0 to 1 of `Header.flags`.
/// v0's packet layout is unchanged; flags 0 is the spec's FP8 encoding.
pub const Encoding = enum(u2) {
    /// FP8 E4M3 ("e4m3fn") values, one E8M0 power-of-two scale per 32 values
    /// in `scales` (OCP MX). Up to 7,168 values.
    fp8_e4m3 = 0,
    /// bfloat16, round to nearest even. Up to 3,584 values.
    bf16 = 1,
    /// float32, exact. Up to 1,792 values.
    f32 = 2,

    pub fn maxValues(self: Encoding) usize {
        return switch (self) {
            .fp8_e4m3 => hidden_bytes,
            .bf16 => hidden_bytes / 2,
            .f32 => hidden_bytes / 4,
        };
    }

    fn bytesPer(self: Encoding) usize {
        return switch (self) {
            .fp8_e4m3 => 1,
            .bf16 => 2,
            .f32 => 4,
        };
    }
};

pub fn encodingOf(h: Header) Encoding {
    return @fromBackingInt(@intCast(@as(u2, @truncate(h.flags))));
}

/// Writes `x` into the payload in `enc`, setting `payload_len`, the encoding
/// bits of `flags`, and (for FP8) the block scales. Unused payload and scale
/// bytes are left as they were; the receiver reads only `payload_len` bytes.
pub fn encodePayload(pkt: *Packet, enc: Encoding, x: []const f32) error{PayloadTooLarge}!void {
    if (x.len > enc.maxValues()) return error.PayloadTooLarge;
    pkt.header.flags = (pkt.header.flags & ~@as(u16, 3)) | @backingInt(enc);
    pkt.header.payload_len = @intCast(x.len * enc.bytesPer());
    switch (enc) {
        .f32 => for (x, 0..) |v, i| std.mem.writeInt(u32, pkt.payload[i * 4 ..][0..4], @bitCast(v), .little),
        .bf16 => for (x, 0..) |v, i| std.mem.writeInt(u16, pkt.payload[i * 2 ..][0..2], f32ToBf16(v), .little),
        .fp8_e4m3 => {
            var b: usize = 0;
            while (b * 32 < x.len) : (b += 1) {
                const blk = x[b * 32 .. @min(x.len, b * 32 + 32)];
                var amax: f32 = 0;
                for (blk) |v| amax = @max(amax, @abs(v));
                // Smallest power of two that keeps the block within E4M3's 448.
                const e: i32 = if (amax == 0) 0 else @intFromFloat(@ceil(std.math.log2(amax / 448.0)));
                const se = std.math.clamp(e, -127, 127);
                pkt.scales[b] = @intCast(se + 127);
                const inv = std.math.ldexp(@as(f32, 1), -se);
                for (blk, 0..) |v, i| pkt.payload[b * 32 + i] = f32ToE4m3(v * inv);
            }
        },
    }
}

/// A vector longer than one packet holds (2,048 f32 values, say) goes as
/// several packets of `enc.maxValues()` values each; bits 2 to 5 of `flags`
/// number the chunk.
pub fn chunkCount(enc: Encoding, n: usize) usize {
    return (n + enc.maxValues() - 1) / enc.maxValues();
}

pub fn chunkOf(h: Header) usize {
    return (h.flags >> 2) & 0xF;
}

/// Encodes chunk `chunk` of `x` (see `chunkCount`).
pub fn encodeChunk(pkt: *Packet, enc: Encoding, x: []const f32, chunk: usize) error{PayloadTooLarge}!void {
    const m = enc.maxValues();
    if (chunk > 15 or chunk * m >= x.len) return error.PayloadTooLarge;
    try encodePayload(pkt, enc, x[chunk * m .. @min(x.len, (chunk + 1) * m)]);
    pkt.header.flags = (pkt.header.flags & ~@as(u16, 0xF << 2)) | @as(u16, @intCast(chunk << 2));
}

/// Decodes a packet's chunk into its place in `out` (the whole vector) and
/// returns the chunk number.
pub fn decodeChunk(pkt: *const Packet, out: []f32) error{BadPayload}!usize {
    const enc = encodingOf(pkt.header);
    const chunk = chunkOf(pkt.header);
    const m = enc.maxValues();
    if (chunk * m >= out.len) return error.BadPayload;
    try decodeInto(pkt, out[chunk * m .. @min(out.len, (chunk + 1) * m)]);
    return chunk;
}

/// Bit 6 of `flags` (TASK-11): a prefetch hint rather than an expert call.
/// `token_id` numbers the forward pass, `layer` is the layer the hint is for,
/// `expert` counts the experts listed, and the payload holds their ids.
pub const flag_prefetch: u16 = 1 << 6;

pub fn isPrefetch(h: Header) bool
{
    return h.flags & flag_prefetch != 0;
}

/// Starts an empty prefetch hint for `layer` in forward pass `step`.
pub fn beginPrefetch(pkt: *Packet, step: u64, layer: u16) void
{
    pkt.header = .{ .token_id = step, .generation = 0, .src_rank = 0, .layer = layer, .expert = 0, .flags = flag_prefetch, .gate_weight = 0, .payload_len = 0 };
}

pub fn addPrefetch(pkt: *Packet, expert: u16) error{PayloadTooLarge}!void
{
    const i = pkt.header.expert;
    if ((@as(usize, i) + 1) * 2 > hidden_bytes)
    {
        return error.PayloadTooLarge;
    }
    std.mem.writeInt(u16, pkt.payload[@as(usize, i) * 2 ..][0..2], expert, .little);
    pkt.header.expert = i + 1;
    pkt.header.payload_len = (@as(u32, i) + 1) * 2;
}

/// The experts a prefetch hint lists, after checking its length.
pub fn prefetchCount(pkt: *const Packet) error{BadPayload}!usize
{
    const n: usize = pkt.header.expert;
    if (!isPrefetch(pkt.header) or pkt.header.payload_len != n * 2 or n * 2 > hidden_bytes)
    {
        return error.BadPayload;
    }
    return n;
}

pub fn prefetchExpert(pkt: *const Packet, i: usize) u16
{
    return std.mem.readInt(u16, pkt.payload[i * 2 ..][0..2], .little);
}

/// Reads a single-packet payload into `out`, which must hold exactly the
/// packet's value count (`payload_len` / bytes per value).
pub fn decodePayload(pkt: *const Packet, out: []f32) error{BadPayload}!void {
    if (chunkOf(pkt.header) != 0) return error.BadPayload;
    return decodeInto(pkt, out);
}

fn decodeInto(pkt: *const Packet, out: []f32) error{BadPayload}!void {
    const enc = encodingOf(pkt.header);
    if (pkt.header.flags >> 6 != 0 or pkt.header.payload_len != out.len * enc.bytesPer() or out.len > enc.maxValues()) return error.BadPayload;
    switch (enc) {
        .f32 => for (out, 0..) |*o, i| {
            o.* = @bitCast(std.mem.readInt(u32, pkt.payload[i * 4 ..][0..4], .little));
        },
        .bf16 => for (out, 0..) |*o, i| {
            o.* = @bitCast(@as(u32, std.mem.readInt(u16, pkt.payload[i * 2 ..][0..2], .little)) << 16);
        },
        .fp8_e4m3 => for (out, 0..) |*o, i| {
            const s: i32 = @as(i32, pkt.scales[i / 32]) - 127;
            o.* = std.math.ldexp(e4m3ToF32(pkt.payload[i]), s);
        },
    }
}

fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// Rounds `x` to the nearest even integer.
fn roundEven(x: f32) f32 {
    const f = @floor(x);
    const diff = x - f;
    if (diff > 0.5) return f + 1;
    if (diff < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// float32 to E4M3 ("e4m3fn": bias 7, max 448, no infinities), round to
/// nearest even, saturating.
pub fn f32ToE4m3(x: f32) u8 {
    if (std.math.isNan(x)) return 0x7F;
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @abs(x);
    if (a >= 464) return sign | 0x7E; // 464 is halfway past 448; below it rounds to 448 at most
    if (a < 0x1p-6) { // subnormal: m * 2^-9
        const m: u8 = @intFromFloat(roundEven(a * 512));
        return sign | m; // m == 8 is the smallest normal, encoded the same way
    }
    var e: i32 = @intFromFloat(@floor(std.math.log2(a)));
    var m = roundEven((std.math.ldexp(a, -e) - 1) * 8);
    if (m == 8) {
        m = 0;
        e += 1;
    }
    if (e > 8 or (e == 8 and m > 6)) return sign | 0x7E;
    return sign | @as(u8, @intCast((e + 7) << 3)) | @as(u8, @intFromFloat(m));
}

pub fn e4m3ToF32(b: u8) f32 {
    if (b & 0x7F == 0x7F) return std.math.nan(f32);
    const e: i32 = (b >> 3) & 0xF;
    const m: f32 = @floatFromInt(b & 7);
    const mag = if (e == 0) m * 0x1p-9 else std.math.ldexp(1 + m / 8, e - 7);
    return if (b & 0x80 != 0) -mag else mag;
}

comptime {
    std.debug.assert(@sizeOf(Header) == 32);
    std.debug.assert(@sizeOf(Tail) == 16);
    std.debug.assert(@sizeOf(Packet) == packet_bytes);
    std.debug.assert(packet_bytes % 128 == 0);
    std.debug.assert(@offsetOf(Header, "generation") == 0x08);
    std.debug.assert(@offsetOf(Header, "src_rank") == 0x10);
    std.debug.assert(@offsetOf(Header, "gate_weight") == 0x18);
    std.debug.assert(@offsetOf(Header, "payload_len") == 0x1C);
    std.debug.assert(@offsetOf(Packet, "payload") == 0x0020);
    std.debug.assert(@offsetOf(Packet, "scales") == 0x1C20);
    std.debug.assert(@offsetOf(Packet, "pad") == 0x1D00);
    std.debug.assert(@offsetOf(Packet, "tail") == 0x1D70);
    std.debug.assert(@bitOffsetOf(Imm, "kind") == 16);
    std.debug.assert(@bitOffsetOf(Imm, "gen8") == 18);
    std.debug.assert(@bitOffsetOf(Imm, "src_rank") == 26);
}

/// Mirrors `tf_wire_layout` in csrc/tf_wire.c, in the same order.
pub const layout = [_]usize{
    @sizeOf(Packet),
    @offsetOf(Header, "generation"),
    @offsetOf(Header, "src_rank"),
    @offsetOf(Header, "gate_weight"),
    @offsetOf(Header, "payload_len"),
    @offsetOf(Packet, "payload"),
    @offsetOf(Packet, "scales"),
    @offsetOf(Packet, "tail"),
};

extern fn tf_wire_layout(out: [*]usize, n: usize) usize;

test "C and Zig agree on the packet layout" {
    var c_layout: [layout.len]usize = undefined;
    try std.testing.expectEqual(layout.len, tf_wire_layout(&c_layout, c_layout.len));
    try std.testing.expectEqualSlices(usize, &layout, &c_layout);
}

test "immediate round-trips and places fields at the documented bits" {
    const imm: Imm = .{ .slot = 513, .kind = .combine, .gen8 = 0xAB, .src_rank = 5 };
    const raw = imm.encode();
    try std.testing.expectEqual(@as(u32, 513 | (1 << 16) | (0xAB << 18) | (5 << 26)), raw);
    try std.testing.expectEqual(imm, Imm.decode(raw));
}

test "CRC32C matches the standard check value and Zig's reference" {
    try std.testing.expectEqual(@as(u32, 0xE3069283), crc32c("123456789"));
    var buf: [1027]u8 = undefined;
    for (&buf, 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
    try std.testing.expectEqual(std.hash.crc.@"CRC-32/ISCSI".hash(&buf), crc32c(&buf));
}

test "E4M3 encoding round-trips every finite code" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7F == 0x7F) continue;
        const v = e4m3ToF32(b);
        const back = f32ToE4m3(v);
        // +0 and -0 both encode as themselves.
        try std.testing.expectEqual(b, back);
    }
    try std.testing.expectEqual(@as(u8, 0x7E), f32ToE4m3(1000)); // saturates at 448
    try std.testing.expectEqual(@as(u8, 0x38), f32ToE4m3(1.03)); // 1.0 is nearest
    try std.testing.expectEqual(@as(u8, 0x3C), f32ToE4m3(1.5625)); // halfway 1.5/1.625: even mantissa 1.5
}

test "payload encodings round-trip within their precision" {
    const pkt = try std.testing.allocator.create(Packet);
    defer std.testing.allocator.destroy(pkt);
    pkt.* = std.mem.zeroes(Packet);
    var x: [2048]f32 = undefined;
    var s: u32 = 12345;
    for (&x, 0..) |*v, i| {
        s = s *% 1664525 +% 1013904223;
        v.* = (@as(f32, @floatFromInt(s >> 8)) / 16777216.0 - 0.5) * @as(f32, @floatFromInt(1 + i % 7)) * 3;
    }
    var y: [2048]f32 = undefined;
    for ([_]Encoding{ .fp8_e4m3, .bf16, .f32 }) |enc| {
        if (x.len > enc.maxValues()) {
            try std.testing.expectError(error.PayloadTooLarge, encodePayload(pkt, enc, &x));
            continue;
        }
        try encodePayload(pkt, enc, &x);
        try std.testing.expectEqual(enc, encodingOf(pkt.header));
        try decodePayload(pkt, &y);
        // Worst relative error against each block's largest value.
        const tol: f32 = switch (enc) {
            .fp8_e4m3 => 1.0 / 16.0, // 3 mantissa bits
            .bf16 => 1.0 / 256.0,
            .f32 => 0,
        };
        for (0..x.len / 32) |b| {
            var amax: f32 = 0;
            for (x[b * 32 ..][0..32]) |v| amax = @max(amax, @abs(v));
            for (x[b * 32 ..][0..32], y[b * 32 ..][0..32]) |a, c| try std.testing.expect(@abs(a - c) <= tol * amax);
        }
    }
}

test "a vector larger than one packet goes as numbered chunks" {
    const pkt = try std.testing.allocator.create(Packet);
    defer std.testing.allocator.destroy(pkt);
    pkt.* = std.mem.zeroes(Packet);
    var x: [2048]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.37 - 100;
    var y: [2048]f32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 2), chunkCount(.f32, x.len));
    for (0..2) |ch| {
        try encodeChunk(pkt, .f32, &x, ch);
        try std.testing.expectEqual(ch, try decodeChunk(pkt, &y));
    }
    try std.testing.expectEqualSlices(f32, &x, &y);
    try std.testing.expectError(error.PayloadTooLarge, encodeChunk(pkt, .f32, &x, 2));
}

test "prefetch hints list their experts and never decode as a hidden state" {
    var pkt = std.mem.zeroes(Packet);
    beginPrefetch(&pkt, 7, 3);
    try addPrefetch(&pkt, 40);
    try addPrefetch(&pkt, 63);
    try std.testing.expect(isPrefetch(pkt.header));
    try std.testing.expectEqual(@as(usize, 2), try prefetchCount(&pkt));
    try std.testing.expectEqual(@as(u16, 63), prefetchExpert(&pkt, 1));
    try std.testing.expectEqual(@as(u64, 7), pkt.header.token_id);
    var out: [1]f32 = undefined;
    try std.testing.expectError(error.BadPayload, decodeChunk(&pkt, &out));
}

test "sealed packet passes; corruption and stale generation are caught" {
    const pkt = try std.testing.allocator.create(Packet);
    defer std.testing.allocator.destroy(pkt);
    pkt.* = std.mem.zeroes(Packet);
    pkt.header = .{ .token_id = 42, .generation = 7, .src_rank = 1, .layer = 3, .expert = 9, .flags = 0, .gate_weight = 0.25, .payload_len = hidden_bytes };
    for (&pkt.payload, 0..) |*b, i| b.* = @truncate(i);
    pkt.seal();
    try pkt.check(7);
    try std.testing.expectError(error.StaleGeneration, pkt.check(8));
    pkt.payload[100] ^= 1;
    try std.testing.expectError(error.BadCrc, pkt.check(7));
}
