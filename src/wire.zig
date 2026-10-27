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
const Crc32c = std.hash.crc.@"CRC-32/ISCSI";

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
        return Crc32c.hash(bytes[0..@offsetOf(Packet, "tail")]);
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

test "CRC32C matches the standard check value" {
    try std.testing.expectEqual(@as(u32, 0xE3069283), Crc32c.hash("123456789"));
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
