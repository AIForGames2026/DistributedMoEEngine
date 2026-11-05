//! safetensors header parser (TASK-08). Layout: u64 header length N
//! (little-endian), N bytes of JSON mapping each tensor name to its dtype,
//! shape, and [begin, end) byte offsets relative to the end of the header,
//! then the data. An optional "__metadata__" entry holds strings.
const std = @import("std");

pub const Error = error{ BadHeader, BadTensor, Truncated, OutOfMemory };

pub const DType = enum { u8, i8, f8_e4m3, f16, bf16, f32, i32, i64, f64 };

fn parseDType(s: []const u8) ?DType {
    const names = .{ .{ "U8", DType.u8 }, .{ "I8", DType.i8 }, .{ "F8_E4M3", DType.f8_e4m3 }, .{ "F16", DType.f16 }, .{ "BF16", DType.bf16 }, .{ "F32", DType.f32 }, .{ "I32", DType.i32 }, .{ "I64", DType.i64 }, .{ "F64", DType.f64 } };
    inline for (names) |n| if (std.mem.eql(u8, s, n[0])) return n[1];
    return null;
}

pub fn dtypeBytes(d: DType) u64 {
    return switch (d) {
        .u8, .i8, .f8_e4m3 => 1,
        .f16, .bf16 => 2,
        .f32, .i32 => 4,
        .i64, .f64 => 8,
    };
}

pub const TensorInfo = struct {
    name: []const u8,
    dtype: DType,
    /// Outermost dimension first (PyTorch order); a scalar has ndim 0.
    shape: [4]u64,
    ndim: u32,
    /// From the start of the file.
    offset: u64,
    bytes: u64,
};

pub const Header = struct {
    arena: std.heap.ArenaAllocator,
    tensors: []TensorInfo,
    data_offset: u64,

    pub fn deinit(self: *Header) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn tensor(self: Header, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }
};

/// Bytes of the file needed to parse the header, from its first 8 bytes.
pub fn headerBytes(first8: *const [8]u8) u64 {
    return 8 + std.mem.readInt(u64, first8, .little);
}

/// Parses the header from `prefix` (at least `headerBytes` long) of a file of
/// `file_bytes` bytes. Checks dtypes, that each byte range matches its shape,
/// and that ranges lie inside the file without overlapping.
pub fn parse(gpa: std.mem.Allocator, prefix: []const u8, file_bytes: u64) Error!Header {
    if (prefix.len < 8) return error.Truncated;
    const n = std.mem.readInt(u64, prefix[0..8], .little);
    if (n > 100 << 20) return error.BadHeader;
    if (prefix.len < 8 + n) return error.Truncated;
    const data_offset = 8 + n;

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, prefix[8..][0..@intCast(n)], .{}) catch return error.BadHeader;
    if (root != .object) return error.BadHeader;

    var list: std.ArrayList(TensorInfo) = .empty;
    var it = root.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        const obj = switch (entry.value_ptr.*) {
            .object => |o| o,
            else => return error.BadHeader,
        };
        const dtype_s = switch (obj.get("dtype") orelse return error.BadHeader) {
            .string => |s| s,
            else => return error.BadHeader,
        };
        var t: TensorInfo = .{
            .name = entry.key_ptr.*,
            .dtype = parseDType(dtype_s) orelse return error.BadTensor,
            .shape = .{ 1, 1, 1, 1 },
            .ndim = 0,
            .offset = 0,
            .bytes = 0,
        };
        const shape = switch (obj.get("shape") orelse return error.BadHeader) {
            .array => |arr| arr.items,
            else => return error.BadHeader,
        };
        if (shape.len > 4) return error.BadTensor;
        t.ndim = @intCast(shape.len);
        var elems: u64 = 1;
        for (shape, 0..) |d, i| {
            const v: u64 = switch (d) {
                .integer => |x| if (x >= 0) @intCast(x) else return error.BadTensor,
                else => return error.BadHeader,
            };
            t.shape[i] = v;
            elems = std.math.mul(u64, elems, v) catch return error.BadTensor;
        }
        const offs = switch (obj.get("data_offsets") orelse return error.BadHeader) {
            .array => |arr| arr.items,
            else => return error.BadHeader,
        };
        if (offs.len != 2 or offs[0] != .integer or offs[1] != .integer) return error.BadHeader;
        if (offs[0].integer < 0 or offs[1].integer < offs[0].integer) return error.BadTensor;
        const begin: u64 = @intCast(offs[0].integer);
        const end: u64 = @intCast(offs[1].integer);
        t.offset = data_offset + begin;
        t.bytes = end - begin;
        if (t.bytes != std.math.mul(u64, elems, dtypeBytes(t.dtype)) catch return error.BadTensor) return error.BadTensor;
        if (t.offset > file_bytes or t.bytes > file_bytes - t.offset) return error.BadTensor;
        try list.append(a, t);
    }

    const sorted = try a.dupe(TensorInfo, list.items);
    std.mem.sort(TensorInfo, sorted, {}, struct {
        fn lt(_: void, x: TensorInfo, y: TensorInfo) bool {
            return x.offset < y.offset;
        }
    }.lt);
    for (sorted[0..sorted.len -| 1], sorted[@min(1, sorted.len)..]) |x, y| {
        if (x.offset + x.bytes > y.offset) return error.BadTensor;
    }
    return .{ .arena = arena, .tensors = list.items, .data_offset = data_offset };
}

test "rejects a header whose sizes disagree with the shape" {
    const json = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,12]}}";
    var buf: [8 + json.len + 16]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], json.len, .little);
    @memcpy(buf[8..][0..json.len], json);
    try std.testing.expectError(error.BadTensor, parse(std.testing.allocator, &buf, buf.len));
}
