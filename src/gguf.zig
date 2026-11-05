//! GGUF v2/v3 header parser (TASK-08): metadata and the tensor table, so the
//! loader can find each tensor's bytes. Parses a prefix of the file held in
//! memory; tensor data itself is read separately. Strings and arrays point into
//! the prefix, so it must outlive the `Header`.
//!
//! Layout (little-endian): "GGUF", u32 version, u64 tensor count, u64 metadata
//! count, the metadata (key, type, value), then per tensor its name, u32 ndim,
//! u64 dims[ndim] (innermost first), u32 ggml type, u64 offset into the data
//! section. The data section starts at the next multiple of
//! `general.alignment` (default 32) after the table.
const std = @import("std");
const quant = @import("quant.zig");

pub const Error = error{ BadMagic, BadVersion, Truncated, BadValueType, BadTensor, UnsupportedType, OutOfMemory };

pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
};

pub const Value = union(enum) {
    int: i128,
    float: f64,
    boolean: bool,
    string: []const u8,
    /// Element type and count; elements are not decoded.
    array: struct { elem: ValueType, len: u64 },
};

pub const KV = struct { key: []const u8, value: Value };

/// ggml tensor types the engine can load, by GGUF type id.
pub fn formatOf(ggml_type: u32) ?quant.Format {
    return switch (ggml_type) {
        0 => .f32,
        1 => .f16,
        8 => .q8_0,
        30 => .bf16,
        39 => .mxfp4,
        40 => .nvfp4,
        else => null,
    };
}

pub const TensorInfo = struct {
    name: []const u8,
    /// Innermost dimension first (ggml order); unused dims are 1.
    dims: [4]u64,
    ndim: u32,
    ggml_type: u32,
    format: quant.Format,
    /// From the start of the file.
    offset: u64,
    bytes: u64,

    pub fn elements(self: TensorInfo) u64 {
        return self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3];
    }
};

pub const Header = struct {
    version: u32,
    kvs: []KV,
    tensors: []TensorInfo,
    alignment: u64,
    data_offset: u64,

    pub fn deinit(self: *Header, gpa: std.mem.Allocator) void {
        gpa.free(self.kvs);
        gpa.free(self.tensors);
        self.* = undefined;
    }

    pub fn get(self: Header, key: []const u8) ?Value {
        for (self.kvs) |kv| if (std.mem.eql(u8, kv.key, key)) return kv.value;
        return null;
    }

    pub fn getInt(self: Header, key: []const u8) ?u64 {
        const v = self.get(key) orelse return null;
        return switch (v) {
            .int => |i| if (i >= 0 and i <= std.math.maxInt(u64)) @intCast(i) else null,
            else => null,
        };
    }

    pub fn getString(self: Header, key: []const u8) ?[]const u8 {
        const v = self.get(key) orelse return null;
        return switch (v) {
            .string => |s| s,
            else => null,
        };
    }

    pub fn tensor(self: Header, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }
};

/// Parses the header from `prefix`, the first bytes of a file of `file_bytes`
/// bytes. Returns `error.Truncated` if the prefix ends before the tensor table
/// does (read more and retry). Checks that every tensor is a supported type,
/// that its byte size follows from its shape, that it is aligned, and that
/// tensors lie inside the file without overlapping.
pub fn parse(gpa: std.mem.Allocator, prefix: []const u8, file_bytes: u64) Error!Header {
    var r: Reader = .{ .buf = prefix };
    if (!std.mem.eql(u8, try r.bytes(4), "GGUF")) return error.BadMagic;
    const version = try r.int(u32);
    if (version != 2 and version != 3) return error.BadVersion;
    const n_tensors = try r.int(u64);
    const n_kv = try r.int(u64);
    if (n_tensors > 1 << 20 or n_kv > 1 << 20) return error.Truncated;

    const kvs = try gpa.alloc(KV, @intCast(n_kv));
    errdefer gpa.free(kvs);
    for (kvs) |*kv| {
        kv.key = try r.string();
        kv.value = try r.value(try r.valueType());
    }

    var alignment: u64 = 32;
    for (kvs) |kv| {
        if (std.mem.eql(u8, kv.key, "general.alignment")) switch (kv.value) {
            .int => |a| alignment = if (a > 0 and a <= 1 << 20) @intCast(a) else return error.BadTensor,
            else => return error.BadTensor,
        };
    }
    if (!std.math.isPowerOfTwo(alignment)) return error.BadTensor;

    const tensors = try gpa.alloc(TensorInfo, @intCast(n_tensors));
    errdefer gpa.free(tensors);
    for (tensors) |*t| {
        t.name = try r.string();
        t.ndim = try r.int(u32);
        if (t.ndim == 0 or t.ndim > 4) return error.BadTensor;
        t.dims = .{ 1, 1, 1, 1 };
        for (t.dims[0..t.ndim]) |*d| d.* = try r.int(u64);
        t.ggml_type = try r.int(u32);
        t.offset = try r.int(u64); // relative for now
        t.format = formatOf(t.ggml_type) orelse return error.UnsupportedType;
        const n = std.math.mul(u64, t.dims[0], t.dims[1]) catch return error.BadTensor;
        const n2 = std.math.mul(u64, n, t.dims[2] * t.dims[3]) catch return error.BadTensor;
        // Blocks run along the innermost dimension, so rows must be whole blocks.
        if (t.dims[0] % t.format.blockValues() != 0) return error.BadTensor;
        t.bytes = t.format.bytesFor(n2) catch return error.BadTensor;
    }

    const data_offset = std.mem.alignForward(u64, r.pos, alignment);
    for (tensors) |*t| {
        if (t.offset % alignment != 0) return error.BadTensor;
        t.offset += data_offset;
        if (t.offset > file_bytes or t.bytes > file_bytes - t.offset) return error.BadTensor;
    }
    // No two tensors may overlap.
    const sorted = try gpa.dupe(TensorInfo, tensors);
    defer gpa.free(sorted);
    std.mem.sort(TensorInfo, sorted, {}, struct {
        fn lt(_: void, a: TensorInfo, b: TensorInfo) bool {
            return a.offset < b.offset;
        }
    }.lt);
    for (sorted[0..sorted.len -| 1], sorted[@min(1, sorted.len)..]) |a, b| {
        if (a.offset + a.bytes > b.offset) return error.BadTensor;
    }

    return .{ .version = version, .kvs = kvs, .tensors = tensors, .alignment = alignment, .data_offset = data_offset };
}

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn bytes(self: *Reader, n: u64) Error![]const u8 {
        if (n > self.buf.len - self.pos) return error.Truncated;
        const s = self.buf[self.pos..][0..@intCast(n)];
        self.pos += @intCast(n);
        return s;
    }

    fn int(self: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try self.bytes(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn string(self: *Reader) Error![]const u8 {
        return self.bytes(try self.int(u64));
    }

    fn valueType(self: *Reader) Error!ValueType {
        return std.enums.fromInt(ValueType, try self.int(u32)) orelse error.BadValueType;
    }

    fn value(self: *Reader, t: ValueType) Error!Value {
        return switch (t) {
            .u8 => .{ .int = try self.int(u8) },
            .i8 => .{ .int = try self.int(i8) },
            .u16 => .{ .int = try self.int(u16) },
            .i16 => .{ .int = try self.int(i16) },
            .u32 => .{ .int = try self.int(u32) },
            .i32 => .{ .int = try self.int(i32) },
            .u64 => .{ .int = try self.int(u64) },
            .i64 => .{ .int = try self.int(i64) },
            .f32 => .{ .float = @as(f32, @bitCast(try self.int(u32))) },
            .f64 => .{ .float = @bitCast(try self.int(u64)) },
            .bool => .{ .boolean = (try self.int(u8)) != 0 },
            .string => .{ .string = try self.string() },
            .array => blk: {
                const elem = try self.valueType();
                const len = try self.int(u64);
                if (elem == .array) return error.BadValueType; // nested arrays are not used
                if (elem == .string) {
                    for (0..len) |_| _ = try self.string();
                } else {
                    _ = try self.bytes(std.math.mul(u64, len, scalarSize(elem)) catch return error.Truncated);
                }
                break :blk .{ .array = .{ .elem = elem, .len = len } };
            },
        };
    }

    fn scalarSize(t: ValueType) u64 {
        return switch (t) {
            .u8, .i8, .bool => 1,
            .u16, .i16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
            .string, .array => unreachable,
        };
    }
};

test "rejects files that are not GGUF" {
    try std.testing.expectError(error.BadMagic, parse(std.testing.allocator, "GGML\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 24));
    try std.testing.expectError(error.Truncated, parse(std.testing.allocator, "GGUF\x03\x00", 6));
}
