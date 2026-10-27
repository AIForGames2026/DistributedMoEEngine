//! Reader for .tfref files written by tools/ref/ref_moe.py (TASK-06): named
//! f32/i32 tensors in one little-endian blob. The layout is documented in
//! ref_moe.py. Tensor data is 64-byte aligned in the file, but the blob itself
//! may not be (e.g. @embedFile), so element slices are align(1).
const std = @import("std");

pub const DType = enum(u8) { f32 = 0, i32 = 1 };

pub const Tensor = struct {
    dtype: DType,
    dims: []align(1) const u32,
    data: []const u8,

    pub fn len(self: Tensor) usize {
        var n: usize = 1;
        for (self.dims) |d| n *= d;
        return n;
    }

    pub fn f32s(self: Tensor) []align(1) const f32 {
        std.debug.assert(self.dtype == .f32);
        return std.mem.bytesAsSlice(f32, self.data);
    }

    pub fn i32s(self: Tensor) []align(1) const i32 {
        std.debug.assert(self.dtype == .i32);
        return std.mem.bytesAsSlice(i32, self.data);
    }
};

pub const Error = error{ BadMagic, BadVersion, Truncated, NotFound };

pub const File = struct {
    bytes: []const u8,
    count: u32,

    pub fn parse(bytes: []const u8) Error!File {
        if (bytes.len < 16) return error.Truncated;
        if (!std.mem.eql(u8, bytes[0..8], "TFREF\x00\x00\x00")) return error.BadMagic;
        if (std.mem.readInt(u32, bytes[8..12], .little) != 1) return error.BadVersion;
        return .{ .bytes = bytes, .count = std.mem.readInt(u32, bytes[12..16], .little) };
    }

    /// Linear scan of the entry table; files are small.
    pub fn get(self: File, name: []const u8) Error!Tensor {
        var pos: usize = 16;
        for (0..self.count) |_| {
            const nlen = try self.int(u16, pos);
            const entry_name = try self.slice(pos + 2, nlen);
            pos += 2 + nlen;
            const header = try self.slice(pos, 2);
            const ndim = header[1];
            const dims_bytes = try self.slice(pos + 2, 4 * @as(usize, ndim));
            const off = try self.int(u64, pos + 2 + 4 * @as(usize, ndim));
            const nbytes = try self.int(u64, pos + 10 + 4 * @as(usize, ndim));
            pos += 2 + 4 * @as(usize, ndim) + 16;
            if (std.mem.eql(u8, entry_name, name)) {
                return .{
                    .dtype = switch (header[0]) {
                        0 => .f32,
                        1 => .i32,
                        else => return error.Truncated,
                    },
                    .dims = std.mem.bytesAsSlice(u32, dims_bytes),
                    .data = try self.slice(@intCast(off), @intCast(nbytes)),
                };
            }
        }
        return error.NotFound;
    }

    fn slice(self: File, start: usize, n: usize) Error![]const u8 {
        if (start > self.bytes.len or n > self.bytes.len - start) return error.Truncated;
        return self.bytes[start..][0..n];
    }

    fn int(self: File, comptime T: type, pos: usize) Error!T {
        return std.mem.readInt(T, (try self.slice(pos, @sizeOf(T)))[0..@sizeOf(T)], .little);
    }
};

test "rejects files that are not .tfref" {
    try std.testing.expectError(error.BadMagic, File.parse("NOTTFREF\x01\x00\x00\x00\x00\x00\x00\x00"));
    try std.testing.expectError(error.Truncated, File.parse("TFREF"));
}
