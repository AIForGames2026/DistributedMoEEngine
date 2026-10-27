//! TensorFold-Swarm engine entry point (TASK-01 skeleton).
const std = @import("std");
const gemm = @import("gemm.zig");
const Arena = @import("arena.zig").Arena;

// C ABI boundary (include/tensorfold.h). Declared directly rather than through
// @cImport so the build does not depend on translate-c.
extern fn tf_version() c_uint;

/// Usage: tensorfold [--n N] [--first-kernel]
///   --n N           GEMM size (default 1024)
///   --first-kernel  exit right after the first kernel call; used by
///                   tools/bench/startup.c to time process start to first kernel
pub fn main(init: std.process.Init) !void {
    var n: usize = 1024;
    var first_kernel = false;
    var it = init.minimal.args.iterate();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--first-kernel")) {
            first_kernel = true;
        } else if (std.mem.eql(u8, arg, "--n")) {
            n = try std.fmt.parseInt(usize, it.next() orelse return error.MissingValue, 10);
        } else {
            std.debug.print("unknown argument '{s}'\n", .{arg});
            std.process.exit(2);
        }
    }

    if (first_kernel) {
        var tiny_arena = try Arena.init(3 * 64 * 64 * @sizeOf(f32));
        defer tiny_arena.deinit();
        const t = try tiny_arena.alloc(f32, 3 * 64 * 64);
        gemm.tiled(t[0 .. 64 * 64], t[64 * 64 .. 2 * 64 * 64], t[2 * 64 * 64 ..], 64, 64, 64);
        return;
    }

    const v = tf_version();
    std.debug.print("TensorFold-Swarm engine v{d}.{d}.{d}\n", .{ v >> 16, (v >> 8) & 0xff, v & 0xff });

    var arena = try Arena.init(4 * n * n * @sizeOf(f32));
    defer arena.deinit();
    const a = try arena.alloc(f32, n * n);
    const b = try arena.alloc(f32, n * n);
    const ref = try arena.alloc(f32, n * n);
    const out = try arena.alloc(f32, n * n);

    fill(a, 1);
    fill(b, 2);
    gemm.naive(a, b, ref, n, n, n);
    gemm.tiled(a, b, out, n, n, n);
    std.debug.print("{d}x{d} tiled GEMM max relative error vs naive: {e}\n", .{ n, n, gemm.maxRelErr(ref, out) });
    std.debug.print("arena: {d} KiB used of {d} KiB, base 2 MiB aligned\n", .{ arena.high_water / 1024, arena.capacity() / 1024 });
}

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(buf: []f32, seed: u32) void {
    var s: u32 = seed *% 2654435761 +% 1;
    for (buf) |*x| {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x.* = @as(f32, @floatFromInt(s >> 8)) / 8388608.0 - 1.0;
    }
}

test {
    _ = @import("arena.zig");
    _ = @import("wire.zig");
    _ = @import("transport.zig");
    _ = @import("tfref.zig");
    _ = @import("moe_ref.zig");
}

test "C ABI reports version 0.1.0" {
    try std.testing.expectEqual(@as(c_uint, 0x000100), tf_version());
}

test "tiled GEMM matches naive within 1e-5" {
    const m = 67;
    const n = 45;
    const k = 33;
    var a: [m * k]f32 = undefined;
    var b: [k * n]f32 = undefined;
    var ref: [m * n]f32 = undefined;
    var out: [m * n]f32 = undefined;
    fill(&a, 3);
    fill(&b, 4);
    gemm.naive(&a, &b, &ref, m, n, k);
    gemm.tiled(&a, &b, &out, m, n, k);
    try std.testing.expect(gemm.maxRelErr(&ref, &out) <= 1e-5);
}
