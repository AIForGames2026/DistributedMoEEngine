//! Toolchain gate (TASK-01, risk R1): proves this machine can build and run a
//! Zig binary that links C, liburing, and the CUDA runtime. Run with
//! `zig build gate`; exits nonzero if any check fails.
const std = @import("std");

const CudaInfo = extern struct {
    name: [96]u8,
    major: c_int,
    minor: c_int,
    sm_count: c_int,
    total_mem: usize,
};

extern fn tf_version() c_uint;
extern fn tf_uring_selftest() c_int;
extern fn tf_cuda_device_count() c_int;
extern fn tf_cuda_device_info(dev: c_int, out: *CudaInfo) c_int;

pub fn main() !void {
    var failed = false;

    const v = tf_version();
    std.debug.print("[ok]   C ABI: tensorfold v{d}.{d}.{d}\n", .{ v >> 16, (v >> 8) & 0xff, v & 0xff });

    const uring = tf_uring_selftest();
    if (uring == 0) {
        std.debug.print("[ok]   io_uring: NOP submitted and reaped\n", .{});
    } else {
        std.debug.print("[FAIL] io_uring: error {d}\n", .{uring});
        failed = true;
    }

    const count = tf_cuda_device_count();
    if (count <= 0) {
        std.debug.print("[FAIL] CUDA: cudaGetDeviceCount returned {d}\n", .{count});
        failed = true;
    } else {
        std.debug.print("[ok]   CUDA: {d} device(s)\n", .{count});
        for (0..@intCast(count)) |dev| {
            var info: CudaInfo = undefined;
            const rc = tf_cuda_device_info(@intCast(dev), &info);
            if (rc != 0) {
                std.debug.print("[FAIL] CUDA device {d}: error {d}\n", .{ dev, rc });
                failed = true;
                continue;
            }
            const name = std.mem.sliceTo(&info.name, 0);
            std.debug.print("[ok]   CUDA device {d}: {s}, compute {d}.{d}, {d} SMs, {d} MiB, malloc+memset ok\n", .{
                dev, name, info.major, info.minor, info.sm_count, info.total_mem >> 20,
            });
        }
    }

    if (failed) {
        std.debug.print("gate: FAILED\n", .{});
        std.process.exit(1);
    }
    std.debug.print("gate: PASSED\n", .{});
}
