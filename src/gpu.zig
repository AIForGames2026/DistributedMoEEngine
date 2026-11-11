//! GPU matmuls for the OLMoE forward pass (TASK-10): csrc/tf_gpu.c on the CUDA
//! driver API, with the kernels from csrc/cuda/kernels.cu embedded as a
//! fatbin. Built only with `zig build olmoe -Dgpu=true`.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const nowNs = @import("model_file.zig").nowNs;

const Handle = opaque {};
const Info = extern struct {
    name: [96]u8,
    major: c_int,
    minor: c_int,
    integrated: c_int,
};

extern fn tf_gpu_open(out: *?*Handle, fatbin: *const anyopaque, err: [*]u8, err_len: usize) c_int;
extern fn tf_gpu_get_info(g: *const Handle, info: *Info) void;
extern fn tf_gpu_map(g: *Handle, host: *anyopaque, len: usize, dev: *u64) c_int;
extern fn tf_gpu_q8_0_matmul(g: *Handle, w: u64, x: [*]const f32, n: c_int, rows: c_int, cols: c_int, y: [*]f32) c_int;
extern fn tf_gpu_close(g: *Handle) void;

const fatbin align(64) = @embedFile("kernels.fatbin").*;

/// One process-wide GPU; `olmoe.gpu_matmul` points at it once `start` returns.
var handle: ?*Handle = null;
var host_base: usize = 0;
var host_len: usize = 0;
var dev_base: u64 = 0;
var live_base: usize = 0;
var live_len: usize = 0;
var live_dev: u64 = 0;
var failed = false;

/// Opens the GPU and loads the kernels. Call before loading weights: on a
/// shared-memory GPU (GB10) next to a co-tenant, the context needs memory
/// that the weights would otherwise take.
pub fn open() !void {
    var err: [256]u8 = @splat(0);
    var h: ?*Handle = null;
    if (tf_gpu_open(&h, &fatbin, &err, err.len) != 0) {
        std.log.err("gpu: {s}", .{std.mem.sliceTo(&err, 0)});
        return error.GpuOpenFailed;
    }
    handle = h;
}

/// Makes `region` (where every weight was loaded) readable by kernels, then
/// routes Q8_0 matmuls there.
pub fn map(region: []u8) !void {
    const h = handle orelse return error.GpuNotOpen;
    if (std.c.getenv("TF_GPU_CHECK")) |v| check = v[0] == '1';
    var info: Info = undefined;
    tf_gpu_get_info(h, &info);
    const t0 = nowNs();
    if (tf_gpu_map(h, region.ptr, region.len, &dev_base) != 0) return error.GpuMapFailed;
    const t1 = nowNs();
    host_base = @intFromPtr(region.ptr);
    host_len = region.len;
    olmoe.gpu_matmul = matmul;
    std.debug.print("gpu: {s} (compute {d}.{d}), weights {s} ({d:.2} GB in {d:.2} s)\n", .{
        std.mem.sliceTo(&info.name, 0),
        info.major,
        info.minor,
        if (info.integrated != 0) "used in place (shared memory)" else "copied to device memory",
        @as(f64, @floatFromInt(region.len)) / 1e9,
        @as(f64, @floatFromInt(t1 - t0)) / 1e9,
    });
}

/// Also makes `region` readable by kernels if the GPU reads host memory in
/// place (GB10), so later writes to it stay visible (the expert cache).
/// Returns false on a discrete GPU, where matmuls on it stay on the CPU.
pub fn mapLive(region: []u8) !bool
{
    const h = handle orelse return error.GpuNotOpen;
    var info: Info = undefined;
    tf_gpu_get_info(h, &info);
    if (info.integrated == 0)
    {
        return false;
    }
    if (tf_gpu_map(h, region.ptr, region.len, &live_dev) != 0)
    {
        return error.GpuMapFailed;
    }
    live_base = @intFromPtr(region.ptr);
    live_len = region.len;
    return true;
}

pub fn stop() void {
    olmoe.gpu_matmul = null;
    if (handle) |h| tf_gpu_close(h);
    handle = null;
}

fn matmul(m: olmoe.Mat, x: []const f32, n: usize, y: []f32) bool {
    if (m.format != .q8_0 or failed) return false;
    const p = @intFromPtr(m.bytes.ptr);
    var dev: u64 = 0;
    if (p >= host_base and p + m.bytes.len <= host_base + host_len)
    {
        dev = dev_base + (p - host_base);
    }
    else if (p >= live_base and p + m.bytes.len <= live_base + live_len)
    {
        dev = live_dev + (p - live_base);
    }
    else
    {
        return false;
    }
    const rc = tf_gpu_q8_0_matmul(handle.?, dev, x.ptr, @intCast(n), @intCast(m.rows), @intCast(m.cols), y.ptr);
    if (rc != 0) {
        // Fall back to the CPU for the rest of the run rather than abort.
        std.log.err("gpu: matmul failed ({d}); using the CPU from here on", .{-rc});
        failed = true;
        return false;
    }
    if (check) verify(m, x, n, y);
    return true;
}

/// TF_GPU_CHECK=1: recompute every GPU matmul on the CPU and report the first
/// mismatch (debugging; very slow).
pub var check = false;
var reported = false;

fn verify(m: olmoe.Mat, x: []const f32, n: usize, y: []f32) void {
    const gpa = std.heap.page_allocator;
    const want = gpa.alloc(f32, n * m.rows) catch return;
    defer gpa.free(want);
    const scratch = gpa.alloc(f32, m.cols) catch return;
    defer gpa.free(scratch);
    const saved = olmoe.gpu_matmul;
    olmoe.gpu_matmul = null;
    m.matmul(x, n, want, scratch);
    olmoe.gpu_matmul = saved;
    var worst: f32 = 0;
    var where: usize = 0;
    for (y, want, 0..) |a, b, i| {
        const e = @abs(a - b);
        if (std.math.isNan(e) or e > worst) {
            worst = if (std.math.isNan(e)) std.math.inf(f32) else e;
            where = i;
        }
    }
    if (worst > 1e-3 and !reported) {
        reported = true;
        std.log.err("gpu check: {d}x{d} matmul, n {d}: output {d} is {e} on the GPU, {e} on the CPU", .{ m.rows, m.cols, n, where, y[where], want[where] });
    }
    @memcpy(y, want);
}
