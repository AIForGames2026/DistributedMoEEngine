//! GPU matmuls (TASK-10) and attention (TASK-12) for the forward pass:
//! csrc/tf_gpu.c on the CUDA driver API, with the kernels from
//! csrc/cuda/kernels.cu embedded as a fatbin. Built only with
//! `zig build olmoe -Dgpu=true`. TF_GPU_ATTENTION=0 keeps attention on the CPU.
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
extern fn tf_gpu_register(g: *Handle, host: *anyopaque, len: usize, dev: *u64) c_int;
extern fn tf_gpu_unregister(g: *Handle, host: *anyopaque) void;
extern fn tf_gpu_attention(g: *Handle, q: [*]const f32, n: c_int, pos0: c_int, heads: c_int, kv_heads: c_int, kvd: c_int, scale: f32, tiles_k: [*]const u64, tiles_v: [*]const u64, tiles: c_int, splits: c_int, out: [*]f32) c_int;
extern fn tf_gpu_experts(g: *Handle, experts: c_int, gate: [*]const u64, up: [*]const u64, down: [*]const u64, first: [*]const c_int, x: [*]const f32, d: c_int, f: c_int, y: [*]f32) c_int;
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
    const off = if (std.c.getenv("TF_GPU_ATTENTION")) |v| v[0] == '0' else false;
    if (!off)
    {
        olmoe.gpu_register = register;
        olmoe.gpu_unregister = unregister;
        olmoe.gpu_attention = attention;
    }
}

fn register(buf: []u8) ?u64
{
    var dev: u64 = 0;
    const rc = tf_gpu_register(handle orelse return null, buf.ptr, buf.len, &dev);
    if (rc != 0)
    {
        std.log.err("gpu: could not map {d:.2} GB for attention ({d}); attention stays on the CPU", .{ @as(f64, @floatFromInt(buf.len)) / 1e9, -rc });
        return null;
    }
    return dev;
}

fn unregister(buf: []u8) void
{
    tf_gpu_unregister(handle orelse return, buf.ptr);
}

fn attention(job: olmoe.AttentionJob) bool
{
    if (attention_failed)
    {
        return false;
    }
    const rc = tf_gpu_attention(handle.?, job.q.ptr, @intCast(job.n), @intCast(job.pos0), @intCast(job.heads), @intCast(job.kv_heads), @intCast(job.kvd), job.scale, job.tiles_k.ptr, job.tiles_v.ptr, @intCast(job.tiles_k.len), @intCast(job.splits), job.out.ptr);
    if (rc != 0)
    {
        std.log.err("gpu: attention failed ({d}); using the CPU from here on", .{-rc});
        attention_failed = true;
        return false;
    }
    return true;
}

var attention_failed = false;

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
    olmoe.gpu_experts = experts;
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
    // A process holding only cached experts never calls `map`.
    if (std.c.getenv("TF_GPU_CHECK")) |v| check = v[0] == '1';
    olmoe.gpu_matmul = matmul;
    olmoe.gpu_experts = experts;
    return true;
}

pub fn stop() void {
    olmoe.gpu_matmul = null;
    olmoe.gpu_experts = null;
    olmoe.gpu_attention = null;
    olmoe.gpu_register = null;
    olmoe.gpu_unregister = null;
    if (handle) |h| tf_gpu_close(h);
    handle = null;
}

/// Where kernels read `m`, if it is a Q8_0 weight in a mapped region.
fn devAddr(m: olmoe.Mat) ?u64
{
    if (m.format != .q8_0)
    {
        return null;
    }
    const p = @intFromPtr(m.bytes.ptr);
    if (p >= host_base and p + m.bytes.len <= host_base + host_len)
    {
        return dev_base + (p - host_base);
    }
    if (p >= live_base and p + m.bytes.len <= live_base + live_len)
    {
        return live_dev + (p - live_base);
    }
    return null;
}

/// A group of experts in one call (TASK-12): one copy in, every matmul and
/// SiLU queued on the GPU, one copy out. TF_GPU_CHECK compares with the CPU.
fn experts(job: olmoe.ExpertJob) bool
{
    if (failed or check or job.experts.len > max_group)
    {
        return false;
    }
    var gate: [max_group]u64 = undefined;
    var up: [max_group]u64 = undefined;
    var down: [max_group]u64 = undefined;
    var first: [max_group + 1]c_int = undefined;
    for (job.experts, 0..) |e, i|
    {
        gate[i] = devAddr(e.gate) orelse return false;
        up[i] = devAddr(e.up) orelse return false;
        down[i] = devAddr(e.down) orelse return false;
        first[i] = @intCast(job.first[i]);
    }
    first[job.experts.len] = @intCast(job.first[job.experts.len]);
    const e0 = job.experts[0];
    const rc = tf_gpu_experts(handle.?, @intCast(job.experts.len), &gate, &up, &down, &first, job.x.ptr, @intCast(e0.gate.cols), @intCast(e0.gate.rows), job.y.ptr);
    if (rc != 0)
    {
        std.log.err("gpu: expert batch failed ({d}); using the CPU from here on", .{-rc});
        failed = true;
        return false;
    }
    return true;
}

const max_group = 1024;

fn matmul(m: olmoe.Mat, x: []const f32, n: usize, y: []f32) bool {
    if (failed) return false;
    const dev = devAddr(m) orelse return false;
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
