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
extern fn tf_gpu_matmul(g: *Handle, w: u64, format: c_int, x: [*]const f32, n: c_int, rows: c_int, cols: c_int, y: [*]f32) c_int;
extern fn tf_gpu_matmuls(g: *Handle, count: c_int, w: [*]const u64, format: c_int, rows: [*]const c_int, x: [*]const f32, n: c_int, cols: c_int, y: [*]const [*]f32) c_int;
extern fn tf_gpu_register(g: *Handle, host: *anyopaque, len: usize, dev: *u64) c_int;
extern fn tf_gpu_unregister(g: *Handle, host: *anyopaque) void;
extern fn tf_gpu_attention(g: *Handle, q: [*]const f32, n: c_int, pos0: c_int, heads: c_int, kv_heads: c_int, kvd: c_int, scale: f32, tiles_k: [*]const u64, tiles_v: [*]const u64, tiles: c_int, splits: c_int, out: [*]f32, fmt: c_int, hd: c_int) c_int;
extern fn tf_gpu_qsa(g: *Handle, q: [*]const f32, iq: [*]const f32, n: c_int, pos0: c_int, heads: c_int, kv_heads: c_int, kvd: c_int, ih: c_int, dim: c_int, bsize: c_int, top: c_int, scale: f32, blocks: u64, tiles_k: [*]const u64, tiles_v: [*]const u64, tiles: c_int, out: [*]f32, kept: ?[*]i32) c_int;
extern fn tf_gpu_delta_rule(g: *Handle, mixed: [*]const f32, beta: [*]const f32, decay: [*]const f32, n: c_int, cd: c_int, kh: c_int, dk: c_int, vh: c_int, dv: c_int, state: u64, core: [*]f32) c_int;
extern fn tf_gpu_experts(g: *Handle, groups: c_int, gate: [*]const u64, up: [*]const u64, down: [*]const u64, first: [*]const c_int, src: [*]const c_int, inv: [*]const c_int, w: [*]const f32, x: [*]const f32, n: c_int, k: c_int, d: c_int, f: c_int, y: [*]f32, fmt: c_int, limit: f32) c_int;
extern fn tf_gpu_close(g: *Handle) void;
extern fn tf_gpu_device_bytes() usize;

/// Device memory held outside the resident set that still comes out of
/// system memory: all of it on an integrated GPU (GB10), none on a discrete one.
pub fn systemBytes() u64
{
    const h = handle orelse return 0;
    var info: Info = undefined;
    tf_gpu_get_info(h, &info);
    return if (info.integrated != 0) tf_gpu_device_bytes() else 0;
}

const fatbin align(64) = @embedFile("kernels.fatbin").*;

/// One process-wide GPU; `olmoe.gpu_matmul` points at it once `start` returns.
var handle: ?*Handle = null;
var host_base: usize = 0;
var host_len: usize = 0;
var dev_base: u64 = 0;
var live_base: usize = 0;
var live_len: usize = 0;
var live_dev: u64 = 0;
/// Other buffers kernels read weights from (TASK-32's widened BF16 matrices).
var extra_base: [8]usize = @splat(0);
var extra_len: [8]usize = @splat(0);
var extra_dev: [8]u64 = @splat(0);
var extras: usize = 0;
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
        olmoe.gpu_adopt = adopt;
        olmoe.gpu_attention = attention;
        olmoe.gpu_delta_rule = deltaRule;
        olmoe.gpu_qsa = qsa;
    }
}

/// QSA attention (TASK-31) on the GPU, for 256-dim heads over an f32 cache.
fn qsa(job: olmoe.QsaJob) bool
{
    if (qsa_failed)
    {
        return false;
    }
    const c = job.cfg;
    if (c.head_dim != 256 or job.blocks_dev == 0)
    {
        return false;
    }
    const kept: ?[*]i32 = if (job.kept.len > 0) job.kept.ptr else null;
    const rc = tf_gpu_qsa(handle.?, job.q.ptr, job.iq.ptr, @intCast(job.n), @intCast(job.pos0), @intCast(c.heads), @intCast(c.kv_heads), @intCast(c.kvDim()), @intCast(c.qsa_heads), @intCast(c.qsa_dim), @intCast(c.qsa_block), @intCast(c.qsaTop()), job.scale, job.blocks_dev, job.tiles_k.ptr, job.tiles_v.ptr, @intCast(job.tiles_k.len), job.out.ptr, kept);
    if (rc != 0)
    {
        std.log.err("gpu: QSA attention failed ({d}); using the CPU from here on", .{-rc});
        qsa_failed = true;
        return false;
    }
    return true;
}

var qsa_failed = false;

fn deltaRule(job: olmoe.DeltaJob) bool
{
    if (delta_failed)
    {
        return false;
    }
    const c = job.cfg;
    if (c.ssm_dk != 128 or c.ssm_dv > 256)
    {
        return false;
    }
    const rc = tf_gpu_delta_rule(handle.?, job.mixed.ptr, job.beta.ptr, job.decay.ptr, @intCast(job.n), @intCast(c.convDim()), @intCast(c.ssm_k_heads), @intCast(c.ssm_dk), @intCast(c.ssm_v_heads), @intCast(c.ssm_dv), job.state_dev, job.core.ptr);
    if (rc != 0)
    {
        std.log.err("gpu: delta rule failed ({d}); using the CPU from here on", .{-rc});
        delta_failed = true;
        return false;
    }
    return true;
}

var delta_failed = false;

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

/// Maps `buf` for the rest of the run so matmuls read the weights in it in place.
fn adopt(buf: []u8) bool
{
    if (extras == extra_base.len)
    {
        return false;
    }
    const dev = register(buf) orelse return false;
    extra_base[extras] = @intFromPtr(buf.ptr);
    extra_len[extras] = buf.len;
    extra_dev[extras] = dev;
    extras += 1;
    return true;
}

fn attention(job: olmoe.AttentionJob) bool
{
    if (attention_failed)
    {
        return false;
    }
    const rc = tf_gpu_attention(handle.?, job.q.ptr, @intCast(job.n), @intCast(job.pos0), @intCast(job.heads), @intCast(job.kv_heads), @intCast(job.kvd), job.scale, job.tiles_k.ptr, job.tiles_v.ptr, @intCast(job.tiles_k.len), @intCast(job.splits), job.out.ptr, @intFromEnum(job.format), @intCast(job.hd));
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
    olmoe.gpu_matmuls = matmuls;
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
    olmoe.gpu_matmuls = matmuls;
    olmoe.gpu_experts = experts;
    return true;
}

pub fn stop() void {
    olmoe.gpu_matmul = null;
    olmoe.gpu_matmuls = null;
    olmoe.gpu_experts = null;
    olmoe.gpu_attention = null;
    olmoe.gpu_delta_rule = null;
    olmoe.gpu_qsa = null;
    olmoe.gpu_register = null;
    olmoe.gpu_unregister = null;
    olmoe.gpu_adopt = null;
    if (handle) |h| tf_gpu_close(h);
    handle = null;
}

const format_q8_0: c_int = 0;
const format_f32: c_int = 1;

/// The kernel format code for `m`, if a kernel reads it.
fn formatOf(m: olmoe.Mat) ?c_int
{
    // The kernels take widths in whole blocks of 32; narrower matrices stay on the CPU.
    if (m.cols % 32 != 0)
    {
        return null;
    }
    return switch (m.format)
    {
        .q8_0 => format_q8_0,
        .f32 => if (@intFromPtr(m.bytes.ptr) % 16 == 0) format_f32 else null,
        else => null,
    };
}

/// Where kernels read `m`, if it is a Q8_0, f32 or (for experts) Q4_K weight in a mapped region.
fn devAddr(m: olmoe.Mat) ?u64
{
    if (m.format != .q4_k and formatOf(m) == null)
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
    for (0..extras) |i|
    {
        if (p >= extra_base[i] and p + m.bytes.len <= extra_base[i] + extra_len[i])
        {
            return extra_dev[i] + (p - extra_base[i]);
        }
    }
    return null;
}

/// A layer of experts in one call (TASK-12): inputs gathered, every matmul
/// and SiLU queued, outputs combined on the GPU. TF_GPU_CHECK uses the CPU.
fn experts(job: olmoe.ExpertJob) bool
{
    // The Q8_0 kernels have no SwiGLU clamp (TASK-32), so a clamped Q8_0 model stays on the CPU.
    if (failed or check or job.experts.len > max_group or (job.experts[0].gate.format == .q8_0 and olmoe.swiglu_limit > 0))
    {
        return false;
    }
    var gate: [max_group]u64 = undefined;
    var up: [max_group]u64 = undefined;
    var down: [max_group]u64 = undefined;
    var first: [max_group + 1]c_int = undefined;
    for (job.experts, 0..) |e, i|
    {
        const f0 = job.experts[0].gate.format;
        if ((f0 != .q8_0 and f0 != .q4_k) or e.gate.format != f0 or e.up.format != f0 or e.down.format != f0)
        {
            return false;
        }
        gate[i] = devAddr(e.gate) orelse return false;
        up[i] = devAddr(e.up) orelse return false;
        down[i] = devAddr(e.down) orelse return false;
        first[i] = @intCast(job.first[i]);
    }
    first[job.experts.len] = @intCast(job.first[job.experts.len]);
    const e0 = job.experts[0];
    const rc = tf_gpu_experts(handle.?, @intCast(job.experts.len), &gate, &up, &down, &first, @ptrCast(job.src.ptr), @ptrCast(job.inv.ptr), job.w.ptr, job.x.ptr, @intCast(job.n), @intCast(job.k), @intCast(e0.gate.cols), @intCast(e0.gate.rows), job.y.ptr, if (e0.gate.format == .q4_k) 2 else 0, olmoe.swiglu_limit);
    if (rc != 0)
    {
        std.log.err("gpu: expert batch failed ({d}); using the CPU from here on", .{-rc});
        failed = true;
        return false;
    }
    return true;
}

const max_group = 1024;

fn matmul(m: olmoe.Mat, x: []const f32, n: usize, y: []f32) bool
{
    // A small f32 matmul for a few tokens (the router in decode) is quicker on the CPU.
    if (failed or (m.format == .f32 and n <= 8 and m.rows * m.cols <= 1 << 20))
    {
        return false;
    }
    const format = formatOf(m) orelse return false;
    const dev = devAddr(m) orelse return false;
    const rc = tf_gpu_matmul(handle.?, dev, format, x.ptr, @intCast(n), @intCast(m.rows), @intCast(m.cols), y.ptr);
    if (rc != 0)
    {
        // Fall back to the CPU for the rest of the run rather than abort.
        std.log.err("gpu: matmul failed ({d}); using the CPU from here on", .{-rc});
        failed = true;
        return false;
    }
    if (check)
    {
        verify(m, x, n, y);
    }
    return true;
}

/// Matmuls of one format on the same inputs with one copy in (q, k, and v).
fn matmuls(mats: []const olmoe.Mat, x: []const f32, n: usize, ys: []const []f32) bool
{
    const most = 8;
    if (failed or check or mats.len > most)
    {
        return false;
    }
    var dev: [most]u64 = undefined;
    var rows: [most]c_int = undefined;
    var outs: [most][*]f32 = undefined;
    const format = formatOf(mats[0]) orelse return false;
    for (mats, 0..) |m, i|
    {
        if (formatOf(m) != format or m.cols != mats[0].cols)
        {
            return false;
        }
        dev[i] = devAddr(m) orelse return false;
        rows[i] = @intCast(m.rows);
        outs[i] = ys[i].ptr;
    }
    const rc = tf_gpu_matmuls(handle.?, @intCast(mats.len), &dev, format, &rows, x.ptr, @intCast(n), @intCast(mats[0].cols), &outs);
    if (rc != 0)
    {
        std.log.err("gpu: matmuls failed ({d}); using the CPU from here on", .{-rc});
        failed = true;
        return false;
    }
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
