//! Model files on Linux (TASK-08, TASK-10): opening a GGUF or ModelOpt NVFP4
//! safetensors checkpoint, parsing its header into a `loader.Model`, and
//! reading a `loader.Plan` into memory through the TASK-03 io_uring O_DIRECT
//! reader. Shared by `tfload` and `olmoe`.
const std = @import("std");
const c = std.c;
const gguf = @import("gguf.zig");
const safetensors = @import("safetensors.zig");
const loader = @import("loader.zig");

pub const Reader = opaque {};
pub const ReaderOpts = extern struct { queue_depth: c_uint, block_bytes: usize, sqpoll: c_int };
pub const ReaderInfo = extern struct {
    file_bytes: u64,
    dio_mem_align: u32,
    dio_offset_align: u32,
    fixed_buffer: c_int,
    fixed_file: c_int,
    sqpoll: c_int,
};
extern fn tf_reader_open(out: *?*Reader, path: [*:0]const u8, opts: *const ReaderOpts, region: *anyopaque, region_len: usize) c_int;
extern fn tf_reader_get_info(r: *const Reader, info: *ReaderInfo) void;
extern fn tf_reader_read(r: *Reader, offset: u64, len: usize, dst: *anyopaque, lat_ns: ?[*]u32, lat_count: ?*usize, lat_cap: usize) c_int;
extern fn tf_reader_close(r: *Reader) void;

/// Plans are aligned to this; it must be a multiple of the file system's
/// direct-I/O alignment (512 or 4096 in practice).
pub const alignment = 4096;
/// Gaps up to this size between wanted ranges are read rather than skipped.
pub const max_gap = 64 * 1024;

/// A file opened for buffered reads (headers, the end-of-file tail).
pub const File = struct {
    fd: c.fd_t,
    bytes: u64,

    pub fn open(path: [:0]const u8) !File {
        const fd = c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) {
            std.debug.print("cannot open {s}\n", .{path});
            return error.FileNotFound;
        }
        const end = c.lseek(fd, 0, c.SEEK.END);
        if (end < 0) return error.Unexpected;
        return .{ .fd = fd, .bytes = @intCast(end) };
    }

    pub fn close(self: File) void {
        _ = c.close(self.fd);
    }

    pub fn readAt(self: File, offset: u64, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            const n = c.pread(self.fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
            if (n <= 0) return error.ShortRead;
            done += @intCast(n);
        }
    }
};

pub fn readWhole(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const f = try File.open(z);
    defer f.close();
    const buf = try gpa.alloc(u8, @intCast(f.bytes));
    errdefer gpa.free(buf);
    try f.readAt(0, buf);
    return buf;
}

pub const ModelState = struct {
    kind: []const u8,
    model: loader.Model,
    prefix: []u8,
    gguf_header: ?gguf.Header = null,
    st_header: ?safetensors.Header = null,

    pub fn deinit(self: *ModelState, gpa: std.mem.Allocator) void {
        self.model.deinit(gpa);
        if (self.gguf_header) |*h| h.deinit(gpa);
        if (self.st_header) |*h| h.deinit();
        gpa.free(self.prefix);
    }
};

/// Reads as much of the file start as the header needs and builds the model.
pub fn openModel(gpa: std.mem.Allocator, file: File, config_path: ?[]const u8) !ModelState {
    var magic: [8]u8 = undefined;
    try file.readAt(0, &magic);
    if (std.mem.eql(u8, magic[0..4], "GGUF")) {
        var want: u64 = 4 << 20;
        while (true) {
            const n = @min(want, file.bytes);
            const prefix = try gpa.alloc(u8, @intCast(n));
            try file.readAt(0, prefix);
            var h = gguf.parse(gpa, prefix, file.bytes) catch |err| {
                gpa.free(prefix);
                if (err == error.Truncated and n < file.bytes) {
                    want *= 4;
                    continue;
                }
                return err;
            };
            errdefer h.deinit(gpa);
            return .{ .kind = "GGUF", .model = try loader.Model.fromGguf(gpa, h), .prefix = prefix, .gguf_header = h };
        }
    }
    const n = safetensors.headerBytes(&magic);
    if (n > file.bytes) return error.UnknownFormat;
    const prefix = try gpa.alloc(u8, @intCast(n));
    errdefer gpa.free(prefix);
    try file.readAt(0, prefix);
    var h = try safetensors.parse(gpa, prefix, file.bytes);
    errdefer h.deinit();
    const config = try readWhole(gpa, config_path orelse {
        std.debug.print("safetensors checkpoints need --config config.json\n", .{});
        return error.MissingConfig;
    });
    defer gpa.free(config);
    return .{ .kind = "safetensors (ModelOpt NVFP4)", .model = try loader.Model.fromSafetensors(gpa, h, config), .prefix = prefix, .st_header = h };
}

pub fn describe(m: loader.Model, path: []const u8, kind: []const u8) void {
    var expert_bytes: u64 = 0;
    for (m.weights[0..3]) |w| {
        for (w.parts) |r| expert_bytes += r.len;
    }
    std.debug.print("{s}: {s}, {d} layers x {d} experts, d_model {d}, d_ff {d}; gate/up/down {t}/{t}/{t}; {d:.2} MB per expert per layer\n", .{
        std.fs.path.basename(path), kind,                        m.layers,                        m.experts,                                   m.d_model, m.d_ff,
        m.weights[0].format,        m.weights[m.experts].format, m.weights[2 * m.experts].format, @as(f64, @floatFromInt(expert_bytes)) / 1e6,
    });
}

/// Reads every span of `plan` into `region` through io_uring O_DIRECT; a span
/// that runs into the last partial block of the file reads that tail with a
/// plain pread instead.
pub fn readPlan(path: [:0]const u8, file: File, plan: loader.Plan, region: []align(4096) u8) !ReaderInfo {
    var r: ?*Reader = null;
    const ropts: ReaderOpts = .{ .queue_depth = 16, .block_bytes = 1 << 20, .sqpoll = 0 };
    const rc = tf_reader_open(&r, path, &ropts, region.ptr, region.len);
    if (rc != 0) {
        std.debug.print("tf_reader_open: error {d}\n", .{-rc});
        return error.ReaderOpenFailed;
    }
    defer tf_reader_close(r.?);
    var info: ReaderInfo = undefined;
    tf_reader_get_info(r.?, &info);
    if (alignment % info.dio_offset_align != 0 or alignment % info.dio_mem_align != 0) return error.UnsupportedAlignment;
    const direct_end = std.mem.alignBackward(u64, file.bytes, alignment);
    for (plan.spans) |s| {
        const dst = region[@intCast(s.mem_offset)..][0..@intCast(s.len)];
        const direct = @min(s.len, direct_end -| s.file_offset);
        if (direct > 0) {
            const e = tf_reader_read(r.?, s.file_offset, @intCast(direct), dst.ptr, null, null, 0);
            if (e != 0) {
                std.debug.print("tf_reader_read at {d}: error {d}\n", .{ s.file_offset, -e });
                return error.ReadFailed;
            }
        }
        if (direct < s.len) {
            const tail_end = @min(s.file_offset + s.len, file.bytes);
            try file.readAt(s.file_offset + direct, dst[@intCast(direct)..][0..@intCast(tail_end - s.file_offset - direct)]);
        }
    }
    return info;
}

pub fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
