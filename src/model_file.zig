//! Model files (TASK-08, TASK-10): opening a GGUF or ModelOpt NVFP4
//! safetensors checkpoint, parsing its header into a `loader.Model`, and
//! reading a `loader.Plan` into memory through the TASK-03 io_uring O_DIRECT
//! reader, or on Windows its unbuffered ReadFile twin (TASK-40). Shared by `tfload` and `olmoe`.
const std = @import("std");
const c = std.c;
const is_windows = @import("builtin").os.tag == .windows;
const win32 = @import("win32.zig");
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
    fd: Fd,
    bytes: u64,

    pub const Fd = if (is_windows) win32.HANDLE else c.fd_t;

    pub fn open(path: [:0]const u8) !File {
        if (is_windows)
        {
            const h = win32.openFile(path, win32.GENERIC_READ, win32.OPEN_EXISTING) orelse {
                std.debug.print("cannot open {s}\n", .{path});
                return error.FileNotFound;
            };
            var size: i64 = 0;
            if (win32.GetFileSizeEx(h, &size) == 0)
            {
                _ = win32.CloseHandle(h);
                return error.Unexpected;
            }
            return .{ .fd = h, .bytes = @intCast(size) };
        }
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
        if (is_windows)
        {
            _ = win32.CloseHandle(self.fd);
            return;
        }
        _ = c.close(self.fd);
    }

    pub fn readAt(self: File, offset: u64, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            if (is_windows)
            {
                const got = win32.readAt(self.fd, offset + done, dst[done..]) orelse return error.ShortRead;
                if (got == 0)
                {
                    return error.ShortRead;
                }
                done += got;
                continue;
            }
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
    /// A split GGUF's first part, whose metadata the header's keys point into.
    meta_prefix: ?[]u8 = null,

    pub fn deinit(self: *ModelState, gpa: std.mem.Allocator) void {
        self.model.deinit(gpa);
        if (self.gguf_header) |*h| h.deinit(gpa);
        if (self.st_header) |*h| h.deinit();
        gpa.free(self.prefix);
        if (self.meta_prefix) |p|
        {
            gpa.free(p);
        }
    }
};

/// Reads a GGUF header from the start of `file`, reading more until it fits.
fn readGgufHeader(gpa: std.mem.Allocator, file: File) !struct { h: gguf.Header, prefix: []u8 } {
    var want: u64 = 4 << 20;
    while (true) {
        const n = @min(want, file.bytes);
        const prefix = try gpa.alloc(u8, @intCast(n));
        try file.readAt(0, prefix);
        const h = gguf.parse(gpa, prefix, file.bytes) catch |err| {
            gpa.free(prefix);
            if (err == error.Truncated and n < file.bytes) {
                want *= 4;
                continue;
            }
            return err;
        };
        return .{ .h = h, .prefix = prefix };
    }
}

/// The path of part `part` of the split GGUF that `path` belongs to (`...-00002-of-00002.gguf`).
fn splitPart(gpa: std.mem.Allocator, path: []const u8, part: u32) ![:0]u8 {
    const at = std.mem.lastIndexOf(u8, path, "-of-") orelse return error.NotSplitName;
    if (at < 5)
    {
        return error.NotSplitName;
    }
    const out = try gpa.dupeSentinel(u8, path, 0);
    _ = std.fmt.bufPrint(out[at - 5 .. at], "{d:0>5}", .{part}) catch return error.NotSplitName;
    return out;
}

/// llama.cpp splits a large GGUF so that the first part holds the metadata and
/// later parts the tensors. Given the part holding every tensor, adds the
/// first part's keys to `h`; the engine then reads weights from this one file.
fn addSplitMetadata(gpa: std.mem.Allocator, path: []const u8, h: *gguf.Header) !?[]u8 {
    const count = h.getInt("split.count") orelse return null;
    if (count < 2 or h.getString("general.architecture") != null)
    {
        return null;
    }
    if (h.getInt("split.tensors.count")) |all|
    {
        if (all != h.tensors.len)
        {
            std.debug.print("{s}: holds {d} of the model's {d} tensors; the engine needs them all in one part\n", .{ path, h.tensors.len, all });
            return error.SplitTensors;
        }
    }
    const first = try splitPart(gpa, path, 1);
    defer gpa.free(first);
    const f = try File.open(first);
    defer f.close();
    var meta = try readGgufHeader(gpa, f);
    defer meta.h.deinit(gpa);
    errdefer gpa.free(meta.prefix);
    if (meta.h.tensors.len != 0)
    {
        return error.SplitTensors;
    }
    const kvs = try gpa.alloc(gguf.KV, meta.h.kvs.len + h.kvs.len);
    @memcpy(kvs[0..meta.h.kvs.len], meta.h.kvs);
    @memcpy(kvs[meta.h.kvs.len..], h.kvs);
    gpa.free(h.kvs);
    h.kvs = kvs;
    return meta.prefix;
}

/// Reads as much of the file start as the header needs and builds the model.
/// For a split GGUF, `path` names the part with the tensors.
pub fn openModel(gpa: std.mem.Allocator, path: []const u8, file: File, config_path: ?[]const u8) !ModelState {
    var magic: [8]u8 = undefined;
    try file.readAt(0, &magic);
    if (std.mem.eql(u8, magic[0..4], "GGUF")) {
        var got = try readGgufHeader(gpa, file);
        errdefer gpa.free(got.prefix);
        errdefer got.h.deinit(gpa);
        if (got.h.tensors.len == 0 and (got.h.getInt("split.count") orelse 1) > 1)
        {
            std.debug.print("{s} is the metadata part of a split GGUF; pass the part with the tensors\n", .{path});
            return error.SplitTensors;
        }
        const meta = try addSplitMetadata(gpa, path, &got.h);
        errdefer if (meta) |p| gpa.free(p);
        return .{ .kind = "GGUF", .model = try loader.Model.fromGguf(gpa, got.h), .prefix = got.prefix, .gguf_header = got.h, .meta_prefix = meta };
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
    // The first layer with experts: leading dense layers (DeepSeek, GLM) have none.
    var layer: u32 = 0;
    while (layer + 1 < m.layers and m.weight(layer, .gate, 0).parts[0].len == 0)
    {
        layer += 1;
    }
    var expert_bytes: u64 = 0;
    for ([_]loader.Projection{ .gate, .up, .down }) |p|
    {
        for (m.weight(layer, p, 0).parts) |r|
        {
            expert_bytes += r.len;
        }
    }
    std.debug.print("{s}: {s}, {d} layers x {d} experts, d_model {d}, d_ff {d}; gate/up/down {t}/{t}/{t}; {d:.2} MB per expert per layer\n", .{
        std.fs.path.basename(path), kind, m.layers, m.experts, m.d_model, m.d_ff,
        m.weight(layer, .gate, 0).format, m.weight(layer, .up, 0).format, m.weight(layer, .down, 0).format, @as(f64, @floatFromInt(expert_bytes)) / 1e6,
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

/// Reads single byte ranges into a fixed memory region through the io_uring
/// O_DIRECT reader (TASK-11's expert cache), widening each to whole blocks.
pub const DirectSource = struct {
    reader: *Reader,
    file: File,
    direct_end: u64,

    pub fn open(path: [:0]const u8, file: File, region: []u8) !DirectSource
    {
        var r: ?*Reader = null;
        const ropts: ReaderOpts = .{ .queue_depth = 16, .block_bytes = 1 << 20, .sqpoll = 0 };
        const rc = tf_reader_open(&r, path, &ropts, region.ptr, region.len);
        if (rc != 0)
        {
            std.debug.print("tf_reader_open: error {d}\n", .{-rc});
            return error.ReaderOpenFailed;
        }
        var info: ReaderInfo = undefined;
        tf_reader_get_info(r.?, &info);
        if (alignment % info.dio_offset_align != 0 or alignment % info.dio_mem_align != 0)
        {
            tf_reader_close(r.?);
            return error.UnsupportedAlignment;
        }
        return .{ .reader = r.?, .file = file, .direct_end = std.mem.alignBackward(u64, file.bytes, alignment) };
    }

    pub fn close(self: *DirectSource) void
    {
        tf_reader_close(self.reader);
    }

    /// A range needs this many bytes beyond its length in `dst`.
    pub const pad = 2 * alignment;

    pub fn read(ctx: *anyopaque, r: loader.Range, dst: []u8) anyerror!usize
    {
        const self: *DirectSource = @ptrCast(@alignCast(ctx));
        const start = std.mem.alignBackward(u64, r.offset, alignment);
        const end = std.mem.alignForward(u64, r.offset + r.len, alignment);
        const at: usize = @intCast(r.offset - start);
        if (end > self.direct_end)
        {
            // The file's last partial block cannot be read directly.
            try self.file.readAt(r.offset, dst[at..][0..@intCast(r.len)]);
            return at;
        }
        const rc = tf_reader_read(self.reader, start, @intCast(end - start), dst.ptr, null, null, 0);
        if (rc != 0)
        {
            std.debug.print("tf_reader_read at {d}: error {d}\n", .{ start, -rc });
            return error.ReadFailed;
        }
        return at;
    }
};

pub fn nowNs() u64 {
    if (is_windows)
    {
        return win32.nowNs();
    }
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
