//! Expert loader tool (TASK-08). Linux: reads through the TASK-03 io_uring
//! O_DIRECT reader.
//!
//!   tfload info  MODEL [--config config.json]
//!   tfload index MODEL [--config config.json]      writes MODEL.tfsum
//!   tfload load  MODEL --experts A-B [--config config.json] [--golden G.tfref]
//!
//! MODEL is a GGUF file or a single-file ModelOpt NVFP4 safetensors checkpoint
//! (which needs its config.json). `load` reads only experts A to B (inclusive)
//! of every layer, checks each against MODEL.tfsum, and, with `--golden`,
//! compares the dequantized weights with values from tools/ref/ref_quant.py.
const std = @import("std");
const c = std.c;
const gguf = @import("gguf.zig");
const safetensors = @import("safetensors.zig");
const loader = @import("loader.zig");
const tfref = @import("tfref.zig");
const Arena = @import("arena.zig").Arena;

const Reader = opaque {};
const ReaderOpts = extern struct { queue_depth: c_uint, block_bytes: usize, sqpoll: c_int };
const ReaderInfo = extern struct {
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
const alignment = 4096;
/// Gaps up to this size between wanted ranges are read rather than skipped.
const max_gap = 64 * 1024;

const Options = struct {
    cmd: []const u8,
    model: [:0]const u8,
    config: ?[]const u8 = null,
    golden: ?[]const u8 = null,
    first: u32 = 0,
    last: ?u32 = null,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const opts = parseArgs(init) catch {
        std.debug.print(
            \\usage: tfload info  MODEL [--config config.json]
            \\       tfload index MODEL [--config config.json]
            \\       tfload load  MODEL --experts A-B [--config config.json] [--golden G.tfref]
            \\
        , .{});
        std.process.exit(2);
    };

    const file = try File.open(opts.model);
    defer file.close();
    var model_state = try openModel(gpa, file, opts);
    defer model_state.deinit(gpa);
    const m = model_state.model;
    describe(m, opts.model, model_state.kind);

    if (std.mem.eql(u8, opts.cmd, "info")) return;
    if (std.mem.eql(u8, opts.cmd, "index")) return writeIndex(gpa, file, m, opts.model);
    if (std.mem.eql(u8, opts.cmd, "load")) return load(gpa, file, m, opts);
    std.debug.print("unknown command '{s}'\n", .{opts.cmd});
    std.process.exit(2);
}

fn parseArgs(init: std.process.Init) !Options {
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    var opts: Options = .{ .cmd = it.next() orelse return error.Usage, .model = it.next() orelse return error.Usage };
    while (it.next()) |arg| {
        const value = it.next() orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--config")) {
            opts.config = value;
        } else if (std.mem.eql(u8, arg, "--golden")) {
            opts.golden = value;
        } else if (std.mem.eql(u8, arg, "--experts")) {
            var parts = std.mem.splitScalar(u8, value, '-');
            opts.first = try std.fmt.parseInt(u32, parts.next().?, 10);
            opts.last = try std.fmt.parseInt(u32, parts.next() orelse return error.Usage, 10);
            if (opts.last.? < opts.first) return error.Usage;
        } else return error.Usage;
    }
    return opts;
}

/// A file opened for buffered reads (headers, the end-of-file tail).
const File = struct {
    fd: c.fd_t,
    bytes: u64,

    fn open(path: [:0]const u8) !File {
        const fd = c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) {
            std.debug.print("cannot open {s}\n", .{path});
            return error.FileNotFound;
        }
        const end = c.lseek(fd, 0, c.SEEK.END);
        if (end < 0) return error.Unexpected;
        return .{ .fd = fd, .bytes = @intCast(end) };
    }

    fn close(self: File) void {
        _ = c.close(self.fd);
    }

    fn readAt(self: File, offset: u64, dst: []u8) !void {
        var done: usize = 0;
        while (done < dst.len) {
            const n = c.pread(self.fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
            if (n <= 0) return error.ShortRead;
            done += @intCast(n);
        }
    }
};

fn readWhole(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const f = try File.open(z);
    defer f.close();
    const buf = try gpa.alloc(u8, @intCast(f.bytes));
    errdefer gpa.free(buf);
    try f.readAt(0, buf);
    return buf;
}

const ModelState = struct {
    kind: []const u8,
    model: loader.Model,
    prefix: []u8,
    gguf_header: ?gguf.Header = null,
    st_header: ?safetensors.Header = null,

    fn deinit(self: *ModelState, gpa: std.mem.Allocator) void {
        self.model.deinit(gpa);
        if (self.gguf_header) |*h| h.deinit(gpa);
        if (self.st_header) |*h| h.deinit();
        gpa.free(self.prefix);
    }
};

/// Reads as much of the file start as the header needs and builds the model.
fn openModel(gpa: std.mem.Allocator, file: File, opts: Options) !ModelState {
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
    const config = try readWhole(gpa, opts.config orelse {
        std.debug.print("safetensors checkpoints need --config config.json\n", .{});
        return error.MissingConfig;
    });
    defer gpa.free(config);
    return .{ .kind = "safetensors (ModelOpt NVFP4)", .model = try loader.Model.fromSafetensors(gpa, h, config), .prefix = prefix, .st_header = h };
}

fn describe(m: loader.Model, path: []const u8, kind: []const u8) void {
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
fn readPlan(path: [:0]const u8, file: File, plan: loader.Plan, region: []align(4096) u8) !ReaderInfo {
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

fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn writeIndex(gpa: std.mem.Allocator, file: File, m: loader.Model, path: [:0]const u8) !void {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try out.writer.print("tfsum 1 {d}\n", .{file.bytes});
    const start = nowNs();
    var total: u64 = 0;
    // One (layer, projection) at a time keeps memory to one stacked tensor.
    var group: usize = 0;
    while (group < m.layers * 3) : (group += 1) {
        const ws = m.weights[group * m.experts ..][0..m.experts];
        var plan = try loader.Plan.make(gpa, ws, alignment, max_gap, file.bytes);
        defer plan.deinit(gpa);
        var arena = try Arena.init(@intCast(plan.region_bytes));
        defer arena.deinit();
        const region = arena.buf[0..@intCast(plan.region_bytes)];
        _ = try readPlan(path, file, plan, region);
        try loader.writeIndexLines(&out.writer, plan, region, ws);
        total += plan.wanted_bytes;
    }
    const ns = nowNs() - start;
    const sum_path = try std.fmt.allocPrintSentinel(gpa, "{s}.tfsum", .{path}, 0);
    defer gpa.free(sum_path);
    const fd = c.open(sum_path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0o644));
    if (fd < 0) return error.CannotWriteIndex;
    defer _ = c.close(fd);
    const text = out.written();
    if (c.write(fd, text.ptr, text.len) != @as(isize, @intCast(text.len))) return error.CannotWriteIndex;
    std.debug.print("index: CRC32C of {d} expert weights ({d:.2} GB) in {d:.2} s, written to {s}\n", .{
        m.weights.len, @as(f64, @floatFromInt(total)) / 1e9, @as(f64, @floatFromInt(ns)) / 1e9, sum_path,
    });
}

fn load(gpa: std.mem.Allocator, file: File, m: loader.Model, opts: Options) !void {
    const last = opts.last orelse {
        std.debug.print("load needs --experts A-B\n", .{});
        std.process.exit(2);
    };
    if (last >= m.experts) {
        std.debug.print("--experts: model has experts 0 to {d}\n", .{m.experts - 1});
        std.process.exit(2);
    }
    var selected: std.ArrayList(loader.ExpertWeight) = .empty;
    defer selected.deinit(gpa);
    for (m.weights) |w| if (w.expert >= opts.first and w.expert <= last) try selected.append(gpa, w);

    const sum_path = try std.fmt.allocPrint(gpa, "{s}.tfsum", .{opts.model});
    defer gpa.free(sum_path);
    const sum_text = readWhole(gpa, sum_path) catch {
        std.debug.print("no checksum index {s}; run tfload index first\n", .{sum_path});
        return error.MissingIndex;
    };
    defer gpa.free(sum_text);
    var idx = try loader.Index.parse(gpa, sum_text);
    defer idx.deinit(gpa);
    if (idx.file_bytes != file.bytes) {
        std.debug.print("{s} was made for a {d}-byte file; this one is {d} bytes\n", .{ sum_path, idx.file_bytes, file.bytes });
        return error.BadIndex;
    }

    var plan = try loader.Plan.make(gpa, selected.items, alignment, max_gap, file.bytes);
    defer plan.deinit(gpa);
    var arena = try Arena.init(@intCast(plan.region_bytes));
    defer arena.deinit();
    const region = arena.buf[0..@intCast(plan.region_bytes)];
    // The engine maps its arenas at startup; touch every page now so the timed
    // read measures I/O, not the kernel finding and zeroing pages.
    const ta = nowNs();
    @memset(region, 0);

    const t0 = nowNs();
    const info = try readPlan(opts.model, file, plan, region);
    const t1 = nowNs();
    try idx.verify(plan, region, selected.items);
    const t2 = nowNs();

    const gb = @as(f64, @floatFromInt(plan.region_bytes)) / 1e9;
    std.debug.print("load: experts {d}-{d} of {d}: {d} weights, {d:.2} GB wanted, {d:.2} GB read in {d} spans (of {d:.2} GB file)\n", .{
        opts.first, last, m.experts, selected.items.len, @as(f64, @floatFromInt(plan.wanted_bytes)) / 1e9, gb, plan.spans.len, @as(f64, @floatFromInt(file.bytes)) / 1e9,
    });
    std.debug.print("load: region touched in {d:.3} s; io_uring O_DIRECT read {d:.3} s ({d:.2} GB/s, fixed buffer {d}); CRC32C of every weight OK in {d:.3} s\n", .{
        @as(f64, @floatFromInt(t0 - ta)) / 1e9, @as(f64, @floatFromInt(t1 - t0)) / 1e9, gb / (@as(f64, @floatFromInt(t1 - t0)) / 1e9),
        info.fixed_buffer,                      @as(f64, @floatFromInt(t2 - t1)) / 1e9,
    });

    if (opts.golden) |gpath| {
        const gbytes = try readWhole(gpa, gpath);
        defer gpa.free(gbytes);
        const golden = try tfref.File.parse(gbytes);
        var checked: usize = 0;
        var worst: f32 = 0;
        for (selected.items) |w| {
            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "golden.layer{d}.{t}.expert{d}", .{ w.layer, w.proj, w.expert });
            const want = (golden.get(name) catch continue).f32s();
            const out = try gpa.alloc(f32, @intCast(w.values()));
            defer gpa.free(out);
            try w.dequantize(plan.partsOf(region, w), out);
            if (want.len != out.len) return error.GoldenShapeMismatch;
            for (out, want) |a, b| worst = @max(worst, @abs(a - b));
            checked += 1;
        }
        std.debug.print("load: dequantized {d} expert weights vs {s}: max abs difference {e}\n", .{ checked, std.fs.path.basename(gpath), worst });
        if (checked == 0) return error.NothingToCompare;
        if (worst != 0) return error.GoldenMismatch;
    }
}
