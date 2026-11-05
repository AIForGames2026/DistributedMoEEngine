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
const loader = @import("loader.zig");
const tfref = @import("tfref.zig");
const Arena = @import("arena.zig").Arena;
const mf = @import("model_file.zig");
const File = mf.File;
const alignment = mf.alignment;
const max_gap = mf.max_gap;
const readWhole = mf.readWhole;
const readPlan = mf.readPlan;
const nowNs = mf.nowNs;

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
    var model_state = try mf.openModel(gpa, file, opts.config);
    defer model_state.deinit(gpa);
    const m = model_state.model;
    mf.describe(m, opts.model, model_state.kind);

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
