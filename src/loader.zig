//! Expert loader (TASK-08): describes every expert weight in a GGUF or
//! safetensors MoE checkpoint, checks the shapes against the model metadata,
//! plans aligned reads for only the experts this process owns, and checks each
//! loaded expert against a CRC32C index (`.tfsum`, written once per file by
//! `tfload index`, since neither format stores checksums).
//!
//! GGUF (llama.cpp layout): each layer stores all experts of one projection in
//! a single 3-D tensor `blk.N.ffn_{gate,up,down}_exps.weight`, dims
//! [row_len, rows, experts]; the expert index is outermost, so one expert is
//! one contiguous byte range.
//!
//! safetensors (NVIDIA ModelOpt NVFP4 export, HF naming): one tensor per
//! expert, `model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight` (packed
//! E2M1), plus `.weight_scale` (E4M3 per 16 values) and `.weight_scale_2`
//! (one f32).
const std = @import("std");
const gguf = @import("gguf.zig");
const safetensors = @import("safetensors.zig");
const quant = @import("quant.zig");

extern fn tf_crc32c(data: *const anyopaque, len: usize) u32;

pub const Error = error{ NotMoe, BadShape, BadChecksum, MissingChecksum, BadIndex, OutOfMemory };

pub const Projection = enum { gate, up, down };

pub const Range = struct {
    offset: u64 = 0,
    len: u64 = 0,
};

pub const Layout = enum {
    /// A ggml block format (`ExpertWeight.format`), one byte range.
    gguf,
    /// ModelOpt NVFP4: packed values, block scales, and a tensor scale.
    modelopt_nvfp4,
};

pub const ExpertWeight = struct {
    layer: u32,
    proj: Projection,
    expert: u32,
    /// Output features x input features, row-major (rows of `cols` values).
    rows: u64,
    cols: u64,
    layout: Layout,
    format: quant.Format,
    /// Data, block scales, tensor scale; unused parts have length 0.
    parts: [3]Range,

    pub fn values(self: ExpertWeight) u64 {
        return self.rows * self.cols;
    }

    /// `bytes[i]` holds `parts[i]`.
    pub fn dequantize(self: ExpertWeight, bytes: [3][]const u8, out: []f32) !void {
        switch (self.layout) {
            .gguf => try quant.dequantize(self.format, bytes[0], out),
            .modelopt_nvfp4 => {
                if (bytes[2].len != 4) return error.SizeMismatch;
                const global: f32 = @bitCast(std.mem.readInt(u32, bytes[2][0..4], .little));
                try quant.dequantizeModelOptNvfp4(bytes[0], bytes[1], global, @intCast(self.rows), @intCast(self.cols), out);
            },
        }
    }
};

pub const Model = struct {
    layers: u32,
    experts: u32,
    d_model: u64,
    d_ff: u64,
    /// Index ((layer * 3) + projection) * experts + expert.
    weights: []ExpertWeight,

    pub fn deinit(self: *Model, gpa: std.mem.Allocator) void {
        gpa.free(self.weights);
        self.* = undefined;
    }

    pub fn weight(self: Model, layer: u32, proj: Projection, expert: u32) ExpertWeight {
        return self.weights[(@as(usize, layer) * 3 + @backingInt(proj)) * self.experts + expert];
    }

    /// Gate and up are [d_ff rows x d_model], down is [d_model rows x d_ff].
    fn shapeOf(self: Model, proj: Projection) [2]u64 {
        return if (proj == .down) .{ self.d_model, self.d_ff } else .{ self.d_ff, self.d_model };
    }

    fn alloc(gpa: std.mem.Allocator, layers: u64, experts: u64, d_model: u64, d_ff: u64) Error!Model {
        if (experts == 0 or experts > 1024 or layers == 0 or layers > 1024 or d_model == 0 or d_ff == 0) return error.BadShape;
        return .{
            .layers = @intCast(layers),
            .experts = @intCast(experts),
            .d_model = d_model,
            .d_ff = d_ff,
            .weights = try gpa.alloc(ExpertWeight, @intCast(layers * 3 * experts)),
        };
    }

    /// Reads the shape from `{arch}.expert_count`, `block_count`,
    /// `embedding_length`, and `expert_feed_forward_length`, and checks every
    /// stacked expert tensor against it. Older conversions of all-MoE models
    /// (allenai's OLMoE GGUF files) give the expert width only as
    /// `feed_forward_length`, which llama.cpp also reads that way.
    pub fn fromGguf(gpa: std.mem.Allocator, h: gguf.Header) Error!Model {
        const arch = h.getString("general.architecture") orelse return error.NotMoe;
        var key_buf: [128]u8 = undefined;
        const key = struct {
            fn f(buf: []u8, a: []const u8, k: []const u8) []const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ a, k }) catch unreachable;
            }
        }.f;
        var m = try alloc(
            gpa,
            h.getInt(key(&key_buf, arch, "block_count")) orelse return error.NotMoe,
            h.getInt(key(&key_buf, arch, "expert_count")) orelse return error.NotMoe,
            h.getInt(key(&key_buf, arch, "embedding_length")) orelse return error.NotMoe,
            h.getInt(key(&key_buf, arch, "expert_feed_forward_length")) orelse
                h.getInt(key(&key_buf, arch, "feed_forward_length")) orelse return error.NotMoe,
        );
        errdefer m.deinit(gpa);
        for (0..m.layers) |l| {
            for ([_]Projection{ .gate, .up, .down }) |p| {
                var name_buf: [64]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_{t}_exps.weight", .{ l, p }) catch unreachable;
                const t = h.tensor(name) orelse return error.BadShape;
                const s = m.shapeOf(p);
                const want = [3]u64{ s[1], s[0], m.experts };
                if (t.ndim != 3 or !std.mem.eql(u64, t.dims[0..3], &want)) {
                    std.log.warn("{s}: dims {any}, expected {any} from metadata", .{ name, t.dims[0..t.ndim], want });
                    return error.BadShape;
                }
                const per = t.bytes / m.experts;
                for (0..m.experts) |e| {
                    m.weights[(l * 3 + @backingInt(p)) * m.experts + e] = .{
                        .layer = @intCast(l),
                        .proj = p,
                        .expert = @intCast(e),
                        .rows = s[0],
                        .cols = s[1],
                        .layout = .gguf,
                        .format = t.format,
                        .parts = .{ .{ .offset = t.offset + e * per, .len = per }, .{}, .{} },
                    };
                }
            }
        }
        return m;
    }

    /// Finds the per-expert ModelOpt NVFP4 tensors. The shape comes from the
    /// checkpoint's `config.json` (`num_hidden_layers`, `num_experts` or
    /// `num_local_experts`, `hidden_size`, `moe_intermediate_size` or
    /// `intermediate_size`), and every tensor is checked against it.
    pub fn fromSafetensors(gpa: std.mem.Allocator, h: safetensors.Header, config_json: []const u8) (Error || error{BadConfig})!Model {
        var parsed = std.json.parseFromSlice(std.json.Value, gpa, config_json, .{}) catch return error.BadConfig;
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const cfg = parsed.value.object;
        const int = struct {
            fn f(o: std.json.ObjectMap, names: []const []const u8) ?u64 {
                for (names) |n| if (o.get(n)) |v| if (v == .integer and v.integer > 0) return @intCast(v.integer);
                return null;
            }
        }.f;
        var m = try alloc(
            gpa,
            int(cfg, &.{"num_hidden_layers"}) orelse return error.BadConfig,
            int(cfg, &.{ "num_experts", "num_local_experts" }) orelse return error.BadConfig,
            int(cfg, &.{"hidden_size"}) orelse return error.BadConfig,
            int(cfg, &.{ "moe_intermediate_size", "intermediate_size" }) orelse return error.BadConfig,
        );
        errdefer m.deinit(gpa);
        for (m.weights, 0..) |*w, i| {
            const e: u32 = @intCast(i % m.experts);
            const p: Projection = @fromBackingInt(@intCast((i / m.experts) % 3));
            const l: u32 = @intCast(i / m.experts / 3);
            const s = m.shapeOf(p);
            var buf: [160]u8 = undefined;
            const base = std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{t}_proj.weight", .{ l, e, p }) catch unreachable;
            const data = h.tensor(base) orelse return error.BadShape;
            var sbuf: [176]u8 = undefined;
            const scale = h.tensor(std.fmt.bufPrint(&sbuf, "{s}_scale", .{base}) catch unreachable) orelse return error.BadShape;
            const global = h.tensor(std.fmt.bufPrint(&sbuf, "{s}_scale_2", .{base}) catch unreachable) orelse return error.BadShape;
            if (s[1] % 16 != 0 or
                data.dtype != .u8 or data.ndim != 2 or data.shape[0] != s[0] or data.shape[1] != s[1] / 2 or
                scale.dtype != .f8_e4m3 or scale.ndim != 2 or scale.shape[0] != s[0] or scale.shape[1] != s[1] / 16 or
                global.dtype != .f32 or global.bytes != 4)
            {
                std.log.warn("{s}: shape or dtype does not match config.json", .{base});
                return error.BadShape;
            }
            w.* = .{
                .layer = l,
                .proj = p,
                .expert = e,
                .rows = s[0],
                .cols = s[1],
                .layout = .modelopt_nvfp4,
                .format = .nvfp4,
                .parts = .{
                    .{ .offset = data.offset, .len = data.bytes },
                    .{ .offset = scale.offset, .len = scale.bytes },
                    .{ .offset = global.offset, .len = global.bytes },
                },
            };
        }
        return m;
    }
};

/// The reads needed to bring a set of expert weights into memory. Byte ranges
/// are merged (bridging gaps up to `max_gap`) and widened to `alignment`, as
/// O_DIRECT requires; each span lands at `mem_offset` in the caller's region.
pub const Plan = struct {
    spans: []Span,
    /// Region size needed, a multiple of `alignment`.
    region_bytes: u64,
    /// Bytes the selected weights occupy, before merging and alignment.
    wanted_bytes: u64,

    pub const Span = struct { file_offset: u64, len: u64, mem_offset: u64 };

    pub fn deinit(self: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(self.spans);
        self.* = undefined;
    }

    pub fn make(gpa: std.mem.Allocator, weights: []const ExpertWeight, alignment: u64, max_gap: u64, file_bytes: u64) Error!Plan {
        var ranges: std.ArrayList(Range) = .empty;
        defer ranges.deinit(gpa);
        for (weights) |w| for (w.parts) |r| if (r.len > 0) try ranges.append(gpa, r);
        return fromRanges(gpa, ranges.items, alignment, max_gap, file_bytes);
    }

    /// Like `make`, for any byte ranges (the order of `ranges` is changed).
    pub fn fromRanges(gpa: std.mem.Allocator, ranges_in: []Range, alignment: u64, max_gap: u64, file_bytes: u64) Error!Plan {
        std.debug.assert(std.math.isPowerOfTwo(alignment));
        var wanted: u64 = 0;
        for (ranges_in) |r| wanted += r.len;
        const ranges: struct { items: []Range } = .{ .items = ranges_in };
        std.mem.sort(Range, ranges.items, {}, struct {
            fn lt(_: void, a: Range, b: Range) bool {
                return a.offset < b.offset;
            }
        }.lt);

        var spans: std.ArrayList(Span) = .empty;
        errdefer spans.deinit(gpa);
        var mem: u64 = 0;
        for (ranges.items) |r| {
            const start = std.mem.alignBackward(u64, r.offset, alignment);
            // Reads may run past the end of the file only up to the alignment.
            const end = @min(std.mem.alignForward(u64, r.offset + r.len, alignment), std.mem.alignForward(u64, file_bytes, alignment));
            if (spans.items.len > 0) {
                const last = &spans.items[spans.items.len - 1];
                const last_end = last.file_offset + last.len;
                if (start <= last_end + max_gap) {
                    if (end > last_end) {
                        mem += end - last_end;
                        last.len = end - last.file_offset;
                    }
                    continue;
                }
            }
            try spans.append(gpa, .{ .file_offset = start, .len = end - start, .mem_offset = mem });
            mem += end - start;
        }
        return .{ .spans = try spans.toOwnedSlice(gpa), .region_bytes = mem, .wanted_bytes = wanted };
    }

    /// Where `r` landed in the region, once every span has been read.
    pub fn locate(self: Plan, region: []const u8, r: Range) []const u8 {
        var lo: usize = 0;
        var hi: usize = self.spans.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (self.spans[mid].file_offset <= r.offset) lo = mid else hi = mid;
        }
        const s = self.spans[lo];
        std.debug.assert(r.offset >= s.file_offset and r.offset + r.len <= s.file_offset + s.len);
        return region[@intCast(s.mem_offset + (r.offset - s.file_offset))..][0..@intCast(r.len)];
    }

    /// The three part slices of `w`.
    pub fn partsOf(self: Plan, region: []const u8, w: ExpertWeight) [3][]const u8 {
        var out: [3][]const u8 = .{ &.{}, &.{}, &.{} };
        for (w.parts, 0..) |r, i| if (r.len > 0) {
            out[i] = self.locate(region, r);
        };
        return out;
    }
};

/// The `.tfsum` checksum index: a header line `tfsum 1 <file bytes>`, then one
/// line per byte range, `<offset> <len> <crc32c hex>`, sorted by offset.
pub const Index = struct {
    file_bytes: u64,
    map: std.AutoHashMapUnmanaged(u64, Entry),

    pub const Entry = struct { len: u64, crc: u32 };

    pub fn deinit(self: *Index, gpa: std.mem.Allocator) void {
        self.map.deinit(gpa);
        self.* = undefined;
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) Error!Index {
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        const head = lines.next() orelse return error.BadIndex;
        var hw = std.mem.tokenizeScalar(u8, head, ' ');
        if (!std.mem.eql(u8, hw.next() orelse "", "tfsum") or !std.mem.eql(u8, hw.next() orelse "", "1")) return error.BadIndex;
        var idx: Index = .{
            .file_bytes = std.fmt.parseInt(u64, hw.next() orelse "", 10) catch return error.BadIndex,
            .map = .empty,
        };
        errdefer idx.map.deinit(gpa);
        while (lines.next()) |line| {
            var f = std.mem.tokenizeScalar(u8, line, ' ');
            const off = std.fmt.parseInt(u64, f.next() orelse return error.BadIndex, 10) catch return error.BadIndex;
            const len = std.fmt.parseInt(u64, f.next() orelse return error.BadIndex, 10) catch return error.BadIndex;
            const crc = std.fmt.parseInt(u32, f.next() orelse return error.BadIndex, 16) catch return error.BadIndex;
            try idx.map.put(gpa, off, .{ .len = len, .crc = crc });
        }
        return idx;
    }

    /// Checks every part of `weights` (already loaded per `plan`).
    pub fn verify(self: Index, plan: Plan, region: []const u8, weights: []const ExpertWeight) Error!void {
        for (weights) |w| for (w.parts) |r| if (r.len > 0) {
            const e = self.map.get(r.offset) orelse return error.MissingChecksum;
            if (e.len != r.len) return error.MissingChecksum;
            const bytes = plan.locate(region, r);
            if (tf_crc32c(bytes.ptr, bytes.len) != e.crc) {
                std.log.warn("layer {d} {t} expert {d}: CRC32C mismatch at offset {d}", .{ w.layer, w.proj, w.expert, r.offset });
                return error.BadChecksum;
            }
        };
    }
};

pub fn crc32c(bytes: []const u8) u32 {
    return tf_crc32c(bytes.ptr, bytes.len);
}

/// Writes index lines for `weights`, whose bytes are loaded per `plan`.
pub fn writeIndexLines(w: *std.Io.Writer, plan: Plan, region: []const u8, weights: []const ExpertWeight) !void {
    for (weights) |ew| for (ew.parts) |r| if (r.len > 0) {
        try w.print("{d} {d} {x:0>8}\n", .{ r.offset, r.len, crc32c(plan.locate(region, r)) });
    };
}

// ---------------------------------------------------------------- tests

const tfref = @import("tfref.zig");

fn expectMatchesGolden(m: Model, file: []const u8, golden: tfref.File, owned: []const bool) !void {
    const gpa = std.testing.allocator;
    var selected: std.ArrayList(ExpertWeight) = .empty;
    defer selected.deinit(gpa);
    for (m.weights) |w| if (owned[w.expert]) try selected.append(gpa, w);

    var plan = try Plan.make(gpa, selected.items, 64, 0, file.len);
    defer plan.deinit(gpa);
    const region = try gpa.alloc(u8, @intCast(plan.region_bytes));
    defer gpa.free(region);
    for (plan.spans) |s| {
        const end = @min(s.file_offset + s.len, file.len);
        @memset(region[@intCast(s.mem_offset)..][0..@intCast(s.len)], 0);
        @memcpy(region[@intCast(s.mem_offset)..][0..@intCast(end - s.file_offset)], file[@intCast(s.file_offset)..@intCast(end)]);
    }
    // Only owned experts were read: the plan is smaller than all experts.
    var all: u64 = 0;
    for (m.weights) |w| for (w.parts) |r| {
        all += r.len;
    };
    try std.testing.expect(plan.wanted_bytes < all);

    for (selected.items) |w| {
        const out = try gpa.alloc(f32, @intCast(w.values()));
        defer gpa.free(out);
        try w.dequantize(plan.partsOf(region, w), out);
        var name_buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "golden.layer{d}.{t}.expert{d}", .{ w.layer, w.proj, w.expert });
        const want = (try golden.get(name)).f32s();
        try std.testing.expectEqual(want.len, out.len);
        for (out, want) |got, x| try std.testing.expectEqual(x, got);
    }
}

test "GGUF fixture: every tensor decodes to gguf-py's values" {
    const gpa = std.testing.allocator;
    const file = @embedFile("quant_tiny.gguf");
    const golden = try tfref.File.parse(@embedFile("quant_tiny.tfref"));
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), h.tensors.len);
    for (h.tensors) |t| {
        const out = try gpa.alloc(f32, @intCast(t.elements()));
        defer gpa.free(out);
        try quant.dequantize(t.format, file[@intCast(t.offset)..][0..@intCast(t.bytes)], out);
        var name_buf: [96]u8 = undefined;
        const want = (try golden.get(try std.fmt.bufPrint(&name_buf, "golden.{s}", .{t.name}))).f32s();
        try std.testing.expectEqual(want.len, out.len);
        for (out, want) |got, w| try std.testing.expectEqual(w, got);
    }
}

test "GGUF fixture: loading experts 1 and 2 of 4 gives gguf-py's values" {
    const gpa = std.testing.allocator;
    const file = @embedFile("quant_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    var m = try Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 4), m.experts);
    try std.testing.expectEqual(quant.Format.nvfp4, m.weight(0, .down, 0).format);
    try expectMatchesGolden(m, file, try tfref.File.parse(@embedFile("quant_tiny.tfref")), &.{ false, true, true, false });
}

test "safetensors fixture: loading experts 0 and 3 of 4 gives the reference values" {
    const gpa = std.testing.allocator;
    const file = @embedFile("nvfp4_tiny.safetensors");
    var h = try safetensors.parse(gpa, file, file.len);
    defer h.deinit();
    var m = try Model.fromSafetensors(gpa, h, @embedFile("nvfp4_tiny.config.json"));
    defer m.deinit(gpa);
    try std.testing.expectEqual(Layout.modelopt_nvfp4, m.weight(0, .gate, 0).layout);
    try expectMatchesGolden(m, file, try tfref.File.parse(@embedFile("nvfp4_tiny.tfref")), &.{ true, false, false, true });
}

test "a shape that disagrees with the metadata is rejected" {
    std.testing.log_level = .err; // the warning is the expected outcome
    defer std.testing.log_level = .warn;
    const gpa = std.testing.allocator;
    const file = @embedFile("quant_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    h.tensors[0].dims[2] = 3; // gate tensor now claims 3 experts
    try std.testing.expectError(error.BadShape, Model.fromGguf(gpa, h));
}

test "a header cut short asks for more bytes" {
    const gpa = std.testing.allocator;
    const file = @embedFile("quant_tiny.gguf");
    try std.testing.expectError(error.Truncated, gguf.parse(gpa, file[0..300], file.len));
    try std.testing.expectError(error.BadTensor, gguf.parse(gpa, file, file.len - 1)); // last tensor runs past the end
}

test "plan merges adjacent experts and aligns spans" {
    const gpa = std.testing.allocator;
    const w = struct {
        fn f(off: u64, len: u64) ExpertWeight {
            return .{ .layer = 0, .proj = .gate, .expert = 0, .rows = 1, .cols = 1, .layout = .gguf, .format = .f32, .parts = .{ .{ .offset = off, .len = len }, .{}, .{} } };
        }
    }.f;
    var plan = try Plan.make(gpa, &.{ w(5000, 1000), w(6000, 1000), w(100_000, 10) }, 4096, 0, 200_000);
    defer plan.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), plan.spans.len);
    try std.testing.expectEqual(@as(u64, 4096), plan.spans[0].file_offset);
    try std.testing.expectEqual(@as(u64, 4096), plan.spans[0].len); // 4096..8192 covers 5000..7000
    try std.testing.expectEqual(@as(u64, 98304), plan.spans[1].file_offset);
    try std.testing.expectEqual(@as(u64, 4096), plan.spans[1].mem_offset);
    try std.testing.expectEqual(@as(u64, 2010), plan.wanted_bytes);
}

test "checksum index catches a corrupted expert" {
    std.testing.log_level = .err; // the warning is the expected outcome
    defer std.testing.log_level = .warn;
    const gpa = std.testing.allocator;
    const file = @embedFile("quant_tiny.gguf");
    var h = try gguf.parse(gpa, file, file.len);
    defer h.deinit(gpa);
    var m = try Model.fromGguf(gpa, h);
    defer m.deinit(gpa);
    var plan = try Plan.make(gpa, m.weights, 1, 0, file.len);
    defer plan.deinit(gpa);
    const region = try gpa.alloc(u8, @intCast(plan.region_bytes));
    defer gpa.free(region);
    for (plan.spans) |s| @memcpy(region[@intCast(s.mem_offset)..][0..@intCast(s.len)], file[@intCast(s.file_offset)..][0..@intCast(s.len)]);

    var text: std.Io.Writer.Allocating = .init(gpa);
    defer text.deinit();
    try text.writer.print("tfsum 1 {d}\n", .{file.len});
    try writeIndexLines(&text.writer, plan, region, m.weights);
    var idx = try Index.parse(gpa, text.written());
    defer idx.deinit(gpa);
    try idx.verify(plan, region, m.weights);
    region[plan.locate(region, m.weight(0, .up, 2).parts[0]).ptr - region.ptr + 5] ^= 1;
    try std.testing.expectError(error.BadChecksum, idx.verify(plan, region, m.weights));
}
