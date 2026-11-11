//! Tiered KV cache (TASK-12). Linux, GPU attention only.
//!
//! Per layer, the first `sinks` positions and a ring holding the most recent
//! `window` positions (plus room for one chunk of new ones) stay in memory.
//! Everything between them, the cold middle, goes to a file on NVMe through
//! O_DIRECT writes, tile by tile, once it falls out of the window. Attention
//! still reads every position, so before each layer's attention the layer's
//! middle is read back into a staging buffer through the TASK-03 io_uring
//! reader, on a thread that reads layer L + 1 while layer L computes.
//!
//! The GPU kernel takes a table of tiles, so it reads each tile wherever it
//! lives and adds positions up in the same order as with the whole cache in
//! memory: the output is bit-identical to an all-RAM run of the same shape.
const std = @import("std");
const c = std.c;
const olmoe = @import("olmoe.zig");

const Reader = opaque {};
const ReaderOpts = extern struct { queue_depth: c_uint, block_bytes: usize, sqpoll: c_int };
extern fn tf_reader_open(out: *?*Reader, path: [*:0]const u8, opts: *const ReaderOpts, region: *anyopaque, region_len: usize) c_int;
extern fn tf_reader_read(r: *Reader, offset: u64, len: usize, dst: *anyopaque, lat_ns: ?[*]u32, lat_count: ?*usize, lat_cap: usize) c_int;
extern fn tf_reader_close(r: *Reader) void;

const tile_positions = olmoe.AttentionJob.tile_positions;

pub const Options = struct {
    /// Positions at the start that never leave memory (attention sinks).
    sinks: usize = 128,
    /// The most recent positions kept in memory.
    window: usize = 8192,
    /// The most new positions one forward pass adds.
    max_chunk: usize,
    /// The cache file; created, and removed on close.
    path: [:0]const u8,
};

pub const Stats = struct {
    /// Middle bytes read back, and the reader thread's time doing it.
    read_bytes: u64 = 0,
    read_ns: u64 = 0,
    /// Time attention waited for a layer's middle.
    wait_ns: u64 = 0,
    write_bytes: u64 = 0,
    write_ns: u64 = 0,
};

pub const TierCache = struct {
    cfg: olmoe.Config,
    opts: Options,
    tile_floats: usize,
    tile_bytes: usize,
    sink_tiles: usize,
    ring_tiles: usize,
    max_tiles: usize,
    /// Tiles below this (from `sink_tiles` on) are in the file, the same for every layer.
    flushed: usize,
    sinks_k: []f32,
    sinks_v: []f32,
    ring_k: []f32,
    ring_v: []f32,
    /// Two staging buffers, each with keys then values for one layer's middle.
    stage: []f32,
    dev_sinks_k: u64 = 0,
    dev_sinks_v: u64 = 0,
    dev_ring_k: u64 = 0,
    dev_ring_v: u64 = 0,
    dev_stage: u64 = 0,
    fd: c.fd_t,
    reader: *Reader,
    thread: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    /// The reader thread's job, published by bumping `request`.
    job_layer: usize = 0,
    job_buf: usize = 0,
    job_tiles: usize = 0,
    request: std.atomic.Value(u64) = .init(0),
    /// Per staging buffer: the request it now holds, and the layer that was.
    done: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) },
    holds: [2]?usize = .{ null, null },
    wanted: [2]u64 = .{ 0, 0 },
    stats: Stats = .{},

    pub fn open(gpa: std.mem.Allocator, cfg: olmoe.Config, cap: usize, opts: Options) !*TierCache
    {
        if (cfg.head_dim != olmoe.AttentionJob.head_dim or olmoe.gpu_register == null or olmoe.gpu_attention == null)
        {
            std.debug.print("kv offload: needs GPU attention (--gpu) and 128-wide heads\n", .{});
            return error.OffloadNeedsGpu;
        }
        const pages = std.heap.page_allocator;
        const kvd = cfg.kvDim();
        const tile_floats = tile_positions * kvd;
        const sink_tiles = (opts.sinks + tile_positions - 1) / tile_positions;
        // The window, one chunk, and partial tiles at both ends.
        const ring_tiles = (opts.window + opts.max_chunk) / tile_positions + 4;
        const max_tiles = (cap + tile_positions - 1) / tile_positions;
        const middle = max_tiles -| (sink_tiles + opts.window / tile_positions);
        const self = try gpa.create(TierCache);
        errdefer gpa.destroy(self);
        const sinks_k = try pages.alloc(f32, cfg.layers * sink_tiles * tile_floats);
        errdefer pages.free(sinks_k);
        const sinks_v = try pages.alloc(f32, sinks_k.len);
        errdefer pages.free(sinks_v);
        const ring_k = try pages.alloc(f32, cfg.layers * ring_tiles * tile_floats);
        errdefer pages.free(ring_k);
        const ring_v = try pages.alloc(f32, ring_k.len);
        errdefer pages.free(ring_v);
        const stage = try pages.alloc(f32, 2 * 2 * @max(middle, 1) * tile_floats);
        errdefer pages.free(stage);

        const fd = c.open(opts.path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .DIRECT = true }, @as(c.mode_t, 0o600));
        if (fd < 0)
        {
            std.debug.print("kv offload: cannot create {s}\n", .{opts.path});
            return error.OffloadFileFailed;
        }
        errdefer _ = c.close(fd);
        const file_bytes = cfg.layers * 2 * max_tiles * tile_floats * @sizeOf(f32);
        if (c.ftruncate(fd, @intCast(file_bytes)) != 0)
        {
            return error.OffloadFileFailed;
        }
        var r: ?*Reader = null;
        const ropts: ReaderOpts = .{ .queue_depth = 32, .block_bytes = 1 << 20, .sqpoll = 0 };
        const rc = tf_reader_open(&r, opts.path, &ropts, stage.ptr, stage.len * @sizeOf(f32));
        if (rc != 0)
        {
            std.debug.print("kv offload: tf_reader_open error {d}\n", .{-rc});
            return error.OffloadFileFailed;
        }
        self.* = .{
            .cfg = cfg,
            .opts = opts,
            .tile_floats = tile_floats,
            .tile_bytes = tile_floats * @sizeOf(f32),
            .sink_tiles = sink_tiles,
            .ring_tiles = ring_tiles,
            .max_tiles = max_tiles,
            .flushed = sink_tiles,
            .sinks_k = sinks_k,
            .sinks_v = sinks_v,
            .ring_k = ring_k,
            .ring_v = ring_v,
            .stage = stage,
            .fd = fd,
            .reader = r.?,
        };
        errdefer tf_reader_close(r.?);
        const reg = olmoe.gpu_register.?;
        self.dev_sinks_k = reg(std.mem.sliceAsBytes(sinks_k)) orelse return error.GpuMapFailed;
        self.dev_sinks_v = reg(std.mem.sliceAsBytes(sinks_v)) orelse return error.GpuMapFailed;
        self.dev_ring_k = reg(std.mem.sliceAsBytes(ring_k)) orelse return error.GpuMapFailed;
        self.dev_ring_v = reg(std.mem.sliceAsBytes(ring_v)) orelse return error.GpuMapFailed;
        self.dev_stage = reg(std.mem.sliceAsBytes(stage)) orelse return error.GpuMapFailed;
        self.thread = try std.Thread.spawn(.{}, work, .{self});
        return self;
    }

    pub fn close(self: *TierCache, gpa: std.mem.Allocator) void
    {
        self.stop.store(true, .release);
        self.thread.join();
        if (olmoe.gpu_unregister) |un|
        {
            un(std.mem.sliceAsBytes(self.sinks_k));
            un(std.mem.sliceAsBytes(self.sinks_v));
            un(std.mem.sliceAsBytes(self.ring_k));
            un(std.mem.sliceAsBytes(self.ring_v));
            un(std.mem.sliceAsBytes(self.stage));
        }
        tf_reader_close(self.reader);
        _ = c.close(self.fd);
        _ = c.unlink(self.opts.path);
        const pages = std.heap.page_allocator;
        pages.free(self.sinks_k);
        pages.free(self.sinks_v);
        pages.free(self.ring_k);
        pages.free(self.ring_v);
        pages.free(self.stage);
        gpa.destroy(self);
    }

    /// Memory the cache holds for positions: sinks, ring, and staging.
    pub fn memoryBytes(self: *const TierCache) usize
    {
        return (self.sinks_k.len + self.sinks_v.len + self.ring_k.len + self.ring_v.len + self.stage.len) * @sizeOf(f32);
    }

    pub fn tier(self: *TierCache) olmoe.KvCache.Tier
    {
        return .{ .ctx = self, .store = store, .tiles = tiles, .begin = begin, .end = end };
    }

    fn stageOffset(self: *const TierCache, buf: usize, kind: usize) usize
    {
        const per = self.stage.len / 4;
        return (buf * 2 + kind) * per;
    }

    fn fileOffset(self: *const TierCache, layer: usize, kind: usize, tile: usize) u64
    {
        return ((@as(u64, layer) * 2 + kind) * self.max_tiles + tile) * self.tile_bytes;
    }

    /// Where tile `ti` of `layer` lives in memory (sinks or ring), as a float offset.
    fn memTile(self: *const TierCache, layer: usize, ti: usize) struct { ring: bool, at: usize }
    {
        if (ti < self.sink_tiles)
        {
            return .{ .ring = false, .at = (layer * self.sink_tiles + ti) * self.tile_floats };
        }
        return .{ .ring = true, .at = (layer * self.ring_tiles + ti % self.ring_tiles) * self.tile_floats };
    }

    fn store(ctx: *anyopaque, layer: usize, pos0: usize, n: usize, k: []const f32, v: []const f32) void
    {
        const self: *TierCache = @ptrCast(@alignCast(ctx));
        const kvd = self.cfg.kvDim();
        for (0..n) |i|
        {
            const p = pos0 + i;
            const ti = p / tile_positions;
            std.debug.assert(ti >= self.flushed or ti < self.sink_tiles);
            std.debug.assert(ti < self.sink_tiles or ti - self.flushed < self.ring_tiles);
            const m = self.memTile(layer, ti);
            const row = m.at + (p % tile_positions) * kvd;
            const dk = if (m.ring) self.ring_k else self.sinks_k;
            const dv = if (m.ring) self.ring_v else self.sinks_v;
            @memcpy(dk[row..][0..kvd], k[i * kvd ..][0..kvd]);
            @memcpy(dv[row..][0..kvd], v[i * kvd ..][0..kvd]);
        }
    }

    fn middleTiles(self: *const TierCache) usize
    {
        return self.flushed - self.sink_tiles;
    }

    /// Asks the reader thread for `layer`'s middle in buffer `layer % 2`.
    fn prefetch(self: *TierCache, layer: usize) void
    {
        const buf = layer % 2;
        self.job_layer = layer;
        self.job_buf = buf;
        self.job_tiles = self.middleTiles();
        const seq = self.request.load(.monotonic) + 1;
        self.wanted[buf] = seq;
        self.holds[buf] = layer;
        self.request.store(seq, .release);
    }

    fn begin(ctx: *anyopaque, len: usize) anyerror!void
    {
        const self: *TierCache = @ptrCast(@alignCast(ctx));
        _ = len;
        self.holds = .{ null, null };
        if (self.middleTiles() > 0)
        {
            self.prefetch(0);
        }
    }

    fn tiles(ctx: *anyopaque, layer: usize, len: usize, out_k: []u64, out_v: []u64) anyerror!void
    {
        const self: *TierCache = @ptrCast(@alignCast(ctx));
        const buf = layer % 2;
        const middle = self.middleTiles();
        if (middle > 0)
        {
            if (self.holds[buf] != layer)
            {
                return error.OffloadOutOfOrder;
            }
            const t0 = nowNs();
            while (self.done[buf].load(.acquire) != self.wanted[buf])
            {
                if (self.failed.load(.acquire))
                {
                    return error.OffloadReadFailed;
                }
                std.atomic.spinLoopHint();
            }
            self.stats.wait_ns += nowNs() - t0;
            if (layer + 1 < self.cfg.layers)
            {
                self.prefetch(layer + 1);
            }
        }
        const fb = @sizeOf(f32);
        for (0..(len + tile_positions - 1) / tile_positions) |ti|
        {
            if (ti >= self.sink_tiles and ti < self.flushed)
            {
                const at = (ti - self.sink_tiles) * self.tile_floats;
                out_k[ti] = self.dev_stage + (self.stageOffset(buf, 0) + at) * fb;
                out_v[ti] = self.dev_stage + (self.stageOffset(buf, 1) + at) * fb;
                continue;
            }
            const m = self.memTile(layer, ti);
            out_k[ti] = (if (m.ring) self.dev_ring_k else self.dev_sinks_k) + m.at * fb;
            out_v[ti] = (if (m.ring) self.dev_ring_v else self.dev_sinks_v) + m.at * fb;
        }
    }

    /// Moves every tile that has left the window to the file.
    fn end(ctx: *anyopaque, len: usize) anyerror!void
    {
        const self: *TierCache = @ptrCast(@alignCast(ctx));
        if (len <= self.opts.window)
        {
            return;
        }
        const target = @max(self.flushed, (len - self.opts.window) / tile_positions);
        if (target <= self.flushed)
        {
            return;
        }
        const t0 = nowNs();
        for (0..self.cfg.layers) |layer|
        {
            for (self.flushed..target) |ti|
            {
                const m = self.memTile(layer, ti);
                for ([_][]f32{ self.ring_k, self.ring_v }, 0..) |src, kind|
                {
                    const bytes = std.mem.sliceAsBytes(src[m.at..][0..self.tile_floats]);
                    var off: usize = 0;
                    while (off < bytes.len)
                    {
                        const n = c.pwrite(self.fd, bytes[off..].ptr, bytes.len - off, @intCast(self.fileOffset(layer, kind, ti) + off));
                        if (n <= 0)
                        {
                            return error.OffloadWriteFailed;
                        }
                        off += @intCast(n);
                    }
                }
            }
        }
        self.stats.write_bytes += (target - self.flushed) * self.cfg.layers * 2 * self.tile_bytes;
        self.stats.write_ns += nowNs() - t0;
        self.flushed = target;
    }

    fn work(self: *TierCache) void
    {
        var served: u64 = 0;
        while (!self.stop.load(.acquire))
        {
            const seq = self.request.load(.acquire);
            if (seq == served)
            {
                std.Thread.yield() catch {};
                continue;
            }
            served = seq;
            const t0 = nowNs();
            const len = self.job_tiles * self.tile_bytes;
            for (0..2) |kind|
            {
                const dst = self.stage[self.stageOffset(self.job_buf, kind)..];
                const rc = tf_reader_read(self.reader, self.fileOffset(self.job_layer, kind, self.sink_tiles), len, dst.ptr, null, null, 0);
                if (rc != 0)
                {
                    std.debug.print("kv offload: read error {d}\n", .{-rc});
                    self.failed.store(true, .release);
                    return;
                }
            }
            self.stats.read_ns += nowNs() - t0;
            self.stats.read_bytes += 2 * len;
            self.done[self.job_buf].store(seq, .release);
        }
    }
};

fn nowNs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
