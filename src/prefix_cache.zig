//! Persistent prefix KV cache (TASK-28). Linux, in-memory KV cache only.
//!
//! After a prefill, the prompt's keys and values for every layer, its tokens,
//! and the logits of its last position go to a file in a directory, named by
//! a hash of the run's fingerprint and the tokens. A later prompt that starts
//! with the same tokens reads the shared part back through the TASK-03
//! io_uring reader instead of recomputing it.
//!
//! A restored cache must match a cold run bit for bit. The keys and values of
//! a position depend on the chunk it was prefilled in (kernels are picked by
//! batch size), so a prompt takes only a whole number of chunks from a saved
//! prefix, and the rest is prefilled from there with the same boundaries a
//! cold run uses. A prompt identical to the saved one takes everything,
//! including the saved logits, and skips prefill. The fingerprint (engine
//! build, model file, settings that change results) must match exactly.
const std = @import("std");
const c = std.c;
const olmoe = @import("olmoe.zig");

const Reader = opaque {};
const ReaderOpts = extern struct { queue_depth: c_uint, block_bytes: usize, sqpoll: c_int };
extern fn tf_reader_open(out: *?*Reader, path: [*:0]const u8, opts: *const ReaderOpts, region: *anyopaque, region_len: usize) c_int;
extern fn tf_reader_read(r: *Reader, offset: u64, len: usize, dst: *anyopaque, lat_ns: ?[*]u32, lat_count: ?*usize, lat_cap: usize) c_int;
extern fn tf_reader_close(r: *Reader) void;

const magic = "TFPREFX1".*;
const page = 4096;
const suffix = ".tfpx";

const Header = extern struct {
    magic: [8]u8,
    layers: u32,
    kvd: u32,
    vocab: u32,
    chunk: u32,
    positions: u64,
    fingerprint_len: u32,
    reserved: u32 = 0,
};

const max_fingerprint = page - @sizeOf(Header);

fn roundUp(x: u64) u64
{
    return (x + page - 1) / page * page;
}

/// Where each part of a file of `positions` positions starts.
const Layout = struct {
    positions: u64,
    kvd: u64,
    vocab: u64,

    fn tokens(_: Layout) u64
    {
        return page;
    }

    fn logits(self: Layout) u64
    {
        return page + roundUp(self.positions * 4);
    }

    fn segmentBytes(self: Layout) u64
    {
        return roundUp(self.positions * self.kvd * 4);
    }

    /// Keys (kind 0) or values (kind 1) of `layer`.
    fn segment(self: Layout, layer: usize, kind: usize) u64
    {
        return self.logits() + roundUp(self.vocab * 4) + (@as(u64, layer) * 2 + kind) * self.segmentBytes();
    }

    fn fileBytes(self: Layout, layers: usize) u64
    {
        return self.segment(layers, 0);
    }
};

/// How many leading positions of a saved prefix a prompt of `n` tokens can
/// take when its first `common` tokens match the `saved` saved ones.
pub fn usable(saved: usize, common: usize, n: usize, chunk: usize) struct { positions: usize, full: bool }
{
    if (common == saved and saved == n)
    {
        return .{ .positions = n, .full = true };
    }
    // At least one position is left to prefill, since its logits are needed.
    const most = @min(common, n - 1);
    return .{ .positions = most / chunk * chunk, .full = false };
}

pub const Match = struct {
    /// The file, under the directory.
    name: [64]u8,
    name_len: usize,
    saved: usize,
    positions: usize,
    full: bool,

    pub fn fileName(self: *const Match) []const u8
    {
        return self.name[0..self.name_len];
    }
};

/// What a run needs to agree on with a saved prefix.
pub const Key = struct {
    dir: []const u8,
    fingerprint: []const u8,
    cfg: olmoe.Config,
    chunk: usize,
};

fn zPath(buf: []u8, dir: []const u8, name: []const u8) ![:0]const u8
{
    const text = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}/{s}", .{ dir, name });
    buf[text.len] = 0;
    return buf[0..text.len :0];
}

fn preadAll(fd: c.fd_t, dst: []u8, off: u64) !void
{
    var done: usize = 0;
    while (done < dst.len)
    {
        const got = c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(off + done));
        if (got <= 0)
        {
            return error.PrefixReadFailed;
        }
        done += @intCast(got);
    }
}

fn pwriteAll(fd: c.fd_t, src: []const u8, off: u64) !void
{
    var done: usize = 0;
    while (done < src.len)
    {
        const put = c.pwrite(fd, src[done..].ptr, src.len - done, @intCast(off + done));
        if (put <= 0)
        {
            return error.PrefixWriteFailed;
        }
        done += @intCast(put);
    }
}

/// Reads the header of a saved prefix and checks it belongs to `key`; null if not.
fn readHeader(fd: c.fd_t, key: Key) ?Header
{
    var buf: [page]u8 align(8) = undefined;
    preadAll(fd, &buf, 0) catch return null;
    const h = std.mem.bytesToValue(Header, buf[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &h.magic, &magic) or h.layers != key.cfg.layers or h.kvd != key.cfg.kvDim() or
        h.vocab != key.cfg.vocab or h.chunk != key.chunk or h.fingerprint_len != key.fingerprint.len)
    {
        return null;
    }
    if (!std.mem.eql(u8, buf[@sizeOf(Header)..][0..h.fingerprint_len], key.fingerprint))
    {
        return null;
    }
    return h;
}

/// The saved prefix in `key.dir` that covers the most of `tokens`, if any covers a chunk or more.
pub fn find(gpa: std.mem.Allocator, key: Key, tokens: []const u32) !?Match
{
    var dir_buf: [512]u8 = undefined;
    const dir_z = try zPath(&dir_buf, key.dir, ".");
    const dir = c.opendir(dir_z) orelse return null;
    defer _ = c.closedir(dir);
    var best: ?Match = null;
    var saved_tokens: std.ArrayList(u8) = .empty;
    defer saved_tokens.deinit(gpa);
    while (c.readdir(dir)) |ent|
    {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
        if (!std.mem.endsWith(u8, name, suffix) or name.len > 64)
        {
            continue;
        }
        var path_buf: [600]u8 = undefined;
        const path = try zPath(&path_buf, key.dir, name);
        const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0)
        {
            continue;
        }
        defer _ = c.close(fd);
        const h = readHeader(fd, key) orelse continue;
        const saved: usize = @intCast(h.positions);
        try saved_tokens.resize(gpa, saved * 4);
        preadAll(fd, saved_tokens.items, page) catch continue;
        var common: usize = 0;
        const limit = @min(saved, tokens.len);
        while (common < limit and std.mem.readInt(u32, saved_tokens.items[common * 4 ..][0..4], .little) == tokens[common])
        {
            common += 1;
        }
        const u = usable(saved, common, tokens.len, key.chunk);
        if (u.positions == 0 or (best != null and u.positions <= best.?.positions))
        {
            continue;
        }
        var m: Match = .{ .name = undefined, .name_len = name.len, .saved = saved, .positions = u.positions, .full = u.full };
        @memcpy(m.name[0..name.len], name);
        best = m;
    }
    return best;
}

pub const Transfer = struct {
    bytes: u64,
    ns: u64,
};

/// Reads the first `m.positions` positions of every layer into `cache`, and
/// the saved logits into `logits` when the match is full.
pub fn restore(key: Key, m: Match, cache: *olmoe.KvCache, logits: []f32) !Transfer
{
    std.debug.assert(cache.tier == null);
    const t0 = nowNs();
    var path_buf: [600]u8 = undefined;
    const path = try zPath(&path_buf, key.dir, m.fileName());
    const lay: Layout = .{ .positions = m.saved, .kvd = cache.d, .vocab = key.cfg.vocab };
    const ropts: ReaderOpts = .{ .queue_depth = 32, .block_bytes = 1 << 20, .sqpoll = 0 };
    var readers: [2]*Reader = undefined;
    for ([_][]f32{ cache.k, cache.v }, 0..) |region, kind|
    {
        var r: ?*Reader = null;
        const rc = tf_reader_open(&r, path, &ropts, region.ptr, region.len * @sizeOf(f32));
        if (rc != 0)
        {
            if (kind == 1)
            {
                tf_reader_close(readers[0]);
            }
            std.debug.print("prefix cache: tf_reader_open error {d}\n", .{-rc});
            return error.PrefixReadFailed;
        }
        readers[kind] = r.?;
    }
    defer for (readers) |r|
    {
        tf_reader_close(r);
    };
    const len = roundUp(@as(u64, m.positions) * cache.d * 4);
    for (0..key.cfg.layers) |layer|
    {
        for (0..2) |kind|
        {
            const dst = if (kind == 0) cache.keys(layer) else cache.values(layer);
            const rc = tf_reader_read(readers[kind], lay.segment(layer, kind), @intCast(len), dst.ptr, null, null, 0);
            if (rc != 0)
            {
                std.debug.print("prefix cache: read error {d}\n", .{-rc});
                return error.PrefixReadFailed;
            }
        }
    }
    if (m.full)
    {
        const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0)
        {
            return error.PrefixReadFailed;
        }
        defer _ = c.close(fd);
        try preadAll(fd, std.mem.sliceAsBytes(logits[0..key.cfg.vocab]), lay.logits());
    }
    cache.len = m.positions;
    return .{ .bytes = 2 * key.cfg.layers * len, .ns = nowNs() - t0 };
}

pub const Saved = struct {
    name: [64]u8,
    name_len: usize,
    transfer: Transfer,

    pub fn fileName(self: *const Saved) []const u8
    {
        return self.name[0..self.name_len];
    }
};

/// Writes the first `tokens.len` positions of `cache`, the tokens, and the
/// last position's `logits` to a new file in `key.dir`.
pub fn save(key: Key, tokens: []const u32, cache: *const olmoe.KvCache, logits: []const f32) !Saved
{
    std.debug.assert(cache.tier == null and cache.len >= tokens.len);
    if (key.fingerprint.len > max_fingerprint)
    {
        return error.FingerprintTooLong;
    }
    const t0 = nowNs();
    var hasher = std.hash.Wyhash.init(0x7466707266);
    hasher.update(key.fingerprint);
    hasher.update(std.mem.sliceAsBytes(tokens));
    var out: Saved = .{ .name = undefined, .name_len = 0, .transfer = undefined };
    const name = try std.fmt.bufPrint(&out.name, "{x:0>16}{s}", .{ hasher.final(), suffix });
    out.name_len = name.len;

    const lay: Layout = .{ .positions = tokens.len, .kvd = cache.d, .vocab = key.cfg.vocab };
    const pages = std.heap.page_allocator;
    const head = try pages.alloc(u8, lay.segment(0, 0));
    defer pages.free(head);
    @memset(head, 0);
    const h: Header = .{
        .magic = magic,
        .layers = @intCast(key.cfg.layers),
        .kvd = @intCast(cache.d),
        .vocab = @intCast(key.cfg.vocab),
        .chunk = @intCast(key.chunk),
        .positions = tokens.len,
        .fingerprint_len = @intCast(key.fingerprint.len),
    };
    @memcpy(head[0..@sizeOf(Header)], std.mem.asBytes(&h));
    @memcpy(head[@sizeOf(Header)..][0..key.fingerprint.len], key.fingerprint);
    @memcpy(head[lay.tokens()..][0 .. tokens.len * 4], std.mem.sliceAsBytes(tokens));
    @memcpy(head[lay.logits()..][0 .. key.cfg.vocab * 4], std.mem.sliceAsBytes(logits[0..key.cfg.vocab]));

    var tmp_buf: [600]u8 = undefined;
    var tmp_name: [32]u8 = undefined;
    const tmp = try zPath(&tmp_buf, key.dir, try std.fmt.bufPrint(&tmp_name, ".tmp_{d}", .{c.getpid()}));
    const fd = c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .DIRECT = true }, @as(c.mode_t, 0o644));
    if (fd < 0)
    {
        std.debug.print("prefix cache: cannot create {s}\n", .{tmp});
        return error.PrefixWriteFailed;
    }
    var ok = false;
    defer if (!ok)
    {
        _ = c.close(fd);
        _ = c.unlink(tmp);
    };
    try pwriteAll(fd, head, 0);
    const len = lay.segmentBytes();
    for (0..key.cfg.layers) |layer|
    {
        try pwriteAll(fd, std.mem.sliceAsBytes(cache.keys(layer))[0..len], lay.segment(layer, 0));
        try pwriteAll(fd, std.mem.sliceAsBytes(cache.values(layer))[0..len], lay.segment(layer, 1));
    }
    if (c.fsync(fd) != 0)
    {
        return error.PrefixWriteFailed;
    }
    ok = true;
    _ = c.close(fd);
    var path_buf: [600]u8 = undefined;
    const path = try zPath(&path_buf, key.dir, name);
    if (c.rename(tmp, path) != 0)
    {
        _ = c.unlink(tmp);
        return error.PrefixWriteFailed;
    }
    out.transfer = .{ .bytes = lay.fileBytes(key.cfg.layers), .ns = nowNs() - t0 };
    return out;
}

fn nowNs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "a prompt takes whole chunks of a saved prefix, or all of an identical one" {
    // Identical prompt: everything, with the saved logits.
    try std.testing.expectEqual(@as(usize, 5000), usable(5000, 5000, 5000, 2048).positions);
    try std.testing.expect(usable(5000, 5000, 5000, 2048).full);
    // Longer prompt over the whole saved prefix: whole chunks only.
    try std.testing.expectEqual(@as(usize, 4096), usable(5000, 5000, 9000, 2048).positions);
    try std.testing.expectEqual(@as(usize, 4096), usable(4096, 4096, 9000, 2048).positions);
    // Prompt that ends at a chunk boundary inside the saved one keeps a chunk to prefill.
    const inside = usable(8192, 4096, 4096, 2048);
    try std.testing.expectEqual(@as(usize, 2048), inside.positions);
    try std.testing.expect(!inside.full);
    // Diverges inside the first chunk: nothing usable.
    try std.testing.expectEqual(@as(usize, 0), usable(8192, 1000, 9000, 2048).positions);
}

test "file layout keeps every part on page boundaries" {
    const lay: Layout = .{ .positions = 131041, .kvd = 512, .vocab = 151936 };
    try std.testing.expectEqual(@as(u64, 0), lay.logits() % page);
    try std.testing.expectEqual(@as(u64, 0), lay.segment(3, 1) % page);
    try std.testing.expect(lay.segmentBytes() >= 131041 * 512 * 4);
    try std.testing.expectEqual(lay.segment(1, 0) - lay.segment(0, 1), lay.segmentBytes());
}
