//! Two-process dispatch/combine check (TASK-02 acceptance). Rank 0 (router)
//! dispatches N token packets to rank 1 (expert), which verifies each one and
//! sends a combine packet back; rank 0 verifies every combine.
//!
//! With `--moe`, the packets carry a real model instead of test patterns: the
//! tiny MoE from tests/golden is split so rank 0 owns the norms, router, and
//! LM head, and rank 1 owns every expert FFN. Rank 0 checks the routing and
//! final logits against the PyTorch goldens (the PoC's pass/fail check).
//!
//!   dispatch shm [--packets N] [--slots S] [--moe]        both ranks, two processes
//!   dispatch expert --port P [--packets N] [--slots S] [--moe]   TCP server, rank 1
//!   dispatch router --host IP --port P [--packets N] [--slots S] [--moe]
//!   dispatch rdma [--dev D] [--gid G] ...                 both ranks, RDMA (TASK-07)
//!   dispatch rdma-expert --port P [--dev D] [--gid G] ...
//!   dispatch rdma-router --host IP --port P [--dev D] [--gid G] ...
//!
//! Instead of `--moe`, `--latency` measures one packet's one-way latency
//! (half the round trip, like `ib_write_lat`) over `--packets` round trips, and
//! `--oneway` streams `--packets` dispatches with no combines and reports the
//! one-direction rate (like `ib_write_bw`).
//!
//! On Linux `shm` forks. Windows has no fork, so `shm` starts this program
//! again as `dispatch shm-expert --name <mapping> ...` (internal) for rank 1.
//! The `rdma` modes need Linux, an RDMA device, and `zig build -Drdma=true`;
//! they swap queue-pair details over TCP on `--port`, then send everything
//! over RDMA.
const std = @import("std");
const build_options = @import("build_options");
const c = std.c;
const win32 = @import("win32.zig");
const wire = @import("wire.zig");
const transport = @import("transport.zig");
const Arena = @import("arena.zig").Arena;
const tfref = @import("tfref.zig");
const moe_ref = @import("moe_ref.zig");

const Options = struct {
    mode: []const u8 = "",
    packets: u64 = 4096,
    /// Ring slots per rank. Defaults to 64, or 256 for the `rdma` modes: the
    /// sender can only have a ring's worth in flight, and RDMA needs about
    /// 256 to keep the link full (bench/transport/2026-11-05.md).
    slots: u16 = 64,
    host: []const u8 = "127.0.0.1",
    port: u16 = 7470,
    moe: bool = false,
    latency: bool = false,
    oneway: bool = false,
    /// `shm-expert` only: the shared mapping's name.
    name: []const u8 = "",
    /// RDMA device; null takes the first.
    device: ?[:0]const u8 = null,
    /// RDMA GID index; negative picks the first RoCE v2 IPv4 GID.
    gid: i32 = -1,
};

const is_windows = @import("builtin").os.tag == .windows;

const combine_salt = 1;

pub fn main(init: std.process.Init) !void {
    const opts = parseArgs(init) catch {
        std.debug.print(
            \\usage: dispatch shm [--packets N] [--slots S] [--moe | --latency | --oneway]
            \\       dispatch expert --port P [--packets N] [--slots S] [--moe | --latency | --oneway]
            \\       dispatch router --host IP --port P [--packets N] [--slots S] [--moe | --latency | --oneway]
            \\       dispatch rdma [--dev D] [--gid G] [--packets N] [--slots S] [--moe | --latency | --oneway]
            \\       dispatch rdma-expert --port P [--dev D] [--gid G] [...]
            \\       dispatch rdma-router --host IP --port P [--dev D] [--gid G] [...]
            \\
        , .{});
        std.process.exit(2);
    };

    if (std.mem.eql(u8, opts.mode, "shm")) return if (is_windows) runShmSpawn(init, opts) else runShmFork(opts);
    if (is_windows and std.mem.eql(u8, opts.mode, "shm-expert")) return runShmExpert(opts);
    if (std.mem.startsWith(u8, opts.mode, "rdma")) {
        if (comptime !build_options.rdma) {
            std.debug.print("this build has no RDMA support; rebuild with zig build -Drdma=true (Linux, libibverbs)\n", .{});
            std.process.exit(2);
        } else return runRdma(opts);
    }

    var arena = try Arena.init(@as(usize, opts.slots) * @sizeOf(wire.Packet));
    defer arena.deinit();
    const ring = try arena.alloc(wire.Packet, opts.slots);

    if (std.mem.eql(u8, opts.mode, "expert")) {
        const l = try transport.tcp.listen(opts.port);
        std.debug.print("expert: listening on port {d}\n", .{l.port});
        const fd = try transport.tcp.acceptOn(l.fd);
        transport.tcp.closeSocket(l.fd);
        var link = transport.tcp.Link.init(fd, ring);
        defer link.closeAfterPeer();
        var ep = transport.Endpoint.init(link.backend(), 1, opts.slots);
        try rank1(&ep, opts);
    } else if (std.mem.eql(u8, opts.mode, "router")) {
        const fd = try transport.tcp.connect(opts.host, opts.port, 100);
        var link = transport.tcp.Link.init(fd, ring);
        defer link.close();
        var ep = transport.Endpoint.init(link.backend(), 0, opts.slots);
        try rank0(&ep, opts, "tcp");
    } else {
        std.debug.print("unknown mode '{s}'\n", .{opts.mode});
        std.process.exit(2);
    }
}

fn parseArgs(init: std.process.Init) !Options {
    var opts: Options = .{};
    var slots_given = false;
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    opts.mode = it.next() orelse return error.Usage;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--moe")) {
            opts.moe = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--latency")) {
            opts.latency = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--oneway")) {
            opts.oneway = true;
            continue;
        }
        const value = it.next() orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--packets")) {
            opts.packets = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, arg, "--slots")) {
            opts.slots = try std.fmt.parseInt(u16, value, 10);
            slots_given = true;
        } else if (std.mem.eql(u8, arg, "--host")) {
            opts.host = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            opts.port = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--name")) {
            opts.name = value;
        } else if (std.mem.eql(u8, arg, "--dev")) {
            opts.device = value;
        } else if (std.mem.eql(u8, arg, "--gid")) {
            opts.gid = try std.fmt.parseInt(i32, value, 10);
        } else return error.Usage;
    }
    if (opts.slots == 0) return error.Usage;
    if (!slots_given and std.mem.startsWith(u8, opts.mode, "rdma")) opts.slots = 256;
    if (@as(u8, @intFromBool(opts.moe)) + @intFromBool(opts.latency) + @intFromBool(opts.oneway) > 1) return error.Usage;
    return opts;
}

fn runShmFork(opts: Options) !void {
    var pair = try transport.shm.Pair.create(opts.slots);
    defer pair.deinit();

    const pid = c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        var ep = transport.Endpoint.init(pair.side(1).backend(), 1, opts.slots);
        rank1(&ep, opts) catch |err| {
            std.debug.print("expert: {t}\n", .{err});
            c._exit(1);
        };
        c._exit(0);
    }

    var ep = transport.Endpoint.init(pair.side(0).backend(), 0, opts.slots);
    const result = rank0(&ep, opts, "shm");
    var status: c_int = 0;
    _ = c.waitpid(pid, &status, 0);
    try result;
    if (status != 0) return error.ExpertFailed;
}

/// Windows `shm`: rank 1 is this program started again, mapping the region
/// by name. Both ranks then run exactly as they do after a fork.
fn runShmSpawn(init: std.process.Init, opts: Options) !void {
    var pair = try transport.shm.Pair.create(opts.slots);
    defer pair.deinit();

    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe = exe_buf[0..try std.process.executablePath(init.io, &exe_buf)];
    var slots_buf: [8]u8 = undefined;
    var packets_buf: [24]u8 = undefined;
    var argv_buf: [10][]const u8 = undefined;
    var argv: std.ArrayList([]const u8) = .initBuffer(&argv_buf);
    argv.appendSliceAssumeCapacity(&.{
        exe,         "shm-expert",
        "--name",    pair.mappingName(),
        "--slots",   try std.fmt.bufPrint(&slots_buf, "{d}", .{opts.slots}),
        "--packets", try std.fmt.bufPrint(&packets_buf, "{d}", .{opts.packets}),
    });
    if (opts.moe) argv.appendAssumeCapacity("--moe");
    if (opts.latency) argv.appendAssumeCapacity("--latency");
    if (opts.oneway) argv.appendAssumeCapacity("--oneway");
    var child = try std.process.spawn(init.io, .{ .argv = argv.items });

    var ep = transport.Endpoint.init(pair.side(0).backend(), 0, opts.slots);
    const result = rank0(&ep, opts, "shm");
    if (result) |_| {
        if (!(try child.wait(init.io)).success()) return error.ExpertFailed;
    } else |err| {
        // A failed router leaves the expert waiting for packets that never come.
        child.kill(init.io);
        return err;
    }
}

/// Windows `shm-expert`: rank 1 of `runShmSpawn`.
fn runShmExpert(opts: Options) !void {
    var pair = try transport.shm.Pair.open(opts.name, opts.slots);
    defer pair.deinit();
    var ep = transport.Endpoint.init(pair.side(1).backend(), 1, opts.slots);
    try rank1(&ep, opts);
}

/// The `rdma` modes. Queue-pair details go over a TCP socket (a loopback one
/// between a fork's two halves for plain `rdma`), then all traffic is RDMA.
fn runRdma(opts: Options) !void {
    if (std.mem.eql(u8, opts.mode, "rdma")) {
        const l = try transport.tcp.listen(0);
        const pid = c.fork();
        if (pid < 0) return error.ForkFailed;
        if (pid == 0) {
            transport.tcp.closeSocket(l.fd);
            const fd = transport.tcp.connect("127.0.0.1", l.port, 50) catch c._exit(1);
            rdmaRank(fd, 1, opts) catch |err| {
                std.debug.print("expert: {t}\n", .{err});
                c._exit(1);
            };
            c._exit(0);
        }
        const fd = try transport.tcp.acceptOn(l.fd);
        transport.tcp.closeSocket(l.fd);
        const result = rdmaRank(fd, 0, opts);
        var status: c_int = 0;
        _ = c.waitpid(pid, &status, 0);
        try result;
        if (status != 0) return error.ExpertFailed;
    } else if (std.mem.eql(u8, opts.mode, "rdma-expert")) {
        const l = try transport.tcp.listen(opts.port);
        std.debug.print("expert: listening on port {d} for the RDMA handshake\n", .{l.port});
        const fd = try transport.tcp.acceptOn(l.fd);
        transport.tcp.closeSocket(l.fd);
        try rdmaRank(fd, 1, opts);
    } else if (std.mem.eql(u8, opts.mode, "rdma-router")) {
        try rdmaRank(try transport.tcp.connect(opts.host, opts.port, 100), 0, opts);
    } else {
        std.debug.print("unknown mode '{s}'\n", .{opts.mode});
        std.process.exit(2);
    }
}

fn rdmaRank(fd: transport.tcp.Socket, rank: u6, opts: Options) !void {
    var arena = try Arena.init(@as(usize, opts.slots) * @sizeOf(wire.Packet));
    defer arena.deinit();
    const ring = try arena.alloc(wire.Packet, opts.slots);
    var link = try transport.rdma.Link.connect(fd, rank, ring, .{ .device = opts.device, .gid_index = opts.gid });
    defer link.close();
    var ep = transport.Endpoint.init(link.backend(), rank, opts.slots);
    if (rank == 0) try rank0(&ep, opts, "rdma") else try rank1(&ep, opts);
}

/// Rank 0's side of whichever check `opts` selects.
fn rank0(ep: *transport.Endpoint, opts: Options, name: []const u8) !void {
    if (opts.moe) return moeRouter(ep, name);
    if (opts.latency) return latencyRouter(ep, opts.packets, name);
    if (opts.oneway) return onewayRouter(ep, opts.packets, name);
    return routerReport(ep, opts, name);
}

/// Rank 1's side of whichever check `opts` selects.
fn rank1(ep: *transport.Endpoint, opts: Options) !void {
    if (opts.moe) return moeExpert(ep);
    if (opts.latency) return latencyExpert(ep, opts.packets);
    if (opts.oneway) return onewayExpert(ep, opts.packets);
    try expert(ep, opts.packets);
    std.debug.print("expert: verified {d} dispatch packets, sent {d} combines\n", .{ ep.stats.received, ep.stats.sent });
}

const latency_warmup = 1000;

/// Rank 0 `--latency`: one dispatch out, then wait for its combine, `rounds`
/// times after a warm-up. Reports half of each round trip, as `ib_write_lat`
/// does, so the numbers compare directly. Includes sealing and checking the
/// packet (CRC32C) on both sides.
fn latencyRouter(ep: *transport.Endpoint, rounds: u64, name: []const u8) !void {
    const gpa = std.heap.page_allocator;
    const samples = try gpa.alloc(u64, rounds);
    defer gpa.free(samples);
    var pkt = std.mem.zeroes(wire.Packet);
    pkt.header.payload_len = wire.hidden_bytes;
    transport.fillPayload(&pkt, 0, 0);
    for (0..latency_warmup + rounds) |i| {
        pkt.header.token_id = i;
        const t0 = nowNs();
        try ep.send(&pkt, .dispatch);
        const r = try ep.recv();
        const t1 = nowNs();
        if (r.kind != .combine or r.packet.header.token_id != i) return error.BadCombine;
        try ep.release();
        if (i >= latency_warmup) samples[i - latency_warmup] = t1 - t0;
    }
    try ep.finish();
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const half = struct {
        fn us(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 2e3;
        }
    }.us;
    std.debug.print("{s}: one-way latency over {d} round trips of {d}-byte packets: min {d:.2} us, p50 {d:.2} us, p99 {d:.2} us, p99.9 {d:.2} us\n", .{
        name,
        rounds,
        wire.packet_bytes,
        half(samples[0]),
        half(samples[samples.len / 2]),
        half(samples[samples.len * 99 / 100]),
        half(samples[samples.len * 999 / 1000]),
    });
}

/// Rank 1 `--latency`: answer each dispatch with a combine at once.
fn latencyExpert(ep: *transport.Endpoint, rounds: u64) !void {
    var out = std.mem.zeroes(wire.Packet);
    out.header.payload_len = wire.hidden_bytes;
    for (0..latency_warmup + rounds) |_| {
        const r = try ep.recv();
        if (r.kind != .dispatch) return error.BadDispatch;
        out.header.token_id = r.packet.header.token_id;
        try ep.release();
        try ep.send(&out, .combine);
    }
    try ep.finish();
}

/// Rank 0 `--oneway`: stream `packets` dispatches as fast as credits allow;
/// rank 1 confirms the last one with a single combine. Reports one
/// direction's payload rate, as `ib_write_bw` does.
fn onewayRouter(ep: *transport.Endpoint, packets: u64, name: []const u8) !void {
    var pkt = std.mem.zeroes(wire.Packet);
    pkt.header.payload_len = wire.hidden_bytes;
    transport.fillPayload(&pkt, 0, 0);
    const start = nowNs();
    for (0..packets) |i| {
        pkt.header.token_id = i;
        try ep.send(&pkt, .dispatch);
    }
    const r = try ep.recv();
    const ns = nowNs() - start;
    if (r.kind != .combine or r.packet.header.token_id != packets - 1) return error.BadCombine;
    try ep.release();
    try ep.finish();
    std.debug.print("{s}: one-way {d} x {d}-byte packets in {d:.2} ms: {d:.2} Gb/s, {d:.2} us per packet, {d} polls waiting for credits\n", .{
        name,
        packets,
        wire.packet_bytes,
        @as(f64, @floatFromInt(ns)) / 1e6,
        @as(f64, @floatFromInt(packets * wire.packet_bytes * 8)) / @as(f64, @floatFromInt(ns)),
        @as(f64, @floatFromInt(ns)) / 1e3 / @as(f64, @floatFromInt(packets)),
        ep.stats.blocked_polls,
    });
}

/// Rank 1 `--oneway`: receive and check every dispatch, confirm the last.
fn onewayExpert(ep: *transport.Endpoint, packets: u64) !void {
    var last: u64 = 0;
    for (0..packets) |i| {
        const r = try ep.recv();
        if (r.kind != .dispatch or r.packet.header.token_id != i) return error.BadDispatch;
        last = r.packet.header.token_id;
        try ep.release();
    }
    var out = std.mem.zeroes(wire.Packet);
    out.header.token_id = last;
    try ep.send(&out, .combine);
    try ep.finish();
}

/// Rank 1: receive each dispatch, check it, answer with a combine.
fn expert(ep: *transport.Endpoint, packets: u64) !void {
    var out = std.mem.zeroes(wire.Packet);
    var done: u64 = 0;
    errdefer std.debug.print("expert: failed after {d} packets\n", .{done});
    while (done < packets) : (done += 1) {
        const r = try ep.recv();
        const id = r.packet.header.token_id;
        if (r.kind != .dispatch or !transport.payloadMatches(r.packet, id, 0)) return error.BadDispatch;
        out.header = r.packet.header;
        out.header.payload_len = wire.hidden_bytes;
        try ep.release();
        transport.fillPayload(&out, id, combine_salt);
        try ep.send(&out, .combine);
    }
    try ep.finish();
}

/// Rank 0: keep the pipe full of dispatches while draining combines.
fn router(ep: *transport.Endpoint, packets: u64) !void {
    var pkt = std.mem.zeroes(wire.Packet);
    var sent: u64 = 0;
    var combined: u64 = 0;
    errdefer std.debug.print("router: failed after {d} dispatches sent, {d} combines received, {d} credits held\n", .{ sent, combined, ep.credits });
    while (combined < packets) {
        if (sent < packets) {
            pkt.header = .{ .token_id = sent, .generation = 0, .src_rank = 0, .layer = 0, .expert = @truncate(sent % 64), .flags = 0, .gate_weight = 0.125, .payload_len = wire.hidden_bytes };
            transport.fillPayload(&pkt, sent, 0);
            if (ep.trySend(&pkt, .dispatch)) {
                sent += 1;
            } else |err| switch (err) {
                error.WouldBlock => {},
                else => |e| return e,
            }
        }
        while (try ep.tryRecv()) |r| {
            if (r.kind != .combine or r.packet.header.token_id != combined or
                !transport.payloadMatches(r.packet, combined, combine_salt)) return error.BadCombine;
            combined += 1;
            try ep.release();
        }
        try ep.flushCredits();
    }
    try ep.finish();
}

fn routerReport(ep: *transport.Endpoint, opts: Options, name: []const u8) !void {
    const start = nowNs();
    try router(ep, opts.packets);
    const ns = nowNs() - start;
    const bytes = 2 * opts.packets * wire.packet_bytes;
    std.debug.print(
        "{s}: {d} dispatch + {d} combine packets verified (CRC, generation, payload), {d} slots\n" ++
            "{s}: {d:.1} ms, {d:.2} us per token (pipelined), {d:.2} Gb/s both directions\n",
        .{
            name,                                                                      opts.packets,
            opts.packets,                                                              opts.slots,
            name,                                                                      @as(f64, @floatFromInt(ns)) / 1e6,
            @as(f64, @floatFromInt(ns)) / 1e3 / @as(f64, @floatFromInt(opts.packets)), @as(f64, @floatFromInt(bytes * 8)) / @as(f64, @floatFromInt(ns)),
        },
    );
}

const golden = @embedFile("moe_tiny.tfref");

fn payloadF32(pkt: *wire.Packet, n: usize) []align(1) f32 {
    return std.mem.bytesAsSlice(f32, pkt.payload[0 .. n * @sizeOf(f32)]);
}

/// Rank 1 in `--moe` mode: owns every expert's FFN weights. Each dispatch
/// carries one token's normalized hidden state for one (layer, expert); the
/// combine carries that expert's output back, unweighted.
fn moeExpert(ep: *transport.Endpoint) !void {
    const f = try tfref.File.parse(golden);
    const cfg = try moe_ref.Config.fromFile(f);
    const d = cfg.d_model;
    var weights: [16]moe_ref.Weights = undefined;
    if (cfg.layers > weights.len or d > 64 or cfg.d_ff > 4096) return error.ModelTooBig;
    for (weights[0..cfg.layers], 0..) |*w, l| w.* = try moe_ref.Weights.load(f, l);

    var out = std.mem.zeroes(wire.Packet);
    var n: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    const total = cfg.layers * cfg.tokens * cfg.top_k;
    var done: usize = 0;
    errdefer std.debug.print("expert: failed after {d} of {d} expert calls\n", .{ done, total });
    while (done < total) : (done += 1) {
        const r = try ep.recv();
        const hdr = r.packet.header;
        if (r.kind != .dispatch or hdr.layer >= cfg.layers or hdr.expert >= cfg.experts or
            hdr.payload_len != d * @sizeOf(f32)) return error.BadDispatch;
        for (n[0..d], payloadF32(@constCast(r.packet), d)) |*x, v| x.* = v;
        try ep.release();
        moe_ref.expert(cfg, weights[hdr.layer], hdr.expert, n[0..d], b[0..d]);
        out.header = hdr;
        for (payloadF32(&out, d), b[0..d]) |*x, v| x.* = v;
        try ep.send(&out, .combine);
    }
    try ep.finish();
    std.debug.print("expert: ran {d} expert FFNs ({d} layers x {d} tokens x top-{d}) for rank 0\n", .{ total, cfg.layers, cfg.tokens, cfg.top_k });
}

/// Rank 0 in `--moe` mode: norms, router, and LM head. Per layer, routes every
/// token, dispatches top-k packets, then folds the combines back in arrival
/// order (the same order the single-process reference sums them in).
fn moeRouter(ep: *transport.Endpoint, name: []const u8) !void {
    const f = try tfref.File.parse(golden);
    const cfg = try moe_ref.Config.fromFile(f);
    const d = cfg.d_model;
    const k = cfg.top_k;
    const per_layer = cfg.tokens * k;
    // Every dispatch of a layer goes out before any combine is read.
    if (per_layer > ep.slots) return error.NotEnoughSlots;

    const max_tokens = 64;
    if (cfg.tokens > max_tokens or d > 64 or cfg.experts > 64 or k > 8 or cfg.vocab > 256) return error.ModelTooBig;
    var h: [max_tokens * 64]f32 = undefined;
    var moe: [max_tokens * 64]f32 = undefined;
    var n: [64]f32 = undefined;
    var logits: [64]f32 = undefined;
    var idx: [max_tokens * 8]i32 = undefined;
    var wts: [max_tokens * 8]f32 = undefined;
    for (h[0 .. cfg.tokens * d], (try f.get("input")).f32s()) |*x, v| x.* = v;

    var pkt = std.mem.zeroes(wire.Packet);
    var name_buf: [64]u8 = undefined;
    const start = nowNs();
    var ok = true;
    for (0..cfg.layers) |l| {
        const w = try moe_ref.Weights.load(f, l);
        const layer_start = nowNs();
        for (0..cfg.tokens) |t| {
            const ti = idx[t * k ..][0..k];
            const tw = wts[t * k ..][0..k];
            moe_ref.route(cfg, w, h[t * d ..][0..d], n[0..d], logits[0..cfg.experts], ti, tw);
            for (ti, tw) |e, p| {
                pkt.header = .{ .token_id = t, .generation = 0, .src_rank = 0, .layer = @intCast(l), .expert = @intCast(e), .flags = 0, .gate_weight = p, .payload_len = @intCast(d * @sizeOf(f32)) };
                for (payloadF32(&pkt, d), n[0..d]) |*x, v| x.* = v;
                try ep.send(&pkt, .dispatch);
            }
        }
        @memset(moe[0 .. cfg.tokens * d], 0);
        for (0..per_layer) |i| {
            const r = try ep.recv();
            const hdr = r.packet.header;
            const t = i / k;
            // Combines must come back for exactly the (token, expert) we sent.
            if (r.kind != .combine or hdr.token_id != t or hdr.layer != l or
                hdr.expert != idx[i] or hdr.payload_len != d * @sizeOf(f32)) return error.BadCombine;
            for (moe[t * d ..][0..d], payloadF32(@constCast(r.packet), d)) |*m, v| m.* += wts[i] * v;
            try ep.release();
        }
        try ep.flushCredits();
        for (h[0 .. cfg.tokens * d], moe[0 .. cfg.tokens * d]) |*x, m| x.* += m;
        const layer_ns = nowNs() - layer_start;

        const want_idx = (try f.get(try std.fmt.bufPrint(&name_buf, "golden.layer{d}.topk_idx", .{l}))).i32s();
        const want_h = (try f.get(try std.fmt.bufPrint(&name_buf, "golden.layer{d}.hidden", .{l}))).f32s();
        var routing_ok = want_idx.len == per_layer;
        for (idx[0..per_layer], want_idx) |got, want_e| routing_ok = routing_ok and got == want_e;
        const hidden_ok = moe_ref.allClose(h[0 .. cfg.tokens * d], want_h);
        ok = ok and routing_ok and hidden_ok;
        std.debug.print("{s}: layer {d}: {d} dispatch + {d} combine, routing {s}, hidden max err {e:.2} ({s}), {d:.2} ms\n", .{
            name,
            l,
            per_layer,
            per_layer,
            if (routing_ok) "matches" else "DIFFERS",
            maxErr(h[0 .. cfg.tokens * d], want_h),
            if (hidden_ok) "ok" else "FAIL",
            @as(f64, @floatFromInt(layer_ns)) / 1e6,
        });
    }
    try ep.finish();

    var out_logits: [max_tokens * 256]f32 = undefined;
    const ol = out_logits[0 .. cfg.tokens * cfg.vocab];
    try moe_ref.head(cfg, f, h[0 .. cfg.tokens * d], ol);
    const want = (try f.get("golden.logits")).f32s();
    const logits_ok = moe_ref.allClose(ol, want);
    ok = ok and logits_ok;
    var argmax_match: usize = 0;
    for (0..cfg.tokens) |t| {
        if (argmax(ol[t * cfg.vocab ..][0..cfg.vocab]) == argmax(want[t * cfg.vocab ..][0..cfg.vocab])) argmax_match += 1;
    }
    std.debug.print("{s}: logits vs PyTorch: max err {e:.2} ({s}), next-token argmax {d}/{d}, {d:.2} ms total\n", .{
        name,
        maxErr(ol, want),
        if (logits_ok) "ok" else "FAIL",
        argmax_match,
        cfg.tokens,
        @as(f64, @floatFromInt(nowNs() - start)) / 1e6,
    });
    if (!ok) return error.MoeMismatch;
    std.debug.print("{s}: PASS: two-process MoE matches the PyTorch reference\n", .{name});
}

fn maxErr(x: []const f32, y: []align(1) const f32) f32 {
    var m: f32 = 0;
    for (x, y) |u, v| m = @max(m, @abs(u - v));
    return m;
}

fn argmax(x: []align(1) const f32) usize {
    var best: usize = 0;
    for (x, 0..) |v, i| {
        if (v > x[best]) best = i;
    }
    return best;
}

fn nowNs() u64 {
    if (is_windows) return win32.nowNs();
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
