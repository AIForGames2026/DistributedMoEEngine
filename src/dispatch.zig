//! Two-process dispatch/combine check (TASK-02 acceptance). Rank 0 (router)
//! dispatches N token packets to rank 1 (expert), which verifies each one and
//! sends a combine packet back; rank 0 verifies every combine.
//!
//!   dispatch shm [--packets N] [--slots S]          both ranks, forked
//!   dispatch expert --port P [--packets N] [--slots S]   TCP server, rank 1
//!   dispatch router --host IP --port P [--packets N] [--slots S]
const std = @import("std");
const c = std.c;
const wire = @import("wire.zig");
const transport = @import("transport.zig");
const Arena = @import("arena.zig").Arena;

const Options = struct {
    mode: []const u8 = "",
    packets: u64 = 4096,
    slots: u16 = 64,
    host: []const u8 = "127.0.0.1",
    port: u16 = 7470,
};

const combine_salt = 1;

pub fn main(init: std.process.Init) !void {
    const opts = parseArgs(init) catch {
        std.debug.print(
            \\usage: dispatch shm [--packets N] [--slots S]
            \\       dispatch expert --port P [--packets N] [--slots S]
            \\       dispatch router --host IP --port P [--packets N] [--slots S]
            \\
        , .{});
        std.process.exit(2);
    };

    if (std.mem.eql(u8, opts.mode, "shm")) return runShm(opts);

    var arena = try Arena.init(@as(usize, opts.slots) * @sizeOf(wire.Packet));
    defer arena.deinit();
    const ring = try arena.alloc(wire.Packet, opts.slots);

    if (std.mem.eql(u8, opts.mode, "expert")) {
        const l = try transport.tcp.listen(opts.port);
        std.debug.print("expert: listening on port {d}\n", .{l.port});
        const fd = try transport.tcp.acceptOn(l.fd);
        _ = c.close(l.fd);
        var link = transport.tcp.Link.init(fd, ring);
        defer link.close();
        var ep = transport.Endpoint.init(link.backend(), 1, opts.slots);
        try expert(&ep, opts.packets);
        std.debug.print("expert: verified {d} dispatch packets, sent {d} combines\n", .{ ep.stats.received, ep.stats.sent });
    } else if (std.mem.eql(u8, opts.mode, "router")) {
        const fd = try transport.tcp.connect(opts.host, opts.port, 100);
        var link = transport.tcp.Link.init(fd, ring);
        defer link.close();
        var ep = transport.Endpoint.init(link.backend(), 0, opts.slots);
        try routerReport(&ep, opts, "tcp");
    } else {
        std.debug.print("unknown mode '{s}'\n", .{opts.mode});
        std.process.exit(2);
    }
}

fn parseArgs(init: std.process.Init) !Options {
    var opts: Options = .{};
    var it = init.minimal.args.iterate();
    _ = it.next();
    opts.mode = it.next() orelse return error.Usage;
    while (it.next()) |arg| {
        const value = it.next() orelse return error.Usage;
        if (std.mem.eql(u8, arg, "--packets")) {
            opts.packets = try std.fmt.parseInt(u64, value, 10);
        } else if (std.mem.eql(u8, arg, "--slots")) {
            opts.slots = try std.fmt.parseInt(u16, value, 10);
        } else if (std.mem.eql(u8, arg, "--host")) {
            opts.host = value;
        } else if (std.mem.eql(u8, arg, "--port")) {
            opts.port = try std.fmt.parseInt(u16, value, 10);
        } else return error.Usage;
    }
    if (opts.slots == 0) return error.Usage;
    return opts;
}

fn runShm(opts: Options) !void {
    var pair = try transport.shm.Pair.create(opts.slots);
    defer pair.deinit();

    const pid = c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        var ep = transport.Endpoint.init(pair.side(1).backend(), 1, opts.slots);
        expert(&ep, opts.packets) catch |err| {
            std.debug.print("expert: {t}\n", .{err});
            c._exit(1);
        };
        c._exit(0);
    }

    var ep = transport.Endpoint.init(pair.side(0).backend(), 0, opts.slots);
    const result = routerReport(&ep, opts, "shm");
    var status: c_int = 0;
    _ = c.waitpid(pid, &status, 0);
    try result;
    if (status != 0) return error.ExpertFailed;
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
            name,                                              opts.packets,
            opts.packets,                                      opts.slots,
            name,                                              @as(f64, @floatFromInt(ns)) / 1e6,
            @as(f64, @floatFromInt(ns)) / 1e3 / @as(f64, @floatFromInt(opts.packets)),
            @as(f64, @floatFromInt(bytes * 8)) / @as(f64, @floatFromInt(ns)),
        },
    );
}

fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
