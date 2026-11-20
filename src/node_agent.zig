//! The node agent behind `tfswarm node` (TASK-25). Linux and Windows (TASK-43).
//!
//!   tfswarm node --dashboard IP[:PORT] --secret-file F --models DIR
//!              [--engine PATH] [--name NAME] [--log-dir DIR] [--advertise IP]
//!   tfswarm node --settings FILE [options]
//!
//! With `--settings` (TASK-44, as the Windows tray launcher starts it) the
//! options come from that file's "node" part, and the secret and logs sit beside it.
//!
//! Runs on each machine that may hold experts. It links to the operator
//! dashboard (port 47310 by default), proves it holds the shared node secret and
//! checks that the dashboard does too, then reports what the machine has: its
//! memory, GPU, RDMA devices, and the models in DIR. The dashboard lists it as
//! pending until an operator approves it. On the dashboard's command it starts
//! one expert rank of the engine (`expert` or `rdma-expert`) and stops it again.
//! Commands name a model file and a few numbers, never a path or a command line,
//! and the agent builds the engine's arguments itself. If the link drops, the
//! rank is stopped, since the router it served belongs to the dashboard.
const std = @import("std");
const c = std.c;
const auth = @import("auth.zig");
const proc = @import("proc.zig");
const tcp = @import("transport/tcp.zig");

pub const default_port: u16 = 47310;

const Options = struct {
    host: []const u8 = "",
    port: u16 = default_port,
    secret_file: []const u8 = "node.secret",
    models: []const u8 = "",
    engine: []const u8 = "/proc/self/exe",
    name: ?[]const u8 = null,
    log_dir: []const u8 = "node-logs",
    /// The address the router dials for this node's rank over TCP; default the one the dashboard sees.
    advertise: ?[]const u8 = null,
};

/// What the dashboard asks a node to run.
pub const Start = struct {
    rank: u16,
    ranks: u16,
    model: []const u8,
    transport: []const u8 = "tcp",
    port: u16,
    dev: ?[]const u8 = null,
    gpu: bool = false,
    max_step: ?u32 = null,
    slots: ?u16 = null,
    cache: ?u32 = null,
};

const Rank = struct {
    start: Start,
    child: proc.Child,
    listening: bool = false,
};

const usage =
    \\usage: tfswarm node --dashboard IP[:PORT] --models DIR [--secret-file F] [--engine PATH] [--name NAME] [--log-dir DIR] [--advertise IP]
    \\       tfswarm node --settings FILE [options]
    \\
;

pub fn main(gpa: std.mem.Allocator, args: []const []const u8) !void
{
    var opts: Options = .{};
    for (args, 0..) |arg, j|
    {
        if (std.mem.eql(u8, arg, "--settings") and j + 1 < args.len)
        {
            try applySettings(gpa, &opts, args[j + 1]);
        }
    }
    var i: usize = 0;
    while (i < args.len) : (i += 2)
    {
        if (i + 1 >= args.len)
        {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--dashboard"))
        {
            if (std.mem.indexOfScalar(u8, v, ':')) |colon|
            {
                opts.host = v[0..colon];
                opts.port = try std.fmt.parseInt(u16, v[colon + 1 ..], 10);
            }
            else
            {
                opts.host = v;
            }
        }
        else if (std.mem.eql(u8, args[i], "--secret-file"))
        {
            opts.secret_file = v;
        }
        else if (std.mem.eql(u8, args[i], "--models"))
        {
            opts.models = v;
        }
        else if (std.mem.eql(u8, args[i], "--engine"))
        {
            opts.engine = v;
        }
        else if (std.mem.eql(u8, args[i], "--name"))
        {
            opts.name = v;
        }
        else if (std.mem.eql(u8, args[i], "--log-dir"))
        {
            opts.log_dir = v;
        }
        else if (std.mem.eql(u8, args[i], "--advertise"))
        {
            _ = tcp.parseIp4(v) catch {
                std.debug.print("node: --advertise takes an IPv4 address\n", .{});
                std.process.exit(2);
            };
            opts.advertise = v;
        }
        else if (std.mem.eql(u8, args[i], "--settings"))
        {
            // Already read, before the other options.
        }
        else
        {
            std.debug.print("node: unknown option {s}\n" ++ usage, .{args[i]});
            std.process.exit(2);
        }
    }
    if (opts.host.len == 0 or opts.models.len == 0)
    {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    var host_buf: [64]u8 = undefined;
    const name = opts.name orelse proc.hostName(&host_buf);
    if (!auth.validName(name))
    {
        std.debug.print("node: '{s}' is not a usable name (letters, digits, '.', '-', '_'); give --name\n", .{name});
        std.process.exit(2);
    }
    const secret = try auth.loadSecret(gpa, opts.secret_file, "node secret");
    defer gpa.free(secret);
    // The engine path is resolved once, so the agent keeps starting the same binary.
    var engine_buf: [4096]u8 = undefined;
    const engine_z = try gpa.dupeSentinel(u8, opts.engine, 0);
    defer gpa.free(engine_z);
    const engine = @import("plat.zig").selfExe(engine_z, &engine_buf) orelse {
        std.debug.print("node: cannot find the engine at {s}\n", .{opts.engine});
        std.process.exit(2);
    };
    const log_z = try gpa.dupeSentinel(u8, opts.log_dir, 0);
    defer gpa.free(log_z);
    @import("plat.zig").makeDir(log_z);
    std.debug.print("node: {s}, engine {s}, models in {s}, dashboard {s}:{d}\n", .{ name, engine, opts.models, opts.host, opts.port });

    var rank: ?Rank = null;
    while (true)
    {
        session(gpa, opts, name, secret, engine, &rank) catch |err| {
            std.debug.print("node: link to the dashboard ended: {t}\n", .{err});
        };
        if (rank) |r|
        {
            std.debug.print("node: stopping rank {d}, since its dashboard is gone\n", .{r.start.rank});
            dropRank(gpa, &rank);
        }
        proc.sleepMs(2000);
    }
}

/// Takes the options, and the paths of the secret and logs, from a settings file.
fn applySettings(gpa: std.mem.Allocator, opts: *Options, path: []const u8) !void
{
    const settings = @import("settings.zig");
    const s = settings.load(gpa, path) orelse {
        std.debug.print("node: cannot read the settings in {s}\n", .{path});
        std.process.exit(2);
    };
    const n = s.node;
    if (settings.hostPort(n.dashboard, default_port)) |hp|
    {
        opts.host = hp.host;
        opts.port = hp.port;
    }
    opts.models = n.models;
    opts.name = if (n.name.len > 0) n.name else null;
    opts.advertise = if (n.advertise.len > 0) n.advertise else null;
    opts.secret_file = try settings.beside(gpa, path, "node.secret");
    opts.log_dir = try settings.beside(gpa, path, "node-logs");
}

/// One link to the dashboard, until it fails.
fn session(gpa: std.mem.Allocator, opts: Options, name: []const u8, secret: []const u8, engine: []const u8, rank: *?Rank) !void
{
    const fd = tcp.connect(opts.host, opts.port, 1) catch return error.DashboardUnreachable;
    defer tcp.closeSocket(fd);
    var link = try auth.Link.init(gpa, fd, true);
    defer link.deinit(gpa);
    auth.handshakeNode(&link, secret, name) catch |err| {
        if (err == error.DashboardNotTrusted)
        {
            std.debug.print("node: the dashboard at {s}:{d} could not prove it holds the node secret; not linking\n", .{ opts.host, opts.port });
            proc.sleepMs(3000);
        }
        return err;
    };
    std.debug.print("node: linked to the dashboard as {s}\n", .{name});
    var last_report: u64 = 0;
    var changed = true;
    while (true)
    {
        if (rank.*) |*r|
        {
            if (!r.listening and r.child.logHas(gpa, "listening on port"))
            {
                r.listening = true;
                changed = true;
            }
            const was = r.child.exit;
            if (!r.child.alive() and was == null)
            {
                std.debug.print("node: rank {d} exited with {d}\n", .{ r.start.rank, r.child.exit.? });
                changed = true;
            }
        }
        const now = proc.monoNs();
        if (changed or now - last_report > 2 * std.time.ns_per_s)
        {
            try sendReport(gpa, &link, opts, name, engine, rank.*);
            last_report = now;
            changed = false;
        }
        const f = (try link.recv(250)) orelse continue;
        switch (f.kind)
        {
            .start =>
            {
                var arena: std.heap.ArenaAllocator = .init(gpa);
                defer arena.deinit();
                startRank(gpa, arena.allocator(), opts, engine, f.body, rank) catch |err| {
                    std.debug.print("node: refused a start command: {t}\n", .{err});
                };
                changed = true;
            },
            .stop =>
            {
                if (rank.*) |r|
                {
                    std.debug.print("node: stopping rank {d}\n", .{r.start.rank});
                    dropRank(gpa, rank);
                }
                changed = true;
            },
            .refuse =>
            {
                std.debug.print("node: the dashboard closed the link: {s}\n", .{f.body[0..@min(f.body.len, 200)]});
                return error.Refused;
            },
            else => return error.UnexpectedFrame,
        }
    }
}

/// Checks a start command and starts the rank, replacing any running one.
fn startRank(gpa: std.mem.Allocator, a: std.mem.Allocator, opts: Options, engine: []const u8, body: []const u8, rank: *?Rank) !void
{
    const s = try std.json.parseFromSliceLeaky(Start, a, body, .{});
    const rdma = std.mem.eql(u8, s.transport, "rdma");
    if (!rdma and !std.mem.eql(u8, s.transport, "tcp"))
    {
        return error.BadTransport;
    }
    if (s.ranks < 2 or s.ranks > 64 or s.rank < 1 or s.rank >= s.ranks or s.port < 1024)
    {
        return error.BadRank;
    }
    if (proc.modelSize(a, opts.models, s.model) == null)
    {
        return error.NoSuchModel;
    }
    if (s.dev) |d|
    {
        if (!proc.validDevice(d))
        {
            return error.BadDevice;
        }
    }
    if (s.max_step) |m|
    {
        if (m == 0 or m > 65536)
        {
            return error.BadMaxStep;
        }
    }
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ engine, if (rdma) "rdma-expert" else "expert", try std.fmt.allocPrint(a, "{s}/{s}", .{ opts.models, s.model }) });
    try argv.appendSlice(a, &.{ "--port", try std.fmt.allocPrint(a, "{d}", .{s.port}), "--rank", try std.fmt.allocPrint(a, "{d}", .{s.rank}), "--ranks", try std.fmt.allocPrint(a, "{d}", .{s.ranks}) });
    if (s.dev) |d|
    {
        try argv.appendSlice(a, &.{ "--dev", d });
    }
    if (s.gpu)
    {
        try argv.append(a, "--gpu");
    }
    if (s.max_step) |m|
    {
        try argv.appendSlice(a, &.{ "--max-step", try std.fmt.allocPrint(a, "{d}", .{m}) });
    }
    if (s.slots) |n|
    {
        try argv.appendSlice(a, &.{ "--slots", try std.fmt.allocPrint(a, "{d}", .{n}) });
    }
    if (s.cache) |n|
    {
        if (n > 0)
        {
            try argv.appendSlice(a, &.{ "--cache", try std.fmt.allocPrint(a, "{d}", .{n}) });
        }
    }
    dropRank(gpa, rank);
    const log_path = try std.fmt.allocPrint(a, "{s}/rank{d}.log", .{ opts.log_dir, s.rank });
    var child = try proc.Child.spawn(gpa, argv.items, log_path);
    errdefer child.deinit(gpa);
    // The command's strings live in the arena, so keep copies for the report.
    var kept = s;
    kept.model = try gpa.dupe(u8, s.model);
    kept.transport = if (rdma) "rdma" else "tcp";
    kept.dev = null;
    rank.* = .{ .start = kept, .child = child };
    std.debug.print("node: started rank {d} of {d} ({s} over {s}, port {d}) as pid {d}, log {s}\n", .{ s.rank, s.ranks, s.model, kept.transport, s.port, child.pid, log_path });
}

fn dropRank(gpa: std.mem.Allocator, rank: *?Rank) void
{
    if (rank.*) |*r|
    {
        r.child.stop(5000);
        r.child.deinit(gpa);
        gpa.free(r.start.model);
        rank.* = null;
    }
}

fn sendReport(gpa: std.mem.Allocator, link: *auth.Link, opts: Options, name: []const u8, engine: []const u8, rank: ?Rank) !void
{
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    var host_buf: [64]u8 = undefined;
    const mem = proc.memory(a);
    try out.appendSlice(a, "{\"name\":");
    try proc.jsonString(a, &out, name);
    try out.appendSlice(a, ",\"host\":");
    try proc.jsonString(a, &out, proc.hostName(&host_buf));
    try out.appendSlice(a, ",\"engine\":");
    try proc.jsonString(a, &out, engine);
    try out.print(a, ",\"mem_total_mb\":{d},\"mem_available_mb\":{d},\"gpu\":", .{ mem.total_mb, mem.available_mb });
    if (proc.gpuName(a)) |g|
    {
        try proc.jsonString(a, &out, g);
    }
    else
    {
        try out.appendSlice(a, "null");
    }
    if (opts.advertise) |ip|
    {
        try out.print(a, ",\"advertise\":\"{s}\"", .{ip});
    }
    try out.appendSlice(a, ",\"models\":[");
    for (try proc.listModels(a, opts.models), 0..) |m, i|
    {
        try out.appendSlice(a, if (i == 0) "{\"name\":" else ",{\"name\":");
        try proc.jsonString(a, &out, m.name);
        try out.print(a, ",\"bytes\":{d}}}", .{m.bytes});
    }
    try out.appendSlice(a, "],\"rdma\":[");
    for (try proc.rdmaDevices(a), 0..) |r, i|
    {
        try out.appendSlice(a, if (i == 0) "{\"dev\":" else ",{\"dev\":");
        try proc.jsonString(a, &out, r.dev);
        try out.appendSlice(a, ",\"ips\":[");
        for (r.ips, 0..) |ip, j|
        {
            var ib: [16]u8 = undefined;
            try out.print(a, "{s}\"{s}\"", .{ if (j == 0) "" else ",", proc.formatIp4(ip, &ib) });
        }
        try out.appendSlice(a, "]}");
    }
    try out.appendSlice(a, "],\"rank\":");
    if (rank) |r|
    {
        const state = if (r.child.exit != null) "exited" else if (r.listening) "listening" else "starting";
        try out.print(a, "{{\"rank\":{d},\"ranks\":{d},\"port\":{d},\"pid\":{d},\"state\":\"{s}\",\"exit\":", .{ r.start.rank, r.start.ranks, r.start.port, r.child.pid, state });
        if (r.child.exit) |e|
        {
            try out.print(a, "{d}", .{e});
        }
        else
        {
            try out.appendSlice(a, "null");
        }
        try out.appendSlice(a, ",\"model\":");
        try proc.jsonString(a, &out, r.start.model);
        try out.appendSlice(a, ",\"log\":");
        try proc.jsonString(a, &out, proc.tail(a, r.child.log_path, 6 << 10) catch "");
        try out.append(a, '}');
    }
    else
    {
        try out.appendSlice(a, "null");
    }
    try out.append(a, '}');
    try link.send(.report, out.items);
}
