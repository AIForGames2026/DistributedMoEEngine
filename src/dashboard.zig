//! The operator dashboard behind `olmoe dashboard` (TASK-25). Linux and Windows (TASK-43).
//!
//!   olmoe dashboard --models DIR [--listen [IP:]PORT] [--node-port P]
//!                   [--token-file F] [--node-secret-file F] [--api-key-file F]
//!                   [--nodes-file F] [--engine PATH] [--router-port P]
//!                   [--expert-port P] [--log-dir DIR]
//!
//! A supervisor on the machine that runs rank 0. It serves a page where an
//! operator signs in with the admin token, approves node agents (`olmoe node`)
//! as they link, loads a model across the chosen nodes, and chats with it. It
//! starts the router (rank 0, with `--serve` on a local port) as its own child,
//! tells each node's agent to start one expert rank, and stops them all again.
//! Adding a node takes effect on the next load. The OpenAI-compatible API that
//! harnesses call is the dashboard's own `/v1`, passed through to the router.
//!
//!   GET  /                       the dashboard page
//!   POST /api/login, /api/logout
//!   GET  /api/state              nodes, models and the run       (signed in)
//!   POST /api/nodes/approve      {"name"}                        (signed in)
//!   POST /api/nodes/forget       {"name"}                        (signed in)
//!   POST /api/load               {"model", "nodes", ...}         (signed in)
//!   POST /api/stop                                               (signed in)
//!   GET  /live, /telemetry       the router's live diagram       (signed in)
//!   /v1/...                      the router's API (the API key, if set, or signed in)
//!
//! Every control action needs a session cookie from the login or the admin
//! token as a Bearer header, and POSTs from pages served elsewhere are refused.
//! It binds to 127.0.0.1 unless told otherwise; it speaks plain HTTP, so put a
//! TLS proxy in front of it anywhere but a private network.
const std = @import("std");
const c = std.c;
const http = @import("http.zig");
const sock = @import("sock.zig");
const plat = @import("plat.zig");
const openai = @import("openai.zig");
const auth = @import("auth.zig");
const proc = @import("proc.zig");
const tcp = @import("transport/tcp.zig");
const node_agent = @import("node_agent.zig");

const page_html = @embedFile("dashboard.html");
const page_js = @embedFile("dashboard.js");

const max_nodes = 32;
const max_conns = 64;
const max_head = 16 << 10;
const max_body = 8 << 20;
/// A node that has sent nothing for this long is dropped.
const node_timeout_ns = 8 * std.time.ns_per_s;
const cookie_name = "tf_session";

const Options = struct {
    addr: u32 = @bitCast([4]u8{ 127, 0, 0, 1 }),
    port: u16 = 8080,
    node_port: u16 = node_agent.default_port,
    token_file: []const u8 = "dashboard.token",
    secret_file: []const u8 = "node.secret",
    api_key_file: ?[]const u8 = null,
    nodes_file: []const u8 = "dashboard.nodes",
    models: []const u8 = "",
    engine: []const u8 = "/proc/self/exe",
    router_port: u16 = 18080,
    expert_port: u16 = 47401,
    log_dir: []const u8 = "dashboard-logs",
};

/// A node agent as the dashboard knows it.
const Node = struct {
    name_buf: [64]u8 = undefined,
    name_len: usize = 0,
    approved: bool = false,
    /// The live link, while the agent is connected.
    conn: ?*Conn = null,
    addr: u32 = 0,
    /// Its latest report (JSON), owned.
    report: ?[]u8 = null,
    seen_ns: u64 = 0,

    fn name(self: *const Node) []const u8
    {
        return self.name_buf[0..self.name_len];
    }
};

/// One agent's link; sends are serialized by `mutex`.
const Conn = struct {
    link: auth.Link,
    mutex: plat.Mutex = .{},

    fn send(self: *Conn, kind: auth.Kind, body: []const u8) !void
    {
        self.mutex.acquire();
        defer self.mutex.release();
        try self.link.send(kind, body);
    }
};

const RunState = enum { idle, starting, running, stopping, failed };

/// Everything one load needs, built when the request is checked.
const Plan = struct {
    arena: std.heap.ArenaAllocator,
    model: []const u8,
    /// Node names, rank 1 first.
    nodes: []const []const u8,
    /// The start command for each node.
    starts: []const []const u8,
    router_argv: []const []const u8,
    /// The settings as JSON, for the page.
    settings: []const u8,
};

const State = struct {
    mutex: plat.Mutex = .{},
    nodes: [max_nodes]Node = @splat(.{}),
    run: RunState = .idle,
    message: std.ArrayList(u8) = .empty,
    since_ns: u64 = 0,
    plan: ?*Plan = null,
    /// What the supervisor should do next.
    pending: ?*Plan = null,
    stop_requested: bool = false,
    router_pid: ?i32 = null,

    fn lock(self: *State) void
    {
        self.mutex.acquire();
    }

    fn unlock(self: *State) void
    {
        self.mutex.release();
    }

    fn find(self: *State, name: []const u8) ?*Node
    {
        for (&self.nodes) |*n|
        {
            if (n.name_len > 0 and std.mem.eql(u8, n.name(), name))
            {
                return n;
            }
        }
        return null;
    }

    fn add(self: *State, name: []const u8) ?*Node
    {
        if (self.find(name)) |n|
        {
            return n;
        }
        for (&self.nodes) |*n|
        {
            if (n.name_len == 0)
            {
                n.* = .{ .name_len = name.len };
                @memcpy(n.name_buf[0..name.len], name);
                return n;
            }
        }
        return null;
    }

    /// Sets the run's status line; call with the lock held.
    fn say(self: *State, comptime fmt: []const u8, args: anytype) void
    {
        self.message.clearRetainingCapacity();
        self.message.print(gpa_global, fmt, args) catch {};
        var t: [16]u8 = undefined;
        std.debug.print("{s} dashboard: {s}\n", .{ stamp(&t), self.message.items });
    }
};

var state: State = .{};
var sessions: auth.Sessions = .{};
var opts_global: Options = .{};
var gpa_global: std.mem.Allocator = undefined;
var admin_token: []const u8 = "";
var node_secret: []const u8 = "";
var api_key: ?[]const u8 = null;
var engine_path: []const u8 = "";
var login_mutex: plat.Mutex = .{};
var open_conns: std.atomic.Value(usize) = .init(0);

const usage =
    \\usage: olmoe dashboard --models DIR [--listen [IP:]PORT] [--node-port P] [--token-file F]
    \\                       [--node-secret-file F] [--api-key-file F] [--nodes-file F] [--engine PATH]
    \\                       [--router-port P] [--expert-port P] [--log-dir DIR]
    \\
;

pub fn main(gpa: std.mem.Allocator, args: []const []const u8) !void
{
    gpa_global = gpa;
    var opts: Options = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 2)
    {
        if (i + 1 >= args.len)
        {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--listen"))
        {
            if (std.mem.indexOfScalar(u8, v, ':')) |colon|
            {
                opts.addr = try tcp.parseIp4(v[0..colon]);
                opts.port = try std.fmt.parseInt(u16, v[colon + 1 ..], 10);
            }
            else
            {
                opts.port = try std.fmt.parseInt(u16, v, 10);
            }
        }
        else if (std.mem.eql(u8, k, "--node-port"))
        {
            opts.node_port = try std.fmt.parseInt(u16, v, 10);
        }
        else if (std.mem.eql(u8, k, "--token-file"))
        {
            opts.token_file = v;
        }
        else if (std.mem.eql(u8, k, "--node-secret-file"))
        {
            opts.secret_file = v;
        }
        else if (std.mem.eql(u8, k, "--api-key-file"))
        {
            opts.api_key_file = v;
        }
        else if (std.mem.eql(u8, k, "--nodes-file"))
        {
            opts.nodes_file = v;
        }
        else if (std.mem.eql(u8, k, "--models"))
        {
            opts.models = v;
        }
        else if (std.mem.eql(u8, k, "--engine"))
        {
            opts.engine = v;
        }
        else if (std.mem.eql(u8, k, "--router-port"))
        {
            opts.router_port = try std.fmt.parseInt(u16, v, 10);
        }
        else if (std.mem.eql(u8, k, "--expert-port"))
        {
            opts.expert_port = try std.fmt.parseInt(u16, v, 10);
        }
        else if (std.mem.eql(u8, k, "--log-dir"))
        {
            opts.log_dir = v;
        }
        else
        {
            std.debug.print("dashboard: unknown option {s}\n" ++ usage, .{k});
            std.process.exit(2);
        }
    }
    if (opts.models.len == 0)
    {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    opts_global = opts;
    admin_token = try auth.loadSecret(gpa, opts.token_file, "admin token");
    node_secret = try auth.loadSecret(gpa, opts.secret_file, "node secret");
    if (opts.api_key_file) |f|
    {
        api_key = try auth.loadSecret(gpa, f, "API key");
    }
    var engine_buf: [4096]u8 = undefined;
    const engine_z = try gpa.dupeSentinel(u8, opts.engine, 0);
    defer gpa.free(engine_z);
    engine_path = try gpa.dupe(u8, plat.selfExe(engine_z, &engine_buf) orelse {
        std.debug.print("dashboard: cannot find the engine at {s}\n", .{opts.engine});
        std.process.exit(2);
    });
    const log_z = try gpa.dupeSentinel(u8, opts.log_dir, 0);
    defer gpa.free(log_z);
    plat.makeDir(log_z);
    loadApproved(gpa);

    const web_fd = try listenOn(opts.addr, opts.port);
    const node_fd = try listenOn(0, opts.node_port);
    _ = try std.Thread.spawn(.{}, nodeAccept, .{node_fd});
    _ = try std.Thread.spawn(.{}, webAccept, .{web_fd});
    var ip: [16]u8 = undefined;
    std.debug.print("dashboard: http://{s}:{d}/ (sign in with the token in {s}); node agents link on port {d} with the secret in {s}; engine {s}\n", .{
        proc.formatIp4(opts.addr, &ip), opts.port, opts.token_file, opts.node_port, opts.secret_file, engine_path,
    });
    std.debug.print("dashboard: the OpenAI-compatible API is at http://{s}:{d}/v1 once a model is loaded{s}\n", .{ proc.formatIp4(opts.addr, &ip), opts.port, if (api_key != null) " (with the API key)" else "" });
    const b: [4]u8 = @bitCast(opts.addr);
    if (b[0] != 127)
    {
        std.debug.print("dashboard: listening beyond this machine over plain HTTP; use it only on a private network or behind a TLS proxy\n", .{});
    }
    // The supervisor runs on this thread, which lives as long as the router it starts.
    supervise();
}

fn listenOn(addr: u32, port: u16) !sock.Fd
{
    return sock.listenTcp(addr, port, 64) catch |err|
    {
        if (err == error.AddressInUse)
        {
            std.debug.print("dashboard: port {d} is in use\n", .{port});
        }
        return err;
    };
}

fn stamp(buf: *[16]u8) []const u8
{
    const ms = plat.unixMs();
    const day = ms % (24 * 3600 * 1000);
    return std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}", .{ day / 3600000, day / 60000 % 60, day / 1000 % 60, day % 1000 }) catch unreachable;
}

/// Reads the approved node names, one per line.
fn loadApproved(gpa: std.mem.Allocator) void
{
    const text = proc.readSmall(gpa, opts_global.nodes_file) orelse return;
    defer gpa.free(text);
    var it = std.mem.tokenizeAny(u8, text, " \r\n");
    var count: usize = 0;
    while (it.next()) |name|
    {
        if (auth.validName(name))
        {
            if (state.add(name)) |n|
            {
                n.approved = true;
                count += 1;
            }
        }
    }
    std.debug.print("dashboard: {d} approved node{s} from {s}\n", .{ count, if (count == 1) "" else "s", opts_global.nodes_file });
}

/// Writes the approved node names; call with the lock held.
fn saveApproved() void
{
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa_global);
    for (&state.nodes) |*n|
    {
        if (n.name_len > 0 and n.approved)
        {
            out.print(gpa_global, "{s}\n", .{n.name()}) catch return;
        }
    }
    const tmp = std.fmt.allocPrintSentinel(gpa_global, "{s}.tmp", .{opts_global.nodes_file}, 0) catch return;
    defer gpa_global.free(tmp);
    const dst = gpa_global.dupeSentinel(u8, opts_global.nodes_file, 0) catch return;
    defer gpa_global.free(dst);
    _ = plat.replaceFile(tmp, dst, out.items);
}

fn nodeAccept(lfd: sock.Fd) void
{
    while (true)
    {
        const got = sock.acceptFrom(lfd) orelse continue;
        const fd = got.fd;
        if (open_conns.fetchAdd(1, .acq_rel) >= max_conns)
        {
            _ = open_conns.fetchSub(1, .acq_rel);
            sock.close(fd);
            continue;
        }
        const t = std.Thread.spawn(.{}, nodeConn, .{ fd, got.addr }) catch {
            _ = open_conns.fetchSub(1, .acq_rel);
            sock.close(fd);
            continue;
        };
        t.detach();
    }
}

/// One agent's link: the handshake, then its reports until it goes quiet or away.
fn nodeConn(fd: sock.Fd, addr: u32) void
{
    defer _ = open_conns.fetchSub(1, .acq_rel);
    defer sock.close(fd);
    const gpa = gpa_global;
    const conn = gpa.create(Conn) catch return;
    defer gpa.destroy(conn);
    conn.* = .{ .link = auth.Link.init(gpa, fd, false) catch return };
    defer conn.link.deinit(gpa);
    var ip: [16]u8 = undefined;
    var name_buf: [64]u8 = undefined;
    const got = auth.handshakeDashboard(&conn.link, node_secret, &name_buf) catch |err| {
        std.debug.print("dashboard: refused a node link from {s}: {t}\n", .{ proc.formatIp4(addr, &ip), err });
        return;
    };
    var name_copy: [64]u8 = undefined;
    @memcpy(name_copy[0..got.len], got);
    const name = name_copy[0..got.len];
    {
        state.lock();
        defer state.unlock();
        const n = state.add(name) orelse {
            conn.send(.refuse, "The dashboard has no room for more nodes.") catch {};
            return;
        };
        if (n.conn != null)
        {
            conn.send(.refuse, "A node with this name is already linked.") catch {};
            std.debug.print("dashboard: refused a second link for node {s} from {s}\n", .{ name, proc.formatIp4(addr, &ip) });
            return;
        }
        n.conn = conn;
        n.addr = addr;
        n.seen_ns = proc.monoNs();
        std.debug.print("dashboard: node {s} linked from {s}{s}\n", .{ name, proc.formatIp4(addr, &ip), if (n.approved) "" else ", waiting for approval" });
    }
    var why: []const u8 = "link closed";
    while (true)
    {
        const f = conn.link.recv(1000) catch |err| {
            why = @errorName(err);
            break;
        };
        state.lock();
        const n = state.find(name).?;
        if (f == null)
        {
            const quiet = proc.monoNs() - n.seen_ns > node_timeout_ns;
            state.unlock();
            if (quiet)
            {
                why = "no report for 8 s";
                break;
            }
            continue;
        }
        n.seen_ns = proc.monoNs();
        if (f.?.kind != .report or !validJson(f.?.body))
        {
            state.unlock();
            why = "sent a frame it should not";
            break;
        }
        const copy = gpa.dupe(u8, f.?.body) catch null;
        if (copy) |cp|
        {
            if (n.report) |old|
            {
                gpa.free(old);
            }
            n.report = cp;
        }
        state.unlock();
    }
    state.lock();
    const n = state.find(name).?;
    n.conn = null;
    state.unlock();
    std.debug.print("dashboard: node {s} went offline ({s})\n", .{ name, why });
}

fn validJson(body: []const u8) bool
{
    var arena: std.heap.ArenaAllocator = .init(gpa_global);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch return false;
    return v == .object;
}

/// The parts of a node's report the dashboard acts on.
const Report = struct {
    advertise: ?[]const u8 = null,
    models: []const struct { name: []const u8, bytes: u64 } = &.{},
    rdma: []const struct { dev: []const u8, ips: []const []const u8 } = &.{},
    rank: ?struct { rank: u16, state: []const u8, exit: ?u8 = null } = null,
};

fn parseReport(a: std.mem.Allocator, body: []const u8) ?Report
{
    return std.json.parseFromSliceLeaky(Report, a, body, .{ .ignore_unknown_fields = true }) catch null;
}

fn webAccept(lfd: sock.Fd) void
{
    while (true)
    {
        const fd = sock.accept(lfd) orelse continue;
        if (open_conns.fetchAdd(1, .acq_rel) >= max_conns)
        {
            _ = open_conns.fetchSub(1, .acq_rel);
            sendSimple(fd, 503, "application/json", "{\"error\":\"The dashboard is busy.\"}", "") catch {};
            sock.close(fd);
            continue;
        }
        const t = std.Thread.spawn(.{}, webConn, .{fd}) catch {
            _ = open_conns.fetchSub(1, .acq_rel);
            sock.close(fd);
            continue;
        };
        t.detach();
    }
}

fn webConn(fd: sock.Fd) void
{
    defer _ = open_conns.fetchSub(1, .acq_rel);
    defer sock.close(fd);
    sock.setTimeouts(fd, 10, 30);
    var arena: std.heap.ArenaAllocator = .init(gpa_global);
    defer arena.deinit();
    handle(arena.allocator(), fd) catch {};
}

/// Reads one request and answers it.
fn handle(a: std.mem.Allocator, fd: sock.Fd) !void
{
    var buf: std.ArrayList(u8) = .empty;
    var head: http.Head = undefined;
    var continued = false;
    while (true)
    {
        if (http.parseHead(buf.items, max_head)) |h|
        {
            head = h;
            if (h.content_length > max_body)
            {
                return apiError(fd, 413, "The request body is too large.");
            }
            if (buf.items.len >= h.len + h.content_length)
            {
                break;
            }
            if (h.expect_continue and !continued)
            {
                try sendAll(fd, "HTTP/1.1 100 Continue\r\n\r\n");
                continued = true;
            }
        }
        else |err| switch (err)
        {
            error.Incomplete => {},
            error.LengthRequired => return apiError(fd, 411, "Send the body with a Content-Length."),
            error.TooLarge => return apiError(fd, 413, "The request head is too large."),
            error.BadRequest => return apiError(fd, 400, "Malformed HTTP request."),
        }
        try buf.ensureUnusedCapacity(a, 64 << 10);
        const n = sock.recv(fd, buf.unusedCapacitySlice());
        if (n <= 0)
        {
            return;
        }
        buf.items.len += @intCast(n);
    }
    const raw = buf.items[0 .. head.len + head.content_length];
    const body = raw[head.len..];
    const path = head.path;
    const get = std.mem.eql(u8, head.method, "GET");
    const post = std.mem.eql(u8, head.method, "POST");

    if (std.mem.eql(u8, path, "/") and get)
    {
        return sendSimple(fd, 200, "text/html; charset=utf-8", page_html, page_headers);
    }
    if (std.mem.eql(u8, path, "/dashboard.js") and get)
    {
        return sendSimple(fd, 200, "text/javascript; charset=utf-8", page_js, "");
    }
    if (std.mem.eql(u8, path, "/health"))
    {
        return sendSimple(fd, 200, "application/json", "{\"status\":\"ok\"}", "");
    }
    if (std.mem.startsWith(u8, path, "/v1/"))
    {
        if (api_key) |key|
        {
            const ok_key = if (head.authorization) |h| (if (auth.bearer(h)) |t| auth.secretEql(t, key) else false) else false;
            if (!ok_key and !signedIn(head))
            {
                return sendOpenAiError(fd, 401, "This endpoint needs the API key as a Bearer token.", "authentication_error");
            }
        }
        if (!routerUp())
        {
            return sendOpenAiError(fd, 503, "No model is loaded; load one from the dashboard first.", "server_error");
        }
        return proxy(a, fd, raw, null);
    }
    if (std.mem.eql(u8, path, "/api/login"))
    {
        if (!post)
        {
            return apiError(fd, 405, "Use POST.");
        }
        if (!http.sameOrigin(head))
        {
            return apiError(fd, 403, "Sign in from the dashboard's own page.");
        }
        return login(a, fd, body);
    }
    // Everything below needs a signed-in operator.
    if (std.mem.startsWith(u8, path, "/api/") or std.mem.eql(u8, path, "/live") or std.mem.eql(u8, path, "/telemetry"))
    {
        if (!signedIn(head))
        {
            return apiError(fd, 401, "Sign in first.");
        }
    }
    else
    {
        return apiError(fd, 404, "Not found.");
    }
    if (std.mem.eql(u8, path, "/live") or std.mem.eql(u8, path, "/telemetry"))
    {
        if (!routerUp())
        {
            return apiError(fd, 503, "No model is loaded.");
        }
        return proxy(a, fd, raw, if (std.mem.eql(u8, path, "/live")) "/" else null);
    }
    if (std.mem.eql(u8, path, "/api/state"))
    {
        if (!get)
        {
            return apiError(fd, 405, "Use GET.");
        }
        return sendSimple(fd, 200, "application/json", try stateJson(a), "");
    }
    if (!post)
    {
        return apiError(fd, 405, "Use POST.");
    }
    if (!http.sameOrigin(head))
    {
        return apiError(fd, 403, "Control requests must come from the dashboard's own page.");
    }
    if (std.mem.eql(u8, path, "/api/logout"))
    {
        if (head.cookie) |ck|
        {
            if (auth.cookieValue(ck, cookie_name)) |id|
            {
                sessions.revoke(id, proc.monoNs());
            }
        }
        return sendSimple(fd, 200, "application/json", "{\"ok\":true}", "Set-Cookie: " ++ cookie_name ++ "=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0\r\n");
    }
    if (std.mem.eql(u8, path, "/api/nodes/approve") or std.mem.eql(u8, path, "/api/nodes/forget"))
    {
        return approve(a, fd, body, std.mem.endsWith(u8, path, "approve"));
    }
    if (std.mem.eql(u8, path, "/api/load"))
    {
        return load(a, fd, body);
    }
    if (std.mem.eql(u8, path, "/api/stop"))
    {
        state.lock();
        defer state.unlock();
        if (state.run == .idle or state.run == .failed)
        {
            return apiError(fd, 409, "Nothing is running.");
        }
        state.stop_requested = true;
        return sendSimple(fd, 202, "application/json", "{\"ok\":true}", "");
    }
    return apiError(fd, 404, "Not found.");
}

const page_headers = "Content-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'; form-action 'none'\r\nX-Frame-Options: SAMEORIGIN\r\n";

fn signedIn(head: http.Head) bool
{
    if (head.cookie) |ck|
    {
        if (auth.cookieValue(ck, cookie_name)) |id|
        {
            if (sessions.valid(id, proc.monoNs()))
            {
                return true;
            }
        }
    }
    if (head.authorization) |h|
    {
        if (auth.bearer(h)) |t|
        {
            return auth.secretEql(t, admin_token);
        }
    }
    return false;
}

fn login(a: std.mem.Allocator, fd: sock.Fd, body: []const u8) !void
{
    const req = std.json.parseFromSliceLeaky(struct { token: []const u8 }, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "Send {\"token\": \"...\"}.");
    };
    // One attempt at a time, and a wrong one costs a second, so guessing is slow.
    login_mutex.acquire();
    const ok = auth.secretEql(std.mem.trim(u8, req.token, " \t\r\n"), admin_token);
    if (!ok)
    {
        proc.sleepMs(1000);
    }
    login_mutex.release();
    if (!ok)
    {
        std.debug.print("dashboard: a sign-in with a wrong token was refused\n", .{});
        return apiError(fd, 401, "That is not the admin token.");
    }
    var id: [auth.id_len]u8 = undefined;
    sessions.create(proc.monoNs(), &id);
    const cookie = try std.fmt.allocPrint(a, "Set-Cookie: {s}={s}; HttpOnly; SameSite=Strict; Path=/; Max-Age={d}\r\n", .{ cookie_name, id, auth.session_ttl_ns / std.time.ns_per_s });
    std.debug.print("dashboard: an operator signed in\n", .{});
    return sendSimple(fd, 200, "application/json", "{\"ok\":true}", cookie);
}

fn approve(a: std.mem.Allocator, fd: sock.Fd, body: []const u8, yes: bool) !void
{
    const req = std.json.parseFromSliceLeaky(struct { name: []const u8 }, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "Send {\"name\": \"...\"}.");
    };
    state.lock();
    defer state.unlock();
    const n = state.find(req.name) orelse return apiError(fd, 404, "No node by that name.");
    if (!yes and state.plan != null and state.run != .idle and state.run != .failed)
    {
        for (state.plan.?.nodes) |used|
        {
            if (std.mem.eql(u8, used, req.name))
            {
                return apiError(fd, 409, "That node is part of the loaded model; stop it first.");
            }
        }
    }
    n.approved = yes;
    if (!yes and n.conn == null)
    {
        if (n.report) |r|
        {
            gpa_global.free(r);
        }
        n.* = .{};
    }
    saveApproved();
    std.debug.print("dashboard: node {s} {s} by the operator\n", .{ req.name, if (yes) "approved" else "forgotten" });
    return sendSimple(fd, 200, "application/json", "{\"ok\":true}", "");
}

const LoadRequest = struct {
    model: []const u8,
    nodes: []const []const u8 = &.{},
    transport: []const u8 = "tcp",
    gpu: bool = false,
    ctx: u32 = 4096,
    payload: []const u8 = "fp8",
    kv_format: []const u8 = "f32",
    reuse: bool = true,
    max_step: ?u32 = null,
    cache: u32 = 0,
};

fn load(a: std.mem.Allocator, fd: sock.Fd, body: []const u8) !void
{
    const req = std.json.parseFromSliceLeaky(LoadRequest, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "The load request is malformed.");
    };
    const plan = gpa_global.create(Plan) catch return apiError(fd, 500, "Out of memory.");
    plan.* = .{ .arena = .init(gpa_global), .model = "", .nodes = &.{}, .starts = &.{}, .router_argv = &.{}, .settings = "" };
    var handed = false;
    defer if (!handed)
    {
        plan.arena.deinit();
        gpa_global.destroy(plan);
    };
    var why: std.ArrayList(u8) = .empty;
    state.lock();
    defer state.unlock();
    if (state.run == .starting or state.run == .stopping or state.pending != null)
    {
        return apiError(fd, 409, "A load or stop is already under way.");
    }
    buildPlan(plan, req, &why) catch |err| {
        if (err == error.Refused)
        {
            return apiError(fd, 400, why.items);
        }
        return apiError(fd, 500, "Could not build the plan.");
    };
    state.pending = plan;
    handed = true;
    state.run = .starting;
    state.since_ns = proc.monoNs();
    state.say("loading {s} on {d} process{s}", .{ plan.model, plan.nodes.len + 1, if (plan.nodes.len == 0) "" else "es" });
    return sendSimple(fd, 202, "application/json", "{\"ok\":true}", "");
}

fn refuse(why: *std.ArrayList(u8), comptime fmt: []const u8, args: anytype) error{Refused}
{
    why.clearRetainingCapacity();
    why.print(gpa_global, fmt, args) catch {};
    return error.Refused;
}

/// Checks a load request against what the nodes reported and builds its
/// commands; call with the lock held.
fn buildPlan(plan: *Plan, req: LoadRequest, why: *std.ArrayList(u8)) !void
{
    const a = plan.arena.allocator();
    const opts = opts_global;
    const size = proc.modelSize(a, opts.models, req.model) orelse return refuse(why, "{s} is not a model in {s} (a .gguf with its .tfsum index).", .{ req.model, opts.models });
    const rdma = std.mem.eql(u8, req.transport, "rdma");
    if (!rdma and !std.mem.eql(u8, req.transport, "tcp"))
    {
        return refuse(why, "The transport is tcp or rdma.", .{});
    }
    const payloads = [_][]const u8{ "fp8", "bf16", "f32" };
    const kv_formats = [_][]const u8{ "f32", "bf16", "fp8", "bf16_fp8" };
    if (!oneOf(req.payload, &payloads))
    {
        return refuse(why, "The payload is fp8, bf16 or f32.", .{});
    }
    if (!oneOf(req.kv_format, &kv_formats))
    {
        return refuse(why, "The KV format is f32, bf16, fp8 or bf16_fp8.", .{});
    }
    if (req.ctx < 256 or req.ctx > 1 << 20)
    {
        return refuse(why, "The context must be 256 to 1,048,576 tokens.", .{});
    }
    if (req.nodes.len + 1 > 64)
    {
        return refuse(why, "At most 63 nodes.", .{});
    }
    const local_rdma = if (rdma) try proc.rdmaDevices(a) else &.{};
    var starts: std.ArrayList([]const u8) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    var peers: std.ArrayList([]const u8) = .empty;
    const ranks = req.nodes.len + 1;
    for (req.nodes, 1..) |name, rank|
    {
        for (names.items) |seen|
        {
            if (std.mem.eql(u8, seen, name))
            {
                return refuse(why, "Node {s} is listed twice.", .{name});
            }
        }
        const n = state.find(name) orelse return refuse(why, "No node is named {s}.", .{name});
        if (!n.approved)
        {
            return refuse(why, "Node {s} has not been approved.", .{name});
        }
        if (n.conn == null or n.report == null)
        {
            return refuse(why, "Node {s} is offline.", .{name});
        }
        const r = parseReport(a, n.report.?) orelse return refuse(why, "Node {s} sent a report the dashboard cannot read.", .{name});
        var has = false;
        for (r.models) |m|
        {
            has = has or (std.mem.eql(u8, m.name, req.model) and m.bytes == size);
        }
        if (!has)
        {
            return refuse(why, "Node {s} does not have {s} (the same file, {d} bytes) in its models directory.", .{ name, req.model, size });
        }
        const port = opts.expert_port + @as(u16, @intCast(rank - 1));
        var host: []const u8 = undefined;
        var node_dev: ?[]const u8 = null;
        var local_dev: []const u8 = "";
        if (rdma)
        {
            var best: u6 = 0;
            for (local_rdma) |l|
            {
                for (l.ips) |lip|
                {
                    for (r.rdma) |nd|
                    {
                        for (nd.ips) |text|
                        {
                            const nip = tcp.parseIp4(text) catch continue;
                            const p = proc.commonPrefix(lip, nip);
                            if (p > best and lip != nip)
                            {
                                best = p;
                                host = text;
                                node_dev = nd.dev;
                                local_dev = l.dev;
                            }
                        }
                    }
                }
            }
            if (best < 24)
            {
                return refuse(why, "No RDMA device here shares a subnet with one on node {s}.", .{name});
            }
            try peers.append(a, try std.fmt.allocPrint(a, "rdma:{s}:{d}@{s}", .{ host, port, local_dev }));
        }
        else
        {
            var ib: [16]u8 = undefined;
            // A node behind NAT or a tunnel says where to reach it.
            host = proc.formatIp4(n.addr, &ib);
            if (r.advertise) |ip|
            {
                if (tcp.parseIp4(ip)) |_|
                {
                    host = ip;
                }
                else |_|
                {
                    return refuse(why, "Node {s} advertises an address that is not IPv4.", .{name});
                }
            }
            host = try a.dupe(u8, host);
            try peers.append(a, try std.fmt.allocPrint(a, "tcp:{s}:{d}", .{ host, port }));
        }
        var s: std.ArrayList(u8) = .empty;
        try s.print(a, "{{\"rank\":{d},\"ranks\":{d},\"model\":", .{ rank, ranks });
        try proc.jsonString(a, &s, req.model);
        try s.print(a, ",\"transport\":\"{s}\",\"port\":{d},\"gpu\":{},\"cache\":{d}", .{ if (rdma) "rdma" else "tcp", port, req.gpu, req.cache });
        if (node_dev) |d|
        {
            try s.print(a, ",\"dev\":\"{s}\"", .{d});
        }
        if (req.max_step) |m|
        {
            try s.print(a, ",\"max_step\":{d}", .{m});
        }
        try s.append(a, '}');
        try starts.append(a, s.items);
        try names.append(a, try a.dupe(u8, name));
    }
    var argv: std.ArrayList([]const u8) = .empty;
    const model_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ opts.models, req.model });
    const mode: []const u8 = if (names.items.len == 0) "local" else if (rdma) "rdma-router" else "router";
    try argv.appendSlice(a, &.{ engine_path, mode, model_path });
    for (peers.items) |p|
    {
        try argv.appendSlice(a, &.{ "--peer", p });
    }
    try argv.appendSlice(a, &.{ "--serve", try std.fmt.allocPrint(a, "127.0.0.1:{d}", .{opts.router_port}), "--ctx", try std.fmt.allocPrint(a, "{d}", .{req.ctx}) });
    // The request's strings go away with it, so the plan keeps its own copies.
    if (names.items.len > 0)
    {
        try argv.appendSlice(a, &.{ "--payload", try a.dupe(u8, req.payload) });
    }
    if (!std.mem.eql(u8, req.kv_format, "f32"))
    {
        try argv.appendSlice(a, &.{ "--kv-format", try a.dupe(u8, req.kv_format) });
    }
    if (req.gpu)
    {
        try argv.append(a, "--gpu");
    }
    if (!req.reuse)
    {
        try argv.append(a, "--no-reuse");
    }
    if (req.max_step) |m|
    {
        try argv.appendSlice(a, &.{ "--max-step", try std.fmt.allocPrint(a, "{d}", .{m}) });
    }
    if (req.cache > 0)
    {
        try argv.appendSlice(a, &.{ "--cache", try std.fmt.allocPrint(a, "{d}", .{req.cache}) });
    }
    var settings: std.ArrayList(u8) = .empty;
    try settings.print(a, "{{\"transport\":\"{s}\",\"gpu\":{},\"ctx\":{d},\"payload\":\"{s}\",\"kv_format\":\"{s}\",\"reuse\":{},\"cache\":{d}}}", .{ if (rdma) "rdma" else "tcp", req.gpu, req.ctx, req.payload, req.kv_format, req.reuse, req.cache });
    plan.model = try a.dupe(u8, req.model);
    plan.nodes = names.items;
    plan.starts = starts.items;
    plan.router_argv = argv.items;
    plan.settings = settings.items;
}

fn oneOf(s: []const u8, set: []const []const u8) bool
{
    for (set) |x|
    {
        if (std.mem.eql(u8, s, x))
        {
            return true;
        }
    }
    return false;
}

fn routerUp() bool
{
    state.lock();
    defer state.unlock();
    return state.run == .running;
}

/// The page's view of everything.
fn stateJson(a: std.mem.Allocator) ![]const u8
{
    var out: std.ArrayList(u8) = .empty;
    var host_buf: [64]u8 = undefined;
    const mem = proc.memory(a);
    try out.appendSlice(a, "{\"local\":{\"name\":");
    try proc.jsonString(a, &out, proc.hostName(&host_buf));
    try out.print(a, ",\"mem_total_mb\":{d},\"mem_available_mb\":{d},\"gpu\":", .{ mem.total_mb, mem.available_mb });
    if (proc.gpuName(a)) |g|
    {
        try proc.jsonString(a, &out, g);
    }
    else
    {
        try out.appendSlice(a, "null");
    }
    try out.appendSlice(a, ",\"models\":[");
    for (try proc.listModels(a, opts_global.models), 0..) |m, i|
    {
        try out.appendSlice(a, if (i == 0) "{\"name\":" else ",{\"name\":");
        try proc.jsonString(a, &out, m.name);
        try out.print(a, ",\"bytes\":{d}}}", .{m.bytes});
    }
    try out.appendSlice(a, "],\"rdma\":[");
    for (try proc.rdmaDevices(a), 0..) |r, i|
    {
        try out.appendSlice(a, if (i == 0) "\"" else ",\"");
        try out.appendSlice(a, r.dev);
        try out.append(a, '"');
    }
    try out.print(a, "]}},\"api_key\":{},\"node_port\":{d},\"nodes\":[", .{ api_key != null, opts_global.node_port });
    const router_log = try std.fmt.allocPrint(a, "{s}/router.log", .{opts_global.log_dir});
    state.lock();
    {
        defer state.unlock();
        const now = proc.monoNs();
        var first = true;
        for (&state.nodes) |*n|
        {
            if (n.name_len == 0)
            {
                continue;
            }
            try out.appendSlice(a, if (first) "{\"name\":" else ",{\"name\":");
            first = false;
            var ib: [16]u8 = undefined;
            try proc.jsonString(a, &out, n.name());
            try out.print(a, ",\"approved\":{},\"online\":{},\"addr\":\"{s}\",\"seen_s\":{d:.1},\"report\":{s}}}", .{
                n.approved,
                n.conn != null,
                proc.formatIp4(n.addr, &ib),
                if (n.seen_ns == 0) -1.0 else @as(f64, @floatFromInt(now - n.seen_ns)) / 1e9,
                if (n.report) |r| r else "null",
            });
        }
        try out.print(a, "],\"run\":{{\"state\":\"{t}\",\"since_s\":{d:.1},\"message\":", .{ state.run, if (state.since_ns == 0) 0 else @as(f64, @floatFromInt(now - state.since_ns)) / 1e9 });
        try proc.jsonString(a, &out, state.message.items);
        const shown = state.pending orelse state.plan;
        if (shown) |p|
        {
            try out.appendSlice(a, ",\"model\":");
            try proc.jsonString(a, &out, p.model);
            try out.appendSlice(a, ",\"nodes\":[");
            for (p.nodes, 0..) |nm, i|
            {
                if (i > 0)
                {
                    try out.append(a, ',');
                }
                try proc.jsonString(a, &out, nm);
            }
            try out.print(a, "],\"settings\":{s},\"command\":", .{p.settings});
            const cmd = try std.mem.join(a, " ", p.router_argv);
            try proc.jsonString(a, &out, cmd);
        }
    }
    try out.appendSlice(a, ",\"router_log\":");
    try proc.jsonString(a, &out, try proc.tail(a, router_log, 8 << 10));
    try out.appendSlice(a, "}}");
    return out.items;
}

/// Runs loads and stops one at a time, and watches the running model.
fn supervise() noreturn
{
    const gpa = gpa_global;
    var router: ?proc.Child = null;
    while (true)
    {
        proc.sleepMs(200);
        state.lock();
        const next = state.pending;
        const stop = state.stop_requested;
        state.pending = null;
        state.stop_requested = false;
        state.unlock();
        if (next != null or stop)
        {
            stopAll(gpa, &router, if (next != null) "replacing the loaded model" else "stopped by the operator");
            if (next) |p|
            {
                state.lock();
                state.plan = p;
                state.run = .starting;
                state.unlock();
                startAll(gpa, p, &router) catch |err| {
                    if (err == error.Cancelled)
                    {
                        stopAll(gpa, &router, "");
                        continue;
                    }
                    state.lock();
                    const msg = gpa.dupe(u8, state.message.items) catch "";
                    state.unlock();
                    stopAll(gpa, &router, "");
                    state.lock();
                    state.run = .failed;
                    state.say("load failed ({t}): {s}", .{ err, msg });
                    state.unlock();
                    gpa.free(msg);
                };
            }
            else
            {
                state.lock();
                state.run = .idle;
                state.say("stopped", .{});
                state.unlock();
            }
            continue;
        }
        state.lock();
        const running = state.run == .running;
        state.unlock();
        if (!running)
        {
            continue;
        }
        if (router) |*r|
        {
            if (!r.alive())
            {
                const code = r.exit.?;
                stopAll(gpa, &router, "");
                state.lock();
                state.run = .failed;
                state.say("the router exited with {d}; see its log", .{code});
                state.unlock();
                continue;
            }
        }
        if (brokenNode()) |why|
        {
            stopAll(gpa, &router, "");
            state.lock();
            state.run = .failed;
            state.say("{s}", .{why});
            state.unlock();
        }
    }
}

var broken_buf: [256]u8 = undefined;

/// Why the running model lost a node, if it did.
fn brokenNode() ?[]const u8
{
    var arena: std.heap.ArenaAllocator = .init(gpa_global);
    defer arena.deinit();
    state.lock();
    defer state.unlock();
    const p = state.plan orelse return null;
    for (p.nodes) |name|
    {
        const n = state.find(name) orelse return "a node of the loaded model was forgotten";
        if (n.conn == null)
        {
            return std.fmt.bufPrint(&broken_buf, "node {s} went offline, so its rank is gone", .{name}) catch "a node went offline";
        }
        const r = parseReport(arena.allocator(), n.report orelse continue) orelse continue;
        if (r.rank == null)
        {
            return std.fmt.bufPrint(&broken_buf, "the rank on node {s} is gone", .{name}) catch "a rank is gone";
        }
        if (r.rank.?.exit) |code|
        {
            return std.fmt.bufPrint(&broken_buf, "the rank on node {s} exited with {d}; its log is rank{d}.log in that agent's log directory", .{ name, code, r.rank.?.rank }) catch "a rank exited";
        }
    }
    return null;
}

/// Stops the router and every node rank of the current plan.
fn stopAll(gpa: std.mem.Allocator, router: *?proc.Child, why: []const u8) void
{
    state.lock();
    const p = state.plan;
    if (p != null or router.* != null)
    {
        state.run = .stopping;
        if (why.len > 0)
        {
            state.say("stopping: {s}", .{why});
        }
    }
    state.unlock();
    if (router.*) |*r|
    {
        r.stop(10_000);
        r.deinit(gpa);
        router.* = null;
    }
    state.lock();
    state.router_pid = null;
    if (p) |plan|
    {
        for (plan.nodes) |name|
        {
            if (state.find(name)) |n|
            {
                if (n.conn) |cn|
                {
                    cn.send(.stop, "") catch {};
                }
            }
        }
    }
    state.unlock();
    // Waits for the agents to report their ranks gone, so ports are free for the next load.
    var waited: u32 = 0;
    while (p != null and waited < 15_000) : (waited += 200)
    {
        if (!anyRank(p.?))
        {
            break;
        }
        proc.sleepMs(200);
    }
    state.lock();
    if (state.plan == p)
    {
        state.plan = null;
    }
    state.unlock();
    if (p) |plan|
    {
        plan.arena.deinit();
        gpa.destroy(plan);
    }
}

fn anyRank(p: *Plan) bool
{
    var arena: std.heap.ArenaAllocator = .init(gpa_global);
    defer arena.deinit();
    state.lock();
    defer state.unlock();
    for (p.nodes) |name|
    {
        const n = state.find(name) orelse continue;
        if (n.conn == null)
        {
            continue;
        }
        const r = parseReport(arena.allocator(), n.report orelse continue) orelse continue;
        if (r.rank != null and r.rank.?.exit == null)
        {
            return true;
        }
    }
    return false;
}

/// Starts each node's rank, waits for it to listen, then starts the router and
/// waits for its API to answer.
fn startAll(gpa: std.mem.Allocator, p: *Plan, router: *?proc.Child) !void
{
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    for (p.nodes, p.starts, 1..) |name, body, rank|
    {
        state.lock();
        state.say("starting rank {d} on node {s}", .{ rank, name });
        const n = state.find(name);
        const ok = if (n != null and n.?.conn != null) blk: {
            n.?.conn.?.send(.start, body) catch break :blk false;
            break :blk true;
        } else false;
        state.unlock();
        if (!ok)
        {
            return error.NodeOffline;
        }
    }
    // Each rank listens before it loads its share, so this is quick.
    var waited: u32 = 0;
    while (true) : (waited += 200)
    {
        proc.sleepMs(200);
        _ = arena.reset(.retain_capacity);
        var ready: usize = 0;
        state.lock();
        for (p.nodes, 1..) |name, rank|
        {
            const n = state.find(name) orelse {
                state.unlock();
                return error.NodeForgotten;
            };
            if (n.conn == null)
            {
                state.say("node {s} went offline while starting rank {d}", .{ name, rank });
                state.unlock();
                return error.NodeOffline;
            }
            const r = parseReport(arena.allocator(), n.report orelse "{}") orelse continue;
            const rk = r.rank orelse continue;
            if (rk.rank != rank)
            {
                continue;
            }
            if (rk.exit) |code|
            {
                state.say("the rank on node {s} exited with {d} while starting; see its log", .{ name, code });
                state.unlock();
                return error.RankExited;
            }
            if (std.mem.eql(u8, rk.state, "listening"))
            {
                ready += 1;
            }
        }
        const pending = state.pending != null or state.stop_requested;
        state.unlock();
        if (pending)
        {
            return error.Cancelled;
        }
        if (ready == p.nodes.len)
        {
            break;
        }
        if (waited > 60_000)
        {
            state.lock();
            state.say("the node ranks did not start listening within 60 s", .{});
            state.unlock();
            return error.Timeout;
        }
    }
    const log_path = try std.fmt.allocPrint(arena.allocator(), "{s}/router.log", .{opts_global.log_dir});
    router.* = try proc.Child.spawn(gpa, p.router_argv, log_path);
    state.lock();
    state.router_pid = router.*.?.pid;
    state.say("the router (pid {d}) is loading {s}; its log is below", .{ router.*.?.pid, p.model });
    state.unlock();
    while (true)
    {
        proc.sleepMs(500);
        if (!router.*.?.alive())
        {
            state.lock();
            state.say("the router exited with {d} while loading; see its log", .{router.*.?.exit.?});
            state.unlock();
            return error.RouterExited;
        }
        state.lock();
        const pending = state.pending != null or state.stop_requested;
        state.unlock();
        if (pending)
        {
            return error.Cancelled;
        }
        if (brokenNode()) |why|
        {
            state.lock();
            state.say("{s}", .{why});
            state.unlock();
            return error.NodeLost;
        }
        if (routerHealthy())
        {
            break;
        }
    }
    state.lock();
    state.run = .running;
    state.since_ns = proc.monoNs();
    state.say("{s} is loaded on {d} process{s}", .{ p.model, p.nodes.len + 1, if (p.nodes.len == 0) "" else "es" });
    state.unlock();
}

/// True once the router's server answers /health.
fn routerHealthy() bool
{
    const fd = tcp.connect("127.0.0.1", opts_global.router_port, 1) catch return false;
    defer tcp.closeSocket(fd);
    sendAll(fd, "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n") catch return false;
    var buf: [64]u8 = undefined;
    if (!(tcp.waitReadable(fd, 2000) catch false))
    {
        return false;
    }
    const n = sock.recv(fd, &buf);
    return n >= 12 and std.mem.startsWith(u8, buf[0..@intCast(n)], "HTTP/1.1 200");
}

/// Passes the request to the router and its answer back, both ways until
/// either side closes (so a WebSocket works too). `path` replaces the request's path.
fn proxy(a: std.mem.Allocator, fd: sock.Fd, raw: []const u8, path: ?[]const u8) !void
{
    const up = tcp.connect("127.0.0.1", opts_global.router_port, 1) catch {
        return apiError(fd, 502, "The router did not answer.");
    };
    defer tcp.closeSocket(up);
    if (path) |p|
    {
        const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return;
        const first = raw[0..line_end];
        const sp1 = std.mem.indexOfScalar(u8, first, ' ') orelse return;
        const sp2 = std.mem.lastIndexOfScalar(u8, first, ' ') orelse return;
        const rewritten = try std.fmt.allocPrint(a, "{s} {s}{s}{s}", .{ first[0..sp1], p, first[sp2..], raw[line_end..] });
        try sendAll(up, rewritten);
    }
    else
    {
        try sendAll(up, raw);
    }
    // Reads may now wait as long as the stream lasts.
    sock.setTimeouts(fd, 0, 30);
    var buf: [64 << 10]u8 = undefined;
    var pfds: [2]sock.PollFd = .{
        .{ .fd = fd, .events = sock.POLL_IN, .revents = 0 },
        .{ .fd = up, .events = sock.POLL_IN, .revents = 0 },
    };
    while (true)
    {
        if (sock.poll(&pfds, -1) == 0)
        {
            continue;
        }
        for (pfds, 0..) |p, i|
        {
            if (p.revents == 0)
            {
                continue;
            }
            const from = if (i == 0) fd else up;
            const to = if (i == 0) up else fd;
            const n = sock.recv(from, &buf);
            if (n <= 0)
            {
                return;
            }
            sendAll(to, buf[0..@intCast(n)]) catch return;
        }
    }
}

fn sendAll(fd: sock.Fd, bytes: []const u8) !void
{
    return sock.sendAll(fd, bytes);
}

/// A whole response; `extra` is more header lines, each ending in CRLF.
fn sendSimple(fd: sock.Fd, status: u16, content_type: []const u8, body: []const u8, extra: []const u8) !void
{
    var head: [1024]u8 = undefined;
    const h = try std.fmt.bufPrint(&head, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n{s}Connection: close\r\n\r\n", .{ status, http.reason(status), content_type, body.len, extra });
    try sendAll(fd, h);
    try sendAll(fd, body);
}

fn apiError(fd: sock.Fd, status: u16, message: []const u8) !void
{
    const gpa = gpa_global;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    try body.appendSlice(gpa, "{\"error\":");
    try openai.writeString(gpa, &body, message);
    try body.append(gpa, '}');
    try sendSimple(fd, status, "application/json", body.items, "");
}

fn sendOpenAiError(fd: sock.Fd, status: u16, message: []const u8, kind: []const u8) !void
{
    const gpa = gpa_global;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    try openai.writeError(gpa, &body, message, kind, null);
    try sendSimple(fd, status, "application/json", body.items, "");
}
