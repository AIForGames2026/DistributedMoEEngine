//! First-run setup behind `tfswarm setup` (TASK-44). Linux and Windows.
//!
//!   tfswarm setup [--settings FILE] [--port P]
//!
//! Serves a page on 127.0.0.1 that asks whether this machine hosts the
//! operator dashboard or joins one as an expert node, then writes the settings
//! file (and, for a node, the pasted node secret beside it) and exits. It prints
//! a one-time link to open; the Windows tray launcher opens it and then starts
//! the dashboard or agent. Every request but that link needs the cookie it sets.
const std = @import("std");
const http = @import("http.zig");
const sock = @import("sock.zig");
const plat = @import("plat.zig");
const auth = @import("auth.zig");
const proc = @import("proc.zig");
const tcp = @import("transport/tcp.zig");
const settings_file = @import("settings.zig");
const node_agent = @import("node_agent.zig");

const page_html = @embedFile("setup.html");
const page_js = @embedFile("setup.js");
const cookie_name = "tf_setup";
const max_head = 16 << 10;
const max_body = 64 << 10;
const page_headers = "Content-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'\r\nX-Frame-Options: DENY\r\n";

const usage =
    \\usage: tfswarm setup [--settings FILE] [--port P]
    \\
;

var code: [32]u8 = undefined;
var settings_path: []const u8 = "";
var done = false;

pub fn main(gpa: std.mem.Allocator, args: []const []const u8) !void
{
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var port: u16 = 0;
    var path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 2)
    {
        if (i + 1 >= args.len)
        {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
        if (std.mem.eql(u8, args[i], "--settings"))
        {
            path = args[i + 1];
        }
        else if (std.mem.eql(u8, args[i], "--port"))
        {
            port = try std.fmt.parseInt(u16, args[i + 1], 10);
        }
        else
        {
            std.debug.print("setup: unknown option {s}\n" ++ usage, .{args[i]});
            std.process.exit(2);
        }
    }
    settings_path = path orelse try settings_file.defaultPath(a);
    var raw: [16]u8 = undefined;
    auth.randomBytes(&raw);
    code = std.fmt.bytesToHex(raw, .lower);
    const lfd = sock.listenTcp(@bitCast([4]u8{ 127, 0, 0, 1 }), port, 16) catch |err| {
        std.debug.print("setup: cannot listen on port {d}: {t}\n", .{ port, err });
        std.process.exit(2);
    };
    const bound = sock.localPort(lfd) orelse port;
    std.debug.print("setup: settings go to {s}\n", .{settings_path});
    std.debug.print("setup: open http://127.0.0.1:{d}/setup/{s}\n", .{ bound, code });
    while (!done)
    {
        const fd = sock.accept(lfd) orelse continue;
        sock.setTimeouts(fd, 10, 30);
        var req_arena: std.heap.ArenaAllocator = .init(gpa);
        handle(req_arena.allocator(), fd) catch {};
        req_arena.deinit();
        sock.close(fd);
    }
    std.debug.print("setup: saved; exiting\n", .{});
}

fn handle(a: std.mem.Allocator, fd: sock.Fd) !void
{
    var buf: std.ArrayList(u8) = .empty;
    var head: http.Head = undefined;
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
        }
        else |err| switch (err)
        {
            error.Incomplete => {},
            else => return apiError(fd, 400, "Malformed HTTP request."),
        }
        try buf.ensureUnusedCapacity(a, 16 << 10);
        const n = sock.recv(fd, buf.unusedCapacitySlice());
        if (n <= 0)
        {
            return;
        }
        buf.items.len += @intCast(n);
    }
    const body = buf.items[head.len .. head.len + head.content_length];
    const path = head.path;
    const get = std.mem.eql(u8, head.method, "GET");
    if (std.mem.startsWith(u8, path, "/setup/") and get)
    {
        if (!auth.secretEql(path["/setup/".len..], &code))
        {
            return sendSimple(fd, 403, "text/plain; charset=utf-8", "This setup link is not current; open setup again from the tray icon.", "");
        }
        const extra = try std.fmt.allocPrint(a, "Set-Cookie: {s}={s}; HttpOnly; SameSite=Strict; Path=/\r\nLocation: /\r\n", .{ cookie_name, code });
        return sendSimple(fd, 303, "text/plain", "", extra);
    }
    if (!hasCookie(head))
    {
        return sendSimple(fd, 403, "text/plain; charset=utf-8", "Open setup from the link it printed, or from the tray icon.", "");
    }
    if (std.mem.eql(u8, path, "/") and get)
    {
        return sendSimple(fd, 200, "text/html; charset=utf-8", page_html, page_headers);
    }
    if (std.mem.eql(u8, path, "/setup.js") and get)
    {
        return sendSimple(fd, 200, "text/javascript; charset=utf-8", page_js, "");
    }
    if (std.mem.eql(u8, path, "/api/setup") and get)
    {
        return sendSimple(fd, 200, "application/json", try currentJson(a), "");
    }
    if (!std.mem.eql(u8, head.method, "POST"))
    {
        return apiError(fd, 405, "Use POST.");
    }
    if (!http.sameOrigin(head))
    {
        return apiError(fd, 403, "Requests must come from the setup page.");
    }
    if (std.mem.eql(u8, path, "/api/models"))
    {
        return checkModels(a, fd, body);
    }
    if (std.mem.eql(u8, path, "/api/test-node"))
    {
        return testNode(a, fd, body);
    }
    if (std.mem.eql(u8, path, "/api/save"))
    {
        return save(a, fd, body);
    }
    return apiError(fd, 404, "Not found.");
}

fn hasCookie(head: http.Head) bool
{
    const ck = head.cookie orelse return false;
    const v = auth.cookieValue(ck, cookie_name) orelse return false;
    return auth.secretEql(v, &code);
}

/// The saved settings (or the defaults), and whether a node secret is already in place.
fn currentJson(a: std.mem.Allocator) ![]const u8
{
    const s = settings_file.load(a, settings_path) orelse settings_file.Settings{};
    const secret_path = try settings_file.beside(a, settings_path, "node.secret");
    var host_buf: [64]u8 = undefined;
    var out: std.ArrayList(u8) = .empty;
    try out.print(a, "{{\"settings\":{f},\"file\":", .{std.json.fmt(s, .{})});
    try proc.jsonString(a, &out, settings_path);
    try out.appendSlice(a, ",\"host\":");
    try proc.jsonString(a, &out, proc.hostName(&host_buf));
    try out.print(a, ",\"has_node_secret\":{},\"windows\":{}}}", .{ proc.readSmall(a, secret_path) != null, @import("builtin").os.tag == .windows });
    return out.items;
}

/// What a models folder holds: the loadable models, and the .gguf files without a .tfsum index.
fn checkModels(a: std.mem.Allocator, fd: sock.Fd, body: []const u8) !void
{
    const req = std.json.parseFromSliceLeaky(struct { dir: []const u8 }, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "Send {\"dir\": \"...\"}.");
    };
    var out: std.ArrayList(u8) = .empty;
    var buf: [4096]u8 = undefined;
    const dir = std.mem.trim(u8, req.dir, " \t\"");
    if (dir.len == 0 or plat.realPath(try a.dupeSentinel(u8, dir, 0), &buf) == null)
    {
        return sendSimple(fd, 200, "application/json", "{\"exists\":false,\"models\":[],\"unindexed\":0}", "");
    }
    const models = proc.listModels(a, dir) catch &.{};
    try out.appendSlice(a, "{\"exists\":true,\"models\":[");
    for (models, 0..) |m, i|
    {
        try out.appendSlice(a, if (i == 0) "{\"name\":" else ",{\"name\":");
        try proc.jsonString(a, &out, m.name);
        try out.print(a, ",\"bytes\":{d}}}", .{m.bytes});
    }
    try out.print(a, "],\"unindexed\":{d}}}", .{unindexed(a, dir)});
    return sendSimple(fd, 200, "application/json", out.items, "");
}

fn unindexed(a: std.mem.Allocator, dir: []const u8) usize
{
    const names = proc.fileNames(a, dir) catch return 0;
    var n: usize = 0;
    for (names) |name|
    {
        if (std.mem.endsWith(u8, name, ".gguf") and proc.modelSize(a, dir, name) == null)
        {
            n += 1;
        }
    }
    return n;
}

const NodeTest = struct {
    dashboard: []const u8,
    secret: []const u8 = "",
    name: []const u8 = "",
};

/// Links to the dashboard once with the pasted secret, so mistakes show before saving.
fn testNode(a: std.mem.Allocator, fd: sock.Fd, body: []const u8) !void
{
    const req = std.json.parseFromSliceLeaky(NodeTest, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "The test request is malformed.");
    };
    const hp = settings_file.hostPort(req.dashboard, node_agent.default_port) orelse return testResult(fd, false, "The dashboard address is IP or IP:PORT.");
    _ = tcp.parseIp4(hp.host) catch return testResult(fd, false, "The dashboard address must be an IPv4 address, such as 192.168.1.20.");
    var secret = std.mem.trim(u8, req.secret, " \t\r\n");
    if (secret.len == 0)
    {
        const secret_path = try settings_file.beside(a, settings_path, "node.secret");
        const saved = proc.readSmall(a, secret_path) orelse return testResult(fd, false, "Paste the node secret from the dashboard's Settings panel.");
        secret = std.mem.trim(u8, saved, " \t\r\n");
    }
    var host_buf: [64]u8 = undefined;
    const name = if (req.name.len > 0) req.name else proc.hostName(&host_buf);
    if (!auth.validName(name))
    {
        return testResult(fd, false, "The node name may hold letters, digits, '.', '-' and '_'.");
    }
    const conn = tcp.connect(hp.host, hp.port, 1) catch return testResult(fd, false, "Nothing answered at that address and port. Is the dashboard running, and does its firewall allow the node port?");
    defer tcp.closeSocket(conn);
    var link = try auth.Link.init(a, conn, true);
    defer link.deinit(a);
    auth.handshakeNode(&link, secret, name) catch |err| {
        return testResult(fd, false, switch (err)
        {
            error.DashboardNotTrusted => "The dashboard answered, but with a different node secret. Copy it again from the dashboard's Settings panel.",
            else => "The dashboard answered but refused the link; check the node secret.",
        });
    };
    return testResult(fd, true, "The dashboard accepted the secret. Once saved, this PC shows on the dashboard as waiting for approval.");
}

fn testResult(fd: sock.Fd, ok: bool, message: []const u8) !void
{
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{{\"ok\":{},\"message\":", .{ok});
    try std.json.Stringify.value(message, .{}, &w);
    try w.writeByte('}');
    return sendSimple(fd, 200, "application/json", w.buffered(), "");
}

const SaveRequest = struct {
    role: []const u8,
    dashboard: settings_file.Dashboard = .{},
    node: settings_file.Node = .{},
    /// A pasted node secret; empty keeps the one already saved.
    node_secret: []const u8 = "",
};

fn save(a: std.mem.Allocator, fd: sock.Fd, body: []const u8) !void
{
    const req = std.json.parseFromSliceLeaky(SaveRequest, a, body, .{ .ignore_unknown_fields = true }) catch {
        return apiError(fd, 400, "The settings are malformed.");
    };
    const host_role = std.mem.eql(u8, req.role, "dashboard");
    if (!host_role and !std.mem.eql(u8, req.role, "node"))
    {
        return apiError(fd, 400, "Choose whether this machine hosts the dashboard or joins one.");
    }
    var s = settings_file.load(a, settings_path) orelse settings_file.Settings{};
    s.role = if (host_role) "dashboard" else "node";
    const models = if (host_role) req.dashboard.models else req.node.models;
    var buf: [4096]u8 = undefined;
    if (std.mem.trim(u8, models, " \t").len == 0 or plat.realPath(try a.dupeSentinel(u8, std.mem.trim(u8, models, " \t\""), 0), &buf) == null)
    {
        return apiError(fd, 400, "That models folder does not exist.");
    }
    if (host_role)
    {
        s.dashboard = req.dashboard;
        s.dashboard.models = std.mem.trim(u8, models, " \t\"");
    }
    else
    {
        var n = req.node;
        n.models = std.mem.trim(u8, models, " \t\"");
        const hp = settings_file.hostPort(n.dashboard, node_agent.default_port) orelse return apiError(fd, 400, "The dashboard address is IP or IP:PORT.");
        _ = tcp.parseIp4(hp.host) catch return apiError(fd, 400, "The dashboard address must be an IPv4 address.");
        if (n.name.len > 0 and !auth.validName(n.name))
        {
            return apiError(fd, 400, "The node name may hold letters, digits, '.', '-' and '_'.");
        }
        if (n.advertise.len > 0)
        {
            _ = tcp.parseIp4(n.advertise) catch return apiError(fd, 400, "The advertised address must be an IPv4 address.");
        }
        const secret_path = try settings_file.beside(a, settings_path, "node.secret");
        if (req.node_secret.len > 0)
        {
            auth.saveSecret(a, secret_path, req.node_secret) catch |err| {
                return apiError(fd, 400, switch (err)
                {
                    error.SecretTooShort => "The node secret is at least 32 characters; copy all of it.",
                    error.SecretMalformed => "The node secret holds only letters and digits; copy it again.",
                    error.SecretExposed => "The settings folder is outside your user profile, so the secret cannot be kept safe there.",
                    else => "Could not write the node secret.",
                });
            };
        }
        else if (proc.readSmall(a, secret_path) == null)
        {
            return apiError(fd, 400, "Paste the node secret from the dashboard's Settings panel.");
        }
        s.node = n;
    }
    settings_file.save(a, settings_path, s) catch {
        return apiError(fd, 500, "Could not write the settings file.");
    };
    std.debug.print("setup: this machine {s}; settings saved to {s}\n", .{ if (host_role) "hosts the dashboard" else "joins a dashboard as a node", settings_path });
    done = true;
    return sendSimple(fd, 200, "application/json", "{\"ok\":true}", "");
}

fn sendSimple(fd: sock.Fd, status: u16, content_type: []const u8, body: []const u8, extra: []const u8) !void
{
    var head: [1024]u8 = undefined;
    const h = try std.fmt.bufPrint(&head, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n{s}Connection: close\r\n\r\n", .{ status, http.reason(status), content_type, body.len, extra });
    try sock.sendAll(fd, h);
    try sock.sendAll(fd, body);
}

fn apiError(fd: sock.Fd, status: u16, message: []const u8) !void
{
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.writeAll("{\"error\":");
    try std.json.Stringify.value(message, .{}, &w);
    try w.writeByte('}');
    return sendSimple(fd, status, "application/json", w.buffered(), "");
}
