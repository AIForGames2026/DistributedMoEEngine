//! Authentication for the operator dashboard (TASK-25). Linux and Windows (TASK-43).
//!
//! Operators: one admin token, kept in a file only its owner can read. A login
//! with it opens a session, held in an HttpOnly, SameSite=Strict cookie; scripts
//! may send the token itself as a Bearer header instead.
//!
//! Node agents: a shared secret, also in a file. An agent and the dashboard
//! prove to each other that they hold it (HMAC-SHA256 over fresh nonces from
//! both sides), derive a session key, and then sign every frame with it and a
//! sequence number, so frames cannot be forged, changed or replayed. Frames are
//! not encrypted; the dashboard is meant for a private network.
const std = @import("std");
const c = std.c;
const proc = @import("proc.zig");
const plat = @import("plat.zig");
const sock = @import("sock.zig");
const win32 = @import("win32.zig");
const is_windows = @import("builtin").os.tag == .windows;

const Hmac = std.crypto.auth.hmac.sha2.HmacSha256;
pub const mac_len = Hmac.mac_length;
pub const nonce_len = 32;
/// Session ids and generated tokens: 32 random bytes as 64 hex digits.
pub const id_len = 64;
pub const session_ttl_ns: u64 = 12 * 3600 * std.time.ns_per_s;
pub const max_sessions = 64;

pub fn randomBytes(buf: []u8) void
{
    plat.randomBytes(buf);
}

pub fn monoNs() u64
{
    return plat.monoNs();
}

/// Compares two secrets in time that depends on neither their contents nor their lengths.
pub fn secretEql(a: []const u8, b: []const u8) bool
{
    var key: [32]u8 = undefined;
    randomBytes(&key);
    var ma: [mac_len]u8 = undefined;
    var mb: [mac_len]u8 = undefined;
    Hmac.create(&ma, a, &key);
    Hmac.create(&mb, b, &key);
    return std.crypto.timing_safe.eql([mac_len]u8, ma, mb);
}

/// Reads a secret from `path`, creating it with a fresh random value (mode 0600)
/// when it does not exist. Refuses a file others can read, or a short secret.
/// On Windows the file must be inside the user's profile folder instead.
pub fn loadSecret(gpa: std.mem.Allocator, path: []const u8, label: []const u8) ![]u8
{
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    if (is_windows)
    {
        return loadSecretWindows(gpa, z, label);
    }
    const made = c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    if (made >= 0)
    {
        var raw: [32]u8 = undefined;
        randomBytes(&raw);
        const text = std.fmt.bytesToHex(raw, .lower) ++ [_]u8{'\n'};
        const ok = c.write(made, &text, text.len) == text.len;
        _ = c.close(made);
        if (!ok)
        {
            return error.WriteFailed;
        }
        say("{s}: created a new {s} in {s} (readable by its owner only)\n", .{ label, label, path });
    }
    const fd = c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0)
    {
        std.debug.print("{s}: cannot read {s}\n", .{ label, path });
        return error.SecretUnreadable;
    }
    defer _ = c.close(fd);
    const st = proc.statAt(fd, "") orelse return error.SecretUnreadable;
    if (st.mode & 0o077 != 0)
    {
        say("{s}: {s} can be read by other users; run chmod 600 on it\n", .{ label, path });
        return error.SecretExposed;
    }
    var buf: [512]u8 = undefined;
    const n = c.read(fd, &buf, buf.len);
    if (n < 0)
    {
        return error.SecretUnreadable;
    }
    const text = std.mem.trim(u8, buf[0..@intCast(n)], " \t\r\n");
    if (text.len < 32)
    {
        std.debug.print("{s}: the secret in {s} is shorter than 32 characters\n", .{ label, path });
        return error.SecretTooShort;
    }
    return gpa.dupe(u8, text);
}

/// Writes a secret someone pasted (such as the node secret, TASK-44), owner-only on
/// Linux and only inside the profile folder on Windows; `loadSecret` then reads it.
pub fn saveSecret(gpa: std.mem.Allocator, path: []const u8, secret: []const u8) error{ SecretTooShort, SecretMalformed, SecretExposed, WriteFailed, OutOfMemory }!void
{
    const text = std.mem.trim(u8, secret, " \t\r\n");
    if (text.len < 32 or text.len > 256)
    {
        return error.SecretTooShort;
    }
    for (text) |ch|
    {
        if (ch < 0x21 or ch > 0x7e)
        {
            return error.SecretMalformed;
        }
    }
    const dir = try gpa.dupeSentinel(u8, std.fs.path.dirname(path) orelse ".", 0);
    defer gpa.free(dir);
    if (is_windows and !plat.inUserProfile(dir))
    {
        return error.SecretExposed;
    }
    const tmp = try std.fmt.allocPrintSentinel(gpa, "{s}.tmp", .{path}, 0);
    defer gpa.free(tmp);
    const dst = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(dst);
    const line = try std.fmt.allocPrint(gpa, "{s}\n", .{text});
    defer gpa.free(line);
    if (!plat.replaceFile(tmp, dst, line))
    {
        return error.WriteFailed;
    }
}

fn loadSecretWindows(gpa: std.mem.Allocator, path: [:0]const u8, label: []const u8) ![]u8
{
    // The folder is checked first, so no secret is ever written outside the profile.
    const dir = try gpa.dupeSentinel(u8, std.fs.path.dirname(path) orelse ".", 0);
    defer gpa.free(dir);
    if (!plat.inUserProfile(dir))
    {
        say("{s}: {s} must be inside your user profile folder ({s}), which only you, SYSTEM and Administrators can read\n", .{ label, path, plat.env("USERPROFILE") orelse "%USERPROFILE%" });
        return error.SecretExposed;
    }
    if (win32.openFile(path, win32.GENERIC_WRITE, win32.CREATE_NEW)) |made|
    {
        var raw: [32]u8 = undefined;
        randomBytes(&raw);
        const text = std.fmt.bytesToHex(raw, .lower) ++ [_]u8{'\n'};
        var n: win32.DWORD = 0;
        const ok = win32.WriteFile(made, &text, text.len, &n, null) != 0 and n == text.len;
        _ = win32.CloseHandle(made);
        if (!ok)
        {
            return error.WriteFailed;
        }
        say("{s}: created a new {s} in {s}\n", .{ label, label, path });
    }
    const text = proc.readSmall(gpa, path) orelse {
        std.debug.print("{s}: cannot read {s}\n", .{ label, path });
        return error.SecretUnreadable;
    };
    defer gpa.free(text);
    const secret = std.mem.trim(u8, text[0..@min(text.len, 512)], " \t\r\n");
    if (secret.len < 32)
    {
        std.debug.print("{s}: the secret in {s} is shorter than 32 characters\n", .{ label, path });
        return error.SecretTooShort;
    }
    return gpa.dupe(u8, secret);
}

/// Prints outside tests, where the messages would only be noise.
fn say(comptime fmt: []const u8, args: anytype) void
{
    if (!@import("builtin").is_test)
    {
        std.debug.print(fmt, args);
    }
}

/// Operator sessions, each a random id with an expiry. Thread-safe.
pub const Sessions = struct {
    mutex: plat.Mutex = .{},
    ids: [max_sessions][32]u8 = undefined,
    expires: [max_sessions]u64 = @splat(0),
    ttl_ns: u64 = session_ttl_ns,

    /// Opens a session and writes its id as hex; the oldest goes when all are in use.
    pub fn create(self: *Sessions, now: u64, out: *[id_len]u8) void
    {
        self.mutex.acquire();
        defer self.mutex.release();
        var slot: usize = 0;
        for (self.expires, 0..) |e, i|
        {
            if (e < self.expires[slot])
            {
                slot = i;
            }
        }
        randomBytes(&self.ids[slot]);
        self.expires[slot] = now + self.ttl_ns;
        out.* = std.fmt.bytesToHex(self.ids[slot], .lower);
    }

    /// True if `hex` names a live session. Compares every slot, so timing reveals nothing.
    pub fn valid(self: *Sessions, hex: []const u8, now: u64) bool
    {
        return self.find(hex, now, false);
    }

    pub fn revoke(self: *Sessions, hex: []const u8, now: u64) void
    {
        _ = self.find(hex, now, true);
    }

    fn find(self: *Sessions, hex: []const u8, now: u64, drop: bool) bool
    {
        var raw: [32]u8 = undefined;
        if (hex.len != id_len)
        {
            return false;
        }
        _ = std.fmt.hexToBytes(&raw, hex) catch return false;
        self.mutex.acquire();
        defer self.mutex.release();
        var found = false;
        for (&self.ids, &self.expires) |*id, *e|
        {
            const same = std.crypto.timing_safe.eql([32]u8, id.*, raw);
            if (same and e.* > now)
            {
                found = true;
                if (drop)
                {
                    e.* = 0;
                }
            }
        }
        return found;
    }
};

/// The value of cookie `name` in a Cookie header.
pub fn cookieValue(header: []const u8, name: []const u8) ?[]const u8
{
    var it = std.mem.splitScalar(u8, header, ';');
    while (it.next()) |part|
    {
        const p = std.mem.trim(u8, part, " \t");
        if (p.len > name.len and std.mem.startsWith(u8, p, name) and p[name.len] == '=')
        {
            return p[name.len + 1 ..];
        }
    }
    return null;
}

/// The token in an `Authorization: Bearer ...` header.
pub fn bearer(header: []const u8) ?[]const u8
{
    const prefix = "Bearer ";
    if (header.len > prefix.len and std.ascii.eqlIgnoreCase(header[0..prefix.len], prefix))
    {
        return std.mem.trim(u8, header[prefix.len..], " \t");
    }
    return null;
}

/// What both sides compute from the secret and the two nonces.
pub const Proofs = struct {
    /// Sent by the dashboard, checked by the agent.
    dashboard: [mac_len]u8,
    /// Sent by the agent, checked by the dashboard.
    node: [mac_len]u8,
    key: [mac_len]u8,

    pub fn compute(secret: []const u8, node_nonce: *const [nonce_len]u8, dash_nonce: *const [nonce_len]u8) Proofs
    {
        var p: Proofs = undefined;
        macParts(&p.dashboard, secret, &.{ "tf-dashboard", node_nonce, dash_nonce });
        macParts(&p.node, secret, &.{ "tf-node", dash_nonce, node_nonce });
        macParts(&p.key, secret, &.{ "tf-key", node_nonce, dash_nonce });
        return p;
    }
};

fn macParts(out: *[mac_len]u8, key: []const u8, parts: []const []const u8) void
{
    var h = Hmac.init(key);
    for (parts) |p|
    {
        h.update(p);
    }
    h.final(out);
}

pub const Kind = enum(u8) {
    /// Agent to dashboard: its nonce and name.
    hello = 1,
    /// Dashboard to agent: its nonce and proof.
    challenge = 2,
    /// Agent to dashboard: its proof.
    proof = 3,
    /// Dashboard to agent: the agent is accepted.
    welcome = 4,
    /// Agent to dashboard: what the node has and runs (JSON), also its heartbeat.
    report = 5,
    /// Dashboard to agent: start an expert rank (JSON).
    start = 6,
    /// Dashboard to agent: stop its rank.
    stop = 7,
    /// Either way: why the link is being closed (text).
    refuse = 8,
    _,
};

const link_magic: u32 = 0x4E4B4654; // "TFKN"
pub const max_body = 256 << 10;

pub const FrameHead = extern struct {
    magic: u32 = link_magic,
    kind: Kind,
    /// Which side sent it, so a frame cannot be reflected back to its sender.
    from_node: u8,
    _pad: u16 = 0,
    len: u32,
    seq: u64,
};

/// One end of a node link over a blocking socket. Frames before the handshake
/// completes carry no signature; after it, every frame is signed.
pub const Link = struct {
    fd: sock.Fd,
    is_node: bool,
    key: ?[mac_len]u8 = null,
    send_seq: u64 = 0,
    recv_seq: u64 = 0,
    buf: []u8,
    len: usize = 0,
    /// Bytes of the frame the last `recv` returned, dropped on the next call.
    returned: usize = 0,

    pub fn init(gpa: std.mem.Allocator, fd: sock.Fd, is_node: bool) !Link
    {
        return .{ .fd = fd, .is_node = is_node, .buf = try gpa.alloc(u8, @sizeOf(FrameHead) + max_body + mac_len) };
    }

    pub fn deinit(self: *Link, gpa: std.mem.Allocator) void
    {
        gpa.free(self.buf);
    }

    pub fn send(self: *Link, kind: Kind, body: []const u8) !void
    {
        if (body.len > max_body)
        {
            return error.FrameTooLarge;
        }
        self.send_seq += 1;
        const head: FrameHead = .{ .kind = kind, .from_node = @intFromBool(self.is_node), .len = @intCast(body.len), .seq = self.send_seq };
        var mac: [mac_len]u8 = @splat(0);
        if (self.key) |k|
        {
            macParts(&mac, &k, &.{ std.mem.asBytes(&head), body });
        }
        try sendAll(self.fd, std.mem.asBytes(&head));
        try sendAll(self.fd, body);
        try sendAll(self.fd, &mac);
    }

    /// The next frame, waiting up to `timeout_ms` (null on timeout). The body
    /// stays valid until the next call. Errors on a closed link or a frame that
    /// is malformed, unsigned when it must be signed, forged, or out of order.
    pub fn recv(self: *Link, timeout_ms: c_int) !?struct { kind: Kind, body: []const u8 }
    {
        if (self.returned > 0)
        {
            std.mem.copyForwards(u8, self.buf[0 .. self.len - self.returned], self.buf[self.returned..self.len]);
            self.len -= self.returned;
            self.returned = 0;
        }
        while (true)
        {
            if (self.len >= @sizeOf(FrameHead))
            {
                const head = std.mem.bytesToValue(FrameHead, self.buf[0..@sizeOf(FrameHead)]);
                if (head.magic != link_magic or head.len > max_body or head.from_node == @intFromBool(self.is_node))
                {
                    return error.BadFrame;
                }
                const total = @sizeOf(FrameHead) + head.len + mac_len;
                if (self.len >= total)
                {
                    const body = self.buf[@sizeOf(FrameHead)..][0..head.len];
                    const mac = self.buf[@sizeOf(FrameHead) + head.len ..][0..mac_len];
                    if (head.seq != self.recv_seq + 1)
                    {
                        return error.BadSequence;
                    }
                    if (self.key) |k|
                    {
                        var want: [mac_len]u8 = undefined;
                        macParts(&want, &k, &.{ self.buf[0..@sizeOf(FrameHead)], body });
                        if (!std.crypto.timing_safe.eql([mac_len]u8, want, mac.*))
                        {
                            return error.BadSignature;
                        }
                    }
                    self.recv_seq = head.seq;
                    self.returned = total;
                    return .{ .kind = head.kind, .body = body };
                }
            }
            const ready = sock.waitReadable(self.fd, timeout_ms) orelse continue;
            if (!ready)
            {
                return null;
            }
            const n = sock.recv(self.fd, self.buf[self.len..]);
            if (n <= 0)
            {
                return error.LinkClosed;
            }
            self.len += @intCast(n);
        }
    }
};

/// A name a node may go by: letters, digits, dot, dash and underscore.
pub fn validName(name: []const u8) bool
{
    if (name.len == 0 or name.len > 63)
    {
        return false;
    }
    for (name) |ch|
    {
        if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-' or ch == '_'))
        {
            return false;
        }
    }
    return true;
}

/// The agent's half of the handshake. Fails if the dashboard cannot prove it holds the secret.
pub fn handshakeNode(link: *Link, secret: []const u8, name: []const u8) !void
{
    var nn: [nonce_len]u8 = undefined;
    randomBytes(&nn);
    var hello: [nonce_len + 64]u8 = undefined;
    @memcpy(hello[0..nonce_len], &nn);
    @memcpy(hello[nonce_len..][0..name.len], name);
    try link.send(.hello, hello[0 .. nonce_len + name.len]);
    const ch = (try link.recv(10_000)) orelse return error.HandshakeTimeout;
    if (ch.kind == .refuse)
    {
        std.debug.print("node: the dashboard refused the link: {s}\n", .{ch.body[0..@min(ch.body.len, 200)]});
        return error.Refused;
    }
    if (ch.kind != .challenge or ch.body.len != nonce_len + mac_len)
    {
        return error.BadHandshake;
    }
    const nd = ch.body[0..nonce_len].*;
    const p = Proofs.compute(secret, &nn, &nd);
    if (!std.crypto.timing_safe.eql([mac_len]u8, p.dashboard, ch.body[nonce_len..][0..mac_len].*))
    {
        return error.DashboardNotTrusted;
    }
    try link.send(.proof, &p.node);
    link.key = p.key;
    const w = (try link.recv(10_000)) orelse return error.HandshakeTimeout;
    if (w.kind != .welcome)
    {
        return error.Refused;
    }
}

/// The dashboard's half. Returns the node's name, valid until the next `recv`.
pub fn handshakeDashboard(link: *Link, secret: []const u8, name_out: *[64]u8) ![]const u8
{
    const hello = (try link.recv(10_000)) orelse return error.HandshakeTimeout;
    if (hello.kind != .hello or hello.body.len <= nonce_len or !validName(hello.body[nonce_len..]))
    {
        return error.BadHandshake;
    }
    const nn = hello.body[0..nonce_len].*;
    const name_len = hello.body.len - nonce_len;
    @memcpy(name_out[0..name_len], hello.body[nonce_len..]);
    var nd: [nonce_len]u8 = undefined;
    randomBytes(&nd);
    const p = Proofs.compute(secret, &nn, &nd);
    var ch: [nonce_len + mac_len]u8 = undefined;
    @memcpy(ch[0..nonce_len], &nd);
    @memcpy(ch[nonce_len..], &p.dashboard);
    try link.send(.challenge, &ch);
    const proof = (try link.recv(10_000)) orelse return error.HandshakeTimeout;
    if (proof.kind != .proof or proof.body.len != mac_len or !std.crypto.timing_safe.eql([mac_len]u8, p.node, proof.body[0..mac_len].*))
    {
        return error.NodeNotTrusted;
    }
    link.key = p.key;
    try link.send(.welcome, "");
    return name_out[0..name_len];
}

fn sendAll(fd: sock.Fd, bytes: []const u8) !void
{
    sock.sendAll(fd, bytes) catch return error.LinkClosed;
}

test "secrets compare equal only when identical" {
    try std.testing.expect(secretEql("abc", "abc"));
    try std.testing.expect(!secretEql("abc", "abd"));
    try std.testing.expect(!secretEql("abc", "abcd"));
}

test "sessions open, check, expire and revoke" {
    var s: Sessions = .{ .ttl_ns = 100 };
    var id: [id_len]u8 = undefined;
    s.create(1000, &id);
    try std.testing.expect(s.valid(&id, 1050));
    try std.testing.expect(!s.valid(&id, 1100));
    var other = id;
    other[0] = if (other[0] == 'a') 'b' else 'a';
    try std.testing.expect(!s.valid(&other, 1050));
    try std.testing.expect(!s.valid("short", 1050));
    try std.testing.expect(!s.valid("zz" ++ id[2..], 1050));
    s.create(2000, &id);
    s.revoke(&id, 2001);
    try std.testing.expect(!s.valid(&id, 2002));
}

test "reads cookies and bearer tokens" {
    try std.testing.expectEqualStrings("xyz", cookieValue("a=1; tf_session=xyz; b=2", "tf_session").?);
    try std.testing.expect(cookieValue("tf_sessionx=1", "tf_session") == null);
    try std.testing.expectEqualStrings("tok", bearer("bearer tok").?);
    try std.testing.expect(bearer("Basic abc") == null);
}

test "both sides derive the same proofs, and a wrong secret gives others" {
    var nn: [nonce_len]u8 = undefined;
    var nd: [nonce_len]u8 = undefined;
    randomBytes(&nn);
    randomBytes(&nd);
    const a = Proofs.compute("secret-one", &nn, &nd);
    const b = Proofs.compute("secret-one", &nn, &nd);
    const w = Proofs.compute("secret-two", &nn, &nd);
    try std.testing.expectEqualSlices(u8, &a.key, &b.key);
    try std.testing.expect(!std.mem.eql(u8, &a.node, &w.node));
    try std.testing.expect(!std.mem.eql(u8, &a.node, &a.dashboard));
}

test "signed frames pass and forged ones are refused" {
    const gpa = std.testing.allocator;
    const fds = try sock.socketPair();
    defer sock.close(fds[0]);
    defer sock.close(fds[1]);
    var node = try Link.init(gpa, fds[0], true);
    defer node.deinit(gpa);
    var dash = try Link.init(gpa, fds[1], false);
    defer dash.deinit(gpa);
    const key: [mac_len]u8 = @splat(7);
    node.key = key;
    dash.key = key;

    try node.send(.report, "{\"a\":1}");
    try node.send(.report, "{\"a\":2}");
    const f1 = (try dash.recv(1000)).?;
    try std.testing.expectEqualStrings("{\"a\":1}", f1.body);
    const f2 = (try dash.recv(1000)).?;
    try std.testing.expectEqualStrings("{\"a\":2}", f2.body);
    try std.testing.expect((try dash.recv(10)) == null);

    // A frame signed with another key.
    var forger = try Link.init(gpa, fds[0], true);
    defer forger.deinit(gpa);
    forger.key = @splat(9);
    forger.send_seq = node.send_seq;
    try forger.send(.report, "{}");
    try std.testing.expectError(error.BadSignature, dash.recv(1000));
}

test "a replayed frame is refused" {
    const gpa = std.testing.allocator;
    const fds = try sock.socketPair();
    defer sock.close(fds[0]);
    defer sock.close(fds[1]);
    var node = try Link.init(gpa, fds[0], true);
    defer node.deinit(gpa);
    var dash = try Link.init(gpa, fds[1], false);
    defer dash.deinit(gpa);
    node.key = @splat(3);
    dash.key = @splat(3);
    try node.send(.report, "x");
    _ = (try dash.recv(1000)).?;
    node.send_seq -= 1;
    try node.send(.report, "x");
    try std.testing.expectError(error.BadSequence, dash.recv(1000));
}

test "a frame sent back to its sender is refused" {
    const gpa = std.testing.allocator;
    const fds = try sock.socketPair();
    defer sock.close(fds[0]);
    defer sock.close(fds[1]);
    var a = try Link.init(gpa, fds[0], true);
    defer a.deinit(gpa);
    var b = try Link.init(gpa, fds[1], true);
    defer b.deinit(gpa);
    try a.send(.report, "x");
    try std.testing.expectError(error.BadFrame, b.recv(1000));
}

fn handshakeThread(link: *Link, secret: []const u8, err: *?anyerror) void
{
    handshakeNode(link, secret, "node-a") catch |e| {
        err.* = e;
        sock.shutdownBoth(link.fd);
    };
}

test "the handshake succeeds with one secret and fails with two" {
    const gpa = std.testing.allocator;
    for ([_][]const u8{ "the-shared-secret", "another-secret!!" }) |node_secret|
    {
        const fds = try sock.socketPair();
        defer sock.close(fds[0]);
        var node = try Link.init(gpa, fds[0], true);
        defer node.deinit(gpa);
        var dash = try Link.init(gpa, fds[1], false);
        defer dash.deinit(gpa);
        var node_err: ?anyerror = null;
        const t = try std.Thread.spawn(.{}, handshakeThread, .{ &node, node_secret, &node_err });
        var name: [64]u8 = undefined;
        const r = handshakeDashboard(&dash, "the-shared-secret", &name);
        // The agent stops waiting once the dashboard hangs up.
        sock.close(fds[1]);
        t.join();
        if (std.mem.eql(u8, node_secret, "the-shared-secret"))
        {
            try std.testing.expectEqualStrings("node-a", try r);
            try std.testing.expect(node_err == null);
        }
        else
        {
            try std.testing.expect(std.meta.isError(r));
            try std.testing.expect(node_err != null);
        }
    }
}

test "a new secret file is created private and read back" {
    const gpa = std.testing.allocator;
    if (is_windows)
    {
        // Inside the profile (TEMP is) it is accepted; in the working folder, outside it, refused.
        const temp = plat.env("TEMP") orelse return error.SkipZigTest;
        const inside = try std.fmt.allocPrintSentinel(gpa, "{s}\\tf_auth_test.secret", .{temp}, 0);
        defer gpa.free(inside);
        plat.deleteFile(inside);
        defer plat.deleteFile(inside);
        const a = try loadSecret(gpa, inside, "test secret");
        defer gpa.free(a);
        const b = try loadSecret(gpa, inside, "test secret");
        defer gpa.free(b);
        try std.testing.expectEqual(@as(usize, 64), a.len);
        try std.testing.expectEqualStrings(a, b);
        const outside = "tf_auth_test_outside.secret";
        if (plat.inUserProfile("."))
        {
            return;
        }
        try std.testing.expectError(error.SecretExposed, loadSecret(gpa, outside, "test secret"));
        try std.testing.expect(plat.regularFileBytes(outside) == null); // refused before writing anything
        return;
    }
    const path = "/tmp/tf_auth_test.secret";
    _ = c.unlink(path);
    defer _ = c.unlink(path);
    const a = try loadSecret(gpa, path, "test secret");
    defer gpa.free(a);
    const b = try loadSecret(gpa, path, "test secret");
    defer gpa.free(b);
    try std.testing.expectEqual(@as(usize, 64), a.len);
    try std.testing.expectEqualStrings(a, b);
    _ = c.chmod(path, 0o644);
    try std.testing.expectError(error.SecretExposed, loadSecret(gpa, path, "test secret"));
}
