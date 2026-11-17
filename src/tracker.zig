//! Swarm node tracker (TASK-09). Linux.
//!
//!   tracker serve [--port P] [--beat-ms B] [--timeout-ms T]
//!   tracker watch IP[:PORT]                       print the map on every change
//!   tracker bench IP[:PORT] [--peers N] [--rounds R]
//!
//! The control plane, after Lumabri's index-only tracker: engine processes join
//! over TCP (default port 47300) with their data address and the experts they
//! own, send a heartbeat every `beat_ms`, and receive the whole process map
//! every time it changes. Token payloads never pass through it. A process is
//! marked unhealthy when its connection closes without a goodbye (a killed
//! process) or when no heartbeat arrives for `timeout_ms` (a stopped or hung
//! one), and the new map goes out at once. If a client loses the tracker it
//! fails closed: every peer counts as unhealthy, since health is unknown.
const std = @import("std");
const c = std.c;
const tcp = @import("transport/tcp.zig");

pub const default_port: u16 = 47300;
pub const max_procs = 64;
/// One bit per expert, up to the engine's 1024.
pub const bitmap_bytes = 128;

const magic: u32 = 0x4B544654; // "TFTK"
const version: u16 = 1;

pub const Role = enum(u8) {
    router = 0,
    expert = 1,
    watcher = 2,
};

pub const Health = enum(u8) {
    healthy = 0,
    unhealthy = 1,
    left = 2,
};

/// What a process tells the tracker about itself.
pub const Info = extern struct {
    version: u16 = version,
    role: Role,
    rank: u8 = 0,
    pid: u32 = 0,
    /// IPv4 in network order, or 0 for the address the tracker sees.
    addr: u32 = 0,
    /// Where peers reach this process's data link.
    port: u16 = 0,
    /// Experts in the model, so `owned` can be checked.
    experts: u16 = 0,
    node: [32]u8 = @splat(0),
    owned: [bitmap_bytes]u8 = @splat(0),

    pub fn owns(self: *const Info, e: usize) bool
    {
        return e < self.experts and self.owned[e / 8] & (@as(u8, 1) << @intCast(e % 8)) != 0;
    }

    pub fn setOwned(self: *Info, e: usize) void
    {
        self.owned[e / 8] |= @as(u8, 1) << @intCast(e % 8);
    }

    pub fn ownedCount(self: *const Info) usize
    {
        var n: usize = 0;
        for (0..self.experts) |e|
        {
            n += @intFromBool(self.owns(e));
        }
        return n;
    }

    pub fn nodeName(self: *const Info) []const u8
    {
        return std.mem.sliceTo(&self.node, 0);
    }

    /// Fills `node` with this machine's host name and `pid` with this process.
    pub fn setLocal(self: *Info) void
    {
        _ = c.gethostname(&self.node, self.node.len - 1);
        self.node[self.node.len - 1] = 0;
        self.pid = @intCast(c.getpid());
    }
};

pub const Entry = extern struct {
    id: u32,
    health: Health,
    _pad: [3]u8 = @splat(0),
    info: Info,
};

const Kind = enum(u8) {
    join = 1,
    beat = 2,
    bye = 3,
    welcome = 4,
    map = 5,
    _,
};

const Head = extern struct {
    magic: u32 = magic,
    kind: Kind,
    _pad: [3]u8 = @splat(0),
    len: u32,
};

const Welcome = extern struct {
    id: u32,
    beat_ms: u32,
    timeout_ms: u32,
};

const MapHead = extern struct {
    version: u64,
    count: u32,
    _pad: u32 = 0,
};

const max_frame = @sizeOf(Head) + @sizeOf(MapHead) + max_procs * @sizeOf(Entry);

/// Splits `IP[:PORT]`.
pub fn parseAddress(text: []const u8) !struct { host: []const u8, port: u16 }
{
    if (std.mem.indexOfScalar(u8, text, ':')) |i|
    {
        return .{ .host = text[0..i], .port = try std.fmt.parseInt(u16, text[i + 1 ..], 10) };
    }
    return .{ .host = text, .port = default_port };
}

pub fn formatIp4(addr: u32, buf: *[16]u8) []const u8
{
    const b: [4]u8 = @bitCast(addr);
    return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}", .{ b[0], b[1], b[2], b[3] }) catch unreachable;
}

pub fn monoNs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Wall clock in milliseconds, for logs compared across processes.
pub fn wallMs() u64
{
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / std.time.ns_per_ms;
}

/// Time of day (UTC) to the millisecond, for logs.
pub fn stamp(buf: *[16]u8) []const u8
{
    const ms = wallMs();
    const day = ms % (24 * 3600 * 1000);
    return std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}", .{ day / 3600000, day / 60000 % 60, day / 1000 % 60, day % 1000 }) catch unreachable;
}

/// Sends a whole frame without blocking; false if the socket could not take
/// all of it (the stream is then unusable and the caller closes it).
fn sendFrame(fd: tcp.Socket, kind: Kind, parts: []const []const u8, block: bool) bool
{
    var buf: [max_frame]u8 = undefined;
    var len: usize = @sizeOf(Head);
    for (parts) |p|
    {
        @memcpy(buf[len..][0..p.len], p);
        len += p.len;
    }
    const head: Head = .{ .kind = kind, .len = @intCast(len - @sizeOf(Head)) };
    @memcpy(buf[0..@sizeOf(Head)], std.mem.asBytes(&head));
    const flags: u32 = c.MSG.NOSIGNAL | if (block) 0 else @as(u32, c.MSG.DONTWAIT);
    var off: usize = 0;
    while (off < len)
    {
        const n = c.send(fd, buf[off..].ptr, len - off, flags);
        if (n < 0 and c.errno(n) == .INTR)
        {
            continue;
        }
        if (n <= 0)
        {
            return false;
        }
        off += @intCast(n);
        if (!block and off < len)
        {
            return false;
        }
    }
    return true;
}

/// Bytes received on one connection, cut into whole frames.
const Inbox = struct {
    buf: [max_frame]u8 = undefined,
    len: usize = 0,

    /// Reads what has arrived. False when the peer closed or the read failed.
    fn fill(self: *Inbox, fd: tcp.Socket) bool
    {
        const n = c.recv(fd, self.buf[self.len..].ptr, self.buf.len - self.len, c.MSG.DONTWAIT);
        if (n < 0)
        {
            const e = c.errno(n);
            return e == .INTR or e == .AGAIN;
        }
        if (n == 0)
        {
            return false;
        }
        self.len += @intCast(n);
        return true;
    }

    /// The next whole frame, or null. `error.BadFrame` if the stream is not ours.
    fn next(self: *Inbox) !?struct { kind: Kind, body: []const u8 }
    {
        if (self.len < @sizeOf(Head))
        {
            return null;
        }
        const head = std.mem.bytesToValue(Head, self.buf[0..@sizeOf(Head)]);
        if (head.magic != magic or head.len > max_frame - @sizeOf(Head))
        {
            return error.BadFrame;
        }
        const total = @sizeOf(Head) + head.len;
        if (self.len < total)
        {
            return null;
        }
        return .{ .kind = head.kind, .body = self.buf[@sizeOf(Head)..total] };
    }

    /// Drops the frame `next` returned.
    fn consume(self: *Inbox) void
    {
        const head = std.mem.bytesToValue(Head, self.buf[0..@sizeOf(Head)]);
        const total = @sizeOf(Head) + head.len;
        std.mem.copyForwards(u8, self.buf[0 .. self.len - total], self.buf[total..self.len]);
        self.len -= total;
    }
};

pub const Server = struct {
    lfd: tcp.Socket,
    port: u16,
    beat_ms: u32,
    timeout_ms: u32,
    slots: [max_procs]Slot = @splat(.{}),
    next_id: u32 = 1,
    map_version: u64 = 0,
    log: bool = true,

    /// Unhealthy and departed processes stay in the map this long.
    const keep_ns = 30 * std.time.ns_per_s;

    const Slot = struct {
        fd: ?tcp.Socket = null,
        inbox: Inbox = .{},
        /// 0 until the process joins.
        id: u32 = 0,
        info: Info = .{ .role = .watcher },
        health: Health = .healthy,
        last_beat: u64 = 0,
        down_at: u64 = 0,

        fn listed(self: *const Slot) bool
        {
            return self.id != 0 and self.info.role != .watcher;
        }
    };

    pub fn open(port: u16, beat_ms: u32, timeout_ms: u32) !Server
    {
        const l = try tcp.listenMany(port, max_procs);
        return .{ .lfd = l.fd, .port = l.port, .beat_ms = beat_ms, .timeout_ms = timeout_ms };
    }

    pub fn close(self: *Server) void
    {
        for (&self.slots) |*s|
        {
            if (s.fd) |fd|
            {
                tcp.closeSocket(fd);
            }
        }
        tcp.closeSocket(self.lfd);
    }

    /// Serves until `stop` is set.
    pub fn run(self: *Server, stop: *const std.atomic.Value(bool)) void
    {
        while (!stop.load(.acquire))
        {
            self.step(20);
        }
    }

    /// One round: waits up to `wait_ms` for traffic, handles it, checks
    /// heartbeats, and sends the map to everyone if anything changed.
    pub fn step(self: *Server, wait_ms: c_int) void
    {
        var pfds: [max_procs + 1]c.pollfd = undefined;
        var which: [max_procs + 1]usize = undefined;
        pfds[0] = .{ .fd = self.lfd, .events = c.POLL.IN, .revents = 0 };
        var n: usize = 1;
        for (&self.slots, 0..) |*s, i|
        {
            if (s.fd) |fd|
            {
                pfds[n] = .{ .fd = fd, .events = c.POLL.IN, .revents = 0 };
                which[n] = i;
                n += 1;
            }
        }
        _ = c.poll(&pfds, @intCast(n), wait_ms);
        var changed = false;
        const now = monoNs();
        if (pfds[0].revents & c.POLL.IN != 0)
        {
            changed = self.accept();
        }
        for (pfds[1..n], which[1..n]) |p, i|
        {
            if (p.revents != 0)
            {
                changed = self.read(&self.slots[i], now) or changed;
            }
        }
        for (&self.slots) |*s|
        {
            if (s.listed() and s.health == .healthy and now - s.last_beat > @as(u64, self.timeout_ms) * std.time.ns_per_ms)
            {
                self.markDown(s, .unhealthy, "no heartbeat", now);
                changed = true;
            }
            if (s.listed() and s.health != .healthy and s.fd == null and now - s.down_at > keep_ns)
            {
                self.note(s, "dropped from the map");
                s.* = .{};
                changed = true;
            }
        }
        // A client that cannot take the map is closed, which changes the map again.
        var rounds: usize = 0;
        while (changed and rounds < 4) : (rounds += 1)
        {
            changed = self.broadcast(now);
        }
    }

    /// Takes a new connection. When every slot is in use, the process that
    /// went down longest ago leaves the map to make room. True if the map changed.
    fn accept(self: *Server) bool
    {
        const fd = tcp.acceptOn(self.lfd) catch return false;
        for (&self.slots) |*s|
        {
            if (s.fd == null and s.id == 0)
            {
                s.* = .{ .fd = fd };
                return false;
            }
        }
        var oldest: ?*Slot = null;
        for (&self.slots) |*s|
        {
            if (s.listed() and s.health != .healthy and s.fd == null and (oldest == null or s.down_at < oldest.?.down_at))
            {
                oldest = s;
            }
        }
        if (oldest) |s|
        {
            self.note(s, "dropped from the map to make room");
            s.* = .{ .fd = fd };
            return true;
        }
        std.debug.print("tracker: full ({d} live processes), refusing a connection\n", .{max_procs});
        tcp.closeSocket(fd);
        return false;
    }

    /// Handles what arrived on `s`. True if the map changed.
    fn read(self: *Server, s: *Slot, now: u64) bool
    {
        const fd = s.fd.?;
        if (!s.inbox.fill(fd))
        {
            return self.drop(s, "connection closed without a goodbye", now);
        }
        var changed = false;
        while (true)
        {
            const f = (s.inbox.next() catch return self.drop(s, "sent a malformed frame", now)) orelse break;
            switch (f.kind)
            {
                .join =>
                {
                    if (f.body.len != @sizeOf(Info))
                    {
                        return self.drop(s, "sent a malformed join", now);
                    }
                    var info = std.mem.bytesToValue(Info, f.body);
                    if (info.version != version or info.experts > bitmap_bytes * 8)
                    {
                        return self.drop(s, "speaks another tracker version", now);
                    }
                    if (info.addr == 0)
                    {
                        info.addr = peerAddress(fd);
                    }
                    const first = s.id == 0;
                    if (first)
                    {
                        s.id = self.next_id;
                        self.next_id += 1;
                    }
                    s.info = info;
                    s.health = .healthy;
                    s.last_beat = now;
                    if (first)
                    {
                        const w: Welcome = .{ .id = s.id, .beat_ms = self.beat_ms, .timeout_ms = self.timeout_ms };
                        if (!sendFrame(fd, .welcome, &.{std.mem.asBytes(&w)}, false))
                        {
                            return self.drop(s, "could not be greeted", now);
                        }
                    }
                    self.note(s, if (first) "joined" else "updated");
                    changed = changed or s.listed() or first;
                },
                .beat =>
                {
                    s.last_beat = now;
                    if (s.listed() and s.health == .unhealthy)
                    {
                        s.health = .healthy;
                        self.note(s, "heartbeat again, healthy");
                        changed = true;
                    }
                },
                .bye =>
                {
                    s.inbox.consume();
                    if (!s.listed())
                    {
                        tcp.closeSocket(fd);
                        s.* = .{};
                        return changed;
                    }
                    self.markDown(s, .left, "left", now);
                    return true;
                },
                else => return self.drop(s, "sent an unknown frame", now),
            }
            s.inbox.consume();
        }
        return changed;
    }

    /// Closes `s` and marks it unhealthy. True if the map changed.
    fn drop(self: *Server, s: *Slot, why: []const u8, now: u64) bool
    {
        if (s.id == 0 or s.info.role == .watcher)
        {
            if (s.fd) |fd|
            {
                tcp.closeSocket(fd);
            }
            s.* = .{};
            return false;
        }
        self.markDown(s, if (s.health == .left) .left else .unhealthy, why, now);
        return true;
    }

    fn markDown(self: *Server, s: *Slot, health: Health, why: []const u8, now: u64) void
    {
        s.health = health;
        s.down_at = now;
        if (health == .left or why.len > 0 and !std.mem.eql(u8, why, "no heartbeat"))
        {
            if (s.fd) |fd|
            {
                tcp.closeSocket(fd);
            }
            s.fd = null;
        }
        self.note(s, if (health == .left) "left" else why);
    }

    /// Sends the map to every connected process. True if a send failed, which
    /// closed that connection and changed the map.
    fn broadcast(self: *Server, now: u64) bool
    {
        self.map_version += 1;
        var entries: [max_procs]Entry = undefined;
        var count: u32 = 0;
        for (&self.slots) |*s|
        {
            if (s.listed())
            {
                entries[count] = .{ .id = s.id, .health = s.health, .info = s.info };
                count += 1;
            }
        }
        const head: MapHead = .{ .version = self.map_version, .count = count };
        const body = std.mem.sliceAsBytes(entries[0..count]);
        var failed = false;
        for (&self.slots) |*s|
        {
            if (s.fd == null or s.id == 0)
            {
                continue;
            }
            if (!sendFrame(s.fd.?, .map, &.{ std.mem.asBytes(&head), body }, false))
            {
                failed = self.drop(s, "stopped reading the map", now) or failed;
            }
        }
        if (self.log)
        {
            var t: [16]u8 = undefined;
            var healthy: usize = 0;
            for (entries[0..count]) |e|
            {
                healthy += @intFromBool(e.health == .healthy);
            }
            std.debug.print("{s} tracker: map v{d}: {d} processes, {d} healthy\n", .{ stamp(&t), self.map_version, count, healthy });
        }
        return failed;
    }

    fn note(self: *Server, s: *const Slot, what: []const u8) void
    {
        if (!self.log)
        {
            return;
        }
        var t: [16]u8 = undefined;
        var ip: [16]u8 = undefined;
        std.debug.print("{s} tracker: id {d} {t} rank {d} pid {d} on {s} ({s}:{d}, {d} of {d} experts): {s}\n", .{
            stamp(&t), s.id, s.info.role, s.info.rank, s.info.pid, s.info.nodeName(), formatIp4(s.info.addr, &ip), s.info.port, s.info.ownedCount(), s.info.experts, what,
        });
    }
};

fn peerAddress(fd: tcp.Socket) u32
{
    var sa: c.sockaddr.in = undefined;
    var len: c.socklen_t = @sizeOf(c.sockaddr.in);
    if (c.getpeername(fd, @ptrCast(&sa), &len) != 0)
    {
        return 0;
    }
    return sa.addr;
}

/// The latest map a client received.
pub const View = struct {
    version: u64 = 0,
    count: usize = 0,
    entries: [max_procs]Entry = undefined,
    /// When it arrived (monotonic).
    at_ns: u64 = 0,

    pub fn find(self: *const View, id: u32) ?*const Entry
    {
        for (self.entries[0..self.count]) |*e|
        {
            if (e.id == id)
            {
                return e;
            }
        }
        return null;
    }
};

/// One process's link to the tracker: joins, then a thread sends heartbeats
/// and keeps the latest map.
pub const Client = struct {
    fd: tcp.Socket,
    id: u32,
    beat_ms: u32,
    timeout_ms: u32,
    thread: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    /// Set when the tracker connection fails; health is unknown from then on.
    lost: std.atomic.Value(bool) = .init(false),
    /// While set, no heartbeats go out (tests a hung process).
    paused: std.atomic.Value(bool) = .init(false),
    /// The process whose health `peerHealthy` reports, and that health.
    watch_id: std.atomic.Value(u32) = .init(0),
    watch_ok: std.atomic.Value(bool) = .init(false),
    locked: std.atomic.Value(bool) = .init(false),
    view: View = .{},
    inbox: Inbox = .{},

    /// Connects, joins, and waits for the tracker's greeting.
    pub fn start(gpa: std.mem.Allocator, host: []const u8, port: u16, info: Info) !*Client
    {
        const fd = try tcp.connect(host, port, 50);
        errdefer tcp.closeSocket(fd);
        if (!sendFrame(fd, .join, &.{std.mem.asBytes(&info)}, true))
        {
            return error.TrackerClosed;
        }
        const self = try gpa.create(Client);
        errdefer gpa.destroy(self);
        self.* = .{ .fd = fd, .id = 0, .beat_ms = 0, .timeout_ms = 0 };
        // The greeting comes first, possibly with a map behind it.
        while (self.id == 0)
        {
            if (!(tcp.waitReadable(fd, 5000) catch false) or !self.inbox.fill(fd))
            {
                return error.TrackerClosed;
            }
            while (try self.inbox.next()) |f|
            {
                if (f.kind == .welcome and f.body.len == @sizeOf(Welcome))
                {
                    const w = std.mem.bytesToValue(Welcome, f.body);
                    self.id = w.id;
                    self.beat_ms = w.beat_ms;
                    self.timeout_ms = w.timeout_ms;
                }
                else if (f.kind == .map)
                {
                    self.takeMap(f.body);
                }
                self.inbox.consume();
            }
        }
        self.thread = try std.Thread.spawn(.{}, loop, .{self});
        return self;
    }

    /// Says goodbye (unless `abrupt`, which just drops the connection like a
    /// killed process) and stops the thread.
    pub fn close(self: *Client, gpa: std.mem.Allocator, abrupt: bool) void
    {
        self.stop.store(true, .release);
        self.thread.join();
        if (!abrupt and !self.lost.load(.acquire))
        {
            _ = sendFrame(self.fd, .bye, &.{}, true);
            tcp.closeAfterPeerSocket(self.fd, false);
        }
        else
        {
            tcp.closeSocket(self.fd);
        }
        gpa.destroy(self);
    }

    /// Sends a new join with changed details; the id stays the same.
    pub fn update(self: *Client, info: Info) !void
    {
        self.lock();
        defer self.unlock();
        if (!sendFrame(self.fd, .join, &.{std.mem.asBytes(&info)}, true))
        {
            return error.TrackerClosed;
        }
    }

    pub fn snapshot(self: *Client, out: *View) void
    {
        self.lock();
        defer self.unlock();
        out.* = self.view;
    }

    /// Makes `peerHealthy` report on process `id`.
    pub fn watch(self: *Client, id: u32) void
    {
        self.lock();
        defer self.unlock();
        self.watch_id.store(id, .release);
        self.refreshWatch();
    }

    /// Whether the watched process is healthy in the latest map. Cheap enough
    /// to call on every dispatch.
    pub fn peerHealthy(self: *const Client) bool
    {
        return self.watch_ok.load(.acquire);
    }

    /// Waits up to `timeout_ms` for a healthy process with `role` that owns
    /// at least one expert.
    pub fn waitFor(self: *Client, role: Role, timeout_ms: u64) !Entry
    {
        const end = monoNs() + timeout_ms * std.time.ns_per_ms;
        var v: View = .{};
        while (monoNs() < end)
        {
            if (self.lost.load(.acquire))
            {
                return error.TrackerLost;
            }
            self.snapshot(&v);
            for (v.entries[0..v.count]) |e|
            {
                if (e.info.role == role and e.health == .healthy and e.info.ownedCount() > 0)
                {
                    return e;
                }
            }
            sleepMs(5);
        }
        return error.NoHealthyPeer;
    }

    fn lock(self: *Client) void
    {
        while (self.locked.swap(true, .acquire))
        {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(self: *Client) void
    {
        self.locked.store(false, .release);
    }

    fn refreshWatch(self: *Client) void
    {
        const id = self.watch_id.load(.acquire);
        const e = self.view.find(id);
        self.watch_ok.store(!self.lost.load(.acquire) and e != null and e.?.health == .healthy, .release);
    }

    fn takeMap(self: *Client, body: []const u8) void
    {
        if (body.len < @sizeOf(MapHead))
        {
            return;
        }
        const head = std.mem.bytesToValue(MapHead, body[0..@sizeOf(MapHead)]);
        if (head.count > max_procs or body.len != @sizeOf(MapHead) + head.count * @sizeOf(Entry))
        {
            return;
        }
        self.lock();
        defer self.unlock();
        self.view.version = head.version;
        self.view.count = head.count;
        @memcpy(std.mem.sliceAsBytes(self.view.entries[0..head.count]), body[@sizeOf(MapHead)..]);
        self.view.at_ns = monoNs();
        self.refreshWatch();
    }

    fn loop(self: *Client) void
    {
        var next_beat = monoNs();
        while (!self.stop.load(.acquire))
        {
            const now = monoNs();
            if (now >= next_beat)
            {
                if (!self.paused.load(.acquire))
                {
                    self.lock();
                    const ok = sendFrame(self.fd, .beat, &.{}, true);
                    self.unlock();
                    if (!ok)
                    {
                        return self.fail();
                    }
                }
                next_beat = now + @as(u64, self.beat_ms) * std.time.ns_per_ms;
            }
            const wait: u64 = @min((next_beat -| monoNs()) / std.time.ns_per_ms, 20);
            if (!(tcp.waitReadable(self.fd, @intCast(wait)) catch false))
            {
                continue;
            }
            if (!self.inbox.fill(self.fd))
            {
                return self.fail();
            }
            while (true)
            {
                const f = (self.inbox.next() catch return self.fail()) orelse break;
                if (f.kind == .map)
                {
                    self.takeMap(f.body);
                }
                self.inbox.consume();
            }
        }
    }

    fn fail(self: *Client) void
    {
        if (self.stop.load(.acquire))
        {
            return;
        }
        self.lock();
        self.lost.store(true, .release);
        self.refreshWatch();
        self.unlock();
        std.debug.print("tracker: lost the connection to the tracker; treating every peer as unhealthy\n", .{});
    }
};

fn sleepMs(ms: u32) void
{
    const ts: c.timespec = .{ .sec = 0, .nsec = @as(c_long, ms) * std.time.ns_per_ms };
    _ = c.nanosleep(&ts, null);
}

pub fn main(init: std.process.Init) !void
{
    const gpa = init.gpa;
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    const usage =
        \\usage: tracker serve [--port P] [--beat-ms B] [--timeout-ms T]
        \\       tracker watch IP[:PORT]
        \\       tracker bench IP[:PORT] [--peers N] [--rounds R]
        \\
    ;
    const mode = it.next() orelse
    {
        std.debug.print(usage, .{});
        std.process.exit(2);
    };
    if (std.mem.eql(u8, mode, "serve"))
    {
        var port = default_port;
        var beat_ms: u32 = 250;
        var timeout_ms: u32 = 1000;
        while (it.next()) |arg|
        {
            const value = it.next() orelse
            {
                std.debug.print(usage, .{});
                std.process.exit(2);
            };
            if (std.mem.eql(u8, arg, "--port"))
            {
                port = try std.fmt.parseInt(u16, value, 10);
            }
            else if (std.mem.eql(u8, arg, "--beat-ms"))
            {
                beat_ms = try std.fmt.parseInt(u32, value, 10);
            }
            else if (std.mem.eql(u8, arg, "--timeout-ms"))
            {
                timeout_ms = try std.fmt.parseInt(u32, value, 10);
            }
            else
            {
                std.debug.print(usage, .{});
                std.process.exit(2);
            }
        }
        var server = try Server.open(port, beat_ms, timeout_ms);
        defer server.close();
        std.debug.print("tracker: listening on port {d}, heartbeat every {d} ms, unhealthy after {d} ms without one\n", .{ server.port, beat_ms, timeout_ms });
        const never: std.atomic.Value(bool) = .init(false);
        server.run(&never);
        return;
    }
    const where = parseAddress(it.next() orelse
    {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }) catch
    {
        std.debug.print(usage, .{});
        std.process.exit(2);
    };
    if (std.mem.eql(u8, mode, "watch"))
    {
        return watchMain(gpa, where.host, where.port);
    }
    if (std.mem.eql(u8, mode, "bench"))
    {
        var peers: usize = 8;
        var rounds: usize = 20;
        while (it.next()) |arg|
        {
            const value = it.next() orelse return error.Usage;
            if (std.mem.eql(u8, arg, "--peers"))
            {
                peers = try std.fmt.parseInt(usize, value, 10);
            }
            else if (std.mem.eql(u8, arg, "--rounds"))
            {
                rounds = try std.fmt.parseInt(usize, value, 10);
            }
            else
            {
                return error.Usage;
            }
        }
        return benchMain(gpa, where.host, where.port, peers, rounds);
    }
    std.debug.print(usage, .{});
    std.process.exit(2);
}

fn watchMain(gpa: std.mem.Allocator, host: []const u8, port: u16) !void
{
    var info: Info = .{ .role = .watcher };
    info.setLocal();
    const client = try Client.start(gpa, host, port, info);
    defer client.close(gpa, false);
    var seen: u64 = 0;
    var v: View = .{};
    while (!client.lost.load(.acquire))
    {
        client.snapshot(&v);
        if (v.version != seen)
        {
            seen = v.version;
            var t: [16]u8 = undefined;
            std.debug.print("{s} map v{d}:\n", .{ stamp(&t), v.version });
            for (v.entries[0..v.count]) |e|
            {
                var ip: [16]u8 = undefined;
                std.debug.print("  id {d} {t} {t} rank {d} pid {d} on {s} at {s}:{d}, {d} of {d} experts\n", .{
                    e.id, e.health, e.info.role, e.info.rank, e.info.pid, e.info.nodeName(), formatIp4(e.info.addr, &ip), e.info.port, e.info.ownedCount(), e.info.experts,
                });
            }
        }
        sleepMs(2);
    }
    return error.TrackerLost;
}

/// Measures, from one clock, how long until every peer's map shows a process
/// that joined, one whose connection dropped, and one that stopped beating.
fn benchMain(gpa: std.mem.Allocator, host: []const u8, port: u16, peers: usize, rounds: usize) !void
{
    const watchers = try gpa.alloc(*Client, peers);
    defer gpa.free(watchers);
    for (watchers) |*w|
    {
        var info: Info = .{ .role = .watcher };
        info.setLocal();
        w.* = try Client.start(gpa, host, port, info);
    }
    defer for (watchers) |w|
    {
        w.close(gpa, false);
    };
    const join_ms = try gpa.alloc(f64, rounds);
    defer gpa.free(join_ms);
    const drop_ms = try gpa.alloc(f64, rounds);
    defer gpa.free(drop_ms);
    const hang_ms = try gpa.alloc(f64, rounds);
    defer gpa.free(hang_ms);
    for (0..rounds) |r|
    {
        for ([_]bool{ false, true }) |hang|
        {
            var info: Info = .{ .role = .expert, .rank = 1, .port = 7480, .experts = 64 };
            info.setLocal();
            for (32..64) |e|
            {
                info.setOwned(e);
            }
            const t0 = monoNs();
            const p = try Client.start(gpa, host, port, info);
            const joined = try untilAll(watchers, p.id, .healthy, 10_000);
            if (!hang)
            {
                join_ms[r] = @as(f64, @floatFromInt(joined - t0)) / 1e6;
            }
            const t1 = monoNs();
            if (hang)
            {
                p.paused.store(true, .release);
                hang_ms[r] = @as(f64, @floatFromInt(try untilAll(watchers, p.id, .unhealthy, 10_000) - t1)) / 1e6;
                p.close(gpa, true);
            }
            else
            {
                const id = p.id;
                p.close(gpa, true);
                drop_ms[r] = @as(f64, @floatFromInt(try untilAll(watchers, id, .unhealthy, 10_000) - t1)) / 1e6;
            }
        }
    }
    std.debug.print("bench: {d} peers, {d} rounds, tracker at {s}:{d}, heartbeat {d} ms, timeout {d} ms\n", .{ peers, rounds, host, port, watchers[0].beat_ms, watchers[0].timeout_ms });
    report("join seen by every peer", join_ms);
    report("dropped connection marked unhealthy everywhere", drop_ms);
    report("stopped heartbeat marked unhealthy everywhere", hang_ms);
}

/// When the last of `watchers` saw process `id` with `health`.
fn untilAll(watchers: []const *Client, id: u32, health: Health, timeout_ms: u64) !u64
{
    const end = monoNs() + timeout_ms * std.time.ns_per_ms;
    var last: u64 = 0;
    var v: View = .{};
    for (watchers) |w|
    {
        while (true)
        {
            w.snapshot(&v);
            if (v.find(id)) |e|
            {
                if (e.health == health)
                {
                    last = @max(last, v.at_ns);
                    break;
                }
            }
            if (monoNs() > end)
            {
                return error.Timeout;
            }
            std.atomic.spinLoopHint();
        }
    }
    return last;
}

fn report(what: []const u8, ms: []f64) void
{
    std.mem.sort(f64, ms, {}, std.sort.asc(f64));
    std.debug.print("  {s}: p50 {d:.1} ms, max {d:.1} ms\n", .{ what, ms[ms.len / 2], ms[ms.len - 1] });
}

test "join, drop, stalled heartbeat and goodbye reach every peer" {
    const gpa = std.testing.allocator;
    var server = try Server.open(0, 20, 150);
    defer server.close();
    server.log = false;
    var stop: std.atomic.Value(bool) = .init(false);
    const t = try std.Thread.spawn(.{}, Server.run, .{ &server, &stop });
    defer
    {
        stop.store(true, .release);
        t.join();
    }
    const watcher = try Client.start(gpa, "127.0.0.1", server.port, .{ .role = .watcher });
    defer watcher.close(gpa, false);

    var info: Info = .{ .role = .expert, .rank = 1, .port = 9, .experts = 64 };
    info.setOwned(40);
    const a = try Client.start(gpa, "127.0.0.1", server.port, info);
    watcher.watch(a.id);
    _ = try untilAll(&.{watcher}, a.id, .healthy, 2000);
    try std.testing.expect(watcher.peerHealthy());
    var v: View = .{};
    watcher.snapshot(&v);
    try std.testing.expect(v.find(a.id).?.info.owns(40));
    try std.testing.expect(!v.find(a.id).?.info.owns(41));
    try std.testing.expectEqual(@as(u32, @bitCast([4]u8{ 127, 0, 0, 1 })), v.find(a.id).?.info.addr);
    const a_id = a.id;
    a.close(gpa, true);
    _ = try untilAll(&.{watcher}, a_id, .unhealthy, 2000);
    try std.testing.expect(!watcher.peerHealthy());

    const b = try Client.start(gpa, "127.0.0.1", server.port, info);
    _ = try untilAll(&.{watcher}, b.id, .healthy, 2000);
    b.paused.store(true, .release);
    _ = try untilAll(&.{watcher}, b.id, .unhealthy, 2000);
    b.paused.store(false, .release);
    _ = try untilAll(&.{watcher}, b.id, .healthy, 2000);
    const b_id = b.id;
    b.close(gpa, false);
    _ = try untilAll(&.{watcher}, b_id, .left, 2000);
}
