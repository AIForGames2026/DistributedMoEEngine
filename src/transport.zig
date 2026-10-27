//! Token transport (TASK-02). Every backend offers the same one-sided model as
//! RDMA WRITE_WITH_IMM: the sender writes a whole packet into a slot of the
//! receiver's ring, then the receiver sees a 32-bit immediate saying which slot
//! landed. Backends: shared memory (`shm.zig`) and TCP (`tcp.zig`); RDMA
//! (TASK-07) plugs in behind the same `Backend` vtable.
//!
//! `Endpoint` adds what all backends share: slot credits, slot generations, and
//! the receiver checks from docs/wire-format.md.
const std = @import("std");
const wire = @import("wire.zig");

pub const shm = @import("transport/shm.zig");
pub const tcp = @import("transport/tcp.zig");

pub const Error = error{ ConnectionClosed, ProtocolError, Unexpected };

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Copies `pkt` into slot `slot` of the peer's ring, then delivers `imm`
        /// to the peer. The data must be visible before the immediate is.
        post_write_imm: *const fn (ptr: *anyopaque, slot: u16, pkt: *const wire.Packet, imm: u32) Error!void,
        /// Delivers an immediate with no data (credit grants, control).
        post_imm: *const fn (ptr: *anyopaque, imm: u32) Error!void,
        /// Non-blocking. Writes up to `out.len` arrived immediates, in order.
        poll_completions: *const fn (ptr: *anyopaque, out: []u32) Error!usize,
        /// The local receive slot `i` (the registered ring).
        slot: *const fn (ptr: *anyopaque, i: u16) *wire.Packet,
    };
};

pub const Received = struct {
    kind: wire.Kind,
    slot: u16,
    packet: *const wire.Packet,
};

pub const Stats = struct {
    sent: u64 = 0,
    received: u64 = 0,
    credits_granted: u64 = 0,
    blocked_polls: u64 = 0,
};

/// One side of a point-to-point link. Slots are used in order, so slot and
/// generation for the n-th packet are `n % slots` and `n / slots + 1` on both
/// sides, and the sender can never write a slot the receiver has not released.
pub const Endpoint = struct {
    backend: Backend,
    rank: u6,
    slots: u16,
    /// Slots the peer has granted that we have not yet written.
    credits: u32,
    send_count: u64 = 0,
    recv_count: u64 = 0,
    /// Received packets the caller has finished with but not yet credited back.
    pending_release: u32 = 0,
    /// Arrived data immediates not yet handed to the caller.
    inbox: Inbox = .{},
    /// The peer has sent its goodbye and will send nothing more.
    peer_done: bool = false,
    stats: Stats = .{},

    /// Control immediates carry their message in the slot field.
    const control_bye = 1;

    const Inbox = struct {
        buf: [256]u32 = undefined,
        head: usize = 0,
        len: usize = 0,
    };

    pub fn init(backend: Backend, rank: u6, slots: u16) Endpoint {
        std.debug.assert(slots > 0);
        return .{ .backend = backend, .rank = rank, .slots = slots, .credits = slots };
    }

    fn generation(self: *const Endpoint, n: u64) u64 {
        return n / self.slots + 1;
    }

    /// Stamps the header (generation, source rank), seals the packet, and
    /// writes it to the peer. Returns `error.WouldBlock` when the peer has no
    /// free slot; the packet is not sent and no slot is touched.
    pub fn trySend(self: *Endpoint, pkt: *wire.Packet, kind: wire.Kind) (Error || error{WouldBlock})!void {
        try self.pump();
        if (self.credits == 0) return error.WouldBlock;
        const slot: u16 = @intCast(self.send_count % self.slots);
        const gen = self.generation(self.send_count);
        pkt.header.generation = gen;
        pkt.header.src_rank = self.rank;
        pkt.seal();
        const imm: wire.Imm = .{ .slot = slot, .kind = kind, .gen8 = @truncate(gen), .src_rank = self.rank };
        try self.backend.vtable.post_write_imm(self.backend.ptr, slot, pkt, imm.encode());
        self.credits -= 1;
        self.send_count += 1;
        self.stats.sent += 1;
    }

    /// Like `trySend`, but waits for a credit. Releases our own pending
    /// credits first so two endpoints sending to each other cannot deadlock.
    pub fn send(self: *Endpoint, pkt: *wire.Packet, kind: wire.Kind) Error!void {
        while (true) {
            self.trySend(pkt, kind) catch |err| switch (err) {
                error.WouldBlock => {
                    try self.flushCredits();
                    self.stats.blocked_polls += 1;
                    std.atomic.spinLoopHint();
                    continue;
                },
                else => |e| return e,
            };
            return;
        }
    }

    /// Returns the next received packet, already checked, or null if none has
    /// arrived. The packet stays valid until `release` is called for it.
    pub fn tryRecv(self: *Endpoint) Error!?Received {
        try self.pump();
        if (self.inbox.len == 0) return null;
        const imm = wire.Imm.decode(self.inbox.buf[self.inbox.head]);
        self.inbox.head = (self.inbox.head + 1) % self.inbox.buf.len;
        self.inbox.len -= 1;

        const expected_slot: u16 = @intCast(self.recv_count % self.slots);
        const gen = self.generation(self.recv_count);
        if (imm.slot != expected_slot or imm.gen8 != @as(u8, @truncate(gen))) return error.ProtocolError;
        const pkt = self.backend.vtable.slot(self.backend.ptr, imm.slot);
        pkt.check(gen) catch return error.ProtocolError;
        self.recv_count += 1;
        self.stats.received += 1;
        return .{ .kind = imm.kind, .slot = imm.slot, .packet = pkt };
    }

    pub fn recv(self: *Endpoint) Error!Received {
        while (true) {
            if (try self.tryRecv()) |r| return r;
            std.atomic.spinLoopHint();
        }
    }

    /// Marks one received packet as consumed. Credits go back to the peer in
    /// batches of a quarter ring, or immediately on `flushCredits`.
    pub fn release(self: *Endpoint) Error!void {
        self.pending_release += 1;
        if (self.pending_release >= @max(1, self.slots / 4)) try self.flushCredits();
    }

    pub fn flushCredits(self: *Endpoint) Error!void {
        if (self.pending_release == 0) return;
        try self.grantCredit(self.pending_release);
        self.pending_release = 0;
    }

    /// Tells the peer it may write `n` more slots. Credit count rides in the
    /// immediate's slot field.
    pub fn grantCredit(self: *Endpoint, n: u32) Error!void {
        std.debug.assert(n <= self.slots);
        const imm: wire.Imm = .{ .slot = @intCast(n), .kind = .credit, .gen8 = 0, .src_rank = self.rank };
        try self.backend.vtable.post_imm(self.backend.ptr, imm.encode());
        self.stats.credits_granted += n;
    }

    /// Drains the backend: credit grants update `credits`, data immediates
    /// queue for `tryRecv`.
    fn pump(self: *Endpoint) Error!void {
        var buf: [64]u32 = undefined;
        const room = @min(buf.len, self.inbox.buf.len - self.inbox.len);
        const n = try self.backend.vtable.poll_completions(self.backend.ptr, buf[0..room]);
        for (buf[0..n]) |raw| {
            const imm = wire.Imm.decode(raw);
            switch (imm.kind) {
                .credit => {
                    self.credits += imm.slot;
                    if (self.credits > self.slots) return error.ProtocolError;
                },
                .dispatch, .combine => {
                    const tail = (self.inbox.head + self.inbox.len) % self.inbox.buf.len;
                    self.inbox.buf[tail] = raw;
                    self.inbox.len += 1;
                },
                .control => {
                    if (imm.slot != control_bye or self.peer_done) return error.ProtocolError;
                    self.peer_done = true;
                },
            }
        }
    }

    /// Returns any held credits and tells the peer we are done sending.
    pub fn sendBye(self: *Endpoint) Error!void {
        try self.flushCredits();
        const imm: wire.Imm = .{ .slot = control_bye, .kind = .control, .gen8 = 0, .src_rank = self.rank };
        try self.backend.vtable.post_imm(self.backend.ptr, imm.encode());
    }

    /// Waits for the peer's goodbye. After this, the link can be torn down
    /// without losing anything in flight, whatever the backend's close does.
    pub fn awaitBye(self: *Endpoint) Error!void {
        while (!self.peer_done) {
            try self.pump();
            std.atomic.spinLoopHint();
        }
    }

    /// `sendBye` then `awaitBye`. Call on both sides before closing the link.
    pub fn finish(self: *Endpoint) Error!void {
        try self.sendBye();
        try self.awaitBye();
    }
};

/// Deterministic test payload: every 8-byte word depends on token id and position.
pub fn fillPayload(pkt: *wire.Packet, token_id: u64, salt: u8) void {
    var s: u64 = token_id *% 0x9E3779B97F4A7C15 +% salt;
    for (payloadWords(pkt)) |*w| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        w.* = s;
    }
}

pub fn payloadMatches(pkt: *const wire.Packet, token_id: u64, salt: u8) bool {
    var s: u64 = token_id *% 0x9E3779B97F4A7C15 +% salt;
    for (payloadWords(@constCast(pkt))) |w| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        if (w != s) return false;
    }
    return true;
}

fn payloadWords(pkt: *wire.Packet) *[wire.hidden_bytes / 8]u64 {
    // The payload starts at offset 0x20 of an 8-byte aligned packet.
    return @ptrCast(@alignCast(&pkt.payload));
}

fn newPacket(token_id: u64) wire.Packet {
    var pkt = std.mem.zeroes(wire.Packet);
    pkt.header.token_id = token_id;
    pkt.header.payload_len = wire.hidden_bytes;
    fillPayload(&pkt, token_id, 0);
    return pkt;
}

test {
    _ = shm;
    _ = tcp;
}

test "packets arrive in order with generations across ring wraparound" {
    var pair = try shm.Pair.create(4);
    defer pair.deinit();
    var a = Endpoint.init(pair.side(0).backend(), 0, 4);
    var b = Endpoint.init(pair.side(1).backend(), 1, 4);

    for (0..20) |i| {
        var pkt = newPacket(i);
        try a.send(&pkt, .dispatch);
        const r = try b.recv();
        try std.testing.expectEqual(wire.Kind.dispatch, r.kind);
        try std.testing.expectEqual(@as(u64, i), r.packet.header.token_id);
        try std.testing.expectEqual(@as(u64, i / 4 + 1), r.packet.header.generation);
        try std.testing.expect(payloadMatches(r.packet, i, 0));
        try b.release();
        try b.flushCredits();
    }
}

test "withheld credits block the sender instead of overwriting a slot" {
    var pair = try shm.Pair.create(4);
    defer pair.deinit();
    var a = Endpoint.init(pair.side(0).backend(), 0, 4);
    var b = Endpoint.init(pair.side(1).backend(), 1, 4);

    for (0..4) |i| {
        var pkt = newPacket(i);
        try a.trySend(&pkt, .dispatch);
    }
    var extra = newPacket(99);
    try std.testing.expectError(error.WouldBlock, a.trySend(&extra, .dispatch));

    // The receiver still sees the original four, untouched.
    for (0..4) |i| {
        const r = (try b.tryRecv()).?;
        try std.testing.expectEqual(@as(u64, i), r.packet.header.token_id);
    }
    try std.testing.expectEqual(@as(?Received, null), try b.tryRecv());

    // One credit back lets exactly one more packet through.
    try b.grantCredit(1);
    try a.trySend(&extra, .dispatch);
    try std.testing.expectError(error.WouldBlock, a.trySend(&extra, .dispatch));
}

test "goodbye handshake completes on both sides" {
    var pair = try shm.Pair.create(4);
    defer pair.deinit();
    var a = Endpoint.init(pair.side(0).backend(), 0, 4);
    var b = Endpoint.init(pair.side(1).backend(), 1, 4);
    try a.sendBye();
    try b.sendBye();
    try a.awaitBye();
    try b.awaitBye();
    try std.testing.expect(a.peer_done and b.peer_done);
}

test "a corrupted slot is reported as a protocol error" {
    var pair = try shm.Pair.create(4);
    defer pair.deinit();
    var a = Endpoint.init(pair.side(0).backend(), 0, 4);
    var b = Endpoint.init(pair.side(1).backend(), 1, 4);

    var pkt = newPacket(1);
    try a.send(&pkt, .dispatch);
    pair.side(1).backend().vtable.slot(pair.side(1), 0).payload[10] ^= 0xFF;
    try std.testing.expectError(error.ProtocolError, b.tryRecv());
}
