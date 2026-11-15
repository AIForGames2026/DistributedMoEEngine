//! The generation loop behind `olmoe serve` (TASK-30): prefills a prompt
//! (reusing what the cache already holds of it), then samples tokens one at a
//! time and hands their text to a sink as it becomes final. Kept apart from
//! sockets and the model, so it is tested with a fake engine.
//!
//! Text handling: stop tokens end the reply unprinted; special tokens print
//! nothing; for Qwen3, text between <think> and </think> goes to the reasoning
//! channel; text that could still become a stop string is held back until it
//! cannot, and a stop string ends the reply before it.
const std = @import("std");
const olmoe = @import("olmoe.zig");
const tokenizer = @import("tokenizer.zig");
const openai = @import("openai.zig");

/// The model, behind its KV cache.
pub const Engine = struct {
    ctx: *anyopaque,
    vocab: usize,
    /// Positions the cache can hold.
    cap: usize,
    /// Drops cached positions from `len` on.
    truncate: *const fn (ctx: *anyopaque, len: usize) void,
    /// Runs `tokens` after the cached positions, adding them; `logits` gets the last one's row.
    step: *const fn (ctx: *anyopaque, tokens: []const u32, logits: []f32) anyerror!void,
};

pub const Sink = struct {
    ctx: *anyopaque,
    /// Takes a piece of reply text; an error (the client went away) stops generation.
    emit: *const fn (ctx: *anyopaque, channel: openai.Channel, text: []const u8) anyerror!void,
};

pub const Params = struct {
    max_tokens: ?usize = null,
    sampler: olmoe.Sampler = .{},
    stop: []const []const u8 = &.{},
    stop_ids: []const u32 = &.{},
    /// Qwen3's reasoning markers, which switch the channel and print nothing.
    think: ?struct { open: u32, close: u32 } = null,
    /// Prefill this many positions per engine call.
    chunk: usize = 2048,
    /// Start from the longest prefix the cache shares with the prompt.
    reuse: bool = true,
    /// Monotonic nanoseconds, for the timings.
    clock: *const fn () u64,
};

/// What the engine's cache holds, so the next request can reuse a shared prefix.
pub const Session = struct {
    tokens: std.ArrayList(u32) = .empty,

    pub fn deinit(self: *Session, gpa: std.mem.Allocator) void {
        self.tokens.deinit(gpa);
    }
};

pub const Result = struct {
    finish: openai.Finish,
    stats: openai.Stats,
};

/// Generates a reply to `prompt`, appending its token ids to `completion`.
pub fn run(gpa: std.mem.Allocator, engine: Engine, session: *Session, tok: *const tokenizer.Tokenizer, prompt: []const u32, p: Params, sink: Sink, completion: *std.ArrayList(u32)) !Result {
    if (prompt.len == 0 or prompt.len >= engine.cap)
    {
        return error.PromptTooLong;
    }
    const t0 = p.clock();
    const row = try gpa.alloc(f32, 2 * engine.vocab);
    defer gpa.free(row);
    const logits = row[0..engine.vocab];

    var common: usize = 0;
    if (p.reuse)
    {
        const held = session.tokens.items;
        while (common < held.len and common < prompt.len and held[common] == prompt[common])
        {
            common += 1;
        }
        // The last prompt position must run again to give its logits.
        common = @min(common, prompt.len - 1);
    }
    engine.truncate(engine.ctx, common);
    session.tokens.shrinkRetainingCapacity(common);
    var pos = common;
    while (pos < prompt.len)
    {
        const e = @min(pos + p.chunk, prompt.len);
        try engine.step(engine.ctx, prompt[pos..e], logits);
        try session.tokens.appendSlice(gpa, prompt[pos..e]);
        pos = e;
    }

    var stats: openai.Stats = .{ .prompt_tokens = prompt.len, .cached_tokens = common, .ttft_ns = p.clock() - t0 };
    var text: Text = .{ .sink = sink, .stop = p.stop };
    defer text.pending.deinit(gpa);
    var sampler = p.sampler;
    const budget = @min(p.max_tokens orelse std.math.maxInt(usize), engine.cap - prompt.len);
    var finish: openai.Finish = .length;
    var decode_ns: u64 = 0;
    while (true)
    {
        const id = sampler.pick(logits, row[engine.vocab..]);
        try completion.append(gpa, id);
        if (std.mem.indexOfScalar(u32, p.stop_ids, id) != null)
        {
            finish = .stop;
            break;
        }
        if (try text.take(gpa, tok, id, p.think))
        {
            finish = .stop;
            break;
        }
        if (completion.items.len >= budget)
        {
            break;
        }
        const s0 = p.clock();
        try engine.step(engine.ctx, &.{id}, logits);
        decode_ns += p.clock() - s0;
        stats.decode_steps += 1;
        try session.tokens.append(gpa, id);
    }
    if (!text.stopped)
    {
        try text.flush(gpa);
    }
    stats.completion_tokens = completion.items.len;
    stats.decode_ns = decode_ns;
    return .{ .finish = finish, .stats = stats };
}

/// Turns token ids into reply text on the right channel, holding back what a stop string may still claim.
const Text = struct {
    sink: Sink,
    stop: []const []const u8,
    stream: tokenizer.Stream = .{},
    channel: openai.Channel = .content,
    pending: std.ArrayList(u8) = .empty,
    /// Drop the line breaks that follow a reasoning marker.
    trim: bool = false,
    stopped: bool = false,

    /// Adds token `id`; true if a stop string ended the reply.
    fn take(self: *Text, gpa: std.mem.Allocator, tok: *const tokenizer.Tokenizer, id: u32, think: anytype) !bool {
        if (think) |t|
        {
            if (id == t.open or id == t.close)
            {
                try self.flush(gpa);
                self.channel = if (id == t.open) .reasoning else .content;
                self.trim = true;
                return false;
            }
        }
        if (tok.isSpecial(id))
        {
            return false;
        }
        const start = self.pending.items.len;
        try self.stream.push(gpa, tok, id, &self.pending);
        if (self.trim)
        {
            var i = start;
            while (i < self.pending.items.len and self.pending.items[i] == '\n')
            {
                i += 1;
            }
            try self.pending.replaceRange(gpa, start, i - start, "");
            self.trim = self.pending.items.len == start;
        }
        const buf = self.pending.items;
        var first: ?usize = null;
        for (self.stop) |s|
        {
            if (std.mem.indexOf(u8, buf, s)) |at|
            {
                first = if (first) |f| @min(f, at) else at;
            }
        }
        if (first) |at|
        {
            try self.send(gpa, at);
            self.stopped = true;
            return true;
        }
        var hold: usize = 0;
        for (self.stop) |s|
        {
            var n = @min(s.len - 1, buf.len);
            while (n > hold)
            {
                if (std.mem.endsWith(u8, buf, s[0..n]))
                {
                    hold = n;
                    break;
                }
                n -= 1;
            }
        }
        try self.send(gpa, buf.len - hold);
        return false;
    }

    /// Sends everything still held, including an unfinished character.
    fn flush(self: *Text, gpa: std.mem.Allocator) !void {
        try self.stream.finish(gpa, &self.pending);
        try self.send(gpa, self.pending.items.len);
    }

    fn send(self: *Text, gpa: std.mem.Allocator, n: usize) !void {
        if (n > 0)
        {
            try self.sink.emit(self.sink.ctx, self.channel, self.pending.items[0..n]);
        }
        try self.pending.replaceRange(gpa, 0, n, "");
    }
};

const Fake = struct {
    /// What the model "says" once it holds `base` positions (the prompt), one token per position after.
    script: []const u32,
    base: usize,
    len: usize = 0,
    steps: usize = 0,

    fn truncate(ctx: *anyopaque, len: usize) void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.len = len;
    }

    fn step(ctx: *anyopaque, tokens: []const u32, logits: []f32) anyerror!void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.len += tokens.len;
        self.steps += 1;
        @memset(logits, 0);
        const i = self.len -% self.base;
        logits[if (self.len >= self.base and i < self.script.len) self.script[i] else 'z'] = 1;
    }

    fn engine(self: *Fake) Engine {
        return .{ .ctx = self, .vocab = 300, .cap = 64, .truncate = truncate, .step = step };
    }
};

const Collect = struct {
    gpa: std.mem.Allocator,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    pieces: usize = 0,

    fn emit(ctx: *anyopaque, channel: openai.Channel, text: []const u8) anyerror!void {
        const self: *Collect = @ptrCast(@alignCast(ctx));
        self.pieces += 1;
        try (if (channel == .content) &self.content else &self.reasoning).appendSlice(self.gpa, text);
    }

    fn sink(self: *Collect) Sink {
        return .{ .ctx = self, .emit = emit };
    }

    fn deinit(self: *Collect) void {
        self.content.deinit(self.gpa);
        self.reasoning.deinit(self.gpa);
    }
};

var fake_now: u64 = 0;
fn fakeClock() u64 {
    fake_now += 10;
    return fake_now;
}

/// A byte-level tokenizer whose ids are the byte values, plus three added tokens.
fn testTokenizer(gpa: std.mem.Allocator) !tokenizer.Tokenizer {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.appendSlice(gpa,
        \\{"pre_tokenizer": {"type": "ByteLevel", "use_regex": true},
        \\ "added_tokens": [{"id": 256, "content": "<think>", "special": false},
        \\                  {"id": 257, "content": "</think>", "special": false},
        \\                  {"id": 258, "content": "<|end|>", "special": true},
        \\                  {"id": 259, "content": "<|other|>", "special": true}],
        \\ "model": {"type": "BPE", "merges": [], "vocab": {"<placeholder>": 299
    );
    for (0..256) |b|
    {
        // Rebuild GPT-2's alphabet: printable bytes are themselves, the rest count up from 256.
        var cp: u21 = @intCast(b);
        if (!((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF)))
        {
            var n: u21 = 0;
            for (0..b) |x|
            {
                if (!((x >= '!' and x <= '~') or (x >= 0xA1 and x <= 0xAC) or (x >= 0xAE and x <= 0xFF)))
                {
                    n += 1;
                }
            }
            cp = 256 + n;
        }
        var enc: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(cp, &enc);
        const esc = if (cp == '"' or cp == '\\') "\\" else "";
        const line = try std.fmt.allocPrint(gpa, ", \"{s}{s}\": {d}", .{ esc, enc[0..len], b });
        defer gpa.free(line);
        try json.appendSlice(gpa, line);
    }
    try json.appendSlice(gpa, "}}}");
    return tokenizer.Tokenizer.fromJson(gpa, json.items);
}

fn expectRun(script: []const u32, p: Params, want_content: []const u8, want_reasoning: []const u8, want_finish: openai.Finish, want_tokens: usize) !void {
    const gpa = std.testing.allocator;
    var tok = try testTokenizer(gpa);
    defer tok.deinit();
    const prompt = [_]u32{ 'h', 'i' };
    var fake: Fake = .{ .script = script, .base = prompt.len };
    var session: Session = .{};
    defer session.deinit(gpa);
    var got: Collect = .{ .gpa = gpa };
    defer got.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    const r = try run(gpa, fake.engine(), &session, &tok, &prompt, p, got.sink(), &ids);
    try std.testing.expectEqualStrings(want_content, got.content.items);
    try std.testing.expectEqualStrings(want_reasoning, got.reasoning.items);
    try std.testing.expectEqual(want_finish, r.finish);
    try std.testing.expectEqual(want_tokens, r.stats.completion_tokens);
}

test "splits reasoning from content and stops on a stop token" {
    const script = [_]u32{ 256, '\n', 'h', 'm', 257, '\n', '\n', 'H', 259, 'i', 258, 'x' };
    try expectRun(&script, .{ .stop_ids = &.{258}, .think = .{ .open = 256, .close = 257 }, .clock = fakeClock }, "Hi", "hm", .stop, 11);
}

test "stop strings end the reply before them, and a partial one is sent at the end" {
    const script = [_]u32{ 'a', 'b', 'S', 'T', 'O', 'P', 'c' };
    try expectRun(&script, .{ .stop = &.{"STOP"}, .clock = fakeClock }, "ab", "", .stop, 6);
    try expectRun(&script, .{ .stop = &.{"STOP"}, .max_tokens = 3, .clock = fakeClock }, "abS", "", .length, 3);
}

test "a second request reuses the shared prefix" {
    const gpa = std.testing.allocator;
    var tok = try testTokenizer(gpa);
    defer tok.deinit();
    const script = [_]u32{ 'o', 'k', 258 };
    var fake: Fake = .{ .script = &script, .base = 4 };
    var session: Session = .{};
    defer session.deinit(gpa);
    var got: Collect = .{ .gpa = gpa };
    defer got.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    const p: Params = .{ .stop_ids = &.{258}, .chunk = 2, .clock = fakeClock };
    const first = try run(gpa, fake.engine(), &session, &tok, &.{ 'a', 'b', 'c', 'd' }, p, got.sink(), &ids);
    try std.testing.expectEqual(@as(usize, 0), first.stats.cached_tokens);
    try std.testing.expectEqualSlices(u32, &.{ 'a', 'b', 'c', 'd', 'o', 'k' }, session.tokens.items);
    // The next turn repeats the conversation so far and adds to it.
    ids.clearRetainingCapacity();
    fake.base = 8;
    const second = try run(gpa, fake.engine(), &session, &tok, &.{ 'a', 'b', 'c', 'd', 'o', 'k', 258, 'e' }, p, got.sink(), &ids);
    try std.testing.expectEqual(@as(usize, 6), second.stats.cached_tokens);
    try std.testing.expectEqualStrings("okok", got.content.items);
    var cold: Session = .{};
    defer cold.deinit(gpa);
    const p_cold: Params = .{ .stop_ids = &.{258}, .reuse = false, .clock = fakeClock };
    ids.clearRetainingCapacity();
    const third = try run(gpa, fake.engine(), &session, &tok, &.{ 'a', 'b', 'c', 'd', 'o', 'k', 258, 'e' }, p_cold, got.sink(), &ids);
    try std.testing.expectEqual(@as(usize, 0), third.stats.cached_tokens);
}
