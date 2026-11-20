//! The HTTP server behind `olmoe ... --serve [IP:]PORT` (TASK-30), on Linux and Windows (TASK-40, `sock.zig`).
//!
//! A front thread accepts connections, reads each request, and answers
//! everything but chat completions itself. Chat requests pass through a pipe
//! to the calling thread, which owns the engine (its transport and GPU state)
//! and generates them one at a time; a few more may wait, and beyond that the
//! front thread answers 429. Each connection carries one request and is closed
//! after the answer.
//!
//!   POST /v1/chat/completions   OpenAI chat completions, streamed or not
//!   GET  /v1/models             the one model this server runs
//!   GET  /health                200 once the model is loaded
//!   GET  /                      the built-in chat page with the live diagram
//!   GET  /telemetry             its WebSocket feed (TASK-16b)
const std = @import("std");
const sock = @import("sock.zig");
const plat = @import("plat.zig");
const http = @import("http.zig");
const openai = @import("openai.zig");
const chat = @import("chat.zig");
const jinja = @import("jinja.zig");
const completion = @import("completion.zig");
const tokenizer = @import("tokenizer.zig");
const olmoe = @import("olmoe.zig");
const telemetry = @import("telemetry.zig");

pub const Options = struct {
    /// IPv4 address in network byte order; 127.0.0.1 unless asked otherwise.
    addr: u32,
    port: u16,
    model_id: []const u8,
    chunk: usize = 2048,
    reuse: bool = true,
    /// Chat requests that may wait while one generates.
    queue: usize = 4,
    clock: *const fn () u64,
    name: []const u8 = "serve",
    /// The built-in page and its live feed, if served.
    page: ?[]const u8 = null,
    live: ?*telemetry.Broadcaster = null,
};

const max_head = 16 << 10;
const max_body = 8 << 20;

/// A chat request handed from the front thread to the engine thread.
const Job = struct {
    fd: sock.Fd,
    arena: std.heap.ArenaAllocator,
    req: openai.Request,
};

const Shared = struct {
    opts: Options,
    listen_fd: sock.Fd,
    pipe: sock.Pipe,
    waiting: std.atomic.Value(usize) = .init(0),
};

/// Serves until the process is stopped or the engine fails.
pub fn run(gpa: std.mem.Allocator, engine: completion.Engine, tok: *const tokenizer.Tokenizer, prompts: *const chat.Chat, opts: Options) !void {
    const lfd = try sock.listenTcp(opts.addr, opts.port, 16);
    defer sock.close(lfd);
    const pipe = try sock.Pipe.open();
    defer sock.close(pipe.read_end);
    var shared: Shared = .{ .opts = opts, .listen_fd = lfd, .pipe = pipe };
    if (opts.live) |b|
    {
        try b.start();
    }
    const front = try std.Thread.spawn(.{}, frontLoop, .{&shared});
    front.detach();
    const ip: [4]u8 = @bitCast(opts.addr);
    std.debug.print("{s}: OpenAI-compatible API at http://{d}.{d}.{d}.{d}:{d}/v1 (model \"{s}\", prompts from {s}, context {d} tokens{s})\n", .{
        opts.name, ip[0], ip[1], ip[2], ip[3], opts.port, opts.model_id, prompts.describe(), engine.cap, if (opts.reuse) ", reusing shared prompt prefixes" else "",
    });
    if (opts.page != null)
    {
        std.debug.print("{s}: chat page with the live diagram at http://{d}.{d}.{d}.{d}:{d}/\n", .{ opts.name, ip[0], ip[1], ip[2], ip[3], opts.port });
    }

    var session: completion.Session = .{};
    defer session.deinit(gpa);
    var count: u64 = 0;
    while (true)
    {
        var ptr: usize = 0;
        if (!pipe.take(std.mem.asBytes(&ptr)))
        {
            return error.FrontStopped;
        }
        const job: *Job = @ptrFromInt(ptr);
        count += 1;
        const result = answer(gpa, engine, &session, tok, prompts, &shared, job, count);
        sock.close(job.fd);
        job.arena.deinit();
        std.heap.c_allocator.destroy(job);
        _ = shared.waiting.fetchSub(1, .acq_rel);
        result catch |err| {
            if (err == error.ClientGone)
            {
                continue;
            }
            std.debug.print("{s}: request {d} failed: {t}; stopping\n", .{ opts.name, count, err });
            return err;
        };
    }
}

fn frontLoop(shared: *Shared) void {
    while (true)
    {
        const fd = sock.accept(shared.listen_fd) orelse continue;
        // A slow or silent client gives up its turn instead of holding the server.
        sock.setTimeouts(fd, 10, 30);
        const handed = route(shared, fd) catch false;
        if (!handed)
        {
            sock.close(fd);
        }
    }
}

/// Reads and answers one request; true if it went to the engine thread, which then owns `fd`.
fn route(shared: *Shared, fd: sock.Fd) !bool {
    const gpa = std.heap.c_allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    var handed = false;
    defer if (!handed) arena.deinit();
    const a = arena.allocator();

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
                try sendError(fd, 413, "The request body is too large.", "invalid_request_error", null);
                return false;
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
            error.LengthRequired => {
                try sendError(fd, 411, "Send the body with a Content-Length.", "invalid_request_error", null);
                return false;
            },
            error.TooLarge => {
                try sendError(fd, 413, "The request head is too large.", "invalid_request_error", null);
                return false;
            },
            error.BadRequest => {
                try sendError(fd, 400, "Malformed HTTP request.", "invalid_request_error", null);
                return false;
            },
        }
        try buf.ensureUnusedCapacity(a, 64 << 10);
        const n = sock.recv(fd, buf.unusedCapacitySlice());
        if (n <= 0)
        {
            return false;
        }
        buf.items.len += @intCast(n);
    }
    const body = buf.items[head.len..][0..head.content_length];
    const opts = shared.opts;

    if (std.mem.eql(u8, head.path, "/") and opts.page != null)
    {
        try sendBody(fd, 200, "text/html; charset=utf-8", opts.page.?);
        return false;
    }
    if (std.mem.eql(u8, head.path, "/telemetry") and opts.live != null)
    {
        return upgrade(fd, head, opts.live.?);
    }
    if (std.mem.eql(u8, head.path, "/health"))
    {
        try sendBody(fd, 200, "application/json", "{\"status\":\"ok\"}");
        return false;
    }
    if (std.mem.eql(u8, head.path, "/v1/models"))
    {
        if (!std.mem.eql(u8, head.method, "GET"))
        {
            try sendError(fd, 405, "Use GET.", "invalid_request_error", null);
            return false;
        }
        var out: std.ArrayList(u8) = .empty;
        try openai.writeModels(a, &out, opts.model_id, unixTime());
        try sendBody(fd, 200, "application/json", out.items);
        return false;
    }
    if (!std.mem.eql(u8, head.path, "/v1/chat/completions"))
    {
        try sendError(fd, 404, "Not found. This server answers /v1/chat/completions and /v1/models.", "invalid_request_error", null);
        return false;
    }
    if (!std.mem.eql(u8, head.method, "POST"))
    {
        try sendError(fd, 405, "Use POST.", "invalid_request_error", null);
        return false;
    }
    var why: openai.Invalid = .{};
    const req = openai.parseChat(a, body, &why) catch |err| {
        if (err == error.Invalid)
        {
            try sendError(fd, 400, why.message, "invalid_request_error", why.param);
            return false;
        }
        return err;
    };
    if (shared.waiting.fetchAdd(1, .acq_rel) > opts.queue)
    {
        _ = shared.waiting.fetchSub(1, .acq_rel);
        try sendError(fd, 429, "The server is busy with other requests; try again shortly.", "rate_limit_error", null);
        return false;
    }
    const job = try gpa.create(Job);
    job.* = .{ .fd = fd, .arena = arena, .req = req };
    handed = true;
    const ptr: usize = @intFromPtr(job);
    if (!shared.pipe.put(std.mem.asBytes(&ptr)))
    {
        return error.EngineStopped;
    }
    return true;
}

/// Finishes the WebSocket handshake and gives the socket to the broadcaster.
fn upgrade(fd: sock.Fd, head: http.Head, live: *telemetry.Broadcaster) !bool {
    const key = head.websocket_key orelse {
        try sendError(fd, 400, "Connect with a WebSocket.", "invalid_request_error", null);
        return false;
    };
    // A page from another site must not read the feed.
    if (!http.sameOrigin(head))
    {
        try sendError(fd, 403, "The live feed is only for pages this server serves.", "invalid_request_error", null);
        return false;
    }
    var buf: [28]u8 = undefined;
    var reply: [160]u8 = undefined;
    try sendAll(fd, try std.fmt.bufPrint(&reply, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {s}\r\n\r\n", .{telemetry.acceptKey(key, &buf)}));
    return live.add(fd);
}

/// Generates one chat completion and writes it to the job's connection.
fn answer(gpa: std.mem.Allocator, engine: completion.Engine, session: *completion.Session, tok: *const tokenizer.Tokenizer, prompts: *const chat.Chat, shared: *Shared, job: *Job, count: u64) !void {
    const opts = shared.opts;
    const a = job.arena.allocator();
    const req = job.req;
    const fd = job.fd;

    var text: std.ArrayList(u8) = .empty;
    const chat_opts: chat.Options = .{ .thinking = req.thinking, .effort = req.effort };
    var diag: jinja.Diag = .{};
    prompts.render(a, req.messages, chat_opts, &diag, &text) catch |err| switch (err) {
        error.TemplateRefused => return sendError(fd, 400, try std.fmt.allocPrint(a, "The chat template refused these messages: {s}", .{diag.msg}), "invalid_request_error", "messages"),
        error.TemplateFailed => return sendError(fd, 400, try std.fmt.allocPrint(a, "The chat template failed on these messages: {s}", .{diag.msg}), "invalid_request_error", "messages"),
        error.SystemNotFirst => return sendError(fd, 400, "A system message must come first.", "invalid_request_error", "messages"),
        error.NoUserQuery => return sendError(fd, 400, "The messages need a user turn that is not a tool response.", "invalid_request_error", "messages"),
        error.UnsupportedEffort => return sendError(fd, 400, "This model does not take that reasoning_effort.", "invalid_request_error", "chat_template_kwargs"),
        else => return err,
    };
    var prompt: std.ArrayList(u32) = .empty;
    tok.encode(a, text.items, &prompt) catch |err| {
        if (err == error.InvalidUtf8)
        {
            return sendError(fd, 400, "Message text must be valid UTF-8.", "invalid_request_error", "messages");
        }
        return err;
    };
    if (prompt.items.len >= engine.cap)
    {
        const msg = try std.fmt.allocPrint(a, "The prompt is {d} tokens; this server's context holds {d} (start it with a larger --ctx).", .{ prompt.items.len, engine.cap });
        return sendError(fd, 400, msg, "invalid_request_error", "messages");
    }

    var stop_buf: [4]u32 = undefined;
    var sampler = olmoe.Sampler.seeded(req.temperature, req.top_k, req.seed orelse opts.clock());
    sampler.top_p = req.top_p;
    const think_open = tok.addedId("<think>");
    const think_close = tok.addedId("</think>");
    const p: completion.Params = .{
        .max_tokens = req.max_tokens,
        .sampler = sampler,
        .stop = req.stop,
        .stop_ids = prompts.stopIds(tok, &stop_buf),
        .think = if (prompts.thinks and think_open != null and think_close != null) .{ .open = think_open.?, .close = think_close.? } else null,
        .reasoning_first = chat.promptOpensReasoning(text.items),
        .harmony = completion.harmonyOf(tok),
        .chunk = opts.chunk,
        .reuse = opts.reuse,
        .clock = opts.clock,
    };
    const ids: openai.Ids = .{ .id = try std.fmt.allocPrint(a, "chatcmpl-tf{x}{d}", .{ opts.clock() & 0xffffff, count }), .created = unixTime(), .model = opts.model_id };

    var out: Out = .{ .fd = fd, .a = a, .ids = ids, .stream = req.stream, .live = opts.live };
    if (opts.live) |b|
    {
        var n: std.ArrayList(u8) = .empty;
        try n.print(a, "{{\"t\":\"chat\",\"id\":{d},\"messages\":[", .{count});
        for (req.messages, 0..) |m, i|
        {
            try n.print(a, "{s}{{\"role\":\"{t}\",\"content\":", .{ if (i == 0) "" else ",", m.role });
            try openai.writeString(a, &n, m.content);
            try n.append(a, '}');
        }
        try n.print(a, "],\"prompt_tokens\":{d},\"max_tokens\":{?d},\"temperature\":{d}}}", .{ prompt.items.len, req.max_tokens, req.temperature });
        b.note(n.items);
    }
    if (req.stream)
    {
        var head: std.ArrayList(u8) = .empty;
        try http.writeHead(a, &head, 200, "text/event-stream", null);
        try openai.writeChunk(a, &head, ids, .role);
        try out.send(head.items);
    }
    var generated: std.ArrayList(u32) = .empty;
    const t0 = opts.clock();
    const r = completion.run(gpa, engine, session, tok, prompt.items, p, .{ .ctx = &out, .emit = Out.emit }, &generated) catch |err| {
        if (err == error.ClientGone)
        {
            std.debug.print("{s}: request {d}: the client went away after {d} tokens\n", .{ opts.name, count, generated.items.len });
            if (opts.live) |b|
            {
                const n = try std.fmt.allocPrint(a, "{{\"t\":\"done\",\"id\":{d},\"finish\":\"client_gone\",\"completion_tokens\":{d}}}", .{ count, generated.items.len });
                b.note(n);
            }
            return err;
        }
        if (!req.stream)
        {
            sendError(fd, 500, "The engine failed while generating.", "server_error", null) catch {};
        }
        return err;
    };
    var stats = r.stats;
    if (req.return_tokens)
    {
        stats.prompt_ids = prompt.items;
        stats.completion_ids = generated.items;
    }
    var tail: std.ArrayList(u8) = .empty;
    if (req.stream)
    {
        try openai.writeChunk(a, &tail, ids, .{ .finish = r.finish });
        try openai.writeUsageChunk(a, &tail, ids, stats, req.include_usage);
        out.send(tail.items) catch {};
    }
    else
    {
        var body: std.ArrayList(u8) = .empty;
        try openai.writeCompletion(a, &body, ids, out.content.items, out.reasoning.items, r.finish, stats);
        try sendBody(fd, 200, "application/json", body.items);
    }
    const decode_rate = if (stats.decode_ns > 0) @as(f64, @floatFromInt(stats.decode_steps)) / (@as(f64, @floatFromInt(stats.decode_ns)) / 1e9) else 0;
    if (opts.live) |b|
    {
        const n = try std.fmt.allocPrint(a, "{{\"t\":\"done\",\"id\":{d},\"finish\":\"{t}\",\"prompt_tokens\":{d},\"cached_tokens\":{d},\"completion_tokens\":{d},\"ttft_ms\":{d:.1},\"decode_tokens_per_s\":{d:.2}}}", .{
            count, r.finish, stats.prompt_tokens, stats.cached_tokens, stats.completion_tokens, @as(f64, @floatFromInt(stats.ttft_ns)) / 1e6, decode_rate,
        });
        b.note(n);
    }
    const secs = @as(f64, @floatFromInt(opts.clock() - t0)) / 1e9;
    std.debug.print("{s}: request {d}: {d} prompt tokens ({d} reused), {d} generated in {d:.2} s, first token {d:.0} ms, {d:.1} tokens/s decoding, finish {t}\n", .{
        opts.name,
        count,
        stats.prompt_tokens,
        stats.cached_tokens,
        stats.completion_tokens,
        secs,
        @as(f64, @floatFromInt(stats.ttft_ns)) / 1e6,
        decode_rate,
        r.finish,
    });
}

/// Where reply text goes: straight to the socket as events when streaming, else collected.
const Out = struct {
    fd: sock.Fd,
    a: std.mem.Allocator,
    ids: openai.Ids,
    stream: bool,
    /// Where reply text is also noted when recording.
    live: ?*telemetry.Broadcaster = null,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    scratch: std.ArrayList(u8) = .empty,

    fn emit(ctx: *anyopaque, channel: openai.Channel, text: []const u8) anyerror!void {
        const self: *Out = @ptrCast(@alignCast(ctx));
        if (self.live) |b|
        {
            if (b.recording)
            {
                self.scratch.clearRetainingCapacity();
                try self.scratch.print(self.a, "{{\"t\":\"text\",\"channel\":\"{t}\",\"s\":", .{channel});
                try openai.writeString(self.a, &self.scratch, text);
                try self.scratch.append(self.a, '}');
                b.note(self.scratch.items);
            }
        }
        if (!self.stream)
        {
            try (if (channel == .content) &self.content else &self.reasoning).appendSlice(self.a, text);
            return;
        }
        self.scratch.clearRetainingCapacity();
        try openai.writeChunk(self.a, &self.scratch, self.ids, .{ .text = .{ .channel = channel, .text = text } });
        try self.send(self.scratch.items);
    }

    fn send(self: *Out, bytes: []const u8) !void {
        sendAll(self.fd, bytes) catch return error.ClientGone;
    }
};

fn unixTime() i64 {
    return plat.unixNow();
}

fn sendAll(fd: sock.Fd, bytes: []const u8) !void {
    return sock.sendAll(fd, bytes);
}

fn sendBody(fd: sock.Fd, status: u16, content_type: []const u8, body: []const u8) !void {
    const gpa = std.heap.c_allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try http.writeHead(gpa, &out, status, content_type, body.len);
    try out.appendSlice(gpa, body);
    try sendAll(fd, out.items);
}

fn sendError(fd: sock.Fd, status: u16, message: []const u8, kind: []const u8, param: ?[]const u8) !void {
    const gpa = std.heap.c_allocator;
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    try openai.writeError(gpa, &body, message, kind, param);
    try sendBody(fd, status, "application/json", body.items);
}
