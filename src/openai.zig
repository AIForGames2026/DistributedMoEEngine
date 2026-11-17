//! OpenAI chat completions format for `olmoe serve` (TASK-30): reads a
//! `POST /v1/chat/completions` body and writes responses, stream chunks,
//! model lists, and errors, so `curl`, the `openai` package, and existing chat
//! clients work unchanged.
//!
//! Beyond OpenAI's fields it accepts `top_k`, vLLM's
//! `chat_template_kwargs.enable_thinking` (and Qwen3.8's `reasoning_effort`) and assistant `reasoning_content`,
//! and `return_tokens`, which adds the prompt and completion token ids to the
//! `tensorfold` object every response carries (with timings and reuse).
const std = @import("std");
const chat = @import("chat.zig");

pub const Request = struct {
    model: []const u8 = "",
    messages: []const chat.Message,
    stream: bool = false,
    include_usage: bool = false,
    max_tokens: ?usize = null,
    temperature: f32 = 1,
    top_p: f32 = 1,
    top_k: usize = 0,
    seed: ?u64 = null,
    stop: []const []const u8 = &.{},
    thinking: ?bool = null,
    effort: chat.Effort = .xhigh,
    return_tokens: bool = false,
};

/// Why a request was rejected, for the error response.
pub const Invalid = struct {
    message: []const u8 = "",
    param: ?[]const u8 = null,
};

/// Parses a request body; strings point into memory from `a`, which should be an arena.
/// On `error.Invalid`, `why` says what was wrong.
pub fn parseChat(a: std.mem.Allocator, body: []const u8, why: *Invalid) !Request {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch |e| {
        if (e == error.OutOfMemory)
        {
            return error.OutOfMemory;
        }
        why.* = .{ .message = "The body is not valid JSON." };
        return error.Invalid;
    };
    const o = switch (root) {
        .object => |o| o,
        else => return invalid(why, "The body must be a JSON object.", null),
    };

    var req: Request = .{ .messages = &.{} };
    if (o.get("model")) |v|
    {
        req.model = switch (v) {
            .string => |s| s,
            else => return invalid(why, "'model' must be a string.", "model"),
        };
    }
    const msgs = switch (o.get("messages") orelse return invalid(why, "'messages' is required.", "messages")) {
        .array => |arr| arr.items,
        else => return invalid(why, "'messages' must be an array.", "messages"),
    };
    if (msgs.len == 0)
    {
        return invalid(why, "'messages' must not be empty.", "messages");
    }
    const out = try a.alloc(chat.Message, msgs.len);
    for (msgs, out) |m, *dst|
    {
        dst.* = try parseMessage(a, m, why);
    }
    req.messages = out;

    req.stream = try optBool(o, "stream", why) orelse false;
    if (o.get("stream_options")) |so|
    {
        if (so == .object)
        {
            req.include_usage = try optBool(so.object, "include_usage", why) orelse false;
        }
    }
    req.max_tokens = try optUint(o, "max_completion_tokens", why) orelse try optUint(o, "max_tokens", why);
    if (req.max_tokens) |m|
    {
        if (m == 0)
        {
            return invalid(why, "'max_tokens' must be at least 1.", "max_tokens");
        }
    }
    if (try optFloat(o, "temperature", why)) |t|
    {
        if (t < 0 or t > 2)
        {
            return invalid(why, "'temperature' must be between 0 and 2.", "temperature");
        }
        req.temperature = t;
    }
    if (try optFloat(o, "top_p", why)) |p|
    {
        if (p <= 0 or p > 1)
        {
            return invalid(why, "'top_p' must be above 0 and at most 1.", "top_p");
        }
        req.top_p = p;
    }
    req.top_k = try optUint(o, "top_k", why) orelse 0;
    if (try optUint(o, "seed", why)) |s|
    {
        req.seed = s;
    }
    if (try optUint(o, "n", why)) |n|
    {
        if (n != 1)
        {
            return invalid(why, "Only one choice per request is supported ('n' must be 1).", "n");
        }
    }
    if (o.get("tools")) |t|
    {
        if (t != .null and !(t == .array and t.array.items.len == 0))
        {
            return invalid(why, "Tool calling is not supported.", "tools");
        }
    }
    req.stop = try parseStop(a, o.get("stop"), why);
    if (o.get("chat_template_kwargs")) |k|
    {
        if (k == .object)
        {
            req.thinking = try optBool(k.object, "enable_thinking", why);
            if (k.object.get("reasoning_effort")) |e|
            {
                const name = if (e == .string) e.string else "";
                req.effort = std.meta.stringToEnum(chat.Effort, name) orelse
                    return invalid(why, "'reasoning_effort' must be \"xhigh\", \"high\", \"medium\", \"low\" or \"max\".", "reasoning_effort");
            }
        }
    }
    req.return_tokens = try optBool(o, "return_tokens", why) orelse false;
    return req;
}

fn invalid(why: *Invalid, message: []const u8, param: ?[]const u8) error{Invalid} {
    why.* = .{ .message = message, .param = param };
    return error.Invalid;
}

fn parseMessage(a: std.mem.Allocator, v: std.json.Value, why: *Invalid) !chat.Message {
    const o = switch (v) {
        .object => |o| o,
        else => return invalid(why, "Each message must be an object.", "messages"),
    };
    const role_text = switch (o.get("role") orelse return invalid(why, "Each message needs a 'role'.", "messages")) {
        .string => |s| s,
        else => return invalid(why, "A message 'role' must be a string.", "messages"),
    };
    const role: chat.Role = if (std.mem.eql(u8, role_text, "system") or std.mem.eql(u8, role_text, "developer"))
        .system
    else if (std.mem.eql(u8, role_text, "user"))
        .user
    else if (std.mem.eql(u8, role_text, "assistant"))
        .assistant
    else
        return invalid(why, "Only system, developer, user, and assistant messages are supported.", "messages");
    if (o.get("tool_calls")) |t|
    {
        if (t != .null and !(t == .array and t.array.items.len == 0))
        {
            return invalid(why, "Tool calls are not supported.", "messages");
        }
    }
    var msg: chat.Message = .{ .role = role, .content = "" };
    switch (o.get("content") orelse .null)
    {
        .string => |s| msg.content = s,
        .null => {},
        .array => |parts| {
            // Text parts are joined, as OpenAI does.
            var text: std.ArrayList(u8) = .empty;
            for (parts.items) |p|
            {
                const po = if (p == .object) p.object else return invalid(why, "Content parts must be objects.", "messages");
                const kind = po.get("type") orelse return invalid(why, "Content parts need a 'type'.", "messages");
                if (kind != .string or !std.mem.eql(u8, kind.string, "text"))
                {
                    return invalid(why, "Only text content parts are supported.", "messages");
                }
                const t = po.get("text") orelse return invalid(why, "Text parts need 'text'.", "messages");
                if (t != .string)
                {
                    return invalid(why, "'text' must be a string.", "messages");
                }
                try text.appendSlice(a, t.string);
            }
            msg.content = text.items;
        },
        else => return invalid(why, "Message 'content' must be a string or an array of text parts.", "messages"),
    }
    if (o.get("reasoning_content")) |r|
    {
        if (r == .string)
        {
            msg.reasoning = r.string;
        }
    }
    return msg;
}

fn parseStop(a: std.mem.Allocator, v: ?std.json.Value, why: *Invalid) ![]const []const u8 {
    const value = v orelse return &.{};
    switch (value)
    {
        .null => return &.{},
        .string => |s| {
            if (s.len == 0)
            {
                return &.{};
            }
            const one = try a.alloc([]const u8, 1);
            one[0] = s;
            return one;
        },
        .array => |arr| {
            if (arr.items.len > 4)
            {
                return invalid(why, "At most 4 stop sequences are allowed.", "stop");
            }
            var list: std.ArrayList([]const u8) = .empty;
            for (arr.items) |s|
            {
                if (s != .string)
                {
                    return invalid(why, "Stop sequences must be strings.", "stop");
                }
                if (s.string.len > 0)
                {
                    try list.append(a, s.string);
                }
            }
            return list.items;
        },
        else => return invalid(why, "'stop' must be a string or an array of strings.", "stop"),
    }
}

fn optBool(o: std.json.ObjectMap, name: []const u8, why: *Invalid) !?bool {
    return switch (o.get(name) orelse return null) {
        .null => null,
        .bool => |b| b,
        else => invalid(why, "Expected true or false.", name),
    };
}

fn optUint(o: std.json.ObjectMap, name: []const u8, why: *Invalid) !?usize {
    return switch (o.get(name) orelse return null) {
        .null => null,
        .integer => |i| if (i >= 0) @intCast(i) else invalid(why, "Expected a non-negative integer.", name),
        else => invalid(why, "Expected an integer.", name),
    };
}

fn optFloat(o: std.json.ObjectMap, name: []const u8, why: *Invalid) !?f32 {
    return switch (o.get(name) orelse return null) {
        .null => null,
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => invalid(why, "Expected a number.", name),
    };
}

/// What a finished (or streaming) completion reports beyond its text.
pub const Stats = struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    /// Prompt positions reused from the previous request's cache.
    cached_tokens: usize = 0,
    ttft_ns: u64 = 0,
    decode_ns: u64 = 0,
    decode_steps: usize = 0,
    prompt_ids: ?[]const u32 = null,
    completion_ids: ?[]const u32 = null,
};

pub const Finish = enum { stop, length };

/// The fields every response and chunk starts with.
pub const Ids = struct {
    id: []const u8,
    created: i64,
    model: []const u8,
};

pub const Channel = enum { content, reasoning };

/// Appends `s` as a JSON string.
pub fn writeString(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(gpa, '"');
    for (s) |ch|
    {
        switch (ch)
        {
            '"' => try out.appendSlice(gpa, "\\\""),
            '\\' => try out.appendSlice(gpa, "\\\\"),
            '\n' => try out.appendSlice(gpa, "\\n"),
            '\r' => try out.appendSlice(gpa, "\\r"),
            '\t' => try out.appendSlice(gpa, "\\t"),
            0...0x08, 0x0B, 0x0C, 0x0E...0x1F, 0x7F => try out.print(gpa, "\\u{x:0>4}", .{ch}),
            else => try out.append(gpa, ch),
        }
    }
    try out.append(gpa, '"');
}

fn writeIds(gpa: std.mem.Allocator, out: *std.ArrayList(u8), ids: Ids, object: []const u8) !void {
    try out.appendSlice(gpa, "{\"id\":");
    try writeString(gpa, out, ids.id);
    try out.print(gpa, ",\"object\":\"{s}\",\"created\":{d},\"model\":", .{ object, ids.created });
    try writeString(gpa, out, ids.model);
}

fn writeUsage(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: Stats) !void {
    try out.print(gpa, ",\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d},\"prompt_tokens_details\":{{\"cached_tokens\":{d}}}}}", .{
        s.prompt_tokens, s.completion_tokens, s.prompt_tokens + s.completion_tokens, s.cached_tokens,
    });
}

fn writeTensorfold(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: Stats) !void {
    const decode_rate = if (s.decode_ns > 0) @as(f64, @floatFromInt(s.decode_steps)) / (@as(f64, @floatFromInt(s.decode_ns)) / 1e9) else 0;
    try out.print(gpa, ",\"tensorfold\":{{\"cached_tokens\":{d},\"ttft_ms\":{d:.1},\"decode_tokens_per_s\":{d:.2}", .{
        s.cached_tokens, @as(f64, @floatFromInt(s.ttft_ns)) / 1e6, decode_rate,
    });
    inline for (.{ .{ "prompt_ids", s.prompt_ids }, .{ "completion_ids", s.completion_ids } }) |f|
    {
        if (f[1]) |list|
        {
            try out.print(gpa, ",\"{s}\":[", .{f[0]});
            for (list, 0..) |id, i|
            {
                try out.print(gpa, "{s}{d}", .{ if (i == 0) "" else ",", id });
            }
            try out.append(gpa, ']');
        }
    }
    try out.append(gpa, '}');
}

/// A whole (non-streamed) completion.
pub fn writeCompletion(gpa: std.mem.Allocator, out: *std.ArrayList(u8), ids: Ids, content: []const u8, reasoning: []const u8, finish: Finish, s: Stats) !void {
    try writeIds(gpa, out, ids, "chat.completion");
    try out.appendSlice(gpa, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    try writeString(gpa, out, content);
    if (reasoning.len > 0)
    {
        try out.appendSlice(gpa, ",\"reasoning_content\":");
        try writeString(gpa, out, reasoning);
    }
    try out.print(gpa, "}},\"logprobs\":null,\"finish_reason\":\"{t}\"}}]", .{finish});
    try writeUsage(gpa, out, s);
    try writeTensorfold(gpa, out, s);
    try out.append(gpa, '}');
}

/// A server-sent event carrying one chunk: the opening role, a piece of text, or the finish.
pub fn writeChunk(gpa: std.mem.Allocator, out: *std.ArrayList(u8), ids: Ids, delta: union(enum) { role, text: struct { channel: Channel, text: []const u8 }, finish: Finish }) !void {
    try out.appendSlice(gpa, "data: ");
    try writeIds(gpa, out, ids, "chat.completion.chunk");
    try out.appendSlice(gpa, ",\"choices\":[{\"index\":0,\"delta\":");
    switch (delta)
    {
        .role => try out.appendSlice(gpa, "{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null"),
        .text => |t| {
            try out.print(gpa, "{{\"{s}\":", .{if (t.channel == .content) "content" else "reasoning_content"});
            try writeString(gpa, out, t.text);
            try out.appendSlice(gpa, "},\"logprobs\":null,\"finish_reason\":null");
        },
        .finish => |f| try out.print(gpa, "{{}},\"logprobs\":null,\"finish_reason\":\"{t}\"", .{f}),
    }
    try out.appendSlice(gpa, "}]}\n\n");
}

/// The last chunk before `[DONE]`: usage (if asked for) and the tensorfold object.
pub fn writeUsageChunk(gpa: std.mem.Allocator, out: *std.ArrayList(u8), ids: Ids, s: Stats, include_usage: bool) !void {
    try out.appendSlice(gpa, "data: ");
    try writeIds(gpa, out, ids, "chat.completion.chunk");
    try out.appendSlice(gpa, ",\"choices\":[]");
    if (include_usage)
    {
        try writeUsage(gpa, out, s);
    }
    try writeTensorfold(gpa, out, s);
    try out.appendSlice(gpa, "}\n\ndata: [DONE]\n\n");
}

pub fn writeModels(gpa: std.mem.Allocator, out: *std.ArrayList(u8), model: []const u8, created: i64) !void {
    try out.appendSlice(gpa, "{\"object\":\"list\",\"data\":[{\"id\":");
    try writeString(gpa, out, model);
    try out.print(gpa, ",\"object\":\"model\",\"created\":{d},\"owned_by\":\"tensorfold\"}}]}}", .{created});
}

pub fn writeError(gpa: std.mem.Allocator, out: *std.ArrayList(u8), message: []const u8, kind: []const u8, param: ?[]const u8) !void {
    try out.appendSlice(gpa, "{\"error\":{\"message\":");
    try writeString(gpa, out, message);
    try out.print(gpa, ",\"type\":\"{s}\",\"param\":", .{kind});
    if (param) |p|
    {
        try writeString(gpa, out, p);
    }
    else
    {
        try out.appendSlice(gpa, "null");
    }
    try out.appendSlice(gpa, ",\"code\":null}}");
}

test "parses a chat request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var why: Invalid = .{};
    const req = try parseChat(arena.allocator(),
        \\{"model": "m", "stream": true, "stream_options": {"include_usage": true}, "temperature": 0,
        \\ "max_tokens": 16, "stop": "END", "seed": 3, "chat_template_kwargs": {"enable_thinking": false},
        \\ "messages": [{"role": "developer", "content": "Be brief."},
        \\              {"role": "user", "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]},
        \\              {"role": "assistant", "content": "Hello", "reasoning_content": "r"}]}
    , &why);
    try std.testing.expect(req.stream and req.include_usage and req.thinking == false);
    try std.testing.expectEqual(@as(f32, 0), req.temperature);
    try std.testing.expectEqual(@as(?usize, 16), req.max_tokens);
    try std.testing.expectEqualStrings("END", req.stop[0]);
    try std.testing.expectEqual(chat.Role.system, req.messages[0].role);
    try std.testing.expectEqualStrings("Hi there", req.messages[1].content);
    try std.testing.expectEqualStrings("r", req.messages[2].reasoning.?);
}

test "rejects what it cannot serve" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const bad = [_][]const u8{
        "not json",
        "{\"messages\": []}",
        "{\"messages\": [{\"role\": \"tool\", \"content\": \"x\"}]}",
        "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"n\": 2}",
        "{\"messages\": [{\"role\": \"user\", \"content\": \"x\"}], \"temperature\": 3}",
        "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\"}]}]}",
    };
    for (bad) |body|
    {
        var why: Invalid = .{};
        try std.testing.expectError(error.Invalid, parseChat(arena.allocator(), body, &why));
        try std.testing.expect(why.message.len > 0);
    }
}

test "writes chunks that parse as JSON" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const ids: Ids = .{ .id = "chatcmpl-1", .created = 5, .model = "m" };
    try writeChunk(gpa, &out, ids, .{ .text = .{ .channel = .reasoning, .text = "a\"b\n\x01" } });
    try std.testing.expect(std.mem.startsWith(u8, out.items, "data: ") and std.mem.endsWith(u8, out.items, "\n\n"));
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, out.items[6 .. out.items.len - 2], .{});
    defer parsed.deinit();
    const delta = parsed.value.object.get("choices").?.array.items[0].object.get("delta").?.object;
    try std.testing.expectEqualStrings("a\"b\n\x01", delta.get("reasoning_content").?.string);

    out.clearRetainingCapacity();
    try writeCompletion(gpa, &out, ids, "hi", "", .stop, .{ .prompt_tokens = 3, .completion_tokens = 2, .prompt_ids = &.{ 1, 2, 3 } });
    const whole = try std.json.parseFromSlice(std.json.Value, gpa, out.items, .{});
    defer whole.deinit();
    try std.testing.expectEqual(@as(i64, 5), whole.value.object.get("usage").?.object.get("total_tokens").?.integer);
    try std.testing.expectEqual(@as(usize, 3), whole.value.object.get("tensorfold").?.object.get("prompt_ids").?.array.items.len);
}
