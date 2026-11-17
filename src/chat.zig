//! Chat prompt formatting (TASK-30): renders a conversation the way each
//! model's Jinja chat template does in transformers' `apply_chat_template`,
//! for text conversations (system, user, and assistant turns; no tools).
//! The templates are reimplemented rather than interpreted; `detect` picks one
//! from the template text the model ships, and refuses templates it does not know.
//!
//!   tulu (OLMoE-1B-7B-Instruct): <|endoftext|><|user|>\nHi\n<|assistant|>\n
//!   qwen3 (Qwen3-MoE): <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
//!   qwen3_release: the template Qwen3 first shipped with, still in older GGUF
//!     files; it spots tool responses by containment instead of by prefix and suffix
//!   qwen38 (Qwen3.8-Flash-Next, TASK-31): Qwen3's turns with trimmed text, a
//!     reasoning-effort system note, every turn's reasoning kept, and a prompt
//!     that opens the reply's <think>
const std = @import("std");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const Format = enum { tulu, qwen3, qwen3_release, qwen38 };

/// Qwen3.8's `reasoning_effort`; xhigh and low add a note to the system turn, medium adds none.
pub const Effort = enum { xhigh, medium, low };

pub const Role = enum { system, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
    /// Qwen3 only: an assistant turn's reasoning, if the client sent it apart from the content.
    reasoning: ?[]const u8 = null,
};

pub const Options = struct {
    /// End with the start of an assistant turn, for generating a reply.
    add_generation_prompt: bool = true,
    /// Qwen3's `enable_thinking`: false asks for a reply without reasoning; null leaves it unset.
    thinking: ?bool = null,
    /// Qwen3.8 only, and only while thinking is on.
    effort: Effort = .xhigh,
};

/// Whether the prompt itself opens the reply's reasoning, so generation starts inside it.
pub fn opensReasoning(format: Format, opts: Options) bool
{
    return format == .qwen38 and opts.add_generation_prompt and opts.thinking != false;
}

/// The format matching a model's chat template, or null for one we do not reimplement.
pub fn detect(tok: *const Tokenizer) ?Format {
    const t = tok.chat_template orelse return null;
    const has = struct {
        fn f(s: []const u8, part: []const u8) bool {
            return std.mem.indexOf(u8, s, part) != null;
        }
    }.f;
    // OLMoE's template has real line breaks inside its strings; Qwen3's has the escape.
    if (tok.pattern == .gpt2 and has(t, "'<|user|>\n' + message['content']") and has(t, "'<|assistant|>\n'  + message['content'] + eos_token"))
    {
        return .tulu;
    }
    if (tok.pattern == .qwen2 and has(t, "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>'") and has(t, "enable_thinking is false"))
    {
        return .qwen3;
    }
    if (tok.pattern == .qwen2 and has(t, "'<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>'") and has(t, "enable_thinking is false") and
        has(t, "not('<tool_response>' in message.content and '</tool_response>' in message.content)"))
    {
        return .qwen3_release;
    }
    if (tok.pattern == .qwen35 and has(t, "reasoning_effort|default('xhigh')") and has(t, "preserve_thinking is undefined or preserve_thinking is true") and
        has(t, "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>'"))
    {
        return .qwen38;
    }
    return null;
}

/// Token ids that end an assistant reply.
pub fn stopIds(format: Format, tok: *const Tokenizer, buf: *[2]u32) []const u32 {
    var n: usize = 0;
    const names: []const []const u8 = switch (format) {
        .tulu => &.{"<|endoftext|>"},
        .qwen3, .qwen3_release, .qwen38 => &.{ "<|im_end|>", "<|endoftext|>" },
    };
    for (names) |name|
    {
        if (tok.addedId(name)) |id|
        {
            buf[n] = id;
            n += 1;
        }
    }
    return buf[0..n];
}

/// Appends the prompt text for `messages` to `out`; tokenize it with `Tokenizer.encode`.
pub fn render(gpa: std.mem.Allocator, format: Format, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) !void {
    if (messages.len == 0)
    {
        return error.NoMessages;
    }
    switch (format)
    {
        .tulu => try renderTulu(gpa, messages, opts, out),
        .qwen3 => try renderQwen3(gpa, messages, opts, false, out),
        .qwen3_release => try renderQwen3(gpa, messages, opts, true, out),
        .qwen38 => try renderQwen38(gpa, messages, opts, out),
    }
}

/// Refuses what Qwen3.8's template raises an exception for: a system turn after the first, or no user query.
fn renderQwen38(gpa: std.mem.Allocator, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) !void {
    const note: []const u8 = if (opts.thinking == false) "" else switch (opts.effort) {
        .xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        .medium => "",
        .low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
    };
    const system = if (messages[0].role == .system) pyTrim(messages[0].content) else "";
    if (system.len > 0)
    {
        try appendAll(gpa, out, &.{ "<|im_start|>system\n", note, if (note.len > 0) "\n\n" else "", system, "<|im_end|>\n" });
    }
    else if (note.len > 0)
    {
        try appendAll(gpa, out, &.{ "<|im_start|>system\n", note, "<|im_end|>\n" });
    }
    var query = false;
    for (messages) |m|
    {
        const c = pyTrim(m.content);
        if (m.role == .user and !(std.mem.startsWith(u8, c, "<tool_response>") and std.mem.endsWith(u8, c, "</tool_response>")))
        {
            query = true;
        }
    }
    if (!query)
    {
        return error.NoUserQuery;
    }
    for (messages, 0..) |m, idx|
    {
        const content = pyTrim(m.content);
        switch (m.role)
        {
            .system => {
                if (idx > 0)
                {
                    return error.SystemNotFirst;
                }
            },
            .user => try appendAll(gpa, out, &.{ "<|im_start|>user\n", content, "<|im_end|>\n" }),
            .assistant => try appendAll(gpa, out, &.{ "<|im_start|>assistant\n<think>\n", pyTrim(m.reasoning orelse ""), "\n</think>\n\n", content, "<|im_end|>\n" }),
        }
    }
    if (opts.add_generation_prompt)
    {
        try out.appendSlice(gpa, if (opts.thinking == false) "<|im_start|>assistant\n<think>\n\n</think>\n\n" else "<|im_start|>assistant\n<think>\n");
    }
}

/// Python's `str.strip()`, which Jinja's `trim` filter calls: Unicode whitespace off both ends.
fn pyTrim(s: []const u8) []const u8 {
    var a: usize = 0;
    var b = s.len;
    while (a < b)
    {
        const n = std.unicode.utf8ByteSequenceLength(s[a]) catch break;
        if (a + n > b or !pySpace(std.unicode.utf8Decode(s[a..][0..n]) catch break))
        {
            break;
        }
        a += n;
    }
    while (b > a)
    {
        var start = b - 1;
        while (start > a and s[start] & 0xC0 == 0x80)
        {
            start -= 1;
        }
        if (!pySpace(std.unicode.utf8Decode(s[start..b]) catch break))
        {
            break;
        }
        b = start;
    }
    return s[a..b];
}

fn pySpace(c: u21) bool {
    return switch (c) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

fn renderTulu(gpa: std.mem.Allocator, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) !void {
    try out.appendSlice(gpa, "<|endoftext|>");
    for (messages, 0..) |m, i|
    {
        switch (m.role)
        {
            .system => try appendAll(gpa, out, &.{ "<|system|>\n", m.content, "\n" }),
            .user => try appendAll(gpa, out, &.{ "<|user|>\n", m.content, "\n" }),
            .assistant => try appendAll(gpa, out, &.{ "<|assistant|>\n", m.content, "<|endoftext|>\n" }),
        }
        if (i + 1 == messages.len and opts.add_generation_prompt)
        {
            try out.appendSlice(gpa, "<|assistant|>\n");
        }
    }
}

fn renderQwen3(gpa: std.mem.Allocator, messages: []const Message, opts: Options, by_containment: bool, out: *std.ArrayList(u8)) !void {
    if (messages[0].role == .system)
    {
        try appendAll(gpa, out, &.{ "<|im_start|>system\n", messages[0].content, "<|im_end|>\n" });
    }
    // The last user turn that is not a tool response; reasoning before it is dropped.
    var last_query = messages.len - 1;
    var i = messages.len;
    while (i > 0)
    {
        i -= 1;
        const c = messages[i].content;
        const tool_response = if (by_containment)
            std.mem.indexOf(u8, c, "<tool_response>") != null and std.mem.indexOf(u8, c, "</tool_response>") != null
        else
            std.mem.startsWith(u8, c, "<tool_response>") and std.mem.endsWith(u8, c, "</tool_response>");
        if (messages[i].role == .user and !tool_response)
        {
            last_query = i;
            break;
        }
    }
    for (messages, 0..) |m, idx|
    {
        switch (m.role)
        {
            .system, .user => {
                if (m.role == .user or idx > 0)
                {
                    try appendAll(gpa, out, &.{ "<|im_start|>", @tagName(m.role), "\n", m.content, "<|im_end|>\n" });
                }
            },
            .assistant => {
                var content = m.content;
                var reasoning: []const u8 = "";
                if (m.reasoning) |r|
                {
                    reasoning = r;
                }
                else if (std.mem.indexOf(u8, content, "</think>")) |first_end|
                {
                    const before = std.mem.trimEnd(u8, content[0..first_end], "\n");
                    const open = std.mem.lastIndexOf(u8, before, "<think>");
                    reasoning = std.mem.trimStart(u8, if (open) |o| before[o + "<think>".len ..] else before, "\n");
                    const last_end = std.mem.lastIndexOf(u8, content, "</think>").?;
                    content = std.mem.trimStart(u8, content[last_end + "</think>".len ..], "\n");
                }
                if (idx > last_query and (idx + 1 == messages.len or reasoning.len > 0))
                {
                    try appendAll(gpa, out, &.{ "<|im_start|>assistant\n<think>\n", std.mem.trim(u8, reasoning, "\n"), "\n</think>\n\n", std.mem.trimStart(u8, content, "\n") });
                }
                else
                {
                    try appendAll(gpa, out, &.{ "<|im_start|>assistant\n", content });
                }
                try out.appendSlice(gpa, "<|im_end|>\n");
            },
        }
    }
    if (opts.add_generation_prompt)
    {
        try out.appendSlice(gpa, "<|im_start|>assistant\n");
        if (opts.thinking == false)
        {
            try out.appendSlice(gpa, "<think>\n\n</think>\n\n");
        }
    }
}

fn appendAll(gpa: std.mem.Allocator, out: *std.ArrayList(u8), parts: []const []const u8) !void {
    for (parts) |p|
    {
        try out.appendSlice(gpa, p);
    }
}

fn expectRender(format: Format, messages: []const Message, opts: Options, want: []const u8) !void {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try render(gpa, format, messages, opts, &out);
    try std.testing.expectEqualStrings(want, out.items);
}

test "tulu renders like OLMoE's template" {
    try expectRender(.tulu, &.{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello." },
        .{ .role = .user, .content = "Bye" },
    }, .{}, "<|endoftext|><|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\nHello.<|endoftext|>\n<|user|>\nBye\n<|assistant|>\n");
}

test "qwen3 drops old reasoning and keeps it after the last user turn" {
    try expectRender(.qwen3, &.{
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\nhmm\n</think>\n\nHello." },
        .{ .role = .user, .content = "Bye" },
    }, .{ .thinking = false }, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    try expectRender(.qwen3, &.{
        .{ .role = .system, .content = "S" },
        .{ .role = .user, .content = "Q" },
        .{ .role = .assistant, .content = "A", .reasoning = "\nr\n" },
    }, .{ .add_generation_prompt = false }, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\nr\n</think>\n\nA<|im_end|>\n");
}

test "qwen38 keeps every turn's reasoning and opens the reply's" {
    const note = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    try expectRender(.qwen38, &.{
        .{ .role = .system, .content = " S\n" },
        .{ .role = .user, .content = "Hi\n\n" },
        .{ .role = .assistant, .content = "Hello.", .reasoning = "r" },
        .{ .role = .user, .content = "Bye" },
    }, .{}, "<|im_start|>system\n" ++ note ++ "\n\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nr\n</think>\n\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n");
    try expectRender(.qwen38, &.{
        .{ .role = .user, .content = "Q\u{3000}" },
        .{ .role = .assistant, .content = "A" },
    }, .{ .thinking = false }, "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    try expectRender(.qwen38, &.{.{ .role = .user, .content = "Q" }}, .{ .effort = .medium }, "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n");
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try std.testing.expectError(error.SystemNotFirst, render(std.testing.allocator, .qwen38, &.{ .{ .role = .user, .content = "a" }, .{ .role = .system, .content = "b" } }, .{}, &out));
    try std.testing.expectError(error.NoUserQuery, render(std.testing.allocator, .qwen38, &.{.{ .role = .user, .content = "<tool_response>x</tool_response>" }}, .{}, &out));
}
