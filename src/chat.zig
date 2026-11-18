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
//!   glm (GLM-5.3-Flash, TASK-32): [gMASK]<sop>, a reasoning-effort system turn,
//!     <|user|>Hi, <|assistant|><think>...</think>reply, and a prompt that opens <think>
//!
//! Since TASK-38 `Chat` renders the GGUF's own Jinja template (src/jinja.zig) and
//! keeps these formats as a fallback for a template the interpreter refuses.
const std = @import("std");
const jinja = @import("jinja.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const Format = enum { tulu, qwen3, qwen3_release, qwen38, glm };

/// `reasoning_effort`. Qwen3.8 takes xhigh, medium or low (xhigh and low add a system note);
/// GLM-5.3-Flash takes low or high, and anything else is its default, max.
pub const Effort = enum { xhigh, high, medium, low, max };

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
    /// `reasoning_effort`, when the client set it; each template has its own default.
    effort: ?Effort = null,
};

/// Whether the prompt itself opens the reply's reasoning, so generation starts inside it.
pub fn opensReasoning(format: Format, opts: Options) bool
{
    return switch (format) {
        .qwen38 => opts.add_generation_prompt and opts.thinking != false,
        // GLM's template opens it whatever the client asks.
        .glm => opts.add_generation_prompt,
        else => false,
    };
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
    if (tok.pattern == .glm4 and std.mem.startsWith(u8, t, "[gMASK]<sop>") and has(t, "Reasoning Effort: {{ effective_reasoning_effort | capitalize }}") and
        has(t, "<|assistant|>{{- '<think>' -}}") and has(t, "{{ '<think>' + reasoning_content +  '</think>'}}"))
    {
        return .glm;
    }
    return null;
}

/// Token ids that end an assistant reply.
pub fn stopIds(format: Format, tok: *const Tokenizer, buf: *[4]u32) []const u32 {
    var n: usize = 0;
    const names: []const []const u8 = switch (format) {
        .tulu => &.{"<|endoftext|>"},
        .qwen3, .qwen3_release, .qwen38 => &.{ "<|im_end|>", "<|endoftext|>" },
        .glm => &.{ "<|endoftext|>", "<|user|>", "<|observation|>" },
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
        .glm => try renderGlm(gpa, messages, opts, out),
    }
}

/// A model's chat prompt (TASK-38): the GGUF's own Jinja template, rendered as
/// transformers does, or else one of the hand-written formats above.
pub const Chat = struct {
    arena: std.heap.ArenaAllocator,
    template: ?jinja.Template = null,
    format: ?Format = null,
    /// Why the template could not be used, when it could not.
    refused: []const u8 = "",
    /// bos_token, eos_token and the like, as transformers passes them to the template.
    specials: jinja.Dict = .empty,
    /// Whether replies may hold reasoning between <think> and </think>.
    thinks: bool = false,

    /// From the tokenizer's chat template and special tokens.
    pub fn init(gpa: std.mem.Allocator, tok: *const Tokenizer) !Chat {
        var c: Chat = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        const names = [_]struct { []const u8, ?u32 }{ .{ "bos_token", tok.bos }, .{ "eos_token", tok.eos }, .{ "unk_token", tok.unk }, .{ "pad_token", tok.pad } };
        for (names) |n|
        {
            if (n[1]) |id|
            {
                if (tok.piece(id).len > 0)
                {
                    try c.specials.put(a, n[0], .{ .string = tok.piece(id) });
                }
            }
        }
        if (tok.chat_template) |t|
        {
            try c.useTemplate(t);
        }
        c.format = detect(tok);
        c.thinks = if (c.format) |f| f != .tulu else if (tok.chat_template) |t| std.mem.indexOf(u8, t, "<think>") != null else false;
        if (c.template == null and c.format == null and c.refused.len == 0)
        {
            c.refused = "the model has no chat template";
        }
        return c;
    }

    /// From a template's text alone, with special tokens given by name (for checking templates of models we cannot load).
    pub fn fromTemplate(gpa: std.mem.Allocator, text: []const u8, specials: []const [2][]const u8) !Chat {
        var c: Chat = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer c.arena.deinit();
        for (specials) |s|
        {
            try c.specials.put(c.arena.allocator(), try c.arena.allocator().dupe(u8, s[0]), .{ .string = try c.arena.allocator().dupe(u8, s[1]) });
        }
        try c.useTemplate(text);
        c.thinks = std.mem.indexOf(u8, text, "<think>") != null;
        return c;
    }

    pub fn deinit(self: *Chat) void {
        self.arena.deinit();
        self.* = undefined;
    }

    fn useTemplate(self: *Chat, text: []const u8) !void {
        const a = self.arena.allocator();
        var diag: jinja.Diag = .{};
        self.template = jinja.Template.parse(a, try a.dupe(u8, text), &diag) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => blk: {
                self.refused = try std.fmt.allocPrint(a, "{s} (template line {d})", .{ diag.msg, diag.line });
                break :blk null;
            },
        };
    }

    /// Whether a prompt can be made at all.
    pub fn usable(self: *const Chat) bool {
        return self.template != null or self.format != null;
    }

    /// How prompts are made, for the startup line.
    pub fn describe(self: *const Chat) []const u8 {
        if (self.template != null)
        {
            return "the GGUF's chat template";
        }
        return if (self.format) |f| @tagName(f) else "none";
    }

    /// Renders `messages`; with a template, its refusals (raise_exception) come back as
    /// error.TemplateRefused with the template's message in `diag`.
    pub fn render(self: *const Chat, a: std.mem.Allocator, messages: []const Message, opts: Options, diag: *jinja.Diag, out: *std.ArrayList(u8)) !void {
        if (self.template == null)
        {
            return chatRender(a, self.format orelse return error.NoChatTemplate, messages, opts, out);
        }
        const list = try a.alloc(jinja.Value, messages.len);
        for (messages, list) |m, *v|
        {
            const d = try a.create(jinja.Dict);
            d.* = .empty;
            try d.put(a, "role", .{ .string = @tagName(m.role) });
            try d.put(a, "content", .{ .string = m.content });
            if (m.reasoning) |r|
            {
                try d.put(a, "reasoning_content", .{ .string = r });
            }
            v.* = .{ .dict = d };
        }
        return self.renderValues(a, .{ .list = list }, opts, diag, out);
    }

    /// Renders messages given as template values, so any fields they carry reach the template.
    pub fn renderValues(self: *const Chat, a: std.mem.Allocator, messages: jinja.Value, opts: Options, diag: *jinja.Diag, out: *std.ArrayList(u8)) !void {
        const t = self.template orelse return error.NoChatTemplate;
        var vars: jinja.Dict = .empty;
        var it = self.specials.iterator();
        while (it.next()) |e|
        {
            try vars.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        try vars.put(a, "messages", messages);
        try vars.put(a, "tools", .none);
        try vars.put(a, "documents", .none);
        try vars.put(a, "add_generation_prompt", .{ .boolean = opts.add_generation_prompt });
        if (opts.thinking) |th|
        {
            try vars.put(a, "enable_thinking", .{ .boolean = th });
        }
        if (opts.effort) |e|
        {
            try vars.put(a, "reasoning_effort", .{ .string = @tagName(e) });
        }
        t.render(a, &vars, diag, out) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Raised => return error.TemplateRefused,
            else => return error.TemplateFailed,
        };
    }

    /// Token ids that end an assistant reply.
    pub fn stopIds(self: *const Chat, tok: *const Tokenizer, buf: *[4]u32) []const u32 {
        if (self.format) |f|
        {
            return chatStopIds(f, tok, buf);
        }
        var n: usize = 0;
        for ([_]?u32{ tok.eos, tok.eot }) |id|
        {
            if (id != null and (n == 0 or buf[0] != id.?))
            {
                buf[n] = id.?;
                n += 1;
            }
        }
        return buf[0..n];
    }
};

/// Whether a rendered prompt opens the reply's reasoning, so generation starts inside it.
pub fn promptOpensReasoning(prompt: []const u8) bool {
    return std.mem.endsWith(u8, std.mem.trimEnd(u8, prompt, " \t\r\n"), "<think>");
}

const chatRender = render;
const chatStopIds = stopIds;

/// GLM-5.3-Flash's template: turns as they are, an assistant turn's reasoning taken from
/// `reasoning` or else from a </think> in its content, and its reply trimmed.
fn renderGlm(gpa: std.mem.Allocator, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) !void {
    const effort: []const u8 = switch (opts.effort orelse .max) {
        .low => "Low",
        .high => "High",
        else => "Max",
    };
    try appendAll(gpa, out, &.{ "[gMASK]<sop><|system|>Reasoning Effort: ", effort });
    for (messages) |m|
    {
        switch (m.role)
        {
            .system => try appendAll(gpa, out, &.{ "<|system|>", m.content }),
            .user => try appendAll(gpa, out, &.{ "<|user|>", m.content }),
            .assistant => {
                var content = m.content;
                var reasoning: ?[]const u8 = m.reasoning;
                if (reasoning == null)
                {
                    if (std.mem.indexOf(u8, content, "</think>")) |end|
                    {
                        const before = content[0..end];
                        const open = if (std.mem.lastIndexOf(u8, before, "<think>")) |i| i + "<think>".len else 0;
                        reasoning = before[open..];
                        content = content[std.mem.lastIndexOf(u8, content, "</think>").? + "</think>".len ..];
                    }
                }
                try appendAll(gpa, out, &.{ "<|assistant|><think>", reasoning orelse "", "</think>", pyTrim(content) });
            },
        }
    }
    if (opts.add_generation_prompt)
    {
        try out.appendSlice(gpa, "<|assistant|><think>");
    }
}

/// Refuses what Qwen3.8's template raises an exception for: a system turn after the first, or no user query.
fn renderQwen38(gpa: std.mem.Allocator, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) !void {
    const effort = opts.effort orelse .xhigh;
    if (opts.thinking != false and (effort == .high or effort == .max))
    {
        return error.UnsupportedEffort;
    }
    const note: []const u8 = if (opts.thinking == false) "" else switch (effort) {
        .xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        .low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        else => "",
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

test "glm takes reasoning from a </think> and opens the reply's" {
    try expectRender(.glm, &.{ .{ .role = .user, .content = "Hi" }, .{ .role = .assistant, .content = "<think>r</think>\n Hello \n" }, .{ .role = .user, .content = "Again" } }, .{}, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>r</think>Hello<|user|>Again<|assistant|><think>");
    try expectRender(.glm, &.{.{ .role = .user, .content = "Q" }}, .{ .effort = .low, .add_generation_prompt = false }, "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Q");
    try expectRender(.glm, &.{ .{ .role = .user, .content = "Q" }, .{ .role = .assistant, .content = "x</think>y</think>z" } }, .{ .add_generation_prompt = false }, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Q<|assistant|><think>x</think>z");
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

test "Chat renders a template with the special tokens and refuses one it cannot run" {
    const gpa = std.testing.allocator;
    var c = try Chat.fromTemplate(gpa, "{{ bos_token }}{% for m in messages %}[{{ m.role }}] {{ m.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}", &.{.{ "bos_token", "<s>" }});
    defer c.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var out: std.ArrayList(u8) = .empty;
    var diag: jinja.Diag = .{};
    try c.render(arena.allocator(), &.{.{ .role = .user, .content = "Hi" }}, .{}, &diag, &out);
    try std.testing.expectEqualStrings("<s>[user] Hi[assistant]", out.items);

    var bad = try Chat.fromTemplate(gpa, "{% include 'x' %}", &.{});
    defer bad.deinit();
    try std.testing.expect(!bad.usable() and std.mem.indexOf(u8, bad.refused, "include") != null);
    try std.testing.expect(promptOpensReasoning("<|im_start|>assistant\n<think>\n") and !promptOpensReasoning("<think>\n\n</think>\n\n"));
}
