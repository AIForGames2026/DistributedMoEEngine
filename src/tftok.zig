//! Tokenizer check tool (TASK-30): tokenizes text files with the engine's
//! tokenizer, for comparison with Hugging Face's (tools/ref/check_tokenizer.py).
//!
//!   tftok TOKENIZER FILE...    print each file's token ids, one line per file
//!   tftok TOKENIZER --info     print what was loaded
//!   tftok TOKENIZER [--config tokenizer_config.json] --chat CONV.json...
//!                              print each conversation's prompt token ids
//!                              (tools/ref/check_chat.py)
//!
//! TOKENIZER is a GGUF model (its `tokenizer.*` metadata) or a `tokenizer.json`;
//! a `tokenizer.json` has no chat template, so `--config` supplies one.
//! Also checks that decoding a text file's ids gives back its NFC text, and
//! exits with status 1 if not.
const std = @import("std");
const chat = @import("chat.zig");
const gguf = @import("gguf.zig");
const unicode = @import("unicode.zig");
const tokenizer = @import("tokenizer.zig");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var it = try init.minimal.args.iterateAllocator(init.arena.allocator());
    _ = it.next();
    const source = it.next() orelse {
        std.debug.print("usage: tftok TOKENIZER (FILE... | --info)\n", .{});
        std.process.exit(2);
    };
    var tok = try load(gpa, io, source);
    defer tok.deinit();

    var out_buf: [1 << 16]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    const out = &stdout.interface;
    var failed = false;
    while (it.next()) |arg|
    {
        if (std.mem.eql(u8, arg, "--info"))
        {
            const format = chat.detect(&tok);
            try out.print("pattern {s}, {d} ids, {d} added tokens, merges {d}, bos {?d}, eos {?d}, chat format {s}\n", .{
                @tagName(tok.pattern), tok.vocabSize(), tok.added.len, tok.merges.count(), tok.bos, tok.eos,
                if (format) |f| @tagName(f) else if (tok.chat_template != null) "unknown" else "none",
            });
            continue;
        }
        if (std.mem.eql(u8, arg, "--config"))
        {
            try setTemplate(gpa, io, &tok, it.next() orelse return error.MissingValue);
            continue;
        }
        if (std.mem.eql(u8, arg, "--chat"))
        {
            const format = chat.detect(&tok) orelse return error.UnknownChatTemplate;
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(gpa);
            try renderChat(gpa, io, &tok, format, it.next() orelse return error.MissingValue, &ids);
            for (ids.items, 0..) |id, i|
            {
                try out.print("{s}{d}", .{ if (i == 0) "" else " ", id });
            }
            try out.print("\n", .{});
            continue;
        }
        const text = try std.Io.Dir.cwd().readFileAlloc(io, arg, gpa, .limited(1 << 30));
        defer gpa.free(text);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        try tok.encode(gpa, text, &ids);
        for (ids.items, 0..) |id, i|
        {
            try out.print("{s}{d}", .{ if (i == 0) "" else " ", id });
        }
        try out.print("\n", .{});

        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(gpa);
        try unicode.nfc(gpa, text, &want);
        var back: std.ArrayList(u8) = .empty;
        defer back.deinit(gpa);
        try tokenizer.decode(gpa, &tok, ids.items, &back);
        if (!std.mem.eql(u8, want.items, back.items))
        {
            std.debug.print("{s}: decoding the ids does not give back the text\n", .{arg});
            failed = true;
        }
    }
    try out.flush();
    if (failed)
    {
        std.process.exit(1);
    }
}

/// Takes the chat template from a Hugging Face `tokenizer_config.json`.
fn setTemplate(gpa: std.mem.Allocator, io: std.Io, tok: *tokenizer.Tokenizer, path: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(struct { chat_template: []const u8 }, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    tok.chat_template = try tok.arena.allocator().dupe(u8, parsed.value.chat_template);
}

/// Renders a conversation file and tokenizes it. The file holds `messages`
/// (role, content, optional reasoning_content), `add_generation_prompt`, and
/// optionally `enable_thinking`.
fn renderChat(gpa: std.mem.Allocator, io: std.Io, tok: *const tokenizer.Tokenizer, format: chat.Format, path: []const u8, ids: *std.ArrayList(u32)) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    const Conversation = struct {
        messages: []const struct { role: chat.Role, content: []const u8, reasoning_content: ?[]const u8 = null },
        add_generation_prompt: bool = true,
        enable_thinking: ?bool = null,
    };
    const parsed = try std.json.parseFromSlice(Conversation, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const msgs = try gpa.alloc(chat.Message, parsed.value.messages.len);
    defer gpa.free(msgs);
    for (parsed.value.messages, msgs) |m, *dst|
    {
        dst.* = .{ .role = m.role, .content = m.content, .reasoning = m.reasoning_content };
    }
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try chat.render(gpa, format, msgs, .{ .add_generation_prompt = parsed.value.add_generation_prompt, .thinking = parsed.value.enable_thinking }, &text);
    try tok.encode(gpa, text.items, ids);
}

fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !tokenizer.Tokenizer {
    if (!std.mem.endsWith(u8, path, ".gguf"))
    {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
        defer gpa.free(bytes);
        return tokenizer.Tokenizer.fromJson(gpa, bytes);
    }
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size = try file.length(io);
    // The metadata sits at the front; read more until the header fits.
    var want: u64 = 16 << 20;
    while (true)
    {
        const n: usize = @intCast(@min(want, size));
        const prefix = try gpa.alloc(u8, n);
        defer gpa.free(prefix);
        _ = try file.readPositionalAll(io, prefix, 0);
        var h = gguf.parse(gpa, prefix, size) catch |e| {
            if (e == error.Truncated and n < size)
            {
                want *= 2;
                continue;
            }
            return e;
        };
        defer h.deinit(gpa);
        return tokenizer.Tokenizer.fromGguf(gpa, h);
    }
}
