//! Tokenizer check tool (TASK-30): tokenizes text files with the engine's
//! tokenizer, for comparison with Hugging Face's (tools/ref/check_tokenizer.py).
//!
//!   tftok TOKENIZER FILE...    print each file's token ids, one line per file
//!   tftok TOKENIZER --info     print what was loaded
//!   tftok TOKENIZER [--config tokenizer_config.json] --chat CONV.json...
//!                              print each conversation's prompt token ids
//!                              (tools/ref/check_chat.py)
//!   tftok TOKENIZER --render CONV.json...
//!                              print each conversation's prompt text, as JSON
//!   tftok --template FILE [--config tokenizer_config.json] --render CONV.json...
//!                              the same from a template file alone, with the
//!                              special tokens from the config (TASK-38)
//!
//! Prompts come from the chat template through src/jinja.zig, or from the
//! built-in formats when the interpreter refuses the template.
//! TOKENIZER is a GGUF model (its `tokenizer.*` metadata) or a `tokenizer.json`;
//! a `tokenizer.json` has no chat template, so `--config` supplies one.
//! Also checks that decoding a text file's ids gives back its text (in NFC when
//! the tokenizer normalizes), and exits with status 1 if not.
const std = @import("std");
const chat = @import("chat.zig");
const gguf = @import("gguf.zig");
const jinja = @import("jinja.zig");
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
    var template_text: ?[]const u8 = null;
    defer if (template_text) |t| gpa.free(t);
    var tok_opt: ?tokenizer.Tokenizer = null;
    if (std.mem.eql(u8, source, "--template"))
    {
        template_text = try std.Io.Dir.cwd().readFileAlloc(io, it.next() orelse return error.MissingValue, gpa, .limited(1 << 26));
    }
    else
    {
        tok_opt = try load(gpa, io, source);
    }
    defer if (tok_opt) |*t| t.deinit();
    var specials: std.ArrayList([2][]const u8) = .empty;
    defer specials.deinit(gpa);
    var config_arena = std.heap.ArenaAllocator.init(gpa);
    defer config_arena.deinit();
    var prompts: ?chat.Chat = null;
    defer if (prompts) |*p| p.deinit();

    var out_buf: [1 << 16]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &out_buf);
    const out = &stdout.interface;
    var failed = false;
    while (it.next()) |arg|
    {
        if (std.mem.eql(u8, arg, "--config"))
        {
            const path = it.next() orelse return error.MissingValue;
            if (tok_opt) |*t|
            {
                try setTemplate(gpa, io, t, path);
            }
            else
            {
                try readSpecials(config_arena.allocator(), io, path, &specials, gpa);
            }
            if (prompts) |*p|
            {
                p.deinit();
                prompts = null;
            }
            continue;
        }
        if (std.mem.eql(u8, arg, "--chat") or std.mem.eql(u8, arg, "--render"))
        {
            if (prompts == null)
            {
                prompts = if (tok_opt) |*t| try chat.Chat.init(gpa, t) else try chat.Chat.fromTemplate(gpa, template_text.?, specials.items);
                if (prompts.?.refused.len > 0)
                {
                    std.debug.print("tftok: the chat template cannot be rendered: {s}\n", .{prompts.?.refused});
                }
                if (!prompts.?.usable())
                {
                    return error.UnknownChatTemplate;
                }
            }
            const path = it.next() orelse return error.MissingValue;
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const a = arena.allocator();
            var text: std.ArrayList(u8) = .empty;
            var diag: jinja.Diag = .{};
            renderChat(a, io, &prompts.?, path, &diag, &text) catch |err| switch (err) {
                // The template refuses this conversation; check_chat.py expects the same refusal.
                error.TemplateRefused => {
                    try out.print("refused {s}\n", .{diag.msg});
                    continue;
                },
                error.SystemNotFirst, error.NoUserQuery, error.UnsupportedEffort => {
                    try out.print("refused {t}\n", .{err});
                    continue;
                },
                error.TemplateFailed => {
                    std.debug.print("tftok: {s}: the template failed: {s} (line {d})\n", .{ path, diag.msg, diag.line });
                    return err;
                },
                else => return err,
            };
            if (std.mem.eql(u8, arg, "--render"))
            {
                try writeJsonString(out, text.items);
                try out.print("\n", .{});
                continue;
            }
            const tok = &(tok_opt orelse return error.NoTokenizer);
            var ids: std.ArrayList(u32) = .empty;
            try tok.encode(a, text.items, &ids);
            for (ids.items, 0..) |id, i|
            {
                try out.print("{s}{d}", .{ if (i == 0) "" else " ", id });
            }
            try out.print("\n", .{});
            continue;
        }
        const tok = &(tok_opt orelse return error.NoTokenizer);
        if (std.mem.eql(u8, arg, "--info"))
        {
            const format = chat.detect(tok);
            try out.print("pattern {s}, {d} ids, {d} added tokens, merges {d}, bos {?d} (added: {}), eos {?d}, chat format {s}\n", .{
                @tagName(tok.pattern), tok.vocabSize(), tok.added.len, tok.merges.count(), tok.bos, tok.add_bos, tok.eos,
                if (format) |f| @tagName(f) else if (tok.chat_template != null) "unknown" else "none",
            });
            continue;
        }
        const text = try std.Io.Dir.cwd().readFileAlloc(io, arg, gpa, .limited(1 << 30));
        defer gpa.free(text);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        // Hugging Face's encode starts with BOS when the tokenizer adds one.
        const bos: usize = if (tok.add_bos) 1 else 0;
        if (tok.add_bos)
        {
            try ids.append(gpa, tok.bos.?);
        }
        try tok.encode(gpa, text, &ids);
        for (ids.items, 0..) |id, i|
        {
            try out.print("{s}{d}", .{ if (i == 0) "" else " ", id });
        }
        try out.print("\n", .{});

        var want: std.ArrayList(u8) = .empty;
        defer want.deinit(gpa);
        if (tok.nfc)
        {
            try unicode.nfc(gpa, text, &want);
        }
        else if (tok.pattern == .none)
        {
            // SentencePiece decodes U+2581 as a space and drops the first space.
            const spaced = try std.mem.replaceOwned(u8, gpa, text, "\u{2581}", " ");
            defer gpa.free(spaced);
            const strip = tok.strip_space and spaced.len > 0 and spaced[0] == ' ';
            try want.appendSlice(gpa, spaced[@intFromBool(strip)..]);
        }
        else
        {
            try want.appendSlice(gpa, text);
        }
        var back: std.ArrayList(u8) = .empty;
        defer back.deinit(gpa);
        try tokenizer.decode(gpa, tok, ids.items[bos..], &back);
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

/// Renders a conversation file (`messages`, `add_generation_prompt`, and optionally
/// `enable_thinking` and `reasoning_effort`); a template sees every message field.
fn renderChat(a: std.mem.Allocator, io: std.Io, prompts: *const chat.Chat, path: []const u8, diag: *jinja.Diag, text: *std.ArrayList(u8)) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    const conv = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (conv != .object)
    {
        return error.BadConversation;
    }
    const o = conv.object;
    var opts: chat.Options = .{};
    if (o.get("add_generation_prompt")) |v|
    {
        opts.add_generation_prompt = v == .bool and v.bool;
    }
    if (o.get("enable_thinking")) |v|
    {
        opts.thinking = v == .bool and v.bool;
    }
    if (o.get("reasoning_effort")) |v|
    {
        opts.effort = std.meta.stringToEnum(chat.Effort, if (v == .string) v.string else "") orelse return error.UnsupportedEffort;
    }
    const messages = o.get("messages") orelse return error.BadConversation;
    if (prompts.template != null)
    {
        return prompts.renderValues(a, try jinja.fromJson(a, messages), opts, diag, text);
    }
    const parsed = try std.json.parseFromValueLeaky([]const struct { role: chat.Role, content: []const u8, reasoning_content: ?[]const u8 = null }, a, messages, .{ .ignore_unknown_fields = true });
    const msgs = try a.alloc(chat.Message, parsed.len);
    for (parsed, msgs) |m, *dst|
    {
        dst.* = .{ .role = m.role, .content = m.content, .reasoning = m.reasoning_content };
    }
    return prompts.render(a, msgs, opts, diag, text);
}

/// The special tokens (bos_token and the like) from a Hugging Face `tokenizer_config.json`.
fn readSpecials(a: std.mem.Allocator, io: std.Io, path: []const u8, out: *std.ArrayList([2][]const u8), gpa: std.mem.Allocator) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
    const cfg = try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{});
    if (cfg != .object)
    {
        return error.BadConfig;
    }
    for ([_][]const u8{ "bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token" }) |name|
    {
        const v = cfg.object.get(name) orelse continue;
        const text = switch (v) {
            .string => |t| t,
            .object => |ob| if (ob.get("content")) |c| (if (c == .string) c.string else continue) else continue,
            else => continue,
        };
        try out.append(gpa, .{ name, text });
    }
}

/// `s` as a JSON string.
fn writeJsonString(out: *std.Io.Writer, s: []const u8) !void {
    try out.writeByte('"');
    for (s) |c|
    {
        switch (c)
        {
            '"' => try out.writeAll("\\\""),
            '\\' => try out.writeAll("\\\\"),
            '\n' => try out.writeAll("\\n"),
            '\r' => try out.writeAll("\\r"),
            '\t' => try out.writeAll("\\t"),
            0...0x08, 0x0B, 0x0C, 0x0E...0x1F, 0x7F => try out.print("\\u{x:0>4}", .{c}),
            else => try out.writeByte(c),
        }
    }
    try out.writeByte('"');
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
