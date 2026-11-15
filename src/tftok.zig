//! Tokenizer check tool (TASK-30): tokenizes text files with the engine's
//! tokenizer, for comparison with Hugging Face's (tools/ref/check_tokenizer.py).
//!
//!   tftok TOKENIZER FILE...    print each file's token ids, one line per file
//!   tftok TOKENIZER --info     print what was loaded
//!
//! TOKENIZER is a GGUF model (its `tokenizer.*` metadata) or a `tokenizer.json`.
//! Also checks that decoding the ids gives back the file's NFC text, and exits
//! with status 1 if not.
const std = @import("std");
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
            try out.print("pattern {s}, {d} ids, {d} added tokens, merges {d}, bos {?d}, eos {?d}, chat template {s}\n", .{
                @tagName(tok.pattern), tok.vocabSize(), tok.added.len, tok.merges.count(), tok.bos, tok.eos, if (tok.chat_template != null) "yes" else "no",
            });
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
