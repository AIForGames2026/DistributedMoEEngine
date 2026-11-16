//! Byte-level BPE tokenizer (TASK-30) for OLMoE-1B-7B, Qwen3-MoE and Qwen3.5-MoE, read
//! from the model's GGUF metadata or from a Hugging Face `tokenizer.json`.
//! Matches Hugging Face `tokenizers`: added tokens are split out first (those
//! marked not normalized from the raw text, the rest after NFC), the remaining
//! text is split by the model's pre-tokenizer pattern, and each piece is
//! merged by BPE over its UTF-8 bytes.
//!
//! The patterns are matched by hand instead of with a regex engine:
//!   gpt2 (OLMoE): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!   qwen2 (Qwen3): (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//!                  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   qwen35 (Qwen3.5, TASK-29): qwen2 with combining marks (\p{M}) counted
//!                  with letters, and so left out of the punctuation runs.
const std = @import("std");
const gguf = @import("gguf.zig");
const unicode = @import("unicode.zig");

pub const Pattern = enum { gpt2, qwen2, qwen35 };

pub const Error = error{ UnsupportedTokenizer, BadTokenizer, InvalidUtf8, OutOfMemory };

const Added = struct { id: u32, text: []const u8, normalized: bool };
const Merge = struct { rank: u32, id: u32 };

const qwen2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const qwen35_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    pattern: Pattern,
    /// Each id's bytes; added tokens are their literal text, unused ids empty.
    pieces: []const []const u8,
    special: []const bool,
    byte_ids: [256]u32,
    merges: std.AutoHashMapUnmanaged(u64, Merge),
    added: []const Added,
    bos: ?u32 = null,
    eos: ?u32 = null,
    chat_template: ?[]const u8 = null,

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// From a GGUF header's `tokenizer.*` keys. Copies what it keeps, so the
    /// header may be freed afterwards.
    pub fn fromGguf(gpa: std.mem.Allocator, h: gguf.Header) Error!Tokenizer {
        const model = h.getString("tokenizer.ggml.model") orelse return error.BadTokenizer;
        if (!std.mem.eql(u8, model, "gpt2"))
        {
            return error.UnsupportedTokenizer;
        }
        const pre = h.getString("tokenizer.ggml.pre") orelse return error.UnsupportedTokenizer;
        const pattern: Pattern = if (std.mem.eql(u8, pre, "olmo"))
            .gpt2
        else if (std.mem.eql(u8, pre, "qwen2"))
            .qwen2
        else if (std.mem.eql(u8, pre, "qwen35"))
            .qwen35
        else
            return error.UnsupportedTokenizer;

        var b = try Builder.init(gpa, pattern);
        errdefer b.arena.deinit();
        const a = b.arena.allocator();
        var toks = h.strings("tokenizer.ggml.tokens") orelse return error.BadTokenizer;
        var id: u32 = 0;
        while (toks.next() catch return error.BadTokenizer) |t|
        {
            // 3 is a control token, 4 user-defined, 5 unused padding.
            const kind = h.intAt("tokenizer.ggml.token_type", id) orelse 1;
            switch (kind)
            {
                3, 4 => try b.addAdded(id, try a.dupe(u8, t), kind == 4, kind == 3),
                5 => try b.vocab.append(a, null),
                else => try b.vocab.append(a, try a.dupe(u8, t)),
            }
            id += 1;
        }
        var merges = h.strings("tokenizer.ggml.merges") orelse return error.BadTokenizer;
        try b.indexVocab();
        var rank: u32 = 0;
        while (merges.next() catch return error.BadTokenizer) |m|
        {
            const sp = std.mem.indexOfScalar(u8, m, ' ') orelse return error.BadTokenizer;
            try b.addMerge(rank, m[0..sp], m[sp + 1 ..]);
            rank += 1;
        }
        var tok = try b.finish();
        tok.bos = if (h.getInt("tokenizer.ggml.bos_token_id")) |v| @intCast(v) else null;
        tok.eos = if (h.getInt("tokenizer.ggml.eos_token_id")) |v| @intCast(v) else null;
        if (h.getString("tokenizer.chat_template")) |t|
        {
            tok.chat_template = try tok.arena.allocator().dupe(u8, t);
        }
        return tok;
    }

    /// From a Hugging Face `tokenizer.json` with one of the two supported setups.
    pub fn fromJson(gpa: std.mem.Allocator, bytes: []const u8) Error!Tokenizer {
        const parsed = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.BadTokenizer,
        };
        defer parsed.deinit();
        const root = asObject(parsed.value) orelse return error.BadTokenizer;

        if (root.get("normalizer")) |n|
        {
            if (n != .null and !std.mem.eql(u8, typeOf(n) orelse "", "NFC"))
            {
                return error.UnsupportedTokenizer;
            }
        }
        const pattern = jsonPattern(root.get("pre_tokenizer") orelse return error.UnsupportedTokenizer) orelse return error.UnsupportedTokenizer;
        const model = asObject(root.get("model") orelse return error.BadTokenizer) orelse return error.BadTokenizer;
        if (!std.mem.eql(u8, typeOf(.{ .object = model }) orelse "", "BPE"))
        {
            return error.UnsupportedTokenizer;
        }

        var b = try Builder.init(gpa, pattern);
        errdefer b.arena.deinit();
        const a = b.arena.allocator();
        const vocab = asObject(model.get("vocab") orelse return error.BadTokenizer) orelse return error.BadTokenizer;
        var it = vocab.iterator();
        while (it.next()) |e|
        {
            const id = asId(e.value_ptr.*) orelse return error.BadTokenizer;
            try b.setVocab(id, try a.dupe(u8, e.key_ptr.*));
        }
        const added = switch (root.get("added_tokens") orelse std.json.Value{ .null = {} }) {
            .array => |arr| arr.items,
            .null => &.{},
            else => return error.BadTokenizer,
        };
        for (added) |v|
        {
            const o = asObject(v) orelse return error.BadTokenizer;
            const id = asId(o.get("id") orelse return error.BadTokenizer) orelse return error.BadTokenizer;
            const text = switch (o.get("content") orelse return error.BadTokenizer) {
                .string => |s| s,
                else => return error.BadTokenizer,
            };
            const normalized = asBool(o.get("normalized")) orelse false;
            const special = asBool(o.get("special")) orelse false;
            for ([_][]const u8{ "lstrip", "rstrip", "single_word" }) |flag|
            {
                if (asBool(o.get(flag)) orelse false)
                {
                    return error.UnsupportedTokenizer;
                }
            }
            if (id < b.vocab.items.len)
            {
                b.vocab.items[id] = null;
            }
            try b.addAdded(id, try a.dupe(u8, text), normalized, special);
        }

        try b.indexVocab();
        const merges = switch (model.get("merges") orelse return error.BadTokenizer) {
            .array => |arr| arr.items,
            else => return error.BadTokenizer,
        };
        for (merges, 0..) |m, rank|
        {
            switch (m)
            {
                .string => |s| {
                    const sp = std.mem.indexOfScalar(u8, s, ' ') orelse return error.BadTokenizer;
                    try b.addMerge(@intCast(rank), s[0..sp], s[sp + 1 ..]);
                },
                .array => |pair| {
                    if (pair.items.len != 2 or pair.items[0] != .string or pair.items[1] != .string)
                    {
                        return error.BadTokenizer;
                    }
                    try b.addMerge(@intCast(rank), pair.items[0].string, pair.items[1].string);
                },
                else => return error.BadTokenizer,
            }
        }
        return b.finish();
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.pieces.len;
    }

    /// The bytes token `id` stands for; empty for ids outside the vocabulary.
    pub fn piece(self: *const Tokenizer, id: u32) []const u8 {
        return if (id < self.pieces.len) self.pieces[id] else "";
    }

    pub fn isSpecial(self: *const Tokenizer, id: u32) bool {
        return id < self.special.len and self.special[id];
    }

    /// The id of an added token by its exact text, such as "<|im_end|>".
    pub fn addedId(self: *const Tokenizer, text: []const u8) ?u32 {
        for (self.added) |t|
        {
            if (std.mem.eql(u8, t.text, text))
            {
                return t.id;
            }
        }
        return null;
    }

    /// Appends the token ids of UTF-8 `text` to `out`. Added tokens in the text,
    /// special ones included, become their ids, as in Hugging Face's `encode`.
    pub fn encode(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, out: *std.ArrayList(u32)) Error!void {
        if (!std.unicode.utf8ValidateSlice(text))
        {
            return error.InvalidUtf8;
        }
        var s: Scratch = .{};
        defer s.deinit(gpa);
        var norm: std.ArrayList(u8) = .empty;
        defer norm.deinit(gpa);

        var pos: usize = 0;
        while (pos < text.len)
        {
            const m = self.findAdded(text, pos, false);
            const end = if (m) |x| x.start else text.len;
            if (end > pos)
            {
                norm.clearRetainingCapacity();
                try unicode.nfc(gpa, text[pos..end], &norm);
                var npos: usize = 0;
                while (npos < norm.items.len)
                {
                    const nm = self.findAdded(norm.items, npos, true);
                    const nend = if (nm) |x| x.start else norm.items.len;
                    try self.encodePlain(gpa, norm.items[npos..nend], &s, out);
                    if (nm) |x|
                    {
                        try out.append(gpa, x.id);
                        npos = x.start + x.len;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            if (m) |x|
            {
                try out.append(gpa, x.id);
                pos = x.start + x.len;
            }
            else
            {
                break;
            }
        }
    }

    const Found = struct { start: usize, len: usize, id: u32 };

    /// Leftmost-longest added token at or after `from` with the given flag.
    fn findAdded(self: *const Tokenizer, text: []const u8, from: usize, normalized: bool) ?Found {
        for (from..text.len) |start|
        {
            var best: ?Found = null;
            for (self.added) |t|
            {
                if (t.normalized != normalized or t.text.len == 0 or t.text[0] != text[start])
                {
                    continue;
                }
                if (std.mem.startsWith(u8, text[start..], t.text) and (best == null or t.text.len > best.?.len))
                {
                    best = .{ .start = start, .len = t.text.len, .id = t.id };
                }
            }
            if (best != null)
            {
                return best;
            }
        }
        return null;
    }

    fn encodePlain(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        if (text.len == 0)
        {
            return;
        }
        try split(gpa, self.pattern, text, s);
        for (s.bounds.items[0 .. s.bounds.items.len - 1], s.bounds.items[1..]) |lo, hi|
        {
            try self.bpe(gpa, text[lo..hi], s, out);
        }
    }

    fn bpe(self: *const Tokenizer, gpa: std.mem.Allocator, word: []const u8, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        const none = std.math.maxInt(u32);
        s.ids.clearRetainingCapacity();
        s.next.clearRetainingCapacity();
        s.prev.clearRetainingCapacity();
        s.heap.clearRetainingCapacity();
        for (word, 0..) |byte, i|
        {
            try s.ids.append(gpa, self.byte_ids[byte]);
            try s.prev.append(gpa, if (i == 0) none else @intCast(i - 1));
            try s.next.append(gpa, if (i + 1 == word.len) none else @intCast(i + 1));
        }
        const ids = s.ids.items;
        const prev = s.prev.items;
        const next = s.next.items;
        for (0..word.len -| 1) |i|
        {
            try self.pushPair(gpa, s, @intCast(i), ids[i], ids[i + 1]);
        }
        while (s.pop()) |it|
        {
            const p = it.pos;
            const q = next[p];
            if (ids[p] != it.a or q == none or ids[q] != it.b)
            {
                continue;
            }
            ids[p] = it.id;
            ids[q] = none;
            next[p] = next[q];
            if (next[q] != none)
            {
                prev[next[q]] = p;
            }
            if (prev[p] != none)
            {
                try self.pushPair(gpa, s, prev[p], ids[prev[p]], ids[p]);
            }
            if (next[p] != none)
            {
                try self.pushPair(gpa, s, p, ids[p], ids[next[p]]);
            }
        }
        var i: u32 = 0;
        while (i != none)
        {
            try out.append(gpa, ids[i]);
            i = next[i];
        }
    }

    fn pushPair(self: *const Tokenizer, gpa: std.mem.Allocator, s: *Scratch, pos: u32, a: u32, b: u32) !void {
        if (self.merges.get(pairKey(a, b))) |m|
        {
            try s.push(gpa, .{ .rank = m.rank, .pos = pos, .a = a, .b = b, .id = m.id });
        }
    }
};

fn pairKey(a: u32, b: u32) u64 {
    return @as(u64, a) << 32 | b;
}

/// Reusable buffers for one `encode` call.
const Scratch = struct {
    cps: std.ArrayList(u21) = .empty,
    offs: std.ArrayList(u32) = .empty,
    bounds: std.ArrayList(u32) = .empty,
    ids: std.ArrayList(u32) = .empty,
    prev: std.ArrayList(u32) = .empty,
    next: std.ArrayList(u32) = .empty,
    heap: std.ArrayList(Item) = .empty,

    const Item = struct { rank: u32, pos: u32, a: u32, b: u32, id: u32 };

    fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        self.cps.deinit(gpa);
        self.offs.deinit(gpa);
        self.bounds.deinit(gpa);
        self.ids.deinit(gpa);
        self.prev.deinit(gpa);
        self.next.deinit(gpa);
        self.heap.deinit(gpa);
    }

    fn less(x: Item, y: Item) bool {
        return x.rank < y.rank or (x.rank == y.rank and x.pos < y.pos);
    }

    fn push(self: *Scratch, gpa: std.mem.Allocator, it: Item) !void {
        try self.heap.append(gpa, it);
        const h = self.heap.items;
        var i = h.len - 1;
        while (i > 0)
        {
            const parent = (i - 1) / 2;
            if (!less(h[i], h[parent]))
            {
                break;
            }
            std.mem.swap(Item, &h[i], &h[parent]);
            i = parent;
        }
    }

    fn pop(self: *Scratch) ?Item {
        const h = self.heap.items;
        if (h.len == 0)
        {
            return null;
        }
        const top = h[0];
        h[0] = h[h.len - 1];
        self.heap.items.len -= 1;
        const n = self.heap.items.len;
        var i: usize = 0;
        while (true)
        {
            var m = i;
            const l = 2 * i + 1;
            if (l < n and less(h[l], h[m]))
            {
                m = l;
            }
            if (l + 1 < n and less(h[l + 1], h[m]))
            {
                m = l + 1;
            }
            if (m == i)
            {
                break;
            }
            std.mem.swap(Item, &h[i], &h[m]);
            i = m;
        }
        return top;
    }
};

const class_letter = 1;
const class_number = 2;
const class_space = 4;
const class_newline = 8;
const class_mark = 16;

fn classOf(cp: u21) u8 {
    if (cp < 0x80)
    {
        return switch (cp) {
            'a'...'z', 'A'...'Z' => class_letter,
            '0'...'9' => class_number,
            '\r', '\n' => class_space | class_newline,
            0x09, 0x0B, 0x0C, ' ' => class_space,
            else => 0,
        };
    }
    var c: u8 = 0;
    if (unicode.isLetter(cp))
    {
        c |= class_letter;
    }
    if (unicode.isNumber(cp))
    {
        c |= class_number;
    }
    if (unicode.isMark(cp))
    {
        c |= class_mark;
    }
    if (unicode.isSpace(cp))
    {
        c |= class_space;
    }
    return c;
}

/// Splits valid UTF-8 `text` by `pattern` into `s.bounds`: byte offsets of
/// each piece's start, plus the end.
fn split(gpa: std.mem.Allocator, pattern: Pattern, text: []const u8, s: *Scratch) !void {
    s.cps.clearRetainingCapacity();
    s.offs.clearRetainingCapacity();
    s.bounds.clearRetainingCapacity();
    var view = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (true)
    {
        const at: u32 = @intCast(view.i);
        const cp = view.nextCodepoint() orelse break;
        try s.cps.append(gpa, cp);
        try s.offs.append(gpa, at);
    }
    try s.offs.append(gpa, @intCast(text.len));
    const cps = s.cps.items;
    var i: usize = 0;
    while (i < cps.len)
    {
        try s.bounds.append(gpa, s.offs.items[i]);
        const end = switch (pattern) {
            .gpt2 => matchGpt2(cps, i),
            .qwen2 => matchQwen(cps, i, class_letter),
            .qwen35 => matchQwen(cps, i, class_letter | class_mark),
        };
        std.debug.assert(end > i);
        i = end;
    }
    try s.bounds.append(gpa, @intCast(text.len));
}

fn is(cps: []const u21, i: usize, class: u8) bool {
    return i < cps.len and classOf(cps[i]) & class != 0;
}

fn isOther(cps: []const u21, i: usize) bool {
    return isOtherThan(cps, i, class_letter);
}

/// Neither `letters`, a number, nor white space.
fn isOtherThan(cps: []const u21, i: usize, letters: u8) bool {
    return i < cps.len and classOf(cps[i]) & (letters | class_number | class_space) == 0;
}

fn runOf(cps: []const u21, from: usize, class: u8) usize {
    var j = from;
    while (is(cps, j, class))
    {
        j += 1;
    }
    return j;
}

fn runOfOther(cps: []const u21, from: usize) usize {
    return runOfOtherThan(cps, from, class_letter);
}

fn runOfOtherThan(cps: []const u21, from: usize, letters: u8) usize {
    var j = from;
    while (isOtherThan(cps, j, letters))
    {
        j += 1;
    }
    return j;
}

/// End of the contraction at `i` ('s, 't, 're, 've, 'm, 'll, 'd), or null.
fn contraction(cps: []const u21, i: usize, fold: bool) ?usize {
    if (cps[i] != '\'' or i + 1 >= cps.len)
    {
        return null;
    }
    const a = lower(cps[i + 1], fold);
    if (a == 's' or a == 't' or a == 'm' or a == 'd')
    {
        return i + 2;
    }
    if (i + 2 < cps.len)
    {
        const b = lower(cps[i + 2], fold);
        if ((a == 'r' and b == 'e') or (a == 'v' and b == 'e') or (a == 'l' and b == 'l'))
        {
            return i + 3;
        }
    }
    return null;
}

fn lower(cp: u21, fold: bool) u21 {
    if (!fold)
    {
        return cp;
    }
    if (cp >= 'A' and cp <= 'Z')
    {
        return cp + 32;
    }
    return if (cp == 0x17F) 's' else cp; // long s case-folds to s
}

/// White space: all of it at the end of the text, else all but the last
/// character, so a following word keeps its leading space.
fn spaceRun(cps: []const u21, i: usize) usize {
    const j = runOf(cps, i, class_space);
    if (j == cps.len or j - i < 2)
    {
        return j;
    }
    return j - 1;
}

fn matchGpt2(cps: []const u21, i: usize) usize {
    if (contraction(cps, i, false)) |end|
    {
        return end;
    }
    const s = if (cps[i] == ' ') i + 1 else i;
    if (is(cps, s, class_letter))
    {
        return runOf(cps, s, class_letter);
    }
    if (is(cps, s, class_number))
    {
        return runOf(cps, s, class_number);
    }
    if (isOther(cps, s))
    {
        return runOfOther(cps, s);
    }
    return spaceRun(cps, i);
}

/// The qwen2 and qwen35 patterns, which differ only in what counts as a letter (`letters`).
fn matchQwen(cps: []const u21, i: usize, letters: u8) usize {
    if (contraction(cps, i, true)) |end|
    {
        return end;
    }
    const c = classOf(cps[i]);
    if (c & (class_newline | class_letter | class_number) == 0 and is(cps, i + 1, letters))
    {
        return runOf(cps, i + 1, letters);
    }
    if (c & letters != 0)
    {
        return runOf(cps, i, letters);
    }
    if (c & class_number != 0)
    {
        return i + 1;
    }
    const s = if (cps[i] == ' ') i + 1 else i;
    if (isOtherThan(cps, s, letters))
    {
        return runOf(cps, runOfOtherThan(cps, s, letters), class_newline);
    }
    // A white space run containing a line break ends after its last break.
    const j = runOf(cps, i, class_space);
    var k = j;
    while (k > i)
    {
        k -= 1;
        if (classOf(cps[k]) & class_newline != 0)
        {
            return k + 1;
        }
    }
    return spaceRun(cps, i);
}

/// Collects vocabulary, added tokens, and merges, then builds the tokenizer.
const Builder = struct {
    arena: std.heap.ArenaAllocator,
    pattern: Pattern,
    /// Byte-level strings by id; null for ids that are added or unused.
    vocab: std.ArrayList(?[]const u8) = .empty,
    added: std.ArrayList(Added) = .empty,
    special_ids: std.ArrayList(u32) = .empty,
    by_text: std.StringHashMapUnmanaged(u32) = .empty,
    merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty,
    joined: std.ArrayList(u8) = .empty,

    fn init(gpa: std.mem.Allocator, pattern: Pattern) !Builder {
        return .{ .arena = std.heap.ArenaAllocator.init(gpa), .pattern = pattern };
    }

    fn setVocab(self: *Builder, id: u32, text: []const u8) !void {
        const a = self.arena.allocator();
        while (self.vocab.items.len <= id)
        {
            try self.vocab.append(a, null);
        }
        self.vocab.items[id] = text;
    }

    fn addAdded(self: *Builder, id: u32, text: []const u8, normalized: bool, special: bool) !void {
        const a = self.arena.allocator();
        while (self.vocab.items.len <= id)
        {
            try self.vocab.append(a, null);
        }
        try self.added.append(a, .{ .id = id, .text = text, .normalized = normalized });
        if (special)
        {
            try self.special_ids.append(a, id);
        }
    }

    fn indexVocab(self: *Builder) !void {
        const a = self.arena.allocator();
        for (self.vocab.items, 0..) |v, id|
        {
            if (v) |text|
            {
                try self.by_text.put(a, text, @intCast(id));
            }
        }
    }

    fn addMerge(self: *Builder, rank: u32, left: []const u8, right: []const u8) !void {
        const a = self.arena.allocator();
        const l = self.by_text.get(left) orelse return error.BadTokenizer;
        const r = self.by_text.get(right) orelse return error.BadTokenizer;
        self.joined.clearRetainingCapacity();
        try self.joined.appendSlice(a, left);
        try self.joined.appendSlice(a, right);
        const m = self.by_text.get(self.joined.items) orelse return error.BadTokenizer;
        const gop = try self.merges.getOrPut(a, pairKey(l, r));
        if (!gop.found_existing)
        {
            gop.value_ptr.* = .{ .rank = rank, .id = m };
        }
    }

    fn finish(self: *Builder) Error!Tokenizer {
        const a = self.arena.allocator();
        const n = self.vocab.items.len;
        const pieces = try a.alloc([]const u8, n);
        for (self.vocab.items, pieces) |v, *p|
        {
            p.* = if (v) |text| try byteLevelDecode(a, text) else "";
        }
        for (self.added.items) |t|
        {
            pieces[t.id] = t.text;
        }
        const special = try a.alloc(bool, n);
        @memset(special, false);
        for (self.special_ids.items) |id|
        {
            special[id] = true;
        }
        var byte_ids: [256]u32 = undefined;
        for (&byte_ids, 0..) |*id, byte|
        {
            var enc: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(byte_chars[byte], &enc) catch unreachable;
            // Bytes that never occur in UTF-8 may be missing; encode only takes valid UTF-8.
            id.* = self.by_text.get(enc[0..len]) orelse
                if (byte == 0xC0 or byte == 0xC1 or byte >= 0xF5) std.math.maxInt(u32) else return error.BadTokenizer;
        }
        return .{
            .arena = self.arena,
            .pattern = self.pattern,
            .pieces = pieces,
            .special = special,
            .byte_ids = byte_ids,
            .merges = self.merges,
            .added = self.added.items,
        };
    }
};

/// GPT-2's byte-level alphabet: printable bytes stand for themselves, the rest
/// map to code points from 256 up, in byte order.
const byte_chars: [256]u21 = blk: {
    var t: [256]u21 = undefined;
    var n: u21 = 0;
    for (0..256) |b|
    {
        if ((b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF))
        {
            t[b] = @intCast(b);
        }
        else
        {
            t[b] = 256 + n;
            n += 1;
        }
    }
    break :blk t;
};

const char_bytes: [324]i16 = blk: {
    var t: [324]i16 = @splat(-1);
    for (byte_chars, 0..) |cp, b|
    {
        t[cp] = b;
    }
    break :blk t;
};

/// A vocabulary string's bytes; characters outside the alphabet pass through as UTF-8.
fn byteLevelDecode(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var view = (std.unicode.Utf8View.init(text) catch return error.BadTokenizer).iterator();
    while (view.nextCodepointSlice()) |slice|
    {
        const cp = std.unicode.utf8Decode(slice) catch unreachable;
        if (cp < char_bytes.len and char_bytes[cp] >= 0)
        {
            try out.append(a, @intCast(char_bytes[cp]));
        }
        else
        {
            try out.appendSlice(a, slice);
        }
    }
    return out.toOwnedSlice(a);
}

/// Turns tokens into text as they arrive, holding back the bytes of a
/// character split across tokens. Invalid bytes become U+FFFD.
pub const Stream = struct {
    pending: [4]u8 = undefined,
    len: u8 = 0,

    pub fn push(self: *Stream, gpa: std.mem.Allocator, tok: *const Tokenizer, id: u32, out: *std.ArrayList(u8)) !void {
        for (tok.piece(id)) |byte|
        {
            self.pending[self.len] = byte;
            self.len += 1;
            try self.drain(gpa, out);
        }
    }

    /// Ends the stream, replacing an unfinished character.
    pub fn finish(self: *Stream, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        if (self.len > 0)
        {
            try out.appendSlice(gpa, "\u{FFFD}");
            self.len = 0;
        }
    }

    fn drain(self: *Stream, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        while (self.len > 0)
        {
            const need = std.unicode.utf8ByteSequenceLength(self.pending[0]) catch {
                try self.replaceFirst(gpa, out);
                continue;
            };
            if (self.len < need)
            {
                // Wait for more unless what we have already cannot start a character.
                if (self.len >= 2 and self.pending[self.len - 1] & 0xC0 != 0x80)
                {
                    try self.replaceFirst(gpa, out);
                    continue;
                }
                return;
            }
            if (std.unicode.utf8Decode(self.pending[0..need])) |_|
            {
                try out.appendSlice(gpa, self.pending[0..need]);
                self.shift(need);
            }
            else |_|
            {
                try self.replaceFirst(gpa, out);
            }
        }
    }

    fn replaceFirst(self: *Stream, gpa: std.mem.Allocator, out: *std.ArrayList(u8)) !void {
        try out.appendSlice(gpa, "\u{FFFD}");
        self.shift(1);
    }

    fn shift(self: *Stream, n: usize) void {
        std.mem.copyForwards(u8, self.pending[0 .. self.len - n], self.pending[n..self.len]);
        self.len -= @intCast(n);
    }
};

/// Appends the text of `ids` to `out`.
pub fn decode(gpa: std.mem.Allocator, tok: *const Tokenizer, ids: []const u32, out: *std.ArrayList(u8)) !void {
    var s: Stream = .{};
    for (ids) |id|
    {
        try s.push(gpa, tok, id, out);
    }
    try s.finish(gpa, out);
}

fn asObject(v: std.json.Value) ?std.json.ObjectMap {
    return switch (v) {
        .object => |o| o,
        else => null,
    };
}

fn typeOf(v: std.json.Value) ?[]const u8 {
    const o = asObject(v) orelse return null;
    return switch (o.get("type") orelse return null) {
        .string => |s| s,
        else => null,
    };
}

fn asId(v: std.json.Value) ?u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i < 1 << 24) @intCast(i) else null,
        else => null,
    };
}

fn asBool(v: ?std.json.Value) ?bool {
    return switch (v orelse return null) {
        .bool => |b| b,
        else => null,
    };
}

/// Which supported pre-tokenizer a `tokenizer.json` uses, if either.
fn jsonPattern(v: std.json.Value) ?Pattern {
    const t = typeOf(v) orelse return null;
    const o = asObject(v).?;
    if (std.mem.eql(u8, t, "ByteLevel"))
    {
        return if (asBool(o.get("use_regex")) orelse true) .gpt2 else null;
    }
    if (!std.mem.eql(u8, t, "Sequence"))
    {
        return null;
    }
    const seq = switch (o.get("pretokenizers") orelse return null) {
        .array => |arr| arr.items,
        else => return null,
    };
    if (seq.len != 2 or !std.mem.eql(u8, typeOf(seq[0]) orelse "", "Split") or !std.mem.eql(u8, typeOf(seq[1]) orelse "", "ByteLevel"))
    {
        return null;
    }
    const sp = asObject(seq[0]).?;
    const pat = asObject(sp.get("pattern") orelse return null) orelse return null;
    const re = switch (pat.get("Regex") orelse return null) {
        .string => |s| s,
        else => return null,
    };
    const behavior = switch (sp.get("behavior") orelse return null) {
        .string => |s| s,
        else => return null,
    };
    const which: Pattern = if (std.mem.eql(u8, re, qwen2_regex)) .qwen2 else if (std.mem.eql(u8, re, qwen35_regex)) .qwen35 else return null;
    if (!std.mem.eql(u8, behavior, "Isolated") or (asBool(sp.get("invert")) orelse false))
    {
        return null;
    }
    if (asBool(asObject(seq[1]).?.get("use_regex")) orelse true)
    {
        return null;
    }
    return which;
}

fn expectSplit(pattern: Pattern, text: []const u8, want: []const []const u8) !void {
    const gpa = std.testing.allocator;
    var s: Scratch = .{};
    defer s.deinit(gpa);
    try split(gpa, pattern, text, &s);
    const b = s.bounds.items;
    try std.testing.expectEqual(want.len, b.len - 1);
    for (want, b[0 .. b.len - 1], b[1..]) |w, lo, hi|
    {
        try std.testing.expectEqualStrings(w, text[lo..hi]);
    }
}

test "gpt2 pattern splits like Hugging Face" {
    try expectSplit(.gpt2, "Hello world's  test\n\n x", &.{ "Hello", " world", "'s", " ", " test", "\n\n", " x" });
    try expectSplit(.gpt2, "x 12345 !!\n\n", &.{ "x", " 12345", " !!", "\n\n" });
    try expectSplit(.gpt2, "It'S  ", &.{ "It", "'", "S", "  " });
}

test "qwen35 pattern keeps combining marks with letters" {
    // U+0301 (combining acute) after a letter, and alone after punctuation.
    try expectSplit(.qwen35, "cafe\u{301} !\u{301}x", &.{ "cafe\u{301}", " !", "\u{301}x" });
    try expectSplit(.qwen2, "cafe\u{301} !\u{301}x", &.{ "cafe", "\u{301}", " !\u{301}", "x" });
    try expectSplit(.qwen35, "x 12345 !!\n\n", &.{ "x", " ", "1", "2", "3", "4", "5", " !!\n\n" });
}

test "qwen2 pattern splits like Hugging Face" {
    try expectSplit(.qwen2, "Hello world's  test\n\n x", &.{ "Hello", " world", "'s", " ", " test", "\n\n", " x" });
    try expectSplit(.qwen2, "x 12345 !!\n\n", &.{ "x", " ", "1", "2", "3", "4", "5", " !!\n\n" });
    try expectSplit(.qwen2, "It'S (hi)\n  \n\tok", &.{ "It", "'S", " (", "hi", ")\n", "  \n", "\tok" });
}

test "BPE merges by rank, leftmost first, and splits out added tokens" {
    const gpa = std.testing.allocator;
    // Every byte is a token with its own value as id, then four merged tokens.
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.appendSlice(gpa,
        \\{"normalizer": {"type": "NFC"},
        \\ "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        \\ "added_tokens": [{"id": 260, "content": "<|end|>", "normalized": false, "special": true},
        \\                  {"id": 261, "content": "  ", "normalized": true, "special": false}],
        \\ "model": {"type": "BPE", "merges": ["a b", "a a", "aa b", "Ġ a"],
        \\           "vocab": {"aa": 256, "ab": 257, "aab": 258, "Ġa": 259
    );
    for (byte_chars, 0..) |cp, b|
    {
        var enc: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(cp, &enc);
        const esc = if (cp == '"' or cp == '\\') "\\" else "";
        const line = try std.fmt.allocPrint(gpa, ", \"{s}{s}\": {d}", .{ esc, enc[0..n], b });
        defer gpa.free(line);
        try json.appendSlice(gpa, line);
    }
    try json.appendSlice(gpa, "}}}");

    var tok = try Tokenizer.fromJson(gpa, json.items);
    defer tok.deinit();
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "aaab a<|end|>b  a", &ids);
    try std.testing.expectEqualSlices(u32, &.{ 256, 257, 259, 260, 'b', 261, 'a' }, ids.items);
    try std.testing.expect(tok.isSpecial(260) and !tok.isSpecial(261));
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try decode(gpa, &tok, ids.items, &text);
    try std.testing.expectEqualStrings("aaab a<|end|>b  a", text.items);
}

test "streaming decode holds back split characters" {
    var tok: Tokenizer = undefined;
    const pieces = [_][]const u8{ "caf", "\xC3", "\xA9!", "\xFF" };
    tok.pieces = &pieces;
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var s: Stream = .{};
    try s.push(gpa, &tok, 0, &out);
    try s.push(gpa, &tok, 1, &out);
    try std.testing.expectEqualStrings("caf", out.items);
    try s.push(gpa, &tok, 2, &out);
    try s.push(gpa, &tok, 3, &out);
    try s.finish(gpa, &out);
    try std.testing.expectEqualStrings("caf\u{e9}!\u{FFFD}", out.items);
}
