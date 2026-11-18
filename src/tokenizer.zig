//! Byte-level BPE tokenizer (TASK-30) for OLMoE-1B-7B, Qwen3-MoE and Qwen3.5-MoE, read
//! from the model's GGUF metadata or from a Hugging Face `tokenizer.json`.
//! Matches Hugging Face `tokenizers`: added tokens are split out first (those
//! marked not normalized from the raw text, the rest after NFC), the remaining
//! text is split by the model's pre-tokenizer pattern, and each piece is
//! merged by BPE over its UTF-8 bytes.
//!
//! SentencePiece vocabularies (TASK-37, GGUF model "llama", as Mixtral's) take
//! another path: spaces become U+2581, one is put in front of the text, and BPE
//! merges characters, with bytes as tokens for characters not in the vocabulary.
//! The GGUF has scores instead of merges; every split of a token into two tokens
//! is a merge, ranked by the token's score as Hugging Face's converter does.
//!
//! The patterns are matched by hand instead of with a regex engine:
//!   gpt2 (OLMoE): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!   qwen2 (Qwen3): (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//!                  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   qwen35 (Qwen3.5, TASK-29): qwen2 with combining marks (\p{M}) counted
//!                  with letters, and so left out of the punctuation runs.
//!   glm4 (GLM-5.3-Flash, TASK-32): qwen2 with numbers in runs of up to three
//!                  digits. Its BPE also ignores merges: a piece that is a whole
//!                  vocabulary token is that token.
//!   deepseek_v3 (TASK-37): three splits in turn, each piece of one split by the
//!                  next: \p{N}{1,3}, then runs of CJK ideographs and kana, then
//!                  [!-/:-@[-`{-~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
//!                   ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!                  Text no alternative matches is a piece of its own.
//!   gpt4o (gpt-oss, TASK-37): words split by case,
//!                  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|...)?|
//!                  [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|...)?|
//!                  \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!                  with merges ignored as for glm4.
//!   kimi_k2 (Kimi K2 and Moonlight, TASK-39): gpt4o with Han split out, as
//!                  Moonshot's tiktoken pattern: [\p{Han}]+ first, the word forms' letters
//!                  excluding Han, and punctuation runs ending in [\r\n]* only.
const std = @import("std");
const gguf = @import("gguf.zig");
const unicode = @import("unicode.zig");

/// `none` is SentencePiece's: the text is not split.
pub const Pattern = enum { gpt2, qwen2, qwen35, glm4, deepseek_v3, gpt4o, kimi_k2, none };

/// When SentencePiece puts U+2581 in front of the text: never, only at its start, or before every piece between added tokens.
pub const Prepend = enum { never, first, always };

/// What `tokenizer.ggml.pre` names each pattern; the first spelling is the canonical one.
const pre_names = [_]struct { name: []const u8, pattern: Pattern }{
    .{ .name = "olmo", .pattern = .gpt2 },
    .{ .name = "qwen2", .pattern = .qwen2 },
    .{ .name = "qwen35", .pattern = .qwen35 },
    .{ .name = "glm4", .pattern = .glm4 },
    .{ .name = "deepseek-v3", .pattern = .deepseek_v3 },
    .{ .name = "gpt-4o", .pattern = .gpt4o },
    .{ .name = "kimi-k2", .pattern = .kimi_k2 },
};

fn patternOf(pre: []const u8) ?Pattern {
    for (pre_names) |p|
    {
        if (std.mem.eql(u8, p.name, pre))
        {
            return p.pattern;
        }
    }
    return null;
}

pub const Error = error{ UnsupportedTokenizer, BadTokenizer, InvalidUtf8, OutOfMemory };

const Added = struct { id: u32, text: []const u8, normalized: bool };
const Merge = struct { rank: u32, id: u32 };

const qwen2_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const qwen35_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const glm4_regex = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

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
    /// The unknown, padding and end-of-turn tokens, when the GGUF names them (TASK-38).
    unk: ?u32 = null,
    pad: ?u32 = null,
    eot: ?u32 = null,
    chat_template: ?[]const u8 = null,
    /// With `ignore_merges`, each vocabulary token's bytes to its id: a piece found here skips BPE.
    whole: std.StringHashMapUnmanaged(u32) = .empty,
    /// Whether text is NFC-normalized before splitting; GLM-5.3-Flash's tokenizer has no normalizer.
    nfc: bool = true,
    /// Whether a prompt starts with `bos` (`tokenizer.ggml.add_bos_token`), as Hugging Face's `encode` adds it.
    add_bos: bool = false,
    /// SentencePiece: each character's id, for characters that are vocabulary tokens.
    chars: std.StringHashMapUnmanaged(u32) = .empty,
    prepend: Prepend = .never,
    /// Whether decoding drops one space at the start of the text, as SentencePiece's decoder does.
    strip_space: bool = false,

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// From a GGUF header's `tokenizer.*` keys. Copies what it keeps, so the
    /// header may be freed afterwards.
    pub fn fromGguf(gpa: std.mem.Allocator, h: gguf.Header) Error!Tokenizer {
        const model = h.getString("tokenizer.ggml.model") orelse return error.BadTokenizer;
        if (std.mem.eql(u8, model, "llama"))
        {
            return fromGgufSpm(gpa, h);
        }
        if (!std.mem.eql(u8, model, "gpt2"))
        {
            std.log.err("tokenizer: GGUF tokenizer model \"{s}\" is not supported (gpt2 and llama are)", .{model});
            return error.UnsupportedTokenizer;
        }
        const pre = h.getString("tokenizer.ggml.pre") orelse "(none)";
        const pattern = patternOf(pre) orelse {
            std.log.err("tokenizer: pre-tokenizer \"{s}\" is not supported (olmo, qwen2, qwen35, glm4, deepseek-v3, gpt-4o and kimi-k2 are)", .{pre});
            return error.UnsupportedTokenizer;
        };

        var b = try Builder.init(gpa, pattern);
        // A GGUF does not record ignore_merges; GLM-5.3-Flash's and gpt-oss's tokenizer.json set it.
        b.ignore_merges = pattern == .glm4 or pattern == .gpt4o or pattern == .kimi_k2;
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
        const common = try Common.read(a, h);
        var tok = try b.finish();
        // A GGUF does not record the normalizer: OLMoE's and Qwen's are NFC, the newer ones have none.
        tok.nfc = pattern == .gpt2 or pattern == .qwen2 or pattern == .qwen35;
        common.apply(&tok);
        return tok;
    }

    /// The keys every GGUF tokenizer shares: special ids, BOS, and the chat template.
    const Common = struct {
        bos: ?u32,
        eos: ?u32,
        unk: ?u32,
        pad: ?u32,
        eot: ?u32,
        add_bos: bool,
        chat_template: ?[]const u8,

        fn read(a: std.mem.Allocator, h: gguf.Header) !Common {
            const id = struct {
                fn f(h_: gguf.Header, key: []const u8) ?u32 {
                    const v = h_.getInt(key) orelse return null;
                    return if (v <= std.math.maxInt(u32)) @intCast(v) else null;
                }
            }.f;
            const bos = id(h, "tokenizer.ggml.bos_token_id");
            return .{
                .bos = bos,
                .eos = id(h, "tokenizer.ggml.eos_token_id"),
                .unk = id(h, "tokenizer.ggml.unknown_token_id"),
                .pad = id(h, "tokenizer.ggml.padding_token_id"),
                .eot = id(h, "tokenizer.ggml.eot_token_id"),
                .add_bos = bos != null and (h.getBool("tokenizer.ggml.add_bos_token") orelse false),
                .chat_template = if (h.getString("tokenizer.chat_template")) |t| try a.dupe(u8, t) else null,
            };
        }

        fn apply(self: Common, tok: *Tokenizer) void {
            tok.bos = self.bos;
            tok.eos = self.eos;
            tok.unk = self.unk;
            tok.pad = self.pad;
            tok.eot = self.eot;
            tok.add_bos = self.add_bos;
            tok.chat_template = self.chat_template;
        }
    };

    /// A SentencePiece vocabulary (`tokenizer.ggml.model` "llama") with scores and byte tokens.
    fn fromGgufSpm(gpa: std.mem.Allocator, h: gguf.Header) Error!Tokenizer {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var pieces: std.ArrayList([]const u8) = .empty;
        var special: std.ArrayList(bool) = .empty;
        var added: std.ArrayList(Added) = .empty;
        var chars: std.StringHashMapUnmanaged(u32) = .empty;
        var by_text: std.StringHashMapUnmanaged(u32) = .empty;
        var byte_ids: [256]u32 = @splat(std.math.maxInt(u32));

        var toks = h.strings("tokenizer.ggml.tokens") orelse return error.BadTokenizer;
        var id: u32 = 0;
        while (toks.next() catch return error.BadTokenizer) |t|
        {
            // 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
            const kind = h.intAt("tokenizer.ggml.token_type", id) orelse 1;
            const text = try a.dupe(u8, t);
            var p: []const u8 = text;
            var is_special = false;
            switch (kind)
            {
                2, 3, 4 => {
                    try added.append(a, .{ .id = id, .text = text, .normalized = kind == 4 });
                    is_special = kind != 4;
                },
                5 => p = "",
                6 => {
                    const byte = byteToken(text) orelse return error.BadTokenizer;
                    byte_ids[byte] = id;
                    p = try a.dupe(u8, &.{byte});
                },
                else => {
                    p = try std.mem.replaceOwned(u8, a, text, "\u{2581}", " ");
                    try by_text.put(a, text, id);
                    if (text.len > 0 and (std.unicode.utf8ByteSequenceLength(text[0]) catch 0) == text.len)
                    {
                        try chars.put(a, text, id);
                    }
                },
            }
            try pieces.append(a, p);
            try special.append(a, is_special);
            id += 1;
        }
        for (byte_ids) |b|
        {
            if (b == std.math.maxInt(u32))
            {
                return error.BadTokenizer;
            }
        }

        // Every split of a token into two tokens, ranked by the token's score, then by the ids.
        const Cand = struct { score: f64, merged: u32, l: u32, r: u32 };
        var cands: std.ArrayList(Cand) = .empty;
        var it = by_text.iterator();
        while (it.next()) |e|
        {
            const text = e.key_ptr.*;
            const score = h.floatAt("tokenizer.ggml.scores", e.value_ptr.*) orelse 0;
            var k: usize = 1;
            while (k < text.len)
            {
                const l = by_text.get(text[0..k]);
                const r = by_text.get(text[k..]);
                if (l != null and r != null)
                {
                    try cands.append(a, .{ .score = score, .merged = e.value_ptr.*, .l = l.?, .r = r.? });
                }
                k += 1;
            }
        }
        std.mem.sort(Cand, cands.items, {}, struct {
            fn less(_: void, x: Cand, y: Cand) bool {
                if (x.score != y.score)
                {
                    return x.score > y.score;
                }
                if (x.merged != y.merged)
                {
                    return x.merged < y.merged;
                }
                return x.l < y.l or (x.l == y.l and x.r < y.r);
            }
        }.less);
        var merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty;
        for (cands.items, 0..) |c, rank|
        {
            const gop = try merges.getOrPut(a, pairKey(c.l, c.r));
            if (!gop.found_existing)
            {
                gop.value_ptr.* = .{ .rank = @intCast(rank), .id = c.merged };
            }
        }

        const scheme = h.getString("tokenizer.ggml.metaspace.prepend_scheme") orelse "first";
        const prepend: Prepend = if (h.getBool("tokenizer.ggml.add_space_prefix") == false)
            .never
        else if (std.mem.eql(u8, scheme, "always"))
            .always
        else if (std.mem.eql(u8, scheme, "never"))
            .never
        else
            .first;
        const common = try Common.read(a, h);
        var tok: Tokenizer = .{
            .arena = arena,
            .pattern = .none,
            .pieces = pieces.items,
            .special = special.items,
            .byte_ids = byte_ids,
            .merges = merges,
            .added = added.items,
            .nfc = false,
            .chars = chars,
            .prepend = prepend,
            .strip_space = prepend != .never,
        };
        common.apply(&tok);
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

        var nfc = false;
        if (root.get("normalizer")) |n|
        {
            if (n != .null and !std.mem.eql(u8, typeOf(n) orelse "", "NFC"))
            {
                return error.UnsupportedTokenizer;
            }
            nfc = n != .null;
        }
        const pattern = jsonPattern(root.get("pre_tokenizer") orelse return error.UnsupportedTokenizer) orelse return error.UnsupportedTokenizer;
        const model = asObject(root.get("model") orelse return error.BadTokenizer) orelse return error.BadTokenizer;
        if (!std.mem.eql(u8, typeOf(.{ .object = model }) orelse "", "BPE"))
        {
            return error.UnsupportedTokenizer;
        }

        var b = try Builder.init(gpa, pattern);
        errdefer b.arena.deinit();
        b.ignore_merges = asBool(model.get("ignore_merges")) orelse false;
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
        var tok = try b.finish();
        tok.nfc = nfc;
        return tok;
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
                if (self.nfc)
                {
                    try unicode.nfc(gpa, text[pos..end], &norm);
                }
                else
                {
                    try norm.appendSlice(gpa, text[pos..end]);
                }
                var npos: usize = 0;
                while (npos < norm.items.len)
                {
                    const nm = self.findAdded(norm.items, npos, true);
                    const nend = if (nm) |x| x.start else norm.items.len;
                    try self.encodePlain(gpa, norm.items[npos..nend], pos == 0 and npos == 0, &s, out);
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

    /// Text between added tokens; `first` if it starts the whole text.
    fn encodePlain(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, first: bool, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        if (text.len == 0)
        {
            return;
        }
        if (self.pattern == .none)
        {
            return self.spm(gpa, text, first, s, out);
        }
        try split(gpa, self.pattern, text, s);
        for (s.bounds.items[0 .. s.bounds.items.len - 1], s.bounds.items[1..]) |lo, hi|
        {
            try self.bpe(gpa, text[lo..hi], s, out);
        }
    }

    fn bpe(self: *const Tokenizer, gpa: std.mem.Allocator, word: []const u8, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        if (self.whole.get(word)) |id|
        {
            return out.append(gpa, id);
        }
        s.ids.clearRetainingCapacity();
        for (word) |byte|
        {
            try s.ids.append(gpa, self.byte_ids[byte]);
        }
        try self.merge(gpa, s, out);
    }

    /// SentencePiece: spaces become U+2581, maybe one goes in front, and each
    /// character not in the vocabulary becomes its bytes' tokens before merging.
    fn spm(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, first: bool, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        const space = "\u{2581}";
        s.word.clearRetainingCapacity();
        const lead = text[0] == ' ' or std.mem.startsWith(u8, text, space);
        if (!lead and (self.prepend == .always or (self.prepend == .first and first)))
        {
            try s.word.appendSlice(gpa, space);
        }
        for (text) |byte|
        {
            if (byte == ' ')
            {
                try s.word.appendSlice(gpa, space);
            }
            else
            {
                try s.word.append(gpa, byte);
            }
        }
        s.ids.clearRetainingCapacity();
        var view = std.unicode.Utf8View.initUnchecked(s.word.items).iterator();
        while (view.nextCodepointSlice()) |c|
        {
            if (self.chars.get(c)) |id|
            {
                try s.ids.append(gpa, id);
            }
            else
            {
                for (c) |byte|
                {
                    try s.ids.append(gpa, self.byte_ids[byte]);
                }
            }
        }
        try self.merge(gpa, s, out);
    }

    /// BPE over the ids in `s.ids`: the lowest-ranked mergeable pair first, leftmost on ties.
    fn merge(self: *const Tokenizer, gpa: std.mem.Allocator, s: *Scratch, out: *std.ArrayList(u32)) Error!void {
        const none = std.math.maxInt(u32);
        const n = s.ids.items.len;
        s.next.clearRetainingCapacity();
        s.prev.clearRetainingCapacity();
        s.heap.clearRetainingCapacity();
        for (0..n) |i|
        {
            try s.prev.append(gpa, if (i == 0) none else @intCast(i - 1));
            try s.next.append(gpa, if (i + 1 == n) none else @intCast(i + 1));
        }
        const ids = s.ids.items;
        const prev = s.prev.items;
        const next = s.next.items;
        for (0..n -| 1) |i|
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
    word: std.ArrayList(u8) = .empty,

    const Item = struct { rank: u32, pos: u32, a: u32, b: u32, id: u32 };

    fn deinit(self: *Scratch, gpa: std.mem.Allocator) void {
        self.word.deinit(gpa);
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

const Class = u16;
const class_letter: Class = 1;
const class_number: Class = 2;
const class_space: Class = 4;
const class_newline: Class = 8;
const class_mark: Class = 16;
const class_upper: Class = 32;
const class_lower: Class = 64;
const class_other_letter: Class = 128;
const class_punct: Class = 256;
const class_symbol: Class = 512;
const class_han: Class = 1024;
/// gpt-oss's two letter classes: the start of a word, and the rest of it.
const class_cased_start = class_upper | class_other_letter | class_mark;
const class_cased_rest = class_lower | class_other_letter | class_mark;

fn classOf(cp: u21) Class {
    if (cp < 0x80)
    {
        return switch (cp) {
            'a'...'z' => class_letter | class_lower,
            'A'...'Z' => class_letter | class_upper,
            '0'...'9' => class_number,
            '\r', '\n' => class_space | class_newline,
            0x09, 0x0B, 0x0C, ' ' => class_space,
            '$', '+', '<', '=', '>', '^', '`', '|', '~' => class_symbol,
            '!', '"', '#', '%', '&', '\'', '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '@', '[', '\\', ']', '_', '{', '}' => class_punct,
            else => 0,
        };
    }
    var c: Class = 0;
    if (unicode.isLetter(cp))
    {
        c |= class_letter;
        if (unicode.isUpper(cp))
        {
            c |= class_upper;
        }
        else if (unicode.isLower(cp))
        {
            c |= class_lower;
        }
        else if (unicode.isOtherLetter(cp))
        {
            c |= class_other_letter;
        }
    }
    else if (unicode.isNumber(cp))
    {
        c |= class_number;
    }
    else if (unicode.isMark(cp))
    {
        c |= class_mark;
    }
    else if (unicode.isPunctuation(cp))
    {
        c |= class_punct;
    }
    else if (unicode.isSymbol(cp))
    {
        c |= class_symbol;
    }
    if (unicode.isSpace(cp))
    {
        c |= class_space;
    }
    if (unicode.isHan(cp))
    {
        c |= class_han;
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
    if (pattern == .deepseek_v3)
    {
        try splitDeepseek(gpa, cps, s);
        try s.bounds.append(gpa, @intCast(text.len));
        return;
    }
    var i: usize = 0;
    while (i < cps.len)
    {
        try s.bounds.append(gpa, s.offs.items[i]);
        const end = switch (pattern) {
            .gpt2 => matchGpt2(cps, i),
            .qwen2 => matchQwen(cps, i, class_letter),
            .qwen35 => matchQwen(cps, i, class_letter | class_mark),
            .glm4 => matchQwenDigits(cps, i, class_letter, 3),
            .gpt4o => matchGpt4o(cps, i),
            .kimi_k2 => matchKimi(cps, i),
            .deepseek_v3, .none => unreachable,
        };
        std.debug.assert(end > i);
        i = end;
    }
    try s.bounds.append(gpa, @intCast(text.len));
}

/// DeepSeek V3's three splits: numbers, then CJK runs in what is left, then
/// its pattern in what is left of that. Each split sees only its own piece.
fn splitDeepseek(gpa: std.mem.Allocator, cps: []const u21, s: *Scratch) !void {
    var i: usize = 0;
    while (i < cps.len)
    {
        if (is(cps, i, class_number))
        {
            try s.bounds.append(gpa, s.offs.items[i]);
            i = @min(runOf(cps, i, class_number), i + 3);
            continue;
        }
        var j = i;
        while (j < cps.len and !is(cps, j, class_number))
        {
            j += 1;
        }
        while (i < j)
        {
            if (isCjk(cps[i]))
            {
                try s.bounds.append(gpa, s.offs.items[i]);
                while (i < j and isCjk(cps[i]))
                {
                    i += 1;
                }
                continue;
            }
            var k = i;
            while (k < j and !isCjk(cps[k]))
            {
                k += 1;
            }
            try splitBy(gpa, cps[0..k], i, s, matchDeepseek);
            i = k;
        }
    }
}

/// Splits `cps[from..]` by a matcher; text where nothing matches is a piece of its own.
fn splitBy(gpa: std.mem.Allocator, cps: []const u21, from: usize, s: *Scratch, comptime match: fn ([]const u21, usize) ?usize) !void {
    var i = from;
    while (i < cps.len)
    {
        try s.bounds.append(gpa, s.offs.items[i]);
        if (match(cps, i)) |end|
        {
            i = end;
            continue;
        }
        i += 1;
        while (i < cps.len and match(cps, i) == null)
        {
            i += 1;
        }
    }
}

/// DeepSeek V3's CJK class: the common ideographs to U+9FA5, hiragana and katakana.
fn isCjk(cp: u21) bool {
    return (cp >= 0x4E00 and cp <= 0x9FA5) or (cp >= 0x3040 and cp <= 0x30FF);
}

fn isAsciiLetter(cps: []const u21, i: usize) bool {
    return i < cps.len and cps[i] < 0x80 and std.ascii.isAlphabetic(@intCast(cps[i]));
}

fn matchDeepseek(cps: []const u21, i: usize) ?usize {
    const c = classOf(cps[i]);
    // ASCII punctuation or symbol, then ASCII letters.
    if (cps[i] < 0x80 and c & (class_punct | class_symbol) != 0 and isAsciiLetter(cps, i + 1))
    {
        var j = i + 1;
        while (isAsciiLetter(cps, j))
        {
            j += 1;
        }
        return j;
    }
    const word = class_letter | class_mark;
    if (c & (class_newline | class_letter | class_punct | class_symbol) == 0 and is(cps, i + 1, word))
    {
        return runOf(cps, i + 1, word);
    }
    if (c & word != 0)
    {
        return runOf(cps, i, word);
    }
    const ps = class_punct | class_symbol;
    const s = if (cps[i] == ' ' and is(cps, i + 1, ps)) i + 1 else i;
    if (is(cps, s, ps))
    {
        return runOf(cps, runOf(cps, s, ps), class_newline);
    }
    if (c & class_space != 0)
    {
        return spaceOrBreaks(cps, i);
    }
    return null;
}

/// White space at `i`: through its last line break if it has one, else as `spaceRun`.
fn spaceOrBreaks(cps: []const u21, i: usize) usize {
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

/// One character that may lead a gpt-oss word: not a line break, letter or number.
fn isWordLead(cps: []const u21, i: usize) bool {
    return i < cps.len and classOf(cps[i]) & (class_newline | class_letter | class_number) == 0;
}

/// Whether `cps[i]` is in `class` and not in `except`.
fn isBut(cps: []const u21, i: usize, class: Class, except: Class) bool {
    return i < cps.len and classOf(cps[i]) & class != 0 and classOf(cps[i]) & except == 0;
}

fn runOfBut(cps: []const u21, from: usize, class: Class, except: Class) usize {
    var j = from;
    while (isBut(cps, j, class, except))
    {
        j += 1;
    }
    return j;
}

/// gpt-oss's first word form: optional capitals, then at least one lower-case letter, none in `except`.
fn casedLower(cps: []const u21, s: usize, except: Class) ?usize {
    const u = runOfBut(cps, s, class_cased_start, except);
    var end: usize = undefined;
    if (isBut(cps, u, class_cased_rest, except))
    {
        end = runOfBut(cps, u, class_cased_rest, except);
    }
    else
    {
        // Give back capitals until one can end the word.
        var p = u;
        while (true)
        {
            if (p == s)
            {
                return null;
            }
            p -= 1;
            if (isBut(cps, p, class_cased_rest, except))
            {
                break;
            }
        }
        end = p + 1;
    }
    return contraction(cps, end, true) orelse end;
}

/// gpt-oss's second word form: at least one capital, then any lower-case letters, none in `except`.
fn casedUpper(cps: []const u21, s: usize, except: Class) ?usize {
    if (!isBut(cps, s, class_cased_start, except))
    {
        return null;
    }
    const end = runOfBut(cps, runOfBut(cps, s, class_cased_start, except), class_cased_rest, except);
    return contraction(cps, end, true) orelse end;
}

/// gpt-oss's two word forms at `i`, each with an optional lead, with no letter in `except`.
fn casedWord(cps: []const u21, i: usize, except: Class) ?usize {
    const lead = isWordLead(cps, i);
    if (lead)
    {
        if (casedLower(cps, i + 1, except)) |end|
        {
            return end;
        }
    }
    if (casedLower(cps, i, except)) |end|
    {
        return end;
    }
    if (lead)
    {
        if (casedUpper(cps, i + 1, except)) |end|
        {
            return end;
        }
    }
    return casedUpper(cps, i, except);
}

/// Kimi's pattern: a run of Han first, then gpt4o's alternatives with Han left out of
/// words and punctuation runs ending in line breaks only.
fn matchKimi(cps: []const u21, i: usize) usize {
    if (is(cps, i, class_han))
    {
        return runOf(cps, i, class_han);
    }
    if (casedWord(cps, i, class_han)) |end|
    {
        return end;
    }
    if (is(cps, i, class_number))
    {
        return @min(runOf(cps, i, class_number), i + 3);
    }
    const s = if (cps[i] == ' ' and isOther(cps, i + 1)) i + 1 else i;
    if (isOther(cps, s))
    {
        var j = runOfOther(cps, s);
        while (j < cps.len and (cps[j] == '\r' or cps[j] == '\n'))
        {
            j += 1;
        }
        return j;
    }
    return spaceOrBreaks(cps, i);
}

fn matchGpt4o(cps: []const u21, i: usize) usize {
    if (casedWord(cps, i, 0)) |end|
    {
        return end;
    }
    if (is(cps, i, class_number))
    {
        return @min(runOf(cps, i, class_number), i + 3);
    }
    const s = if (cps[i] == ' ' and isOther(cps, i + 1)) i + 1 else i;
    if (isOther(cps, s))
    {
        var j = runOfOther(cps, s);
        while (j < cps.len and (cps[j] == '\r' or cps[j] == '\n' or cps[j] == '/'))
        {
            j += 1;
        }
        return j;
    }
    return spaceOrBreaks(cps, i);
}

fn is(cps: []const u21, i: usize, class: Class) bool {
    return i < cps.len and classOf(cps[i]) & class != 0;
}

fn isOther(cps: []const u21, i: usize) bool {
    return isOtherThan(cps, i, class_letter);
}

/// Neither `letters`, a number, nor white space.
fn isOtherThan(cps: []const u21, i: usize, letters: Class) bool {
    return i < cps.len and classOf(cps[i]) & (letters | class_number | class_space) == 0;
}

fn runOf(cps: []const u21, from: usize, class: Class) usize {
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

fn runOfOtherThan(cps: []const u21, from: usize, letters: Class) usize {
    var j = from;
    while (isOtherThan(cps, j, letters))
    {
        j += 1;
    }
    return j;
}

/// End of the contraction at `i` ('s, 't, 're, 've, 'm, 'll, 'd), or null.
fn contraction(cps: []const u21, i: usize, fold: bool) ?usize {
    if (i + 1 >= cps.len or cps[i] != '\'')
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
fn matchQwen(cps: []const u21, i: usize, letters: Class) usize {
    return matchQwenDigits(cps, i, letters, 1);
}

/// The qwen2 family with numbers in runs of up to `digits` (1 for qwen2, 3 for glm4).
fn matchQwenDigits(cps: []const u21, i: usize, letters: Class, digits: usize) usize {
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
        return @min(runOf(cps, i, class_number), i + digits);
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
    ignore_merges: bool = false,

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
        var whole: std.StringHashMapUnmanaged(u32) = .empty;
        if (self.ignore_merges)
        {
            for (self.vocab.items, 0..) |v, id|
            {
                if (v != null and pieces[id].len > 0)
                {
                    try whole.put(a, pieces[id], @intCast(id));
                }
            }
        }
        return .{
            .arena = self.arena,
            .pattern = self.pattern,
            .pieces = pieces,
            .special = special,
            .byte_ids = byte_ids,
            .merges = self.merges,
            .added = self.added.items,
            .whole = whole,
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

/// The byte a SentencePiece byte token such as "<0x0A>" stands for.
fn byteToken(text: []const u8) ?u8 {
    if (text.len != 6 or !std.mem.startsWith(u8, text, "<0x") or text[5] != '>')
    {
        return null;
    }
    return std.fmt.parseInt(u8, text[3..5], 16) catch null;
}

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
    started: bool = false,

    pub fn push(self: *Stream, gpa: std.mem.Allocator, tok: *const Tokenizer, id: u32, out: *std.ArrayList(u8)) !void {
        var p = tok.piece(id);
        if (!self.started and p.len > 0)
        {
            self.started = true;
            if (tok.strip_space and p[0] == ' ')
            {
                p = p[1..];
            }
        }
        for (p) |byte|
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
    const which: Pattern = if (std.mem.eql(u8, re, qwen2_regex)) .qwen2 else if (std.mem.eql(u8, re, qwen35_regex)) .qwen35 else if (std.mem.eql(u8, re, glm4_regex)) .glm4 else return null;
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

test "deepseek_v3 pattern splits like Hugging Face" {
    // Pieces from Hugging Face's pre_tokenize_str on DeepSeek-V3's tokenizer.json.
    try expectSplit(.deepseek_v3, "a1b22c333d4444 x", &.{ "a", "1", "b", "22", "c", "333", "d", "444", "4", " x" });
    try expectSplit(.deepseek_v3, "日本語abc #include <x.h>", &.{ "日本語", "abc", " #", "include", " <", "x", ".h", ">" });
    try expectSplit(.deepseek_v3, "!!abc ?? \n\n x\x01\x02y", &.{ "!!", "abc", " ??", " \n\n", " x", "\x01", "\x02y" });
}

test "gpt4o pattern splits like Hugging Face" {
    // Pieces from Hugging Face's pre_tokenize_str on gpt-oss's tokenizer.json.
    try expectSplit(.gpt4o, "HelloWORLDfoo DON'T can't", &.{ "Hello", "WORLDfoo", " DON'T", " can't" });
    try expectSplit(.gpt4o, "path/to\n x  y", &.{ "path", "/to", "\n", " x", " ", " y" });
    try expectSplit(.gpt4o, "\u{1C5}ungla \u{2B0}ello 12345", &.{ "\u{1C5}ungla", " \u{2B0}ello", " ", "123", "45" });
}

test "kimi_k2 pattern splits like Moonshot's" {
    // Pieces from the regex package (V1, for its && sets) on tokenization_moonshot.py's pat_str.
    try expectSplit(.kimi_k2, "Hello\u{4E16}\u{754C}abc\u{4E2D}\u{6587}DEF can'T", &.{ "Hello", "\u{4E16}\u{754C}", "abc", "\u{4E2D}\u{6587}", "DEF", " can'T" });
    try expectSplit(.kimi_k2, "\u{2E80}!x\u{3007}\u{3007}1 \u{4E00}\u{4E01}. a/b\n\n c", &.{ "\u{2E80}", "!x", "\u{3007}\u{3007}", "1", " ", "\u{4E00}\u{4E01}", ".", " a", "/b", "\n\n", " c" });
}

test "an unknown pre-tokenizer is refused by name" {
    try std.testing.expectEqual(Pattern.deepseek_v3, patternOf("deepseek-v3").?);
    try std.testing.expectEqual(@as(?Pattern, null), patternOf("llama-bpe"));
}

/// A GGUF header with only the given metadata, for the SentencePiece test.
fn testGguf(gpa: std.mem.Allocator, tokens: []const []const u8, scores: []const f32, types: []const i32, add_bos: bool) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(gpa);
    const put = struct {
        fn int(l: *std.ArrayList(u8), g: std.mem.Allocator, comptime T: type, v: T) !void {
            var buf: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &buf, v, .little);
            try l.appendSlice(g, &buf);
        }
        fn str(l: *std.ArrayList(u8), g: std.mem.Allocator, s: []const u8) !void {
            try int(l, g, u64, s.len);
            try l.appendSlice(g, s);
        }
    };
    try b.appendSlice(gpa, "GGUF");
    try put.int(&b, gpa, u32, 3);
    try put.int(&b, gpa, u64, 0);
    try put.int(&b, gpa, u64, 7);
    try put.str(&b, gpa, "tokenizer.ggml.model");
    try put.int(&b, gpa, u32, 8);
    try put.str(&b, gpa, "llama");
    try put.str(&b, gpa, "tokenizer.ggml.tokens");
    try put.int(&b, gpa, u32, 9);
    try put.int(&b, gpa, u32, 8);
    try put.int(&b, gpa, u64, tokens.len);
    for (tokens) |t|
    {
        try put.str(&b, gpa, t);
    }
    try put.str(&b, gpa, "tokenizer.ggml.scores");
    try put.int(&b, gpa, u32, 9);
    try put.int(&b, gpa, u32, 6);
    try put.int(&b, gpa, u64, scores.len);
    for (scores) |s|
    {
        try put.int(&b, gpa, u32, @bitCast(s));
    }
    try put.str(&b, gpa, "tokenizer.ggml.token_type");
    try put.int(&b, gpa, u32, 9);
    try put.int(&b, gpa, u32, 5);
    try put.int(&b, gpa, u64, types.len);
    for (types) |t|
    {
        try put.int(&b, gpa, i32, t);
    }
    try put.str(&b, gpa, "tokenizer.ggml.bos_token_id");
    try put.int(&b, gpa, u32, 4);
    try put.int(&b, gpa, u32, 1);
    try put.str(&b, gpa, "tokenizer.ggml.add_bos_token");
    try put.int(&b, gpa, u32, 7);
    try b.append(gpa, @intFromBool(add_bos));
    try put.str(&b, gpa, "tokenizer.ggml.metaspace.prepend_scheme");
    try put.int(&b, gpa, u32, 8);
    try put.str(&b, gpa, "first");
    return b.toOwnedSlice(gpa);
}

test "SentencePiece merges by score, falls back to bytes, and puts a space in front of the first piece only" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList([]const u8) = .empty;
    defer tokens.deinit(gpa);
    var names: std.ArrayList(u8) = .empty;
    defer names.deinit(gpa);
    var scores: std.ArrayList(f32) = .empty;
    defer scores.deinit(gpa);
    var types: std.ArrayList(i32) = .empty;
    defer types.deinit(gpa);
    for (0..256) |byte|
    {
        try names.print(gpa, "<0x{X:0>2}>", .{byte});
    }
    try tokens.appendSlice(gpa, &.{ "<unk>", "<s>", "</s>" });
    try types.appendSlice(gpa, &.{ 2, 3, 3 });
    for (0..256) |byte|
    {
        try tokens.append(gpa, names.items[byte * 6 ..][0..6]);
        try types.append(gpa, 6);
    }
    // Ids 259 to 264; the space-then-h merge outranks h-then-i.
    try tokens.appendSlice(gpa, &.{ "\u{2581}", "h", "i", "\u{2581}h", "hi", "\u{2581}hi" });
    try types.appendSlice(gpa, &.{ 1, 1, 1, 1, 1, 1 });
    try scores.appendNTimes(gpa, 0, 259);
    try scores.appendSlice(gpa, &.{ -10, -10, -10, -1, -2, -3 });

    const bytes = try testGguf(gpa, tokens.items, scores.items, types.items, true);
    defer gpa.free(bytes);
    var h = try gguf.parse(gpa, bytes, bytes.len);
    defer h.deinit(gpa);
    var tok = try Tokenizer.fromGguf(gpa, h);
    defer tok.deinit();
    try std.testing.expect(tok.add_bos and tok.bos.? == 1 and tok.pattern == .none);

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "hi</s>hi \u{e9}", &ids);
    try std.testing.expectEqualSlices(u32, &.{ 264, 2, 263, 259, 3 + 0xC3, 3 + 0xA9 }, ids.items);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try decode(gpa, &tok, ids.items, &text);
    try std.testing.expectEqualStrings("hi</s>hi \u{e9}", text.items);
}

test "glm4 pattern keeps up to three digits together" {
    try expectSplit(.glm4, "x 12345 !!\n\n", &.{ "x", " ", "123", "45", " !!\n\n" });
    try expectSplit(.glm4, "It'S 7 (hi)\n", &.{ "It", "'S", " ", "7", " (", "hi", ")\n" });
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
    tok.strip_space = false;
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
