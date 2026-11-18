//! A Jinja interpreter for chat templates (TASK-38), matching what transformers'
//! `apply_chat_template` renders: Jinja2 with `trim_blocks`, `lstrip_blocks`, the
//! loop controls, `raise_exception`, `strftime_now`, its own `tojson`, and the
//! `generation` block, in a sandbox where lists and dicts cannot be changed.
//!
//! It covers the subset chat templates use: text, `{{ }}`, `{# #}`, if/elif/else,
//! for (with loop variables, an `if` filter, else, break and continue), set (of
//! names and namespace attributes), macros, and expressions with Python's values
//! and operators, the common filters and tests, and string and dict methods.
//! Anything else is refused by name when the template is parsed, or when a
//! method is called, so a template either renders as transformers does or fails.
const std = @import("std");

pub const Error = error{ Syntax, Unsupported, Undefined, Type, Raised, OutOfMemory };

/// Why parsing or rendering failed: the message and the template line.
pub const Diag = struct {
    buf: [400]u8 = undefined,
    msg: []const u8 = "",
    line: u32 = 0,

    fn set(self: *Diag, line: u32, comptime fmt: []const u8, args: anytype) void {
        self.line = line;
        self.msg = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
    }
};

pub const Dict = std.StringArrayHashMapUnmanaged(Value);

pub const Value = union(enum) {
    /// The name that was not found, for the error message.
    undefined: []const u8,
    none,
    boolean: bool,
    int: i64,
    float: f64,
    string: []const u8,
    /// Lists and tuples alike; never changed after creation.
    list: []const Value,
    dict: *Dict,
    namespace: *Dict,
    macro: *const Macro,
    builtin: Builtin,
    method: *const Method,

    pub fn str(s: []const u8) Value {
        return .{ .string = s };
    }

    fn truthy(self: Value) bool {
        return switch (self) {
            .undefined, .none => false,
            .boolean => |b| b,
            .int => |i| i != 0,
            .float => |f| f != 0,
            .string => |s| s.len > 0,
            .list => |l| l.len > 0,
            .dict => |d| d.count() > 0,
            .namespace, .macro, .builtin, .method => true,
        };
    }
};

const Builtin = enum { raise_exception, range, namespace, strftime_now, dict };
const Method = struct { self: Value, name: []const u8 };

pub const Macro = struct {
    name: []const u8,
    params: []const Param,
    body: []const Node,
};

const Param = struct { name: []const u8, default: ?*const Expr };
const Kw = struct { name: []const u8, value: *const Expr };

const BinOp = enum { add, sub, mul, div, floordiv, mod, pow, concat, eq, ne, lt, le, gt, ge, in, not_in, and_, or_ };

const Expr = union(enum) {
    lit: Value,
    name: []const u8,
    list: []const *const Expr,
    dict: []const [2]*const Expr,
    attr: struct { obj: *const Expr, name: []const u8 },
    item: struct { obj: *const Expr, key: *const Expr },
    slice: struct { obj: *const Expr, start: ?*const Expr, stop: ?*const Expr, step: ?*const Expr },
    call: struct { callee: *const Expr, args: []const *const Expr, kwargs: []const Kw },
    filter: struct { obj: *const Expr, name: []const u8, args: []const *const Expr, kwargs: []const Kw },
    @"test": struct { obj: *const Expr, name: []const u8, args: []const *const Expr, negate: bool },
    neg: *const Expr,
    not_: *const Expr,
    binary: struct { op: BinOp, l: *const Expr, r: *const Expr },
    cond: struct { cond: *const Expr, yes: *const Expr, no: ?*const Expr },
};

const Branch = struct { cond: *const Expr, body: []const Node };

const Node = struct {
    line: u32,
    kind: union(enum) {
        text: []const u8,
        out: *const Expr,
        if_: struct { branches: []const Branch, else_: []const Node },
        for_: struct { targets: []const []const u8, iter: *const Expr, cond: ?*const Expr, body: []const Node, else_: []const Node },
        set: struct { name: []const u8, attr: ?[]const u8, value: *const Expr },
        macro: *const Macro,
        scope: []const Node,
        brk,
        cont,
    },
};

const filters = [_][]const u8{
    "length",  "count",   "trim",   "tojson",  "string", "items",   "safe",    "default", "d",     "join",  "list",
    "sort",    "unique",  "capitalize", "format", "upper", "lower", "replace", "first",   "last",  "int",   "float",
    "abs",     "reverse", "map",    "selectattr", "rejectattr", "title",
};
const tests_known = [_][]const u8{
    "defined", "undefined", "none",  "string", "mapping", "iterable", "sequence", "number", "integer", "float",
    "boolean", "true",      "false", "odd",    "even",    "divisibleby", "equalto", "eq",  "callable", "lower", "upper",
};

pub const Template = struct {
    nodes: []const Node,

    /// Parses `src`; everything it keeps is allocated from `a`.
    pub fn parse(a: std.mem.Allocator, src: []const u8, diag: *Diag) Error!Template {
        const chunks = try split(a, src, diag);
        var p: Parser = .{ .a = a, .chunks = chunks, .diag = diag };
        const body = try p.parseBody(&.{});
        return .{ .nodes = body };
    }

    /// Renders with the given variables; values it makes are allocated from `a`.
    pub fn render(self: Template, a: std.mem.Allocator, vars: *const Dict, diag: *Diag, out: *std.ArrayList(u8)) Error!void {
        var top: Scope = .{ .parent = null };
        var it = vars.iterator();
        while (it.next()) |e|
        {
            try top.vars.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        var r: Renderer = .{ .a = a, .out = out, .top = &top, .diag = diag };
        _ = try r.exec(self.nodes, &top);
    }
};

const ChunkKind = enum { text, out, block };
const Chunk = struct { kind: ChunkKind, src: []const u8, line: u32 };

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0B or c == 0x0C;
}

/// Splits the template into text, `{{ }}` and `{% %}` chunks, applying the whitespace rules.
fn split(a: std.mem.Allocator, raw: []const u8, diag: *Diag) Error![]Chunk {
    // Jinja turns every line break into \n and drops one at the very end.
    var norm: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < raw.len)
    {
        if (raw[i] == '\r')
        {
            try norm.append(a, '\n');
            i += if (i + 1 < raw.len and raw[i + 1] == '\n') 2 else 1;
            continue;
        }
        try norm.append(a, raw[i]);
        i += 1;
    }
    var src: []const u8 = norm.items;
    if (src.len > 0 and src[src.len - 1] == '\n')
    {
        src = src[0 .. src.len - 1];
    }

    var chunks: std.ArrayList(Chunk) = .empty;
    var pos: usize = 0;
    var line_starting = true;
    while (true)
    {
        const p = nextTag(src, pos) orelse {
            if (pos < src.len)
            {
                try chunks.append(a, .{ .kind = .text, .src = src[pos..], .line = lineAt(src, pos) });
            }
            break;
        };
        const kind = src[p + 1];
        const mod: u8 = if (p + 2 < src.len and (src[p + 2] == '-' or src[p + 2] == '+')) src[p + 2] else 0;
        var text = src[pos..p];
        if (mod == '-')
        {
            text = std.mem.trimEnd(u8, text, " \t\n\r\x0b\x0c");
        }
        else if (mod != '+' and kind != '{')
        {
            const l = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |n| n + 1 else 0;
            if ((l > 0 or line_starting) and l < text.len and std.mem.trim(u8, text[l..], " \t\x0b\x0c\r").len == 0)
            {
                text = text[0..l];
            }
        }
        if (text.len > 0)
        {
            try chunks.append(a, .{ .kind = .text, .src = text, .line = lineAt(src, pos) });
        }
        const inner_start = p + 2 + @intFromBool(mod != 0);
        const close: []const u8 = switch (kind) {
            '{' => "}}",
            '%' => "%}",
            else => "#}",
        };
        const e = findClose(src, inner_start, close, kind != '#') orelse {
            diag.set(lineAt(src, p), "unclosed tag", .{});
            return error.Syntax;
        };
        var inner_end = e;
        var end_mod: u8 = 0;
        if (e > inner_start and (src[e - 1] == '-' or src[e - 1] == '+'))
        {
            end_mod = src[e - 1];
            inner_end = e - 1;
        }
        var after = e + 2;
        const tag_end = after;
        if (end_mod == '-')
        {
            while (after < src.len and isWs(src[after]))
            {
                after += 1;
            }
        }
        else if (end_mod != '+' and kind != '{' and after < src.len and src[after] == '\n')
        {
            after += 1;
        }
        line_starting = after > tag_end and src[after - 1] == '\n';
        if (kind != '#')
        {
            try chunks.append(a, .{ .kind = if (kind == '{') .out else .block, .src = src[inner_start..inner_end], .line = lineAt(src, p) });
        }
        pos = after;
    }
    return chunks.items;
}

fn nextTag(src: []const u8, from: usize) ?usize {
    var i = from;
    while (std.mem.indexOfScalarPos(u8, src, i, '{')) |p|
    {
        if (p + 1 < src.len and (src[p + 1] == '{' or src[p + 1] == '%' or src[p + 1] == '#'))
        {
            return p;
        }
        i = p + 1;
    }
    return null;
}

/// The position of `close`, skipping string literals inside expression tags.
fn findClose(src: []const u8, from: usize, close: []const u8, strings: bool) ?usize {
    var i = from;
    while (i + 1 < src.len)
    {
        const c = src[i];
        if (strings and (c == '\'' or c == '"'))
        {
            i += 1;
            while (i < src.len and src[i] != c)
            {
                i += if (src[i] == '\\') 2 else 1;
            }
            i += 1;
            continue;
        }
        if (c == close[0] and src[i + 1] == close[1])
        {
            return i;
        }
        i += 1;
    }
    return null;
}

fn lineAt(src: []const u8, pos: usize) u32 {
    return @intCast(std.mem.count(u8, src[0..@min(pos, src.len)], "\n") + 1);
}

const Tag = enum { name, string, int, float, op, eof };
const Token = struct { tag: Tag, text: []const u8 = "", int: i64 = 0, float: f64 = 0 };

const ops = [_][]const u8{ "**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "(", ")", "[", "]", "{", "}", ",", ":", ".", "|", "=" };

/// Tokens of one tag's contents.
const Lexer = struct {
    a: std.mem.Allocator,
    src: []const u8,
    pos: usize = 0,
    peeked: ?Token = null,
    line: u32,
    diag: *Diag,

    fn peek(self: *Lexer) Error!Token {
        if (self.peeked == null)
        {
            self.peeked = try self.lex();
        }
        return self.peeked.?;
    }

    fn next(self: *Lexer) Error!Token {
        const t = try self.peek();
        self.peeked = null;
        return t;
    }

    fn isOp(self: *Lexer, op: []const u8) Error!bool {
        const t = try self.peek();
        return t.tag == .op and std.mem.eql(u8, t.text, op);
    }

    fn isName(self: *Lexer, name: []const u8) Error!bool {
        const t = try self.peek();
        return t.tag == .name and std.mem.eql(u8, t.text, name);
    }

    fn skipOp(self: *Lexer, op: []const u8) Error!bool {
        if (try self.isOp(op))
        {
            _ = try self.next();
            return true;
        }
        return false;
    }

    fn skipName(self: *Lexer, name: []const u8) Error!bool {
        if (try self.isName(name))
        {
            _ = try self.next();
            return true;
        }
        return false;
    }

    fn expectOp(self: *Lexer, op: []const u8) Error!void {
        if (!try self.skipOp(op))
        {
            return self.fail("expected '{s}'", .{op});
        }
    }

    fn expectName(self: *Lexer) Error![]const u8 {
        const t = try self.next();
        if (t.tag != .name)
        {
            return self.fail("expected a name", .{});
        }
        return t.text;
    }

    fn fail(self: *Lexer, comptime fmt: []const u8, args: anytype) Error {
        self.diag.set(self.line, fmt, args);
        return error.Syntax;
    }

    fn lex(self: *Lexer) Error!Token {
        const s = self.src;
        while (self.pos < s.len and isWs(s[self.pos]))
        {
            self.pos += 1;
        }
        if (self.pos >= s.len)
        {
            return .{ .tag = .eof };
        }
        const start = self.pos;
        const c = s[start];
        if (std.ascii.isAlphabetic(c) or c == '_')
        {
            while (self.pos < s.len and (std.ascii.isAlphanumeric(s[self.pos]) or s[self.pos] == '_'))
            {
                self.pos += 1;
            }
            return .{ .tag = .name, .text = s[start..self.pos] };
        }
        if (std.ascii.isDigit(c))
        {
            return self.number();
        }
        if (c == '\'' or c == '"')
        {
            return .{ .tag = .string, .text = try self.string() };
        }
        for (ops) |op|
        {
            if (std.mem.startsWith(u8, s[start..], op))
            {
                self.pos += op.len;
                return .{ .tag = .op, .text = op };
            }
        }
        return self.fail("unexpected character '{c}'", .{c});
    }

    fn number(self: *Lexer) Error!Token {
        const s = self.src;
        const start = self.pos;
        var is_float = false;
        while (self.pos < s.len and (std.ascii.isDigit(s[self.pos]) or s[self.pos] == '_'))
        {
            self.pos += 1;
        }
        if (self.pos + 1 < s.len and s[self.pos] == '.' and std.ascii.isDigit(s[self.pos + 1]))
        {
            is_float = true;
            self.pos += 1;
            while (self.pos < s.len and (std.ascii.isDigit(s[self.pos]) or s[self.pos] == '_'))
            {
                self.pos += 1;
            }
        }
        if (self.pos < s.len and (s[self.pos] == 'e' or s[self.pos] == 'E'))
        {
            var j = self.pos + 1;
            if (j < s.len and (s[j] == '+' or s[j] == '-'))
            {
                j += 1;
            }
            if (j < s.len and std.ascii.isDigit(s[j]))
            {
                is_float = true;
                self.pos = j;
                while (self.pos < s.len and std.ascii.isDigit(s[self.pos]))
                {
                    self.pos += 1;
                }
            }
        }
        var digits: std.ArrayList(u8) = .empty;
        for (s[start..self.pos]) |ch|
        {
            if (ch != '_')
            {
                try digits.append(self.a, ch);
            }
        }
        if (is_float)
        {
            const f = std.fmt.parseFloat(f64, digits.items) catch return self.fail("bad number", .{});
            return .{ .tag = .float, .float = f };
        }
        const n = std.fmt.parseInt(i64, digits.items, 10) catch return self.fail("number too large", .{});
        return .{ .tag = .int, .int = n };
    }

    /// A quoted string with Python's escapes, decoded.
    fn string(self: *Lexer) Error![]const u8 {
        const s = self.src;
        const q = s[self.pos];
        self.pos += 1;
        var out: std.ArrayList(u8) = .empty;
        while (true)
        {
            if (self.pos >= s.len)
            {
                return self.fail("unclosed string", .{});
            }
            const c = s[self.pos];
            if (c == q)
            {
                self.pos += 1;
                break;
            }
            if (c != '\\' or self.pos + 1 >= s.len)
            {
                try out.append(self.a, c);
                self.pos += 1;
                continue;
            }
            const e = s[self.pos + 1];
            self.pos += 2;
            switch (e)
            {
                'n' => try out.append(self.a, '\n'),
                't' => try out.append(self.a, '\t'),
                'r' => try out.append(self.a, '\r'),
                '\\' => try out.append(self.a, '\\'),
                '\'' => try out.append(self.a, '\''),
                '"' => try out.append(self.a, '"'),
                'a' => try out.append(self.a, 0x07),
                'b' => try out.append(self.a, 0x08),
                'f' => try out.append(self.a, 0x0C),
                'v' => try out.append(self.a, 0x0B),
                '\n' => {},
                'x', 'u', 'U' => {
                    const n: usize = switch (e) {
                        'x' => 2,
                        'u' => 4,
                        else => 8,
                    };
                    if (self.pos + n > s.len)
                    {
                        return self.fail("bad escape", .{});
                    }
                    const cp = std.fmt.parseInt(u21, s[self.pos .. self.pos + n], 16) catch return self.fail("bad escape", .{});
                    self.pos += n;
                    var buf: [4]u8 = undefined;
                    const len = std.unicode.utf8Encode(cp, &buf) catch return self.fail("bad escape", .{});
                    try out.appendSlice(self.a, buf[0..len]);
                },
                '0'...'7' => {
                    var v: u21 = e - '0';
                    var k: usize = 0;
                    while (k < 2 and self.pos < s.len and s[self.pos] >= '0' and s[self.pos] <= '7')
                    {
                        v = v * 8 + (s[self.pos] - '0');
                        self.pos += 1;
                        k += 1;
                    }
                    var buf: [4]u8 = undefined;
                    const len = std.unicode.utf8Encode(v, &buf) catch unreachable;
                    try out.appendSlice(self.a, buf[0..len]);
                },
                else => {
                    try out.append(self.a, '\\');
                    try out.append(self.a, e);
                },
            }
        }
        return out.items;
    }
};

const Parser = struct {
    a: std.mem.Allocator,
    chunks: []const Chunk,
    i: usize = 0,
    diag: *Diag,
    /// The block chunk that ended the last body, positioned after its keyword.
    end: ?Lexer = null,
    end_name: []const u8 = "",

    fn new(self: *Parser, e: Expr) Error!*const Expr {
        const p = try self.a.create(Expr);
        p.* = e;
        return p;
    }

    /// Nodes until a block whose keyword is in `ends` (or the end of the template if `ends` is empty).
    fn parseBody(self: *Parser, ends: []const []const u8) Error![]const Node {
        var nodes: std.ArrayList(Node) = .empty;
        while (self.i < self.chunks.len)
        {
            const c = self.chunks[self.i];
            self.i += 1;
            switch (c.kind)
            {
                .text => try nodes.append(self.a, .{ .line = c.line, .kind = .{ .text = c.src } }),
                .out => {
                    var lx = self.lexer(c);
                    const e = try self.expr(&lx);
                    try self.done(&lx);
                    try nodes.append(self.a, .{ .line = c.line, .kind = .{ .out = e } });
                },
                .block => {
                    var lx = self.lexer(c);
                    const kw = try lx.expectName();
                    for (ends) |e|
                    {
                        if (std.mem.eql(u8, kw, e))
                        {
                            self.end = lx;
                            self.end_name = kw;
                            return nodes.items;
                        }
                    }
                    try nodes.append(self.a, try self.statement(kw, &lx, c.line));
                },
            }
        }
        if (ends.len > 0)
        {
            self.diag.set(if (self.chunks.len > 0) self.chunks[self.chunks.len - 1].line else 0, "missing '{s}'", .{ends[ends.len - 1]});
            return error.Syntax;
        }
        return nodes.items;
    }

    fn lexer(self: *Parser, c: Chunk) Lexer {
        return .{ .a = self.a, .src = c.src, .line = c.line, .diag = self.diag };
    }

    fn done(self: *Parser, lx: *Lexer) Error!void {
        _ = self;
        const t = try lx.peek();
        if (t.tag != .eof)
        {
            return lx.fail("unexpected '{s}'", .{if (t.tag == .op or t.tag == .name) t.text else "token"});
        }
    }

    fn statement(self: *Parser, kw: []const u8, lx: *Lexer, line: u32) Error!Node {
        if (std.mem.eql(u8, kw, "if"))
        {
            var branches: std.ArrayList(Branch) = .empty;
            var cond = try self.orExpr(lx);
            try self.done(lx);
            while (true)
            {
                const body = try self.parseBody(&.{ "elif", "else", "endif" });
                try branches.append(self.a, .{ .cond = cond, .body = body });
                var end = self.end.?;
                if (std.mem.eql(u8, self.end_name, "elif"))
                {
                    cond = try self.orExpr(&end);
                    try self.done(&end);
                    continue;
                }
                try self.done(&end);
                var else_: []const Node = &.{};
                if (std.mem.eql(u8, self.end_name, "else"))
                {
                    else_ = try self.parseBody(&.{"endif"});
                    var e2 = self.end.?;
                    try self.done(&e2);
                }
                return .{ .line = line, .kind = .{ .if_ = .{ .branches = branches.items, .else_ = else_ } } };
            }
        }
        if (std.mem.eql(u8, kw, "for"))
        {
            var targets: std.ArrayList([]const u8) = .empty;
            try targets.append(self.a, try lx.expectName());
            while (try lx.skipOp(","))
            {
                try targets.append(self.a, try lx.expectName());
            }
            if (!try lx.skipName("in"))
            {
                return lx.fail("expected 'in'", .{});
            }
            const iter = try self.tupleNoCond(lx);
            var cond: ?*const Expr = null;
            if (try lx.skipName("if"))
            {
                cond = try self.condExpr(lx);
            }
            if (try lx.isName("recursive"))
            {
                return self.unsupported(line, "recursive loops", .{});
            }
            try self.done(lx);
            const body = try self.parseBody(&.{ "else", "endfor" });
            var end = self.end.?;
            try self.done(&end);
            var else_: []const Node = &.{};
            if (std.mem.eql(u8, self.end_name, "else"))
            {
                else_ = try self.parseBody(&.{"endfor"});
                var e2 = self.end.?;
                try self.done(&e2);
            }
            return .{ .line = line, .kind = .{ .for_ = .{ .targets = targets.items, .iter = iter, .cond = cond, .body = body, .else_ = else_ } } };
        }
        if (std.mem.eql(u8, kw, "set"))
        {
            const name = try lx.expectName();
            var attr: ?[]const u8 = null;
            if (try lx.skipOp("."))
            {
                attr = try lx.expectName();
            }
            if (try lx.isOp(","))
            {
                return self.unsupported(line, "setting several names at once", .{});
            }
            if (!try lx.skipOp("="))
            {
                return self.unsupported(line, "block 'set'", .{});
            }
            const value = try self.tuple(lx);
            try self.done(lx);
            return .{ .line = line, .kind = .{ .set = .{ .name = name, .attr = attr, .value = value } } };
        }
        if (std.mem.eql(u8, kw, "macro"))
        {
            const name = try lx.expectName();
            try lx.expectOp("(");
            var params: std.ArrayList(Param) = .empty;
            while (!try lx.skipOp(")"))
            {
                const pname = try lx.expectName();
                var default: ?*const Expr = null;
                if (try lx.skipOp("="))
                {
                    default = try self.condExpr(lx);
                }
                try params.append(self.a, .{ .name = pname, .default = default });
                if (!try lx.skipOp(","))
                {
                    try lx.expectOp(")");
                    break;
                }
            }
            try self.done(lx);
            const body = try self.parseBody(&.{"endmacro"});
            var end = self.end.?;
            _ = try end.skipName(name);
            try self.done(&end);
            const m = try self.a.create(Macro);
            m.* = .{ .name = name, .params = params.items, .body = body };
            return .{ .line = line, .kind = .{ .macro = m } };
        }
        if (std.mem.eql(u8, kw, "generation"))
        {
            try self.done(lx);
            const body = try self.parseBody(&.{"endgeneration"});
            var end = self.end.?;
            try self.done(&end);
            return .{ .line = line, .kind = .{ .scope = body } };
        }
        if (std.mem.eql(u8, kw, "break") or std.mem.eql(u8, kw, "continue"))
        {
            try self.done(lx);
            return .{ .line = line, .kind = if (kw[0] == 'b') .brk else .cont };
        }
        return self.unsupported(line, "the '{s}' statement", .{kw});
    }

    fn unsupported(self: *Parser, line: u32, comptime what: []const u8, args: anytype) Error {
        self.diag.set(line, "the template uses " ++ what ++ ", which is not supported", args);
        return error.Unsupported;
    }

    /// An expression, or several separated by commas as a tuple.
    fn tuple(self: *Parser, lx: *Lexer) Error!*const Expr {
        const first = try self.condExpr(lx);
        if (!try lx.isOp(","))
        {
            return first;
        }
        var items: std.ArrayList(*const Expr) = .empty;
        try items.append(self.a, first);
        while (try lx.skipOp(","))
        {
            const t = try lx.peek();
            if (t.tag == .eof)
            {
                break;
            }
            try items.append(self.a, try self.condExpr(lx));
        }
        return self.new(.{ .list = items.items });
    }

    fn tupleNoCond(self: *Parser, lx: *Lexer) Error!*const Expr {
        const first = try self.orExpr(lx);
        if (!try lx.isOp(","))
        {
            return first;
        }
        var items: std.ArrayList(*const Expr) = .empty;
        try items.append(self.a, first);
        while (try lx.skipOp(","))
        {
            try items.append(self.a, try self.orExpr(lx));
        }
        return self.new(.{ .list = items.items });
    }

    fn expr(self: *Parser, lx: *Lexer) Error!*const Expr {
        return self.condExpr(lx);
    }

    fn condExpr(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.orExpr(lx);
        while (try lx.skipName("if"))
        {
            const cond = try self.orExpr(lx);
            var no: ?*const Expr = null;
            if (try lx.skipName("else"))
            {
                no = try self.condExpr(lx);
            }
            e = try self.new(.{ .cond = .{ .cond = cond, .yes = e, .no = no } });
        }
        return e;
    }

    fn orExpr(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.andExpr(lx);
        while (try lx.skipName("or"))
        {
            e = try self.new(.{ .binary = .{ .op = .or_, .l = e, .r = try self.andExpr(lx) } });
        }
        return e;
    }

    fn andExpr(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.notExpr(lx);
        while (try lx.skipName("and"))
        {
            e = try self.new(.{ .binary = .{ .op = .and_, .l = e, .r = try self.notExpr(lx) } });
        }
        return e;
    }

    fn notExpr(self: *Parser, lx: *Lexer) Error!*const Expr {
        if (try lx.skipName("not"))
        {
            return self.new(.{ .not_ = try self.notExpr(lx) });
        }
        return self.compare(lx);
    }

    fn compare(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.math1(lx);
        while (true)
        {
            const t = try lx.peek();
            var op: ?BinOp = null;
            if (t.tag == .op)
            {
                const pairs = [_]struct { []const u8, BinOp }{ .{ "==", .eq }, .{ "!=", .ne }, .{ "<", .lt }, .{ "<=", .le }, .{ ">", .gt }, .{ ">=", .ge } };
                for (pairs) |pr|
                {
                    if (std.mem.eql(u8, t.text, pr[0]))
                    {
                        op = pr[1];
                    }
                }
            }
            else if (t.tag == .name and std.mem.eql(u8, t.text, "in"))
            {
                op = .in;
            }
            else if (t.tag == .name and std.mem.eql(u8, t.text, "not"))
            {
                // Only "not in" continues a comparison.
                const save = lx.*;
                _ = try lx.next();
                if (try lx.skipName("in"))
                {
                    e = try self.new(.{ .binary = .{ .op = .not_in, .l = e, .r = try self.math1(lx) } });
                    continue;
                }
                lx.* = save;
            }
            if (op == null)
            {
                return e;
            }
            _ = try lx.next();
            e = try self.new(.{ .binary = .{ .op = op.?, .l = e, .r = try self.math1(lx) } });
        }
    }

    fn math1(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.concat(lx);
        while (true)
        {
            const op: BinOp = if (try lx.skipOp("+")) .add else if (try lx.skipOp("-")) .sub else return e;
            e = try self.new(.{ .binary = .{ .op = op, .l = e, .r = try self.concat(lx) } });
        }
    }

    fn concat(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.math2(lx);
        while (try lx.skipOp("~"))
        {
            e = try self.new(.{ .binary = .{ .op = .concat, .l = e, .r = try self.math2(lx) } });
        }
        return e;
    }

    fn math2(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.pow(lx);
        while (true)
        {
            const op: BinOp = if (try lx.skipOp("*")) .mul else if (try lx.skipOp("//")) .floordiv else if (try lx.skipOp("/")) .div else if (try lx.skipOp("%")) .mod else return e;
            e = try self.new(.{ .binary = .{ .op = op, .l = e, .r = try self.pow(lx) } });
        }
    }

    fn pow(self: *Parser, lx: *Lexer) Error!*const Expr {
        var e = try self.unary(lx, true);
        while (try lx.skipOp("**"))
        {
            e = try self.new(.{ .binary = .{ .op = .pow, .l = e, .r = try self.unary(lx, true) } });
        }
        return e;
    }

    fn unary(self: *Parser, lx: *Lexer, with_filter: bool) Error!*const Expr {
        var e: *const Expr = undefined;
        if (try lx.skipOp("-"))
        {
            e = try self.new(.{ .neg = try self.unary(lx, false) });
        }
        else if (try lx.skipOp("+"))
        {
            e = try self.unary(lx, false);
        }
        else
        {
            e = try self.postfix(lx, try self.primary(lx));
        }
        if (with_filter)
        {
            e = try self.filterExpr(lx, e);
        }
        return e;
    }

    fn primary(self: *Parser, lx: *Lexer) Error!*const Expr {
        const t = try lx.next();
        switch (t.tag)
        {
            .name => {
                const consts = [_]struct { []const u8, Value }{
                    .{ "true", .{ .boolean = true } },   .{ "True", .{ .boolean = true } },
                    .{ "false", .{ .boolean = false } }, .{ "False", .{ .boolean = false } },
                    .{ "none", .none },                  .{ "None", .none },
                };
                for (consts) |c|
                {
                    if (std.mem.eql(u8, t.text, c[0]))
                    {
                        return self.new(.{ .lit = c[1] });
                    }
                }
                return self.new(.{ .name = t.text });
            },
            .string => {
                var s = t.text;
                // Adjacent strings join, as in Python.
                while ((try lx.peek()).tag == .string)
                {
                    s = try std.mem.concat(self.a, u8, &.{ s, (try lx.next()).text });
                }
                return self.new(.{ .lit = .{ .string = s } });
            },
            .int => return self.new(.{ .lit = .{ .int = t.int } }),
            .float => return self.new(.{ .lit = .{ .float = t.float } }),
            .op => {
                if (std.mem.eql(u8, t.text, "("))
                {
                    if (try lx.skipOp(")"))
                    {
                        return self.new(.{ .list = &.{} });
                    }
                    const first = try self.condExpr(lx);
                    if (try lx.skipOp(")"))
                    {
                        return first;
                    }
                    var items: std.ArrayList(*const Expr) = .empty;
                    try items.append(self.a, first);
                    while (try lx.skipOp(","))
                    {
                        if (try lx.isOp(")"))
                        {
                            break;
                        }
                        try items.append(self.a, try self.condExpr(lx));
                    }
                    try lx.expectOp(")");
                    return self.new(.{ .list = items.items });
                }
                if (std.mem.eql(u8, t.text, "["))
                {
                    var items: std.ArrayList(*const Expr) = .empty;
                    while (!try lx.skipOp("]"))
                    {
                        try items.append(self.a, try self.condExpr(lx));
                        if (!try lx.skipOp(","))
                        {
                            try lx.expectOp("]");
                            break;
                        }
                    }
                    return self.new(.{ .list = items.items });
                }
                if (std.mem.eql(u8, t.text, "{"))
                {
                    var items: std.ArrayList([2]*const Expr) = .empty;
                    while (!try lx.skipOp("}"))
                    {
                        const k = try self.condExpr(lx);
                        try lx.expectOp(":");
                        try items.append(self.a, .{ k, try self.condExpr(lx) });
                        if (!try lx.skipOp(","))
                        {
                            try lx.expectOp("}");
                            break;
                        }
                    }
                    return self.new(.{ .dict = items.items });
                }
                return lx.fail("unexpected '{s}'", .{t.text});
            },
            .eof => return lx.fail("unexpected end of expression", .{}),
        }
    }

    fn postfix(self: *Parser, lx: *Lexer, start: *const Expr) Error!*const Expr {
        var e = start;
        while (true)
        {
            if (try lx.skipOp("."))
            {
                const t = try lx.next();
                e = switch (t.tag) {
                    .name => try self.new(.{ .attr = .{ .obj = e, .name = t.text } }),
                    .int => try self.new(.{ .item = .{ .obj = e, .key = try self.new(.{ .lit = .{ .int = t.int } }) } }),
                    else => return lx.fail("expected a name after '.'", .{}),
                };
            }
            else if (try lx.skipOp("["))
            {
                e = try self.subscript(lx, e);
            }
            else if (try lx.isOp("("))
            {
                e = try self.call(lx, e);
            }
            else
            {
                return e;
            }
        }
    }

    fn subscript(self: *Parser, lx: *Lexer, obj: *const Expr) Error!*const Expr {
        var parts: [3]?*const Expr = .{ null, null, null };
        var n: usize = 0;
        var is_slice = false;
        while (true)
        {
            if (try lx.skipOp("]"))
            {
                break;
            }
            if (try lx.skipOp(":"))
            {
                is_slice = true;
                n += 1;
                if (n > 2)
                {
                    return lx.fail("bad slice", .{});
                }
                continue;
            }
            parts[n] = try self.condExpr(lx);
        }
        if (!is_slice)
        {
            return self.new(.{ .item = .{ .obj = obj, .key = parts[0] orelse return lx.fail("empty subscript", .{}) } });
        }
        return self.new(.{ .slice = .{ .obj = obj, .start = parts[0], .stop = parts[1], .step = parts[2] } });
    }

    const Args = struct { args: []const *const Expr, kwargs: []const Kw };

    fn callArgs(self: *Parser, lx: *Lexer) Error!Args {
        try lx.expectOp("(");
        var pos: std.ArrayList(*const Expr) = .empty;
        var kws: std.ArrayList(Kw) = .empty;
        while (!try lx.skipOp(")"))
        {
            const t = try lx.peek();
            if (t.tag == .op and (std.mem.eql(u8, t.text, "*") or std.mem.eql(u8, t.text, "**")))
            {
                return lx.fail("the template uses *args, which is not supported", .{});
            }
            var is_kw = false;
            if (t.tag == .name)
            {
                const save = lx.*;
                _ = try lx.next();
                if (try lx.skipOp("="))
                {
                    try kws.append(self.a, .{ .name = t.text, .value = try self.condExpr(lx) });
                    is_kw = true;
                }
                else
                {
                    lx.* = save;
                }
            }
            if (!is_kw)
            {
                try pos.append(self.a, try self.condExpr(lx));
            }
            if (!try lx.skipOp(","))
            {
                try lx.expectOp(")");
                break;
            }
        }
        return .{ .args = pos.items, .kwargs = kws.items };
    }

    fn call(self: *Parser, lx: *Lexer, callee: *const Expr) Error!*const Expr {
        const a = try self.callArgs(lx);
        return self.new(.{ .call = .{ .callee = callee, .args = a.args, .kwargs = a.kwargs } });
    }

    fn filterExpr(self: *Parser, lx: *Lexer, start: *const Expr) Error!*const Expr {
        var e = start;
        while (true)
        {
            if (try lx.skipOp("|"))
            {
                const name = try lx.expectName();
                if (!known(&filters, name))
                {
                    return self.unsupported(lx.line, "the filter '{s}'", .{name});
                }
                var a: Args = .{ .args = &.{}, .kwargs = &.{} };
                if (try lx.isOp("("))
                {
                    a = try self.callArgs(lx);
                }
                e = try self.new(.{ .filter = .{ .obj = e, .name = name, .args = a.args, .kwargs = a.kwargs } });
            }
            else if (try lx.skipName("is"))
            {
                const negate = try lx.skipName("not");
                const name = try lx.expectName();
                if (!known(&tests_known, name))
                {
                    return self.unsupported(lx.line, "the test '{s}'", .{name});
                }
                var targs: []const *const Expr = &.{};
                if (try lx.isOp("("))
                {
                    targs = (try self.callArgs(lx)).args;
                }
                else
                {
                    const t = try lx.peek();
                    const ends = [_][]const u8{ "else", "or", "and", "if", "in", "not", "is" };
                    const starts_arg = switch (t.tag) {
                        .string, .int, .float => true,
                        .name => !known(&ends, t.text),
                        .op => std.mem.eql(u8, t.text, "[") or std.mem.eql(u8, t.text, "{"),
                        .eof => false,
                    };
                    if (starts_arg)
                    {
                        const arg = try self.postfix(lx, try self.primary(lx));
                        const one = try self.a.alloc(*const Expr, 1);
                        one[0] = arg;
                        targs = one;
                    }
                }
                e = try self.new(.{ .@"test" = .{ .obj = e, .name = name, .args = targs, .negate = negate } });
            }
            else if (try lx.isOp("("))
            {
                e = try self.call(lx, e);
            }
            else
            {
                return e;
            }
        }
    }
};

fn known(list: []const []const u8, name: []const u8) bool {
    for (list) |n|
    {
        if (std.mem.eql(u8, n, name))
        {
            return true;
        }
    }
    return false;
}

const Scope = struct {
    vars: std.StringHashMapUnmanaged(Value) = .empty,
    parent: ?*Scope,

    fn get(self: *Scope, name: []const u8) ?Value {
        var s: ?*Scope = self;
        while (s) |sc|
        {
            if (sc.vars.get(name)) |v|
            {
                return v;
            }
            s = sc.parent;
        }
        return null;
    }
};

const Flow = enum { normal, brk, cont };

const Renderer = struct {
    a: std.mem.Allocator,
    out: *std.ArrayList(u8),
    top: *Scope,
    diag: *Diag,
    line: u32 = 0,

    fn fail(self: *Renderer, err: Error, comptime fmt: []const u8, args: anytype) Error {
        self.diag.set(self.line, fmt, args);
        return err;
    }

    fn exec(self: *Renderer, nodes: []const Node, scope: *Scope) Error!Flow {
        for (nodes) |n|
        {
            self.line = n.line;
            switch (n.kind)
            {
                .text => |t| try self.out.appendSlice(self.a, t),
                .out => |e| {
                    const v = try self.eval(e, scope);
                    try self.write(v);
                },
                .if_ => |f| {
                    var done = false;
                    for (f.branches) |b|
                    {
                        if ((try self.eval(b.cond, scope)).truthy())
                        {
                            const flow = try self.exec(b.body, scope);
                            if (flow != .normal)
                            {
                                return flow;
                            }
                            done = true;
                            break;
                        }
                    }
                    if (!done)
                    {
                        const flow = try self.exec(f.else_, scope);
                        if (flow != .normal)
                        {
                            return flow;
                        }
                    }
                },
                .for_ => |f| try self.loop(f.targets, f.iter, f.cond, f.body, f.else_, scope),
                .set => |s| {
                    const v = try self.eval(s.value, scope);
                    if (s.attr) |attr|
                    {
                        const target = scope.get(s.name) orelse return self.fail(error.Undefined, "'{s}' is undefined", .{s.name});
                        switch (target)
                        {
                            .namespace => |ns| try ns.put(self.a, attr, v),
                            else => return self.fail(error.Type, "can only set attributes of a namespace", .{}),
                        }
                    }
                    else
                    {
                        try scope.vars.put(self.a, s.name, v);
                    }
                },
                .macro => |m| try scope.vars.put(self.a, m.name, .{ .macro = m }),
                .scope => |body| {
                    var inner: Scope = .{ .parent = scope };
                    const flow = try self.exec(body, &inner);
                    if (flow != .normal)
                    {
                        return flow;
                    }
                },
                .brk => return .brk,
                .cont => return .cont,
            }
        }
        return .normal;
    }

    fn loop(self: *Renderer, targets: []const []const u8, iter_e: *const Expr, cond: ?*const Expr, body: []const Node, else_: []const Node, scope: *Scope) Error!void {
        const line = self.line;
        const all = try self.iterate(try self.eval(iter_e, scope));
        var items = all;
        if (cond) |c|
        {
            var kept: std.ArrayList(Value) = .empty;
            for (all) |item|
            {
                var s: Scope = .{ .parent = scope };
                try self.bind(&s, targets, item);
                if ((try self.eval(c, &s)).truthy())
                {
                    try kept.append(self.a, item);
                }
            }
            items = kept.items;
        }
        if (items.len == 0)
        {
            _ = try self.exec(else_, scope);
            return;
        }
        for (items, 0..) |item, i|
        {
            // Each pass has its own variables, as in Jinja.
            var s: Scope = .{ .parent = scope };
            try self.bind(&s, targets, item);
            const lp = try self.a.create(Dict);
            lp.* = .empty;
            const n: i64 = @intCast(items.len);
            const k: i64 = @intCast(i);
            try lp.put(self.a, "index", .{ .int = k + 1 });
            try lp.put(self.a, "index0", .{ .int = k });
            try lp.put(self.a, "revindex", .{ .int = n - k });
            try lp.put(self.a, "revindex0", .{ .int = n - k - 1 });
            try lp.put(self.a, "first", .{ .boolean = i == 0 });
            try lp.put(self.a, "last", .{ .boolean = i + 1 == items.len });
            try lp.put(self.a, "length", .{ .int = n });
            try lp.put(self.a, "depth", .{ .int = 1 });
            try lp.put(self.a, "depth0", .{ .int = 0 });
            if (i > 0)
            {
                try lp.put(self.a, "previtem", items[i - 1]);
            }
            if (i + 1 < items.len)
            {
                try lp.put(self.a, "nextitem", items[i + 1]);
            }
            try s.vars.put(self.a, "loop", .{ .dict = lp });
            const flow = try self.exec(body, &s);
            self.line = line;
            if (flow == .brk)
            {
                break;
            }
        }
    }

    fn bind(self: *Renderer, s: *Scope, targets: []const []const u8, item: Value) Error!void {
        if (targets.len == 1)
        {
            return s.vars.put(self.a, targets[0], item);
        }
        const parts = try self.iterate(item);
        if (parts.len != targets.len)
        {
            return self.fail(error.Type, "cannot unpack {d} values into {d} names", .{ parts.len, targets.len });
        }
        for (targets, parts) |t, p|
        {
            try s.vars.put(self.a, t, p);
        }
    }

    /// The values a for loop goes over: list items, dict keys, or a string's characters.
    fn iterate(self: *Renderer, v: Value) Error![]const Value {
        return switch (v) {
            .list => |l| l,
            .undefined => &.{},
            .dict, .namespace => |d| blk: {
                const keys = try self.a.alloc(Value, d.count());
                for (d.keys(), keys) |k, *o|
                {
                    o.* = .{ .string = k };
                }
                break :blk keys;
            },
            .string => |s| blk: {
                var out: std.ArrayList(Value) = .empty;
                var it = (std.unicode.Utf8View.init(s) catch return self.fail(error.Type, "invalid UTF-8", .{})).iterator();
                while (it.nextCodepointSlice()) |c|
                {
                    try out.append(self.a, .{ .string = c });
                }
                break :blk out.items;
            },
            else => self.fail(error.Type, "{s} is not iterable", .{typeName(v)}),
        };
    }

    fn write(self: *Renderer, v: Value) Error!void {
        try pyStr(self.a, v, self.out);
    }

    fn eval(self: *Renderer, e: *const Expr, scope: *Scope) Error!Value {
        switch (e.*)
        {
            .lit => |v| return v,
            .name => |n| {
                if (scope.get(n)) |v|
                {
                    return v;
                }
                const globals = [_]struct { []const u8, Builtin }{ .{ "raise_exception", .raise_exception }, .{ "range", .range }, .{ "namespace", .namespace }, .{ "strftime_now", .strftime_now }, .{ "dict", .dict } };
                for (globals) |g|
                {
                    if (std.mem.eql(u8, n, g[0]))
                    {
                        return .{ .builtin = g[1] };
                    }
                }
                return .{ .undefined = n };
            },
            .list => |items| {
                const out = try self.a.alloc(Value, items.len);
                for (items, out) |x, *o|
                {
                    o.* = try self.eval(x, scope);
                }
                return .{ .list = out };
            },
            .dict => |items| {
                const d = try self.a.create(Dict);
                d.* = .empty;
                for (items) |kv|
                {
                    const k = try self.eval(kv[0], scope);
                    if (k != .string)
                    {
                        return self.fail(error.Unsupported, "dict keys must be strings", .{});
                    }
                    try d.put(self.a, k.string, try self.eval(kv[1], scope));
                }
                return .{ .dict = d };
            },
            .attr => |x| return self.getAttr(try self.eval(x.obj, scope), x.name),
            .item => |x| return self.getItem(try self.eval(x.obj, scope), try self.eval(x.key, scope)),
            .slice => |x| {
                const obj = try self.eval(x.obj, scope);
                const start = if (x.start) |s| try self.eval(s, scope) else Value.none;
                const stop = if (x.stop) |s| try self.eval(s, scope) else Value.none;
                const step = if (x.step) |s| try self.eval(s, scope) else Value.none;
                return self.slice(obj, start, stop, step);
            },
            .call => |x| {
                const callee = try self.eval(x.callee, scope);
                const args = try self.evalAll(x.args, scope);
                const kwargs = try self.evalKw(x.kwargs, scope);
                return self.call(callee, args, kwargs);
            },
            .filter => |x| {
                const obj = try self.eval(x.obj, scope);
                return self.filter(x.name, obj, try self.evalAll(x.args, scope), try self.evalKw(x.kwargs, scope));
            },
            .@"test" => |x| {
                const obj = try self.eval(x.obj, scope);
                const r = try self.testValue(x.name, obj, try self.evalAll(x.args, scope));
                return .{ .boolean = r != x.negate };
            },
            .neg => |x| {
                const v = try self.eval(x, scope);
                return switch (v) {
                    .int => |i| .{ .int = -i },
                    .float => |f| .{ .float = -f },
                    .boolean => |b| .{ .int = -@as(i64, @intFromBool(b)) },
                    .undefined => |n| self.fail(error.Undefined, "'{s}' is undefined", .{n}),
                    else => self.fail(error.Type, "bad operand for unary -: {s}", .{typeName(v)}),
                };
            },
            .not_ => |x| return .{ .boolean = !(try self.eval(x, scope)).truthy() },
            .cond => |x| {
                if ((try self.eval(x.cond, scope)).truthy())
                {
                    return self.eval(x.yes, scope);
                }
                return if (x.no) |n| self.eval(n, scope) else .{ .undefined = "" };
            },
            .binary => |x| {
                if (x.op == .and_ or x.op == .or_)
                {
                    const l = try self.eval(x.l, scope);
                    if (l.truthy() == (x.op == .or_))
                    {
                        return l;
                    }
                    return self.eval(x.r, scope);
                }
                return self.binary(x.op, try self.eval(x.l, scope), try self.eval(x.r, scope));
            },
        }
    }

    fn evalAll(self: *Renderer, es: []const *const Expr, scope: *Scope) Error![]const Value {
        const out = try self.a.alloc(Value, es.len);
        for (es, out) |e, *o|
        {
            o.* = try self.eval(e, scope);
        }
        return out;
    }

    const KwVal = struct { name: []const u8, value: Value };

    fn evalKw(self: *Renderer, ks: []const Kw, scope: *Scope) Error![]const KwVal {
        const out = try self.a.alloc(KwVal, ks.len);
        for (ks, out) |k, *o|
        {
            o.* = .{ .name = k.name, .value = try self.eval(k.value, scope) };
        }
        return out;
    }

    fn undefinedError(self: *Renderer, v: Value) Error {
        return self.fail(error.Undefined, "'{s}' is undefined", .{v.undefined});
    }

    fn getAttr(self: *Renderer, obj: Value, name: []const u8) Error!Value {
        switch (obj)
        {
            .undefined => return self.undefinedError(obj),
            .dict => |d| {
                if (known(&dict_methods, name))
                {
                    return self.bound(obj, name);
                }
                return d.get(name) orelse .{ .undefined = name };
            },
            .namespace => |d| return d.get(name) orelse .{ .undefined = name },
            .string => {
                if (known(&string_methods, name))
                {
                    return self.bound(obj, name);
                }
            },
            .list => {
                if (known(&list_methods, name))
                {
                    return self.bound(obj, name);
                }
            },
            else => {},
        }
        return .{ .undefined = name };
    }

    fn bound(self: *Renderer, obj: Value, name: []const u8) Error!Value {
        const m = try self.a.create(Method);
        m.* = .{ .self = obj, .name = name };
        return .{ .method = m };
    }

    fn getItem(self: *Renderer, obj: Value, key: Value) Error!Value {
        switch (obj)
        {
            .undefined => return self.undefinedError(obj),
            .dict, .namespace => |d| {
                if (key == .string)
                {
                    if (d.get(key.string)) |v|
                    {
                        return v;
                    }
                    return self.getAttr(obj, key.string);
                }
                return .{ .undefined = "" };
            },
            .list => |l| {
                if (key == .int)
                {
                    const i = if (key.int < 0) key.int + @as(i64, @intCast(l.len)) else key.int;
                    if (i >= 0 and i < l.len)
                    {
                        return l[@intCast(i)];
                    }
                }
                return .{ .undefined = "" };
            },
            .string => |s| {
                if (key == .int)
                {
                    const cps = try codepoints(self.a, s);
                    const i = if (key.int < 0) key.int + @as(i64, @intCast(cps.len)) else key.int;
                    if (i >= 0 and i < cps.len)
                    {
                        return .{ .string = cps[@intCast(i)] };
                    }
                    return .{ .undefined = "" };
                }
                if (key == .string)
                {
                    return self.getAttr(obj, key.string);
                }
                return .{ .undefined = "" };
            },
            else => return .{ .undefined = "" },
        }
    }

    fn slice(self: *Renderer, obj: Value, start: Value, stop: Value, step_v: Value) Error!Value {
        const step: i64 = switch (step_v) {
            .none => 1,
            .int => |i| i,
            else => return self.fail(error.Type, "slice step must be an integer", .{}),
        };
        if (step == 0)
        {
            return self.fail(error.Type, "slice step cannot be zero", .{});
        }
        switch (obj)
        {
            .undefined => return self.undefinedError(obj),
            .list => |l| {
                const idx = try self.sliceIndices(l.len, start, stop, step);
                const out = try self.a.alloc(Value, idx.len);
                for (idx, out) |i, *o|
                {
                    o.* = l[i];
                }
                return .{ .list = out };
            },
            .string => |s| {
                const cps = try codepoints(self.a, s);
                const idx = try self.sliceIndices(cps.len, start, stop, step);
                var out: std.ArrayList(u8) = .empty;
                for (idx) |i|
                {
                    try out.appendSlice(self.a, cps[i]);
                }
                return .{ .string = out.items };
            },
            else => return self.fail(error.Type, "{s} cannot be sliced", .{typeName(obj)}),
        }
    }

    /// Python's slice indices for a sequence of `len`.
    fn sliceIndices(self: *Renderer, len: usize, start_v: Value, stop_v: Value, step: i64) Error![]const usize {
        const n: i64 = @intCast(len);
        const bound_ = struct {
            fn f(v: Value, n_: i64, default: i64, lo: i64, hi: i64) ?i64 {
                const x = switch (v) {
                    .none => return default,
                    .int => |i| i,
                    else => return null,
                };
                const y = if (x < 0) x + n_ else x;
                return std.math.clamp(y, lo, hi);
            }
        }.f;
        const start = (if (step > 0) bound_(start_v, n, 0, 0, n) else bound_(start_v, n, n - 1, -1, n - 1)) orelse return self.fail(error.Type, "slice indices must be integers", .{});
        const stop = (if (step > 0) bound_(stop_v, n, n, 0, n) else bound_(stop_v, n, -1, -1, n - 1)) orelse return self.fail(error.Type, "slice indices must be integers", .{});
        var out: std.ArrayList(usize) = .empty;
        var i = start;
        while ((step > 0 and i < stop) or (step < 0 and i > stop))
        {
            try out.append(self.a, @intCast(i));
            i += step;
        }
        return out.items;
    }

    fn call(self: *Renderer, callee: Value, args: []const Value, kwargs: []const KwVal) Error!Value {
        switch (callee)
        {
            .macro => |m| return self.callMacro(m, args, kwargs),
            .builtin => |b| return self.callBuiltin(b, args, kwargs),
            .method => |m| return self.callMethod(m.self, m.name, args),
            .undefined => return self.undefinedError(callee),
            else => return self.fail(error.Type, "{s} is not callable", .{typeName(callee)}),
        }
    }

    fn callMacro(self: *Renderer, m: *const Macro, args: []const Value, kwargs: []const KwVal) Error!Value {
        // A macro sees the template's top-level names, not its caller's.
        var s: Scope = .{ .parent = self.top };
        if (args.len > m.params.len)
        {
            return self.fail(error.Type, "macro '{s}' takes {d} arguments", .{ m.name, m.params.len });
        }
        for (m.params, 0..) |p, i|
        {
            var v: ?Value = if (i < args.len) args[i] else null;
            for (kwargs) |k|
            {
                if (std.mem.eql(u8, k.name, p.name))
                {
                    v = k.value;
                }
            }
            if (v == null)
            {
                v = if (p.default) |d| try self.eval(d, &s) else .{ .undefined = p.name };
            }
            try s.vars.put(self.a, p.name, v.?);
        }
        var buf: std.ArrayList(u8) = .empty;
        const saved = self.out;
        const line = self.line;
        self.out = &buf;
        defer self.out = saved;
        _ = try self.exec(m.body, &s);
        self.line = line;
        return .{ .string = buf.items };
    }

    fn callBuiltin(self: *Renderer, b: Builtin, args: []const Value, kwargs: []const KwVal) Error!Value {
        switch (b)
        {
            .raise_exception => {
                var msg: std.ArrayList(u8) = .empty;
                if (args.len > 0)
                {
                    try pyStr(self.a, args[0], &msg);
                }
                return self.fail(error.Raised, "{s}", .{msg.items});
            },
            .range => {
                var lo: i64 = 0;
                var hi: i64 = 0;
                var step: i64 = 1;
                for (args) |x|
                {
                    if (x != .int)
                    {
                        return self.fail(error.Type, "range() takes integers", .{});
                    }
                }
                switch (args.len)
                {
                    1 => hi = args[0].int,
                    2 => {
                        lo = args[0].int;
                        hi = args[1].int;
                    },
                    3 => {
                        lo = args[0].int;
                        hi = args[1].int;
                        step = args[2].int;
                    },
                    else => return self.fail(error.Type, "range() takes 1 to 3 arguments", .{}),
                }
                if (step == 0)
                {
                    return self.fail(error.Type, "range() step cannot be zero", .{});
                }
                var out: std.ArrayList(Value) = .empty;
                var i = lo;
                while ((step > 0 and i < hi) or (step < 0 and i > hi))
                {
                    if (out.items.len > 1 << 20)
                    {
                        return self.fail(error.Type, "range too large", .{});
                    }
                    try out.append(self.a, .{ .int = i });
                    i += step;
                }
                return .{ .list = out.items };
            },
            .namespace, .dict => {
                const d = try self.a.create(Dict);
                d.* = .empty;
                for (args) |x|
                {
                    if (x == .dict)
                    {
                        var it = x.dict.iterator();
                        while (it.next()) |e|
                        {
                            try d.put(self.a, e.key_ptr.*, e.value_ptr.*);
                        }
                    }
                }
                for (kwargs) |k|
                {
                    try d.put(self.a, k.name, k.value);
                }
                return if (b == .namespace) .{ .namespace = d } else .{ .dict = d };
            },
            .strftime_now => {
                if (args.len != 1 or args[0] != .string)
                {
                    return self.fail(error.Type, "strftime_now() takes a format string", .{});
                }
                return .{ .string = try strftime(self.a, args[0].string, localNow()) };
            },
        }
    }

    fn callMethod(self: *Renderer, obj: Value, name: []const u8, args: []const Value) Error!Value {
        const eq = std.mem.eql;
        switch (obj)
        {
            .string => |s| {
                if (eq(u8, name, "strip") or eq(u8, name, "lstrip") or eq(u8, name, "rstrip"))
                {
                    const chars: ?[]const u8 = if (args.len > 0 and args[0] == .string) args[0].string else null;
                    return .{ .string = try pyStrip(self.a, s, chars, name[0] != 'r', name[0] != 'l') };
                }
                if (eq(u8, name, "split") or eq(u8, name, "rsplit"))
                {
                    const sep: ?[]const u8 = if (args.len > 0 and args[0] == .string) args[0].string else null;
                    const maxsplit: i64 = if (args.len > 1 and args[1] == .int) args[1].int else -1;
                    if (name[0] == 'r' and maxsplit >= 0)
                    {
                        return self.fail(error.Unsupported, "rsplit with a limit is not supported", .{});
                    }
                    return self.pySplit(s, sep, maxsplit);
                }
                if (eq(u8, name, "startswith") or eq(u8, name, "endswith"))
                {
                    if (args.len != 1)
                    {
                        return self.fail(error.Unsupported, "{s} with start and end positions is not supported", .{name});
                    }
                    const cands: []const Value = switch (args[0]) {
                        .string => &.{args[0]},
                        .list => |l| l,
                        else => return self.fail(error.Type, "{s} takes a string or a tuple", .{name}),
                    };
                    for (cands) |c|
                    {
                        if (c != .string)
                        {
                            return self.fail(error.Type, "{s} takes strings", .{name});
                        }
                        const hit = if (name[0] == 's') std.mem.startsWith(u8, s, c.string) else std.mem.endsWith(u8, s, c.string);
                        if (hit)
                        {
                            return .{ .boolean = true };
                        }
                    }
                    return .{ .boolean = false };
                }
                if (eq(u8, name, "upper") or eq(u8, name, "lower"))
                {
                    return .{ .string = try caseMap(self.a, s, name[0] == 'u') };
                }
                if (eq(u8, name, "capitalize"))
                {
                    return .{ .string = try capitalize(self.a, s) };
                }
                if (eq(u8, name, "title"))
                {
                    return .{ .string = try title(self.a, s) };
                }
                if (eq(u8, name, "replace"))
                {
                    if (args.len < 2 or args[0] != .string or args[1] != .string)
                    {
                        return self.fail(error.Type, "replace() takes two strings", .{});
                    }
                    const count: i64 = if (args.len > 2 and args[2] == .int) args[2].int else -1;
                    return .{ .string = try replace(self.a, s, args[0].string, args[1].string, count) };
                }
                if (eq(u8, name, "count") or eq(u8, name, "find"))
                {
                    if (args.len != 1 or args[0] != .string)
                    {
                        return self.fail(error.Type, "{s}() takes a string", .{name});
                    }
                    if (name[0] == 'c')
                    {
                        return .{ .int = @intCast(if (args[0].string.len == 0) (try codepoints(self.a, s)).len + 1 else std.mem.count(u8, s, args[0].string)) };
                    }
                    const at = std.mem.indexOf(u8, s, args[0].string) orelse return .{ .int = -1 };
                    return .{ .int = @intCast((try codepoints(self.a, s[0..at])).len) };
                }
            },
            .dict => |d| {
                if (eq(u8, name, "get"))
                {
                    if (args.len < 1 or args[0] != .string)
                    {
                        return if (args.len > 1) args[1] else .none;
                    }
                    return d.get(args[0].string) orelse (if (args.len > 1) args[1] else .none);
                }
                if (eq(u8, name, "items") or eq(u8, name, "keys") or eq(u8, name, "values"))
                {
                    const out = try self.a.alloc(Value, d.count());
                    for (d.keys(), d.values(), out) |k, v, *o|
                    {
                        o.* = switch (name[0]) {
                            'i' => blk: {
                                const pair = try self.a.alloc(Value, 2);
                                pair[0] = .{ .string = k };
                                pair[1] = v;
                                break :blk .{ .list = pair };
                            },
                            'k' => .{ .string = k },
                            else => v,
                        };
                    }
                    return .{ .list = out };
                }
            },
            else => {},
        }
        return self.fail(error.Unsupported, "the method '{s}' of a {s} is not supported", .{ name, typeName(obj) });
    }

    fn pySplit(self: *Renderer, s: []const u8, sep: ?[]const u8, maxsplit: i64) Error!Value {
        var out: std.ArrayList(Value) = .empty;
        if (sep) |sp|
        {
            if (sp.len == 0)
            {
                return self.fail(error.Type, "empty separator", .{});
            }
            var rest = s;
            var n: i64 = 0;
            while (maxsplit < 0 or n < maxsplit)
            {
                const at = std.mem.indexOf(u8, rest, sp) orelse break;
                try out.append(self.a, .{ .string = rest[0..at] });
                rest = rest[at + sp.len ..];
                n += 1;
            }
            try out.append(self.a, .{ .string = rest });
            return .{ .list = out.items };
        }
        // No separator: runs of white space, none empty.
        var i: usize = 0;
        var n: i64 = 0;
        while (true)
        {
            i = skipSpace(s, i);
            if (i >= s.len)
            {
                break;
            }
            if (maxsplit >= 0 and n == maxsplit)
            {
                try out.append(self.a, .{ .string = try pyStrip(self.a, s[i..], null, false, true) });
                break;
            }
            const j = toSpace(s, i);
            try out.append(self.a, .{ .string = s[i..j] });
            i = j;
            n += 1;
        }
        return .{ .list = out.items };
    }

    fn filter(self: *Renderer, name: []const u8, v: Value, args: []const Value, kwargs: []const KwVal) Error!Value {
        const eq = std.mem.eql;
        const kw = struct {
            fn get(ks: []const KwVal, n: []const u8) ?Value {
                for (ks) |k|
                {
                    if (std.mem.eql(u8, k.name, n))
                    {
                        return k.value;
                    }
                }
                return null;
            }
        }.get;
        if (eq(u8, name, "default") or eq(u8, name, "d"))
        {
            const dflt = if (args.len > 0) args[0] else kw(kwargs, "default_value") orelse Value{ .string = "" };
            const boolean = (if (args.len > 1) args[1] else kw(kwargs, "boolean") orelse Value{ .boolean = false }).truthy();
            if (v == .undefined or (boolean and !v.truthy()))
            {
                return dflt;
            }
            return v;
        }
        if (v == .undefined and !eq(u8, name, "string") and !eq(u8, name, "length") and !eq(u8, name, "count") and !eq(u8, name, "list") and !eq(u8, name, "safe") and !eq(u8, name, "trim"))
        {
            return self.undefinedError(v);
        }
        if (eq(u8, name, "safe"))
        {
            return v;
        }
        if (eq(u8, name, "length") or eq(u8, name, "count"))
        {
            return .{ .int = @intCast(switch (v) {
                .string => |s| (try codepoints(self.a, s)).len,
                .list => |l| l.len,
                .dict, .namespace => |d| d.count(),
                .undefined => 0,
                else => return self.fail(error.Type, "{s} has no length", .{typeName(v)}),
            }) };
        }
        if (eq(u8, name, "string"))
        {
            var out: std.ArrayList(u8) = .empty;
            try pyStr(self.a, v, &out);
            return .{ .string = out.items };
        }
        if (eq(u8, name, "trim"))
        {
            var out: std.ArrayList(u8) = .empty;
            try pyStr(self.a, v, &out);
            const chars: ?[]const u8 = if (args.len > 0 and args[0] == .string) args[0].string else null;
            return .{ .string = try pyStrip(self.a, out.items, chars, true, true) };
        }
        if (eq(u8, name, "tojson"))
        {
            const indent = if (args.len > 1) args[1] else kw(kwargs, "indent") orelse Value.none;
            var j: Json = .{
                .a = self.a,
                .ensure_ascii = (if (args.len > 0) args[0] else kw(kwargs, "ensure_ascii") orelse Value{ .boolean = false }).truthy(),
                .indent = if (indent == .int) indent.int else null,
                .sort_keys = (kw(kwargs, "sort_keys") orelse Value{ .boolean = false }).truthy(),
            };
            if (kw(kwargs, "separators")) |sep|
            {
                if (sep != .list or sep.list.len != 2 or sep.list[0] != .string or sep.list[1] != .string)
                {
                    return self.fail(error.Type, "separators must be two strings", .{});
                }
                j.item_sep = sep.list[0].string;
                j.key_sep = sep.list[1].string;
            }
            else if (j.indent != null)
            {
                j.item_sep = ",";
            }
            var out: std.ArrayList(u8) = .empty;
            j.write(v, &out, 0) catch |err| switch (err) {
                error.Type => return self.fail(error.Type, "a {s} cannot be written as JSON", .{typeName(j.bad)}),
                else => |e| return e,
            };
            return .{ .string = out.items };
        }
        if (eq(u8, name, "items"))
        {
            if (v == .none)
            {
                return .{ .list = &.{} };
            }
            if (v != .dict and v != .namespace)
            {
                return self.fail(error.Type, "items needs a mapping", .{});
            }
            return self.callMethod(.{ .dict = if (v == .dict) v.dict else v.namespace }, "items", &.{});
        }
        if (eq(u8, name, "list"))
        {
            return .{ .list = try self.iterate(v) };
        }
        if (eq(u8, name, "join"))
        {
            const sep = if (args.len > 0 and args[0] == .string) args[0].string else "";
            const attr = kw(kwargs, "attribute");
            var out: std.ArrayList(u8) = .empty;
            for (try self.iterate(v), 0..) |item, i|
            {
                if (i > 0)
                {
                    try out.appendSlice(self.a, sep);
                }
                const x = if (attr) |at| try self.attrPath(item, at) else item;
                try pyStr(self.a, x, &out);
            }
            return .{ .string = out.items };
        }
        if (eq(u8, name, "sort") or eq(u8, name, "unique"))
        {
            const is_sort = name[0] == 's';
            const case_sensitive = (kw(kwargs, "case_sensitive") orelse (if (!is_sort and args.len > 0) args[0] else if (is_sort and args.len > 1) args[1] else Value{ .boolean = false })).truthy();
            const reverse = is_sort and (kw(kwargs, "reverse") orelse (if (args.len > 0) args[0] else Value{ .boolean = false })).truthy();
            const attr = kw(kwargs, "attribute") orelse (if (is_sort and args.len > 2) args[2] else if (!is_sort and args.len > 1) args[1] else null);
            const items = try self.iterate(v);
            const keys = try self.a.alloc(Value, items.len);
            for (items, keys) |item, *k|
            {
                var x = if (attr) |at| try self.attrPath(item, at) else item;
                if (!case_sensitive and x == .string)
                {
                    x = .{ .string = try caseMap(self.a, x.string, false) };
                }
                k.* = x;
            }
            if (!is_sort)
            {
                var out: std.ArrayList(Value) = .empty;
                var seen: std.ArrayList(Value) = .empty;
                outer: for (items, keys) |item, k|
                {
                    for (seen.items) |s|
                    {
                        if (try self.equal(s, k))
                        {
                            continue :outer;
                        }
                    }
                    try seen.append(self.a, k);
                    try out.append(self.a, item);
                }
                return .{ .list = out.items };
            }
            const order = try self.a.alloc(usize, items.len);
            for (order, 0..) |*o, i|
            {
                o.* = i;
            }
            // Insertion sort keeps equal keys in order, as Python's sort does.
            var i: usize = 1;
            while (i < order.len)
            {
                var j = i;
                while (j > 0)
                {
                    const c = try self.cmp(keys[order[j - 1]], keys[order[j]]);
                    const swap = if (reverse) c == .lt else c == .gt;
                    if (!swap)
                    {
                        break;
                    }
                    std.mem.swap(usize, &order[j - 1], &order[j]);
                    j -= 1;
                }
                i += 1;
            }
            const out = try self.a.alloc(Value, items.len);
            for (order, out) |o, *x|
            {
                x.* = items[o];
            }
            return .{ .list = out };
        }
        if (eq(u8, name, "capitalize") or eq(u8, name, "upper") or eq(u8, name, "lower") or eq(u8, name, "title"))
        {
            var out: std.ArrayList(u8) = .empty;
            try pyStr(self.a, v, &out);
            return .{ .string = switch (name[0]) {
                'c' => try capitalize(self.a, out.items),
                'u' => try caseMap(self.a, out.items, true),
                'l' => try caseMap(self.a, out.items, false),
                else => try title(self.a, out.items),
            } };
        }
        if (eq(u8, name, "replace"))
        {
            var out: std.ArrayList(u8) = .empty;
            try pyStr(self.a, v, &out);
            return self.callMethod(.{ .string = out.items }, "replace", args);
        }
        if (eq(u8, name, "format"))
        {
            if (v != .string)
            {
                return self.fail(error.Type, "format needs a string", .{});
            }
            return .{ .string = try self.percent(v.string, args) };
        }
        if (eq(u8, name, "first") or eq(u8, name, "last"))
        {
            const items = try self.iterate(v);
            if (items.len == 0)
            {
                return .{ .undefined = name };
            }
            return if (name[0] == 'f') items[0] else items[items.len - 1];
        }
        if (eq(u8, name, "reverse"))
        {
            if (v == .string)
            {
                return self.slice(v, .none, .none, .{ .int = -1 });
            }
            const items = try self.iterate(v);
            const out = try self.a.alloc(Value, items.len);
            for (items, 0..) |x, i|
            {
                out[items.len - 1 - i] = x;
            }
            return .{ .list = out };
        }
        if (eq(u8, name, "int") or eq(u8, name, "float") or eq(u8, name, "abs"))
        {
            const f: f64 = switch (v) {
                .int => |i| @floatFromInt(i),
                .float => |x| x,
                .boolean => |b| @floatFromInt(@intFromBool(b)),
                .string => |s| std.fmt.parseFloat(f64, std.mem.trim(u8, s, " \t\n")) catch 0,
                else => 0,
            };
            if (name[0] == 'i')
            {
                return .{ .int = @intFromFloat(std.math.trunc(f)) };
            }
            if (name[0] == 'f')
            {
                return .{ .float = f };
            }
            return if (v == .int) .{ .int = if (v.int < 0) -v.int else v.int } else .{ .float = @abs(f) };
        }
        if (eq(u8, name, "map") or eq(u8, name, "selectattr") or eq(u8, name, "rejectattr"))
        {
            const items = try self.iterate(v);
            var out: std.ArrayList(Value) = .empty;
            if (name[0] == 'm')
            {
                const attr = kw(kwargs, "attribute") orelse return self.fail(error.Unsupported, "map is supported only with attribute=", .{});
                for (items) |x|
                {
                    try out.append(self.a, try self.attrPath(x, attr));
                }
                return .{ .list = out.items };
            }
            if (args.len < 1)
            {
                return self.fail(error.Type, "{s} needs an attribute", .{name});
            }
            for (items) |x|
            {
                const field = try self.attrPath(x, args[0]);
                const ok = if (args.len > 1 and args[1] == .string) try self.testValue(args[1].string, field, args[2..]) else field.truthy();
                if (ok == (name[0] == 's'))
                {
                    try out.append(self.a, x);
                }
            }
            return .{ .list = out.items };
        }
        return self.fail(error.Unsupported, "the filter '{s}' is not supported", .{name});
    }

    /// An attribute, or dotted path of attributes, as filters' `attribute=` takes.
    fn attrPath(self: *Renderer, v: Value, path: Value) Error!Value {
        if (path == .int)
        {
            return self.getItem(v, path);
        }
        if (path != .string)
        {
            return self.fail(error.Type, "attribute must be a string", .{});
        }
        var x = v;
        var it = std.mem.splitScalar(u8, path.string, '.');
        while (it.next()) |part|
        {
            x = try self.getItem(x, .{ .string = part });
        }
        return x;
    }

    /// Python's % formatting with %s, %d, %i and %%.
    fn percent(self: *Renderer, fmt: []const u8, args: []const Value) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var k: usize = 0;
        var i: usize = 0;
        while (i < fmt.len)
        {
            if (fmt[i] != '%' or i + 1 >= fmt.len)
            {
                try out.append(self.a, fmt[i]);
                i += 1;
                continue;
            }
            const c = fmt[i + 1];
            i += 2;
            if (c == '%')
            {
                try out.append(self.a, '%');
                continue;
            }
            if (k >= args.len)
            {
                return self.fail(error.Type, "not enough arguments for format string", .{});
            }
            const x = args[k];
            k += 1;
            switch (c)
            {
                's' => try pyStr(self.a, x, &out),
                'r' => try pyRepr(self.a, x, &out),
                'd', 'i' => {
                    const n: i64 = switch (x) {
                        .int => |n| n,
                        .float => |f| @intFromFloat(std.math.trunc(f)),
                        .boolean => |b| @intFromBool(b),
                        else => return self.fail(error.Type, "%d needs a number", .{}),
                    };
                    try out.print(self.a, "{d}", .{n});
                },
                else => return self.fail(error.Unsupported, "the format '%{c}' is not supported", .{c}),
            }
        }
        return out.items;
    }

    fn testValue(self: *Renderer, name: []const u8, v: Value, args: []const Value) Error!bool {
        const eq = std.mem.eql;
        if (eq(u8, name, "defined"))
        {
            return v != .undefined;
        }
        if (eq(u8, name, "undefined"))
        {
            return v == .undefined;
        }
        if (eq(u8, name, "none"))
        {
            return v == .none;
        }
        if (eq(u8, name, "string"))
        {
            return v == .string;
        }
        if (eq(u8, name, "mapping"))
        {
            return v == .dict;
        }
        if (eq(u8, name, "iterable"))
        {
            return v == .list or v == .dict or v == .string or v == .namespace or v == .undefined;
        }
        if (eq(u8, name, "sequence"))
        {
            return v == .list or v == .dict or v == .string;
        }
        if (eq(u8, name, "number"))
        {
            return v == .int or v == .float or v == .boolean;
        }
        if (eq(u8, name, "integer"))
        {
            return v == .int;
        }
        if (eq(u8, name, "float"))
        {
            return v == .float;
        }
        if (eq(u8, name, "boolean"))
        {
            return v == .boolean;
        }
        if (eq(u8, name, "true") or eq(u8, name, "false"))
        {
            return v == .boolean and v.boolean == (name[0] == 't');
        }
        if (eq(u8, name, "callable"))
        {
            return v == .macro or v == .builtin or v == .method;
        }
        if (eq(u8, name, "odd") or eq(u8, name, "even"))
        {
            if (v != .int)
            {
                return self.fail(error.Type, "{s} needs an integer", .{name});
            }
            return @mod(v.int, 2) == @intFromBool(name[0] == 'o');
        }
        if (eq(u8, name, "divisibleby"))
        {
            if (v != .int or args.len != 1 or args[0] != .int or args[0].int == 0)
            {
                return self.fail(error.Type, "divisibleby needs integers", .{});
            }
            return @rem(v.int, args[0].int) == 0;
        }
        if (eq(u8, name, "equalto") or eq(u8, name, "eq"))
        {
            if (args.len != 1)
            {
                return self.fail(error.Type, "{s} needs one argument", .{name});
            }
            return self.equal(v, args[0]);
        }
        if (eq(u8, name, "lower") or eq(u8, name, "upper"))
        {
            if (v != .string)
            {
                return false;
            }
            return eq(u8, v.string, try caseMap(self.a, v.string, name[0] == 'u'));
        }
        return self.fail(error.Unsupported, "the test '{s}' is not supported", .{name});
    }

    fn binary(self: *Renderer, op: BinOp, l: Value, r: Value) Error!Value {
        switch (op)
        {
            .eq => return .{ .boolean = try self.equal(l, r) },
            .ne => return .{ .boolean = !try self.equal(l, r) },
            .lt, .le, .gt, .ge => {
                const c = try self.cmp(l, r);
                return .{ .boolean = switch (op) {
                    .lt => c == .lt,
                    .le => c != .gt,
                    .gt => c == .gt,
                    else => c != .lt,
                } };
            },
            .in, .not_in => {
                const hit = try self.contains(r, l);
                return .{ .boolean = hit == (op == .in) };
            },
            .concat => {
                var out: std.ArrayList(u8) = .empty;
                try pyStr(self.a, l, &out);
                try pyStr(self.a, r, &out);
                return .{ .string = out.items };
            },
            else => {},
        }
        if (l == .undefined)
        {
            return self.undefinedError(l);
        }
        if (r == .undefined)
        {
            return self.undefinedError(r);
        }
        if (op == .add)
        {
            if (l == .string and r == .string)
            {
                return .{ .string = try std.mem.concat(self.a, u8, &.{ l.string, r.string }) };
            }
            if (l == .list and r == .list)
            {
                return .{ .list = try std.mem.concat(self.a, Value, &.{ l.list, r.list }) };
            }
        }
        if (op == .mod and l == .string)
        {
            const args: []const Value = if (r == .list) r.list else &.{r};
            return .{ .string = try self.percent(l.string, args) };
        }
        if (op == .mul and (l == .string or r == .string))
        {
            const s = if (l == .string) l.string else r.string;
            const n = if (l == .string) r else l;
            if (n != .int)
            {
                return self.fail(error.Type, "can only repeat a string by an integer", .{});
            }
            var out: std.ArrayList(u8) = .empty;
            var k: i64 = 0;
            while (k < n.int)
            {
                try out.appendSlice(self.a, s);
                k += 1;
            }
            return .{ .string = out.items };
        }
        const ln = number(l) orelse return self.fail(error.Type, "unsupported operand types: {s} and {s}", .{ typeName(l), typeName(r) });
        const rn = number(r) orelse return self.fail(error.Type, "unsupported operand types: {s} and {s}", .{ typeName(l), typeName(r) });
        if (ln == .int and rn == .int and op != .div)
        {
            const x = ln.int;
            const y = rn.int;
            switch (op)
            {
                .add => return .{ .int = std.math.add(i64, x, y) catch return self.fail(error.Type, "integer overflow", .{}) },
                .sub => return .{ .int = std.math.sub(i64, x, y) catch return self.fail(error.Type, "integer overflow", .{}) },
                .mul => return .{ .int = std.math.mul(i64, x, y) catch return self.fail(error.Type, "integer overflow", .{}) },
                .floordiv, .mod => {
                    if (y == 0)
                    {
                        return self.fail(error.Type, "division by zero", .{});
                    }
                    var q = @divTrunc(x, y);
                    var m = @rem(x, y);
                    if (m != 0 and (m < 0) != (y < 0))
                    {
                        q -= 1;
                        m += y;
                    }
                    return .{ .int = if (op == .floordiv) q else m };
                },
                .pow => {
                    if (y >= 0)
                    {
                        return .{ .int = std.math.powi(i64, x, y) catch return self.fail(error.Type, "integer overflow", .{}) };
                    }
                },
                else => {},
            }
        }
        const x = ln.float();
        const y = rn.float();
        return .{ .float = switch (op) {
            .add => x + y,
            .sub => x - y,
            .mul => x * y,
            .div => if (y == 0) return self.fail(error.Type, "division by zero", .{}) else x / y,
            .floordiv => if (y == 0) return self.fail(error.Type, "division by zero", .{}) else @floor(x / y),
            .mod => if (y == 0) return self.fail(error.Type, "division by zero", .{}) else x - y * @floor(x / y),
            .pow => std.math.pow(f64, x, y),
            else => unreachable,
        } };
    }

    const Num = union(enum) {
        int: i64,
        flt: f64,

        fn float(self: Num) f64 {
            return switch (self) {
                .int => |i| @floatFromInt(i),
                .flt => |f| f,
            };
        }
    };

    fn number(v: Value) ?Num {
        return switch (v) {
            .int => |i| .{ .int = i },
            .boolean => |b| .{ .int = @intFromBool(b) },
            .float => |f| .{ .flt = f },
            else => null,
        };
    }

    fn equal(self: *Renderer, l: Value, r: Value) Error!bool {
        if (number(l)) |a|
        {
            if (number(r)) |b|
            {
                if (a == .int and b == .int)
                {
                    return a.int == b.int;
                }
                return a.float() == b.float();
            }
            return false;
        }
        return switch (l) {
            .undefined => r == .undefined,
            .none => r == .none,
            .string => |s| r == .string and std.mem.eql(u8, s, r.string),
            .list => |a| blk: {
                if (r != .list or r.list.len != a.len)
                {
                    break :blk false;
                }
                for (a, r.list) |x, y|
                {
                    if (!try self.equal(x, y))
                    {
                        break :blk false;
                    }
                }
                break :blk true;
            },
            .dict => |a| blk: {
                if (r != .dict or r.dict.count() != a.count())
                {
                    break :blk false;
                }
                var it = a.iterator();
                while (it.next()) |e|
                {
                    const y = r.dict.get(e.key_ptr.*) orelse break :blk false;
                    if (!try self.equal(e.value_ptr.*, y))
                    {
                        break :blk false;
                    }
                }
                break :blk true;
            },
            .namespace => |a| r == .namespace and r.namespace == a,
            .macro => |m| r == .macro and r.macro == m,
            .builtin => |b| r == .builtin and r.builtin == b,
            .method => false,
            else => false,
        };
    }

    fn cmp(self: *Renderer, l: Value, r: Value) Error!std.math.Order {
        if (number(l)) |a|
        {
            if (number(r)) |b|
            {
                if (a == .int and b == .int)
                {
                    return std.math.order(a.int, b.int);
                }
                return std.math.order(a.float(), b.float());
            }
        }
        if (l == .string and r == .string)
        {
            return std.mem.order(u8, l.string, r.string);
        }
        if (l == .list and r == .list)
        {
            for (l.list, 0..) |x, i|
            {
                if (i >= r.list.len)
                {
                    return .gt;
                }
                if (!try self.equal(x, r.list[i]))
                {
                    return self.cmp(x, r.list[i]);
                }
            }
            return std.math.order(l.list.len, r.list.len);
        }
        if (l == .undefined)
        {
            return self.undefinedError(l);
        }
        if (r == .undefined)
        {
            return self.undefinedError(r);
        }
        return self.fail(error.Type, "cannot compare {s} and {s}", .{ typeName(l), typeName(r) });
    }

    fn contains(self: *Renderer, hay: Value, needle: Value) Error!bool {
        switch (hay)
        {
            .string => |s| {
                if (needle != .string)
                {
                    return self.fail(error.Type, "'in <string>' needs a string on the left", .{});
                }
                return std.mem.indexOf(u8, s, needle.string) != null;
            },
            .list => |l| {
                for (l) |x|
                {
                    if (try self.equal(x, needle))
                    {
                        return true;
                    }
                }
                return false;
            },
            .dict, .namespace => |d| return needle == .string and d.contains(needle.string),
            .undefined => return false,
            else => return self.fail(error.Type, "a {s} cannot contain anything", .{typeName(hay)}),
        }
    }
};

const dict_methods = [_][]const u8{ "get", "items", "keys", "values", "pop", "update", "copy", "setdefault", "clear", "popitem" };
const string_methods = [_][]const u8{ "strip", "lstrip", "rstrip", "split", "rsplit", "startswith", "endswith", "upper", "lower", "capitalize", "title", "replace", "count", "find", "join", "format" };
const list_methods = [_][]const u8{ "append", "extend", "insert", "pop", "remove", "index", "count", "sort", "reverse", "copy", "clear" };

fn typeName(v: Value) []const u8 {
    return switch (v) {
        .undefined => "undefined value",
        .none => "None",
        .boolean => "bool",
        .int => "int",
        .float => "float",
        .string => "str",
        .list => "list",
        .dict => "dict",
        .namespace => "namespace",
        .macro => "macro",
        .builtin => "function",
        .method => "method",
    };
}

/// A string's characters as UTF-8 slices; invalid bytes count as one character each.
fn codepoints(a: std.mem.Allocator, s: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < s.len)
    {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        const end = @min(i + n, s.len);
        try out.append(a, s[i..end]);
        i = end;
    }
    return out.items;
}

/// Python's str.isspace for one character.
fn pySpaceCp(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

fn cpAt(s: []const u8, i: usize) struct { cp: u21, len: usize } {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .cp = 0xFFFD, .len = 1 };
    if (i + n > s.len)
    {
        return .{ .cp = 0xFFFD, .len = 1 };
    }
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return .{ .cp = 0xFFFD, .len = 1 };
    return .{ .cp = cp, .len = n };
}

fn skipSpace(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len)
    {
        const c = cpAt(s, i);
        if (!pySpaceCp(c.cp))
        {
            break;
        }
        i += c.len;
    }
    return i;
}

fn toSpace(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len)
    {
        const c = cpAt(s, i);
        if (pySpaceCp(c.cp))
        {
            break;
        }
        i += c.len;
    }
    return i;
}

/// Python's strip, lstrip and rstrip: white space, or any of `chars`.
fn pyStrip(a: std.mem.Allocator, s: []const u8, chars: ?[]const u8, left: bool, right: bool) ![]const u8 {
    const cps = try codepoints(a, s);
    const set = if (chars) |cs| try codepoints(a, cs) else null;
    const strip = struct {
        fn f(c: []const u8, set_: ?[]const []const u8) bool {
            if (set_) |st|
            {
                for (st) |x|
                {
                    if (std.mem.eql(u8, x, c))
                    {
                        return true;
                    }
                }
                return false;
            }
            return pySpaceCp(cpAt(c, 0).cp);
        }
    }.f;
    var lo: usize = 0;
    var hi: usize = cps.len;
    while (left and lo < hi and strip(cps[lo], set))
    {
        lo += 1;
    }
    while (right and hi > lo and strip(cps[hi - 1], set))
    {
        hi -= 1;
    }
    if (lo == hi)
    {
        return "";
    }
    const start = @intFromPtr(cps[lo].ptr) - @intFromPtr(s.ptr);
    const end = @intFromPtr(cps[hi - 1].ptr) - @intFromPtr(s.ptr) + cps[hi - 1].len;
    return s[start..end];
}

/// Upper or lower case; ASCII and the Latin-1, Greek and Cyrillic letters that map one to one.
fn caseMap(a: std.mem.Allocator, s: []const u8, upper: bool) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len)
    {
        const c = cpAt(s, i);
        var buf: [4]u8 = undefined;
        const m = mapCase(c.cp, upper);
        const n = std.unicode.utf8Encode(m, &buf) catch 0;
        if (c.cp == 0xFFFD and c.len == 1)
        {
            try out.append(a, s[i]);
        }
        else
        {
            try out.appendSlice(a, buf[0..n]);
        }
        i += c.len;
    }
    return out.items;
}

fn mapCase(cp: u21, upper: bool) u21 {
    if (upper)
    {
        return switch (cp) {
            'a'...'z' => cp - 32,
            0xE0...0xF6, 0xF8...0xFE => cp - 32,
            0x3B1...0x3C1, 0x3C3...0x3CB => cp - 32,
            0x430...0x44F => cp - 32,
            0x450...0x45F => cp - 80,
            else => cp,
        };
    }
    return switch (cp) {
        'A'...'Z' => cp + 32,
        0xC0...0xD6, 0xD8...0xDE => cp + 32,
        0x391...0x3A1, 0x3A3...0x3AB => cp + 32,
        0x410...0x42F => cp + 32,
        0x400...0x40F => cp + 80,
        else => cp,
    };
}

fn capitalize(a: std.mem.Allocator, s: []const u8) ![]const u8 {
    if (s.len == 0)
    {
        return s;
    }
    const first = cpAt(s, 0);
    const head = try caseMap(a, s[0..first.len], true);
    const rest = try caseMap(a, s[first.len..], false);
    return std.mem.concat(a, u8, &.{ head, rest });
}

fn title(a: std.mem.Allocator, s: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var prev_letter = false;
    var i: usize = 0;
    while (i < s.len)
    {
        const c = cpAt(s, i);
        const letter = mapCase(c.cp, true) != c.cp or mapCase(c.cp, false) != c.cp;
        try out.appendSlice(a, try caseMap(a, s[i .. i + c.len], !prev_letter));
        prev_letter = letter;
        i += c.len;
    }
    return out.items;
}

fn replace(a: std.mem.Allocator, s: []const u8, old: []const u8, new: []const u8, count: i64) ![]const u8 {
    if (old.len == 0)
    {
        return s;
    }
    var out: std.ArrayList(u8) = .empty;
    var rest = s;
    var n: i64 = 0;
    while (count < 0 or n < count)
    {
        const at = std.mem.indexOf(u8, rest, old) orelse break;
        try out.appendSlice(a, rest[0..at]);
        try out.appendSlice(a, new);
        rest = rest[at + old.len ..];
        n += 1;
    }
    try out.appendSlice(a, rest);
    return out.items;
}

/// Python's str() of a value, as Jinja prints it.
pub fn pyStr(a: std.mem.Allocator, v: Value, out: *std.ArrayList(u8)) error{OutOfMemory}!void {
    switch (v)
    {
        .undefined => {},
        .string => |s| try out.appendSlice(a, s),
        else => try pyRepr(a, v, out),
    }
}

/// Python's repr() of a value.
fn pyRepr(a: std.mem.Allocator, v: Value, out: *std.ArrayList(u8)) error{OutOfMemory}!void {
    switch (v)
    {
        .undefined => {},
        .none => try out.appendSlice(a, "None"),
        .boolean => |b| try out.appendSlice(a, if (b) "True" else "False"),
        .int => |i| try out.print(a, "{d}", .{i}),
        .float => |f| try pyFloat(a, f, out),
        .string => |s| try reprString(a, s, out),
        .list => |l| {
            try out.append(a, '[');
            for (l, 0..) |x, i|
            {
                if (i > 0)
                {
                    try out.appendSlice(a, ", ");
                }
                try pyRepr(a, x, out);
            }
            try out.append(a, ']');
        },
        .dict => |d| {
            try out.append(a, '{');
            var it = d.iterator();
            var first = true;
            while (it.next()) |e|
            {
                if (!first)
                {
                    try out.appendSlice(a, ", ");
                }
                first = false;
                try reprString(a, e.key_ptr.*, out);
                try out.appendSlice(a, ": ");
                try pyRepr(a, e.value_ptr.*, out);
            }
            try out.append(a, '}');
        },
        .namespace => try out.appendSlice(a, "<Namespace>"),
        .macro => |m| try out.print(a, "<Macro '{s}'>", .{m.name}),
        .builtin => try out.appendSlice(a, "<function>"),
        .method => try out.appendSlice(a, "<method>"),
    }
}

fn reprString(a: std.mem.Allocator, s: []const u8, out: *std.ArrayList(u8)) !void {
    const q: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try out.append(a, q);
    var i: usize = 0;
    while (i < s.len)
    {
        const c = cpAt(s, i);
        switch (c.cp)
        {
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            0...0x08, 0x0B, 0x0C, 0x0E...0x1F, 0x7F...0xA0, 0xAD => try out.print(a, "\\x{x:0>2}", .{c.cp}),
            0x2028, 0x2029 => try out.print(a, "\\u{x:0>4}", .{c.cp}),
            else => {
                if (c.cp == q)
                {
                    try out.append(a, '\\');
                }
                try out.appendSlice(a, s[i .. i + c.len]);
            },
        }
        i += c.len;
    }
    try out.append(a, q);
}

/// Python's repr() of a float: the shortest digits that read back, in fixed or scientific form.
fn pyFloat(a: std.mem.Allocator, f: f64, out: *std.ArrayList(u8)) !void {
    if (std.math.isNan(f))
    {
        return out.appendSlice(a, "nan");
    }
    if (std.math.isInf(f))
    {
        return out.appendSlice(a, if (f < 0) "-inf" else "inf");
    }
    if (f == 0)
    {
        return out.appendSlice(a, if (std.math.signbit(f)) "-0.0" else "0.0");
    }
    var buf: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&buf, "{e}", .{@abs(f)}) catch unreachable;
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exp = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch 0;
    var digits: [32]u8 = undefined;
    var nd: usize = 0;
    for (sci[0..e_at]) |ch|
    {
        if (ch != '.')
        {
            digits[nd] = ch;
            nd += 1;
        }
    }
    while (nd > 1 and digits[nd - 1] == '0')
    {
        nd -= 1;
    }
    if (f < 0)
    {
        try out.append(a, '-');
    }
    if (exp >= -4 and exp < 16)
    {
        if (exp < 0)
        {
            try out.appendSlice(a, "0.");
            var k: i32 = -1;
            while (k > exp)
            {
                try out.append(a, '0');
                k -= 1;
            }
            try out.appendSlice(a, digits[0..nd]);
            return;
        }
        const int_len: usize = @intCast(exp + 1);
        for (0..int_len) |k|
        {
            try out.append(a, if (k < nd) digits[k] else '0');
        }
        try out.append(a, '.');
        if (nd > int_len)
        {
            try out.appendSlice(a, digits[int_len..nd]);
        }
        else
        {
            try out.append(a, '0');
        }
        return;
    }
    try out.append(a, digits[0]);
    if (nd > 1)
    {
        try out.append(a, '.');
        try out.appendSlice(a, digits[1..nd]);
    }
    try out.print(a, "e{s}{d:0>2}", .{ if (exp < 0) "-" else "+", @abs(exp) });
}

/// Python's json.dumps, as transformers' tojson calls it.
const Json = struct {
    a: std.mem.Allocator,
    ensure_ascii: bool,
    indent: ?i64,
    sort_keys: bool,
    item_sep: []const u8 = ", ",
    key_sep: []const u8 = ": ",
    bad: Value = .none,

    fn newline(self: *Json, out: *std.ArrayList(u8), depth: usize) !void {
        try out.append(self.a, '\n');
        const n: usize = @intCast(@max(self.indent.?, 0));
        try out.appendNTimes(self.a, ' ', n * depth);
    }

    fn write(self: *Json, v: Value, out: *std.ArrayList(u8), depth: usize) error{ OutOfMemory, Type }!void {
        switch (v)
        {
            .none => try out.appendSlice(self.a, "null"),
            .boolean => |b| try out.appendSlice(self.a, if (b) "true" else "false"),
            .int => |i| try out.print(self.a, "{d}", .{i}),
            .float => |f| {
                if (std.math.isNan(f))
                {
                    try out.appendSlice(self.a, "NaN");
                }
                else if (std.math.isInf(f))
                {
                    try out.appendSlice(self.a, if (f < 0) "-Infinity" else "Infinity");
                }
                else
                {
                    try pyFloat(self.a, f, out);
                }
            },
            .string => |s| try self.string(s, out),
            .list => |l| {
                if (l.len == 0)
                {
                    return out.appendSlice(self.a, "[]");
                }
                try out.append(self.a, '[');
                for (l, 0..) |x, i|
                {
                    if (i > 0)
                    {
                        try out.appendSlice(self.a, self.item_sep);
                    }
                    if (self.indent != null)
                    {
                        try self.newline(out, depth + 1);
                    }
                    try self.write(x, out, depth + 1);
                }
                if (self.indent != null)
                {
                    try self.newline(out, depth);
                }
                try out.append(self.a, ']');
            },
            .dict => |d| {
                if (d.count() == 0)
                {
                    return out.appendSlice(self.a, "{}");
                }
                const keys = try self.a.dupe([]const u8, d.keys());
                if (self.sort_keys)
                {
                    std.mem.sort([]const u8, keys, {}, struct {
                        fn less(_: void, x: []const u8, y: []const u8) bool {
                            return std.mem.order(u8, x, y) == .lt;
                        }
                    }.less);
                }
                try out.append(self.a, '{');
                for (keys, 0..) |k, i|
                {
                    if (i > 0)
                    {
                        try out.appendSlice(self.a, self.item_sep);
                    }
                    if (self.indent != null)
                    {
                        try self.newline(out, depth + 1);
                    }
                    try self.string(k, out);
                    try out.appendSlice(self.a, self.key_sep);
                    try self.write(d.get(k).?, out, depth + 1);
                }
                if (self.indent != null)
                {
                    try self.newline(out, depth);
                }
                try out.append(self.a, '}');
            },
            else => {
                self.bad = v;
                return error.Type;
            },
        }
    }

    fn string(self: *Json, s: []const u8, out: *std.ArrayList(u8)) !void {
        try out.append(self.a, '"');
        var i: usize = 0;
        while (i < s.len)
        {
            const c = cpAt(s, i);
            switch (c.cp)
            {
                '"' => try out.appendSlice(self.a, "\\\""),
                '\\' => try out.appendSlice(self.a, "\\\\"),
                '\n' => try out.appendSlice(self.a, "\\n"),
                '\r' => try out.appendSlice(self.a, "\\r"),
                '\t' => try out.appendSlice(self.a, "\\t"),
                0x08 => try out.appendSlice(self.a, "\\b"),
                0x0C => try out.appendSlice(self.a, "\\f"),
                0...0x07, 0x0B, 0x0E...0x1F => try out.print(self.a, "\\u{x:0>4}", .{c.cp}),
                else => {
                    if (self.ensure_ascii and c.cp > 0x7E)
                    {
                        if (c.cp >= 0x10000)
                        {
                            const v = c.cp - 0x10000;
                            try out.print(self.a, "\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF) });
                        }
                        else
                        {
                            try out.print(self.a, "\\u{x:0>4}", .{c.cp});
                        }
                    }
                    else
                    {
                        try out.appendSlice(self.a, s[i .. i + c.len]);
                    }
                },
            }
            i += c.len;
        }
        try out.append(self.a, '"');
    }
};

/// The local date and time, as Python's datetime.now() gives it.
pub const Now = struct { year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8, weekday: u8, yday: u16 };

const Tm = extern struct {
    sec: c_int,
    min: c_int,
    hour: c_int,
    mday: c_int,
    mon: c_int,
    year: c_int,
    wday: c_int,
    yday: c_int,
    isdst: c_int,
    rest: [64]u8,
};

extern "c" fn time(t: ?*i64) i64;
extern "c" fn localtime_r(t: *const i64, tm: *Tm) ?*Tm;
extern "c" fn _localtime64_s(tm: *Tm, t: *const i64) c_int;

fn localNow() Now {
    var t: i64 = time(null);
    var tm: Tm = undefined;
    const ok = if (@import("builtin").os.tag == .windows) _localtime64_s(&tm, &t) == 0 else localtime_r(&t, &tm) != null;
    if (!ok)
    {
        return .{ .year = 1970, .month = 1, .day = 1, .hour = 0, .minute = 0, .second = 0, .weekday = 4, .yday = 1 };
    }
    return .{
        .year = tm.year + 1900,
        .month = @intCast(tm.mon + 1),
        .day = @intCast(tm.mday),
        .hour = @intCast(tm.hour),
        .minute = @intCast(tm.min),
        .second = @intCast(tm.sec),
        .weekday = @intCast(tm.wday),
        .yday = @intCast(tm.yday + 1),
    };
}

/// strftime with the C locale's names, for the directives templates use.
fn strftime(a: std.mem.Allocator, fmt: []const u8, now: Now) ![]const u8 {
    const months = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };
    const days = [_][]const u8{ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < fmt.len)
    {
        if (fmt[i] != '%' or i + 1 >= fmt.len)
        {
            try out.append(a, fmt[i]);
            i += 1;
            continue;
        }
        const c = fmt[i + 1];
        i += 2;
        switch (c)
        {
            'Y' => try out.print(a, "{d}", .{now.year}),
            'y' => try out.print(a, "{d:0>2}", .{@as(u32, @intCast(@mod(now.year, 100)))}),
            'm' => try out.print(a, "{d:0>2}", .{now.month}),
            'd' => try out.print(a, "{d:0>2}", .{now.day}),
            'H' => try out.print(a, "{d:0>2}", .{now.hour}),
            'M' => try out.print(a, "{d:0>2}", .{now.minute}),
            'S' => try out.print(a, "{d:0>2}", .{now.second}),
            'j' => try out.print(a, "{d:0>3}", .{now.yday}),
            'B' => try out.appendSlice(a, months[now.month - 1]),
            'b' => try out.appendSlice(a, months[now.month - 1][0..3]),
            'A' => try out.appendSlice(a, days[now.weekday]),
            'a' => try out.appendSlice(a, days[now.weekday][0..3]),
            '%' => try out.append(a, '%'),
            else => {
                try out.append(a, '%');
                try out.append(a, c);
            },
        }
    }
    return out.items;
}

/// A JSON value as a template value; objects keep their key order.
pub fn fromJson(a: std.mem.Allocator, j: std.json.Value) !Value {
    return switch (j) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .float = std.fmt.parseFloat(f64, s) catch 0 },
        .string => |s| .{ .string = s },
        .array => |arr| blk: {
            const out = try a.alloc(Value, arr.items.len);
            for (arr.items, out) |x, *o|
            {
                o.* = try fromJson(a, x);
            }
            break :blk .{ .list = out };
        },
        .object => |o| blk: {
            const d = try a.create(Dict);
            d.* = .empty;
            var it = o.iterator();
            while (it.next()) |e|
            {
                try d.put(a, e.key_ptr.*, try fromJson(a, e.value_ptr.*));
            }
            break :blk .{ .dict = d };
        },
    };
}

fn expectRender(src: []const u8, vars_json: []const u8, want: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diag: Diag = .{};
    const t = Template.parse(a, src, &diag) catch |e| {
        std.debug.print("parse: {s} (line {d})\n", .{ diag.msg, diag.line });
        return e;
    };
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, vars_json, .{});
    const vars = (try fromJson(a, parsed)).dict;
    var out: std.ArrayList(u8) = .empty;
    t.render(a, vars, &diag, &out) catch |e| {
        std.debug.print("render: {s} (line {d})\n", .{ diag.msg, diag.line });
        return e;
    };
    try std.testing.expectEqualStrings(want, out.items);
}

// Expected outputs below are what Jinja2 3.1.6 renders in transformers' environment.
test "whitespace: trim_blocks, lstrip_blocks and the - and + markers" {
    try expectRender("a\n  {% if true %}\n  b\n  {% endif %}\nc", "{}", "a\n  b\nc");
    try expectRender("a\n  {# c #}\nb", "{}", "a\nb");
    try expectRender("a\n  {{ 1 }}\nb", "{}", "a\n  1\nb");
    try expectRender("a  {%- if true -%}  b  {%- endif -%}  c", "{}", "abc");
    try expectRender("a\n  {%+ if true +%}\nb{% endif %}", "{}", "a\n  \nb");
    try expectRender("{% if true %}  {% if true %}x{% endif %}{% endif %}", "{}", "  x");
    try expectRender("  {% if true %}x{% endif %}", "{}", "x");
    try expectRender("x\n", "{}", "x");
}

test "scopes: loop passes, macros and generation" {
    try expectRender("{% set x = 1 %}{% for i in [1,2,3] %}{{ x }}{% set x = x + 1 %}{{ x }},{% endfor %}|{{ x }}", "{}", "12,12,12,|1");
    try expectRender("{% for i in [1,2] %}{% if i == 1 %}{% set y = 5 %}{% endif %}{{ y is defined }},{% endfor %}", "{}", "True,False,");
    try expectRender("{% macro m() %}{{ z }}{% endmacro %}{{ m() }}{% set z = 3 %}{{ m() }}", "{}", "3");
    try expectRender("{% macro m() %}{{ q }}{% endmacro %}{% for q in [7] %}{{ m() }}{% endfor %}", "{}", "");
    try expectRender("{% macro m(a, b=2, c=none) %}{{ a }}{{ b }}{{ c }}{% endmacro %}{{ m() }}|{{ m(1, c=3) }}", "{}", "2None|123");
    try expectRender("{% generation %}{% set g = 1 %}{% endgeneration %}{{ g is defined }}", "{}", "False");
    try expectRender("{% set ns = namespace(a=1) %}{% for i in [1,2] %}{% set ns.a = ns.a + i %}{% endfor %}{{ ns.a }}", "{}", "4");
}

test "values print as Python prints them" {
    try expectRender("{{ none }}|{{ true }}|{{ [1, 'a', none] }}|{{ {'a': 1} }}|{{ 1.0 }}|{{ 3/2 }}|{{ 7//2 }}|{{ -7 % 3 }}", "{}", "None|True|[1, 'a', None]|{'a': 1}|1.0|1.5|3|2");
    try expectRender("{{ 0.1 + 0.2 }}|{{ 1e20 }}|{{ 2.0 * 3 }}|{{ 10 / 2 }}|{{ 1.5e-7 }}|{{ 123456789012345678.0 }}", "{}", "0.30000000000000004|1e+20|6.0|5.0|1.5e-07|1.2345678901234568e+17");
    try expectRender("{{ [\"it's\", 'a\"b', 'x\\ny'] }}", "{}", "[\"it's\", 'a\"b', 'x\\ny']");
}

test "filters, tests and methods" {
    try expectRender("{{ '  a b \\n'|trim }}|{{ ' x '.strip() }}|{{ 'aXbXc'.split('X') }}|{{ 'a  b'.split() }}", "{}", "a b|x|['a', 'b', 'c']|['a', 'b']");
    try expectRender("{{ {'b': [1, 2.5, none, true, 'é<']}|tojson }}|{{ [1,{'a':2}]|tojson(indent=2) }}|{{ {'a': [1]}|tojson(separators=(',', ':')) }}", "{}", "{\"b\": [1, 2.5, null, true, \"é<\"]}|[\n  1,\n  {\n    \"a\": 2\n  }\n]|{\"a\":[1]}");
    try expectRender("{{ [1,2,3,4][::-1] }}{{ 'abc'[1:] }}{{ [1,2,3][-1] }}{{ [1,2,3][5] is defined }}{{ 'abcdef'[1:5:2] }}", "{}", "[4, 3, 2, 1]bc3Falsebd");
    try expectRender("{{ ['b','a','B','a']|unique|list }}{{ ['b','a','B']|sort }}{{ [{'id':'b'},{'id':'a'}]|sort(attribute='id') }}", "{}", "['b', 'a']['a', 'b', 'B'][{'id': 'a'}, {'id': 'b'}]");
    try expectRender("{{ 'hELLO world'|capitalize }}|{{ '%s-%d'|format('a', 3) }}|{{ 'a' ~ 1 ~ none ~ nope }}", "{}", "Hello world|a-3|a1None");
    try expectRender("{% for x in ['a','b'] %}{{ loop.index }}{{ loop.first }}{{ loop.last }}{{ loop.previtem }}{{ loop.nextitem }};{% endfor %}", "{}", "1TrueFalseb;2FalseTruea;");
    try expectRender("{% for x in [] %}a{% else %}empty{% endfor %}{% for x in [1,2,3] if x > 1 %}{{ loop.index }}{{ x }}{% endfor %}", "{}", "empty1223");
    try expectRender("{{ d.items is defined }} {{ d['items'] }} {{ d.get('k', 2) }} {{ d.get('zz') }}", "{\"d\": {\"items\": 9, \"k\": 1}}", "True 9 1 None");
}

test "refusals: unsupported features and raise_exception" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var diag: Diag = .{};
    try std.testing.expectError(error.Unsupported, Template.parse(arena.allocator(), "{% include 'x' %}", &diag));
    try std.testing.expectError(error.Unsupported, Template.parse(arena.allocator(), "{{ x|wordwrap }}", &diag));
    try std.testing.expectEqualStrings("the template uses the filter 'wordwrap', which is not supported", diag.msg);
    const t = try Template.parse(arena.allocator(), "{{ raise_exception('no ' ~ 1) }}", &diag);
    var out: std.ArrayList(u8) = .empty;
    var vars: Dict = .empty;
    try std.testing.expectError(error.Raised, t.render(arena.allocator(), &vars, &diag, &out));
    try std.testing.expectEqualStrings("no 1", diag.msg);
}
