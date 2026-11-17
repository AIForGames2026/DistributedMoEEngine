//! Minimal HTTP/1.1 for `olmoe serve` (TASK-30): parses a request head and
//! writes response heads. One request per connection; the server closes the
//! connection after answering, so there is no keep-alive or pipelining, and
//! bodies must come with a Content-Length (no chunked uploads).
const std = @import("std");

pub const Head = struct {
    method: []const u8,
    /// Without the query string.
    path: []const u8,
    content_length: usize = 0,
    /// The client waits for "100 Continue" before sending its body (curl does for large ones).
    expect_continue: bool = false,
    /// Asks to switch protocols (WebSocket), with the client's key.
    websocket_key: ?[]const u8 = null,
    host: ?[]const u8 = null,
    /// Set by browsers on WebSocket and cross-site requests.
    origin: ?[]const u8 = null,
    /// For the operator dashboard (TASK-25).
    cookie: ?[]const u8 = null,
    authorization: ?[]const u8 = null,
    /// Bytes up to and including the blank line.
    len: usize,
};

pub const Error = error{ Incomplete, BadRequest, LengthRequired, TooLarge };

/// Parses the head at the start of `buf`; `error.Incomplete` until the blank line arrives.
pub fn parseHead(buf: []const u8, max_head: usize) Error!Head {
    const end = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse {
        return if (buf.len >= max_head) error.TooLarge else error.Incomplete;
    };
    if (end + 4 > max_head)
    {
        return error.TooLarge;
    }
    var lines = std.mem.splitSequence(u8, buf[0..end], "\r\n");
    const first = lines.next() orelse return error.BadRequest;
    var parts = std.mem.splitScalar(u8, first, ' ');
    const method = parts.next() orelse return error.BadRequest;
    const target = parts.next() orelse return error.BadRequest;
    const version = parts.next() orelse return error.BadRequest;
    if (parts.next() != null or method.len == 0 or target.len == 0 or target[0] != '/' or !std.mem.startsWith(u8, version, "HTTP/1."))
    {
        return error.BadRequest;
    }
    var head: Head = .{
        .method = method,
        .path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len],
        .len = end + 4,
    };
    var upgrade = false;
    var key: ?[]const u8 = null;
    while (lines.next()) |line|
    {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.BadRequest;
        const name = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-length"))
        {
            head.content_length = std.fmt.parseInt(usize, value, 10) catch return error.BadRequest;
        }
        else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding"))
        {
            return error.LengthRequired;
        }
        else if (std.ascii.eqlIgnoreCase(name, "expect"))
        {
            head.expect_continue = std.ascii.eqlIgnoreCase(value, "100-continue");
        }
        else if (std.ascii.eqlIgnoreCase(name, "upgrade"))
        {
            upgrade = std.ascii.eqlIgnoreCase(value, "websocket");
        }
        else if (std.ascii.eqlIgnoreCase(name, "sec-websocket-key"))
        {
            key = value;
        }
        else if (std.ascii.eqlIgnoreCase(name, "host"))
        {
            head.host = value;
        }
        else if (std.ascii.eqlIgnoreCase(name, "origin"))
        {
            head.origin = value;
        }
        else if (std.ascii.eqlIgnoreCase(name, "cookie"))
        {
            head.cookie = value;
        }
        else if (std.ascii.eqlIgnoreCase(name, "authorization"))
        {
            head.authorization = value;
        }
    }
    if (upgrade)
    {
        head.websocket_key = key;
    }
    return head;
}

/// True unless a browser says the request comes from a page served elsewhere.
pub fn sameOrigin(head: Head) bool {
    const origin = head.origin orelse return true;
    const host = head.host orelse return false;
    const rest = if (std.mem.startsWith(u8, origin, "http://")) origin[7..] else if (std.mem.startsWith(u8, origin, "https://")) origin[8..] else return false;
    return std.ascii.eqlIgnoreCase(rest, host);
}

pub fn reason(status: u16) []const u8 {
    return switch (status) {
        101 => "Switching Protocols",
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        409 => "Conflict",
        413 => "Content Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        else => "Error",
    };
}

/// Appends a response head. A null `length` means the body runs until the
/// connection closes (server-sent events).
pub fn writeHead(gpa: std.mem.Allocator, out: *std.ArrayList(u8), status: u16, content_type: []const u8, length: ?usize) !void {
    var buf: [256]u8 = undefined;
    try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\n", .{ status, reason(status), content_type }));
    if (length) |n|
    {
        try out.appendSlice(gpa, try std.fmt.bufPrint(&buf, "Content-Length: {d}\r\n", .{n}));
    }
    else
    {
        try out.appendSlice(gpa, "Cache-Control: no-cache\r\n");
    }
    try out.appendSlice(gpa, "Connection: close\r\n\r\n");
}

test "parses a request head" {
    const raw = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: a\r\ncontent-length: 12\r\nExpect: 100-continue\r\n\r\n{\"a\": 1}";
    const h = try parseHead(raw, 4096);
    try std.testing.expectEqualStrings("POST", h.method);
    try std.testing.expectEqualStrings("/v1/chat/completions", h.path);
    try std.testing.expectEqual(@as(usize, 12), h.content_length);
    try std.testing.expect(h.expect_continue and h.websocket_key == null);
    try std.testing.expectEqualStrings("{\"a\": 1}", raw[h.len..]);
}

test "rejects bad and unfinished heads" {
    try std.testing.expectError(error.Incomplete, parseHead("GET / HTTP/1.1\r\nHost: a\r\n", 4096));
    try std.testing.expectError(error.BadRequest, parseHead("GET nothing HTTP/1.1\r\n\r\n", 4096));
    try std.testing.expectError(error.LengthRequired, parseHead("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 4096));
    try std.testing.expectError(error.TooLarge, parseHead("GET / HTTP/1.1\r\nX: aaaaaaaaaaaaaaaa", 16));
    const ws = try parseHead("GET /ws HTTP/1.1\r\nHost: 127.0.0.1:8091\r\nOrigin: http://127.0.0.1:8091\r\nUpgrade: websocket\r\nSec-WebSocket-Key: abc==\r\n\r\n", 4096);
    try std.testing.expectEqualStrings("abc==", ws.websocket_key.?);
    try std.testing.expect(sameOrigin(ws));
    const other = try parseHead("GET /ws HTTP/1.1\r\nHost: 127.0.0.1:8091\r\nOrigin: https://example.com\r\n\r\n", 4096);
    try std.testing.expect(!sameOrigin(other));
}
