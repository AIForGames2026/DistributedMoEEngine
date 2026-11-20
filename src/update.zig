//! Updates for the Windows tray launcher (TASK-45): reads the newest release on
//! GitHub, says whether it is newer than this build, and downloads its
//! installer, checked against the SHA-256 digest GitHub publishes for it.
const std = @import("std");
const is_windows = @import("builtin").os.tag == .windows;

/// The newest release; `TFSWARM_UPDATE_URL` replaces it for tests.
pub const latest_url = "https://api.github.com/repos/AIForGames2026/DistributedMoEEngine/releases/latest";

pub const Release = struct {
    version: []const u8,
    /// The installer asset.
    name: []const u8,
    url: []const u8,
    size: u64,
    sha256: [32]u8,
};

const Version = struct {
    nums: [3]u32,
    pre: bool,
};

/// Reads "v1.2.3", "1.2.3-beta" or git's "1.2.3-beta-4-gabc123" (a build after that tag).
fn parseVersion(text: []const u8) ?Version
{
    var t = std.mem.trimStart(u8, std.mem.trim(u8, text, " \t\r\n"), "v");
    if (std.mem.endsWith(u8, t, "-dirty"))
    {
        t = t[0 .. t.len - "-dirty".len];
    }
    // Drop git's "-N-gHASH", which names a build after the tag.
    if (std.mem.lastIndexOf(u8, t, "-g")) |g|
    {
        if (std.mem.lastIndexOfScalar(u8, t[0..g], '-')) |dash|
        {
            const count_ok = std.fmt.parseInt(u32, t[dash + 1 .. g], 10) catch null;
            const hash_ok = t.len > g + 2 and for (t[g + 2 ..]) |ch|
            {
                if (!std.ascii.isHex(ch))
                {
                    break false;
                }
            } else true;
            if (count_ok != null and hash_ok)
            {
                t = t[0..dash];
            }
        }
    }
    const core_end = std.mem.indexOfScalar(u8, t, '-') orelse t.len;
    var v: Version = .{ .nums = .{ 0, 0, 0 }, .pre = core_end < t.len };
    var it = std.mem.splitScalar(u8, t[0..core_end], '.');
    var i: usize = 0;
    while (it.next()) |part|
    {
        if (i >= 3)
        {
            return null;
        }
        v.nums[i] = std.fmt.parseInt(u32, part, 10) catch return null;
        i += 1;
    }
    if (i == 0)
    {
        return null;
    }
    return v;
}

/// Whether `other` is a later release than `have`; a release beats a pre-release of the same number.
pub fn newer(have: []const u8, other: []const u8) bool
{
    const h = parseVersion(have) orelse return parseVersion(other) != null;
    const o = parseVersion(other) orelse return false;
    for (h.nums, o.nums) |x, y|
    {
        if (x != y)
        {
            return y > x;
        }
    }
    return h.pre and !o.pre;
}

const Json = struct {
    tag_name: []const u8 = "",
    draft: bool = false,
    assets: []const struct {
        name: []const u8 = "",
        browser_download_url: []const u8 = "",
        size: u64 = 0,
        digest: ?[]const u8 = null,
    } = &.{},
};

/// The release's installer, `TFSwarm-Setup-*.exe`; null without one, or without its digest.
pub fn parseRelease(a: std.mem.Allocator, body: []const u8) ?Release
{
    const r = std.json.parseFromSliceLeaky(Json, a, body, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return null;
    if (r.draft or r.tag_name.len == 0)
    {
        return null;
    }
    for (r.assets) |asset|
    {
        if (!std.mem.startsWith(u8, asset.name, "TFSwarm-Setup-") or !std.mem.endsWith(u8, asset.name, ".exe"))
        {
            continue;
        }
        const digest = asset.digest orelse return null;
        if (!std.mem.startsWith(u8, digest, "sha256:") or digest.len != 7 + 64)
        {
            return null;
        }
        var sum: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&sum, digest[7..]) catch return null;
        return .{
            .version = std.mem.trimStart(u8, r.tag_name, "v"),
            .name = asset.name,
            .url = asset.browser_download_url,
            .size = asset.size,
            .sha256 = sum,
        };
    }
    return null;
}

/// Whether `bytes` are the installer the release describes.
pub fn matches(r: Release, bytes: []const u8) bool
{
    if (bytes.len != r.size)
    {
        return false;
    }
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sum, .{});
    return std.mem.eql(u8, &sum, &r.sha256);
}

const W = std.unicode.utf8ToUtf16LeStringLiteral;
const HINTERNET = *anyopaque;

const URL_COMPONENTS = extern struct {
    size: u32 = @sizeOf(URL_COMPONENTS),
    scheme: ?[*]u16 = null,
    scheme_len: u32 = 0,
    scheme_id: c_int = 0,
    host: ?[*]u16 = null,
    host_len: u32 = std.math.maxInt(u32),
    port: u16 = 0,
    user: ?[*]u16 = null,
    user_len: u32 = 0,
    password: ?[*]u16 = null,
    password_len: u32 = 0,
    path: ?[*]u16 = null,
    path_len: u32 = std.math.maxInt(u32),
    extra: ?[*]u16 = null,
    extra_len: u32 = std.math.maxInt(u32),
};

const win = if (is_windows) struct {
    extern "winhttp" fn WinHttpOpen(agent: [*:0]const u16, access: u32, proxy: ?[*:0]const u16, bypass: ?[*:0]const u16, flags: u32) callconv(.winapi) ?HINTERNET;
    extern "winhttp" fn WinHttpCrackUrl(url: [*]const u16, len: u32, flags: u32, parts: *URL_COMPONENTS) callconv(.winapi) c_int;
    extern "winhttp" fn WinHttpConnect(session: HINTERNET, host: [*:0]const u16, port: u16, reserved: u32) callconv(.winapi) ?HINTERNET;
    extern "winhttp" fn WinHttpOpenRequest(conn: HINTERNET, verb: [*:0]const u16, object: [*:0]const u16, version: ?[*:0]const u16, referrer: ?[*:0]const u16, accept: ?*const anyopaque, flags: u32) callconv(.winapi) ?HINTERNET;
    extern "winhttp" fn WinHttpSendRequest(req: HINTERNET, headers: ?[*]const u16, headers_len: u32, optional: ?*anyopaque, optional_len: u32, total: u32, context: usize) callconv(.winapi) c_int;
    extern "winhttp" fn WinHttpReceiveResponse(req: HINTERNET, reserved: ?*anyopaque) callconv(.winapi) c_int;
    extern "winhttp" fn WinHttpQueryHeaders(req: HINTERNET, info: u32, name: ?[*:0]const u16, buf: *anyopaque, len: *u32, index: ?*u32) callconv(.winapi) c_int;
    extern "winhttp" fn WinHttpReadData(req: HINTERNET, buf: [*]u8, to_read: u32, read: *u32) callconv(.winapi) c_int;
    extern "winhttp" fn WinHttpCloseHandle(h: HINTERNET) callconv(.winapi) c_int;
} else struct {};

const WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY = 4;
const WINHTTP_FLAG_SECURE = 0x00800000;
const INTERNET_SCHEME_HTTPS = 2;
const WINHTTP_QUERY_STATUS_CODE = 19;
const WINHTTP_QUERY_FLAG_NUMBER = 0x20000000;

/// GETs `url` over HTTPS (or HTTP, for tests) through the system's proxy settings, following redirects.
pub fn get(a: std.mem.Allocator, url: []const u8, agent: []const u8, max: usize) ![]u8
{
    if (!is_windows)
    {
        return error.Unsupported;
    }
    const wurl = try std.unicode.utf8ToUtf16LeAllocZ(a, url);
    var parts: URL_COMPONENTS = .{};
    if (win.WinHttpCrackUrl(wurl.ptr, @intCast(wurl.len), 0, &parts) == 0)
    {
        return error.BadUrl;
    }
    const host = try a.dupeSentinel(u16, parts.host.?[0..parts.host_len], 0);
    const object = try a.dupeSentinel(u16, parts.path.?[0 .. parts.path_len + parts.extra_len], 0);
    const session = win.WinHttpOpen(try std.unicode.utf8ToUtf16LeAllocZ(a, agent), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, null, null, 0) orelse return error.NoSession;
    defer _ = win.WinHttpCloseHandle(session);
    const conn = win.WinHttpConnect(session, host, parts.port, 0) orelse return error.ConnectFailed;
    defer _ = win.WinHttpCloseHandle(conn);
    const secure: u32 = if (parts.scheme_id == INTERNET_SCHEME_HTTPS) WINHTTP_FLAG_SECURE else 0;
    const req = win.WinHttpOpenRequest(conn, W("GET"), object, null, null, null, secure) orelse return error.RequestFailed;
    defer _ = win.WinHttpCloseHandle(req);
    const headers = W("Accept: application/vnd.github+json\r\nX-GitHub-Api-Version: 2022-11-28\r\n");
    if (win.WinHttpSendRequest(req, headers, headers.len, null, 0, 0, 0) == 0 or win.WinHttpReceiveResponse(req, null) == 0)
    {
        return error.NoResponse;
    }
    var status: u32 = 0;
    var status_len: u32 = @sizeOf(u32);
    if (win.WinHttpQueryHeaders(req, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, null, &status, &status_len, null) == 0)
    {
        return error.NoResponse;
    }
    if (status == 404)
    {
        return error.NotFound;
    }
    if (status != 200)
    {
        return error.BadStatus;
    }
    var out: std.ArrayList(u8) = .empty;
    while (true)
    {
        var buf: [64 << 10]u8 = undefined;
        var n: u32 = 0;
        if (win.WinHttpReadData(req, &buf, buf.len, &n) == 0)
        {
            return error.ReadFailed;
        }
        if (n == 0)
        {
            break;
        }
        if (out.items.len + n > max)
        {
            return error.TooLarge;
        }
        try out.appendSlice(a, buf[0..n]);
    }
    return out.items;
}

test "versions compare as releases" {
    try std.testing.expect(newer("0.3.0-beta", "0.4.0"));
    try std.testing.expect(newer("0.3.0-beta-12-g3aa1a61", "v0.4.0"));
    try std.testing.expect(newer("0.4.0-beta", "0.4.0"));
    try std.testing.expect(newer("0.4.0", "0.4.1"));
    try std.testing.expect(newer("0.9.9", "1.0.0"));
    try std.testing.expect(!newer("0.4.0", "0.4.0"));
    try std.testing.expect(!newer("0.4.0-3-gabc123", "0.4.0"));
    try std.testing.expect(!newer("0.4.0-3-gabc123-dirty", "0.4.0"));
    try std.testing.expect(!newer("0.4.1", "0.4.0"));
    try std.testing.expect(!newer("0.4.0", "0.4.0-beta"));
    try std.testing.expect(!newer("0.4.0", "nonsense"));
    try std.testing.expect(newer("unknown", "0.4.0"));
}

test "a release's installer and digest" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("abc", &sum, .{});
    const body = try std.fmt.allocPrint(a,
        \\{{"tag_name":"v0.4.0","draft":false,"assets":[
        \\{{"name":"notes.txt","browser_download_url":"https://x/notes.txt","size":3,"digest":"sha256:00"}},
        \\{{"name":"TFSwarm-Setup-0.4.0.exe","browser_download_url":"https://x/s.exe","size":3,"digest":"sha256:{s}"}}]}}
    , .{&std.fmt.bytesToHex(sum, .lower)});
    const r = parseRelease(a, body).?;
    try std.testing.expectEqualStrings("0.4.0", r.version);
    try std.testing.expectEqualStrings("https://x/s.exe", r.url);
    try std.testing.expect(matches(r, "abc"));
    try std.testing.expect(!matches(r, "abd"));
    try std.testing.expect(parseRelease(a, "{\"tag_name\":\"v1\",\"draft\":true,\"assets\":[]}") == null);
    try std.testing.expect(parseRelease(a, "{\"tag_name\":\"v1\",\"assets\":[{\"name\":\"TFSwarm-Setup-1.exe\",\"size\":1}]}") == null);
    try std.testing.expect(parseRelease(a, "not json") == null);
}
