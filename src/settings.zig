//! The settings file behind `--settings` (TASK-44): what the tray launcher,
//! `tfswarm setup` and the dashboard's Settings page share. The secrets, the
//! approved nodes and the logs live in the same folder as the file.
const std = @import("std");
const plat = @import("plat.zig");
const is_windows = @import("builtin").os.tag == .windows;

/// The exit code with which the dashboard asks the tray launcher to start it again.
pub const restart_code: u8 = 75;

pub const Dashboard = struct {
    models: []const u8 = "",
    /// Whether other machines may open the page; it binds to 127.0.0.1 otherwise.
    open_to_network: bool = false,
    port: u16 = 8080,
    node_port: u16 = 47310,
    router_port: u16 = 18080,
    expert_port: u16 = 47401,
    /// Whether `/v1` needs the API key in `api.key`.
    api_key: bool = false,
};

pub const Node = struct {
    /// The dashboard's IP[:PORT].
    dashboard: []const u8 = "",
    models: []const u8 = "",
    /// This node's name; empty for the machine's.
    name: []const u8 = "",
    /// The address the router dials; empty for the one the dashboard sees.
    advertise: []const u8 = "",
};

pub const Settings = struct {
    /// "dashboard" or "node"; empty until setup has run.
    role: []const u8 = "",
    dashboard: Dashboard = .{},
    node: Node = .{},
};

/// Reads `path`; null if it is missing or unreadable as settings.
pub fn load(a: std.mem.Allocator, path: []const u8) ?Settings
{
    const text = @import("proc.zig").readSmall(a, path) orelse return null;
    return std.json.parseFromSliceLeaky(Settings, a, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch null;
}

/// Writes `s` to `path` through a temporary file, so a reader never sees half of it.
pub fn save(a: std.mem.Allocator, path: []const u8, s: Settings) !void
{
    const text = try std.json.Stringify.valueAlloc(a, s, .{ .whitespace = .indent_2 });
    const tmp = try std.fmt.allocPrintSentinel(a, "{s}.tmp", .{path}, 0);
    const dst = try a.dupeSentinel(u8, path, 0);
    if (!plat.replaceFile(tmp, dst, text))
    {
        return error.WriteFailed;
    }
}

/// `name` in the folder that holds the settings file.
pub fn beside(a: std.mem.Allocator, settings_path: []const u8, name: []const u8) ![]const u8
{
    const dir = std.fs.path.dirname(settings_path) orelse ".";
    return std.fmt.allocPrint(a, "{s}{c}{s}", .{ dir, if (is_windows) @as(u8, '\\') else '/', name });
}

/// The default settings file: `%LOCALAPPDATA%\TFSwarm\settings.json` on Windows,
/// `~/.config/tfswarm/settings.json` elsewhere, or in `$TFSWARM_HOME`. The folder is made if missing.
pub fn defaultPath(a: std.mem.Allocator) ![]const u8
{
    const base = if (is_windows) plat.env("LOCALAPPDATA") else null;
    const dir = if (plat.env("TFSWARM_HOME")) |h|
        try a.dupe(u8, h)
    else if (base) |b|
        try std.fmt.allocPrint(a, "{s}\\TFSwarm", .{b})
    else if (plat.env("XDG_CONFIG_HOME")) |x|
        try std.fmt.allocPrint(a, "{s}/tfswarm", .{x})
    else
        try std.fmt.allocPrint(a, "{s}/.config/tfswarm", .{plat.env("HOME") orelse return error.NoHome});
    if (!is_windows)
    {
        if (std.fs.path.dirname(dir)) |parent|
        {
            plat.makeDir(try a.dupeSentinel(u8, parent, 0));
        }
    }
    plat.makeDir(try a.dupeSentinel(u8, dir, 0));
    return std.fmt.allocPrint(a, "{s}{c}settings.json", .{ dir, if (is_windows) @as(u8, '\\') else '/' });
}

/// Splits IP[:PORT], with `default_port` when no port is given.
pub fn hostPort(text: []const u8, default_port: u16) ?struct { host: []const u8, port: u16 }
{
    const t = std.mem.trim(u8, text, " \t");
    if (std.mem.indexOfScalar(u8, t, ':')) |colon|
    {
        const port = std.fmt.parseInt(u16, t[colon + 1 ..], 10) catch return null;
        return .{ .host = t[0..colon], .port = port };
    }
    return .{ .host = t, .port = default_port };
}

test "settings round-trip through the file" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = if (is_windows) plat.env("TEMP") orelse "." else "/tmp";
    const path = try std.fmt.allocPrint(a, "{s}{c}tf-settings-{d}.json", .{ dir, if (is_windows) @as(u8, '\\') else '/', plat.pid() });
    defer plat.deleteFile(a.dupeSentinel(u8, path, 0) catch unreachable);
    try std.testing.expect(load(a, path) == null);
    try save(a, path, .{ .role = "node", .node = .{ .dashboard = "10.0.0.5:47310", .models = "E:\\models" } });
    const s = load(a, path).?;
    try std.testing.expectEqualStrings("node", s.role);
    try std.testing.expectEqualStrings("E:\\models", s.node.models);
    try std.testing.expectEqual(@as(u16, 8080), s.dashboard.port);
    const hp = hostPort(s.node.dashboard, 1).?;
    try std.testing.expectEqualStrings("10.0.0.5", hp.host);
    try std.testing.expectEqual(@as(u16, 47310), hp.port);
    try std.testing.expectEqual(@as(u16, 9), hostPort("1.2.3.4", 9).?.port);
    try std.testing.expect(hostPort("1.2.3.4:x", 9) == null);
}
