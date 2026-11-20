//! The Windows tray launcher, `tfswarm-operator.exe` (TASK-44). It has no
//! console window. It runs first-run setup (`tfswarm setup`), then the dashboard
//! or the node agent from the settings file, each as a child of `tfswarm.exe`
//! beside it in a job object, so quitting ends them and every engine process
//! under them. Its icon in the notification area opens the dashboard signed in,
//! copies the admin token, and offers settings, logs, restart and quit. It
//! also looks for a newer release and installs it when asked (TASK-45).
const std = @import("std");
const build_options = @import("build_options");
const update = @import("update.zig");
const win32 = @import("win32.zig");
const plat = @import("plat.zig");
const proc = @import("proc.zig");
const settings_file = @import("settings.zig");
const tcp = @import("transport/tcp.zig");
const Child = @import("win_child.zig").Child;

const W = std.unicode.utf8ToUtf16LeStringLiteral;
const HWND = *anyopaque;
const HANDLE = win32.HANDLE;
const BOOL = win32.BOOL;
const WPARAM = usize;
const LPARAM = isize;
const LRESULT = isize;

const WNDCLASSEXW = extern struct {
    size: u32 = @sizeOf(WNDCLASSEXW),
    style: u32 = 0,
    proc: *const fn (HWND, u32, WPARAM, LPARAM) callconv(.winapi) LRESULT,
    cls_extra: c_int = 0,
    wnd_extra: c_int = 0,
    instance: ?HANDLE,
    icon: ?HANDLE = null,
    cursor: ?HANDLE = null,
    background: ?HANDLE = null,
    menu_name: ?[*:0]const u16 = null,
    class_name: [*:0]const u16,
    icon_small: ?HANDLE = null,
};

const MSG = extern struct {
    hwnd: ?HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    time: u32,
    x: i32,
    y: i32,
    private: u32,
};

const POINT = extern struct {
    x: i32 = 0,
    y: i32 = 0,
};

const NOTIFYICONDATAW = extern struct {
    size: u32 = @sizeOf(NOTIFYICONDATAW),
    hwnd: HWND,
    id: u32 = 1,
    flags: u32 = 0,
    callback: u32 = 0,
    icon: ?HANDLE = null,
    tip: [128]u16 = @splat(0),
    state: u32 = 0,
    state_mask: u32 = 0,
    info: [256]u16 = @splat(0),
    version: u32 = 0,
    info_title: [64]u16 = @splat(0),
    info_flags: u32 = 0,
    guid: [16]u8 = @splat(0),
    balloon_icon: ?HANDLE = null,
};

comptime
{
    std.debug.assert(@sizeOf(NOTIFYICONDATAW) == 976);
}

extern "user32" fn RegisterClassExW(class: *const WNDCLASSEXW) callconv(.winapi) u16;
extern "user32" fn CreateWindowExW(ex: u32, class: [*:0]const u16, name: [*:0]const u16, style: u32, x: i32, y: i32, w: i32, h: i32, parent: ?HWND, menu: ?HANDLE, instance: ?HANDLE, param: ?*anyopaque) callconv(.winapi) ?HWND;
extern "user32" fn DefWindowProcW(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) callconv(.winapi) LRESULT;
extern "user32" fn GetMessageW(msg: *MSG, hwnd: ?HWND, min: u32, max: u32) callconv(.winapi) BOOL;
extern "user32" fn TranslateMessage(msg: *const MSG) callconv(.winapi) BOOL;
extern "user32" fn DispatchMessageW(msg: *const MSG) callconv(.winapi) LRESULT;
extern "user32" fn PostQuitMessage(code: c_int) callconv(.winapi) void;
extern "user32" fn PostMessageW(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) callconv(.winapi) BOOL;
extern "user32" fn FindWindowW(class: [*:0]const u16, name: ?[*:0]const u16) callconv(.winapi) ?HWND;
extern "user32" fn RegisterWindowMessageW(name: [*:0]const u16) callconv(.winapi) u32;
extern "user32" fn SetTimer(hwnd: HWND, id: usize, ms: u32, func: ?*anyopaque) callconv(.winapi) usize;
extern "user32" fn CreatePopupMenu() callconv(.winapi) ?HANDLE;
extern "user32" fn AppendMenuW(menu: HANDLE, flags: u32, id: usize, text: ?[*:0]const u16) callconv(.winapi) BOOL;
extern "user32" fn SetMenuDefaultItem(menu: HANDLE, item: u32, by_pos: u32) callconv(.winapi) BOOL;
extern "user32" fn TrackPopupMenu(menu: HANDLE, flags: u32, x: i32, y: i32, reserved: i32, hwnd: HWND, rect: ?*const anyopaque) callconv(.winapi) BOOL;
extern "user32" fn DestroyMenu(menu: HANDLE) callconv(.winapi) BOOL;
extern "user32" fn SetForegroundWindow(hwnd: HWND) callconv(.winapi) BOOL;
extern "user32" fn GetCursorPos(point: *POINT) callconv(.winapi) BOOL;
extern "user32" fn LoadImageW(instance: ?HANDLE, name: usize, kind: u32, cx: c_int, cy: c_int, flags: u32) callconv(.winapi) ?HANDLE;
extern "user32" fn GetSystemMetrics(index: c_int) callconv(.winapi) c_int;
extern "user32" fn MessageBoxW(hwnd: ?HWND, text: [*:0]const u16, caption: [*:0]const u16, kind: u32) callconv(.winapi) c_int;
extern "user32" fn OpenClipboard(hwnd: ?HWND) callconv(.winapi) BOOL;
extern "user32" fn EmptyClipboard() callconv(.winapi) BOOL;
extern "user32" fn SetClipboardData(format: u32, data: ?*anyopaque) callconv(.winapi) ?*anyopaque;
extern "user32" fn CloseClipboard() callconv(.winapi) BOOL;
extern "shell32" fn Shell_NotifyIconW(message: u32, data: *NOTIFYICONDATAW) callconv(.winapi) BOOL;
extern "shell32" fn ShellExecuteW(hwnd: ?HWND, op: ?[*:0]const u16, file: [*:0]const u16, params: ?[*:0]const u16, dir: ?[*:0]const u16, show: c_int) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetModuleHandleW(name: ?[*:0]const u16) callconv(.winapi) ?HANDLE;
extern "kernel32" fn CreateMutexW(attrs: ?*anyopaque, owner: BOOL, name: [*:0]const u16) callconv(.winapi) ?HANDLE;
extern "kernel32" fn GetEnvironmentVariableW(name: [*:0]const u16, buf: [*]u16, size: u32) callconv(.winapi) u32;
extern "kernel32" fn GlobalAlloc(flags: u32, bytes: usize) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GlobalLock(mem: *anyopaque) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GlobalUnlock(mem: *anyopaque) callconv(.winapi) BOOL;
extern "kernel32" fn GlobalFree(mem: *anyopaque) callconv(.winapi) ?*anyopaque;

const WM_NULL = 0x0000;
const WM_DESTROY = 0x0002;
const WM_TIMER = 0x0113;
const WM_COMMAND = 0x0111;
const WM_LBUTTONUP = 0x0202;
const WM_RBUTTONUP = 0x0205;
const WM_CONTEXTMENU = 0x007B;
const WM_APP = 0x8000;
/// Clicks on the tray icon.
const WM_TRAY = WM_APP + 1;
/// Sent by a second copy of the launcher, which then exits.
const WM_OPEN = WM_APP + 2;
/// Posted by the update threads when they finish; WPARAM says which.
const WM_UPDATE = WM_APP + 3;
const NIM_ADD = 0;
const NIM_MODIFY = 1;
const NIM_DELETE = 2;
const NIF_MESSAGE = 0x1;
const NIF_ICON = 0x2;
const NIF_TIP = 0x4;
const NIF_INFO = 0x10;
const NIIF_INFO = 0x1;
const NIIF_ERROR = 0x3;
const MF_STRING = 0x0;
const MF_GRAYED = 0x1;
const MF_CHECKED = 0x8;
const MF_SEPARATOR = 0x800;
const TPM_RIGHTBUTTON = 0x2;
const TPM_RETURNCMD = 0x100;
const IMAGE_ICON = 1;
const SM_CXSMICON = 49;
const SM_CYSMICON = 50;
const MB_ICONERROR = 0x10;
const MB_ICONQUESTION = 0x20;
const MB_YESNO = 0x4;
const IDYES = 6;
const CF_UNICODETEXT = 13;
const GMEM_MOVEABLE = 0x2;
const ERROR_ALREADY_EXISTS = 183;
const SW_SHOWNORMAL = 1;

const class_name = W("TFSwarmOperatorTray");

const Cmd = enum(usize) {
    open = 1,
    copy_token,
    settings,
    logs,
    restart,
    setup,
    quit,
    start,
    open_setup,
    check_updates,
    install_update,
    auto_updates,
};

const update_timer = 2;
const update_check: usize = 1;
const update_download: usize = 2;
const user_agent = "TFSwarm-Operator/" ++ build_options.version;

const Mode = enum { setup, dashboard, node, stopped };

const App = struct {
    gpa: std.mem.Allocator,
    hwnd: HWND = undefined,
    icon: ?HANDLE = null,
    folder: []const u8 = "",
    settings_path: []const u8 = "",
    engine: []const u8 = "",
    mode: Mode = .stopped,
    child: ?Child = null,
    /// Open the dashboard in the browser once it answers.
    open_pending: bool = false,
    open_since_ns: u64 = 0,
    setup_url: ?[]u8 = null,
    port: u16 = 8080,
    status_buf: [160]u8 = undefined,
    status: []const u8 = "Starting",
    taskbar_created: u32 = 0,
    /// One update check or download at a time; the threads own the fields below while it is set.
    update_busy: std.atomic.Value(bool) = .init(false),
    update_quiet: bool = true,
    update_arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    update_found: ?update.Release = null,
    update_err: ?anyerror = null,
    update_file: []const u8 = "",
    /// What the icon said, for the logs folder; started afresh by each launch.
    log: ?plat.OutFile = null,
};

var app: App = .{ .gpa = std.heap.page_allocator };

pub fn main() void
{
    run() catch |err| {
        var buf: [256]u8 = undefined;
        const text = std.fmt.bufPrint(&buf, "TFSwarm could not start: {t}", .{err}) catch "TFSwarm could not start.";
        fail(text);
    };
}

fn run() !void
{
    const a = app.gpa;
    try findFolder(a);
    // One copy per settings folder; a second double-click opens the dashboard of the first.
    const title = try std.unicode.utf8ToUtf16LeAllocZ(a, try std.fmt.allocPrint(a, "TFSwarm {s}", .{app.folder}));
    const mutex_name = try std.unicode.utf8ToUtf16LeAllocZ(a, try std.fmt.allocPrint(a, "Local\\TFSwarm-{x:0>16}", .{std.hash.Wyhash.hash(0, app.folder)}));
    _ = CreateMutexW(null, 0, mutex_name);
    if (win32.GetLastError() == ERROR_ALREADY_EXISTS)
    {
        if (FindWindowW(class_name, title)) |other|
        {
            _ = PostMessageW(other, WM_OPEN, 0, 0);
        }
        return;
    }
    var exe_buf: [4096]u8 = undefined;
    const self = plat.selfExe("/proc/self/exe", &exe_buf) orelse return error.NoSelfPath;
    app.engine = try std.fmt.allocPrint(a, "{s}\\tfswarm.exe", .{std.fs.path.dirname(self) orelse "."});
    if (plat.regularFileBytes(app.engine) == null)
    {
        const text = try std.fmt.allocPrint(a, "tfswarm.exe is missing. Keep tfswarm-operator.exe in the same folder as tfswarm.exe ({s}).", .{std.fs.path.dirname(self) orelse "."});
        return fail(text);
    }
    app.settings_path = try std.fmt.allocPrint(a, "{s}\\settings.json", .{app.folder});
    // The log keeps the end of the last launch's, which tells how an update went.
    const log_path = try a.dupeSentinel(u8, beside("launcher.log"), 0);
    const earlier = proc.tail(a, log_path, 8 << 10) catch null;
    app.log = plat.OutFile.create(log_path);
    if (earlier) |e|
    {
        if (app.log) |f|
        {
            _ = f.write(e);
        }
    }
    logLine("TFSwarm Operator {s}, settings in {s}", .{ build_options.version, app.folder });
    // An update's installer is not needed once it has run; it may still be open, then the next start removes it.
    plat.deleteFile(try a.dupeSentinel(u8, beside("TFSwarm-Setup.exe"), 0));

    const instance = GetModuleHandleW(null);
    const class: WNDCLASSEXW = .{ .proc = windowProc, .instance = instance, .class_name = class_name };
    if (RegisterClassExW(&class) == 0)
    {
        return error.NoWindowClass;
    }
    // A hidden window that receives the tray icon's clicks and the timer.
    app.hwnd = CreateWindowExW(0, class_name, title, 0, 0, 0, 0, 0, null, null, instance, null) orelse return error.NoWindow;
    app.icon = LoadImageW(instance, 1, IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    app.taskbar_created = RegisterWindowMessageW(W("TaskbarCreated"));
    addIcon();
    startFromSettings();
    _ = SetTimer(app.hwnd, 1, 500, null);
    // The first update check waits a minute, so it does not slow the start.
    _ = SetTimer(app.hwnd, update_timer, 60 * 1000, null);
    var msg: MSG = undefined;
    while (GetMessageW(&msg, null, 0, 0) > 0)
    {
        _ = TranslateMessage(&msg);
        _ = DispatchMessageW(&msg);
    }
}

/// The settings folder: `%LOCALAPPDATA%\TFSwarm`, or `TFSWARM_HOME` for a second setup or a test.
fn findFolder(a: std.mem.Allocator) !void
{
    var env_buf: [1024]u16 = undefined;
    const home = GetEnvironmentVariableW(W("TFSWARM_HOME"), &env_buf, env_buf.len);
    if (home > 0 and home < env_buf.len)
    {
        app.folder = try std.unicode.utf16LeToUtf8Alloc(a, env_buf[0..home]);
    }
    else
    {
        const n = GetEnvironmentVariableW(W("LOCALAPPDATA"), &env_buf, env_buf.len);
        if (n == 0 or n >= env_buf.len)
        {
            return error.NoLocalAppData;
        }
        const local = try std.unicode.utf16LeToUtf8Alloc(a, env_buf[0..n]);
        app.folder = try std.fmt.allocPrint(a, "{s}\\TFSwarm", .{local});
    }
    plat.makeDir(try a.dupeSentinel(u8, app.folder, 0));
}

fn fail(text: []const u8) void
{
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const w = std.unicode.utf8ToUtf16LeAllocZ(arena.allocator(), text) catch return;
    _ = MessageBoxW(null, w, W("TFSwarm"), MB_ICONERROR);
}

fn windowProc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) callconv(.winapi) LRESULT
{
    switch (msg)
    {
        WM_TIMER =>
        {
            if (wparam == update_timer)
            {
                autoCheck();
            }
            else
            {
                tick();
            }
            return 0;
        },
        WM_UPDATE =>
        {
            onUpdate(wparam);
            return 0;
        },
        WM_TRAY =>
        {
            const event: u32 = @truncate(@as(usize, @bitCast(lparam)));
            if (event == WM_LBUTTONUP)
            {
                primary();
            }
            else if (event == WM_RBUTTONUP or event == WM_CONTEXTMENU)
            {
                showMenu();
            }
            return 0;
        },
        WM_OPEN =>
        {
            primary();
            return 0;
        },
        WM_COMMAND =>
        {
            // The menu's commands, also accepted when posted, as scripts and tests do.
            const id = wparam & 0xFFFF;
            if (id >= @intFromEnum(Cmd.open) and id <= @intFromEnum(Cmd.auto_updates))
            {
                runCommand(@enumFromInt(id));
            }
            return 0;
        },
        WM_DESTROY =>
        {
            PostQuitMessage(0);
            return 0;
        },
        else => {},
    }
    if (msg == app.taskbar_created and msg != 0)
    {
        // Explorer restarted and lost the icon.
        addIcon();
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

fn iconData() NOTIFYICONDATAW
{
    return .{ .hwnd = app.hwnd };
}

fn addIcon() void
{
    var d = iconData();
    d.flags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    d.callback = WM_TRAY;
    d.icon = app.icon;
    setWide(&d.tip, "TFSwarm");
    _ = Shell_NotifyIconW(NIM_ADD, &d);
}

/// Copies UTF-8 `text` into a fixed UTF-16 field, cut to fit.
fn setWide(dst: []u16, text: []const u8) void
{
    @memset(dst, 0);
    var view = std.unicode.Utf8View.init(text) catch return;
    var it = view.iterator();
    var i: usize = 0;
    while (it.nextCodepoint()) |cp|
    {
        if (i + 2 >= dst.len)
        {
            break;
        }
        if (cp < 0x10000)
        {
            dst[i] = @intCast(cp);
            i += 1;
        }
        else
        {
            const v = cp - 0x10000;
            dst[i] = @intCast(0xD800 + (v >> 10));
            dst[i + 1] = @intCast(0xDC00 + (v & 0x3FF));
            i += 2;
        }
    }
}

fn setStatus(comptime fmt: []const u8, args: anytype) void
{
    const text = std.fmt.bufPrint(&app.status_buf, fmt, args) catch return;
    if (std.mem.eql(u8, text, app.status))
    {
        return;
    }
    app.status = text;
    var d = iconData();
    d.flags = NIF_TIP;
    var tip: [160]u8 = undefined;
    setWide(&d.tip, std.fmt.bufPrint(&tip, "TFSwarm: {s}", .{text}) catch "TFSwarm");
    _ = Shell_NotifyIconW(NIM_MODIFY, &d);
}

/// One line in launcher.log, from the window's thread.
fn logLine(comptime fmt: []const u8, args: anytype) void
{
    const f = app.log orelse return;
    var buf: [600]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt ++ "\r\n", args) catch return;
    _ = f.write(line);
}

/// A notification from the tray icon.
fn notify(title: []const u8, text: []const u8, is_error: bool) void
{
    logLine("{s}: {s}", .{ title, text });
    var d = iconData();
    d.flags = NIF_INFO;
    d.info_flags = if (is_error) NIIF_ERROR else NIIF_INFO;
    setWide(&d.info_title, title);
    setWide(&d.info, text);
    _ = Shell_NotifyIconW(NIM_MODIFY, &d);
}

fn beside(name: []const u8) []const u8
{
    return std.fmt.allocPrint(app.gpa, "{s}\\{s}", .{ app.folder, name }) catch "";
}

fn stopChild() void
{
    if (app.child) |*c_|
    {
        c_.stop(0);
        c_.deinit(app.gpa);
        app.child = null;
    }
}

fn spawn(argv: []const []const u8, log_name: []const u8) bool
{
    stopChild();
    app.child = Child.spawn(app.gpa, argv, beside(log_name)) catch |err| {
        var buf: [200]u8 = undefined;
        notify("TFSwarm could not start tfswarm.exe", std.fmt.bufPrint(&buf, "{t}", .{err}) catch "", true);
        app.mode = .stopped;
        setStatus("stopped", .{});
        return false;
    };
    return true;
}

fn startSetup() void
{
    if (app.setup_url) |u|
    {
        app.gpa.free(u);
        app.setup_url = null;
    }
    if (spawn(&.{ app.engine, "setup", "--settings", app.settings_path }, "setup.log"))
    {
        app.mode = .setup;
        setStatus("setup", .{});
    }
}

/// Starts what the settings file says this PC does, or setup if it says nothing yet.
fn startFromSettings() void
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const s = settings_file.load(arena.allocator(), app.settings_path) orelse return startSetup();
    if (std.mem.eql(u8, s.role, "dashboard"))
    {
        app.port = s.dashboard.port;
        if (spawn(&.{ app.engine, "dashboard", "--settings", app.settings_path }, "dashboard.log"))
        {
            app.mode = .dashboard;
            app.open_pending = true;
            app.open_since_ns = plat.monoNs();
            setStatus("dashboard starting", .{});
        }
        return;
    }
    if (std.mem.eql(u8, s.role, "node"))
    {
        if (spawn(&.{ app.engine, "node", "--settings", app.settings_path }, "node.log"))
        {
            app.mode = .node;
            setStatus("node agent starting", .{});
        }
        return;
    }
    startSetup();
}

fn tick() void
{
    if (app.child) |*c_|
    {
        if (!c_.alive())
        {
            const code = c_.exit.?;
            c_.deinit(app.gpa);
            app.child = null;
            onExit(code);
            return;
        }
    }
    switch (app.mode)
    {
        .setup =>
        {
            if (app.setup_url == null)
            {
                app.setup_url = findInLog("setup.log", "setup: open ");
                if (app.setup_url) |u|
                {
                    openUrl(u);
                }
            }
        },
        .dashboard =>
        {
            if (app.open_pending)
            {
                if (openDashboard(false))
                {
                    app.open_pending = false;
                    setStatus("dashboard on port {d}", .{app.port});
                }
                else if (plat.monoNs() - app.open_since_ns > 30 * std.time.ns_per_s)
                {
                    app.open_pending = false;
                    notify("The dashboard did not answer", "It may still be starting. Open it from this icon in a moment, or see dashboard.log in the logs folder.", true);
                }
            }
        },
        .node => nodeStatus(),
        .stopped => {},
    }
}

fn onExit(code: u8) void
{
    var buf: [256]u8 = undefined;
    switch (app.mode)
    {
        .setup =>
        {
            if (code == 0)
            {
                return startFromSettings();
            }
            app.mode = .stopped;
            setStatus("setup stopped", .{});
            notify("Setup stopped", std.fmt.bufPrint(&buf, "It exited with {d}; see setup.log in the logs folder.", .{code}) catch "", true);
        },
        .dashboard =>
        {
            if (code == settings_file.restart_code)
            {
                return startFromSettings();
            }
            app.mode = .stopped;
            setStatus("dashboard stopped", .{});
            notify("The dashboard stopped", std.fmt.bufPrint(&buf, "It exited with {d}; see dashboard.log in the logs folder, or start it again from this icon.", .{code}) catch "", true);
        },
        .node =>
        {
            app.mode = .stopped;
            setStatus("node agent stopped", .{});
            notify("The node agent stopped", std.fmt.bufPrint(&buf, "It exited with {d}; see node.log in the logs folder, or run setup again from this icon.", .{code}) catch "", true);
        },
        .stopped => {},
    }
}

/// The rest of the first line in a log that starts with `prefix`.
fn findInLog(log_name: []const u8, prefix: []const u8) ?[]u8
{
    const text = proc.tail(app.gpa, beside(log_name), 16 << 10) catch return null;
    defer app.gpa.free(text);
    const at = std.mem.indexOf(u8, text, prefix) orelse return null;
    const rest = text[at + prefix.len ..];
    const end = std.mem.indexOfAny(u8, rest, "\r\n") orelse return null;
    return app.gpa.dupe(u8, rest[0..end]) catch null;
}

/// What the agent's log says last about its link.
fn nodeStatus() void
{
    const text = proc.tail(app.gpa, beside("node.log"), 8 << 10) catch return;
    defer app.gpa.free(text);
    const marks = [_]struct { needle: []const u8, status: []const u8 }{
        .{ .needle = "node: linked to the dashboard", .status = "node, linked to the dashboard" },
        .{ .needle = "node: link to the dashboard ended", .status = "node, not linked (retrying)" },
        .{ .needle = "could not prove it holds the node secret", .status = "node, the dashboard has another secret" },
        .{ .needle = "node: started rank", .status = "node, holding an expert rank" },
        .{ .needle = "node: stopping rank", .status = "node, linked to the dashboard" },
    };
    var best: ?usize = null;
    var status: []const u8 = "node agent starting";
    for (marks) |m|
    {
        if (std.mem.lastIndexOf(u8, text, m.needle)) |at|
        {
            if (best == null or at > best.?)
            {
                best = at;
                status = m.status;
            }
        }
    }
    setStatus("{s}", .{status});
}

fn openUrl(url: []const u8) void
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const w = std.unicode.utf8ToUtf16LeAllocZ(arena.allocator(), url) catch return;
    _ = ShellExecuteW(null, W("open"), w, null, null, SW_SHOWNORMAL);
}

/// Asks the dashboard for a one-time sign-in link and opens it; false if it does not answer yet.
fn openDashboard(settings_view: bool) bool
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = proc.readSmall(a, beside("dashboard.token")) orelse return false;
    const token = std.mem.trim(u8, raw, " \t\r\n");
    const fd = tcp.connect("127.0.0.1", app.port, 1) catch return false;
    defer tcp.closeSocket(fd);
    const req = std.fmt.allocPrint(a, "POST /api/login-link HTTP/1.1\r\nHost: 127.0.0.1:{d}\r\nAuthorization: Bearer {s}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", .{ app.port, token }) catch return false;
    if (win32.send(fd, req.ptr, @intCast(req.len), 0) != req.len)
    {
        return false;
    }
    var resp: std.ArrayList(u8) = .empty;
    while (resp.items.len < 8192)
    {
        var buf: [2048]u8 = undefined;
        const n = win32.recv(fd, &buf, buf.len, 0);
        if (n <= 0)
        {
            break;
        }
        resp.appendSlice(a, buf[0..@intCast(n)]) catch return false;
    }
    const key = "\"path\":\"";
    const at = std.mem.indexOf(u8, resp.items, key) orelse return false;
    const rest = resp.items[at + key.len ..];
    const end = std.mem.indexOfScalar(u8, rest, '"') orelse return false;
    const url = std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}{s}", .{ app.port, rest[0..end], if (settings_view) "/settings" else "" }) catch return false;
    openUrl(url);
    return true;
}

fn copyToken() void
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = proc.readSmall(a, beside("dashboard.token")) orelse {
        return notify("No admin token yet", "The dashboard makes it the first time it starts.", true);
    };
    const w = std.unicode.utf8ToUtf16LeAllocZ(a, std.mem.trim(u8, raw, " \t\r\n")) catch return;
    const bytes = (w.len + 1) * 2;
    const mem = GlobalAlloc(GMEM_MOVEABLE, bytes) orelse return;
    const p: [*]u8 = @ptrCast(GlobalLock(mem) orelse {
        _ = GlobalFree(mem);
        return;
    });
    @memcpy(p[0..bytes], std.mem.sliceAsBytes(w[0 .. w.len + 1]));
    _ = GlobalUnlock(mem);
    if (OpenClipboard(app.hwnd) == 0)
    {
        _ = GlobalFree(mem);
        return;
    }
    _ = EmptyClipboard();
    // The clipboard owns the memory once this succeeds.
    if (SetClipboardData(CF_UNICODETEXT, mem) == null)
    {
        _ = GlobalFree(mem);
    }
    _ = CloseClipboard();
    notify("Copied the admin token", "Paste it to sign in from another machine. Treat it like a password.", false);
}

/// A click on the icon: whatever this PC is doing, shown.
fn primary() void
{
    switch (app.mode)
    {
        .dashboard =>
        {
            if (!openDashboard(false))
            {
                app.open_pending = true;
                app.open_since_ns = plat.monoNs();
            }
        },
        .setup =>
        {
            if (app.setup_url) |u|
            {
                openUrl(u);
            }
        },
        .node => notify("TFSwarm node", app.status, false),
        .stopped => showMenu(),
    }
}

fn showMenu() void
{
    const menu = CreatePopupMenu() orelse return;
    defer _ = DestroyMenu(menu);
    var status: [200]u8 = undefined;
    var wide: [200]u16 = undefined;
    setWide(&wide, std.fmt.bufPrint(&status, "TFSwarm: {s}", .{app.status}) catch "TFSwarm");
    _ = AppendMenuW(menu, MF_STRING | MF_GRAYED, 0, @ptrCast(&wide));
    _ = AppendMenuW(menu, MF_SEPARATOR, 0, null);
    switch (app.mode)
    {
        .dashboard =>
        {
            item(menu, .open, "Open the dashboard");
            item(menu, .copy_token, "Copy the admin token");
            item(menu, .settings, "Settings");
            _ = AppendMenuW(menu, MF_SEPARATOR, 0, null);
            item(menu, .logs, "Open the logs folder");
            item(menu, .restart, "Restart the dashboard");
            item(menu, .setup, "Run setup again");
            _ = SetMenuDefaultItem(menu, @intFromEnum(Cmd.open), 0);
        },
        .node =>
        {
            item(menu, .setup, "Settings (run setup again)");
            item(menu, .logs, "Open the logs folder");
            item(menu, .restart, "Restart the node agent");
        },
        .setup =>
        {
            item(menu, .open_setup, "Open setup");
            _ = SetMenuDefaultItem(menu, @intFromEnum(Cmd.open_setup), 0);
        },
        .stopped =>
        {
            item(menu, .start, "Start again");
            item(menu, .logs, "Open the logs folder");
            item(menu, .setup, "Run setup again");
        },
    }
    _ = AppendMenuW(menu, MF_SEPARATOR, 0, null);
    var label: [80]u8 = undefined;
    if (app.update_found) |r|
    {
        if (!app.update_busy.load(.acquire))
        {
            item(menu, .install_update, std.fmt.bufPrint(&label, "Update to {s}", .{r.version}) catch "Update");
        }
    }
    item(menu, .check_updates, std.fmt.bufPrint(&label, "Check for updates (this is {s})", .{build_options.version}) catch "Check for updates");
    var wide_auto: [80]u16 = undefined;
    setWide(&wide_auto, "Check for updates automatically");
    _ = AppendMenuW(menu, MF_STRING | (if (autoUpdates()) @as(u32, MF_CHECKED) else 0), @intFromEnum(Cmd.auto_updates), @ptrCast(&wide_auto));
    _ = AppendMenuW(menu, MF_SEPARATOR, 0, null);
    item(menu, .quit, "Quit");
    var pt: POINT = .{};
    _ = GetCursorPos(&pt);
    // Without this the menu stays open when the user clicks elsewhere.
    _ = SetForegroundWindow(app.hwnd);
    const chosen = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_RIGHTBUTTON, pt.x, pt.y, 0, app.hwnd, null);
    _ = PostMessageW(app.hwnd, WM_NULL, 0, 0);
    if (chosen > 0)
    {
        runCommand(@enumFromInt(@as(usize, @intCast(chosen))));
    }
}

fn item(menu: HANDLE, cmd: Cmd, text: []const u8) void
{
    var wide: [80]u16 = undefined;
    setWide(&wide, text);
    _ = AppendMenuW(menu, MF_STRING, @intFromEnum(cmd), @ptrCast(&wide));
}

fn runCommand(cmd: Cmd) void
{
    switch (cmd)
    {
        .open => primary(),
        .open_setup => primary(),
        .copy_token => copyToken(),
        .settings =>
        {
            if (!openDashboard(true))
            {
                notify("The dashboard does not answer", "Start it again from this icon, or see dashboard.log in the logs folder.", true);
            }
        },
        .logs => openUrl(app.folder),
        .restart, .start => startFromSettings(),
        .setup => startSetup(),
        .check_updates => startCheck(false),
        .install_update => askToInstall(),
        .auto_updates => toggleAutoUpdates(),
        .quit =>
        {
            stopChild();
            var d = iconData();
            _ = Shell_NotifyIconW(NIM_DELETE, &d);
            PostQuitMessage(0);
        },
    }
}

/// Whether the settings file allows the daily check; it does until turned off.
fn autoUpdates() bool
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const s = settings_file.load(arena.allocator(), app.settings_path) orelse return true;
    return s.check_updates;
}

fn toggleAutoUpdates() void
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = settings_file.load(a, app.settings_path) orelse settings_file.Settings{};
    s.check_updates = !s.check_updates;
    settings_file.save(a, app.settings_path, s) catch {
        return notify("Could not save the settings", "See whether the settings folder is writable.", true);
    };
}

fn autoCheck() void
{
    _ = SetTimer(app.hwnd, update_timer, 24 * 60 * 60 * 1000, null);
    if (autoUpdates())
    {
        startCheck(true);
    }
}

/// The release feed: GitHub's newest release, or `TFSWARM_UPDATE_URL` for a test.
fn feedUrl(a: std.mem.Allocator) []const u8
{
    var buf: [1024]u16 = undefined;
    const n = GetEnvironmentVariableW(W("TFSWARM_UPDATE_URL"), &buf, buf.len);
    if (n > 0 and n < buf.len)
    {
        return std.unicode.utf16LeToUtf8Alloc(a, buf[0..n]) catch update.latest_url;
    }
    return update.latest_url;
}

/// Looks for a newer release on a thread; `quiet` only speaks up when there is one.
fn startCheck(quiet: bool) void
{
    if (app.update_busy.swap(true, .acq_rel))
    {
        if (!quiet)
        {
            notify("Already looking for updates", "This takes a few seconds.", false);
        }
        return;
    }
    app.update_quiet = quiet;
    app.update_found = null;
    app.update_err = null;
    _ = app.update_arena.reset(.free_all);
    const t = std.Thread.spawn(.{}, checkThread, .{}) catch {
        app.update_busy.store(false, .release);
        return;
    };
    t.detach();
}

fn checkThread() void
{
    const a = app.update_arena.allocator();
    defer _ = PostMessageW(app.hwnd, WM_UPDATE, update_check, 0);
    const body = update.get(a, feedUrl(a), user_agent, 1 << 20) catch |err| {
        app.update_err = err;
        return;
    };
    const r = update.parseRelease(a, body) orelse {
        app.update_err = error.NoInstallerInRelease;
        return;
    };
    if (update.newer(build_options.version, r.version))
    {
        app.update_found = r;
    }
}

fn askToInstall() void
{
    const r = app.update_found orelse return;
    if (app.update_busy.load(.acquire))
    {
        return;
    }
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = std.fmt.allocPrint(a, "TFSwarm {s} is available (this is {s}).\n\nDownload its installer from GitHub ({d} MB) and install it? A running dashboard or node agent stops while it installs, then starts again. Your settings are kept.", .{ r.version, build_options.version, (r.size + (1 << 19)) >> 20 }) catch return;
    const w = std.unicode.utf8ToUtf16LeAllocZ(a, text) catch return;
    const yes = MessageBoxW(app.hwnd, w, W("TFSwarm update"), MB_YESNO | MB_ICONQUESTION) == IDYES;
    logLine("Update to {s}: {s}", .{ r.version, if (yes) "accepted, downloading" else "declined" });
    if (!yes)
    {
        return;
    }
    if (app.update_busy.swap(true, .acq_rel))
    {
        return;
    }
    app.update_err = null;
    app.update_file = beside("TFSwarm-Setup.exe");
    setStatus("downloading TFSwarm {s}", .{r.version});
    const t = std.Thread.spawn(.{}, downloadThread, .{}) catch {
        app.update_busy.store(false, .release);
        return;
    };
    t.detach();
}

fn downloadThread() void
{
    defer _ = PostMessageW(app.hwnd, WM_UPDATE, update_download, 0);
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = app.update_found.?;
    const bytes = update.get(a, r.url, user_agent, 1 << 30) catch |err| {
        app.update_err = err;
        return;
    };
    if (!update.matches(r, bytes))
    {
        app.update_err = error.InstallerDigestMismatch;
        return;
    }
    const dst = a.dupeSentinel(u8, app.update_file, 0) catch return;
    const tmp = std.fmt.allocPrintSentinel(a, "{s}.part", .{app.update_file}, 0) catch return;
    if (!plat.replaceFile(tmp, dst, bytes))
    {
        app.update_err = error.WriteFailed;
    }
}

fn onUpdate(which: usize) void
{
    app.update_busy.store(false, .release);
    var buf: [200]u8 = undefined;
    if (which == update_check)
    {
        if (app.update_err) |err|
        {
            if (!app.update_quiet)
            {
                if (err == error.NotFound)
                {
                    notify("No release yet", "GitHub has no TFSwarm release to update to.", false);
                }
                else
                {
                    notify("Could not check for updates", std.fmt.bufPrint(&buf, "{t}", .{err}) catch "", true);
                }
            }
            return;
        }
        if (app.update_found) |r|
        {
            if (app.update_quiet)
            {
                notify(std.fmt.bufPrint(&buf, "TFSwarm {s} is available", .{r.version}) catch "An update is available", "Choose \"Update\" in this icon's menu to install it.", false);
            }
            else
            {
                askToInstall();
            }
            return;
        }
        if (!app.update_quiet)
        {
            notify("TFSwarm is up to date", std.fmt.bufPrint(&buf, "This is the newest release, {s}.", .{build_options.version}) catch "", false);
        }
        return;
    }
    if (app.update_err) |err|
    {
        notify("The update did not download", std.fmt.bufPrint(&buf, "{t}. Nothing was changed.", .{err}) catch "", true);
        return startFromSettings();
    }
    runInstaller();
}

/// Starts the downloaded installer quietly; it starts the launcher again when it is done.
fn runInstaller() void
{
    var arena: std.heap.ArenaAllocator = .init(app.gpa);
    defer arena.deinit();
    const path = std.unicode.utf8ToUtf16LeAllocZ(arena.allocator(), app.update_file) catch return;
    stopChild();
    const h = ShellExecuteW(null, W("open"), path, W("/SILENT /SUPPRESSMSGBOXES /NORESTART /RELAUNCH"), null, SW_SHOWNORMAL);
    // ShellExecute reports failure as a value of 32 or less.
    logLine("Update: starting {s}", .{app.update_file});
    if (h == null or @intFromPtr(h.?) <= 32)
    {
        notify("The installer did not start", "Run TFSwarm-Setup.exe from the logs folder by hand.", true);
        return startFromSettings();
    }
    runCommand(.quit);
}
