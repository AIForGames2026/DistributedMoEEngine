//! Windows engine processes for the dashboard and its node agents (TASK-43),
//! the `proc.Child` of Linux: output to a log file, and a job object that ends
//! the process and anything it starts when stopped or when this process exits.
const std = @import("std");
const win32 = @import("win32.zig");
const plat = @import("plat.zig");

const HANDLE = win32.HANDLE;
const DWORD = win32.DWORD;
const BOOL = win32.BOOL;

const SECURITY_ATTRIBUTES = extern struct {
    length: DWORD = @sizeOf(SECURITY_ATTRIBUTES),
    descriptor: ?*anyopaque = null,
    inherit: BOOL = 1,
};

const STARTUPINFOW = extern struct {
    cb: DWORD,
    reserved: ?[*:0]u16 = null,
    desktop: ?[*:0]u16 = null,
    title: ?[*:0]u16 = null,
    x: DWORD = 0,
    y: DWORD = 0,
    x_size: DWORD = 0,
    y_size: DWORD = 0,
    x_chars: DWORD = 0,
    y_chars: DWORD = 0,
    fill: DWORD = 0,
    flags: DWORD = 0,
    show_window: u16 = 0,
    reserved2_len: u16 = 0,
    reserved2: ?*u8 = null,
    std_input: ?HANDLE = null,
    std_output: ?HANDLE = null,
    std_error: ?HANDLE = null,
};

const STARTUPINFOEXW = extern struct {
    info: STARTUPINFOW,
    attributes: ?*anyopaque,
};

const PROCESS_INFORMATION = extern struct {
    process: HANDLE,
    thread: HANDLE,
    pid: DWORD,
    tid: DWORD,
};

const JOBOBJECT_EXTENDED_LIMIT_INFORMATION = extern struct {
    per_process_time: i64 = 0,
    per_job_time: i64 = 0,
    limit_flags: DWORD = 0,
    min_working_set: usize = 0,
    max_working_set: usize = 0,
    active_processes: DWORD = 0,
    affinity: usize = 0,
    priority_class: DWORD = 0,
    scheduling_class: DWORD = 0,
    io: [6]u64 = @splat(0),
    process_memory_limit: usize = 0,
    job_memory_limit: usize = 0,
    peak_process_memory: usize = 0,
    peak_job_memory: usize = 0,
};

extern "kernel32" fn CreateProcessW(app: ?[*:0]const u16, cmd: [*:0]u16, pattrs: ?*anyopaque, tattrs: ?*anyopaque, inherit: BOOL, flags: DWORD, env: ?*anyopaque, cwd: ?[*:0]const u16, startup: *STARTUPINFOEXW, info: *PROCESS_INFORMATION) callconv(.winapi) BOOL;
extern "kernel32" fn InitializeProcThreadAttributeList(list: ?*anyopaque, count: DWORD, flags: DWORD, size: *usize) callconv(.winapi) BOOL;
extern "kernel32" fn UpdateProcThreadAttribute(list: *anyopaque, flags: DWORD, attribute: usize, value: *const anyopaque, size: usize, prev: ?*anyopaque, ret: ?*usize) callconv(.winapi) BOOL;
extern "kernel32" fn DeleteProcThreadAttributeList(list: *anyopaque) callconv(.winapi) void;
extern "kernel32" fn CreateJobObjectW(attrs: ?*anyopaque, name: ?[*:0]const u16) callconv(.winapi) ?HANDLE;
extern "kernel32" fn SetInformationJobObject(job: HANDLE, class: c_int, info: *const anyopaque, len: DWORD) callconv(.winapi) BOOL;
extern "kernel32" fn AssignProcessToJobObject(job: HANDLE, process: HANDLE) callconv(.winapi) BOOL;
extern "kernel32" fn TerminateJobObject(job: HANDLE, code: u32) callconv(.winapi) BOOL;
extern "kernel32" fn ResumeThread(thread: HANDLE) callconv(.winapi) DWORD;
extern "kernel32" fn WaitForSingleObject(h: HANDLE, ms: DWORD) callconv(.winapi) DWORD;
extern "kernel32" fn GetExitCodeProcess(process: HANDLE, code: *DWORD) callconv(.winapi) BOOL;
extern "kernel32" fn TerminateProcess(process: HANDLE, code: u32) callconv(.winapi) BOOL;
extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: DWORD, share: DWORD, attrs: ?*const SECURITY_ATTRIBUTES, disposition: DWORD, flags: DWORD, template: ?HANDLE) callconv(.winapi) HANDLE;

const EXTENDED_STARTUPINFO_PRESENT: DWORD = 0x0008_0000;
const CREATE_SUSPENDED: DWORD = 0x0000_0004;
const CREATE_NO_WINDOW: DWORD = 0x0800_0000;
const STARTF_USESTDHANDLES: DWORD = 0x0000_0100;
const PROC_THREAD_ATTRIBUTE_HANDLE_LIST: usize = 0x0002_0002;
const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: DWORD = 0x2000;
const JobObjectExtendedLimitInformation: c_int = 9;
const STILL_ACTIVE: DWORD = 259;
const FILE_SHARE_WRITE: DWORD = 0x2;

pub const Child = struct {
    pid: i32,
    log_path: []u8,
    started_ns: u64,
    /// Set once the process has exited: its exit code (143 when stopped, as SIGTERM on Linux).
    exit: ?u8 = null,
    process: HANDLE,
    job: HANDLE,

    /// Starts `argv` with output going to `log_path`. It ends if this process exits.
    pub fn spawn(gpa: std.mem.Allocator, argv: []const []const u8, log_path: []const u8) !Child
    {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cmd = try std.unicode.utf8ToUtf16LeAllocZ(a, try commandLine(a, argv));
        const log_w = try std.unicode.utf8ToUtf16LeAllocZ(a, log_path);

        const sa: SECURITY_ATTRIBUTES = .{};
        const log = CreateFileW(log_w, win32.GENERIC_WRITE, win32.FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, win32.CREATE_ALWAYS, win32.FILE_ATTRIBUTE_NORMAL, null);
        if (log == win32.INVALID_HANDLE_VALUE)
        {
            return error.LogUnwritable;
        }
        defer _ = win32.CloseHandle(log);
        const nul_w = std.unicode.utf8ToUtf16LeStringLiteral("NUL");
        const nul = CreateFileW(nul_w, win32.GENERIC_READ, win32.FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, win32.OPEN_EXISTING, 0, null);
        if (nul == win32.INVALID_HANDLE_VALUE)
        {
            return error.SystemResources;
        }
        defer _ = win32.CloseHandle(nul);

        // Only the log and NUL are inherited, not this process's sockets and files.
        var size: usize = 0;
        _ = InitializeProcThreadAttributeList(null, 1, 0, &size);
        const list_mem = try a.alignedAlloc(u8, .of(usize), size);
        const list: *anyopaque = list_mem.ptr;
        if (InitializeProcThreadAttributeList(list, 1, 0, &size) == 0)
        {
            return error.SystemResources;
        }
        defer DeleteProcThreadAttributeList(list);
        const handles = [2]HANDLE{ log, nul };
        if (UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, &handles, @sizeOf(@TypeOf(handles)), null, null) == 0)
        {
            return error.SystemResources;
        }
        var si: STARTUPINFOEXW = .{
            .info = .{ .cb = @sizeOf(STARTUPINFOEXW), .flags = STARTF_USESTDHANDLES, .std_input = nul, .std_output = log, .std_error = log },
            .attributes = list,
        };

        const job = CreateJobObjectW(null, null) orelse return error.SystemResources;
        errdefer _ = win32.CloseHandle(job);
        const limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = .{ .limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE };
        if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, @sizeOf(JOBOBJECT_EXTENDED_LIMIT_INFORMATION)) == 0)
        {
            return error.SystemResources;
        }
        var pi: PROCESS_INFORMATION = undefined;
        if (CreateProcessW(null, cmd.ptr, null, null, 1, EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW, null, null, &si, &pi) == 0)
        {
            return error.SpawnFailed;
        }
        defer _ = win32.CloseHandle(pi.thread);
        // In the job before it runs, so everything it starts is in the job too.
        if (AssignProcessToJobObject(job, pi.process) == 0)
        {
            _ = TerminateProcess(pi.process, 1);
            _ = win32.CloseHandle(pi.process);
            return error.SpawnFailed;
        }
        _ = ResumeThread(pi.thread);
        return .{ .pid = @intCast(pi.pid), .log_path = try gpa.dupe(u8, log_path), .started_ns = plat.monoNs(), .process = pi.process, .job = job };
    }

    pub fn deinit(self: *Child, gpa: std.mem.Allocator) void
    {
        gpa.free(self.log_path);
        _ = win32.CloseHandle(self.process);
        _ = win32.CloseHandle(self.job);
    }

    /// Collects the exit code if the process has ended. True while it runs.
    pub fn alive(self: *Child) bool
    {
        if (self.exit != null)
        {
            return false;
        }
        if (WaitForSingleObject(self.process, 0) != 0)
        {
            return true;
        }
        var code: DWORD = 0;
        _ = GetExitCodeProcess(self.process, &code);
        self.exit = @truncate(code);
        return false;
    }

    /// Ends the process and everything it started, and waits for the end. Windows
    /// has no SIGTERM to send first, so `grace_ms` is not used.
    pub fn stop(self: *Child, grace_ms: u32) void
    {
        _ = grace_ms;
        if (!self.alive())
        {
            return;
        }
        _ = TerminateJobObject(self.job, 128 + 15);
        _ = WaitForSingleObject(self.process, 0xFFFF_FFFF);
        _ = self.alive();
    }

    /// True once the log contains `needle`.
    pub fn logHas(self: *const Child, gpa: std.mem.Allocator, needle: []const u8) bool
    {
        const t = @import("proc.zig").tail(gpa, self.log_path, 64 << 10) catch return false;
        defer gpa.free(t);
        return std.mem.indexOf(u8, t, needle) != null;
    }
};

/// Joins `argv` into a command line that CommandLineToArgvW and the C runtime split back the same way.
pub fn commandLine(a: std.mem.Allocator, argv: []const []const u8) ![]const u8
{
    var out: std.ArrayList(u8) = .empty;
    for (argv, 0..) |arg, i|
    {
        if (i > 0)
        {
            try out.append(a, ' ');
        }
        const plain = arg.len > 0 and std.mem.indexOfAny(u8, arg, " \t\n\"") == null;
        if (plain)
        {
            try out.appendSlice(a, arg);
            continue;
        }
        try out.append(a, '"');
        var slashes: usize = 0;
        for (arg) |ch|
        {
            if (ch == '\\')
            {
                slashes += 1;
                continue;
            }
            if (ch == '"')
            {
                try out.appendNTimes(a, '\\', slashes * 2 + 1); // each backslash doubled, then the quote escaped
            }
            else
            {
                try out.appendNTimes(a, '\\', slashes);
            }
            slashes = 0;
            try out.append(a, ch);
        }
        try out.appendNTimes(a, '\\', slashes * 2); // doubled before the closing quote
        try out.append(a, '"');
    }
    return out.items;
}

test "command lines quote what needs it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("tfswarm.exe expert C:\\m\\x.gguf --port 5", try commandLine(a, &.{ "tfswarm.exe", "expert", "C:\\m\\x.gguf", "--port", "5" }));
    try std.testing.expectEqualStrings("\"C:\\Program Files\\o.exe\" \"\" \"a\\\"b\" \"c d\\\\\"", try commandLine(a, &.{ "C:\\Program Files\\o.exe", "", "a\"b", "c d\\" }));
}
