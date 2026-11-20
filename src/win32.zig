//! The few Win32 and Winsock calls the engine makes on Windows. Zig 0.17's
//! std no longer declares most of them, so they are declared here. Only
//! referenced from `builtin.os.tag == .windows` branches.
const std = @import("std");

pub const HANDLE = *anyopaque;
pub const BOOL = c_int;
pub const DWORD = u32;
pub const INVALID_HANDLE_VALUE: HANDLE = @ptrFromInt(std.math.maxInt(usize));

pub const PAGE_READWRITE = 0x04;
pub const MEM_COMMIT = 0x1000;
pub const MEM_RESERVE = 0x2000;
pub const MEM_RELEASE = 0x8000;
pub const FILE_MAP_ALL_ACCESS = 0xF001F;

pub extern "kernel32" fn VirtualAlloc(addr: ?*anyopaque, size: usize, alloc_type: DWORD, protect: DWORD) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn VirtualFree(addr: *anyopaque, size: usize, free_type: DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn CreateFileMappingW(file: HANDLE, attrs: ?*anyopaque, protect: DWORD, size_high: DWORD, size_low: DWORD, name: ?[*:0]const u16) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn OpenFileMappingW(access: DWORD, inherit: BOOL, name: [*:0]const u16) callconv(.winapi) ?HANDLE;
pub extern "kernel32" fn MapViewOfFile(mapping: HANDLE, access: DWORD, offset_high: DWORD, offset_low: DWORD, len: usize) callconv(.winapi) ?*anyopaque;
pub extern "kernel32" fn UnmapViewOfFile(addr: *const anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetCurrentProcessId() callconv(.winapi) DWORD;
pub extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) BOOL;
pub extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.winapi) BOOL;
pub extern "kernel32" fn Sleep(ms: DWORD) callconv(.winapi) void;
pub extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: DWORD, share: DWORD, attrs: ?*anyopaque, disposition: DWORD, flags: DWORD, template: ?HANDLE) callconv(.winapi) HANDLE;
pub extern "kernel32" fn ReadFile(file: HANDLE, buf: [*]u8, len: DWORD, read: ?*DWORD, overlapped: ?*OVERLAPPED) callconv(.winapi) BOOL;
pub extern "kernel32" fn WriteFile(file: HANDLE, buf: [*]const u8, len: DWORD, written: ?*DWORD, overlapped: ?*OVERLAPPED) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetFileSizeEx(file: HANDLE, size: *i64) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetLastError() callconv(.winapi) DWORD;
pub extern "kernel32" fn GetSystemTimeAsFileTime(ft: *u64) callconv(.winapi) void;
pub extern "kernel32" fn GetComputerNameA(buf: [*]u8, len: *DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn GetCurrentProcess() callconv(.winapi) HANDLE;
pub extern "kernel32" fn K32GetProcessMemoryInfo(process: HANDLE, counters: *PROCESS_MEMORY_COUNTERS, cb: DWORD) callconv(.winapi) BOOL;

pub const PROCESS_MEMORY_COUNTERS = extern struct {
    cb: DWORD = 0,
    page_faults: DWORD = 0,
    peak_working_set: usize = 0,
    working_set: usize = 0,
    quota_peak_paged_pool: usize = 0,
    quota_paged_pool: usize = 0,
    quota_peak_nonpaged_pool: usize = 0,
    quota_nonpaged_pool: usize = 0,
    pagefile_usage: usize = 0,
    peak_pagefile_usage: usize = 0,
};

pub const GENERIC_READ: DWORD = 0x80000000;
pub const GENERIC_WRITE: DWORD = 0x40000000;
pub const FILE_SHARE_READ: DWORD = 0x1;
pub const CREATE_ALWAYS: DWORD = 2;
pub const OPEN_EXISTING: DWORD = 3;
pub const FILE_ATTRIBUTE_NORMAL: DWORD = 0x80;

pub const OVERLAPPED = extern struct {
    internal: usize = 0,
    internal_high: usize = 0,
    offset: DWORD = 0,
    offset_high: DWORD = 0,
    event: ?HANDLE = null,
};

/// `path` (UTF-8) as a terminated UTF-16 string in `buf`; null if it does not fit.
pub fn widePath(path: []const u8, buf: *[1024:0]u16) ?[:0]const u16
{
    const n = std.unicode.utf8ToUtf16Le(buf, path) catch return null;
    if (n >= buf.len)
    {
        return null;
    }
    buf[n] = 0;
    return buf[0..n :0];
}

/// Opens `path` (UTF-8) with CreateFileW; null if it cannot. Others may keep
/// reading and writing it, as with a log an engine is still writing.
pub fn openFile(path: []const u8, access: DWORD, disposition: DWORD) ?HANDLE
{
    var wide: [1024:0]u16 = undefined;
    const w = widePath(path, &wide) orelse return null;
    const h = CreateFileW(w, access, FILE_SHARE_READ | 0x2, null, disposition, FILE_ATTRIBUTE_NORMAL, null); // 0x2: FILE_SHARE_WRITE
    return if (h == INVALID_HANDLE_VALUE) null else h;
}

pub extern "kernel32" fn GetFullPathNameW(name: [*:0]const u16, len: DWORD, buf: [*]u16, file_part: ?*?[*:0]u16) callconv(.winapi) DWORD;
pub extern "kernel32" fn GetModuleFileNameW(module: ?*anyopaque, buf: [*]u16, len: DWORD) callconv(.winapi) DWORD;
pub extern "kernel32" fn GetLongPathNameW(short: [*:0]const u16, long: [*]u16, len: DWORD) callconv(.winapi) DWORD;
pub extern "kernel32" fn GetFileAttributesExW(name: [*:0]const u16, level: c_int, info: *FILE_ATTRIBUTE_DATA) callconv(.winapi) BOOL;
pub extern "kernel32" fn CreateDirectoryW(name: [*:0]const u16, attrs: ?*anyopaque) callconv(.winapi) BOOL;
pub extern "kernel32" fn MoveFileExW(from: [*:0]const u16, to: [*:0]const u16, flags: DWORD) callconv(.winapi) BOOL;
pub extern "kernel32" fn DeleteFileW(name: [*:0]const u16) callconv(.winapi) BOOL;
pub extern "kernel32" fn FindFirstFileW(pattern: [*:0]const u16, data: *WIN32_FIND_DATAW) callconv(.winapi) HANDLE;
pub extern "kernel32" fn FindNextFileW(find: HANDLE, data: *WIN32_FIND_DATAW) callconv(.winapi) BOOL;
pub extern "kernel32" fn FindClose(find: HANDLE) callconv(.winapi) BOOL;
pub extern "kernel32" fn GlobalMemoryStatusEx(status: *MEMORYSTATUSEX) callconv(.winapi) BOOL;
pub extern "kernel32" fn AcquireSRWLockExclusive(lock: *?*anyopaque) callconv(.winapi) void;
pub extern "kernel32" fn ReleaseSRWLockExclusive(lock: *?*anyopaque) callconv(.winapi) void;
pub extern "advapi32" fn SystemFunction036(buf: [*]u8, len: u32) callconv(.winapi) u8; // RtlGenRandom
pub extern "user32" fn EnumDisplayDevicesA(device: ?[*:0]const u8, index: DWORD, out: *DISPLAY_DEVICEA, flags: DWORD) callconv(.winapi) BOOL;

pub const FILE_ATTRIBUTE_DIRECTORY: DWORD = 0x10;
pub const MOVEFILE_REPLACE_EXISTING: DWORD = 0x1;
pub const CREATE_NEW: DWORD = 1;

pub const FILE_ATTRIBUTE_DATA = extern struct {
    attributes: DWORD,
    created: u64 align(4),
    accessed: u64 align(4),
    written: u64 align(4),
    size_high: DWORD,
    size_low: DWORD,
};

pub const WIN32_FIND_DATAW = extern struct {
    attributes: DWORD,
    created: u64 align(4),
    accessed: u64 align(4),
    written: u64 align(4),
    size_high: DWORD,
    size_low: DWORD,
    reserved0: DWORD,
    reserved1: DWORD,
    name: [260]u16,
    short_name: [14]u16,
};

pub const MEMORYSTATUSEX = extern struct {
    length: DWORD = @sizeOf(MEMORYSTATUSEX),
    load: DWORD = 0,
    total_phys: u64 = 0,
    avail_phys: u64 = 0,
    total_page: u64 = 0,
    avail_page: u64 = 0,
    total_virtual: u64 = 0,
    avail_virtual: u64 = 0,
    avail_extended: u64 = 0,
};

pub const DISPLAY_DEVICEA = extern struct {
    cb: DWORD = @sizeOf(DISPLAY_DEVICEA),
    name: [32]u8 = undefined,
    string: [128]u8 = undefined,
    flags: DWORD = 0,
    id: [128]u8 = undefined,
    key: [128]u8 = undefined,
};

/// Reads up to `len` bytes at `offset`, like pread; returns the count or null on an error.
pub fn readAt(h: HANDLE, offset: u64, buf: []u8) ?usize
{
    var ov: OVERLAPPED = .{ .offset = @truncate(offset), .offset_high = @intCast(offset >> 32) };
    var got: DWORD = 0;
    const len: DWORD = @intCast(@min(buf.len, 1 << 30));
    if (ReadFile(h, buf.ptr, len, &got, &ov) == 0)
    {
        return if (GetLastError() == 38) 0 else null; // ERROR_HANDLE_EOF
    }
    return got;
}

pub const ws2 = std.os.windows.ws2_32;
pub const SOCKET = usize;
pub const INVALID_SOCKET: SOCKET = std.math.maxInt(usize);
pub const SD_SEND = 1;
/// POLLRDNORM | POLLRDBAND; WSAPoll rejects POLLPRI, so POLLIN is spelled out.
pub const POLLIN: i16 = 0x0100 | 0x0200;
pub const WSAEINTR = 10004;

pub const WSAPOLLFD = extern struct {
    fd: SOCKET,
    events: i16,
    revents: i16,
};

pub extern "ws2_32" fn WSAStartup(version: u16, data: *anyopaque) callconv(.winapi) c_int;
pub extern "ws2_32" fn WSAGetLastError() callconv(.winapi) c_int;
pub extern "ws2_32" fn WSAPoll(fds: [*]WSAPOLLFD, n: u32, timeout_ms: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn socket(af: c_int, kind: c_int, protocol: c_int) callconv(.winapi) SOCKET;
pub extern "ws2_32" fn closesocket(s: SOCKET) callconv(.winapi) c_int;
pub extern "ws2_32" fn bind(s: SOCKET, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn listen(s: SOCKET, backlog: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn accept(s: SOCKET, addr: ?*anyopaque, len: ?*c_int) callconv(.winapi) SOCKET;
pub extern "ws2_32" fn connect(s: SOCKET, addr: *const anyopaque, len: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn send(s: SOCKET, buf: [*]const u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn recv(s: SOCKET, buf: [*]u8, len: c_int, flags: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn shutdown(s: SOCKET, how: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn setsockopt(s: SOCKET, level: c_int, name: c_int, value: *const anyopaque, len: c_int) callconv(.winapi) c_int;
pub extern "ws2_32" fn getsockname(s: SOCKET, addr: *anyopaque, len: *c_int) callconv(.winapi) c_int;

/// Winsock must be started once per process before any socket call.
pub fn startWinsock() !void {
    var data: [64]u64 = undefined; // WSADATA, 408 bytes on x86_64
    if (WSAStartup(0x0202, &data) != 0) return error.SystemResources;
}

/// Monotonic clock in nanoseconds.
pub fn nowNs() u64 {
    var count: i64 = 0;
    var freq: i64 = 1;
    _ = QueryPerformanceCounter(&count);
    _ = QueryPerformanceFrequency(&freq);
    const c: u128 = @intCast(count);
    return @intCast(c * std.time.ns_per_s / @as(u128, @intCast(freq)));
}
