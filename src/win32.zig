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
