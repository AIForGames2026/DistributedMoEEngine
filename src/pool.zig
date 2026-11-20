//! Worker threads for loops over tokens or rows. Each index runs once, so
//! independent iterations give the same result as on one thread.
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;

pub const max_threads = 32;

/// Runs `f(ctx, first, end)` over pieces of 0 to n - 1 on up to `threads` threads.
/// `f` must not call `parallelFor` itself.
pub fn parallelFor(threads: usize, n: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void
{
    const T = @TypeOf(ctx);
    const Thunk = struct {
        fn run(p: *const anyopaque, first: usize, end: usize) void
        {
            const cx: *const T = @ptrCast(@alignCast(p));
            f(cx.*, first, end);
        }
    };
    if (threads <= 1 or n <= 1 or !have_threads)
    {
        f(ctx, 0, n);
        return;
    }
    shared.run(@min(threads, max_threads), n, &ctx, Thunk.run);
}

const is_windows = builtin.os.tag == .windows;
const have_threads = is_windows or builtin.link_libc;

/// A lock and condition variable: pthreads, or Win32's slim reader/writer locks (TASK-40).
const Mutex = if (is_windows) struct {
    lock: ?*anyopaque = null,

    fn acquire(self: *Mutex) void
    {
        AcquireSRWLockExclusive(&self.lock);
    }
    fn release(self: *Mutex) void
    {
        ReleaseSRWLockExclusive(&self.lock);
    }
} else struct {
    m: c.pthread_mutex_t = c.PTHREAD_MUTEX_INITIALIZER,

    fn acquire(self: *Mutex) void
    {
        _ = c.pthread_mutex_lock(&self.m);
    }
    fn release(self: *Mutex) void
    {
        _ = c.pthread_mutex_unlock(&self.m);
    }
};

const Cond = if (is_windows) struct {
    cv: ?*anyopaque = null,

    fn wait(self: *Cond, m: *Mutex) void
    {
        _ = SleepConditionVariableSRW(&self.cv, &m.lock, 0xFFFFFFFF, 0);
    }
    fn signal(self: *Cond) void
    {
        WakeConditionVariable(&self.cv);
    }
    fn broadcast(self: *Cond) void
    {
        WakeAllConditionVariable(&self.cv);
    }
} else struct {
    cv: c.pthread_cond_t = c.PTHREAD_COND_INITIALIZER,

    fn wait(self: *Cond, m: *Mutex) void
    {
        _ = c.pthread_cond_wait(&self.cv, &m.m);
    }
    fn signal(self: *Cond) void
    {
        _ = c.pthread_cond_signal(&self.cv);
    }
    fn broadcast(self: *Cond) void
    {
        _ = c.pthread_cond_broadcast(&self.cv);
    }
};

extern "kernel32" fn AcquireSRWLockExclusive(lock: *?*anyopaque) callconv(.winapi) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *?*anyopaque) callconv(.winapi) void;
extern "kernel32" fn SleepConditionVariableSRW(cv: *?*anyopaque, lock: *?*anyopaque, ms: u32, flags: u32) callconv(.winapi) c_int;
extern "kernel32" fn WakeConditionVariable(cv: *?*anyopaque) callconv(.winapi) void;
extern "kernel32" fn WakeAllConditionVariable(cv: *?*anyopaque) callconv(.winapi) void;

const Func = *const fn (*const anyopaque, usize, usize) void;

const Pool = struct {
    /// Held by the one caller running a loop.
    call: Mutex = .{},
    mutex: Mutex = .{},
    wake: Cond = .{},
    done: Cond = .{},
    spawned: usize = 0,
    generation: u64 = 0,
    /// Workers with an id below this take part in the current loop.
    active: usize = 0,
    /// Taking part and not finished yet.
    busy: usize = 0,
    ctx: *const anyopaque = undefined,
    func: Func = undefined,
    n: usize = 0,
    piece: usize = 1,
    next: std.atomic.Value(usize) = .init(0),

    fn run(self: *Pool, threads: usize, n: usize, ctx: *const anyopaque, func: Func) void
    {
        self.call.acquire();
        defer self.call.release();
        const helpers = threads - 1;
        self.mutex.acquire();
        while (self.spawned < helpers)
        {
            const th = std.Thread.spawn(.{}, worker, .{ self, self.spawned, self.generation }) catch break;
            th.detach();
            self.spawned += 1;
        }
        const use = @min(helpers, self.spawned);
        self.ctx = ctx;
        self.func = func;
        self.n = n;
        self.piece = @max(1, n / (threads * 4));
        self.next.store(0, .monotonic);
        self.active = use;
        self.busy = use;
        self.generation += 1;
        self.wake.broadcast();
        self.mutex.release();
        self.take();
        self.mutex.acquire();
        while (self.busy > 0)
        {
            self.done.wait(&self.mutex);
        }
        self.mutex.release();
    }

    fn take(self: *Pool) void
    {
        while (true)
        {
            const first = self.next.fetchAdd(self.piece, .monotonic);
            if (first >= self.n)
            {
                return;
            }
            self.func(self.ctx, first, @min(first + self.piece, self.n));
        }
    }

    fn worker(self: *Pool, id: usize, start: u64) void
    {
        var seen = start;
        self.mutex.acquire();
        while (true)
        {
            while (self.generation == seen)
            {
                self.wake.wait(&self.mutex);
            }
            seen = self.generation;
            if (id >= self.active)
            {
                continue;
            }
            self.mutex.release();
            self.take();
            self.mutex.acquire();
            self.busy -= 1;
            if (self.busy == 0)
            {
                self.done.signal();
            }
        }
    }
};

var shared: Pool = .{};

test "parallelFor runs every index once" {
    var hits: [1000]std.atomic.Value(u32) = undefined;
    for (&hits) |*h|
    {
        h.* = .init(0);
    }
    const Ctx = struct {
        hits: []std.atomic.Value(u32),
        fn run(self: @This(), first: usize, end: usize) void
        {
            for (first..end) |i|
            {
                _ = self.hits[i].fetchAdd(1, .monotonic);
            }
        }
    };
    for (0..20) |round|
    {
        parallelFor(1 + round % 8, hits.len, Ctx{ .hits = &hits }, Ctx.run);
    }
    for (&hits) |*h|
    {
        try std.testing.expectEqual(@as(u32, 20), h.load(.monotonic));
    }
}
