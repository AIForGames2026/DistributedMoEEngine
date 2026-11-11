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
    if (threads <= 1 or n <= 1 or !have_pthreads)
    {
        f(ctx, 0, n);
        return;
    }
    shared.run(@min(threads, max_threads), n, &ctx, Thunk.run);
}

const have_pthreads = builtin.os.tag != .windows and builtin.link_libc;

const Func = *const fn (*const anyopaque, usize, usize) void;

const Pool = struct {
    /// Held by the one caller running a loop.
    call: c.pthread_mutex_t = c.PTHREAD_MUTEX_INITIALIZER,
    mutex: c.pthread_mutex_t = c.PTHREAD_MUTEX_INITIALIZER,
    wake: c.pthread_cond_t = c.PTHREAD_COND_INITIALIZER,
    done: c.pthread_cond_t = c.PTHREAD_COND_INITIALIZER,
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
        _ = c.pthread_mutex_lock(&self.call);
        defer _ = c.pthread_mutex_unlock(&self.call);
        const helpers = threads - 1;
        _ = c.pthread_mutex_lock(&self.mutex);
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
        _ = c.pthread_cond_broadcast(&self.wake);
        _ = c.pthread_mutex_unlock(&self.mutex);
        self.take();
        _ = c.pthread_mutex_lock(&self.mutex);
        while (self.busy > 0)
        {
            _ = c.pthread_cond_wait(&self.done, &self.mutex);
        }
        _ = c.pthread_mutex_unlock(&self.mutex);
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
        _ = c.pthread_mutex_lock(&self.mutex);
        while (true)
        {
            while (self.generation == seen)
            {
                _ = c.pthread_cond_wait(&self.wake, &self.mutex);
            }
            seen = self.generation;
            if (id >= self.active)
            {
                continue;
            }
            _ = c.pthread_mutex_unlock(&self.mutex);
            self.take();
            _ = c.pthread_mutex_lock(&self.mutex);
            self.busy -= 1;
            if (self.busy == 0)
            {
                _ = c.pthread_cond_signal(&self.done);
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
