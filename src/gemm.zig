//! Host FP32 GEMM reference kernels (TASK-01). The naive kernel is the
//! correctness oracle; the tiled kernel must match it.

/// C[m x n] = A[m x k] * B[k x n], row-major.
pub fn naive(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| acc += a[i * k + p] * b[p * n + j];
            c[i * n + j] = acc;
        }
    }
}

const tile = 32;

/// Same result as `naive`, blocked for cache reuse.
pub fn tiled(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, k: usize) void {
    @memset(c[0 .. m * n], 0);
    var row: usize = 0;
    while (row < m) : (row += tile) {
        var inner: usize = 0;
        while (inner < k) : (inner += tile) {
            var col: usize = 0;
            while (col < n) : (col += tile) {
                for (row..@min(row + tile, m)) |i| {
                    for (inner..@min(inner + tile, k)) |p| {
                        const av = a[i * k + p];
                        for (col..@min(col + tile, n)) |j| c[i * n + j] += av * b[p * n + j];
                    }
                }
            }
        }
    }
}

/// Largest relative error between two result matrices.
pub fn maxRelErr(x: []const f32, y: []const f32) f32 {
    var worst: f32 = 0;
    for (x, y) |u, v| {
        const denom = @max(@abs(u), 1e-6);
        worst = @max(worst, @abs(u - v) / denom);
    }
    return worst;
}
