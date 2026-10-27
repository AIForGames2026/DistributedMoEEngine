const std = @import("std");

const c_sources = [_][]const u8{ "csrc/tf_version.c", "csrc/tf_wire.c", "csrc/tf_crc32c.c" };
const c_flags = [_][]const u8{ "-std=c11", "-Wall", "-Wextra", "-Werror" };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const engine = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    engine.addIncludePath(b.path("include"));
    engine.addCSourceFiles(.{ .files = &c_sources, .flags = &c_flags });

    const exe = b.addExecutable(.{ .name = "tensorfold", .root_module = engine });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    b.step("run", "Run the engine").dependOn(&run.step);

    // Pure C consumer of the public headers, to prove the ABI from the C side.
    const abi = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    abi.addIncludePath(b.path("include"));
    abi.addCSourceFiles(.{ .files = &c_sources, .flags = &c_flags });
    abi.addCSourceFile(.{ .file = b.path("tests/abi_test.c"), .flags = &c_flags });
    const abi_test = b.addExecutable(.{ .name = "abi_test", .root_module = abi });

    const tests = b.addTest(.{ .root_module = engine });
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
    test_step.dependOn(&b.addRunArtifact(abi_test).step);

    // Two-process dispatch/combine check (TASK-02).
    const dispatch_mod = b.createModule(.{
        .root_source_file = b.path("src/dispatch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    dispatch_mod.addIncludePath(b.path("include"));
    dispatch_mod.addCSourceFiles(.{ .files = &c_sources, .flags = &c_flags });
    const dispatch = b.addExecutable(.{ .name = "dispatch", .root_module = dispatch_mod });
    b.installArtifact(dispatch);
    const run_dispatch = b.addRunArtifact(dispatch);
    run_dispatch.addPassthruArgs();
    b.step("dispatch", "Run the dispatch/combine check (args after --)").dependOn(&run_dispatch.step);

    // Process start to first kernel call (TASK-01).
    const startup_mod = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true });
    startup_mod.addCSourceFile(.{ .file = b.path("tools/bench/startup.c"), .flags = &c_flags });
    const startup = b.addExecutable(.{ .name = "startup", .root_module = startup_mod });
    const run_startup = b.addRunArtifact(startup);
    run_startup.addArtifactArg(exe);
    b.step("startup", "Time process start to first kernel (median of 20)").dependOn(&run_startup.step);

    addGate(b, target, optimize);
}

/// Toolchain gate (TASK-01): links liburing and the CUDA runtime, so it is a
/// separate step that only machines with both installed need to build.
fn addGate(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const cuda = b.option([]const u8, "cuda", "CUDA toolkit root (default /usr/local/cuda)") orelse "/usr/local/cuda";
    const uring = b.option([]const u8, "uring", "liburing install prefix, if not in system paths");

    const gate = b.createModule(.{
        .root_source_file = b.path("src/gate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gate.addIncludePath(b.path("include"));
    gate.addCSourceFiles(.{ .files = &(c_sources ++ [_][]const u8{ "csrc/tf_uring.c", "csrc/tf_cuda.c" }), .flags = &c_flags });

    gate.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "include" }) });
    const cuda_lib = b.pathJoin(&.{ cuda, "lib64" });
    gate.addLibraryPath(.{ .cwd_relative = cuda_lib });
    gate.addRPath(.{ .cwd_relative = cuda_lib });
    gate.linkSystemLibrary("cudart", .{});

    if (uring) |prefix| {
        gate.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ prefix, "include" }) });
        const lib = b.pathJoin(&.{ prefix, "lib" });
        gate.addLibraryPath(.{ .cwd_relative = lib });
        gate.addRPath(.{ .cwd_relative = lib });
    }
    gate.linkSystemLibrary("uring", .{});

    const exe = b.addExecutable(.{ .name = "gate", .root_module = gate });
    const install = b.addInstallArtifact(exe, .{});
    const run = b.addRunArtifact(exe);
    run.step.dependOn(&install.step);
    b.step("gate", "Build and run the toolchain gate (needs liburing and CUDA)").dependOn(&run.step);
}
