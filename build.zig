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
    // Golden outputs from tools/ref/ref_moe.py, embedded for the reference test.
    engine.addAnonymousImport("moe_tiny.tfref", .{ .root_source_file = b.path("tests/golden/moe_tiny.tfref") });
    // Quantized-format fixture and its golden values from tools/ref/ref_quant.py.
    engine.addAnonymousImport("quant_tiny.gguf", .{ .root_source_file = b.path("tests/golden/quant_tiny.gguf") });
    engine.addAnonymousImport("quant_tiny.tfref", .{ .root_source_file = b.path("tests/golden/quant_tiny.tfref") });
    engine.addAnonymousImport("nvfp4_tiny.safetensors", .{ .root_source_file = b.path("tests/golden/nvfp4_tiny.safetensors") });
    engine.addAnonymousImport("nvfp4_tiny.config.json", .{ .root_source_file = b.path("tests/golden/nvfp4_tiny.config.json") });
    engine.addAnonymousImport("nvfp4_tiny.tfref", .{ .root_source_file = b.path("tests/golden/nvfp4_tiny.tfref") });
    // Tiny random OLMoE and transformers' outputs for it (tools/ref/ref_olmoe.py --tiny).
    engine.addAnonymousImport("olmoe_tiny.gguf", .{ .root_source_file = b.path("tests/golden/olmoe_tiny.gguf") });
    engine.addAnonymousImport("olmoe_tiny.tfref", .{ .root_source_file = b.path("tests/golden/olmoe_tiny.tfref") });
    // The TCP transport uses Winsock on Windows.
    if (target.result.os.tag == .windows) engine.linkSystemLibrary("ws2_32", .{});

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
    dispatch_mod.addAnonymousImport("moe_tiny.tfref", .{ .root_source_file = b.path("tests/golden/moe_tiny.tfref") });
    if (target.result.os.tag == .windows) dispatch_mod.linkSystemLibrary("ws2_32", .{});
    // RDMA transport (TASK-07): opt in, since it needs libibverbs (Linux).
    const rdma = b.option(bool, "rdma", "Build dispatch with the RDMA transport (needs libibverbs)") orelse false;
    const dispatch_options = b.addOptions();
    dispatch_options.addOption(bool, "rdma", rdma);
    dispatch_mod.addOptions("build_options", dispatch_options);
    if (rdma) {
        dispatch_mod.addCSourceFile(.{ .file = b.path("csrc/tf_rdma.c"), .flags = &c_flags });
        dispatch_mod.linkSystemLibrary("ibverbs", .{});
    }
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

    const uring = b.option([]const u8, "uring", "liburing install prefix, if not in system paths");
    const cuda = b.option([]const u8, "cuda", "CUDA toolkit root (default /usr/local/cuda)") orelse "/usr/local/cuda";
    addIo(b, target, optimize, uring, rdma, cuda);
    addGate(b, target, optimize, uring, cuda);
}

/// Links liburing, from `prefix` if given, else from the system paths.
fn linkUring(b: *std.Build, mod: *std.Build.Module, prefix: ?[]const u8) void {
    if (prefix) |p| {
        mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ p, "include" }) });
        const lib = b.pathJoin(&.{ p, "lib" });
        mod.addLibraryPath(.{ .cwd_relative = lib });
        mod.addRPath(.{ .cwd_relative = lib });
    }
    mod.linkSystemLibrary("uring", .{});
}

/// io_uring expert reader (TASK-03): its unit test and bench need liburing,
/// so they live in their own step, like the gate.
fn addIo(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, uring: ?[]const u8, rdma: bool, cuda: []const u8) void {
    const io_test = b.step("io-test", "Run the io_uring reader unit test (needs liburing)");
    const io_bench = b.step("io-bench", "Build tools/bench/uring_bench (needs liburing)");
    const programs = [_]struct { name: []const u8, src: []const u8, step: *std.Build.Step }{
        .{ .name = "reader_test", .src = "tests/reader_test.c", .step = io_test },
        .{ .name = "uring_bench", .src = "tools/bench/uring_bench.c", .step = io_bench },
    };
    for (programs) |p| {
        const mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
        mod.addIncludePath(b.path("include"));
        mod.addCSourceFiles(.{ .files = &.{ "csrc/io/uring_reader.c", p.src }, .flags = &c_flags });
        linkUring(b, mod, uring);
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        const install = b.addInstallArtifact(exe, .{});
        if (p.step == io_test) {
            p.step.dependOn(&b.addRunArtifact(exe).step);
        } else {
            p.step.dependOn(&install.step);
        }
    }

    // Expert loader tool (TASK-08): GGUF / safetensors through the reader.
    const tfload_mod = b.createModule(.{
        .root_source_file = b.path("src/tfload.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    tfload_mod.addIncludePath(b.path("include"));
    tfload_mod.addCSourceFiles(.{ .files = &(c_sources ++ [_][]const u8{"csrc/io/uring_reader.c"}), .flags = &c_flags });
    linkUring(b, tfload_mod, uring);
    const tfload = b.addExecutable(.{ .name = "tfload", .root_module = tfload_mod });
    b.step("tfload", "Build the expert loader tool (needs liburing)").dependOn(&b.addInstallArtifact(tfload, .{}).step);

    // OLMoE forward pass (TASK-10): loads through the same reader.
    const olmoe_mod = b.createModule(.{
        .root_source_file = b.path("src/olmoe_run.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    olmoe_mod.addIncludePath(b.path("include"));
    olmoe_mod.addCSourceFiles(.{ .files = &(c_sources ++ [_][]const u8{"csrc/io/uring_reader.c"}), .flags = &c_flags });
    linkUring(b, olmoe_mod, uring);
    const gpu = b.option(bool, "gpu", "Build olmoe with CUDA matmuls (needs nvcc and the driver)") orelse false;
    const olmoe_options = b.addOptions();
    olmoe_options.addOption(bool, "rdma", rdma);
    olmoe_options.addOption(bool, "gpu", gpu);
    olmoe_mod.addOptions("build_options", olmoe_options);
    if (rdma) {
        olmoe_mod.addCSourceFile(.{ .file = b.path("csrc/tf_rdma.c"), .flags = &c_flags });
        olmoe_mod.linkSystemLibrary("ibverbs", .{});
    }
    if (gpu) {
        // Kernels for Ampere (sm_86, such as the RTX 3060 Ti and 3070), the RTX 5070 Ti (sm_120),
        // and GB10 (sm_121), plus PTX for anything newer; loaded at run time through the driver API.
        const nvcc = b.addSystemCommand(&.{
            b.pathJoin(&.{ cuda, "bin", "nvcc" }), "-fatbin",
            "-O3",                                 "-gencode",
            "arch=compute_86,code=sm_86",          "-gencode",
            "arch=compute_120,code=sm_120",        "-gencode",
            "arch=compute_121,code=sm_121",        "-gencode",
            "arch=compute_120,code=compute_120",   "-o",
        });
        const fatbin = nvcc.addOutputFileArg("kernels.fatbin");
        nvcc.addFileArg(b.path("csrc/cuda/kernels.cu"));
        olmoe_mod.addAnonymousImport("kernels.fatbin", .{ .root_source_file = fatbin });
        olmoe_mod.addCSourceFile(.{ .file = b.path("csrc/tf_gpu.c"), .flags = &c_flags });
        olmoe_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "include" }) });
        // Link against the toolkit's stub; the driver's libcuda loads at run time.
        olmoe_mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda, "lib64", "stubs" }) });
        olmoe_mod.linkSystemLibrary("cuda", .{});
    }
    const olmoe = b.addExecutable(.{ .name = "olmoe", .root_module = olmoe_mod });
    b.step("olmoe", "Build the OLMoE forward-pass tool (needs liburing)").dependOn(&b.addInstallArtifact(olmoe, .{}).step);
}

/// Toolchain gate (TASK-01): links liburing and the CUDA runtime, so it is a
/// separate step that only machines with both installed need to build.
fn addGate(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, uring: ?[]const u8, cuda: []const u8) void {
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

    linkUring(b, gate, uring);

    const exe = b.addExecutable(.{ .name = "gate", .root_module = gate });
    const install = b.addInstallArtifact(exe, .{});
    const run = b.addRunArtifact(exe);
    run.step.dependOn(&install.step);
    b.step("gate", "Build and run the toolchain gate (needs liburing and CUDA)").dependOn(&run.step);
}
