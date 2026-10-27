const std = @import("std");

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
    engine.addCSourceFile(.{
        .file = b.path("csrc/tf_version.c"),
        .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" },
    });

    const exe = b.addExecutable(.{ .name = "tensorfold", .root_module = engine });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    b.step("run", "Run the engine").dependOn(&run.step);

    const tests = b.addTest(.{ .root_module = engine });
    b.step("test", "Run unit tests").dependOn(&b.addRunArtifact(tests).step);
}
