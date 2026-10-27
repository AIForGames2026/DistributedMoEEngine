/* TASK-01: process start to first kernel call. Spawns the engine with
 * --first-kernel (it exits right after its first GEMM) and reports the median
 * wall time over N runs, after 3 warm-up runs.
 *   startup <path/to/tensorfold> [runs]
 * Exit status 1 if the median is 5 ms or more. */
#define _GNU_SOURCE
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>

extern char **environ;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static double run_once(char *path)
{
    char *argv[] = {path, "--first-kernel", NULL};
    pid_t pid;
    double start = now_ms();
    if (posix_spawn(&pid, path, NULL, NULL, argv, environ) != 0) {
        perror("posix_spawn");
        exit(2);
    }
    int status;
    waitpid(pid, &status, 0);
    double ms = now_ms() - start;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "engine exited abnormally\n");
        exit(2);
    }
    return ms;
}

static int cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tensorfold> [runs]\n", argv[0]);
        return 2;
    }
    int runs = argc > 2 ? atoi(argv[2]) : 20;
    if (runs < 1 || runs > 1000)
        runs = 20;

    for (int i = 0; i < 3; i++)
        run_once(argv[1]);

    double t[1000];
    for (int i = 0; i < runs; i++)
        t[i] = run_once(argv[1]);
    qsort(t, runs, sizeof t[0], cmp);
    double median = runs % 2 ? t[runs / 2] : (t[runs / 2 - 1] + t[runs / 2]) / 2;

    printf("start to first kernel over %d runs: median %.3f ms, min %.3f ms, max %.3f ms (target < 5 ms)\n",
           runs, median, t[0], t[runs - 1]);
    return median < 5.0 ? 0 : 1;
}
