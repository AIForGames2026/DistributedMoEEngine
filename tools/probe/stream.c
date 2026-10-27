/* TASK-05 STREAM-style CPU memory bandwidth test (triad: a = b + s * c).
 *   stream [MiB per array] [threads]
 * Prints one JSON object. Counts 3 arrays of traffic per triad (2 reads and
 * 1 write, no write-allocate), as STREAM does, and reports the best of 10. */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    double *a, *b, *c;
    size_t lo, hi;
    pthread_barrier_t *start, *stop;
    int reps;
} job;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void *worker(void *p)
{
    job *j = p;
    /* First touch: each thread faults in its own slice. */
    for (size_t i = j->lo; i < j->hi; i++) {
        j->a[i] = 0;
        j->b[i] = 1;
        j->c[i] = 2;
    }
    for (int r = 0; r < j->reps; r++) {
        pthread_barrier_wait(j->start);
        for (size_t i = j->lo; i < j->hi; i++)
            j->a[i] = j->b[i] + 3.0 * j->c[i];
        pthread_barrier_wait(j->stop);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    size_t mib = argc > 1 ? (size_t)atol(argv[1]) : 256;
    int threads = argc > 2 ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (mib == 0 || threads < 1 || threads > 1024)
        return 2;
    size_t n = (mib << 20) / sizeof(double);
    double *a = aligned_alloc(4096, n * sizeof(double));
    double *b = aligned_alloc(4096, n * sizeof(double));
    double *c = aligned_alloc(4096, n * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    const int reps = 10;
    pthread_barrier_t start, stop;
    pthread_barrier_init(&start, NULL, threads + 1);
    pthread_barrier_init(&stop, NULL, threads + 1);
    pthread_t *tid = calloc(threads, sizeof *tid);
    job *jobs = calloc(threads, sizeof *jobs);
    for (int t = 0; t < threads; t++) {
        jobs[t] = (job){a, b, c, n * t / threads, n * (t + 1) / threads, &start, &stop, reps};
        pthread_create(&tid[t], NULL, worker, &jobs[t]);
    }

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        pthread_barrier_wait(&start);
        double t0 = now_s();
        pthread_barrier_wait(&stop);
        double dt = now_s() - t0;
        if (dt < best)
            best = dt;
    }
    for (int t = 0; t < threads; t++)
        pthread_join(tid[t], NULL);

    /* Use the result so the compiler cannot drop the loops. */
    double check = a[0] + a[n / 2] + a[n - 1];
    printf("{\"array_mib\": %zu, \"threads\": %d, \"triad_gb_s\": %.1f, \"check\": %.0f}\n", mib, threads,
           3.0 * n * sizeof(double) / best / 1e9, check);
    free(a);
    free(b);
    free(c);
    return check == 21.0 ? 0 : 1;
}
