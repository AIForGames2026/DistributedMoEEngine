/* liburing's header uses POSIX and Linux types that strict C11 hides. */
#define _GNU_SOURCE
#include <liburing.h>

#include "tf_gate.h"

int tf_uring_selftest(void)
{
    struct io_uring ring;
    int rc = io_uring_queue_init(8, &ring, 0);
    if (rc < 0)
        return rc;

    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_nop(sqe);
    io_uring_sqe_set_data64(sqe, 0x7f);
    rc = io_uring_submit(&ring);
    if (rc == 1) {
        struct io_uring_cqe *cqe;
        rc = io_uring_wait_cqe(&ring, &cqe);
        if (rc == 0) {
            rc = (cqe->res == 0 && io_uring_cqe_get_data64(cqe) == 0x7f) ? 0 : -1;
            io_uring_cqe_seen(&ring, cqe);
        }
    } else if (rc >= 0) {
        rc = -1;
    }
    io_uring_queue_exit(&ring);
    return rc;
}
