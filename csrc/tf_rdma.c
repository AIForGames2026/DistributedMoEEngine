/* RoCE v2 RDMA link (TASK-07), on libibverbs. See include/tf_rdma.h.
 * Most verbs entry points are static inline functions in verbs.h, so the Zig
 * transport calls this C layer instead of libibverbs directly. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tf_rdma.h"

#define PORT 1
#define POLL_BATCH 32

struct tf_rdma {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *send_cq;
    struct ibv_cq *recv_cq;
    struct ibv_qp *qp;
    struct ibv_mr *ring_mr;
    struct ibv_mr *stage_mr;
    uint8_t *ring;
    uint8_t *stage;
    uint32_t slots;
    size_t packet_bytes;
    uint32_t sq_depth;
    uint32_t rq_depth;
    int gid_index;
    union ibv_gid gid;
    enum ibv_mtu mtu;
    uint32_t psn;
    uint64_t remote_addr;
    uint32_t remote_rkey;
    /* Sends are numbered from 0 and all signaled; RC completes them in order,
     * so `completed` is one past the newest completed send. */
    uint64_t posted;
    uint64_t completed;
    /* stage_done[i]: `completed` must reach this before staging buffer i is
     * reused (0: never used). */
    uint64_t *stage_done;
    uint32_t stage_next;
    int failed;
    tf_rdma_stats stats;
};

static void set_err(char *err, size_t len, const char *fmt, const char *what, int code)
{
    if (err && len) snprintf(err, len, fmt, what, code);
}

/* First RoCE v2 GID whose address is IPv4-mapped (::ffff:a.b.c.d). */
static int find_roce_v2_ipv4_gid(struct ibv_context *ctx, int table_len)
{
    for (int i = 0; i < table_len; i++) {
        char path[256], type[32] = {0};
        snprintf(path, sizeof path, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d",
                 ibv_get_device_name(ctx->device), PORT, i);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        int ok = fgets(type, sizeof type, f) != NULL;
        fclose(f);
        if (!ok || strncmp(type, "RoCE v2", 7) != 0) continue;
        union ibv_gid gid;
        if (ibv_query_gid(ctx, PORT, i, &gid) != 0) continue;
        static const uint8_t v4_prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
        if (memcmp(gid.raw, v4_prefix, sizeof v4_prefix) == 0) return i;
    }
    return -1;
}

static int post_recvs(tf_rdma *r, uint32_t n)
{
    /* Writes with immediate need a receive but place their data by address,
     * so the receives carry no buffers. */
    struct ibv_recv_wr wrs[POLL_BATCH], *bad;
    while (n > 0) {
        uint32_t k = n < POLL_BATCH ? n : POLL_BATCH;
        for (uint32_t i = 0; i < k; i++) {
            wrs[i] = (struct ibv_recv_wr){.wr_id = 0, .sg_list = NULL, .num_sge = 0,
                                          .next = i + 1 < k ? &wrs[i + 1] : NULL};
        }
        int rc = ibv_post_recv(r->qp, wrs, &bad);
        if (rc) return -rc;
        n -= k;
    }
    return 0;
}

/* Reaps send completions; with `block`, waits for at least one. */
static int reap_sends(tf_rdma *r, int block)
{
    struct ibv_wc wc[POLL_BATCH];
    for (;;) {
        int n = ibv_poll_cq(r->send_cq, POLL_BATCH, wc);
        if (n < 0) return -EIO;
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "rdma: send %llu failed: %s\n", (unsigned long long)wc[i].wr_id,
                        ibv_wc_status_str(wc[i].status));
                r->failed = 1;
                return -EIO;
            }
            r->completed = wc[i].wr_id + 1;
        }
        if (n > 0 || !block) return 0;
    }
}

static int post_write(tf_rdma *r, struct ibv_sge *sge, uint64_t remote_addr, uint32_t imm)
{
    if (r->failed) return -EIO;
    while (r->posted - r->completed >= r->sq_depth) {
        int rc = reap_sends(r, 1);
        if (rc) return rc;
    }
    struct ibv_send_wr wr = {
        .wr_id = r->posted,
        .sg_list = sge,
        .num_sge = sge ? 1 : 0,
        .opcode = IBV_WR_RDMA_WRITE_WITH_IMM,
        .send_flags = IBV_SEND_SIGNALED,
        .imm_data = htonl(imm),
    };
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = r->remote_rkey;
    struct ibv_send_wr *bad;
    int rc = ibv_post_send(r->qp, &wr, &bad);
    if (rc) return -rc;
    r->posted++;
    return 0;
}

int tf_rdma_open(tf_rdma **out, const char *device, int gid_index, void *ring,
                 uint32_t slots, size_t packet_bytes, char *err, size_t err_len)
{
    *out = NULL;
    if (!ring || slots == 0 || packet_bytes == 0) return -EINVAL;
    tf_rdma *r = calloc(1, sizeof *r);
    if (!r) return -ENOMEM;
    r->ring = ring;
    r->slots = slots;
    r->packet_bytes = packet_bytes;
    int rc = -ENODEV;

    int n;
    struct ibv_device **list = ibv_get_device_list(&n);
    if (!list) {
        set_err(err, err_len, "%s: errno %d", "ibv_get_device_list", errno);
        goto fail;
    }
    for (int i = 0; i < n && !r->ctx; i++) {
        if (!device || strcmp(ibv_get_device_name(list[i]), device) == 0) r->ctx = ibv_open_device(list[i]);
    }
    ibv_free_device_list(list);
    if (!r->ctx) {
        set_err(err, err_len, "%s: no such RDMA device (or open failed, %d)", device ? device : "(first)", errno);
        goto fail;
    }

    struct ibv_port_attr port;
    struct ibv_device_attr dev;
    if (ibv_query_port(r->ctx, PORT, &port) || ibv_query_device(r->ctx, &dev)) {
        set_err(err, err_len, "%s: query failed, errno %d", "ibv_query_port", errno);
        goto fail;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        set_err(err, err_len, "%s: port 1 is not ACTIVE (state %d)", ibv_get_device_name(r->ctx->device), port.state);
        goto fail;
    }
    r->mtu = port.active_mtu;
    r->gid_index = gid_index >= 0 ? gid_index : find_roce_v2_ipv4_gid(r->ctx, port.gid_tbl_len);
    if (r->gid_index < 0 || ibv_query_gid(r->ctx, PORT, r->gid_index, &r->gid)) {
        set_err(err, err_len, "%s: no usable RoCE v2 IPv4 GID (index %d)", ibv_get_device_name(r->ctx->device), r->gid_index);
        goto fail;
    }

    uint32_t depth = 2 * slots + 16;
    if (depth > (uint32_t)dev.max_qp_wr) depth = (uint32_t)dev.max_qp_wr;
    r->sq_depth = r->rq_depth = depth;

    rc = -ENOMEM;
    if (!(r->pd = ibv_alloc_pd(r->ctx))) {
        set_err(err, err_len, "%s: errno %d", "ibv_alloc_pd", errno);
        goto fail;
    }
    size_t ring_bytes = (size_t)slots * packet_bytes;
    r->ring_mr = ibv_reg_mr(r->pd, ring, ring_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (posix_memalign((void **)&r->stage, 4096, ring_bytes) != 0) r->stage = NULL;
    r->stage_done = calloc(slots, sizeof *r->stage_done);
    if (r->stage) r->stage_mr = ibv_reg_mr(r->pd, r->stage, ring_bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!r->ring_mr || !r->stage_mr || !r->stage_done) {
        set_err(err, err_len, "%s: errno %d (check ulimit -l)", "ibv_reg_mr", errno);
        goto fail;
    }
    r->send_cq = ibv_create_cq(r->ctx, (int)r->sq_depth, NULL, NULL, 0);
    r->recv_cq = ibv_create_cq(r->ctx, (int)r->rq_depth, NULL, NULL, 0);
    if (!r->send_cq || !r->recv_cq) {
        set_err(err, err_len, "%s: errno %d", "ibv_create_cq", errno);
        goto fail;
    }
    struct ibv_qp_init_attr qa = {
        .send_cq = r->send_cq,
        .recv_cq = r->recv_cq,
        .cap = {.max_send_wr = r->sq_depth, .max_recv_wr = r->rq_depth, .max_send_sge = 1, .max_recv_sge = 1},
        .qp_type = IBV_QPT_RC,
        .sq_sig_all = 1,
    };
    if (!(r->qp = ibv_create_qp(r->pd, &qa))) {
        set_err(err, err_len, "%s: errno %d", "ibv_create_qp", errno);
        goto fail;
    }

    struct ibv_qp_attr a = {.qp_state = IBV_QPS_INIT, .pkey_index = 0, .port_num = PORT,
                            .qp_access_flags = IBV_ACCESS_REMOTE_WRITE};
    if ((rc = -ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))) {
        set_err(err, err_len, "%s: %d", "modify_qp(INIT)", -rc);
        goto fail;
    }
    if ((rc = post_recvs(r, r->rq_depth))) {
        set_err(err, err_len, "%s: %d", "ibv_post_recv", -rc);
        goto fail;
    }
    srand48((long)time(NULL) ^ (long)getpid());
    r->psn = (uint32_t)lrand48() & 0xffffff;
    *out = r;
    return 0;

fail:
    tf_rdma_close(r);
    return rc < 0 ? rc : -EIO;
}

void tf_rdma_local_info(const tf_rdma *r, tf_rdma_conn_info *info)
{
    memset(info, 0, sizeof *info);
    info->qpn = r->qp->qp_num;
    info->psn = r->psn;
    memcpy(info->gid, r->gid.raw, 16);
    info->ring_addr = (uint64_t)(uintptr_t)r->ring;
    info->ring_rkey = r->ring_mr->rkey;
    info->slots = r->slots;
}

int tf_rdma_connect(tf_rdma *r, const tf_rdma_conn_info *peer, char *err, size_t err_len)
{
    if (peer->slots != r->slots) {
        set_err(err, err_len, "%s: peer has %d slots", "tf_rdma_connect", (int)peer->slots);
        return -EINVAL;
    }
    r->remote_addr = peer->ring_addr;
    r->remote_rkey = peer->ring_rkey;

    struct ibv_qp_attr a = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = r->mtu,
        .dest_qp_num = peer->qpn,
        .rq_psn = peer->psn,
        .max_dest_rd_atomic = 1,
        .min_rnr_timer = 12,
        .ah_attr = {.is_global = 1, .port_num = PORT,
                    .grh = {.sgid_index = (uint8_t)r->gid_index, .hop_limit = 64}},
    };
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid, 16);
    int rc = ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) {
        set_err(err, err_len, "%s: %d", "modify_qp(RTR)", rc);
        return -rc;
    }
    memset(&a, 0, sizeof a);
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;   /* 4.096 us * 2^14, about 67 ms per retry */
    a.retry_cnt = 7;
    a.rnr_retry = 7;  /* retry forever if the peer is briefly out of receives */
    a.sq_psn = r->psn;
    a.max_rd_atomic = 1;
    rc = ibv_modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                      IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) {
        set_err(err, err_len, "%s: %d", "modify_qp(RTS)", rc);
        return -rc;
    }
    return 0;
}

int tf_rdma_write_imm(tf_rdma *r, uint32_t slot, const void *pkt, uint32_t imm)
{
    if (slot >= r->slots) return -EINVAL;
    uint32_t s = r->stage_next;
    while (r->completed < r->stage_done[s]) {
        r->stats.staging_waits++;
        int rc = reap_sends(r, 1);
        if (rc) return rc;
    }
    uint8_t *buf = r->stage + (size_t)s * r->packet_bytes;
    memcpy(buf, pkt, r->packet_bytes);
    struct ibv_sge sge = {.addr = (uintptr_t)buf, .length = (uint32_t)r->packet_bytes, .lkey = r->stage_mr->lkey};
    int rc = post_write(r, &sge, r->remote_addr + (uint64_t)slot * r->packet_bytes, imm);
    if (rc) return rc;
    r->stage_done[s] = r->posted; /* reusable once this send has completed */
    r->stage_next = (s + 1) % r->slots;
    r->stats.writes++;
    return 0;
}

int tf_rdma_imm(tf_rdma *r, uint32_t imm)
{
    int rc = post_write(r, NULL, r->remote_addr, imm);
    if (rc == 0) r->stats.imms++;
    return rc;
}

int tf_rdma_poll(tf_rdma *r, uint32_t *out, int max)
{
    if (r->failed) return -EIO;
    int rc = reap_sends(r, 0);
    if (rc) return rc;
    struct ibv_wc wc[POLL_BATCH];
    int n = ibv_poll_cq(r->recv_cq, max < POLL_BATCH ? max : POLL_BATCH, wc);
    if (n < 0) return -EIO;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS || !(wc[i].wc_flags & IBV_WC_WITH_IMM)) {
            fprintf(stderr, "rdma: receive failed: %s\n", ibv_wc_status_str(wc[i].status));
            r->failed = 1;
            return -EIO;
        }
        out[i] = ntohl(wc[i].imm_data);
    }
    if (n > 0 && (rc = post_recvs(r, (uint32_t)n))) return rc;
    return n;
}

int tf_rdma_drain(tf_rdma *r, int timeout_ms)
{
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (r->completed < r->posted) {
        int rc = reap_sends(r, 0);
        if (rc) return rc;
        clock_gettime(CLOCK_MONOTONIC, &t);
        if ((t.tv_sec - t0.tv_sec) * 1000 + (t.tv_nsec - t0.tv_nsec) / 1000000 > timeout_ms) return -ETIMEDOUT;
    }
    return 0;
}

void tf_rdma_get_stats(const tf_rdma *r, tf_rdma_stats *stats)
{
    *stats = r->stats;
}

void tf_rdma_close(tf_rdma *r)
{
    if (!r) return;
    if (r->qp) ibv_destroy_qp(r->qp);
    if (r->send_cq) ibv_destroy_cq(r->send_cq);
    if (r->recv_cq) ibv_destroy_cq(r->recv_cq);
    if (r->ring_mr) ibv_dereg_mr(r->ring_mr);
    if (r->stage_mr) ibv_dereg_mr(r->stage_mr);
    if (r->pd) ibv_dealloc_pd(r->pd);
    if (r->ctx) ibv_close_device(r->ctx);
    free(r->stage);
    free(r->stage_done);
    free(r);
}
