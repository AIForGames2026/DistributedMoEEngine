/* RoCE v2 RDMA link (TASK-07): one Reliable Connection queue pair per peer,
 * carrying the transport's one-sided model directly. A data write is an
 * IBV_WR_RDMA_WRITE_WITH_IMM of one packet into a slot of the peer's
 * registered receive ring; credits and control are zero-length writes with an
 * immediate. The receiver learns of each write from the immediate in its
 * receive completion, after the data has landed. */
#ifndef TF_RDMA_H
#define TF_RDMA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tf_rdma tf_rdma;

/* What each side sends the other before connecting (out of band, over TCP).
 * Fixed layout, little-endian hosts only, so it can be sent as raw bytes. */
typedef struct tf_rdma_conn_info {
    uint32_t qpn;
    uint32_t psn;
    uint8_t gid[16];
    uint64_t ring_addr; /* remote address of the receive ring */
    uint32_t ring_rkey;
    uint32_t slots;
} tf_rdma_conn_info;

typedef struct tf_rdma_stats {
    uint64_t writes;        /* data writes posted */
    uint64_t imms;          /* zero-length immediates posted */
    uint64_t staging_waits; /* times a send waited for a staging buffer */
} tf_rdma_stats;

/* Opens `device` (NULL: the first device) port 1 and registers `ring`
 * (`slots` packets of `packet_bytes`, page aligned) for remote writes.
 * `gid_index` < 0 picks the first RoCE v2 GID with an IPv4 address.
 * Returns 0 or a negative errno; on failure, a message is in `err`. */
int tf_rdma_open(tf_rdma **out, const char *device, int gid_index, void *ring,
                 uint32_t slots, size_t packet_bytes, char *err, size_t err_len);

void tf_rdma_local_info(const tf_rdma *r, tf_rdma_conn_info *info);

/* Moves the queue pair to ready-to-send against `peer`. */
int tf_rdma_connect(tf_rdma *r, const tf_rdma_conn_info *peer, char *err, size_t err_len);

/* Copies `pkt` into a registered staging buffer and writes it to slot `slot`
 * of the peer's ring with immediate `imm`. A staging buffer is reused only
 * after the send that used it has completed. Returns 0 or a negative errno. */
int tf_rdma_write_imm(tf_rdma *r, uint32_t slot, const void *pkt, uint32_t imm);

/* Delivers `imm` with no data. */
int tf_rdma_imm(tf_rdma *r, uint32_t imm);

/* Non-blocking. Writes up to `max` arrived immediates to `out`, in order,
 * reposting a receive for each. Returns the count or a negative errno. */
int tf_rdma_poll(tf_rdma *r, uint32_t *out, int max);

/* Waits (up to `timeout_ms`) until every posted send has completed. */
int tf_rdma_drain(tf_rdma *r, int timeout_ms);

void tf_rdma_get_stats(const tf_rdma *r, tf_rdma_stats *stats);

void tf_rdma_close(tf_rdma *r);

#ifdef __cplusplus
}
#endif

#endif /* TF_RDMA_H */
