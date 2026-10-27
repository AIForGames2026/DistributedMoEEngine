/* TensorFold-Swarm wire format v0 (docs/wire-format.md). Mirrors src/wire.zig;
 * both sides assert the same sizes and offsets at compile time. */
#ifndef TF_WIRE_H
#define TF_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TF_WIRE_VERSION 0
#define TF_HIDDEN_BYTES 7168
#define TF_SCALE_BYTES (TF_HIDDEN_BYTES / 32)
#define TF_PACKET_BYTES 7552

enum tf_kind { TF_KIND_DISPATCH = 0, TF_KIND_COMBINE = 1, TF_KIND_CREDIT = 2, TF_KIND_CONTROL = 3 };

typedef struct tf_header {
    uint64_t token_id;
    uint64_t generation; /* full slot generation, starts at 1 */
    uint16_t src_rank;
    uint16_t layer;
    uint16_t expert;
    uint16_t flags;
    float gate_weight;
    uint32_t payload_len;
} tf_header;

typedef struct tf_tail {
    uint32_t crc32c;        /* CRC32C over every byte before the tail */
    uint32_t generation_lo; /* low 32 bits of header.generation, written last */
    uint8_t reserved[8];
} tf_tail;

typedef struct tf_packet {
    tf_header header;
    uint8_t payload[TF_HIDDEN_BYTES]; /* fp8_e4m3 hidden state */
    uint8_t scales[TF_SCALE_BYTES];   /* e8m0, one per 32-element block */
    uint8_t pad[112];
    tf_tail tail;
} tf_packet;

/* 32-bit immediate: slot[0:15] kind[16:17] gen8[18:25] src_rank[26:31]. */
#define TF_IMM(slot, kind, gen8, src) \
    (((uint32_t)(slot) & 0xFFFFu) | (((uint32_t)(kind) & 0x3u) << 16) | \
     (((uint32_t)(gen8) & 0xFFu) << 18) | (((uint32_t)(src) & 0x3Fu) << 26))
#define TF_IMM_SLOT(imm) ((imm) & 0xFFFFu)
#define TF_IMM_KIND(imm) (((imm) >> 16) & 0x3u)
#define TF_IMM_GEN8(imm) (((imm) >> 18) & 0xFFu)
#define TF_IMM_SRC(imm) (((imm) >> 26) & 0x3Fu)

_Static_assert(sizeof(tf_header) == 32, "tf_header size");
_Static_assert(sizeof(tf_tail) == 16, "tf_tail size");
_Static_assert(sizeof(tf_packet) == TF_PACKET_BYTES, "tf_packet size");
_Static_assert(offsetof(tf_header, generation) == 0x08, "generation offset");
_Static_assert(offsetof(tf_header, src_rank) == 0x10, "src_rank offset");
_Static_assert(offsetof(tf_header, gate_weight) == 0x18, "gate_weight offset");
_Static_assert(offsetof(tf_header, payload_len) == 0x1C, "payload_len offset");
_Static_assert(offsetof(tf_packet, payload) == 0x0020, "payload offset");
_Static_assert(offsetof(tf_packet, scales) == 0x1C20, "scales offset");
_Static_assert(offsetof(tf_packet, pad) == 0x1D00, "pad offset");
_Static_assert(offsetof(tf_packet, tail) == 0x1D70, "tail offset");

/* CRC32C (Castagnoli) of `len` bytes, hardware accelerated where available.
 * tf_crc32c("123456789", 9) == 0xE3069283. */
uint32_t tf_crc32c(const void *data, size_t len);

/* Writes up to n layout values (size, then field offsets, in the order of
 * `layout` in src/wire.zig) and returns how many exist. */
size_t tf_wire_layout(size_t *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* TF_WIRE_H */
