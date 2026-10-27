#include "tf_wire.h"

size_t tf_wire_layout(size_t *out, size_t n)
{
    const size_t layout[] = {
        sizeof(tf_packet),
        offsetof(tf_header, generation),
        offsetof(tf_header, src_rank),
        offsetof(tf_header, gate_weight),
        offsetof(tf_header, payload_len),
        offsetof(tf_packet, payload),
        offsetof(tf_packet, scales),
        offsetof(tf_packet, tail),
    };
    const size_t count = sizeof(layout) / sizeof(layout[0]);
    for (size_t i = 0; i < n && i < count; i++)
        out[i] = layout[i];
    return count;
}
