/* CRC32C (Castagnoli) for packet checks. Uses the ARMv8 CRC instructions on
 * the Sparks and SSE4.2 on x86; Zig's std only accelerates x86, and the table
 * version costs about 10x more per packet on GB10. */
#include <string.h>

#include "tf_wire.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define CRC_U64(crc, v) __crc32cd((crc), (v))
#define CRC_U8(crc, b) __crc32cb((crc), (b))
#elif defined(__x86_64__) && defined(__SSE4_2__)
#include <nmmintrin.h>
#define CRC_U64(crc, v) ((uint32_t)_mm_crc32_u64((crc), (v)))
#define CRC_U8(crc, b) _mm_crc32_u8((crc), (b))
#endif

#ifdef CRC_U64
uint32_t tf_crc32c(const void *data, size_t len)
{
    const uint8_t *p = data;
    uint32_t crc = 0xFFFFFFFFu;
    for (; len >= 8; p += 8, len -= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        crc = CRC_U64(crc, v);
    }
    while (len--)
        crc = CRC_U8(crc, *p++);
    return ~crc;
}
#else
/* Portable fallback, bit by bit. Only used for baseline cross-compiles. */
uint32_t tf_crc32c(const void *data, size_t len)
{
    const uint8_t *p = data;
    uint32_t crc = 0xFFFFFFFFu;
    while (len--) {
        crc ^= *p++;
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0x82F63B78u & (0u - (crc & 1u)));
    }
    return ~crc;
}
#endif
