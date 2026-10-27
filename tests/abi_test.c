/* Plain C consumer of the public ABI (TASK-01): proves tensorfold.h and
 * tf_wire.h work from C, without any Zig on the calling side. */
#include <stdio.h>

#include "tensorfold.h"
#include "tf_wire.h"

static int failures;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                \
        }                                                              \
    } while (0)

int main(void)
{
    CHECK(tf_version() == ((TF_VERSION_MAJOR << 16) | (TF_VERSION_MINOR << 8) | TF_VERSION_PATCH));

    size_t layout[16];
    CHECK(tf_wire_layout(layout, 16) == 8);
    CHECK(layout[0] == TF_PACKET_BYTES);

    CHECK(tf_crc32c("123456789", 9) == 0xE3069283u);

    unsigned imm = TF_IMM(513, TF_KIND_COMBINE, 0xAB, 5);
    CHECK(imm == (513u | (1u << 16) | (0xABu << 18) | (5u << 26)));
    CHECK(TF_IMM_SLOT(imm) == 513);
    CHECK(TF_IMM_KIND(imm) == TF_KIND_COMBINE);
    CHECK(TF_IMM_GEN8(imm) == 0xAB);
    CHECK(TF_IMM_SRC(imm) == 5);

    if (failures == 0)
        printf("abi_test: ok\n");
    return failures != 0;
}
