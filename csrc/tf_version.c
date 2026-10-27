#include "tensorfold.h"

unsigned tf_version(void)
{
    return (TF_VERSION_MAJOR << 16) | (TF_VERSION_MINOR << 8) | TF_VERSION_PATCH;
}
