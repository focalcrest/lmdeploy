// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm70_884_u8()
{
    if constexpr (1) {
        // clang-format off
        // group_axis=0 (Blocked/Indexed striding) — original configs
        using C = Config_U8<kColMajor, 0>;
        Add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        // group_axis=-1 (Flat striding) — better for dense rectangular matrices in decode.
        // K=128 aligns exactly with group_size=128 (one scale per K-tile).
        // N=256 provides wider tiles for better SM utilization on large projections.
        using CF = Config_U8<kColMajor>;
        Add<CF::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type<  8, 128, 128, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type<  8, 256, 128, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 16, 128, 128, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 16, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 32, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        Add<CF::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        Add<CF::Type< 64, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>>();
        Add<CF::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        Add<CF::Type<128, 256,  16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>>();
        // clang-format on
    }
}

}  // namespace turbomind::gemm
