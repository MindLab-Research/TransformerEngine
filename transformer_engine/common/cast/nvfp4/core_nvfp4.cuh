/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file core_nvfp4.cuh
 *  \brief Core functions used in NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_CORE_NVFP4_CUH_

#if !defined(__CUDACC_RTC__)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <limits>

#include "../../common.h"
#include "../../util/curanddx.hpp"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#else
// NVRTC build: common.h (host-only: cuDNN/cutlass) cannot be parsed by NVRTC.
// utils.cuh, util/math.h and ptx.cuh are injected as in-memory headers by the
// RTC dispatch and already provide the integer typedefs, detail::is_same, and
// the fp8 element types. util/type_extrema.h (also injected) provides the
// transformer_engine-namespace fp4 aliases and detail::TypeExtrema
// specializations that would otherwise come from common.h.
#include "ptx.cuh"
#include "util/math.h"
#include "utils.cuh"
#endif  // __CUDACC_RTC__

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

#if defined(__CUDACC_RTC__)
namespace transformer_engine {
#include "util/type_extrema.h"
}  // namespace transformer_engine
#endif  // __CUDACC_RTC__

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

// Central runtime-to-compile-time dispatch for NVFP4 scale storage types.
// SWITCH_FP8UE5M3_TYPE_HANDLE adds UE5M3 when the CUDA toolkit supports it.
#define TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(SCALE_DTYPE, SCALE_TYPE, ...)          \
  switch (SCALE_DTYPE) {                                                                  \
    case DType::kFloat8E8M0: {  /* dsv41-mxfp4-e8m0-block32 */                                          \
      using SCALE_TYPE = fp8e8m0;                                                       \
      { __VA_ARGS__ }                                                                   \
    } break;                                                                            \
    case DType::kFloat8E4M3: {                                                          \
      using SCALE_TYPE = fp8e4m3;                                                         \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
      SWITCH_FP8UE5M3_TYPE_HANDLE(SCALE_TYPE, __VA_ARGS__)                                \
    default: {                                                                            \
      NVTE_ERROR("Unsupported NVFP4 scale dtype ", to_string(SCALE_DTYPE),                \
                 ". Expected Float8E4M3, or Float8UE5M3 when compiled with CUDA 13.4+."); \
    }                                                                                     \
  }

namespace core {

#if FP4_TYPE_SUPPORTED
using namespace ptx;

template <typename ScaleType>
__device__ __forceinline__ ScaleType
compute_decoding_scaling_factor(const float block_amax, const float global_encode_scale) {
  // Compute the per-block decode scale in the selected scale storage type:
  //
  //   block_decode_scale = block_amax / fp4_max
  //   stored_decode_scale = block_decode_scale * global_encode_scale
  //
  // An equivalent, more literal implementation is:
  //
  //   constexpr float rcp_6f = 1.0f / 6.0f;
  //   const float block_decode_scale = block_amax * rcp_6f;
  //   return static_cast<ScaleType>(block_decode_scale * global_encode_scale);
  //
  // Keep the multiplication order below to match the emulation code exactly,
  // while avoiding a direct division by the FP4 maximum.
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp4_max_inv = 1.0f / fp4_max;
  const float decode_scale = block_amax * (global_encode_scale * fp4_max_inv);
  if constexpr (std::is_same_v<ScaleType, fp8e8m0>) {
    // dsv41-mxfp4-e8m0-block32: MXFP4's block scale covers the block -- round UP to a power of two.
    // The reference implementation matches 2^ceil(log2(amax/6)) on 100% of blocks
    // (measured, 1024 blocks). Computed from the bit pattern because exp2f/ceilf/log2f
    // under this build's fast-math stored byte 0 for every block.
    const float _s = fminf(decode_scale, TypeExtrema<float>::max);
    if (_s <= 0.0f) {
      return static_cast<ScaleType>(_s);
    }
    const unsigned _b = __float_as_uint(_s);
    int _e = static_cast<int>((_b >> 23) & 0xFFu) - 127;
    const bool _exact = ((_b & 0x7FFFFFu) == 0u);
    if (!_exact) {
      ++_e;  // 2^ceil: step up unless the value is already a power of two
    }
    const unsigned char _byte = static_cast<unsigned char>(_e + 127);
    ScaleType _out;
    memcpy(&_out, &_byte, sizeof(_byte));
    return _out;
  }
  return static_cast<ScaleType>(fminf(decode_scale, TypeExtrema<float>::max));
}

// Compute the global encode scale factor for a given global amax.
// NVFP4 uses the full scale-type range by default. The explicit SCALE_MAX
// template argument lets recipes such as 4over6 reserve encoding headroom.
// dsv41-mxfp4-e8m0-block32: sentinel default; a float template parameter is not allowed in C++17.
template <typename ScaleType, int SCALE_MAX = -1>
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  static_assert(SCALE_MAX != 0, "FP4 scale maximum must be non-zero.");
  constexpr float fp8_max = (SCALE_MAX > 0) ? static_cast<float>(SCALE_MAX)
                                            : detail::TypeExtrema<ScaleType>::max;
  if constexpr (std::is_same_v<ScaleType, fp8e8m0>) {
    // MXFP4 has no tensor-level global scale: the block scale is already an
    // absolute power of two. Returning anything else made the global factor inf
    // and saturated the stored exponent (measured relerr=inf).
    return 1.0f;
  }
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

#if !defined(__CUDACC_RTC__)
// The RTC 4over6 path rejects stochastic rounding; static transpose kernels use this helper.
__device__ __forceinline__ uint32_t get_rbits(
    transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>
        &rng,
    // philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>: compile-time configurable rounds
    uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    random_uint4 = rng.generate4();
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}
#endif  // !__CUDACC_RTC__

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
