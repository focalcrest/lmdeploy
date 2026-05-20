#
# TurboMind SM70 GEMM kernel static library
#
# Builds the TurboMind SM70 (V100) AWQ GEMM kernels as a static library.
# Designed to be included by downstream projects after setting
# VLLM_TURBOMIND_ROOT to the lmdeploy source directory.
#
# Required variables:
#   VLLM_TURBOMIND_ROOT - Path to lmdeploy source root
#
# Provided targets:
#   turbomind_sm70_gemm - Static library with SM70 GEMM kernels
#

set(TURBOMIND_SM70_GEMM_SRCS
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/check.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/layout.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/context.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/allocator.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/buffer.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/core/stream.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/utils/logger.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/utils/parser.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/gemm.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/kernel.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/dispatch_cache.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/context.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/convert_v3.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/cast.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/unpack.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/tuner/cache_utils.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/tuner/measurer.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/tuner/sampler.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/tuner/stopping_criterion.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/tuner/params.cc
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu
  ${VLLM_TURBOMIND_ROOT}/src/turbomind/kernels/gemm/kernel/sm70_884_16.cu
)

add_library(turbomind_sm70_gemm STATIC ${TURBOMIND_SM70_GEMM_SRCS})

target_include_directories(turbomind_sm70_gemm PUBLIC ${VLLM_TURBOMIND_ROOT})

target_compile_features(turbomind_sm70_gemm PUBLIC cxx_std_17)

set_target_properties(turbomind_sm70_gemm PROPERTIES
  CUDA_ARCHITECTURES "70-real"
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(turbomind_sm70_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>
)
