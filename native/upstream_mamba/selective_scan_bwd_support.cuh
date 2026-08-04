// Copyright (c) 2023 Tri Dao
// Modifications Copyright 2026 Raamesh
//
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <thrust/complex.h>

#define MAX_DSTATE 256

using complex_t = thrust::complex<float>;

constexpr size_t custom_max(std::initializer_list<size_t> values) {
  return std::max(values);
}

template <typename T>
constexpr T constexpr_min(T a, T b) {
  return std::min(a, b);
}

inline __device__ float2 operator+(const float2& a, const float2& b) {
  return {a.x + b.x, a.y + b.y};
}

inline __device__ float3 operator+(const float3& a, const float3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ float4 operator+(const float4& a, const float4& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

template <int Bytes>
struct BytesToType;

template <>
struct BytesToType<16> {
  using Type = uint4;
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
};

template <typename Scalar, int N>
struct Converter {
  static inline __device__ void to_float(
      const Scalar (&source)[N], float (&destination)[N]) {
#pragma unroll
    for (int item = 0; item < N; ++item) {
      destination[item] = source[item];
    }
  }
};

__device__ __forceinline__ complex_t cexp2f(complex_t value) {
  const float magnitude = exp2f(value.real());
  float cosine;
  float sine;
  sincosf(value.imag(), &sine, &cosine);
  return complex_t(cosine * magnitude, sine * magnitude);
}

template <typename Scalar>
struct SSMScanOp;

template <>
struct SSMScanOp<float> {
  __device__ __forceinline__ float2 operator()(
      const float2& ab0, const float2& ab1) const {
    return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
  }
};

template <>
struct SSMScanOp<complex_t> {
  __device__ __forceinline__ float4 operator()(
      const float4& ab0, const float4& ab1) const {
    const complex_t a0(ab0.x, ab0.y);
    const complex_t b0(ab0.z, ab0.w);
    const complex_t a1(ab1.x, ab1.y);
    const complex_t b1(ab1.z, ab1.w);
    const complex_t out_a = a1 * a0;
    const complex_t out_b = a1 * b0 + b1;
    return make_float4(
        out_a.real(), out_a.imag(), out_b.real(), out_b.imag());
  }
};

template <typename Scalar>
struct SSMScanPrefixCallbackOp {
  using scan_t =
      std::conditional_t<std::is_same_v<Scalar, float>, float2, float4>;
  scan_t running_prefix;

  __device__ explicit SSMScanPrefixCallbackOp(scan_t prefix)
      : running_prefix(prefix) {}

  __device__ scan_t operator()(scan_t block_aggregate) {
    const scan_t old_prefix = running_prefix;
    running_prefix =
        SSMScanOp<Scalar>()(running_prefix, block_aggregate);
    return old_prefix;
  }
};

template <typename Ktraits>
inline __device__ void load_input(
    typename Ktraits::input_t* input,
    typename Ktraits::input_t (&values)[Ktraits::kNItems],
    typename Ktraits::BlockLoadT::TempStorage& storage,
    int valid_items) {
  if constexpr (Ktraits::kIsEvenLen) {
    auto& vector_storage =
        reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(
            storage);
    typename Ktraits::BlockLoadVecT(vector_storage).Load(
        reinterpret_cast<typename Ktraits::vec_t*>(input),
        reinterpret_cast<
            typename Ktraits::vec_t (&)[Ktraits::kNLoads]>(values));
  } else {
    typename Ktraits::BlockLoadT(storage).Load(
        input, values, valid_items, 0.0f);
  }
}

template <typename Ktraits>
inline __device__ void load_weight(
    typename Ktraits::input_t* input,
    typename Ktraits::weight_t (&values)[Ktraits::kNItems],
    typename Ktraits::BlockLoadWeightT::TempStorage& storage,
    int valid_items) {
  constexpr int kNItems = Ktraits::kNItems;
  if constexpr (!Ktraits::kIsComplex) {
    typename Ktraits::input_t loaded[kNItems];
    if constexpr (Ktraits::kIsEvenLen) {
      auto& vector_storage = reinterpret_cast<
          typename Ktraits::BlockLoadWeightVecT::TempStorage&>(storage);
      typename Ktraits::BlockLoadWeightVecT(vector_storage).Load(
          reinterpret_cast<typename Ktraits::vec_t*>(input),
          reinterpret_cast<
              typename Ktraits::vec_t (&)[Ktraits::kNLoads]>(loaded));
    } else {
      typename Ktraits::BlockLoadWeightT(storage).Load(
          input, loaded, valid_items, 0.0f);
    }
    Converter<typename Ktraits::input_t, kNItems>::to_float(
        loaded, values);
  } else {
    typename Ktraits::input_t loaded[kNItems * 2];
    typename Ktraits::BlockLoadWeightT(storage).Load(
        input, loaded, valid_items, 0.0f);
#pragma unroll
    for (int item = 0; item < kNItems; ++item) {
      values[item] = complex_t(loaded[2 * item], loaded[2 * item + 1]);
    }
  }
}

template <typename Ktraits>
inline __device__ void store_output(
    typename Ktraits::input_t* output,
    const float (&values)[Ktraits::kNItems],
    typename Ktraits::BlockStoreT::TempStorage& storage,
    int valid_items) {
  typename Ktraits::input_t write_values[Ktraits::kNItems];
#pragma unroll
  for (int item = 0; item < Ktraits::kNItems; ++item) {
    write_values[item] = values[item];
  }
  if constexpr (Ktraits::kIsEvenLen) {
    auto& vector_storage =
        reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(
            storage);
    typename Ktraits::BlockStoreVecT(vector_storage).Store(
        reinterpret_cast<typename Ktraits::vec_t*>(output),
        reinterpret_cast<
            typename Ktraits::vec_t (&)[Ktraits::kNLoads]>(write_values));
  } else {
    typename Ktraits::BlockStoreT(storage).Store(
        output, write_values, valid_items);
  }
}

inline __device__ void gpuAtomicAdd(float* address, float value) {
  atomicAdd(address, value);
}

inline __device__ void gpuAtomicAdd(complex_t* address, complex_t value) {
  float* components = reinterpret_cast<float*>(address);
  atomicAdd(components, value.real());
  atomicAdd(components + 1, value.imag());
}

struct SSMParamsBwd {
  using index_t = uint64_t;

  int batch;
  int dim;
  int seqlen;
  int dstate;
  int n_groups;
  int n_chunks;
  int dim_ngroups_ratio;
  bool is_variable_B;
  bool is_variable_C;
  bool delta_softplus;

  index_t A_d_stride;
  index_t A_dstate_stride;
  index_t B_batch_stride;
  index_t B_d_stride;
  index_t B_dstate_stride;
  index_t B_group_stride;
  index_t C_batch_stride;
  index_t C_d_stride;
  index_t C_dstate_stride;
  index_t C_group_stride;
  index_t u_batch_stride;
  index_t u_d_stride;
  index_t delta_batch_stride;
  index_t delta_d_stride;
  index_t z_batch_stride;
  index_t z_d_stride;
  index_t out_batch_stride;
  index_t out_d_stride;
  index_t out_z_batch_stride;
  index_t out_z_d_stride;

  index_t dout_batch_stride;
  index_t dout_d_stride;
  index_t dA_d_stride;
  index_t dA_dstate_stride;
  index_t dB_batch_stride;
  index_t dB_group_stride;
  index_t dB_d_stride;
  index_t dB_dstate_stride;
  index_t dC_batch_stride;
  index_t dC_group_stride;
  index_t dC_d_stride;
  index_t dC_dstate_stride;
  index_t du_batch_stride;
  index_t du_d_stride;
  index_t dz_batch_stride;
  index_t dz_d_stride;
  index_t ddelta_batch_stride;
  index_t ddelta_d_stride;

  void* A_ptr;
  void* B_ptr;
  void* C_ptr;
  void* D_ptr;
  void* u_ptr;
  void* delta_ptr;
  void* delta_bias_ptr;
  void* out_ptr;
  void* x_ptr;
  void* z_ptr;
  void* out_z_ptr;
  void* initial_x_ptr;
  void* dfinal_x_ptr;

  void* dout_ptr;
  void* dA_ptr;
  void* dB_ptr;
  void* dC_ptr;
  void* dD_ptr;
  void* du_ptr;
  void* dz_ptr;
  void* ddelta_ptr;
  void* ddelta_bias_ptr;
  void* dinitial_x_ptr;
};
