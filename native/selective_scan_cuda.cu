// Copyright (c) 2023 Tri Dao
// Modifications Copyright 2026 Raamesh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// This file ports the float32 forward path from Mamba v2.3.2's
// selective_scan_fwd_kernel.cuh to JAX's typed XLA FFI. The CUB I/O and scan
// structure, launch tuning, vectorization, and occupancy hints are retained.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kMaxDState = 256;

constexpr size_t CustomMax(std::initializer_list<size_t> values) {
  return std::max(values);
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

struct SSMScanOp {
  __device__ __forceinline__ float2 operator()(
      const float2& ab0, const float2& ab1) const {
    return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
  }
};

struct SSMScanPrefixCallbackOp {
  float2 running_prefix;

  __device__ explicit SSMScanPrefixCallbackOp(float2 running_prefix_)
      : running_prefix(running_prefix_) {}

  __device__ float2 operator()(float2 block_aggregate) {
    const float2 old_prefix = running_prefix;
    running_prefix = SSMScanOp{}(running_prefix, block_aggregate);
    return old_prefix;
  }
};

template <int NThreads, int NItems, bool IsEvenLen>
struct SelectiveScanFwdKernelTraits {
  static_assert(NItems % 4 == 0);
  static constexpr int kNThreads = NThreads;
  static constexpr int kNItems = NItems;
  static constexpr int kMinBlocks = NThreads < 128 ? 5 : 3;
  static constexpr int kNElts = 4;
  static constexpr int kNLoads = NItems / kNElts;
  static constexpr bool kIsEvenLen = IsEvenLen;
  static constexpr bool kDirectIO = IsEvenLen && kNLoads == 1;

  using vec_t = typename BytesToType<sizeof(float) * kNElts>::Type;
  using BlockLoadT = cub::BlockLoad<
      float, NThreads, NItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadVecT = cub::BlockLoad<
      vec_t,
      NThreads,
      kNLoads,
      kDirectIO ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadWeightT = cub::BlockLoad<
      float, NThreads, NItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadWeightVecT = cub::BlockLoad<
      vec_t,
      NThreads,
      kNLoads,
      kDirectIO ? cub::BLOCK_LOAD_DIRECT : cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<
      float, NThreads, NItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreVecT = cub::BlockStore<
      vec_t,
      NThreads,
      kNLoads,
      kDirectIO ? cub::BLOCK_STORE_DIRECT : cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScanT =
      cub::BlockScan<float2, NThreads, cub::BLOCK_SCAN_WARP_SCANS>;

  static constexpr int kSmemIOSize = CustomMax(
      {sizeof(typename BlockLoadT::TempStorage),
       2 * sizeof(typename BlockLoadWeightT::TempStorage),
       sizeof(typename BlockStoreT::TempStorage)});
  static constexpr int kSmemSize =
      kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template <typename KTraits>
__device__ __forceinline__ void LoadInput(
    const float* input,
    float (&values)[KTraits::kNItems],
    typename KTraits::BlockLoadT::TempStorage& storage,
    int valid_items) {
  if constexpr (KTraits::kIsEvenLen) {
    auto& vec_storage =
        reinterpret_cast<typename KTraits::BlockLoadVecT::TempStorage&>(
            storage);
    typename KTraits::BlockLoadVecT(vec_storage).Load(
        reinterpret_cast<const typename KTraits::vec_t*>(input),
        reinterpret_cast<typename KTraits::vec_t (&)[KTraits::kNLoads]>(
            values));
  } else {
    typename KTraits::BlockLoadT(storage).Load(
        input, values, valid_items, 0.0f);
  }
}

template <typename KTraits>
__device__ __forceinline__ void LoadWeight(
    const float* input,
    float (&values)[KTraits::kNItems],
    typename KTraits::BlockLoadWeightT::TempStorage& storage,
    int valid_items) {
  if constexpr (KTraits::kIsEvenLen) {
    auto& vec_storage =
        reinterpret_cast<typename KTraits::BlockLoadWeightVecT::TempStorage&>(
            storage);
    typename KTraits::BlockLoadWeightVecT(vec_storage).Load(
        reinterpret_cast<const typename KTraits::vec_t*>(input),
        reinterpret_cast<typename KTraits::vec_t (&)[KTraits::kNLoads]>(
            values));
  } else {
    typename KTraits::BlockLoadWeightT(storage).Load(
        input, values, valid_items, 0.0f);
  }
}

template <typename KTraits>
__device__ __forceinline__ void StoreOutput(
    float* output,
    const float (&values)[KTraits::kNItems],
    typename KTraits::BlockStoreT::TempStorage& storage,
    int valid_items) {
  float write_values[KTraits::kNItems];
#pragma unroll
  for (int item = 0; item < KTraits::kNItems; ++item) {
    write_values[item] = values[item];
  }
  if constexpr (KTraits::kIsEvenLen) {
    auto& vec_storage =
        reinterpret_cast<typename KTraits::BlockStoreVecT::TempStorage&>(
            storage);
    typename KTraits::BlockStoreVecT(vec_storage).Store(
        reinterpret_cast<typename KTraits::vec_t*>(output),
        reinterpret_cast<typename KTraits::vec_t (&)[KTraits::kNLoads]>(
            write_values));
  } else {
    typename KTraits::BlockStoreT(storage).Store(
        output, write_values, valid_items);
  }
}

struct SSMParams {
  int batch;
  int dim;
  int seqlen;
  int dstate;
  int n_chunks;

  const float* a;
  const float* b;
  const float* c;
  const float* u;
  const float* delta;
  const float* initial_x;
  float* out;
  float* final_x;
};

template <typename KTraits, bool Euler>
__global__ __launch_bounds__(KTraits::kNThreads, KTraits::kMinBlocks)
void SelectiveScanFwdKernel(SSMParams params) {
  constexpr int kNThreads = KTraits::kNThreads;
  constexpr int kNItems = KTraits::kNItems;
  constexpr int kChunkSize = kNThreads * kNItems;

  extern __shared__ char smem[];
  auto& smem_load =
      *reinterpret_cast<typename KTraits::BlockLoadT::TempStorage*>(smem);
  auto& smem_load_b =
      *reinterpret_cast<typename KTraits::BlockLoadWeightT::TempStorage*>(
          smem);
  auto& smem_load_c =
      *reinterpret_cast<typename KTraits::BlockLoadWeightT::TempStorage*>(
          smem + sizeof(typename KTraits::BlockLoadWeightT::TempStorage));
  auto& smem_store =
      *reinterpret_cast<typename KTraits::BlockStoreT::TempStorage*>(smem);
  auto& smem_scan =
      *reinterpret_cast<typename KTraits::BlockScanT::TempStorage*>(
          smem + KTraits::kSmemIOSize);
  auto* smem_running_prefix = reinterpret_cast<float2*>(
      smem + KTraits::kSmemSize);

  const int batch_id = blockIdx.x;
  const int dim_id = blockIdx.y;
  const int state_base = (batch_id * params.dim + dim_id) * params.dstate;
  if (threadIdx.x < params.dstate) {
    smem_running_prefix[threadIdx.x] =
        make_float2(1.0f, params.initial_x[state_base + threadIdx.x]);
  }
  __syncthreads();

  const float* u =
      params.u + (batch_id * params.dim + dim_id) * params.seqlen;
  const float* delta =
      params.delta + (batch_id * params.dim + dim_id) * params.seqlen;
  float* out =
      params.out + (batch_id * params.dim + dim_id) * params.seqlen;

  for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
    const int chunk_start = chunk * kChunkSize;
    const int valid_items = params.seqlen - chunk_start;
    float u_values[kNItems];
    float delta_values[kNItems];
    float delta_u_values[kNItems];
    float output_values[kNItems];

    __syncthreads();
    LoadInput<KTraits>(
        u + chunk_start, u_values, smem_load, valid_items);
    __syncthreads();
    LoadInput<KTraits>(
        delta + chunk_start, delta_values, smem_load, valid_items);

#pragma unroll
    for (int item = 0; item < kNItems; ++item) {
      if constexpr (Euler) {
        delta_u_values[item] = delta_values[item] * u_values[item];
      }
      output_values[item] = 0.0f;
    }

    for (int state = 0; state < params.dstate; ++state) {
      float b_values[kNItems];
      float c_values[kNItems];
      const float* b = params.b +
          (batch_id * params.dstate + state) * params.seqlen + chunk_start;
      const float* c = params.c +
          (batch_id * params.dstate + state) * params.seqlen + chunk_start;

      LoadWeight<KTraits>(b, b_values, smem_load_b, valid_items);
      LoadWeight<KTraits>(c, c_values, smem_load_c, valid_items);

      const float a = params.a[dim_id * params.dstate + state];
      const float a_log2e = a * static_cast<float>(M_LOG2E);
      float2 thread_data[kNItems];
#pragma unroll
      for (int item = 0; item < kNItems; ++item) {
        const float delta_a = delta_values[item] * a;
        const float a_bar = exp2f(delta_values[item] * a_log2e);
        float b_u;
        if constexpr (Euler) {
          b_u = b_values[item] * delta_u_values[item];
        } else {
          b_u = b_values[item] * (expm1f(delta_a) / a) * u_values[item];
        }
        thread_data[item] = make_float2(a_bar, b_u);
        if constexpr (!KTraits::kIsEvenLen) {
          if (threadIdx.x * kNItems + item >= valid_items) {
            thread_data[item] = make_float2(1.0f, 0.0f);
          }
        }
      }

      const float2 running_prefix = threadIdx.x % 32 == 0
          ? smem_running_prefix[state]
          : make_float2(1.0f, 0.0f);
      SSMScanPrefixCallbackOp prefix_callback(running_prefix);
      typename KTraits::BlockScanT(smem_scan).InclusiveScan(
          thread_data, thread_data, SSMScanOp{}, prefix_callback);
      if (threadIdx.x == 0) {
        smem_running_prefix[state] = prefix_callback.running_prefix;
        if (chunk == params.n_chunks - 1) {
          params.final_x[state_base + state] =
              prefix_callback.running_prefix.y;
        }
      }

#pragma unroll
      for (int item = 0; item < kNItems; ++item) {
        output_values[item] += c_values[item] * thread_data[item].y;
      }
    }

    __syncthreads();
    StoreOutput<KTraits>(
        out + chunk_start, output_values, smem_store, valid_items);
  }
}

ffi::Error CudaError(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(
      std::string(operation) + ": " + cudaGetErrorString(error));
}

template <int NThreads, int NItems, bool IsEvenLen, bool Euler>
cudaError_t LaunchSelectiveScan(SSMParams params, cudaStream_t stream) {
  using KTraits =
      SelectiveScanFwdKernelTraits<NThreads, NItems, IsEvenLen>;
  const size_t smem_size =
      KTraits::kSmemSize + params.dstate * sizeof(float2);
  auto kernel = &SelectiveScanFwdKernel<KTraits, Euler>;
  if (smem_size >= 48 * 1024) {
    const cudaError_t attribute_error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attribute_error != cudaSuccess) {
      return attribute_error;
    }
  }
  const dim3 grid(params.batch, params.dim);
  kernel<<<grid, NThreads, smem_size, stream>>>(params);
  return cudaPeekAtLastError();
}

template <int NThreads, int NItems, bool Euler>
cudaError_t DispatchEvenLength(SSMParams params, cudaStream_t stream) {
  if (params.seqlen % (NThreads * NItems) == 0) {
    return LaunchSelectiveScan<NThreads, NItems, true, Euler>(params, stream);
  }
  return LaunchSelectiveScan<NThreads, NItems, false, Euler>(params, stream);
}

template <bool Euler>
cudaError_t DispatchLength(SSMParams params, cudaStream_t stream) {
  if (params.seqlen <= 128) {
    return DispatchEvenLength<32, 4, Euler>(params, stream);
  }
  if (params.seqlen <= 256) {
    return DispatchEvenLength<32, 8, Euler>(params, stream);
  }
  if (params.seqlen <= 512) {
    return DispatchEvenLength<32, 16, Euler>(params, stream);
  }
  if (params.seqlen <= 1024) {
    return DispatchEvenLength<64, 16, Euler>(params, stream);
  }
  return DispatchEvenLength<128, 16, Euler>(params, stream);
}

ffi::Error SelectiveScanImplWithRepeats(
    cudaStream_t stream,
    int32_t discretization,
    int32_t repeats,
    int32_t batch,
    int32_t length,
    int32_t dim,
    int32_t dstate,
    const ffi::Buffer<ffi::F32>& a,
    const ffi::Buffer<ffi::F32>& deltas,
    const ffi::Buffer<ffi::F32>& bs,
    const ffi::Buffer<ffi::F32>& cs,
    const ffi::Buffer<ffi::F32>& u,
    const ffi::Buffer<ffi::F32>& initial_x,
    ffi::ResultBuffer<ffi::F32>& y,
    ffi::ResultBuffer<ffi::F32>& final_x) {
  const auto a_dims = a.dimensions();
  const auto u_dims = u.dimensions();
  if (a_dims.size() != 2 || u_dims.size() != 3) {
    return ffi::Error::InvalidArgument("A must have rank 2 and u rank 3");
  }

  if (batch <= 0 || length <= 0 || dim <= 0) {
    return ffi::Error::InvalidArgument(
        "selective scan dimensions must be positive");
  }
  if (dstate <= 0 || dstate > kMaxDState) {
    return ffi::Error::InvalidArgument("N must be between 1 and 256");
  }
  if (discretization != 0 && discretization != 1) {
    return ffi::Error::InvalidArgument(
        "discretization must be 0 (Euler) or 1 (ZOH)");
  }
  if (repeats <= 0) {
    return ffi::Error::InvalidArgument("repeats must be positive");
  }

  const int64_t expected_a = static_cast<int64_t>(dim) * dstate;
  const int64_t expected_bld = static_cast<int64_t>(batch) * length * dim;
  const int64_t expected_bln = static_cast<int64_t>(batch) * length * dstate;
  const int64_t expected_bdn = static_cast<int64_t>(batch) * dim * dstate;
  if (a.element_count() != expected_a ||
      deltas.element_count() != expected_bld ||
      bs.element_count() != expected_bln ||
      cs.element_count() != expected_bln ||
      u.element_count() != expected_bld ||
      initial_x.element_count() != expected_bdn ||
      y->element_count() != expected_bld ||
      final_x->element_count() != expected_bdn) {
    return ffi::Error::InvalidArgument(
        "selective scan buffer shapes do not match");
  }

  SSMParams params{};
  params.batch = static_cast<int>(batch);
  params.dim = static_cast<int>(dim);
  params.seqlen = static_cast<int>(length);
  params.dstate = static_cast<int>(dstate);
  params.n_chunks = static_cast<int>((length + 2047) / 2048);
  params.a = a.typed_data();
  params.b = bs.typed_data();
  params.c = cs.typed_data();
  params.u = u.typed_data();
  params.delta = deltas.typed_data();
  params.initial_x = initial_x.typed_data();
  params.out = y->typed_data();
  params.final_x = final_x->typed_data();

  // The launch dispatch uses the same sequence-length thresholds as upstream.
  // Recompute n_chunks for the selected chunk size inside each branch.
  cudaError_t error;
  if (length <= 128) {
    params.n_chunks = static_cast<int>((length + 127) / 128);
  } else if (length <= 256) {
    params.n_chunks = static_cast<int>((length + 255) / 256);
  } else if (length <= 512) {
    params.n_chunks = static_cast<int>((length + 511) / 512);
  } else if (length <= 1024) {
    params.n_chunks = static_cast<int>((length + 1023) / 1024);
  }
  for (int repeat = 0; repeat < repeats; ++repeat) {
    error = discretization == 0
        ? DispatchLength<true>(params, stream)
        : DispatchLength<false>(params, stream);
    if (error != cudaSuccess) {
      return CudaError("launching upstream selective scan", error);
    }
  }
  return ffi::Error::Success();
}

ffi::Error SelectiveScanImpl(
    cudaStream_t stream,
    int32_t discretization,
    int32_t batch,
    int32_t length,
    int32_t dim,
    int32_t dstate,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> deltas,
    ffi::Buffer<ffi::F32> bs,
    ffi::Buffer<ffi::F32> cs,
    ffi::Buffer<ffi::F32> u,
    ffi::Buffer<ffi::F32> initial_x,
    ffi::ResultBuffer<ffi::F32> y,
    ffi::ResultBuffer<ffi::F32> final_x) {
  return SelectiveScanImplWithRepeats(
      stream,
      discretization,
      1,
      batch,
      length,
      dim,
      dstate,
      a,
      deltas,
      bs,
      cs,
      u,
      initial_x,
      y,
      final_x);
}

ffi::Error SelectiveScanBenchmarkImpl(
    cudaStream_t stream,
    int32_t discretization,
    int32_t repeats,
    int32_t batch,
    int32_t length,
    int32_t dim,
    int32_t dstate,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> deltas,
    ffi::Buffer<ffi::F32> bs,
    ffi::Buffer<ffi::F32> cs,
    ffi::Buffer<ffi::F32> u,
    ffi::Buffer<ffi::F32> initial_x,
    ffi::ResultBuffer<ffi::F32> y,
    ffi::ResultBuffer<ffi::F32> final_x) {
  return SelectiveScanImplWithRepeats(
      stream,
      discretization,
      repeats,
      batch,
      length,
      dim,
      dstate,
      a,
      deltas,
      bs,
      cs,
      u,
      initial_x,
      y,
      final_x);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MambaSelectiveScan,
    SelectiveScanImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("discretization")
        .Attr<int32_t>("batch")
        .Attr<int32_t>("length")
        .Attr<int32_t>("dim")
        .Attr<int32_t>("dstate")
        .Arg<ffi::Buffer<ffi::F32>>()  // A: [D, N]
        .Arg<ffi::Buffer<ffi::F32>>()  // deltas: logical [B, L, D]
        .Arg<ffi::Buffer<ffi::F32>>()  // B: logical [B, L, N]
        .Arg<ffi::Buffer<ffi::F32>>()  // C: logical [B, L, N]
        .Arg<ffi::Buffer<ffi::F32>>()  // u: logical [B, L, D]
        .Arg<ffi::Buffer<ffi::F32>>()  // initial_x: [B, D, N]
        .Ret<ffi::Buffer<ffi::F32>>()  // y: logical [B, L, D]
        .Ret<ffi::Buffer<ffi::F32>>()  // final_x: [B, D, N]
);

// Internal microbenchmark entry point. It amortizes one JAX dispatch over
// repeated launches of the same kernel and is intentionally not part of the
// public Python selective-scan API.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MambaSelectiveScanBenchmark,
    SelectiveScanBenchmarkImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("discretization")
        .Attr<int32_t>("repeats")
        .Attr<int32_t>("batch")
        .Attr<int32_t>("length")
        .Attr<int32_t>("dim")
        .Attr<int32_t>("dstate")
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
