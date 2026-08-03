// Copyright 2023 Tri Dao and Albert Gu
// Modifications Copyright 2026 Raamesh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// This file was modified from the Mamba selective-scan implementation:
// https://github.com/state-spaces/mamba

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include <cub/block/block_scan.cuh>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// The chunk loop, launch sizes, and CUB BlockScan strategy are adapted from
// state-spaces/mamba's Apache-2.0 selective_scan_fwd_kernel.cuh.

namespace ffi = xla::ffi;

namespace {

struct ComposeAffine {
  __device__ float2 operator()(const float2& left, const float2& right) const {
    return make_float2(
        right.x * left.x,
        right.x * left.y + right.y);
  }
};

struct PrefixCallback {
  float2 running_prefix;

  __device__ explicit PrefixCallback(float2 prefix) : running_prefix(prefix) {}

  __device__ float2 operator()(float2 block_aggregate) {
    const float2 old_prefix = running_prefix;
    running_prefix = ComposeAffine{}(running_prefix, block_aggregate);
    return old_prefix;
  }
};

template <int N, int THREADS, int ITEMS>
__global__ __launch_bounds__(THREADS)
void SelectiveScanKernel(
    const float* a,
    const float* deltas,
    const float* bs,
    const float* cs,
    const float* u,
    const float* initial_x,
    float* y,
    float* final_x,
    int64_t length,
    int64_t dim,
    int32_t discretization) {
  using BlockScan = cub::BlockScan<float2, THREADS, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ float2 running_prefix[N];

  const int64_t batch_index = blockIdx.x;
  const int64_t channel = blockIdx.y;
  const int64_t state_offset = (batch_index * dim + channel) * N;
  if (threadIdx.x < N) {
    running_prefix[threadIdx.x] =
        make_float2(1.0f, initial_x[state_offset + threadIdx.x]);
  }
  __syncthreads();

  constexpr int CHUNK_SIZE = THREADS * ITEMS;
  const int64_t chunk_count = (length + CHUNK_SIZE - 1) / CHUNK_SIZE;
  for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const int64_t chunk_start = chunk * CHUNK_SIZE;
    float token_values[ITEMS];
    float delta_values[ITEMS];
    float output_values[ITEMS];

#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int64_t time = chunk_start + threadIdx.x * ITEMS + item;
      if (time < length) {
        const int64_t input_offset = (batch_index * dim + channel) * length + time;
        token_values[item] = u[input_offset];
        delta_values[item] = deltas[input_offset];
      } else {
        token_values[item] = 0.0f;
        delta_values[item] = 0.0f;
      }
      output_values[item] = 0.0f;
    }

#pragma unroll
    for (int state = 0; state < N; ++state) {
      float2 thread_data[ITEMS];
      const float state_a = a[channel * N + state];

#pragma unroll
      for (int item = 0; item < ITEMS; ++item) {
        const int64_t time = chunk_start + threadIdx.x * ITEMS + item;
        if (time < length) {
          const int64_t bc_offset = (batch_index * N + state) * length + time;
          const float delta_a = delta_values[item] * state_a;
          const float a_bar = expf(delta_a);
          const float b_bar =
              discretization == 0
                  ? delta_values[item] * bs[bc_offset]
                  : expm1f(delta_a) * bs[bc_offset] / state_a;
          thread_data[item] = make_float2(a_bar, b_bar * token_values[item]);
        } else {
          thread_data[item] = make_float2(1.0f, 0.0f);
        }
      }

      __syncthreads();
      const float2 prefix = threadIdx.x % 32 == 0
                                ? running_prefix[state]
                                : make_float2(1.0f, 0.0f);
      PrefixCallback prefix_callback(prefix);
      BlockScan(scan_storage).InclusiveScan(
          thread_data, thread_data, ComposeAffine{}, prefix_callback);
      if (threadIdx.x == 0) {
        running_prefix[state] = prefix_callback.running_prefix;
      }

#pragma unroll
      for (int item = 0; item < ITEMS; ++item) {
        const int64_t time = chunk_start + threadIdx.x * ITEMS + item;
        if (time < length) {
          const int64_t bc_offset = (batch_index * N + state) * length + time;
          output_values[item] += cs[bc_offset] * thread_data[item].y;
        }
      }
    }

#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int64_t time = chunk_start + threadIdx.x * ITEMS + item;
      if (time < length) {
        y[(batch_index * dim + channel) * length + time] = output_values[item];
      }
    }
  }

  __syncthreads();
  if (threadIdx.x < N) {
    final_x[state_offset + threadIdx.x] = running_prefix[threadIdx.x].y;
  }
}

ffi::Error CudaError(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(
      std::string(operation) + ": " + cudaGetErrorString(error));
}

template <int N, int THREADS, int ITEMS>
ffi::Error LaunchSelectiveScan(
    cudaStream_t stream,
    const float* a,
    const float* deltas,
    const float* bs,
    const float* cs,
    const float* u,
    const float* initial_x,
    float* y,
    float* final_x,
    int64_t batch,
    int64_t length,
    int64_t dim,
    int32_t discretization) {
  const dim3 grid(batch, dim);
  SelectiveScanKernel<N, THREADS, ITEMS><<<grid, THREADS, 0, stream>>>(
      a, deltas, bs, cs, u, initial_x, y, final_x, length, dim,
      discretization);
  if (cudaError_t error = cudaPeekAtLastError(); error != cudaSuccess) {
    return CudaError("launching chunked selective scan", error);
  }
  return ffi::Error::Success();
}

template <int N>
ffi::Error DispatchLength(
    cudaStream_t stream,
    const float* a,
    const float* deltas,
    const float* bs,
    const float* cs,
    const float* u,
    const float* initial_x,
    float* y,
    float* final_x,
    int64_t batch,
    int64_t length,
    int64_t dim,
    int32_t discretization) {
#define LAUNCH(THREADS, ITEMS)                                                \
  return LaunchSelectiveScan<N, THREADS, ITEMS>(                             \
      stream, a, deltas, bs, cs, u, initial_x, y, final_x, batch, length,   \
      dim, discretization)

  if (length <= 128) {
    LAUNCH(32, 4);
  } else if (length <= 256) {
    LAUNCH(32, 8);
  } else if (length <= 512) {
    LAUNCH(32, 16);
  } else if (length <= 1024) {
    LAUNCH(64, 16);
  } else {
    LAUNCH(128, 16);
  }
#undef LAUNCH
}

ffi::Error SelectiveScanImpl(
    cudaStream_t stream,
    int32_t discretization,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> deltas,
    ffi::Buffer<ffi::F32> bs,
    ffi::Buffer<ffi::F32> cs,
    ffi::Buffer<ffi::F32> u,
    ffi::Buffer<ffi::F32> initial_x,
    ffi::ResultBuffer<ffi::F32> y,
    ffi::ResultBuffer<ffi::F32> final_x) {
  const auto a_dims = a.dimensions();
  const auto u_dims = u.dimensions();
  if (a_dims.size() != 2 || u_dims.size() != 3) {
    return ffi::Error::InvalidArgument("A must have rank 2 and u rank 3");
  }

  const int64_t batch = u_dims[0];
  const int64_t dim = u_dims[1];
  const int64_t length = u_dims[2];
  const int64_t n = a_dims[1];
  if (batch <= 0 || length <= 0 || dim <= 0) {
    return ffi::Error::InvalidArgument("selective scan dimensions must be positive");
  }
  if (a_dims[0] != dim) {
    return ffi::Error::InvalidArgument("A channel dimension does not match u");
  }
  if (discretization != 0 && discretization != 1) {
    return ffi::Error::InvalidArgument("discretization must be 0 (Euler) or 1 (ZOH)");
  }

  const int64_t expected_bld = batch * length * dim;
  const int64_t expected_bln = batch * length * n;
  const int64_t expected_bdn = batch * dim * n;
  if (deltas.element_count() != expected_bld ||
      bs.element_count() != expected_bln ||
      cs.element_count() != expected_bln ||
      initial_x.element_count() != expected_bdn ||
      y->element_count() != expected_bld ||
      final_x->element_count() != expected_bdn) {
    return ffi::Error::InvalidArgument("selective scan buffer shapes do not match");
  }

#define DISPATCH_N(N_VALUE)                                                   \
  case N_VALUE:                                                               \
    return DispatchLength<N_VALUE>(                                           \
        stream, a.typed_data(), deltas.typed_data(), bs.typed_data(),         \
        cs.typed_data(), u.typed_data(), initial_x.typed_data(),              \
        y->typed_data(), final_x->typed_data(), batch, length, dim,           \
        discretization)

  switch (n) {
    DISPATCH_N(1);
    DISPATCH_N(2);
    DISPATCH_N(4);
    DISPATCH_N(8);
    DISPATCH_N(16);
    default:
      return ffi::Error::InvalidArgument("N must be one of 1, 2, 4, 8, or 16");
  }
#undef DISPATCH_N
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MambaSelectiveScan,
    SelectiveScanImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("discretization")
        .Arg<ffi::Buffer<ffi::F32>>()  // A: [D, N]
        .Arg<ffi::Buffer<ffi::F32>>()  // deltas: [B, D, L]
        .Arg<ffi::Buffer<ffi::F32>>()  // B: [B, N, L]
        .Arg<ffi::Buffer<ffi::F32>>()  // C: [B, N, L]
        .Arg<ffi::Buffer<ffi::F32>>()  // u: [B, D, L]
        .Arg<ffi::Buffer<ffi::F32>>()  // initial_x: [B, D, N]
        .Ret<ffi::Buffer<ffi::F32>>()  // y: [B, D, L]
        .Ret<ffi::Buffer<ffi::F32>>()  // final_x: [B, D, N]
);
