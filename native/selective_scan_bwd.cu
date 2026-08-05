// Copyright (c) 2023 Tri Dao
// Modifications Copyright 2026 Raamesh
//
// Licensed under the Apache License, Version 2.0.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "upstream_mamba/selective_scan_bwd_kernel.cuh"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

ffi::Error CudaError(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(
      std::string(operation) + ": " + cudaGetErrorString(error));
}

ffi::Error SelectiveScanBackwardImpl(
    cudaStream_t stream,
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
    ffi::Buffer<ffi::F32> chunk_states,
    ffi::Buffer<ffi::F32> dout,
    ffi::Buffer<ffi::F32> dfinal_x,
    ffi::ResultBuffer<ffi::F32> da,
    ffi::ResultBuffer<ffi::F32> ddeltas,
    ffi::ResultBuffer<ffi::F32> dbs,
    ffi::ResultBuffer<ffi::F32> dcs,
    ffi::ResultBuffer<ffi::F32> du,
    ffi::ResultBuffer<ffi::F32> dinitial_x) {
  if (batch <= 0 || length <= 0 || dim <= 0) {
    return ffi::Error::InvalidArgument(
        "selective scan dimensions must be positive");
  }
  if (dstate <= 0 || dstate > MAX_DSTATE) {
    return ffi::Error::InvalidArgument("N must be between 1 and 256");
  }

  const int64_t chunks = (static_cast<int64_t>(length) + 2047) / 2048;
  const int64_t expected_a = static_cast<int64_t>(dim) * dstate;
  const int64_t expected_bld = static_cast<int64_t>(batch) * length * dim;
  const int64_t expected_bln =
      static_cast<int64_t>(batch) * length * dstate;
  const int64_t expected_bdn =
      static_cast<int64_t>(batch) * dim * dstate;
  const int64_t expected_chunks =
      static_cast<int64_t>(batch) * dim * chunks * dstate * 2;
  if (a.element_count() != expected_a ||
      deltas.element_count() != expected_bld ||
      bs.element_count() != expected_bln ||
      cs.element_count() != expected_bln ||
      u.element_count() != expected_bld ||
      initial_x.element_count() != expected_bdn ||
      chunk_states.element_count() != expected_chunks ||
      dout.element_count() != expected_bld ||
      dfinal_x.element_count() != expected_bdn ||
      da->element_count() != expected_a ||
      ddeltas->element_count() != expected_bld ||
      dbs->element_count() != expected_bln ||
      dcs->element_count() != expected_bln ||
      du->element_count() != expected_bld ||
      dinitial_x->element_count() != expected_bdn) {
    return ffi::Error::InvalidArgument(
        "selective scan backward buffer shapes do not match");
  }

  cudaError_t error = cudaMemsetAsync(
      da->typed_data(), 0, expected_a * sizeof(float), stream);
  if (error != cudaSuccess) {
    return CudaError("zeroing dA", error);
  }
  error = cudaMemsetAsync(
      dbs->typed_data(), 0, expected_bln * sizeof(float), stream);
  if (error != cudaSuccess) {
    return CudaError("zeroing dB", error);
  }
  error = cudaMemsetAsync(
      dcs->typed_data(), 0, expected_bln * sizeof(float), stream);
  if (error != cudaSuccess) {
    return CudaError("zeroing dC", error);
  }

  SSMParamsBwd params{};
  params.batch = batch;
  params.dim = dim;
  params.seqlen = length;
  params.dstate = dstate;
  params.n_groups = 1;
  params.n_chunks = static_cast<int>(chunks);
  params.dim_ngroups_ratio = dim;
  params.is_variable_B = true;
  params.is_variable_C = true;
  params.delta_softplus = false;

  params.A_d_stride = dstate;
  params.A_dstate_stride = 1;
  params.B_batch_stride = static_cast<uint64_t>(dstate) * length;
  params.B_group_stride = static_cast<uint64_t>(dstate) * length;
  params.B_dstate_stride = length;
  params.C_batch_stride = static_cast<uint64_t>(dstate) * length;
  params.C_group_stride = static_cast<uint64_t>(dstate) * length;
  params.C_dstate_stride = length;
  params.u_batch_stride = static_cast<uint64_t>(dim) * length;
  params.u_d_stride = length;
  params.delta_batch_stride = static_cast<uint64_t>(dim) * length;
  params.delta_d_stride = length;
  params.out_batch_stride = static_cast<uint64_t>(dim) * length;
  params.out_d_stride = length;

  params.dout_batch_stride = static_cast<uint64_t>(dim) * length;
  params.dout_d_stride = length;
  params.dA_d_stride = dstate;
  params.dA_dstate_stride = 1;
  params.dB_batch_stride = static_cast<uint64_t>(dstate) * length;
  params.dB_group_stride = static_cast<uint64_t>(dstate) * length;
  params.dB_dstate_stride = length;
  params.dC_batch_stride = static_cast<uint64_t>(dstate) * length;
  params.dC_group_stride = static_cast<uint64_t>(dstate) * length;
  params.dC_dstate_stride = length;
  params.du_batch_stride = static_cast<uint64_t>(dim) * length;
  params.du_d_stride = length;
  params.ddelta_batch_stride = static_cast<uint64_t>(dim) * length;
  params.ddelta_d_stride = length;

  params.A_ptr = const_cast<float*>(a.typed_data());
  params.B_ptr = const_cast<float*>(bs.typed_data());
  params.C_ptr = const_cast<float*>(cs.typed_data());
  params.u_ptr = const_cast<float*>(u.typed_data());
  params.delta_ptr = const_cast<float*>(deltas.typed_data());
  params.out_ptr = const_cast<float*>(dout.typed_data());
  params.x_ptr = const_cast<float*>(chunk_states.typed_data());
  params.initial_x_ptr = const_cast<float*>(initial_x.typed_data());
  params.dfinal_x_ptr = const_cast<float*>(dfinal_x.typed_data());
  params.dout_ptr = const_cast<float*>(dout.typed_data());

  params.dA_ptr = da->typed_data();
  params.dB_ptr = dbs->typed_data();
  params.dC_ptr = dcs->typed_data();
  params.du_ptr = du->typed_data();
  params.ddelta_ptr = ddeltas->typed_data();
  params.dinitial_x_ptr = dinitial_x->typed_data();

  error = selective_scan_bwd_cuda(params, stream);
  if (error != cudaSuccess) {
    return CudaError("launching upstream selective scan backward", error);
  }
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MambaSelectiveScanBackward,
    SelectiveScanBackwardImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Arg<ffi::Buffer<ffi::F32>>()  // chunk states
        .Arg<ffi::Buffer<ffi::F32>>()  // dy: logical [B, L, D]
        .Arg<ffi::Buffer<ffi::F32>>()  // dfinal_x: [B, D, N]
        .Ret<ffi::Buffer<ffi::F32>>()  // dA
        .Ret<ffi::Buffer<ffi::F32>>()  // ddeltas
        .Ret<ffi::Buffer<ffi::F32>>()  // dB
        .Ret<ffi::Buffer<ffi::F32>>()  // dC
        .Ret<ffi::Buffer<ffi::F32>>()  // du
        .Ret<ffi::Buffer<ffi::F32>>()  // dinitial_x
);
