cmake_minimum_required(VERSION 3.24)
project(mamba_selective_scan LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import jax.ffi; print(jax.ffi.include_dir(), end='')"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  COMMAND_ERROR_IS_FATAL ANY
)

add_library(mamba_selective_scan SHARED selective_scan_cuda.cu)
target_include_directories(mamba_selective_scan PRIVATE "${XLA_INCLUDE_DIR}")
target_link_libraries(mamba_selective_scan PRIVATE CUDA::cudart)
target_compile_features(mamba_selective_scan PRIVATE cxx_std_17)
set_target_properties(
  mamba_selective_scan
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE YES
)

target_compile_options(
  mamba_selective_scan
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
