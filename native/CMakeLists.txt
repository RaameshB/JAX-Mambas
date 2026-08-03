cmake_minimum_required(VERSION 3.24)
project(mamba_selective_scan LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)

set(
  MAMBA_CUDA_ARCHITECTURES
  ""
  CACHE STRING
  "CUDA architectures to embed (for example: 80;89;120)"
)

if(
  MAMBA_CUDA_ARCHITECTURES MATCHES "(^|;)120($|;)"
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8
)
  message(FATAL_ERROR "sm_120 requires CUDA 12.8 or newer")
endif()

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

if(MAMBA_CUDA_ARCHITECTURES)
  set_target_properties(
    mamba_selective_scan
    PROPERTIES CUDA_ARCHITECTURES "${MAMBA_CUDA_ARCHITECTURES}"
  )
endif()

target_compile_options(
  mamba_selective_scan
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

install(TARGETS mamba_selective_scan LIBRARY DESTINATION .)
