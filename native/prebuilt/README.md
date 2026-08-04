# Prebuilt CUDA selective scan

`libmamba_selective_scan.so` is the CUDA 13 Linux x86-64 fat binary loaded
automatically by `selective_scan_cuda.register_cuda_kernel()` when
`use_kernel=True`.

The binary was built by `.github/workflows/cuda-fatbin.yml` with CUDA 13.3 and
contains native device code for:

- A100 (`sm_80`)
- L4 (`sm_89`)
- G4 (`sm_120`)

It also contains `compute_120` PTX. The host must provide `libcudart.so.13` and
an NVIDIA driver compatible with CUDA 13. The prebuilt library is not portable
to macOS, Windows, non-x86-64 Linux, or CUDA 12-only environments; use the
source build documented in the repository README on those systems.

The binary's SHA-256 digest is recorded in `SHA256SUMS`. Its licensing and
upstream attribution are covered by the repository's `LICENSE-APACHE` and
`NOTICE` files.
