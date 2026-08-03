# JAX-Mambas
This is a repo where I (attempt to) implement all of the 3 major Mamba varients in JAX + Flax's NNX API. 
Likely order of implementation:
1. Pure mathematical formulations as described in the papers
2. Stability tricks used in the official repos (may or may not get all of them)
3. Pallas implementations of their CUDA kernels

## Implementation Progress:
- [ ] [Mamba](https://arxiv.org/abs/2312.00752):
  - [x] Mathematical Form
  - [x] Stability Tricks
  - [ ] Pallas Kernel
  - [ ] LayerNorm/RMSNorm and also need to add variable length sequence padding support
- [ ] [Mamba-2](https://arxiv.org/abs/2405.21060):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel
- [ ] [Mamba-3](https://arxiv.org/abs/2405.21060):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel

### Notes:
- Models will be called "naive" when they are implemented without their kernels

## Experimental CUDA selective scan

Mamba-1 has an optional float32 CUDA forward path implemented with JAX FFI and
CUB `BlockScan`. It follows the original Mamba kernel's layout: one CUDA block
per batch/channel pair, a serial loop over sequence chunks, and an inclusive
affine scan inside each chunk.

Build the shared library using the Python environment that contains JAX:

```bash
cmake -S native -B native/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build native/build -j
```

The release workflow uses CUDA 13.3 to produce one fat shared library
containing native code for A100 (`sm_80`), L4 (`sm_89`), and G4 (`sm_120`).
The same build can be reproduced with a CUDA 13 Python/build environment:

```bash
cmake -S native -B native/build-fat \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DMAMBA_CUDA_ARCHITECTURES="80;89;120"
cmake --build native/build-fat -j
cmake --install native/build-fat --prefix native/prebuilt
```

Release binaries include `LICENSE`, `LICENSE-APACHE`, and `NOTICE`.
The loader searches `native/build`, `native/prebuilt`, and `native` by default;
a downloaded binary can also be passed explicitly to
`register_cuda_kernel(path)`.

Enable it with `S6(..., use_kernel=True)` or `Mamba(..., use_kernel=True)`.
The current CUDA path supports real float32 inputs and `N` in
`{1, 2, 4, 8, 16}`. Euler and ZOH discretization, nonzero initial states, JIT,
and reverse-mode differentiation are tested. The VJP currently uses the JAX
reference implementation, so only the forward pass is accelerated.

On a Colab L4 at `B=1, D=256, N=16`, the CUDA forward path was approximately
at parity with `lax.associative_scan` for lengths 512 and 2048, and 2.0x faster
at length 8192. Run `python benchmark_selective_scan.py` to benchmark locally.

Against Mamba v2.3.2's official float32 forward kernel on an A100-SXM4-40GB
(`B=1, D=256, N=16`), end-to-end framework call latency is:

| Length | JAX FFI (median) | Official Mamba (median) |
| ---: | ---: | ---: |
| 512 | 0.172 ms | 0.045 ms |
| 2048 | 0.223 ms | 0.072 ms |
| 8192 | 0.412 ms | 0.210 ms |

That comparison includes JAX dispatch, synchronization, output allocation, and
layout handling. To isolate the CUDA work, the benchmark also launches each
kernel 100 times behind one framework call and divides the elapsed time by 100:

| Length | This kernel (median) | Official Mamba (median) | Difference |
| ---: | ---: | ---: | ---: |
| 512 | 0.0282 ms | 0.0262 ms | 7.6% slower |
| 2048 | 0.0560 ms | 0.0510 ms | 9.8% slower |
| 8192 | 0.1584 ms | 0.1502 ms | 5.5% slower |

Both paths used float32 Euler discretization and zero initial state. All 11 GPU
correctness and gradient tests passed, and the benchmark outputs were bitwise
identical. The isolated result shows that most of the apparent end-to-end gap
is outside the CUDA kernel itself.

## License

The original JAX-Mambas code is available under the MIT License. The adapted
CUDA selective-scan implementation is available under Apache-2.0; see
`LICENSE-APACHE` and `NOTICE` for its upstream attribution.
