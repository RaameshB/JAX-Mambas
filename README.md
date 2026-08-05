# JAX-Mambas
This is a repo where I (attempt to) implement all of the 3 major Mamba variants in JAX + Flax's NNX API. 
Likely order of implementation:
1. Pure mathematical formulations as described in the papers
2. Stability tricks used in the official repos (may or may not get all of them)
3. Pallas implementations of their CUDA kernels

## Implementation Progress:
- [x] [Mamba](https://arxiv.org/abs/2312.00752):
  - [x] Mathematical Form
  - [x] Stability Tricks
  - [x] ~~Pallas~~ _CUDA_ Kernel (Pallas doesn't really offer a good blocked prefix scan and attempts to write a custom one have not gone well)
  - [x] LayerNorm/RMSNorm and added variable length sequence padding support
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

Mamba-1 has an optional float32 CUDA forward and backward path implemented with
JAX FFI and CUB `BlockScan`. It adapts Mamba v2.3.2's selective-scan kernels to
JAX's layouts and custom-VJP interface: one CUDA block per batch/channel pair,
a serial loop over sequence chunks, and affine scans inside each chunk.

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
and reverse-mode differentiation are tested. Euler-mode reverse differentiation
uses the adapted CUDA backward kernel, including gradients for nonzero initial
states and the returned final state. ZOH-mode reverse differentiation falls
back to the JAX reference because the original Mamba backward kernel implements
the Euler approximation only.

In a steady-state Colab G4 benchmark shaped like the induction-heads notebook
(`B=8`, `L=256`, `D=64`, two Mamba layers, `expand=2`, `N=16`, `R=16`), the
fully JIT-compiled loss-and-gradient call took 1.132 ms with the CUDA forward
and backward kernels versus 1.600 ms with the JIT-compiled JAX reference scan.
That is 1.42x the throughput (42% more loss/gradient evaluations per second),
or 29% lower latency. Compilation and optimizer updates were excluded from the
timed region; both paths used the same parameters and batch.

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

Both paths used float32 Euler discretization and zero initial state. GPU
correctness and gradient tests pass, including nonzero initial states and
sequences spanning multiple chunks. The isolated result shows that most of the
apparent end-to-end gap is outside the CUDA kernel itself.

## License

The original JAX-Mambas code is available under the MIT License. The adapted
CUDA selective-scan implementation is available under Apache-2.0; see
`LICENSE-APACHE` and `NOTICE` for its upstream attribution.
