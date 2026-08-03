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

Enable it with `S6(..., use_kernel=True)` or `Mamba(..., use_kernel=True)`.
The current CUDA path supports real float32 inputs and `N` in
`{1, 2, 4, 8, 16}`. Euler and ZOH discretization, nonzero initial states, JIT,
and reverse-mode differentiation are tested. The VJP currently uses the JAX
reference implementation, so only the forward pass is accelerated.

On a Colab L4 at `B=1, D=256, N=16`, the CUDA forward path was approximately
at parity with `lax.associative_scan` for lengths 512 and 2048, and 2.0x faster
at length 8192. Run `python benchmark_selective_scan.py` to benchmark locally.
