# JAX-Mambas
This is a repo where I (attempt to) implement all of the 3 major Mamba varients in JAX + Flax's NNX API. 
Likely order of implementation:
1. Pure mathematical formulations as described in the papers
2. Stability tricks used in the official repos (may or may not get all of them)
3. Pallas implementations of their CUDA kernels

## Implementation Progress:
- [ ] [Mamba](https://arxiv.org/abs/2312.00752):
  - [x] Mathematical Form - (Achieves equivalence to original implementation with 1e-4 tolerance)
  - [x] Stability Tricks
  - [x] Pallas Kernel - (Achieves 4.70x speedup forward and 4.43x speedup backward on A100 when compared to JIT)
  - [ ] LayerNorm/RMSNorm and also need to add variable length sequence padding support
- [ ] [Mamba-2](https://arxiv.org/abs/2405.21060):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel
- [ ] [Mamba-3](https://arxiv.org/abs/2405.21060):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel

## Mamba-1

### Pallas GPU Selective Scan Kernel (`mamba/mamba1_kernel.py`)
A custom Pallas Mosaic GPU kernel implementing Mamba-1's S6 selective scan. It streams SSM state vectors directly through GPU Shared Memory (SMEM) and registers using `plgpu.emit_pipeline`, avoiding Global Memory (GMEM) write bottlenecks for intermediate state tensors.

### High-Level Differences from the Original Mamba CUDA Kernel
- **Scan Strategy**: Employs a 2D Blelloch parallel scan in SMEM across sequence chunks ($K$) and state dimensions ($N=16$), compared to intra-warp register shuffle instructions (`__shfl_sync`) in the original Mamba CUDA kernel.
- **Sequence Pipeline**: Streams sequence partitions via `plgpu.emit_pipeline` state carries.
- **Backward Pass**: Leverages native JAX automatic differentiation (`jax.grad` with `@nnx.remat`) instead of the custom hand-written backward CUDA kernel (`selective_scan_bwd_kernel.cuh`) from the original Mamba repository.
  - Note: there is no current plan to implement a backwards pass kernel. Pallas gets traced through by JAX's autodiff, so the forward kernel's speedups largely carry over to the backwards pass already.

### Performance Benchmarks
Evaluated on canonical Mamba-130M hyperparameters: Batch Size $B=4$, Sequence Length $L=16,384$, Inner Dimension $D=1536$, State Size $N=16$.

#### Speedup over Standard JAX JIT (`lax.associative_scan`)
- **NVIDIA A100 (40GB)** ($K=2048$): **4.70x** Forward speedup ($1.81\text{ ms}$ vs $8.52\text{ ms}$), **4.43x** Backward speedup ($6.42\text{ ms}$ vs $28.45\text{ ms}$).
- **NVIDIA L4 / G4** ($K=512$): **3.98x** Forward speedup ($3.12\text{ ms}$ vs $12.41\text{ ms}$), **3.65x** Backward speedup ($11.45\text{ ms}$ vs $41.82\text{ ms}$).

#### Comparison with the Original Mamba CUDA Kernel
On an NVIDIA A100 GPU, the pure Python JAX Pallas kernel achieves **~70% of the raw throughput** of the original Mamba CUDA kernel (`selective_scan_fn` from Tri Dao & Albert Gu's implementation: $1.81\text{ ms}$ vs $1.24\text{ ms}$ forward pass).

### Usage & GPU Hardware Profiling

Identify the optimal chunk length $K$ for your GPU hardware:
```bash
python3 mamba/sweep_mamba1_kernel_lengths.py --batch_size 4 --seq_len 16384 --dim 1536 --d_state 16
```

Benchmark forward and backward latency against standard JAX JIT:
```bash
python3 mamba/benchmark_mamba1_kernel.py --batch_size 4 --seq_len 16384 --dim 1536 --kernel_lengths 512
```