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
  - [x] Pallas Kernel - (Achieves 42.5x forward / 41.3x backward speedup vs lax.associative_scan and ~71-78% of official C++/CUDA kernel throughput on A100)
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
- **Scan Strategy**: Employs a 2D Blelloch parallel scan in SMEM across sequence chunks ($K=512$) and state dimensions ($N=16$), compared to intra-warp register shuffle instructions (`__shfl_sync`) in the original Mamba CUDA kernel.
- **Sequence Pipeline**: Streams sequence partitions via `plgpu.emit_pipeline` state carries.
- **Backward Pass**: Implements a dedicated Pallas GPU backward scan kernel (`_apply_pallas_mamba_bwd_kernel_raw`) registered via `@jax.custom_vjp`, executing time-reversed parallel Blelloch scans directly in GPU SRAM.

### Performance Benchmarks
Evaluated on canonical Mamba-130M hyperparameters: Batch Size $B=4$, Sequence Length $L=16,384$, Inner Dimension $D=1536$, State Size $N=16$.

#### Speedup over Standard JAX JIT (`lax.associative_scan`)
- **NVIDIA A100 (40GB)** ($K=512$): **42.5x** Forward speedup ($7.08\text{ ms}$ vs $301.12\text{ ms}$), **41.3x** Backward speedup ($23.41\text{ ms}$ vs $967.65\text{ ms}$).
- **NVIDIA L4** ($K=512$): **31.0x** Forward speedup ($20.71\text{ ms}$ vs $642.15\text{ ms}$).

#### Comparison with the Original Mamba CUDA Kernel (`mamba_ssm`)
On an NVIDIA A100 GPU ($L=16,384$), our pure Python JAX Pallas kernel achieves **~71-78% of the raw throughput** of Tri Dao & Albert Gu's official C++/CUDA kernels:
- **Forward Pass**: **`7.08 ms`** (Pallas) vs **`5.04 ms`** (Official CUDA) — **`0.71x` of CUDA**
- **Backward Pass**: **`23.41 ms`** (Pallas) vs **`18.24 ms`** (Official CUDA) — **`0.78x` of CUDA**

### Usage & GPU Hardware Profiling

Identify the optimal chunk length $K$ for your GPU hardware:
```bash
python3 mamba/sweep_mamba1_kernel_lengths.py --batch_size 4 --seq_len 16384 --dim 1536 --d_state 16
```

Benchmark forward and backward latency against standard JAX JIT:
```bash
python3 mamba/benchmark_mamba1_kernel.py --batch_size 4 --seq_len 16384 --dim 1536 --kernel_lengths 512
```