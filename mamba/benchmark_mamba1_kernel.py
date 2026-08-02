import os
import sys

# Ensure repository root is on sys.path when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import statistics
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Rngs
from jax.numpy import isclose

from mamba.mamba1 import S6
from icecream import ic
ic.disable()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Pallas GPU Kernel vs. Reference lax.associative_scan for Mamba-1 S6 Layer."
    )
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size B (default: 4)")
    parser.add_argument("-l", "--seq_len", type=int, default=16384, help="Sequence length L (default: 16384)")
    parser.add_argument("-d", "--dim", type=int, default=1536, help="Model inner dimension D (default: 1536)")
    parser.add_argument("-n", "--d_state", type=int, default=16, help="SSM state size N (default: 16)")
    parser.add_argument("-k", "--kernel_len", type=int, default=512, help="Pallas kernel chunk size K (default: 512)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision instead of float32")
    parser.add_argument("--skip_bwd", action="store_true", help="Skip backward pass benchmarking")
    args, _ = parser.parse_known_args()
    return args


def benchmark_fn(fn, x, warmup=5, runs=20):
    for _ in range(warmup):
        y = fn(x)
        jax.block_until_ready(y)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # us

    return {
        "median_us": statistics.median(times),
        "mean_us": statistics.mean(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def main():
    args = parse_args()

    B = args.batch_size
    L = args.seq_len
    D = args.dim
    N = args.d_state
    K = args.kernel_len
    dtype = jnp.bfloat16 if args.use_bf16 else jnp.float32

    print(f"=== Mamba-1 S6 Layer Benchmark (Fully JIT Compiled) ===")
    print(f"Device(s)        : {jax.devices()}")
    print(f"Batch Size (B)   : {B}")
    print(f"Sequence Length (L): {L:,} tokens")
    print(f"Inner Dim (D)    : {D}")
    print(f"SSM State Size(N): {N}")
    print(f"Chunk Size (K)   : {K}")
    print(f"Precision        : {'bfloat16' if args.use_bf16 else 'float32'}")
    print(f"Warmup / Runs    : {args.warmup} / {args.runs}")
    print()

    # Instantiate modules
    rngs = Rngs(1)
    test_inp = rngs.uniform((B, L, D), dtype=dtype)

    reference_s6 = S6(rngs, D=D, N=N, use_kernel=False, use_bf16=args.use_bf16)
    kernel_s6 = S6(rngs, D=D, N=N, use_kernel=True, kernel_seq_len=K, use_bf16=args.use_bf16)

    # Wrap BOTH modules with nnx.jit for a 100% fair comparison
    @nnx.jit
    def ref_fwd(x):
        return reference_s6(x)

    @nnx.jit
    def kernel_fwd(x):
        return kernel_s6(x)

    @nnx.jit
    def ref_bwd(x):
        def loss_fn(inp):
            return jnp.sum(reference_s6(inp))
        return jax.grad(loss_fn)(x)

    @nnx.jit
    def kernel_bwd(x):
        def loss_fn(inp):
            return jnp.sum(kernel_s6(inp))
        return jax.grad(loss_fn)(x)

    # Correctness Verification
    print("Checking numerical correctness...")
    ref_out = ref_fwd(test_inp)
    kernel_out = kernel_fwd(test_inp)
    jax.block_until_ready(ref_out)
    jax.block_until_ready(kernel_out)

    tol_rtol = 1e-2 if args.use_bf16 else 1e-3
    tol_atol = 1e-3 if args.use_bf16 else 1e-4
    assert isclose(kernel_out, ref_out, rtol=tol_rtol, atol=tol_atol).all(), "Outputs do not match!"
    print("Correctness Check: PASSED ✅\n")

    # Benchmark Forward Pass
    print("--- Forward Pass ---")
    ref_fwd_time = benchmark_fn(ref_fwd, test_inp, warmup=args.warmup, runs=args.runs)
    k_fwd_time = benchmark_fn(kernel_fwd, test_inp, warmup=args.warmup, runs=args.runs)

    ref_fwd_ms = ref_fwd_time['median_us'] / 1000.0
    k_fwd_ms = k_fwd_time['median_us'] / 1000.0
    fwd_speedup = ref_fwd_time["median_us"] / k_fwd_time["median_us"]

    print(f"Reference FWD Median: {ref_fwd_ms:7.2f} ms ({ref_fwd_time['median_us']:.2f} us)")
    print(f"Kernel    FWD Median: {k_fwd_ms:7.2f} ms ({k_fwd_time['median_us']:.2f} us)")
    print(f"Forward Speedup: {fwd_speedup:.2f}x\n")

    # Benchmark Backward Pass
    if not args.skip_bwd:
        print("--- Backward Pass (Grad) ---")
        ref_bwd_time = benchmark_fn(ref_bwd, test_inp, warmup=args.warmup, runs=args.runs)
        k_bwd_time = benchmark_fn(kernel_bwd, test_inp, warmup=args.warmup, runs=args.runs)

        ref_bwd_ms = ref_bwd_time['median_us'] / 1000.0
        k_bwd_ms = k_bwd_time['median_us'] / 1000.0
        bwd_speedup = ref_bwd_time["median_us"] / k_bwd_time["median_us"]

        print(f"Reference BWD Median: {ref_bwd_ms:7.2f} ms ({ref_bwd_time['median_us']:.2f} us)")
        print(f"Kernel    BWD Median: {k_bwd_ms:7.2f} ms ({k_bwd_time['median_us']:.2f} us)")
        print(f"Backward Speedup: {bwd_speedup:.2f}x\n")


if __name__ == "__main__":
    main()