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
        description="Sweep Pallas GPU kernel chunk lengths K to identify the optimal configuration for a GPU."
    )
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size B (default: 4)")
    parser.add_argument("-l", "--seq_len", type=int, default=16384, help="Sequence length L (default: 16384)")
    parser.add_argument("-d", "--dim", type=int, default=1536, help="Model inner dimension D (default: 1536)")
    parser.add_argument("-n", "--d_state", type=int, default=16, help="SSM state size N (default: 16)")
    parser.add_argument(
        "-k", "--kernel_lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096],
        help="Kernel chunk lengths K to sweep (default: 128 256 512 1024 2048 4096)"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per chunk size (default: 3)")
    parser.add_argument("--runs", type=int, default=15, help="Benchmark iterations per chunk size (default: 15)")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision instead of float32")
    args, _ = parser.parse_known_args()
    return args


def benchmark_fwd(fn, x, warmup=3, runs=15):
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

    return statistics.median(times)


def main():
    args = parse_args()

    B = args.batch_size
    L = args.seq_len
    D = args.dim
    N = args.d_state
    dtype = jnp.bfloat16 if args.use_bf16 else jnp.float32

    print("=== Mamba-1 Pallas GPU Kernel Length (K) Sweep ===")
    print(f"Device(s)        : {jax.devices()}")
    print(f"Batch Size (B)   : {B}")
    print(f"Sequence Length (L): {L:,} tokens")
    print(f"Inner Dim (D)    : {D}")
    print(f"SSM State Size(N): {N}")
    print(f"Precision        : {'bfloat16' if args.use_bf16 else 'float32'}")
    print(f"Sweeping K       : {args.kernel_lengths}")
    print()

    rngs = Rngs(1)
    test_inp = rngs.uniform((B, L, D), dtype=dtype)

    # Reference JAX benchmark
    ref_s6 = S6(rngs, D=D, N=N, use_kernel=False, use_bf16=args.use_bf16)
    
    @nnx.jit
    def ref_fwd(x):
        return ref_s6(x)

    ref_out = ref_fwd(test_inp)
    jax.block_until_ready(ref_out)
    ref_time_us = benchmark_fwd(ref_fwd, test_inp, warmup=args.warmup, runs=args.runs)
    ref_time_ms = ref_time_us / 1000.0

    print(f"Reference JAX (lax.associative_scan): {ref_time_ms:7.2f} ms ({ref_time_us:.2f} us)")
    print("-" * 75)

    best_k = None
    best_time_us = float("inf")
    best_speedup = 0.0

    for K in args.kernel_lengths:
        try:
            kernel_s6 = S6(rngs, D=D, N=N, use_kernel=True, kernel_len=K, use_bf16=args.use_bf16)
            
            @nnx.jit
            def kernel_fwd(x):
                return kernel_s6(x)

            k_out = kernel_fwd(test_inp)
            jax.block_until_ready(k_out)

            tol_rtol = 1e-2 if args.use_bf16 else 1e-3
            tol_atol = 1e-3 if args.use_bf16 else 1e-4
            match = isclose(k_out, ref_out, rtol=tol_rtol, atol=tol_atol).all()
            if not match:
                print(f"K = {K:4d} | Correctness Check: FAILED ❌")
                continue

            k_time_us = benchmark_fwd(kernel_fwd, test_inp, warmup=args.warmup, runs=args.runs)
            k_time_ms = k_time_us / 1000.0
            speedup = ref_time_us / k_time_us

            if k_time_us < best_time_us:
                best_time_us = k_time_us
                best_k = K
                best_speedup = speedup

            print(f"K = {K:4d} | Kernel FWD: {k_time_ms:7.2f} ms ({k_time_us:8.2f} us) | Speedup: {speedup:.2f}x")
        except Exception as e:
            err_msg = str(e).split('\n')[0]
            print(f"K = {K:4d} | Error: {err_msg}")

    print("-" * 75)
    if best_k is not None:
        print(f"🏆 Optimal Chunk Size for this GPU: K = {best_k} (FWD: {best_time_us/1000:.2f} ms, {best_speedup:.2f}x Speedup)")
    else:
        print("No valid kernel configurations completed.")


if __name__ == "__main__":
    main()
