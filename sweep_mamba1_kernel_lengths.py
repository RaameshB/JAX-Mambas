import time
import statistics
import jax
import jax.numpy as jnp
from flax.nnx import Rngs
from jax.numpy import isclose
from mamba1 import S6

D = 16
N = 16
L = 2**20  # 1,048,576 sequence length
B = 3

WARMUP = 3
RUNS = 15

print("=== A100 GPU Kernel Length (K) Sweep ===")
print("Device:", jax.devices())
print(f"B={B}, L={L:,}, D={D}, N={N}")
print()

def benchmark_fwd(fn, x):
    for _ in range(WARMUP):
        y = fn(x)
        jax.block_until_ready(y)

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    return statistics.median(times)

rngs = Rngs(1)
test_inp = rngs.uniform((B, L, D))

ref_s6 = S6(rngs, D=D, N=N, use_kernel=False)
ref_out = ref_s6(test_inp)
jax.block_until_ready(ref_out)
ref_time = benchmark_fwd(ref_s6, test_inp)

print(f"Reference JAX (lax.associative_scan): {ref_time:.2f} us ({ref_time/1000:.2f} ms)")
print("-" * 65)

for K in [128, 256, 512, 1024, 2048, 4096]:
    try:
        kernel_s6 = S6(rngs, D=D, N=N, use_kernel=True, kernel_seq_len=K)
        k_out = kernel_s6(test_inp)
        jax.block_until_ready(k_out)

        match = isclose(k_out, ref_out, rtol=1e-3, atol=1e-4).all()
        if not match:
            print(f"K = {K:4d} | Correctness FAILED ❌")
            continue

        k_time = benchmark_fwd(kernel_s6, test_inp)
        speedup = ref_time / k_time
        print(f"K = {K:4d} | Kernel FWD: {k_time:8.2f} us ({k_time/1000:.2f} ms) | Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"K = {K:4d} | Error: {e}")
