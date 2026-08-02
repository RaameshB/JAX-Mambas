if __name__ == "__main__":
    import time
    import statistics
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.nnx import Rngs
    from jax.numpy import isclose

    from mamba1 import S6

    from icecream import ic
    ic.disable()

    D = 16
    N = 16
    K = 512
    L = 2**20  # 1,048,576 sequence length
    B = 3

    WARMUP = 5
    RUNS = 20

    print(f"=== Mamba-1 Benchmark (Fully JIT Compiled) ===")
    print(f"Batch Size (B): {B}")
    print(f"Sequence Length (L): {L:,} tokens")
    print(f"Model Dimension (D): {D}")
    print(f"SSM State Size (N): {N}")
    print(f"Kernel Chunk Size (K): {K}")
    print()

    # Instantiate modules
    rngs = Rngs(1)
    test_inp = rngs.uniform((B, L, D))

    reference_s6 = S6(rngs, D=D, N=N, use_kernel=False)
    kernel_s6 = S6(rngs, D=D, N=N, use_kernel=True, kernel_seq_len=K)

    # Wrap BOTH modules with nnx.jit for a fair comparison
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

    # Correctness Check
    ref_out = ref_fwd(test_inp)
    kernel_out = kernel_fwd(test_inp)
    jax.block_until_ready(ref_out)
    jax.block_until_ready(kernel_out)

    assert isclose(kernel_out, ref_out, rtol=1e-3, atol=1e-4).all()
    print("Correctness Check: PASSED ✅")
    print()

    def benchmark_fn(fn, x, warmup=WARMUP, runs=RUNS):
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

    # --------------------
    # Benchmark Forward
    # --------------------
    print("--- Forward Pass (Fully JITed) ---")
    ref_fwd_time = benchmark_fn(ref_fwd, test_inp)
    k_fwd_time = benchmark_fn(kernel_fwd, test_inp)

    print(f"Reference FWD Median: {ref_fwd_time['median_us']:.2f} us ({ref_fwd_time['median_us']/1000:.2f} ms)")
    print(f"Kernel    FWD Median: {k_fwd_time['median_us']:.2f} us ({k_fwd_time['median_us']/1000:.2f} ms)")
    fwd_speedup = ref_fwd_time["median_us"] / k_fwd_time["median_us"]
    print(f"Forward Speedup: {fwd_speedup:.2f}x")
    print()

    # --------------------
    # Benchmark Backward
    # --------------------
    print("--- Backward Pass (Fully JITed Grad) ---")
    ref_bwd_time = benchmark_fn(ref_bwd, test_inp)
    k_bwd_time = benchmark_fn(kernel_bwd, test_inp)

    print(f"Reference BWD Median: {ref_bwd_time['median_us']:.2f} us ({ref_bwd_time['median_us']/1000:.2f} ms)")
    print(f"Kernel    BWD Median: {k_bwd_time['median_us']:.2f} us ({k_bwd_time['median_us']/1000:.2f} ms)")
    bwd_speedup = ref_bwd_time["median_us"] / k_bwd_time["median_us"]
    print(f"Backward Speedup: {bwd_speedup:.2f}x")