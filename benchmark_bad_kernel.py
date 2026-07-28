if __name__ == "__main__":
    import time
    import statistics

    import jax
    from flax.nnx import Rngs
    from jax.numpy import isclose

    from mamba1 import S6

    from icecream import ic
    ic.disable()

    D = 16
    N = 8
    K = 32
    L = 32
    B = 3

    WARMUP = 5
    RUNS = 100

    def benchmark(fn, x, warmup=WARMUP, runs=RUNS):
        # Compile + warm caches.
        for _ in range(warmup):
            jax.block_until_ready(fn(x))

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
    # Reference
    # --------------------

    rngs = Rngs(1)
    test_inp = rngs.uniform((B, L, D))

    reference_s6 = S6(
        rngs,
        D=D,
        N=N,
        use_kernel=False,
    )

    reference_out = reference_s6(test_inp)
    jax.block_until_ready(reference_out)

    # --------------------
    # Kernel
    # --------------------

    rngs = Rngs(1)
    test_inp = rngs.uniform((B, L, D))

    kernel_s6 = S6(
        rngs,
        D=D,
        N=N,
        use_kernel=True,
        kernel_seq_len=K,
    )

    kernel_out = kernel_s6(test_inp)
    jax.block_until_ready(kernel_out)

    # --------------------
    # Correctness
    # --------------------

    assert isclose(
        kernel_out,
        reference_out,
        rtol=1e-4,
        atol=1e-5,
    ).all()

    print("correctness passed")

    # --------------------
    # Benchmark
    # --------------------

    reference_time = benchmark(reference_s6, test_inp)
    kernel_time = benchmark(kernel_s6, test_inp)

    print()
    print("Reference:")
    print(reference_time)

    print()
    print("Kernel:")
    print(kernel_time)

    speedup = (
        reference_time["median_us"]
        / kernel_time["median_us"]
    )

    print()
    print(f"Speedup: {speedup:.2f}x")