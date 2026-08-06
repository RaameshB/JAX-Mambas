import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from mamba import selective_scan_cuda as scan_cuda
from mamba.selective_scan_cuda import selective_scan


_BENCHMARK_TARGET = "mamba_selective_scan_cuda_benchmark"
_BENCHMARK_REGISTERED = False


def block_until_ready(tree):
    jax.tree.map(lambda value: value.block_until_ready(), tree)


def benchmark(function, args, repeats):
    compiled = jax.jit(function)
    output = compiled(*args)
    block_until_ready(output)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = compiled(*args)
        block_until_ready(output)
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples), sum(samples) / len(samples)


def repeated_cuda_scan(*args, launches):
    """Launch the CUDA kernel repeatedly behind one JAX FFI dispatch."""
    global _BENCHMARK_REGISTERED
    scan_cuda.register_cuda_kernel()
    if not _BENCHMARK_REGISTERED:
        jax.ffi.register_ffi_target(
            _BENCHMARK_TARGET,
            jax.ffi.pycapsule(scan_cuda._LIBRARY.MambaSelectiveScanBenchmark),
            platform="CUDA",
        )
        _BENCHMARK_REGISTERED = True

    A, deltas, Bs, Cs, u, initial_x = args
    batch, length, dim = u.shape
    n = A.shape[1]
    n_chunks = (length + 2047) // 2048
    call = jax.ffi.ffi_call(
        _BENCHMARK_TARGET,
        (
            jax.ShapeDtypeStruct((batch, length, dim), jnp.float32),
            jax.ShapeDtypeStruct((batch, dim, n), jnp.float32),
            jax.ShapeDtypeStruct((batch, dim, n_chunks, n, 2), jnp.float32),
        ),
        input_layouts=(
            (0, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 1, 2),
        ),
        output_layouts=((0, 2, 1), (0, 1, 2), (0, 1, 2, 3, 4)),
    )
    y, final_x, _ = call(
        A,
        deltas,
        Bs,
        Cs,
        u,
        initial_x,
        discretization=np.int32(0),
        repeats=np.int32(launches),
        batch=np.int32(batch),
        length=np.int32(length),
        dim=np.int32(dim),
        dstate=np.int32(n),
    )
    return y, final_x


def make_inputs(batch, length, dim, n):
    keys = jax.random.split(jax.random.key(0), 6)
    return (
        -jnp.exp(jax.random.normal(keys[0], (dim, n))),
        jax.nn.softplus(jax.random.normal(keys[1], (batch, length, dim))),
        jax.random.normal(keys[2], (batch, length, n)),
        jax.random.normal(keys[3], (batch, length, n)),
        jax.random.normal(keys[4], (batch, length, dim)),
        jax.random.normal(keys[5], (batch, dim, n)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kernel-launches", type=int, default=100)
    args = parser.parse_args()

    print("devices", jax.devices())
    print("B", args.batch, "D", args.dim, "N", args.n)
    for length in args.lengths:
        values = make_inputs(args.batch, length, args.dim, args.n)
        reference = lambda *xs: selective_scan(*xs, use_cuda=False)
        cuda = lambda *xs: selective_scan(*xs, use_cuda=True)
        repeated_cuda = lambda *xs: repeated_cuda_scan(
            *xs, launches=args.kernel_launches
        )
        reference_min, reference_mean = benchmark(reference, values, args.repeats)
        cuda_min, cuda_mean = benchmark(cuda, values, args.repeats)
        repeated_min, repeated_mean = benchmark(
            repeated_cuda, values, args.repeats
        )
        print(
            f"L={length:5d} "
            f"reference={reference_min:8.3f}/{reference_mean:8.3f} ms "
            f"cuda={cuda_min:8.3f}/{cuda_mean:8.3f} ms "
            f"cuda_amortized="
            f"{repeated_min / args.kernel_launches:8.3f}/"
            f"{repeated_mean / args.kernel_launches:8.3f} ms "
            f"speedup={reference_min / cuda_min:6.2f}x"
        )


if __name__ == "__main__":
    main()
