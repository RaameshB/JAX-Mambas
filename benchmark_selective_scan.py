import argparse
import time

import jax
import jax.numpy as jnp

from selective_scan_cuda import selective_scan


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
    args = parser.parse_args()

    print("devices", jax.devices())
    print("B", args.batch, "D", args.dim, "N", args.n)
    for length in args.lengths:
        values = make_inputs(args.batch, length, args.dim, args.n)
        reference = lambda *xs: selective_scan(*xs, use_cuda=False)
        cuda = lambda *xs: selective_scan(*xs, use_cuda=True)
        reference_min, reference_mean = benchmark(reference, values, args.repeats)
        cuda_min, cuda_mean = benchmark(cuda, values, args.repeats)
        print(
            f"L={length:5d} "
            f"reference={reference_min:8.3f}/{reference_mean:8.3f} ms "
            f"cuda={cuda_min:8.3f}/{cuda_mean:8.3f} ms "
            f"speedup={reference_min / cuda_min:6.2f}x"
        )


if __name__ == "__main__":
    main()
