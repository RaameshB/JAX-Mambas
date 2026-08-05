from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mamba.selective_scan_cuda import selective_scan, selective_scan_reference


def inputs(batch=2, length=17, dim=8, n=16):
    keys = jax.random.split(jax.random.key(7), 6)
    A = -jnp.exp(jax.random.normal(keys[0], (dim, n)))
    deltas = jax.nn.softplus(jax.random.normal(keys[1], (batch, length, dim)))
    Bs = jax.random.normal(keys[2], (batch, length, n))
    Cs = jax.random.normal(keys[3], (batch, length, n))
    u = jax.random.normal(keys[4], (batch, length, dim))
    initial_x = jax.random.normal(keys[5], (batch, dim, n))
    return A, deltas, Bs, Cs, u, initial_x


@pytest.mark.parametrize("use_euler", [True, False])
def test_reference_matches_sequential_recurrence(use_euler):
    A, deltas, Bs, Cs, u, initial_x = inputs()
    expected_y, expected_final = selective_scan_reference(
        A,
        deltas,
        Bs,
        Cs,
        u,
        initial_x,
        use_euler_barB_approx=use_euler,
    )

    delta_a = jnp.einsum("bld,dn->bldn", deltas, A)
    a_bars = jnp.exp(delta_a)
    if use_euler:
        b_bars = jnp.einsum("bld,bln->bldn", deltas, Bs)
    else:
        b_bars = jnp.expm1(delta_a) * jnp.einsum(
            "dn,bln->bldn", jnp.reciprocal(A), Bs
        )

    state = initial_x
    outputs = []
    for step in range(u.shape[1]):
        state = a_bars[:, step] * state + b_bars[:, step] * u[:, step, :, None]
        outputs.append(jnp.einsum("bn,bdn->bd", Cs[:, step], state))
    actual_y = jnp.stack(outputs, axis=1)

    np.testing.assert_allclose(expected_y, actual_y, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(expected_final, state, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="CUDA test requires a GPU",
)
@pytest.mark.parametrize("length", [1, 17, 128, 513])
@pytest.mark.parametrize("use_euler", [True, False])
def test_cuda_matches_reference(length, use_euler):
    args = inputs(length=length)
    expected = selective_scan(*args, use_euler_barB_approx=use_euler)
    actual = jax.jit(
        lambda *xs: selective_scan(
            *xs, use_euler_barB_approx=use_euler, use_cuda=True
        )
    )(*args)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=2e-4, atol=2e-4
        )


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="CUDA test requires a GPU",
)
@pytest.mark.parametrize("length", [1, 31, 513, 2051])
def test_cuda_vjp_matches_reference(length):
    args = inputs(batch=1, length=length, dim=4, n=8)

    def loss(use_cuda, *values):
        y, final_x = selective_scan(*values, use_cuda=use_cuda)
        return jnp.mean(y**2) + jnp.mean(final_x**2)

    expected = jax.grad(partial(loss, False), argnums=range(6))(*args)
    actual = jax.jit(jax.grad(partial(loss, True), argnums=range(6)))(*args)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=1e-3, atol=1e-3
        )


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="CUDA test requires a GPU",
)
def test_zoh_cuda_vjp_matches_reference():
    args = inputs(batch=1, length=31, dim=4, n=8)

    def loss(use_cuda, *values):
        y, final_x = selective_scan(
            *values, use_euler_barB_approx=False, use_cuda=use_cuda
        )
        return jnp.mean(y**2) + jnp.mean(final_x**2)

    expected = jax.grad(partial(loss, False), argnums=range(6))(*args)
    actual = jax.jit(jax.grad(partial(loss, True), argnums=range(6)))(*args)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=2e-4, atol=2e-4
        )
