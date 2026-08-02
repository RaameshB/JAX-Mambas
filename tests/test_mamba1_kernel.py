import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Rngs
from jax.numpy import isclose

from mamba import S6, apply_pallas_mamba_kernel


def has_gpu():
    """Returns True if an NVIDIA GPU device is available in JAX."""
    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def ref_ssm_fn(A_p, Delta_p, B_p, C_p, u_p, use_euler=True, dtype=jnp.float32):
    """Pure reference S6 SSM scan implementation for accuracy comparison."""
    Delta_flt = Delta_p.astype(jnp.float32)
    u_flt = u_p.astype(jnp.float32)
    B_flt = B_p.astype(jnp.float32)
    C_flt = C_p.astype(jnp.float32)

    mulDeltaA = jnp.einsum("bld,dn->bldn", Delta_flt, A_p)
    barAs = jnp.exp(mulDeltaA)
    if use_euler:
        barBs = jnp.einsum("bld,bln->bldn", Delta_flt, B_flt)
    else:
        barBs = jnp.expm1(mulDeltaA) * (1.0 / A_p[None, None, :, :]) * B_flt

    Bu = barBs * u_flt[..., jnp.newaxis]

    def combine(a, b):
        return b[0] * a[0], b[0] * a[1] + b[1]

    _, xs = jax.lax.associative_scan(combine, (barAs, Bu), axis=1)
    ys = jnp.einsum("bln,bldn->bld", C_flt, xs)
    return ys.astype(dtype), xs[:, -1]


@pytest.mark.skipif(not has_gpu(), reason="Pallas Mosaic GPU kernel requires an NVIDIA GPU.")
@pytest.mark.parametrize("L", [1, 511, 512, 513, 1024])
@pytest.mark.parametrize("use_euler", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_kernel_forward_accuracy(L, use_euler, dtype):
    """Tests Pallas GPU kernel forward output against reference lax.associative_scan."""
    B, D, N, K = 2, 8, 16, 512
    rngs = Rngs(42)
    key = rngs.params()
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    A = -jnp.exp(jax.random.normal(k1, (D, N), dtype=jnp.float32))
    Deltas = jax.nn.softplus(jax.random.normal(k2, (B, L, D), dtype=dtype))
    Bs = jax.random.normal(k3, (B, L, N), dtype=dtype)
    Cs = jax.random.normal(k4, (B, L, N), dtype=dtype)
    u = jax.random.normal(k5, (B, L, D), dtype=dtype)

    y_ref, x_ref = ref_ssm_fn(A, Deltas, Bs, Cs, u, use_euler=use_euler, dtype=dtype)
    y_kernel, x_kernel = apply_pallas_mamba_kernel(
        A, Deltas, Bs, Cs, u, N=N, use_euler_barB_approx=use_euler, K=K
    )

    tol_rtol = 1e-2 if dtype == jnp.bfloat16 else 1e-3
    tol_atol = 1e-3 if dtype == jnp.bfloat16 else 1e-4

    assert isclose(y_kernel, y_ref, rtol=tol_rtol, atol=tol_atol).all(), f"Forward y mismatch at L={L}"
    assert isclose(x_kernel, x_ref, rtol=tol_rtol, atol=tol_atol).all(), f"Final state x mismatch at L={L}"


@pytest.mark.skipif(not has_gpu(), reason="Pallas Mosaic GPU kernel requires an NVIDIA GPU.")
@pytest.mark.parametrize("L", [1, 511, 512, 513, 1024])
@pytest.mark.parametrize("use_euler", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_kernel_vjp_gradient_accuracy(L, use_euler, dtype):
    """Tests Pallas GPU kernel VJP reverse-mode gradients against reference for all parameters."""
    B, D, N, K = 2, 8, 16, 512
    rngs = Rngs(42)
    key = rngs.params()
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    A = -jnp.exp(jax.random.normal(k1, (D, N), dtype=jnp.float32))
    Deltas = jax.nn.softplus(jax.random.normal(k2, (B, L, D), dtype=dtype))
    Bs = jax.random.normal(k3, (B, L, N), dtype=dtype)
    Cs = jax.random.normal(k4, (B, L, N), dtype=dtype)
    u = jax.random.normal(k5, (B, L, D), dtype=dtype)
    cotangent = jax.random.normal(k6, (B, L, D), dtype=dtype)

    def ref_fn(A_p, Delta_p, B_p, C_p, u_p):
        return ref_ssm_fn(A_p, Delta_p, B_p, C_p, u_p, use_euler=use_euler, dtype=dtype)

    def kernel_fn(A_p, Delta_p, B_p, C_p, u_p):
        return apply_pallas_mamba_kernel(
            A_p, Delta_p, B_p, C_p, u_p, N=N, use_euler_barB_approx=use_euler, K=K
        )

    _, ref_vjp = jax.vjp(ref_fn, A, Deltas, Bs, Cs, u)
    _, kernel_vjp = jax.vjp(kernel_fn, A, Deltas, Bs, Cs, u)

    _, dummy_x = ref_ssm_fn(A, Deltas, Bs, Cs, u, use_euler=use_euler, dtype=dtype)
    g_tuple = (cotangent, jnp.zeros_like(dummy_x))

    ref_grads = ref_vjp(g_tuple)
    kernel_grads = kernel_vjp(g_tuple)

    tol_rtol = 1e-2 if dtype == jnp.bfloat16 else 1e-3
    tol_atol = 1e-3 if dtype == jnp.bfloat16 else 1e-4

    for name, g_ref, g_kernel in zip(["A", "Deltas", "Bs", "Cs", "u"], ref_grads, kernel_grads):
        assert isclose(g_kernel, g_ref, rtol=tol_rtol, atol=tol_atol).all(), f"Gradient mismatch for {name} at L={L}"


@pytest.mark.skipif(not has_gpu(), reason="Pallas Mosaic GPU kernel requires an NVIDIA GPU.")
def test_kernel_initial_x_continuation():
    """Tests streaming chunk continuation with non-zero initial_x state vectors."""
    B, D, N, K, L = 2, 8, 16, 512, 512
    rngs = Rngs(1)
    k1, k2, k3, k4, k5, k6 = jax.random.split(rngs.params(), 6)

    A = -jnp.exp(jax.random.normal(k1, (D, N)))
    Deltas = jax.nn.softplus(jax.random.normal(k2, (B, L, D)))
    Bs = jax.random.normal(k3, (B, L, N))
    Cs = jax.random.normal(k4, (B, L, N))
    u = jax.random.normal(k5, (B, L, D))
    initial_x = jax.random.normal(k6, (B, D, N))

    ys, final_x = apply_pallas_mamba_kernel(
        A, Deltas, Bs, Cs, u, N=N, K=K, initial_x=initial_x
    )
    assert ys.shape == (B, L, D)
    assert final_x.shape == (B, D, N)


def test_s6_module_forward():
    """Tests S6 Flax NNX module initialization and execution."""
    B, D, N, L = 2, 8, 16, 64
    rngs = Rngs(1)
    u = rngs.uniform((B, L, D))

    s6_ref = S6(rngs, D=D, N=N, use_kernel=False)
    out_ref = s6_ref(u)
    assert out_ref.shape == (B, L, D)


def has_official_mamba_ssm():
    """Returns True if torch, CUDA, and official mamba_ssm package are available."""
    try:
        import torch
        import mamba_ssm
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not has_official_mamba_ssm(), reason="Requires PyTorch, CUDA GPU, and official mamba_ssm package.")
def test_official_cuda_kernel_equivalence():
    """Tests numerical equivalence of JAX Pallas GPU kernel against Tri Dao's official C++/CUDA kernel (selective_scan_fn)."""
    import torch
    import numpy as np
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    B, L, D, N = 2, 512, 32, 16
    torch.manual_seed(42)
    np.random.seed(42)

    A_np = -np.exp(np.random.randn(D, N).astype(np.float32))
    delta_np = np.abs(np.random.randn(B, D, L).astype(np.float32)) + 0.1
    u_np = np.random.randn(B, D, L).astype(np.float32)
    B_np = np.random.randn(B, N, L).astype(np.float32)
    C_np = np.random.randn(B, N, L).astype(np.float32)

    # Official CUDA execution
    u_torch = torch.tensor(u_np, device="cuda", dtype=torch.float32)
    delta_torch = torch.tensor(delta_np, device="cuda", dtype=torch.float32)
    A_torch = torch.tensor(A_np, device="cuda", dtype=torch.float32)
    B_torch = torch.tensor(B_np, device="cuda", dtype=torch.float32)
    C_torch = torch.tensor(C_np, device="cuda", dtype=torch.float32)

    y_official_torch = selective_scan_fn(
        u_torch, delta_torch, A_torch, B_torch, C_torch,
        D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
    )
    y_official_np = y_official_torch.detach().cpu().numpy()

    # JAX Pallas execution
    A_jax = jnp.array(A_np)
    Deltas_jax = jnp.array(delta_np.transpose(0, 2, 1))
    u_jax = jnp.array(u_np.transpose(0, 2, 1))
    Bs_jax = jnp.array(B_np.transpose(0, 2, 1))
    Cs_jax = jnp.array(C_np.transpose(0, 2, 1))

    y_pallas, _ = apply_pallas_mamba_kernel(
        A_jax, Deltas_jax, Bs_jax, Cs_jax, u_jax, N=N, use_euler_barB_approx=False, K=512
    )
    y_pallas_np = np.array(jnp.transpose(y_pallas, (0, 2, 1)))

    assert np.allclose(y_pallas_np, y_official_np, rtol=1e-4, atol=1e-4), "Numerical mismatch vs official Mamba CUDA kernel"

