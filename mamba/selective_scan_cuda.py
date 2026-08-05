from __future__ import annotations

import ctypes
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


_TARGET = "mamba_selective_scan_cuda"
_BACKWARD_TARGET = "mamba_selective_scan_cuda_backward"
_LIBRARY = None
_REGISTERED = False


def _library_candidates():
    repository_root = Path(__file__).resolve().parents[1]
    native_root = repository_root / "native"
    yield native_root / "build" / "libmamba_selective_scan.so"
    yield native_root / "prebuilt" / "libmamba_selective_scan.so"
    yield native_root / "libmamba_selective_scan.so"


def register_cuda_kernel(library_path: str | Path | None = None):
    """Load and register the compiled CUDA FFI handler."""
    global _LIBRARY, _REGISTERED
    if _REGISTERED:
        return

    candidates = [Path(library_path)] if library_path is not None else list(
        _library_candidates()
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"The selective-scan CUDA library is not built; searched: {searched}"
        )

    _LIBRARY = ctypes.cdll.LoadLibrary(path)
    jax.ffi.register_ffi_target(
        _TARGET,
        jax.ffi.pycapsule(_LIBRARY.MambaSelectiveScan),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        _BACKWARD_TARGET,
        jax.ffi.pycapsule(_LIBRARY.MambaSelectiveScanBackward),
        platform="CUDA",
    )
    _REGISTERED = True


def selective_scan_reference(
    A,
    deltas,
    Bs,
    Cs,
    u,
    initial_x=None,
    *,
    use_euler_barB_approx=True,
):
    """Reference selective scan with the same contract as the CUDA kernel."""
    if initial_x is None:
        initial_x = jnp.zeros(
            (u.shape[0], u.shape[2], A.shape[1]), dtype=jnp.result_type(A, u)
        )

    delta_a = jnp.einsum("bld,dn->bldn", deltas, A)
    a_bars = jnp.exp(delta_a)
    if use_euler_barB_approx:
        b_bars = jnp.einsum("bld,bln->bldn", deltas, Bs)
    else:
        b_bars = jnp.expm1(delta_a) * jnp.einsum(
            "dn,bln->bldn", jnp.reciprocal(A), Bs
        )
    bu = b_bars * u[..., None]
    bu = bu.at[:, 0].add(a_bars[:, 0] * initial_x)

    def compose(left, right):
        left_a, left_h = left
        right_a, right_h = right
        return right_a * left_a, right_a * left_h + right_h

    _, states = jax.lax.associative_scan(compose, (a_bars, bu), axis=1)
    y = jnp.einsum("bln,bldn->bld", Cs, states)
    return y, states[:, -1]


def _validate_inputs(A, deltas, Bs, Cs, u, initial_x):
    if any(value.dtype != jnp.float32 for value in (A, deltas, Bs, Cs, u, initial_x)):
        raise TypeError("The CUDA selective scan currently supports float32 only")
    if A.ndim != 2 or u.ndim != 3:
        raise ValueError("A must have shape (D, N) and u must have shape (B, L, D)")

    batch, length, dim = u.shape
    n = A.shape[1]
    if length == 0:
        raise ValueError("The sequence length must be positive")
    if n not in (1, 2, 4, 8, 16):
        raise ValueError("N must be one of 1, 2, 4, 8, or 16")
    expected = {
        "A": (dim, n),
        "deltas": (batch, length, dim),
        "Bs": (batch, length, n),
        "Cs": (batch, length, n),
        "initial_x": (batch, dim, n),
    }
    actual = {
        "A": A.shape,
        "deltas": deltas.shape,
        "Bs": Bs.shape,
        "Cs": Cs.shape,
        "initial_x": initial_x.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")


def _selective_scan_ffi(A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx):
    register_cuda_kernel()
    _validate_inputs(A, deltas, Bs, Cs, u, initial_x)
    batch, length, dim = u.shape
    n = A.shape[1]
    n_chunks = (length + 2047) // 2048
    outputs = (
        jax.ShapeDtypeStruct((batch, length, dim), jnp.float32),
        jax.ShapeDtypeStruct((batch, dim, n), jnp.float32),
        jax.ShapeDtypeStruct((batch, dim, n_chunks, n, 2), jnp.float32),
    )
    call = jax.ffi.ffi_call(
        _TARGET,
        outputs,
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
    return call(
        A,
        deltas,
        Bs,
        Cs,
        u,
        initial_x,
        discretization=np.int32(0 if use_euler_barB_approx else 1),
        batch=np.int32(batch),
        length=np.int32(length),
        dim=np.int32(dim),
        dstate=np.int32(n),
    )


def _selective_scan_bwd_ffi(
    A, deltas, Bs, Cs, u, initial_x, chunk_states, dout, dfinal_x
):
    register_cuda_kernel()
    _validate_inputs(A, deltas, Bs, Cs, u, initial_x)
    batch, length, dim = u.shape
    n = A.shape[1]
    call = jax.ffi.ffi_call(
        _BACKWARD_TARGET,
        (
            jax.ShapeDtypeStruct(A.shape, jnp.float32),
            jax.ShapeDtypeStruct(deltas.shape, jnp.float32),
            jax.ShapeDtypeStruct(Bs.shape, jnp.float32),
            jax.ShapeDtypeStruct(Cs.shape, jnp.float32),
            jax.ShapeDtypeStruct(u.shape, jnp.float32),
            jax.ShapeDtypeStruct(initial_x.shape, jnp.float32),
        ),
        input_layouts=(
            (0, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 1, 2),
            (0, 1, 2, 3, 4),
            (0, 2, 1),
            (0, 1, 2),
        ),
        output_layouts=(
            (0, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 2, 1),
            (0, 1, 2),
        ),
    )
    return call(
        A,
        deltas,
        Bs,
        Cs,
        u,
        initial_x,
        chunk_states,
        dout,
        dfinal_x,
        batch=np.int32(batch),
        length=np.int32(length),
        dim=np.int32(dim),
        dstate=np.int32(n),
    )


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _selective_scan_cuda(A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx):
    y, final_x, _ = _selective_scan_ffi(
        A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx
    )
    return y, final_x


def _selective_scan_cuda_fwd(
    A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx
):
    y, final_x, chunk_states = _selective_scan_ffi(
        A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx
    )
    return (y, final_x), (A, deltas, Bs, Cs, u, initial_x, chunk_states)


def _selective_scan_cuda_bwd(use_euler_barB_approx, residuals, cotangents):
    A, deltas, Bs, Cs, u, initial_x, chunk_states = residuals
    if use_euler_barB_approx:
        dout, dfinal_x = cotangents
        return _selective_scan_bwd_ffi(
            A,
            deltas,
            Bs,
            Cs,
            u,
            initial_x,
            chunk_states,
            dout,
            dfinal_x,
        )

    def reference(*args):
        return selective_scan_reference(
            *args, use_euler_barB_approx=use_euler_barB_approx
        )

    _, pullback = jax.vjp(reference, A, deltas, Bs, Cs, u, initial_x)
    return pullback(cotangents)


_selective_scan_cuda.defvjp(_selective_scan_cuda_fwd, _selective_scan_cuda_bwd)


def selective_scan(
    A,
    deltas,
    Bs,
    Cs,
    u,
    initial_x=None,
    *,
    use_euler_barB_approx=True,
    use_cuda=False,
):
    """Run the reference scan or the optional CUB CUDA implementation."""
    if initial_x is None:
        initial_x = jnp.zeros(
            (u.shape[0], u.shape[2], A.shape[1]), dtype=jnp.result_type(A, u)
        )
    if use_cuda:
        return _selective_scan_cuda(
            A, deltas, Bs, Cs, u, initial_x, use_euler_barB_approx
        )
    return selective_scan_reference(
        A,
        deltas,
        Bs,
        Cs,
        u,
        initial_x,
        use_euler_barB_approx=use_euler_barB_approx,
    )
