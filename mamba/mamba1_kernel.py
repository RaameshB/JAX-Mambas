import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax import ShapeDtypeStruct


def _mamba_kernel_chunk_size(seq_len):
    if seq_len <= 128:
        return 128
    if seq_len <= 256:
        return 256
    return 512


def blelloch_scan(A_seq, h_seq):
    """Work-efficient 2D (Blelloch) inclusive scan over time axis (axis 0).
    
    A_seq: (block_len, N)
    h_seq: (block_len, N)
    """
    def combine(earlier, later):
        A_e, h_e = earlier
        A_l, h_l = later
        return A_l * A_e, A_l * h_e + h_l

    def deinterleave(v):
        T, N = v.shape
        pair = v.reshape(T // 2, 2, N)
        return pair[:, 0, :], pair[:, 1, :]

    def interleave(a, b):
        T_half, N = a.shape
        pair = jnp.zeros((T_half, 2, N), dtype=a.dtype)
        pair = pair.at[:, 0, :].set(a)
        pair = pair.at[:, 1, :].set(b)
        return pair.reshape(T_half * 2, N)

    def _scan(A_e, h_e):
        T = A_e.shape[0]
        if T == 1:
            return A_e, h_e
        if T == 2:
            A0, A1 = A_e[:1], A_e[1:]
            h0, h1 = h_e[:1], h_e[1:]
            odd_A, odd_h = combine((A0, h0), (A1, h1))
            return interleave(A0, odd_A), interleave(h0, odd_h)

        A0, A1 = deinterleave(A_e)
        h0, h1 = deinterleave(h_e)
        odd_A, odd_h = _scan(*combine((A0, h0), (A1, h1)))
        even_rest_A, even_rest_h = combine((odd_A[:-1], odd_h[:-1]), (A0[1:], h0[1:]))

        even_A = A0.at[1:].set(even_rest_A)
        even_h = h0.at[1:].set(even_rest_h)
        return interleave(even_A, odd_A), interleave(even_h, odd_h)

    return _scan(A_seq, h_seq)


def apply_reference_ssm(A, Deltas, Bs, Cs, u, N=16, use_euler_barB_approx=True, initial_x=None):
    mulDeltaA = jnp.einsum("bld,dn->bldn", Deltas, A)
    barAs = jnp.exp(mulDeltaA)
    if use_euler_barB_approx:
        barBs = jnp.einsum("bld,bln->bldn", Deltas, Bs)
    else:
        barBs = jnp.expm1(mulDeltaA) * jnp.einsum("dn,bln->bldn", jnp.reciprocal(A), Bs)

    Bu = barBs * u[..., jnp.newaxis]

    def binary_operator(Aht_prev, Aht):
        At_prev, ht_prev = Aht_prev
        At, ht = Aht
        return At * At_prev, At * ht_prev + ht

    if initial_x is not None:
        Bu = Bu.at[:, 0, ...].add(barAs[:, 0, ...] * initial_x[:, None, ...])

    _, xs = jax.lax.associative_scan(binary_operator, (barAs, Bu), axis=1)
    ys = jnp.einsum("bln,bldn->bld", Cs, xs)
    return ys, xs[:, -1]


def _apply_pallas_mamba_kernel_raw(
    A, Deltas, Bs, Cs, u,
    N: int = 16,
    use_euler_barB_approx: bool = True,
    K: int = 512,
    initial_x = None,
    scan_dtype = jnp.float32,
    max_concurrent_steps: int = 1,
):
    batch, seq_len, dim = u.shape

    assert A.shape == (dim, N), f"A shape must be ({dim}, {N}), got {A.shape}"
    assert Deltas.shape == (batch, seq_len, dim), f"Deltas shape must be ({batch}, {seq_len}, {dim}), got {Deltas.shape}"
    assert Bs.shape == (batch, seq_len, N), f"Bs shape must be ({batch}, {seq_len}, {N}), got {Bs.shape}"
    assert Cs.shape == (batch, seq_len, N), f"Cs shape must be ({batch}, {seq_len}, {N}), got {Cs.shape}"
    assert u.shape == (batch, seq_len, dim), f"u shape must be ({batch}, {seq_len}, {dim}), got {u.shape}"
    if initial_x is not None:
        assert initial_x.shape == (batch, dim, N), f"initial_x shape must be ({batch}, {dim}, {N}), got {initial_x.shape}"

    block_len = _mamba_kernel_chunk_size(seq_len) if K is None else K
    if block_len & (block_len - 1):
        raise ValueError("K (kernel_len) must be a power of two for Blelloch scan.")

    padded_len = ((seq_len + block_len - 1) // block_len) * block_len
    n_chunks = padded_len // block_len
    kernel_n_state = N

    y_dtype = u.dtype

    if padded_len != seq_len:
        time_padding = padded_len - seq_len
        u = jnp.pad(u, ((0, 0), (0, time_padding), (0, 0)))
        Deltas = jnp.pad(Deltas, ((0, 0), (0, time_padding), (0, 0)))
        Bs = jnp.pad(Bs, ((0, 0), (0, time_padding), (0, 0)))
        Cs = jnp.pad(Cs, ((0, 0), (0, time_padding), (0, 0)))

    Deltas_bdl = jnp.transpose(Deltas, (0, 2, 1))
    u_bdl = jnp.transpose(u, (0, 2, 1))
    has_init_x = initial_x is not None
    init_x_val = initial_x if has_init_x else jnp.zeros((batch, dim, N), dtype=scan_dtype)

    def ssm_kernel(A_ref, Delta_ref, B_ref, C_ref, u_ref, init_x_ref, ys_ref, final_x_ref):
        b = pl.program_id(0)
        d = pl.program_id(1)

        A_states = jnp.array([A_ref[d, n].astype(scan_dtype) for n in range(kernel_n_state)])
        init_x = init_x_ref[b, d, :].astype(scan_dtype)

        def chunk_body(_, Delta_smem, B_smem, C_smem, u_smem, y_smem, x):
            deltas = Delta_smem[:].astype(scan_dtype)
            tokens = u_smem[:].astype(scan_dtype)
            B_mat = B_smem[:, :].astype(scan_dtype)
            C_mat = C_smem[:, :].astype(scan_dtype)

            delta_A = deltas[:, None] * A_states[None, :]
            A_bars = jnp.exp(delta_A)

            if use_euler_barB_approx:
                B_bars = deltas[:, None] * B_mat
            else:
                B_bars = jnp.expm1(delta_A) * (1.0 / A_states[None, :]) * B_mat

            Bu = B_bars * tokens[:, None]

            A_cum, h_cum = blelloch_scan(A_bars, Bu)
            xs = A_cum * x[None, :] + h_cum

            y_smem[:] = jnp.sum(C_mat * xs, axis=-1).astype(y_dtype)
            return xs[-1]

        pipeline = plgpu.emit_pipeline(
            chunk_body,
            grid=(n_chunks,),
            in_specs=[
                plgpu.BlockSpec((block_len,), lambda chunk: (chunk,)),
                plgpu.BlockSpec((block_len, kernel_n_state), lambda chunk: (chunk, 0)),
                plgpu.BlockSpec((block_len, kernel_n_state), lambda chunk: (chunk, 0)),
                plgpu.BlockSpec((block_len,), lambda chunk: (chunk,)),
            ],
            out_specs=[
                plgpu.BlockSpec((block_len,), lambda chunk: (chunk,)),
            ],
            max_concurrent_steps=max_concurrent_steps,
            init_carry=init_x,
        )
        final_x = pipeline(
            Delta_ref.at[b, d],
            B_ref.at[b],
            C_ref.at[b],
            u_ref.at[b, d],
            ys_ref.at[b, d],
        )
        final_x_ref[b, d, :] = final_x

    out_structs = (
        ShapeDtypeStruct((batch, dim, padded_len), dtype=y_dtype),
        ShapeDtypeStruct((batch, dim, N), dtype=scan_dtype),
    )
    try:
        ys_bdl, final_x = plgpu.kernel(
            ssm_kernel,
            out_type=out_structs,
            grid=(batch, dim),
            grid_names=("batch", "dim"),
        )(A, Deltas_bdl, Bs, Cs, u_bdl, init_x_val)
    except TypeError:
        ys_bdl, final_x = plgpu.kernel(
            ssm_kernel,
            out_shape=out_structs,
            grid=(batch, dim),
            grid_names=("batch", "dim"),
        )(A, Deltas_bdl, Bs, Cs, u_bdl, init_x_val)

    ys = jnp.transpose(ys_bdl, (0, 2, 1))
    return ys[:, :seq_len, :], final_x


def _apply_pallas_mamba_bwd_kernel_raw(
    A, Deltas, Bs, Cs, u, g_ys,
    N: int = 16,
    use_euler_barB_approx: bool = True,
    K: int = 512,
    scan_dtype = jnp.float32,
    max_concurrent_steps: int = 1,
):
    batch, seq_len, dim = u.shape
    block_len = _mamba_kernel_chunk_size(seq_len) if K is None else K
    padded_len = ((seq_len + block_len - 1) // block_len) * block_len
    n_chunks = padded_len // block_len

    if padded_len != seq_len:
        time_padding = padded_len - seq_len
        u = jnp.pad(u, ((0, 0), (0, time_padding), (0, 0)))
        Deltas = jnp.pad(Deltas, ((0, 0), (0, time_padding), (0, 0)))
        Bs = jnp.pad(Bs, ((0, 0), (0, time_padding), (0, 0)))
        Cs = jnp.pad(Cs, ((0, 0), (0, time_padding), (0, 0)))
        g_ys = jnp.pad(g_ys, ((0, 0), (0, time_padding), (0, 0)))

    Deltas_rev = jnp.flip(Deltas, axis=1)
    Cs_rev = jnp.flip(Cs, axis=1)
    g_ys_rev = jnp.flip(g_ys, axis=1)

    Deltas_bdl = jnp.transpose(Deltas_rev, (0, 2, 1))
    g_ys_bdl = jnp.transpose(g_ys_rev, (0, 2, 1))
    init_dx_val = jnp.zeros((batch, dim, N), dtype=scan_dtype)

    def ssm_bwd_kernel(A_ref, Delta_ref, C_ref, g_y_ref, init_dx_ref, dx_ref):
        b = pl.program_id(0)
        d = pl.program_id(1)

        A_states = jnp.array([A_ref[d, n].astype(scan_dtype) for n in range(N)])
        init_dx = init_dx_ref[b, d, :].astype(scan_dtype)

        def chunk_body(_, Delta_smem, C_smem, g_y_smem, dx_smem, dx_carry):
            deltas = Delta_smem[:].astype(scan_dtype)
            g_y = g_y_smem[:].astype(scan_dtype)
            C_mat = C_smem[:, :].astype(scan_dtype)

            delta_A = deltas[:, None] * A_states[None, :]
            A_bars = jnp.exp(delta_A)
            v_inputs = C_mat * g_y[:, None]

            A_cum, h_cum = blelloch_scan(A_bars, v_inputs)
            dxs = A_cum * dx_carry[None, :] + h_cum

            dx_smem[:, :] = dxs.astype(scan_dtype)
            return dxs[-1]

        pipeline = plgpu.emit_pipeline(
            chunk_body,
            grid=(n_chunks,),
            in_specs=[
                plgpu.BlockSpec((block_len,), lambda chunk: (chunk,)),
                plgpu.BlockSpec((block_len, N), lambda chunk: (chunk, 0)),
                plgpu.BlockSpec((block_len,), lambda chunk: (chunk,)),
            ],
            out_specs=[
                plgpu.BlockSpec((block_len, N), lambda chunk: (chunk, 0)),
            ],
            max_concurrent_steps=max_concurrent_steps,
            init_carry=init_dx,
        )
        final_dx = pipeline(
            Delta_ref.at[b, d],
            C_ref.at[b],
            g_y_ref.at[b, d],
            dx_ref.at[b, d],
        )

    out_structs = ShapeDtypeStruct((batch, dim, padded_len, N), dtype=scan_dtype)
    try:
        dx_bdl = plgpu.kernel(
            ssm_bwd_kernel,
            out_type=out_structs,
            grid=(batch, dim),
            grid_names=("batch", "dim"),
        )(A, Deltas_bdl, Cs_rev, g_ys_bdl, init_dx_val)
    except TypeError:
        dx_bdl = plgpu.kernel(
            ssm_bwd_kernel,
            out_shape=out_structs,
            grid=(batch, dim),
            grid_names=("batch", "dim"),
        )(A, Deltas_bdl, Cs_rev, g_ys_bdl, init_dx_val)

    dx_rev = jnp.transpose(dx_bdl, (0, 2, 1, 3))
    return jnp.flip(dx_rev[:, :seq_len, :, :], axis=1)


from functools import partial

@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def apply_pallas_mamba_kernel(
    A, Deltas, Bs, Cs, u,
    N: int = 16,
    use_euler_barB_approx: bool = True,
    K: int = 512,
    scan_dtype = jnp.float32,
    max_concurrent_steps: int = 1,
    initial_x = None,
):
    ys, final_x = _apply_pallas_mamba_kernel_raw(
        A, Deltas, Bs, Cs, u, N, use_euler_barB_approx, K, initial_x, scan_dtype, max_concurrent_steps
    )
    return ys, final_x


def _apply_pallas_mamba_kernel_fwd(
    A, Deltas, Bs, Cs, u,
    N, use_euler_barB_approx, K, scan_dtype, max_concurrent_steps,
    initial_x=None
):
    ys, final_x = _apply_pallas_mamba_kernel_raw(
        A, Deltas, Bs, Cs, u, N, use_euler_barB_approx, K, initial_x, scan_dtype, max_concurrent_steps
    )
    res = (A, Deltas, Bs, Cs, u, initial_x)
    return (ys, final_x), res


def _apply_pallas_mamba_kernel_bwd(
    N, use_euler_barB_approx, K, scan_dtype, max_concurrent_steps,
    res, g
):
    A, Deltas, Bs, Cs, u, initial_x = res
    g_ys, g_final_x = g

    dx_rev = _apply_pallas_mamba_bwd_kernel_raw(
        A, Deltas, Bs, Cs, u, g_ys, N=N, use_euler_barB_approx=use_euler_barB_approx, K=K, scan_dtype=scan_dtype, max_concurrent_steps=max_concurrent_steps
    )

    mulDeltaA = jnp.einsum("bld,dn->bldn", Deltas, A)
    barAs = jnp.exp(mulDeltaA)
    if use_euler_barB_approx:
        barBs = jnp.einsum("bld,bln->bldn", Deltas, Bs)
    else:
        barBs = jnp.expm1(mulDeltaA) * jnp.einsum("dn,bln->bldn", jnp.reciprocal(A), Bs)

    g_u = jnp.einsum("bldn,bldn->bld", barBs, dx_rev * Cs[..., None, :])
    g_Cs = jnp.einsum("bld,bldn->bln", g_ys, dx_rev * barBs)
    g_Bs = jnp.einsum("bld,bldn->bln", u, dx_rev * Deltas[..., None])
    g_Deltas = jnp.einsum("bldn,bldn->bld", A[None, None, ...], dx_rev * barAs)
    g_A = jnp.einsum("bld,bldn->dn", Deltas, dx_rev * barAs)
    g_initial_x = dx_rev[:, 0, ...] * barAs[:, 0, ...]

    return g_A, g_Deltas, g_Bs, g_Cs, g_u, g_initial_x


apply_pallas_mamba_kernel.defvjp(_apply_pallas_mamba_kernel_fwd, _apply_pallas_mamba_kernel_bwd)
