import jax
from flax import nnx
import jax.numpy as jnp
import jax.random as jrand
from jax import lax
import jax.experimental.pallas as pl
from icecream import ic
import jax.experimental.pallas.mosaic_gpu as plgpu
plgpu.CompilerParams(
    lowering_semantics=plgpu.LoweringSemantics.Lane
)


class S6(nnx.Module):
    def __init__(self, rngs:nnx.Rngs, D, N:int=64, R:int=1,
                 complex_ssm:bool=False, use_euler_barB_approx:bool=True, use_log_A_stability_trick:bool=True,
                 use_bf16=False, cache_states=True,
                 use_kernel=False, kernel_seq_len=128):

        self.cache_states = cache_states

        self.euler_barB_approx = use_euler_barB_approx
        self.log_A = use_log_A_stability_trick
        real_dtype = jnp.float32 if not use_bf16 else jnp.bfloat16
        general_dtype = real_dtype if not complex_ssm else jnp.complex64

        # the nth eigenvalue is initialized as -(n+1)
        if not complex_ssm:
            A_init = jnp.log(jnp.arange(N)+1) if self.log_A else -(jnp.arange(N)+1)
            A_init = A_init.astype(real_dtype)
        else:
            A_init = jnp.log(1/2 - jnp.arange(N, dtype=jnp.complex64) * 1j) if self.log_A else -1/2 + jnp.arange(N, dtype=jnp.complex64) * 1j
        # here we have D SSMs applied to every element in the input, so we now broadcast to init all of them
        self.A = nnx.Param(jnp.broadcast_to(A_init[jnp.newaxis,:], (D,) + A_init.shape))

        # the inputs mix to select the multipliers for each SSM
        self.s_B = nnx.Linear(in_features=D, out_features=N, use_bias=False, rngs=rngs, dtype=general_dtype)
        self.s_C = nnx.Linear(in_features=D, out_features=N, use_bias=False, rngs=rngs, dtype=general_dtype)

        # using the shorthand mappings the paper uses to avoid confusion during implementation
        self.tau_Delta = nnx.softplus

        # while this isn't technically necessary, the paper found that this initialization is good for the Delta bias
        def s_Delta_bias_initializer(rng_key, shape, dtype=jnp.float32):
            tau_Delta_inv = lambda x: jnp.log(jnp.expm1(x))
            uniform = jrand.uniform(rng_key, shape, dtype, minval=1e-3, maxval=1e-1).astype(real_dtype)
            return tau_Delta_inv(uniform)

        # Each SSM has a scalar Delta value, despite using vector-valued states (this differs from S5 in this way)
        if R==1:
            self.Linear_1 = nnx.Linear(in_features=D, out_features=1, use_bias=False, rngs=rngs, dtype=real_dtype)
            self.delta_bias = nnx.Param(s_Delta_bias_initializer(rngs.params(), (D,)))
            # broadcast dropped because addition auto-broadcasts the scalars
            self.biased_s_Delta = lambda x: self.delta_bias + self.Linear_1(x)
        elif R>1:
            self.Linear_R = nnx.Linear(in_features=D, out_features=R, use_bias=False, rngs=rngs, dtype=real_dtype)
            self.Linear_Delta = nnx.Linear(in_features=R, out_features=D, use_bias=False, rngs=rngs, bias_init=s_Delta_bias_initializer, dtype=real_dtype)
        else:
            raise ValueError("R must be 1 or greater.")

        self.complex_ssm = complex_ssm

        self.state_caches = nnx.Variable

        self.use_kernel = use_kernel

        self.kernel_seq_len = kernel_seq_len

    def discretize(self, A, Bs, Deltas):
        mulDeltaA = jnp.einsum("bld,dn->bldn", Deltas, A)
        barAs = jnp.exp(mulDeltaA)
        if self.euler_barB_approx:
            barBs = jnp.einsum("bld,bln->bldn", Deltas, Bs)
        else:
            # barBs = jnp.reciprocal(mulDeltaA) * jnp.expm1(mulDeltaA) * jnp.einsum("bld,bln->bldn", Deltas, Bs)
            # slight optimization. we take an elementwise reciprocal of multDeltaA only to multiply by delta again later, so we eliminate the redundancy
            # we're able to do this because we're working with vectors instead of the matrices ZOH was designed for
            barBs =  jnp.expm1(mulDeltaA) * jnp.einsum("dn,bln->bldn", jnp.reciprocal(A), Bs)
        return barAs, barBs

    def binary_operator(Aht_prev, Aht):
        At_prev, ht_prev = Aht_prev
        At, ht = Aht
        return At * At_prev, At * ht_prev + ht

    # def euler_approx_kernel(A_ref, B_block_ref,
    #            Delta_block_ref,
    #            u_block_ref, A_muls_block_ref, xs_block_ref):
    #     delta_SRAM = Delta_block_ref[...]
    #     mulDeltaA = jnp.einsum("ld,dn->ldn", delta_SRAM, A_ref[...])
    #     ic(mulDeltaA.shape)
    #     barAs = jnp.exp(mulDeltaA)
    #     barBs = jnp.einsum("ld,ln->ldn", delta_SRAM, B_block_ref[...])
    #     Bus = barBs * u_block_ref[...][..., None]
    #     As, xs = lax.associative_scan(S6.binary_operator, (barAs, Bus), axis=0)
    #     ic(As.shape)
    #     ic(xs.shape)
    #     A_muls_block_ref[...], xs_block_ref[...] = lax.associative_scan(S6.binary_operator, (barAs, Bus), axis=0)
    #
    # def zoh_kernel(A_ref, B_block_ref,
    #            Delta_block_ref,
    #            u_block_ref, A_muls_block_ref, xs_block_ref):
    #     delta_SRAM = Delta_block_ref[...]
    #     A_SRAM = A_ref[...]
    #     mulDeltaA = jnp.einsum("ld,dn->ldn", delta_SRAM, A_SRAM)
    #     ic(mulDeltaA.shape)
    #     barAs = jnp.exp(mulDeltaA)
    #     barBs =  jnp.expm1(mulDeltaA) * jnp.einsum("dn,ln->ldn", jnp.reciprocal(A_SRAM), B_block_ref[...])
    #     Bus = barBs * u_block_ref[...][..., None]
    #     As, xs = lax.associative_scan(S6.binary_operator, (barAs, Bus), axis=0)
    #     ic(As.shape)
    #     ic(xs.shape)
    #     A_muls_block_ref[...], xs_block_ref[...] = lax.associative_scan(S6.binary_operator, (barAs, Bus), axis=0)


    def apply_with_mosaic_kernel(self, A, Bs, Deltas, Cs, u):

        B, L, D = u.shape
        N = Bs.shape[-1]
        K = self.kernel_seq_len

        chunk_count = (L + K - 1) // K

        def mosiac_kernel(A_ref, B_ref, Delta_ref, C_ref, u_ref,
                          ys_ref, state_ref,
                          A_smem, A_barrier):
            batch_idx = lax.axis_index("batch")
            plgpu.copy_gmem_to_smem(A_ref, A_smem, A_barrier)
            plgpu.barrier_wait(A_barrier)

            B_spec = plgpu.BlockSpec(
                block_shape=(None, K, N),
                index_map=lambda q: (batch_idx, q, 0),
            )

            Delta_spec = plgpu.BlockSpec(
                block_shape=(None, K, D),
                index_map=lambda q: (batch_idx, q, 0),
            )

            C_spec = plgpu.BlockSpec(
                block_shape=(None, K, N),
                index_map=lambda q: (batch_idx, q, 0),
            )

            u_spec = plgpu.BlockSpec(
                block_shape=(None, K, D),
                index_map=lambda q: (batch_idx, q, 0),
            )

            y_spec = plgpu.BlockSpec(
                block_shape=(None, K, D),
                index_map=lambda q: (batch_idx, q, 0),
            )


            def pipeline_body(_,
                              B_block_ref,Delta_block_ref,
                              C_block_ref,u_block_ref,
                              ys_block_ref,
                              carry):
                # euler approx discretization
                delta = Delta_block_ref[...]  # [K, D]
                A_reg = A_smem[...]  # [D, N]
                B_block = B_block_ref[...]  # [K, N]
                u_block = u_block_ref[...]  # [K, D]

                mulDeltaA = (
                        delta[:, :, None]
                        * A_reg[None, :, :]
                )  # [K, D, N]

                barAs = jnp.exp(mulDeltaA)

                barBs = (
                        delta[:, :, None]
                        * B_block[:, None, :]
                )  # [K, D, N]

                Bus = barBs * u_block[:, :, None]
                # delta_SRAM = Delta_block_ref[...]
                # mulDeltaA = jnp.einsum("ld,dn->ldn", delta_SRAM, A_smem[...])
                # ic(mulDeltaA.shape)
                # barAs = jnp.exp(mulDeltaA)
                # barBs = jnp.einsum("ld,ln->ldn", delta_SRAM, B_block_ref[...])
                # Bus = barBs * u_block_ref[...][..., None]
                #Bus = Bus.at[0].set(Bus[0] + barAs[0] * carry)

                mask = (jnp.arange(K) == 0)[:, None, None]
                Bus = Bus + mask * (barAs[0] * carry)[None, :, :]

                def hillis_steele_scan(As, bs):
                    # As, bs: [K, D, N]

                    K = As.shape[0]
                    offset = 1

                    while offset < K:
                        # Shift the previous-stage values right by `offset`.
                        #
                        # The affine identity is:
                        #     A = 1
                        #     b = 0
                        #
                        # so the first `offset` entries remain unchanged.
                        shifted_As = jnp.concatenate(
                            (
                                jnp.ones_like(As[:offset]),
                                As[:-offset],
                            ),
                            axis=0,
                        )

                        shifted_bs = jnp.concatenate(
                            (
                                jnp.zeros_like(bs[:offset]),
                                bs[:-offset],
                            ),
                            axis=0,
                        )

                        # IMPORTANT: left/earlier prefix goes first.
                        As, bs = S6.binary_operator(
                            (shifted_As, shifted_bs),
                            (As, bs),
                        )

                        offset *= 2

                    return As, bs

                # _, xs = lax.associative_scan(S6.binary_operator, (barAs, Bus), axis=0)

                _, xs = hillis_steele_scan(barAs, Bus)
                ys_block_ref[...] = jnp.einsum("ln,ldn->ld", C_block_ref[...], xs)
                return xs[-1]

            pipeline = plgpu.emit_pipeline(
                body=pipeline_body,
                grid=(chunk_count,),
                init_carry=jnp.zeros((D,N), dtype=A.dtype),
                in_specs=(B_spec, Delta_spec, C_spec, u_spec),
                out_specs=(y_spec,)
            )
            last_state = pipeline(B_ref, Delta_ref, C_ref, u_ref, ys_ref)
            state_ref[batch_idx, ...] = last_state

        kernel = plgpu.kernel(
            body=mosiac_kernel,
            out_type=(
                jax.ShapeDtypeStruct.like(u),
                jax.ShapeDtypeStruct((B,D,N), A.dtype)
            ),
            grid=(B,),
            scratch_types=(
                plgpu.SMEM((D, N), A.dtype),
                plgpu.Barrier()
            ),
            grid_names=("batch",),
            # compiler_params=plgpu.CompilerParams(
            #     lowering_semantics=plgpu.LoweringSemantics.Lane
            # )
        )

        return kernel(A, Bs, Deltas, Cs, u)


    # def apply_with_kernel(self, A, Bs, Deltas, u):
    #     needs_padding = u.shape[1] % self.kernel_seq_len != 0
    #     if needs_padding:
    #         pad_len = (u.shape[1] // self.kernel_seq_len + 1) * self.kernel_seq_len - u.shape[1]
    #         pad_dims = (
    #             (0,0),
    #             (0,pad_len),
    #             (0,0),
    #         )
    #         Bs = jnp.pad(Bs, pad_dims)
    #         Deltas = jnp.pad(Deltas, pad_dims)
    #         u = jnp.pad(u, pad_dims)
    #     block_count = u.shape[1] // self.kernel_seq_len
    #     ic(block_count)
    #     B_split = jnp.stack(jnp.split(Bs, block_count, axis=1)).transpose((1,0,2,3))
    #     ic(B_split.shape)
    #     Delta_split = jnp.stack(jnp.split(Deltas, block_count, axis=1)).transpose((1,0,2,3))
    #     ic(Delta_split.shape)
    #     u_split = jnp.stack(jnp.split(u, block_count, axis=1)).transpose((1,0,2,3))
    #     ic(u_split.shape)
    #     if self.euler_barB_approx:
    #         block_As, block_zero_init_xs = nnx.vmap(
    #             lambda B, Delta, u: nnx.vmap(
    #                 lambda B_block, Delta_block, u_block: pl.pallas_call(
    #                     kernel=S6.euler_approx_kernel,
    #                     out_shape=(
    #                         jax.ShapeDtypeStruct((self.kernel_seq_len, u_block.shape[1], B_block.shape[1]), jnp.float32),
    #                         jax.ShapeDtypeStruct((self.kernel_seq_len, u_block.shape[1], B_block.shape[1]), jnp.float32)
    #                     ),
    #                     interpret=True
    #                 )(A.astype(jnp.float32), B_block, Delta_block, u_block)
    #             )(B, Delta, u)
    #         )(B_split, Delta_split, u_split)
    #     else:
    #         block_As, block_zero_init_xs = nnx.vmap(
    #             lambda B, Delta, u: nnx.vmap(
    #                 lambda B_block, Delta_block, u_block: pl.pallas_call(
    #                     kernel=S6.zoh_kernel,
    #                     out_shape=(
    #                         jax.ShapeDtypeStruct((self.kernel_seq_len, u_block.shape[1], B_block.shape[1]),
    #                                              jnp.float32),
    #                         jax.ShapeDtypeStruct((self.kernel_seq_len, u_block.shape[1], B_block.shape[1]), jnp.float32)
    #                     ),
    #                     interpret=True
    #                 )(A.astype(jnp.float32), B_block, Delta_block, u_block)
    #             )(B, Delta, u)
    #         )(B_split, Delta_split, u_split)
    #     ic(block_As.shape)
    #     _, exiting_states = lax.associative_scan(S6.binary_operator, (block_As[:,:,-1], block_zero_init_xs[:,:,-1]), axis=1)
    #     ic(exiting_states.shape)
    #     entering_states = jnp.pad(exiting_states, ((0,0),(1,0),(0,0),(0,0)))[:,:-1]
    #     ic(entering_states.shape)
    #     macro_scanned_xs = block_As * entering_states[:,:, None] + block_zero_init_xs
    #     xs = jnp.reshape(macro_scanned_xs, u.shape + (Bs.shape[2],))
    #     if needs_padding: xs = xs[:,:-pad_len]
    #     return xs


    # @nnx.jit
    def __call__(self, x):
        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        Bs = self.s_B(x)
        Cs = self.s_C(x)
        Deltas = self.tau_Delta(self.biased_s_Delta(x))

        ic(A.shape)

        ic(x.shape)
        ic(Bs.shape)
        ic(Deltas.shape)
        xs=None
        if not self.use_kernel:
            A_bars, B_bars = self.discretize(A, Bs, Deltas)
            Bx = B_bars * x[..., jnp.newaxis]
            _, xs = lax.associative_scan(S6.binary_operator, (A_bars, Bx), axis=1)
            self.state_caches.value = xs[:, -1:, ...]
            ys = jnp.einsum("bln,bldn->bld", Cs, xs)
        else:
            # xs = self.apply_with_kernel(A, Bs, Deltas, x)
            ys, last_states = self.apply_with_mosaic_kernel(A, Bs, Deltas, Cs, x)
            self.state_caches.value = last_states
        ic(xs.shape)

        return ys if not self.complex_ssm else ys.real

    def step(self, token, prev_state=None):
        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        B = self.s_B(token)
        C = self.s_C(token)
        Deltas = self.tau_Delta(self.biased_s_Delta(token))
        A_bar, B_bar = self.discretize(A, B, Deltas)
        if self.state_caches[...] is None and prev_state is None:
            prev_state = jnp.zeros_like(A_bar)
        elif self.state_caches[...] is not None and prev_state is None:
            prev_state = self.state_caches
        x = A_bar * prev_state + B_bar * token[..., jnp.newaxis]
        if self.cache_states:
            self.state_caches.value = x
        y = jnp.einsum("bln,bldn->bld", C, x)
        return y if not self.complex_ssm else y.real

class Mamba(nnx.Module):
    def __init__(self, rngs:nnx.Rngs,
                 in_features:int, out_features:int,
                 D:int, N:int=64, R:int=1,
                 causal_conv_kernel_size:int=4,
                    use_euler_barB_approx:bool=True, complex_ssm:bool=False,
                    use_log_A_stability_trick:bool=True, bf16=False, cache_states=True):
        dtype = jnp.bfloat16 if bf16 else jnp.float32
        self.main_proj_up = nnx.Linear(in_features=in_features, out_features=D, rngs=rngs, dtype=dtype)
        self.skip_proj_up = nnx.Linear(in_features=in_features, out_features=D, rngs=rngs, dtype=dtype)
        self.conv = nnx.Conv(in_features=D, out_features=D, kernel_size=causal_conv_kernel_size, feature_group_count=D,
                             padding="CAUSAL", use_bias=False, rngs=rngs, dtype=dtype)
        self.sigma = nnx.silu
        self.s6 = S6(rngs, D, N=N, R=R,
                     use_euler_barB_approx=use_euler_barB_approx, complex_ssm=complex_ssm,
                     use_log_A_stability_trick=use_log_A_stability_trick, use_bf16=bf16, cache_states=cache_states)
        self.proj_down = nnx.Linear(in_features=D, out_features=out_features, rngs=rngs, dtype=dtype)
        self.cache = nnx.Variable
        self.cache_states = cache_states

    # @nnx.jit
    def __call__(self, x):
        projed = self.main_proj_up(x)
        skip = self.sigma(self.skip_proj_up(x))

        if self.cache_states:
            kernel_size = self.conv.kernel.shape[0]
            self.cache.value = projed[:,-(kernel_size-1):, ...]

        conved = self.sigma(self.conv(projed))
        ssm_out = self.s6(conved)
        muled = ssm_out * skip
        logits = self.proj_down(muled)
        return logits

    def step(self, token):
        projed = self.main_proj_up(token)
        skip = self.sigma(self.skip_proj_up(token))

        if self.cache[...] is None:
            kernel_size = self.conv.kernel.shape[0]
            cache_concat = jnp.pad(projed,
                                   pad_width=(
                                        (0,0),
                                        (kernel_size-1, 0),
                                        (0,0)
                                    )
                                   )
        else:
            cache_concat = jnp.concatenate([self.cache, projed], axis=1)
        self.cache = cache_concat[:,1:,...]

        conved = self.sigma(self.conv(cache_concat)[0,-1:,...])
        ssm_out = self.s6.step(conved)
        muled = ssm_out * skip
        logits = self.proj_down(muled)

        return logits


