from flax import nnx
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

from .selective_scan_cuda import selective_scan


def _uniform_initializer(bound):
    """Match torch.nn.Linear/Conv1d's default Kaiming-uniform bounds."""
    def init(key, shape, dtype=jnp.float32):
        return jrand.uniform(
            key,
            shape,
            dtype,
            minval=-bound,
            maxval=bound,
        )

    return init


class S6(nnx.Module):
    def __init__(self, rngs:nnx.Rngs, D, N:int=16, R:int | str="auto",
                 complex_ssm:bool=False, use_euler_barB_approx:bool=True,
                 use_log_A_stability_trick:bool=True, use_bf16=False,
                 kernel_len:int=512, use_kernel:bool=False, use_D_Param=True, use_broadcast_Delta=False):

        # Kept for compatibility with Mamba's current constructor. The scan
        # implementation does not chunk the sequence.
        del kernel_len
        self.use_kernel = use_kernel
        self.N = N

        self.euler_barB_approx = use_euler_barB_approx
        self.log_A = use_log_A_stability_trick
        real_dtype = jnp.float32 if not use_bf16 else jnp.bfloat16
        general_dtype = real_dtype if not complex_ssm else jnp.complex64

        # the nth eigenvalue is initialized as -(n+1)
        if not complex_ssm:
            A_init = jnp.log(jnp.arange(N)+1) if self.log_A else -(jnp.arange(N)+1)
            # The reference implementation keeps A_log in float32 even when
            # the rest of the block is evaluated in lower precision.
            A_init = A_init.astype(jnp.float32)
        else:
            A_init = jnp.log(1/2 - jnp.arange(N, dtype=jnp.complex64) * 1j) if self.log_A else -1/2 + jnp.arange(N, dtype=jnp.complex64) * 1j
        # here we have D SSMs applied to every element in the input, so we now broadcast to init all of them
        self.A = nnx.Param(jnp.broadcast_to(A_init[jnp.newaxis,:], (D,) + A_init.shape))

        # the inputs mix to select the multipliers for each SSM
        x_proj_kernel_init = _uniform_initializer(D ** -0.5)
        self.s_B = nnx.Linear(
            in_features=D,
            out_features=N,
            use_bias=False,
            kernel_init=x_proj_kernel_init,
            rngs=rngs,
            dtype=general_dtype,
        )
        self.s_C = nnx.Linear(
            in_features=D,
            out_features=N,
            use_bias=False,
            kernel_init=x_proj_kernel_init,
            rngs=rngs,
            dtype=general_dtype,
        )

        self.use_D_param = use_D_Param
        if use_D_Param:
            # Like A_log, the intra-SSM skip parameter is always float32.
            self.D = nnx.Param(jnp.ones((D,), dtype=jnp.float32))

        # using the shorthand mappings the paper uses to avoid confusion during implementation
        self.tau_Delta = nnx.softplus

        # while this isn't technically necessary, the paper found that this initialization is good for the Delta bias
        def s_Delta_bias_initializer(rng_key, shape, dtype=jnp.float32):
            log_dt = jrand.uniform(
                rng_key,
                shape,
                jnp.float32,
                minval=jnp.log(1e-3),
                maxval=jnp.log(1e-1),
            )
            dt = jnp.maximum(jnp.exp(log_dt), 1e-4)
            # Numerically stable inverse softplus, matching mamba_simple.py.
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(dtype)

        if R == "auto":
            R = max(1, (D + 15) // 16)

        if R < 1:
            raise ValueError("R must be 1 or greater.")
        elif use_broadcast_Delta and R==1:
            self.Linear_Delta = nnx.Linear(
                in_features=D,
                out_features=1,
                use_bias=True,
                kernel_init=_uniform_initializer(R ** -0.5),
                bias_init=s_Delta_bias_initializer,
                rngs=rngs,
                dtype=real_dtype,
            )
            self.biased_s_Delta = lambda x: jnp.broadcast_to(self.Linear_Delta(x), x.shape[:2]+(D,))
        else:
            self.Linear_R = nnx.Linear(
                in_features=D,
                out_features=R,
                use_bias=False,
                kernel_init=x_proj_kernel_init,
                rngs=rngs,
                dtype=real_dtype,
            )
            self.Linear_Delta = nnx.Linear(
                in_features=R,
                out_features=D,
                use_bias=True,
                kernel_init=_uniform_initializer(R ** -0.5),
                bias_init=s_Delta_bias_initializer,
                rngs=rngs,
                dtype=real_dtype,
            )
            self.biased_s_Delta = lambda x: self.Linear_Delta(self.Linear_R(x))
        self.R = R

        self.complex_ssm = complex_ssm

        self.has_cache = False

    def initialize_state(self, input_shape, state_init_value=None):
        cache_shape = (input_shape[0],) + input_shape[2:] + (self.N,)
        initial_state = (
            jnp.zeros(cache_shape)
            if state_init_value is None
            else state_init_value
        )
        self.state_cache = nnx.Variable(initial_state)
        self.has_cache = True

    def discretize(self, A, Bs, Deltas):
        mulDeltaA = jnp.einsum("bld,dn->bldn", Deltas, A)
        barAs = jnp.exp(mulDeltaA)
        if self.euler_barB_approx:
            barBs = jnp.einsum("bld,bln->bldn", Deltas, Bs)
        else:
            # barBs = jnp.reciprocal(mulDeltaA) * jnp.expm1(mulDeltaA) * jnp.einsum("bld,bln->bldn", Deltas, Bs)
            # slight optimization. we take an elementwise reciprocal of multDeltaA only to multiply by delta again later, so we eliminate the redundancy
            # we're able to do this because we're working with vectors instead of the matrices ZOH was designed for
            barBs = (jnp.expm1(mulDeltaA) / A[None, None, :, :]) * Bs[:, :, None, :]
        return barAs, barBs

    def binary_operator(Aht_prev, Aht):
        At_prev, ht_prev = Aht_prev
        At, ht = Aht
        return At * At_prev, At * ht_prev + ht

    @nnx.jit
    def __call__(self, x, padding_mask=None):
        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        Bs = self.s_B(x)
        Cs = self.s_C(x)
        Deltas = self.tau_Delta(self.biased_s_Delta(x))

        if padding_mask is not None:
            Deltas *= padding_mask

        if self.complex_ssm:
            A_bars, B_bars = self.discretize(A, Bs, Deltas)
            Bx = B_bars * x[..., jnp.newaxis]
            _, xs = lax.associative_scan(S6.binary_operator, (A_bars, Bx), axis=1)
            if self.has_cache:
                self.state_cache.value = xs[:, -1]
            ys = jnp.einsum("bln,bldn->bld", Cs, xs)
        else:
            initial_x = self.state_cache[...] if self.has_cache else None
            # The official scan performs its recurrence in float32 and casts
            # the output back to the input dtype.
            ys, final_x = selective_scan(
                A.astype(jnp.float32),
                Deltas.astype(jnp.float32),
                Bs.astype(jnp.float32),
                Cs.astype(jnp.float32),
                x.astype(jnp.float32),
                initial_x,
                use_euler_barB_approx=self.euler_barB_approx,
                use_cuda=self.use_kernel,
            )
            if self.has_cache:
                self.state_cache.value = final_x

        if self.use_D_param:
            ys += self.D * x

        return ys.astype(x.dtype) if not self.complex_ssm else ys.real

    def step(self, token):
        if not self.has_cache:
            self.initialize_state(token.shape)

        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        B = self.s_B(token)
        C = self.s_C(token)
        Deltas = self.tau_Delta(self.biased_s_Delta(token))
        A_bar, B_bar = self.discretize(A, B, Deltas)
        previous_state = self.state_cache[:, jnp.newaxis, ...]
        x = A_bar * previous_state + B_bar * token[..., jnp.newaxis]
        self.state_cache.value = x[:, -1]
        y = jnp.einsum("bln,bldn->bld", C, x)
        y += self.D * token
        return y.astype(token.dtype) if not self.complex_ssm else y.real


class Mamba(nnx.Module):
    def __init__(self, rngs:nnx.Rngs,
                 D:int, expand:int, N:int=16, R:int | str="auto",
                 causal_conv_kernel_size:int=4,
                 use_norm=True, norm_type='layernorm',
                    kernel_len=512,
                    use_kernel=False,
                    num_layers:int=1,
                    use_euler_barB_approx:bool=True, complex_ssm:bool=False,
                    use_log_A_stability_trick:bool=True, bf16=False,
                    use_D_Param=True, use_broadcast_Delta=False):
        dtype = jnp.bfloat16 if bf16 else jnp.float32
        if R == "auto":
            # The reference computes dt_rank from d_model, not d_inner.
            R = max(1, (D + 15) // 16)

        in_proj_kernel_init = _uniform_initializer(D ** -0.5)
        self.main_proj_up = nnx.Linear(
            in_features=D,
            out_features=D*expand,
            use_bias=False,
            kernel_init=in_proj_kernel_init,
            rngs=rngs,
            dtype=dtype,
        )
        self.skip_proj_up = nnx.Linear(
            in_features=D,
            out_features=D*expand,
            use_bias=False,
            kernel_init=in_proj_kernel_init,
            rngs=rngs,
            dtype=dtype,
        )
        conv_bound = causal_conv_kernel_size ** -0.5
        self.conv = nnx.Conv(in_features=D*expand, out_features=D*expand, kernel_size=causal_conv_kernel_size, feature_group_count=D*expand,
                             padding="CAUSAL", use_bias=True,
                             kernel_init=_uniform_initializer(conv_bound),
                             bias_init=_uniform_initializer(conv_bound),
                             rngs=rngs, dtype=dtype)
        self.sigma = nnx.silu
        self.s6 = S6(rngs, D*expand, N=N, R=R, kernel_len=kernel_len,
                     use_kernel=use_kernel,
                     use_euler_barB_approx=use_euler_barB_approx, complex_ssm=complex_ssm,
                     use_log_A_stability_trick=use_log_A_stability_trick, use_bf16=bf16,
                     use_D_Param=use_D_Param, use_broadcast_Delta=use_broadcast_Delta)
        self.proj_down = nnx.Linear(
            in_features=D*expand,
            out_features=D,
            use_bias=False,
            kernel_init=_uniform_initializer(
                (D * expand) ** -0.5 * num_layers ** -0.5
            ),
            rngs=rngs,
            dtype=dtype,
        )
        self.has_cache = False
        self.D = D
        self.expand = expand
        self.has_norm = use_norm
        if use_norm:
            if norm_type == 'layernorm':
                self.norm = nnx.LayerNorm(D, epsilon=1e-5, rngs=rngs)
            elif norm_type == 'rmsnorm':
                self.norm = nnx.RMSNorm(D, epsilon=1e-5, rngs=rngs)
            else:
                raise ValueError(f"Unsupported norm type: {norm_type!r}")

    def initialize_state(self, input_shape, state_init_value=None):
        kernel_size = self.conv.kernel.shape[0]
        cache_size = kernel_size - 1
        cache_shape = (input_shape[0], cache_size, self.D*self.expand)
        self.cache = nnx.Variable(jnp.zeros(cache_shape))
        if state_init_value is not None:
            cache_concat = jnp.concatenate([self.cache[...], state_init_value], axis=1)
            self.cache.value = (
                cache_concat[:, -cache_size:, ...]
                if cache_size
                else cache_concat[:, :0, ...]
            )
        s6_input_shape = (input_shape[0], input_shape[1], self.D * self.expand)
        self.s6.initialize_state(s6_input_shape)
        self.has_cache = True

    # @nnx.jit
    def __call__(self, x, padding_mask=None):
        res = x

        if self.has_norm:
            x = self.norm(x)

        projed = self.main_proj_up(x)
        skip = self.sigma(self.skip_proj_up(x))

        if padding_mask is not None:
            projed *= padding_mask

        kernel_size = self.conv.kernel.shape[0]
        if self.has_cache:
            cache_size = kernel_size - 1
            cache_concat = jnp.concatenate([self.cache[...], projed], axis=1)
            self.cache.value = (
                cache_concat[:, -cache_size:, ...]
                if cache_size
                else cache_concat[:, :0, ...]
            )
            conv_input = cache_concat
        else:
            conv_input = projed

        conved = self.sigma(self.conv(conv_input)[:, -projed.shape[1]:])

        if padding_mask is not None:
            conved *= padding_mask

        ssm_out = self.s6(conved, padding_mask)
        muled = ssm_out * skip
        logits = self.proj_down(muled)

        logits += res

        return logits

    def step(self, token):

        res = token

        if self.has_norm:
            token = self.norm(token)

        projed = self.main_proj_up(token)
        skip = self.sigma(self.skip_proj_up(token))

        if not self.has_cache:
            self.initialize_state(token.shape)

        cache_concat = jnp.concatenate([self.cache[...], projed], axis=1)
        self.cache.value = cache_concat[:,1:,...]

        conved = self.sigma(self.conv(cache_concat)[:,-1:,...])

        ssm_out = self.s6.step(conved)
        muled = ssm_out * skip
        logits = self.proj_down(muled)

        logits+=res

        return logits
