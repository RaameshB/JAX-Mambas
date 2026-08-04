from flax import nnx
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

from selective_scan_cuda import selective_scan

class S6(nnx.Module):
    def __init__(self, rngs:nnx.Rngs, D, N:int=16, R:int | str="auto",
                 complex_ssm:bool=False, use_euler_barB_approx:bool=True,
                 use_log_A_stability_trick:bool=True, use_bf16=False,
                 kernel_len:int=512, use_kernel:bool=False):

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

        if R == "auto":
            R = max(1, (D + 15) // 16)

        # Each SSM has a scalar Delta value, despite using vector-valued states (this differs from S5 in this way)
        if R==1:
            self.Linear_1 = nnx.Linear(in_features=D, out_features=1, use_bias=False, rngs=rngs, dtype=real_dtype)
            self.delta_bias = nnx.Param(s_Delta_bias_initializer(rngs.params(), (D,)))
            # broadcast dropped because addition auto-broadcasts the scalars
            self.biased_s_Delta = lambda x: self.delta_bias + self.Linear_1(x)
        elif R>1:
            self.Linear_R = nnx.Linear(in_features=D, out_features=R, use_bias=False, rngs=rngs, dtype=real_dtype)
            self.Linear_Delta = nnx.Linear(in_features=R, out_features=D, use_bias=False, rngs=rngs, bias_init=s_Delta_bias_initializer, dtype=real_dtype)
            self.biased_s_Delta = lambda x: self.Linear_Delta(self.Linear_R(x))
        else:
            raise ValueError("R must be 1 or greater.")

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
            barBs =  jnp.expm1(mulDeltaA) * jnp.einsum("dn,bln->bldn", jnp.reciprocal(A), Bs)
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
            Deltas *= jnp.inf * padding_mask

        if self.complex_ssm:
            A_bars, B_bars = self.discretize(A, Bs, Deltas)
            Bx = B_bars * x[..., jnp.newaxis]
            _, xs = lax.associative_scan(S6.binary_operator, (A_bars, Bx), axis=1)
            if self.has_cache:
                self.state_cache.value = xs[:, -1]
            ys = jnp.einsum("bln,bldn->bld", Cs, xs)
        else:
            initial_x = self.state_cache[...] if self.has_cache else None
            ys, final_x = selective_scan(
                A,
                Deltas,
                Bs,
                Cs,
                x,
                initial_x,
                use_euler_barB_approx=self.euler_barB_approx,
                use_cuda=self.use_kernel,
            )
            if self.has_cache:
                self.state_cache.value = final_x

        return ys if not self.complex_ssm else ys.real

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
        return y if not self.complex_ssm else y.real


class Mamba(nnx.Module):
    def __init__(self, rngs:nnx.Rngs,
                 D:int, expand:int, N:int=16, R:int | str="auto",
                 causal_conv_kernel_size:int=4,
                 use_norm=True, norm_type='layernorm',
                    kernel_len=512,
                    use_kernel=False,
                    use_euler_barB_approx:bool=True, complex_ssm:bool=False,
                    use_log_A_stability_trick:bool=True, bf16=False):
        dtype = jnp.bfloat16 if bf16 else jnp.float32
        self.main_proj_up = nnx.Linear(in_features=D, out_features=D*expand, rngs=rngs, dtype=dtype)
        self.skip_proj_up = nnx.Linear(in_features=D, out_features=D*expand, rngs=rngs, dtype=dtype)
        self.conv = nnx.Conv(in_features=D*expand, out_features=D*expand, kernel_size=causal_conv_kernel_size, feature_group_count=D*expand,
                             padding="CAUSAL", use_bias=False, rngs=rngs, dtype=dtype)
        self.sigma = nnx.silu
        self.s6 = S6(rngs, D*expand, N=N, R=R, kernel_len=kernel_len,
                     use_kernel=use_kernel,
                     use_euler_barB_approx=use_euler_barB_approx, complex_ssm=complex_ssm,
                     use_log_A_stability_trick=use_log_A_stability_trick, use_bf16=bf16)
        self.proj_down = nnx.Linear(in_features=D*expand, out_features=D, rngs=rngs, dtype=dtype)
        self.has_cache = False
        self.D = D
        self.expand = expand
        self.has_norm = use_norm
        if use_norm:
            if norm_type == 'layernorm':
                self.norm = nnx.LayerNorm(D, rngs=rngs)
            elif norm_type == 'rmsnorm':
                self.norm = nnx.RMSNorm(D, rngs=rngs)

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

        conved = self.sigma(self.conv(projed))
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
