from flax import nnx
import jax.numpy as jnp
import jax.random as jrand
from jax import lax

class S6(nnx.Module):
    def __init__(self, rngs:nnx.Rngs, D, N:int=64, R:int=1, complex_ssm:bool=True, use_euler_barB_approx:bool=True, use_log_A_stability_trick:bool=True, use_bf16=False, cache_states=True):

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

        self.state_caches = None

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
    def __call__(self, x):
        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        Bs = self.s_B(x)
        Cs = self.s_C(x)
        Deltas = self.tau_Delta(self.biased_s_Delta(x))
        A_bars, B_bars = self.discretize(A, Bs, Deltas)
        Bx = B_bars * x[..., jnp.newaxis]
        _, xs = lax.associative_scan(S6.binary_operator, (A_bars, Bx), axis=1)
        ys = jnp.einsum("bln,bldn->bld", Cs, xs)

        self.state_caches = xs[:,-1:,...]

        return ys if not self.complex_ssm else ys.real / 2
    def step(self, token, prev_state=None):
        A = -jnp.exp(self.A.real) + (self.A.imag * 1j if self.complex_ssm else 0) if self.log_A else self.A
        B = self.s_B(token)
        C = self.s_C(token)
        Deltas = self.tau_Delta(self.biased_s_Delta(token))
        A_bar, B_bar = self.discretize(A, B, Deltas)
        if self.state_caches is None and prev_state is None:
            prev_state = jnp.zeros_like(A_bar)
        elif self.state_caches is not None and prev_state is None:
            prev_state = self.state_caches
        x = A_bar * prev_state + B_bar * token[..., jnp.newaxis]
        if self.cache_states:
            self.state_caches = x
        y = jnp.einsum("bln,bldn->bld", C, x)
        return y

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
        self.cache = None


    @nnx.jit
    def __call__(self, x):
        projed = self.main_proj_up(x)
        skip = self.sigma(self.skip_proj_up(x))

        if self.cache_states:
            kernel_size = self.conv.kernel.shape[0]
            self.cache = projed[:,-(kernel_size-1):, ...]

        conved = self.sigma(self.conv(projed))
        ssm_out = self.s6(conved)
        muled = ssm_out * skip
        logits = self.proj_down(muled)
        return logits

    def step(self, token):
        projed = self.main_proj_up(token)
        skip = self.sigma(self.skip_proj_up(token))

        if self.cache is None:
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


