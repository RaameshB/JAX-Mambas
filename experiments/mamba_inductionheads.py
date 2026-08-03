# %%
import sys
sys.path.append("/content/JAX-Mambas")
# %%
import jax
from jax import numpy as jnp
from jax import random
from functools import partial
from flax import nnx
from mamba.mamba1 import Mamba
import optax
from icecream import ic
# %%
def generate_induction_heads(rng_key, seq_len=256, vocab_size=16):
    special_key, content_key = random.split(rng_key)
    special_token = jnp.array([vocab_size-1])
    sequence = jnp.concat((random.randint(content_key, (seq_len-1,), minval=0, maxval=vocab_size-1), special_token))
    special_loc = random.randint(special_key, (1,), minval=0, maxval=seq_len-2)
    sequence_with_key = sequence.at[special_loc[...]].set(jnp.array(special_token))
    value = sequence_with_key[special_loc[...]+1]
    return sequence_with_key, value[0]


def create_batch(key, bsz, seq_len=256, vocab_size=16):
    induction_heads_batch, values = jax.vmap(
        partial(generate_induction_heads, seq_len=seq_len, vocab_size=vocab_size)
    )(random.split(key, bsz))
    one_hot_y = jax.nn.one_hot(values, vocab_size)
    return induction_heads_batch, one_hot_y
# %%
class MambaInductionHeads(nnx.Module):
    def __init__(self, rngs, vocab_size=16, D=64, expand=2, num_layers=2):
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=D, rngs=rngs)
        self.mambas = nnx.List([Mamba(D=D, expand=expand, rngs=rngs) for _ in range(num_layers)])
        self.proj_down = nnx.Linear(in_features=D, out_features=vocab_size, rngs=rngs)
    def __call__(self, x):
        hidden = self.embed(x)
        for mamba in self.mambas:
            hidden = mamba(hidden)
        return self.proj_down(hidden)
# %%
lr = 1e-3
bsz = 32
train_steps = 50
# %%
rngs = nnx.Rngs(0)
model = MambaInductionHeads(rngs=rngs)
graphdef, params = nnx.split(model, nnx.Param)
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)
# %%
@nnx.jit
def train_step(rngs, graphdef, params, opt_state):
    def compute_loss(params, inputs, labels):
        model = nnx.merge(graphdef, params)
        logits = model(inputs)[:,-1]
        loss = jnp.mean(optax.losses.safe_softmax_cross_entropy(logits, labels))
        return loss
    batch_x, batch_y = create_batch(rngs.inputs(), bsz=bsz)
    loss, grads = jax.value_and_grad(compute_loss)(params, batch_x, batch_y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
# %%
for step in range(train_steps):
    params, opt_state, loss = train_step(rngs, graphdef, params, opt_state)
    print(f'Step: {step}, Loss: {loss}')
# %%

