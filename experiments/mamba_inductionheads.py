import jax
from jax import numpy as jnp
from jax import random
from functools import partial
from mamba import mamba1

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
    one_hot_x = jax.nn.one_hot(induction_heads_batch, vocab_size, axis=-1)
    one_hot_y = jax.nn.one_hot(values, vocab_size)
    return one_hot_x, one_hot_y




if __name__ == "__main__":
    from icecream import ic
    key = random.key(42)
    ic(create_batch(key, 4, 6, 5))