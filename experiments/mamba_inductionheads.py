# %%
from pathlib import Path
import sys

# Running this file directly makes Python search `experiments/`, rather than
# the repository root. Add the root so `mamba.mamba1` resolves in Colab and
# from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jnp
from jax import random
from functools import partial
from flax import nnx
from mamba.mamba1 import Mamba
import optax
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


# %%
LR = 1e-3
BSZ = 8
TRAIN_STEPS = 204800
VALIDATION_INTERVAL = 8192
VALIDATION_LENGTH = 8192
# Converting a JAX scalar to Python synchronizes the device, so refresh the
# displayed training loss periodically instead of on every asynchronous step.
LOSS_DISPLAY_INTERVAL = 128


# %%
def generate_induction_heads(rng_key, max_seq_len=256, min_seq_len=None, vocab_size=16):
    length_key, special_key, content_key = random.split(rng_key, 3)

    if min_seq_len is None:
        min_seq_len = max_seq_len

    # Scalar tracer, but it does not determine any array shapes.
    # maxval is exclusive, hence max_seq_len + 1.
    seq_len = random.randint(
        length_key,
        shape=(),
        minval=min_seq_len,
        maxval=max_seq_len + 1,
    )

    special_token = jnp.asarray(vocab_size - 1, dtype=jnp.int32)

    sequence = random.randint(
        content_key,
        shape=(max_seq_len,),
        minval=0,
        maxval=vocab_size - 1,
        dtype=jnp.int32,
    )

    sequence_start = max_seq_len - seq_len
    positions = jnp.arange(max_seq_len)
    padding_mask = positions >= sequence_start

    sequence = jnp.where(padding_mask, sequence, 0)
    sequence = sequence.at[-1].set(special_token)

    special_offset = random.randint(
        special_key,
        shape=(),
        minval=0,
        maxval=seq_len - 2,
    )

    special_index = sequence_start + special_offset
    value = sequence[special_index + 1]
    sequence = sequence.at[special_index].set(special_token)

    return sequence, value, padding_mask


def create_batch(key, bsz, seq_len=256, vocab_size=16, min_seq_len=None):
    induction_heads_batch, values, paddings = jax.vmap(
        partial(generate_induction_heads, max_seq_len=seq_len, min_seq_len=min_seq_len, vocab_size=vocab_size)
    )(random.split(key, bsz))
    one_hot_y = jax.nn.one_hot(values, vocab_size)
    return induction_heads_batch, one_hot_y, paddings[..., None]
# %%
class MambaInductionHeads(nnx.Module):
    def __init__(self, rngs, vocab_size=16, D=64, expand=2, num_layers=2):
        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=D,
            embedding_init=jax.nn.initializers.normal(stddev=0.02),
            rngs=rngs,
        )
        self.mambas = nnx.List([
            Mamba(
                D=D,
                N=16,
                R=16,
                expand=expand,
                rngs=rngs,
                use_kernel=True,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(D, epsilon=1e-5, rngs=rngs)
        self.proj_down = nnx.Linear(
            in_features=D,
            out_features=vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.uniform(scale=D ** -0.5),
            rngs=rngs,
        )

    def __call__(self, x, paddings):
        hidden = self.embed(x)
        for mamba in self.mambas:
            hidden = mamba(hidden, paddings)
        return self.proj_down(self.final_norm(hidden))

# %%
rngs = nnx.Rngs(0)
model = MambaInductionHeads(rngs=rngs)
graphdef, params = nnx.split(model, nnx.Param)
# optimizer = optax.adam(learning_rate=LR)
optimizer = optax.contrib.cocob()
opt_state = optimizer.init(params)
# %%
@nnx.jit
def train_step(rngs, graphdef, params, opt_state):
    def compute_loss(params, inputs, labels, padding_mask):
        model = nnx.merge(graphdef, params)
        logits = model(inputs, padding_mask)[:,-1]
        loss = jnp.mean(optax.losses.safe_softmax_cross_entropy(logits, labels))
        return loss
    batch_x, batch_y, padding_mask = create_batch(rngs.inputs(), bsz=BSZ)
    loss, grads = jax.value_and_grad(compute_loss)(params, batch_x, batch_y, padding_mask)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# This is deliberately separate from train_step: it gets its own compilation
# for the longer sequence shape and never contributes gradients or updates.
@nnx.jit
def validation_step(batch_rng, graphdef, params):
    batch_x, batch_y, padding_mask = create_batch(
        batch_rng,
        bsz=BSZ,
        seq_len=VALIDATION_LENGTH,
        min_seq_len=VALIDATION_LENGTH,
    )
    model = nnx.merge(graphdef, params)
    logits = model(batch_x, padding_mask)[:, -1]
    loss = jnp.mean(optax.losses.safe_softmax_cross_entropy(logits, batch_y))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(batch_y, axis=-1))
    return loss, accuracy


# %%
validation_rng = random.key(1)
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("Epoch {task.fields[epoch]:>6}/{task.total}"),
    TextColumn("Loss {task.fields[loss]}"),
    TextColumn("Val@8192 {task.fields[validation]}"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task(
        "Training...",
        total=TRAIN_STEPS,
        epoch=0,
        loss="--",
        validation="--",
    )
    for step in range(TRAIN_STEPS):
        params, opt_state, loss = train_step(rngs, graphdef, params, opt_state)
        epoch = step + 1
        fields = {"epoch": epoch}
        if epoch % LOSS_DISPLAY_INTERVAL == 0 or epoch == TRAIN_STEPS:
            fields["loss"] = f"{float(loss):.6g}"
        if epoch % VALIDATION_INTERVAL == 0:
            validation_rng, batch_rng = random.split(validation_rng)
            validation_loss, validation_accuracy = validation_step(
                batch_rng, graphdef, params
            )
            fields["validation"] = (
                f"loss={float(validation_loss):.6g}, "
                f"acc={float(validation_accuracy):.2%}"
            )
        progress.update(task, advance=1, **fields)
print(f"Final Loss: {float(loss):.6g}")

# %%
model = nnx.merge(graphdef, params)
# %%
num_samples = 128
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Training Accuracy: {correct_proportion*100:.2f}')

# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=256)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 256 Accuracy: {correct_proportion*100:.2f}')
# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=512)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 512 Accuracy: {correct_proportion*100:.2f}')
# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=1024)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 1024 Accuracy: {correct_proportion*100:.2f}')
# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=2048)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 2048 Accuracy: {correct_proportion*100:.2f}')
# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=4096)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 4096 Accuracy: {correct_proportion*100:.2f}')
# %%
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=8192)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 8192 Accuracy: {correct_proportion*100:.2f}')
# %%
num_samples = 128
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=2**14)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 16384 Accuracy: {correct_proportion*100:.2f}')
# %%
num_samples = 128
batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=2**15)
logits = jnp.argmax(model(batch, paddings)[:,-1], axis=-1)
reference = jnp.argmax(labels, axis=-1)
correct_proportion = jnp.sum(logits==reference)/num_samples
print(f'Seq Len 32768 Accuracy: {correct_proportion*100:.2f}')
