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
LR = 1e-4
BSZ = 64
TRAIN_STEPS = 400 * 10**3
VALIDATION_INTERVAL = 8192
VALIDATION_LENGTH = 8192
# Converting a JAX scalar to Python synchronizes the device, so refresh the
# displayed training loss periodically instead of on every asynchronous step.
LOSS_DISPLAY_INTERVAL = 128
VOCAB_SIZE = 16


# %%
def selective_copying(rng_key, seq_len=256, vocab_size=VOCAB_SIZE):
    shuffle_key, element_key = random.split(rng_key)
    num_elements = vocab_size - 2
    space_token = vocab_size - 2
    copy_token = vocab_size - 1
    elements = jnp.arange(num_elements)
    num_spaces = seq_len - 2 * num_elements
    prefix = jnp.concatenate((
        elements,
        jnp.full((num_spaces,), space_token),
    ))
    prefix = random.permutation(shuffle_key, prefix)
    inputs = jnp.concatenate((
        prefix,
        jnp.full((num_elements,), copy_token),
    ))
    element_positions = jnp.nonzero(
        prefix != space_token,
        size=num_elements,
    )[0]
    labels = prefix[element_positions]
    return inputs, jax.nn.one_hot(labels, num_classes=16)
def create_batch(key, bsz, seq_len=256, vocab_size=16):
    induction_heads_batch, one_hot_y = jax.vmap(
        partial(selective_copying, seq_len=seq_len, vocab_size=vocab_size)
    )(random.split(key, bsz))
    return induction_heads_batch, one_hot_y
# %%
class MambaSelectiveCopying(nnx.Module):
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
                R=1,
                expand=expand,
                rngs=rngs,
                use_kernel=True,
                use_D_Param=False,
                use_broadcast_Delta=True,
                use_euler_barB_approx=True
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.RMSNorm(D, epsilon=1e-5, rngs=rngs)
        self.proj_down = nnx.Linear(
            in_features=D,
            out_features=vocab_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.uniform(scale=D ** -0.5),
            rngs=rngs,
        )

    def __call__(self, x):
        hidden = self.embed(x)
        for mamba in self.mambas:
            hidden = mamba(hidden)
        return self.proj_down(self.final_norm(hidden))

# %%
rngs = nnx.Rngs(47)
model = MambaSelectiveCopying(rngs=rngs)
graphdef, params = nnx.split(model, nnx.Param)
optimizer = optax.adam(learning_rate=LR)
opt_state = optimizer.init(params)
# %%
@nnx.jit
def train_step(rngs, graphdef, params, opt_state):
    def compute_loss(params, inputs, labels):
        model = nnx.merge(graphdef, params)
        logits = model(inputs)[:,-VOCAB_SIZE]
        loss = jnp.mean(nnx.vmap(lambda logit, label: jnp.mean(optax.losses.safe_softmax_cross_entropy(logit, label)))(logits, labels))
        return loss
    batch_x, batch_y = create_batch(rngs.inputs(), bsz=BSZ)
    loss, grads = jax.value_and_grad(compute_loss)(params, batch_x, batch_y)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# This is deliberately separate from train_step: it gets its own compilation
# for the longer sequence shape and never contributes gradients or updates.
@nnx.jit
def validation_step(batch_rng, graphdef, params):
    batch_x, batch_y = create_batch(
        batch_rng,
        bsz=64,
        seq_len=VALIDATION_LENGTH
    )
    model = nnx.merge(graphdef, params)
    logits = model(batch_x)[:,-VOCAB_SIZE]
    loss = jnp.mean(optax.losses.safe_softmax_cross_entropy(logits, batch_y))
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(batch_y, axis=-1))
    return loss, accuracy


# %%
validation_rng = random.key(1)
with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("Step {task.fields[epoch]:>6}/{task.total}"),
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


for exponent in range(8,16):
    seq_length = 2**exponent
    batch, labels, paddings = create_batch(rngs.inputs(), bsz=num_samples, seq_len=seq_length)
    logits = jnp.argmax(model(batch, paddings)[:, -1], axis=-1)
    reference = jnp.argmax(labels, axis=-1)
    correct_proportion = jnp.sum(logits == reference) / num_samples
    print(f'Seq Len {seq_length} Accuracy: {correct_proportion * 100:.2f}')
