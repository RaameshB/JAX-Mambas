pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt jaxlib
pip install -U -y flax optax icecream jax[cuda13] rich
git clone https://github.com/RaameshB/JAX-Mambas.git
cd JAX-Mambas && git switch dev && git pull
python experiments/mamba_inductionheads.py