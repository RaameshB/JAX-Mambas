if __name__ == "__main__":
    from mamba1 import S6
    from flax.nnx import Rngs
    from jax.numpy import isclose
    from icecream import ic

    rngs = Rngs(1)
    test_inp = rngs.uniform((3, 32, 2))

    s6 = S6(rngs, D=2, N=4, use_kernel=True, kernel_seq_len=8)
    kernel_out = s6(test_inp)
    ic(kernel_out)

    rngs = Rngs(1)
    test_inp = rngs.uniform((3, 32, 2))
    s6 = S6(rngs, D=2, N=4, use_kernel=False)
    reference_out = s6(test_inp)
    ic(reference_out)

    assert isclose(kernel_out, reference_out).all()
