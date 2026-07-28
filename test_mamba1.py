if __name__ == "__main__":
    from mamba1 import S6
    from flax.nnx import Rngs
    from jax.numpy import isclose
    from icecream import ic

    D = 16
    N = 8
    K = 32
    L = 32

    rngs = Rngs(1)
    test_inp = rngs.uniform((3, L, D))
    s6 = S6(rngs, D=D, N=N, use_kernel=False)
    reference_out = s6(test_inp)
    # ic(reference_out)

    rngs = Rngs(1)
    test_inp = rngs.uniform((3, L, D))
    s6 = S6(rngs, D=D, N=N, use_kernel=True, kernel_seq_len=K)
    kernel_out = s6(test_inp)
    # ic(kernel_out)

    assert isclose(kernel_out, reference_out).all()

    print("passed!")
