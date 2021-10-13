try:
    import jax
    import jax.numpy as np
except ImportError:
    import numpy as np


def n_periodic(n, T, key, minval=-1., maxval=1.):
    try:
        # jax
        arr = jax.random.uniform(key, shape=(n,), minval=minval, maxval=maxval)
    except NameError:
        # numpy
        arr = key.uniform(low=minval, high=maxval, size=(n,))
    return np.tile(arr, T // n + 1)[:T].reshape(-1, 1)


def n_sine(n, T, amplitude=1., phase=0.):
    x = np.arange(0, T) * 2 * np.pi / n
    return amplitude * np.sin(x - 2 * np.pi * phase).reshape(-1, 1)


def concatenate_patterns(xt, washout, shift=0):
    return np.concatenate([
        xt[i][washout + shift: xt[i].shape[0]+shift, :]
        for i in range(len(xt))
    ])


def jax_random_normal_wrapper(key, shape, loc=0., scale=1.):
    return jax.random.normal(key, shape) * scale + loc


def numpy_random_normal_wrapper(key, shape, loc=0., scale=1.):
    return key.normal(loc=loc, scale=scale, size=shape)
