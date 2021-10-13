import numpy as np


def n_periodic(n, T, key, minval=-1., maxval=1.):
    arr = key.uniform(low=minval, high=maxval, size=(n,))
    arr -= arr.min()
    arr /= arr.max()
    arr = arr * 2. - 1.
    return np.tile(arr, T // n + 1)[:T].reshape(-1, 1)


def n_sine(n, T, amplitude=1., phase=0.):
    x = np.arange(0, T) * 2 * np.pi / n
    return amplitude * np.sin(x - 2 * np.pi * phase).reshape(-1, 1)
