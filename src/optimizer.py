"""Optimizers for the output weights of the ESN"""
import jax.numpy as jnp


class Optimizer:
    def __init__(self):
        pass

    def fit(self, xt, ut, yt_hat, skip_connections=False):
        pass


class LinearRegression(Optimizer):
    def __init__(self):
        """Initialize linear regression optimizer."""
        super().__init__()

    def fit(self, xt, ut, yt_hat, skip_connections=False):
        """
        Fit the linear regression.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :return W: weight matrix of size (N, N)
        """
        if skip_connections:
            S = jnp.concatenate([xt, ut], axis=1)
        else:
            S = xt.copy()
        w_out = jnp.dot(jnp.linalg.pinv(S), yt_hat).T
        return w_out


class RidgeRegression(Optimizer):
    def __init__(self, alpha: float = 1e-8):
        """
        Initialize ridge regression optimizer.

        :param alpha: regularization parameter.
        """
        super().__init__()
        self.alpha = alpha

    def fit(self, xt, ut, yt_hat, skip_connections=False):
        """
        Fit the ridge regression.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :return W: weight matrix of size (N, N)
        """
        if skip_connections:
            S = jnp.concatenate([xt, ut], axis=1)
        else:
            S = xt.copy()
        R = jnp.dot(S.T, S) / xt.shape[0]
        D = yt_hat
        P = jnp.dot(S.T, D) / xt.shape[0]
        w_out = jnp.dot(
            jnp.linalg.inv(R + self.alpha * jnp.eye(R.shape[0])),
            P).T
        return w_out
