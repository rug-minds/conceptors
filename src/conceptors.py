"""Functions for implementing conceptors"""
import jax.numpy as jnp


def loading_ridge_report(X, X_, B, regularizer: float = 1e-4):
    """Solving reservoir loading problem with ridge regression.

    :param X: state trajectory, shape (T, N)
    :param X_: state trajectory, time-shifted by -1
    :param B: bias (T, 1)
    :param regularizer_w: regularizer for ridge regression
    """
    N = X.shape[1]
    return jnp.dot(
        jnp.linalg.inv(jnp.dot(X_.T, X_) + regularizer * jnp.eye(N)),
        jnp.dot(X_.T, jnp.arctanh(X)-B)
    )


def compute_conceptor(X, aperture: float = 10.):
    """Compute conceptors from state trajectory.

    :param X: array, shape (T, N)
    :param aperture: aperture of conceptor computation, see Jaeger2014
    :return conceptor: array (N, N)
    """
    R = jnp.dot(X.T, X) / X.shape[0]
    return jnp.dot(
        R,
        jnp.linalg.inv(
            R + aperture ** (-2) * jnp.eye(R.shape[0])
        )
    )
