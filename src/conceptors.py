"""Functions for implementing conceptors"""
import numpy as np


def loading_ridge_report(X, X_, B, regularizer: float = 1e-4):
    """Solving reservoir loading problem with ridge regression.

    :param X: state trajectory, shape (T, N)
    :param X_: state trajectory, time-shifted by -1
    :param B: bias (T, 1)
    :param regularizer_w: regularizer for ridge regression
    """
    N = X.shape[1]
    return np.dot(
        np.linalg.inv(np.dot(X_.T, X_) + regularizer * np.eye(N)),
        np.dot(X_.T, np.arctanh(X)-B)
    )


def compute_conceptor(X, aperture: float = 10.):
    """Compute conceptors from state trajectory.

    :param X: array, shape (T, N)
    :param aperture: aperture of conceptor computation, see Jaeger2014
    :return conceptor: array (N, N)
    """
    R = np.dot(X.T, X) / X.shape[0]
    return np.dot(
        R,
        np.linalg.inv(
            R + aperture ** (-2) * np.eye(R.shape[0])
        )
    )
