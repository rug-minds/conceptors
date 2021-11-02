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
        np.dot(X_.T, (np.arctanh(X.T)-B.T).T)
    ).T


def ridge_regression(X,Y,regularizer: float = 1e-4):
    """ Generic ridge regression solving W x = y for T couple of vectors
    x and y of respective size Nx and Ny. They are concatenated into 
    matrices X and Y. 
    
    :param X: shape (T, Nx)
    :param Y: shape (T, Ny)
    :param regularizer_w: regularizer for ridge 
    
    :return matrix solution of above equation: array (N, N)
    """
    XTX = np.dot(X.T, X)
    
    w = np.dot(
        np.linalg.inv(XTX + regularizer * np.eye(XTX.shape[0])), 
        np.dot(X.T,Y)).T
    return w


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

def compute_conceptor_diag(X, aperture: float = 10.):
    """Compute conceptors diagonal from state trajectory.

    :param X: array, shape (T, N)
    :param aperture: aperture of conceptor computation, see Jaeger2014
    :return diagonal conceptor: array (N, N) 
    """
    

    R = np.sum(X*X, axis=0) / X.shape[0]
    C = R / (R + aperture ** (-2))
    
    return  np.diag(C)
