"""Implementation of the echo state network.
"""
# local imports
from __future__ import annotations
from .utils.config import ESNConfig
from typing import Any, Callable, Optional, Tuple, TypeVar
import numpy as np


# type variable for array type (varies)
Array = TypeVar('Array')


class ESN:
    def __init__(self, config: ESNConfig, prng: Optional[Any]) -> None:
        """
        Set up ESN and initialize the weight matrices (and bias).

        :param config: ESNConfig configuration
        :param prng: numpy random generator
        """
        # save configuration
        self.config = config

        # save some elements of the configuration directly
        self.input_size = config.input_size
        self.reservoir_size = config.reservoir_size
        self.output_size = config.output_size
        self.spectral_radius = config.spectral_radius
        self.feedback = config.feedback
        self.skip_connections = config.skip_connections

        # PRNG key
        if prng is None:
            self.prng = np.random.default_rng()
        else:
            assert isinstance(prng, np.random.Generator)
            self.prng = prng

        # shortcut
        K, N, L = self.get_sizes()

        # initialize input weights
        args = self.config.init_weights_in__args
        fun = getattr(self.prng, self.config.init_weights_in)
        self.w_in = fun(size=(N, K), **args)
        if self.config.init_weights_in_density < 1.0:
            density = self.config.init_weights_in_density
            filt = prng.uniform(size=self.w_in.shape) > density
            self.w_in[filt] = 0.

        # initialize internal weights
        args = self.config.init_weights__args
        fun = getattr(self.prng, self.config.init_weights_in)
        self.w = fun(size=(N, N), **args)
        if self.config.init_weights_density < 1.0:
            density = self.config.init_weights_density
            filt = prng.uniform(size=self.w.shape) > density
            self.w[filt] = 0.

        # initialize bias
        args = self.config.init_weights_b__args
        fun = getattr(self.prng, self.config.init_weights_b)
        self.b = fun(size=(N, 1), **args)

        # initialize feedback weights (if set, otherwise zero)
        fun = getattr(self.prng, self.config.init_weights)
        self.w_fb = fun(size=(N, L)) if self.feedback else np.zeros((N, L))

        # initialize output weights (with possible skip connections)
        # the initialization here does not matter (will be trained)
        size = (L, N+K) if self.skip_connections else (L, N)
        fun = getattr(self.prng, self.config.init_weights)
        self.w_out = fun(size=size)

        # normalize spectral radius (if set)
        if self.spectral_radius is not None:
            self.normalize_spectral_radius(self.spectral_radius)

    def get_sizes(self) -> tuple[int, int, int]:
        """
        Simple helper function to get dimensions of the ESN.
        :return K, N, L: input size, reservoir size, output size
        """
        return self.input_size, self.reservoir_size, self.output_size

    def normalize_spectral_radius(self, rho: float = 1.0) -> None:
        """
        Normalize the reservoir's internal weight matrix to a desired
        spectral radius. This helps to keep the reservoir in a stable
        regime. See [TODO: reference].

        :param rho: desired spectral radius
        """
        # compute current spectral radius
        current_rho = max(abs(np.linalg.eig(self.w)[0]))
        # scale weight matrix to desired spectral radius
        self.w *= rho / current_rho

    def _forward(self, ut: Array, x_init: Optional[Array] = None,
                 collect_states: bool = True, C: Optional[Array] = None)\
            -> Tuple[Array, ...]:
        """
        Forward pass for training, collects all reservoir states and outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return xt: (T, N)
        :return yt: (T, L)
        """
        _, N, L = self.get_sizes()
        T = ut.shape[0]
        if collect_states:
            xt = []
        yt = []
        # initial reservoir state (default: zero)
        x = np.zeros((N, 1)) if x_init is None else x_init.copy()
        y = np.zeros((L, 1))
        # time loop
        for t in range(T):
            u = ut[t:t+1, :].T
            # state update (with or without feedback)
            x = np.dot(self.w_in, u) + np.dot(self.w, x) + self.b
            if self.feedback:
                x += np.dot(self.w_fb, y)
            x = np.tanh(x)
            # use conceptor, if given
            if C is not None:
                x = np.dot(C, x)
            # compute output
            if self.skip_connections:
                y = np.dot(self.w_out, np.concatenate([x, u], axis=0))
            else:
                y = np.dot(self.w_out, x)
            # collect state (if desired) and output
            if collect_states:
                xt.append(x)
            yt.append(y)
        # collect outputs and reservoir states into matrices
        yt = np.concatenate(yt, axis=1).T
        if collect_states:
            xt = np.concatenate(xt, axis=1).T
            return xt, yt
        else:
            return yt

    def harvest_states(self, ut: Array, x_init: Optional[Callable] = None,
                       C: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Forward pass for training, collects all reservoir states and outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return xt: (T, N)
        :return yt: (T, L)
        """
        return self._forward(ut, x_init, collect_states=True, C=C)

    def forward(self, ut: Array, x_init: Optional[Callable] = None,
                C: Optional[Array] = None) -> Array:
        """
        Forward pass function, only collects and returns outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return yt: (T, L)
        """
        return self._forward(ut, x_init, collect_states=False, C=C)

    def _ridge_regression(self, xt: Array, ut: Array, yt_hat: Array,
                          alpha: Optional[float] = None) -> Array:
        if self.skip_connections:
            S = np.concatenate([xt, ut], axis=1)
        else:
            S = xt.copy()
        if alpha is None or alpha == 0.:
            # linear regression
            w_out = np.dot(np.linalg.pinv(S), yt_hat).T
        else:
            # ridge regression
            R = np.dot(S.T, S) / xt.shape[0]
            D = yt_hat
            P = np.dot(S.T, D) / xt.shape[0]
            w_out = np.dot(
                np.linalg.inv(R + alpha * np.eye(R.shape[0])), P).T
        return w_out

    def compute_weights(self, xt: Array, ut: Array, yt_hat: Array,
                        alpha: Optional[float] = None) -> Array:
        """
        Compute updated weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        :return W: weight matrix of size (N, N)
        """
        return self._ridge_regression(xt, ut, yt_hat, alpha=alpha)

    def update_weights(self, xt: Array, ut: Array, yt_hat: Array,
                       alpha: Optional[float] = None):
        """
        Compute and update the weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        """
        self.w_out = self._ridge_regression(xt, ut, yt_hat, alpha=alpha)
