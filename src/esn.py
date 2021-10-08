"""Implementation of the echo state network.
"""
# local imports
from __future__ import annotations
from typing import Callable, Optional, Tuple, TypeVar
from configs import ESNConfig
from optimizer import Optimizer, LinearRegression
import jax
import jax.numpy as jnp
# TODO: import numpy as np ?


# type variable for jax array type (varies)
Array = TypeVar('Array')


class ESN:
    def __init__(self, key: Array, config: ESNConfig) -> None:
        """
        Set up ESN and initialize the weight matrices (and bias).

        :param key: JAX PRNG key
        :param config: ESNConfig configuration
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
        self.key = key

        # shortcut
        K, N, L = self.get_sizes()

        # initialize input weights
        args = self.config.init_weights_in__args
        self.w_in = self.config.init_weights_in(key, (N, K), **args)
        if self.config.init_weights_in_density < 1.0:
            density = self.config.init_weights_in_density
            filter = jax.random.uniform(key, self.w_in.shape) > density
            self.w_in = self.w_in.at[filter].set(0.)

        # initialize internal weights
        args = self.config.init_weights__args
        self.w = self.config.init_weights(key, (N, N), **args)
        if self.config.init_weights_density < 1.0:
            density = self.config.init_weights_density
            filter = jax.random.uniform(key, self.w.shape) > density
            self.w = self.w.at[filter].set(0.)

        # initialize bias
        args = self.config.init_weights_b__args
        self.b = self.config.init_weights_b(key, (N, 1), **args)

        # initialize feedback weights (if set, otherwise zero)
        self.w_fb = self.config.init_weights(key, (N, L))\
            if self.feedback else jnp.zeros((N, L))

        # initialize output weights (with possible skip connections)
        # NOTE: the initialization here does not matter (will be trained)
        if self.skip_connections:
            self.w_out = self.config.init_weights(key, (L, N + K))
        else:
            self.w_out = self.config.init_weights(key, (L, N))

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
        current_rho = max(abs(jnp.linalg.eig(self.w)[0]))
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
        x = jnp.zeros((N, 1)) if x_init is None else x_init.copy()
        y = jnp.zeros((L, 1))
        # time loop
        for t in range(T):
            u = ut[t:t+1, :].T
            if collect_states:
                xt.append(x)
            # state update (with or without feedback)
            x = jnp.dot(self.w_in, u) + jnp.dot(self.w, x) + self.b
            if self.feedback:
                x += jnp.dot(self.w_fb, y)
            x = jnp.tanh(x)
            # use conceptor, if given
            if C is not None:
                x = jnp.dot(C, x)
            # compute output
            if self.skip_connections:
                y = jnp.dot(self.w_out, jnp.concatenate([x, u], axis=0))
            else:
                y = jnp.dot(self.w_out, x)
            yt.append(y)
        # collect outputs and reservoir states into matrices
        yt = jnp.concatenate(yt, axis=1).T
        if collect_states:
            xt = jnp.concatenate(xt, axis=1).T
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

    def compute_weights(self, xt: Array, ut: Array, yt_hat: Array,
                        optimizer: Optimizer = LinearRegression()) -> Array:
        """
        Compute updated weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        :return W: weight matrix of size (N, N)
        """
        return optimizer.fit(xt, ut, yt_hat,
                             skip_connections=self.skip_connections)

    def update_weights(self, xt: Array, ut: Array, yt_hat: Array,
                       optimizer: Optimizer = LinearRegression()):
        """
        Compute and update the weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        """
        self.w_out = self.compute_weights(xt, ut, yt_hat, optimizer=optimizer)
