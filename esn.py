"""
TODO:
- washout
- add tests?
"""
from __future__ import annotations
import jax.numpy as jnp


class Optimizer:
    def __init__(self):
        pass

    def fit(self, xt, ut, yt_hat):
        pass


class LinearRegression(Optimizer):
    def __init__(self):
        """Initialize linear regression optimizer."""
        super().__init__()
    
    def fit(self, xt, ut, yt_hat):
        """
        Fit the linear regression.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :return W: weight matrix of size (N, N)
        """
        S = jnp.concatenate([xt, ut], axis=1)
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

    def fit(self, xt, ut, yt_hat):
        """
        Fit the ridge regression.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :return W: weight matrix of size (N, N)
        """
        S = jnp.concatenate([xt, ut], axis=1)
        R = jnp.dot(S.T, S) / xt.shape[0]
        D = yt_hat
        P = jnp.dot(S.T, D)
        w_out = jnp.dot(jnp.linalg.inv(R + self.alpha * jnp.eye(R.shape[0])), P).T
        return w_out


class ESN:
    def __init__(self, key, input_size: int, reservoir_size: int, output_size: int, 
        init_weights: function, rho: float = None, feedback: bool = False) -> None:
        """
        Set up echo state network and initialize the weight matrices (and bias).

        :param input_size: int, input dimension
        :param reservoir_size: int, number of neurons in reservoir
        :param output_size: int, output dimension
        :param init_weights: function, to initialize all weights
            e.g. jax.random.uniform or jax.random.normal
        :param rho: float?, desired spectral radius of reservoir weight matrix
            if None, the spectral radius is not modified.
        :param feedback: bool (False), whether or not to include feedback
            from output to reservoir
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.feedback = feedback

        # PRNG key
        self.key = key

        # shortcut
        K, N, L = self.get_sizes()

        # initialize weights
        self.w_in = init_weights(key, (N, K))
        self.w = init_weights(key, (N, N))
        self.b = init_weights(key, (N, 1))
        self.w_fb = init_weights(key, (N, L)) if feedback else jnp.zeros((N, L))
        self.w_out = init_weights(key, (L, N + K))

        # normalize spectral radius (if desired)
        if rho is not None:
            self.normalize_spectral_radius(rho)
    
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
    
    def _forward(self, ut, x_init = None, collect_states = True):
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
            # compute output
            y = jnp.dot(self.w_out, jnp.concatenate([x, u], axis=0))
            yt.append(y)
        # collect outputs and reservoir states into matrices
        yt = jnp.concatenate(yt, axis=1).T
        if collect_states:
            xt = jnp.concatenate(xt, axis=1).T
            return xt, yt
        else:
            return yt
    
    def harvest_states(self, ut, x_init = None):
        """
        Forward pass for training, collects all reservoir states and outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return xt: (T, N)
        :return yt: (T, L)
        """
        return self._forward(ut, x_init, collect_states=True)
        
    
    def forward(self, ut, x_init = None):
        """
        Forward pass function, only collects and returns outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return yt: (T, L)
        """
        return self._forward(ut, x_init, collect_states=False)

    def compute_weights(self, xt, ut, yt_hat, optimizer: Optimizer = LinearRegression()):
        """
        Compute updated weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        :return W: weight matrix of size (N, N)
        """
        return optimizer.fit(xt, ut, yt_hat)
    
    def update_weights(self, xt, ut, yt_hat, optimizer: Optimizer = LinearRegression()):
        """
        Compute and update the weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        """
        self.w_out = self.compute_weights(xt, ut, yt_hat, optimizer=optimizer)
