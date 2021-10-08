"""Configurations"""
from typing import NamedTuple, Callable, Optional


class ESNConfig(NamedTuple):
    """A configuration for the ESN.

    :param input_size: dimension of the input
    :param reservoir_size: number of neurons in the reservoir
    :param output_size: dimension of the output
    :param init_weights: function to initialize reservoir weights (W)
        e.g. jax.random.uniform or jax.random.normal
        (key, shape) -> array
    :param init_weights_in: function to initialize input weights (W_in)
        e.g. jax.random.uniform or jax.random.normal
        (key, shape) -> array
    :param rho: desired spectral radius of reservoir weight matrix
        if None, spectral radius is not modified
    :param feedback: whether or not to include feedback connections
        from the output to the reservoir
        defaults to False
    """
    # dimensions
    input_size: int
    reservoir_size: int
    output_size: int
    # weight initializations
    init_weights: Callable
    init_weights_in: Callable
    init_weights_b: Callable
    # sparsity
    init_weights_density: float = 1.0
    init_weights_in_density: float = 1.0
    # arguments to init functions
    init_weights__args: dict = {}
    init_weights_b__args: dict = {}
    init_weights_in__args: dict = {}
    # optional parameters
    feedback: bool = False
    skip_connections: bool = False
    spectral_radius: Optional[float] = None


class TrainingConfig(NamedTuple):
    """A configuration for training the ESN with conceptors.

    :params washout: number of timesteps to wash out from beginning
    :params optimizer_wout: function that initializes an Optimizer object
        (see optimizer.py)
    :params compute_wout__args: arguments to the function (optional)
    :params compute_conceptor: function that computes conceptor
    :params compute_conceptor__args: arguments to the function (optional)
        e.g. aperture
    :params compute_loading: function that computes loaded weight matrix
    :params compute_loading__args: arguments to the function (optional)
        e.g. regularizer
    :params init_states: function that initializes the reservoir state
    :params init_states__arg: argument to the function (optional)
    """
    washout: int
    optimizer_wout: Callable
    compute_conceptor: Callable
    compute_loading: Callable
    init_states: Callable
    init_states__args: dict = {}
    optimizer_wout__args: dict = {}
    compute_loading__args: dict = {}
    compute_conceptor__args: dict = {}
