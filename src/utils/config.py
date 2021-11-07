"""Configurations"""
from typing import NamedTuple, Callable, Optional


class ESNConfig(NamedTuple):
    """A configuration for the ESN.

    :param input_size: dimension of the input
    :param reservoir_size: number of neurons in the reservoir
    :param output_size: dimension of the output
    :param random_feature_size: size of the expansion when using random 
    feature conceptor.
    
    :param init_weights: function to initialize reservoir weights (W)
        should be a function that exists in np.random (e.g. normal, uniform)
        will be called using a PRNG, passing size=(...) as argument, and
        any additional arguments from init_weights__args.
    :param init_weights_g: function to initialize reservoir weights (G)
        should be a function that exists in np.random (e.g. normal, uniform)
        will be called using a PRNG, passing size=(...) as argument, and
        any additional arguments from init_weights__args.
    :param init_weights_f: function to initialize reservoir weights (F)
        should be a function that exists in np.random (e.g. normal, uniform)
        will be called using a PRNG, passing size=(...) as argument, and
        any additional arguments from init_weights__args.        
    :param init_weights_in: function to initialize input weights (W_in)
        should be a function that exists in np.random (e.g. normal, uniform)
        will be called using a PRNG, passing size=(...) as argument, and
        any additional arguments from init_weights_in__args.
    :param init_weights_b: function to initialize bias (W_b)
        should be a function that exists in np.random (e.g. normal, uniform)
        will be called using a PRNG, passing size=(...) as argument, and
        any additional arguments from init_weights_in_b__args.
    
    :param init_weights_density: density (=1.0-sparsity) of reservoir weights.
    :param init_weights_in_density: density of input weight matrix.
    :param init_weights_g_density: density of G weight matrix.
    :param init_weights_f_density: density of F weight matrix.
    
    :param init_weights__args: dictionary of other arguments that will be
        passed to the init_weights function.
    :param init_weights_in__args: dictionary of other arguments that will be
        passed to the init_weights_in function.
    :param init_weights_b__args: dictionary of other arguments that will be
        passed to the init_weights_b function.
        
    :param architecture: whether or not this esn will be used in architecture
        (multiple layers of esn) or not
    :para top_layer: whether it will be the top layer or not the architecture
    :para random_feature: whether or not it will be a random feature conceptor
    
    :param feedback: whether or not to include feedback connections
        from the output to the reservoir
        defaults to False
    :param skip_connections: whether or not to include skip connections, i.e.
        connections from input layer directly to output layer.
    :param spectral_radius: desired spectral radius of reservoir weight matrix
        if None, spectral radius is not modified
    """
    # dimensions
    input_size: int
    reservoir_size: int
    output_size: int
    random_feature_size: int = 0
    # weight initializations
    init_weights: str = 'normal'
    init_weights_g: str = 'normal'
    init_weights_f: str = 'normal'
    init_weights_in: str = 'normal'
    init_weights_b: str = 'normal'
    # sparsity
    init_weights_density: float = 1.0
    init_weights_g_density: float = 1.0
    init_weights_f_density: float = 1.0
    init_weights_in_density: float = 1.0
    # arguments to init functions
    init_weights__args: dict = {}
    init_weights_g__args: dict = {}
    init_weights_f__args: dict = {}
    init_weights_b__args: dict = {}
    init_weights_in__args: dict = {}
    # architecture parameter
    is_architecture: bool = False
    is_top_layer: bool = False
    is_random_feature: bool = False
    # optional parameters
    has_feedback: bool = False
    skip_connections: bool = False
    spectral_radius: Optional[float] = None
    


class TrainingConfig(NamedTuple):
    """A configuration for training the ESN with conceptors.

    :params washout: number of timesteps to wash out from beginning
    :params optimizer_wout_alpha: float, regularization for ridge regression
    :params init_states: function that initializes the reservoir state
    :params compute_conceptor: function that computes conceptor
    :params compute_loading: function that computes loaded weight matrix
    :params compute_wout__args: arguments to the function (optional)
    :params init_states__arg: argument to the function (optional)
    :params compute_conceptor__args: arguments to the function (optional)
        e.g. aperture
    :params compute_loading__args: arguments to the function (optional)
        e.g. regularizer
    """
    washout: int
    aperture: float
    wout_regularizer: float
    compute_conceptor: Optional[Callable] = None
    compute_loading: Optional[Callable] = None
    init_states: Optional[Callable] = None
    learning_rate_recall: float = 0
    learning_rate_cue: float = 0
    cue: int = 0
    recall: int = 0
    adapt: int = 0
    w_regularizer: float = 0
    d_regularizer: float = 0
    g_regularizer: float = 0
    optimizer_wout_alpha: float = None
    init_states__args: dict = {}
    optimizer_wout__args: dict = {}
    compute_conceptor__args: dict = {}
    compute_loading__args: dict = {}
