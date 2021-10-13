import json
import os
import pickle
import matplotlib.pyplot as plt
from typing import Any, Callable, Iterable, Tuple
from utils import concatenate_patterns, v_harvest_states
from ..esn import ESN
from .config import ESNConfig, TrainingConfig
try:
    import jax.numpy as np
except ImportError:
    import numpy as np


def pipeline_identity(prng, ut, esn_config: ESNConfig, config: TrainingConfig):
    """
    Trains ESN on identity function.

    :param prng: PRNG, numpy generator
    :param ut: list of input patterns, each of (T, L)
    :param esn_config: ESNConfig tuple
    :param config: TrainingConfig tuple
    """
    esn = ESN(prng, esn_config)

    # pass each pattern through network
    xt, yt = v_harvest_states(esn, ut)
    # concatenate patterns along time
    X = concatenate_patterns(xt, config.washout)
    U = concatenate_patterns(ut, config.washout)

    # compute output weights
    Y_T = U.copy()
    optimizer = config.optimizer_wout(**config.optimizer_wout__args)
    esn.update_weights(X, U, Y_T, optimizer=optimizer)
    # harvest trained states and outputs
    xt_trained, yt_trained = v_harvest_states(esn, ut)

    # load the weight matrix (need shifted X_tilde and B)
    X_ = concatenate_patterns(xt, config.washout, shift=-1)
    B = np.repeat(esn.b, X_.shape[0], axis=1).T
    W_loaded = config.compute_loading(X, X_, B, **config.compute_loading__args)
    W_before_loading = esn.w.copy()
    esn.w = W_loaded.copy()
    # test the loaded reservoir with zero input
    ut_loaded_zero = np.zeros_like(U)
    xt_loaded_zero, yt_loaded_zero = esn.harvest_states(ut_loaded_zero)

    # compute conceptor
    xt_loaded, yt_loaded = v_harvest_states(esn, ut)
    Ci = [
        config.compute_conceptor(xt_loaded_i[config.washout:, :],
                                 **config.compute_conceptor__args)
        for xt_loaded_i in xt_loaded
    ]
    # test the loaded reservoir with the conceptor
    xt_conceptor, yt_conceptor = [], []
    for i in range(len(ut)):
        xt_tmp, yt_tmp = esn.harvest_states(
            np.zeros_like(ut[i]),
            x_init=config.init_states(prng, (esn_config.reservoir_size, 1),
                                      **config.init_states__args),
            C=Ci[i]
        )
        xt_conceptor.append(xt_tmp)
        yt_conceptor.append(yt_tmp)

    return (
        xt, yt,                             # lists
        xt_trained, yt_trained,             # lists
        xt_loaded_zero, yt_loaded_zero,     # single arrays
        xt_loaded, yt_loaded,               # lists
        xt_conceptor, yt_conceptor,         # lists
        W_before_loading, esn               # single objects
    )


def config_to_dict(config):
    d_config = config._asdict()
    for key, val in d_config.items():
        if callable(val):
            mod = val.__module__
            name = val.__name__
            d_config[key] = f'{mod}.{name}'
    return d_config


def run_experiment(config_list: Iterable[Tuple[ESNConfig, TrainingConfig]],
                   ut: Any, folder: str, pipeline: Callable, prng: Any):
    """Runs pipeline with different configurations.

    :param config_list: list of (esnConfig, trainingConfig)
    :param ut: input timeseries
    :param folder: folder for data
    :param pipeline: pipeline function
    :param prng: PRNG, numpy generator
    """
    if os.path.exists(folder):
        print('Folder already exists. Aborting..')
        return
    os.makedirs(folder)

    _, axs = plt.subplots(4, 1, figsize=(4, 8), sharex=True, sharey=True)
    for i in range(4):
        axs[i].plot(ut[i][:20])
    plt.savefig(os.path.join(folder, 'ut.png'), dpi=200)
    plt.close()

    with open(os.path.join(folder, 'ut.pkl'), 'wb') as f:
        pickle.dump(ut, f)

    print(f'Starting experiments. Total number: {len(config_list)}')
    for idx, (esnConfig, trainingConfig) in enumerate(config_list):
        print('Running experiment {:3}/{:3}'.format(
            idx+1, len(config_list)
        ))
        # save hyperparameters to file
        d = {
            'esnConfig': config_to_dict(esnConfig),
            'trainingConfig': config_to_dict(trainingConfig)
        }
        with open(os.path.join(folder, f'exp{idx}.json'), 'w') as f:
            json.dump(d, f, indent=4)
        # run pipeline
        data = pipeline(prng, ut, esnConfig, trainingConfig)
        # save data to file
        with open(os.path.join(folder, f'exp{idx}.pkl'), 'wb') as f:
            pickle.dump(data, f)
