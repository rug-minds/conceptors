from optimizer import LinearRegression
from configs import ESNConfig, TrainingConfig
from pipeline import pipeline_identity
from utils import sine
from conceptors import loading_ridge_report, compute_conceptor
import jax
import os
import pickle


def jax_random_normal_wrapper(key, shape, loc=0., scale=1.):
    return jax.random.normal(key, shape) * scale + loc


# set up hyperparameter list

config_list = []
list_reg_loading = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
list_aperture = [0.5, 1., 2.5, 5., 10., 15., 20.]
for aperture in list_aperture:
    for reg_loading in list_reg_loading:
        config_list.append((
            ESNConfig(
                input_size=1,
                reservoir_size=100,
                output_size=1,
                feedback=False,
                spectral_radius=1.5,
                init_weights=jax_random_normal_wrapper,
                init_weights_b=jax_random_normal_wrapper,
                init_weights_in=jax_random_normal_wrapper,
                init_weights_density=0.1,
                init_weights_in_density=1.0,
                init_weights__args={'loc': 0., 'scale': 1.},
                init_weights_b__args={'loc': 0., 'scale': 0.2},
                init_weights_in__args={'loc': 0., 'scale': 1.5},
            ),
            TrainingConfig(
                washout=100,
                optimizer_wout=LinearRegression,
                compute_conceptor=compute_conceptor,
                compute_loading=loading_ridge_report,
                init_states=jax.random.uniform,
                init_states__args={'minval': -1., 'maxval': 1.},
                optimizer_wout__args={},
                compute_loading__args={'regularizer': reg_loading},
                compute_conceptor__args={'aperture': aperture}
            )
        ))

# set up random key

key = jax.random.PRNGKey(123)

# set up input pattern

T_pattern = 1500
n_pattern = 3
T = T_pattern * n_pattern
dt = 0.5  # before: 0.1
ut = [
    sine(T_pattern*dt, dt, 0.6, 1.0, 1.0),  # before: b=0.5
    sine(T_pattern*dt, dt, 1.0, 1.4, 1.0),  # before: b=1.0
    sine(T_pattern*dt, dt, 1.2, 2.2, 1.0)   # before: b=1.8
]


def ensure_folder_exists(experiment_name):
    folder = '../data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder += '/experiments'
    if not os.path.exists(folder):
        os.makedirs(folder)
    folder += f'/{experiment_name}'
    if os.path.exists(folder):
        print('experiment folder already exists, please choose another name.')
        return None
    else:
        os.makedirs(folder)
        return folder


if __name__ == '__main__':
    experiment_name = input('What is the name of this experiment? ')
    experiment_name = experiment_name.replace(' ', '_')
    folder = ensure_folder_exists(experiment_name)
    if folder is not None:
        print(f'Starting experiments. Total number: {len(config_list)}')
        for idx, (esnConfig, trainingConfig) in enumerate(config_list):
            print('Running experiment {:3}/{:3}'.format(
                idx+1, len(config_list)
            ))
            # save hyperparameters to file
            with open(os.path.join(folder, f'exp{idx}.txt'), 'w') as f:
                f.write(str(esnConfig) + '\n\n' + str(trainingConfig))
            # run pipeline
            data = pipeline_identity(key, ut, esnConfig, trainingConfig)
            # save data to file
            with open(os.path.join(folder, f'exp{idx}.pkl'), 'wb') as f:
                pickle.dump(data, f)
