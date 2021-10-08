import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# General helper functions below


def sine(T, dt, a=1.0, b=0.0, c=1.0):
    return a * jnp.sin(b * jnp.arange(0, T, dt) - c * jnp.pi).reshape(-1, 1)


def sparsify(arr, key, density: float):
    return arr.at[jax.random.uniform(key, arr.shape) > density].set(0.)


def v_harvest_states(esn, ut):
    xt, yt = [None]*3, [None]*3
    for i in range(len(ut)):
        xt[i], yt[i] = esn.harvest_states(ut[i])
    return xt, yt


def concatenate_patterns(xt, washout, shift=0):
    return jnp.concatenate([
        xt[i][washout + shift: xt[i].shape[0]+shift, :]
        for i in range(len(xt))
    ])


# Plotting functions below


def plot_states(xt, washout=0, maxT=-1):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2), sharex=True, dpi=200)
    ax.plot(xt[washout: maxT, :], linestyle='-')
    plt.tight_layout()


def plot_states_list(xt, washout=0, maxT=-1):
    fig, axs = plt.subplots(len(xt), 1, figsize=(12, 3), sharex=True, dpi=200)
    for i in range(len(xt)):
        axs[i].plot(xt[i][washout: maxT, :], linestyle='-')
    plt.tight_layout()


def plot_inputs(ut):
    fig, axs = plt.subplots(len(ut), 1, figsize=(12, 3), dpi=200,
                            sharex=True, sharey=True)
    for i in range(len(ut)):
        axs[i].plot(ut[i])
    plt.tight_layout()


def plot_input_output_list(ut: list, yt: list):
    fig, axs = plt.subplots(len(ut), 1, figsize=(12, 3), sharex=True, dpi=200)
    for i in range(len(ut)):
        axs[i].plot(ut[i][10:, :], linestyle='-')
        axs[i].plot(yt[i][10:, :], linestyle='--')
    plt.tight_layout()


def plot_input_output(ut: list, yt: list):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2), sharex=True, dpi=200)
    ax.plot(ut, linestyle='-')
    ax.plot(yt, linestyle='--')
    plt.tight_layout()
