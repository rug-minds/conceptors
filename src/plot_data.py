# from optimizer import LinearRegression
# from configs import ESNConfig, TrainingConfig
# from pipeline import pipeline
# from utils import sine
# from conceptor_fun import loading_ridge_report, compute_conceptor
import jax
import os
import pickle
import matplotlib.pyplot as plt


def jax_random_normal_wrapper(key, shape, loc=0., scale=1.):
    return jax.random.normal(key, shape) * scale + loc


_, ax = plt.subplots(1, 1, figsize=(12, 4))
filename = os.path.join('..', 'data', 'experiments', 'fast', 'exp1.pkl')
with open(filename, 'rb') as f:
    data = pickle.load(f)

# (
#     xt, yt,                             # lists
#     xt_trained, yt_trained,             # lists
#     xt_loaded_zero, yt_loaded_zero,     # single arrays
#     xt_loaded, yt_loaded,               # lists
#     xt_conceptor, yt_conceptor,         # lists
#     W_before_loading, esn               # single objects
# )

maxT = 10
yt = data[-3][0][:maxT, :]
ax.set_title('yt_conceptor')
ax.plot(yt)
# xt = data[-3][0][:maxT, :]
# ax.set_title('xt_conceptor')
# ax.plot(xt)
plt.show()
