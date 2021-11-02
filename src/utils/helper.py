import numpy as np


def n_periodic(n, T, key, minval=-1., maxval=1., snr=0.):
    arr = key.uniform(low=minval, high=maxval, size=(n,))
    arr -= arr.min()
    arr /= arr.max()
    arr = (arr * 2. - 1.) * 0.9
    arr = np.tile(arr, T // n + 1)[:T].reshape(-1, 1)
    if snr != 0:
       arr = arr + np.random.normal(0, snr, arr.shape)
    return arr 

def n_periodic_jong(nbr ,n, T, snr=0):
    if nbr == 0:
        arr = np.array([-0.9,-0.8,0.5,0.1,0.9])
    if nbr == 1:
        arr = np.array([-0.9,-0.5,0.5,0.4,0.9])
    if snr != 0:
       arr = arr + np.random.normal(0, snr, arr.shape)
    return np.tile(arr, T // n + 1)[:T].reshape(-1, 1)


def n_sine(n, T, amplitude=1., phase=0., snr=0.):
    x = np.arange(0, T) * 2 * np.pi / n
    arr = amplitude * np.sin(x - 2 * np.pi * phase).reshape(-1, 1)
    if snr != 0:
       arr = arr + np.random.normal(0, snr, arr.shape)
    return arr


def concatenate_patterns(xt, washout, shift=0):
    return np.concatenate([
        xt[i][washout + shift: xt[i].shape[0]+shift, :]
        for i in range(len(xt))
    ])

def testLRMSE(ut, yt_trained, d_shift, K):
    """
    Compute the error between ut and yt. It's calculated on two windows of size
    K, one at the begining and one at the end. The phase is compensated by 
    shifting ut in the range [0,d_shift].
    """
    RMSE = []
    d = []
    for u,y in zip(ut,yt_trained):
        rmse_ini = [1/(K)*
        sum([ float(u[(i+d)]-y[i]) **2 for i in range (K)])  
        for d in range (d_shift)
        ]
        
        d.append(np.argmin(rmse_ini))
        
        rmse_final = [1/(K)*
        sum([ float(u[(i+d)]-y[i]) **2 for i in range (len(y)-d_shift-K,len(y)-d_shift)])  
        for d in range (d_shift)
        ]
        
        RMSE.append(min(rmse_ini)+min(rmse_final))
    return RMSE,d