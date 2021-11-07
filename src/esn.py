"""Implementation of the echo state network.
"""
# local imports
from __future__ import annotations

from .utils.config import ESNConfig, TrainingConfig
from .conceptors import loading_ridge_regression, compute_conceptor, ridge_regression, compute_conceptor_diag
from .utils import helper


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
        self.has_feedback = config.has_feedback
        self.skip_connections = config.skip_connections
        self.is_architecture = config.is_architecture
        self.is_top_layer = config.is_top_layer
        self.is_random_feature = config.is_random_feature
        self.random_feature_size = config.random_feature_size

        # PRNG key
        assert isinstance(prng, np.random.Generator)
        self.prng = prng

        # shortcut
        K, N, M, L = self.get_sizes()

        # initialize input weights
        args = self.config.init_weights_in__args
        fun = getattr(self.prng, self.config.init_weights_in)
        self.w_in = fun(size=(N, K), **args)
        if self.config.init_weights_in_density < 1.0:
            density = self.config.init_weights_in_density
            filt = prng.uniform(size=self.w_in.shape) > density
            self.w_in[filt] = 0.
            
        # initialize internal weights
        if self.is_random_feature:
            # matrix of weight g
            args = self.config.init_weights_g__args
            fun = getattr(self.prng, self.config.init_weights_g)
            self.g = fun(size=(N, M), **args)
            if self.config.init_weights_density < 1.0:
                density = self.config.init_weights_g_density
                filt = prng.uniform(size=self.w.shape) > density
                self.g[filt] = 0.
            
            # matrix of expansion f
            args = self.config.init_weights_f__args
            fun = getattr(self.prng, self.config.init_weights_f)
            self.f = fun(size=(M, N), **args)
            if self.config.init_weights_density < 1.0:
                density = self.config.init_weights_f_density
                filt = prng.uniform(size=self.w.shape) > density
                self.f[filt] = 0.
            
            # matrix of memory d
            self.d: Optional[Array] = None
            
        else:
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
        self.w_fb = fun(size=(N, L)) if self.has_feedback else np.zeros((N, L))

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
        return self.input_size, self.reservoir_size, self.random_feature_size, self.output_size

    def normalize_spectral_radius(self, rho: float = 1.0) -> None:
        """
        Normalize the reservoir's internal weight matrix to a desired
        spectral radius. This helps to keep the reservoir in a stable
        regime. See [TODO: reference].

        :param rho: desired spectral radius
        """
        if self.is_random_feature:
            # compute current spectral radius
            current_rho = max(abs(np.linalg.eig(np.dot(self.g, self.f))[0]))
            # scale weight matrix to desired spectral radius
            self.f *= np.sqrt(rho / current_rho)
            self.g *= np.sqrt(rho / current_rho)
        else:
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
        _, N, _, L = self.get_sizes()
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
            if self.has_feedback:
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
            yt.append(y)     
            if collect_states:
                xt.append(x)
        # collect outputs and reservoir states into matrices
        yt = np.concatenate(yt, axis=1).T
        
        if collect_states:
            xt = np.concatenate(xt, axis=1).T
            return xt, yt
        else:
            return yt

    def _forwardRFC(self, ut: Array, x_init: Optional[Array] = None, 
                 z_init: Optional[Array] = None, collect_states: bool = True, 
                 C: Optional[Array] = None)\
            -> Tuple[Array, ...]:
        """
        Forward pass for training, collects all reservoir states and outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param z_init: (M, 1)
        :return xt: (T, N)
        :return zt: (T, M)
        :return yt: (T, L)
        """
        _, N, M, L = self.get_sizes()
        T = ut.shape[0]
        if collect_states:
            xt = []
            zt = []
        yt = []
        # initial reservoir state (default: zero)
        x = np.zeros((N, 1)) if x_init is None else x_init.copy()
        z = np.zeros((M, 1)) if z_init is None else z_init.copy()
        y = np.zeros((L, 1))
        # time loop
        for t in range(T):
            u = ut[t:t+1, :].T
            # state update (with or without feedback)
            x = np.dot(self.w_in, u) + np.dot(self.g, z) + self.b
            
            if self.has_feedback:
                x += np.dot(self.w_fb, y)
            x = np.tanh(x)
            
            # use conceptor, if given
            if C is not None:
                z = C * np.dot(self.f,x)
            else:
                z = np.dot(self.f,x)
            # compute output
            if self.skip_connections:
                y = np.dot(self.w_out, np.concatenate([x, u], axis=0))
            else:
                y = np.dot(self.w_out, x)
            yt.append(y)     
            if collect_states:
                xt.append(x)
                zt.append(z)
        # collect outputs and reservoir states into matrices
        yt = np.concatenate(yt, axis=1).T
        
        if collect_states:
            xt = np.concatenate(xt, axis=1).T
            zt = np.concatenate(zt, axis=1).T
            return xt, zt, yt
        else:
            return yt    
    
    def forward(self, ut: Array, x_init: Optional[Callable] = None,
                z_init: Optional[Array] = None,
                C: Optional[Array] = None) -> Array:
        """
        Forward pass function, only collects and returns outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return yt: (T, L)
        """
        if self.is_random_feature:
            return self._forwardRFC(ut, x_init, z_init, collect_states=False, C=C)
        else:
            return self._forward(ut, x_init, collect_states=False, C=C)

    def harvest_states(self, ut: Array, x_init: Optional[Callable] = None, 
                       z_init: Optional[Array] = None,
                       C: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Forward pass for training, collects all reservoir states and outputs.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :return xt: (T, N)
        :return yt: (T, L)
        """
        if self.is_random_feature:
            return self._forwardRFC(ut, x_init, z_init, collect_states=True, C=C)
        else:
            return self._forward(ut, x_init, collect_states=True, C=C)

    def _wout_ridge_regression(self, xt: Array, ut: Array, yt_hat: Array,
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
            R = np.dot(S.T, S) 
            D = yt_hat
            P = np.dot(S.T, D) 
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
        return self._wout_ridge_regression(xt, ut, yt_hat, alpha=alpha)

    def update_weights(self, xt: Array, ut: Array, yt_hat: Array,
                       alpha: Optional[float] = None):
        """
        Compute and update the weights with the given optimizer.

        :param xt: collected reservoir states (T, N)
        :param ut: input (T, K)
        :param yt_hat: desired output (T, L)
        :param optimizer: optimizer object, e.g. linear regression.
        """
        self.w_out = self._wout_ridge_regression(xt, ut, yt_hat, alpha=alpha)
    
    def full_procedure(self, ut, trainingConfig: TrainingConfig):  
        """
        Run the whole procedure of loading and computing the conceptors.
        
        :param ut: (T, K)
        :return Ci: conceptor matrix full (N,N) or scalar (M, 1)

        """
        
        if self.is_random_feature:
            xt, zt, yt = list(zip(*map(self.harvest_states, ut)))
            # concatenate patterns along time
            X = helper.concatenate_patterns(xt, trainingConfig.washout)
            U = helper.concatenate_patterns(ut, trainingConfig.washout)
            Y_T = U.copy()
            self.update_weights(X, U, Y_T, alpha=trainingConfig.wout_regularizer)
            # xt_conceptor, zt_conceptor, yt_trained = list(zip(*map(self.harvest_states, ut)))
            
            Z_ = helper.concatenate_patterns(zt, trainingConfig.washout, shift=-1)
            Z = helper.concatenate_patterns(zt, trainingConfig.washout, shift=0)
            
            D_loaded = ridge_regression(Z_, np.dot(self.w_in, U.T).T, regularizer=trainingConfig.d_regularizer)
            G_loaded = ridge_regression(Z, np.dot(self.g, Z.T).T, regularizer=trainingConfig.g_regularizer)
            self.g = G_loaded + D_loaded.copy()
            
            Ci = [
                np.diag(compute_conceptor_diag(zt_i[trainingConfig.washout:, :], aperture=trainingConfig.aperture)).reshape(-1, 1)
                for zt_i in zt
            ]
            
        else:
    
            xt, yt = list(zip(*map(self.harvest_states, ut)))
            # concatenate patterns along time
            X = helper.concatenate_patterns(xt, trainingConfig.washout)
            U = helper.concatenate_patterns(ut, trainingConfig.washout)
        
            Y_T = U.copy()
            self.update_weights(X, U, Y_T, alpha=trainingConfig.wout_regularizer)
        
            X_ = helper.concatenate_patterns(xt, trainingConfig.washout, shift=-1)
            B = np.repeat(self.b, X_.shape[0], axis=1).T
        
            W_loaded = loading_ridge_regression(X, X_, B, regularizer=trainingConfig.w_regularizer)
            
            self.w = W_loaded.copy()
        
            Ci = [
            compute_conceptor(xt_i[trainingConfig.washout:, :], aperture=trainingConfig.aperture)
            for xt_i in xt
            ]
        return Ci
    
    def full_procedure_diag(self, ut, trainingConfig: TrainingConfig):  
        """
        Run the whole procedure of loading and computing the diagonale conceptors. This procedure is different
        and more involved than the one for classical non-diagonale conceptor. It is inspired by Jong (2021).
        
        :param ut: (T, K)
        :return Ci: conceptor matrix full (N,N) or scalar (M, 1)
        
        """
        # Diagonal "trick": initial biais of the input driven reservoir with random conceptor
        C_diag = []
        for i in range (4):
            diag = self.prng.uniform(size=(self.config.reservoir_size))
            C_diag.append(np.diag(diag))
            
        #Adaptation (changing C)
        x_init = self.prng.uniform(low=-1., high=1., size=(self.reservoir_size, 1))
        xt_conceptor, yt_conceptor = list(zip(*[
            self.harvest_states(ut[i][:trainingConfig.washout+trainingConfig.adapt].copy(), x_init=x_init.copy(), C=C_diag[i])
            for i in range(len(ut))
        ]))
        
        Ci = [
            
            compute_conceptor_diag(xt_i[trainingConfig.washout:, :], aperture=trainingConfig.aperture)
            for xt_i in xt_conceptor
        ]

        #Learning (Wout and W with C_diag on the loop)
        xt_conceptor, yt_conceptor = list(zip(*[
            self.harvest_states(ut[i][trainingConfig.washout+trainingConfig.adapt:].copy(), x_init=xt_conceptor[i][[-1],:].T.copy(), C=Ci[i])
            for i in range(len(ut))
        ]))
        
        ut_learn = [ut[i][trainingConfig.washout+trainingConfig.adapt:] for i in range (len(ut))]

        X = helper.concatenate_patterns(xt_conceptor, 1)
        U = helper.concatenate_patterns(ut_learn, 1)
        Y_T = U.copy() 
        
        X_ = helper.concatenate_patterns(xt_conceptor, 1, shift=-1)
        B = np.repeat(self.b, X_.shape[0], axis=1).T
        
        W_loaded = ridge_regression(X_, (np.dot(self.w, X_.T) + np.dot(self.w_in, U.T)).T, regularizer=trainingConfig.w_regularizer)
        self.w = W_loaded.copy()
        
        self.update_weights(X, U, Y_T, alpha=trainingConfig.wout_regularizer)
        
        return Ci

# Auto-conceptor parts: to factorize (and combined with RCF) after debugged

    def _auto(self, ut: Array, learning_rate: float, aperture: float, x_init: Optional[Array] = None, 
              y_init: Optional[Array] = None, C_init: Optional[Array] = None,
             collect_states: bool = False, recall: bool = False)\
            -> Tuple[Array, ...]:
        """
        Forward pass for cueing, collects all reservoir states, outputs, and conceptors.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param y_init: (L, 1)
        :param C_init: (N, N)
        :param learning_rate: float, learning rate for incremental adapatation of C
        :param aperture: float
        :return xt: (T, N)
        :return yt: (T, L)
        :return Ct: (T*N, N)

        If collect_states == False, only returns the cued conceptor and the last reservoir state vector.
        :return x: (N, 1)
        :return y: (L, 1)
        :return C: (N, N)
        """

        _, N, _, L = self.get_sizes()
        T = ut.shape[0]
        if collect_states:
            xt = []
            yt = []
            Ct = []
        # initial reservoir state (default: zero)
        x = np.zeros((N, 1)) if x_init is None else x_init.copy()
        y = np.zeros((L, 1)) if y_init is None else y_init.copy()
        C = np.zeros((N, N)) if C_init is None else C_init.copy()
        loaded_input = np.zeros((N, 1))
        # time loop
        for t in range(T):
            u = ut[t:t+1, :].T
            # conceptor adaptation
            C = C + learning_rate*( np.dot(x - np.dot(C, x), x.T) - (aperture ** (-2))*C )
            if recall:
                loaded_input = np.dot(self.d, x)
            # state update (with or without feedback)
            x = np.dot(self.w_in, u) + np.dot(self.w, x) + self.b + loaded_input

            if self.has_feedback:
                x += np.dot(self.w_fb, y)
            x = np.tanh(x)
            if recall:
                x = np.dot(C, x)
            # compute output
            if self.skip_connections:
                y = np.dot(self.w_out, np.concatenate([x, u], axis=0))
            else:
                y = np.dot(self.w_out, x)
            
            if collect_states:
                xt.append(x)
                yt.append(y)
                Ct.append(C)

        if collect_states:
            xt = np.concatenate(xt, axis=1).T
            yt = np.concatenate(yt, axis=1).T
            return xt, yt, Ct
        else:
            return x, y, C

    def harvest_cueing(self, ut: Array, learning_rate: float, aperture: float,
               x_init: Optional[Array] = None, y_init: Optional[Array] = None)\
            -> Tuple[Array, Array, Array]:
        """
        Forward pass for cueing, collects all reservoir states, outputs, and conceptors.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param y_init: (L, 1)
        :param learning_rate: float, learning rate for incremental adapatation of C
        :param aperture: float
        :return xt: (T, N)
        :return yt: (T, L)
        :return Ct: (T*N, N)
        """
        return self._auto(ut, learning_rate, aperture, x_init, y_init,
                          collect_states=True, recall=False)
    
    def forward_cueing(self, ut: Array, learning_rate: float, aperture: float,
                       x_init: Optional[Array] = None, y_init: Optional[Array] = None)\
            -> Tuple[Array, Array, Array]:
        """
        Forward pass for cueing, only returns the last reservoir state vector and output vector, and 
        the cue conceptor.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param y_init: (L, 1)
        :param learning_rate: float, learning rate for incremental adapatation of C
        :param aperture: float
        :return x: (N, 1)
        :return y: (L,1)
        :return C: (N, N), cue conceptor
        """
        return self._auto(ut, learning_rate, aperture, x_init, y_init,
                          collect_states=False, recall=False)

    def harvest_recall(self, ut: Array, learning_rate: float, aperture: float,
               x_init: Optional[Array] = None, y_init: Optional[Array] = None,
              C_init: Optional[Array]= None)\
            -> Tuple[Array, Array, Array]:
        """
        Forward pass for recall, collects all reservoir states, outputs, and conceptors.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param y_init: (L, 1)
        :param C_init: (N, N)
        :param learning_rate: float, learning rate for incremental adapatation of C
        :param aperture: float
        :return xt: (T, N)
        :return yt: (T, L)
        :return Ct: (T*N, N)
        """
        return self._auto(ut, learning_rate, aperture, x_init, y_init, C_init,
                          collect_states=True, recall=True)

    def forward_recall(self, ut: Array, learning_rate: float, aperture: float,
                       x_init: Optional[Array] = None, y_init: Optional[Array] = None,
                       C_init: Optional[Array] = None)\
            -> Tuple[Array, Array, Array]:
        """
        Forward pass for recall, only returns the last reservoir state vector and output vector,
        and the recall conceptor.

        :param ut: (T, K)
        :param x_init: (N, 1)
        :param y_init: (L, 1)
        :param C_init: (N, N)
        :param learning_rate: float, learning rate for incremental adapatation of C
        :param aperture: float
        :return x: (N, 1)
        :return y: (L,1)
        :return C: (N, N), recall conceptor
        """
        return self._auto(ut, learning_rate, aperture, x_init, y_init, C_init,
                          collect_states=False, recall=True)
    
    def full_procedure_auto(self, ut, trainingConfig: TrainingConfig):
        
        """
        Run the whole procedure of loading and computing the conceptors.
        
        :param ut: (T, K)
        :return Ci: conceptor matrix full (N,N).
        :return xt_recall: (N,T_recall)
        :return yt_recall: (L,T_recall)

        """
        
        # Initial recording
        xt, yt = list(zip(*map(self.harvest_states, ut)))
        X = np.concatenate([x[trainingConfig.washout:, :] for x in xt])
        U = np.concatenate([u[trainingConfig.washout:, :] for u in ut])
        
        # Learning of the readouts
        
        Y_T = U.copy()
        self.update_weights(X, U, Y_T, alpha=trainingConfig.wout_regularizer)
        # _, yt_trained = list(zip(*map(self.harvest_states, ut)))
        
        # Loading the input
        X_ = helper.concatenate_patterns(xt, trainingConfig.washout, shift=-1)
        
        D_loaded = ridge_regression(X_, np.dot(self.w_in, U.T).T, regularizer=trainingConfig.w_regularizer)
        self.d = D_loaded.copy()
        
        # Washout
        ut0 = [u[:trainingConfig.washout, :] for u in ut] 
        xt_ini, yt_ini = list(zip(*map(self.harvest_states, ut0)))
        
        # Cueing
        ut1 = [u[:trainingConfig.washout+trainingConfig.cue, :] for u in ut]
        xt_cue, yt_cue, Ct_cue = list(zip(*[
            self.harvest_cueing(ut1[i][trainingConfig.washout:,:], trainingConfig.learning_rate_cue, trainingConfig.aperture,
                    x_init=xt_ini[i][-1,:][np.newaxis].T.copy(),
                    y_init=yt_ini[i][-1,:].copy() ) 
            for i in range(len(ut))
        ]))
        
        ut_zero = np.zeros((trainingConfig.recall,1))
        xt_recall, yt_recall, Ct_recall = list(zip(*[
            self.harvest_recall(ut_zero.copy(), trainingConfig.learning_rate_recall, trainingConfig.aperture,
                    x_init=xt_cue[i][-1,:][np.newaxis].T.copy(), 
                    y_init=yt_cue[i][-1,:].copy(),
                    # Conceptor calculated with correlation matrix during cueing
                    #C_init=Ci[i].copy() 
                    # Conceptor calculated during cueing with gradient descent
                    C_init=Ct_cue[i][-1].copy()
                      ) 
            for i in range(len(ut))
        ]))
        
        return xt_recall, yt_recall, Ct_recall 
