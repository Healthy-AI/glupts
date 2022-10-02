from models.base import model
from sklearn.linear_model import LinearRegression as skLR
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import colour
from sklearn.metrics import pairwise_kernels
from numpy.linalg import pinv
from helpers.hyperparameters import hyperparameter
from helpers.parameters import *
from helpers.helpers import kernel_mat
from helpers.random_maps import random_map

class LUPTS_custom_kernel(model):
    def __init__(self, kernel, color='#BB8FCE', scaling=True, *args, **kwargs):
        super(LUPTS_custom_kernel, self).__init__()
        self.__init_args = args
        self.scaling = scaling
        self.__init_kwargs = kwargs
        self.name = 'LuPTS+ ' + kernel.__name__
        self.kernel = kernel
        self.color = color

    def fit(self, X, y, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        y_ = y_[:, 0, :]
        n, T, f = X_.shape
        self.x0 = X_[:, 0]
        self.x0_pinv = np.linalg.pinv(kernel_mat(self.kernel, self.x0))
        self.alpha = np.eye(n)
        for t in range(1, T):
            K = kernel_mat(self.kernel, X_[:, t])
            self.alpha = self.alpha @ K @ np.linalg.pinv(K)
        self.alpha = self.alpha @ y_

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if len(X_.shape) == 3:
            X_ = X_[:, 0, :]
        Knew = kernel_mat(self.kernel, self.x0, X_)
        y_hat = (self.alpha.T @ self.x0_pinv @ Knew).T
        return self.scaler_inverse_transform_Y(y_hat, self.scaling)


class LUPTS_custom_map(model):
    def __init__(self, map, color='#BB8FCE', scaling=True, *args, **kwargs):
        super(LUPTS_custom_kernel, self).__init__()
        self.__init_args = args
        self.scaling = scaling
        self.__init_kwargs = kwargs
        self.name = 'LuPTS+ ' + map.__name__
        self.map = map
        self.color = color

    def kernel_mat(self, x, y=None):
        if y is None:
            return self.map(x) @ self.map(x).T
        else:
            return self.map(x) @ self.map(y).T

    def fit(self, X, y, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        y_ = y_[:, 0, :]
        n, T, f = X_.shape
        self.x0 = X_[:, 0]
        self.x0_pinv = np.linalg.pinv(self.kernel_mat(self.x0))
        self.alpha = np.eye(n)
        for t in range(1, T):
            K = self.kernel_mat(X_[:, t])
            self.alpha = self.alpha @ K @ np.linalg.pinv(K)
        self.alpha = self.alpha @ y_

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if len(X_.shape) == 3:
            X_ = X_[:, 0, :]
        Knew = self.kernel_mat(self.x0, X_)
        y_hat = (self.alpha.T @ self.x0_pinv @ Knew).T
        return self.scaler_inverse_transform_Y(y_hat, self.scaling)


class LUPTS_RF(model):
    def __init__(self, random_feature_map, color='#CB0040', scaling=True, *args, **kwargs):
        super(LUPTS_RF, self).__init__()
        self.name = f'LuPTS {random_feature_map.name}'
        assert isinstance(random_feature_map, random_map), 'Not a valid random map class'
        self.random_feature_map = random_feature_map
        self.__init_args = args

        self.scaling = scaling
        self.__init_kwargs = kwargs
        self.color = color
        self.trained = False

    def get_hyperparams(self):
        return self.random_feature_map.get_hyperparams()

    def hyperparameter_search(self, X, y, random_seed, trials=None):
        N = float(X.shape[0])
        self.random_feature_map.Random_Feats = hyperparameter(0.5*N, (0.05*N, 0.8*N), tuning=True, integer=True)
        super(LUPTS_RF, self).hyperparameter_search(X, y, random_seed, trials)


    def fit(self, X, y, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        y_ = y_[:, 0, :]
        n, T, f = X_.shape

        self.random_feature_map.sample(X_.shape[-1])
        z = self.random_feature_map.apply(X_)
        coefficients = []
        sklearn_ols = skLR(fit_intercept=False, copy_X=False)

        for t in range(T-1):
            sklearn_ols.fit(z[:,t], z[:,t+1])
            coefficients.append(sklearn_ols.coef_.copy())
        sklearn_ols.fit(z[:,-1], y_)
        coefficients.append(sklearn_ols.coef_.copy())

        self.coefficients = coefficients.pop(0).T
        for c in coefficients:
            self.coefficients = self.coefficients @ c.T
        self.trained = True

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if len(X_.shape) == 3:
            X_ = X_[:, 0, :]
        y_hat = self.random_feature_map.apply(X_) @ self.coefficients
        return self.scaler_inverse_transform_Y(y_hat, self.scaling)

    
    def predict_latent(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if not self.trained:
            raise Exception('Must fit model before predicting latent states.')
        predictions = self.random_feature_map.apply(X_)
        return predictions
        
class LUPTS(model):
    def __init__(self, stationary=False, color='#1B1B1B', scaling=True, *args, **kwargs):
        super(LUPTS, self).__init__()
        self.__init_args = args
        self.scaling = scaling
        self.__init_kwargs = kwargs
        self.stationary = stationary
        self.name = 'LuPTS (Linear)'
        self.color = color
        if stationary:
            color = colour.Color(hex=self.color)
            color.set_luminance(color.get_luminance() * 0.5)
            self.color = color.get_hex().upper()
            self.name += ' stat'
        self.stationary = stationary

    def fit(self, X, y, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)

        coefficients = None
        if len(y_.shape) == 3:
            y_ = y_[:, 0, :]

        if not self.stationary:
            T = X_.shape[1] - 1
            for t in range(X_.shape[1]):

                lin_mod = skLR(fit_intercept=False, copy_X=False)

                if t < T:
                    lin_mod.fit(X_[:, t, :], X_[:, t + 1, :])
                else:
                    lin_mod.fit(X_[:, t, :], y_)

                if t == 0:
                    coefficients = lin_mod.coef_
                else:
                    coefficients = lin_mod.coef_ @ coefficients



        else:
            lin_mod = skLR(fit_intercept=False, copy_X=False)
            inputs = np.reshape(X_[:, :-1, :], (-1, X_.shape[-1]))
            targets = np.reshape(X_[:, 1:, :], (-1, X_.shape[-1]))

            # fit the coefficients of the square matrix
            lin_mod.fit(inputs, targets)
            A = lin_mod.coef_

            lin_mod = skLR(fit_intercept=False, copy_X=False)
            lin_mod.fit(X_[:, -1, :], y_)
            beta = lin_mod.coef_
            self.b = lin_mod.intercept_
            coefficients = beta @ A

        self.W = coefficients

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if len(X_.shape) == 3:
            X_ = X_[:, 0, :]
        y_hat = np.einsum('oi, bi-> bo', self.W, X_)  # + self.b
        y_hat = np.expand_dims(y_hat, 1)
        return self.scaler_inverse_transform_Y(y_hat, self.scaling)

    def get_info(self):
        other = super(LUPTS, self).get_info()
        return {**other, 'stationary': self.stationary}

