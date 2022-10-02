from sklearn.linear_model import LinearRegression as skLR
from sklearn.linear_model import Ridge as skRidge
from models.base import model
import numpy as np
import abc
from helpers.hyperparameters import hyperparameter
from helpers.parameters import *
from helpers.helpers import kernel_mat
from helpers.random_maps import random_map

SKL_hyperparam_blacklist = ['rff_scale']

class sklearn_predictor(model):
    @abc.abstractmethod
    def __init__(self, skl_model, *args, **kwargs):
        super(sklearn_predictor, self).__init__()
        self.skl_model_class = skl_model

    def fit(self, X, y, *args, **kwargs):
        hyperparams = self.get_hyperparams()
        skl_hyperparams = {h:p.get() for h,p in hyperparams.items() if h not in SKL_hyperparam_blacklist}
        self.skl_model = self.skl_model_class(**skl_hyperparams)
        X_scaled, y_scaled = self.scaler_fit_transform(X, y, self.get_scaling_status())
        self.skl_model.fit(X_scaled[:, 0, :], y_scaled[:, 0, :])

    def predict(self, X, *args, **kwargs):
        scaling = self.get_scaling_status()
        X_scaled = self.scaler_transform_X(X, scaling)
        y_unscaled = np.expand_dims(self.skl_model.predict(X_scaled[:, 0, :]), 1)
        return self.scaler_inverse_transform_Y(y_unscaled, scaling)

    def get_scaling_status(self):
        if not hasattr(self, 'scaling'):
            return True
        else:
            return self.scaling

    def get_info(self):
        other = super(sklearn_predictor, self).get_info()
        return {**other}


class OLS_RF(model):
    def __init__(self, random_feature_map, color='#CB0040', scaling=True, *args, **kwargs):
        super(OLS_RF, self).__init__()
        self.name = f'OLS {random_feature_map.name}'
        self.__init_args = args
        self.scaling = scaling
        self.__init_kwargs = kwargs
        self.color = color
        assert isinstance(random_feature_map, random_map), 'Not a valid random map class'
        self.random_feature_map = random_feature_map
        self.trained =False
        
    def fit(self, X, y, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        y_ = y_[:, 0, :]
        X_ = X_[:,0]

        self.random_feature_map.sample(X_.shape[-1])
        z = self.random_feature_map.apply(X_)
        sklearn_ols = skLR(fit_intercept=False, copy_X=False)
        sklearn_ols.fit(z, y_)
        self.ols = sklearn_ols
        self.trained = True

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if len(X_.shape) == 3:
            X_ = X_[:, 0, :]
        z = self.random_feature_map.apply(X_)
        y_hat = self.ols.predict(z)
        return self.scaler_inverse_transform_Y(y_hat, self.scaling)

    def get_hyperparams(self):
        return self.random_feature_map.get_hyperparams()

    def hyperparameter_search(self, X, y, random_seed, trials=None):
        N = float(X.shape[0])
        self.random_feature_map.Random_Feats = hyperparameter(0.5 * N, (0.05 * N, 0.8 * N), tuning=True, integer=True)
        super(OLS_RF, self).hyperparameter_search(X, y, random_seed, trials)
    
    def predict_latent(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if not self.trained:
            raise Exception('Must fit model before predicting latent states.')
        predictions = self.random_feature_map.apply(X_)
        return predictions


class linearRegression(sklearn_predictor):

    def __init__(self, *args, color='#6A6A6A', **kwargs):
        self.name = 'OLS'
        super(linearRegression, self).__init__(skLR, *args, **kwargs)
        self.color = color

    def fit(self, X, y, *args, **kwargs):
        hyperparams = self.get_hyperparams()
        skl_hyperparams = {h:p.get() for h,p in hyperparams.items() if h not in SKL_hyperparam_blacklist}
        self.skl_model = self.skl_model_class(fit_intercept = False, **skl_hyperparams)
        X_scaled, y_scaled = self.scaler_fit_transform(X, y, self.get_scaling_status())
        self.skl_model.fit(X_scaled[:, 0, :], y_scaled[:, 0, :])

