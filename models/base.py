import abc
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from helpers.parameters import *

class model(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        self._initialize_hyperparams()
        self.color = '#000000'


    def _initialize_hyperparams(self):
        if not hasattr(self, 'hyperparams'):
            self.hyperparams = []
        if not hasattr(self, 'hyperparam_trials'):
            self.hyperparam_trials = DEFAULT_HYPERPARAM_TRIALS

    def get_hyperparams(self):
        return {name: getattr(self, name) for name in self.hyperparams}

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    def get_info(self):
        hyperparams = {h:p.get() for h,p in self.get_hyperparams().items()}
        return hyperparams

    def get_color(self):
        return self.color

    def scaler_fit_transform(self, X, y, scale=None):
        if scale:
            if str(scale).lower() == 'robust':
                self.scaler_X = RobustScaler()
                self.scaler_Y = RobustScaler()
            else:
                self.scaler_X = StandardScaler()
                self.scaler_Y = StandardScaler()


            X = np.reshape(self.scaler_X.fit_transform(np.reshape(X, (-1, X.shape[-1]))), X.shape)
            y = np.reshape(self.scaler_Y.fit_transform(np.reshape(y, (-1, y.shape[-1]))), y.shape)
        return X, y

    def scaler_transform_X(self, X, scale=True):
        if scale:
            X = np.reshape(self.scaler_X.transform(np.reshape(X, (-1, X.shape[-1]))), X.shape)
        return X

    def scaler_inverse_transform_Y(self, y, scale=True):
        if len(y.shape)<3:
            y = np.expand_dims(y,1)
        if scale:
            y_ = np.reshape(self.scaler_Y.inverse_transform(np.reshape(y, (-1, y.shape[-1]))), y.shape)
            return y_
        else:
            return y

    def hyperparameter_search(self, X, y, random_seed, trials=None):

        self._initialize_hyperparams()

        hyperparams = self.get_hyperparams()

        if trials is None:
            trials = self.hyperparam_trials

        if len(hyperparams) > 0:
            kf = KFold(CROSS_VALIDATION_K, shuffle=True, random_state=random_seed)

            np.random.seed(random_seed)
            random_numbers = np.random.randint(0, 2**16,trials*len(hyperparams)).reshape(trials,len(hyperparams))

            best_score = np.inf
            best_trial = 0
            for trial in range(trials):
                for i, (_, h) in enumerate(hyperparams.items()):
                    h.set_random(random_numbers[trial,i])

                mse_sum = 0
                for train_idx, test_idx in kf.split(X, y):
                    self.fit(X[train_idx], y[train_idx])
                    y_hat = self.predict(X[test_idx])
                    if len(y_hat.shape) == 3:
                        y_hat = y_hat[:, 0, :]
                    y_test = y[test_idx, 0, :]
                    mse_sum += mean_squared_error(y_test, y_hat)
                if mse_sum < best_score:
                    best_trial = trial
                    best_score = mse_sum

            for i, (_, h) in enumerate(hyperparams.items()):
                h.set_random(random_numbers[best_trial,i])





