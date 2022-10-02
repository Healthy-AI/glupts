from abc import ABC
import abc
from .hyperparameters import hyperparameter
import numpy as np


class random_map(ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_hyperparams(self):
        pass

    @abc.abstractmethod
    def sample(self, dim):
        pass

    @abc.abstractmethod
    def apply(self, x):
        pass


class random_relu_features(random_map):
    gamma_defaults = (1, (0.01, 10))
    b_scale_defaults = (1, (0.01, 5))
    w_scale_defaults = (1, (-5, 5))

    def __init__(self, N):
        self.W = None
        self.b = None
        self.name = 'ReLU RF'
        self.hyperparams = ['gamma', 'Random_Feats']
        self.gamma = hyperparameter(*random_relu_features.gamma_defaults, tuning=True)
        self.Random_Feats = hyperparameter(N, (5, 1000), tuning=True)

        self.fitted = False

    def get_hyperparams(self):
        return {name: getattr(self, name) for name in self.hyperparams}

    @staticmethod
    def relu(x):
        negative = x < 0.0
        x[negative] = 0.0
        return x

    def sample(self, dim):
        N = self.Random_Feats.get()
        # Uniform on [-1,1]
        self.W = (np.random.rand(dim, N) - 1 / 2) * 2
        self.b = (np.random.rand(1, N) - 1 / 2) * 2
        self.fitted = True

    def apply(self, x):
        if not self.fitted:
            raise Exception('No random features sampled before using the feature map.')
        gamma = self.gamma.get()
        return random_relu_features.relu(x @ self.W * gamma + self.b * gamma)



class random_fourier_features(random_map):
    gamma_defaults = (0.1, (0.001, 0.2))

    def __init__(self, N):
        self.name = 'Fourier RF'
        self.hyperparams = ['gamma', 'Random_Feats']
        self.gamma = hyperparameter(*random_fourier_features.gamma_defaults, tuning=True)
        self.Random_Feats = hyperparameter(N, (5, 1000), tuning=True)
        self.fitted = False

    def get_hyperparams(self):
        return {name: getattr(self, name) for name in self.hyperparams}

    def sample(self, dim):
        N = self.Random_Feats.get()
        self.W = np.random.randn(dim, N)
        self.b = np.random.rand(1, N) * 2 * np.pi
        self.fitted = True

    def apply(self, x):
        if not self.fitted:
            raise Exception('No random features sampled before using the feature map.')
        assert x.shape[-1] == self.W.shape[0], 'Shape mismatch'
        n = self.Random_Feats.get()
        gamma = self.gamma.get()
        return np.sqrt(2 / n) * np.cos(np.sqrt(2 * gamma) * x @ self.W + self.b)

