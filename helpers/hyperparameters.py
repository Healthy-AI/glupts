import numpy as np


class hyperparameter():
    def __init__(self, value, range, tuning=True, integer=False, *args, **kwargs):
        self.value = value
        self.range = range
        self.tuning = tuning
        self.integer = integer
        self.N = None

    def get(self):
        return self.value

    def set(self, value, *args, **kwargs):
        if self.integer:
            self.value = int(value)

    def set_random(self, seed):
        if self.tuning:
            np.random.seed(seed)
            random_num = float(np.random.rand())
            param_vals = random_num * (self.range[1] - self.range[0]) + self.range[0]
            self.value = param_vals
            if self.integer:
                self.value = int(param_vals)



