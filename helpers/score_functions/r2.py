from helpers.score_functions.base import score_fun
import numpy as np
from sklearn.metrics import r2_score

class r2(score_fun):
    def score(self, y_hat, y):
        if len(y.shape) == 3:
            y = np.squeeze(y,1)
        if len(y_hat.shape) == 3:
            y_hat = np.squeeze(y_hat, 1)

        return {'R2': float(r2_score(y, y_hat))}

    def get_score_cols(self):
        return ['R2']