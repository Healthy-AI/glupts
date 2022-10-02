from helpers.score_functions.base import score_fun
import numpy as np

class mse_loss(score_fun):
    def score(self, y_hat, y):
        return {'MSE': float(np.mean(((y_hat - y)**2).flatten()))}

    def get_score_cols(self):
        return ['MSE']