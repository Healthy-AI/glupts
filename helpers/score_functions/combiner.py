from helpers.score_functions.base import score_fun


class combined_scores(score_fun):
    def __init__(self, *args):
        self.funcs = args
        self.score_cols = []
        for fun in args:
            self.score_cols += fun.get_score_cols()

    def score(self, y_hat, y):
        scores = {}
        for fun in self.funcs:
            s = fun.score(y_hat, y)
            scores = {**scores, **s}
        return scores

    def get_score_cols(self):
        return self.score_cols
