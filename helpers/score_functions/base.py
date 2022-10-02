import abc

class score_fun(abc.ABC):
    @abc.abstractmethod
    def score(self, y_hat, y, *args, **kwargs):
        pass

    def get_score_cols(self):
        return ['Score']
