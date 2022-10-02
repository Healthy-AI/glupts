from experiment.experiment import experiment
from models import *
from helpers.score_functions import *
from data import *
import os

if __name__ == '__main__':
    workers = 8
    repetitions = 25
    savepath = os.path.join(os.getenv('HOME'), 'Privileged_Time_Series_Learning')

    params = {
        'verbose': False,
        'LeNet': False,
        'batch_size': 30,
        'early_stopping_range': 200,
        'latent_size': 25
    }
    models = [CRL(**params),
              GRL(**params),
              SRL(**params),
              baseline_network(**params),
              generalized_distillation_net(**params, width_teacher=100)
              ]

    sf = combined_scores(r2(), mse_loss())

    data_inst = fiveCities.get_all_cities()
    sample_sizes = [100, 150, 200, 300, 400, 500, 700, 1000, 1200]
    ex = experiment(data_instances=data_inst, models=models, sample_sizes=sample_sizes,
                    seq_lengths=[4], seq_steps=[2], seq_gap=7, test_ratio=0.2,
                    random_seed=100, repetitions=repetitions, score_function=sf, hp_tuning=True)
    ex.run_single_node(savepath, workers)
    ex.visualize_scores()
