from experiment.experiment import experiment
from models import *
from helpers.score_functions import *
from data import *
from helpers.random_maps import *
import os

if __name__ == '__main__':
    workers = 8
    repetitions = 60
    savepath = os.path.join(os.getenv('HOME'), 'Privileged_Time_Series_Learning')

    models = [linearRegression(),
              LUPTS(),
              LUPTS_RF(random_fourier_features(10), color='#2F6899'),
              LUPTS_RF(random_relu_features(10), color='#CB0040'),
              OLS_RF(random_fourier_features(10), color='#A0CEEA'),
              OLS_RF(random_relu_features(10), color='#FFA797')
              ]

    sf = combined_scores(r2(), mse_loss())

    # AIR QUALITY
    print('Air Quality')
    data_inst = fiveCities.get_all_cities()

    sample_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 900, 1100, 1300, 1500]
    ex = experiment(data_instances=data_inst, models=models, sample_sizes=sample_sizes,
                    seq_lengths=[5], seq_steps=[2], seq_gap=7, test_ratio=0.2,
                    random_seed=100, repetitions=repetitions, score_function=sf, hp_tuning=True)
    ex.run_single_node(savepath, workers)
    ex.visualize_scores()
