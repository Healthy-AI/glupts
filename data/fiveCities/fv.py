import pandas as pd
import pkg_resources
from data.dataset import dataset
import numpy as np


class fiveCities(dataset):
    # List of cities with available data
    available_cities = ['beijing', 'shanghai', 'shenyang', 'chengdu', 'guangzhou']

    # suffix of data files
    suffix = 'PM20100101_20151231.csv'

    # numerical features to use
    feature_list = ['DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws', 'precipitation', 'Iprec']

    # time columns
    time_cols = ['year', 'month', 'day', 'hour']

    # categorical features to use
    categorical_list = ['season', 'cbwd']

    # default target col
    default_target_cols = ['PM_US Post']

    def __init__(self, city, target_cols=None, impute=False):
        """
        :param cities: string of city name
        """

        if target_cols is None:
            target_cols = fiveCities.default_target_cols
        self.target_cols = target_cols

        super(fiveCities, self).__init__()
        # check city exists
        assert city.lower() in fiveCities.available_cities, "City does not exist."

        # reformat city string to match path
        self.city = city[0].upper() + city[1:].lower()
        self.name = 'Five Cities ' + self.city
        # get path of the csv file corresponding to the city
        self.csv_path = pkg_resources.resource_filename('data.fiveCities', self.city + fiveCities.suffix)

        # load data
        csv_data = pd.read_csv(self.csv_path)

        # run preprocessing
        self.__preprocess(csv_data, impute)

    def __preprocess(self, csv_data, impute):
        print(f'Preprocessing {self.city} Air Quality Data...')

        # select features
        features = set(fiveCities.feature_list + self.target_cols + fiveCities.categorical_list \
                       + fiveCities.time_cols)
        data = csv_data[list(features)]
        num_feats = fiveCities.feature_list + self.target_cols
        for col in num_feats:
            assert col in data.columns.tolist(), 'Column not found'
            mean = np.mean(data[col])
            std = np.std(data[col])
            idx = np.abs(data[col] - mean) > std * 6
            data = data.loc[~idx]



        # If categorical features are missing, the row gets thrown away
        idx = ~np.max(pd.isnull(data[fiveCities.categorical_list]), axis=1)
        data = data.loc[idx, :].copy()
        data['season'] = data['season'].astype(int)  # Guangzou has some issues with a season value missing

        # mean imputation
        if impute:
            columns = set(data.columns.tolist()) - set(fiveCities.categorical_list)
            data = dataset.simple_mean_imputation(data, list(columns))
        else:
            data = data.loc[~np.max(pd.isnull(data), axis=1), :]

        # convert to one hot encodings
        data = pd.get_dummies(data, columns=fiveCities.categorical_list, drop_first=True)
        data['time_of_day_sin'] = np.sin(data['hour'] / 12 * np.pi)
        data['time_of_day_cos'] = np.cos(data['hour'] / 12 * np.pi)

        # add timestamps
        self.preprocessed_data = dataset.create_timestamp(data, fiveCities.time_cols)

    def prepare_for_experiment(self, seq_lengths, seq_steps, seq_gap, *args):
        super(fiveCities, self).prepare_for_experiment('timestamp', seq_lengths, seq_steps, seq_gap)

    @staticmethod
    def get_all_cities(target_cols=None, impute=True):
        sets = []
        for city in fiveCities.available_cities:
            data_obj = fiveCities(city, target_cols, impute)
            sets.append(data_obj)
        return sets


