import datetime

from data.dataset import dataset
import pkg_resources
import pandas as pd
import numpy as np





class mp_traffic(dataset):
    default_target_cols = ['traffic_volume']
    feature_list = ['holiday', 'temp', 'rain_1h', 'snow_1h','clouds_all', 'weather_main',
                    'date_time', 'traffic_volume']
    categorical_features = ['holiday', 'weather_main']
    numerical_features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']

    def __init__(self, target_cols=None):
        if target_cols is None:
            target_cols = mp_traffic.default_target_cols
        self.target_cols = target_cols
        super(mp_traffic, self).__init__()
        self.name = 'Minneapolis_Traffic_Data'

        self.csv_path = pkg_resources.resource_filename('data.minneapolis_traffic',
                                                        'Metro_Interstate_Traffic_Volume.csv')

        # load data
        csv_data = pd.read_csv(self.csv_path)

        # run preprocessing
        self.__preprocess(csv_data)

    def __preprocess(self, csv_data):
        print('Preprocessing Minneapolis Traffic Data...')
        # convert data to datetime format
        data = csv_data[mp_traffic.feature_list].copy()
        data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
        data['date_time'].dt.tz_localize('US/Central', ambiguous='NaT')
        data['weekday'] = data.date_time.dt.weekday


        # grouping together the weather conditions
        idx = data['weather_main'].isin(['Drizzle', 'Rain', 'Squall', 'Thunderstorm'])
        data.loc[idx,'weather_main'] = 'Rain'

        idx = data['weather_main'].isin(['Fog', 'Mist', 'Haze', 'Smoke'])
        data.loc[idx, 'weather_main'] = 'Fog'

        # get rid of outliers which are more six sigma away from the mean
        for column in mp_traffic.numerical_features:
            sigma = np.std(data.loc[:,column])
            mean = np.mean(data.loc[:, column])
            idx = np.abs(data[column] - mean) < 6 * sigma
            data = data.loc[idx,:]
            if np.std(data.loc[:,column]) == 0.0:
                data = data.drop(columns=[column])

        # sort ascending by time
        data = data.sort_values(by='date_time', ascending=True).reset_index(inplace=False, drop=True)

        # make sure the holiday property is assigned to the full day
        gb = data.groupby(pd.Grouper(key='date_time', freq='D'))
        for day, idx in gb.indices.items():
            data.loc[idx, 'holiday'] = data.loc[idx[0], 'holiday']

        # assign different types of holidays to the same label
        data.loc[data['holiday'] != 'None', 'holiday'] = 'holiday'

        # produce dummy columns
        data = pd.get_dummies(data, prefix='DUM', columns=mp_traffic.categorical_features+['weekday'], drop_first=True)
#        data = dataset.prevent_dummy_trap(mp_traffic.categorical_features, data)

        # add time of day information
        hours = data.date_time.dt.hour
        time_of_day_sin = np.sin(hours / 12 * np.pi)
        time_of_day_cos = np.cos(hours / 12 * np.pi)
        data['time_sin'] = time_of_day_sin
        data['time_cos'] = time_of_day_cos

        # take care of duplicates and merge the weather information

        # get list of dummy cols of categorical variables
        dummy_cols = [c for c in data.columns.values.tolist() if c[:3] == 'DUM']

        # go through each hour
        gb = data.groupby(pd.Grouper(key='date_time', freq='H'))

        for group, idx in gb.indices.items():
            if len(idx) > 1:
                ar = np.expand_dims(np.max(data.loc[idx, dummy_cols].values, axis=0), 0)
                data.loc[idx, dummy_cols] = np.repeat(ar, repeats=len(idx), axis=0)
        data = data.drop_duplicates(subset='date_time', ignore_index=True).copy()

        # get time stamp
        data['year'] = data.date_time.dt.year
        data['month'] = data.date_time.dt.month
        data['day'] = data.date_time.dt.day
        data['hour'] = data.date_time.dt.hour
        data.drop(columns='date_time', inplace=True)
        self.preprocessed_data = dataset.create_timestamp(data, ['year', 'month', 'day', 'hour'])

    def prepare_for_experiment(self, seq_lengths, seq_steps, seq_gap, *args):
        super(mp_traffic, self).prepare_for_experiment('timestamp', seq_lengths, seq_steps,
                                                       seq_gap)


