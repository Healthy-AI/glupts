import datetime
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import abc
from tqdm import tqdm


class dataset(abc.ABC):
    def __init__(self):
        self.preprocessing_settings = locals()
        self.time = datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def create_timestamp(dataframe, columns, drop_cols=True):
        """
        Creates hour time stamp based on year, month, day, hour columns. Columns must be in the correct order.
        :param dataframe: pandas dataframe including the required columns
        :param columns: names of the columns corresponding to year, month, day, hour
        :return: dataframe with added timestamp column
        """
        tz = datetime.timezone.utc
        assert isinstance(dataframe, pd.DataFrame), f'Dataframe expected, got {type(dataframe)}'
        timestamp_fun = lambda row: datetime.datetime(year=int(row[0]),
                                                      month=int(row[1]),
                                                      day=int(row[2]),
                                                      hour=int(row[3]),
                                                      tzinfo=tz).timestamp()
        # convert to hours in integers and start at zero
        timestamps = (dataframe[columns].apply(timestamp_fun, axis=1) / 3600).astype(int)
        timestamps = timestamps - np.min(timestamps)

        # check that there are no duplicates
        assert np.sum(timestamps.duplicated()) == 0, 'Duplicates found in timestamps'

        # add column to the dataframe and return
        dataframe['timestamp'] = timestamps
        if drop_cols:
            dataframe = dataframe.drop(columns=columns)
        return dataframe

    @staticmethod
    def filter_sequences(length, timestep_h, dataframe, index_col, distance_h=0):
        """
        filters out only the data corresponding to a full sequence that has the required length (3 means one
        baseline point, one intermediate timestep and one target step) with the required step size
        :param length: int, at least 2
        :param timestep_h: int, at least 1
        :param distance_h: int, how many hours to skip between sequences, 0 means no values skipped
        """
        assert length >= 2, f'Length must be at least 2, currently is {length}'
        assert timestep_h >= 1, f'timestep_h must be at least 2, currently is {timestep_h}'
        assert distance_h >= 0, f'distance_h must be at least 0, currently is {distance_h}'

        # get rid of values where the target or features are missing
        idx = ~np.max(pd.isnull(dataframe), axis=1)
        dataframe = dataframe.loc[idx, :]

        # set index column as index and sort
        dataframe.set_index(index_col, drop=True, inplace=True)
        dataframe.sort_index(inplace=True)
        df_index_list = dataframe.index.tolist()

        # remember feature order and sort the order
        feature_order = {}
        features_ordered = dataframe.columns.sort_values().tolist()
        for i, feature in enumerate(features_ordered):
            feature_order[feature] = i
        dataframe = dataframe.loc[:, features_ordered]

        # variable to save the data
        sequences_filtered = []

        # remember max hour to get non-overlapping time series, start with a negative number,
        # so the first iteration always goes through
        max_h = -distance_h * 2

        # iterate over index
        # try starting a timeseries at index h
        for h in dataframe.index:
            chosen_h = []
            # if h is at least max_h plus distance, this is a usable value
            if h > max_h + distance_h:
                # add this as the first step
                chosen_h.append(h)

                # now look whether the other values needed exist
                aborted = False
                for offset in range(timestep_h, length * timestep_h, timestep_h):
                    h_next = h + offset
                    # if the next value exists add it
                    if h_next in df_index_list:
                        chosen_h.append(h_next)
                    # if not we need to start a new attempt at a different starting point
                    else:
                        aborted = True
                        break

                # if the sequence is complete, save the data
                if not aborted:
                    # update max_h
                    max_h = max(chosen_h)

                    # save the sequence
                    new_seq = []
                    for idx in chosen_h:
                        new_seq.append(dataframe.loc[idx, :].tolist())
                    sequences_filtered.append(new_seq)

        sequences_filtered = np.array(sequences_filtered)
        return sequences_filtered, feature_order

    @staticmethod
    def train_test_split(array, test_ratio, random_seed=0):
        array = array.copy()
        assert isinstance(array, np.ndarray), f'Expected a numpy array as input, got {type(array)}'
        test_size = round(array.shape[0] * test_ratio)
        assert test_size > 0
        # np.random.seed(random_seed)
        # np.random.shuffle(array)
        rs = np.random.RandomState(random_seed)
        rs.shuffle(array)
        test_array = array[:test_size]
        train_array = array[test_size:]
        return train_array, test_array

    @staticmethod
    def sample_from_data(array, size, random_seed=0):
        # np.random.seed(random_seed)
        # choice = np.random.choice(np.arange(array.shape[0]),size,False)
        rs = np.random.RandomState(random_seed)
        choice = rs.choice(np.arange(array.shape[0]), size, False)
        return array[choice]

    @staticmethod
    def separate_out_targets(array, feature_pos, keep_in_x=True, copy_y=False):
        """
         Splits an array of data into two arrays X and y.
         Data must be shaped N X TIME X FEATURES
        :param array: array to be split
        :param feature_pos: which features to extract (provide a list of indeces)
        :param keep_in_x: whether to keep the positions of pos in X
        :return: X, y
        """
        # for convenience one can give the position of a single target feature as a number
        if copy_y:
            keep_in_x = True

        if not isinstance(feature_pos, list):
            feature_pos = [feature_pos]

        # pick out the target values
        y = array[:, -1, feature_pos]
        # make sure y is in the 3 dimensional shape
        if len(y.shape) < 3:
            y = np.expand_dims(y, 1)

        # if one wants to keep the target column in X, one can just pick out X
        if keep_in_x and not copy_y:
            X = array[:, :-1, :]
        # if copy_y then X must remain the same

        # otherwise one needs to leave out the features
        elif not keep_in_x:
            idx = [i for i in range(array.shape[-1]) if i not in feature_pos]
            X = array[:, :-1, idx]
        else:
            X = array

        return X, y

    def get_train_data(self):
        return self.train_x, self.test_y

    def get_test_data(self):
        return self.test_x, self.test_y

    @staticmethod
    def simple_mean_imputation(dataframe, numerical_columns):
        imp = SimpleImputer()
        imputed = imp.fit_transform(dataframe[numerical_columns])
        dataframe.loc[:, numerical_columns] = imputed
        return dataframe

    def get_train_test_data(self, seq_length, seq_step, test_ratio, sample_size, seed):
        """
        From the city object it gets a training and test set and has the chosen properties
        :param seq_length: how long is the sequence (including the target and baseline value), minimum 2
        :param seq_step:  how far apart are two steps in the sequence, minimum 1
        :param seq_gap: how many steps to skip between sequences, minimum 0
        :param test_ratio: which proportion of all available data to keep for testing
        :param sample_size: how large shall the training set be
        :param seed: random seed for sampling and splitting
        :param copy_instead_of_split: If true y will be copied out of the last time step of X, if false X and y will be exclusive
        :return: (train x, train y), (test x, test y)
        """
        # filter out the sequences as wanted
        assert hasattr(self, 'experiment_prep_data'), 'Need to run prepare_for_experiment first...'
        data_x, data_y, col_name_number_map = self.experiment_prep_data[(seq_length, seq_step)]

        # train test split
        tr_x, te_x = dataset.train_test_split(data_x, test_ratio, random_seed=seed)
        tr_y, te_y = dataset.train_test_split(data_y, test_ratio, random_seed=seed)


        # sample the desired size if it is not too large
        if sample_size is not None:
            if tr_x.shape[0] <= sample_size:
                raise SampleSizeError(Sample_Size_Wanted=sample_size, Sequence_Length=seq_length,
                                      Sequence_Step_Size=seq_step,
                                      Seed=seed, Data_Object=self.name, Actual_Sample_Size=tr_x.shape[0])
            else:
                tr_x_sampled = dataset.sample_from_data(tr_x, sample_size, random_seed=seed)
                tr_y_sampled = dataset.sample_from_data(tr_y, sample_size, random_seed=seed)

        return (tr_x_sampled, tr_y_sampled), (te_x, te_y)

    def prepare_for_experiment(self, timestamp_col, seq_lengths, seq_steps, seq_gap, *args):
        self.experiment_prep_data = {}
        print(f'Preparing data for {self.name:}')
        for sl in tqdm(seq_lengths):
            for ss in seq_steps:
                data, col_name_number_map = dataset.filter_sequences(sl, ss, self.preprocessed_data, timestamp_col,
                                                                     seq_gap)
                assert hasattr(self, 'target_cols'), "No target columns found."
                target_cols = [col_name_number_map[t] for t in self.target_cols]
                data_x, data_y = dataset.separate_out_targets(data, target_cols, keep_in_x=True, copy_y=False)
                self.experiment_prep_data[(sl, ss)] = (data_x, data_y, col_name_number_map)



    def get_info(self):
        return {}

class SampleSizeError(Exception):
    def __init__(self, *args, **kwargs):
        self.info = kwargs
        super(SampleSizeError, self).__init__()

    """Raised when sample size too small"""

    def __str__(self):
        info = ''
        for k, v in self.info.items():
            info += str(k) + ':' + str(v) + '\t'
        return info
