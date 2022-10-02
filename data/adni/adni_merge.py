from data.dataset import dataset
import pandas as pd
import os
import numpy as np
import functools
from sklearn.impute import SimpleImputer
import warnings
from data.dataset import SampleSizeError

cols_sel = ['MMSE', 'PTGENDER', 'APOE4', 'AGE', 'PTEDUCAT', 'FDG',
                'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11', 'ADAS13', 'ADASQ4', 'RAVLT_immediate',
                'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL',
                'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
                'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal',
                'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV']

cols_categorical = ['PTGENDER', 'APOE4']

data_viscodes = ['bl', 'm12', 'm24', 'm36', 'm48']

nan_threshold = 0.7

class adni(dataset):
    def __init__(self, path):
        self.name = 'ADNI_MMSE'
        self.path = path
        self.target_cols = ['MMSE']

    def prepare_for_experiment(self, seq_lengths, *args):
        target = 'MMSE'
        self.experiment_prep_data = {}
        print(f'Preparing data for {self.name}...')
        for seq_len in seq_lengths:
            priv_points = seq_len - 2

            data_viscodes = ['bl', 'm12', 'm24', 'm36', 'm48']
            if priv_points == 1:
                selection_viscodes = ['bl', 'm24', 'm48']
            elif priv_points == 3:
                selection_viscodes = data_viscodes
            else:
                raise ValueError('priv_points invalid value, expected 3 or 1 and got ' + str(priv_points))

            D = pd.read_csv(self.path)

            D['AD'] = D['DX'] == 'Dementia'
            D['MCI'] = D['DX'] == 'MCI'
            D.loc[D['DX'].isna(), ['AD', 'MCI']] = np.nan

            D.loc[:, 'ABETA'] = D.loc[:, 'ABETA'].replace('>1700', 1700, regex=True) \
                .replace('<900', 900, regex=True) \
                .replace('<200', 200, regex=True).astype(np.float32)

            D.loc[:, 'TAU'] = D.loc[:, 'TAU'].replace('>1300', 1300, regex=True) \
                .replace('<80', 80, regex=True).astype(np.float32)

            D.loc[:, 'PTAU'] = D.loc[:, 'PTAU'].replace('>120', 120, regex=True) \
                .replace('<8', 8, regex=True).astype(np.float32)

            D = D.loc[:, ['VISCODE', 'RID', 'MCI', 'AD'] + cols_sel]
            D = pd.get_dummies(D, columns=cols_categorical, drop_first=True)

            # Drop features with more than nan_threshold% of the observations missing
            to_be_removed = []
            for code in data_viscodes:
                count = len(D[D['VISCODE'] == code])
                l = D[D['VISCODE'] == code].isna().sum()
                for i, e in enumerate(l):
                    if nan_threshold < e / count:
                        if D.columns[i] not in to_be_removed:
                            to_be_removed += [D.columns[i]]
            D = D.drop(to_be_removed, axis=1)

            # Start to packet data into X, Y
            frames = {}
            for code in data_viscodes:
                if code == data_viscodes[-1]:
                    frames[code] = D[D['VISCODE'] == code].dropna(subset=[target])
                else:
                    frames[code] = D[D['VISCODE'] == code]


            patient_ID = {}
            for code in data_viscodes:
                patient_ID[code] = frames[code]['RID'].unique()
            I = functools.reduce(lambda a, b: np.intersect1d(a, b), [patient_ID[k] for k in patient_ID.keys()])



            data = {}
            for code in selection_viscodes:
                data[code] = frames[code][frames[code]['RID'].isin(I)]

            features = [e for e in D.columns if e not in ['RID', 'VISCODE', 'MCI', 'AD']]


            X = np.zeros((len(I), len(selection_viscodes) - 1, len(features)))
            data[selection_viscodes[-1]] = data[selection_viscodes[-1]].sort_values(by=['RID'])
            Y = data[selection_viscodes[-1]][target].values

            feature_index = {}

            for j, code in enumerate(selection_viscodes[0:len(selection_viscodes) - 1]):
                data[code] = data[code].sort_values(by=['RID'])
                data[code] = data[code].loc[:, features]
                for feature in features:
                    feature_index[feature] = data[code].columns.get_loc(feature)
                X[:, j, :] = data[code].values

            imputer = SimpleImputer()
            shape = X.shape
            X = imputer.fit_transform(X.reshape(shape[0],-1)).reshape(shape)

            self.experiment_prep_data[(seq_len,1)] = (X, Y)


    def get_train_test_data(self, seq_len, seq_step, test_ratio, sample_size, seed):
        if seq_step != 1:
            warnings.warn(f'Sequence Step set to 1 automatically for {self.name}')
            seq_step = 1
        assert hasattr(self, 'experiment_prep_data'), 'Need to run prepare_for_experiment first...'
        data_x, data_y = self.experiment_prep_data[(seq_len, seq_step)]

        # train test split
        tr_x, te_x = dataset.train_test_split(data_x, test_ratio, random_seed=seed)
        tr_y, te_y = dataset.train_test_split(data_y, test_ratio, random_seed=seed)

        # sample the desired size if it is not too large
        if sample_size is not None:
            if tr_x.shape[0] <= sample_size:
                raise SampleSizeError(Sample_Size_Wanted=sample_size, Sequence_Length=seq_len,
                                      Sequence_Step_Size=seq_step,
                                      Seed=seed, Data_Object=self.name, Actual_Sample_Size=tr_x.shape[0])
            else:
                tr_x_sampled = dataset.sample_from_data(tr_x, sample_size, random_seed=seed)
                tr_y_sampled = dataset.sample_from_data(tr_y, sample_size, random_seed=seed)

        return (tr_x_sampled, tr_y_sampled.reshape(-1,1,1)), (te_x, te_y.reshape(-1,1,1))


if __name__ == '__main__':
    path = '~/ADNI_DATA/ADNIMERGE.csv'
    dataset = adni(path)
    dataset.prepare_for_experiment([3,5])
    (x_tr, y_tr),(x_te, y_te) = dataset.get_train_test_data(5, 1, 0.2, 200,1)
    print(x_tr.shape)