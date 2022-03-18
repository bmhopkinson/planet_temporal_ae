import torch
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer

class SR_Timeseries():
    def __init__(self, datapath):
        self.n_channels = 4
        df = pd.read_csv(datapath, delimiter='\t', header=None)
        df = self.clean_and_impute_data(df, 'Simple')

        rows, cols = df.shape
        n_timepts = int((cols - 2) / 4)

        ts_data_raw = df.iloc[:, 2:].to_numpy(copy=True)
        ts_data = []
        for i, row in enumerate(ts_data_raw):
            position = df.iloc[i, [0, 1]]
            sr_data = np.zeros([n_timepts, self.n_channels])

            for j in range(n_timepts):
                data_t = row[j*self.n_channels:(j+1)*self.n_channels]
                sr_data[j, :] = data_t

            ts_data.append([position, sr_data])

        self.ts_data = ts_data

    def __len__(self):
        return len(self.ts_data)

    def __getitem__(self, idx):
        elm = self.ts_data[idx]
        pos = elm[0]
        sr_data = torch.tensor(elm[1])

        return {'pos': pos, 'sr_data': sr_data}


    def clean_and_impute_data(self, df, imputer_choice):
        rows, cols = df.shape
        n_timepts = int((cols - 2) / 4)

        names1 = ['lat', 'lon']
        names2 = ['{}_{}'.format(channel, i) for i in range(0, n_timepts) for channel in ['b', 'g', 'r', 'nir']]
        names = names1 + names2
        df.columns = names

        # cleaning and imputation
        n_nans_to_drop = int(0.5 * n_timepts * self.n_channels)  # if nans exceed half of the timeseries -> drop row
        df = df.dropna(thresh=n_nans_to_drop)
        ts_data = df.iloc[:, 2:].to_numpy(copy=True)  # drop
        print('number of points that need to be imputed: {}'.format(np.count_nonzero(np.isnan(ts_data))))

        if imputer_choice == 'Simple':
            imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        elif imputer_choice == 'KNN':
            imputer = KNNImputer(n_neighbors=5, weights='uniform', missing_values=np.nan)
        else:
            raise NotImplementedError

        ts_data = imputer.fit_transform(ts_data)
        df.iloc[:, 2:] = ts_data

        return df

##### end SR_Timeseries class #####

def pad_collate(batch_raw):
    positions = [x['pos'] for x in batch_raw]

    lens = [x['sr_data'].size(0) for x in batch_raw]
    max_len = max(lens)
    lens = torch.tensor(lens)

    batch_size = len(batch_raw)
    n_channels = batch_raw[0]['sr_data'].size(1)
    padded_data = torch.zeros(batch_size, max_len, n_channels)

    for i, sample in enumerate(batch_raw):
        data = sample['sr_data']
        n_timepts = data.size(0)
        padded_data[i, 0:n_timepts, :] = data

    return {'pos': positions, 'sr_data': padded_data, 'lens': lens}


