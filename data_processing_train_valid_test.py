import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import interpolate
from torch.utils.data import Dataset
import torch
from config import MAX_RUL, WIN_SIZE

scaler_global = preprocessing.MinMaxScaler()

def process_data(data_dir, data_id, win_size=WIN_SIZE):
    scaler = preprocessing.MinMaxScaler()
    RUL_arr = np.loadtxt(f'{data_dir}/RUL_{data_id}.txt')

    col_names = ['id','cycle','setting1','setting2','setting3'] + \
                [f's{i}' for i in range(1, 22)]

    tr = pd.read_csv(f'{data_dir}/train_{data_id}.txt', sep=r'\s+', header=None)
    tr.drop(columns=[26, 27], inplace=True, errors='ignore') 
    tr.columns = col_names
    tr.sort_values(['id','cycle'], inplace=True)

    te = pd.read_csv(f'{data_dir}/test_{data_id}.txt', sep=r'\s+', header=None)
    te.drop(columns=[26, 27], inplace=True, errors='ignore')
    te.columns = col_names
    te.sort_values(['id','cycle'], inplace=True)

    if data_id in ['FD002', 'FD004']:
        for df in [tr, te]:
            df.loc[df['setting1'].between(0.0, 3e-3),   'setting1'] = 0.0
            df.loc[df['setting1'].between(9.998, 10.008),'setting1'] = 10.0
            df.loc[df['setting1'].between(19.998,20.008),'setting1'] = 20.0
            df.loc[df['setting1'].between(24.998,25.008),'setting1'] = 25.0
            df.loc[df['setting1'].between(34.998,35.008),'setting1'] = 35.0
            df.loc[df['setting1'].between(41.998,42.008),'setting1'] = 42.0

        train_norm_parts, test_norm_parts = [], []
        tr_sensor = tr.iloc[:, 2:]; te_sensor = te.iloc[:, 2:]
        tr_norm    = pd.DataFrame(columns=tr_sensor.columns[3:], dtype=object)
        te_norm    = pd.DataFrame(columns=te_sensor.columns[3:], dtype=object)

        for t_idx, t_grp in tr_sensor.groupby('setting1'):
            sc = preprocessing.MinMaxScaler()
            sc_tr = sc.fit_transform(t_grp.iloc[:, 3:])
            tr_norm = pd.concat([tr_norm,
                pd.DataFrame(sc_tr, index=t_grp.index, columns=tr_sensor.columns[3:])])
            for te_idx, te_grp in te_sensor.groupby('setting1'):
                if t_idx == te_idx:
                    sc_te = sc.transform(te_grp.iloc[:, 3:])
                    te_norm = pd.concat([te_norm,
                        pd.DataFrame(sc_te, index=te_grp.index, columns=te_sensor.columns[3:])])

        tr_norm = tr_norm.sort_index(); te_norm = te_norm.sort_index()
        tr.iloc[:, 2:5] = scaler.fit_transform(tr.iloc[:, 2:5])
        te.iloc[:, 2:5] = scaler.transform(te.iloc[:, 2:5])
        tr_settings = tr.iloc[:, :5]; te_settings = te.iloc[:, :5]
        tr_nor = pd.concat([tr_settings, tr_norm], axis=1).values
        te_nor = pd.concat([te_settings, te_norm], axis=1).values
    else:
        tr['setting1'] = 0.0
        tr.iloc[:, 2:] = scaler.fit_transform(tr.iloc[:, 2:])
        te.iloc[:, 2:] = scaler.transform(te.iloc[:, 2:])
        tr_nor = tr.values; te_nor = te.values

    drop_idx = [5, 9, 10, 14, 20, 22, 23]
    tr_nor = np.delete(tr_nor, drop_idx, axis=1)
    te_nor = np.delete(te_nor, drop_idx, axis=1)

    train_data, train_labels, valid_data, valid_labels = _get_train_valid(tr_nor, win_size, MAX_RUL)

    testX, testY = [], []
    n_engines = int(np.max(te_nor[:, 0]))
    for i in range(1, n_engines + 1):
        idx     = np.where(te_nor[:, 0] == i)[0]
        d_temp  = te_nor[idx, :]
        if len(d_temp) < win_size:
            d_interp = []
            for col in range(d_temp.shape[1]):
                x1    = np.linspace(0, win_size - 1, len(d_temp))
                x_new = np.linspace(0, win_size - 1, win_size)
                tck   = interpolate.splrep(x1, d_temp[:, col])
                d_interp.append(interpolate.splev(x_new, tck))
            d_temp = np.array(d_interp).T
            d_temp = d_temp[:, 1:] 
        else:
            d_temp = d_temp[-win_size:, 1:]
        testX.append(d_temp.reshape(1, d_temp.shape[0], d_temp.shape[1]))
        testY.append(min(RUL_arr[i - 1], MAX_RUL))

    testX = np.concatenate(testX, axis=0)

    sc2 = preprocessing.MinMaxScaler()
    train_data[:, :, 0] = sc2.fit_transform(train_data[:, :, 0])
    valid_data[:, :, 0] = sc2.fit_transform(valid_data[:, :, 0])
    testX[:, :, 0]      = sc2.transform(testX[:, :, 0])

    train_data  = train_data[:, :, 4:]
    valid_data  = valid_data[:, :, 4:]
    testX       = testX[:, :, 4:]

    return train_data, valid_data, testX, train_labels, valid_labels, np.array(testY)

def _get_train_valid(data, win_size, max_rul):
    n_engines = int(np.max(data[:, 0]))
    n_train   = int(0.9 * n_engines)
    n_val     = n_engines - n_train
    torch.manual_seed(42)
    idx_train, idx_val = torch.utils.data.random_split(
        np.arange(n_engines), [n_train, n_val])
    X_tr, y_tr = _split_data(idx_train.indices, win_size, data, max_rul)
    X_va, y_va = _split_data(idx_val.indices,   win_size, data, max_rul)
    return X_tr, y_tr, X_va, y_va

def _split_data(indices, win_size, data, max_rul):
    Xs, ys = [], []
    for i in indices:
        idx    = np.where(data[:, 0] == i)[0]
        d_temp = data[idx, :]
        for j in range(len(d_temp) - win_size + 1):
            Xs.append(d_temp[j:j+win_size, 1:].tolist())
            rul = len(d_temp) - win_size - j
            ys.append(min(rul, max_rul))
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.float32) / max_rul 
    return X, y

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], i
