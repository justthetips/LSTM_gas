import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from sklearn.externals import joblib

import os

from src.mlearn.utils import series_to_supervised, name_model


class PCAModel(object):
    _model = None
    _scaler = None

    def __init__(self, *args, **kwargs):
        if 'modelpath' in kwargs:
            self._model = load_model(filepath=kwargs.get('modelpath'))

        if 'scalerpath' in kwargs:
            self._scaler = joblib.load(filename=kwargs.get('scalerpath'))

    def getModel(self, train_X: np.array, train_Y: np.array, test_X: np.array, test_Y: np.array,
                 force_refit: bool = False) -> Sequential:
        if (self._model is None) or force_refit:
            self._model = Sequential()
            self._model.add(LSTM(898, input_shape=(train_X.shape[1], train_X.shape[2])))
            self._model.add(Dense(898 * 4))
            self._model.add(Dense(898))
            self._model.add(Dense(14))
            self._model.compile(loss='mae', optimizer='adam')
            # fit network
            self._model.fit(train_X, train_y, epochs=500, batch_size=182, validation_data=(test_X, test_y),
                            verbose=2,
                            shuffle=False)
        return self._model

    def getScaler(self, data: pd.DataFrame, force_refit: bool = False) -> MinMaxScaler:
        if (self._scaler is None) or force_refit:
            self._scaler = MinMaxScaler(feature_range=(0, 1))
            self._scaler.fit(data)
        return self._scaler

    def saveScaler(self):
        if self._scaler is None:
            raise AttributeError('Scaler Does Not Exist')
        joblib.dump(self._scaler, name_model('pca_scaler', 'save'))

    def saveModel(self):
        if self._model is None:
            raise AttributeError('Model Does Not Exist')
        self._model.save(filepath=name_model('pcamodel'))


def parse(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')


def load_data(path: str) -> pd.DataFrame:
    dataset = pd.read_csv(path, index_col=0, date_parser=parse)
    # take the first difference
    return dataset


def pca_analysis(data: pd.DataFrame):
    data_chg = data.diff().dropna()
    known_cols = [0, 8, 13]

    N = data_chg.shape[0]
    M = data_chg.shape[1]

    covar_M = np.cov(data_chg, rowvar=False)
    std_M = np.diagflat(covar_M.diagonal() ** .5)

    eig_values, eig_vectors = np.linalg.eig(covar_M)

    last_chg = data_chg[-1:].values
    last_swaps = data.iloc[N - 1]

    V = np.zeros([3, M])
    DeltaQ = np.zeros(3)
    W_hat = eig_vectors[:, 0:3]
    for i, v in enumerate(known_cols):
        V[i][v] = 1
        DeltaQ[i] = last_chg[0][v]

    exp_chg = np.matmul(np.matmul(std_M, np.matmul(W_hat, np.linalg.inv(np.matmul(np.matmul(V, std_M), W_hat)))),
                        DeltaQ)
    exp_swap = last_swaps + exp_chg

    return exp_swap


data = load_data('~/projects/mlearn/data/swapdata.csv')
pca_model = PCAModel(modelpath='/home/jacob/projects/mlearn/models/pcamodel.h5',
                     scalerpath='/home/jacob/projects/mlearn/models/pca_scaler.save')
data_chg = data.diff().dropna()

values = data.values
values = values.astype('float32')
# frame as supervised learning
reframed = series_to_supervised(values, 63, 1)
X = reframed.iloc[:, 1:reframed.shape[1] - 14]
Y = reframed.iloc[:, -14:]

# now to add in the t changes
chg = data_chg.iloc[62:]
chg = chg.iloc[:, [0, 8, 13]]
allX = np.hstack((X, chg))
allData = np.hstack((allX, Y))
reframed = pd.DataFrame(allData)

# scale
scaler = pca_model.getScaler(reframed)
pca_model.saveScaler()
scaled = scaler.transform(reframed)

# split into train and test sets
values = scaled
n_train_days = 3 * 251
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, 0:train.shape[1] - 14], train[:, -14:]
test_X, test_y = test[:, 0:train.shape[1] - 14], test[:, -14:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# build our model

model = pca_model.getModel(train_X, train_y, test_X, test_y, force_refit=True)
pca_model.saveModel()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -14:]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 14))
inv_y = np.concatenate((test_X[:, 0:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -14:]
# calculate RMSE
print(pd.DataFrame(inv_yhat).tail())
print(pd.DataFrame(inv_y).tail())