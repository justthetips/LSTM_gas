from math import sqrt
from matplotlib import pyplot

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.utils import plot_model

from src.mlearn.rboba import name_chart


def parse(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = pd.read_csv('~/projects/mlearn/data/gasoline.csv', index_col=0, date_parser=parse)
print(dataset.head())
dataset = dataset[:-1]
datadates = dataset.index.values
datamonths = pd.Series(data=[pd.to_datetime(x).month for x in datadates], index=datadates, name='month')
datadays = pd.Series([pd.to_datetime(x).day for x in datadates], index=datadates, name='day')
datamonths = datamonths.to_frame().join(datadays.to_frame())
dataset = datamonths.join(dataset)
print(dataset.head())

values = dataset.values
values = values.astype('float32')
# frame as supervised learning
reframed = series_to_supervised(values, 12, 1)
# drop columns we don't want to predict (ie month, day, xb2 on day t)
reframed.drop(reframed.columns[[48, 49, 50]], axis=1, inplace=True)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(reframed)
print(reframed.head())

# split into train and test sets
values = scaled
n_train_days = 2 * 365
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=91, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

plot_model(model, to_file=name_chart('model_rep'))

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig(name_chart('loss'))
pyplot.close()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, 0:], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

pyplot.plot(inv_yhat)
pyplot.plot(inv_y)
pyplot.legend(['predict', 'actual'])
pyplot.title('Neural Net Retail Gas Prediction')
pyplot.savefig(name_chart('prediction'))
pyplot.close()

pyplot.plot(inv_yhat, inv_y, '.')
pyplot.xlabel('predicted')
pyplot.ylabel('actual')
m, b = np.polyfit(inv_yhat, inv_y, 1)
pyplot.plot(inv_yhat, m * inv_yhat + b, '-')
pyplot.savefig(name_chart('scatter_pred'))
pyplot.close()

