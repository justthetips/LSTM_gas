import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Any, List
from math import sqrt
from pathlib import Path

import os


def name_chart(chart_type: str) -> str:
    base_path = Path(__file__)
    chrt_path = base_path.parent.parent.parent.joinpath('charts')
    if not chrt_path.exists():
        chrt_path.mkdir()
    file_name = '-'.join([chart_type, '-chart.png'])
    file_path = os.path.join(chrt_path, file_name)
    return file_path


def name_model(model_name: str) -> str:
    base_path = Path(__file__)
    chrt_path = base_path.parent.parent.parent.joinpath('models')
    if not chrt_path.exists():
        chrt_path.mkdir()
    file_name = '.'.join([model_name, 'h5'])
    file_path = os.path.join(chrt_path, file_name)
    return file_path


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


def price(yld: float, maturity: int, cpn: float, cpn_per_year: int = 2, maturity_value: int = 100) -> float:
    payments: int = maturity * cpn_per_year
    n: List[float] = [(1 / cpn_per_year) * x for x in range(1, payments + 1)]
    cf: List[float] = [(cpn / cpn_per_year) * maturity_value for x in n]
    cf[-1] += maturity_value
    dcf: List[float] = [cf[i] / ((1 + yld) ** x) for i, x in enumerate(n)]
    return sum(dcf)


def price_dist(yld: float, maturity: int, cpn: float, target_price: float, cpn_per_year: int = 2,
               maturity_value: int = 100) -> float:
    return target_price - price(yld, maturity, cpn, cpn_per_year, maturity_value)


def ytm(price: float, maturity: int, cpn: float, cpn_per_year: int = 2, maturity_value: int = 100) -> Tuple[float, Any]:
    return brentq(price_dist, a=-cpn, b=2 * cpn, args=(maturity, cpn, price))


def generate_bond_row():
    m = np.random.randint(1, 51)
    c = np.random.randint(0, 500) / 10000
    y = np.random.randint(0, 500) / 10000
    p = price(y, m, c)
    if (p > 175) or (p < 50):
        return generate_bond_row()
    else:
        return c, y, m, p


def generate_bond_inputs(n: int) -> pd.DataFrame:
    rows = []
    while len(rows) < n:
        rows.append(generate_bond_row())
    df = pd.DataFrame.from_records(data=rows,columns=['Coupon','Yield','Maturity','Price'])
    return df

np.random.seed(7)

# number of random bonds to create
num_bonds = 50000
# create our bond terms
dataset = generate_bond_inputs(num_bonds)
values = dataset.values

# create our imputs
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

train_size = int(0.75 * num_bonds)
train = scaled[:train_size, :]
test = scaled[train_size:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# create the model
model = Sequential()
model.add(Dense(768, input_dim=3, activation="relu"))
model.add(Dense(384, activation="relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# Compile model
model.compile(loss='mse', optimizer='adam')
# Fit the model
model.fit(train_X, train_y, epochs=300, batch_size=5000, validation_data=(test_X, test_y), verbose=2,
          shuffle=False)
model.save(filepath=name_model('duration'))

# model = load_model(filepath=name_model('duration'))


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
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

plt.plot(inv_yhat, inv_y, '.')
plt.xlabel('predicted')
plt.ylabel('actual')
m, b = np.polyfit(inv_yhat, inv_y, 1)
plt.plot(inv_yhat, m * inv_yhat + b, '-')
plt.show()
plt.close()
