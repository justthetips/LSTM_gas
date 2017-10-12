import os
from math import sqrt
from pathlib import Path
from typing import Tuple, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from scipy.optimize import brentq
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from src.mlearn.utils import price, name_model, name_chart


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
    df = pd.DataFrame.from_records(data=rows, columns=['Coupon', 'Yield', 'Maturity', 'Price'])
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
# save our scaler
joblib.dump(scaler, name_model('scaler', 'save'))

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
#model = Sequential()
#model.add(Dense(768, input_dim=3, activation="relu"))
#model.add(Dense(384, activation="relu"))
#model.add(Dense(1))
#model.add(Activation("sigmoid"))
# Compile model
#model.compile(loss='mse', optimizer='adam')
# Fit the model
#model.fit(train_X, train_y, epochs=1000, batch_size=7500, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#model.save(filepath=name_model('duration'))

model = load_model(filepath=name_model('duration'))


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
plt.title('Predicted vs Actual Bond Prices')
m, b = np.polyfit(inv_yhat, inv_y, 1)
plt.plot(inv_yhat, m * inv_yhat + b, '-')
plt.savefig(name_chart('bond_pricing'))
plt.show()
plt.close()
