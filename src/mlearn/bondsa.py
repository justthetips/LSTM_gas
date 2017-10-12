import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.utils import plot_model
from sklearn.externals import joblib
from src.mlearn.utils import price

from src.mlearn.utils import name_chart

# model location
base_path = '/home/jacob/projects/mlearn/models'

# load our model
scaler = joblib.load(os.path.join(base_path, 'scaler.save'))
model = load_model(os.path.join(base_path, 'duration.h5'))


def duration_estimate(p0: float, pp: float, pm: float, yc: float) -> float:
    d = (pp - pm) / (2 * p0 * yc)
    return d


def convexity_estimate(p0: float, pp: float, pm: float, yc: float) -> float:
    c = ((pp + pm) - (2 * p0)) / (2 * p0 * (yc ** 2))
    return c


# create our test input
# first our duration test
m = 10
c = .03
y = [.029, .03, .031]
p = [price(yld=x, maturity=m, cpn=c) for x in y]
bonds = [(c, y[i], m, p[i]) for i in range(len(p))]
bonds = df = pd.DataFrame.from_records(data=bonds, columns=['Coupon', 'Yield', 'Maturity', 'Price'])
values = bonds.values
bond_dur = duration_estimate(p[1], p[0], p[2], .001)
bond_cx = convexity_estimate(p[1], p[0], p[2], .001)
# now use our model
scaled = scaler.transform(values)
test_X = scaled[:, :-1]
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
test_dur = duration_estimate(inv_yhat[1], inv_yhat[0], inv_yhat[2], .001)
text_cx = convexity_estimate(inv_yhat[1], inv_yhat[0], inv_yhat[2], .001)

print("Math Prices: {}".format(p))
print("NN Prices: {}".format(inv_yhat))
print("Math Duration: {0:.2f}".format(bond_dur))
print("NN Duration {0:.2f}".format(test_dur))
print("Math Convexity: {0:.2f}".format(bond_cx))
print("NN Convexity: {0:.2f}".format(text_cx))

yrange = [x / 10000 for x in range(0, 500)]
prange = [price(yld=x,maturity=30,cpn=.025) for x in yrange]
brange = [(.025,yrange[i],30,prange[i]) for i in range(len(prange))]
brange = pd.DataFrame.from_records(data=brange, columns=['Coupon', 'Yield', 'Maturity', 'Price'])
values = brange.values
scaled = scaler.transform(values)
test_X = scaled[:, :-1]
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, 0:], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]

plt.plot(yrange, inv_yhat, '.')
plt.xlabel('yield')
plt.ylabel('price')
plt.title('NN Price vs. Yld of a 30yr 2.5% Cpn Bond')
plt.savefig(name_chart('px_yld'))
plt.show()

plot_model(model, to_file=name_chart('duration_model_rep'))