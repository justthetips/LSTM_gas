import os
from pathlib import Path
from typing import List, Tuple, Any

import pandas as pd
from scipy.optimize import brentq


def name_chart(chart_type: str) -> str:
    base_path = Path(__file__)
    chrt_path = base_path.parent.parent.parent.joinpath('charts')
    if not chrt_path.exists():
        chrt_path.mkdir()
    file_name = '-'.join([chart_type, '-chart.png'])
    file_path = os.path.join(chrt_path, file_name)
    return file_path


def name_model(model_name: str, extension: str = 'h5') -> str:
    base_path = Path(__file__)
    chrt_path = base_path.parent.parent.parent.joinpath('models')
    if not chrt_path.exists():
        chrt_path.mkdir()
    file_name = '.'.join([model_name, extension])
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


def ytm(px: float, maturity: int, cpn: float, cpn_per_year: int = 2, maturity_value: int = 100) -> Tuple[float, Any]:
    return brentq(price_dist, a=-cpn, b=2 * cpn, args=(maturity, cpn, px))
