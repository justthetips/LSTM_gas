import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
import os
import time

from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose


def parse(x):
    return pd.datetime.strptime(x, '%Y-%m-%d')


def name_chart(chart_type: str) -> str:
    base_path = Path(__file__)
    chrt_path = base_path.parent.parent.parent.joinpath('charts')
    if not chrt_path.exists():
        chrt_path.mkdir()
    file_name = '-'.join([chart_type, '-chart.png'])
    file_path = os.path.join(chrt_path, file_name)
    return file_path


matplotlib.style.use('ggplot')

# load dataset
dataset = pd.read_csv('~/projects/mlearn/data/gasoline.csv', index_col=0, date_parser=parse).dropna()

retail_prices = dataset['Retail']
rp_chart = retail_prices.plot()
rp_chart.set_ylabel('Retail Price')
rp_chart.set_title('US Average Retail Gasoline Price')
plt.savefig(name_chart('retail_price'))
plt.close()

spread = dataset['Retail'] - dataset['XB2']
sp_chart = spread.plot()
sp_chart.set_ylabel('Spread')
sp_chart.set_title('Spread between Average Retail and XB2')
plt.savefig(name_chart('spread'))
plt.close()

sspread = seasonal_decompose(spread)
sspread.plot()
plt.savefig(name_chart('seasonal'))
plt.close()