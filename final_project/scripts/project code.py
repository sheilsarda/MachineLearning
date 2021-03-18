import pandas as pd
import datetime
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import io
import keras
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import MinMaxScaler

symbol_url = "https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(symbol_url).content
symbols = pd.read_csv(io.StringIO(s.decode('utf-8')))['Symbol'].tolist()
print("Stock Symbols", symbols)

# Comment this to get the all-symbols
symbols = ["AAL", "AAPL", "DAL", "FB", "AMZN", "TSLA", "MSFT", "CRM"]
stock_data = pd.DataFrame()
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2020, 11, 28)


def sma(data, n):
    sma_values = pd.Series(data['Close'].rolling(n).mean(), name='Sma')
    data = data.join(sma_values)
    return data


def ewma(data, n):
    ema = pd.Series(data['Close'].ewm(span=n, min_periods=n - 1).mean(),
                    name='Ewma_' + str(n))
    data = data.join(ema)
    return data


def ATR(data, n):
    df = data.copy()
    df['H_L'] = abs(data['High']-data['Low'])
    df['H_PC'] = abs(data['High']-data['Close'].shift(1))
    df['L_PC'] = abs(data['Low']-data['Close'].shift(1))
    TR = df[['H_L', 'H_PC', 'L_PC']].max(axis=1,skipna=False)
    atr_value = TR.rolling(n).mean()
    atr_value = round(atr_value[-1], 2)
    atr_value = pd.Series(atr_value, name='ATR')
    data = data.join(atr_value)
    return data


def CAGR(data, n):
    p_change = data['Close'].pct_change()
    cum_return = (1 + p_change).cumprod()
    n = len(data) / 252
    cagr = (cum_return[-1]) ** (1 / n) - 1
    cagr = pd.Series(cagr, name='CAGR')
    data = data.join(cagr)
    return data


def volatility(data):
    daily_ret = pd.Series(data['Close'].pct_change(), name='daily_ret')
    data = data.join(daily_ret)
    return data


def cci(data, n):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    cci_values = pd.Series((TP - TP.rolling(n).mean()) / (0.015 * TP.rolling(n).std()),
                           name='Cci')
    data = data.join(cci_values)
    return data


for symbol in symbols:
    try:
        s = []
        n = 15
        s = yf.download(symbol, start=start, end=end)
        if s is not None and len(s) > 0:
            s['Name'] = symbol

            # Getting simple moving average
            s = sma(s, n)
            s = s.dropna()

            # Exponentially weighted moving average
            s = ewma(s, n)
            s = s.dropna()

            # Commodity Channel Index
            s = cci(s, n)
            s = s.dropna()

            s = ATR(s, 3)

            # volatility
            s = volatility(s)

            # volatility
            s = CAGR(s, 3)

            stock_data = stock_data.append(s, sort=False)
    except Exception:
        None

print(stock_data)


def get_model():
    model = Sequential()
    n_input_layer = 3
    model.add(Dense(n_input_layer, input_dim=n_input_layer,
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


X = stock_data[["Sma", "Ewma_15", "Cci"]]
Y = stock_data['Close']

estimator = KerasRegressor(build_fn=get_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("MSE: %.2f" % (results.mean()))
