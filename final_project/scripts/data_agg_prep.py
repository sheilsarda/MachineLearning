import pandas as pd
import datetime
import yfinance as yf
import requests
import io

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

            stock_data = stock_data.append(s, sort=False)
    except Exception:
        None

print(stock_data)
