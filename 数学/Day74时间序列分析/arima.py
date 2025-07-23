import  yfinance as yf
data = yf.download('AAPL', start='2020-01-01',end='2025-07-23')

print(data.head())

close_prices = data['Close']

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(close_prices)
plt.title('Apple Stock Close Prices')
plt.savefig('Apple-Stock-Close-Prices.png')

from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] > 0.05:
        print("该系列是非平稳的")
    else:
        print("该系列是平稳的")
check_stationarity(close_prices)

diff_close_prices = close_prices.diff().dropna()
check_stationarity(diff_close_prices)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(diff_close_prices)
plot_pacf(diff_close_prices)
plt.savefig('acf.png')


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(close_prices,order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=5)
print(forecast)
