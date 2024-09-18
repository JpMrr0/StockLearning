import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import yfinance as yf

symbol = 'ETH-USD'
data = yf.download(tickers=symbol, period='5d', interval='1m')

if data.empty:
    raise ValueError("Veri seti bos")

data['Date'] = data.index.date
today = data.index[-1].date()

training_data = data[data['Date'] < today].copy()

training_data['Target'] = training_data['Close'].shift(-1)
training_data.dropna(inplace=True)

X_train = training_data[['Close']]
y_train = training_data['Target']

model = LinearRegression()
model.fit(X_train, y_train)
last_close = training_data['Close'].iloc[-1]
predicted_prices = []

for _ in range(len(data[data['Date'] == today])):
    next_pred = model.predict(np.array([[last_close]]))[0]
    predicted_prices.append(next_pred)
    last_close = next_pred

plt.figure(figsize=(14, 7))
time_range = pd.date_range(start=pd.Timestamp(today), periods=len(predicted_prices), freq='T')
plt.plot(time_range, predicted_prices, label='Tahmin Edilen Değer', color='orange')
plt.title('ETH Bugünkü Tahmin Edilen Fiyatlar')
plt.xlabel('Saat')
plt.ylabel('Tahmin Edilen Fiyat (USD)')
plt.legend()
plt.show()
