import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 📊 Simulate two interrelated time series
np.random.seed(42)
n = 100
time = pd.date_range(start='2020-01-01', periods=n, freq='D')

# Simulate two series with some correlation
y1 = np.cumsum(np.random.randn(n)) + np.random.randn(n)
y2 = np.cumsum(np.random.randn(n)) + 0.5 * y1 + np.random.randn(n)

# 🧹 Create DataFrame
df = pd.DataFrame({'y1': y1, 'y2': y2}, index=time)

# 🧠 Check stationarity (optional: use ADF test) and difference if needed
df_diff = df.diff().dropna()

# 🔧 Fit VAR model
model = VAR(df_diff)
results = model.fit(maxlags=15, ic='aic')  # Automatically selects optimal lag

# 📈 Forecast next 10 steps
forecast = results.forecast(df_diff.values[-results.k_ar:], steps=10)

# 🗓️ Create forecast index
forecast_index = pd.date_range(df.index[-1], periods=11, freq='D')[1:]
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['y1_forecast', 'y2_forecast'])

# 📊 Plot results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['y1'], label='y1 Actual')
plt.plot(df.index, df['y2'], label='y2 Actual')
plt.plot(forecast_df.index, forecast_df['y1_forecast'], label='y1 Forecast', linestyle='--')
plt.plot(forecast_df.index, forecast_df['y2_forecast'], label='y2 Forecast', linestyle='--')
plt.legend()
plt.title('VAR Forecast')
plt.show()
