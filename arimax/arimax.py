import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 📊 Simulate some data
np.random.seed(42)
n = 100
time = pd.date_range(start='2020-01-01', periods=n, freq='D')
y = np.cumsum(np.random.randn(n))  # endogenous variable
x = np.random.randn(n)             # exogenous variable

# 🧹 Create DataFrame
df = pd.DataFrame({'y': y, 'x': x}, index=time)

print(df)

# 🧠 Fit ARIMAX model (ARIMA(1,1,1) with exogenous variable)
model = SARIMAX(df['y'], order=(1, 1, 1), exog=df[['x']])
results = model.fit(disp=False)

# 📈 Forecast
forecast = results.get_forecast(steps=10, exog=np.random.randn(10).reshape(-1, 1))
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# 📊 Plot results
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['y'], label='Observed')
plt.plot(df.index, df['x'], label='Exogenous', alpha=0.5)
plt.plot(pd.date_range(df.index[-1], periods=11, freq='D')[1:], forecast_mean, label='Forecast')
plt.fill_between(pd.date_range(df.index[-1], periods=11, freq='D')[1:], 
                 conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('ARIMAX Forecast')
plt.show()
