import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------

# Create a DataFrame and load a CSV
df = pd.read_csv("hurst-exponent\\data.csv")

# Define column names in the CSV file
# Change these as per your CSV file
start_date = '1940-01-01'
date = 'Date'
price = 'SP500'

# Select the required columns
df=df[[date,price]]

# Change type of date column to datetime
df[date] = pd.to_datetime(df[date], format='%Y-%m-%d', errors='coerce')

#set date column as index
df.set_index(date,inplace=True)

# Filter from a starting date
df = df[df.index >= start_date]

# set frequency
# df = df.to_period(freq='M')

#plot the time series
#df.plot()

#view the time series
print(df)

#view the time series information
print(df.info())
print("\n")

# additional columns
df['log_price'] = np.log(df[price])
df['return'] = df[price].pct_change().dropna()
df['ma_10_return'] = df['return'].rolling(window=10).mean().dropna()
df['log_return'] = np.log(df[price] / df[price].shift(1)).dropna()
df['ma_10_log_return'] = df['log_return'].rolling(window=10).mean().dropna()

# Select the column, Variance Ratio input series should be log_return
series = df['log_return']

series.plot()

# ------------------------------------------------------------------------------------
#Parameters
max_lag = 30

# Function to compute variance ratio
def variance_ratio(series, q):
    var_1 = np.var(series, ddof=1)
    q_returns = [np.sum(series[i:i+q]) for i in range(len(series) - q)]
    var_q = np.var(q_returns, ddof=1)
    return var_q / (q * var_1)

# Compute variance ratios for a range of lags
lags = list(range(2, max_lag))
vr_values = [variance_ratio(series, q) for q in lags]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lags, vr_values, marker='o', label='Variance Ratio')
plt.axhline(y=1, color='red', linestyle='--', label='Random Walk Benchmark')  # Red benchmark line
plt.xticks(ticks=lags)  # Set x-axis ticks to match lag values
plt.xlabel('Lag (q)')
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio Test')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
