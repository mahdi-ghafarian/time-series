import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------------

# Create a DataFrame and load a CSV
df = pd.read_csv("hurst-exponent\\data.csv")

# Define column names in the CSV file
# Change these as per your CSV file
date = 'Date'
price = 'SP500'

# Select the required columns
df=df[[date,price]]

# Change type of date column to datetime
df[date] = pd.to_datetime(df[date], format='%Y-%m-%d', errors='coerce')

#set date column as index
df.set_index(date,inplace=True)

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
df['Log_Price'] = np.log(df[price])
df['Return'] = df[price].pct_change()
df=df.dropna()
df['MA_10_Return'] = df['Return'].rolling(window=10).mean()
df=df.dropna()

# ------------------------------------------------------------------------------------
# ts is the df target column to calculate Hurst 
ts = df['MA_10_Return']

def hurst_exponent(ts):
    """Calculate the Hurst exponent and return regression data"""
    N = len(ts)
    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = np.std(ts)
    RS = R / S
    log_RS = np.log(RS[1:])
    log_T = np.log(T[1:])
    hurst, intercept = np.polyfit(log_T, log_RS, 1)
    return hurst, intercept, log_T, log_RS

# Calculate Hurst exponent and regression data
H, intercept, log_T, log_RS = hurst_exponent(ts)

# Print result and interpretation
print(f"Hurst Exponent: {H:.4f}")
if H < 0.5:
    print("The time series is mean-reverting.")
elif H == 0.5:
    print("The time series is a random walk.")
else:
    print("The time series is trending.")

# Plot the time series
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title("Synthetic Random Walk Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.plot(log_T, log_RS, 'o', label='log(R/S) vs log(T)', markersize=3)
plt.plot(log_T, intercept + H * log_T, 'r', label=f'Regression Line (H = {H:.4f})')
plt.title("Hurst Exponent Regression Line")
plt.xlabel("log(T)")
plt.ylabel("log(R/S)")
plt.legend()
plt.grid(True)
plt.show()
