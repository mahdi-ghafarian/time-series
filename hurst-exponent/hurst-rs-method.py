import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
df['return'] = df[price].pct_change()
df=df.dropna()
df['ma_10_return'] = df['return'].rolling(window=10).mean()
df=df.dropna()

# Select the column for hurst exponent calculation
# ts = df['log_price']

# ------------------------------------------------------------------------------------
# TEST: DATA FOR RANDOM WALK AND WHITE NOISE SERIES

# Parameters
n_steps = 100       # Number of time steps
dt = 1               # Time increment
mu = 0               # Drift (mean change per step)
sigma = 1            # Volatility (standard deviation of change per step)

# Generate Brownian motion
np.random.seed(42)   # For reproducibility
white_noise = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt), size=n_steps)
random_walk = np.cumsum(white_noise)

# ts = random_walk

# PASSED: got the expected values for H
# H of white noise = 0.5
# H of random walk = 1.0

# ------------------------------------------------------------------------------------

#TEST: DATA FOR MEAN REVERTING SERIES
# synthetic mean-reverting time series generated using the Ornstein-Uhlenbeck process

# Parameters
theta = 0.5   # Speed of mean reversion
mu = 0.0       # Long-term mean
sigma = 0.3    # Volatility
dt = 0.01      # Time step
T = 200.0        # Total time
N = int(T / dt)  # Number of steps

# Initialize the series
x = np.zeros(N)
x[0] = 1.0  # Initial value

# Generate the time series
for i in range(1, N):
    dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    x[i] = x[i-1] + dx
    
# ts = x

# FAILED: values near 1 for OU process

# ------------------------------------------------------------------------------------
#TEST: AR(1) process

# Parameters 
n = 1000  # Number of observations 
phi = -0.5  # AR(1) parameter 
sigma = 1  # Standard deviation of the noise
mean = 0  # Theoretical mean
# Initialize the time series 
ar1_process = np.zeros(n) 

# Generate AR(1) process 
for t in range(1, n): 
    ar1_process[t] = phi * ar1_process[t-1] + np.random.normal(mean, sigma) 

ts = ar1_process

# PASSED: only works for -1 < phi < 0
# ------------------------------------------------------------------------------------
def hurst_exponent(ts):
    """Calculate the Hurst exponent using R/S analysis"""
    N = len(ts)
    max_k = int(np.floor(N / 2))
    RS = []
    T = []

    for k in range(10, max_k, 100):  # step size can be adjusted
        chunks = [ts[i:i+k] for i in range(0, N, k) if len(ts[i:i+k]) == k]
        RS_chunk = []
        for chunk in chunks:
            Z = chunk - np.mean(chunk)
            Y = np.cumsum(Z)
            R = np.max(Y) - np.min(Y)
            S = np.std(chunk)
            RS_chunk.append(R / S if S != 0 else 0)
        RS.append(np.mean(RS_chunk))
        T.append(k)

    log_RS = np.log(RS)
    log_T = np.log(T)
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
plt.title("Time Series")
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
