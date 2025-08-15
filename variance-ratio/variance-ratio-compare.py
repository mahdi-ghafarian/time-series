import numpy as np
import matplotlib.pyplot as plt

# Function to generate time series
def generate_series(n, model='white_noise', phi=0.0):
    series = np.zeros(n)
    if model == 'white_noise':
        series = np.random.normal(0, 1, n)
    elif model == 'ar1':
        for t in range(1, n):
            series[t] = phi * series[t-1] + np.random.normal(0, 1)
    elif model == 'random_walk':
        noise = np.random.normal(0, 1, n)
        series = np.cumsum(noise)
    return series

# Function to compute variance ratio
def variance_ratio(series, q):
    n = len(series)
    # If you want to use log return
    # Ensure series is strictly positive to avoid log of zero or negative
    # series = series + np.abs(np.min(series)) + 1e-6
    # returns = np.diff(np.log(series))
    # returns = returns[~np.isnan(returns)]
    
    returns = np.diff(series)
    
    var_1 = np.var(returns, ddof=1)
    
    # q-period returns
    q_returns = [np.sum(returns[i:i+q]) for i in range(len(returns) - q)]
    var_q = np.var(q_returns, ddof=1)
    
    return var_q / (q * var_1)

# Parameters
n = 500
lags = range(2, 21)

# Generate series
white_noise = generate_series(n, model='white_noise')
ar1_positive = generate_series(n, model='ar1', phi=0.8)
ar1_negative = generate_series(n, model='ar1', phi=-0.8)
random_walk = generate_series(n, model='random_walk')

# Compute variance ratios
vr_white_noise = [variance_ratio(white_noise, q) for q in lags]
vr_ar1_positive = [variance_ratio(ar1_positive, q) for q in lags]
vr_ar1_negative = [variance_ratio(ar1_negative, q) for q in lags]
vr_random_walk = [variance_ratio(random_walk, q) for q in lags]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lags, vr_white_noise, label='White Noise', marker='o')
plt.plot(lags, vr_ar1_positive, label='AR(1) φ=0.8 (Momentum)', marker='s')
plt.plot(lags, vr_ar1_negative, label='AR(1) φ=-0.8 (Mean Reversion)', marker='^')
plt.plot(lags, vr_random_walk, label='Random Walk', marker='d')
plt.axhline(y=1, color='gray', linestyle='--', label='Random Walk Benchmark')
plt.xlabel('Lag (q)')
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio Test for Different Time Series Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
