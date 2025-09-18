#-----------------------------------------------------------
# This script simulates a price path using Geometric Brownian Motion (GBM). 
# GBM is commonly used in financial mathematics to model stock prices.
# It assumes that the logarithm of the price follows a Brownian motion with drift.
#-----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Parameters
P0 = 100       # Initial price
mu = 0.05      # Drift (expected return)
sigma = 0.2    # Volatility
T = 1          # Time horizon in years
N = 252        # Number of time steps (e.g., trading days in a year)

# Time increment
dt = T / N

# Generate random standard normal variables
Z = np.random.standard_normal(N)

# Simulate Brownian motion
W = np.cumsum(Z) * np.sqrt(dt)

# Time vector
t = np.linspace(0, T, N)

# Simulate GBM price path
P_t = P0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

# Plot the simulated price path
plt.figure(figsize=(10, 6))
plt.plot(t, P_t, label='GBM Price Path')
plt.title('Geometric Brownian Motion Simulation')
plt.xlabel('Time (Years)')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()