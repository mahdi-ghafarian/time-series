import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------------

# Create a DataFrame and load a CSV
df = pd.read_csv("hurst-exponent\\data.csv")

# Define column names in the CSV file
# Change these as per your CSV file
start_date = '2000-01-01'
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
df['log_return'] = np.log(df[price] / df[price].shift(1)).dropna()
df['ma_10_return'] = df['return'].rolling(window=10).mean().dropna()

# Select the column for hurst exponent calculation
ts = df['return']

# ------------------------------------------------------------------------------------
# Set your threshold
threshold = 0.01

# Overall statistics
total_observations = len(ts)
mean_return = ts.mean()
std_dev_return = ts.std()
max_return = ts.max()
min_return = ts.min()
greater_than_zero = (ts > 0).sum()
prob_greater_than_zero = (ts > 0).mean()
greater_than_threshold = (ts > threshold).sum()
prob_greater_than_threshold = (ts > threshold).mean()

print("\nOverall Statistics:\n")
print(f"N: {total_observations}")
print(f"Mean: {mean_return:.2%}")
print(f"Std Dev: {std_dev_return:.2%}")
print(f"Max: {max_return:.2%}")
print(f"Min: {min_return:.2%}\n")
print(f"N (R > 0): {greater_than_zero}")
print(f"P (R > 0): {prob_greater_than_zero:.4f}\n")
print(f"N (R > {threshold}): {greater_than_threshold}")
print(f"P (R > {threshold}): {prob_greater_than_threshold:.4f}")


# Create a mask for current bar return > threshold
# shift(1) to refer to previous bar
condition = ts.shift(1) > threshold

# Filter next ts based on condition
next_ts = ts[condition]

# Calculate conditional probability and statistics
frequency = condition.sum()
probability = (next_ts > 0).mean()
mean = next_ts.mean()
std_dev = next_ts.std()

# Output results
print("\nConditional Statistics (Condition: Previous Return > Threshold):\n")
print(f"N (Condition): {frequency}")
print(f"P (Positive Rturn | Condition): {probability:.2f}\n")
print(f"Mean (Return | Condition): {mean:.2%}")
print(f"Std Dev (Return | Condition): {std_dev:.2%}")

# ------------------------------------------------------------------------------------
# Plot next_ts.mean() vs threshold

# Define a range of thresholds to test
threshold_range = np.arange(0, 0.12, 0.01)  # Adjust range and step as needed
mean_returns = []

for thresh in threshold_range:
    condition = ts.shift(1) > thresh
    next_ts = ts[condition]
    mean_returns.append(next_ts.mean())

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(threshold_range, mean_returns, marker='o', linestyle='-', color='darkgreen')
plt.title("Mean of Next Return vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Mean of Next Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------