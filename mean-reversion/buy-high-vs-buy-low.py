import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------------------------
# Load CSV and prepare DataFrame
df = pd.read_csv("mean-reversion\\data.csv")

# Define column names
start_date = '1940-01-01'
date = 'Date'
price = 'SP500'

# Select and format columns
df = df[[date, price]]
df[date] = pd.to_datetime(df[date], format='%Y-%m-%d', errors='coerce')
df.set_index(date, inplace=True)
df = df[df.index >= start_date]

# Add return-related columns
df['log_price'] = np.log(df[price])
df['return'] = df[price].pct_change()
df['log_return'] = np.log(df[price] / df[price].shift(1))
df['ma_10_return'] = df['return'].rolling(window=10).mean()

# Select return series
ts = df['return'].dropna()

# ------------------------------------------------------------------------------------
# Parameters
threshold = 0.05  # 5% threshold for conditional statistics
grid = 0.02  # Grid size for plotting

# ------------------------------------------------------------------------------------
# Overall statistics
print("\nOverall Statistics:\n")
print(f"N: {len(ts)}")
print(f"Mean: {ts.mean():.2%}")
print(f"Std Dev: {ts.std():.2%}")
print(f"Max: {ts.max():.2%}")
print(f"Min: {ts.min():.2%}")
print("\n")
print(f"N (R > 0): {(ts > 0).sum()}")
print(f"P (R > 0): {(ts > 0).mean():.4f}")
print(f"Mean (R > 0): {ts[ts > 0].mean():.2%}")
print("\n")
print(f"N (R < 0): {(ts < 0).sum()}")
print(f"P (R < 0): {(ts < 0).mean():.4f}")
print(f"Mean (R < 0): {ts[ts < 0].mean():.2%}")

# ------------------------------------------------------------------------------------
# Generalized conditional statistics function
def conditional_stats(ts, threshold, direction='greater'):
    if direction == 'greater':
        condition = ts.shift(1) > threshold
        label = f"Previous Return > {threshold:.2%}"
    elif direction == 'less':
        condition = ts.shift(1) < threshold
        label = f"Previous Return < {threshold:.2%}"
    else:
        raise ValueError("Direction must be 'greater' or 'less'")

    next_ts = ts[condition]

    print(f"\nConditional Statistics ({label}):\n")
    print(f"N (Condition): {condition.sum()}")
    print(f"P (Positive Return | Condition): {(next_ts > 0).mean():.2f}")
    print(f"Mean (Return | Condition): {next_ts.mean():.2%}")
    print(f"Std Dev (Return | Condition): {next_ts.std():.2%}")

# ------------------------------------------------------------------------------------
# Example usage for both directions
conditional_stats(ts, threshold=threshold, direction='greater')
conditional_stats(ts, threshold=-1 * threshold, direction='less')

# ------------------------------------------------------------------------------------
# Generalized plotting function
def plot_mean_next_return(ts, threshold_range, direction='greater'):
    mean_returns = []

    for thresh in threshold_range:
        if direction == 'greater':
            condition = ts.shift(1) > thresh
        elif direction == 'less':
            condition = ts.shift(1) < thresh
        else:
            raise ValueError("Direction must be 'greater' or 'less'")

        next_ts = ts[condition]
        mean_returns.append(next_ts.mean())

    label = f"Return (t) {('>' if direction == 'greater' else '<')} Threshold"
    color = 'darkgreen' if direction == 'greater' else 'darkred'
    marker = 'o' if direction == 'greater' else 'x'
    linestyle = '-' if direction == 'greater' else '--'

    plt.plot(threshold_range, mean_returns, marker=marker, linestyle=linestyle, label=label, color=color)

# ------------------------------------------------------------------------------------
# Plot both directions
threshold_range = np.arange(ts.min(), ts.max(), 0.01)
threshold_range = np.round(threshold_range, 2)

plt.figure(figsize=(10, 6))
plot_mean_next_return(ts, threshold_range, direction='greater')
plot_mean_next_return(ts, threshold_range, direction='less')
plt.title("Mean Reversion Profile at Market Tops and Bottoms")
plt.xlabel("Return (t)")
plt.ylabel("Return (t+1)")
plt.legend()
plt.grid(True)

# Make grid rectangular and ticks equal
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Add darker lines at (0,0)
ax.axhline(y=0, color='gray', linewidth=1)
ax.axvline(x=0, color='gray', linewidth=1)

# Add a horizontal line at overall mean
ax.axhline(y=ts.mean(), color='blue', linestyle='dotted', linewidth=1, label='Overall Mean Return')

# Round limits to nearest 0.01
x_min = np.floor(ts.min() * 100) / 100
x_max = np.ceil(ts.max() * 100) / 100
y_min = np.floor(ax.get_ylim()[0] * 100) / 100
y_max = np.ceil(ax.get_ylim()[1] * 100) / 100

# Set ticks as multiples of grid
x_ticks = np.arange(x_min, x_max + grid, grid)
y_ticks = np.arange(y_min, y_max + grid, grid)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

# Format axes as percentages
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))

plt.tight_layout()
plt.show()