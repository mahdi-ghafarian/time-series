import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------------------------
# Parameters
date = 'Date'
price = 'SP500'
start_date = '1940-01-01'
backward_window = 12  # Look-back window for signal
forward_window = 60  # Look-ahead window for outcome
custom_grid = False  # Set to True to use custom ticks, False for automatic
grid_x=0.10   # Grid spacing for ticks, can be adjusted based on data range
grid_y=0.10   # Grid spacing for ticks, can be adjusted based on data range

# ------------------------------------------------------------------------------------
# Load CSV and prepare DataFrame
df = pd.read_csv("mean-reversion\\data.csv")
df[date] = pd.to_datetime(df[date], format='%Y-%m-%d', errors='coerce')
df.set_index(date, inplace=True)
df = df[df.index >= start_date]

# ------------------------------------------------------------------------------------
# Compute signal and outcome returns
df['signal'] = df[price] / df[price].shift(backward_window) - 1
df['outcome'] = df[price].shift(-forward_window) / df[price] - 1

# Drop rows with NaNs in either column
df.dropna(subset=['signal', 'outcome'], inplace=True)

# Round signal for binning
ts_signal = df['signal'].round(2)
ts_outcome = df['outcome']

# ------------------------------------------------------------------------------------
# Summary Statistics
print("\nPast Return Summary:")
print(f"Count: {ts_signal.count()}")
print(f"Mean: {ts_signal.mean():.2%}")
print(f"Std Dev: {ts_signal.std():.2%}")
print(f"Min: {ts_signal.min():.2%}")
print(f"Max: {ts_signal.max():.2%}")

print("\nFuture Return Summary:")
print(f"Count: {ts_outcome.count()}")
print(f"Mean: {ts_outcome.mean():.2%}")
print(f"Std Dev: {ts_outcome.std():.2%}")
print(f"Min: {ts_outcome.min():.2%}")
print(f"Max: {ts_outcome.max():.2%}")

# ------------------------------------------------------------------------------------
# Threshold range for signal binning
threshold_range = np.round(np.arange(ts_signal.min(), ts_signal.max(), 0.01), 2)

# ------------------------------------------------------------------------------------
# Conditional statistics for signal → outcome
def conditional_stats_equal(signal, outcome, threshold):
    condition = signal == threshold
    next_ts = outcome[condition]
    stats = {
        'Threshold': threshold,
        'Count': condition.sum(),
        'P(Positive)': round((next_ts > 0).mean(), 2),
        'Mean Return': round(next_ts.mean(), 4),
        'Std Dev': round(next_ts.std(), 4)
    }
    return stats

# ------------------------------------------------------------------------------------
# Plotting Function with Tabular Output
def plot_return(signal, outcome, threshold_range, custom_grid):
    stats_list = [conditional_stats_equal(signal, outcome, thresh) for thresh in threshold_range]
    stats_df = pd.DataFrame(stats_list)

    # Drop rows with NaN mean returns
    stats_df.dropna(subset=['Mean Return'], inplace=True)

    # figure setup
    plt.figure(figsize=(10, 6), dpi=100)

    # Scatterplot of all individual returns
    plt.scatter(signal, outcome, color='lightgray', alpha=0.6, s=100, label='')

    # Scatterplot of mean returns per threshold
    plt.scatter(stats_df['Threshold'], stats_df['Mean Return'], color='blue', 
                label='', s=100, alpha=0.6)

    # Add regression line
    coeffs = np.polyfit(signal, outcome, deg=1)
    reg_x = np.linspace(signal.min(), signal.max(), 100)
    reg_y = coeffs[0] * reg_x + coeffs[1]
    plt.plot(reg_x, reg_y, color='blue', linestyle='solid', linewidth=1.5,
             label=f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')

    # Add labels and title
    plt.title(f"Mean Reversion Profile ({backward_window},{forward_window})")
    plt.xlabel(f"Past Return (t-{backward_window} to t)")
    plt.ylabel(f"Future Return (t to t+{forward_window})")
    plt.grid(True)

    ax = plt.gca()
    
    # Make grid rectangular and ticks equal, uncomment if needed
    # ax.set_aspect('equal', adjustable='box')
    
    if custom_grid:
        # Axis limits and adaptive ticks
        x_min, x_max = np.floor(ts_signal.min() * 100) / 100, np.ceil(ts_signal.max() * 100) / 100
        y_min, y_max = np.floor(outcome.min() * 100) / 100, np.ceil(outcome.max() * 100) / 100

        # Set ticks and grid
        plt.xticks(np.arange(x_min, x_max , grid_x))
        plt.yticks(np.arange(y_min, y_max, grid_y))
    
    # Format ticks as percentages
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    
    # Add annotations lines to the chart
    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.axhline(y=ts_outcome.mean(), color='red', linestyle='dotted', linewidth=2, 
               label=f'Mean Future Return = {ts_outcome.mean():.2%}')

    # Add legend and layout adjustments
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    # Print table
    print("\nConditional Statistics Table:")
    print(stats_df.to_string(index=False))

# ------------------------------------------------------------------------------------
# Run the plot
plot_return(ts_signal, ts_outcome, threshold_range, custom_grid=custom_grid)

#-- End of script
# ------------------------------------------------------------------------------------