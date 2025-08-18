import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------------------------
# Parameters
#-----------------------------------------------------------------------------------
# data parameters
file = "mean-reversion\\sp500.csv" # Path to the CSV file containing data
date = 'Date' # Column name for date
price = 'SP500' # Column name for price (close or average price)
start_date = '1943-01-01' # Start date for analysis, format 'YYYY-MM-DD'

# signal/outcome parameters
backward_window = 12  # Look-back window for signal
forward_window = 60  # Look-ahead window for outcome
bin_size = 0.01  # Size of bins for signal values

# plotting parameters
fig_size = (10, 6) # Size of the figure
fig_dpi = 100 # DPI for the figure
custom_grid = True  # Set to True to use custom ticks, False for automatic
grid_x=0.10   # Grid spacing for ticks, can be adjusted based on data range
grid_y=0.20   # Grid spacing for ticks, can be adjusted based on data range

# ------------------------------------------------------------------------------------
# Load CSV and prepare DataFrame
df = pd.read_csv(file)
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
print(f"Mean: {ts_signal.mean():.1%}")
print(f"Std Dev: {ts_signal.std():.1%}")
print(f"Min: {ts_signal.min():.1%}")
print(f"Max: {ts_signal.max():.1%}")

print("\nFuture Return Summary:")
print(f"Count: {ts_outcome.count()}")
print(f"Mean: {ts_outcome.mean():.1%}")
print(f"Std Dev: {ts_outcome.std():.1%}")
print(f"Min: {ts_outcome.min():.1%}")
print(f"Max: {ts_outcome.max():.1%}")

# ------------------------------------------------------------------------------------
# Threshold range for signal binning
threshold_range = np.round(np.arange(ts_signal.min(), ts_signal.max(), bin_size), 2)

# ------------------------------------------------------------------------------------
# Conditional statistics for signal → outcome
def conditional_stats_equal(signal, outcome, threshold):
    condition = signal == threshold
    next_ts = outcome[condition]
    stats = {
        'Threshold': threshold,
        'Count': condition.sum(),
        'P(Positive)': round((next_ts > 0).mean(), 2),
        'Mean Return': round(next_ts.mean(), 2),
        'Std Dev': round(next_ts.std(), 2)
    }
    return stats

# ------------------------------------------------------------------------------------
# Plotting Function with Tabular Output
def plot_return(signal, outcome, threshold_range, custom_grid,figsize=(10, 6), dpi=100):
    stats_list = [conditional_stats_equal(signal, outcome, thresh) for thresh in threshold_range]
    stats_df = pd.DataFrame(stats_list)

    # Drop rows with NaN mean returns
    stats_df.dropna(subset=['Mean Return'], inplace=True)

    # figure setup
    plt.figure(figsize=fig_size, dpi=dpi)

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
        x_min, x_max = np.floor(ts_signal.min() * 10) / 10, np.ceil(ts_signal.max() * 10) / 10
        y_min, y_max = np.floor(outcome.min() * 10) / 10, np.ceil(outcome.max() * 10) / 10

        # Set ticks and grid
        plt.xticks(np.arange(x_min, x_max , grid_x))
        plt.yticks(np.arange(y_min, y_max, grid_y))
    
    # Format ticks as percentages
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    
    # Add annotations lines to the chart
    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.axhline(y=outcome.mean(), color='red', linestyle='dotted', linewidth=2, 
               label=f'Mean (FR) = {ts_outcome.mean():.2%}')
    ax.axvline(x=signal.mean(), color='red', linestyle='dotted', linewidth=2, 
               label=f'Mean (PR) = {ts_signal.mean():.2%}')
    # confidence interval of ts_signal
    ax.fill_betweenx(y=np.linspace(outcome.min(), outcome.max(), 100), 
                     x1=signal.mean() - signal.std(), 
                     x2=signal.mean() + signal.std(), 
                     color='blue', alpha=0.1, label='1 std dev (PR)')
    ax.fill_betweenx(y=np.linspace(outcome.min(), outcome.max(), 100), 
                     x1=signal.mean() - 2 * signal.std(), 
                     x2=signal.mean() - 1 * signal.std(),
                     color='red', alpha=0.1, label='2 std dev (PR)')
    ax.fill_betweenx(y=np.linspace(outcome.min(), outcome.max(), 100), 
                     x1=signal.mean() + 1 * signal.std(), 
                     x2=signal.mean() + 2 * signal.std(),
                     color='red', alpha=0.1, label='')    

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
plot_return(ts_signal, ts_outcome, threshold_range, custom_grid=custom_grid,
            figsize=fig_size, dpi=fig_dpi)

#-- End of script
# ------------------------------------------------------------------------------------