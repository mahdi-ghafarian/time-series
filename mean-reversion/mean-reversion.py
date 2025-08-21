import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import linregress

# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
# data parameters
file = "mean-reversion\\sp500-monthly.csv" # Path to the CSV file
date = 'Date' # Date column name
price = 'SP500' # Price column name
start_date = '1943-01-01' # Start date for analysis, format 'YYYY-MM-DD'

# signal/outcome parameters
backward_window = 60 # backward window for signal calculation
forward_window = 60 # forward window for outcome calculation
bin_size = 0.01 # bin size for signal binning (blue dots on plot)

# plotting parameters
fig_size = (10,6) # figure width and height in inches (window size)
fig_dpi = 100 # figure resolution in dots per inch
aspect_ratio = 0.6 # height/width ratio for the plot
custom_grid = True # whether to use custom grid spacing
grid_x = 0.2 # grid spacing for x-axis
grid_y = 0.2 # grid spacing for y-axis

# std dev ranges, used for range-based statistics, add or remove as needed
std_dev_ranges = [-3, -2, -1, 0, 1, 2, 3]

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
df.dropna(subset=['signal', 'outcome'], inplace=True)
# Round signal to nearest bin size
ts_signal = ((df['signal'] / bin_size).round() * bin_size).round(2)
# Extract the outcome series
ts_outcome = df['outcome']

# ------------------------------------------------------------------------------------
# Summary Statistics
print("\nPast Return Summary:")
print(f"Count: {ts_signal.count()}")
print(f"Mean: {ts_signal.mean():.1%}")
print(f"Std Dev: {ts_signal.std():.1%}")
print(f"Min: {ts_signal.min():.1%}")
print(f"Max: {ts_signal.max():.1%}")
print(f"Percentage > 0: {(ts_signal > 0).mean():.1%}")
print(f"Mean (Positive): {ts_signal[ts_signal > 0].mean():.1%}")
print(f"Mean (Negative): {ts_signal[ts_signal < 0].mean():.1%}")

print("\nFuture Return Summary:")
print(f"Count: {ts_outcome.count()}")
print(f"Mean: {ts_outcome.mean():.1%}")
print(f"Std Dev: {ts_outcome.std():.1%}")
print(f"Min: {ts_outcome.min():.1%}")
print(f"Max: {ts_outcome.max():.1%}")
print(f"Percentage > 0: {(ts_outcome > 0).mean():.1%}")
print(f"Mean (Positive): {ts_outcome[ts_outcome > 0].mean():.1%}")
print(f"Mean (Negative): {ts_outcome[ts_outcome < 0].mean():.1%}")

# ------------------------------------------------------------------------------------
# Detailed Stats Function
def detailed_stats(subset, label):
    count = subset.count()
    mean = subset.mean()
    min_val = subset.min()
    max_val = subset.max()
    pct_positive = (subset > 0).mean()
    mean_positive = subset[subset > 0].mean()
    mean_negative = subset[subset < 0].mean()

    print(f"\n{label}:")
    print(f"Count: {count}")
    print(f"Mean: {mean:.1%}")
    print(f"Std Dev: {subset.std():.1%}")
    print(f"Min: {min_val:.1%}")
    print(f"Max: {max_val:.2%}")
    print(f"Percentage > 0: {pct_positive:.1%}")
    print(f"Mean (Positive): {mean_positive:.1%}")
    print(f"Mean (Negative): {mean_negative:.1%}")

# ------------------------------------------------------------------------------------
# Range-Based Signal Statistics
signal_mean = ts_signal.mean()
signal_std = ts_signal.std()

# Below lower bound
lower_extreme_mask = ts_signal < signal_mean + std_dev_ranges[0] * signal_std
lower_extreme = ts_outcome[lower_extreme_mask]
if not lower_extreme.empty:
    detailed_stats(lower_extreme, f"Signal < {std_dev_ranges[0]}σ")

# Between lower and upper bounds
for i in range(len(std_dev_ranges) - 1):
    lower = signal_mean + std_dev_ranges[i] * signal_std
    upper = signal_mean + std_dev_ranges[i + 1] * signal_std
    mask = (ts_signal >= lower) & (ts_signal < upper)
    label = f"Signal in [{std_dev_ranges[i]}σ, {std_dev_ranges[i+1]}σ)"
    subset = ts_outcome[mask]
    if not subset.empty:
        detailed_stats(subset, label)

# Beyond upper bound
upper_extreme_mask = ts_signal >= signal_mean + std_dev_ranges[-1] * signal_std
upper_extreme = ts_outcome[upper_extreme_mask]
if not upper_extreme.empty:
    detailed_stats(upper_extreme, f"Signal > {std_dev_ranges[-1]}σ")

# ------------------------------------------------------------------------------------
# Conditional statistics for signal → outcome
def conditional_stats_equal(signal, outcome, bin):
    condition = signal == bin
    next_ts = outcome[condition]
    stats = {
        'Bin': bin,
        'Count': condition.sum(),
        'P(Positive)': round((next_ts > 0).mean(), 2),
        'Mean Return': round(next_ts.mean(), 2),
        'Std Dev': round(next_ts.std(), 2)
    }
    return stats

# ------------------------------------------------------------------------------------
# Plotting Function with Tabular Output
def plot_return(signal, outcome, bin_range, custom_grid, figsize=(10, 6), dpi=100):
    
    # Compute conditional statistics for each bin
    stats_list = [conditional_stats_equal(signal, outcome, bin) for bin in bin_range]
    stats_df = pd.DataFrame(stats_list) # Convert list of dicts to DataFrame
    stats_df.dropna(subset=['Mean Return'], inplace=True) # Remove bins with no data

    # Create the plot
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Scatter all points in light gray
    plt.scatter(signal, outcome, color='lightgray', alpha=0.6, s=100)
    
    # Overlay bin means in green and red
    # colors = ['green' if val >= 0 else 'red' for val in stats_df['Mean Return']]
    # plt.scatter(stats_df['Bin'], stats_df['Mean Return'], color=colors, s=100, alpha=0.6)
   
    # Overlay bin means in blue
    plt.scatter(stats_df['Bin'], stats_df['Mean Return'], color='blue', s=100, alpha=0.6)

    # Perform linear regression using scipy
    slope, intercept, r_value, p_value, std_err = linregress(signal, outcome)
    reg_x = np.linspace(signal.min(), signal.max(), 100)
    reg_y = slope * reg_x + intercept

    # Add regression line with stats to plot
    plt.plot(reg_x, reg_y, color='blue', linestyle='solid', linewidth=1.5,
            label=f'y = {slope:.2f}x + {intercept:.3f}\nR = {r_value:.2f}, p = {p_value:.4f}')

    # Set plot title and labels
    plt.title(f"Mean Reversion Profile ({backward_window},{forward_window})")
    plt.xlabel(f"Past Return (t-{backward_window} to t)")
    plt.ylabel(f"Future Return (t to t+{forward_window})")
    plt.grid(True)

    # Get current axis for further customization
    ax = plt.gca()
    
    # Set aspect ratio
    ax.set_box_aspect(aspect_ratio)
        
    # Set custom grid if specified
    if custom_grid:
        x_min, x_max = np.floor(ts_signal.min() * 10) / 10, np.ceil(ts_signal.max() * 10) / 10
        y_min, y_max = np.floor(outcome.min() * 10) / 10, np.ceil(outcome.max() * 10) / 10
        plt.xticks(np.arange(x_min, x_max, grid_x))
        plt.yticks(np.arange(y_min, y_max, grid_y))
    
    # Set axis limits
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    # Add horizontal and vertical lines for mean
    ax.axhline(y=0, color='black', linewidth=1.0)
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.axhline(y=outcome.mean(), color='red', linestyle='dotted', linewidth=2,
               label=f'Mean (FR) = {ts_outcome.mean():.1%}')
    ax.axvline(x=signal.mean(), color='red', linestyle='dotted', linewidth=2,
               label=f'Mean (PR) = {ts_signal.mean():.1%}')
    
    # Fill between for standard deviation ranges``
    ax.fill_betweenx(np.linspace(outcome.min(), outcome.max(), 100),
                     signal.mean() - signal.std(),
                     signal.mean() + signal.std(),
                     color='green', alpha=0.1, label='1 std dev (PR)')
    ax.fill_betweenx(np.linspace(outcome.min(), outcome.max(), 100),
                     signal.mean() - 2 * signal.std(),
                     signal.mean() - 1 * signal.std(),
                     color='orange', alpha=0.1, label='2 std dev (PR)')
    ax.fill_betweenx(np.linspace(outcome.min(), outcome.max(), 100),
                     signal.mean() + 1 * signal.std(),
                     signal.mean() + 2 * signal.std(),
                     color='orange', alpha=0.1)
    ax.fill_betweenx(np.linspace(outcome.min(), outcome.max(), 100),
                     signal.mean() - 3 * signal.std(),
                     signal.mean() - 2 * signal.std(),
                     color='red', alpha=0.1, label='3 std dev (PR)')
    ax.fill_betweenx(np.linspace(outcome.min(), outcome.max(), 100),
                     signal.mean() + 2 * signal.std(),
                     signal.mean() + 3 * signal.std(),
                     color='red', alpha=0.1)

    plt.legend(loc='upper right') # Add legend
    plt.tight_layout() # Adjust layout to prevent overlap
    
    # Show the plot
    plt.show()

    # Print table
    print("\nConditional Statistics Table:")
    print(stats_df.to_string(index=False))

# ------------------------------------------------------------------------------------

# Bin range for signal binning
bin_range = np.round(np.arange(ts_signal.min(), ts_signal.max(), bin_size), 2)

# Run the plot
plot_return(ts_signal, ts_outcome, bin_range, custom_grid=custom_grid,
            figsize=fig_size, dpi=fig_dpi)

# ------------------------------------------------------------------------------------
#-- End of script
# ------------------------------------------------------------------------------------