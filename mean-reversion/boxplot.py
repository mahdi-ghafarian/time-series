import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
file = "mean-reversion\\sp500-monthly.csv"
date_col = 'Date'
price_col = 'SP500'
start_date = '1943-01-01' # Oldest data available for SP500, 1871-01-01

backward_window = 12
forward_window = 12
bin_size = 0.05 # 0.05 works weel up to 60 months, after that 0.1 is better

fig_size = (12, 6)
fig_dpi = 100

# ------------------------------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------------------------------
df = pd.read_csv(file)
df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
df.set_index(date_col, inplace=True)
df = df[df.index >= start_date]

# ------------------------------------------------------------------------------------
# Compute signal and outcome
# ------------------------------------------------------------------------------------
df['past_return'] = df[price_col] / df[price_col].shift(backward_window) - 1
df['future_return'] = df[price_col].shift(-forward_window) / df[price_col] - 1
df.dropna(subset=['past_return', 'future_return'], inplace=True)

# ------------------------------------------------------------------------------------
# Bin past returns
# ------------------------------------------------------------------------------------
x_min = np.floor(df['past_return'].min() / bin_size) * bin_size
x_max = np.ceil(df['past_return'].max() / bin_size) * bin_size
bins = np.arange(x_min, x_max + bin_size, bin_size)
df['past_return_bin'] = pd.cut(df['past_return'], bins)

# ------------------------------------------------------------------------------------
# Compute mean future return per bin and map to colors
# ------------------------------------------------------------------------------------
bin_means = df.groupby('past_return_bin')['future_return'].mean()
norm = mcolors.Normalize(vmin=bin_means.min(), vmax=bin_means.max())
cmap = cm.get_cmap('RdYlGn')
bin_colors = [cmap(norm(val)) for val in bin_means]

# Create a palette mapping each bin to its color
palette = dict(zip(bin_means.index, bin_colors))

# ------------------------------------------------------------------------------------
# Plot boxplot using Seaborn
# ------------------------------------------------------------------------------------
# plt.style.use('default')  # Resets to default style
# sns.set_style("whitegrid")

# Set up the figure
plt.figure(figsize=fig_size, dpi=fig_dpi)

# Create boxplot with color mapping
ax = sns.boxplot(
    data=df, x='past_return_bin', y='future_return', palette=palette, showfliers=True, 
    flierprops=dict(marker='o', markerfacecolor='black', markersize=10,linestyle='none', alpha=0.25)
)

# Set title and labels
plt.title(f"Mean Reversion Profile ({backward_window},{forward_window})")
plt.xlabel(f"Past Return (t-{backward_window} to t)")
plt.ylabel(f"Future Return (t to t+{forward_window})")
plt.grid(True)

# Format x-axis tick labels
# plt.xticks(rotation=90) # have (a,b] formt by itself
xtick_labels = [f"{interval.left:.2f} to {interval.right:.2f}" for interval in df['past_return_bin'].cat.categories]
plt.xticks(ticks=np.arange(len(xtick_labels)), labels=xtick_labels, rotation=90)

# Plot mean future return line
mean_future_return = df['future_return'].mean()
plt.axhline(mean_future_return, color='red', linestyle='dotted', linewidth=2,
            label=f"Mean Future Return ({mean_future_return:.2%})")

# Plot horizontal line at zero pasfuture return
plt.axhline(0, color='black', linestyle='dotted', linewidth=2,
            label="")

# Find the bin index where zero falls
zero_bin_index = None
for i, interval in enumerate(df['past_return_bin'].cat.categories):
    if interval.left < 0 and interval.right > 0:
        zero_bin_index = i
        break

# Draw vertical line between boxes around zero
if zero_bin_index is not None and zero_bin_index + 1 < len(df['past_return_bin'].cat.categories):
    # Position between the two boxes
    x_pos = zero_bin_index + 0.5
    plt.axvline(x=x_pos, color='black', linestyle='dotted', linewidth=2, label='')

# count observations in each bin
bin_counts = df['past_return_bin'].value_counts().sort_index()

# y-axis tick distance for annotation
offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 5% of y-axis range
y_position = ax.get_ylim()[0] + offset # y-position (slightly above the minimum y limit)

# Annotate number of observations per bin
for i, count in enumerate(bin_counts):
    ax.text(
        i,  # x-position (bin index)
        y_position, 
        f"{count}",     # text to display
        ha='center', va='top', fontsize=9, color='black'
    )

# Add colorbar to show mapping
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='Mean Future Return')

plt.legend()
plt.tight_layout()
plt.show()
# ------------------------------------------------------------------------------------