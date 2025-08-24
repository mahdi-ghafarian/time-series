import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # for color normalization
import matplotlib.cm as cm # for color maps
from matplotlib.ticker import FuncFormatter # for custom tick formatting

# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
# Data parameters
# These parameters control the data fetching from Yahoo Finance.
# You can specify either a data window (e.g., last 1 year) or a specific date range.
# If use_data_window is True, data_window is used; otherwise, start_date and end_date are used.
ticker_symbol = 'LULU'  # e.g., 'AAPL', 'MSFT', 'GOOGL'
use_data_window = True # if True, use data_window; if False, use start_date and end_date
interval = '1mo' # '1d','1wk','1mo','3mo'
data_window = 'max' # ['3mo','6mo','1y','2y','5y','10y','ytd','max']
start_date = '2000-01-01' # 'YYYY-MM-DD'
end_date = '2025-01-01'  # 'YYYY-MM-DD'

# Parameters for mean reversion analysis
# These parameters control the backward and forward windows for calculating returns
# and the bin size for categorizing past returns.
backward_window = 12
forward_window = 12
auto_bin = True  # if True, bin_size is set automatically based on backward_window
number_of_bins = 20 # used only if auto_bin is True
bin_size = 0.1 # adjust based on BACKWARD_WINDOW, 0.05 works well up to 36 months 

# Figure settings
# These settings control the size and resolution of the output figure.
# Adjust these values to fit your display or publication requirements.
fig_size = (12, 6)
fig_dpi = 100

# ------------------------------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------------------------------
# Create a ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical data
if (use_data_window==True): 
    data = ticker.history(period=data_window,interval=interval)
else:
    data = ticker.history(start=start_date, end=end_date,interval=interval)

# Convert data to DataFrame
df = pd.DataFrame(data) # Date column set as index automatically

# handle empty dataframe
if df.empty:
    raise ValueError("No data fetched. Please check the ticker symbol and date range.")
else:
    print(f"Fetched {len(df)} rows of data for {ticker_symbol} from Yahoo Finance.")
    print(df)

# ------------------------------------------------------------------------------------
# Plot line chart of log average price
# ------------------------------------------------------------------------------------
df['AVG'] = (df['High'] + df['Low'] + df['Close']) / 3
df['LOG AVG'] = np.log(df['AVG'])  # log price
df['LOG AVG'].plot(title=f"{ticker_symbol} {interval}", grid=True)
plt.ylabel("Log Avg Price (HLC3)")
plt.xlabel("Date")

# ------------------------------------------------------------------------------------
# Compute past return and future return
# ------------------------------------------------------------------------------------
# log return = ln(price_2 / price_1)
# Example: price_1 = 100, price_2 = 120, log return = ln(120/100) = 0.1823
# e^0.1823 = 120/100 = 1.2, which is a 20% return
df['past_return'] = np.log(df['Close'] / df['Close'].shift(backward_window))
df['future_return'] = np.log(df['Close'].shift(-forward_window) / df['Close'])
df.dropna(subset=['past_return', 'future_return'], inplace=True)

# ------------------------------------------------------------------------------------
# Calculate bin size automatically if auto_bin is True
# ------------------------------------------------------------------------------------
if auto_bin:
    past_return_min = df['past_return'].min()
    past_return_max = df['past_return'].max()
    bin_size = (past_return_max - past_return_min) / number_of_bins
    # Round bin_size to nearest 0.01 for better readability
    bin_size = round(bin_size, 2)
    # print(f"Auto-calculated bin_size: {bin_size}")

# ------------------------------------------------------------------------------------
# Bin past returns
# ------------------------------------------------------------------------------------
x_min = np.floor(df['past_return'].min() / bin_size) * bin_size
x_max = np.ceil(df['past_return'].max() / bin_size) * bin_size
bins = np.arange(x_min, x_max + bin_size, bin_size)
df['past_return_bin'] = pd.cut(df['past_return'], bins)

# ------------------------------------------------------------------------------------
# Compute mean future return per bin and map to colors of boxees
# ------------------------------------------------------------------------------------
bin_mean = df.groupby('past_return_bin',observed=False)['future_return'].mean()
norm = mcolors.Normalize(vmin=bin_mean.min(), vmax=bin_mean.max())
cmap = plt.colormaps['RdYlGn']  # updated for matplotlib >= 3.7
bin_colors = [cmap(norm(val)) for val in bin_mean]

# Create a palette mapping each bin to its color
palette = dict(zip(bin_mean.index.astype(str), bin_colors))

# ------------------------------------------------------------------------------------
# Plot boxplot using Seaborn
# ------------------------------------------------------------------------------------
# sns.set_style("whitegrid")

# Convert bin labels to string and preserve order
df['past_return_bin_str'] = df['past_return_bin'].astype(str)
sorted_categories = df['past_return_bin'].cat.categories
df['past_return_bin_str'] = pd.Categorical(df['past_return_bin_str'],
                                           categories=[str(cat) for cat in sorted_categories],
                                           ordered=True)

# Set up the figure
fig, ax = plt.subplots(figsize=fig_size, dpi=fig_dpi)

# format y-axis as percentage, rounded to zero decimal places
# ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

# Create boxplot with color mapping
sns.boxplot(
    data=df, x='past_return_bin_str', y='future_return', palette=palette, ax=ax, showfliers=True,
    # hue='past_return_bin_str', legend=False, # future warning
    flierprops=dict(marker='o', markerfacecolor='black', markersize=10, linestyle='none', alpha=0.25)
)

# Set title and labels
plt.title(f"Mean Reversion Profile ({backward_window},{forward_window},{bin_size}) — {ticker_symbol} {interval} (Log Return)")
plt.xlabel(f"Past Log Return (t-{backward_window} to t)")
plt.ylabel(f"Future Log Return (t to t+{forward_window})")
plt.grid(True)

#  --------------------------------------------------------------------------------------
# plot customizations
#  --------------------------------------------------------------------------------------
# Set x-axis limits
ax.set_xlim(-0.5, len(sorted_categories) - 0.5)

# Format x-axis tick labels
# plt.xticks(rotation=90) # have (a,b] format by itself
xtick_labels = [f"{interval.left:.2f} to {interval.right:.2f}" for interval in sorted_categories]
ax.set_xticks(np.arange(len(xtick_labels)))
ax.set_xticklabels(xtick_labels, rotation=90)

# Format y-axis to show two decimal places
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2f}"))

# Plot mean future return line (red horizontal line)
mean_future_return = df['future_return'].mean()
ax.axhline(mean_future_return, color='red', linestyle='dotted', linewidth=2,
           label=f"Mean Future Log Return ({mean_future_return:.2f})")

# Plot horizontal line at zero future return
ax.axhline(0, color='black', linestyle='dotted', linewidth=2, label="")

# Find the bin index where zero falls
zero_bin_index = None
epsilon = 1e-6  # small value to handle floating point precision issues
for i, interval in enumerate(sorted_categories):
    if interval.left < 0 and interval.right > -1 * epsilon:  # Check if the bin contains zero
        zero_bin_index = i
        break

# Draw vertical line between boxes around zero
if zero_bin_index is not None and zero_bin_index + 1 < len(sorted_categories):
    # Position between the two boxes
    x_pos = zero_bin_index + 0.5
    ax.axvline(x=x_pos, color='black', linestyle='dotted', linewidth=2, label='Zero Return')

# Count observations in each bin
bin_counts = df['past_return_bin_str'].value_counts().sort_index()

# y-axis tick distance for annotation
offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # 5% of y-axis range
y_position = ax.get_ylim()[0] + offset  # y-position (slightly above the minimum y limit)

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
cbar = fig.colorbar(sm, ax=ax, label='Mean Future Log Return')
# two decimal places for colorbar ticks

# Add legend 
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# show the plot
plt.show()
# ------------------------------------------------------------------------------------