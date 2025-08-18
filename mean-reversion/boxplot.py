import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
# Data parameters
file = "mean-reversion\\sp500-monthly.csv"
date_col = 'Date'
price_col = 'SP500'
start_date = '1943-01-01'

# Signal/outcome parameters
backward_window = 12
forward_window = 12
bin_size = 0.05

# Plotting parameters
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
# Plot boxplot using Seaborn
# ------------------------------------------------------------------------------------
plt.figure(figsize=fig_size, dpi=fig_dpi)
sns.boxplot(data=df, x='past_return_bin', y='future_return', palette='Blues', showfliers=True)

plt.title(f"MRP Box Plot ({backward_window},{forward_window})")
plt.xlabel(f"Past Return (t-{backward_window} to t)")
plt.ylabel(f"Future Return (t to t+{forward_window})")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()