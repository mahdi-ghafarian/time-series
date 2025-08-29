# --------------------------------------------------------------------------------
# import necessary libraries
# --------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf # update package (pip install --upgrade yfinance) to remove connection errors
import matplotlib.pyplot as plt
import io
import sys
import inspect
import warnings

# Set display option to show all columns
# pd.set_option('display.max_columns', None)


# --------------------------------------------------------------------------------
# Ignore all warnings 
# --------------------------------------------------------------------------------
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
# Data parameters
ticker_symbol = 'SPY'  # e.g., 'AAPL', 'MSFT', 'GOOGL'
use_data_window = True # if True, use data_window; if False, use start_date and end_date
interval = '1mo' # '1d','1wk','1mo','3mo'
data_window = 'max' # ['3mo','6mo','1y','2y','5y','10y','ytd','max']
start_date = '2024-01-01' # 'YYYY-MM-DD'
end_date = '2025-01-01'  # 'YYYY-MM-DD'

# ATR 
atr_window = 30 # window for ATR calculation


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
    raise ValueError("\nNo data fetched. Please check the ticker symbol and date range.\n")
else:
    print(f"\nFetched {len(df)} rows of data for {ticker_symbol} from Yahoo Finance.\n")
    print(df)

# ------------------------------------------------------------------------------------
# Plot line chart of log average price
# ------------------------------------------------------------------------------------
df['AVG'] = (df['High'] + df['Low'] + df['Close']) / 3
df['LOG AVG'] = np.log(df['AVG'])  # log price
df['LOG AVG'].plot(title=f"{ticker_symbol} {interval}", grid=True)
plt.ylabel("Log Avg Price (HLC3)")
plt.xlabel("Date")

# plt.show()

# ------------------------------------------------------------------------------------
# Calculate true range and ATR
# ------------------------------------------------------------------------------------  
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=atr_window).mean()
df['TR%'] = df['TR'] / df['Close'] * 100
df['ATR%'] = df['TR%'].rolling(window=atr_window).mean()
df.drop(columns=['H-L', 'H-PC', 'L-PC'], inplace=True)

#------------------------------------------------------------------------------------
# Calculate average price
#------------------------------------------------------------------------------------
df['AVG'] = (df['High'] + df['Low'] + df['Close']) / 3


#------------------------------------------------------------------------------------
#calculate returns
#------------------------------------------------------------------------------------
# return as simple percentage
df['Return_C%'] = df['Close'].pct_change() * 100
#absolute of return%
df['Abs_Return_C%'] = df['Return_C%'].abs()

# return as ratio of ATR%
df['Return_C_Ratio'] = df['Return_C%'] / df['ATR%'] 
df['Abs_Return_C_Ratio'] = df['Return_C_Ratio'].abs()

# return of average price
df['Return_A%'] = df['AVG'].pct_change() * 100
df['Abs_Return_A%'] = df['Return_A%'].abs()

# return of average price as ratio of ATR%
df['Return_A_Ratio'] = df['Return_A%'] / (df['ATR%'])
df['Abs_Return_A_Ratio'] = df['Return_A_Ratio'].abs()

#------------------------------------------------------------------------------------
# Determine swing state based on average price movement
#------------------------------------------------------------------------------------
# Initial swing state assignment: UP, DOWN, or FLAT
df['swing_state'] = np.where(df['AVG'] > df['AVG'].shift(1), 'UP',
                     np.where(df['AVG'] < df['AVG'].shift(1), 'DOWN',
                     'FLAT'))

# Resolve FLAT by inheriting the previous swing_state
df['swing_state'] = df['swing_state'].where(df['swing_state'] != 'FLAT', df['swing_state'].shift(1))

# print(df[['AVG','swing_state']].tail(20))

# --------------------------------------------------------------------------------
# Extract pivot points where swing direction changes
# --------------------------------------------------------------------------------
pivot_df = df[df['swing_state'] != df['swing_state'].shift(-1)].copy()
pivot_df.reset_index(inplace=True)

# print(pivot_df[['Date','AVG','swing_state','ATR%']].tail(10))

# --------------------------------------------------------------------------------
# Construct swing_df with start/end data and swing metrics
# --------------------------------------------------------------------------------
swing_df = pd.DataFrame({
    'start_date': pivot_df['Date'].shift(1),
    'end_date': pivot_df['Date'],
    'start_price': pivot_df['AVG'].shift(1),
    'end_price': pivot_df['AVG'],
    'ATR%': pivot_df['ATR%'].shift(1),
    'swing_type': pivot_df['swing_state']
    })

swing_df.dropna(inplace=True)


# Calculate swing metrics
swing_df['Swing_Size'] = abs(swing_df['end_price'] - swing_df['start_price'])
swing_df['Swing_Size%'] = abs(swing_df['Swing_Size'] / swing_df['start_price'] * 100)
swing_df['Swing_Size_Ratio'] = swing_df['Swing_Size%'] / swing_df['ATR%']
swing_df['Swing_Duration'] = (swing_df['end_date'] - swing_df['start_date']).dt.days / (df.index.to_series().diff().dt.days.median())

# Display last few swings
# print(swing_df.tail(3))

# -----------------------------------------------------------------------------
# up swings and down swings
# -----------------------------------------------------------------------------
up_swing_df = swing_df[swing_df['swing_type'] == 'UP']
down_swing_df = swing_df[swing_df['swing_type'] == 'DOWN']

#------------------------------------------------------------------------------------
# function to print descriptive statistics
#------------------------------------------------------------------------------------
def print_descriptive_stats(df,column_name):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    df_name = next((name for name, val in callers_local_vars if val is df), "DataFrame")
    print(f"\nDescriptive statistics for {df_name}[{column_name}]:")
    #count of non-null values
    count = df[column_name].count()
    mean = df[column_name].mean()
    #standard deviation
    std = df[column_name].std()
    min_val = df[column_name].min()
    percentile_1 = df[column_name].quantile(0.01)
    percentile_5 = df[column_name].quantile(0.05)
    q1 = df[column_name].quantile(0.25)
    q2 = df[column_name].quantile(0.5) 
    q3 = df[column_name].quantile(0.75)
    percentile_95 = df[column_name].quantile(0.95)
    percentile_99 = df[column_name].quantile(0.99)
    max_val = df[column_name].max()
    print(f"Count: {count}")
    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Min: {min_val:.4f}")
    print(f"1st Percentile: {percentile_1:.4f}")
    print(f"5th Percentile: {percentile_5:.4f}")    
    print(f"25th Percentile (Q1): {q1:.4f}")
    print(f"50th Percentile (Q2): {q2:.4f}")
    print(f"75th Percentile (Q3): {q3:.4f}")
    print(f"95th Percentile: {percentile_95:.4f}")
    print(f"99th Percentile: {percentile_99:.4f}")
    print(f"Max: {max_val:.4f}")


# -------------------------------------------------------------------------------------
# Print descriptive statistics for swing metrics
# -------------------------------------------------------------------------------------
# Create a text file to write the output
with open('./swing-study/swing-stats.txt', 'w') as f:
    # Redirect stdout to a StringIO buffer
    buffer = io.StringIO()
    sys.stdout = buffer

    # Run your functions
    print_descriptive_stats(df, 'TR%')
    print_descriptive_stats(df, 'ATR%')

    print_descriptive_stats(swing_df, 'Swing_Size_Ratio')
    print_descriptive_stats(up_swing_df, 'Swing_Size_Ratio')
    print_descriptive_stats(down_swing_df, 'Swing_Size_Ratio')
    
    print_descriptive_stats(swing_df, 'Swing_Size%')
    print_descriptive_stats(up_swing_df, 'Swing_Size%')
    print_descriptive_stats(down_swing_df, 'Swing_Size%')
    
    print_descriptive_stats(swing_df, 'Swing_Duration')
    print_descriptive_stats(up_swing_df, 'Swing_Duration')
    print_descriptive_stats(down_swing_df, 'Swing_Duration')
    
    # Reset stdout to default
    sys.stdout = sys.__stdout__

    # Write buffer contents to file
    f.write(buffer.getvalue())

# ------------------------------------------------------------------------------------
# Box plots for swing metrics
# ------------------------------------------------------------------------------------
# Swing Size Ratio

# Determine common y-axis limits
# y_min = swing_df['Swing_Size_Ratio'].min()
y_min = 0  # set minimum to 0 for better visualization
y_max = swing_df['Swing_Size_Ratio'].max()

y_ticks = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 0.5, 0.5)


plt.figure(figsize=(12, 6))

# First subplot
plt.subplot(1, 3, 1)
sns.boxplot(data=swing_df, y='Swing_Size_Ratio')
plt.title('Swing Size Ratio')
plt.ylabel('ATR%')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Second subplot
plt.subplot(1, 3, 2)
sns.boxplot(data=up_swing_df, y='Swing_Size_Ratio')
plt.title('Up Swing Size Ratio')
plt.ylabel('ATR%')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Third subplot
plt.subplot(1, 3, 3)
sns.boxplot(data=down_swing_df, y='Swing_Size_Ratio')
plt.title('Down Swing Size Ratio')
plt.ylabel('ATR%')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)

plt.tight_layout()
plt.savefig('./swing-study/swing-size-ratio-boxplot.png')
# plt.show()

# ------------------------------------------------------------------------------------
# Swing Size %
# Determine common y-axis limits
# y_min = swing_df['Swing_Size%'].min()
y_min = 0
y_max = swing_df['Swing_Size%'].max()
# y_ticks = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1, 1)
plt.figure(figsize=(12, 6))
# First subplot
plt.subplot(1, 3, 1)
sns.boxplot(data=swing_df, y='Swing_Size%')
plt.title('Swing Size %')
plt.ylabel('Percentage %')
plt.ylim(y_min, y_max)
# plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)
# Second subplot
plt.subplot(1, 3, 2)
sns.boxplot(data=up_swing_df, y='Swing_Size%')
plt.title('Up Swing Size %')
plt.ylabel('Percentage %')
plt.ylim(y_min, y_max)
# plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)
# Third subplot
plt.subplot(1, 3, 3)
sns.boxplot(data=down_swing_df, y='Swing_Size%')
plt.title('Down Swing Size %')
plt.ylabel('Percentage %')
plt.ylim(y_min, y_max)
# plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('./swing-study/swing-size-percentage-boxplot.png')

# ------------------------------------------------------------------------------------
# Duration

# Determine common y-axis limits
# y_min = swing_df['Swing_Duration'].min()
y_min = 0
y_max = swing_df['Swing_Duration'].max()

y_ticks = np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1, 1)


plt.figure(figsize=(12, 6))

# First subplot
plt.subplot(1, 3, 1)
sns.boxplot(data=swing_df, y='Swing_Duration')
plt.title('Swing Duration')
plt.ylabel('Periods')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Second subplot
plt.subplot(1, 3, 2)
sns.boxplot(data=up_swing_df, y='Swing_Duration')
plt.title('Up Swing Duration')
plt.ylabel('Periods')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)

# Third subplot
plt.subplot(1, 3, 3)
sns.boxplot(data=down_swing_df, y='Swing_Duration')
plt.title('Down Swing Duration')
plt.ylabel('Periods')
plt.ylim(y_min, y_max)
plt.yticks(y_ticks)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('./swing-study/swing-duration-boxplot.png')
plt.show()
# ------------------------------------------------------------------------------------
# End of code

