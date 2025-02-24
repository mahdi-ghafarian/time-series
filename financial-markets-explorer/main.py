import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd 
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import warnings
import streamlit as st
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------

# Set page layout to wide
# Set title which is used as tab title and also as app name in 
# community cloud
st.set_page_config(layout="wide", # centered
                   page_title="Financial Markets Explorer",
                    initial_sidebar_state="expanded")

# Title
# st.title('Financial Markets Explorer')

# Get parameters from user
st.sidebar.header('Parameters')

# create three tabs for parameters
tab_data, tab_smoothing, tab_plot = \
st.sidebar.tabs(["Data", "Smoothing", "Plot"])

with tab_data: # first tab
    # ticker symbol: find from Yahoo Finance
    ticker_symbol = st.text_input(label='Ticker',value='BTC-USD',
        key='ticker_symbol',
        help = '''
            Use [Yahoo Finance](https://finance.yahoo.com) to find the ticker.
            Examples include:
            - **Crypto**: SPY, BTC-USD, ETH-USD
            - **ETF**: SPY, QQQ, GLD
            - **Index**:  ^DJI, ^IXIC, ^GSPC
            - **Stock**: GOOG, MSFT, NVDA, AAPL, AMZN, META, TSLA
            '''
        )
    # Timeframe
    tf_options = {'1d':'Daily', '1wk':'Weekly', '1mo':'Monthly'}
    interval = st.pills(label="Timeframe", options=tf_options, 
        default='1wk', selection_mode="single",
        help = 'Duration of time that each data point on the chart \
            represents.')
    
    # Data range: can be either a data window or from and to date
    # if from date is empty, first data point is used
    # if end date is empty, last data point is used
    use_data_window = st.checkbox(label='Use data window',value=True,
        help='Select data range method: data window or dates')
    
    # use a data window
    if(use_data_window):
        dw_options = ['3mo','6mo','1y','2y','5y'
            ,'10y','ytd','max'] # '1d','5d', '1mo' are also available in YF
        data_window = st.pills(label="", options=dw_options, 
            default='2y', selection_mode="single",
            label_visibility="collapsed")
    # use start date and end date
    else:
        # Get today's date
        today = datetime.now()
        # Calculate the date two years ago
        two_years_ago = today - timedelta(days=2*365)
        # set start date and end date
        start_date = st.date_input("Start date",format='YYYY-MM-DD',
            value = two_years_ago)
        end_date = st.date_input("End date",format='YYYY-MM-DD',
            value = today)

with tab_smoothing: # second tab
    # LOWESS parameters
    # no of reweightings (default=3)
    it = st.number_input('Number of reweightings',value=3) 
    # amount of data used for smoothing 
    use_period = st.checkbox(label='Use smoothing period',value=True,
        help='Select smoothing window method: period or bandwidth')
    if use_period:
        # monthly: 12, weekly: 52 
        lt_period = st.number_input('Long-term period',value=50,
            min_value=1)
        # monthly: 1, weekly: 4
        st_period = st.number_input('Short-term period',value=10,
            min_value=1)
    else:
        # adjust, common values: 0.25, 0.5
        lt_bw = st.number_input('Long-term bandwidth',value=0.5,
            min_value=0.01, max_value=1.0)
        # 0.1 x lt_bw
        st_bw = st.number_input('Long-term bandwidth',value=0.05,
            min_value=0.01, max_value=1.0)

with tab_plot: # third tab
    # Plot parameters
    height = st.number_input('Heigth',value=1000, min_value=0)  
    # 0: program selects the y_grid
    y_grid = st.number_input('Y-axis grid',value=0.0,
        min_value=0.0, step=0.1, help = 'Enter `0` for automatic selection') 

# About the App
st.sidebar.header('About')
with st.sidebar.expander(label='About this application...'):
    st.markdown('''
                - Retrieves data for the selected ticker from Yahoo Finance.
                - Applies LOWESS smoothing to the time series to identify the trend.
                - Calculates and standardize residuals, which can be used as an oscillator.
                - Plots the smoothed series and the oscillator
                ''')
# ------------------------------------------------------------------------------ 
# Get data
# ------------------------------------------------------------------------------
# Ignore all warnings 
warnings.filterwarnings('ignore')

# Create a divider
# st.divider()

# Create a ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical data
if (use_data_window==True): 
    data = ticker.history(period=data_window,interval=interval)
else:
    data = ticker.history(start=start_date, end=end_date,interval=interval)

# Convert data to DataFrame
df = pd.DataFrame(data)

# Error handling
if(len(df)==0):
    st.warning('Ticker does not exist.\
        Check [Yahoo Finance](https://finance.yahoo.com) to find the \
            correct format.', icon="⚠️")
    st.stop()

# ------------------------------------------------------------------------------ 
# Calculations
# ------------------------------------------------------------------------------

# Transformation
df['avg'] = (df['High']+df['Low']+df['Close']) / 3
df['l_avg'] = np.log(df['avg'])

# number of data points
n_rows = len(df)

#calculate badwidth from smoothing periods
if(
    (use_period == True) and
    (lt_period < n_rows ) and
    (lt_period < n_rows )
):
    lt_bw = lt_period/n_rows
    st_bw = st_period/n_rows
else:
    st.warning('Not enough data. Increase data window.', icon="⚠️")
    st.stop()  

#LOESS
lt_loess = sm.nonparametric.lowess(df['l_avg'], np.arange(len(df)),it=it, frac=lt_bw)
st_loess = sm.nonparametric.lowess(df['l_avg'], np.arange(len(df)),it=it, frac=st_bw)

# Add results to the DF
df['lt_lowess']=lt_loess[:,1]
df['st_lowess']=st_loess[:,1]

# Residual
df['residual']=df['st_lowess'] - df['lt_lowess']

#Standardization
res_mean = df['residual'].mean()
res_std = df['residual'].std()
df['std_residual'] = (df['residual'] - res_mean)/res_std

# ------------------------------------------------------------------------------ 
# Main Figure
# ------------------------------------------------------------------------------
# Set the default template
pio.templates.default = "seaborn"

# Create chart title
if(use_period == True):
    title = f'{ticker_symbol}: LOWESS Regression ({interval},{lt_period},{st_period})'
else:
    title = f'{ticker_symbol}: LOWESS Regression ({interval},{lt_bw:.2f},{st_bw:.2f})'

# Display the title for main figure
st.header(title)

# Create a figure with 2 rows and 1 column, sharing the x-axis
main_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    row_heights=[0.75, 0.25], vertical_spacing=0.01)

# Update the figure size 
main_fig.update_layout(height=height)

# Add vertical lines at the start of each year
for year in range(df.index.year.min()+1, df.index.year.max() + 1):
    main_fig.add_shape(
        type='line',
        x0=pd.Timestamp(f'{year}-01-01'),
        y0=0,
        x1=pd.Timestamp(f'{year}-01-01'),
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color='gray', width=0.5, dash='solid')
    )

# ------------------------------------------------------------------------------ 
# Top Figure
# ------------------------------------------------------------------------------
# colors of lines
colors = ['#00CC96','#EF553B','#636EFA']

# Plot time series, short-term and long-term smoothed series
top_fig = px.line(df, x=df.index, y=['l_avg','st_lowess','lt_lowess'], 
              title=title, labels={'variable': 'Time Series'},
              color_discrete_sequence=colors)

# Change the y-axis label 
top_fig.update_layout(yaxis_title='Log Price')
    
# Update the y-axis to have grid lines at each integer
top_fig.update_yaxes(dtick=y_grid)

# Center the title
# fig1.update_layout(title_x=0.5)

# Update the visibility of one of the traces to be 'legendonly'
for trace in top_fig.data:
    if trace.name == 'st_lowess':
        trace.visible = 'legendonly'  # Hide l_avg by default
        
# Update the legend labels for each line
top_fig.for_each_trace(lambda trace: trace.update( 
    name=trace.name.replace('l_avg', 'Log Price').
    replace('lt_lowess', 'Long-term Trend').
    replace('st_lowess', 'Short-term Trend')
    )
)

# ------------------------------------------------------------------------------ 
# Bottom Figure
# ------------------------------------------------------------------------------

# Standard residual figure
bot_fig = px.line(df, x=df.index, y=['std_residual'], 
              labels={'variable': 'Time Series'},
              title = 'Residual Oscillator')

# Change the y-axis label 
#fig2.update_layout(yaxis_title='Standard Residual')


# Update the y-axis to have grid lines at each integer
bot_fig.update_yaxes(dtick=1)

# Add hrizontal lines
# y = 1
bot_fig.add_shape(
    type='line',
    x0=df.index.min(),
    y0=1,
    x1=df.index.max(),
    y1=1,
    line=dict(color='black', width=0.5, dash='dot'),
)
# y = -1
bot_fig.add_shape(
    type='line',
    x0=df.index.min(),
    y0=-1,
    x1=df.index.max(),
    y1=-1,
    line=dict(color='black', width=0.5, dash='dot'),
)
# y = 0
bot_fig.add_shape(
    type='line',
    x0=df.index.min(),
    y0=0,
    x1=df.index.max(),
    y1=0,
    line=dict(color='black', width=0.5, dash='solid'),
)

# Update the legend labels for each line
bot_fig.for_each_trace(lambda trace: trace.update( 
    name=trace.name.replace('std_residual', 'Std. Residual')
    )
)

# ------------------------------------------------------------------------------ 
# Display Output
# ------------------------------------------------------------------------------

# Add the trace from the top_fig the subplot (main_fig)
for trace in top_fig.data:
    main_fig.add_trace(trace, row=1, col=1)
# Add the trace from the bot_fig to the subplot (main_fig)
for trace in bot_fig.data:
    main_fig.add_trace(trace, row=2, col=1)

# show the plot
st.plotly_chart(main_fig)

# Display a divider
st.divider()

# Display Statistics
st.subheader('Statistics')
# Residuals
st.write('Residuals Mean:',round(res_mean,2))
st.write('Residuals Standard Deviation:',round(res_std,2))

# Display latest data
st.subheader('Latest Data')
st.write('Number of data points: ', df.shape[0])
st.write(df[['avg','l_avg']].tail(10))
