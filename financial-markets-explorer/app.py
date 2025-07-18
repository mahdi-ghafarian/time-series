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
# UI Configuration
# ------------------------------------------------------------------------------

# Set page layout to wide
# Set title which is used as tab title and also as app name in 
# community cloud
st.set_page_config(layout="wide", # centered
                   page_title="Explorer",
                    initial_sidebar_state="expanded")

# ------------------------------------------------------------------------------
# Create sidebar to get parameters
# ------------------------------------------------------------------------------

# Get parameters from user
st.sidebar.header('Parameters')

# Initialize session state
# This is not necessary in this app, because all widgets are created
# and theirs keys stored in session_state when on_change
# callback is triggered for timeframe. Kept for future reference.

# if ('interval' not in st.session_state):
#     st.session_state.interval='1wk'
# if ('data_window' not in st.session_state):
#     st.session_state.data_window='2y'

# Set the default data_window based on interval value
# by using session state, this is a callback function
def set_data_window():
    if(st.session_state.interval == '1d'):
        st.session_state.data_window = '3mo'
    elif (st.session_state.interval == '1wk'):
        st.session_state.data_window = '2y'
    elif (st.session_state.interval == '1mo'):
        st.session_state.data_window = '10y'
    elif (st.session_state.interval == '3mo'):
        st.session_state.data_window = 'max'

# create three tabs for parameters
tab_data, tab_smoothing, tab_format = \
st.sidebar.tabs(["Data", "Smoothing", "Format"])

with tab_data: # first tab
    # ticker symbol: find from Yahoo Finance
    ticker_symbol = st.text_input(label='Ticker',value='BTC-USD',
        key='ticker_symbol',
        help = '''
            Use [Yahoo Finance](https://finance.yahoo.com) to find the ticker.
            Examples include:
            - **Crypto**: BTC-USD, ETH-USD, PAXG-USD
            - **ETF**: SPY, QQQ, GLD
            - **Futures**: GC=F, CL=F, DX=F
            - **Forex**: EURUSD=X, USDCAD=X
            - **Index**:  ^DJI, ^IXIC, ^GSPC
            - **Stock**: GOOG, MSFT, NVDA, AAPL, AMZN, META, TSLA
            '''
        )
    # Timeframe
    tf_options = ['1d','1wk','1mo','3mo']
    interval = st.pills(label="Timeframe", options=tf_options, 
        default='1d', selection_mode="single", key='interval',
        help = 'Duration of time that each data point on the chart \
            represents.', on_change = set_data_window)
    
    # Data range: can be either a data window or from and to date
    # if from date is empty, first data point is used
    # if end date is empty, last data point is used
    use_data_window = st.checkbox(label='Use data window',value=True,
        help='Select data range method: data window or dates')
    
    # use a data window
    if(use_data_window):
        dw_options = ['3mo','6mo','1y','2y','5y'
            ,'10y','ytd','max'] # '1d','5d', '1mo' are also available in YF
        data_window = st.pills(label="Data Window", options=dw_options, 
            default='3mo', selection_mode="single", key='data_window',
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

with tab_format: # third tab
    # Format parameters
    # Chart height
    height = st.number_input('Plot heigth',value=700, min_value=0, step=25)
    
    # y-axis ticks distance for top plot
    # 0: program selects the y_grid
    y_grid = st.number_input('Y-axis grid',value=0.0,
        min_value=0.0, step=0.1, help = 'Enter `0` for automatic selection') 
    
    # Decimal places, override streamlit four decimals
    decimal_places = st.number_input('Decimal places',value=2, 
            min_value=0, max_value = 4, step=1)

# Back Transformation
st.sidebar.divider()
st.sidebar.header('Back Transformation',help='Convert the figures from the chart back \
    to their original prices.')
# get input from user
log_price = st.sidebar.number_input('Log Price', value=1.0)
# write the back transformation
if (log_price):
    st.sidebar.write(f'Price: `{np.exp(log_price):.4f}`')

# About the App
st.sidebar.divider()
st.sidebar.header('About')
with st.sidebar.expander(label='About this application...'):
    st.markdown('''
                - Retrieves data for the selected ticker from Yahoo Finance.
                - Applies LOWESS smoothing to the time series to identify the trend.
                - Calculates and standardize residuals, which can be used as an oscillator.
                - Plots the smoothed series and the oscillator
                ''')
# ------------------------------------------------------------------------------ 
# Get data from Yahoo Finance
# ------------------------------------------------------------------------------
# Ignore all warnings 
warnings.filterwarnings('ignore')

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

# Number of data points
n_rows = len(df)

# Calculate badwidth from smoothing periods whe use_period is true
if (use_period == True):
    enough_data = (lt_period < n_rows ) and (st_period < n_rows )
    # enough data exists
    if(enough_data):
        lt_bw = lt_period/n_rows
        st_bw = st_period/n_rows
    # lt_period > n_rows or  st_period > n_rows
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

#Upper and lower bounds
df['upper_bound'] = df['lt_lowess'] + 1 * res_std
df['lower_bound'] = df['lt_lowess'] - 1 * res_std

# ------------------------------------------------------------------------------ 
# Create Main Figure
# ------------------------------------------------------------------------------
# Set the default template
pio.templates.default = "seaborn"

# Create a figure with 2 rows and 1 column, sharing the x-axis
main_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    row_heights=[0.75, 0.25], vertical_spacing=0.01)

# Create chart title
if(use_period == True):
    title = f'{ticker_symbol} Trends ({interval}, {lt_period}, {st_period})'
else:
    title = f'{ticker_symbol} Trends ({interval}, {lt_bw:.2f}, {st_bw:.2f})'
    
# Update the figure title
main_fig.update_layout(
    title={
        'text': title,
        'x': 0.5,
        'xanchor': 'center'
    }
)

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

# Update the y-axis tick distance for top figure
main_fig.update_yaxes(dtick=y_grid)

# Update y-axis tick distance for bottom figure
main_fig.update_yaxes(dtick=1, row=2, col=1)

# Update y-axis label
main_fig.update_yaxes(title_text='Log Price', row=1, col=1)
main_fig.update_yaxes(title_text='Std. Residual', row=2, col=1)

# last close and its log
last_close = df.iloc[-1]['Close']
log_last_close = np.log(last_close)

# Add a horizontal line at Latest Close
main_fig.add_shape(
    type="line",
    x0=df.index.min(),
    x1=df.index.max(),
    y0=log_last_close,
    y1=log_last_close,
    line=dict(
        color="red",
        width=1,
        dash="dot"
    )
)

# Add annotation for the horizontal line
main_fig.add_annotation(
    x=df.index.max(),
    y=log_last_close,
    text=f"{log_last_close:.{decimal_places}f} ({last_close:.{decimal_places}f})",
    showarrow=False,
    yshift=10,
    xshift=50,
    bgcolor='red',
    font=dict(color="white")           
)

# ------------------------------------------------------------------------------ 
# Create Top Figure
# ------------------------------------------------------------------------------
# colors of lines
colors = ['#00CC96','#EF553B','#636EFA','#AB63FA','#AB63FA']

# Plot time series, short-term and long-term smoothed series
top_fig = px.line(df, x=df.index, y=['l_avg','st_lowess','lt_lowess','upper_bound','lower_bound'], 
              title=title, labels={'variable': 'Time Series'},
              color_discrete_sequence=colors)

# Update the visibility of one of the traces to be 'legendonly'
for trace in top_fig.data:
    if trace.name == 'st_lowess':
        trace.visible = 'legendonly'  # Hide l_avg by default
        
# Update the legend labels for each line
top_fig.for_each_trace(lambda trace: trace.update( 
    name=trace.name.replace('l_avg', 'Log Price').
    replace('lt_lowess', 'Long-term Trend').
    replace('st_lowess', 'Short-term Trend').
    replace('upper_bound', 'Upper Bound').
    replace('lower_bound', 'Lower Bound'))
)

# ------------------------------------------------------------------------------ 
# Create Bottom Figure
# ------------------------------------------------------------------------------

# Standard residual figure
bot_fig = px.line(df, x=df.index, y=['std_residual'], 
              labels={'variable': 'Time Series'},
              title = 'Residual Oscillator')

# Update the y-axis to have grid lines at each integer
bot_fig.update_yaxes(dtick=1)

# Update the legend labels for each line
bot_fig.for_each_trace(lambda trace: trace.update( 
    name=trace.name.replace('std_residual', 'Std. Residual')
    )
)

# ------------------------------------------------------------------------------ 
# Show Results  
# ------------------------------------------------------------------------------

# Top section - Chart

# Display section header
# st.subheader('Long-Term Trend')
st.subheader('Explorer')

# Add the trace from the top_fig the subplot (main_fig)
for trace in top_fig.data:
    main_fig.add_trace(trace, row=1, col=1)
# Add the trace from the bot_fig to the subplot (main_fig)
for trace in bot_fig.data:
    main_fig.add_trace(trace, row=2, col=1)
    
# Hide the legend for the bottom subplot
main_fig.update_traces(showlegend=False, row=2, col=1)

# show the plot
st.plotly_chart(main_fig)

# Display a divider
# st.divider()

# ------------------------------------------------------------------------------
# Bottom section - Data and other extra information

# New section
st.subheader('Data')

# Create Dataframe to display latest data
df2 = df[['Open','High','Low','Close','avg','l_avg']]
# Calculate change
df2['Change'] = df2['Close'].diff()
# Calculate percentage change
df2['Change (%)'] = df2['Close'].pct_change() * 100
# Change index display format
df2.index = df2.index.strftime('%Y-%m-%d')
# Rename columns
df2 = df2.rename(columns={'avg': 'Avg (HLC3)', 'l_avg': 'Log Avg'})
# Keep the tail
# df2 = df2.tail(10)
#Display format
column_config={
'Open': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'High': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Low': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Close': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Avg (HLC3)': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Log Avg': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Change': st.column_config.NumberColumn(format=f'%.{decimal_places}f'),
'Change (%)': st.column_config.NumberColumn(format='%.1f %%')
}

# Display latest data
with st.expander('Latest Data'):
    # Display number of data points and latest data
    st.write('Number of data points: ', df.shape[0])
    # Display latest data
    st.dataframe(df2,column_config = column_config)
    
# Display Statistics
with st.expander('Statistics'):
    # Residuals
    st.write('Residuals Mean:',round(res_mean,2))
    st.write('Residuals Standard Deviation:',round(res_std,2))

# ------------------------------------------------------------------------------
# Debug
# ------------------------------------------------------------------------------
# st.session_state
# st.write(df.tail())

# ------------------------------------------------------------------------------