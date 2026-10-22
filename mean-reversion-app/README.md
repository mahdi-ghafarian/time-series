# Mean Reversion Profile

This Streamlit app adapts [`rtm_boxplot_lr_yf.py`](./rtm_boxplot_lr_yf.py) into
an interactive analysis of historical prices and forward returns by past-return
bin. Data is fetched from Yahoo Finance.

From this directory, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a ticker, interval, history range, return windows, and bin settings in
the sidebar. The app displays a log average-price chart, future-return boxplot,
bin statistics including quartiles, whiskers, and outlier counts, and a CSV
download of the calculated observations.
