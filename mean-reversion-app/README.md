# Mean Reversion Profile

This Streamlit app adapts [`rtm_boxplot_lr_yf.py`](./rtm_boxplot_lr_yf.py) into
an interactive analysis of historical prices and forward returns by past-return
bin. Data is fetched from Yahoo Finance.

From this directory, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

In the sidebar's ticker field, type a company name or ticker and press Enter to
search Yahoo Finance, then choose a matching result. You can also enter a
ticker directly. Set the interval, history range, return windows, and bin
settings in the sidebar. The app displays a log average-price chart,
future-return boxplot, bin statistics including quartiles, whiskers, and
outlier counts, and a CSV download of the calculated observations.
