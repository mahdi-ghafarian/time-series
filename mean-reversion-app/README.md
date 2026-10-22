# Mean Reversion Profile

This Streamlit app adapts [`rtm_boxplot_lr_yf.py`](./rtm_boxplot_lr_yf.py) into
an interactive analysis of historical prices and forward returns by past-return
bin. Data is fetched from Yahoo Finance.

From this directory, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

In the sidebar, search Yahoo Finance by company name or ticker in **Search for
ticker**, then choose a `TICKER | NAME | EXCHANGE` result to fill the separate
ticker field with its symbol; the search field clears after selection. You can
also edit the ticker directly. Unselected search text is retained when
switching between apps in the Market Analysis Suite. Set the interval, history
range, return windows, bin settings, and line-chart settings in the sidebar.
The price chart includes a raw log-return range control with configurable
decimal precision (default one place) that highlights matching dates without
changing the future-return boxplot or bin statistics. Select a row in the bin
statistics table to set the chart's highlight range to that bin. The app
also displays those statistics, including quartiles, whiskers, and outlier
counts, and provides a CSV download of the calculated observations.
