# Swing Statistics

This Streamlit app turns [`swing-study.py`](./swing-study.py) into an
interactive analysis of historical price swings, true range, and ATR. Market
data is fetched from Yahoo Finance.

From this directory, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a ticker, interval, history range, and ATR window in the sidebar. The app
shows log average price, swing distribution boxplots, descriptive statistics,
swing observations, and CSV downloads.
