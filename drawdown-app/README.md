# Drawdown Calculator

This Streamlit app adapts [`drawdown.ipynb`](./drawdown.ipynb) into an
interactive drawdown analysis. It includes bundled S&P 500 and Bitcoin datasets,
and also accepts CSV uploads with selectable date and price columns.

From this directory, install the app dependencies and start Streamlit:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

Drawdown episodes start at the first observation below a prior peak and end when
the price recovers to a previous high. As in the notebook, only recovered
episodes are included in the episode table and summary statistics; any current
unrecovered drawdown appears in the current metric and chart.
