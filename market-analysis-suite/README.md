# Market Analysis Suite

This Streamlit app brings the Swing Statistics, Mean Reversion Profile, and
Financial Markets Explorer apps together behind a sidebar selector. It reuses
their existing app files, so their analyses and controls remain consistent
with the standalone apps.

Each app provides a **Search for ticker** field that searches Yahoo Finance by
ticker or company name. Select a `TICKER | NAME | EXCHANGE` result to fill the
separate ticker field with its symbol. Search text is retained when switching
between apps.

From the repository root, run:

```powershell
streamlit run market-analysis-suite/app.py
```

Install dependencies from this folder's `requirements.txt` if they are not
already available in your Python environment.
