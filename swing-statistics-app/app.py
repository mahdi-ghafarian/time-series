from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="Swing Statistics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(
    ticker_symbol: str,
    interval: str,
    use_data_window: bool,
    data_window: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_symbol)
    if use_data_window:
        history = ticker.history(period=data_window, interval=interval)
    else:
        history = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
        )
    return pd.DataFrame(history)


def analyze_swings(
    history: pd.DataFrame, atr_window: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {"High", "Low", "Close"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(
            "Yahoo Finance data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prices = history.copy().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.index = pd.to_datetime(prices.index)
    for column in required_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["High", "Low", "Close"]
    )
    prices = prices.loc[prices["Close"] > 0].copy()
    if len(prices) < 2:
        raise ValueError("Not enough valid price observations to analyze swings.")

    prices["AVG"] = (prices["High"] + prices["Low"] + prices["Close"]) / 3
    prices = prices.loc[prices["AVG"] > 0].copy()
    prices["LOG AVG"] = np.log(prices["AVG"])

    high_low = prices["High"] - prices["Low"]
    high_previous_close = (prices["High"] - prices["Close"].shift(1)).abs()
    low_previous_close = (prices["Low"] - prices["Close"].shift(1)).abs()
    prices["TR"] = pd.concat(
        [high_low, high_previous_close, low_previous_close], axis=1
    ).max(axis=1)
    prices["ATR"] = prices["TR"].rolling(window=atr_window).mean()
    prices["TR%"] = prices["TR"] / prices["Close"] * 100
    prices["ATR%"] = prices["TR%"].rolling(window=atr_window).mean()

    prices["Return_C%"] = prices["Close"].pct_change() * 100
    prices["Abs_Return_C%"] = prices["Return_C%"].abs()
    prices["Return_C_Ratio"] = prices["Return_C%"] / prices["ATR%"]
    prices["Abs_Return_C_Ratio"] = prices["Return_C_Ratio"].abs()
    prices["Return_A%"] = prices["AVG"].pct_change() * 100
    prices["Abs_Return_A%"] = prices["Return_A%"].abs()
    prices["Return_A_Ratio"] = prices["Return_A%"] / prices["ATR%"]
    prices["Abs_Return_A_Ratio"] = prices["Return_A_Ratio"].abs()

    prices["swing_state"] = np.where(
        prices["AVG"] > prices["AVG"].shift(1),
        "UP",
        np.where(prices["AVG"] < prices["AVG"].shift(1), "DOWN", "FLAT"),
    )
    prices["swing_state"] = prices["swing_state"].where(
        prices["swing_state"] != "FLAT", prices["swing_state"].shift(1)
    )

    pivots = prices.loc[
        prices["swing_state"] != prices["swing_state"].shift(-1),
        ["AVG", "ATR%", "swing_state"],
    ]
    swing_df = pd.DataFrame(
        {
            "start_date": pd.Series(pivots.index).shift(1).to_numpy(),
            "end_date": pivots.index.to_numpy(),
            "start_price": pivots["AVG"].shift(1).to_numpy(),
            "end_price": pivots["AVG"].to_numpy(),
            "ATR%": pivots["ATR%"].shift(1).to_numpy(),
            "swing_type": pivots["swing_state"].to_numpy(),
        }
    ).dropna()

    if swing_df.empty:
        raise ValueError(
            "No complete swings were found. Choose a longer date range or a "
            "different interval."
        )

    swing_df["Swing_Size"] = (
        swing_df["end_price"] - swing_df["start_price"]
    ).abs()
    swing_df["Swing_Size%"] = (
        swing_df["Swing_Size"] / swing_df["start_price"] * 100
    )
    swing_df["Swing_Size_Ratio"] = swing_df["Swing_Size%"] / swing_df["ATR%"]

    median_period_days = prices.index.to_series().diff().dt.days.median()
    if not np.isfinite(median_period_days) or median_period_days <= 0:
        raise ValueError(
            "Could not determine the typical observation interval for swing "
            "duration."
        )
    swing_days = (
        pd.to_datetime(swing_df["end_date"]) - pd.to_datetime(swing_df["start_date"])
    ).dt.days
    swing_df["Swing_Duration"] = swing_days / median_period_days

    return prices, swing_df


def descriptive_statistics(
    price_history: pd.DataFrame, swings: pd.DataFrame
) -> pd.DataFrame:
    records = []
    groups = {
        "All swings": swings,
        "Up swings": swings.loc[swings["swing_type"] == "UP"],
        "Down swings": swings.loc[swings["swing_type"] == "DOWN"],
    }
    metrics = {
        "True range (%)": (price_history["TR%"], "Price history"),
        "ATR (%)": (price_history["ATR%"], "Price history"),
    }
    for metric in ("Swing_Size_Ratio", "Swing_Size%", "Swing_Duration"):
        for group_name, group_data in groups.items():
            metrics[f"{metric} · {group_name}"] = (
                group_data[metric],
                group_name,
            )

    for metric_label, (values, group_name) in metrics.items():
        values = values.dropna()
        records.append(
            {
                "Metric": metric_label,
                "Group": group_name,
                "Count": int(values.count()),
                "Mean": values.mean(),
                "Std. dev.": values.std(),
                "Skewness": values.skew(),
                "Kurtosis": values.kurtosis(),
                "Min": values.min(),
                "1st pct.": values.quantile(0.01),
                "5th pct.": values.quantile(0.05),
                "25th pct.": values.quantile(0.25),
                "Median": values.median(),
                "75th pct.": values.quantile(0.75),
                "95th pct.": values.quantile(0.95),
                "99th pct.": values.quantile(0.99),
                "Max": values.max(),
            }
        )
    return pd.DataFrame(records)


st.title("Swing Statistics")
st.write(
    "Explore price swings, average true range, swing size, and swing duration "
    "using historical market data from Yahoo Finance."
)

with st.sidebar:
    st.header("Data")
    ticker_symbol = st.text_input(
        "Ticker",
        value="LULU",
        help="Examples: AAPL, MSFT, SPY, BTC-USD, or ^GSPC.",
    ).strip().upper()
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo", "3mo"], index=2)
    use_data_window = st.checkbox("Use data window", value=True)
    if use_data_window:
        data_window = st.selectbox(
            "Data window",
            ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=6,
        )
        start_date = date.today() - timedelta(days=365 * 10)
        end_date = date.today() + timedelta(days=1)
    else:
        data_window = "max"
        start_date = st.date_input(
            "Start date", value=date.today() - timedelta(days=365 * 10)
        )
        end_date = st.date_input("End date", value=date.today())

    st.divider()
    atr_window = st.number_input(
        "ATR window (observations)", min_value=2, value=30, step=1
    )

if not ticker_symbol:
    st.warning("Enter a ticker symbol to load market data.")
    st.stop()
if not use_data_window and start_date >= end_date:
    st.warning("The start date must be earlier than the end date.")
    st.stop()

try:
    with st.spinner(f"Loading {ticker_symbol} data..."):
        history = fetch_history(
            ticker_symbol,
            interval,
            use_data_window,
            data_window,
            start_date,
            end_date,
        )
except Exception as exc:
    st.error(f"Could not retrieve data for {ticker_symbol}: {exc}")
    st.stop()

if history.empty:
    st.warning("No data was returned. Check the ticker, date range, and interval.")
    st.stop()

try:
    prices, swings = analyze_swings(history, int(atr_window))
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

up_swings = swings.loc[swings["swing_type"] == "UP"]
down_swings = swings.loc[swings["swing_type"] == "DOWN"]
latest_atr = prices["ATR%"].dropna()
metric_columns = st.columns(4)
metric_columns[0].metric("Price observations", f"{len(prices):,}")
metric_columns[1].metric("Total swings", f"{len(swings):,}")
metric_columns[2].metric("Up / down swings", f"{len(up_swings):,} / {len(down_swings):,}")
metric_columns[3].metric(
    f"Latest ATR ({int(atr_window)} observations)",
    f"{latest_atr.iloc[-1]:.2f}%" if not latest_atr.empty else "Not available",
)

st.subheader(f"{ticker_symbol} log average price ({interval})")
price_figure = go.Figure(
    go.Scatter(x=prices.index, y=prices["LOG AVG"], mode="lines")
)
price_figure.update_layout(
    xaxis_title="Date",
    yaxis_title="Log average price (HLC3)",
)
st.plotly_chart(price_figure, width="stretch", config={"scrollZoom": False})

st.subheader("Swing distributions")
plot_metrics = [
    (
        "Swing_Size_Ratio",
        "Swing size / ATR (%)",
        "ATR%",
        np.arange(
            0,
            np.ceil(swings["Swing_Size_Ratio"].max()) + 0.5,
            0.5,
        ),
    ),
    ("Swing_Size%", "Swing size (%)", "Percentage %", None),
    (
        "Swing_Duration",
        "Swing duration (periods)",
        "Periods",
        np.arange(0, np.ceil(swings["Swing_Duration"].max()) + 1, 1),
    ),
]
for metric, title, y_label, y_ticks in plot_metrics:
    figure, axes = plt.subplots(1, 3, figsize=(12, 6))
    swing_groups = [
        (swings, title),
        (up_swings, f"Up {title[0].lower()}{title[1:]}"),
        (down_swings, f"Down {title[0].lower()}{title[1:]}"),
    ]
    y_max = float(swings[metric].max())
    for axis, (group_data, panel_title) in zip(axes, swing_groups):
        plot_data = group_data[metric].dropna()
        if plot_data.empty:
            axis.text(0.5, 0.5, "No swings available", ha="center", va="center")
        else:
            sns.boxplot(y=plot_data, ax=axis)
        axis.set_title(panel_title)
        axis.set_ylabel(y_label)
        axis.set_ylim(0, y_max)
        if y_ticks is not None:
            axis.set_yticks(y_ticks)
        axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

st.subheader("Descriptive statistics")
statistics = descriptive_statistics(prices, swings)
metric_families = statistics["Metric"].str.partition(" · ")[0]
for metric_family in metric_families.drop_duplicates():
    st.subheader(metric_family)
    metric_rows = statistics.loc[metric_families == metric_family]
    st.dataframe(
        metric_rows.drop(columns=["Metric"]),
        hide_index=True,
        width="stretch",
        height="content",
    )
st.download_button(
    "Download descriptive statistics (CSV)",
    data=statistics.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker_symbol.lower()}_{interval}_swing_statistics.csv",
    mime="text/csv",
)

st.subheader("Swing observations")
st.dataframe(
    swings.sort_values("end_date", ascending=False),
    hide_index=True,
    width="stretch",
)
st.download_button(
    "Download swing observations (CSV)",
    data=swings.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker_symbol.lower()}_{interval}_swings.csv",
    mime="text/csv",
)

with st.expander("Price and indicator observations"):
    st.dataframe(prices, width="stretch")
