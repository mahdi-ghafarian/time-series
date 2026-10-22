from datetime import date, timedelta

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter


st.set_page_config(
    page_title="Mean Reversion Profile",
    page_icon="📈",
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


def analyze_returns(
    history: pd.DataFrame,
    backward_window: int,
    forward_window: int,
    auto_bin: bool,
    number_of_bins: int,
    requested_bin_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    required_columns = {"High", "Low", "Close"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        raise ValueError(
            "Yahoo Finance data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prices = history.copy()
    prices["AVG"] = (prices["High"] + prices["Low"] + prices["Close"]) / 3
    prices["LOG AVG"] = np.log(prices["AVG"].where(prices["AVG"] > 0))
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(subset=["LOG AVG"])
    if len(prices) <= backward_window + forward_window:
        raise ValueError(
            "Not enough valid price observations for the selected backward and "
            "forward windows. Choose a longer data range or smaller windows."
        )

    price_history = prices[["LOG AVG"]].copy()
    prices["past_return"] = np.log(
        prices["Close"] / prices["Close"].shift(backward_window)
    )
    prices["future_return"] = np.log(
        prices["Close"].shift(-forward_window) / prices["Close"]
    )
    returns = prices.dropna(subset=["past_return", "future_return"]).copy()
    if returns.empty:
        raise ValueError("No valid past/future return pairs are available.")

    return_range = float(returns["past_return"].max() - returns["past_return"].min())
    if auto_bin:
        bin_size = round(return_range / number_of_bins, 2)
        if bin_size <= 0:
            bin_size = return_range / number_of_bins
    else:
        bin_size = requested_bin_size

    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError(
            "Past returns do not vary enough to create bins. Try a different "
            "ticker or data range."
        )

    lower = np.floor(returns["past_return"].min() / bin_size) * bin_size
    upper = np.ceil(returns["past_return"].max() / bin_size) * bin_size
    if upper <= lower:
        lower -= bin_size / 2
        upper += bin_size / 2
    bin_span = (upper - lower) / bin_size
    nearest_bin_count = round(bin_span)
    bin_count = max(
        1,
        int(
            nearest_bin_count
            if np.isclose(bin_span, nearest_bin_count)
            else np.ceil(bin_span)
        ),
    )
    edges = lower + np.arange(bin_count + 1) * bin_size
    if edges[-1] < returns["past_return"].max():
        edges = np.append(edges, edges[-1] + bin_size)

    returns["past_return_bin"] = pd.cut(
        returns["past_return"], bins=edges, include_lowest=True
    )
    categories = returns["past_return_bin"].cat.categories
    labels = [f"{interval.left:.2f} to {interval.right:.2f}" for interval in categories]
    returns["bin_label"] = returns["past_return_bin"].map(
        dict(zip(categories, labels))
    )

    summary = (
        returns.groupby("past_return_bin", observed=False)["future_return"]
        .agg(observations="count", mean="mean", median="median")
        .reindex(categories)
    )
    summary.insert(0, "bin", labels)
    summary["observations"] = summary["observations"].fillna(0).astype(int)
    return price_history, returns, bin_size, summary


st.title("Mean Reversion Profile")
st.write(
    "Explore how returns over a past window relate to returns over a future "
    "window. Prices and return data are retrieved from Yahoo Finance."
)

with st.sidebar:
    st.header("Parameters")
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
            ["3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            index=7,
        )
        start_date = date.today() - timedelta(days=365 * 10)
        end_date = date.today()
    else:
        data_window = "max"
        start_date = st.date_input(
            "Start date", value=date.today() - timedelta(days=365 * 10)
        )
        end_date = st.date_input("End date", value=date.today())

    st.divider()
    st.subheader("Return windows")
    backward_window = st.number_input(
        "Backward window (observations)", min_value=1, value=12, step=1
    )
    forward_window = st.number_input(
        "Forward window (observations)", min_value=1, value=12, step=1
    )

    st.divider()
    st.subheader("Past-return bins")
    auto_bin = st.checkbox("Calculate bin size automatically", value=True)
    if auto_bin:
        number_of_bins = st.number_input(
            "Target number of bins", min_value=2, max_value=30, value=5, step=1
        )
        requested_bin_size = 0.1
    else:
        number_of_bins = 5
        requested_bin_size = st.number_input(
            "Bin size (log return)",
            min_value=0.001,
            value=0.1,
            step=0.01,
            format="%.3f",
        )

st.sidebar.caption("Past and future windows are measured in data observations.")

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
    st.warning(
        "No data was returned. Check the ticker symbol, date range, and interval."
    )
    st.stop()

try:
    price_history, prices, bin_size, summary = analyze_returns(
        history,
        int(backward_window),
        int(forward_window),
        auto_bin,
        int(number_of_bins),
        float(requested_bin_size),
    )
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

mean_future_return = float(prices["future_return"].mean())
metric_columns = st.columns(3)
metric_columns[0].metric("Valid observations", f"{len(prices):,}")
metric_columns[1].metric("Mean future log return", f"{mean_future_return:.2%}")
metric_columns[2].metric("Past-return bin size", f"{bin_size:.3f}")

st.subheader(f"{ticker_symbol} log average price ({interval})")
price_figure = go.Figure(
    go.Scatter(
        x=price_history.index,
        y=price_history["LOG AVG"],
        mode="lines",
    )
)
log_tick_min = float(np.floor(price_history["LOG AVG"].min()))
log_tick_max = float(np.ceil(price_history["LOG AVG"].max()))
if log_tick_max <= log_tick_min:
    log_tick_max = log_tick_min + 1
log_price_ticks = np.arange(log_tick_min, log_tick_max + 1, 1)
natural_price_ticks = np.exp(log_price_ticks)
log_price_range = [log_tick_min, log_tick_max]
price_figure.add_trace(
    go.Scatter(
        x=price_history.index,
        y=price_history["LOG AVG"],
        mode="lines",
        yaxis="y2",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
)
price_figure.update_layout(
    xaxis_title="Date",
    showlegend=False,
    yaxis={
        "title": "Log average price (HLC3)",
        "range": log_price_range,
        "tickmode": "array",
        "tickvals": log_price_ticks,
        "ticktext": [f"{value:.0f}" for value in log_price_ticks],
        "showgrid": True,
    },
    yaxis2={
        "overlaying": "y",
        "side": "right",
        "title": "Average price (HLC3)",
        "range": log_price_range,
        "tickmode": "array",
        "tickvals": log_price_ticks,
        "ticktext": [f"{price:,.0f}" for price in natural_price_ticks],
        "showgrid": False,
    },
)
st.plotly_chart(price_figure, width="stretch", config={"scrollZoom": False})

st.subheader("Future returns by past-return bin")
st.caption(
    f"Past window: {int(backward_window)} observations · "
    f"Future window: {int(forward_window)} observations · "
    "Boxplot values are log returns."
)

category_intervals = prices["past_return_bin"].cat.categories
category_labels = [
    f"{interval.left:.2f} to {interval.right:.2f}"
    for interval in category_intervals
]
bin_means = prices.groupby("past_return_bin", observed=False)["future_return"].mean()
valid_means = bin_means.dropna()
color_min = float(valid_means.min())
color_max = float(valid_means.max())
if color_min == color_max:
    color_min -= 0.01
    color_max += 0.01
color_norm = mcolors.Normalize(vmin=color_min, vmax=color_max)
color_map = plt.colormaps["RdYlGn"]
palette = {
    label: (
        color_map(color_norm(bin_means.iloc[index]))
        if pd.notna(bin_means.iloc[index])
        else "#d3d3d3"
    )
    for index, label in enumerate(category_labels)
}

figure, axis = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=prices,
    x="bin_label",
    y="future_return",
    hue="bin_label",
    order=category_labels,
    hue_order=category_labels,
    palette=palette,
    dodge=False,
    legend=False,
    ax=axis,
    showfliers=True,
    flierprops={
        "marker": "o",
        "markerfacecolor": "black",
        "markersize": 6,
        "linestyle": "none",
        "alpha": 0.25,
    },
)
axis.set_title(
    f"Mean Reversion Profile ({int(backward_window)}, {int(forward_window)}, "
    f"{bin_size:.3f}) — {ticker_symbol} {interval}"
)
axis.set_xlabel(f"Past log return (t-{int(backward_window)} to t)")
axis.set_ylabel(f"Future log return (t to t+{int(forward_window)})")
axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
axis.tick_params(axis="x", labelrotation=45)
negative_bin_index = max(
    (
        idx
        for idx, interval in enumerate(category_intervals)
        if interval.right <= 0
    ),
    default=-1,
)
positive_bin_index = min(
    (
        idx
        for idx, interval in enumerate(category_intervals)
        if interval.left >= 0
    ),
    default=len(category_intervals),
)
if 0 <= negative_bin_index < len(category_intervals) - 1:
    boundary_x = negative_bin_index + 0.5
elif 0 <= negative_bin_index < len(category_intervals) and positive_bin_index < len(category_intervals):
    boundary_x = (negative_bin_index + positive_bin_index) / 2
else:
    boundary_x = (len(category_intervals) - 1) / 2
axis.axvline(
    x=boundary_x,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label="Zero return boundary",
)
axis.set_xlim(-0.5, len(category_labels) - 0.5)
axis.axhline(
    mean_future_return,
    color="red",
    linestyle="dotted",
    linewidth=2,
    label=f"Mean future log return ({mean_future_return:.2f})",
)
axis.axhline(0, color="black", linestyle="dotted", linewidth=1)
axis.legend()
axis.grid(True, axis="y", alpha=0.3)

scalar_mappable = plt.cm.ScalarMappable(cmap=color_map, norm=color_norm)
scalar_mappable.set_array([])
figure.colorbar(
    scalar_mappable, ax=axis, label="Mean future log return by past-return bin"
)
figure.tight_layout()
st.pyplot(figure, width="stretch")
plt.close(figure)

st.subheader("Bin statistics")
st.dataframe(
    summary.rename(
        columns={
            "bin": "Past log-return bin",
            "observations": "Observations",
            "mean": "Mean future log return",
            "median": "Median future log return",
        }
    ),
    hide_index=True,
    width="stretch",
    column_config={
        "Mean future log return": st.column_config.NumberColumn(format="%.4f"),
        "Median future log return": st.column_config.NumberColumn(format="%.4f"),
    },
)

csv_data = prices[
    ["Close", "past_return", "future_return", "bin_label"]
].to_csv(index_label="Date").encode("utf-8")
st.download_button(
    "Download return observations (CSV)",
    data=csv_data,
    file_name=f"{ticker_symbol.lower()}_mean_reversion_returns.csv",
    mime="text/csv",
)
