from datetime import date, timedelta

import matplotlib.colors as mcolors
from matplotlib.cbook import boxplot_stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter


if not st.session_state.get("_market_analysis_suite"):
    st.set_page_config(
        page_title="Mean Reversion Profile",
        page_icon="📈",
        layout="wide",
    )


def persist_suite_preferences() -> None:
    if not st.session_state.get("_market_analysis_suite"):
        return

    preferences = dict(
        st.session_state.get("_suite_saved_preferences", {})
    )
    ticker_key = "mean_reversion_ticker"
    if ticker_key in st.session_state:
        shared_ticker = st.session_state[ticker_key].strip().upper()
        st.session_state[ticker_key] = shared_ticker
        st.session_state["_suite_shared_ticker"] = shared_ticker
    search_key = "mean_reversion_search_ticker"
    search_label_key = "mean_reversion_search_ticker_label"
    if search_key in st.session_state:
        search_value = st.session_state[search_key]
        selected_label = st.session_state.get(search_label_key, "")
        if selected_label and search_value == selected_label:
            search_value = ""
            st.session_state[search_key] = search_value
            st.session_state[search_label_key] = ""
        st.session_state["_suite_ticker_search"] = search_value
    if search_label_key in st.session_state:
        st.session_state["_suite_ticker_search_label"] = st.session_state[
            search_label_key
        ]
    for key in st.session_state:
        if key.startswith("mean_reversion_"):
            preferences[key] = st.session_state[key]
    st.session_state["_suite_saved_preferences"] = preferences


@st.cache_data(ttl=3600, show_spinner=False)
def search_tickers(query: str) -> list[dict[str, str]]:
    quotes = yf.Search(
        query,
        max_results=8,
        news_count=0,
        lists_count=0,
        include_cb=False,
    ).quotes
    matches = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol:
            continue
        matches.append(
            {
                "symbol": str(symbol),
                "name": str(
                    quote.get("longname") or quote.get("shortname") or symbol
                ),
                "exchange": str(
                    quote.get("exchDisp") or quote.get("exchange") or ""
                ),
            }
        )
    return matches


def select_ticker_result(symbol: str) -> None:
    st.session_state["mean_reversion_ticker"] = symbol.upper()
    st.session_state["mean_reversion_search_ticker"] = ""
    st.session_state["mean_reversion_search_ticker_label"] = ""
    st.session_state["_suite_ticker_search"] = ""
    st.session_state["_suite_ticker_search_label"] = ""
    persist_suite_preferences()


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
        .agg(
            observations="count",
            mean="mean",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reindex(categories)
    )
    summary.insert(0, "bin", labels)
    summary["observations"] = summary["observations"].fillna(0).astype(int)
    boxplot_summaries = []
    for category in categories:
        values = returns.loc[
            returns["past_return_bin"] == category, "future_return"
        ].to_numpy()
        stats = boxplot_stats(values, whis=1.5)[0] if len(values) else {}
        boxplot_summaries.append(
            {
                "lower_whisker": stats.get("whislo", np.nan),
                "q1": stats.get("q1", np.nan),
                "q3": stats.get("q3", np.nan),
                "upper_whisker": stats.get("whishi", np.nan),
                "outliers": len(stats.get("fliers", [])),
            }
        )
    summary = summary.join(pd.DataFrame(boxplot_summaries, index=categories))
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["range"] = summary["maximum"] - summary["minimum"]
    summary = summary[
        [
            "bin",
            "observations",
            "mean",
            "range",
            "iqr",
            "lower_whisker",
            "q1",
            "median",
            "q3",
            "upper_whisker",
            "outliers",
        ]
    ]
    return price_history, returns, bin_size, summary


st.title("Mean Reversion Profile")
st.write(
    "Explore how returns over a past window relate to returns over a future "
    "window."
)

with st.sidebar:
    st.header("Data")
    ticker_query = st.text_input(
        "Search for ticker",
        placeholder="Type a ticker or company name",
        key="mean_reversion_search_ticker",
        live="500ms",
        on_change=persist_suite_preferences,
    ).strip()
    ticker_matches = []
    search_failed = False
    selected_label = st.session_state.get(
        "mean_reversion_search_ticker_label", ""
    )
    if ticker_query != selected_label and len(ticker_query) >= 2:
        try:
            with st.spinner("Searching Yahoo Finance..."):
                ticker_matches = search_tickers(ticker_query)
        except Exception as exc:
            search_failed = True
            st.warning(
                f"Yahoo Finance ticker search failed: {exc} "
                "You can still enter a ticker manually."
            )
    if ticker_matches:
        for match in ticker_matches:
            symbol = match["symbol"]
            label = (
                f'{symbol} | {match["name"]} | '
                f'{match["exchange"] or "Unknown exchange"}'
            )
            st.button(
                label,
                key=f"_mean_reversion_ticker_result_{symbol}",
                on_click=select_ticker_result,
                args=(symbol,),
                width="stretch",
            )
    else:
        if (
            ticker_query != selected_label
            and ticker_query
            and len(ticker_query) >= 2
            and not search_failed
        ):
            st.caption("No Yahoo Finance matches found.")
        elif ticker_query != selected_label and ticker_query:
            st.caption("Enter at least two characters to search.")

    ticker_symbol = st.text_input(
        "Ticker",
        value=st.session_state.get("mean_reversion_ticker", "BTC-USD"),
        help="Examples: AAPL, MSFT, SPY, BTC-USD, or ^GSPC.",
        key="mean_reversion_ticker",
        on_change=persist_suite_preferences,
    ).strip().upper()

    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo", "3mo"],
        index=2,
        key="mean_reversion_interval",
        on_change=persist_suite_preferences,
    )
    use_data_window = st.checkbox(
        "Use data window",
        value=True,
        key="mean_reversion_use_data_window",
        on_change=persist_suite_preferences,
    )
    if use_data_window:
        data_window = st.selectbox(
            "Data window",
            ["3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            index=7,
            key="mean_reversion_data_window",
            on_change=persist_suite_preferences,
        )
        start_date = date.today() - timedelta(days=365 * 10)
        end_date = date.today()
    else:
        data_window = "max"
        start_date = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=365 * 10),
            key="mean_reversion_start_date",
            on_change=persist_suite_preferences,
        )
        end_date = st.date_input(
            "End date",
            value=date.today(),
            key="mean_reversion_end_date",
            on_change=persist_suite_preferences,
        )

    st.divider()
    st.header("Box plot")
    backward_window = st.number_input(
        "Backward window (observations)",
        min_value=1,
        value=12,
        step=1,
        key="mean_reversion_backward_window",
        on_change=persist_suite_preferences,
    )
    forward_window = st.number_input(
        "Forward window (observations)",
        min_value=1,
        value=12,
        step=1,
        key="mean_reversion_forward_window",
        on_change=persist_suite_preferences,
    )

    auto_bin = st.checkbox(
        "Calculate bin size automatically",
        value=True,
        key="mean_reversion_auto_bin",
        on_change=persist_suite_preferences,
    )
    if auto_bin:
        number_of_bins = st.number_input(
            "Target number of bins",
            min_value=2,
            max_value=30,
            value=5,
            step=1,
            key="mean_reversion_number_of_bins",
            on_change=persist_suite_preferences,
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
            key="mean_reversion_requested_bin_size",
            on_change=persist_suite_preferences,
        )
    st.divider()
    st.header("Line chart")
    log_grid_interval = st.number_input(
        "Y-axis grid interval",
        min_value=0.1,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Spacing between log-price grid lines.",
        key="mean_reversion_log_grid_interval",
        on_change=persist_suite_preferences,
    )
    slider_decimal_places = st.number_input(
        "Slider precision",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
        key="mean_reversion_slider_decimal_places",
        on_change=persist_suite_preferences,
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
metric_columns[1].metric("Mean future log return", f"{mean_future_return:.2f}")
metric_columns[2].metric("Past-return bin size", f"{bin_size:.3f}")

# Calculate chart highlights independently from the future-return pairs used
# by the box plot and bin statistics.
chart_close = history["Close"].reindex(price_history.index)
chart_past_returns = np.log(
    chart_close / chart_close.shift(int(backward_window))
).replace([np.inf, -np.inf], np.nan).dropna()
return_min = float(chart_past_returns.min())
return_max = float(chart_past_returns.max())
if return_min == return_max:
    return_min -= 0.0001
    return_max += 0.0001
range_key = "mean_reversion_past_log_return_range"

price_figure = go.Figure(
    go.Scatter(
        x=price_history.index,
        y=price_history["LOG AVG"],
        mode="lines",
    )
)
log_tick_min = float(
    np.floor(price_history["LOG AVG"].min() / log_grid_interval)
    * log_grid_interval
)
log_tick_max = float(
    np.ceil(price_history["LOG AVG"].max() / log_grid_interval)
    * log_grid_interval
)
if log_tick_max <= log_tick_min:
    log_tick_max = log_tick_min + log_grid_interval
tick_count = int(round((log_tick_max - log_tick_min) / log_grid_interval))
log_price_ticks = (
    log_tick_min + np.arange(tick_count + 1) * log_grid_interval
)
log_tick_decimals = max(0, int(np.ceil(-np.log10(log_grid_interval))))
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
        "ticktext": [
            f"{value:.{log_tick_decimals}f}" for value in log_price_ticks
        ],
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
st.subheader("How Past Returns Relate to Future Returns")
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
    whis=1.5,
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
    # label="Zero return boundary",
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
st.caption("Return statistics are based on future log return.")
st.caption(
    "Select a bin row to set the price-chart highlight range to that bin."
)
bin_ranges = [
    (
        max(float(interval.left), return_min),
        min(float(interval.right), return_max),
    )
    for interval in category_intervals
]
st.session_state["_mean_reversion_bin_ranges"] = bin_ranges
bin_statistics = summary.rename(
    columns={
        "bin": "Past log-return bin",
        "observations": "Observations",
        "mean": "Mean",
        "range": "Range",
        "iqr": "IQR",
        "lower_whisker": "Lower whisker",
        "q1": "Q1",
        "median": "Median",
        "q3": "Q3",
        "upper_whisker": "Upper whisker",
        "outliers": "Outliers",
    }
)
table_state = st.dataframe(
    bin_statistics,
    hide_index=True,
    width="stretch",
    key="_mean_reversion_bin_table",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Mean": st.column_config.NumberColumn(format="%.4f"),
        "Range": st.column_config.NumberColumn(format="%.4f"),
        "IQR": st.column_config.NumberColumn(format="%.4f"),
        "Lower whisker": st.column_config.NumberColumn(format="%.4f"),
        "Q1": st.column_config.NumberColumn(format="%.4f"),
        "Median": st.column_config.NumberColumn(format="%.4f"),
        "Q3": st.column_config.NumberColumn(format="%.4f"),
        "Upper whisker": st.column_config.NumberColumn(format="%.4f"),
    },
)

bin_selection = table_state.selection.rows
previous_bin_selection = st.session_state.get(
    "_mean_reversion_last_bin_selection", []
)
if bin_selection != previous_bin_selection:
    st.session_state["_mean_reversion_last_bin_selection"] = bin_selection
    if bin_selection and bin_selection[0] < len(bin_ranges):
        st.session_state[range_key] = bin_ranges[bin_selection[0]]

if range_key in st.session_state:
    saved_range = tuple(st.session_state[range_key])
    clamped_range = (
        min(max(float(saved_range[0]), return_min), return_max),
        min(max(float(saved_range[1]), return_min), return_max),
    )
    if saved_range != clamped_range:
        st.session_state[range_key] = clamped_range
selected_return_range = st.slider(
    f"Highlight points with past {int(backward_window)}-period log returns "
    "in this range",
    min_value=return_min,
    max_value=return_max,
    value=(return_min, return_max),
    step=10 ** -int(slider_decimal_places),
    format=f"%.{int(slider_decimal_places)}f",
    key=range_key,
)
selected_dates = chart_past_returns.index[
    chart_past_returns.between(*selected_return_range)
]
selected_prices = price_history.reindex(selected_dates)["LOG AVG"]
st.caption(
    f"Highlighted points: {len(selected_dates):,} of "
    f"{len(chart_past_returns):,}."
)
if not selected_dates.empty:
    price_figure.add_trace(
        go.Scatter(
            x=selected_dates,
            y=selected_prices,
            mode="markers",
            name="Matching past returns",
            marker={
                "color": "#e63946",
                "size": 8,
                "line": {"color": "white", "width": 1},
            },
            customdata=chart_past_returns.reindex(selected_dates),
            hovertemplate=(
                "Date: %{x}<br>Past log return: %{customdata:.2f}"
                "<extra></extra>"
            ),
        )
    )

st.subheader(f"{ticker_symbol} log average price ({interval})")
st.plotly_chart(price_figure, width="stretch", config={"scrollZoom": False})

csv_data = prices[
    ["Close", "past_return", "future_return", "bin_label"]
].to_csv(index_label="Date").encode("utf-8")
st.download_button(
    "Download return observations (CSV)",
    data=csv_data,
    file_name=f"{ticker_symbol.lower()}_mean_reversion_returns.csv",
    mime="text/csv",
)
