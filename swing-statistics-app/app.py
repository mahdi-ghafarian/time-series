from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import yfinance as yf


if not st.session_state.get("_market_analysis_suite"):
    st.set_page_config(
        page_title="Swing Trading",
        page_icon="📊",
        layout="wide",
    )


def persist_suite_preferences() -> None:
    if not st.session_state.get("_market_analysis_suite"):
        return

    preferences = dict(
        st.session_state.get("_suite_saved_preferences", {})
    )
    ticker_key = "swing_ticker"
    if ticker_key in st.session_state:
        shared_ticker = st.session_state[ticker_key].strip().upper()
        st.session_state[ticker_key] = shared_ticker
        st.session_state["_suite_shared_ticker"] = shared_ticker
    search_key = "swing_search_ticker"
    search_label_key = "swing_search_ticker_label"
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
        if key.startswith("swing_"):
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


def select_swing_ticker(symbol: str) -> None:
    st.session_state["swing_ticker"] = symbol.upper()
    st.session_state["swing_search_ticker"] = ""
    st.session_state["swing_search_ticker_label"] = ""
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
        "Indicator · True range": (price_history["TR"], "True range"),
        "Indicator · Average true range": (
            price_history["ATR"],
            "Average true range",
        ),
        "Indicator · True range (%)": (
            price_history["TR%"],
            "True range (%)",
        ),
        "Indicator · Average true range (%)": (
            price_history["ATR%"],
            "Average true range (%)",
        ),
        "Indicator · Close return (%)": (
            price_history["Return_C%"],
            "Close return (%)",
        ),
        "Indicator · Absolute close return (%)": (
            price_history["Abs_Return_C%"],
            "Absolute close return (%)",
        ),
        "Indicator · Close return (ATR multiples)": (
            price_history["Return_C_Ratio"],
            "Close return (ATR multiples)",
        ),
        "Indicator · Absolute close return (ATR multiples)": (
            price_history["Abs_Return_C_Ratio"],
            "Absolute close return (ATR multiples)",
        ),
        "Indicator · HLC3 return (%)": (
            price_history["Return_A%"],
            "HLC3 return (%)",
        ),
        "Indicator · Absolute HLC3 return (%)": (
            price_history["Abs_Return_A%"],
            "Absolute HLC3 return (%)",
        ),
        "Indicator · HLC3 return (ATR multiples)": (
            price_history["Return_A_Ratio"],
            "HLC3 return (ATR multiples)",
        ),
        "Indicator · Absolute HLC3 return (ATR multiples)": (
            price_history["Abs_Return_A_Ratio"],
            "Absolute HLC3 return (ATR multiples)",
        ),
    }
    metric_titles = {
        "Swing_Size_Ratio": "Swing size ratio",
        "Swing_Size%": "Swing size (%)",
        "Swing_Duration": "Swing duration (bars)",
    }
    for metric, title in metric_titles.items():
        for group_name, group_data in groups.items():
            metrics[f"{title} · {group_name}"] = (
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
    "using historical market data."
)

with st.sidebar:
    st.header("Data")
    ticker_query = st.text_input(
        "Search for ticker",
        placeholder="Type a ticker or company name",
        key="swing_search_ticker",
        live="500ms",
        on_change=persist_suite_preferences,
    ).strip()
    selected_label = st.session_state.get("swing_search_ticker_label", "")
    if ticker_query != selected_label and len(ticker_query) >= 2:
        search_failed = False
        try:
            with st.spinner("Searching Yahoo Finance..."):
                ticker_matches = search_tickers(ticker_query)
        except Exception as exc:
            search_failed = True
            st.warning(
                f"Yahoo Finance ticker search failed: {exc} "
                "You can still enter a ticker manually."
            )
            ticker_matches = []
        if ticker_matches:
            for match in ticker_matches:
                symbol = match["symbol"]
                label = (
                    f'{symbol} | {match["name"]} | '
                    f'{match["exchange"] or "Unknown exchange"}'
                )
                st.button(
                    label,
                    key=f"_swing_ticker_result_{symbol}",
                    on_click=select_swing_ticker,
                    args=(symbol,),
                    width="stretch",
                )
        elif not search_failed:
            st.caption("No Yahoo Finance matches found.")
    elif ticker_query != selected_label and ticker_query:
        st.caption("Enter at least two characters to search.")

    ticker_symbol = st.text_input(
        "Ticker",
        value="BTC-USD",
        help="Examples: AAPL, MSFT, SPY, BTC-USD, or ^GSPC.",
        key="swing_ticker",
        on_change=persist_suite_preferences,
    ).strip().upper()
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo", "3mo"],
        index=2,
        key="swing_interval",
        on_change=persist_suite_preferences,
    )
    use_data_window = st.checkbox(
        "Use data window",
        value=True,
        key="swing_use_data_window",
        on_change=persist_suite_preferences,
    )
    if use_data_window:
        data_window = st.selectbox(
            "Data window",
            ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=6,
            key="swing_data_window",
            on_change=persist_suite_preferences,
        )
        start_date = date.today() - timedelta(days=365 * 10)
        end_date = date.today() + timedelta(days=1)
    else:
        data_window = "max"
        start_date = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=365 * 10),
            key="swing_start_date",
            on_change=persist_suite_preferences,
        )
        end_date = st.date_input(
            "End date",
            value=date.today(),
            key="swing_end_date",
            on_change=persist_suite_preferences,
        )

    st.divider()
    st.header("Bar Chart")
    atr_window = st.number_input(
        "ATR window (observations)",
        min_value=2,
        value=30,
        step=1,
        key="swing_atr_window",
        on_change=persist_suite_preferences,
    )
    log_grid_interval = st.number_input(
        "Y-axis grid interval",
        min_value=0.1,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="Spacing between log-price grid lines.",
        key="swing_log_grid_interval",
        on_change=persist_suite_preferences,
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

price_figure = go.Figure()

# Price series
price_figure.add_trace(
    go.Scatter(
        x=prices.index,
        y=prices["LOG AVG"],
        mode="lines",
        name="HLC3",
        # line=dict(color="lightgray", width=1),
    )
)

# swing line
zigzag_points = pd.concat(
    [
        swings[["start_date", "start_price"]]
        .rename(
            columns={
                "start_date": "date",
                "start_price": "price",
            }
        ),
        swings[["end_date", "end_price"]]
        .rename(
            columns={
                "end_date": "date",
                "end_price": "price",
            }
        ),
    ],
    ignore_index=True,
)

zigzag_points = (
    zigzag_points
    .drop_duplicates(subset="date", keep="last")
    .sort_values("date")
)

zigzag_points["log_price"] = np.log(zigzag_points["price"])

price_figure.add_trace(
    go.Scatter(
        x=zigzag_points["date"],
        y=zigzag_points["log_price"],
        mode="lines+markers",
        name="Swing",
        line=dict(color="orange", width=1),
        marker=dict(size=4),
        visible="legendonly",
    )
)

log_tick_min = float(
    np.floor(
        prices["LOG AVG"].min() / log_grid_interval
    ) * log_grid_interval
)

log_tick_max = float(
    np.ceil(
        prices["LOG AVG"].max() / log_grid_interval
    ) * log_grid_interval
)

if log_tick_max <= log_tick_min:
    log_tick_max = log_tick_min + log_grid_interval

tick_count = int(
    round(
        (log_tick_max - log_tick_min)
        / log_grid_interval
    )
)

log_price_ticks = (
    log_tick_min
    + np.arange(tick_count + 1)
    * log_grid_interval
)

log_tick_decimals = max(
    0,
    int(np.ceil(-np.log10(log_grid_interval)))
)

natural_price_ticks = np.exp(log_price_ticks)
log_price_range = [log_tick_min, log_tick_max]

price_figure.add_trace(
    go.Scatter(
        x=prices.index,
        y=prices["LOG AVG"],
        yaxis="y2",
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
)

price_figure.update_layout(
    xaxis_title="Date",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    yaxis={
        "title": "Log average price (HLC3)",
        "range": log_price_range,
        "tickmode": "array",
        "tickvals": log_price_ticks,
        "ticktext": [
            f"{value:.{log_tick_decimals}f}"
            for value in log_price_ticks
        ],
        "showgrid": True,
    },
    yaxis2={
        "overlaying": "y",
        "side": "right",
        "title": "Average price (HLC3)",
        "showticklabels": True,
        "range": log_price_range,
        "tickmode": "array",
        "tickvals": log_price_ticks,
        "ticktext": [
            f"{price:,.0f}"
            for price in natural_price_ticks
        ],
        "showgrid": False,
    },
)

st.plotly_chart(
    price_figure,
    width="stretch",
    config={"scrollZoom": False},
)

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
chart_tabs = st.tabs([title for _, title, _, _ in plot_metrics])
for chart_tab, (metric, title, y_label, y_ticks) in zip(chart_tabs, plot_metrics):
    with chart_tab:
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
                axis.text(
                    0.5, 0.5, "No swings available", ha="center", va="center"
                )
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
st.caption(
    "Indicator summaries cover true range and ATR (absolute and percentage), "
    "close and HLC3 returns (including absolute returns), and their ATR multiples. "
    "Swing summaries cover size / ATR, size (%), and duration, each grouped across all, up, and "
    "down swings. Return-to-ATR values are multiples, not percentages."
)
statistics = descriptive_statistics(prices, swings)
metric_families = statistics["Metric"].str.partition(" · ")[0]
for metric_family in metric_families.drop_duplicates():
    metric_rows = statistics.loc[metric_families == metric_family]
    with st.expander(metric_family, expanded=False):
        st.dataframe(
            metric_rows.drop(columns=["Metric"]),
            hide_index=True,
            width="stretch",
            height="content",
        )

st.subheader("Observations")
display_swings = swings.sort_values("end_date", ascending=False).copy()
for column in ("start_date", "end_date"):
    display_swings[column] = pd.to_datetime(display_swings[column]).dt.date
with st.expander("Swing observations", expanded=False):
    st.dataframe(
        display_swings,
        hide_index=True,
        width="stretch",
    )
price_observation_columns = {
    "AVG": "Average price (HLC3)",
    "LOG AVG": "Log average price",
    "TR": "True range",
    "ATR": "Average true range",
    "TR%": "True range (%)",
    "ATR%": "Average true range (%)",
    "Return_C%": "Close return (%)",
    "Abs_Return_C%": "Absolute close return (%)",
    "Return_C_Ratio": "Close return (ATR multiples)",
    "Abs_Return_C_Ratio": "Absolute close return (ATR multiples)",
    "Return_A%": "HLC3 return (%)",
    "Abs_Return_A%": "Absolute HLC3 return (%)",
    "Return_A_Ratio": "HLC3 return (ATR multiples)",
    "Abs_Return_A_Ratio": "Absolute HLC3 return (ATR multiples)",
}
price_observations = (
    prices.rename(columns=price_observation_columns)
    .rename_axis("Date")
    .reset_index()
    .sort_values("Date", ascending=False)
)
price_observations["Date"] = pd.to_datetime(price_observations["Date"]).dt.date
preferred_observation_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Average price (HLC3)",
    "Log average price",
    "True range",
    "Average true range",
    "True range (%)",
    "Average true range (%)",
    "Close return (%)",
    "Absolute close return (%)",
    "Close return (ATR multiples)",
    "Absolute close return (ATR multiples)",
    "HLC3 return (%)",
    "Absolute HLC3 return (%)",
    "HLC3 return (ATR multiples)",
    "Absolute HLC3 return (ATR multiples)",
]
ordered_observation_columns = [
    column
    for column in preferred_observation_columns
    if column in price_observations.columns
]
ordered_observation_columns.extend(
    column
    for column in price_observations.columns
    if column not in ordered_observation_columns
)
price_observations = price_observations[ordered_observation_columns]

with st.expander("Price and indicator observations", expanded=False):
    st.dataframe(
        price_observations,
        hide_index=True,
        width="stretch",
        height=500,
    )
st.subheader("Download")
download_columns = st.columns(3)
download_columns[0].download_button(
    "Descriptive statistics CSV",
    data=statistics.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker_symbol.lower()}_{interval}_swing_statistics.csv",
    mime="text/csv",
    width="stretch",
)
download_columns[1].download_button(
    "Swing observations CSV",
    data=swings.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker_symbol.lower()}_{interval}_swings.csv",
    mime="text/csv",
    width="stretch",
)
download_columns[2].download_button(
    "Price and indicator observations CSV",
    data=price_observations.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker_symbol.lower()}_{interval}_price_observations.csv",
    mime="text/csv",
    width="stretch",
)
