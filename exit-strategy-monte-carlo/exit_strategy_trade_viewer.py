"""Interactive trade explorer for Exit Strategy Monte Carlo *_trades.csv files.

Run:
    pip install streamlit plotly yfinance pandas numpy
    streamlit run trade_viewer.py

Pick a trades file (upload it or choose one from a results folder), filter by
stop / target / direction / exit reason, click rows in the table to select
trades, and see them drawn on the price chart with entry, exit, stop and
target levels.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Observer", layout="wide")

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "visualization"
REQUIRED_COLUMNS = {
    "EntryDate",
    "ExitDate",
    "Direction",
    "EntryReferencePrice",
    "ExitReferencePrice",
    "NetReturnPct",
}
TABLE_COLUMNS = [
    "Scenario",
    "StopParameter",
    "TargetParameter",
    "Direction",
    "EntryDate",
    "ExitDate",
    "BarsHeld",
    "ExitReason",
    "NetReturnPct",
    "RealizedR",
]
GREEN, RED, BLUE, GREY = "#2e9e5b", "#d6453d", "#3b82c4", "#8a8f98"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_trades(source) -> pd.DataFrame:
    trades = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(trades.columns)
    if missing:
        raise ValueError(f"Not a trades file; missing columns: {sorted(missing)}")
    for column in ("EntryDate", "ExitDate"):
        trades[column] = pd.to_datetime(trades[column]).dt.tz_localize(None)
    return trades.reset_index(drop=True)


@st.cache_data(show_spinner="Downloading price data...")
def load_prices(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("No price data returned. Check ticker / interval.")
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close"]].dropna().sort_index()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None)
    return data


def guess_from_filename(name: str) -> tuple[str, str]:
    """Filenames look like qqq_1d_pct_n1000_...; return ticker, interval."""
    parts = Path(name).stem.split("_")
    ticker = parts[0].upper() if parts else "QQQ"
    interval = parts[1] if len(parts) > 1 and parts[1] in {"1d", "1wk", "1mo"} else "1d"
    return ticker, interval


def add_levels(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute stop / target price levels from entry reference price."""
    trades = trades.copy()
    sign = np.where(trades["Direction"] == "Long", 1.0, -1.0)
    entry = trades["EntryReferencePrice"].to_numpy(float)

    def level(column: str, side: float) -> np.ndarray:
        if column not in trades:
            return np.full(len(trades), np.nan)
        pct = trades[column].to_numpy(float)
        price = entry * (1 + side * sign * pct / 100)
        return np.where(pct > 0, price, np.nan)  # -1 means "no stop/target"

    trades["StopPrice"] = level("StopLossPct", -1.0)
    trades["TargetPrice"] = level("TargetPct", 1.0)
    return trades


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def build_chart(
    prices: pd.DataFrame,
    selected: pd.DataFrame,
    ticker: str,
    candles: bool,
    show_levels: bool,
    log_y: bool,
    x_range: tuple | None,
) -> go.Figure:
    fig = go.Figure()
    if candles:
        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices["Open"],
                high=prices["High"],
                low=prices["Low"],
                close=prices["Close"],
                name=ticker,
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices["Close"],
                mode="lines",
                name=f"{ticker} close",
                line=dict(color=GREY, width=1.4),
            )
        )

    if not selected.empty:
        win = selected["NetReturnPct"] > 0

        # Connector lines entry -> exit (None separates segments in one trace).
        for mask, color, label in ((win, GREEN, "Winner"), (~win, RED, "Loser")):
            part = selected[mask]
            if part.empty:
                continue
            xs, ys = [], []
            for row in part.itertuples():
                xs += [row.EntryDate, row.ExitDate, None]
                ys += [row.EntryReferencePrice, row.ExitReferencePrice, None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} path",
                    hoverinfo="skip",
                )
            )

        if show_levels:
            for column, color, label in (
                ("StopPrice", RED, "Stop"),
                ("TargetPrice", GREEN, "Target"),
            ):
                xs, ys = [], []
                for row in selected.itertuples():
                    level = getattr(row, column)
                    if np.isfinite(level):
                        xs += [row.EntryDate, row.ExitDate, None]
                        ys += [level, level, None]
                if xs:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(color=color, width=1, dash="dot"),
                            name=f"{label} level",
                            hoverinfo="skip",
                        )
                    )

        def hover(frame: pd.DataFrame, kind: str) -> list[str]:
            return [
                f"{kind} {r.Direction}<br>Entry {r.EntryDate:%Y-%m-%d} @ {r.EntryReferencePrice:.2f}"
                f"<br>Exit {r.ExitDate:%Y-%m-%d} @ {r.ExitReferencePrice:.2f}"
                f"<br>Return {r.NetReturnPct:+.2f}%"
                f"<br>Reason {getattr(r, 'ExitReason', '?')}"
                for r in frame.itertuples()
            ]

        fig.add_trace(
            go.Scatter(
                x=selected["EntryDate"],
                y=selected["EntryReferencePrice"],
                mode="markers",
                marker=dict(
                    symbol=np.where(
                        selected["Direction"] == "Long", "triangle-up", "triangle-down"
                    ),
                    size=12,
                    color=BLUE,
                    line=dict(color="white", width=1),
                ),
                name="Entry",
                text=hover(selected, "Entry"),
                hoverinfo="text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=selected["ExitDate"],
                y=selected["ExitReferencePrice"],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=11,
                    color=np.where(win, GREEN, RED),
                    line=dict(color="white", width=1),
                ),
                name="Exit",
                text=hover(selected, "Exit"),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        height=640,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        xaxis=dict(rangeslider=dict(visible=False), range=x_range),
        yaxis=dict(type="log" if log_y else "linear", title="Price"),
        legend=dict(orientation="h", y=1.04, x=0),
        title=f"{ticker} - {len(selected)} selected trade(s)",
    )
    return fig


def y_range_for(prices: pd.DataFrame, x_range, log_y: bool):
    """Fit the y axis to the visible x window (plotly does not autoscale on set range)."""
    if x_range is None:
        return None
    window = prices.loc[x_range[0] : x_range[1]]
    if window.empty:
        return None
    lo, hi = float(window["Low"].min()), float(window["High"].max())
    pad = (hi - lo) * 0.05 or hi * 0.02
    lo, hi = lo - pad, hi + pad
    if log_y:
        return [np.log10(max(lo, 1e-9)), np.log10(hi)]
    return [lo, hi]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("Observer")

with st.sidebar:
    st.header("1. Trades file")
    source_mode = st.radio("Source", ["Choose from folder", "Upload CSV"], horizontal=True)
    trades_source, file_name = None, ""
    if source_mode == "Upload CSV":
        upload = st.file_uploader("Trades CSV", type="csv")
        if upload is not None:
            trades_source, file_name = upload, upload.name
    else:
        folder = Path(
            st.text_input("Folder", str(DEFAULT_RESULTS_DIR)).strip().strip('"')
        )
        files = sorted(folder.glob("*trades*.csv")) if folder.is_dir() else []
        if not folder.is_dir():
            st.warning("Folder not found.")
        elif not files:
            st.info("No *trades*.csv files in this folder.")
        else:
            choice = st.selectbox("File", [f.name for f in files])
            trades_source, file_name = folder / choice, choice

if trades_source is None:
    st.info("Select a trades CSV in the sidebar to begin.")
    st.stop()

try:
    trades = add_levels(load_trades(trades_source))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read file: {exc}")
    st.stop()

guess_ticker, guess_interval = guess_from_filename(file_name)

with st.sidebar:
    st.header("2. Price data")
    ticker = st.text_input("Ticker", guess_ticker).strip().upper()
    interval = st.selectbox(
        "Interval", ["1d", "1wk", "1mo"], index=["1d", "1wk", "1mo"].index(guess_interval)
    )
    pad_days = st.slider("Padding around trades (days)", 0, 365, 30)

    st.header("3. Filter trades")

    def multi(label: str, column: str, default_first: bool = False):
        if column not in trades:
            return None
        values = sorted(trades[column].dropna().unique().tolist())
        default = values[:1] if default_first else values
        return st.multiselect(label, values, default=default)

    scenarios = multi("Scenario", "Scenario", default_first=True)
    stops = multi("Stop parameter", "StopParameter", default_first=True)
    targets = multi("Target parameter", "TargetParameter", default_first=True)
    directions = multi("Direction", "Direction")
    reasons = multi("Exit reason", "ExitReason")
    outcome = st.radio("Outcome", ["All", "Winners", "Losers"], horizontal=True)

filtered = trades
for column, chosen in (
    ("Scenario", scenarios),
    ("StopParameter", stops),
    ("TargetParameter", targets),
    ("Direction", directions),
    ("ExitReason", reasons),
):
    if chosen is not None:
        filtered = filtered[filtered[column].isin(chosen)]
if outcome == "Winners":
    filtered = filtered[filtered["NetReturnPct"] > 0]
elif outcome == "Losers":
    filtered = filtered[filtered["NetReturnPct"] <= 0]
filtered = filtered.sort_values("EntryDate").reset_index(drop=True)

st.caption(f"{file_name} - {len(trades):,} trades in file, {len(filtered):,} after filters")
if filtered.empty:
    st.warning("No trades match the current filters.")
    st.stop()

left, right = st.columns([2, 3], gap="medium")

with left:
    st.subheader("Trades")
    st.caption("Click rows to select (Ctrl/Cmd or Shift for several).")
    columns = [c for c in TABLE_COLUMNS if c in filtered.columns]
    event = st.dataframe(
        filtered[columns],
        hide_index=False,
        use_container_width=True,
        height=520,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "EntryDate": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "ExitDate": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
            "NetReturnPct": st.column_config.NumberColumn("Net %", format="%.2f"),
            "RealizedR": st.column_config.NumberColumn("R", format="%.2f"),
        },
        key="trade_table",
    )
    plot_all = st.checkbox("Plot every trade in the table", value=False)
    max_plot = st.number_input("Max trades to plot", 1, 5000, 300, step=50)

selected_rows = list(event.selection.rows)
selected = filtered if plot_all else filtered.iloc[selected_rows]
if len(selected) > max_plot:
    st.warning(f"Showing the first {max_plot} of {len(selected)} trades.")
    selected = selected.head(int(max_plot))

with right:
    st.subheader("Chart")
    o1, o2, o3, o4 = st.columns(4)
    candles = o1.toggle("Candles", value=False)
    show_levels = o2.toggle("Stop/target", value=True)
    log_y = o3.toggle("Log scale", value=False)
    zoom = o4.toggle("Zoom to selection", value=True)

    # Price window: selected trades (or filtered trades) plus padding.
    basis = selected if not selected.empty else filtered
    first = basis["EntryDate"].min()
    last = basis["ExitDate"].max()
    fetch_start = (filtered["EntryDate"].min() - timedelta(days=pad_days)).date()
    fetch_end = (filtered["ExitDate"].max() + timedelta(days=pad_days + 1)).date()
    try:
        prices = load_prices(ticker, fetch_start.isoformat(), fetch_end.isoformat(), interval)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Price download failed: {exc}")
        st.stop()

    x_range = None
    if zoom and not selected.empty:
        x_range = (first - timedelta(days=pad_days), last + timedelta(days=pad_days))

    fig = build_chart(prices, selected, ticker, candles, show_levels, log_y, x_range)
    y_range = y_range_for(prices, x_range, log_y)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    st.plotly_chart(fig, use_container_width=True)

    if selected.empty:
        st.info("Select one or more rows in the table to draw them on the chart.")
    else:
        wins = (selected["NetReturnPct"] > 0).mean() * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", f"{len(selected):,}")
        m2.metric("Win rate", f"{wins:.1f}%")
        m3.metric("Avg net return", f"{selected['NetReturnPct'].mean():+.2f}%")
        if "RealizedR" in selected:
            m4.metric("Total R", f"{selected['RealizedR'].sum():+.2f}")
