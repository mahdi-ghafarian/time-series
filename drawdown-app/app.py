from pathlib import Path
from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
BUNDLED_DATASETS = {
    "S&P 500 monthly": APP_DIRECTORY / "data" / "sp500-monthly.csv",
    "Bitcoin weekly": APP_DIRECTORY / "data" / "btc-weekly.csv",
}
DRAWDOWN_COLUMNS = [
    "Start",
    "Trough",
    "End",
    "Drawdown (%)",
    "Duration (days)",
]


st.set_page_config(
    page_title="Drawdown Calculator",
    page_icon="📉",
    layout="wide",
)


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    try:
        return pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read the selected CSV file: {exc}") from exc


def calculate_drawdowns(
    source: pd.DataFrame,
    date_column: str,
    price_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = {date_column, price_column}.difference(source.columns)
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    prices = source[[date_column, price_column]].copy()
    prices[date_column] = pd.to_datetime(prices[date_column], errors="coerce")
    prices[price_column] = pd.to_numeric(prices[price_column], errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[date_column, price_column]
    )
    prices = prices.loc[prices[price_column] > 0]
    prices = prices.sort_values(date_column).drop_duplicates(
        subset=[date_column], keep="last"
    )
    prices = prices.set_index(date_column)
    prices.index.name = "Date"

    if len(prices) < 2:
        raise ValueError("At least two valid, positive price observations are required.")

    prices["Peak"] = prices[price_column].cummax()
    prices["Drawdown"] = (prices[price_column] - prices["Peak"]) / prices[
        "Peak"
    ] * 100

    drawdown_periods = []
    in_drawdown = False
    start: pd.Timestamp | None = None
    date_index = pd.DatetimeIndex(prices.index)

    for current_date, drawdown_value in zip(
        date_index, prices["Drawdown"].to_numpy()
    ):
        if not in_drawdown and drawdown_value < 0:
            in_drawdown = True
            start = current_date
        elif in_drawdown and start is not None and drawdown_value >= 0:
            end = current_date
            episode = prices.loc[start:end, "Drawdown"]
            trough = episode.idxmin()
            drawdown_periods.append(
                {
                    "Start": start,
                    "Trough": trough,
                    "End": end,
                    "Drawdown (%)": prices.at[trough, "Drawdown"],
                    "Duration (days)": (end - start).days,
                }
            )
            in_drawdown = False
            start = None

    completed_drawdowns = pd.DataFrame(
        drawdown_periods, columns=DRAWDOWN_COLUMNS
    )
    return prices, completed_drawdowns


st.title("Drawdown Calculator")
st.write(
    "Analyze historical peak-to-trough declines from a CSV price series. "
    "Drawdown episodes are counted when price recovers to a previous high."
)

with st.sidebar:
    st.header("Data")
    data_source = st.radio(
        "Source",
        ["Bundled dataset", "Upload CSV"],
        horizontal=True,
    )
    uploaded_file = None
    if data_source == "Bundled dataset":
        dataset_name = st.selectbox(
            "Dataset",
            list(BUNDLED_DATASETS),
        )
        csv_source = BUNDLED_DATASETS[dataset_name]
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        csv_source = uploaded_file

if csv_source is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    raw_data = load_csv(csv_source)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if raw_data.empty or len(raw_data.columns) < 2:
    st.error("The CSV must contain a date column and at least one price column.")
    st.stop()

with st.sidebar:
    column_names = list(raw_data.columns)
    date_default = column_names.index("Date") if "Date" in column_names else 0
    date_column = st.selectbox(
        "Date column",
        column_names,
        index=date_default,
    )
    price_candidates = [
        column for column in column_names if column != date_column
    ]
    preferred_price = next(
        (
            column
            for column in ("SP500", "Close", "Adj Close")
            if column in price_candidates
        ),
        price_candidates[0],
    )
    price_column = st.selectbox(
        "Price column",
        price_candidates,
        index=price_candidates.index(preferred_price),
    )

try:
    prices, drawdowns = calculate_drawdowns(
        raw_data,
        date_column,
        price_column,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

top_drawdowns = drawdowns.sort_values("Drawdown (%)").head(10)
ongoing = (
    not prices.empty
    and float(prices["Drawdown"].iloc[-1]) < 0
)

metric_columns = st.columns(4)
metric_columns[0].metric("Valid observations", f"{len(prices):,}")
metric_columns[1].metric("Recovered drawdowns", f"{len(drawdowns):,}")
metric_columns[2].metric(
    "Worst recovered drawdown",
    f"{drawdowns['Drawdown (%)'].min():.2f}%"
    if not drawdowns.empty
    else "Not available",
)
metric_columns[3].metric(
    "Current drawdown",
    f"{prices['Drawdown'].iloc[-1]:.2f}%",
)
if ongoing:
    st.info(
        "The latest observation remains below its previous peak. This active "
        "drawdown is shown in the chart and current metric, but excluded from "
        "the recovered-episode table and summary statistics, matching the notebook."
    )

st.subheader(f"{price_column} price history")
figure, axis = plt.subplots(figsize=(12, 4))
axis.plot(prices.index, prices[price_column], color="blue")
axis.set_title("Price Over Time (Log Scale)")
axis.set_ylabel(price_column)
axis.set_yscale("log")
axis.grid(True, alpha=0.35)
figure.tight_layout()
st.pyplot(figure, width="stretch")
plt.close(figure)

st.subheader("Drawdown over time")
figure, axis = plt.subplots(figsize=(12, 4))
axis.plot(prices.index, prices["Drawdown"], color="red")
axis.set_title("Drawdown Over Time (%)")
axis.set_ylabel("Drawdown (%)")
for _, row in top_drawdowns.iterrows():
    axis.axvspan(row["Start"], row["End"], color="gray", alpha=0.3)
axis.grid(True, alpha=0.35)
figure.tight_layout()
st.pyplot(figure, width="stretch")
plt.close(figure)

st.subheader("Top 10 recovered drawdowns")
if drawdowns.empty:
    st.info("No recovered drawdown episodes were found in this data.")
else:
    display_top = top_drawdowns.copy()
    display_top["Start"] = pd.to_datetime(display_top["Start"]).dt.date
    display_top["Trough"] = pd.to_datetime(display_top["Trough"]).dt.date
    display_top["End"] = pd.to_datetime(display_top["End"]).dt.date
    st.dataframe(display_top, hide_index=True, width="stretch")

    st.subheader("Drawdown summary statistics")
    summary_columns = st.columns(2)
    summary_columns[0].caption("Drawdown depth (%)")
    summary_columns[0].dataframe(
        drawdowns["Drawdown (%)"].describe().to_frame().T,
        hide_index=True,
        width="stretch",
    )
    summary_columns[1].caption("Duration (days)")
    summary_columns[1].dataframe(
        drawdowns["Duration (days)"].describe().to_frame().T,
        hide_index=True,
        width="stretch",
    )

st.download_button(
    "Download recovered drawdowns (CSV)",
    data=drawdowns.to_csv(index=False).encode("utf-8"),
    file_name="drawdown_episodes.csv",
    mime="text/csv",
)

with st.expander("Validated price observations"):
    st.dataframe(prices, width="stretch")
