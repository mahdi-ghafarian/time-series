from datetime import date, timedelta
from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.patches import Rectangle


# ============================================================
# Configuration
# ============================================================

STARTUP_RESULT_FILE = Path(__file__).parent / "startup_result.pkl"

IN_SUITE = st.session_state.get(
    "_market_analysis_suite",
    False,
)

if not IN_SUITE:
    st.set_page_config(
        page_title="Exit Strategy",
        page_icon="📊",
        layout="wide",
    )


DEFAULT_STOPS = [
    1,
    2,
    3,
    4,
    5,
    7.5,
    10,
    12.5,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    -1,
]

DEFAULT_TARGETS = [
    1,
    2,
    3,
    4,
    5,
    7.5,
    10,
    12.5,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    100,
    -1,
]

METRICS = [
    "StopLossPct",
    "TargetPct",
    "NumberOfTrades",
    "WinRatePct",
    "AverageWinPct",
    "AverageLossPct",
    "RealizedRRR",
    "BreakEvenWinRatePct",
    "PlannedRRR",
    "PlannedBreakEvenWinRatePct",
    "ExpectancyPct",
    "ExpectancyR",
    "AverageBarsHeld",
    "StopExitPct",
    "TargetExitPct",
    "EndOfDataExitPct",
    "AverageCommissionPct",
    "Score",
]


# ============================================================
# Ticker search
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def search_tickers(query):

    quotes = yf.Search(
        query,
        max_results=8,
        news_count=0,
        lists_count=0,
        include_cb=False,
    ).quotes

    return [
        {
            "symbol": str(q["symbol"]),
            "name": str(
                q.get("longname")
                or q.get("shortname")
                or q["symbol"]
            ),
            "exchange": str(
                q.get("exchDisp")
                or q.get("exchange")
                or ""
            ),
        }
        for q in quotes
        if q.get("symbol")
    ]


def select_ticker(symbol, label=""):

    symbol = symbol.upper()

    st.session_state["exit_ticker"] = symbol
    st.session_state["exit_search_ticker"] = ""
    st.session_state["exit_search_ticker_label"] = label

    if st.session_state.get(
        "_market_analysis_suite",
        False,
    ):
        st.session_state[
            "_suite_shared_ticker"
        ] = symbol


# ============================================================
# Historical data
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def download_data(
    ticker,
    start_date,
    end_date,
    interval,
):

    # yfinance treats the end date as exclusive.
    yf_end = end_date + timedelta(days=1)

    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=yf_end.isoformat(),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data downloaded. Check the ticker, dates, and interval."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if ticker in data.columns.get_level_values(-1):
            data = data.xs(
                ticker,
                axis=1,
                level=-1,
            )
        else:
            data.columns = data.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    data = (
        data[required]
        .dropna()
        .sort_index()
    )

    if len(data) < 2:
        raise ValueError(
            "At least two valid price bars are required."
        )

    if (data[required] <= 0).any().any():
        raise ValueError(
            "Price data contains zero or negative prices."
        )

    return data


# ============================================================
# Random entries
# ============================================================

def select_random_entries(
    data,
    number_of_entries,
    seed,
):

    maximum_entries = len(data) - 1

    if maximum_entries < 1:
        raise ValueError(
            "Not enough price data to select entries."
        )

    actual_entries = min(
        int(number_of_entries),
        maximum_entries,
    )

    rng = np.random.default_rng(
        int(seed)
    )

    indexes = np.sort(
        rng.choice(
            np.arange(maximum_entries),
            size=actual_entries,
            replace=False,
        )
    )

    return pd.DataFrame(
        {
            "EntryIndex": indexes,
            "EntryDate": data.index[indexes],
            "EntryClose":
                data["Close"].to_numpy()[indexes],
        }
    )


# ============================================================
# Trade simulation
# ============================================================

def simulate_trade(
    data,
    entry_index,
    stop_loss_pct,
    target_pct,
    commission_pct,
    slippage_pct,
    conservative,
):

    close_prices = data[
        "Close"
    ].to_numpy()

    high_prices = data[
        "High"
    ].to_numpy()

    low_prices = data[
        "Low"
    ].to_numpy()

    dates = data.index

    entry_reference = float(
        close_prices[entry_index]
    )

    no_stop = stop_loss_pct == -1
    no_target = target_pct == -1

    entry_execution = (
        entry_reference
        * (
            1
            + float(slippage_pct) / 100
        )
    )

    stop_price = (
        None
        if no_stop
        else entry_reference
        * (
            1
            - float(stop_loss_pct) / 100
        )
    )

    target_price = (
        None
        if no_target
        else entry_reference
        * (
            1
            + float(target_pct) / 100
        )
    )

    exit_index = len(data) - 1

    exit_reference = float(
        close_prices[exit_index]
    )

    exit_reason = "EndOfData"

    for current_index in range(
        entry_index + 1,
        len(data),
    ):

        stop_hit = (
            stop_price is not None
            and float(
                low_prices[current_index]
            ) <= stop_price
        )

        target_hit = (
            target_price is not None
            and float(
                high_prices[current_index]
            ) >= target_price
        )

        if stop_hit and target_hit:

            if conservative:
                exit_reference = stop_price
                exit_reason = "Stop"
            else:
                exit_reference = target_price
                exit_reason = "Target"

            exit_index = current_index
            break

        if stop_hit:
            exit_reference = stop_price
            exit_reason = "Stop"
            exit_index = current_index
            break

        if target_hit:
            exit_reference = target_price
            exit_reason = "Target"
            exit_index = current_index
            break

    exit_execution = (
        exit_reference
        * (
            1
            - float(slippage_pct) / 100
        )
    )

    gross_return = (
        exit_execution
        / entry_execution
        - 1
    ) * 100

    total_commission = (
        2
        * float(commission_pct)
    )

    net_return = (
        gross_return
        - total_commission
    )

    risk = (
        100
        if no_stop
        else float(stop_loss_pct)
    )

    return {
        "EntryDate":
            dates[entry_index],

        "ExitDate":
            dates[exit_index],

        "EntryIndex":
            entry_index,

        "ExitIndex":
            exit_index,

        "EntryReferencePrice":
            entry_reference,

        "EntryExecutionPrice":
            entry_execution,

        "ExitReferencePrice":
            exit_reference,

        "ExitExecutionPrice":
            exit_execution,

        "StopLossPct":
            stop_loss_pct,

        "TargetPct":
            target_pct,

        "GrossReturnPct":
            gross_return,

        "NetReturnPct":
            net_return,

        "RealizedR":
            net_return / risk,

        "BarsHeld":
            exit_index - entry_index,

        "ExitReason":
            exit_reason,

        "CommissionPct":
            total_commission,
    }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    trades,
    stop,
    target,
):

    returns = trades[
        "NetReturnPct"
    ].to_numpy(float)

    wins = returns > 0
    losses = returns < 0

    winning_returns = returns[wins]
    losing_returns = returns[losses]

    number_of_trades = len(returns)

    win_rate = (
        float(wins.mean() * 100)
        if number_of_trades
        else np.nan
    )

    average_win = (
        float(winning_returns.mean())
        if len(winning_returns)
        else np.nan
    )

    average_loss = (
        float(losing_returns.mean())
        if len(losing_returns)
        else np.nan
    )

    expectancy = (
        float(returns.mean())
        if number_of_trades
        else np.nan
    )

    risk = (
        100
        if stop == -1
        else float(stop)
    )

    expectancy_r = (
        expectancy / risk
    )

    planned_rrr = (
        np.nan
        if target == -1
        else float(target) / risk
    )

    realized_rrr = (
        average_win / abs(average_loss)
        if (
            len(winning_returns)
            and len(losing_returns)
            and average_loss != 0
        )
        else np.nan
    )

    break_even_win_rate = (
        abs(average_loss)
        / (
            average_win
            + abs(average_loss)
        )
        * 100
        if (
            np.isfinite(average_win)
            and np.isfinite(average_loss)
            and average_win > 0
            and average_loss < 0
        )
        else np.nan
    )

    planned_break_even_win_rate = (
        1
        / (
            1
            + planned_rrr
        )
        * 100
        if (
            np.isfinite(planned_rrr)
            and planned_rrr > 0
        )
        else np.nan
    )

    exit_counts = (
        trades["ExitReason"]
        .value_counts()
    )

    return {
        "StopLossPct":
            stop,

        "TargetPct":
            target,

        "NumberOfTrades":
            number_of_trades,

        "WinRatePct":
            win_rate,

        "AverageWinPct":
            average_win,

        "AverageLossPct":
            average_loss,

        "RealizedRRR":
            realized_rrr,

        "BreakEvenWinRatePct":
            break_even_win_rate,

        "PlannedRRR":
            planned_rrr,

        "PlannedBreakEvenWinRatePct":
            planned_break_even_win_rate,

        "ExpectancyPct":
            expectancy,

        "ExpectancyR":
            expectancy_r,

        "AverageBarsHeld":
            float(
                trades[
                    "BarsHeld"
                ].mean()
            )
            if number_of_trades
            else np.nan,

        "StopExitPct":
            exit_counts.get(
                "Stop",
                0,
            )
            / number_of_trades
            * 100
            if number_of_trades
            else np.nan,

        "TargetExitPct":
            exit_counts.get(
                "Target",
                0,
            )
            / number_of_trades
            * 100
            if number_of_trades
            else np.nan,

        "EndOfDataExitPct":
            exit_counts.get(
                "EndOfData",
                0,
            )
            / number_of_trades
            * 100
            if number_of_trades
            else np.nan,

        "AverageCommissionPct":
            float(
                trades[
                    "CommissionPct"
                ].mean()
            )
            if number_of_trades
            else np.nan,
    }


# ============================================================
# Compare combinations
# ============================================================

def compare_combinations(
    data,
    entries,
    stops,
    targets,
    commission,
    slippage,
    conservative,
    rank_metric,
):

    result_records = []
    trade_records = []

    total_combinations = (
        len(stops)
        * len(targets)
    )

    combinations_completed = 0

    progress = st.progress(
        0,
        text="Testing combinations...",
    )

    for stop in stops:

        for target in targets:

            combination_trades = [
                simulate_trade(
                    data=data,
                    entry_index=int(
                        entry.EntryIndex
                    ),
                    stop_loss_pct=stop,
                    target_pct=target,
                    commission_pct=commission,
                    slippage_pct=slippage,
                    conservative=conservative,
                )
                for entry
                in entries.itertuples(
                    index=False
                )
            ]

            combination_frame = pd.DataFrame(
                combination_trades
            )

            result_records.append(
                calculate_metrics(
                    combination_frame,
                    stop,
                    target,
                )
            )

            trade_records.extend(
                combination_trades
            )

            combinations_completed += 1

            progress.progress(
                combinations_completed
                / total_combinations,
                text=(
                    "Testing combinations: "
                    f"{combinations_completed}"
                    f"/{total_combinations}"
                ),
            )

    progress.empty()

    results = pd.DataFrame(
        result_records
    )

    trades = pd.DataFrame(
        trade_records
    )

    for metric in [
        "ExpectancyR",
        "WinRatePct",
    ]:

        standard_deviation = (
            results[metric].std()
        )

        if (
            standard_deviation
            and np.isfinite(
                standard_deviation
            )
        ):
            results[
                f"z_{metric}"
            ] = (
                results[metric]
                - results[metric].mean()
            ) / standard_deviation
        else:
            results[
                f"z_{metric}"
            ] = 0.0

    results["ScoreRaw"] = (
        2
        * results[
            "z_ExpectancyR"
        ]
        + results[
            "z_WinRatePct"
        ]
    )

    minimum_score = results[
        "ScoreRaw"
    ].min()

    maximum_score = results[
        "ScoreRaw"
    ].max()

    if maximum_score != minimum_score:

        results["Score"] = (
            100
            * (
                results["ScoreRaw"]
                - minimum_score
            )
            / (
                maximum_score
                - minimum_score
            )
        )

    else:
        results["Score"] = 50.0

    results = (
        results
        .sort_values(
            rank_metric,
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    results.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(results) + 1,
        ),
    )

    return results, trades


# ============================================================
# Heatmap
# ============================================================

def create_heatmap(
    results,
    stops,
    targets,
    ticker,
    start_date,
    end_date,
    number_of_entries,
    top_metric,
    bottom_metric,
    color_metric,
    vmin,
    vmax,
    win_rate_minimum,
    expectancy_r_minimum,
):

    color_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=color_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    top_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=top_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    bottom_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=bottom_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    highlighted = (
        pd.concat(
            [
                results[
                    [
                        "TargetPct",
                        "StopLossPct",
                    ]
                ],
                (
                    (
                        results[
                            "WinRatePct"
                        ]
                        > win_rate_minimum
                    )
                    &
                    (
                        results[
                            "ExpectancyR"
                        ]
                        > expectancy_r_minimum
                    )
                ).rename(
                    "Condition"
                ),
            ],
            axis=1,
        )
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values="Condition",
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    matrix = color_values.to_numpy(
        float
    )

    finite_values = matrix[
        np.isfinite(matrix)
    ]

    if not finite_values.size:
        return None

    if vmin is None:
        vmin = float(
            np.min(finite_values)
        )

    if vmax is None:
        vmax = float(
            np.max(finite_values)
        )

    if vmin == vmax:

        padding = max(
            abs(vmin) * 0.01,
            1e-9,
        )

        vmin -= padding
        vmax += padding

    figure, axis = plt.subplots(
        figsize=(
            max(
                12,
                len(stops) * 0.85,
            ),
            max(
                8,
                len(targets) * 0.55,
            ),
        )
    )

    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    figure.colorbar(
        image,
        ax=axis,
    ).set_label(
        color_metric
    )

    axis.set_title(
        f"{ticker} | "
        f"{start_date} to {end_date} | "
        f"{number_of_entries:,} Random Entries\n"
        f"{top_metric} (top), "
        f"{bottom_metric} (bottom)"
    )

    axis.set_xlabel(
        "Stop-Loss (%)"
    )

    axis.set_ylabel(
        "Target (%)"
    )

    axis.set_xticks(
        np.arange(
            len(stops)
        )
    )

    axis.set_xticklabels(
        [
            "NS"
            if value == -1
            else str(value)
            for value in stops
        ]
    )

    axis.set_yticks(
        np.arange(
            len(targets)
        )
    )

    axis.set_yticklabels(
        [
            "NT"
            if value == -1
            else str(value)
            for value in targets
        ]
    )

    formats = {
        "ExpectancyR": ".2f",
        "ExpectancyPct": ".2f",
        "WinRatePct": ".1f",
        "RealizedRRR": ".2f",
        "Score": ".0f",
    }

    top_format = formats.get(
        top_metric,
        ".2f",
    )

    bottom_format = formats.get(
        bottom_metric,
        ".2f",
    )

    for row in range(
        len(targets)
    ):

        for column in range(
            len(stops)
        ):

            top_value = top_values.iloc[
                row,
                column,
            ]

            bottom_value = bottom_values.iloc[
                row,
                column,
            ]

            color_value = matrix[
                row,
                column,
            ]

            if (
                np.isfinite(top_value)
                and np.isfinite(
                    bottom_value
                )
            ):
                label = (
                    f"{top_value:{top_format}}\n"
                    f"{bottom_value:{bottom_format}}"
                )
            else:
                label = "N/A"

            if np.isfinite(color_value):
                rgba = image.cmap(
                    image.norm(
                        color_value
                    )
                )
            else:
                rgba = (
                    1,
                    1,
                    1,
                    1,
                )

            brightness = (
                0.299 * rgba[0]
                + 0.587 * rgba[1]
                + 0.114 * rgba[2]
            )

            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=(
                    "black"
                    if brightness > 0.55
                    else "white"
                ),
                fontsize=8,
            )

            if bool(
                highlighted.iloc[
                    row,
                    column,
                ]
            ):
                axis.add_patch(
                    Rectangle(
                        (
                            column - 0.5,
                            row - 0.5,
                        ),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                    )
                )

    axis.set_xticks(
        np.arange(
            -0.5,
            len(stops),
            1,
        ),
        minor=True,
    )

    axis.set_yticks(
        np.arange(
            -0.5,
            len(targets),
            1,
        ),
        minor=True,
    )

    axis.grid(
        which="minor",
        color="gray",
        linewidth=0.4,
        alpha=0.4,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    figure.tight_layout()

    return figure


# ============================================================
# Helper functions
# ============================================================

def parse_values(text):

    return [
        float(value.strip())
        for value in text.split(",")
        if value.strip()
    ]


def run_simulation_now(
    ticker,
    start_date,
    end_date,
    interval,
    number_of_entries,
    seed,
    commission,
    slippage,
    conservative,
    stops,
    targets,
):

    with st.spinner(
        f"Loading {ticker} data..."
    ):

        data = download_data(
            ticker,
            start_date,
            end_date,
            interval,
        )

    entries = select_random_entries(
        data,
        int(number_of_entries),
        int(seed),
    )

    start_time = time.perf_counter()

    results, trades = compare_combinations(
        data=data,
        entries=entries,
        stops=stops,
        targets=targets,
        commission=float(commission),
        slippage=float(slippage),
        conservative=conservative,
        rank_metric="ExpectancyR",
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return (
        data,
        entries,
        results,
        trades,
        stops,
        targets,
        elapsed,
        ticker,
        start_date,
        end_date,
        int(number_of_entries),
    )


def load_startup_result():

    if not STARTUP_RESULT_FILE.exists():
        return None

    try:

        with open(
            STARTUP_RESULT_FILE,
            "rb",
        ) as result_file:

            return pickle.load(
                result_file
            )

    except Exception as exc:

        st.warning(
            "The startup result could not be loaded: "
            f"{exc}"
        )

        return None


# ============================================================
# Application title
# ============================================================

st.title(
    "Exit Strategy"
)

st.write(
    "Test stop-loss and target combinations "
    "across randomly selected historical entries."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "Data"
    )

    search_query = st.text_input(
        "Search for ticker",
        placeholder="Type a ticker or company name",
        key="exit_search_ticker",
        live="500ms",
    ).strip()

    if len(search_query) >= 2:

        try:

            search_results = search_tickers(
                search_query
            )

            for match in search_results:

                result_label = (
                    f"{match['symbol']} | "
                    f"{match['name']} | "
                    f"{match['exchange'] or 'Unknown exchange'}"
                )

                st.button(
                    result_label,
                    key=(
                        "exit_search_result_"
                        f"{match['symbol']}"
                    ),
                    on_click=select_ticker,
                    args=(
                        match["symbol"],
                        result_label,
                    ),
                    width="stretch",
                )

        except Exception as exc:

            st.warning(
                "Yahoo Finance ticker search failed: "
                f"{exc}"
            )

    # suite.py initializes exit_ticker before running this app.
    # When running independently, initialize it here.
    if "exit_ticker" not in st.session_state:

        st.session_state[
            "exit_ticker"
        ] = st.session_state.get(
            "_suite_shared_ticker",
            "QQQ",
        )

    ticker = st.text_input(
        "Ticker",
        key="exit_ticker",
    ).strip().upper()

    interval = st.selectbox(
        "Interval",
        [
            "1d",
            "1wk",
            "1mo",
        ],
        key="exit_interval",
    )

    st.subheader(
        "Date range"
    )

    start_date = st.date_input(
        "Start date",
        value=date(
            2010,
            1,
            1,
        ),
        key="exit_start_date",
    )

    end_date = st.date_input(
        "End date",
        value=date(
            2026,
            10,
            30,
        ),
        key="exit_end_date",
    )

    st.divider()

    st.header(
        "Simulation"
    )

    number_of_entries = st.number_input(
        "Random entries",
        min_value=1,
        value=1000,
        step=100,
        key="exit_n_entries",
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=27,
        step=1,
        key="exit_seed",
    )

    commission = st.number_input(
        "Commission (%)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.3f",
        key="exit_commission",
    )

    slippage = st.number_input(
        "Slippage (%)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.3f",
        key="exit_slippage",
    )

    conservative = st.checkbox(
        "Stop first when stop and target hit same candle",
        value=True,
        key="exit_conservative",
    )

    st.divider()

    st.header(
        "Heatmap"
    )

    heatmap_top_metric = st.selectbox(
        "Top metric",
        METRICS,
        index=METRICS.index(
            "ExpectancyR"
        ),
        key="exit_hm_top",
    )

    heatmap_bottom_metric = st.selectbox(
        "Bottom metric",
        METRICS,
        index=METRICS.index(
            "WinRatePct"
        ),
        key="exit_hm_bottom",
    )

    heatmap_color_metric = st.selectbox(
        "Color metric",
        METRICS,
        index=METRICS.index(
            "ExpectancyR"
        ),
        key="exit_hm_color",
    )

    automatic_color_scale = st.checkbox(
        "Auto color scale (use min/max)",
        value=False,
        key="exit_auto_scale",
    )

    color_column_1, color_column_2 = (
        st.columns(2)
    )

    heatmap_vmin = color_column_1.number_input(
        "Color min",
        value=0.0,
        step=0.1,
        disabled=automatic_color_scale,
        key="exit_hm_vmin",
    )

    heatmap_vmax = color_column_2.number_input(
        "Color max",
        value=1.0,
        step=0.1,
        disabled=automatic_color_scale,
        key="exit_hm_vmax",
    )

    if automatic_color_scale:
        heatmap_vmin = None
        heatmap_vmax = None

    highlight_column_1, highlight_column_2 = (
        st.columns(2)
    )

    highlight_win_rate = (
        highlight_column_1.number_input(
            "Highlight WR min",
            value=50.0,
            step=1.0,
            key="exit_hl_wr",
        )
    )

    highlight_expectancy_r = (
        highlight_column_2.number_input(
            "Highlight ExpR min",
            value=0.5,
            step=0.1,
            key="exit_hl_er",
        )
    )

    rank_metric = st.selectbox(
        "Rank metric",
        METRICS,
        index=METRICS.index(
            "ExpectancyR"
        ),
        key="exit_rank_metric",
    )

    st.divider()

    st.header(
        "Combinations"
    )

    with st.expander(
        "Stop / target values"
    ):

        stop_text = st.text_area(
            "Stops (%)",
            ", ".join(
                map(
                    str,
                    DEFAULT_STOPS,
                )
            ),
            help=(
                "Use -1 for no stop (NS)."
            ),
            key="exit_stop_text",
        )

        target_text = st.text_area(
            "Targets (%)",
            ", ".join(
                map(
                    str,
                    DEFAULT_TARGETS,
                )
            ),
            help=(
                "Use -1 for no target (NT)."
            ),
            key="exit_target_text",
        )

    st.divider()

    # IMPORTANT:
    # Do not give this button an exit_* key.
    # suite.py saves and restores all exit_* state.
    # Button trigger states must not be restored.
    run_simulation = st.button(
        "Run simulation",
        type="primary",
        width="stretch",
    )


# ============================================================
# Synchronize ticker with suite
# ============================================================

if IN_SUITE:

    st.session_state[
        "_suite_shared_ticker"
    ] = ticker


# ============================================================
# Validate inputs
# ============================================================

if not ticker:

    st.warning(
        "Enter a ticker symbol."
    )

    st.stop()


if start_date >= end_date:

    st.warning(
        "The start date must be earlier than the end date."
    )

    st.stop()


# ============================================================
# Run simulation only when requested
# ============================================================

if run_simulation:

    try:

        stops = parse_values(
            stop_text
        )

        targets = parse_values(
            target_text
        )

        if not stops or not targets:

            raise ValueError(
                "Provide at least one stop "
                "and one target value."
            )

        st.session_state[
            "_exit_output"
        ] = run_simulation_now(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            number_of_entries=number_of_entries,
            seed=seed,
            commission=commission,
            slippage=slippage,
            conservative=conservative,
            stops=stops,
            targets=targets,
        )

        st.caption(
            "Showing this session's latest simulation. "
            "No user result file was created."
        )

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()


# ============================================================
# Initial startup result
# ============================================================

elif "_exit_output" not in st.session_state:

    startup_output = (
        load_startup_result()
    )

    if startup_output is not None:

        st.session_state[
            "_exit_output"
        ] = startup_output

        st.caption(
            "Showing the precomputed startup result."
        )

    else:

        try:

            initial_stops = parse_values(
                stop_text
            )

            initial_targets = parse_values(
                target_text
            )

            st.info(
                "No bundled startup result was found. "
                "Running the initial display simulation "
                "for this session."
            )

            # Do not write a file here.
            # This avoids public users creating files
            # on Streamlit Community Cloud.
            st.session_state[
                "_exit_output"
            ] = run_simulation_now(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                number_of_entries=number_of_entries,
                seed=seed,
                commission=commission,
                slippage=slippage,
                conservative=conservative,
                stops=initial_stops,
                targets=initial_targets,
            )

        except Exception as exc:

            st.error(
                str(exc)
            )

            st.stop()

else:

    st.caption(
        "Change parameters as needed, then click "
        "Run simulation. The displayed results remain "
        "unchanged until you run again."
    )


# ============================================================
# Display results
# ============================================================

if "_exit_output" in st.session_state:

    (
        data,
        entries,
        results,
        trades,
        result_stops,
        result_targets,
        elapsed,
        result_ticker,
        result_start_date,
        result_end_date,
        result_number_of_entries,
    ) = st.session_state[
        "_exit_output"
    ]


    # ========================================================
    # Summary
    # ========================================================

    summary_columns = st.columns(4)

    summary_columns[0].metric(
        "Bars",
        f"{len(data):,}",
    )

    summary_columns[1].metric(
        "Random entries",
        f"{len(entries):,}",
    )

    summary_columns[2].metric(
        "Combinations",
        f"{len(results):,}",
    )

    summary_columns[3].metric(
        "Run time",
        f"{elapsed:.1f}s",
    )


    # ========================================================
    # Heatmap
    # ========================================================

    st.subheader(
        "Heatmap"
    )

    heatmap_figure = create_heatmap(
        results=results,
        stops=result_stops,
        targets=result_targets,
        ticker=result_ticker,
        start_date=result_start_date,
        end_date=result_end_date,
        number_of_entries=len(entries),
        top_metric=heatmap_top_metric,
        bottom_metric=heatmap_bottom_metric,
        color_metric=heatmap_color_metric,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
        win_rate_minimum=highlight_win_rate,
        expectancy_r_minimum=
            highlight_expectancy_r,
    )

    if heatmap_figure is not None:

        st.pyplot(
            heatmap_figure,
            width="stretch",
        )

        plt.close(
            heatmap_figure
        )


    # ========================================================
    # Ranked combinations
    # ========================================================

    ranked = results.copy()

    if "Rank" in ranked.columns:
        ranked = ranked.drop(
            columns="Rank"
        )

    ranked = (
        ranked
        .sort_values(
            rank_metric,
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    ranked.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranked) + 1,
        ),
    )

    display_columns = [
        "Rank",
        "StopLossPct",
        "TargetPct",
        "NumberOfTrades",
        "ExpectancyR",
        "ExpectancyPct",
        "WinRatePct",
        "AverageWinPct",
        "AverageLossPct",
        "RealizedRRR",
        "StopExitPct",
        "TargetExitPct",
        "EndOfDataExitPct",
        "Score",
    ]

    with st.expander(
        f"Ranked combinations "
        f"(by {rank_metric})",
        expanded=False,
    ):

        st.dataframe(
            ranked[
                display_columns
            ].round(2),
            hide_index=True,
            width="stretch",
            height=600,
        )


    # ========================================================
    # Random entry bars
    # ========================================================

    with st.expander(
        "Random entry bars"
    ):

        st.dataframe(
            entries,
            hide_index=True,
            width="stretch",
        )


    # ========================================================
    # Trade details
    # ========================================================

    with st.expander(
        "Trade details"
    ):

        st.dataframe(
            trades.round(4),
            hide_index=True,
            width="stretch",
            height=600,
        )


    # ========================================================
    # Downloads
    # ========================================================

    st.subheader(
        "Download"
    )

    download_columns = st.columns(3)

    download_columns[0].download_button(
        "Results CSV",
        ranked
        .round(4)
        .to_csv(
            index=False
        )
        .encode(),
        (
            f"{result_ticker.lower()}"
            "_stop_target_results.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    download_columns[1].download_button(
        "Trade details CSV",
        trades
        .round(4)
        .to_csv(
            index=False
        )
        .encode(),
        (
            f"{result_ticker.lower()}"
            "_trade_details.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    download_columns[2].download_button(
        "Random entries CSV",
        entries
        .to_csv(
            index=False
        )
        .encode(),
        (
            f"{result_ticker.lower()}"
            "_random_entries.csv"
        ),
        mime="text/csv",
        width="stretch",
    )