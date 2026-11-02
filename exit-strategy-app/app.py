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

IN_SUITE = st.session_state.get(
    "_market_analysis_suite",
    False,
)

STARTUP_TICKER = "QQQ"

STARTUP_RESULT_FILE = (
    Path(__file__).parent
    / "startup_result.pkl"
)

if not IN_SUITE:
    st.set_page_config(
        page_title="Exit Strategy",
        page_icon="📊",
        layout="wide",
    )


DEFAULT_STOPS = [
    1, 2, 3, 4, 5, 7.5, 10, 12.5, 15,
    20, 25, 30, 35, 40, 45, 50, -1,
]

DEFAULT_TARGETS = [
    1, 2, 3, 4, 5, 7.5, 10, 12.5, 15,
    20, 25, 30, 35, 40, 45, 50, 100, -1,
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
    "MovingAverageExitPct",
    "EndOfDataExitPct",
    "AverageCommissionPct",
    "Score",
]


# ============================================================
# Exit Strategy preference persistence
# ============================================================

def persist_exit_preferences() -> None:
    """
    Save current exit_* widget values before the parent suite
    reruns.

    Do not assign values back to widget-owned keys here.
    Only read them and copy them to suite-level state.
    """

    if not st.session_state.get(
        "_market_analysis_suite",
        False,
    ):
        return

    preferences = dict(
        st.session_state.get(
            "_suite_saved_preferences",
            {},
        )
    )

    # --------------------------------------------------------
    # Synchronize ticker
    #
    # Read exit_ticker but do not assign back to exit_ticker.
    # --------------------------------------------------------

    if "exit_ticker" in st.session_state:

        ticker_value = (
            str(
                st.session_state[
                    "exit_ticker"
                ]
            )
            .strip()
            .upper()
        )

        if ticker_value:

            st.session_state[
                "_suite_shared_ticker"
            ] = ticker_value

    # --------------------------------------------------------
    # Synchronize ticker-search state
    # --------------------------------------------------------

    if (
        "exit_search_ticker"
        in st.session_state
    ):

        st.session_state[
            "_suite_ticker_search"
        ] = st.session_state[
            "exit_search_ticker"
        ]

    if (
        "exit_search_ticker_label"
        in st.session_state
    ):

        st.session_state[
            "_suite_ticker_search_label"
        ] = st.session_state[
            "exit_search_ticker_label"
        ]

    # --------------------------------------------------------
    # Save every persistent Exit Strategy widget
    # --------------------------------------------------------

    for key in list(
        st.session_state.keys()
    ):

        if key.startswith("exit_"):

            preferences[key] = (
                st.session_state[key]
            )

    st.session_state[
        "_suite_saved_preferences"
    ] = preferences


# ============================================================
# Ticker selection callback
# ============================================================

def select_exit_ticker(
    symbol: str,
) -> None:
    """
    Called by a ticker-search result button.

    This callback runs before widget creation, so assigning
    exit_ticker here is permitted by Streamlit.
    """

    symbol = (
        symbol
        .strip()
        .upper()
    )

    st.session_state[
        "exit_ticker"
    ] = symbol

    st.session_state[
        "exit_search_ticker"
    ] = ""

    st.session_state[
        "exit_search_ticker_label"
    ] = ""

    if IN_SUITE:

        st.session_state[
            "_suite_shared_ticker"
        ] = symbol

        st.session_state[
            "_suite_ticker_search"
        ] = ""

        st.session_state[
            "_suite_ticker_search_label"
        ] = ""

    persist_exit_preferences()


# ============================================================
# Ticker search
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def search_tickers(
    query: str,
) -> list[dict[str, str]]:

    quotes = yf.Search(
        query,
        max_results=8,
        news_count=0,
        lists_count=0,
        include_cb=False,
    ).quotes

    matches = []

    for quote in quotes:

        symbol = quote.get(
            "symbol"
        )

        if not symbol:
            continue

        matches.append(
            {
                "symbol": str(
                    symbol
                ),

                "name": str(
                    quote.get(
                        "longname"
                    )
                    or quote.get(
                        "shortname"
                    )
                    or symbol
                ),

                "exchange": str(
                    quote.get(
                        "exchDisp"
                    )
                    or quote.get(
                        "exchange"
                    )
                    or ""
                ),
            }
        )

    return matches


# ============================================================
# Historical data
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def download_data(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> pd.DataFrame:

    yahoo_end_date = (
        end_date
        + timedelta(days=1)
    )

    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=yahoo_end_date.isoformat(),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:

        raise ValueError(
            "No data was downloaded. Check the ticker, "
            "date range, and interval."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        if ticker in data.columns.get_level_values(
            -1
        ):

            data = data.xs(
                ticker,
                axis=1,
                level=-1,
            )

        else:

            data.columns = (
                data.columns
                .get_level_values(0)
            )

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = (
        data[required_columns]
        .dropna()
        .sort_index()
    )

    if len(data) < 2:

        raise ValueError(
            "At least two valid price bars are required."
        )

    if (
        data[required_columns]
        <= 0
    ).any().any():

        raise ValueError(
            "Price data contains zero or negative values."
        )

    return data


# ============================================================
# Random entries
# ============================================================

def select_random_entries(
    data: pd.DataFrame,
    number_of_entries: int,
    seed: int,
    sma_period: int,
    trade_direction: str,
    ma_enabled: bool = True,
) -> pd.DataFrame:

    maximum_entries = (
        len(data) - 1
    )

    if maximum_entries < 1:

        raise ValueError(
            "Not enough historical data to select entries."
        )

    if sma_period < 1:

        raise ValueError(
            "The moving-average period must be at least 1."
        )

    if trade_direction not in {
        "Long",
        "Short",
        "Both",
    }:

        raise ValueError(
            "Trade direction must be Long, Short, or Both."
        )

    if ma_enabled:

        moving_average_values = (
            data["Close"]
            .rolling(
                window=int(sma_period),
                min_periods=int(sma_period),
            )
            .mean()
            .to_numpy()
        )

    else:

        moving_average_values = np.full(
            len(data),
            np.nan,
        )

    close_prices = (
        data["Close"]
        .to_numpy()
    )

    eligible_indexes = np.arange(
        maximum_entries
    )

    above_average = (
        close_prices
        > moving_average_values
    )

    below_average = (
        close_prices
        < moving_average_values
    )

    if ma_enabled and trade_direction == "Long":

        eligible_indexes = eligible_indexes[
            above_average[
                eligible_indexes
            ]
        ]

    elif ma_enabled and trade_direction == "Short":

        eligible_indexes = eligible_indexes[
            below_average[
                eligible_indexes
            ]
        ]

    elif ma_enabled:

        eligible_indexes = eligible_indexes[
            above_average[
                eligible_indexes
            ]
            | below_average[
                eligible_indexes
            ]
        ]

    if not len(eligible_indexes):

        raise ValueError(
            "No eligible random entries match the selected "
            "moving-average period and trade direction."
        )

    actual_entries = min(
        int(number_of_entries),
        len(eligible_indexes),
    )

    rng = np.random.default_rng(
        int(seed)
    )

    entry_indexes = np.sort(
        rng.choice(
            eligible_indexes,
            size=actual_entries,
            replace=False,
        )
    )

    entry_directions = np.where(
        above_average[
            entry_indexes
        ],
        "Long",
        "Short",
    )

    if not ma_enabled:

        if trade_direction == "Long":

            entry_directions = np.full(
                len(entry_indexes),
                "Long",
            )

        elif trade_direction == "Short":

            entry_directions = np.full(
                len(entry_indexes),
                "Short",
            )

        else:

            entry_directions = rng.choice(
                [
                    "Long",
                    "Short",
                ],
                size=len(entry_indexes),
            )

    return pd.DataFrame(
        {
            "EntryIndex":
                entry_indexes,

            "EntryDate":
                data.index[
                    entry_indexes
                ],

            "EntryClose":
                data[
                    "Close"
                ].to_numpy()[
                    entry_indexes
                ],

            "EntrySMA":
                moving_average_values[
                    entry_indexes
                ],

            "Direction":
                entry_directions,
        }
    )


# ============================================================
# Individual trade simulation
# ============================================================

def simulate_trade(
    data: pd.DataFrame,
    entry_index: int,
    stop_loss_pct: float,
    target_pct: float,
    commission_pct: float,
    slippage_pct: float,
    conservative: bool,
    direction: str = "Long",
    sma_values: np.ndarray | None = None,
) -> dict:

    if direction not in {
        "Long",
        "Short",
    }:

        raise ValueError(
            "Trade direction must be Long or Short."
        )

    direction_sign = (
        1
        if direction == "Long"
        else -1
    )

    close_prices = (
        data["Close"]
        .to_numpy()
    )

    high_prices = (
        data["High"]
        .to_numpy()
    )

    low_prices = (
        data["Low"]
        .to_numpy()
    )

    dates = data.index

    entry_reference = float(
        close_prices[
            entry_index
        ]
    )

    no_stop = (
        stop_loss_pct == -1
    )

    no_target = (
        target_pct == -1
    )

    entry_execution = (
        entry_reference
        * (
            1
            + direction_sign
            * float(
                slippage_pct
            )
            / 100
        )
    )

    stop_price = (
        None
        if no_stop
        else entry_reference
        * (
            1
            - direction_sign
            * float(
                stop_loss_pct
            )
            / 100
        )
    )

    target_price = (
        None
        if no_target
        else entry_reference
        * (
            1
            + direction_sign
            * float(
                target_pct
            )
            / 100
        )
    )

    exit_index = (
        len(data) - 1
    )

    exit_reference = float(
        close_prices[
            exit_index
        ]
    )

    exit_reason = (
        "EndOfData"
    )

    for current_index in range(
        entry_index + 1,
        len(data),
    ):

        stop_hit = (
            stop_price is not None
            and (
                float(
                    low_prices[
                        current_index
                    ]
                ) <= stop_price
                if direction == "Long"
                else float(
                    high_prices[
                        current_index
                    ]
                ) >= stop_price
            )
        )

        target_hit = (
            target_price is not None
            and (
                float(
                    high_prices[
                        current_index
                    ]
                ) >= target_price
                if direction == "Long"
                else float(
                    low_prices[
                        current_index
                    ]
                ) <= target_price
            )
        )

        if stop_hit and target_hit:

            if conservative:

                exit_reference = (
                    stop_price
                )

                exit_reason = (
                    "Stop"
                )

            else:

                exit_reference = (
                    target_price
                )

                exit_reason = (
                    "Target"
                )

            exit_index = (
                current_index
            )

            break

        if stop_hit:

            exit_reference = (
                stop_price
            )

            exit_reason = (
                "Stop"
            )

            exit_index = (
                current_index
            )

            break

        if target_hit:

            exit_reference = (
                target_price
            )

            exit_reason = (
                "Target"
            )

            exit_index = (
                current_index
            )

            break

        if (
            sma_values is not None
            and np.isfinite(
                sma_values[
                    current_index - 1
                ]
            )
            and np.isfinite(
                sma_values[
                    current_index
                ]
            )
        ):

            previous_close = float(
                close_prices[
                    current_index - 1
                ]
            )

            current_close = float(
                close_prices[
                    current_index
                ]
            )

            previous_sma = float(
                sma_values[
                    current_index - 1
                ]
            )

            current_sma = float(
                sma_values[
                    current_index
                ]
            )

            crossed_against_position = (
                previous_close >= previous_sma
                and current_close < current_sma
                if direction == "Long"
                else previous_close <= previous_sma
                and current_close > current_sma
            )

            if crossed_against_position:

                exit_reference = (
                    current_close
                )

                exit_reason = (
                    "MovingAverage"
                )

                exit_index = (
                    current_index
                )

                break

    exit_execution = (
        exit_reference
        * (
            1
            - direction_sign
            * float(
                slippage_pct
            )
            / 100
        )
    )

    gross_return_pct = (
        direction_sign
        * (
            exit_execution
            / entry_execution
            - 1
        )
        * 100
    )

    total_commission_pct = (
        2
        * float(
            commission_pct
        )
    )

    net_return_pct = (
        gross_return_pct
        - total_commission_pct
    )

    risk_pct = (
        100
        if no_stop
        else float(
            stop_loss_pct
        )
    )

    return {
        "EntryDate":
            dates[
                entry_index
            ],

        "Direction":
            direction,

        "ExitDate":
            dates[
                exit_index
            ],

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
            gross_return_pct,

        "NetReturnPct":
            net_return_pct,

        "RealizedR":
            net_return_pct
            / risk_pct,

        "BarsHeld":
            exit_index
            - entry_index,

        "ExitReason":
            exit_reason,

        "CommissionPct":
            total_commission_pct,
    }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    trades: pd.DataFrame,
    stop: float,
    target: float,
) -> dict:

    returns = (
        trades[
            "NetReturnPct"
        ]
        .to_numpy(float)
    )

    number_of_trades = (
        len(returns)
    )

    winning_returns = (
        returns[
            returns > 0
        ]
    )

    losing_returns = (
        returns[
            returns < 0
        ]
    )

    win_rate = (
        float(
            (
                returns > 0
            ).mean()
            * 100
        )
        if number_of_trades
        else np.nan
    )

    average_win = (
        float(
            winning_returns.mean()
        )
        if len(
            winning_returns
        )
        else np.nan
    )

    average_loss = (
        float(
            losing_returns.mean()
        )
        if len(
            losing_returns
        )
        else np.nan
    )

    expectancy = (
        float(
            returns.mean()
        )
        if number_of_trades
        else np.nan
    )

    risk = (
        100
        if stop == -1
        else float(
            stop
        )
    )

    expectancy_r = (
        expectancy / risk
    )

    planned_rrr = (
        np.nan
        if target == -1
        else float(
            target
        )
        / risk
    )

    realized_rrr = (
        average_win
        / abs(
            average_loss
        )
        if (
            len(
                winning_returns
            )
            and len(
                losing_returns
            )
            and average_loss != 0
        )
        else np.nan
    )

    break_even_win_rate = (
        abs(
            average_loss
        )
        / (
            average_win
            + abs(
                average_loss
            )
        )
        * 100
        if (
            np.isfinite(
                average_win
            )
            and np.isfinite(
                average_loss
            )
            and average_win > 0
            and average_loss < 0
        )
        else np.nan
    )

    planned_break_even = (
        1
        / (
            1
            + planned_rrr
        )
        * 100
        if (
            np.isfinite(
                planned_rrr
            )
            and planned_rrr > 0
        )
        else np.nan
    )

    exit_counts = (
        trades[
            "ExitReason"
        ]
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
            planned_break_even,

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

        "MovingAverageExitPct":
            exit_counts.get(
                "MovingAverage",
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


def calculate_directional_statistics(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    if "Direction" not in trades.columns:

        raise ValueError(
            "Trade direction data is required to calculate "
            "directional statistics."
        )

    statistics = []

    for (
        stop,
        target,
    ), combination_trades in trades.groupby(
        [
            "StopLossPct",
            "TargetPct",
        ],
        sort=False,
    ):

        for direction in [
            "All",
            "Long",
            "Short",
        ]:

            direction_trades = (
                combination_trades
                if direction == "All"
                else combination_trades.loc[
                    combination_trades[
                        "Direction"
                    ]
                    == direction
                ]
            )

            metrics = calculate_metrics(
                direction_trades,
                float(stop),
                float(target),
            )

            metrics[
                "Direction"
            ] = direction

            statistics.append(
                metrics
            )

    return pd.DataFrame(
        statistics
    )


# ============================================================
# Compare combinations
# ============================================================

def compare_combinations(
    data: pd.DataFrame,
    entries: pd.DataFrame,
    stops: list[float],
    targets: list[float],
    commission: float,
    slippage: float,
    conservative: bool,
    sma_period: int,
    ma_exit_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    result_records = []
    trade_records = []

    sma_values = (
        data["Close"]
        .rolling(
            window=int(sma_period),
            min_periods=int(sma_period),
        )
        .mean()
        .to_numpy()
        if ma_exit_enabled
        else None
    )

    total_combinations = (
        len(stops)
        * len(targets)
    )

    completed = 0

    progress_bar = st.progress(
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

                    stop_loss_pct=
                        stop,

                    target_pct=
                        target,

                    commission_pct=
                        commission,

                    slippage_pct=
                        slippage,

                    conservative=
                        conservative,

                    direction=
                        str(
                            entry.Direction
                        ),

                    sma_values=
                        sma_values,
                )

                for entry
                in entries.itertuples(
                    index=False
                )
            ]

            combination_frame = (
                pd.DataFrame(
                    combination_trades
                )
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

            completed += 1

            progress_bar.progress(
                completed
                / total_combinations,

                text=(
                    "Testing combinations: "
                    f"{completed}"
                    f"/{total_combinations}"
                ),
            )

    progress_bar.empty()

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
            results[
                metric
            ].std()
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
                results[
                    metric
                ]
                - results[
                    metric
                ].mean()
            ) / standard_deviation

        else:

            results[
                f"z_{metric}"
            ] = 0.0

    results[
        "ScoreRaw"
    ] = (
        2
        * results[
            "z_ExpectancyR"
        ]
        + results[
            "z_WinRatePct"
        ]
    )

    minimum_score = (
        results[
            "ScoreRaw"
        ].min()
    )

    maximum_score = (
        results[
            "ScoreRaw"
        ].max()
    )

    if (
        maximum_score
        != minimum_score
    ):

        results[
            "Score"
        ] = (
            100
            * (
                results[
                    "ScoreRaw"
                ]
                - minimum_score
            )
            / (
                maximum_score
                - minimum_score
            )
        )

    else:

        results[
            "Score"
        ] = 50.0

    results = (
        results
        .sort_values(
            "ExpectancyR",
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

    return (
        results,
        trades,
    )


# ============================================================
# Heatmap
# ============================================================

def create_heatmap(
    results: pd.DataFrame,
    stops: list[float],
    targets: list[float],
    ticker: str,
    start_date: date,
    end_date: date,
    number_of_entries: int,
    top_metric: str,
    bottom_metric: str,
    color_metric: str,
    vmin,
    vmax,
    win_rate_minimum: float,
    expectancy_r_minimum: float,
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

    matrix = (
        color_values
        .to_numpy(
            float
        )
    )

    finite_values = (
        matrix[
            np.isfinite(
                matrix
            )
        ]
    )

    if not finite_values.size:

        return None

    if vmin is None:

        vmin = float(
            np.min(
                finite_values
            )
        )

    if vmax is None:

        vmax = float(
            np.max(
                finite_values
            )
        )

    if vmin == vmax:

        padding = max(
            abs(vmin)
            * 0.01,
            1e-9,
        )

        vmin -= padding
        vmax += padding

    figure, axis = (
        plt.subplots(
            figsize=(
                max(
                    12,
                    len(stops)
                    * 0.85,
                ),

                max(
                    8,
                    len(targets)
                    * 0.55,
                ),
            )
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

            for value
            in stops
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

            for value
            in targets
        ]
    )

    formats = {
        "ExpectancyR":
            ".2f",

        "ExpectancyPct":
            ".2f",

        "WinRatePct":
            ".1f",

        "RealizedRRR":
            ".2f",

        "Score":
            ".0f",
    }

    top_format = (
        formats.get(
            top_metric,
            ".2f",
        )
    )

    bottom_format = (
        formats.get(
            bottom_metric,
            ".2f",
        )
    )

    for row in range(
        len(targets)
    ):

        for column in range(
            len(stops)
        ):

            top_value = (
                top_values.iloc[
                    row,
                    column,
                ]
            )

            bottom_value = (
                bottom_values.iloc[
                    row,
                    column,
                ]
            )

            color_value = (
                matrix[
                    row,
                    column,
                ]
            )

            if (
                np.isfinite(
                    top_value
                )
                and
                np.isfinite(
                    bottom_value
                )
            ):

                label = (
                    f"{format(top_value, top_format)}\n"
                    f"{format(bottom_value, bottom_format)}"
                )

            else:

                label = "N/A"

            if np.isfinite(
                color_value
            ):

                rgba = (
                    image.cmap(
                        image.norm(
                            color_value
                        )
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
                0.299
                * rgba[0]

                + 0.587
                * rgba[1]

                + 0.114
                * rgba[2]
            )

            axis.text(
                column,
                row,
                label,

                ha="center",
                va="center",

                color=(
                    "black"
                    if brightness
                    > 0.55
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

                        edgecolor=
                            "black",

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
# Helpers
# ============================================================

def parse_values(
    text: str,
) -> list[float]:

    return [
        float(
            value.strip()
        )

        for value
        in text.split(",")

        if value.strip()
    ]


def result_ticker(
    output,
):

    if output is None:

        return None

    try:

        return (
            str(
                output[7]
            )
            .strip()
            .upper()
        )

    except (
        IndexError,
        TypeError,
    ):

        return None


def result_has_directional_entries(
    output,
) -> bool:

    if output is None:

        return False

    try:

        entries = output[1]
        results = output[2]
        trades = output[3]

    except (
        IndexError,
        TypeError,
    ):

        return False

    return (
        hasattr(
            entries,
            "columns",
        )
        and {
            "EntrySMA",
            "Direction",
        }.issubset(
            entries.columns
        )
        and hasattr(
            results,
            "columns",
        )
        and "MovingAverageExitPct" in results.columns
        and hasattr(
            trades,
            "columns",
        )
        and "Direction" in trades.columns
    )


def upgrade_startup_result(
    output,
) -> tuple:

    (
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
        number_of_entries,
    ) = output

    entries = entries.copy()
    entries["EntrySMA"] = np.nan
    entries["Direction"] = "Long"

    results = results.copy()
    results["MovingAverageExitPct"] = 0.0

    trades = trades.copy()
    trades["Direction"] = "Long"

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
        number_of_entries,
    )


def startup_result_matches_settings(
    output,
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str,
    number_of_entries: int,
    seed: int,
    commission: float,
    slippage: float,
    conservative: bool,
    stop_text: str,
    target_text: str,
    trade_direction: str,
    ma_enabled: bool,
    ma_exit_enabled: bool,
) -> bool:

    if output is None:

        return False

    try:

        result_start_date = output[8]
        result_end_date = output[9]
        result_entry_count = output[10]
        result_stops = output[4]
        result_targets = output[5]

    except (
        IndexError,
        TypeError,
    ):

        return False

    if (
        ticker.strip().upper() != STARTUP_TICKER
        or result_ticker(output) != STARTUP_TICKER
        or start_date != result_start_date
        or end_date != result_end_date
        or interval != "1d"
        or int(number_of_entries) != result_entry_count
        or int(seed) != 27
        or float(commission) != 0
        or float(slippage) != 0
        or not conservative
        or trade_direction != "Long"
        or ma_enabled
        or ma_exit_enabled
    ):

        return False

    try:

        selected_stops = parse_values(
            stop_text
        )

        selected_targets = parse_values(
            target_text
        )

    except ValueError:

        return False

    if (
        selected_stops != result_stops
        or selected_targets != result_targets
    ):

        return False

    expected_entries = select_random_entries(
        output[0],
        result_entry_count,
        27,
        1,
        "Long",
        False,
    )

    return np.array_equal(
        expected_entries["EntryIndex"].to_numpy(),
        output[1]["EntryIndex"].to_numpy(),
    )


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
    sma_period,
    trade_direction,
    ma_enabled,
    ma_exit_enabled,
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

    entries = (
        select_random_entries(
            data,
            number_of_entries,
            seed,
            sma_period,
            trade_direction,
            ma_enabled,
        )
    )

    start_time = (
        time.perf_counter()
    )

    results, trades = (
        compare_combinations(
            data=data,

            entries=
                entries,

            stops=
                stops,

            targets=
                targets,

            commission=
                float(
                    commission
                ),

            slippage=
                float(
                    slippage
                ),

            conservative=
                conservative,

            sma_period=
                sma_period,

            ma_exit_enabled=
                ma_exit_enabled,
        )
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
        int(
            number_of_entries
        ),
    )


def load_startup_result():

    if not (
        STARTUP_RESULT_FILE
        .exists()
    ):

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
            "The startup result "
            "could not be loaded: "
            f"{exc}"
        )

        return None


# ============================================================
# Heading
# ============================================================

st.title(
    "Exit Strategy"
)

st.write(
    "Test stop-loss and target combinations "
    "using random historical entries in the direction "
    "of the selected simple moving average."
)


# ============================================================
# Initialize widget state BEFORE widget creation
# ============================================================

if "exit_ticker" not in st.session_state:

    st.session_state[
        "exit_ticker"
    ] = st.session_state.get(
        "_suite_shared_ticker",
        STARTUP_TICKER,
    )


if (
    "exit_search_ticker"
    not in st.session_state
):

    st.session_state[
        "exit_search_ticker"
    ] = ""


if (
    "exit_search_ticker_label"
    not in st.session_state
):

    st.session_state[
        "exit_search_ticker_label"
    ] = ""


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "Data"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    ticker_query = (
        st.text_input(
            "Search for ticker",

            placeholder=(
                "Type a ticker "
                "or company name"
            ),

            key=
                "exit_search_ticker",

            live=
                "500ms",

            on_change=
                persist_exit_preferences,
        )
        .strip()
    )


    selected_label = (
        st.session_state.get(
            "exit_search_ticker_label",
            "",
        )
    )


    if (
        ticker_query
        != selected_label

        and

        len(
            ticker_query
        ) >= 2
    ):

        search_failed = False

        try:

            with st.spinner(
                "Searching Yahoo Finance..."
            ):

                ticker_matches = (
                    search_tickers(
                        ticker_query
                    )
                )

        except Exception as exc:

            search_failed = True

            ticker_matches = []

            st.warning(
                "Yahoo Finance ticker "
                "search failed: "
                f"{exc}. "
                "You can still enter "
                "a ticker manually."
            )


        if ticker_matches:

            for match in ticker_matches:

                symbol = (
                    match[
                        "symbol"
                    ]
                )

                label = (
                    f"{symbol} | "
                    f"{match['name']} | "
                    f"{match['exchange'] or 'Unknown exchange'}"
                )

                st.button(
                    label,

                    key=(
                        "_exit_ticker_result_"
                        f"{symbol}"
                    ),

                    on_click=
                        select_exit_ticker,

                    args=(
                        symbol,
                    ),

                    width=
                        "stretch",
                )

        elif not search_failed:

            st.caption(
                "No Yahoo Finance matches found."
            )


    elif (
        ticker_query
        != selected_label

        and

        ticker_query
    ):

        st.caption(
            "Enter at least two "
            "characters to search."
        )


    # --------------------------------------------------------
    # Ticker
    #
    # No value= argument because the state key was initialized
    # before widget creation.
    # --------------------------------------------------------

    ticker = (
        st.text_input(
            "Ticker",

            help=(
                "Examples: AAPL, MSFT, SPY, "
                "QQQ, BTC-USD, or ^GSPC."
            ),

            key=
                "exit_ticker",

            on_change=
                persist_exit_preferences,
        )
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # Simulation settings
    # --------------------------------------------------------

    interval = (
        st.selectbox(
            "Interval",

            [
                "1d",
                "1wk",
                "1mo",
            ],

            index=0,

            key=
                "exit_interval",

            on_change=
                persist_exit_preferences,
        )
    )


    st.subheader(
        "Date range"
    )


    start_date = (
        st.date_input(
            "Start date",

            value=date(
                2010,
                1,
                1,
            ),

            key=
                "exit_start_date",

            on_change=
                persist_exit_preferences,
        )
    )


    end_date = (
        st.date_input(
            "End date",

            value=date(
                2026,
                10,
                30,
            ),

            key=
                "exit_end_date",

            on_change=
                persist_exit_preferences,
        )
    )


    st.divider()


    st.header(
        "Simulation"
    )


    sma_period = (
        st.number_input(
            "SMA period (bars)",

            min_value=1,

            value=200,

            step=1,

            help=(
                "Random entries are long when the close is "
                "above this simple moving average and short "
                "when it is below."
            ),

            key=
                "exit_sma_period",

            on_change=
                persist_exit_preferences,
        )
    )

    ma_enabled = (
        st.checkbox(
            "Use SMA for entry direction",

            value=False,

            help=(
                "When enabled, entries are long above the SMA "
                "and short below it."
            ),

            key=
                "exit_ma_enabled",

            on_change=
                persist_exit_preferences,
        )
    )

    ma_exit_enabled = (
        st.checkbox(
            "Exit on opposite SMA crossover",

            value=False,

            help=(
                "Close an open trade when price crosses the "
                "SMA against its direction. Stop-loss and "
                "target exits remain active and take priority."
            ),

            key=
                "exit_ma_exit_enabled",

            on_change=
                persist_exit_preferences,
        )
    )


    trade_direction = (
        st.selectbox(
            "Trade direction",

            [
                "Both",
                "Long",
                "Short",
            ],

            index=1,

            help=(
                "When SMA logic is on, entries follow the SMA. "
                "When it is off, Both randomly assigns long "
                "or short to each entry."
            ),

            key=
                "exit_trade_direction",

            on_change=
                persist_exit_preferences,
        )
    )


    number_of_entries = (
        st.number_input(
            "Random entries",

            min_value=1,

            value=1000,

            step=100,

            key=
                "exit_number_of_entries",

            on_change=
                persist_exit_preferences,
        )
    )


    seed = (
        st.number_input(
            "Random seed",

            min_value=0,

            value=27,

            step=1,

            key=
                "exit_seed",

            on_change=
                persist_exit_preferences,
        )
    )


    commission = (
        st.number_input(
            "Commission (%)",

            min_value=0.0,

            value=0.0,

            step=0.01,

            format=
                "%.3f",

            key=
                "exit_commission",

            on_change=
                persist_exit_preferences,
        )
    )


    slippage = (
        st.number_input(
            "Slippage (%)",

            min_value=0.0,

            value=0.0,

            step=0.01,

            format=
                "%.3f",

            key=
                "exit_slippage",

            on_change=
                persist_exit_preferences,
        )
    )


    conservative = (
        st.checkbox(
            "Stop first when stop and "
            "target hit same candle",

            value=True,

            key=
                "exit_conservative",

            on_change=
                persist_exit_preferences,
        )
    )


    st.divider()


    # --------------------------------------------------------
    # Heatmap controls
    # --------------------------------------------------------

    st.header(
        "Heatmap"
    )


    top_metric = (
        st.selectbox(
            "Top metric",

            METRICS,

            index=
                METRICS.index(
                    "ExpectancyR"
                ),

            key=
                "exit_top_metric",

            on_change=
                persist_exit_preferences,
        )
    )


    bottom_metric = (
        st.selectbox(
            "Bottom metric",

            METRICS,

            index=
                METRICS.index(
                    "WinRatePct"
                ),

            key=
                "exit_bottom_metric",

            on_change=
                persist_exit_preferences,
        )
    )


    color_metric = (
        st.selectbox(
            "Color metric",

            METRICS,

            index=
                METRICS.index(
                    "ExpectancyR"
                ),

            key=
                "exit_color_metric",

            on_change=
                persist_exit_preferences,
        )
    )


    automatic_color_scale = (
        st.checkbox(
            "Auto color scale "
            "(use min/max)",

            value=False,

            key=
                "exit_auto_color_scale",

            on_change=
                persist_exit_preferences,
        )
    )


    (
        color_column_1,
        color_column_2,
    ) = st.columns(2)


    color_minimum = (
        color_column_1
        .number_input(
            "Color min",

            value=0.0,

            step=0.1,

            disabled=
                automatic_color_scale,

            key=
                "exit_color_minimum",

            on_change=
                persist_exit_preferences,
        )
    )


    color_maximum = (
        color_column_2
        .number_input(
            "Color max",

            value=1.0,

            step=0.1,

            disabled=
                automatic_color_scale,

            key=
                "exit_color_maximum",

            on_change=
                persist_exit_preferences,
        )
    )


    if automatic_color_scale:

        color_minimum = None
        color_maximum = None


    (
        highlight_column_1,
        highlight_column_2,
    ) = st.columns(2)


    highlight_win_rate = (
        highlight_column_1
        .number_input(
            "Highlight WR min",

            value=50.0,

            step=1.0,

            key=
                "exit_highlight_win_rate",

            on_change=
                persist_exit_preferences,
        )
    )


    highlight_expectancy_r = (
        highlight_column_2
        .number_input(
            "Highlight ExpR min",

            value=0.5,

            step=0.1,

            key=
                "exit_highlight_expectancy_r",

            on_change=
                persist_exit_preferences,
        )
    )


    rank_metric = (
        st.selectbox(
            "Rank metric",

            METRICS,

            index=
                METRICS.index(
                    "ExpectancyR"
                ),

            key=
                "exit_rank_metric",

            on_change=
                persist_exit_preferences,
        )
    )


    st.divider()


    # --------------------------------------------------------
    # Stop / target combinations
    # --------------------------------------------------------

    st.header(
        "Combinations"
    )


    with st.expander(
        "Stop / target values"
    ):

        stop_text = (
            st.text_area(
                "Stops (%)",

                ", ".join(
                    map(
                        str,
                        DEFAULT_STOPS,
                    )
                ),

                help=(
                    "Use -1 for "
                    "no stop (NS)."
                ),

                key=
                    "exit_stop_text",

                on_change=
                    persist_exit_preferences,
            )
        )


        target_text = (
            st.text_area(
                "Targets (%)",

                ", ".join(
                    map(
                        str,
                        DEFAULT_TARGETS,
                    )
                ),

                help=(
                    "Use -1 for "
                    "no target (NT)."
                ),

                key=
                    "exit_target_text",

                on_change=
                    persist_exit_preferences,
            )
        )


    st.divider()


    # No exit_* key because button trigger state must not be
    # saved and restored as a preference.
    run_simulation = (
        st.button(
            "Run simulation",

            type=
                "primary",

            width=
                "stretch",
        )
    )


# ============================================================
# Input validation
# ============================================================

valid_inputs = True


if not ticker:

    st.warning(
        "Enter a ticker symbol."
    )

    valid_inputs = False


elif start_date >= end_date:

    st.warning(
        "The start date must be "
        "earlier than the end date."
    )

    valid_inputs = False


# ============================================================
# Run simulation only after button click
# ============================================================

if (
    valid_inputs
    and run_simulation
):

    try:

        stops = parse_values(
            stop_text
        )

        targets = parse_values(
            target_text
        )


        if (
            not stops
            or not targets
        ):

            raise ValueError(
                "Provide at least one stop "
                "and one target value."
            )


        persist_exit_preferences()


        st.session_state[
            "_exit_output"
        ] = run_simulation_now(
            ticker=
                ticker,

            start_date=
                start_date,

            end_date=
                end_date,

            interval=
                interval,

            number_of_entries=
                number_of_entries,

            seed=
                seed,

            commission=
                commission,

            slippage=
                slippage,

            conservative=
                conservative,

            stops=
                stops,

            targets=
                targets,

            sma_period=
                sma_period,

            trade_direction=
                trade_direction,

            ma_enabled=
                ma_enabled,

            ma_exit_enabled=
                ma_exit_enabled,
        )


    except Exception as exc:

        st.error(
            str(exc)
        )


# ============================================================
# Load QQQ startup result only when current ticker is QQQ
# ============================================================

if (
    "_exit_output"
    not in st.session_state
):

    if (
        ticker
        == STARTUP_TICKER
    ):

        startup_output = (
            load_startup_result()
        )


        if (
            startup_output
            is not None

            and

            result_ticker(
                startup_output
            )
            == STARTUP_TICKER

            and

            result_has_directional_entries(
                startup_output
            )
        ):

            st.session_state[
                "_exit_output"
            ] = startup_output

        elif (
            startup_result_matches_settings(
                output=startup_output,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                number_of_entries=number_of_entries,
                seed=seed,
                commission=commission,
                slippage=slippage,
                conservative=conservative,
                stop_text=stop_text,
                target_text=target_text,
                trade_direction=trade_direction,
                ma_enabled=ma_enabled,
                ma_exit_enabled=ma_exit_enabled,
            )
        ):

            st.session_state[
                "_exit_output"
            ] = startup_output


        elif (
            startup_output
            is None
        ):

            st.info(
                "No precomputed QQQ startup result "
                "is available. Click Run simulation."
            )

        elif (
            startup_output
            is not None

            and

            not result_has_directional_entries(
                startup_output
            )
        ):

            st.info(
                "The bundled QQQ result is available for "
                "daily data from 2010-01-01 to 2026-10-30, "
                "1,000 entries with seed 27, Long direction, "
                "SMA entry and exit disabled, zero commission "
                "and slippage, and the default stop/target "
                "values. Select those settings to display it, "
                "or run a new simulation."
            )


        else:

            st.warning(
                "The bundled startup result "
                "is not for QQQ."
            )


    else:

        st.info(
            f"Ticker is {ticker}. "
            "The bundled startup results are for QQQ "
            "and are therefore hidden. "
            f"Click Run simulation to generate "
            f"results for {ticker}."
        )


# ============================================================
# Determine whether output matches current ticker
# ============================================================

output_to_display = (
    st.session_state.get(
        "_exit_output"
    )
)

if (
    output_to_display is not None

    and

    not result_has_directional_entries(
        output_to_display
    )

    and

    startup_result_matches_settings(
        output=output_to_display,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        number_of_entries=number_of_entries,
        seed=seed,
        commission=commission,
        slippage=slippage,
        conservative=conservative,
        stop_text=stop_text,
        target_text=target_text,
        trade_direction=trade_direction,
        ma_enabled=ma_enabled,
        ma_exit_enabled=ma_exit_enabled,
    )
):

    output_to_display = upgrade_startup_result(
        output_to_display
    )


output_ticker = (
    result_ticker(
        output_to_display
    )
)


show_results = (
    valid_inputs

    and

    output_to_display
    is not None

    and

    result_has_directional_entries(
        output_to_display
    )

    and

    output_ticker
    == ticker
)


if (
    valid_inputs

    and

    output_to_display
    is not None

    and

    not result_has_directional_entries(
        output_to_display
    )

):

    st.info(
        "The bundled result is for QQQ daily data from "
        "2010-01-01 to 2026-10-30, 1,000 Long entries with "
        "seed 27, SMA entry and exit disabled, zero "
        "commission/slippage, and the default stop/target "
        "values. Select those settings to display it, or "
        "click Run simulation."
    )


if (
    valid_inputs

    and

    output_to_display
    is not None

    and

    result_has_directional_entries(
        output_to_display
    )

    and

    output_ticker
    != ticker
):

    st.info(
        f"Ticker changed from "
        f"{output_ticker or 'the previous ticker'} "
        f"to {ticker}. "
        f"Click Run simulation to generate "
        f"results for {ticker}."
    )


# ============================================================
# Display results
# ============================================================

if show_results:

    (
        data,
        entries,
        results,
        trades,
        result_stops,
        result_targets,
        elapsed,
        result_ticker_value,
        result_start_date,
        result_end_date,
        result_entry_count,
    ) = output_to_display


    if (
        result_ticker_value
        .strip()
        .upper()
        == STARTUP_TICKER

        and

        not run_simulation
    ):

        st.caption(
            "Showing precomputed QQQ startup results."
        )


    else:

        st.caption(
            "Showing simulation results for "
            f"{result_ticker_value}."
        )


    # ========================================================
    # Summary
    # ========================================================

    summary_columns = (
        st.columns(4)
    )


    summary_columns[
        0
    ].metric(
        "Bars",
        f"{len(data):,}",
    )


    summary_columns[
        1
    ].metric(
        "Random entries",
        f"{len(entries):,}",
    )


    summary_columns[
        2
    ].metric(
        "Combinations",
        f"{len(results):,}",
    )


    summary_columns[
        3
    ].metric(
        "Run time",
        f"{elapsed:.1f}s",
    )


    # ========================================================
    # Heatmap
    # ========================================================

    st.subheader(
        "Heatmap"
    )


    heatmap = (
        create_heatmap(
            results=
                results,

            stops=
                result_stops,

            targets=
                result_targets,

            ticker=
                result_ticker_value,

            start_date=
                result_start_date,

            end_date=
                result_end_date,

            number_of_entries=
                len(entries),

            top_metric=
                top_metric,

            bottom_metric=
                bottom_metric,

            color_metric=
                color_metric,

            vmin=
                color_minimum,

            vmax=
                color_maximum,

            win_rate_minimum=
                highlight_win_rate,

            expectancy_r_minimum=
                highlight_expectancy_r,
        )
    )


    if heatmap is not None:

        st.pyplot(
            heatmap,
            width="stretch",
        )

        plt.close(
            heatmap
        )


    # ========================================================
    # Ranked results
    # ========================================================

    ranked = (
        results.copy()
    )


    if (
        "Rank"
        in ranked.columns
    ):

        ranked = (
            ranked.drop(
                columns=
                    "Rank"
            )
        )


    ranked = (
        ranked
        .sort_values(
            rank_metric,

            ascending=
                False,

            na_position=
                "last",
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
        "MovingAverageExitPct",
        "EndOfDataExitPct",
        "Score",
    ]


    with st.expander(
        (
            "Ranked combinations "
            f"(by {rank_metric})"
        ),

        expanded=False,
    ):

        st.dataframe(
            ranked[
                display_columns
            ].round(
                2
            ),

            hide_index=True,

            width=
                "stretch",

            height=
                600,
        )


    # ========================================================
    # Directional statistics
    # ========================================================

    directional_statistics = (
        calculate_directional_statistics(
            trades
        )
    )

    statistics_columns = [
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
        "MovingAverageExitPct",
        "EndOfDataExitPct",
    ]

    st.subheader(
        "Statistics by direction"
    )

    all_statistics, long_statistics, short_statistics = (
        st.tabs(
            [
                "All trades",
                "Long trades",
                "Short trades",
            ]
        )
    )

    for tab, direction in [
        (all_statistics, "All"),
        (long_statistics, "Long"),
        (short_statistics, "Short"),
    ]:

        with tab:

            direction_statistics = (
                directional_statistics.loc[
                    directional_statistics[
                        "Direction"
                    ]
                    == direction
                ]
                .sort_values(
                    rank_metric,
                    ascending=False,
                    na_position="last",
                )
            )

            st.dataframe(
                direction_statistics[
                    statistics_columns
                ].round(
                    2
                ),

                hide_index=True,

                width=
                    "stretch",

                height=
                    400,
            )


    # ========================================================
    # Detail tables
    # ========================================================

    with st.expander(
        "Random entry bars"
    ):

        st.dataframe(
            entries,

            hide_index=True,

            width=
                "stretch",
        )


    with st.expander(
        "Trade details"
    ):

        st.dataframe(
            trades.round(
                4
            ),

            hide_index=True,

            width=
                "stretch",

            height=
                600,
        )


    # ========================================================
    # Downloads
    # ========================================================

    st.subheader(
        "Download"
    )


    download_columns = (
        st.columns(4)
    )


    download_columns[
        0
    ].download_button(
        "Results CSV",

        ranked
        .round(4)
        .to_csv(
            index=False
        )
        .encode(),

        (
            f"{result_ticker_value.lower()}"
            "_stop_target_results.csv"
        ),

        mime=
            "text/csv",

        width=
            "stretch",
    )


    download_columns[
        1
    ].download_button(
        "Trade details CSV",

        trades
        .round(4)
        .to_csv(
            index=False
        )
        .encode(),

        (
            f"{result_ticker_value.lower()}"
            "_trade_details.csv"
        ),

        mime=
            "text/csv",

        width=
            "stretch",
    )


    download_columns[
        2
    ].download_button(
        "Directional stats CSV",

        directional_statistics
        .round(4)
        .to_csv(
            index=False
        )
        .encode(),

        (
            f"{result_ticker_value.lower()}"
            "_directional_statistics.csv"
        ),

        mime=
            "text/csv",

        width=
            "stretch",
    )


    download_columns[
        3
    ].download_button(
        "Random entries CSV",

        entries
        .to_csv(
            index=False
        )
        .encode(),

        (
            f"{result_ticker_value.lower()}"
            "_random_entries.csv"
        ),

        mime=
            "text/csv",

        width=
            "stretch",
    )