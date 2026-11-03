"""Data preparation and direction-aware trade calculations.

The public functions in this module are independent of page layout. Historical
prices are normalized to the OHLC columns used by the simulator; random entries
can be filtered by an SMA and assigned a direction; trade simulation applies
stop/target exits, optional opposite-SMA-cross exits, and execution costs.
Metric helpers summarize the resulting trade records by combination and
direction.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


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
    """Download validated OHLC history for a ticker and date range.

    The returned frame is sorted by date and contains positive Open, High, Low,
    and Close values. Yahoo Finance's end date is exclusive, so the requested
    final date is included in the download.
    """

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

def select_random_entries(
    data: pd.DataFrame,
    number_of_entries: int,
    seed: int,
    sma_period: int,
    trade_direction: str,
    ma_enabled: bool = True,
) -> pd.DataFrame:
    """Select reproducible entry bars and assign each a trade direction.

    The random baseline is selected before SMA filtering, so enabled Long and
    Short entries are subsets of the same seeded baseline used with filtering
    disabled. Long entries must close above the SMA and short entries below it;
    bars equal to or lacking an SMA are excluded. When disabled, every selected
    bar can be used and ``Both`` assigns random directions. The final data bar
    is excluded because no later exit bar exists.
    """

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

    above_average = (
        close_prices
        > moving_average_values
    )

    below_average = (
        close_prices
        < moving_average_values
    )

    if ma_enabled and trade_direction == "Long":

        entry_indexes = entry_indexes[
            above_average[
                entry_indexes
            ]
        ]

    elif ma_enabled and trade_direction == "Short":

        entry_indexes = entry_indexes[
            below_average[
                entry_indexes
            ]
        ]

    elif ma_enabled:

        entry_indexes = entry_indexes[
            above_average[
                entry_indexes
            ]
            | below_average[
                entry_indexes
            ]
        ]

    if ma_enabled and actual_entries and not len(entry_indexes):

        raise ValueError(
            "None of the selected baseline entries match the "
            "moving-average filter and trade direction. Increase "
            "the random entry count or adjust the moving average."
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
    """Simulate one long or short trade and return its outcome record.

    Stop and target levels are mirrored for short trades. When supplied,
    ``sma_values`` enables an exit after a confirmed close crosses the SMA
    against the position; stop/target exits are checked first.
    """

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

def calculate_metrics(
    trades: pd.DataFrame,
    stop: float,
    target: float,
) -> dict:
    """Calculate return, risk/reward, and exit-reason metrics for trades."""

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
    """Calculate combination metrics for all, long, and short trades."""

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

def parse_values(
    text: str,
) -> list[float]:
    """Parse comma-separated numeric stop or target values."""

    return [
        float(
            value.strip()
        )

        for value
        in text.split(",")

        if value.strip()
    ]
