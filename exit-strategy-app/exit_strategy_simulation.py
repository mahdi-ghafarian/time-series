"""Simulation orchestration for Exit Strategy runs."""

import time

import numpy as np
import pandas as pd
import streamlit as st

from exit_strategy_core import (
    calculate_metrics,
    download_data,
    select_random_entries,
    simulate_trade,
)


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
    """Simulate each stop/target pair for the supplied entry bars.

    Returns the combination metrics and the corresponding individual
    trade records. The SMA exit series is calculated once per run and passed
    to each trade only when opposite-crossover exits are enabled.
    """

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

    base_score = (
        (results["WinRatePct"] / 100) ** 2
        * results["ExpectancyR"] ** 1
        *(results["StopLossPct"] / 100) ** -1
    )
    
    results["ScoreRaw"] = np.where(
        (results["WinRatePct"] < 50)
        | (results["ExpectancyR"] < 0.25)
        | (results["StopLossPct"] == -1)
        | (results["TargetPct"] == -1),
        0.0,
        base_score,
    )

    maximum_score = (
        results[
            "ScoreRaw"
        ].max()
    )

    if np.isfinite(maximum_score) and maximum_score > 0:

        results[
            "Score"
        ] = (
            100 * results["ScoreRaw"] / maximum_score
        ).clip(
            lower=0
        )

    else:

        results[
            "Score"
        ] = 0.0

    return (
        results,
        trades,
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
    """Download data, select entries, simulate combinations, and package results."""

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
