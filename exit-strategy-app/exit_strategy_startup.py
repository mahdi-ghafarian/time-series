"""Bundled-result loading, compatibility, and settings matching."""

from datetime import date
from pathlib import Path
import pickle

import numpy as np
import streamlit as st

from exit_strategy_core import parse_values, select_random_entries


STARTUP_TICKER = "QQQ"
STARTUP_RESULT_FILE = Path(__file__).parent / "startup_result.pkl"


def result_ticker(
    output,
):
    """Read the ticker from a saved simulation result, if available."""

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
    """Return whether a saved result has the current SMA/direction schema."""

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
    """Add current display columns to the legacy bundled result in memory."""

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
    """Check whether current controls match the bundled QQQ simulation.

    The bundle represents daily long-only entries selected with seed 27 and
    SMA entry/exit logic disabled. Its date range, costs, and stop/target grids
    must also match before it can stand in for a newly run simulation.
    """

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


def load_startup_result():
    """Load the bundled QQQ result, reporting deserialization errors in the UI."""

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
