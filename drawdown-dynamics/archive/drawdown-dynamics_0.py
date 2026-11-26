"""
QQQ Drawdown / Recovery Analysis
================================

Purpose
-------
Test drawdown/stop behavior independently of a trading strategy.

For each random entry:

    Entry
      |
      |----------- W1 ----------->
      |
      | Did price fall X%?
      |
      +---- Stop / drawdown hit
                  |
                  |----------- W2 ----------->
                  |
                  | Did price recover to the
                  | original entry price?

Definitions
-----------
W1:
    Maximum period after entry during which the drawdown threshold
    can be reached.

W2:
    Recovery period starting from the FIRST day the drawdown
    threshold is reached.

StopHitProbability:
    P(drawdown X% is reached within W1)

RecoveryProbability:
    P(recovery to original entry within W2 | drawdown was reached)

For cases that DO NOT recover within W2, the script also calculates:

    MedianWorstDropNonRecovered
        Median worst drawdown from the original entry price during W2.

    MedianAdditionalDropAfterStop
        Median additional decline beyond the stop threshold.

        Example:
            Stop = -20%
            Worst drawdown = -32%
            Additional decline = -12%

    MedianEndReturnNonRecovered
        Median return relative to entry at the END of W2.

Output
------
Only one analytical file is created:

    results/qqq_stop_summary.csv

Install dependencies
--------------------
pip install yfinance pandas numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# Configuration
# =============================================================================

TICKER = "QQQ"

START_DATE = "1999-03-10"
END_DATE = None


# Drawdown thresholds:
#
# 5%, 10%, 15%, ... 50%
#
STOP_PCTS = np.arange(
    0.05,
    0.51,
    0.05,
)


# -----------------------------------------------------------------------------
# W1
#
# Maximum period AFTER ENTRY during which the stop/drawdown
# threshold is allowed to occur.
# -----------------------------------------------------------------------------

W1_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}


# -----------------------------------------------------------------------------
# W2
#
# Recovery period AFTER THE STOP IS FIRST HIT.
#
# The recovery clock starts on the stop-hit date, not the
# original entry date.
# -----------------------------------------------------------------------------

W2_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}


# Number of random entry dates.
#
# Set to None to use every eligible trading day.
#
N_RANDOM_ENTRIES = 5_000

RANDOM_SEED = 42


# RESULTS directory beside this Python script.
#
# This avoids dependence on the VS Code working directory.
#
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
)


# =============================================================================
# Download data
# =============================================================================

def download_data() -> pd.DataFrame:
    """
    Download adjusted daily OHLC data from Yahoo Finance.

    auto_adjust=True:
        Prices are adjusted for splits and dividends.

    High and Low are used because we need to know whether a
    drawdown/recovery threshold was touched intraday.
    """

    print(
        f"Downloading {TICKER}..."
    )

    data = yf.download(
        tickers=TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        repair=True,
        actions=False,
        progress=False,
        multi_level_index=False,
        timeout=30,
    )

    if data.empty:
        raise RuntimeError(
            f"No data returned for {TICKER}."
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
        raise RuntimeError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    data = (
        data[required_columns]
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna()
        .sort_index()
    )

    # Remove duplicate trading dates if any.
    data = data[
        ~data.index.duplicated(
            keep="first"
        )
    ]

    # Make timezone handling consistent.
    if data.index.tz is not None:
        data.index = (
            data.index.tz_localize(None)
        )

    print(
        f"History: "
        f"{data.index.min().date()} "
        f"to "
        f"{data.index.max().date()}"
    )

    print(
        f"Trading days: "
        f"{len(data):,}"
    )

    return data


# =============================================================================
# Select random entries
# =============================================================================

def select_entries(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Select random entry dates.

    Every selected entry must have enough future history for:

        longest W1
        +
        longest W2

    This prevents recent entries from having incomplete observation
    or recovery periods.
    """

    maximum_w1 = max(
        W1_WINDOWS.values()
    )

    maximum_w2 = max(
        W2_WINDOWS.values()
    )

    required_future_bars = (
        maximum_w1
        + maximum_w2
    )

    last_eligible_entry = (
        len(data)
        - required_future_bars
        - 1
    )

    if last_eligible_entry < 1:
        raise RuntimeError(
            "Not enough price history for "
            "the configured W1 and W2 windows."
        )

    eligible_entries = np.arange(
        1,
        last_eligible_entry + 1,
    )

    # Use every eligible trading day.
    if (
        N_RANDOM_ENTRIES is None
        or N_RANDOM_ENTRIES
        >= len(eligible_entries)
    ):
        entries = eligible_entries

    else:

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        entries = rng.choice(
            eligible_entries,
            size=N_RANDOM_ENTRIES,
            replace=False,
        )

        entries = np.sort(
            entries
        )

    print(
        f"Eligible entries: "
        f"{len(eligible_entries):,}"
    )

    print(
        f"Entries tested: "
        f"{len(entries):,}"
    )

    return entries


# =============================================================================
# Analysis
# =============================================================================

def analyze(
    data: pd.DataFrame,
    entry_indices: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate the complete:

        Stop x W1 x W2

    summary.

    No strategy entry signal or profit target is used.
    """

    opens = data[
        "Open"
    ].to_numpy(
        dtype=float
    )

    highs = data[
        "High"
    ].to_numpy(
        dtype=float
    )

    lows = data[
        "Low"
    ].to_numpy(
        dtype=float
    )

    closes = data[
        "Close"
    ].to_numpy(
        dtype=float
    )

    total_entries = len(
        entry_indices
    )

    summary_records = []

    # =========================================================================
    # Stop depth
    # =========================================================================

    for stop_pct in STOP_PCTS:

        print(
            f"Analyzing stop "
            f"{stop_pct:.0%}..."
        )

        # =====================================================================
        # W1
        # =====================================================================

        for w1_name, w1_bars in (
            W1_WINDOWS.items()
        ):

            # -----------------------------------------------------------------
            # First determine which entries hit this stop during W1.
            #
            # We calculate this ONCE for each Stop x W1 combination.
            # W2 has nothing to do with StopHitProbability.
            # -----------------------------------------------------------------

            hit_events = []

            for entry_idx in entry_indices:

                entry_price = opens[
                    entry_idx
                ]

                stop_price = (
                    entry_price
                    * (1.0 - stop_pct)
                )

                # W1 begins on the trading day AFTER entry.
                future_lows = lows[
                    entry_idx + 1:
                    entry_idx + w1_bars + 1
                ]

                hit_positions = (
                    np.flatnonzero(
                        future_lows
                        <= stop_price
                    )
                )

                # Stop never reached during W1.
                if len(hit_positions) == 0:
                    continue

                # First time the stop threshold is reached.
                first_hit_offset = (
                    int(hit_positions[0])
                    + 1
                )

                hit_idx = (
                    entry_idx
                    + first_hit_offset
                )

                hit_events.append(
                    {
                        "EntryIndex":
                            entry_idx,

                        "EntryPrice":
                            entry_price,

                        "StopPrice":
                            stop_price,

                        "HitIndex":
                            hit_idx,

                        "DaysToStop":
                            first_hit_offset,
                    }
                )

            # -----------------------------------------------------------------
            # Stop-hit statistics
            # -----------------------------------------------------------------

            stop_hits = len(
                hit_events
            )

            stop_hit_probability = (
                stop_hits
                / total_entries
                if total_entries
                else np.nan
            )

            if hit_events:

                median_days_to_stop = (
                    float(
                        np.median(
                            [
                                event[
                                    "DaysToStop"
                                ]
                                for event
                                in hit_events
                            ]
                        )
                    )
                )

            else:

                median_days_to_stop = (
                    np.nan
                )

            # =================================================================
            # W2
            # =================================================================

            for w2_name, w2_bars in (
                W2_WINDOWS.items()
            ):

                recovered_count = 0

                recovery_times = []

                # Only for cases that FAIL to recover.
                non_recovered_worst_drops = []
                non_recovered_additional_drops = []
                non_recovered_end_returns = []

                # -------------------------------------------------------------
                # Evaluate every stop-hit event.
                # -------------------------------------------------------------

                for event in hit_events:

                    entry_price = event[
                        "EntryPrice"
                    ]

                    hit_idx = event[
                        "HitIndex"
                    ]

                    w2_end_idx = (
                        hit_idx
                        + w2_bars
                    )

                    # ---------------------------------------------------------
                    # Recovery test
                    #
                    # Recovery begins AFTER the stop-hit day.
                    #
                    # A recovery occurs when adjusted High reaches or exceeds
                    # the ORIGINAL entry price.
                    # ---------------------------------------------------------

                    future_highs = highs[
                        hit_idx + 1:
                        w2_end_idx + 1
                    ]

                    recovery_positions = (
                        np.flatnonzero(
                            future_highs
                            >= entry_price
                        )
                    )

                    recovered = (
                        len(
                            recovery_positions
                        )
                        > 0
                    )

                    # =========================================================
                    # Recovered
                    # =========================================================

                    if recovered:

                        recovered_count += 1

                        first_recovery_offset = (
                            int(
                                recovery_positions[0]
                            )
                            + 1
                        )

                        recovery_times.append(
                            first_recovery_offset
                        )

                    # =========================================================
                    # Did not recover
                    # =========================================================

                    else:

                        # Include the stop-hit day in the W2 drawdown analysis.
                        w2_lows = lows[
                            hit_idx:
                            w2_end_idx + 1
                        ]

                        worst_price = float(
                            np.min(
                                w2_lows
                            )
                        )

                        # -----------------------------------------------------
                        # Worst drop relative to ORIGINAL ENTRY.
                        #
                        # Example:
                        #
                        #     Entry = 100
                        #     Worst = 68
                        #
                        #     WorstDrop = -32%
                        # -----------------------------------------------------

                        worst_drop = (
                            worst_price
                            / entry_price
                            - 1.0
                        )

                        # -----------------------------------------------------
                        # Additional decline beyond stop.
                        #
                        # Example:
                        #
                        #     Stop = -20%
                        #     Worst = -32%
                        #
                        #     AdditionalDrop = -12%
                        #
                        # Negative means it continued lower beyond the stop.
                        # -----------------------------------------------------

                        additional_drop = (
                            worst_drop
                            + stop_pct
                        )

                        # -----------------------------------------------------
                        # Where was the position at the END of W2?
                        #
                        # This is different from worst drawdown.
                        #
                        # Example:
                        #
                        #     Worst during W2 = -40%
                        #     End of W2       = -18%
                        #
                        # The position recovered substantially,
                        # but did not reach the original entry.
                        # -----------------------------------------------------

                        end_price = closes[
                            w2_end_idx
                        ]

                        end_return = (
                            end_price
                            / entry_price
                            - 1.0
                        )

                        non_recovered_worst_drops.append(
                            worst_drop
                        )

                        non_recovered_additional_drops.append(
                            additional_drop
                        )

                        non_recovered_end_returns.append(
                            end_return
                        )

                # =============================================================
                # W2 statistics
                # =============================================================

                not_recovered_count = (
                    stop_hits
                    - recovered_count
                )

                recovery_probability = (
                    recovered_count
                    / stop_hits
                    if stop_hits
                    else np.nan
                )

                non_recovery_probability = (
                    not_recovered_count
                    / stop_hits
                    if stop_hits
                    else np.nan
                )

                # -------------------------------------------------------------
                # Median recovery time among cases that DID recover.
                # -------------------------------------------------------------

                if recovery_times:

                    median_days_to_recovery = (
                        float(
                            np.median(
                                recovery_times
                            )
                        )
                    )

                else:

                    median_days_to_recovery = (
                        np.nan
                    )

                # -------------------------------------------------------------
                # Statistics ONLY among cases that failed to recover.
                # -------------------------------------------------------------

                if non_recovered_worst_drops:

                    median_worst_drop = (
                        float(
                            np.median(
                                non_recovered_worst_drops
                            )
                        )
                    )

                    median_additional_drop = (
                        float(
                            np.median(
                                non_recovered_additional_drops
                            )
                        )
                    )

                    median_end_return = (
                        float(
                            np.median(
                                non_recovered_end_returns
                            )
                        )
                    )

                else:

                    median_worst_drop = (
                        np.nan
                    )

                    median_additional_drop = (
                        np.nan
                    )

                    median_end_return = (
                        np.nan
                    )

                # =============================================================
                # Final summary row
                # =============================================================

                summary_records.append(
                    {
                        "Ticker":
                            TICKER,

                        "StopPct":
                            stop_pct,

                        "W1":
                            w1_name,

                        "W2":
                            w2_name,

                        "Entries":
                            total_entries,

                        "StopHits":
                            stop_hits,

                        "StopHitProbability":
                            stop_hit_probability,

                        "MedianDaysToStop":
                            median_days_to_stop,

                        "Recovered":
                            recovered_count,

                        "NotRecovered":
                            not_recovered_count,

                        "RecoveryProbability":
                            recovery_probability,

                        "NonRecoveryProbability":
                            non_recovery_probability,

                        "MedianDaysToRecovery":
                            median_days_to_recovery,

                        "MedianWorstDropNonRecovered":
                            median_worst_drop,

                        "MedianAdditionalDropAfterStop":
                            median_additional_drop,

                        "MedianEndReturnNonRecovered":
                            median_end_return,
                    }
                )

    # =========================================================================
    # Final DataFrame
    # =========================================================================

    summary = pd.DataFrame(
        summary_records
    )

    # Preserve logical ordering rather than alphabetical W1/W2 ordering.
    w1_order = {
        name: i
        for i, name
        in enumerate(
            W1_WINDOWS.keys()
        )
    }

    w2_order = {
        name: i
        for i, name
        in enumerate(
            W2_WINDOWS.keys()
        )
    }

    summary["_W1Order"] = (
        summary["W1"]
        .map(w1_order)
    )

    summary["_W2Order"] = (
        summary["W2"]
        .map(w2_order)
    )

    summary = (
        summary.sort_values(
            [
                "StopPct",
                "_W1Order",
                "_W2Order",
            ]
        )
        .drop(
            columns=[
                "_W1Order",
                "_W2Order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Run the complete study and produce one summary CSV.
    """

    # -------------------------------------------------------------------------
    # Results folder
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------------

    data = download_data()

    # -------------------------------------------------------------------------
    # Entries
    # -------------------------------------------------------------------------

    entry_indices = (
        select_entries(
            data
        )
    )

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    print(
        "\nCalculating "
        "Stop x W1 x W2..."
    )

    summary = analyze(
        data=data,
        entry_indices=entry_indices,
    )

    # -------------------------------------------------------------------------
    # Save one output file
    # -------------------------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "qqq_stop_summary.csv"
    )

    summary = summary.round({
    "StopHitProbability": 2,
    "RecoveryProbability": 2,
    "NonRecoveryProbability": 2,
    "MedianWorstDropNonRecovered":2,
    "MedianAdditionalDropAfterStop":2,
    "MedianEndReturnNonRecovered":2
    })

    summary.to_csv(
        output_file,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print(
        "\nAnalysis complete."
    )

    print(
        f"Summary rows: "
        f"{len(summary):,}"
    )

    print(
        "\nSaved:"
    )

    print(
        output_file.resolve()
    )

    # -------------------------------------------------------------------------
    # Show a useful slice in the console:
    #
    # W1 = 12 months
    # W2 = 12 months
    # -------------------------------------------------------------------------

    example = summary[
        (summary["W1"] == "12M")
        & (summary["W2"] == "12M")
    ].copy()

    display_columns = [
        "StopPct",
        "StopHitProbability",
        "RecoveryProbability",
        "MedianDaysToStop",
        "MedianDaysToRecovery",
        "MedianWorstDropNonRecovered",
        "MedianAdditionalDropAfterStop",
        "MedianEndReturnNonRecovered",
    ]

    print(
        "\n"
        "W1 = 12M / W2 = 12M"
    )

    print(
        example[
            display_columns
        ].to_string(
            index=False
        )
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()