"""
QQQ Drawdown Threshold Analysis
===============================

Tests stop/drawdown behavior independently of a trading strategy.

Experiment
----------
For each arbitrary entry date and stop depth:

    ENTRY
      |
      |<------------- W1 ------------->
      |
      +---- Did price hit the stop?
                    |
                    |<-------- W2 -------->
                    |
                    +---- Did price recover
                          to the original
                          entry price?

W1:
    Probability that the stop is hit within W1 after entry.

W2:
    Conditional probability that price recovers to the original
    entry price within W2 AFTER the stop was first hit.

The script calculates the full:

    STOP x W1 x W2

cube.

It therefore also lets us investigate whether the speed at which a
drawdown develops matters. For example:

    P(recovery in 12M | -20% reached within 3M)

versus:

    P(recovery in 12M | -20% reached within 12M)

Install
-------
pip install yfinance pandas numpy matplotlib seaborn

Outputs
-------
RESULTS/
    qqq_daily.csv
    qqq_random_entries.csv
    qqq_stop_w1_w2_summary.csv
    qqq_stop_hit_probability.csv
    qqq_recovery_probability.csv
    qqq_recovery_12m_heatmap.png
    qqq_stop_hit_heatmap.png
    qqq_recovery_by_w1.png
    qqq_recovery_by_w2.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# =============================================================================
# Configuration
# =============================================================================

TICKER = "QQQ"
START_DATE = "1999-03-10"
END_DATE = None

# Drawdown thresholds to test.
STOP_PCTS = np.arange(0.05, 0.51, 0.05)

# W1: Maximum time allowed for the stop to be reached after entry.
STOP_HIT_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}

# W2: Maximum time allowed for recovery AFTER the stop is first hit.
RECOVERY_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}

# Random entries.
# None = use every eligible trading day.
N_RANDOM_ENTRIES = 5_000

RANDOM_SEED = 42

# Results folder beside the Python script.
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


# =============================================================================
# Download data
# =============================================================================

def download_data() -> pd.DataFrame:
    """Download adjusted daily OHLC data."""

    data = yf.download(
        TICKER,
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
        raise RuntimeError(f"No data returned for {TICKER}")

    required = ["Open", "High", "Low", "Close"]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    data = data[required].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    data = data.sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data


# =============================================================================
# Select entries
# =============================================================================

def select_entries(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Select entry dates with enough future data for:

        maximum W1 + maximum W2
    """

    required_future = (
        max(STOP_HIT_WINDOWS.values())
        + max(RECOVERY_WINDOWS.values())
    )

    last_entry = len(data) - required_future - 1

    if last_entry < 1:
        raise RuntimeError(
            "Not enough history for configured W1 and W2."
        )

    eligible = np.arange(1, last_entry + 1)

    if (
        N_RANDOM_ENTRIES is None
        or N_RANDOM_ENTRIES >= len(eligible)
    ):
        return eligible

    rng = np.random.default_rng(RANDOM_SEED)

    entries = rng.choice(
        eligible,
        size=N_RANDOM_ENTRIES,
        replace=False,
    )

    return np.sort(entries)


# =============================================================================
# Main calculation
# =============================================================================

def analyze(
    data: pd.DataFrame,
    entry_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate individual stop events and Stop x W1 x W2 summary.
    """

    dates = data.index.to_numpy()

    opens = data["Open"].to_numpy(dtype=float)
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)

    event_records = []

    # -------------------------------------------------------------------------
    # Loop through entries
    # -------------------------------------------------------------------------

    for entry_number, entry_idx in enumerate(
        entry_indices,
        start=1,
    ):

        entry_date = pd.Timestamp(dates[entry_idx])
        entry_price = opens[entry_idx]

        # ---------------------------------------------------------------------
        # Test every W1
        # ---------------------------------------------------------------------

        for w1_name, w1_bars in STOP_HIT_WINDOWS.items():

            w1_lows = lows[
                entry_idx + 1:
                entry_idx + w1_bars + 1
            ]

            # -----------------------------------------------------------------
            # Test every stop
            # -----------------------------------------------------------------

            for stop_pct in STOP_PCTS:

                stop_price = entry_price * (
                    1.0 - stop_pct
                )

                hit_positions = np.flatnonzero(
                    w1_lows <= stop_price
                )

                stop_hit = len(hit_positions) > 0

                # -------------------------------------------------------------
                # Stop not hit
                # -------------------------------------------------------------

                if not stop_hit:

                    for w2_name, w2_bars in (
                        RECOVERY_WINDOWS.items()
                    ):

                        event_records.append({
                            "EntryNumber": entry_number,
                            "EntryDate": entry_date,
                            "EntryPrice": entry_price,

                            "StopPct": stop_pct,
                            "StopLabel": f"{stop_pct:.0%}",
                            "StopPrice": stop_price,

                            "W1": w1_name,
                            "W1Bars": w1_bars,

                            "StopHit": False,
                            "StopHitDate": pd.NaT,
                            "BarsToStop": np.nan,

                            "W2": w2_name,
                            "W2Bars": w2_bars,

                            "Recovered": np.nan,
                            "RecoveryDate": pd.NaT,
                            "BarsToRecovery": np.nan,
                        })

                    continue

                # -------------------------------------------------------------
                # First stop hit
                # -------------------------------------------------------------

                first_hit_offset = (
                    int(hit_positions[0]) + 1
                )

                hit_idx = (
                    entry_idx
                    + first_hit_offset
                )

                hit_date = pd.Timestamp(
                    dates[hit_idx]
                )

                bars_to_stop = (
                    hit_idx - entry_idx
                )

                # -------------------------------------------------------------
                # Evaluate every W2 starting FROM stop-hit date
                # -------------------------------------------------------------

                for w2_name, w2_bars in (
                    RECOVERY_WINDOWS.items()
                ):

                    recovery_end_idx = (
                        hit_idx + w2_bars
                    )

                    future_highs = highs[
                        hit_idx + 1:
                        recovery_end_idx + 1
                    ]

                    recovery_positions = (
                        np.flatnonzero(
                            future_highs >= entry_price
                        )
                    )

                    recovered = (
                        len(recovery_positions) > 0
                    )

                    if recovered:

                        recovery_offset = (
                            int(
                                recovery_positions[0]
                            )
                            + 1
                        )

                        recovery_idx = (
                            hit_idx
                            + recovery_offset
                        )

                        recovery_date = (
                            pd.Timestamp(
                                dates[recovery_idx]
                            )
                        )

                        bars_to_recovery = (
                            recovery_idx
                            - hit_idx
                        )

                    else:

                        recovery_date = pd.NaT
                        bars_to_recovery = np.nan

                    event_records.append({
                        "EntryNumber": entry_number,
                        "EntryDate": entry_date,
                        "EntryPrice": entry_price,

                        "StopPct": stop_pct,
                        "StopLabel": f"{stop_pct:.0%}",
                        "StopPrice": stop_price,

                        "W1": w1_name,
                        "W1Bars": w1_bars,

                        "StopHit": True,
                        "StopHitDate": hit_date,
                        "BarsToStop": bars_to_stop,

                        "W2": w2_name,
                        "W2Bars": w2_bars,

                        "Recovered": recovered,
                        "RecoveryDate": recovery_date,
                        "BarsToRecovery": bars_to_recovery,
                    })

    events = pd.DataFrame(event_records)

    # =========================================================================
    # Summary
    # =========================================================================

    summary_records = []

    for stop_pct in STOP_PCTS:

        for w1_name, w1_bars in (
            STOP_HIT_WINDOWS.items()
        ):

            # One W2 is sufficient to identify unique entries because
            # StopHit is identical across W2 values.
            base = events[
                (events["StopPct"] == stop_pct)
                & (events["W1"] == w1_name)
                & (
                    events["W2"]
                    == next(iter(RECOVERY_WINDOWS))
                )
            ]

            entries = len(base)

            stop_hits = int(
                base["StopHit"].sum()
            )

            stop_hit_probability = (
                stop_hits / entries
                if entries
                else np.nan
            )

            # -----------------------------------------------------------------
            # W2
            # -----------------------------------------------------------------

            for w2_name, w2_bars in (
                RECOVERY_WINDOWS.items()
            ):

                group = events[
                    (events["StopPct"] == stop_pct)
                    & (events["W1"] == w1_name)
                    & (events["W2"] == w2_name)
                ]

                hits = group[
                    group["StopHit"]
                ]

                recovered = int(
                    hits["Recovered"]
                    .fillna(False)
                    .sum()
                )

                recovery_probability = (
                    recovered / len(hits)
                    if len(hits)
                    else np.nan
                )

                median_bars_to_stop = (
                    hits["BarsToStop"].median()
                    if len(hits)
                    else np.nan
                )

                recovered_rows = hits[
                    hits["Recovered"] == True
                ]

                median_bars_to_recovery = (
                    recovered_rows[
                        "BarsToRecovery"
                    ].median()
                    if len(recovered_rows)
                    else np.nan
                )

                summary_records.append({
                    "Ticker": TICKER,

                    "StopPct": stop_pct,
                    "StopLabel": f"{stop_pct:.0%}",

                    "W1": w1_name,
                    "W1Bars": w1_bars,

                    "Entries": entries,
                    "StopHits": stop_hits,

                    "StopHitProbabilityWithinW1":
                        stop_hit_probability,

                    "W2": w2_name,
                    "W2Bars": w2_bars,

                    "Recovered": recovered,

                    "RecoveryProbabilityWithinW2GivenHit":
                        recovery_probability,

                    "MedianBarsEntryToStop":
                        median_bars_to_stop,

                    "MedianBarsStopToRecovery":
                        median_bars_to_recovery,
                })

    summary = pd.DataFrame(
        summary_records
    )

    return events, summary


# =============================================================================
# Stop-hit table
# =============================================================================

def create_stop_hit_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    # Stop-hit probability doesn't depend on W2,
    # so take one W2 per Stop/W1 combination.

    table = (
        summary
        .drop_duplicates(
            ["StopPct", "W1"]
        )
        [
            [
                "StopPct",
                "StopLabel",
                "W1",
                "W1Bars",
                "Entries",
                "StopHits",
                "StopHitProbabilityWithinW1",
                "MedianBarsEntryToStop",
            ]
        ]
        .sort_values(
            ["StopPct", "W1Bars"]
        )
    )

    return table


# =============================================================================
# Recovery table
# =============================================================================

def create_recovery_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:

    return (
        summary[
            [
                "StopPct",
                "StopLabel",
                "W1",
                "W1Bars",
                "W2",
                "W2Bars",
                "StopHits",
                "Recovered",
                "RecoveryProbabilityWithinW2GivenHit",
                "MedianBarsStopToRecovery",
            ]
        ]
        .sort_values(
            [
                "StopPct",
                "W1Bars",
                "W2Bars",
            ]
        )
    )


# =============================================================================
# Charts
# =============================================================================

def plot_stop_hit_heatmap(
    stop_table: pd.DataFrame,
) -> None:

    matrix = stop_table.pivot(
        index="StopLabel",
        columns="W1",
        values="StopHitProbabilityWithinW1",
    )

    column_order = list(
        STOP_HIT_WINDOWS.keys()
    )

    matrix = matrix.reindex(
        columns=column_order
    )

    stop_order = [
        f"{x:.0%}"
        for x in STOP_PCTS
    ]

    matrix = matrix.reindex(
        stop_order
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )

    plt.title(
        f"{TICKER}: Probability Stop Is Hit Within W1"
    )

    plt.xlabel(
        "W1: Time after entry"
    )

    plt.ylabel(
        "Drawdown / Stop"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qqq_stop_hit_heatmap.png",
        dpi=180,
    )

    plt.close()


def plot_recovery_heatmap(
    recovery_table: pd.DataFrame,
    selected_w2: str = "12M",
) -> None:

    subset = recovery_table[
        recovery_table["W2"]
        == selected_w2
    ]

    matrix = subset.pivot(
        index="StopLabel",
        columns="W1",
        values=(
            "RecoveryProbabilityWithinW2GivenHit"
        ),
    )

    matrix = matrix.reindex(
        columns=list(
            STOP_HIT_WINDOWS.keys()
        )
    )

    matrix = matrix.reindex(
        [
            f"{x:.0%}"
            for x in STOP_PCTS
        ]
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )

    plt.title(
        f"{TICKER}: Recovery Probability\n"
        f"W2 = {selected_w2} after stop is hit"
    )

    plt.xlabel(
        "W1: Maximum time allowed for stop to occur"
    )

    plt.ylabel(
        "Drawdown / Stop"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / f"qqq_recovery_{selected_w2.lower()}_heatmap.png",
        dpi=180,
    )

    plt.close()


def plot_recovery_by_w1(
    recovery_table: pd.DataFrame,
    selected_w2: str = "12M",
) -> None:

    subset = recovery_table[
        recovery_table["W2"]
        == selected_w2
    ]

    plt.figure(
        figsize=(12, 7)
    )

    sns.lineplot(
        data=subset,
        x="StopPct",
        y=(
            "RecoveryProbabilityWithinW2GivenHit"
        ),
        hue="W1",
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{TICKER}: Recovery Probability by Drawdown Speed\n"
        f"Recovery window W2 = {selected_w2}"
    )

    plt.xlabel(
        "Drawdown / Stop"
    )

    plt.ylabel(
        "P(Recovery within W2 | Stop Hit within W1)"
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.0%}"
        )
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, _: f"{y:.0%}"
        )
    )

    plt.ylim(0, 1)
    plt.grid(alpha=0.25)

    plt.legend(
        title="W1"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qqq_recovery_by_w1.png",
        dpi=180,
    )

    plt.close()


def plot_recovery_by_w2(
    recovery_table: pd.DataFrame,
    selected_w1: str = "12M",
) -> None:

    subset = recovery_table[
        recovery_table["W1"]
        == selected_w1
    ]

    plt.figure(
        figsize=(12, 7)
    )

    sns.lineplot(
        data=subset,
        x="StopPct",
        y=(
            "RecoveryProbabilityWithinW2GivenHit"
        ),
        hue="W2",
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{TICKER}: Recovery Probability by Recovery Horizon\n"
        f"Stop must occur within W1 = {selected_w1}"
    )

    plt.xlabel(
        "Drawdown / Stop"
    )

    plt.ylabel(
        "P(Recovery within W2 | Stop Hit)"
    )

    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"{x:.0%}"
        )
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, _: f"{y:.0%}"
        )
    )

    plt.ylim(0, 1)
    plt.grid(alpha=0.25)

    plt.legend(
        title="W2"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "qqq_recovery_by_w2.png",
        dpi=180,
    )

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sns.set_theme(
        style="whitegrid"
    )

    print(
        f"Downloading {TICKER}..."
    )

    data = download_data()

    entry_indices = select_entries(
        data
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

    print(
        f"Entries: "
        f"{len(entry_indices):,}"
    )

    # -------------------------------------------------------------------------
    # Entry table
    # -------------------------------------------------------------------------

    entry_table = pd.DataFrame({
        "EntryIndex": entry_indices,
        "EntryDate":
            data.index[entry_indices],
        "EntryOpen":
            data["Open"]
            .iloc[entry_indices]
            .to_numpy(),
    })

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    print(
        "\nCalculating "
        "Stop x W1 x W2 cube..."
    )

    events, summary = analyze(
        data,
        entry_indices,
    )

    stop_table = (
        create_stop_hit_table(
            summary
        )
    )

    recovery_table = (
        create_recovery_table(
            summary
        )
    )

    # -------------------------------------------------------------------------
    # Save CSV files
    # -------------------------------------------------------------------------

    data.to_csv(
        OUTPUT_DIR
        / "qqq_daily.csv"
    )

    entry_table.to_csv(
        OUTPUT_DIR
        / "qqq_random_entries.csv",
        index=False,
    )

    events.to_csv(
        OUTPUT_DIR
        / "qqq_stop_w1_w2_events.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "qqq_stop_w1_w2_summary.csv",
        index=False,
    )

    stop_table.to_csv(
        OUTPUT_DIR
        / "qqq_stop_hit_probability.csv",
        index=False,
    )

    recovery_table.to_csv(
        OUTPUT_DIR
        / "qqq_recovery_probability.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Charts
    # -------------------------------------------------------------------------

    plot_stop_hit_heatmap(
        stop_table
    )

    # Generate one recovery heatmap
    # for every W2.
    for w2 in RECOVERY_WINDOWS:

        plot_recovery_heatmap(
            recovery_table,
            selected_w2=w2,
        )

    plot_recovery_by_w1(
        recovery_table,
        selected_w2="12M",
    )

    plot_recovery_by_w2(
        recovery_table,
        selected_w1="12M",
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "STOP HIT PROBABILITY"
    )

    print(
        "============================================================"
    )

    console_stop = (
        stop_table.copy()
    )

    console_stop[
        "StopHitProbabilityWithinW1"
    ] = (
        console_stop[
            "StopHitProbabilityWithinW1"
        ]
        .map(
            lambda x: f"{x:.1%}"
        )
    )

    print(
        console_stop.to_string(
            index=False
        )
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "RECOVERY: W2 = 12M"
    )

    print(
        "============================================================"
    )

    console_recovery = (
        recovery_table[
            recovery_table["W2"]
            == "12M"
        ]
        .copy()
    )

    console_recovery[
        "RecoveryProbabilityWithinW2GivenHit"
    ] = (
        console_recovery[
            "RecoveryProbabilityWithinW2GivenHit"
        ]
        .map(
            lambda x: f"{x:.1%}"
        )
    )

    print(
        console_recovery.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()