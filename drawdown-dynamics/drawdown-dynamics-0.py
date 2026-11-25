"""
QQQ stop-behavior study using random entries.

Purpose
-------
Test stop-loss behavior independently of a trading strategy.

The experiment asks:

1. How often is a stop hit after an arbitrary entry?
2. If the stop is hit, how often does QQQ recover to the entry price?
3. How long does recovery take?
4. How much further does price fall after the stop is hit?
5. Is there a nonlinear "noise-to-regime" boundary at a particular stop depth?

Definitions
-----------
Entry:
    Adjusted opening price on a randomly selected trading day.

Stop hit:
    The adjusted daily Low touches entry_price * (1 - stop_pct)
    within the observation horizon.

Recovery:
    After the first stop hit, the adjusted daily High reaches or exceeds
    the original entry price within a selected recovery horizon.

Further drawdown:
    The worst adjusted Low after the stop was first hit, measured relative
    to the original entry price.

Important
---------
The script does not claim that a stop creates edge. It measures whether
different drawdown depths carry different information about subsequent
price behavior.

Install:
    pip install yfinance pandas numpy matplotlib seaborn

Outputs
-------
qqq_stop_analysis/
    qqq_adjusted_daily.csv
    qqq_random_entries.csv
    qqq_stop_event_details.csv
    qqq_stop_summary.csv
    qqq_stop_hit_rates.csv
    qqq_recovery_probabilities.csv
    qqq_recovery_curve.png
    qqq_stop_hit_curve.png
    qqq_time_to_recovery.png
    qqq_post_stop_drawdown.png
    qqq_stop_dashboard.png
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

# Test stops from 5% through 50%.
STOP_PCTS = np.arange(0.05, 0.51, 0.05)

# Approximate trading-day horizons.
OBSERVATION_HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}

# Recovery is measured from the first stop-hit date.
RECOVERY_HORIZONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "12M": 252,
    "24M": 504,
}

# Main observation horizon used for the detailed recovery analysis.
MAIN_OBSERVATION_HORIZON = "12M"

# Number of random entry dates.
# Set to None to use every eligible trading date instead.
N_RANDOM_ENTRIES = 5_000

RANDOM_SEED = 42

# Require enough future history for the observation period plus the longest
# recovery period. This prevents recent entries from having truncated outcomes.
MAX_REQUIRED_FUTURE_BARS = (
    OBSERVATION_HORIZONS[MAIN_OBSERVATION_HORIZON]
    + max(RECOVERY_HORIZONS.values())
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# =============================================================================
# Data download and preparation
# =============================================================================

def download_data(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """
    Download adjusted daily OHLC data from Yahoo Finance.

    auto_adjust=True adjusts prices for splits and dividends.
    multi_level_index=False avoids a MultiIndex for a single ticker.
    """
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
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
            f"No data was downloaded for {ticker}. "
            "Check the ticker, internet connection, and yfinance installation."
        )

    required_columns = ["Open", "High", "Low", "Close"]
    missing = [column for column in required_columns if column not in data.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    data = data[required_columns].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    data = data[~data.index.duplicated(keep="first")]
    data = data.sort_index()

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    return data


def select_entry_indices(
    data: pd.DataFrame,
    required_future_bars: int,
    n_entries: int | None,
    random_seed: int,
) -> np.ndarray:
    """
    Select random entry indices with enough future data for a complete test.

    Entries begin at index 1 so that the sample does not rely on the first
    available record.
    """
    last_eligible_index = len(data) - required_future_bars - 1

    if last_eligible_index < 1:
        raise RuntimeError(
            "Not enough data for the configured observation and recovery horizons."
        )

    eligible_indices = np.arange(1, last_eligible_index + 1)

    if n_entries is None or n_entries >= len(eligible_indices):
        return eligible_indices

    rng = np.random.default_rng(random_seed)

    return np.sort(
        rng.choice(
            eligible_indices,
            size=n_entries,
            replace=False,
        )
    )


# =============================================================================
# Core calculations
# =============================================================================

def calculate_stop_hit_rates(
    data: pd.DataFrame,
    entry_indices: np.ndarray,
    stop_pcts: np.ndarray,
    observation_horizons: dict[str, int],
) -> pd.DataFrame:
    """
    Calculate the probability that each stop is touched within each horizon.
    """
    lows = data["Low"].to_numpy(dtype=float)
    opens = data["Open"].to_numpy(dtype=float)

    records = []

    for stop_pct in stop_pcts:
        for horizon_name, horizon_bars in observation_horizons.items():
            hit_count = 0

            for entry_idx in entry_indices:
                entry_price = opens[entry_idx]
                stop_price = entry_price * (1.0 - stop_pct)

                future_lows = lows[entry_idx + 1: entry_idx + horizon_bars + 1]
                stop_hit = np.any(future_lows <= stop_price)

                hit_count += int(stop_hit)

            total_entries = len(entry_indices)

            records.append({
                "Ticker": TICKER,
                "StopPct": stop_pct,
                "StopLabel": f"{stop_pct:.0%}",
                "ObservationHorizon": horizon_name,
                "ObservationBars": horizon_bars,
                "Entries": total_entries,
                "StopHits": hit_count,
                "StopHitProbability": hit_count / total_entries,
            })

    return pd.DataFrame(records)


def analyze_stop_events(
    data: pd.DataFrame,
    entry_indices: np.ndarray,
    stop_pcts: np.ndarray,
    observation_bars: int,
    recovery_horizons: dict[str, int],
) -> pd.DataFrame:
    """
    Create one record per entry-stop-recovery-horizon combination.

    A stop must first be hit within observation_bars.

    Recovery is then evaluated from the stop-hit date forward. Recovery means
    that adjusted High reaches the original adjusted entry Open.
    """
    dates = data.index.to_numpy()
    opens = data["Open"].to_numpy(dtype=float)
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)

    records = []

    for entry_number, entry_idx in enumerate(entry_indices, start=1):
        entry_date = pd.Timestamp(dates[entry_idx])
        entry_price = opens[entry_idx]

        observation_end_idx = entry_idx + observation_bars
        observation_lows = lows[entry_idx + 1: observation_end_idx + 1]

        for stop_pct in stop_pcts:
            stop_price = entry_price * (1.0 - stop_pct)

            hit_positions = np.flatnonzero(observation_lows <= stop_price)
            stop_hit = len(hit_positions) > 0

            if not stop_hit:
                for recovery_name, recovery_bars in recovery_horizons.items():
                    records.append({
                        "Ticker": TICKER,
                        "EntryNumber": entry_number,
                        "EntryDate": entry_date,
                        "EntryPrice": entry_price,
                        "StopPct": stop_pct,
                        "StopLabel": f"{stop_pct:.0%}",
                        "StopPrice": stop_price,
                        "ObservationBars": observation_bars,
                        "StopHit": False,
                        "StopHitDate": pd.NaT,
                        "BarsToStop": np.nan,
                        "RecoveryHorizon": recovery_name,
                        "RecoveryBarsAllowed": recovery_bars,
                        "Recovered": np.nan,
                        "RecoveryDate": pd.NaT,
                        "BarsFromStopToRecovery": np.nan,
                        "WorstPriceAfterStop": np.nan,
                        "WorstDrawdownAfterStop": np.nan,
                        "AdditionalDrawdownBeyondStop": np.nan,
                        "EndPrice": closes[observation_end_idx],
                        "EndReturn": (
                            closes[observation_end_idx] / entry_price - 1.0
                        ),
                    })

                continue

            # +1 because observation_lows begins on the day after entry.
            first_hit_offset = int(hit_positions[0]) + 1
            stop_hit_idx = entry_idx + first_hit_offset
            stop_hit_date = pd.Timestamp(dates[stop_hit_idx])
            bars_to_stop = stop_hit_idx - entry_idx

            for recovery_name, recovery_bars in recovery_horizons.items():
                recovery_end_idx = stop_hit_idx + recovery_bars

                post_hit_highs = highs[
                    stop_hit_idx + 1: recovery_end_idx + 1
                ]
                post_hit_lows = lows[
                    stop_hit_idx: recovery_end_idx + 1
                ]

                recovery_positions = np.flatnonzero(
                    post_hit_highs >= entry_price
                )
                recovered = len(recovery_positions) > 0

                if recovered:
                    recovery_offset = int(recovery_positions[0]) + 1
                    recovery_idx = stop_hit_idx + recovery_offset
                    recovery_date = pd.Timestamp(dates[recovery_idx])
                    bars_to_recovery = recovery_idx - stop_hit_idx
                else:
                    recovery_date = pd.NaT
                    bars_to_recovery = np.nan

                worst_price = float(np.min(post_hit_lows))
                worst_drawdown = worst_price / entry_price - 1.0

                # Example:
                # stop = -20%, worst drawdown = -27%
                # additional decline beyond stop = -7%
                additional_drawdown = worst_drawdown + stop_pct

                end_price = closes[recovery_end_idx]
                end_return = end_price / entry_price - 1.0

                records.append({
                    "Ticker": TICKER,
                    "EntryNumber": entry_number,
                    "EntryDate": entry_date,
                    "EntryPrice": entry_price,
                    "StopPct": stop_pct,
                    "StopLabel": f"{stop_pct:.0%}",
                    "StopPrice": stop_price,
                    "ObservationBars": observation_bars,
                    "StopHit": True,
                    "StopHitDate": stop_hit_date,
                    "BarsToStop": bars_to_stop,
                    "RecoveryHorizon": recovery_name,
                    "RecoveryBarsAllowed": recovery_bars,
                    "Recovered": recovered,
                    "RecoveryDate": recovery_date,
                    "BarsFromStopToRecovery": bars_to_recovery,
                    "WorstPriceAfterStop": worst_price,
                    "WorstDrawdownAfterStop": worst_drawdown,
                    "AdditionalDrawdownBeyondStop": additional_drawdown,
                    "EndPrice": end_price,
                    "EndReturn": end_return,
                })

    return pd.DataFrame(records)


def create_recovery_summary(events: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize recovery behavior conditional on the stop being hit.
    """
    hit_events = events.loc[events["StopHit"]].copy()

    summary = (
        hit_events
        .groupby(
            [
                "StopPct",
                "StopLabel",
                "RecoveryHorizon",
                "RecoveryBarsAllowed",
            ],
            as_index=False,
        )
        .agg(
            StopHits=("StopHit", "size"),
            RecoveredCount=("Recovered", "sum"),
            RecoveryProbability=("Recovered", "mean"),
            MedianBarsToStop=("BarsToStop", "median"),
            MedianBarsToRecovery=("BarsFromStopToRecovery", "median"),
            MedianWorstDrawdown=("WorstDrawdownAfterStop", "median"),
            P10WorstDrawdown=(
                "WorstDrawdownAfterStop",
                lambda x: x.quantile(0.10),
            ),
            MedianAdditionalDrawdown=(
                "AdditionalDrawdownBeyondStop",
                "median",
            ),
            MedianEndReturn=("EndReturn", "median"),
            P10EndReturn=("EndReturn", lambda x: x.quantile(0.10)),
        )
    )

    return summary


def create_combined_summary(
    hit_rates: pd.DataFrame,
    recovery_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine the main observation-horizon hit rate with recovery statistics.
    """
    main_hit_rates = hit_rates.loc[
        hit_rates["ObservationHorizon"] == MAIN_OBSERVATION_HORIZON,
        [
            "StopPct",
            "StopLabel",
            "Entries",
            "StopHits",
            "StopHitProbability",
        ],
    ].copy()

    combined = recovery_summary.merge(
        main_hit_rates,
        how="left",
        on=["StopPct", "StopLabel"],
        suffixes=("", "_HitRate"),
    )

    combined["UnconditionalRecoveryProbability"] = (
        combined["StopHitProbability"]
        * combined["RecoveryProbability"]
    )

    return combined.sort_values(
        ["RecoveryBarsAllowed", "StopPct"]
    ).reset_index(drop=True)


# =============================================================================
# Statistical diagnostics
# =============================================================================

def bootstrap_recovery_intervals(
    events: pd.DataFrame,
    n_bootstraps: int = 2_000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for recovery probabilities.

    Resampling is performed by entry date rather than individual stop rows.
    This retains the dependence among stop levels calculated from the same
    entry event.
    """
    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - confidence

    hit_events = events.loc[events["StopHit"]].copy()
    results = []

    for (stop_pct, horizon), group in hit_events.groupby(
        ["StopPct", "RecoveryHorizon"]
    ):
        outcomes = group["Recovered"].astype(float).to_numpy()

        if len(outcomes) == 0:
            continue

        bootstrap_means = np.empty(n_bootstraps)

        for i in range(n_bootstraps):
            sample = rng.choice(
                outcomes,
                size=len(outcomes),
                replace=True,
            )
            bootstrap_means[i] = sample.mean()

        results.append({
            "StopPct": stop_pct,
            "StopLabel": f"{stop_pct:.0%}",
            "RecoveryHorizon": horizon,
            "StopHits": len(outcomes),
            "RecoveryProbability": outcomes.mean(),
            "LowerCI": np.quantile(bootstrap_means, alpha / 2.0),
            "UpperCI": np.quantile(
                bootstrap_means,
                1.0 - alpha / 2.0,
            ),
        })

    return pd.DataFrame(results)


# =============================================================================
# Charts
# =============================================================================

def plot_recovery_curve(
    recovery_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot P(recovery within N months | stop was hit).
    """
    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=recovery_summary,
        x="StopPct",
        y="RecoveryProbability",
        hue="RecoveryHorizon",
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{TICKER}: Probability of Recovery After Stop Is Hit\n"
        f"Stop must be hit within {MAIN_OBSERVATION_HORIZON}"
    )
    plt.xlabel("Stop depth")
    plt.ylabel("Conditional recovery probability")
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(title="Recovery within")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_stop_hit_curve(
    hit_rates: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot P(stop hit within observation horizon).
    """
    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=hit_rates,
        x="StopPct",
        y="StopHitProbability",
        hue="ObservationHorizon",
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{TICKER}: Probability That Stop Is Hit After Random Entry"
    )
    plt.xlabel("Stop depth")
    plt.ylabel("Stop-hit probability")
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(title="Hit within")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_time_to_recovery(
    recovery_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot median recovery time among positions that recovered.
    """
    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=recovery_summary,
        x="StopPct",
        y="MedianBarsToRecovery",
        hue="RecoveryHorizon",
        marker="o",
        linewidth=2,
    )

    plt.title(
        f"{TICKER}: Median Trading Days to Recover After Stop Is Hit"
    )
    plt.xlabel("Stop depth")
    plt.ylabel("Median trading days from stop hit to recovery")
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.grid(alpha=0.25)
    plt.legend(title="Recovery window")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_post_stop_drawdown(
    recovery_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot median and adverse-decile worst drawdown after the stop was hit.
    Use the 12-month recovery window for a consistent comparison.
    """
    selected = recovery_summary.loc[
        recovery_summary["RecoveryHorizon"] == "12M"
    ].sort_values("StopPct")

    plt.figure(figsize=(12, 7))

    plt.plot(
        selected["StopPct"],
        selected["MedianWorstDrawdown"],
        marker="o",
        linewidth=2,
        label="Median worst drawdown",
    )

    plt.plot(
        selected["StopPct"],
        selected["P10WorstDrawdown"],
        marker="o",
        linewidth=2,
        label="Adverse 10th percentile",
    )

    plt.plot(
        selected["StopPct"],
        -selected["StopPct"],
        linestyle="--",
        color="black",
        alpha=0.65,
        label="Stop threshold",
    )

    plt.title(
        f"{TICKER}: Further Drawdown After Stop Is Hit\n"
        "Measured over the following 12 months"
    )
    plt.xlabel("Stop depth")
    plt.ylabel("Worst drawdown from original entry")
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_dashboard(
    hit_rates: pd.DataFrame,
    recovery_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create a four-panel summary dashboard.
    """
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # Panel 1: stop-hit probability
    sns.lineplot(
        data=hit_rates,
        x="StopPct",
        y="StopHitProbability",
        hue="ObservationHorizon",
        marker="o",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Probability stop is hit")
    axes[0, 0].set_xlabel("Stop depth")
    axes[0, 0].set_ylabel("Probability")
    axes[0, 0].set_ylim(0, 1)

    # Panel 2: conditional recovery probability
    sns.lineplot(
        data=recovery_summary,
        x="StopPct",
        y="RecoveryProbability",
        hue="RecoveryHorizon",
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("P(recovery | stop hit)")
    axes[0, 1].set_xlabel("Stop depth")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].set_ylim(0, 1)

    # Panel 3: time to recovery
    sns.lineplot(
        data=recovery_summary,
        x="StopPct",
        y="MedianBarsToRecovery",
        hue="RecoveryHorizon",
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Median time to recovery")
    axes[1, 0].set_xlabel("Stop depth")
    axes[1, 0].set_ylabel("Trading days")

    # Panel 4: additional decline after stop
    selected = recovery_summary.loc[
        recovery_summary["RecoveryHorizon"] == "12M"
    ].sort_values("StopPct")

    axes[1, 1].plot(
        selected["StopPct"],
        selected["MedianAdditionalDrawdown"],
        marker="o",
        linewidth=2,
        label="Median additional decline",
    )
    axes[1, 1].axhline(
        0,
        color="black",
        linestyle="--",
        alpha=0.65,
    )
    axes[1, 1].set_title(
        "Additional decline beyond stop over next 12 months"
    )
    axes[1, 1].set_xlabel("Stop depth")
    axes[1, 1].set_ylabel("Additional return after stop threshold")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

    for axis in [axes[0, 0], axes[0, 1], axes[1, 1]]:
        axis.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )

    fig.suptitle(
        f"{TICKER} Random-Entry Stop Behavior",
        fontsize=18,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Downloading adjusted daily data for {TICKER}...")

    data = download_data(
        ticker=TICKER,
        start=START_DATE,
        end=END_DATE,
    )

    entry_indices = select_entry_indices(
        data=data,
        required_future_bars=MAX_REQUIRED_FUTURE_BARS,
        n_entries=N_RANDOM_ENTRIES,
        random_seed=RANDOM_SEED,
    )

    entry_table = pd.DataFrame({
        "EntryIndex": entry_indices,
        "EntryDate": data.index[entry_indices],
        "EntryOpen": data["Open"].iloc[entry_indices].to_numpy(),
    })

    print(
        f"Downloaded {len(data):,} daily observations from "
        f"{data.index.min().date()} through {data.index.max().date()}."
    )
    print(f"Testing {len(entry_indices):,} random entry dates.")

    hit_rates = calculate_stop_hit_rates(
        data=data,
        entry_indices=entry_indices,
        stop_pcts=STOP_PCTS,
        observation_horizons=OBSERVATION_HORIZONS,
    )

    events = analyze_stop_events(
        data=data,
        entry_indices=entry_indices,
        stop_pcts=STOP_PCTS,
        observation_bars=OBSERVATION_HORIZONS[
            MAIN_OBSERVATION_HORIZON
        ],
        recovery_horizons=RECOVERY_HORIZONS,
    )

    recovery_summary = create_recovery_summary(events)

    combined_summary = create_combined_summary(
        hit_rates=hit_rates,
        recovery_summary=recovery_summary,
    )

    bootstrap_intervals = bootstrap_recovery_intervals(
        events=events,
        n_bootstraps=2_000,
        confidence=0.95,
        random_seed=RANDOM_SEED,
    )

    # Save tables.
    data.to_csv(OUTPUT_DIR / "qqq_adjusted_daily.csv")
    entry_table.to_csv(
        OUTPUT_DIR / "qqq_random_entries.csv",
        index=False,
    )
    events.to_csv(
        OUTPUT_DIR / "qqq_stop_event_details.csv",
        index=False,
    )
    hit_rates.to_csv(
        OUTPUT_DIR / "qqq_stop_hit_rates.csv",
        index=False,
    )
    recovery_summary.to_csv(
        OUTPUT_DIR / "qqq_recovery_probabilities.csv",
        index=False,
    )
    combined_summary.to_csv(
        OUTPUT_DIR / "qqq_stop_summary.csv",
        index=False,
    )
    bootstrap_intervals.to_csv(
        OUTPUT_DIR / "qqq_recovery_confidence_intervals.csv",
        index=False,
    )

    # Create charts.
    plot_recovery_curve(
        recovery_summary,
        OUTPUT_DIR / "qqq_recovery_curve.png",
    )
    plot_stop_hit_curve(
        hit_rates,
        OUTPUT_DIR / "qqq_stop_hit_curve.png",
    )
    plot_time_to_recovery(
        recovery_summary,
        OUTPUT_DIR / "qqq_time_to_recovery.png",
    )
    plot_post_stop_drawdown(
        recovery_summary,
        OUTPUT_DIR / "qqq_post_stop_drawdown.png",
    )
    plot_dashboard(
        hit_rates,
        recovery_summary,
        OUTPUT_DIR / "qqq_stop_dashboard.png",
    )

    # Print the most useful table to the console.
    display_columns = [
        "StopLabel",
        "RecoveryHorizon",
        "Entries",
        "StopHits",
        "StopHitProbability",
        "RecoveryProbability",
        "MedianBarsToStop",
        "MedianBarsToRecovery",
        "MedianWorstDrawdown",
        "MedianAdditionalDrawdown",
    ]

    console_table = combined_summary.loc[
        combined_summary["RecoveryHorizon"].isin(
            ["6M", "12M", "24M"]
        ),
        display_columns,
    ].copy()

    percent_columns = [
        "StopHitProbability",
        "RecoveryProbability",
        "MedianWorstDrawdown",
        "MedianAdditionalDrawdown",
    ]

    for column in percent_columns:
        console_table[column] = console_table[column].map(
            lambda value: f"{value:.2%}"
        )

    print("\nMain results")
    print(console_table.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print(
        "\nPrimary chart:",
        (OUTPUT_DIR / "qqq_stop_dashboard.png").resolve(),
    )


if __name__ == "__main__":
    main()