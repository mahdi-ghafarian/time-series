"""Run Monte Carlo stop/target analysis and create an app-style heatmap."""

import argparse
from datetime import date
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import exit_strategy_monte_carlo_analysis as sensitivity
import exit_strategy_fast_engine as engine


# region USER CONFIGURATION
# =============================================================================
# Edit these defaults or override them with command-line options.
# =============================================================================

# region DATA
# Market ticker.
TICKER = "QQQ"
# Start of the historical data range, inclusive.
START_DATE = date(2000, 1, 1)
# End of the historical data range, inclusive.
END_DATE = date(2026, 12, 31)
# Trading candle interval.
INTERVAL = "1d"
# endregion

# region MONTE CARLO
# Statistic aggregated across Monte Carlo scenarios.
STATISTIC = "p75"
# Seed for the first scenario.
SCENARIO_START = 0
# Number of Monte Carlo scenarios. Seeds run from SCENARIO_START sequentially.
SCENARIO_COUNT = 1000
# endregion

#region TRADING
# Exit mode: Percent, ATRP, or TRP.
EXIT_MODE = "Percent"
# Candle timeframe used to calculate ATRP and TRP.
RANGE_TIMEFRAME = "3mo"
# ATR Wilder period or TRP history warm-up in range timeframe bars.
RANGE_PERIOD = 1
# Maximum random trades sampled from each calendar year per scenario.
MAX_TRADES_PER_YEAR = 12
# Maximum simultaneous open trades; None disables the concurrency limit.
MAX_OPEN_TRADES: int | None = 1
# Trade direction: Long, Short, or Both.
TRADE_DIRECTION = "Long"
# Commission percentage per trade side.
COMMISSION = 0.0
# Slippage percentage per trade side.
SLIPPAGE = 0.02
# If both exit levels hit in one candle, assume the stop hit first.
CONSERVATIVE = True
# Filter entries by the moving average.
MA_ENTRY_ENABLED = True
# Exit trades on an opposite SMA crossover.
MA_EXIT_ENABLED = False
# Moving-average period in trading bars.
SMA_PERIOD = 200
# Optional stop override; None uses the default stop grid.
STOPS: list[float] | None = None
# Optional target override; None uses the default target grid.
TARGETS: list[float] | None = None
# Fixed-percentage stop grid; -1 disables the stop.
DEFAULT_PERCENT_STOPS = [
    10, 15, 20, 25, 30, 35, 40, 45, 50
]
# Fixed-percentage target grid; -1 disables the target.
DEFAULT_PERCENT_TARGETS = [
    10, 15, 20, 25, 30, 35, 40, 45, 50,
    60, 70, 80, 90, 100, -1
]
# Normalized stop-multiplier grid used by ATRP and TRP modes.
RANGE_STOP_MULTIPLIERS = [
    0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, -1
]
# Normalized target-multiplier grid used by ATRP and TRP modes.
RANGE_TARGET_MULTIPLIERS = [
0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 4.5, 5
]
# endregion

# region RANKING IN REPORT
# Metric used to rank stop/target combinations.
RANK_METRIC = "CalmarRatio"
# Which RANK_METRIC values rank first: "higher" or "lower" (e.g. drawdown).
RANK_DIRECTION = "higher"
#endregion

# region HEATMAP
# HEATMAP COLOR
# Metric used for heatmap colors.
COLOR_METRIC = "CalmarRatio"
# Which COLOR_METRIC values are better: "higher" (green = high) or "lower"
# (green = low, e.g. drawdown).
COLOR_DIRECTION = "higher"
# Minimum heatmap color value; None uses the observed minimum.
COLOR_MIN: float | None = None #  None = 0.0 
# Maximum heatmap color value; None uses the observed maximum.
COLOR_MAX: float | None = None #  None = 1.0

# TOP METRIC
# Metric printed in the top section of each heatmap cell.
TOP_METRIC = "AnnualizedReturnPct"
# TOP_METRIC threshold for a highlighted heatmap cell.
HIGHLIGHT_TOP_THRESHOLD = 8 # QQQ CAGR
# "above" highlights TOP_METRIC > threshold; "below" highlights < threshold.
HIGHLIGHT_TOP_DIRECTION = "above"

# BOTTOM METRIC
# Metric printed in the bottom section of each heatmap cell.
BOTTOM_METRIC = "MaxDrawdownPct"
# BOTTOM_METRIC threshold for a highlighted heatmap cell.
HIGHLIGHT_BOTTOM_THRESHOLD = 25
# "above" highlights BOTTOM_METRIC > threshold; "below" highlights < threshold.
HIGHLIGHT_BOTTOM_DIRECTION = "below"
#endregion

# region OTHER
# Directory for generated reports and charts.
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "visualization"
# endregion

# region DOCS
# ------------------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------------------
# Available values for TOP_METRIC, BOTTOM_METRIC, COLOR_METRIC, and RANK_METRIC:
#
# Trade count: NumberOfTrades, MaxConsecutiveLosses
#
# Outcomes: WinRatePct, AverageWinPct, AverageLossPct, AverageWinATRP,
# AverageLossATRP
#
# Reward/risk: RealizedRRR, BreakEvenWinRatePct, PlannedRRR,
# PlannedBreakEvenWinRatePct
#
# Total return:
# CumulativeReturnPct compounds each trade as if full capital were reinvested
# sequentially; it is not a portfolio equity curve when trades overlap.
# TotalRealizedR: Sum of realized R

# Expectancy:
# Percentage expectancy: ExpectancyPct, ExpectancyPctPerBar
# ATRP expectancy: ExpectancyATRP, ExpectancyATRPPerBar
# TRP expectancy: ExpectancyTRP, ExpectancyTRPPerBar
# R expectancy: ExpectancyR, ExpectancyRPerBar, ExpectancyRPer100Bars
#
# Drawdown: MaxDrawdownR, MaxDrawdownPct
#
# Return/drawdown (higher is better, calculated per scenario):
# ReturnToDrawdownPct (CumulativeReturnPct / |MaxDrawdownPct|),
# ReturnToDrawdownR (TotalRealizedR / |MaxDrawdownR|)
#
# Annual (calculated per scenario from the length of the downloaded data):
# AnnualizedReturnPct (CAGR of CumulativeReturnPct), CalmarRatio (CAGR /
# |MaxDrawdownPct|), TradesPerYear, RPerYear (TotalRealizedR / years),
# TimeInMarketPct (AverageBarsHeld x NumberOfTrades / total bars; can exceed
# 100 with several open trades), AnnualizedReturnPerBarHeldPct
# (ExpectancyPctPerBar x bars per year: 252 daily, 52 weekly, 12 monthly)
#
# Risk and distance: AverageEntryATRP, AverageEntryTRP, AverageStopDistancePct,
# AverageTargetDistancePct
#
# Duration: AverageBarsHeld
#
# Exit distribution: StopExitPct, TargetExitPct, MovingAverageExitPct,
# EndOfDataExitPct
#
# Costs: AverageCommissionPct

# ------------------------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------------------------
# Available values for STATISTIC:
# min, median, average, max, p1, p5, p25, p75, p95, p99.

# ------------------------------------------------------------------------------
# NOTES
# ------------------------------------------------------------------------------
# 1. When using TRP, RANGE_PERIOD = 1 waits one RANGE_TIMEFRAME before 
# starting simulation
# endregion

# =============================================================================
# endregion


# region INTERNAL REPORT SCHEMAS
# Derived return-to-drawdown metrics, calculated per scenario before the
# statistic is applied: name -> (return column, drawdown column).
# Drawdown is used as an absolute value, and zero drawdown gives NaN.
DERIVED_METRICS = {
    "ReturnToDrawdownPct": ("CumulativeReturnPct", "MaxDrawdownPct"),
    "ReturnToDrawdownR": ("TotalRealizedR", "MaxDrawdownR"),
}
# Derived annual metrics. They need the length of the sampled history.
ANNUAL_METRICS = [
    "AnnualizedReturnPct",
    "CalmarRatio",
    "TradesPerYear",
    "RPerYear",
    "TimeInMarketPct",
    "AnnualizedReturnPerBarHeldPct",
]
ALL_METRICS = [
    *sensitivity.METRIC_COLUMNS,
    *DERIVED_METRICS,
    *ANNUAL_METRICS,
]
# Trading bars per year for each supported interval.
BARS_PER_YEAR = {"1d": 252, "1wk": 52, "1mo": 12}
ANNUAL_SOURCE_COLUMNS = {
    "CumulativeReturnPct",
    "MaxDrawdownPct",
    "NumberOfTrades",
    "TotalRealizedR",
    "AverageBarsHeld",
    "ExpectancyPctPerBar",
}
STATISTICS = {
    "min": ("Minimum", "min"),
    "median": ("Median", "median"),
    "average": ("Average", "mean"),
    "max": ("Maximum", "max"),
    "p1": ("1st percentile", 0.01),
    "p5": ("5th percentile", 0.05),
    "p25": ("25th percentile", 0.25),
    "p75": ("75th percentile", 0.75),
    "p95": ("95th percentile", 0.95),
    "p99": ("99th percentile", 0.99),
}
# Metrics where a bigger number is worse. For these the statistic is taken from
# the opposite tail, so "p5"/"min" always mean the pessimistic end of every
# metric (e.g. highest drawdown) and "p95"/"max" the optimistic end.
# AverageLossPct and AverageLossATRP are stored as negative numbers, so a
# bigger loss is already a lower value and they need no flipping.
LOWER_IS_BETTER = {
    "MaxDrawdownPct",
    "MaxDrawdownR",
    "MaxConsecutiveLosses",
    "AverageCommissionPct",
}
# Metrics that are meaningless (not zero) when a scenario made no trades.
NO_TRADE_NAN_METRICS = ["MaxDrawdownPct", "MaxDrawdownR", "MaxConsecutiveLosses"]
# Tail described by each statistic: "pessimistic", "optimistic" or None.
STATISTIC_TAIL = {
    "min": "pessimistic",
    "p1": "pessimistic",
    "p5": "pessimistic",
    "p25": "pessimistic",
    "p75": "optimistic",
    "p95": "optimistic",
    "p99": "optimistic",
    "max": "optimistic",
}
#endregion


def create_scenario_heatmap(
    results: pd.DataFrame,
    stops: list[float],
    targets: list[float],
    ticker: str,
    start_date: date,
    end_date: date,
    number_of_entries: int,
    requested_entries: int,
    top_metric: str,
    bottom_metric: str,
    color_metric: str,
    vmin: float | None,
    vmax: float | None,
    top_threshold: float,
    top_direction: str,
    bottom_threshold: float,
    bottom_direction: str,
    color_direction: str = "higher",
    ma_enabled: bool = False,
):
    """Create the annotated stop/target heatmap without app dependencies."""

    if not stops:
        raise ValueError("The heatmap requires at least one stop setting.")

    color_values = results.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=color_metric,
    ).reindex(index=targets, columns=stops)
    top_values = results.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=top_metric,
    ).reindex(index=targets, columns=stops)
    bottom_values = results.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=bottom_metric,
    ).reindex(index=targets, columns=stops)
    top_passes = (
        top_values > top_threshold
        if top_direction == "above"
        else top_values < top_threshold
    )
    bottom_passes = (
        bottom_values > bottom_threshold
        if bottom_direction == "above"
        else bottom_values < bottom_threshold
    )
    highlighted = top_passes & bottom_passes
    top_symbol = ">" if top_direction == "above" else "<"
    bottom_symbol = ">" if bottom_direction == "above" else "<"

    matrix = color_values.to_numpy(dtype=float)
    finite_values = matrix[np.isfinite(matrix)]
    if not finite_values.size:
        raise ValueError(f"No finite values for heatmap metric {color_metric!r}.")

    if vmin is None:
        vmin = float(np.min(finite_values))
    if vmax is None:
        vmax = float(np.max(finite_values))
    if vmin == vmax:
        padding = max(abs(vmin) * 0.01, 1e-9)
        vmin -= padding
        vmax += padding

    figure, axis = plt.subplots(
        figsize=(
            max(12, len(stops) * 0.85),
            max(8, len(targets) * 0.55),
        )
    )
    image = axis.imshow(
        matrix,
        cmap="RdYlGn" if color_direction == "higher" else "RdYlGn_r",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axis).set_label(color_metric)
    axis.set_title(
        f"{ticker} | {start_date} to {end_date} | "
        f"{number_of_entries:,} Random Trades "
        f"(up to {requested_entries:,} per Year) | "
        f"{'MA Enabled' if ma_enabled else 'MA Disabled'}\n"
        f"{top_metric} (top), {bottom_metric} (bottom)"
    )
    axis.set_xticks(np.arange(len(stops)))
    axis.set_xticklabels(
        ["NS" if value == -1 else str(value) for value in stops]
    )
    axis.set_yticks(np.arange(len(targets)))
    axis.set_yticklabels(["NT" if value == -1 else str(value) for value in targets])

    formats = {
        "ExpectancyR": ".2f",
        "ExpectancyPct": ".2f",
        "WinRatePct": ".1f",
        "RealizedRRR": ".2f",
        "Score": ".0f",
    }
    top_format = formats.get(top_metric, ".2f")
    bottom_format = formats.get(bottom_metric, ".2f")
    for row in range(len(targets)):
        for column in range(len(stops)):
            top_value = top_values.iloc[row, column]
            bottom_value = bottom_values.iloc[row, column]
            color_value = matrix[row, column]
            if np.isfinite(top_value) and np.isfinite(bottom_value):
                label = (
                    f"{format(top_value, top_format)}\n"
                    f"{format(bottom_value, bottom_format)}"
                )
            else:
                label = "N/A"

            rgba = (
                image.cmap(image.norm(color_value))
                if np.isfinite(color_value)
                else (1, 1, 1, 1)
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
                color="black" if brightness > 0.55 else "white",
                fontsize=8,
            )

    for row in range(len(targets)):
        for column in range(len(stops)):
            stop_value = stops[column]
            target_value = targets[row]
            is_highlighted = bool(highlighted.iloc[row, column])
            if (
                not is_highlighted
                and stop_value != -1
                and target_value != -1
                and np.isclose(stop_value, target_value)
            ):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=(0.20, 0.55, 0.85, 1.0),
                        linewidth=2,
                        zorder=3,
                    )
                )
            if is_highlighted:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                        zorder=4,
                    )
                )

    axis.set_xticks(np.arange(-0.5, len(stops), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    axis.grid(which="minor", color="gray", linewidth=0.4, alpha=0.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    parameter_unit = (
        str(results["ParameterUnit"].iloc[0])
        if "ParameterUnit" in results and not results.empty
        else "parameter units"
    )
    axis.set_xlabel(
        f"Stop-Loss ({parameter_unit})\n"
        "Borders: blue = stop equals target; "
        f"black = {top_metric} {top_symbol} {top_threshold:g} and "
        f"{bottom_metric} {bottom_symbol} {bottom_threshold:g}",
        fontsize=10,
    )
    axis.set_ylabel(f"Target ({parameter_unit})")
    figure.tight_layout()
    return figure


def sample_span(data: pd.DataFrame, interval: str) -> tuple[float, int]:
    """Return the years covered by the data and the bars per year."""

    bars_per_year = BARS_PER_YEAR[interval]
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        years = (data.index[-1] - data.index[0]).days / 365.25
    else:
        years = len(data) / bars_per_year
    if years <= 0:
        raise ValueError("The downloaded data does not span any time.")
    return years, bars_per_year


def add_derived_metrics(
    scenario_results: pd.DataFrame,
    years: float,
    bars_per_year: int,
    total_bars: int,
) -> pd.DataFrame:
    """Return a copy of the scenario results with the derived columns."""

    result = scenario_results.copy()
    required = ANNUAL_SOURCE_COLUMNS.union(
        column for pair in DERIVED_METRICS.values() for column in pair
    )
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(
            "Cannot calculate derived metrics; missing columns: "
            + ", ".join(sorted(missing))
        )

    # A scenario without trades has no drawdown; keep it out of the statistic
    # (like its NaN return) instead of counting it as a perfect zero.
    no_trades = result["NumberOfTrades"] == 0
    for column in NO_TRADE_NAN_METRICS:
        result[column] = result[column].astype("float64").mask(no_trades)

    for name, (return_column, drawdown_column) in DERIVED_METRICS.items():
        drawdown = result[drawdown_column].abs().replace(0, np.nan)
        result[name] = result[return_column] / drawdown

    growth = 1 + result["CumulativeReturnPct"] / 100
    annualized = (growth.where(growth > 0) ** (1 / years) - 1) * 100
    result["AnnualizedReturnPct"] = annualized
    result["CalmarRatio"] = annualized / (
        result["MaxDrawdownPct"].abs().replace(0, np.nan)
    )
    result["TradesPerYear"] = result["NumberOfTrades"] / years
    result["RPerYear"] = result["TotalRealizedR"] / years
    result["TimeInMarketPct"] = (
        result["AverageBarsHeld"] * result["NumberOfTrades"] / total_bars * 100
    )
    result["AnnualizedReturnPerBarHeldPct"] = (
        result["ExpectancyPctPerBar"] * bars_per_year
    )
    return result


def statistic_label(statistic: str) -> str:
    """Return the display name, marking pessimistic/optimistic tails."""

    label = STATISTICS[statistic][0]
    tail = STATISTIC_TAIL.get(statistic)
    return f"{label} ({tail} end)" if tail else label


def flipped_aggregation(aggregation):
    """Return the opposite-tail aggregation for lower-is-better metrics."""

    if isinstance(aggregation, float):
        return 1 - aggregation
    return {"min": "max", "max": "min"}.get(aggregation, aggregation)


def aggregate_scenarios(
    scenario_results: pd.DataFrame,
    statistic: str,
    years: float,
    bars_per_year: int,
    total_bars: int,
) -> pd.DataFrame:
    """Aggregate metrics per stop/target pair across random scenarios."""

    if statistic not in STATISTICS:
        raise ValueError(
            f"Unsupported statistic {statistic!r}. "
            f"Choose from: {', '.join(STATISTICS)}."
        )

    required_columns = {
        *sensitivity.SUMMARY_KEYS,
        "Seed",
        *sensitivity.METRIC_COLUMNS,
    }
    missing_columns = required_columns.difference(scenario_results.columns)
    if missing_columns:
        raise ValueError(
            "Scenario results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    group = add_derived_metrics(
        scenario_results,
        years=years,
        bars_per_year=bars_per_year,
        total_bars=total_bars,
    ).groupby(
        sensitivity.SUMMARY_KEYS,
        sort=False,
        dropna=False,
    )
    aggregation = STATISTICS[statistic][1]

    def aggregate(columns: list[str], how) -> pd.DataFrame:
        metrics = group[columns]
        if isinstance(how, float):
            return metrics.quantile(how)
        return metrics.agg(how)

    normal_columns = [m for m in ALL_METRICS if m not in LOWER_IS_BETTER]
    flipped_columns = [m for m in ALL_METRICS if m in LOWER_IS_BETTER]
    summary = pd.concat(
        [
            aggregate(normal_columns, aggregation),
            aggregate(flipped_columns, flipped_aggregation(aggregation)),
        ],
        axis=1,
    )[ALL_METRICS].reset_index()

    robustness_metric = (
        "ExpectancyATRP"
        if scenario_results["ExitMode"].eq("ATRP").all()
        else (
            "ExpectancyTRP"
            if scenario_results["ExitMode"].eq("TRP").all()
            else "ExpectancyR"
        )
    )
    robustness = sensitivity.summarize_runs(
        scenario_results,
        robustness_metric=robustness_metric,
    )[
        [
            *sensitivity.SUMMARY_KEYS,
            "SeedCount",
            "RunsWithTrades",
            "ExpectancyR_PositivePct",
            "ExpectancyATRP_PositivePct",
            "ExpectancyTRP_PositivePct",
            "RobustRank",
        ]
    ].rename(
        columns={
            "SeedCount": "ScenarioCount",
            "RunsWithTrades": "ScenariosWithTrades",
        }
    )
    return summary.merge(
        robustness,
        on=sensitivity.SUMMARY_KEYS,
        how="left",
        validate="one_to_one",
    )


def parse_values(value: str) -> list[float]:
    """Parse comma-separated numeric stop/target values."""

    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated numeric values."
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one numeric value.")
    if any(not np.isfinite(item) or (item <= 0 and item != -1) for item in values):
        raise argparse.ArgumentTypeError(
            "Values must be positive, except -1 for no stop or target."
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must not contain duplicates.")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Monte Carlo command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize Monte Carlo stop/target results across random-entry "
            "scenarios, using a selected statistic for the heatmap and ranking."
        ),
        epilog=(
            "Sampling: --scenario-count is the number of independent scenario "
            "runs. --scenario-start is the first seed, and scenarios use "
            "consecutive seeds. --trades-per-year is the maximum number of "
            "eligible random trades drawn from each calendar year in every "
            "scenario. For example, --scenario-start 0 --scenario-count 200 "
            "--trades-per-year 25 creates 200 reproducible scenarios; each "
            "year contributes up to 25 trades to each scenario. "
            "--max-open-trades can cap simultaneous positions separately for "
            "each stop/target pair."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=START_DATE,
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=END_DATE,
    )
    parser.add_argument(
        "--interval",
        choices=["1d", "1wk", "1mo"],
        default=INTERVAL,
    )
    exit_mode = parser.add_mutually_exclusive_group()
    exit_mode.add_argument(
        "--exit-mode",
        choices=["ATRP", "TRP", "Percent"],
        default=EXIT_MODE,
    )
    exit_mode.add_argument(
        "--use-atrp",
        action="store_const",
        const="ATRP",
        dest="exit_mode",
    )
    exit_mode.add_argument(
        "--use-trp",
        action="store_true",
        dest="use_trp",
    )
    exit_mode.add_argument(
        "--percent-exits",
        action="store_false",
        dest="use_atrp",
    )
    parser.set_defaults(
        use_atrp=EXIT_MODE == "ATRP",
        use_trp=EXIT_MODE == "TRP",
        atr_timeframe=RANGE_TIMEFRAME,
        atr_period=RANGE_PERIOD,
    )
    parser.add_argument(
        "--atr-timeframe",
        "--range-timeframe",
        dest="range_timeframe",
        choices=["1d", "1wk", "1mo", "3mo"],
        default=RANGE_TIMEFRAME,
        help="Range candle timeframe for ATRP and TRP.",
    )
    parser.add_argument(
        "--atr-period",
        "--range-period",
        dest="range_period",
        type=int,
        default=RANGE_PERIOD,
        help="ATR Wilder period or number of pre-start TRP history bars.",
    )
    parser.add_argument(
        "--scenario-count",
        "--seed-count",
        dest="scenario_count",
        type=int,
        default=SCENARIO_COUNT,
        help=(
            "Number of independent scenario runs. The program creates "
            "consecutive seeds from scenario-start through "
            "scenario-start + scenario-count - 1 "
            "(--seed-count is a legacy alias)."
        ),
    )
    parser.add_argument(
        "--scenario-start",
        "--seed-start",
        dest="scenario_start",
        type=int,
        default=SCENARIO_START,
        help=(
            "Seed for the first scenario. Each following scenario uses the "
            "next consecutive seed (--seed-start is a legacy alias)."
        ),
    )
    parser.add_argument(
        "--trades-per-year",
        "--entry-count",
        dest="trades_per_year",
        type=int,
        default=MAX_TRADES_PER_YEAR,
        help=(
            "Maximum number of eligible random trades to select independently "
            "from each calendar year for every scenario seed. Filters or "
            "insufficient data in a year can reduce the count "
            "(--entry-count is a legacy alias)."
        ),
    )
    parser.add_argument(
        "--max-open-trades",
        type=int,
        default=MAX_OPEN_TRADES,
        help=(
            "Maximum number of simultaneous open trades for each "
            "stop/target pair (default: 1)."
        ),
    )
    entry_filter = parser.add_mutually_exclusive_group()
    entry_filter.add_argument(
        "--ma-entry-enabled",
        action="store_true",
        dest="ma_entry_enabled",
        default=MA_ENTRY_ENABLED,
    )
    entry_filter.add_argument(
        "--no-ma-entry-enabled",
        action="store_false",
        dest="ma_entry_enabled",
    )
    parser.add_argument("--sma-period", type=int, default=SMA_PERIOD)
    parser.add_argument(
        "--direction",
        choices=["Long", "Short", "Both"],
        default=TRADE_DIRECTION,
    )
    parser.add_argument("--commission", type=float, default=COMMISSION)
    parser.add_argument("--slippage", type=float, default=SLIPPAGE)
    parser.add_argument(
        "--non-conservative",
        action="store_false",
        dest="conservative",
        default=CONSERVATIVE,
    )
    parser.add_argument(
        "--ma-exit",
        action="store_true",
        default=MA_EXIT_ENABLED,
    )
    parser.add_argument("--stops", type=parse_values, default=STOPS)
    parser.add_argument("--targets", type=parse_values, default=TARGETS)
    parser.add_argument(
        "--statistic",
        choices=list(STATISTICS),
        default=STATISTIC,
        help=(
            "Statistic across scenarios: min, median, average, max, p1, p5, "
            "p25, p75, p95, or p99."
        ),
    )
    parser.add_argument(
        "--top-metric",
        choices=ALL_METRICS,
        default=TOP_METRIC,
        help="Metric printed at the top of each heatmap cell.",
    )
    parser.add_argument(
        "--bottom-metric",
        choices=ALL_METRICS,
        default=BOTTOM_METRIC,
        help="Metric printed at the bottom of each heatmap cell.",
    )
    parser.add_argument(
        "--color-metric",
        choices=ALL_METRICS,
        default=COLOR_METRIC,
        help="Metric used for heatmap colors.",
    )
    parser.add_argument(
        "--color-direction",
        choices=["higher", "lower"],
        default=COLOR_DIRECTION,
        help="Which color-metric values are better (green).",
    )
    parser.add_argument(
        "--rank-metric",
        choices=ALL_METRICS,
        default=RANK_METRIC,
        help="Metric used to rank stop/target combinations.",
    )
    parser.add_argument(
        "--rank-direction",
        choices=["higher", "lower"],
        default=RANK_DIRECTION,
        help="Which rank-metric values rank first.",
    )
    parser.add_argument("--color-min", type=float, default=COLOR_MIN)
    parser.add_argument("--color-max", type=float, default=COLOR_MAX)
    parser.add_argument(
        "--highlight-top-threshold",
        type=float,
        default=HIGHLIGHT_TOP_THRESHOLD,
        help="Threshold applied to the top metric for highlighting.",
    )
    parser.add_argument(
        "--highlight-top-direction",
        choices=["above", "below"],
        default=HIGHLIGHT_TOP_DIRECTION,
        help="Highlight when the top metric is above or below its threshold.",
    )
    parser.add_argument(
        "--highlight-bottom-threshold",
        type=float,
        default=HIGHLIGHT_BOTTOM_THRESHOLD,
        help="Threshold applied to the bottom metric for highlighting.",
    )
    parser.add_argument(
        "--highlight-bottom-direction",
        choices=["above", "below"],
        default=HIGHLIGHT_BOTTOM_DIRECTION,
        help=(
            "Highlight when the bottom metric is above or below its "
            "threshold."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes for scenario simulation (1 = single process, 0 = all CPU cores).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser


def simulate_representative_scenario(
    data: pd.DataFrame,
    scenario: int,
    args: argparse.Namespace,
    stops: list[float],
    targets: list[float],
    atrp_values: pd.Series | np.ndarray | None,
    trp_values: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create app-style entries, trade details, and direction statistics."""

    trades_per_year = getattr(
        args,
        "trades_per_year",
        getattr(args, "entry_count", None),
    )
    if trades_per_year is None:
        raise ValueError("A trades-per-year setting is required.")

    atrp_array = None
    trp_array = None
    if args.use_atrp:
        if atrp_values is None:
            raise ValueError("ATRP values are required when ATRP exits are enabled.")
        atrp_array = np.asarray(atrp_values, dtype=float)
        if len(atrp_array) != len(data):
            raise ValueError("ATRP values must align one-to-one with trading bars.")
    elif getattr(args, "use_trp", False):
        if trp_values is None:
            raise ValueError("TRP values are required when TRP exits are enabled.")
        trp_array = np.asarray(trp_values, dtype=float)
        if len(trp_array) != len(data):
            raise ValueError("TRP values must align one-to-one with trading bars.")

    entries = sensitivity.select_entries(
        data=data,
        number_of_entries=trades_per_year,
        seed=scenario,
        sma_period=args.sma_period,
        trade_direction=args.direction,
        ma_enabled=args.ma_entry_enabled,
        atrp_values=atrp_array,
        trp_values=trp_array,
        trades_per_year=trades_per_year,
    )
    sma_values = (
        data["Close"]
        .rolling(
            window=int(args.sma_period),
            min_periods=int(args.sma_period),
        )
        .mean()
        .to_numpy()
        if args.ma_exit
        else None
    )

    trade_records = []
    directional_records = []
    exit_mode = (
        "ATRP"
        if args.use_atrp
        else ("TRP" if getattr(args, "use_trp", False) else "Percent")
    )
    for stop in stops:
        for target in targets:
            candidate_trades = []
            for entry in entries.itertuples(index=False):
                trade = engine.simulate_trade(
                    data=data,
                    entry_index=int(entry.EntryIndex),
                    stop_parameter=stop,
                    target_parameter=target,
                    commission_pct=args.commission,
                    slippage_pct=args.slippage,
                    conservative=args.conservative,
                    direction=str(entry.Direction),
                    sma_values=sma_values,
                    entry_atrp=(
                        float(entry.EntryATRP) if args.use_atrp else None
                    ),
                    entry_trp=(
                        float(entry.EntryTRP)
                        if getattr(args, "use_trp", False)
                        else None
                    ),
                    use_atrp=args.use_atrp,
                    use_trp=getattr(args, "use_trp", False),
                )
                candidate_trades.append(
                    {
                        "Direction": str(entry.Direction),
                        **trade,
                    }
                )
            combination_trades = sensitivity.limit_concurrent_trades(
                candidate_trades,
                getattr(args, "max_open_trades", MAX_OPEN_TRADES),
            )
            for trade in combination_trades:
                trade_records.append(
                    {
                        "Scenario": scenario,
                        "StopParameter": stop,
                        "TargetParameter": target,
                        **trade,
                    }
                )

            combination_frame = pd.DataFrame(
                combination_trades,
                columns=(
                    ["Direction", *sensitivity.TRADE_METRIC_COLUMNS]
                    if not combination_trades
                    else None
                ),
            )
            for direction in ("All", "Long", "Short"):
                selected_trades = (
                    combination_frame
                    if direction == "All"
                    else combination_frame.loc[
                        combination_frame["Direction"] == direction
                    ]
                )
                directional_records.append(
                    {
                        "Scenario": scenario,
                        "Direction": direction,
                        **engine.calculate_metrics(
                            selected_trades.drop(columns="Direction"),
                            stop,
                            target,
                            exit_mode,
                        ),
                    }
                )

    return (
        entries,
        pd.DataFrame.from_records(trade_records),
        pd.DataFrame.from_records(directional_records),
    )


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid configuration before downloading market data."""

    if not args.ticker.strip():
        raise ValueError("Ticker must not be empty.")
    if args.start_date >= args.end_date:
        raise ValueError("--start-date must be earlier than --end-date.")
    if args.scenario_count < 1:
        raise ValueError("--scenario-count must be at least 1.")
    if args.trades_per_year < 1:
        raise ValueError("--trades-per-year must be at least 1.")
    if (
        args.max_open_trades is not None
        and args.max_open_trades < 1
    ):
        raise ValueError("--max-open-trades must be at least 1.")
    if args.sma_period < 1:
        raise ValueError("--sma-period must be at least 1.")
    if (args.use_atrp or getattr(args, "use_trp", False)) and args.range_period < 1:
        raise ValueError("--range-period must be at least 1.")
    if args.commission < 0 or args.slippage < 0:
        raise ValueError("--commission and --slippage must not be negative.")
    for name, allowed in (
        ("highlight_top_direction", ("above", "below")),
        ("highlight_bottom_direction", ("above", "below")),
        ("color_direction", ("higher", "lower")),
        ("rank_direction", ("higher", "lower")),
    ):
        if getattr(args, name) not in allowed:
            raise ValueError(
                f"{name} must be one of: {', '.join(allowed)}."
            )
    if args.color_min is not None and args.color_max is not None:
        if args.color_min >= args.color_max:
            raise ValueError("--color-min must be less than --color-max.")


def main() -> None:
    args = build_parser().parse_args()
    args.atr_timeframe = args.range_timeframe
    args.atr_period = args.range_period
    if args.exit_mode != EXIT_MODE:
        args.use_atrp = args.exit_mode == "ATRP"
        args.use_trp = args.exit_mode == "TRP"
    elif getattr(args, "use_trp", False):
        args.exit_mode = "TRP"
        args.use_atrp = False
    elif not args.use_atrp:
        args.exit_mode = "Percent"
        args.use_trp = False
    validate_args(args)

    stops = args.stops or STOPS or (
        RANGE_STOP_MULTIPLIERS
        if args.use_atrp or getattr(args, "use_trp", False)
        else DEFAULT_PERCENT_STOPS
    )
    targets = args.targets or TARGETS or (
        RANGE_TARGET_MULTIPLIERS
        if args.use_atrp or getattr(args, "use_trp", False)
        else DEFAULT_PERCENT_TARGETS
    )
    start_time = time.perf_counter()

    print(f"Downloading {args.ticker} trading data...")
    data = sensitivity.download_data(
        args.ticker,
        args.start_date,
        args.end_date,
        args.interval,
    )
    atrp_values = None
    trp_values = None
    if args.use_atrp:
        atr_start_date = sensitivity.atr_warmup_start_date(
            args.start_date,
            args.atr_timeframe,
            args.atr_period,
        )
        print(
            f"Downloading {args.atr_timeframe} ATR history "
            f"from {atr_start_date}..."
        )
        atr_data = sensitivity.download_data(
            args.ticker,
            atr_start_date,
            args.end_date,
            args.atr_timeframe,
        )
        atrp_values = sensitivity.align_atrp_to_trading_data(
            trading_data=data,
            atr_data=atr_data,
            trading_timeframe=args.interval,
            atr_timeframe=args.atr_timeframe,
            atr_period=args.atr_period,
        )
    elif getattr(args, "use_trp", False):
        trp_start_date = sensitivity.atr_warmup_start_date(
            args.start_date,
            args.range_timeframe,
            args.range_period,
        )
        print(
            f"Downloading {args.range_timeframe} TRP history "
            f"from {trp_start_date}..."
        )
        range_data = sensitivity.download_data(
            args.ticker,
            trp_start_date,
            args.end_date,
            args.range_timeframe,
        )
        trp_values = sensitivity.align_trp_to_trading_data(
            trading_data=data,
            range_data=range_data,
            trading_timeframe=args.interval,
            range_timeframe=args.range_timeframe,
        )

    scenarios = range(
        args.scenario_start,
        args.scenario_start + args.scenario_count,
    )

    def report_progress(
        completed: int,
        total: int,
        scenario: int,
        _ma_enabled: bool,
    ) -> None:
        print(f"Completed {completed}/{total} scenarios.")

    scenario_results = engine.run_sensitivity(
        data=data,
        seeds=scenarios,
        number_of_entries=args.trades_per_year,
        trades_per_year=args.trades_per_year,
        stops=stops,
        targets=targets,
        sma_period=args.sma_period,
        trade_direction=args.direction,
        commission=args.commission,
        slippage=args.slippage,
        conservative=args.conservative,
        ma_entry_enabled=args.ma_entry_enabled,
        ma_exit_enabled=args.ma_exit,
        use_atrp=args.use_atrp,
        atrp_values=atrp_values,
        use_trp=getattr(args, "use_trp", False),
        trp_values=trp_values,
        atr_timeframe=args.atr_timeframe,
        atr_period=args.atr_period,
        progress_callback=report_progress,
        ticker=args.ticker.strip().upper(),
        trading_timeframe=args.interval,
        max_open_trades=args.max_open_trades,
        workers=args.workers,
    )
    years, bars_per_year = sample_span(data, args.interval)
    summary = aggregate_scenarios(
        scenario_results,
        args.statistic,
        years=years,
        bars_per_year=bars_per_year,
        total_bars=len(data),
    )
    if args.ma_entry_enabled:
        summary = summary.loc[summary["EntryMAEnabled"]].copy()

    detail_scenario = args.scenario_start
    entries, trades, directional_statistics = simulate_representative_scenario(
        data=data,
        scenario=detail_scenario,
        args=args,
        stops=stops,
        targets=targets,
        atrp_values=atrp_values,
        trp_values=trp_values,
    )

    summary = summary.rename(
        columns={
            "StopParameter": "StopLossPct",
            "TargetParameter": "TargetPct",
        }
    )
    ranked = (
        summary.sort_values(
            args.rank_metric,
            ascending=args.rank_direction == "lower",
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))

    selected_statistic = statistic_label(args.statistic)
    heatmap_statistic = STATISTICS[args.statistic][0]
    ticker = args.ticker.strip().upper()
    heatmap = create_scenario_heatmap(
        results=summary,
        stops=stops,
        targets=targets,
        ticker=ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        number_of_entries=len(entries),
        requested_entries=args.trades_per_year,
        top_metric=args.top_metric,
        bottom_metric=args.bottom_metric,
        color_metric=args.color_metric,
        vmin=args.color_min,
        vmax=args.color_max,
        top_threshold=args.highlight_top_threshold,
        top_direction=args.highlight_top_direction,
        bottom_threshold=args.highlight_bottom_threshold,
        bottom_direction=args.highlight_bottom_direction,
        color_direction=args.color_direction,
        ma_enabled=args.ma_entry_enabled,
    )
    heatmap.axes[0].set_title(
        heatmap.axes[0].get_title()
        + f"\nAcross {args.scenario_count} scenarios: {heatmap_statistic}"
    )
    heatmap.axes[1].set_ylabel(
        f"{args.color_metric} ({heatmap_statistic})"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ticker_slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in ticker
    ).strip("_")
    mode_slug = (
        f"{args.exit_mode.lower()}{args.range_timeframe}{args.range_period}"
        if args.use_atrp
        else "pct"
    )
    open_trades_slug = (
        f"_open{args.max_open_trades}"
        if args.max_open_trades is not None
        else ""
    )
    filter_slug = "ma1" if args.ma_entry_enabled else "ma0"
    statistic_slug = {
        "median": "med",
        "average": "avg",
    }.get(args.statistic, args.statistic)
    output_slug = (
        f"{ticker_slug}_{args.interval}_{mode_slug}_"
        f"n{args.scenario_count}_s{args.scenario_start}_"
        f"{filter_slug}{open_trades_slug}_{statistic_slug}"
    )
    representative_scenario = f"scenario{detail_scenario}"
    files = {
        "heatmap": args.output_dir / f"{output_slug}_heatmap.png",
        "summary": args.output_dir / f"{output_slug}_summary.csv",
        "scenarios": args.output_dir / f"{output_slug}_scenarios.csv",
        "entries": (
            args.output_dir
            / f"{output_slug}_{representative_scenario}_entries.csv"
        ),
        "trades": (
            args.output_dir
            / f"{output_slug}_{representative_scenario}_trades.csv"
        ),
        "directions": (
            args.output_dir
            / f"{output_slug}_{representative_scenario}_directions.csv"
        ),
    }
    heatmap.savefig(files["heatmap"], dpi=180, bbox_inches="tight")
    plt.close(heatmap)

    ranked.to_csv(files["summary"], index=False)
    scenario_results.rename(columns={"Seed": "Scenario"}).to_csv(
        files["scenarios"],
        index=False,
    )
    entries.to_csv(files["entries"], index=False)
    trades.to_csv(files["trades"], index=False)
    directional_statistics.to_csv(files["directions"], index=False)

    elapsed = time.perf_counter() - start_time
    print(
        f"\nCompleted {args.scenario_count} scenarios using the "
        f"{selected_statistic.lower()} across scenarios."
    )
    print(
        f"Bars: {len(data):,} | entries in scenario {detail_scenario}: "
        f"{len(entries):,} (up to {args.trades_per_year:,}/year) | "
        f"combinations: {len(summary):,} | run time: {elapsed:.1f}s"
    )
    for label, path in files.items():
        print(f"{label.title()}: {path}")
    print(
        "\nTop stop/target combinations:\n"
        + ranked[
            [
                "Rank",
                "StopLossPct",
                "TargetPct",
                "ScenarioCount",
                "ScenariosWithTrades",
                args.rank_metric,
                args.bottom_metric,
                "ExpectancyR_PositivePct",
                "RobustRank",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()