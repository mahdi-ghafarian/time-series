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


# =============================================================================
# USER CONFIGURATION
# Edit these defaults or override them with command-line options.
# =============================================================================
TICKER = "QQQ"
START_DATE = date(2000, 1, 1)
END_DATE = date(2026, 12, 31)
INTERVAL = "1d"
USE_ATRP = True
ATR_TIMEFRAME = "3mo"
ATR_PERIOD = 12
SCENARIO_COUNT = 200
SCENARIO_START = 0
ENTRY_COUNT = 25
MA_ENTRY_ENABLED = False
SMA_PERIOD = 200
TRADE_DIRECTION = "Long"
COMMISSION = 0.0
SLIPPAGE = 0.0
CONSERVATIVE = True
MA_EXIT_ENABLED = False
STOPS: list[float] | None = None
TARGETS: list[float] | None = None
DEFAULT_PERCENT_STOPS = [
    5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 90, -1,
]
DEFAULT_PERCENT_TARGETS = [
    5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, -1,
]
DEFAULT_ATRP_STOP_MULTIPLIERS = [
    0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, -1,
]
DEFAULT_ATRP_TARGET_MULTIPLIERS = [
    0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, -1,
]
STATISTIC = "p5"
TOP_METRIC = "ExpectancyR"
BOTTOM_METRIC = "WinRatePct"
COLOR_METRIC = "ExpectancyR"
RANK_METRIC = "ExpectancyR"
COLOR_MIN: float | None = 0.0
COLOR_MAX: float | None = 1.0
HIGHLIGHT_WIN_RATE = 50.0
HIGHLIGHT_EXPECTANCY_R = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "visualization"


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
    win_rate_minimum: float,
    expectancy_r_minimum: float,
):
    """Create the annotated stop/target heatmap without app dependencies."""

    stops = [stop for stop in stops if stop != -1]
    if not stops:
        raise ValueError("The heatmap requires at least one stop other than -1.")

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
    highlighted = results.assign(
        _Highlighted=(
            (results["WinRatePct"] > win_rate_minimum)
            & (results["ExpectancyR"] > expectancy_r_minimum)
        )
    ).pivot(
        index="TargetPct",
        columns="StopLossPct",
        values="_Highlighted",
    ).reindex(index=targets, columns=stops)

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
        cmap="RdYlGn",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axis).set_label(color_metric)
    axis.set_title(
        f"{ticker} | {start_date} to {end_date} | "
        f"{number_of_entries:,} / {requested_entries:,} Random Entries\n"
        f"{top_metric} (top), {bottom_metric} (bottom)"
    )
    axis.set_xticks(np.arange(len(stops)))
    axis.set_xticklabels([str(value) for value in stops])
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
                        linewidth=3,
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
        "black = meets highlight thresholds",
        fontsize=10,
    )
    axis.set_ylabel(f"Target ({parameter_unit})")
    figure.tight_layout()
    return figure


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


def aggregate_scenarios(
    scenario_results: pd.DataFrame,
    statistic: str,
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

    group = scenario_results.groupby(
        sensitivity.SUMMARY_KEYS,
        sort=False,
        dropna=False,
    )
    aggregation = STATISTICS[statistic][1]
    metrics = group[sensitivity.METRIC_COLUMNS]
    if isinstance(aggregation, float):
        summary = metrics.quantile(aggregation).reset_index()
    else:
        summary = metrics.agg(aggregation).reset_index()

    robustness_metric = (
        "ExpectancyATRP"
        if scenario_results["ExitMode"].eq("ATRP").all()
        else "ExpectancyR"
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
        )
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
        "--use-atrp",
        action="store_true",
        dest="use_atrp",
        default=USE_ATRP,
    )
    exit_mode.add_argument(
        "--percent-exits",
        action="store_false",
        dest="use_atrp",
    )
    parser.add_argument(
        "--atr-timeframe",
        choices=["1d", "1wk", "1mo", "3mo"],
        default=ATR_TIMEFRAME,
    )
    parser.add_argument("--atr-period", type=int, default=ATR_PERIOD)
    parser.add_argument(
        "--scenario-count",
        "--seed-count",
        dest="scenario_count",
        type=int,
        default=SCENARIO_COUNT,
        help="Number of scenarios to simulate (--seed-count is a legacy alias).",
    )
    parser.add_argument(
        "--scenario-start",
        "--seed-start",
        dest="scenario_start",
        type=int,
        default=SCENARIO_START,
        help="First scenario number (--seed-start is a legacy alias).",
    )
    parser.add_argument("--entry-count", type=int, default=ENTRY_COUNT)
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
        choices=sensitivity.METRIC_COLUMNS,
        default=TOP_METRIC,
        help="Metric printed at the top of each heatmap cell.",
    )
    parser.add_argument(
        "--bottom-metric",
        choices=sensitivity.METRIC_COLUMNS,
        default=BOTTOM_METRIC,
        help="Metric printed at the bottom of each heatmap cell.",
    )
    parser.add_argument(
        "--color-metric",
        choices=sensitivity.METRIC_COLUMNS,
        default=COLOR_METRIC,
        help="Metric used for heatmap colors.",
    )
    parser.add_argument(
        "--rank-metric",
        choices=sensitivity.METRIC_COLUMNS,
        default=RANK_METRIC,
        help="Metric used to rank stop/target combinations.",
    )
    parser.add_argument("--color-min", type=float, default=COLOR_MIN)
    parser.add_argument("--color-max", type=float, default=COLOR_MAX)
    parser.add_argument(
        "--highlight-win-rate",
        type=float,
        default=HIGHLIGHT_WIN_RATE,
    )
    parser.add_argument(
        "--highlight-expectancy-r",
        type=float,
        default=HIGHLIGHT_EXPECTANCY_R,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create app-style entries, trade details, and direction statistics."""

    atrp_array = None
    if args.use_atrp:
        if atrp_values is None:
            raise ValueError("ATRP values are required when ATRP exits are enabled.")
        atrp_array = np.asarray(atrp_values, dtype=float)
        if len(atrp_array) != len(data):
            raise ValueError("ATRP values must align one-to-one with trading bars.")

    entries = sensitivity.select_entries(
        data=data,
        number_of_entries=args.entry_count,
        seed=scenario,
        sma_period=args.sma_period,
        trade_direction=args.direction,
        ma_enabled=args.ma_entry_enabled,
        atrp_values=atrp_array,
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
    exit_mode = "ATRP" if args.use_atrp else "Percent"
    for stop in stops:
        for target in targets:
            combination_trades = []
            for entry in entries.itertuples(index=False):
                trade = sensitivity.simulate_trade(
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
                    use_atrp=args.use_atrp,
                )
                trade_records.append(
                    {
                        "Scenario": scenario,
                        "StopParameter": stop,
                        "TargetParameter": target,
                        "Direction": str(entry.Direction),
                        **trade,
                    }
                )
                combination_trades.append(
                    {
                        "Direction": str(entry.Direction),
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
                        **sensitivity.calculate_metrics(
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
    if args.entry_count < 1:
        raise ValueError("--entry-count must be at least 1.")
    if args.sma_period < 1:
        raise ValueError("--sma-period must be at least 1.")
    if args.use_atrp and args.atr_period < 1:
        raise ValueError("--atr-period must be at least 1.")
    if args.commission < 0 or args.slippage < 0:
        raise ValueError("--commission and --slippage must not be negative.")
    if args.color_min is not None and args.color_max is not None:
        if args.color_min >= args.color_max:
            raise ValueError("--color-min must be less than --color-max.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    stops = args.stops or STOPS or (
        DEFAULT_ATRP_STOP_MULTIPLIERS
        if args.use_atrp
        else DEFAULT_PERCENT_STOPS
    )
    targets = args.targets or TARGETS or (
        DEFAULT_ATRP_TARGET_MULTIPLIERS
        if args.use_atrp
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

    scenarios = range(
        args.scenario_start,
        args.scenario_start + args.scenario_count,
    )

    def report_progress(
        completed: int,
        total: int,
        scenario: int,
        ma_enabled: bool,
    ) -> None:
        filter_state = "enabled" if ma_enabled else "disabled"
        print(
            f"Completed {completed}/{total} scenarios "
            f"(scenario {scenario}, entry MA filter {filter_state})."
        )

    scenario_results = sensitivity.run_sensitivity(
        data=data,
        seeds=scenarios,
        number_of_entries=args.entry_count,
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
        atr_timeframe=args.atr_timeframe,
        atr_period=args.atr_period,
        progress_callback=report_progress,
        ticker=args.ticker.strip().upper(),
        trading_timeframe=args.interval,
    )
    summary = aggregate_scenarios(scenario_results, args.statistic)
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
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))

    selected_statistic = STATISTICS[args.statistic][0]
    ticker = args.ticker.strip().upper()
    heatmap = create_scenario_heatmap(
        results=summary,
        stops=stops,
        targets=targets,
        ticker=ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        number_of_entries=len(entries),
        requested_entries=args.entry_count,
        top_metric=args.top_metric,
        bottom_metric=args.bottom_metric,
        color_metric=args.color_metric,
        vmin=args.color_min,
        vmax=args.color_max,
        win_rate_minimum=args.highlight_win_rate,
        expectancy_r_minimum=args.highlight_expectancy_r,
    )
    heatmap.axes[0].set_title(
        heatmap.axes[0].get_title()
        + f"\nAcross {args.scenario_count} scenarios: {selected_statistic}"
    )
    heatmap.axes[1].set_ylabel(
        f"{args.color_metric} ({selected_statistic})"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ticker_slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in ticker
    ).strip("_")
    mode_slug = (
        f"atrp{args.atr_timeframe}{args.atr_period}"
        if args.use_atrp
        else "pct"
    )
    filter_slug = "ma1" if args.ma_entry_enabled else "ma0"
    statistic_slug = {
        "median": "med",
        "average": "avg",
    }.get(args.statistic, args.statistic)
    output_slug = (
        f"{ticker_slug}_{args.interval}_{mode_slug}_"
        f"n{args.scenario_count}_s{args.scenario_start}_"
        f"{filter_slug}_{statistic_slug}"
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
        f"{len(entries):,}/{args.entry_count:,} | "
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
