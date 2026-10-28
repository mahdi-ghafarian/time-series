
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from matplotlib.patches import Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "QQQ"
START_DATE = "2010-01-01"
END_DATE = None                # None = through the latest available data
INTERVAL = "1d"

NUMBER_OF_RANDOM_ENTRIES = 1000
RANDOM_SEED = 27

# Metrics:
# StopLossPct, TargetPct, NumberOfTrades, WinRatePct,
# AverageWinPct, AverageLossPct, RealizedRRR,
# BreakEvenWinRatePct, PlannedRRR, PlannedBreakEvenWinRatePct,
# ExpectancyPct, ExpectancyR, AverageBarsHeld,
# StopExitPct, TargetExitPct, EndOfDataExitPct,AverageCommissionPct,
# Score

HEATMAP_TOP_METRIC = "ExpectancyR"
HEATMAP_BOTTOM_METRIC = "WinRatePct"

HEATMAP_COLOR_METRIC = "ExpectancyR"
HEATMAP_VMIN = 0        # None = Use min value
HEATMAP_VMAX = 1        # None = Use max value

HIGHLIGHT_WR_MIN = 50
HIGHLIGHT_EXPECTANCYR_MIN = 0.5

RANK_METRIC = "ExpectancyR"         # CSV file

# Commission and slippage are percentages.
# Example: 0.05 means 0.05%, not 5%.
COMMISSION_PCT = 0.0
SLIPPAGE_PCT = 0.0

# STOCK
STOP_LOSS_VALUES = [
    1, 2, 3, 4, 5, 7.5, 10,
    15, 20, 25, 30, 35, 40, 45, 50,
    -1  # NS
]

TARGET_VALUES = [
    1, 2, 3, 4, 5, 7.5, 10,
    15, 20, 25, 30, 35, 40, 45, 50, 100,
    -1  # NT
]

# BTC-USD
# STOP_LOSS_VALUES = [
#     5, 10, 15, 20, 25, 30, 35, 40, 45,
#     50, 60, 70, 80, 90,
#     -1  # NS
# ]

# TARGET_VALUES = [
#     5, 10, 15, 20, 25, 30, 35, 40, 45,
#     50, 60, 70, 80, 90, 100, 200, 300, 400, 500,
#     -1  # NT
# ]


# If both stop and target are reached in the same candle,
# assume the stop was hit first.
CONSERVATIVE_SAME_CANDLE = True

# Metric formating
HEATMAP_METRIC_FORMATS = {
    "ExpectancyR": ".2f",
    "ExpectancyPct": ".2f",
    "WinRatePct": ".1f",
    "RealizedRRR": ".2f",
    "Score": ".0f",
}

# Output files are saved alongside this script.
# In a Jupyter notebook, files are saved to the current directory.
try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()


# ============================================================
# DOWNLOAD AND PREPARE DATA
# ============================================================

def download_data():
    print(f"Downloading {TICKER} data...")

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "No data downloaded. Check the ticker, dates, and interval."
        )

    # Handle yfinance MultiIndex columns, which may occur for
    # single-ticker downloads in some yfinance versions.
    if isinstance(data.columns, pd.MultiIndex):
        if TICKER in data.columns.get_level_values(-1):
            data = data.xs(TICKER, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_columns if c not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data[required_columns].copy()
    data = data.dropna().sort_index()

    if len(data) < 2:
        raise ValueError("At least two valid price bars are required.")

    if (data[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Price data contains zero or negative prices.")

    print(f"Downloaded {len(data):,} bars.")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data


# ============================================================
# SELECT RANDOM ENTRY BARS
# ============================================================

def select_random_entries(data):
    """
    Select entry bars once and reuse them for every stop/target
    combination, allowing a fairer comparison.

    Entries are at the selected bar's close. A trade becomes
    eligible for stop/target exits on the following bar.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    number_of_entries = min(
        NUMBER_OF_RANDOM_ENTRIES,
        len(data) - 1
    )

    entry_indices = rng.choice(
        np.arange(len(data) - 1),
        size=number_of_entries,
        replace=False
    )

    entry_indices = np.sort(entry_indices)

    entries = pd.DataFrame({
        "EntryIndex": entry_indices,
        "EntryDate": data.index[entry_indices],
        "EntryClose": data["Close"].to_numpy()[entry_indices]
    })

    entries.to_csv(
        OUTPUT_DIR / "random_entry_bars.csv",
        index=False
    )

    print(f"Selected {len(entries)} random entry bars.")

    return entries


# ============================================================
# SIMULATE ONE TRADE
# ============================================================

def simulate_trade(
    data,
    entry_index,
    stop_loss_pct,
    target_pct
):
    """
    Simulate a long trade.

    Entry:
        Selected bar's close, with buy-side slippage.

    Exit:
        Stop, target, or final available close.

    Stop/target checks start on the bar after entry.

    If both stop and target are touched in the same candle,
    the configured conservative assumption is applied.

    Net return includes slippage and commissions.
    """

    close_prices = data["Close"].to_numpy()
    high_prices = data["High"].to_numpy()
    low_prices = data["Low"].to_numpy()
    dates = data.index

    entry_reference_price = float(close_prices[entry_index])
    no_stop = stop_loss_pct == -1
    no_target = target_pct == -1

    # Buy-side slippage increases the entry execution price.
    entry_execution_price = (
        entry_reference_price * (1 + SLIPPAGE_PCT / 100)
    )
    stop_price = None
    target_price = None

    if not no_stop:
        stop_price = (
            entry_reference_price *
            (1 - float(stop_loss_pct) / 100)
        )

    if not no_target:
        target_price = (
            entry_reference_price *
            (1 + float(target_pct) / 100)
        )

    exit_index = len(data) - 1
    exit_reference_price = float(close_prices[exit_index])
    exit_reason = "EndOfData"

    # The entry candle is deliberately excluded from exit checks.
    for i in range(entry_index + 1, len(data)):
        bar_low = float(low_prices[i])
        bar_high = float(high_prices[i])

        stop_hit = (
            stop_price is not None
            and bar_low <= stop_price
        )

        target_hit = (
            target_price is not None
            and bar_high >= target_price
        )

        if stop_hit and target_hit:
            if CONSERVATIVE_SAME_CANDLE:
                exit_reference_price = stop_price
                exit_reason = "Stop"
            else:
                exit_reference_price = target_price
                exit_reason = "Target"

            exit_index = i
            break

        if stop_hit:
            exit_reference_price = stop_price
            exit_reason = "Stop"
            exit_index = i
            break

        if target_hit:
            exit_reference_price = target_price
            exit_reason = "Target"
            exit_index = i
            break

    # Sell-side slippage decreases the exit execution price.
    exit_execution_price = (
        exit_reference_price * (1 - SLIPPAGE_PCT / 100)
    )

    gross_return_pct = (
        exit_execution_price / entry_execution_price - 1
    ) * 100

    # Commission charged on entry and exit, expressed as a
    # percentage of the trade's notional value on each side.
    total_commission_pct = 2 * COMMISSION_PCT

    net_return_pct = gross_return_pct - total_commission_pct

    # One R is defined as the planned stop distance measured
    # from the entry reference price. For no-stop trades,
    # realized R is undefined.
    risk_pct = 100 if no_stop else float(stop_loss_pct)
    realized_r = net_return_pct / risk_pct

    bars_held = exit_index - entry_index

    return {
        "EntryDate": dates[entry_index],
        "ExitDate": dates[exit_index],
        "EntryIndex": entry_index,
        "ExitIndex": exit_index,
        "EntryReferencePrice": entry_reference_price,
        "EntryExecutionPrice": entry_execution_price,
        "ExitReferencePrice": exit_reference_price,
        "ExitExecutionPrice": exit_execution_price,
        "StopLossPct": stop_loss_pct,
        "TargetPct": target_pct,
        "GrossReturnPct": gross_return_pct,
        "NetReturnPct": net_return_pct,
        "RealizedR": realized_r,
        "BarsHeld": bars_held,
        "ExitReason": exit_reason,
        "CommissionPct": total_commission_pct
    }


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(trades, stop_loss_pct, target_pct):
    net_returns = trades["NetReturnPct"].to_numpy(dtype=float)

    wins = net_returns > 0
    losses = net_returns < 0

    win_returns = net_returns[wins]
    loss_returns = net_returns[losses]

    win_rate = float(wins.mean() * 100) if len(net_returns) else np.nan

    average_win = (
        float(win_returns.mean()) if len(win_returns) else np.nan
    )

    average_loss = (
        float(loss_returns.mean()) if len(loss_returns) else np.nan
    )

    expectancy_pct = (
        float(net_returns.mean()) if len(net_returns) else np.nan
    )

    # Expectancy in R:
    # average net return per trade / planned stop-loss risk.
    risk_pct = (
        100 if stop_loss_pct == -1
        else float(stop_loss_pct)
    )

    expectancy_r = expectancy_pct / risk_pct

    if target_pct == -1:
        planned_rrr = np.nan
    else:
        planned_rrr = (
            float(target_pct) / risk_pct
        )

    # Realized reward-to-risk ratio based on average winning
    # and losing trades. Average loss is negative.
    if (
        len(win_returns) > 0
        and len(loss_returns) > 0
        and average_loss != 0
    ):
        realized_rrr = average_win / abs(average_loss)
    else:
        realized_rrr = np.nan

    # The break-even win rate implied by average win/loss sizes.
    if (
        np.isfinite(average_win)
        and np.isfinite(average_loss)
        and average_win > 0
        and average_loss < 0
    ):
        break_even_win_rate = (
            abs(average_loss)
            / (average_win + abs(average_loss))
        ) * 100
    else:
        break_even_win_rate = np.nan

    if np.isfinite(planned_rrr) and planned_rrr > 0:
        planned_break_even_win_rate = (
            1 / (1 + planned_rrr)
        ) * 100
    else:
        planned_break_even_win_rate = np.nan

    exit_counts = trades["ExitReason"].value_counts()
    total_trades = len(trades)

    stop_exit_pct = (
        exit_counts.get("Stop", 0) / total_trades * 100
        if total_trades else np.nan
    )

    target_exit_pct = (
        exit_counts.get("Target", 0) / total_trades * 100
        if total_trades else np.nan
    )

    end_of_data_exit_pct = (
        exit_counts.get("EndOfData", 0) / total_trades * 100
        if total_trades else np.nan
    )

    return {
        "StopLossPct": stop_loss_pct,
        "TargetPct": target_pct,
        "NumberOfTrades": total_trades,
        "WinRatePct": win_rate,
        "AverageWinPct": average_win,
        "AverageLossPct": average_loss,
        "RealizedRRR": realized_rrr,
        "BreakEvenWinRatePct": break_even_win_rate,
        "PlannedRRR": planned_rrr,
        "PlannedBreakEvenWinRatePct": planned_break_even_win_rate,
        "ExpectancyPct": expectancy_pct,
        "ExpectancyR": expectancy_r,
        "AverageBarsHeld": (
            float(trades["BarsHeld"].mean())
            if total_trades else np.nan
        ),
        "StopExitPct": stop_exit_pct,
        "TargetExitPct": target_exit_pct,
        "EndOfDataExitPct": end_of_data_exit_pct,
        "AverageCommissionPct": (
            float(trades["CommissionPct"].mean())
            if total_trades else np.nan
        )
    }


# ============================================================
# COMPARE ALL STOP/TARGET COMBINATIONS
# ============================================================

def compare_combinations(data, entries):
    total_combinations = (
        len(STOP_LOSS_VALUES) * len(TARGET_VALUES)
    )

    results = []
    all_trade_details = []

    start_time = time.perf_counter()

    with tqdm(
        total=total_combinations,
        desc="Testing combinations",
        unit="combination"
    ) as progress:

        for stop_loss_pct in STOP_LOSS_VALUES:
            for target_pct in TARGET_VALUES:

                combination_trades = []

                for entry in entries.itertuples(index=False):
                    trade = simulate_trade(
                        data=data,
                        entry_index=int(entry.EntryIndex),
                        stop_loss_pct=stop_loss_pct,
                        target_pct=target_pct
                    )

                    trade["StopLossPct"] = stop_loss_pct
                    trade["TargetPct"] = target_pct

                    combination_trades.append(trade)

                trades_df = pd.DataFrame(combination_trades)

                metrics = calculate_metrics(
                    trades=trades_df,
                    stop_loss_pct=stop_loss_pct,
                    target_pct=target_pct
                )

                results.append(metrics)

                all_trade_details.extend(combination_trades)

                progress.update(1)

                elapsed = time.perf_counter() - start_time
                completed = progress.n
                remaining = (
                    elapsed / completed * (total_combinations - completed)
                    if completed > 0
                    else 0
                )

                progress.set_postfix({
                    "Stop": stop_loss_pct,
                    "Target": target_pct,
                    "Elapsed": f"{elapsed:.0f}s",
                    "ETA": f"{remaining:.0f}s"
                })

    results_df = pd.DataFrame(results)
    trade_details_df = pd.DataFrame(all_trade_details)
    
    # Calculate score
    score_metrics = [
    "ExpectancyR",
    "WinRatePct",
    ]

    for metric in score_metrics:
        results_df[f"z_{metric}"] = (
            results_df[metric] - results_df[metric].mean()
        ) / results_df[metric].std()
    
    # Adjust weights if needed to give a metric more weight
    results_df["ScoreRaw"] = (
    2 * results_df["z_ExpectancyR"] +
    1 * results_df["z_WinRatePct"]
    )
    
    # Convert score to 0-100 range
    score_min = results_df["ScoreRaw"].min()
    score_max = results_df["ScoreRaw"].max()

    results_df["Score"] = (
        100
        * (results_df["ScoreRaw"] - score_min)
        / (score_max - score_min)
    )
    

    # Rank by Expectancy in R, highest first.
    # NaN values (e.g. StopLossPct = 0) are placed at the bottom.
    results_df = results_df.sort_values(
        by=RANK_METRIC,
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    results_df.insert(
        0,
        "Rank",
        np.arange(1, len(results_df) + 1)
    )

    return results_df, trade_details_df


# ============================================================
# CREATE EXPECTANCY IN R HEATMAP
# ============================================================
def format_stop_label(x):
    return "NS" if x == -1 else str(x)

def format_target_label(x):
    return "NT" if x == -1 else str(x)
    
def create_heatmap(results_df):
    color_data = results_df.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=HEATMAP_COLOR_METRIC
    ).reindex(
        index=TARGET_VALUES,
        columns=STOP_LOSS_VALUES
    )

    heatmap_data = results_df.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=HEATMAP_TOP_METRIC
    ).reindex(
        index=TARGET_VALUES,
        columns=STOP_LOSS_VALUES
    )    

    bottom_metric_data  = results_df.pivot(
        index="TargetPct",
        columns="StopLossPct",
        values=HEATMAP_BOTTOM_METRIC
    ).reindex(
        index=TARGET_VALUES,
        columns=STOP_LOSS_VALUES
    )
    
    condition_data = (
        (
            (results_df["WinRatePct"] > HIGHLIGHT_WR_MIN)
            &
            (results_df["ExpectancyR"] > HIGHLIGHT_EXPECTANCYR_MIN)
        )
        .rename("Condition")
    )

    condition_data = (
        pd.concat(
            [results_df[["TargetPct", "StopLossPct"]], condition_data],
            axis=1
        )
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values="Condition"
        )
        .reindex(
            index=TARGET_VALUES,
            columns=STOP_LOSS_VALUES
        )
    )
    display_values = heatmap_data.to_numpy(dtype=float)
    color_values = color_data.to_numpy(dtype=float)
    finite_values = color_values[np.isfinite(color_values)]

    if finite_values.size == 0:
        print(
            "Heatmap not created: no valid Expectancy in R values. "
            "A positive stop-loss value is required."
        )
        return

    # Use the actual minimum and maximum for the colour scale.
    # This works on any metrics
    vmin = HEATMAP_VMIN
    if vmin is None:
        vmin = float(np.min(finite_values))

    vmax = HEATMAP_VMAX
    if vmax is None:
        vmax = float(np.max(finite_values))
    
    # Avoid identical bounds if all finite values are equal.
    if vmin == vmax:
        pad = max(abs(vmin) * 0.01, 1e-9)
        vmin -= pad
        vmax += pad

    fig, ax = plt.subplots(
        figsize=(
            max(12, len(TARGET_VALUES) * 0.85),
            max(8, len(STOP_LOSS_VALUES) * 0.55)
        )
    )

    image = ax.imshow(
        color_values,
        cmap="RdYlGn",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(HEATMAP_COLOR_METRIC)

    ax.set_title(
        f"{TICKER} | {START_DATE} to {END_DATE or 'Latest'} | "
        f"{NUMBER_OF_RANDOM_ENTRIES:,} Random Entries\n"
        f"{HEATMAP_TOP_METRIC} (top), "
        f"{HEATMAP_BOTTOM_METRIC} (bottom)"
    )
    ax.set_xlabel("Stop-Loss (%)")
    ax.set_ylabel("Target (%)")

    ax.set_xticks(np.arange(len(STOP_LOSS_VALUES)))
    ax.set_xticklabels(
        [format_stop_label(x) for x in STOP_LOSS_VALUES],
        rotation=0
    )
    ax.set_yticks(np.arange(len(TARGET_VALUES)))
    ax.set_yticklabels(
        [format_target_label(x) for x in TARGET_VALUES]
    )
    
    

    # Annotate each cell with Expectancy in R.
    # Use a dash for undefined values, such as StopLossPct = 0.
    
    top_fmt = HEATMAP_METRIC_FORMATS.get(
    HEATMAP_TOP_METRIC,
    ".2f"
    )

    bottom_fmt = HEATMAP_METRIC_FORMATS.get(
    HEATMAP_BOTTOM_METRIC,
    ".2f"
    )
    
    midpoint = (vmin + vmax) / 2

    for row in range(display_values.shape[0]):
        for col in range(display_values.shape[1]):
            value = display_values[row, col]
            color_value = color_values[row, col]
            bottom_metric = bottom_metric_data.iloc[row, col]

            if np.isfinite(value):
                label = (
                    f"{value:{top_fmt}}\n"
                    f"{bottom_metric:{bottom_fmt}}"
                )
                rgba = image.cmap(image.norm(color_value))
                r, g, b, _ = rgba
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if brightness > 0.55 else "white"
            else:
                label = "N/A"
                text_color = "black"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )
            if condition_data.iloc[row, col]:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2
                    )
                )
    ax.set_xticks(
        np.arange(-0.5, len(STOP_LOSS_VALUES), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(TARGET_VALUES), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="gray",
        linestyle="-",
        linewidth=0.4,
        alpha=0.4
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "expectancy_r_heatmap.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Heatmap saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    data = download_data()

    entries = select_random_entries(data)

    results_df, trade_details_df = compare_combinations(
        data=data,
        entries=entries
    )

    results_path = (
        OUTPUT_DIR / "stop_target_randomized_results.csv"
    )
    details_path = (
        OUTPUT_DIR / "randomized_trade_details.csv"
    )

    results_df.round(2).to_csv(results_path, index=False)
    trade_details_df.round(2).to_csv(details_path, index=False)

    create_heatmap(results_df)

    print("\n" + "=" * 80)
    print("TOP 20 STOP/TARGET COMBINATIONS — RANKED BY EXPECTANCY IN R")
    print("=" * 80)

    columns_to_display = [
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
        "EndOfDataExitPct"
    ]

    print(
        results_df[columns_to_display]
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.2f}")
    )

    print("\nFiles saved:")
    print(f"  Random entry bars: {OUTPUT_DIR / 'random_entry_bars.csv'}")
    print(f"  Results:           {results_path}")
    print(f"  Trade details:     {details_path}")
    print(f"  Heatmap:           {OUTPUT_DIR / 'expectancy_r_heatmap.png'}")


if __name__ == "__main__":
    main()
