"""Run Exit Strategy simulations across reproducible random entry samples."""

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# USER CONFIGURATION
# Edit this section to change the default analysis; CLI arguments can override it.
# =============================================================================
# Yahoo Finance symbol and inclusive simulation date range.
TICKER = "QQQ"
START_DATE = date(2000, 1, 1)
END_DATE = date(2026, 12, 31)
# Candle interval used to choose entries and simulate exits.
INTERVAL = "1d"
# True uses entry ATRP multiples for exits; False uses fixed price percentages.
USE_ATRP = True
# ATR candles and Wilder ATR lookback. The ATR value is frozen at each entry.
ATRP_TIMEFRAME = "3mo"
ATRP_PERIOD = 12
# Seeds are SEED_START through SEED_START + SEED_COUNT - 1.
SEED_COUNT = 200
SEED_START = 0
# Requested entries per seed; filters or unavailable ATRP can reduce the count.
ENTRY_COUNT = 25
# Entry/exit moving-average lookback, in trading bars.
SMA_PERIOD = 200
# Long, Short, or Both; Both randomizes direction when the entry filter is off.
TRADE_DIRECTION = "Long"
# Commission and slippage are percentages per side.
COMMISSION = 0.0
SLIPPAGE = 0.0
# If stop and target hit in the same candle, True assumes the stop was hit first.
CONSERVATIVE = True
# Run only the chosen entry-filter mode: True enables the SMA entry filter.
MA_ENTRY_ENABLED = False
# Independently exit a trade on an opposite SMA crossover.
MA_EXIT_ENABLED = False
# Directory for CSV reports and heatmap PNG.
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "analysis"
# Each heatmap config selects one summary metric and its color bounds.
HEATMAP_CONFIGS = {
    "win_rate": {
        "title": "Win rate",
        "metric": "WinRatePct_Median",
        "min": 50,
        "max": 100,
    },
    "expectancy_r": {
        "title": "Expectancy (R)",
        "metric": "ExpectancyR_Median",
        "min": 0,
        "max": 1,
    },
    "expectancy_r_per_bar": {
        "title": "Expectancy (R per 100 bars)",
        "metric": "ExpectancyRPer100Bars_Median",
        "min": None,
        "max": None,
    },
    "bars_held": {
        "title": "Bars held",
        "metric": "AverageBarsHeld_Median",
        "min": 0,
        "max": None,
    },
}
# Fixed-percentage stop/target grids; -1 means no stop or target.
DEFAULT_PERCENT_STOPS = [
    5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 90, -1,
]
DEFAULT_PERCENT_TARGETS = [
    5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, -1,
]
# ATRP-multiple stop/target grids; -1 means no stop or target.
DEFAULT_ATRP_STOP_MULTIPLIERS = [
    0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, -1,
]
DEFAULT_ATRP_TARGET_MULTIPLIERS = [
    0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, -1,
]


# =============================================================================
# INTERNAL REPORT SCHEMAS
# =============================================================================
TRADE_METRIC_COLUMNS = [
    "NetReturnPct",
    "NetReturnATRP",
    "NetReturnPctPerBar",
    "NetReturnATRPPerBar",
    "RealizedR",
    "EntryATRP",
    "StopDistancePct",
    "TargetDistancePct",
    "BarsHeld",
    "ExitReason",
    "CommissionPct",
]
METRIC_COLUMNS = [
    "NumberOfTrades",
    "WinRatePct",
    "AverageWinPct",
    "AverageLossPct",
    "AverageWinATRP",
    "AverageLossATRP",
    "RealizedRRR",
    "BreakEvenWinRatePct",
    "PlannedRRR",
    "PlannedBreakEvenWinRatePct",
    "ExpectancyPct",
    "ExpectancyPctPerBar",
    "ExpectancyATRP",
    "ExpectancyATRPPerBar",
    "ExpectancyR",
    "ExpectancyRPerBar",
    "ExpectancyRPer100Bars",
    "AverageEntryATRP",
    "AverageStopDistancePct",
    "AverageTargetDistancePct",
    "AverageBarsHeld",
    "StopExitPct",
    "TargetExitPct",
    "MovingAverageExitPct",
    "EndOfDataExitPct",
    "AverageCommissionPct",
]
HEATMAP_AGGREGATIONS = ("Mean", "Median", "Std", "Min", "Max", "Count")
HEATMAP_METRICS = [
    f"{metric}_{aggregation}"
    for metric in METRIC_COLUMNS
    for aggregation in HEATMAP_AGGREGATIONS
] + [
    "ExpectancyR_PositivePct",
    "ExpectancyATRP_PositivePct",
    "SeedCount",
    "RunsWithTrades",
    "RobustRank",
]
SUMMARY_KEYS = [
    "Ticker",
    "TradingTimeframe",
    "EntryMAEnabled",
    "ExitMode",
    "ParameterUnit",
    "ATRTimeframe",
    "ATRPeriod",
    "StopParameter",
    "TargetParameter",
]


def download_data(
    ticker: str,
    start_date: date,
    end_date: date,
    interval: str,
) -> pd.DataFrame:
    """Download and validate OHLC history directly from Yahoo Finance."""

    data = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(
            "No data was downloaded. Check the ticker, date range, and interval."
        )

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[required_columns].dropna().sort_index()
    if len(data) < 2:
        raise ValueError("At least two valid price bars are required.")
    if (data[required_columns] <= 0).any().any():
        raise ValueError("Price data contains zero or negative values.")
    return data


def atr_warmup_start_date(
    start_date: date,
    timeframe: str,
    period: int,
) -> date:
    """Include extra bars before trading starts to initialize Wilder ATR."""

    if timeframe == "3mo":
        offset = pd.DateOffset(months=3 * (period + 2))
        return (pd.Timestamp(start_date) - offset).date()
    if timeframe == "1mo":
        offset = pd.DateOffset(months=period + 2)
        return (pd.Timestamp(start_date) - offset).date()
    if timeframe == "1d":
        return start_date - timedelta(days=period * 2 + 14)
    days_per_bar = {"1wk": 7}
    if timeframe not in days_per_bar:
        raise ValueError(f"Unsupported ATR timeframe: {timeframe}")
    return start_date - timedelta(
        days=days_per_bar[timeframe] * (period + 2)
    )


def calculate_atrp(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Wilder ATR as a percentage of the ATR candle's close."""

    if period < 1:
        raise ValueError("The ATR period must be at least 1.")

    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_values = true_range.to_numpy(dtype=float)
    atr_values = np.full(len(data), np.nan)

    if len(data) >= period:
        first_valid_index = period - 1
        atr_values[first_valid_index] = true_range_values[:period].mean()
        for index in range(period, len(data)):
            atr_values[index] = (
                atr_values[index - 1] * (period - 1)
                + true_range_values[index]
            ) / period

    return pd.Series(
        atr_values / data["Close"].to_numpy(dtype=float) * 100,
        index=data.index,
        name="ATRP",
    )


def _naive_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    datetime_index = pd.DatetimeIndex(pd.to_datetime(index))
    if datetime_index.tz is not None:
        datetime_index = datetime_index.tz_localize(None)
    return datetime_index


def align_atrp_to_trading_data(
    trading_data: pd.DataFrame,
    atr_data: pd.DataFrame,
    trading_timeframe: str,
    atr_timeframe: str,
    atr_period: int,
) -> pd.Series:
    """Align ATRP to trading bars without using an unfinished ATR candle."""

    atrp = calculate_atrp(atr_data, atr_period)
    trading_index = _naive_datetime_index(trading_data.index)
    atr_index = _naive_datetime_index(atr_data.index)

    if trading_timeframe == atr_timeframe:
        atrp.index = atr_index
        aligned = atrp.reindex(trading_index)
    else:
        completed_atr_values = pd.DataFrame(
            {
                "AvailableAt": atr_index[1:],
                "ATRP": atrp.to_numpy(dtype=float)[:-1],
            }
        )
        trading_dates = pd.DataFrame({"TradingDate": trading_index})
        aligned_frame = pd.merge_asof(
            trading_dates.sort_values("TradingDate"),
            completed_atr_values.sort_values("AvailableAt"),
            left_on="TradingDate",
            right_on="AvailableAt",
            direction="backward",
        )
        aligned = pd.Series(
            aligned_frame["ATRP"].to_numpy(dtype=float),
            index=trading_index,
            name="ATRP",
        )

    aligned.index = trading_data.index
    aligned.name = "ATRP"
    if not np.isfinite(aligned.to_numpy(dtype=float)).any():
        raise ValueError(
            "No completed ATRP values overlap the trading data. "
            "Use an earlier start date, a shorter ATR period, or a finer ATR timeframe."
        )
    return aligned


def select_entries(
    data: pd.DataFrame,
    number_of_entries: int,
    seed: int,
    sma_period: int,
    trade_direction: str,
    ma_enabled: bool,
    atrp_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Select reproducible entry bars and optionally apply the entry SMA filter."""

    maximum_entries = len(data) - 1
    if maximum_entries < 1:
        raise ValueError("Not enough historical data to select entries.")
    if sma_period < 1:
        raise ValueError("The moving-average period must be at least 1.")
    if trade_direction not in {"Long", "Short", "Both"}:
        raise ValueError("Trade direction must be Long, Short, or Both.")

    close_values = data["Close"].to_numpy()
    sma_values = (
        data["Close"]
        .rolling(window=int(sma_period), min_periods=int(sma_period))
        .mean()
        .to_numpy()
        if ma_enabled
        else np.full(len(data), np.nan)
    )
    rng = np.random.default_rng(int(seed))
    actual_entries = min(int(number_of_entries), maximum_entries)
    eligible_indexes = np.arange(maximum_entries)
    if atrp_values is not None:
        eligible_indexes = eligible_indexes[
            np.isfinite(atrp_values[eligible_indexes])
        ]
    actual_entries = min(int(number_of_entries), len(eligible_indexes))
    entry_indexes = np.sort(
        rng.choice(eligible_indexes, size=actual_entries, replace=False)
    )

    above_sma = close_values > sma_values
    below_sma = close_values < sma_values
    if ma_enabled and trade_direction == "Long":
        entry_indexes = entry_indexes[above_sma[entry_indexes]]
    elif ma_enabled and trade_direction == "Short":
        entry_indexes = entry_indexes[below_sma[entry_indexes]]
    elif ma_enabled:
        entry_indexes = entry_indexes[
            above_sma[entry_indexes] | below_sma[entry_indexes]
        ]

    entry_directions = np.where(
        above_sma[entry_indexes],
        "Long",
        "Short",
    )
    if not ma_enabled and trade_direction == "Long":
        entry_directions = np.full(len(entry_indexes), "Long")
    elif not ma_enabled and trade_direction == "Short":
        entry_directions = np.full(len(entry_indexes), "Short")
    elif not ma_enabled:
        entry_directions = rng.choice(
            ["Long", "Short"],
            size=len(entry_indexes),
        )

    entries = pd.DataFrame(
        {
            "EntryIndex": entry_indexes,
            "EntryDate": data.index[entry_indexes],
            "EntryClose": close_values[entry_indexes],
            "EntrySMA": sma_values[entry_indexes],
            "Direction": entry_directions,
            "EntryATRP": (
                atrp_values[entry_indexes]
                if atrp_values is not None
                else np.full(len(entry_indexes), np.nan)
            ),
        }
    )
    return entries


def simulate_trade(
    data: pd.DataFrame,
    entry_index: int,
    stop_parameter: float,
    target_parameter: float,
    commission_pct: float,
    slippage_pct: float,
    conservative: bool,
    direction: str,
    sma_values: np.ndarray | None = None,
    entry_atrp: float | None = None,
    use_atrp: bool = False,
) -> dict:
    """Simulate a trade with percentage or entry-ATRP-scaled exits."""

    if direction not in {"Long", "Short"}:
        raise ValueError("Trade direction must be Long or Short.")
    if use_atrp and (
        entry_atrp is None
        or not np.isfinite(entry_atrp)
        or entry_atrp <= 0
    ):
        raise ValueError("A positive entry ATRP is required for ATRP exits.")

    direction_sign = 1 if direction == "Long" else -1
    close_prices = data["Close"].to_numpy()
    high_prices = data["High"].to_numpy()
    low_prices = data["Low"].to_numpy()
    dates = data.index
    entry_reference = float(close_prices[entry_index])
    no_stop = stop_parameter == -1
    no_target = target_parameter == -1
    stop_distance_pct = (
        None
        if no_stop
        else (
            float(stop_parameter) * float(entry_atrp)
            if use_atrp
            else float(stop_parameter)
        )
    )
    target_distance_pct = (
        None
        if no_target
        else (
            float(target_parameter) * float(entry_atrp)
            if use_atrp
            else float(target_parameter)
        )
    )
    entry_execution = entry_reference * (
        1 + direction_sign * float(slippage_pct) / 100
    )
    stop_price = (
        None
        if stop_distance_pct is None
        else entry_reference
        * (1 - direction_sign * stop_distance_pct / 100)
    )
    target_price = (
        None
        if target_distance_pct is None
        else entry_reference
        * (1 + direction_sign * target_distance_pct / 100)
    )

    exit_index = len(data) - 1
    exit_reference = float(close_prices[exit_index])
    exit_reason = "EndOfData"

    for current_index in range(entry_index + 1, len(data)):
        stop_hit = stop_price is not None and (
            float(low_prices[current_index]) <= stop_price
            if direction == "Long"
            else float(high_prices[current_index]) >= stop_price
        )
        target_hit = target_price is not None and (
            float(high_prices[current_index]) >= target_price
            if direction == "Long"
            else float(low_prices[current_index]) <= target_price
        )

        if stop_hit and target_hit:
            if conservative:
                exit_reference = stop_price
                exit_reason = "Stop"
            else:
                exit_reference = target_price
                exit_reason = "Target"
            exit_index = current_index
            break
        if stop_hit:
            exit_reference = stop_price
            exit_reason = "Stop"
            exit_index = current_index
            break
        if target_hit:
            exit_reference = target_price
            exit_reason = "Target"
            exit_index = current_index
            break

        if (
            sma_values is not None
            and np.isfinite(sma_values[current_index - 1])
            and np.isfinite(sma_values[current_index])
        ):
            previous_close = float(close_prices[current_index - 1])
            current_close = float(close_prices[current_index])
            previous_sma = float(sma_values[current_index - 1])
            current_sma = float(sma_values[current_index])
            crossed_against_position = (
                previous_close >= previous_sma and current_close < current_sma
                if direction == "Long"
                else previous_close <= previous_sma and current_close > current_sma
            )
            if crossed_against_position:
                exit_reference = current_close
                exit_reason = "MovingAverage"
                exit_index = current_index
                break

    exit_execution = exit_reference * (
        1 - direction_sign * float(slippage_pct) / 100
    )
    gross_return_pct = (
        direction_sign * (exit_execution / entry_execution - 1) * 100
    )
    total_commission_pct = 2 * float(commission_pct)
    net_return_pct = gross_return_pct - total_commission_pct
    bars_held = exit_index - entry_index
    net_return_atrp = (
        net_return_pct / float(entry_atrp)
        if entry_atrp is not None
        and np.isfinite(entry_atrp)
        and entry_atrp > 0
        else np.nan
    )
    net_return_pct_per_bar = (
        net_return_pct / bars_held if bars_held else np.nan
    )
    net_return_atrp_per_bar = (
        net_return_atrp / bars_held
        if bars_held and np.isfinite(net_return_atrp)
        else np.nan
    )
    realized_r = (
        net_return_pct / stop_distance_pct
        if stop_distance_pct is not None
        else np.nan
    )
    return {
        "EntryDate": dates[entry_index],
        "Direction": direction,
        "ExitDate": dates[exit_index],
        "EntryIndex": entry_index,
        "ExitIndex": exit_index,
        "EntryReferencePrice": entry_reference,
        "EntryExecutionPrice": entry_execution,
        "ExitReferencePrice": exit_reference,
        "ExitExecutionPrice": exit_execution,
        "ExitMode": "ATRP" if use_atrp else "Percent",
        "StopParameter": stop_parameter,
        "TargetParameter": target_parameter,
        "StopLossPct": (
            stop_distance_pct if stop_distance_pct is not None else -1
        ),
        "TargetPct": (
            target_distance_pct if target_distance_pct is not None else -1
        ),
        "StopDistancePct": (
            stop_distance_pct if stop_distance_pct is not None else np.nan
        ),
        "TargetDistancePct": (
            target_distance_pct if target_distance_pct is not None else np.nan
        ),
        "EntryATRP": entry_atrp if entry_atrp is not None else np.nan,
        "GrossReturnPct": gross_return_pct,
        "NetReturnPct": net_return_pct,
        "NetReturnATRP": net_return_atrp,
        "NetReturnPctPerBar": net_return_pct_per_bar,
        "NetReturnATRPPerBar": net_return_atrp_per_bar,
        "RealizedR": realized_r,
        "BarsHeld": bars_held,
        "ExitReason": exit_reason,
        "CommissionPct": total_commission_pct,
    }


def calculate_metrics(
    trades: pd.DataFrame,
    stop: float,
    target: float,
    exit_mode: str,
) -> dict:
    """Calculate percentage, ATRP, and initial-risk metrics per combination."""

    returns = trades["NetReturnPct"].to_numpy(float)
    atr_returns = trades["NetReturnATRP"].to_numpy(float)
    pct_bar_returns = trades["NetReturnPctPerBar"].to_numpy(float)
    atrp_bar_returns = trades["NetReturnATRPPerBar"].to_numpy(float)
    realized_r = trades["RealizedR"].to_numpy(float)
    bars_held = trades["BarsHeld"].to_numpy(float)
    number_of_trades = len(returns)
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    valid_atr_returns = atr_returns[np.isfinite(atr_returns)]
    valid_pct_bar_returns = pct_bar_returns[np.isfinite(pct_bar_returns)]
    valid_atrp_bar_returns = atrp_bar_returns[np.isfinite(atrp_bar_returns)]
    valid_r_per_bar_mask = (
        np.isfinite(realized_r)
        & np.isfinite(bars_held)
        & (bars_held > 0)
    )
    valid_realized_r = realized_r[valid_r_per_bar_mask]
    valid_r_bars_held = bars_held[valid_r_per_bar_mask]
    winning_atr_returns = valid_atr_returns[valid_atr_returns > 0]
    losing_atr_returns = valid_atr_returns[valid_atr_returns < 0]
    win_rate = (
        float((returns > 0).mean() * 100)
        if number_of_trades
        else np.nan
    )
    average_win = (
        float(winning_returns.mean()) if len(winning_returns) else np.nan
    )
    average_loss = (
        float(losing_returns.mean()) if len(losing_returns) else np.nan
    )
    expectancy = float(returns.mean()) if number_of_trades else np.nan
    expectancy_pct_per_bar = (
        float(valid_pct_bar_returns.mean())
        if len(valid_pct_bar_returns)
        else np.nan
    )
    expectancy_atrp = (
        float(valid_atr_returns.mean())
        if len(valid_atr_returns)
        else np.nan
    )
    expectancy_atrp_per_bar = (
        float(valid_atrp_bar_returns.mean())
        if len(valid_atrp_bar_returns)
        else np.nan
    )
    expectancy_r = (
        float(valid_realized_r.mean()) if len(valid_realized_r) else np.nan
    )
    expectancy_r_per_bar = (
        float(valid_realized_r.sum() / valid_r_bars_held.sum())
        if len(valid_realized_r) and valid_r_bars_held.sum() > 0
        else np.nan
    )
    expectancy_r_per_100_bars = (
        expectancy_r_per_bar * 100
        if np.isfinite(expectancy_r_per_bar)
        else np.nan
    )
    planned_rrr = (
        float(target) / float(stop)
        if stop != -1 and target != -1 and stop > 0
        else np.nan
    )
    realized_rrr = (
        average_win / abs(average_loss)
        if (
            len(winning_returns)
            and len(losing_returns)
            and average_loss != 0
        )
        else np.nan
    )
    break_even_win_rate = (
        abs(average_loss) / (average_win + abs(average_loss)) * 100
        if (
            np.isfinite(average_win)
            and np.isfinite(average_loss)
            and average_win > 0
            and average_loss < 0
        )
        else np.nan
    )
    planned_break_even = (
        1 / (1 + planned_rrr) * 100
        if np.isfinite(planned_rrr) and planned_rrr > 0
        else np.nan
    )
    exit_counts = trades["ExitReason"].value_counts()
    return {
        "ExitMode": exit_mode,
        "StopParameter": stop,
        "TargetParameter": target,
        "NumberOfTrades": number_of_trades,
        "WinRatePct": win_rate,
        "AverageWinPct": average_win,
        "AverageLossPct": average_loss,
        "AverageWinATRP": (
            float(winning_atr_returns.mean())
            if len(winning_atr_returns)
            else np.nan
        ),
        "AverageLossATRP": (
            float(losing_atr_returns.mean())
            if len(losing_atr_returns)
            else np.nan
        ),
        "RealizedRRR": realized_rrr,
        "BreakEvenWinRatePct": break_even_win_rate,
        "PlannedRRR": planned_rrr,
        "PlannedBreakEvenWinRatePct": planned_break_even,
        "ExpectancyPct": expectancy,
        "ExpectancyPctPerBar": expectancy_pct_per_bar,
        "ExpectancyATRP": expectancy_atrp,
        "ExpectancyATRPPerBar": expectancy_atrp_per_bar,
        "ExpectancyR": expectancy_r,
        "ExpectancyRPerBar": expectancy_r_per_bar,
        "ExpectancyRPer100Bars": expectancy_r_per_100_bars,
        "AverageEntryATRP": (
            float(trades["EntryATRP"].mean()) if number_of_trades else np.nan
        ),
        "AverageStopDistancePct": (
            float(trades["StopDistancePct"].mean())
            if number_of_trades
            else np.nan
        ),
        "AverageTargetDistancePct": (
            float(trades["TargetDistancePct"].mean())
            if number_of_trades
            else np.nan
        ),
        "AverageBarsHeld": (
            float(trades["BarsHeld"].mean()) if number_of_trades else np.nan
        ),
        "StopExitPct": (
            exit_counts.get("Stop", 0) / number_of_trades * 100
            if number_of_trades
            else np.nan
        ),
        "TargetExitPct": (
            exit_counts.get("Target", 0) / number_of_trades * 100
            if number_of_trades
            else np.nan
        ),
        "MovingAverageExitPct": (
            exit_counts.get("MovingAverage", 0) / number_of_trades * 100
            if number_of_trades
            else np.nan
        ),
        "EndOfDataExitPct": (
            exit_counts.get("EndOfData", 0) / number_of_trades * 100
            if number_of_trades
            else np.nan
        ),
        "AverageCommissionPct": (
            float(trades["CommissionPct"].mean())
            if number_of_trades
            else np.nan
        ),
    }
def run_sensitivity(
    data: pd.DataFrame,
    seeds: Iterable[int],
    number_of_entries: int = 25,
    stops: list[float] | None = None,
    targets: list[float] | None = None,
    sma_period: int = 200,
    trade_direction: str = "Long",
    commission: float = 0.0,
    slippage: float = 0.0,
    conservative: bool = True,
    ma_entry_enabled: bool = MA_ENTRY_ENABLED,
    ma_exit_enabled: bool = False,
    use_atrp: bool = USE_ATRP,
    atrp_values: pd.Series | np.ndarray | None = None,
    atr_timeframe: str = ATRP_TIMEFRAME,
    atr_period: int = ATRP_PERIOD,
    progress_callback: Callable[[int, int, int, bool], None] | None = None,
    ticker: str = TICKER,
    trading_timeframe: str = INTERVAL,
) -> pd.DataFrame:
    """Return one metrics row per seed and stop/target pair."""

    seed_values = [int(seed) for seed in seeds]
    default_stops = (
        DEFAULT_ATRP_STOP_MULTIPLIERS if use_atrp else DEFAULT_PERCENT_STOPS
    )
    default_targets = (
        DEFAULT_ATRP_TARGET_MULTIPLIERS if use_atrp else DEFAULT_PERCENT_TARGETS
    )
    stop_values = default_stops if stops is None else stops
    target_values = default_targets if targets is None else targets

    if not seed_values:
        raise ValueError("At least one random seed is required.")
    if number_of_entries < 1:
        raise ValueError("The number of entries must be at least 1.")
    if not stop_values or not target_values:
        raise ValueError("At least one stop and one target are required.")
    if use_atrp and atrp_values is None:
        raise ValueError("ATRP values are required when ATRP exits are enabled.")

    atrp_array = (
        np.asarray(atrp_values, dtype=float)
        if atrp_values is not None
        else None
    )
    if atrp_array is not None and len(atrp_array) != len(data):
        raise ValueError("ATRP values must align one-to-one with trading bars.")

    sma_values = (
        data["Close"]
        .rolling(window=int(sma_period), min_periods=int(sma_period))
        .mean()
        .to_numpy()
        if ma_exit_enabled
        else None
    )
    records = []
    completed_runs = 0
    total_runs = len(seed_values)

    for seed in seed_values:
        entries = select_entries(
            data=data,
            number_of_entries=number_of_entries,
            seed=seed,
            sma_period=sma_period,
            trade_direction=trade_direction,
            ma_enabled=ma_entry_enabled,
            atrp_values=atrp_array if use_atrp else None,
        )
        for stop in stop_values:
            for target in target_values:
                trades = [
                    simulate_trade(
                        data=data,
                        entry_index=int(entry.EntryIndex),
                        stop_parameter=stop,
                        target_parameter=target,
                        commission_pct=commission,
                        slippage_pct=slippage,
                        conservative=conservative,
                        direction=str(entry.Direction),
                        sma_values=sma_values,
                        entry_atrp=(
                            float(entry.EntryATRP)
                            if use_atrp
                            else None
                        ),
                        use_atrp=use_atrp,
                    )
                    for entry in entries.itertuples(index=False)
                ]
                trade_frame = pd.DataFrame(
                    trades,
                    columns=(
                        list(trades[0])
                        if trades
                        else TRADE_METRIC_COLUMNS
                    ),
                )
                metrics = calculate_metrics(
                    trade_frame,
                    stop,
                    target,
                    "ATRP" if use_atrp else "Percent",
                )
                records.append(
                    {
                        "Seed": seed,
                        "Ticker": ticker,
                        "TradingTimeframe": trading_timeframe,
                        "EntryMAEnabled": ma_entry_enabled,
                        "ParameterUnit": (
                            "ATRP multiples"
                            if use_atrp
                            else "Percent of entry price"
                        ),
                        "ATRTimeframe": (
                            atr_timeframe if use_atrp else "N/A"
                        ),
                        "ATRPeriod": atr_period if use_atrp else np.nan,
                        "SelectedEntries": len(entries),
                        **metrics,
                    }
                )
        completed_runs += 1
        if progress_callback is not None:
            progress_callback(
                completed_runs,
                total_runs,
                seed,
                ma_entry_enabled,
            )

    return pd.DataFrame.from_records(records)


def summarize_runs(
    run_results: pd.DataFrame,
    robustness_metric: str = "ExpectancyR",
) -> pd.DataFrame:
    """Aggregate performance distributions and rank stable combinations."""

    if robustness_metric not in {"ExpectancyR", "ExpectancyATRP"}:
        raise ValueError(
            "Robustness metric must be ExpectancyR or ExpectancyATRP."
        )
    required_columns = {"Seed", *SUMMARY_KEYS, *METRIC_COLUMNS}
    missing_columns = required_columns.difference(run_results.columns)
    if missing_columns:
        raise ValueError(
            "Run results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    results = run_results.copy()
    for metric in ("ExpectancyR", "ExpectancyATRP"):
        results[f"_{metric}Positive"] = results[metric].gt(0).astype(int)
        results[f"_{metric}Valid"] = results[metric].notna().astype(int)
    grouped = results.groupby(SUMMARY_KEYS, sort=False, dropna=False)

    summary = grouped[METRIC_COLUMNS].agg(
        ["mean", "median", "std", "min", "max", "count"]
    )
    summary.columns = [
        f"{metric}_{statistic.title()}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()

    run_counts = grouped.agg(
        SeedCount=("Seed", "nunique"),
        RunsWithTrades=("NumberOfTrades", lambda values: int(values.gt(0).sum())),
        PositiveRExpectancyRuns=("_ExpectancyRPositive", "sum"),
        ValidRExpectancyRuns=("_ExpectancyRValid", "sum"),
        PositiveATRPExpectancyRuns=("_ExpectancyATRPPositive", "sum"),
        ValidATRPExpectancyRuns=("_ExpectancyATRPValid", "sum"),
    ).reset_index()
    summary = summary.merge(
        run_counts,
        on=SUMMARY_KEYS,
        how="left",
        validate="one_to_one",
    )
    summary["ExpectancyR_PositivePct"] = (
        summary["PositiveRExpectancyRuns"]
        .div(summary["ValidRExpectancyRuns"].replace(0, np.nan))
        * 100
    )
    summary["ExpectancyATRP_PositivePct"] = (
        summary["PositiveATRPExpectancyRuns"]
        .div(summary["ValidATRPExpectancyRuns"].replace(0, np.nan))
        * 100
    )
    summary = summary.drop(
        columns=[
            "PositiveRExpectancyRuns",
            "ValidRExpectancyRuns",
            "PositiveATRPExpectancyRuns",
            "ValidATRPExpectancyRuns",
        ]
    )

    positive_rate_column = f"{robustness_metric}_PositivePct"
    median_column = f"{robustness_metric}_Median"
    std_column = f"{robustness_metric}_Std"
    summary = summary.sort_values(
        [
            "Ticker",
            "TradingTimeframe",
            "EntryMAEnabled",
            "ExitMode",
            "ParameterUnit",
            "ATRTimeframe",
            "ATRPeriod",
            positive_rate_column,
            median_column,
            std_column,
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)
    summary["RobustMetric"] = robustness_metric
    summary.insert(
        0,
        "RobustRank",
        summary.groupby(
            [
                "EntryMAEnabled",
                "Ticker",
                "TradingTimeframe",
                "ExitMode",
                "ParameterUnit",
                "ATRTimeframe",
                "ATRPeriod",
            ],
            sort=False,
            dropna=False,
        ).cumcount()
        + 1,
    )
    return summary


def create_heatmap(
    results: pd.DataFrame,
    stops: list[float],
    targets: list[float],
    metric: str,
    ticker: str,
    trading_timeframe: str,
    entry_ma_enabled: bool,
    exit_mode: str,
    atr_timeframe: str,
    atr_period: int,
    color_min: float | None = None,
    color_max: float | None = None,
    metric_label: str | None = None,
):
    """Create one annotated stop/target heatmap for a single metric."""

    required_columns = {
        "StopParameter",
        "TargetParameter",
        "EntryMAEnabled",
        metric,
    }
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        raise ValueError(
            "Heatmap results are missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if not pd.api.types.is_numeric_dtype(results[metric]):
        raise ValueError(f"Heatmap metric must be numeric: {metric}")

    mode_results = results.loc[
        results["EntryMAEnabled"] == entry_ma_enabled
    ]
    matrix_frame = (
        mode_results.pivot(
            index="TargetParameter",
            columns="StopParameter",
            values=metric,
        )
        .reindex(index=targets, columns=stops)
    )
    matrix = matrix_frame.to_numpy(dtype=float)
    all_metric_values = results[metric].to_numpy(dtype=float)
    finite_values = all_metric_values[np.isfinite(all_metric_values)]
    if not finite_values.size:
        raise ValueError(
            f"No finite values for heatmap metric {metric!r} "
            f"with entry MA filter {entry_ma_enabled}."
        )
    if color_min is not None and color_max is not None and color_min >= color_max:
        raise ValueError("Heatmap minimum must be less than heatmap maximum.")

    vmin = float(np.min(finite_values)) if color_min is None else color_min
    vmax = float(np.max(finite_values)) if color_max is None else color_max
    if vmin > vmax or (
        vmin == vmax
        and color_min is not None
        and color_max is not None
    ):
        raise ValueError("Heatmap minimum must be less than heatmap maximum.")
    if vmin == vmax:
        padding = max(abs(vmin) * 0.01, 1e-9)
        if color_min is None:
            vmin -= padding
        if color_max is None:
            vmax += padding

    figure, axis = plt.subplots(
        figsize=(max(10, len(stops) * 0.8), max(6, len(targets) * 0.55))
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
    display_metric = metric_label or metric
    figure.colorbar(image, ax=axis).set_label(display_metric)

    for row_index, column_index in zip(*np.where(np.isfinite(matrix))):
        axis.text(
            column_index,
            row_index,
            f"{matrix[row_index, column_index]:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

    axis.set_xticks(np.arange(len(stops)))
    axis.set_xticklabels(
        ["NS" if value == -1 else str(value) for value in stops]
    )
    axis.set_yticks(np.arange(len(targets)))
    axis.set_yticklabels(
        ["NT" if value == -1 else str(value) for value in targets]
    )
    axis.set_xticks(np.arange(-0.5, len(stops), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    parameter_unit = (
        "ATRP multiples" if exit_mode == "ATRP" else "Percent of entry price"
    )
    axis.set_xlabel(f"Stop ({parameter_unit})")
    axis.set_ylabel(f"Target ({parameter_unit})")
    filter_label = "enabled" if entry_ma_enabled else "disabled"
    mode_label = (
        f"ATRP {atr_timeframe} / {atr_period}"
        if exit_mode == "ATRP"
        else "percentage exits"
    )
    axis.set_title(
        f"{ticker} | Trading {trading_timeframe} | {mode_label}\n"
        f"Entry MA filter {filter_label} | {display_metric}"
    )
    figure.tight_layout()
    return figure


def build_heatmap_requests(
    use_atrp: bool,
    selected_configs: list[str] | None = None,
    metric_override: str | None = None,
    color_min: float | None = None,
    color_max: float | None = None,
) -> list[dict[str, object]]:
    """Resolve configured heatmaps and mode-specific metrics and bounds."""

    if metric_override is not None:
        return [
            {
                "name": "custom",
                "title": metric_override,
                "metric": metric_override,
                "color_min": color_min,
                "color_max": color_max,
            }
        ]

    exit_mode = "ATRP" if use_atrp else "Percent"
    config_names = (
        selected_configs
        if selected_configs is not None
        else list(HEATMAP_CONFIGS)
    )
    requests = []
    for name in config_names:
        if name not in HEATMAP_CONFIGS:
            raise ValueError(f"Unknown heatmap config: {name}")
        config = HEATMAP_CONFIGS[name]
        metric_config = config["metric"]
        metric = (
            metric_config.get(exit_mode)
            if isinstance(metric_config, dict)
            else metric_config
        )
        if not isinstance(metric, str):
            raise ValueError(f"Invalid metric in heatmap config {name!r}.")

        def resolve_bound(bound_name: str) -> float | None:
            bound_config = config[bound_name]
            if isinstance(bound_config, dict):
                bound = bound_config.get(exit_mode)
            else:
                bound = bound_config
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, (int, float))
            ):
                raise ValueError(
                    f"Invalid {bound_name} bound in heatmap config {name!r}."
                )
            return float(bound) if bound is not None else None

        minimum = resolve_bound("min")
        maximum = resolve_bound("max")
        title = config["title"]
        if not isinstance(title, str):
            raise ValueError(f"Invalid title in heatmap config {name!r}.")
        if name == "win_rate":
            title = f"{title} (%)"
        requests.append(
            {
                "name": name,
                "title": title,
                "metric": metric,
                "color_min": color_min if color_min is not None else minimum,
                "color_max": color_max if color_max is not None else maximum,
            }
        )
    return requests


def _parse_values(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated numeric values."
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one numeric value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Exit Strategy stop/target combinations across "
            "random-entry seeds using the selected entry SMA filter setting. "
            "Exit distances can use percentages or entry ATRP multiples."
        )
    )
    parser.add_argument("--ticker", default=TICKER)
    parser.add_argument("--start-date", type=date.fromisoformat, default=START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=END_DATE)
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
        default=ATRP_TIMEFRAME,
    )
    parser.add_argument("--atr-period", type=int, default=ATRP_PERIOD)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--seed-start", type=int, default=SEED_START)
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
    parser.add_argument("--stops", type=_parse_values)
    parser.add_argument("--targets", type=_parse_values)
    heatmap_selection = parser.add_mutually_exclusive_group()
    heatmap_selection.add_argument(
        "--heatmap-metrics",
        nargs="+",
        choices=list(HEATMAP_CONFIGS),
        default=None,
        help="Configured metrics to plot; defaults to all configured heatmaps.",
    )
    heatmap_selection.add_argument(
        "--heatmap-metric",
        default=None,
        help=(
            "Plot only this numeric summary metric, e.g. "
            "ExpectancyATRP_Median or WinRatePct_Mean."
        ),
    )
    parser.add_argument(
        "--heatmap-min",
        type=float,
        default=None,
        help="Override the lower color bound for each selected heatmap.",
    )
    parser.add_argument(
        "--heatmap-max",
        type=float,
        default=None,
        help="Override the upper color bound for each selected heatmap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be at least 1.")
    if args.entry_count < 1:
        raise SystemExit("--entry-count must be at least 1.")
    if args.sma_period < 1:
        raise SystemExit("--sma-period must be at least 1.")
    if args.use_atrp and args.atr_period < 1:
        raise SystemExit("--atr-period must be at least 1.")
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must be on or after --start-date.")
    if (
        args.heatmap_min is not None
        and args.heatmap_max is not None
        and args.heatmap_min >= args.heatmap_max
    ):
        raise SystemExit("--heatmap-min must be less than --heatmap-max.")
    heatmap_requests = build_heatmap_requests(
        use_atrp=args.use_atrp,
        selected_configs=args.heatmap_metrics,
        metric_override=args.heatmap_metric,
        color_min=args.heatmap_min,
        color_max=args.heatmap_max,
    )
    for request in heatmap_requests:
        color_min = request["color_min"]
        color_max = request["color_max"]
        if (
            isinstance(color_min, (int, float))
            and isinstance(color_max, (int, float))
            and color_min >= color_max
        ):
            raise SystemExit(
                f"Invalid color bounds for heatmap {request['name']!r}: "
                "minimum must be less than maximum."
            )

    print(f"Downloading {args.ticker} data...")
    data = download_data(
        args.ticker,
        args.start_date,
        args.end_date,
        args.interval,
    )
    atrp_values = None
    if args.use_atrp:
        atr_start_date = atr_warmup_start_date(
            args.start_date,
            args.atr_timeframe,
            args.atr_period,
        )
        print(
            f"Downloading {args.atr_timeframe} ATR history "
            f"from {atr_start_date}..."
        )
        atr_data = download_data(
            args.ticker,
            atr_start_date,
            args.end_date,
            args.atr_timeframe,
        )
        atrp_values = align_atrp_to_trading_data(
            trading_data=data,
            atr_data=atr_data,
            trading_timeframe=args.interval,
            atr_timeframe=args.atr_timeframe,
            atr_period=args.atr_period,
        )

    stops = args.stops
    targets = args.targets
    if stops is None:
        stops = (
            DEFAULT_ATRP_STOP_MULTIPLIERS
            if args.use_atrp
            else DEFAULT_PERCENT_STOPS
        )
    if targets is None:
        targets = (
            DEFAULT_ATRP_TARGET_MULTIPLIERS
            if args.use_atrp
            else DEFAULT_PERCENT_TARGETS
        )

    seeds = range(args.seed_start, args.seed_start + args.seed_count)

    def report_progress(
        completed: int,
        total: int,
        seed: int,
        ma_enabled: bool,
    ) -> None:
        filter_state = "enabled" if ma_enabled else "disabled"
        print(
            f"Completed {completed}/{total} runs "
            f"(seed {seed}, entry MA filter {filter_state})."
        )

    run_results = run_sensitivity(
        data=data,
        seeds=seeds,
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
        ticker=args.ticker,
        trading_timeframe=args.interval,
    )
    robustness_metric = "ExpectancyATRP" if args.use_atrp else "ExpectancyR"
    summary = summarize_runs(run_results, robustness_metric=robustness_metric)
    for request in heatmap_requests:
        metric = request["metric"]
        if (
            not isinstance(metric, str)
            or metric not in HEATMAP_METRICS
            or metric not in summary.columns
        ):
            raise ValueError(
                f"Unsupported heatmap metric {metric!r}. "
                "Choose a numeric summary metric such as "
                "ExpectancyATRP_Median or WinRatePct_Mean."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ticker_slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in args.ticker
    ).strip("_")
    mode_slug = (
        f"atrp_{args.atr_timeframe}_{args.atr_period}"
        if args.use_atrp
        else "percent"
    )
    filter_slug = "enabled" if args.ma_entry_enabled else "disabled"
    output_slug = (
        f"{ticker_slug}_{args.interval}_{mode_slug}_"
        f"entry-ma-{filter_slug}_sensitivity"
    )
    run_file = args.output_dir / f"{output_slug}_runs.csv"
    summary_file = args.output_dir / f"{output_slug}_summary.csv"
    run_results.to_csv(run_file, index=False)
    summary.to_csv(summary_file, index=False)
    heatmap_files = []
    for request in heatmap_requests:
        metric = request["metric"]
        title = request["title"]
        name = request["name"]
        color_min = request["color_min"]
        color_max = request["color_max"]
        if (
            not isinstance(metric, str)
            or not isinstance(title, str)
            or not isinstance(name, str)
        ):
            raise ValueError("Invalid heatmap request.")
        heatmap = create_heatmap(
            results=summary,
            stops=stops,
            targets=targets,
            metric=metric,
            ticker=args.ticker,
            trading_timeframe=args.interval,
            entry_ma_enabled=args.ma_entry_enabled,
            exit_mode="ATRP" if args.use_atrp else "Percent",
            atr_timeframe=args.atr_timeframe,
            atr_period=args.atr_period,
            color_min=color_min if isinstance(color_min, (int, float)) else None,
            color_max=color_max if isinstance(color_max, (int, float)) else None,
            metric_label=title,
        )
        heatmap_suffix = (
            "_heatmap.png" if name == "custom" else f"_heatmap_{name}.png"
        )
        heatmap_file = args.output_dir / f"{output_slug}{heatmap_suffix}"
        heatmap.savefig(heatmap_file, dpi=180, bbox_inches="tight")
        plt.close(heatmap)
        heatmap_files.append((metric, heatmap_file))

    filter_state = "enabled" if args.ma_entry_enabled else "disabled"
    print(
        f"Completed {args.seed_count} seeds "
        f"(entry MA filter {filter_state})."
    )
    print(f"Per-seed results: {run_file}")
    print(f"Robustness summary: {summary_file}")
    print("Heatmaps:")
    for metric, heatmap_file in heatmap_files:
        print(f"  {metric}: {heatmap_file}")
    print(
        summary[
            [
                "RobustRank",
                "RobustMetric",
                "EntryMAEnabled",
                "ExitMode",
                "ParameterUnit",
                "ATRTimeframe",
                "ATRPeriod",
                "StopParameter",
                "TargetParameter",
                "ExpectancyATRP_PositivePct",
                "ExpectancyATRP_Median",
                "ExpectancyATRP_Mean",
                "ExpectancyATRP_Std",
                "ExpectancyR_Median",
                "NumberOfTrades_Mean",
            ]
        ]
        .groupby("EntryMAEnabled", sort=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
