"""Fast drop-in engine for the exit-strategy Monte Carlo analysis.

Same results as exit_strategy_monte_carlo_analysis.run_sensitivity (up to
floating-point summation order), but much faster:

* Instead of walking the price bars once per (entry, stop, target) trade, each
  entry gets a running-min / running-max of the future lows / highs. The first
  bar that touches every stop level and every target level is then found with
  one binary search per level, and the whole stop x target grid is resolved
  with array operations.
* The max-open-trades filter runs vectorized across the whole grid.
* Metrics are computed from NumPy arrays rather than per-combination DataFrames.
* Scenarios (seeds) are independent, so they can run in parallel processes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import heapq
import os
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import exit_strategy_monte_carlo_analysis as base

limit_concurrent_trades = base.limit_concurrent_trades

# Exit reason codes used internally.
_STOP, _TARGET, _MA, _EOD = 0, 1, 2, 3


# =============================================================================
# Single-trade simulation (same output as base.simulate_trade, vectorized scan)
# =============================================================================
_ARRAY_CACHE: dict = {}


def _price_arrays(data: pd.DataFrame):
    cached = _ARRAY_CACHE.get("arrays")
    if cached is not None and cached[0] is data:
        return cached[1]
    arrays = (
        data["Close"].to_numpy(dtype=float),
        data["High"].to_numpy(dtype=float),
        data["Low"].to_numpy(dtype=float),
    )
    _ARRAY_CACHE["arrays"] = (data, arrays)
    return arrays


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
    entry_trp: float | None = None,
    use_atrp: bool = False,
    use_trp: bool = False,
) -> dict:
    """Drop-in replacement for base.simulate_trade."""

    if direction not in {"Long", "Short"}:
        raise ValueError("Trade direction must be Long or Short.")
    if use_atrp and (
        entry_atrp is None or not np.isfinite(entry_atrp) or entry_atrp <= 0
    ):
        raise ValueError("A positive entry ATRP is required for ATRP exits.")
    if use_trp and (
        entry_trp is None or not np.isfinite(entry_trp) or entry_trp <= 0
    ):
        raise ValueError("A positive entry TRP is required for TRP exits.")
    if use_atrp and use_trp:
        raise ValueError("Use ATRP or TRP exits, not both.")

    sign = 1 if direction == "Long" else -1
    close, high, low = _price_arrays(data)
    n = len(close)
    dates = data.index
    entry_reference = float(close[entry_index])
    no_stop = stop_parameter == -1
    no_target = target_parameter == -1
    scale = (
        float(entry_atrp) if use_atrp else (float(entry_trp) if use_trp else None)
    )
    stop_distance_pct = (
        None
        if no_stop
        else (float(stop_parameter) * scale if scale is not None else float(stop_parameter))
    )
    target_distance_pct = (
        None
        if no_target
        else (float(target_parameter) * scale if scale is not None else float(target_parameter))
    )
    entry_execution = entry_reference * (1 + sign * float(slippage_pct) / 100)
    stop_price = (
        None
        if stop_distance_pct is None
        else entry_reference * (1 - sign * stop_distance_pct / 100)
    )
    target_price = (
        None
        if target_distance_pct is None
        else entry_reference * (1 + sign * target_distance_pct / 100)
    )

    start = entry_index + 1
    stop_idx = target_idx = n
    if start < n:
        if stop_price is not None:
            hits = (
                low[start:] <= stop_price
                if sign == 1
                else high[start:] >= stop_price
            )
            k = int(hits.argmax())
            if hits[k]:
                stop_idx = start + k
        if target_price is not None:
            hits = (
                high[start:] >= target_price
                if sign == 1
                else low[start:] <= target_price
            )
            k = int(hits.argmax())
            if hits[k]:
                target_idx = start + k
    first = min(stop_idx, target_idx)

    ma_idx = n
    if sma_values is not None and first > start:
        previous_close = close[start - 1 : first - 1]
        current_close = close[start:first]
        previous_sma = sma_values[start - 1 : first - 1]
        current_sma = sma_values[start:first]
        crossed = np.isfinite(previous_sma) & np.isfinite(current_sma)
        if sign == 1:
            crossed &= (previous_close >= previous_sma) & (current_close < current_sma)
        else:
            crossed &= (previous_close <= previous_sma) & (current_close > current_sma)
        k = int(crossed.argmax())
        if crossed[k]:
            ma_idx = start + k

    exit_index = n - 1
    exit_reference = float(close[exit_index])
    exit_reason = "EndOfData"
    if ma_idx < first:
        exit_index = ma_idx
        exit_reference = float(close[ma_idx])
        exit_reason = "MovingAverage"
    elif first < n:
        exit_index = first
        stop_hit = stop_idx == first
        target_hit = target_idx == first
        if stop_hit and (not target_hit or conservative):
            exit_reference, exit_reason = stop_price, "Stop"
        else:
            exit_reference, exit_reason = target_price, "Target"

    exit_execution = exit_reference * (1 - sign * float(slippage_pct) / 100)
    gross_return_pct = sign * (exit_execution / entry_execution - 1) * 100
    total_commission_pct = 2 * float(commission_pct)
    net_return_pct = gross_return_pct - total_commission_pct
    bars_held = exit_index - entry_index
    net_return_atrp = (
        net_return_pct / float(entry_atrp)
        if entry_atrp is not None and np.isfinite(entry_atrp) and entry_atrp > 0
        else np.nan
    )
    net_return_trp = (
        net_return_pct / float(entry_trp)
        if entry_trp is not None and np.isfinite(entry_trp) and entry_trp > 0
        else np.nan
    )
    net_return_pct_per_bar = net_return_pct / bars_held if bars_held else np.nan
    net_return_atrp_per_bar = (
        net_return_atrp / bars_held
        if bars_held and np.isfinite(net_return_atrp)
        else np.nan
    )
    net_return_trp_per_bar = (
        net_return_trp / bars_held
        if bars_held and np.isfinite(net_return_trp)
        else np.nan
    )
    r_risk_pct = (
        stop_distance_pct if stop_distance_pct is not None else base.NO_STOP_RISK_PCT
    )
    realized_r = net_return_pct / r_risk_pct
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
        "ExitMode": "ATRP" if use_atrp else ("TRP" if use_trp else "Percent"),
        "StopParameter": stop_parameter,
        "TargetParameter": target_parameter,
        "StopLossPct": stop_distance_pct if stop_distance_pct is not None else -1,
        "TargetPct": target_distance_pct if target_distance_pct is not None else -1,
        "StopDistancePct": (
            stop_distance_pct if stop_distance_pct is not None else np.nan
        ),
        "TargetDistancePct": (
            target_distance_pct if target_distance_pct is not None else np.nan
        ),
        "EntryATRP": entry_atrp if entry_atrp is not None else np.nan,
        "EntryTRP": entry_trp if entry_trp is not None else np.nan,
        "GrossReturnPct": gross_return_pct,
        "NetReturnPct": net_return_pct,
        "NetReturnATRP": net_return_atrp,
        "NetReturnTRP": net_return_trp,
        "NetReturnPctPerBar": net_return_pct_per_bar,
        "NetReturnATRPPerBar": net_return_atrp_per_bar,
        "NetReturnTRPPerBar": net_return_trp_per_bar,
        "RealizedR": realized_r,
        "BarsHeld": bars_held,
        "ExitReason": exit_reason,
        "CommissionPct": total_commission_pct,
    }


# =============================================================================
# Metrics
# =============================================================================
def calculate_compounded_return_metrics(
    returns_pct: np.ndarray,
) -> tuple[float, float]:
    """Vectorized equivalent of base.calculate_compounded_return_metrics."""

    returns_pct = np.asarray(returns_pct, dtype=float)
    if not len(returns_pct):
        return np.nan, 0.0
    if not np.isfinite(returns_pct).all():
        return np.nan, np.nan
    with np.errstate(all="ignore"):
        equity = np.cumprod(1 + returns_pct / 100)
        bad = (equity <= 0) | ~np.isfinite(equity)
        if bad.any():
            first = int(bad.argmax())
            if equity[first] <= 0:
                return -100.0, 100.0
            return np.nan, np.nan
        high_water_mark = np.maximum.accumulate(np.maximum(equity, 1.0))
        drawdown = (high_water_mark - equity) / high_water_mark * 100
    return float((equity[-1] - 1) * 100), float(max(drawdown.max(), 0.0))


def _max_consecutive(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    breaks = np.concatenate(([-1], np.flatnonzero(~mask), [len(mask)]))
    return int(np.diff(breaks).max() - 1)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else np.nan


def _mean_skipna(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else np.nan


def _metrics_from_arrays(
    stop,
    target,
    exit_mode,
    returns,
    atr_returns,
    trp_returns,
    pct_bar_returns,
    atrp_bar_returns,
    trp_bar_returns,
    realized_r,
    bars_held,
    entry_atrp,
    entry_trp,
    stop_distance,
    target_distance,
    commission,
    exit_counts,
) -> dict:
    n = len(returns)
    winning = returns[returns > 0]
    losing = returns[returns < 0]
    valid_atr = atr_returns[np.isfinite(atr_returns)]
    valid_pct_bar = pct_bar_returns[np.isfinite(pct_bar_returns)]
    valid_atrp_bar = atrp_bar_returns[np.isfinite(atrp_bar_returns)]
    valid_trp = trp_returns[np.isfinite(trp_returns)]
    valid_trp_bar = trp_bar_returns[np.isfinite(trp_bar_returns)]
    r_mask = np.isfinite(realized_r) & np.isfinite(bars_held) & (bars_held > 0)
    valid_r = realized_r[r_mask]
    valid_r_bars = bars_held[r_mask]
    finite_r = realized_r[np.isfinite(realized_r)]

    win_rate = float((returns > 0).mean() * 100) if n else np.nan
    average_win = float(winning.mean()) if len(winning) else np.nan
    average_loss = float(losing.mean()) if len(losing) else np.nan
    expectancy = float(returns.mean()) if n else np.nan
    cumulative_return_pct, max_drawdown_pct = calculate_compounded_return_metrics(
        returns
    )
    if np.isfinite(realized_r).all():
        cumulative_r = np.concatenate(([0.0], np.cumsum(realized_r)))
        max_drawdown_r = float(
            np.max(np.maximum.accumulate(cumulative_r) - cumulative_r)
        )
    else:
        max_drawdown_r = np.nan
    expectancy_r = _mean(valid_r)
    expectancy_r_per_bar = (
        float(valid_r.sum() / valid_r_bars.sum())
        if len(valid_r) and valid_r_bars.sum() > 0
        else np.nan
    )
    planned_rrr = (
        float(target) / float(stop)
        if stop != -1 and target != -1 and stop > 0
        else np.nan
    )
    realized_rrr = (
        average_win / abs(average_loss)
        if len(winning) and len(losing) and average_loss != 0
        else np.nan
    )
    break_even = (
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
    stop_n, target_n, ma_n, eod_n = exit_counts
    return {
        "ExitMode": exit_mode,
        "StopParameter": stop,
        "TargetParameter": target,
        "NumberOfTrades": n,
        "MaxConsecutiveLosses": _max_consecutive(returns < 0),
        "WinRatePct": win_rate,
        "AverageWinPct": average_win,
        "AverageLossPct": average_loss,
        "AverageWinATRP": _mean(valid_atr[valid_atr > 0]),
        "AverageLossATRP": _mean(valid_atr[valid_atr < 0]),
        "RealizedRRR": realized_rrr,
        "BreakEvenWinRatePct": break_even,
        "PlannedRRR": planned_rrr,
        "PlannedBreakEvenWinRatePct": planned_break_even,
        "ExpectancyPct": expectancy,
        "ExpectancyPctPerBar": _mean(valid_pct_bar),
        "CumulativeReturnPct": cumulative_return_pct,
        "ExpectancyATRP": _mean(valid_atr),
        "ExpectancyATRPPerBar": _mean(valid_atrp_bar),
        "ExpectancyTRP": _mean(valid_trp),
        "ExpectancyTRPPerBar": _mean(valid_trp_bar),
        "ExpectancyR": expectancy_r,
        "ExpectancyRPerBar": expectancy_r_per_bar,
        "ExpectancyRPer100Bars": (
            expectancy_r_per_bar * 100
            if np.isfinite(expectancy_r_per_bar)
            else np.nan
        ),
        "TotalRealizedR": float(finite_r.sum()) if len(finite_r) else np.nan,
        "MaxDrawdownR": max_drawdown_r,
        "MaxDrawdownPct": max_drawdown_pct,
        "AverageEntryATRP": _mean_skipna(entry_atrp) if n else np.nan,
        "AverageEntryTRP": _mean_skipna(entry_trp) if n else np.nan,
        "AverageStopDistancePct": _mean_skipna(stop_distance) if n else np.nan,
        "AverageTargetDistancePct": _mean_skipna(target_distance) if n else np.nan,
        "AverageBarsHeld": float(bars_held.mean()) if n else np.nan,
        "StopExitPct": stop_n / n * 100 if n else np.nan,
        "TargetExitPct": target_n / n * 100 if n else np.nan,
        "MovingAverageExitPct": ma_n / n * 100 if n else np.nan,
        "EndOfDataExitPct": eod_n / n * 100 if n else np.nan,
        "AverageCommissionPct": float(commission.mean()) if n else np.nan,
    }


def calculate_metrics(
    trades: pd.DataFrame,
    stop: float,
    target: float,
    exit_mode: str,
) -> dict:
    """Drop-in replacement for base.calculate_metrics (DataFrame input)."""

    n = len(trades)

    def column(name: str) -> np.ndarray:
        return (
            trades[name].to_numpy(float) if name in trades else np.full(n, np.nan)
        )

    counts = trades["ExitReason"].value_counts()
    exit_counts = tuple(
        int(counts.get(reason, 0))
        for reason in ("Stop", "Target", "MovingAverage", "EndOfData")
    )
    return _metrics_from_arrays(
        stop,
        target,
        exit_mode,
        column("NetReturnPct"),
        column("NetReturnATRP"),
        column("NetReturnTRP"),
        column("NetReturnPctPerBar"),
        column("NetReturnATRPPerBar"),
        column("NetReturnTRPPerBar"),
        column("RealizedR"),
        column("BarsHeld"),
        column("EntryATRP"),
        column("EntryTRP"),
        column("StopDistancePct"),
        column("TargetDistancePct"),
        column("CommissionPct"),
        exit_counts,
    )


# =============================================================================
# Grid simulation per scenario
# =============================================================================
_CTX: dict = {}


def _init_worker(context: dict) -> None:
    global _CTX
    _CTX = context


def _next_cross(cross: np.ndarray) -> np.ndarray:
    """next[j] = first index >= j where cross is True, else len(cross)."""

    n = len(cross)
    positions = np.where(cross, np.arange(n), n)
    result = np.full(n + 1, n, dtype=np.int64)
    result[:n] = np.minimum.accumulate(positions[::-1])[::-1]
    return result


def _accept_mask(exit_idx: np.ndarray, entry_idx: np.ndarray, max_open) -> np.ndarray:
    """Open-trade limit for every stop/target pair at once."""

    entries, stops, targets = exit_idx.shape
    if max_open is None:
        return np.ones(exit_idx.shape, dtype=bool)
    accept = np.zeros(exit_idx.shape, dtype=bool)
    if max_open == 1:
        last_exit = np.full((stops, targets), -1, dtype=np.int64)
        for e in range(entries):
            take = last_exit <= entry_idx[e]
            accept[e] = take
            last_exit = np.where(take, exit_idx[e], last_exit)
        return accept
    entry_list = entry_idx.tolist()
    for s in range(stops):
        for t in range(targets):
            exits = exit_idx[:, s, t].tolist()
            heap: list[int] = []
            for e in range(entries):
                while heap and heap[0] <= entry_list[e]:
                    heapq.heappop(heap)
                if len(heap) >= max_open:
                    continue
                accept[e, s, t] = True
                heapq.heappush(heap, exits[e])
    return accept


def _run_seed(seed: int) -> list[dict]:
    c = _CTX
    data = c["data"]
    close, high, nlow = c["close"], c["high"], c["nlow"]
    n = len(close)
    stops_arr, targets_arr = c["stops_arr"], c["targets_arr"]
    no_stop, no_target = stops_arr == -1, targets_arr == -1
    S, T = len(stops_arr), len(targets_arr)
    use_atrp, use_trp = c["use_atrp"], c["use_trp"]
    slippage, commission = c["slippage"], c["commission"]
    conservative = c["conservative"]

    entries = base.select_entries(
        data=data,
        number_of_entries=c["number_of_entries"],
        seed=seed,
        sma_period=c["sma_period"],
        trade_direction=c["trade_direction"],
        ma_enabled=c["ma_entry_enabled"],
        atrp_values=c["atrp_array"] if use_atrp else None,
        trp_values=c["trp_array"] if use_trp else None,
        trades_per_year=c["trades_per_year"],
    )
    idx = entries["EntryIndex"].to_numpy(dtype=np.int64)
    is_long = entries["Direction"].to_numpy() == "Long"
    E = len(idx)
    entry_atrp = (
        entries["EntryATRP"].to_numpy(float) if use_atrp else np.full(E, np.nan)
    )
    if use_trp:
        if "EntryTRP" not in entries:
            raise ValueError("TRP exits require per-year entry selection.")
        entry_trp = entries["EntryTRP"].to_numpy(float)
    else:
        entry_trp = np.full(E, np.nan)
    if E and np.any(np.diff(idx) < 0):
        order = np.argsort(idx, kind="stable")
        idx, is_long = idx[order], is_long[order]
        entry_atrp, entry_trp = entry_atrp[order], entry_trp[order]

    scale = entry_atrp if use_atrp else (entry_trp if use_trp else None)
    if scale is not None and not (np.isfinite(scale) & (scale > 0)).all():
        name = "ATRP" if use_atrp else "TRP"
        raise ValueError(f"A positive entry {name} is required for {name} exits.")
    if scale is None:
        stop_dist = np.broadcast_to(stops_arr[None, :], (E, S))
        target_dist = np.broadcast_to(targets_arr[None, :], (E, T))
    else:
        stop_dist = stops_arr[None, :] * scale[:, None]
        target_dist = targets_arr[None, :] * scale[:, None]

    exit_idx = np.empty((E, S, T), dtype=np.int64)
    exit_ref = np.empty((E, S, T))
    reason = np.empty((E, S, T), dtype=np.int8)
    next_long, next_short = c["next_long"], c["next_short"]

    for e in range(E):
        i = int(idx[e])
        start = i + 1
        ref = close[i]
        neg_cummin = np.maximum.accumulate(nlow[start:])
        cummax = np.maximum.accumulate(high[start:])
        if is_long[e]:
            sp = np.where(no_stop, -np.inf, ref * (1 - stop_dist[e] / 100))
            tp = np.where(no_target, np.inf, ref * (1 + target_dist[e] / 100))
            sk = np.searchsorted(neg_cummin, -sp, side="left")
            tk = np.searchsorted(cummax, tp, side="left")
            ma = next_long[start] if next_long is not None else n
        else:
            sp = np.where(no_stop, np.inf, ref * (1 + stop_dist[e] / 100))
            tp = np.where(no_target, -np.inf, ref * (1 - target_dist[e] / 100))
            sk = np.searchsorted(cummax, sp, side="left")
            tk = np.searchsorted(neg_cummin, -tp, side="left")
            ma = next_short[start] if next_short is not None else n
        sidx, tidx = start + sk, start + tk
        first = np.minimum(sidx[:, None], tidx[None, :])
        stop_hit = sidx[:, None] == first
        target_hit = tidx[None, :] == first
        use_stop = stop_hit & (~target_hit | conservative)
        ex_ref = np.where(use_stop, sp[:, None], tp[None, :])
        ex_reason = np.where(use_stop, _STOP, _TARGET)
        if ma < n:
            ma_mask = first > ma
            ex_idx = np.where(ma_mask, ma, first)
            ex_ref = np.where(ma_mask, close[ma], ex_ref)
            ex_reason = np.where(ma_mask, _MA, ex_reason)
        else:
            eod = first >= n
            ex_idx = np.where(eod, n - 1, first)
            ex_ref = np.where(eod, close[n - 1], ex_ref)
            ex_reason = np.where(eod, _EOD, ex_reason)
        exit_idx[e], exit_ref[e], reason[e] = ex_idx, ex_ref, ex_reason

    accept = _accept_mask(exit_idx, idx, c["max_open_trades"])

    # Vectorized trade economics for every (entry, stop, target).
    sign = np.where(is_long, 1.0, -1.0)
    ref_e = close[idx]
    sign3 = sign[:, None, None]
    entry_execution = (ref_e * (1 + sign * slippage / 100))[:, None, None]
    with np.errstate(all="ignore"):
        exit_execution = exit_ref * (1 - sign3 * slippage / 100)
        gross = sign3 * (exit_execution / entry_execution - 1) * 100
        net = gross - 2 * commission
        bars = (exit_idx - idx[:, None, None]).astype(float)
        positive_bars = bars > 0
        net_atrp = net / entry_atrp[:, None, None] if use_atrp else np.full(net.shape, np.nan)
        net_trp = net / entry_trp[:, None, None] if use_trp else np.full(net.shape, np.nan)
        pct_bar = np.where(positive_bars, net / bars, np.nan)
        atrp_bar = np.where(positive_bars & np.isfinite(net_atrp), net_atrp / bars, np.nan)
        trp_bar = np.where(positive_bars & np.isfinite(net_trp), net_trp / bars, np.nan)
        risk = np.where(no_stop[None, :], base.NO_STOP_RISK_PCT, stop_dist)
        realized = net / risk[:, :, None]
    stop_col = np.where(no_stop[None, :], np.nan, stop_dist)
    target_col = np.where(no_target[None, :], np.nan, target_dist)

    def to_grid_first(array):
        return np.ascontiguousarray(np.moveaxis(array, 0, -1))

    net_g, net_atrp_g, net_trp_g = map(to_grid_first, (net, net_atrp, net_trp))
    pct_bar_g, atrp_bar_g, trp_bar_g = map(to_grid_first, (pct_bar, atrp_bar, trp_bar))
    realized_g, bars_g = to_grid_first(realized), to_grid_first(bars)
    reason_g, accept_g = to_grid_first(reason), to_grid_first(accept)
    stop_col_g, target_col_g = np.ascontiguousarray(stop_col.T), np.ascontiguousarray(target_col.T)
    commission_total = np.full(E, 2 * commission)

    exit_mode = "ATRP" if use_atrp else ("TRP" if use_trp else "Percent")
    unit = (
        "ATRP multiples"
        if use_atrp
        else ("TRP multiples" if use_trp else "Percent of entry price")
    )
    constants = {
        "Seed": seed,
        "Ticker": c["ticker"],
        "TradingTimeframe": c["trading_timeframe"],
        "EntryMAEnabled": c["ma_entry_enabled"],
        "ParameterUnit": unit,
        "ATRTimeframe": c["atr_timeframe"] if use_atrp else "N/A",
        "ATRPeriod": c["atr_period"] if use_atrp else np.nan,
    }
    records = []
    stop_list, target_list = c["stop_values"], c["target_values"]
    for s in range(S):
        for t in range(T):
            m = accept_g[s, t]
            counts = np.bincount(reason_g[s, t][m], minlength=4)
            metrics = _metrics_from_arrays(
                stop_list[s],
                target_list[t],
                exit_mode,
                net_g[s, t][m],
                net_atrp_g[s, t][m],
                net_trp_g[s, t][m],
                pct_bar_g[s, t][m],
                atrp_bar_g[s, t][m],
                trp_bar_g[s, t][m],
                realized_g[s, t][m],
                bars_g[s, t][m],
                entry_atrp[m],
                entry_trp[m],
                stop_col_g[s][m],
                target_col_g[t][m],
                commission_total[m],
                (int(counts[_STOP]), int(counts[_TARGET]), int(counts[_MA]), int(counts[_EOD])),
            )
            records.append(
                {**constants, "SelectedEntries": int(m.sum()), **metrics}
            )
    return records


def run_sensitivity(
    data: pd.DataFrame,
    seeds: Iterable[int],
    number_of_entries: int = 25,
    trades_per_year: int | None = None,
    stops: list[float] | None = None,
    targets: list[float] | None = None,
    sma_period: int = 200,
    trade_direction: str = "Long",
    commission: float = 0.0,
    slippage: float = 0.0,
    conservative: bool = True,
    ma_entry_enabled: bool = base.MA_ENTRY_ENABLED,
    ma_exit_enabled: bool = False,
    use_atrp: bool = True,
    atrp_values: pd.Series | np.ndarray | None = None,
    use_trp: bool = False,
    trp_values: pd.Series | np.ndarray | None = None,
    atr_timeframe: str = base.RANGE_TIMEFRAME,
    atr_period: int = base.RANGE_PERIOD,
    progress_callback: Callable[[int, int, int, bool], None] | None = None,
    ticker: str = base.TICKER,
    trading_timeframe: str = base.INTERVAL,
    max_open_trades: int | None = base.MAX_OPEN_TRADES,
    workers: int | None = 1,
) -> pd.DataFrame:
    """Drop-in replacement for base.run_sensitivity.

    workers: number of processes (1 = in-process, 0/None = all CPU cores).
    """

    seed_values = [int(seed) for seed in seeds]
    default_stops = base.RANGE_STOP_MULTIPLIERS if (use_atrp or use_trp) else base.DEFAULT_PERCENT_STOPS
    default_targets = base.RANGE_TARGET_MULTIPLIERS if (use_atrp or use_trp) else base.DEFAULT_PERCENT_TARGETS
    stop_values = default_stops if stops is None else stops
    target_values = default_targets if targets is None else targets

    if not seed_values:
        raise ValueError("At least one random seed is required.")
    if number_of_entries < 1:
        raise ValueError("The number of entries must be at least 1.")
    if max_open_trades is not None and max_open_trades < 1:
        raise ValueError("--max-open-trades must be at least 1.")
    if not stop_values or not target_values:
        raise ValueError("At least one stop and one target are required.")
    if use_atrp and atrp_values is None:
        raise ValueError("ATRP values are required when ATRP exits are enabled.")
    if use_trp and trp_values is None:
        raise ValueError("TRP values are required when TRP exits are enabled.")
    if use_atrp and use_trp:
        raise ValueError("Use ATRP or TRP exits, not both.")
    if trade_direction not in {"Long", "Short", "Both"}:
        raise ValueError("Trade direction must be Long, Short, or Both.")

    atrp_array = None if atrp_values is None else np.asarray(atrp_values, dtype=float)
    trp_array = None if trp_values is None else np.asarray(trp_values, dtype=float)
    if atrp_array is not None and len(atrp_array) != len(data):
        raise ValueError("ATRP values must align one-to-one with trading bars.")
    if trp_array is not None and len(trp_array) != len(data):
        raise ValueError("TRP values must align one-to-one with trading bars.")

    close = data["Close"].to_numpy(dtype=float)
    high = data["High"].to_numpy(dtype=float)
    low = data["Low"].to_numpy(dtype=float)
    n = len(close)
    next_long = next_short = None
    if ma_exit_enabled:
        sma = (
            data["Close"]
            .rolling(window=int(sma_period), min_periods=int(sma_period))
            .mean()
            .to_numpy()
        )
        valid = np.zeros(n, dtype=bool)
        valid[1:] = np.isfinite(sma[:-1]) & np.isfinite(sma[1:])
        previous_close = np.concatenate(([np.nan], close[:-1]))
        previous_sma = np.concatenate(([np.nan], sma[:-1]))
        with np.errstate(invalid="ignore"):
            cross_long = valid & (previous_close >= previous_sma) & (close < sma)
            cross_short = valid & (previous_close <= previous_sma) & (close > sma)
        next_long, next_short = _next_cross(cross_long), _next_cross(cross_short)

    context = {
        "data": data,
        "close": close,
        "high": high,
        "nlow": -low,
        "next_long": next_long,
        "next_short": next_short,
        "stop_values": list(stop_values),
        "target_values": list(target_values),
        "stops_arr": np.asarray(stop_values, dtype=float),
        "targets_arr": np.asarray(target_values, dtype=float),
        "use_atrp": use_atrp,
        "use_trp": use_trp,
        "atrp_array": atrp_array,
        "trp_array": trp_array,
        "slippage": float(slippage),
        "commission": float(commission),
        "conservative": bool(conservative),
        "number_of_entries": number_of_entries,
        "trades_per_year": trades_per_year,
        "sma_period": sma_period,
        "trade_direction": trade_direction,
        "ma_entry_enabled": ma_entry_enabled,
        "max_open_trades": max_open_trades,
        "ticker": ticker,
        "trading_timeframe": trading_timeframe,
        "atr_timeframe": atr_timeframe,
        "atr_period": atr_period,
    }

    worker_count = (os.cpu_count() or 1) if not workers else int(workers)
    worker_count = max(1, min(worker_count, len(seed_values)))
    records: list[dict] = []
    total = len(seed_values)

    def finish(done: int, seed: int, seed_records: list[dict]) -> None:
        records.extend(seed_records)
        if progress_callback is not None:
            progress_callback(done, total, seed, ma_entry_enabled)

    if worker_count == 1:
        _init_worker(context)
        for done, seed in enumerate(seed_values, start=1):
            finish(done, seed, _run_seed(seed))
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_worker,
            initargs=(context,),
        ) as pool:
            for done, (seed, seed_records) in enumerate(
                zip(seed_values, pool.map(_run_seed, seed_values)), start=1
            ):
                finish(done, seed, seed_records)

    return pd.DataFrame.from_records(records)
