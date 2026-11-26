"""
QQQ Drawdown / Recovery and Stop-Loss Analysis
==============================================

Studies stop-loss behaviour independently of any entry signal.

Price paths (--method)
    historical   Entries on the one observed price series.
    stationary   Stationary bootstrap (random block lengths, circular).
    circular     Circular block bootstrap (fixed block length).
    fhs          Filtered historical simulation (GJR-GARCH).

Analyses (--analysis)
    recovery     P(stop hit within W1), then P(recovery to entry within W2).
    compare      Stop versus Hold over the same horizon H.
    walkforward  Choose the stop on past data, apply it to the next year.

Conventions
    * Entry is at the Open of bar 0.
    * Bar 0's Low counts toward the stop.
    * A window of N bars means bar 0 plus the following N bars.
    * Recovery is only tested AFTER the stop-hit bar, because daily data
      does not show whether the High came before or after the Low.

See README.md. Run  python drawdown-dynamics.py --help
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Defaults (all can be overridden on the command line)
# =============================================================================

TICKER = "QQQ"
START_DATE = "1999-03-10"

WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252, "24M": 504}

DEFAULT_STOPS = [round(0.05 * i, 2) for i in range(1, 11)]
DEFAULT_BLOCKS = ["21", "63", "126", "252"]
METHODS = ["historical", "stationary", "circular", "fhs"]

OBJECTIVES = {
    # name: (column, description). Larger is always better.
    "mean_log": ("MeanLogGrowth", "mean log growth"),
    "median_log": ("MedianLogGrowth", "median log growth"),
    "return_over_es": ("ReturnOverES", "mean return / |expected shortfall|"),
    "es": ("ES", "expected shortfall (least negative is best)"),
}

OHLC = ["Open", "High", "Low", "Close"]
NO_STOP = 0.0          # StopPct value that means "Hold"
EPS = 1e-12            # tolerance for threshold comparisons


# =============================================================================
# Data
# =============================================================================

def clean_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    """Validate, sort and de-duplicate daily OHLC data."""
    missing = [c for c in OHLC if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    data = data[OHLC].apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    before = len(data)
    data = data.dropna().sort_index()
    data = data[~data.index.duplicated(keep="first")]
    if len(data) < before:
        print(f"WARNING: dropped {before - len(data)} rows "
              "(missing values or duplicate dates).")

    if len(data) < 2 or (data <= 0).any().any():
        raise RuntimeError("Need at least 2 rows of strictly positive prices.")

    # High must be the largest and Low the smallest value of each bar.
    high = data[OHLC].max(axis=1)
    low = data[OHLC].min(axis=1)
    bad = ((high / data["High"] - 1 > 1e-6) | (1 - low / data["Low"] > 1e-6))
    if bad.any():
        print(f"WARNING: repaired High/Low on {int(bad.sum())} inconsistent "
              f"bars (first: {data.index[bad][0].date()}). Check your data.")
    data["High"] = high
    data["Low"] = low

    gaps = data.index.to_series().diff().dt.days
    if gaps.max() > 10:
        print(f"WARNING: largest gap between rows is {int(gaps.max())} days.")
    return data


def load_data(args, output_dir: Path) -> pd.DataFrame:
    """Read a CSV, or download adjusted daily OHLC from Yahoo Finance."""
    if args.csv:
        raw = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        raw.columns = [str(c).strip().title() for c in raw.columns]
        if args.end:
            raw = raw[pd.to_datetime(raw.index) < pd.Timestamp(args.end)]
        data = clean_ohlc(raw)
    else:
        import yfinance as yf

        print(f"Downloading {args.ticker}...")
        raw = yf.download(
            tickers=args.ticker, start=args.start, end=args.end,
            interval="1d", auto_adjust=True, repair=True, actions=False,
            progress=False, multi_level_index=False, timeout=30,
        )
        if raw.empty:
            raise RuntimeError(f"No data returned for {args.ticker}.")

        # Drop today's bar while it may still be incomplete.
        now = pd.Timestamp.now(tz="America/New_York")
        if (len(raw) and raw.index[-1].date() == now.date() and now.hour < 17):
            raw = raw.iloc[:-1]
        data = clean_ohlc(raw)

        # Snapshot: adjusted prices change over time, so keep what was used.
        data.to_csv(output_dir / f"{args.ticker.lower()}_prices.csv",
                    index_label="Date")

    print(f"History: {data.index[0].date()} to {data.index[-1].date()} "
          f"({len(data):,} trading days)")
    return data


# =============================================================================
# Entry selection (historical method)
# =============================================================================

def select_entries(data: pd.DataFrame, per_month, seed: int) -> np.ndarray:
    """
    Return entry bar indices: `per_month` random days from each calendar
    month, or every day when per_month is None.

    No entries are removed here. Each analysis keeps only the entries that
    have enough future history for the windows it is evaluating.
    """
    if per_month is None:
        return np.arange(len(data))

    months = data.index.to_period("M")
    rng = np.random.default_rng(seed)
    chosen = []
    for month in months.unique():
        days = np.flatnonzero(months == month)
        chosen.append(rng.choice(days, min(per_month, len(days)),
                                 replace=False))
    return np.sort(np.concatenate(chosen))


# =============================================================================
# Path construction
#
# Every method produces the same object: a dict of arrays "o","h","l","c",
# each shaped (n_trades, n_bars), expressed as a multiple of the entry price.
# The same analysis code then runs on all of them.
# =============================================================================

def historical_paths(data, entries, n_bars: int) -> dict:
    """Windows starting at each entry. Bars past the end of data are NaN."""
    idx = entries[:, None] + np.arange(n_bars)[None, :]
    valid = idx < len(data)
    safe = np.where(valid, idx, 0)
    entry_price = data["Open"].to_numpy(float)[entries][:, None]

    paths = {}
    for key, column in zip("ohlc", OHLC):
        values = data[column].to_numpy(float)[safe] / entry_price
        values[~valid] = np.nan
        paths[key] = values
    # Number of bars that really exist for each entry.
    paths["length"] = valid.sum(axis=1)
    return paths


def daily_relatives(data: pd.DataFrame) -> np.ndarray:
    """Each day's O, H, L, C divided by the previous Close. Shape (n-1, 4)."""
    previous_close = data["Close"].to_numpy(float)[:-1][:, None]
    return data[OHLC].to_numpy(float)[1:] / previous_close


def bootstrap_indices(n_obs, n_paths, n_bars, block, method, rng):
    """
    Row indices for a circular bootstrap.

    stationary: each bar starts a new block with probability 1/block, so
                block lengths are geometric with mean `block`.
    circular:   a new block starts every `block` bars.
    """
    if block < 1:
        raise ValueError("Block length must be >= 1.")

    positions = np.arange(n_bars)
    if method == "stationary":
        new_block = rng.random((n_paths, n_bars)) < 1.0 / block
        new_block[:, 0] = True
    elif method == "circular":
        size = max(1, int(round(block)))
        new_block = np.broadcast_to(positions % size == 0,
                                    (n_paths, n_bars))
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")

    # Position in the path where each bar's current block began.
    block_start = np.maximum.accumulate(
        np.where(new_block, positions, 0), axis=1)
    # A random source row for every possible block start.
    draws = rng.integers(0, n_obs, size=(n_paths, n_bars))
    origin = np.take_along_axis(draws, block_start, axis=1)
    return (origin + positions - block_start) % n_obs


def relatives_to_paths(rel: np.ndarray) -> dict:
    """Chain (n_paths, n_bars, 4) relatives into entry-normalised prices."""
    close = np.cumprod(rel[:, :, 3], axis=1)
    previous = np.ones_like(close)
    previous[:, 1:] = close[:, :-1]
    entry = rel[:, 0, 0][:, None]          # Open of bar 0

    paths = {key: previous * rel[:, :, i] / entry
             for i, key in enumerate("ohl")}
    paths["c"] = close / entry
    paths["length"] = np.full(len(close), close.shape[1])
    return paths


def bootstrap_paths(data, method, block, n_paths, n_bars, seed) -> dict:
    rel = daily_relatives(data)
    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(len(rel), n_paths, n_bars, block, method, rng)
    return relatives_to_paths(rel[idx])


# ----------------------------------------------------------------------------
# Filtered historical simulation
# ----------------------------------------------------------------------------

def fit_fhs(data: pd.DataFrame) -> dict:
    """
    Fit constant-mean GJR-GARCH(1,1) to close-to-close log returns, then
    standardise each day's O, H, L, C log relatives by that day's volatility.
    The four values of a day are resampled together.
    """
    try:
        from arch import arch_model
    except ImportError as error:
        raise RuntimeError("--method fhs needs:  pip install arch") from error

    logs = 100.0 * np.log(daily_relatives(data))
    result = arch_model(logs[:, 3], mean="Constant", vol="GARCH",
                        p=1, o=1, q=1, dist="t", rescale=False).fit(disp="off")
    if result.convergence_flag != 0:
        print("WARNING: GARCH fit did not converge; do not trust FHS results.")

    p = result.params
    sigma = np.asarray(result.conditional_volatility, float)
    model = {
        "mu": float(p["mu"]), "omega": float(p["omega"]),
        "alpha": float(p["alpha[1]"]), "gamma": float(p["gamma[1]"]),
        "beta": float(p["beta[1]"]),
        "sigma": sigma,
        "z": (logs - float(p["mu"])) / sigma[:, None],
    }
    model["persistence"] = (model["alpha"] + 0.5 * model["gamma"]
                            + model["beta"])
    print(f"GJR-GARCH persistence: {model['persistence']:.4f}")
    if model["persistence"] >= 0.999:
        print("WARNING: persistence is close to 1; long FHS paths may be "
              "unrealistic.")

    # Next-day variance after the last observation.
    shock = logs[-1, 3] - model["mu"]
    model["next_variance"] = (
        model["omega"]
        + (model["alpha"] + model["gamma"] * (shock < 0)) * shock ** 2
        + model["beta"] * sigma[-1] ** 2)
    return model


def fhs_paths(model, n_paths, n_bars, seed, start="random") -> dict:
    """
    start="random":  starting volatility drawn from history (a typical entry).
    start="current": starting volatility is today's forecast.
    """
    rng = np.random.default_rng(seed)
    z = model["z"][rng.integers(0, len(model["z"]), (n_paths, n_bars))]

    if start == "current":
        variance = np.full(n_paths, model["next_variance"])
    else:
        variance = rng.choice(model["sigma"], n_paths) ** 2

    rel = np.empty((n_paths, n_bars, 4))
    for t in range(n_bars):
        sigma = np.sqrt(variance)
        logs = model["mu"] + sigma[:, None] * z[:, t, :]
        rel[:, t, :] = np.exp(logs / 100.0)
        shock = logs[:, 3] - model["mu"]
        variance = (model["omega"]
                    + (model["alpha"] + model["gamma"] * (shock < 0))
                    * shock ** 2
                    + model["beta"] * variance)
    return relatives_to_paths(rel)


# =============================================================================
# Analysis 1: drawdown and recovery
# =============================================================================

def _median(values):
    return float(np.median(values)) if len(values) else np.nan


def recovery_table(paths, stops, w1_windows, w2_windows,
                   eligibility="per-window") -> pd.DataFrame:
    """
    Stop x W1 x W2 summary.

    eligibility="per-window": a row uses every trade with W1+W2 bars of
        future data. Short windows keep recent entries. Rows can therefore
        cover different periods (see the Entries column).
    eligibility="common": all rows use the trades that have enough data for
        the longest W1 plus the longest W2, so rows are comparable.
    """
    lows, highs, closes = paths["l"], paths["h"], paths["c"]
    longest = max(w1_windows.values()) + max(w2_windows.values())
    records = []

    for stop in stops:
        for w1_name, w1 in w1_windows.items():
            for w2_name, w2 in w2_windows.items():
                need = (longest if eligibility == "common" else w1 + w2) + 1
                rows = np.flatnonzero(paths["length"] >= need)
                record = {"StopPct": stop, "W1": w1_name, "W2": w2_name,
                          "Entries": len(rows)}

                if len(rows):
                    touched = lows[rows, :w1 + 1] <= (1 - stop) + EPS
                    hit = touched.any(axis=1)
                    first = touched.argmax(axis=1)[hit]    # 0 = entry day
                    rows_hit = rows[hit]
                else:
                    first = rows_hit = np.array([], dtype=int)

                n_hit = len(rows_hit)
                cols = first[:, None] + np.arange(w2 + 1)[None, :]
                take = rows_hit[:, None]

                # Highs strictly after the stop-hit bar.
                back = highs[take, cols[:, 1:]] >= 1 - EPS
                recovered = back.any(axis=1)
                days_back = back.argmax(axis=1)[recovered] + 1
                failed = ~recovered

                worst = lows[take, cols].min(axis=1)[failed] - 1.0
                end = closes[rows_hit, first + w2][failed] - 1.0
                n_rec = int(recovered.sum())

                record.update({
                    "StopHits": n_hit,
                    "StopHitProbability":
                        n_hit / len(rows) if len(rows) else np.nan,
                    "MedianDaysToStop": _median(first),
                    "Recovered": n_rec,
                    "NotRecovered": n_hit - n_rec,
                    "RecoveryProbability": n_rec / n_hit if n_hit else np.nan,
                    "NonRecoveryProbability":
                        (n_hit - n_rec) / n_hit if n_hit else np.nan,
                    "MedianDaysToRecovery": _median(days_back),
                    "MedianWorstDropNonRecovered": _median(worst),
                    # 10th percentile = the bad tail of the worst drops.
                    "P10WorstDropNonRecovered":
                        float(np.quantile(worst, 0.10))
                        if len(worst) else np.nan,
                    "MedianAdditionalDropAfterStop": _median(worst + stop),
                    "MedianEndReturnNonRecovered": _median(end),
                })
                records.append(record)
    return pd.DataFrame(records)


# =============================================================================
# Analysis 2: stop versus Hold
# =============================================================================

def simulate_stop(paths, rows, stop, horizon, slippage, cash_rate):
    """
    Terminal wealth per 1 unit invested at the entry Open.

    Hold: sell at the Close of bar `horizon`.
    Stop: if a Low touches the stop, sell at the stop price, or at the Open
          if the bar opened at or below it (gap). Apply slippage, then hold
          cash to the horizon. The exit horizon is identical to Hold.

    Returns (wealth, triggered, fill, exit_bar).
    """
    hold = paths["c"][rows, horizon]
    none = np.zeros(len(rows), dtype=bool)
    if stop == NO_STOP:
        return hold.copy(), none, np.full(len(rows), np.nan), \
            np.full(len(rows), horizon)

    level = 1.0 - stop
    touched = paths["l"][rows, :horizon + 1] <= level + EPS
    triggered = touched.any(axis=1)
    bar = np.where(triggered, touched.argmax(axis=1), horizon)

    open_at_bar = paths["o"][rows, bar]
    fill = np.where((bar > 0) & (open_at_bar <= level), open_at_bar, level)
    interest = (1.0 + cash_rate) ** ((horizon - bar) / 252.0)
    wealth = np.where(triggered, fill * (1.0 - slippage) * interest, hold)
    return wealth, triggered, np.where(triggered, fill, np.nan), bar


def max_drawdowns(paths, rows, horizon, triggered, wealth, exit_bar):
    """
    Close-to-close maximum drawdown of each trade, starting from the entry
    price. After a stop, the equity is fixed at the exit value.
    """
    equity = paths["c"][rows, :horizon + 1].copy()
    after = triggered[:, None] & (np.arange(horizon + 1) >= exit_bar[:, None])
    equity = np.where(after, wealth[:, None], equity)
    equity = np.concatenate([np.ones((len(rows), 1)), equity], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    return (equity / peak - 1.0).min(axis=1)


def expected_shortfall(returns, alpha):
    """Mean of the worst `alpha` share of returns."""
    k = max(1, int(np.ceil(alpha * len(returns) - 1e-9)))
    return float(np.sort(returns)[:k].mean())


def compare_stops(paths, stops, horizon, slippage=0.0005, cash_rate=0.0,
                  es_alpha=0.05, rows=None) -> pd.DataFrame:
    """One row per stop, plus StopPct 0 = Hold. All use the same trades."""
    if rows is None:
        rows = np.flatnonzero(paths["length"] >= horizon + 1)
    if len(rows) == 0:
        raise RuntimeError("No trade has a complete holding horizon.")

    hold_wealth = paths["c"][rows, horizon]
    hold_log = np.log(hold_wealth)
    records = []

    for stop in [NO_STOP] + [s for s in stops if s != NO_STOP]:
        wealth, triggered, fill, bar = simulate_stop(
            paths, rows, stop, horizon, slippage, cash_rate)
        returns = wealth - 1.0
        logs = np.log(wealth)
        es = expected_shortfall(returns, es_alpha)
        drawdown = max_drawdowns(paths, rows, horizon, triggered, wealth, bar)
        n_trig = int(triggered.sum())

        records.append({
            "StopPct": stop,
            "Trades": len(rows),
            "TriggerRate": n_trig / len(rows),
            # Triggered, yet Hold finished above the stop fill price.
            "WhipsawRate":
                float((hold_wealth[triggered] > fill[triggered]).mean())
                if n_trig else np.nan,
            "MeanReturn": float(returns.mean()),
            "MedianReturn": float(np.median(returns)),
            "MeanLogGrowth": float(logs.mean()),
            "MedianLogGrowth": float(np.median(logs)),
            "ProbLoss": float((returns < 0).mean()),
            "P05Return": float(np.quantile(returns, 0.05)),
            "ES": es,
            "WorstReturn": float(returns.min()),
            "ReturnOverES":
                float(returns.mean() / -es) if es < 0 else np.nan,
            "MedianMaxDrawdown": float(np.median(drawdown)),
            "P05MaxDrawdown": float(np.quantile(drawdown, 0.05)),
            "ProbBeatsHold": float((wealth > hold_wealth).mean()),
            "ProbWorseThanHold": float((wealth < hold_wealth).mean()),
            "DiffMeanLogVsHold": float((logs - hold_log).mean()),
            "DiffESVsHold":
                es - expected_shortfall(hold_wealth - 1.0, es_alpha),
        })
    return pd.DataFrame(records)


def best_stop(table, objective, min_es=None) -> float:
    """
    Best StopPct under the objective. Hold (0) competes and wins ties.
    min_es, e.g. -0.25, keeps only rows whose ES is at least that good.
    Returns NaN if nothing satisfies the constraint.
    """
    column = OBJECTIVES[objective][0]
    candidates = table
    if min_es is not None:
        candidates = candidates[candidates["ES"] >= min_es]
    candidates = candidates.dropna(subset=[column])
    if candidates.empty:
        return np.nan
    # idxmax returns the first maximum; Hold is the first row.
    return float(candidates.loc[candidates[column].idxmax(), "StopPct"])


# ----------------------------------------------------------------------------
# Sampling uncertainty for the historical method
# ----------------------------------------------------------------------------

def historical_diff_intervals(data, paths, entries, stops, horizon, args):
    """
    Approximate 95% interval for DiffMeanLogVsHold.

    Overlapping trades are not independent, so whole blocks of consecutive
    entry MONTHS are resampled (circular block bootstrap of monthly sums and
    counts). The block spans at least twice the holding horizon. With few
    such blocks in the data, treat the interval as a rough guide that is
    likely too narrow.
    """
    rows = np.flatnonzero(paths["length"] >= horizon + 1)
    codes, _ = pd.factorize(data.index[entries[rows]].to_period("M"))
    n_months = codes.max() + 1
    block = max(12, int(np.ceil(2 * horizon / 21)))
    if n_months < 2 * block:
        return {}

    rng = np.random.default_rng(args.seed + 999)
    n_blocks = int(np.ceil(n_months / block))
    starts = rng.integers(0, n_months, (2000, n_blocks))
    picks = (starts[:, :, None] + np.arange(block)) % n_months
    picks = picks.reshape(2000, -1)[:, :n_months]
    counts = np.bincount(codes, minlength=n_months)
    hold_log = np.log(paths["c"][rows, horizon])

    intervals = {}
    for stop in stops:
        wealth = simulate_stop(paths, rows, stop, horizon,
                               args.slippage, args.cash_rate)[0]
        sums = np.bincount(codes, np.log(wealth) - hold_log, n_months)
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
        intervals[stop] = tuple(np.quantile(means, [0.025, 0.975]))
    return intervals


# =============================================================================
# Analysis 3: walk-forward
# =============================================================================

def build_training_paths(train, method, block, args, seed, fhs_start):
    """Paths built ONLY from the training slice."""
    n_bars = args.horizon + 1
    if method == "historical":
        # Every day; entries without a full horizon inside `train` are
        # excluded by compare_stops, which prevents lookahead.
        return historical_paths(train, np.arange(len(train)), n_bars)
    if method == "fhs":
        return fhs_paths(fit_fhs(train), args.paths, n_bars, seed, fhs_start)
    return bootstrap_paths(train, method, block, args.paths, n_bars, seed)


def walk_forward(data, entries, scenarios, args):
    """
    For each calendar year Y: choose the stop using prices dated before Y,
    then apply it to the historical entries made in Y. Test entries need a
    complete horizon. The training window expands each year.
    """
    n_bars = args.horizon + 1
    test_paths = historical_paths(data, entries, n_bars)
    complete = test_paths["length"] >= n_bars
    entry_years = data.index[entries].year
    details, trades = [], []

    for method, block_label in scenarios:
        label = scenario_name(method, block_label)
        print(f"Walk-forward: {label}")
        for year in sorted(set(entry_years)):
            train = data[data.index.year < year]
            if len(train) < args.min_train_years * 252:
                continue
            rows = np.flatnonzero(complete & (entry_years == year))
            if len(rows) == 0:
                continue

            block = resolve_block(block_label, method, train, args)
            paths = build_training_paths(
                train, method, block, args, args.seed + int(year),
                args.fhs_start)
            table = compare_stops(paths, args.stops, args.horizon,
                                  args.slippage, args.cash_rate,
                                  args.es_alpha)
            chosen = best_stop(table, args.objective, args.min_es)
            # If no stop meets the constraint, fall back to Hold and say so.
            met = not np.isnan(chosen)
            applied = chosen if met else NO_STOP

            wealth = simulate_stop(test_paths, rows, applied, args.horizon,
                                   args.slippage, args.cash_rate)[0]
            hold = test_paths["c"][rows, args.horizon]
            details.append({
                "Scenario": label, "TestYear": int(year),
                "TrainEnd": train.index[-1].date(),
                "ChosenStopPct": applied, "ConstraintMet": met,
                "Block": block, "TestTrades": len(rows),
                "MeanReturn": float(wealth.mean() - 1),
                "HoldMeanReturn": float(hold.mean() - 1),
                "MeanLogGrowth": float(np.log(wealth).mean()),
                "HoldMeanLogGrowth": float(np.log(hold).mean()),
            })
            trades.append(pd.DataFrame(
                {"Scenario": label, "Wealth": wealth, "Hold": hold}))

    if not details:
        raise RuntimeError("Not enough history for walk-forward. "
                           "Lower --min-train-years.")

    trades = pd.concat(trades, ignore_index=True)
    summary = []
    for label, group in trades.groupby("Scenario", sort=False):
        for name in ["Wealth", "Hold"]:
            returns = group[name].to_numpy() - 1.0
            summary.append({
                "Scenario": label,
                "Policy": "WalkForwardStop" if name == "Wealth" else "Hold",
                "Trades": len(returns),
                "MeanReturn": float(returns.mean()),
                "MedianReturn": float(np.median(returns)),
                "MeanLogGrowth": float(np.log1p(returns).mean()),
                "MedianLogGrowth": float(np.median(np.log1p(returns))),
                "ES": expected_shortfall(returns, args.es_alpha),
                "WorstReturn": float(returns.min()),
            })
    return pd.DataFrame(details), pd.DataFrame(summary)


# =============================================================================
# Scenarios and block lengths
# =============================================================================

def auto_block(data, method, series="squared") -> float:
    """Politis-White (2004) block length with the 2009 correction (arch)."""
    try:
        from arch.bootstrap import optimal_block_length
    except ImportError as error:
        raise RuntimeError("Block 'auto' needs:  pip install arch") from error
    returns = np.log(daily_relatives(data)[:, 3])
    if series == "squared":
        returns = returns ** 2
    value = float(optimal_block_length(returns)[method].iloc[0])
    value = max(1.0, value)
    return float(np.ceil(value)) if method == "circular" else value


def resolve_block(label, method, data, args):
    if method not in ("stationary", "circular"):
        return np.nan
    if label == "auto":
        return auto_block(data, method, args.auto_series)
    return float(label)


def scenario_name(method, block_label):
    return method if block_label is None else f"{method}_L{block_label}"


def build_scenarios(args):
    scenarios = []
    for method in args.method:
        if method in ("stationary", "circular"):
            scenarios += [(method, b) for b in args.block_lengths]
        else:
            scenarios.append((method, None))
    return scenarios


# =============================================================================
# Command line
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Drawdown/recovery and stop-loss analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--ticker", default=TICKER)
    g.add_argument("--csv", help="CSV with Date,Open,High,Low,Close "
                                 "(adjusted prices) instead of downloading")
    g.add_argument("--start", default=START_DATE)
    g.add_argument("--end", help="exclusive end date, YYYY-MM-DD")
    g.add_argument("--output-dir",
                   default=str(Path(__file__).resolve().parent / "results"))

    g = p.add_argument_group("what to run")
    g.add_argument("--method", nargs="+", default=["historical"],
                   choices=METHODS + ["all"])
    g.add_argument("--analysis", nargs="+",
                   default=["recovery", "compare"],
                   choices=["recovery", "compare", "walkforward", "all"])

    g = p.add_argument_group("sampling")
    g.add_argument("--entries-per-month", default="5",
                   help="historical entries per month, or 'all'")
    g.add_argument("--eligibility", choices=["per-window", "common"],
                   default="per-window",
                   help="recovery table, historical method only")
    g.add_argument("--paths", type=int, default=5000,
                   help="simulated paths (one trade per path)")
    g.add_argument("--block-lengths", nargs="+", default=DEFAULT_BLOCKS,
                   help="trading days, or 'auto'")
    g.add_argument("--auto-series", choices=["squared", "returns"],
                   default="squared")
    g.add_argument("--fhs-start", choices=["random", "current"],
                   default="random")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--n-seeds", type=int, default=1,
                   help="repeat 'compare' with this many consecutive seeds")

    g = p.add_argument_group("stops and execution")
    g.add_argument("--stops", nargs="+", type=float, default=DEFAULT_STOPS,
                   help="fractions, e.g. 0.05 0.10")
    g.add_argument("--horizon", type=int, default=252,
                   help="holding horizon H in trading days")
    g.add_argument("--slippage", type=float, default=0.0005,
                   help="fraction lost on a stop exit")
    g.add_argument("--cash-rate", type=float, default=0.0,
                   help="constant annual rate on cash after a stop")

    g = p.add_argument_group("selection")
    g.add_argument("--objective", choices=list(OBJECTIVES),
                   default="mean_log")
    g.add_argument("--min-es", type=float,
                   help="constraint, e.g. -0.25: ES must be >= this")
    g.add_argument("--es-alpha", type=float, default=0.05)
    g.add_argument("--periods", type=int, default=3,
                   help="historical sub-periods for stability (0 = off)")
    g.add_argument("--min-train-years", type=int, default=8)

    args = p.parse_args(argv)

    if "all" in args.method:
        args.method = METHODS
    if "all" in args.analysis:
        args.analysis = ["recovery", "compare", "walkforward"]
    args.method = list(dict.fromkeys(args.method))
    args.stops = sorted({round(s, 6) for s in args.stops})

    if str(args.entries_per_month).lower() == "all":
        args.entries_per_month = None
    else:
        try:
            args.entries_per_month = int(args.entries_per_month)
        except ValueError:
            p.error("--entries-per-month must be a whole number or 'all'")
        if args.entries_per_month < 1:
            p.error("--entries-per-month must be >= 1")

    for b in args.block_lengths:
        if b != "auto":
            try:
                ok = float(b) >= 1
            except ValueError:
                ok = False
            if not ok:
                p.error("--block-lengths values must be >= 1 or 'auto'")
    if not all(0 < s < 1 for s in args.stops):
        p.error("--stops must be between 0 and 1")
    if min(args.paths, args.n_seeds, args.horizon, args.min_train_years) < 1:
        p.error("--paths, --n-seeds, --horizon and --min-train-years "
                "must be positive")
    if not 0 <= args.slippage < 1 or args.cash_rate <= -1:
        p.error("invalid --slippage or --cash-rate")
    if not 0 < args.es_alpha < 1:
        p.error("--es-alpha must be between 0 and 1")
    return args


def save(frame, path):
    frame.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved: {path}")


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(args, output)
    prefix = args.ticker.lower()

    entries = select_entries(data, args.entries_per_month, args.seed)
    print(f"Historical entry dates selected: {len(entries):,}")
    scenarios = build_scenarios(args)

    recovery, compare, periods, best = [], [], [], []
    blocks_used = {}
    fhs_model = None
    recovery_bars = 2 * max(WINDOWS.values()) + 1

    if {"recovery", "compare"} & set(args.analysis):
        for method, block_label in scenarios:
            label = scenario_name(method, block_label)
            block = resolve_block(block_label, method, data, args)
            if not np.isnan(block):
                blocks_used[label] = block
            if method == "fhs" and fhs_model is None:
                fhs_model = fit_fhs(data)

            n_seeds = 1 if method == "historical" else args.n_seeds
            for k in range(n_seeds):
                seed = args.seed + k
                print(f"Running {label}"
                      + ("" if method == "historical" else f" (seed {seed})"))

                # Each task builds paths of the length it needs.
                def make(n_bars):
                    if method == "historical":
                        return historical_paths(data, entries, n_bars)
                    if method == "fhs":
                        return fhs_paths(fhs_model, args.paths, n_bars,
                                         seed, args.fhs_start)
                    return bootstrap_paths(data, method, block, args.paths,
                                           n_bars, seed)

                tags = {"Ticker": args.ticker, "Scenario": label,
                        "Seed": seed}

                if "recovery" in args.analysis and k == 0:
                    eligibility = (args.eligibility
                                   if method == "historical" else "common")
                    table = recovery_table(
                        make(recovery_bars), args.stops, WINDOWS, WINDOWS,
                        eligibility)
                    recovery.append(table.assign(**tags))

                if "compare" not in args.analysis:
                    continue

                paths = make(args.horizon + 1)
                table = compare_stops(
                    paths, args.stops, args.horizon, args.slippage,
                    args.cash_rate, args.es_alpha)

                # Uncertainty of the paired difference from Hold.
                table["DiffCILow"] = np.nan
                table["DiffCIHigh"] = np.nan
                table["SimulationSE"] = np.nan
                if method == "historical":
                    ci = historical_diff_intervals(
                        data, paths, entries, args.stops, args.horizon, args)
                    for stop, (low, high) in ci.items():
                        mask = table["StopPct"] == stop
                        table.loc[mask, ["DiffCILow", "DiffCIHigh"]] = \
                            [low, high]
                else:
                    rows = np.arange(args.paths)
                    hold_log = np.log(paths["c"][rows, args.horizon])
                    for i, stop in enumerate(table["StopPct"]):
                        wealth = simulate_stop(
                            paths, rows, stop, args.horizon,
                            args.slippage, args.cash_rate)[0]
                        diff = np.log(wealth) - hold_log
                        table.loc[i, "SimulationSE"] = (
                            diff.std(ddof=1) / np.sqrt(len(diff))
                            if len(diff) > 1 else np.nan)
                compare.append(table.assign(**tags))

                chosen = best_stop(table, args.objective, args.min_es)
                best.append({**tags, "Objective": args.objective,
                             "MinES": args.min_es,
                             "BestStopPct": chosen,
                             "Result": "no row meets --min-es"
                             if np.isnan(chosen)
                             else "Hold (no stop)" if chosen == NO_STOP
                             else f"{chosen:.1%} stop"})

                if method == "historical" and args.periods > 1:
                    rows = np.flatnonzero(
                        paths["length"] >= args.horizon + 1)
                    for part in np.array_split(rows, args.periods):
                        if len(part) == 0:
                            continue
                        sub = compare_stops(
                            paths, args.stops, args.horizon, args.slippage,
                            args.cash_rate, args.es_alpha, rows=part)
                        dates = data.index[entries[part]]
                        periods.append(sub.assign(
                            FirstEntry=dates[0].date(),
                            LastEntry=dates[-1].date(), **tags))

    def ordered(frames):
        frame = pd.concat(frames, ignore_index=True)
        front = [c for c in ["Ticker", "Scenario", "Seed",
                             "FirstEntry", "LastEntry"] if c in frame]
        return frame[front + [c for c in frame if c not in front]]

    if recovery:
        save(ordered(recovery), output / f"{prefix}_recovery_summary.csv")
    if compare:
        save(ordered(compare), output / f"{prefix}_stop_vs_hold.csv")
        best = pd.DataFrame(best)
        save(best, output / f"{prefix}_best_stop.csv")
        print(f"\nBest by {OBJECTIVES[args.objective][1]} "
              "(in-sample, not an out-of-sample result):")
        print(best[["Scenario", "Seed", "Result"]].to_string(index=False))
    if periods:
        save(ordered(periods), output / f"{prefix}_by_period.csv")

    if "walkforward" in args.analysis:
        detail, summary = walk_forward(data, entries, scenarios, args)
        save(detail, output / f"{prefix}_walk_forward.csv")
        save(summary, output / f"{prefix}_walk_forward_summary.csv")
        print("\nOut-of-sample summary:")
        print(summary.to_string(index=False))

    config = {**vars(args), "data_first": str(data.index[0].date()),
              "data_last": str(data.index[-1].date()), "rows": len(data),
              "block_lengths_used": blocks_used,
              "numpy": np.__version__, "pandas": pd.__version__,
              "python": sys.version.split()[0]}
    (output / f"{prefix}_run_config.json").write_text(
        json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
