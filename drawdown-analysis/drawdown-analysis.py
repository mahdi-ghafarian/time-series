"""Yahoo Finance Drawdown and Stop-Tolerance Analyzer.

Key conventions:
- Peak drawdown: adjusted Close -> adjusted Close.
- Entry MAE: entry adjusted Close -> subsequent adjusted Low.
- Stop breach: first subsequent Low <= stop level.
- Post-breach recovery: first High >= entry price AFTER breach day.
  The breach day is excluded because daily OHLC cannot reveal intraday order.
- Stop-selection statistics use full holding windows by default to avoid right-censoring bias.
"""
from __future__ import annotations

# =============================================================================
# USER CONFIGURATION
# =============================================================================
SYMBOL = "QQQ"
START_DATE = "1999-03-10"
END_DATE = None
INTERVAL = "1d"
AUTO_ADJUST = True

BIN_WIDTH_PCT = 5.0
MAX_BIN_PCT = None
MIN_EPISODE_DRAWDOWN_PCT = 0.0

ANALYZE_ENTRY_MAE = True
ENTRY_FREQUENCY = "daily"            # daily, weekly, monthly
MAX_HOLDING_BARS = 252               # in days
RECOVERY_THRESHOLD_PCT = 0.0         # MAE recovery level vs entry
PROFIT_TARGET_PCT = None
EXCLUDE_INCOMPLETE_ENTRY_WINDOWS = True

ATR_PERIOD = 20
TREND_MA_PERIOD = 200
HIGH_VOL_QUANTILE = 0.75
ATR_STOP_MULTIPLES = [2, 3, 4, 5, 6, 8, 10]
CANDIDATE_STOPS_PCT = [5, 10, 15, 20, 25, 30, 35, 40]
SURVIVAL_PERCENTILES = [50, 75, 80, 90, 95, 99]

OUTPUT_FOLDER = "results"
OUTPUT_BASENAME = "drawdown_stop_analysis"
SAVE_EXCEL = True
SAVE_CHART = True
SHOW_CHART = True
CHART_DPI = 160
FIGURE_SIZE = (18, 13)

# NOTES
# MAX_HOLDING_BARS = 1M: 21, 3M: 63, 6M: 126, 12M: 252, 24M: 504

# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Settings:
    symbol: str = SYMBOL
    start_date: str = START_DATE
    end_date: Optional[str] = END_DATE
    interval: str = INTERVAL
    auto_adjust: bool = AUTO_ADJUST
    bin_width_pct: float = BIN_WIDTH_PCT
    max_bin_pct: Optional[float] = MAX_BIN_PCT
    min_episode_drawdown_pct: float = MIN_EPISODE_DRAWDOWN_PCT
    analyze_entry_mae: bool = ANALYZE_ENTRY_MAE
    entry_frequency: str = ENTRY_FREQUENCY
    max_holding_bars: int = MAX_HOLDING_BARS
    recovery_threshold_pct: float = RECOVERY_THRESHOLD_PCT
    profit_target_pct: Optional[float] = PROFIT_TARGET_PCT
    exclude_incomplete_entry_windows: bool = EXCLUDE_INCOMPLETE_ENTRY_WINDOWS
    atr_period: int = ATR_PERIOD
    trend_ma_period: int = TREND_MA_PERIOD
    high_vol_quantile: float = HIGH_VOL_QUANTILE


def validate_settings(s: Settings) -> None:
    if s.interval != "1d":
        raise ValueError("This version is designed for INTERVAL='1d'.")
    if s.bin_width_pct <= 0:
        raise ValueError("BIN_WIDTH_PCT must be > 0.")
    if s.max_bin_pct is not None and s.max_bin_pct <= 0:
        raise ValueError("MAX_BIN_PCT must be None or > 0.")
    if s.entry_frequency.lower() not in {"daily", "weekly", "monthly"}:
        raise ValueError("ENTRY_FREQUENCY must be daily, weekly, or monthly.")
    if s.max_holding_bars < 1:
        raise ValueError("MAX_HOLDING_BARS must be >= 1.")
    if not 0 < s.high_vol_quantile < 1:
        raise ValueError("HIGH_VOL_QUANTILE must be between 0 and 1.")
    if s.atr_period < 2 or s.trend_ma_period < 2:
        raise ValueError("ATR_PERIOD and TREND_MA_PERIOD must be >= 2.")
    if any(x <= 0 for x in CANDIDATE_STOPS_PCT + ATR_STOP_MULTIPLES):
        raise ValueError("Candidate stops and ATR multiples must be positive.")


def download_data(s: Settings) -> pd.DataFrame:
    df = yf.download(
        s.symbol, start=s.start_date, end=s.end_date, interval=s.interval,
        auto_adjust=s.auto_adjust, progress=False, actions=False, threads=True,
    )
    if df.empty:
        raise RuntimeError(f"No Yahoo Finance data returned for {s.symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        if s.symbol in df.columns.get_level_values(-1):
            df = df.xs(s.symbol, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    missing = {"Open", "High", "Low", "Close"}.difference(df.columns)
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {sorted(missing)}")
    df = df.sort_index().loc[~df.index.duplicated(keep="last")].copy()
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < max(s.atr_period, s.trend_ma_period) + 2:
        raise RuntimeError("Not enough data for configured indicators.")
    return df


def add_indicators(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    out = df.copy()
    pc = out["Close"].shift(1)
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - pc).abs(),
        (out["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(alpha=1/s.atr_period, adjust=False, min_periods=s.atr_period).mean()
    out["ATRPct"] = out["ATR"] / out["Close"] * 100.0
    out["TrendMA"] = out["Close"].rolling(s.trend_ma_period, min_periods=s.trend_ma_period).mean()
    cutoff = out["ATRPct"].quantile(s.high_vol_quantile)
    out["TrendRegime"] = np.where(out["Close"] >= out["TrendMA"], "Above MA", "Below MA")
    out.loc[out["TrendMA"].isna(), "TrendRegime"] = "Insufficient history"
    out["VolatilityRegime"] = np.where(out["ATRPct"] >= cutoff, "High ATRP", "Normal ATRP")
    out.loc[out["ATRPct"].isna(), "VolatilityRegime"] = "Insufficient history"
    return out


def identify_drawdown_episodes(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    close = df["Close"].to_numpy(float)
    dates = df.index
    atrp = df["ATRPct"].to_numpy(float)
    trend = df["TrendRegime"].to_numpy()
    vol = df["VolatilityRegime"].to_numpy()
    records = []
    peak_i = trough_i = 0
    underwater = False

    def add_episode(recovery_i: Optional[int]):
        dd = (1.0 - close[trough_i] / close[peak_i]) * 100.0
        if dd + 1e-12 < s.min_episode_drawdown_pct:
            return
        end_i = recovery_i if recovery_i is not None else len(close)-1
        pa = atrp[peak_i]
        records.append({
            "PeakDate": dates[peak_i], "PeakPrice": close[peak_i],
            "TroughDate": dates[trough_i], "TroughPrice": close[trough_i],
            "RecoveryDate": dates[recovery_i] if recovery_i is not None else pd.NaT,
            "Recovered": recovery_i is not None, "IsCensored": recovery_i is None,
            "DrawdownPct": dd, "DeclineBars": trough_i-peak_i,
            "RecoveryBars": recovery_i-trough_i if recovery_i is not None else np.nan,
            "TotalBars": end_i-peak_i,
            "DeclineCalendarDays": (dates[trough_i]-dates[peak_i]).days,
            "RecoveryCalendarDays": (dates[recovery_i]-dates[trough_i]).days if recovery_i is not None else np.nan,
            "TotalCalendarDays": (dates[end_i]-dates[peak_i]).days,
            "PeakATRPct": pa,
            "DrawdownATRUnits": dd/pa if np.isfinite(pa) and pa > 0 else np.nan,
            "PeakTrendRegime": trend[peak_i], "PeakVolatilityRegime": vol[peak_i],
        })

    for i in range(1, len(close)):
        if not underwater:
            if close[i] >= close[peak_i]:
                peak_i = trough_i = i
            else:
                underwater = True
                trough_i = i
        else:
            if close[i] < close[trough_i]:
                trough_i = i
            if close[i] >= close[peak_i]:
                add_episode(i)
                peak_i = trough_i = i
                underwater = False
    if underwater:
        add_episode(None)
    return pd.DataFrame(records)


def bin_values(values: pd.Series, s: Settings) -> pd.Categorical:
    observed = float(values.max()) if len(values) and values.notna().any() else s.bin_width_pct
    ceiling = s.max_bin_pct if s.max_bin_pct is not None else max(s.bin_width_pct, np.ceil(observed/s.bin_width_pct)*s.bin_width_pct)
    edges = np.arange(0.0, max(ceiling, s.bin_width_pct) + s.bin_width_pct + 1e-9, s.bin_width_pct)
    if s.max_bin_pct is not None and observed >= edges[-1]:
        edges = np.append(edges, np.inf)
    labels = [f"{a:g}%+" if np.isinf(b) else f"{a:g}-{b:g}%" for a,b in zip(edges[:-1], edges[1:])]
    return pd.cut(values, edges, labels=labels, right=False, include_lowest=True)


def summarize_episodes(ep: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if ep.empty: return pd.DataFrame()
    w = ep.copy(); w["DrawdownBin"] = bin_values(w["DrawdownPct"], s)
    o = w.groupby("DrawdownBin", observed=False).agg(
        Episodes=("DrawdownPct","size"), RecoveredEpisodes=("Recovered","sum"),
        MedianDrawdownPct=("DrawdownPct","median"), MedianDeclineBars=("DeclineBars","median"),
        MedianRecoveryBars=("RecoveryBars","median"), MedianTotalBars=("TotalBars","median"),
        MaxTotalBars=("TotalBars","max"), MedianDrawdownATRUnits=("DrawdownATRUnits","median"),
    ).reset_index()
    o["EpisodePct"] = o["Episodes"] / o["Episodes"].sum() * 100
    o["CumulativeEpisodePct"] = o["EpisodePct"].cumsum()
    o["RecoveryRatePct"] = np.where(o["Episodes"]>0, o["RecoveredEpisodes"]/o["Episodes"]*100, np.nan)
    return o


def select_entry_positions(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily": return np.arange(len(index), dtype=int)
    period = index.to_period("W" if frequency == "weekly" else "M")
    return np.flatnonzero(~pd.Series(period).duplicated().to_numpy())


def analyze_entry_mae(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    close, low, high = [df[c].to_numpy(float) for c in ["Close","Low","High"]]
    atrp = df["ATRPct"].to_numpy(float)
    dates = df.index; records = []
    for start in select_entry_positions(dates, s.entry_frequency):
        if start >= len(df)-1: continue
        end = min(start+s.max_holding_bars, len(df)-1)
        lows = low[start+1:end+1]
        if not len(lows): continue
        entry = close[start]
        trough = start+1+int(np.argmin(lows))
        mae = max(0.0, (1-low[trough]/entry)*100)
        recovery_level = entry*(1+s.recovery_threshold_pct/100)
        rh = np.flatnonzero(high[trough+1:end+1] >= recovery_level) if trough+1 <= end else np.array([])
        recovery = trough+1+int(rh[0]) if len(rh) else None
        target = None
        if s.profit_target_pct is not None:
            level = entry*(1+s.profit_target_pct/100)
            th = np.flatnonzero(high[start+1:end+1] >= level)
            target = start+1+int(th[0]) if len(th) else None
        ea = atrp[start]
        records.append({
            "EntryIndex": start, "WindowEndIndex": end,
            "EntryDate": dates[start], "EntryPrice": entry, "WindowEndDate": dates[end],
            "ObservedBars": end-start, "FullWindowObserved": end-start == s.max_holding_bars,
            "MAETroughDate": dates[trough], "MAEPct": mae, "BarsToMAETrough": trough-start,
            "RecoveredAfterTroughWithinWindow": recovery is not None,
            "RecoveryAfterTroughDate": dates[recovery] if recovery is not None else pd.NaT,
            "BarsFromTroughToRecovery": recovery-trough if recovery is not None else np.nan,
            "BarsFromEntryToPostTroughRecovery": recovery-start if recovery is not None else np.nan,
            "TargetHitWithinWindow": target is not None if s.profit_target_pct is not None else np.nan,
            "TargetDate": dates[target] if target is not None else pd.NaT,
            "BarsToTarget": target-start if target is not None else np.nan,
            "TerminalReturnPct": (close[end]/entry-1)*100,
            "EntryATRPct": ea, "MAEATRUnits": mae/ea if np.isfinite(ea) and ea>0 else np.nan,
            "EntryTrendRegime": df["TrendRegime"].iloc[start],
            "EntryVolatilityRegime": df["VolatilityRegime"].iloc[start],
        })
    return pd.DataFrame(records)


def stop_event_table(df: pd.DataFrame, mae: pd.DataFrame) -> pd.DataFrame:
    """Detailed audit trail: one row per entry x candidate stop."""
    if mae.empty: return pd.DataFrame()
    low = df["Low"].to_numpy(float); high = df["High"].to_numpy(float); dates = df.index
    records = []
    for r in mae.itertuples(index=False):
        start = int(r.EntryIndex); end = int(r.WindowEndIndex); entry = float(r.EntryPrice)
        for stop in sorted(set(CANDIDATE_STOPS_PCT)):
            level = entry*(1-stop/100)
            bh = np.flatnonzero(low[start+1:end+1] <= level)
            breach = start+1+int(bh[0]) if len(bh) else None
            recovery = None
            if breach is not None and breach+1 <= end:
                rh = np.flatnonzero(high[breach+1:end+1] >= entry)
                recovery = breach+1+int(rh[0]) if len(rh) else None
            records.append({
                "StopPct": stop, "EntryDate": r.EntryDate, "EntryPrice": entry,
                "FullWindowObserved": bool(r.FullWindowObserved), "StopLevel": level,
                "Breached": breach is not None, "BreachDate": dates[breach] if breach is not None else pd.NaT,
                "BarsFromEntryToBreach": breach-start if breach is not None else np.nan,
                "RecoveredAfterBreach": recovery is not None,
                "PostBreachRecoveryDate": dates[recovery] if recovery is not None else pd.NaT,
                "BarsFromBreachToRecovery": recovery-breach if recovery is not None else np.nan,
                "BarsFromEntryToPostBreachRecovery": recovery-start if recovery is not None else np.nan,
                "MAEPct": r.MAEPct, "MAEATRUnits": r.MAEATRUnits,
                "TerminalReturnPct": r.TerminalReturnPct, "EntryATRPct": r.EntryATRPct,
                "EntryTrendRegime": r.EntryTrendRegime, "EntryVolatilityRegime": r.EntryVolatilityRegime,
            })
    return pd.DataFrame(records)


def eligible_mae(mae: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if mae.empty: return mae.copy()
    return mae.loc[mae["FullWindowObserved"]].copy() if s.exclude_incomplete_entry_windows else mae.copy()


def eligible_events(events: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if events.empty: return events.copy()
    return events.loc[events["FullWindowObserved"]].copy() if s.exclude_incomplete_entry_windows else events.copy()


def summarize_mae(mae: pd.DataFrame, s: Settings) -> pd.DataFrame:
    if mae.empty: return pd.DataFrame()
    w = mae.copy(); w["MAEBin"] = bin_values(w["MAEPct"], s)
    o = w.groupby("MAEBin", observed=False).agg(
        Entries=("MAEPct","size"), MedianMAEPct=("MAEPct","median"),
        MedianBarsToTrough=("BarsToMAETrough","median"),
        RecoveryRateAfterTroughPct=("RecoveredAfterTroughWithinWindow",lambda x:x.mean()*100),
        MedianBarsFromTroughToRecovery=("BarsFromTroughToRecovery","median"),
        MedianTerminalReturnPct=("TerminalReturnPct","median"), MedianMAEATRUnits=("MAEATRUnits","median"),
    ).reset_index()
    o["EntryPct"] = o["Entries"]/o["Entries"].sum()*100; o["CumulativeEntryPct"] = o["EntryPct"].cumsum()
    return o


def build_stop_selection(events: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Combined Stop Survival + Stop Selection decision sheet."""
    e = eligible_events(events, s)
    if e.empty: return pd.DataFrame()
    rows = []; previous_survival = np.nan; previous_stop = np.nan
    for stop, g in e.groupby("StopPct", sort=True):
        breached = g["Breached"]; b = g.loc[breached]; survival = (~breached).mean()*100
        recovery_bars = b.loc[b["RecoveredAfterBreach"], "BarsFromBreachToRecovery"]
        breach_bars = b["BarsFromEntryToBreach"]
        valid_atr = g.loc[g["EntryATRPct"].notna() & (g["EntryATRPct"]>0), "EntryATRPct"]
        increment = np.nan if pd.isna(previous_survival) else survival-previous_survival
        stop_increment = np.nan if pd.isna(previous_stop) else stop-previous_stop
        row = {
            "StopPct": stop, "Entries": len(g), "BreachedEntries": int(breached.sum()),
            "BreachRatePct": breached.mean()*100, "SurvivalRatePct": survival,
            "IncrementalSurvivalGainPctPoints": increment,
            "SurvivalGainPer5PctStop": (increment/stop_increment*5) if pd.notna(increment) and stop_increment>0 else np.nan,
            "MedianBarsFromEntryToBreach": breach_bars.median(),
            "P25BarsFromEntryToBreach": breach_bars.quantile(.25),
            "P75BarsFromEntryToBreach": breach_bars.quantile(.75),
            "RecoveryRateAfterBreachPct": b["RecoveredAfterBreach"].mean()*100 if len(b) else np.nan,
            "MedianBarsFromBreachToRecovery": recovery_bars.median(),
            "P75BarsFromBreachToRecovery": recovery_bars.quantile(.75),
            "MedianTerminalReturnAmongBreachesPct": b["TerminalReturnPct"].median(),
            "MedianMAEAmongBreachesPct": b["MAEPct"].median(),
            "P90MAEAmongBreachesPct": b["MAEPct"].quantile(.90),
            "MedianStopATRUnits": (stop/valid_atr).median() if len(valid_atr) else np.nan,
        }
        regime_cols = []
        for col in ["EntryTrendRegime","EntryVolatilityRegime"]:
            for name, sg in g.groupby(col, observed=True):
                if name == "Insufficient history": continue
                key = f"Survival_{str(name).replace(' ','_')}Pct"
                row[key] = (~sg["Breached"]).mean()*100
                regime_cols.append(key)
        row["WorstRegimeSurvivalPct"] = min((row[k] for k in regime_cols), default=np.nan)
        row["Interpretation"] = (
            "Exceptional-event tolerance (>=95%)" if survival >= 95 else
            "High-coverage candidate (>=90%)" if survival >= 90 else
            "Minimum practical candidate (>=80%)" if survival >= 80 else
            "Below 80% historical coverage"
        )
        rows.append(row); previous_survival = survival; previous_stop = stop
    return pd.DataFrame(rows)


def atr_stop_table(mae: pd.DataFrame) -> pd.DataFrame:
    v = mae.dropna(subset=["MAEATRUnits"])
    if v.empty: return pd.DataFrame()
    return pd.DataFrame([{
        "ATRMultiple": m, "Entries": len(v), "BreachedEntries": int((v["MAEATRUnits"]>=m).sum()),
        "BreachRatePct": (v["MAEATRUnits"]>=m).mean()*100,
        "SurvivalRatePct": (v["MAEATRUnits"]<m).mean()*100,
    } for m in sorted(set(ATR_STOP_MULTIPLES))])


def percentile_table(ep: pd.DataFrame, mae: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Percentile": p,
        "PeakBasedDrawdownPct": ep["DrawdownPct"].quantile(p/100) if not ep.empty else np.nan,
        "EntryMAEPct": mae["MAEPct"].quantile(p/100) if not mae.empty else np.nan,
        "EntryMAEATRUnits": mae["MAEATRUnits"].quantile(p/100) if not mae.empty else np.nan,
    } for p in SURVIVAL_PERCENTILES])


def regime_table(mae: pd.DataFrame) -> pd.DataFrame:
    if mae.empty: return pd.DataFrame()
    return mae.groupby(["EntryTrendRegime","EntryVolatilityRegime"], observed=True).agg(
        Entries=("MAEPct","size"), MedianMAEPct=("MAEPct","median"),
        P90MAEPct=("MAEPct",lambda x:x.quantile(.90)), P95MAEPct=("MAEPct",lambda x:x.quantile(.95)),
        MedianMAEATRUnits=("MAEATRUnits","median"),
        RecoveryRateAfterTroughPct=("RecoveredAfterTroughWithinWindow",lambda x:x.mean()*100),
        MedianBarsFromTroughToRecovery=("BarsFromTroughToRecovery","median"),
        MedianTerminalReturnPct=("TerminalReturnPct","median"),
    ).reset_index()


def create_overview(df, ep, mae, comparable, s):
    completed = ep[ep["Recovered"]] if not ep.empty else ep
    latest = ep.iloc[-1] if not ep.empty and not ep.iloc[-1]["Recovered"] else None
    vals = [
        ("Symbol",s.symbol),("Data start",df.index.min().date()),("Data end",df.index.max().date()),
        ("Trading bars",len(df)),("Adjusted prices",s.auto_adjust),("Bin width (%)",s.bin_width_pct),
        ("Drawdown episodes",len(ep)),("Completed episodes",len(completed)),
        ("Open/censored episodes",int((~ep["Recovered"]).sum()) if not ep.empty else 0),
        ("Largest drawdown (%)",ep["DrawdownPct"].max() if not ep.empty else np.nan),
        ("Current open drawdown (%)",latest["DrawdownPct"] if latest is not None else 0.0),
        ("Entry frequency",s.entry_frequency),("All entry records",len(mae)),("Stop-stat entry records",len(comparable)),
        ("Holding window (bars)",s.max_holding_bars),("Incomplete windows excluded",s.exclude_incomplete_entry_windows),
        ("Median entry MAE (%)",comparable["MAEPct"].median() if not comparable.empty else np.nan),
        ("P90 entry MAE (%)",comparable["MAEPct"].quantile(.90) if not comparable.empty else np.nan),
        ("P95 entry MAE (%)",comparable["MAEPct"].quantile(.95) if not comparable.empty else np.nan),
        ("P99 entry MAE (%)",comparable["MAEPct"].quantile(.99) if not comparable.empty else np.nan),
    ]
    return pd.DataFrame(vals, columns=["Metric","Value"])


def plot_dashboard(es, ms, selection, regimes, path, s):
    fig, axes = plt.subplots(2,2,figsize=FIGURE_SIZE,constrained_layout=True)
    fig.suptitle(f"{s.symbol} Drawdown and Stop-Tolerance Analysis",fontsize=17,fontweight="bold")
    def style(ax):
        ax.grid(axis="y",alpha=.25); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(labelsize=9)
    def sparse(labels, limit=9):
        n=len(labels)
        if n<=limit:return list(range(n)),labels
        step=int(np.ceil(n/limit)); pos=list(range(0,n,step))
        if pos[-1]!=n-1:pos.append(n-1)
        return pos,[labels[i] for i in pos]
    ax=axes[0,0]
    if not es.empty:
        labels=es["DrawdownBin"].astype(str).tolist();x=np.arange(len(labels));ax.bar(x,es["EpisodePct"]);p,l=sparse(labels);ax.set_xticks(p,l,rotation=35,ha="right");ax.set(title="Peak-Based Drawdown Distribution",ylabel="Episodes (%)");style(ax)
    else: ax.set_axis_off()
    ax=axes[0,1]
    if not ms.empty:
        labels=ms["MAEBin"].astype(str).tolist();x=np.arange(len(labels));ax.bar(x,ms["EntryPct"]);p,l=sparse(labels);ax.set_xticks(p,l,rotation=35,ha="right");ax.set(title=f"Entry MAE ({s.entry_frequency.title()})",ylabel="Entries (%)");style(ax)
    else: ax.set_axis_off()
    ax=axes[1,0]
    if not selection.empty:
        ax.plot(selection["StopPct"],selection["SurvivalRatePct"],marker="o",lw=2,label="Overall")
        if "WorstRegimeSurvivalPct" in selection: ax.plot(selection["StopPct"],selection["WorstRegimeSurvivalPct"],marker="s",lw=1.5,label="Worst regime")
        ax.axhline(90,ls="--",lw=1);ax.axhline(95,ls=":",lw=1);ax.set(title="Stop Survival and Worst-Regime Survival",xlabel="Stop width (%)",ylabel="Survival (%)",ylim=(0,102));ax.legend(fontsize=8);style(ax)
    else: ax.set_axis_off()
    ax=axes[1,1]
    if not regimes.empty:
        labels=[textwrap.fill(x,20) for x in (regimes["EntryTrendRegime"].astype(str)+" / "+regimes["EntryVolatilityRegime"].astype(str))];y=np.arange(len(regimes));h=.34
        ax.barh(y-h/2,regimes["P90MAEPct"],height=h,label="P90");ax.barh(y+h/2,regimes["P95MAEPct"],height=h,label="P95");ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(title="MAE by Entry Regime",xlabel="MAE (%)");ax.legend(fontsize=8);ax.grid(axis="x",alpha=.25)
    else: ax.set_axis_off()
    if SAVE_CHART: fig.savefig(path,dpi=CHART_DPI,facecolor="white")
    if SHOW_CHART: plt.show()
    else: plt.close(fig)


def autosize_excel(writer, sheet_name, frame):
    ws=writer.sheets[sheet_name];ws.freeze_panes="A2";ws.auto_filter.ref=ws.dimensions
    for i,c in enumerate(frame.columns,1):
        lengths=frame[c].map(lambda v:0 if pd.isna(v) else len(str(v)));m=int(lengths.max()) if not lengths.empty else 0
        ws.column_dimensions[ws.cell(1,i).column_letter].width=min(45,max(12,len(str(c))+2,m+2))


def export_excel(path, tables):
    with pd.ExcelWriter(path,engine="openpyxl",datetime_format="yyyy-mm-dd") as writer:
        for name,frame in tables.items():
            n=name[:31];frame.to_excel(writer,sheet_name=n,index=False);autosize_excel(writer,n,frame)


def main():
    s=Settings();validate_settings(s)
    out=Path(__file__).resolve().parent/OUTPUT_FOLDER;out.mkdir(parents=True,exist_ok=True)
    print(f"Downloading {s.symbol} from Yahoo Finance...")
    data=add_indicators(download_data(s),s)
    episodes=identify_drawdown_episodes(data,s)
    mae=analyze_entry_mae(data,s) if s.analyze_entry_mae else pd.DataFrame()
    comparable=eligible_mae(mae,s)
    events=stop_event_table(data,mae)
    selection=build_stop_selection(events,s)
    es=summarize_episodes(episodes,s);ms=summarize_mae(comparable,s);regimes=regime_table(comparable)
    if not episodes.empty: episodes["DrawdownBin"]=bin_values(episodes["DrawdownPct"],s)
    if not mae.empty: mae["MAEBin"]=bin_values(mae["MAEPct"],s)
    tables={
        "Overview":create_overview(data,episodes,mae,comparable,s),
        "Stop Selection":selection,
        "Stop Events":events,
        "Drawdown Episodes":episodes,
        "Drawdown Distribution":es,
        "Entry MAE":mae,
        "MAE Distribution":ms,
        "ATR Stop Survival":atr_stop_table(comparable),
        "Percentiles":percentile_table(episodes,comparable),
        "Regime Analysis":regimes,
    }
    safe=s.symbol.replace("^","");xlsx=out/f"{OUTPUT_BASENAME}_{safe}.xlsx";png=out/f"{OUTPUT_BASENAME}_{safe}.png"
    if SAVE_EXCEL:export_excel(xlsx,tables)
    plot_dashboard(es,ms,selection,regimes,png,s)
    print("\nAnalysis complete.")
    if SAVE_EXCEL:print(f"Workbook: {xlsx.resolve()}")
    if SAVE_CHART:print(f"Chart:    {png.resolve()}")


if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=FutureWarning)
        main()
