"""
Decision charts from the result CSVs of drawdown-dynamics.py

    python plot_results.py                       # reads ./results
    python plot_results.py --results-dir DIR --ticker QQQ

Writes PNG charts and <ticker>_decision_table.csv to <results-dir>/charts.
A chart is skipped (with a message) if its input file is missing.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dd", HERE / "drawdown-dynamics.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
# Diverging: red = worse than Hold, gray = same, blue = better.
DIVERGING = LinearSegmentedColormap.from_list("rg", [RED, NEUTRAL, BLUE])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#eef5fd", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def stop_label(stop):
    return "Hold" if stop == 0 else f"{100 * stop:g}%"


def finish(fig, path, note):
    """Add a how-to-read footnote below the plot, then save."""
    fig.tight_layout()
    fig.text(0.01, -0.01, note, ha="left", va="top", fontsize=8.5,
             color=MUTED)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def heatmap(ax, values, rows, cols, cmap, norm=None, text=None,
            vmin=None, vmax=None):
    values = np.asarray(values, float)
    kwargs = {"norm": norm} if norm else {"vmin": vmin, "vmax": vmax}
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto",
                      **kwargs)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Surface-colored gaps between cells.
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    if text is not None:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not text[i][j]:
                    continue
                color = INK
                if np.isfinite(values[i, j]):
                    r, g, b, _ = image.cmap(image.norm(values[i, j]))
                    if 0.299 * r + 0.587 * g + 0.114 * b < 0.55:
                        color = "white"
                ax.text(j, i, text[i][j], ha="center", va="center",
                        fontsize=7.5, color=color)
    return image


def diverging_norm(values):
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(finite.max()) if finite.size else 0.0, 1e-6)
    return TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)


def seed_average(frame, keys):
    """Average numeric columns over seeds."""
    numeric = [c for c in frame.select_dtypes("number").columns
               if c not in keys + ["Seed"]]
    return frame.groupby(keys, sort=False)[numeric].mean().reset_index()


def grid_of(frame, index, column, value, order=None):
    table = frame.pivot_table(index=index, columns=column, values=value,
                              sort=False)
    if order is not None:
        table = table.reindex(order)
    return table[sorted(table.columns)]


def difference_heatmap(table, title, note, path, scale=100):
    values = table.to_numpy(float) * scale
    text = [["" if not np.isfinite(v) else f"{v:+.1f}" for v in row]
            for row in values]
    fig, ax = plt.subplots(
        figsize=(max(8, .8 * values.shape[1] + 3),
                 .42 * values.shape[0] + 1.8))
    heatmap(ax, values, table.index.tolist(),
            [stop_label(s) for s in table.columns], DIVERGING,
            norm=diverging_norm(values), text=text)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Stop")
    finish(fig, path, note)


# =============================================================================
# Decision table
# =============================================================================

def decision_table(compare, objective, min_es=None) -> pd.DataFrame:
    """
    One row per stop (plus Hold), summarising all scenarios.

    Seeds are averaged first. "Balanced" figures average within each method
    family (historical, stationary, circular, fhs) and then across families,
    so that four block lengths do not count four times.
    """
    column = dd.OBJECTIVES[objective][0]
    data = seed_average(compare, ["Scenario", "StopPct"])
    data["Family"] = data["Scenario"].str.split("_L").str[0]

    hold = data[data.StopPct == 0].set_index("Scenario")
    data["Gain"] = data[column] - data["Scenario"].map(hold[column])
    data["ESGain"] = data["ES"] - data["Scenario"].map(hold["ES"])
    data["Regret"] = (data.groupby("Scenario")[column].transform("max")
                      - data[column])

    def balanced(group, name):
        return group.groupby("Family")[name].mean().mean(skipna=False)

    records = []
    for stop, group in data.groupby("StopPct"):
        defined = group.dropna(subset=["Gain"])
        record = {
            "StopPct": stop,
            "Policy": stop_label(stop),
            "Scenarios": len(group),
            "ScoredScenarios": len(defined),
            "BalancedGain": balanced(group, "Gain"),
            "MedianGain": defined["Gain"].median(),
            "WorstGain": defined["Gain"].min(),
            "ShareBeatsHold": (defined["Gain"] > 1e-12).mean()
                if len(defined) else np.nan,
            "MaxRegret": group["Regret"].max(skipna=False),
            "BalancedESGain": balanced(group, "ESGain"),
            "ShareBetterES": (group["ESGain"] > 1e-12).mean(),
            "WorstES": group["ES"].min(),
            "BalancedTriggerRate": balanced(group, "TriggerRate"),
        }
        if min_es is not None:
            record["ShareMeetsMinES"] = (group["ES"] >= min_es).mean()
        records.append(record)
    return pd.DataFrame(records)


# =============================================================================
# Charts
# =============================================================================

def chart_scorecard(table, objective, path):
    """Chart 1: the robustness summary."""
    labels = table["Policy"].tolist()
    x = np.arange(len(table))
    unit = "ratio units" if objective == "return_over_es" else "% points"
    scale = 1 if objective == "return_over_es" else 100
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    panels = [
        ("BalancedGain", scale,
         f"Objective gain over Hold, family-balanced ({unit})"),
        ("WorstGain", scale,
         f"Objective gain over Hold, WORST scenario ({unit})"),
        ("ShareBeatsHold", 100, "Share of scenarios beating Hold (%)"),
        ("BalancedESGain", 100,
         "Expected shortfall gain over Hold (% points)"),
    ]
    for ax, (name, factor, title) in zip(axes, panels):
        values = table[name].to_numpy(float) * factor
        share = name == "ShareBeatsHold"
        colors = BLUE if share else [BLUE if v >= 0 else RED
                                     for v in np.nan_to_num(values)]
        ax.bar(x, np.nan_to_num(values), width=.6, color=colors)
        ax.axhline(0, color=MUTED, linewidth=.8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if share:
            ax.set_ylim(0, 118)
        else:
            finite = np.r_[values[np.isfinite(values)], 0.0]
            span = max(np.ptp(finite), 0.1)
            ax.set_ylim(finite.min() - .18 * span, finite.max() + .22 * span)
        for i, v in enumerate(values):
            if np.isfinite(v):
                ax.annotate(f"{v:.0f}" if share else
                            "0" if v == 0 else
                            f"{v:+.2f}" if abs(v) < .1 else f"{v:+.1f}", (i, v),
                            xytext=(0, 3 if v >= 0 else -3),
                            textcoords="offset points", ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=7.5, color=MUTED)
            else:
                ax.annotate("n/a", (i, 0), ha="center", va="bottom",
                            fontsize=7.5, color=MUTED)
    axes[-1].set_xticks(x, labels)
    finish(fig, path,
           "Above zero is better than Hold (blue); below is worse (red). "
           "In panel 3, Hold is 0 by definition.\n"
           "Look for several NEIGHBORING stops that do well. These are "
           "in-sample summaries, not proof.")


def chart_tradeoff(compare, path):
    """Chart 3: growth against tail risk, one panel per scenario."""
    data = seed_average(compare, ["Scenario", "StopPct"])
    names = data["Scenario"].unique()
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows),
                             squeeze=False, sharex=True, sharey=True)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.flat, names):
        group = data[data.Scenario == name].sort_values("StopPct")
        x, y = group.ES * 100, group.MeanLogGrowth * 100
        stops = group.StopPct.to_numpy()
        is_hold = stops == 0
        ax.plot(x[~is_hold], y[~is_hold], color=BLUE, linewidth=1.5,
                marker="o", markersize=5, label="Stops")
        ax.plot(x[is_hold], y[is_hold], color=INK, marker="D",
                markersize=8, linestyle="none", label="Hold")
        # Label a few points only, to avoid collisions. The widest stop is
        # skipped because it usually sits on top of Hold.
        positive = stops[~is_hold]
        shown = {0.0}
        if len(positive):
            shown |= {positive[0], positive[len(positive) // 3],
                      positive[len(positive) // 2]}
        for xi, yi, stop in zip(x, y, stops):
            if stop in shown:
                below = stop == 0
                ax.annotate(stop_label(stop), (xi, yi),
                            xytext=(2, -14) if below else (4, 6),
                            textcoords="offset points", fontsize=7.5,
                            color=MUTED)
        ax.set_title(name, loc="left")
        ax.grid(True)
        ax.margins(.12)
        ax.tick_params(labelbottom=True, labelleft=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean log growth (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Expected shortfall (%)  right = safer")
    axes[0, 0].legend(frameon=False, loc="best")
    finish(fig, path,
           "Up = more growth; right = less severe tail. A point both below "
           "and left of another is dominated on these two measures.")


def chart_historical(compare, path):
    """Chart 4: historical difference from Hold, with its interval."""
    group = compare[compare.Scenario == "historical"]
    group = group[group.StopPct > 0].sort_values("StopPct")
    if group.empty:
        print("Skipped historical interval chart (no historical scenario).")
        return
    x = np.arange(len(group))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    has = group.DiffCILow.notna().to_numpy()
    ax.vlines(x[has], group.DiffCILow[has] * 100, group.DiffCIHigh[has] * 100,
              color=BLUE, linewidth=3, alpha=.35,
              label="Approx. 95% interval")
    ax.plot(x, group.DiffMeanLogVsHold * 100, "o", color=BLUE, markersize=8,
            label="Estimate")
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x, [stop_label(s) for s in group.StopPct])
    ax.set_xlabel("Stop")
    ax.set_ylabel("Mean log growth minus Hold (% points)")
    ax.set_title("Historical: stop versus Hold, with sampling uncertainty",
                 loc="left")
    ax.grid(axis="y")
    ax.legend(frameon=False)
    finish(fig, path,
           "An interval that includes zero is inconclusive. Intervals are "
           "rough, likely too narrow with little history,\n"
           "and not adjusted for testing many stops. "
           + ("" if has.all() else "Missing intervals: history too short."))


def chart_periods(periods, path):
    """Chart 5: stability through time."""
    data = periods[(periods.Scenario == "historical") & (periods.StopPct > 0)]
    data = data.assign(Period=data.FirstEntry.astype(str) + " to "
                       + data.LastEntry.astype(str))
    table = grid_of(data, "Period", "StopPct", "DiffMeanLogVsHold")
    difference_heatmap(
        table.sort_index(),
        "Historical periods: mean log growth minus Hold (% points)",
        "Blue = stop did better. A good candidate should not depend on one "
        "period. Adjacent periods share some price data.", path)


def chart_mechanics(compare, path):
    """Chart 6: how often stops trigger and whipsaw."""
    data = seed_average(compare[compare.StopPct > 0],
                        ["Scenario", "StopPct"])
    names = data.Scenario.unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, column, title in zip(
            axes, ["TriggerRate", "WhipsawRate"],
            ["Trigger rate: share of trades stopped",
             "Whipsaw rate: share of stopped trades where Hold ended higher"]):
        table = grid_of(data, "Scenario", "StopPct", column, names) * 100
        stops = table.columns.to_numpy() * 100
        # Range across scenarios, plus the median and the historical line.
        ax.fill_between(stops, table.min(), table.max(), color=BLUE,
                        alpha=.18, linewidth=0, label="Range of scenarios")
        ax.plot(stops, table.median(), color=BLUE, linewidth=2,
                label="Median scenario")
        if "historical" in table.index:
            ax.plot(stops, table.loc["historical"], color=INK, linewidth=2,
                    linestyle="--", label="Historical")
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("Stop (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False)
    finish(fig, path,
           "These explain WHY results differ. They do not by themselves say "
           "which stop is best. Deep stops may rest on few triggers.")


def chart_walk_forward(detail, summary, path):
    """Chart 7: chosen stop by year, and out-of-sample gain."""
    names = detail.Scenario.unique().tolist()
    chosen = detail.pivot(index="Scenario", columns="TestYear",
                          values="ChosenStopPct").reindex(names)
    met = detail.pivot(index="Scenario", columns="TestYear",
                       values="ConstraintMet").reindex(names)
    values = chosen.to_numpy(float) * 100
    text = [["" if not np.isfinite(v) else
             ("H*" if not met.iloc[i, j] else "H" if v == 0 else f"{v:g}")
             for j, v in enumerate(row)] for i, row in enumerate(values)]
    shaded = np.where(values == 0, np.nan, values)   # Hold stays neutral

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, .5 * values.shape[1] + 3),
                       .75 * len(names) + 3),
        )
    cmap = SEQUENTIAL.copy()
    cmap.set_bad(NEUTRAL)
    heatmap(axes[0], shaded, names, chosen.columns.tolist(), cmap, text=text,
            vmin=0, vmax=max(50, np.nanmax(values)))
    axes[0].set_title("Stop chosen using only earlier data (%, H = Hold)",
                      loc="left")

    wide = summary.pivot(index="Scenario", columns="Policy").reindex(names)
    gains = pd.DataFrame({
        "Mean log growth": wide["MeanLogGrowth"]["WalkForwardStop"]
        - wide["MeanLogGrowth"]["Hold"],
        "Expected shortfall": wide["ES"]["WalkForwardStop"]
        - wide["ES"]["Hold"]}) * 100
    y = np.arange(len(names))
    axes[1].barh(y - .18, gains.iloc[:, 0], height=.34, color=BLUE,
                 label=gains.columns[0])
    axes[1].barh(y + .18, gains.iloc[:, 1], height=.34, color="#eb6834",
                 label=gains.columns[1])
    for offset, column in [(-.18, 0), (.18, 1)]:
        for i, v in enumerate(gains.iloc[:, column]):
            axes[1].annotate(f"{v:+.1f}", (v, i + offset),
                             xytext=(3 if v >= 0 else -3, 0),
                             textcoords="offset points", va="center",
                             ha="left" if v >= 0 else "right",
                             fontsize=7, color=MUTED)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=INK, linewidth=1)
    axes[1].margins(x=.15)
    axes[1].grid(axis="x")
    axes[1].set_xlabel("Walk-forward minus Hold (% points). Right = better")
    axes[1].set_title("Out-of-sample result, pooled over test trades",
                      loc="left")
    axes[1].legend(frameon=False, loc="upper left",
                   bbox_to_anchor=(1.01, 1))
    finish(fig, path,
           "H* = no row met --min-es, so Hold was used. This tests the "
           "SELECTION RULE, not one fixed stop. Trades overlap.")


def chart_recovery(recovery, path, w1="12M"):
    """Chart 8 (context): recovery probability after a stop level is hit."""
    names = recovery.Scenario.unique()
    scenario = "historical" if "historical" in names else names[0]
    data = recovery[(recovery.Scenario == scenario) & (recovery.W1 == w1)]
    order = [w for w in dd.WINDOWS if w in set(data.W2)]
    prob = data.pivot(index="StopPct", columns="W2",
                      values="RecoveryProbability")[order]
    hits = data.pivot(index="StopPct", columns="W2", values="StopHits")[order]
    values = prob.to_numpy(float) * 100
    text = [[f"{v:.0f}% (n={int(hits.iloc[i, j])})" if np.isfinite(v)
             else "no hits" for j, v in enumerate(row)]
            for i, row in enumerate(values)]
    fig, ax = plt.subplots(figsize=(10, .45 * len(prob) + 1.8))
    heatmap(ax, values, [stop_label(s) for s in prob.index], order,
            SEQUENTIAL, text=text, vmin=0, vmax=100)
    ax.set_title(f"{scenario}: probability of recovering to entry after the "
                 f"drop (W1 = {w1})", loc="left")
    ax.set_xlabel("W2: time allowed after the stop level is first hit")
    ax.set_ylabel("Drop")
    finish(fig, path,
           "Context only: a high recovery rate does NOT show that a stop "
           "loses money, or a low one that it helps.\n"
           "n = hits (overlapping, not independent). Columns can use "
           "different entry periods with per-window eligibility.")


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot stop-loss results.")
    parser.add_argument("--results-dir", default=str(HERE / "results"))
    parser.add_argument("--ticker", default=dd.TICKER)
    parser.add_argument("--objective", choices=list(dd.OBJECTIVES),
                        help="default: the objective used in the run")
    args = parser.parse_args(argv)

    results = Path(args.results_dir)
    prefix = args.ticker.lower()
    charts = results / "charts"

    config_path = results / f"{prefix}_run_config.json"
    config = json.loads(config_path.read_text()) \
        if config_path.exists() else {}
    objective = args.objective or config.get("objective", "mean_log")
    analyses = set(config.get("analysis", ["recovery", "compare",
                                           "walkforward"]))

    def read(name, analysis):
        path = results / f"{prefix}_{name}.csv"
        # Ignore files left over from an earlier run in the same folder.
        if analysis not in analyses or not path.exists():
            print(f"No {name} from the latest run; related charts skipped.")
            return None
        return pd.read_csv(path)

    compare = read("stop_vs_hold", "compare")
    periods = read("by_period", "compare")
    if config and not ("historical" in config.get("method", [])
                       and config.get("periods", 0) > 1):
        periods = None
    detail = read("walk_forward", "walkforward")
    summary = read("walk_forward_summary", "walkforward")
    recovery = read("recovery_summary", "recovery")
    if all(f is None for f in (compare, detail, recovery)):
        raise SystemExit(f"No result files for '{prefix}' in {results}")

    charts.mkdir(parents=True, exist_ok=True)
    # Remove this ticker's old charts so stale ones are not mistaken as new.
    for old in charts.glob(f"{prefix}_[1-8]_*.png"):
        old.unlink()
    (charts / f"{prefix}_decision_table.csv").unlink(missing_ok=True)

    def out(name):
        return charts / f"{prefix}_{name}.png"

    if compare is not None:
        table = decision_table(compare, objective, config.get("min_es"))
        table.to_csv(charts / f"{prefix}_decision_table.csv", index=False,
                     float_format="%.6f")
        chart_scorecard(table, objective, out("1_scorecard"))

        data = seed_average(compare[compare.StopPct > 0],
                            ["Scenario", "StopPct"])
        names = data.Scenario.unique().tolist()
        column, description = dd.OBJECTIVES[objective]
        hold = seed_average(compare[compare.StopPct == 0],
                            ["Scenario", "StopPct"]).set_index("Scenario")
        data["Gain"] = data[column] - data.Scenario.map(hold[column])
        ratio = objective == "return_over_es"
        difference_heatmap(
            grid_of(data, "Scenario", "StopPct", "Gain", names),
            f"{description.capitalize()}: stop minus Hold "
            f"({'ratio units' if ratio else '% points'})",
            "Blue = better than Hold, red = worse, blank = undefined. Look "
            "for a blue vertical BAND across rows and neighboring stops.",
            out("2_robustness_heatmap"), scale=1 if ratio else 100)
        chart_tradeoff(compare, out("3_growth_vs_tail_risk"))
        chart_historical(compare, out("4_historical_uncertainty"))
        chart_mechanics(compare, out("6_trigger_whipsaw"))
    if periods is not None:
        chart_periods(periods, out("5_period_stability"))
    if detail is not None and summary is not None:
        chart_walk_forward(detail, summary, out("7_walk_forward"))
    if recovery is not None:
        chart_recovery(recovery, out("8_recovery_context"))


if __name__ == "__main__":
    main()
