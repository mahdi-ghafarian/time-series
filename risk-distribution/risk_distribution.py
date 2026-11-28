"""
Holding-period RISK analysis over the whole price history -> one Excel workbook.

Install:  pip install -U yfinance pandas openpyxl
Run:      python risk_distribution.py                       (settings below)
          python risk_distribution.py --ticker ^GSPC
          python risk_distribution.py --csv prices.csv      (Date + 'Adj Close' or 'Close')
"""
import argparse
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.formula import ArrayFormula

from stop_lab import add_stop_lab, entry_allowed, filter_text

# ---------------- SETTINGS (edit these) ----------------
TICKER = "BTC-USD"

STOP_MONTHS = 12                 # holding period used for the stop analysis
STOP_LEVELS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
HIST_BINS = 30                   # approximate number of histogram bars
HOLDING_MONTHS = [1, 3, 6, 12, 24, 36, 60, 120]   # how long each investment is held, in months

MA_FILTER = "all"                # test entries: "all" (any day), "above" the MA, or "below" (at/below)
MA_PERIOD = 200                  # simple moving average length in TRADING days (used by above/below)

RUN_STOP_LAB = False             # switch for the stop testing lab: True = run, False = skip
MAX_CVAR = 0.50                  # your loss limit for the stop scorecard. CHANGE to suit you
BOOTSTRAP_RESAMPLES = 1000       # per block length; 0 switches the bootstrap off


# -------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent / "results"   # next to this script

HEAD_FILL = PatternFill("solid", fgColor="1F3864")
PCT = '0.0%;-0.0%;0.0%'


def load_prices(ticker, csv=None):
    if csv:
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                      index=pd.to_datetime(df["Date"]))
        source = f"CSV file: {csv} (column '{col}')"
    else:
        import yfinance as yf
        df = yf.download(ticker, period="max", interval="1d", auto_adjust=True,
                         multi_level_index=False, progress=False)
        if df is None or df.empty:
            raise SystemExit(f"No data returned for {ticker}. Check the symbol and connection, "
                             "run 'pip install -U yfinance', or wait if rate-limited.")
        s = df["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        # Drop today's bar, which may be an unfinished trading session
        s = s[s.index.normalize() < pd.Timestamp.today().normalize()]
        source = f"Yahoo Finance via yfinance {yf.__version__}, adjusted close (auto_adjust=True)"
    s = s.sort_index().dropna()
    s = s[~s.index.duplicated(keep="last")]
    s = s[s > 0]
    if len(s) < 300:
        raise SystemExit("Need at least ~300 daily prices.")
    return s, source


def tag(m):
    return f"{m // 12}Y" if m % 12 == 0 else f"{m}M"


def start_positions(dates, months):
    """For each end date, row of the last trading day on/before (end date - months); -1 if none."""
    targets = dates - pd.DateOffset(months=months)
    return dates.searchsorted(targets, side="right") - 1


def header(ws, row, values, col=1):
    for j, v in enumerate(values):
        c = ws.cell(row, col + j, v)
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def put(ws, row, col, value, fmt=None, bold=False, color="000000"):
    c = ws.cell(row, col, value)
    c.font = Font(name="Arial", bold=bold, color=color)
    if fmt:
        c.number_format = fmt
    return c


def widths(ws, values):
    for i, w in enumerate(values, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def style_chart(ch, title, ytitle, xtitle):
    ch.title, ch.width, ch.height = title, 26, 9.5
    ch.y_axis.title, ch.x_axis.title = ytitle, xtitle
    ch.roundedCorners = False
    for t in (ch.title, ch.y_axis.title, ch.x_axis.title):
        t.overlay = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.y_axis.number_format = '0%'
    ch.legend = None


def path_drawdowns(p, starts):
    """Largest peak-to-trough fall inside each window (negative number), NaN if no window."""
    out = np.full(len(p), np.nan)
    for i, s in enumerate(starts):
        if s >= 0:
            w = p[s:i + 1]
            out[i] = (w / np.maximum.accumulate(w) - 1).min()
    return out


def worst_episodes(returns, starts, k=5):
    """End rows of the k worst windows that do not overlap each other."""
    chosen = []
    for i in np.argsort(returns, kind="stable"):
        if np.isnan(returns[i]):
            continue
        if all(i <= starts[j] or starts[i] >= j for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


def stopped_returns(p, starts, level, trailing):
    """Exit at the FIRST close at/below the stop (fill = that close), else hold to the end."""
    ret = np.full(len(p), np.nan)
    hit = np.zeros(len(p), dtype=int)
    for i, s in enumerate(starts):
        if s < 0:
            continue
        w = p[s:i + 1]
        ref = np.maximum.accumulate(w) if trailing else w[0]
        breach = np.flatnonzero(w / ref - 1 <= -level + 1e-12)
        hit[i] = int(len(breach) > 0)
        ret[i] = (w[breach[0]] if len(breach) else w[-1]) / w[0] - 1
    return ret, hit


def count_separate(sp):
    """Greedy count of windows that do not overlap (a window may start where one ends)."""
    count, reach = 0, -1
    for end, start in enumerate(sp):
        if start >= 0 and start >= reach:
            count, reach = count + 1, end
    return count


def check_levels(values):
    """Sorted, de-duplicated stop levels (the lab compares neighbors, so order matters)."""
    levels = sorted(set(values))
    if not levels or not all(0 < v < 1 for v in levels):
        raise SystemExit("STOP_LEVELS must be decimals between 0 and 1 (0.10 = 10%).")
    return levels


DEPTH_SHEETS = {
    "mae": dict(
        sheet="MAE Analysis", position=1, column=2, name="MAE",
        title="Maximum Adverse Excursion (MAE)",
        definition="MAE = lowest daily CLOSE during the window / entry price - 1. "
                   "0% = never closed below entry.",
        never="Never closed below entry", row="Fell {:g}% or more",
        chance="Chance of falling this far below entry at some point",
        after="After falling this far, share of windows that still ENDED UP"),
    "drawdown": dict(
        sheet="Drawdown Analysis", position=2, column=3, name="max drawdown",
        title="Maximum drawdown",
        definition="Max drawdown = largest fall from the highest CLOSE so far in the window to a later "
                   "close. 0% = never closed below an earlier high.",
        never="Never closed below an earlier high", row="Drawdown of {:g}% or more",
        chance="Chance of a drawdown this large at some point",
        after="After a drawdown this large, share of windows that still ENDED UP"),
}


def add_depth_analysis(wb, holds, rng, levels, kind, subtitle="full history"):
    """One statistics sheet about how deep windows fell. kind: "mae" or "drawdown" (see DEPTH_SHEETS).
    rng(h, m) gives a Data range: m = 0 return, 2 lowest vs entry, 3 max drawdown."""
    k = DEPTH_SHEETS[kind]
    name = k["name"]
    ws = wb.create_sheet(k["sheet"], k["position"])
    last_col = len(holds) + 1
    put(ws, 1, 1, f"{k['title']}, {subtitle}").font = Font(
        name="Arial", bold=True, size=14)
    put(ws, 2, 1, k["definition"] + " No stops applied. Intraday lows are not included.")
    put(ws, 3, 1, "'75% / 90% / 95% / 99% of windows' = that share fell no further than the value shown. "
        "Windows overlap, so they are not independent." + (
            " Drawdowns in Data are script values: re-run after changing prices."
            if kind == "drawdown" else ""))
    header(ws, 4, ["Measure"] + [f"Hold {tag(h)}" for h in holds])
    row = 5

    def section(title):
        nonlocal row
        put(ws, row, 1, title, bold=True)
        for c in range(1, last_col + 1):
            ws.cell(row, c).fill = PatternFill("solid", fgColor="D9E1F2")
        row += 1

    def line(label, make, fmt=PCT, array=False):
        nonlocal row
        put(ws, row, 1, label)
        for c, h in enumerate(holds, 2):
            formula = make(rng(h, k['column']), rng(h, 0))
            if array:      # needed for IF() over a whole range
                formula = ArrayFormula(f"{get_column_letter(c)}{row}", formula)
            put(ws, row, c, formula, fmt).alignment = Alignment(horizontal="right")
        row += 1

    depths = [(f"{q}% of windows fell no further than", (100 - q) / 100) for q in (75, 90, 95, 99)]

    section(f"{name} distribution: all windows")
    line("Windows", lambda L, R: f"=COUNT({L})", "#,##0")
    line(k["never"], lambda L, R: f'=COUNTIF({L},">=-0.000000000001")/COUNT({L})')
    line(f"Mean {name}", lambda L, R: f"=AVERAGE({L})")
    line(f"Median {name}", lambda L, R: f"=MEDIAN({L})")
    line(f"Std dev of {name}", lambda L, R: f"=STDEV({L})")
    for label, q in depths:
        line(label, lambda L, R, q=q: f"=PERCENTILE({L},{q})")
    line(f"Worst {name}", lambda L, R: f"=MIN({L})")

    section(k["chance"])
    for lv in levels:
        line(k["row"].format(lv * 100), lambda L, R, lv=lv: f'=COUNTIF({L},"<="&{-lv})/COUNT({L})')

    # ISNUMBER keeps blank rows (dropped by the MA filter) out of both groups
    for title, op in ((f"{name} of windows that ENDED UP", ">0"),
                      (f"{name} of windows that ENDED FLAT OR DOWN", "<=0")):
        section(title)

        def stat(func, extra="", op=op):
            return lambda L, R: (f'=IFERROR({func}(IF(ISNUMBER({R}),IF({R}{op},{L}))'
                                 f'{extra}),"-")')

        count_row = row
        line("Windows", lambda L, R, op=op: f'=COUNTIF({R},"{op}")', "#,##0")
        line(f"Mean {name}", stat("AVERAGE"), array=True)
        line(f"Median {name}", stat("MEDIAN"), array=True)
        for label, q in depths:
            line(label, stat("PERCENTILE", f",{q}"), array=True)
        line(f"Worst {name}", stat("MIN"), array=True)
        # MIN of no values returns 0, so show a dash when the group is empty
        for c in range(2, last_col + 1):
            x = ws.cell(row - 1, c)
            L_c = get_column_letter(c)
            x.value = ArrayFormula(x.coordinate, f'=IF({L_c}{count_row}=0,"-",{x.value.text[1:]})')

    section(k["after"])
    for lv in levels:
        line(k["row"].format(lv * 100), lambda L, R, lv=lv:
             f'=IFERROR(COUNTIFS({L},"<="&{-lv},{R},">0")/COUNTIF({L},"<="&{-lv}),"-")')
    section("Number of windows behind each share above")
    for lv in levels:
        line(k["row"].format(lv * 100), lambda L, R, lv=lv: f'=COUNTIF({L},"<="&{-lv})', "#,##0")
    widths(ws, [46] + [14] * len(holds))
    ws.freeze_panes = "B5"


def add_stop_analysis(wb, dates, p, sp, ret_letter, low_letter, levels, subtitle="full history"):
    name = tag(STOP_MONTHS)
    ws = wb.create_sheet("Stop Analysis", 1)
    sd = wb.create_sheet("Stop Data", wb.sheetnames.index("Data") + 1)
    valid = np.flatnonzero(sp >= 0)
    first, last = 2, len(valid) + 1
    half = len(valid) // 2

    combos = [(t, lv) for t in (False, True) for lv in levels]
    header(sd, 1, ["End date", "No stop"] + [
        f"{'Trail' if t else 'Fixed'} {lv:.1%} {x}" for t, lv in combos for x in ("return", "hit")])
    results = [stopped_returns(p, sp, lv, t) for t, lv in combos]
    for k, i in enumerate(valid):
        r = k + 2
        put(sd, r, 1, f"=Data!A{i + 2}", "yyyy-mm-dd")
        put(sd, r, 2, f"=Data!{ret_letter}{i + 2}", PCT)
        for j, (ret, hit) in enumerate(results):
            put(sd, r, 3 + 2 * j, float(ret[i]), PCT, color="0000FF")
            put(sd, r, 4 + 2 * j, int(hit[i]), "0", color="0000FF")
    sd.freeze_panes = "C2"
    widths(sd, [13, 11] + [17] * (2 * len(combos)))

    def col(c, a=first, b=last):
        L = get_column_letter(c)
        return f"'Stop Data'!${L}${a}:${L}${b}"

    def tail(R):
        return f'-AVERAGEIF({R},"<="&(SMALL({R},ROUNDUP(COUNT({R})*0.05,0))+1E-12),{R})'

    put(ws, 1, 1, f"Stop-loss comparison: {name} holding window, {subtitle}").font = Font(
        name="Arial", bold=True, size=14)
    put(ws, 2, 1, "Trigger and fill at the daily CLOSE. After a stop: stay in cash at 0% until "
        "the window ends. No re-entry, costs or slippage.")
    labels = ["Stop distance", "% windows stopped", "% of stopped that would have ended up",
              "Mean return", "Median return", "% ending with a loss", "5th percentile",
              "95% CVaR (loss; negative = gain)", "Worst", "Mean change vs no stop",
              "% of stopped where stop helped", "Mean / CVaR",
              "Mean / CVaR: 1st half", "Mean / CVaR: 2nd half"]
    fmts = [PCT] * 11 + ["0.00"] * 3
    row = 4
    for trailing, title in ((False, "FIXED stop: distance below entry price"),
                            (True, "TRAILING stop: distance below highest close since entry")):
        put(ws, row, 1, title, bold=True)
        header(ws, row + 1, labels)
        ws.row_dimensions[row + 1].height = 45
        base_row = row + 2
        entries = [None] + [j for j, (t, _) in enumerate(combos) if t == trailing]
        for k, j in enumerate(entries):
            r = base_row + k
            c = 2 if j is None else 3 + 2 * j
            R, H, B = col(c), None if j is None else col(c + 1), col(2)
            put(ws, r, 1, "No stop" if j is None else combos[j][1],
                None if j is None else "0.0%")
            f = {4: f"=AVERAGE({R})", 5: f"=MEDIAN({R})", 6: f'=COUNTIF({R},"<0")/COUNT({R})',
                 7: f"=PERCENTILE({R},0.05)", 8: f"={tail(R)}", 9: f"=MIN({R})",
                 10: f"=D{r}-$D${base_row}"}
            if j is None:
                for q in (2, 3, 11):
                    put(ws, r, q, "-").alignment = Alignment(horizontal="right")
            else:
                f[2] = f"=AVERAGE({H})"
                f[3] = f'=IF(SUM({H})=0,"-",COUNTIFS({H},1,{B},">0")/SUM({H}))'
                f[11] = f'=IF(SUM({H})=0,"-",SUMPRODUCT({H},--({R}>{B}+1E-9))/SUM({H}))'
            f[12] = f'=IF(H{r}>0,D{r}/H{r},"n/a")'
            for q, (a, b) in ((13, (first, first + half - 1)), (14, (first + half, last))):
                x = col(c, a, b)
                f[q] = f'=IF({tail(x)}>0,AVERAGE({x})/({tail(x)}),"n/a")'
            for q, formula in f.items():
                put(ws, r, q, formula, fmts[q - 1])
        row = base_row + len(entries) + 2

    # How far did eventual winners and losers fall below entry?
    put(ws, row, 1, "How far below entry did windows go? Split by how they ENDED without a stop",
        bold=True)
    header(ws, row + 1, ["Lowest point vs entry", "Ended up", "Ended flat or down"])
    low = f"Data!${low_letter}${valid[0] + 2}:${low_letter}${len(p) + 1}"
    end = f"Data!${ret_letter}${valid[0] + 2}:${ret_letter}${len(p) + 1}"
    r = row + 2
    for lv in levels:
        put(ws, r, 1, f"Share that fell {lv:.1%} or more")
        for c, cond in ((2, '">0"'), (3, '"<=0"')):
            put(ws, r, c, f'=IFERROR(COUNTIFS({low},"<="&{-lv},{end},{cond})/COUNTIF({end},{cond}),0)', PCT)
        r += 1
    put(ws, r + 1, 1, f"1st half = windows ending {dates[valid[0]]:%Y-%m-%d} to "
        f"{dates[valid[half - 1]]:%Y-%m-%d}; 2nd half = the rest. Halves share some price history.")
    put(ws, r + 2, 1, "Stopped returns are values calculated by the script (blue in Stop Data). "
        "Re-run after changing prices or stop levels.")
    widths(ws, [34] + [15] * 13)
    ws.freeze_panes = "B4"


def build(prices, name, source, out, max_cvar=MAX_CVAR, reps=BOOTSTRAP_RESAMPLES,
          run_lab=RUN_STOP_LAB, ma_filter=MA_FILTER, ma_period=MA_PERIOD):
    dates, p = prices.index, prices.values
    levels = check_levels(STOP_LEVELS)
    n = len(p)
    last = n + 1
    filtered = ma_filter != "all"
    if filtered and ma_period >= n:
        raise SystemExit(f"MA period {ma_period} is too long for {n} prices.")
    allowed = entry_allowed(p, ma_filter, ma_period)
    side = "above" if ma_filter == "above" else "at or below"
    subtitle = f"entries {side} {ma_period}-day MA" if filtered else "full history"
    holds, starts = [], {}
    for h in HOLDING_MONTHS:
        sp = start_positions(dates, h)
        if filtered:     # drop windows whose ENTRY day does not meet the condition
            sp = np.where((sp >= 0) & allowed[np.maximum(sp, 0)], sp, -1)
        if (sp >= 0).sum() >= 60:
            holds.append(h)
            starts[h] = sp
        else:
            print(f"Skipping {tag(h)} holding period: fewer than 60 windows"
                  + (" after the MA filter." if filtered else "."))
    if not holds:
        raise SystemExit("History is too short for every holding period.")

    wb = Workbook()
    summ = wb.active
    summ.title = "Risk Summary"
    hist = wb.create_sheet("Distribution")
    worst = wb.create_sheet("Worst Periods")
    data = wb.create_sheet("Data")
    notes = wb.create_sheet("Notes")

    # ---- Data ----
    blocks = ["return", "annualized", "lowest vs entry", "max drawdown"]
    header(data, 1, ["Date", "Adj. price"] + [f"{tag(h)} {b}" for h in holds for b in blocks])
    cols = {h: [get_column_letter(3 + 4 * k + m) for m in range(4)] for k, h in enumerate(holds)}
    rets, dd = {}, {}
    for h in holds:
        sp = starts[h]
        rets[h] = np.where(sp >= 0, p / p[np.maximum(sp, 0)] - 1, np.nan)
        dd[h] = path_drawdowns(p, sp)
    for i, d in enumerate(dates):
        r = i + 2
        put(data, r, 1, d.to_pydatetime(), "yyyy-mm-dd")
        put(data, r, 2, float(p[i]), "#,##0.00", color="0000FF")
        for k, h in enumerate(holds):
            s = starts[h][i]
            if s < 0:
                continue
            c = 3 + 4 * k
            ret = cols[h][0]
            put(data, r, c, f"=B{r}/B{s + 2}-1", PCT)
            if h >= 12:   # annualizing sub-1-year returns is misleading, so leave blank
                put(data, r, c + 1, f"=(1+{ret}{r})^(12/{h})-1", PCT)
            put(data, r, c + 2, f"=MIN(B{s + 2}:B{r})/B{s + 2}-1", PCT)
            # Static value: a path drawdown needs a running peak, impractical as a cell formula
            put(data, r, c + 3, float(dd[h][i]), PCT, color="0000FF")
    data.freeze_panes = "B2"
    widths(data, [13, 13] + [17] * (4 * len(holds)))
    if filtered:   # extra columns at the far right so the existing column positions do not move
        c = 3 + 4 * len(holds)
        header(data, 1, [f"{ma_period}-day SMA", "Qualifies as entry day?"], col=c)
        for i in range(ma_period - 1, n):
            r = i + 2
            put(data, r, c, f"=AVERAGE(B{r - ma_period + 1}:B{r})", "#,##0.00")
            put(data, r, c + 1, "Yes" if allowed[i] else "No", color="0000FF")
        for k in (c, c + 1):
            data.column_dimensions[get_column_letter(k)].width = 24

    def rng(h, m):
        first = int(np.argmax(starts[h] >= 0)) + 2
        return f"Data!${cols[h][m]}${first}:${cols[h][m]}${last}"

    # ---- Risk Summary: one column per holding period, full history ----
    def cvar(q):
        return ('=MAX(0,-AVERAGEIF({R},"<="&(SMALL({R},MAX(1,ROUNDUP(COUNT({R})*%s,0)))+1E-12),{R}))' % q)

    def prob(key, x):
        return '=COUNTIF({%s},"%s")/COUNT({%s})' % (key, x, key)

    sections = [
        ("Sample", [
            ("Windows (overlapping)", "=COUNT({R})", "#,##0"),
            ("Approx. non-overlapping periods", "=({LASTDATE}-{FIRSTDATE})/365.25*12/{H}", "0.0")]),
        ("Distribution of holding-period return (cumulative)", [
            ("Mean", "=AVERAGE({R})", PCT), ("Median", "=MEDIAN({R})", PCT),
            ("Std dev", "=STDEV({R})", PCT), ("Skew", "=SKEW({R})", "0.00"),
            ("Excess kurtosis", "=KURT({R})", "0.00"), ("Worst", "=MIN({R})", PCT)]
            + [(f"{q}th percentile" if q != 1 else "1st percentile",
                "=PERCENTILE({R},%s)" % (q / 100), PCT) for q in (1, 5, 10, 25, 75, 90, 95, 99)]
            + [("Best", "=MAX({R})", PCT)]),
        ("Chance of ENDING with a loss", [("Any loss", prob("R", "<0"), PCT)]
            + [(f"Loss worse than {x}%", prob("R", f"<-{x / 100}"), PCT) for x in (10, 20, 30, 40)]),
        ("Tail risk (loss shown as a positive number; 0% = no loss at that level)", [
            ("95% VaR", "=MAX(0,-PERCENTILE({R},0.05))", PCT),
            ("95% CVaR (average of worst 5%)", cvar(0.05), PCT),
            ("99% VaR", "=MAX(0,-PERCENTILE({R},0.01))", PCT),
            ("99% CVaR (average of worst 1%)", cvar(0.01), PCT)]),
        ("Annualized returns", [
            ("Median", "=MEDIAN({A})", PCT), ("5th percentile", "=PERCENTILE({A},0.05)", PCT),
            ("1st percentile", "=PERCENTILE({A},0.01)", PCT), ("Worst", "=MIN({A})", PCT)]),
        ("DURING the holding period: fall below entry price (daily closes)", [
            (f"Chance of being down {x}% or more at some point", prob("L", f"<=-{x / 100}"), PCT)
            for x in (10, 20, 30, 40)]
            + [("Median lowest point", "=MEDIAN({L})", PCT),
               ("5th percentile lowest point", "=PERCENTILE({L},0.05)", PCT),
               ("Worst lowest point", "=MIN({L})", PCT)]),
        ("DURING the holding period: peak-to-trough drawdown (daily closes)", [
            ("Median max drawdown", "=MEDIAN({D})", PCT),
            ("Severe case (5th percentile)", "=PERCENTILE({D},0.05)", PCT),
            ("Worst max drawdown", "=MIN({D})", PCT)]
            + [(f"Chance of a drawdown of {x}% or more", prob("D", f"<=-{x / 100}"), PCT)
               for x in (10, 20, 30, 50)]),
    ]
    put(summ, 1, 1, f"{name}: holding-period risk, {subtitle}").font = Font(
        name="Arial", bold=True, size=14)
    put(summ, 2, 1, f"Prices {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}. Each trading day (once history allows) "
        "ends one window. Historical frequencies, not forecasts.")
    if filtered:
        put(summ, 3, 1, "ENTRY FILTER: " + filter_text(ma_filter, ma_period) + ".", bold=True,
            color="C00000")
    header(summ, 4, ["Measure"] + [f"Hold {tag(h)}" for h in holds])
    row = 5
    for title, items in sections:
        put(summ, row, 1, title, bold=True)
        for c in range(1, len(holds) + 2):
            summ.cell(row, c).fill = PatternFill("solid", fgColor="D9E1F2")
        row += 1
        for label, f, fmt in items:
            put(summ, row, 1, label)
            for c, h in enumerate(holds, 2):
                first = int(np.argmax(starts[h] >= 0)) + 2
                if title == "Annualized returns" and h < 12:
                    put(summ, row, c, "n/a").alignment = Alignment(horizontal="right")
                    continue
                if filtered and label == "Approx. non-overlapping periods":
                    # the date-span formula would overstate the evidence when windows are filtered out
                    put(summ, row, c, count_separate(starts[h]), "0", color="0000FF")
                    continue
                put(summ, row, c, f.format(R=rng(h, 0), A=rng(h, 1), L=rng(h, 2), D=rng(h, 3),
                                           H=h, FIRSTDATE=f"Data!$A${first}",
                                           LASTDATE=f"Data!$A${last}"), fmt)
            row += 1
    widths(summ, [50] + [14] * len(holds))
    summ.freeze_panes = "B5"

    for kind in DEPTH_SHEETS:
        add_depth_analysis(wb, holds, rng, levels, kind, subtitle)

    # ---- Distribution: histogram + cumulative probability ----
    row = 1
    for h in holds:
        vals = rets[h][~np.isnan(rets[h])]
        lo, hi = np.percentile(vals, [0.5, 99.5])
        raw = max((hi - lo) / HIST_BINS, 0.001)
        mag = 10 ** math.floor(math.log10(raw))
        width = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        low = math.floor(lo / width) * width
        nb = max(int(math.ceil((hi - low) / width)), 5)
        put(hist, row, 1, f"{tag(h)} holding period", bold=True)
        put(hist, row + 1, 1, "First edge"); put(hist, row + 1, 2, round(low, 6), PCT, color="0000FF")
        put(hist, row + 2, 1, "Bin width"); put(hist, row + 2, 2, round(width, 6), PCT, color="0000FF")
        header(hist, row + 3, ["From", "To (below)", "Bin", "Windows", "Frequency", "Cumulative"])
        t = row + 4
        end = t + nb + 1
        x = rng(h, 0)
        for b in range(nb + 2):
            r = t + b
            cond = []
            if b > 0:
                put(hist, r, 1, f"=$B${row + 1}" if b == 1 else f"=B{r - 1}", PCT)
                cond.append(f'{x},">="&$A{r}')
            if b < nb + 1:
                put(hist, r, 2, f"=$B${row + 1}" if b == 0 else f"=A{r}+$B${row + 2}", PCT)
                cond.append(f'{x},"<"&$B{r}')
            lf = "0%" if abs(width * 100 - round(width * 100)) < 1e-9 else "0.0%"
            label = (f'="<"&TEXT(B{r},"{lf}")' if b == 0 else
                     f'=">="&TEXT(A{r},"{lf}")' if b == nb + 1 else f'=TEXT(A{r},"{lf}")')
            put(hist, r, 3, label).alignment = Alignment(horizontal="center")
            put(hist, r, 4, f"=COUNTIFS({','.join(cond)})", "#,##0")
            put(hist, r, 5, f"=D{r}/SUM($D${t}:$D${end})", PCT)
            put(hist, r, 6, f"=SUM($D${t}:D{r})/SUM($D${t}:$D${end})", PCT)
        ch = BarChart()
        ch.add_data(Reference(hist, min_col=5, min_row=t, max_row=end))
        ch.set_categories(Reference(hist, min_col=3, min_row=t, max_row=end))
        ch.gapWidth = 10
        ch.series[0].graphicalProperties.solidFill = "2F5597"
        chart_name = f"{name}: distribution of {tag(h)} returns"
        if filtered:
            chart_name += f" ({subtitle})"
        style_chart(ch, chart_name, "Share of windows",
                    "Holding-period return (bin lower edge)")
        hist.add_chart(ch, f"H{row}")
        row = max(end + 3, row + 22)
    widths(hist, [11, 11, 10, 11, 12, 12])

    # ---- Worst Periods ----
    put(worst, 1, 1, "Five worst windows per holding period. Windows in each list do not overlap "
        "(selected when the script runs).", bold=True)
    row = 3
    for h in holds:
        put(worst, row, 1, f"{tag(h)} holding period", bold=True)
        header(worst, row + 1, ["Rank", "Start date", "End date", "Return",
                                "Lowest vs entry", "Max drawdown"])
        row += 2
        for rank, i in enumerate(worst_episodes(rets[h], starts[h]), 1):
            r = i + 2
            put(worst, row, 1, rank)
            put(worst, row, 2, f"=Data!A{starts[h][i] + 2}", "yyyy-mm-dd")
            put(worst, row, 3, f"=Data!A{r}", "yyyy-mm-dd")
            for c, m in zip((4, 5, 6), (0, 2, 3)):
                put(worst, row, c, f"=Data!{cols[h][m]}{r}", PCT)
            row += 1
        row += 2
    widths(worst, [8, 14, 14, 13, 17, 15])

    # ---- Notes ----
    info = [
        ("Instrument", name), ("Data source", source), ("Run date", f"{dt.date.today():%Y-%m-%d}"),
        ("Window", "End price / start price - 1. Start = last trading day on or before the same "
                   "date N months earlier. A new window ends every trading day."),
        ("Overlap", "Windows overlap heavily, so they are not independent. Chances are shares of "
                    "historical windows. Long horizons and 99% tail figures rest on few episodes."),
        ("Annualized", "Shown only for holding periods of 1 year or more."),
        ("VaR / CVaR", "Historical, over the whole holding period (not scaled from daily data). "
                       "CVaR averages the worst 5% or 1% of windows (count rounded up; ties included)."),
        ("Lowest vs entry", "Lowest daily close in the window relative to the starting price."),
        ("Max drawdown", "Largest fall from a running peak to a later close within the window. "
                         "Stored as values calculated by the script."),
        ("Not captured", "Intraday lows, inflation, taxes, costs, currency effects."),
        ("Colors / editing", "Blue = values. Black = formulas. After changing prices or dates, "
                             "re-run the script (drawdowns and worst-period lists are not live)."),
    ]
    if filtered:
        info.append(("Entry filter", filter_text(ma_filter, ma_period).capitalize() + ". Checked on the "
                     "window's start date only; the holding period and stops then run as usual. "
                     "'Full history' and 'all windows' elsewhere mean all QUALIFYING windows. "
                     "'Approx. non-overlapping periods' is a count of qualifying windows that do not overlap. "
                     "Changing prices needs a re-run."))
    header(notes, 1, ["Item", "Detail"])
    for i, (a, b) in enumerate(info, 2):
        put(notes, i, 1, a, bold=True)
        put(notes, i, 2, b).alignment = Alignment(wrap_text=True, vertical="top")
    widths(notes, [18, 100])

    if STOP_MONTHS in holds:
        add_stop_analysis(wb, dates, p, starts[STOP_MONTHS], cols[STOP_MONTHS][0],
                          cols[STOP_MONTHS][2], levels, subtitle)
        if run_lab:
            add_stop_lab(wb, dates, p, starts[STOP_MONTHS], STOP_MONTHS, max_cvar, reps,
                         levels, levels, ma_filter, ma_period)
        else:
            print("Stop testing lab: OFF (use --stop-lab to run it)")

    for ws in wb.worksheets:
        ws.page_setup.orientation = "landscape"
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToHeight = 0
    wb.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=TICKER)
    ap.add_argument("--csv", help="use a local price file instead of downloading")
    ap.add_argument("--out", help="output .xlsx name (saved in results/) or a path with folders")
    ap.add_argument("--max-cvar", type=float, default=MAX_CVAR,
                    help="max acceptable 95%% CVaR as a decimal, e.g. 0.12")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES,
                    help="bootstrap resamples per block length (0 = off)")
    ap.add_argument("--stop-lab", action=argparse.BooleanOptionalAction, default=RUN_STOP_LAB,
                    help="run (--stop-lab) or skip (--no-stop-lab) the stop testing lab")
    ap.add_argument("--ma-filter", choices=["all", "above", "below"], type=str.lower, default=MA_FILTER,
                    help="which entry days to test: all, above the MA, or below (at/below) it")
    ap.add_argument("--ma-period", type=int, default=MA_PERIOD,
                    help="simple moving average length in trading days")
    a = ap.parse_args()
    if a.ma_filter not in ("all", "above", "below"):
        ap.error('MA_FILTER must be "all", "above" or "below"')
    if a.ma_period < 2:
        ap.error("the MA period must be at least 2")
    if not 0 < a.max_cvar < 1:
        ap.error("--max-cvar must be a decimal between 0 and 1 (0.12 = 12%)")
    label = a.ticker if not a.csv else (a.ticker if a.ticker != TICKER else "CSV data")
    series, src = load_prices(a.ticker, a.csv)
    safe = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "output"
    if a.ma_filter != "all":        # separate default file name, so runs do not overwrite each other
        safe += f"_{'above' if a.ma_filter == 'above' else 'atbelow'}{a.ma_period}MA"
    out = Path(a.out or f"risk_distribution_{safe}.xlsx")
    if out.parent == Path("."):          # bare file name -> results folder
        out = RESULTS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    build(series, label, src, out, a.max_cvar, a.bootstrap, a.stop_lab, a.ma_filter, a.ma_period)
