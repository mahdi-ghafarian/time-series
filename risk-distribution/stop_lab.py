"""
Stop testing lab: robustness tests that help choose between stop rules.
Used by risk_distribution.py. Keep both files in the same folder.
"""
import math

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# ---------------- SETTINGS (edit these) ----------------
# Fixed and trailing levels come from risk_distribution.py: full STOP_LADDER in auto mode,
# or your manual list
VOL_MULTIPLES = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
VOL_DAYS = 63                 # past daily returns used to size volatility stops
SLIPPAGE = [0.01, 0.02]       # extra loss on the exit price (first one can be a required test)
CASH_RATE = 0.03              # ASSUMED yearly cash yield for the "cash interest" test
MIN_DECADE_WINDOWS = 750      # fewer windows than this: decade is not used
CRISES = 3                    # deepest price declines used in the leave-one-out test
BLOCK_DAYS = [63, 126, 252]   # average bootstrap block lengths (trading days)
BOOT_STEP = 5                 # bootstrap evaluates an entry every 5th day (speed)
BOOT_CONFIDENCE = 0.80        # share of resamples that should be within the loss limit
WF_MIN_TRAIN_YEARS = 10
SEED = 12345
# -------------------------------------------------------

PCT = '0.0%;-0.0%;0.0%'
NAVY = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")

def candidate_list(fixed_levels, trail_levels):
    """Every rule the lab tests. 'No stop' must stay first."""
    return ([("None", 0.0)] + [("Fixed", x) for x in fixed_levels]
            + [("Trailing", x) for x in trail_levels] + [("Volatility", k) for k in VOL_MULTIPLES])


def label(c):
    kind, x = c
    if kind == "None":
        return "No stop"
    return f"Volatility {x:.2f}x" if kind == "Volatility" else f"{kind} {x * 100:g}%"


# ---------------- calculation engine ----------------
def entry_allowed(p, mode, period):
    """True where a window may START: close above, or at/below, its simple moving average.
    The average includes the entry day's close. Days without a full average are excluded."""
    if mode == "all":
        return np.ones(len(p), dtype=bool)
    ma = pd.Series(p).rolling(period).mean().values
    return p > ma if mode == "above" else p <= ma      # comparisons with NaN are False


def filter_text(mode, period):
    if mode == "all":
        return "all entry days"
    side = "ABOVE" if mode == "above" else "AT OR BELOW"
    return f"only entries with the close {side} its {period}-day simple moving average"


def window_matrix(p, s, e):
    """One row per window. Short rows repeat the end price, which cannot add a new trigger."""
    width = int((e - s).max()) + 1
    idx = np.minimum(s[:, None] + np.arange(width), e[:, None])
    return p[idx]


def entry_vol(p, s, months):
    """Volatility from returns known at entry, scaled to the holding period."""
    lr = pd.Series(np.diff(np.log(p), prepend=np.nan))
    v = lr.rolling(VOL_DAYS).std().values
    return v[s] * math.sqrt(252 * months / 12)


def run_stops(W, last, vol, candidates, delay=0):
    """Returns (return, hit, exit offset) arrays shaped [candidates, windows].

    last = offset of each window's end; delay = trading days between signal and exit."""
    entry = W[:, :1]
    from_entry = W / entry - 1
    from_peak = W / np.maximum.accumulate(W, axis=1) - 1
    n = len(W)
    rows = np.arange(n)
    R = np.empty((len(candidates), n))
    H = np.zeros((len(candidates), n), dtype=bool)
    X = np.empty((len(candidates), n), dtype=np.int32)
    for c, (kind, x) in enumerate(candidates):
        if kind == "None":
            hit = np.zeros(n, dtype=bool)
            exit_at = last
        else:
            if kind == "Fixed":
                breach = from_entry <= -x + 1e-12
            elif kind == "Trailing":
                breach = from_peak <= -x + 1e-12
            else:   # fixed distance, sized by volatility at entry
                dist = np.clip(x * vol, 0.01, 0.95)
                breach = from_entry <= -dist[:, None] + 1e-12
            hit = breach.any(axis=1)
            exit_at = np.where(hit, np.minimum(breach.argmax(axis=1) + delay, last), last)
        R[c] = W[rows, exit_at] / entry[:, 0] - 1
        H[c], X[c] = hit, exit_at
    return R, H, X


def tail_loss(R, q=0.05):
    """CVaR along the last axis: average of the worst q share, as a loss (negative = gain)."""
    k = max(1, math.ceil(R.shape[-1] * q - 1e-9))
    return -np.partition(R, k - 1, axis=-1)[..., :k].mean(axis=-1)


def choose(mean, cvar, limit):
    """The selection rule: highest mean among those within the limit; else lowest tail loss.
    Ties go to the earliest candidate (No stop is first)."""
    ok = np.flatnonzero(cvar <= limit + 1e-12)
    if len(ok):
        return int(ok[np.argmax(np.round(mean[ok], 10))]), True
    return int(np.argmin(np.round(cvar, 10))), False


def crisis_list(p, k=CRISES):
    """Deepest declines, each from a running high to the low before the next new high."""
    high = np.maximum.accumulate(p)
    new_high = np.flatnonzero(p >= high)
    out = []
    for a, b in zip(new_high, np.append(new_high[1:], len(p))):
        if b - a > 1:
            low = a + int(np.argmin(p[a:b]))
            out.append((p[low] / p[a] - 1, int(a), low))
    return sorted(out)[:k]


def episodes(idx, s, e):
    """Group windows whose date spans overlap. Returns (groups, share in the largest group)."""
    order = idx[np.argsort(s[idx], kind="stable")]
    sizes, reach = [], -1
    for i in order:
        if s[i] >= reach:
            sizes.append(0)
        sizes[-1] += 1
        reach = max(reach, e[i])
    return len(sizes), max(sizes) / len(order)


def bootstrap(p, months, reps, rng, candidates, ma_filter="all", ma_period=200):
    """Stationary block bootstrap: rebuild the daily history, re-test every candidate.
    Every candidate sees the same resampled histories. With an MA filter, the average is
    recalculated on each resampled history and only matching entry days are tested."""
    daily = p[1:] / p[:-1] - 1
    n, length = len(daily), int(round(months * 21))
    warmup = VOL_DAYS if ma_filter == "all" else max(VOL_DAYS, ma_period - 1)
    if n < length + warmup + 100:
        return {}
    all_starts = np.arange(warmup, n - length + 1, BOOT_STEP)
    out = {}
    for block in BLOCK_DAYS:
        mean = np.empty((reps, len(candidates)))
        cvar = np.empty_like(mean)
        b = redraws = 0
        while b < reps:
            jump = rng.random(n) < 1 / block
            jump[0] = True
            run = np.cumsum(jump) - 1
            first = np.flatnonzero(jump)
            origin = rng.integers(0, n, len(first))
            idx = (origin[run] + np.arange(n) - first[run]) % n
            path = np.concatenate([[1.0], np.cumprod(1 + daily[idx])])
            starts = all_starts[entry_allowed(path, ma_filter, ma_period)[all_starts]]
            if len(starts) < 20:      # too few matching entries in this history: draw another
                redraws += 1
                if redraws > reps:
                    print("  bootstrap skipped: resampled histories rarely meet the MA condition.")
                    return {}
                continue
            ends = starts + length
            R, _, _ = run_stops(window_matrix(path, starts, ends), ends - starts,
                                entry_vol(path, starts, months), candidates)
            mean[b], cvar[b] = R.mean(axis=1), tail_loss(R)
            b += 1
        out[block] = (mean, cvar)
        extra = f", {redraws} redrawn for too few matching entries" if redraws else ""
        print(f"  bootstrap: {block}-day blocks done ({reps} resamples{extra})")
    return out


def walk_forward(dates, s, e, R, limit):
    """Each year: choose a stop using only windows that ENDED before that year, then apply
    it to windows ENTERED during that year."""
    folds, chosen_ret, base_ret = [], [], []
    first_year = (dates[s[0]] + pd.DateOffset(years=WF_MIN_TRAIN_YEARS)).year + 1
    for year in range(first_year, dates[s[-1]].year + 1):
        boundary = pd.Timestamp(year=year, month=1, day=1)
        train = np.flatnonzero(dates[e] < boundary)
        test = np.flatnonzero((dates[s] >= boundary) & (dates[s] < boundary + pd.DateOffset(years=1)))
        if len(train) < 750 or len(test) == 0:
            continue
        T = R[:, train]
        c, met = choose(T.mean(axis=1), tail_loss(T), limit)
        folds.append(dict(year=year, train_end=dates[e[train[-1]]], first_entry=dates[s[test[0]]],
                          n=len(test), pick=c, met=met, train_cvar=tail_loss(T[c]),
                          test_mean=R[c, test].mean(), base_mean=R[0, test].mean(),
                          test_worst=R[c, test].min(), base_worst=R[0, test].min()))
        chosen_ret.append(R[c, test])
        base_ret.append(R[0, test])
    if folds:
        return folds, np.concatenate(chosen_ret), np.concatenate(base_ret)
    return folds, None, None


# ---------------- Excel output ----------------
def cell(ws, r, c, value, fmt=None, bold=False, color="000000"):
    x = ws.cell(r, c, value)
    x.font = Font(name="Arial", bold=bold, color=color)
    if fmt:
        x.number_format = fmt
    return x


def val(ws, r, c, v, fmt=PCT):
    """A value calculated by the script (blue)."""
    if isinstance(v, (float, np.floating)):
        v = None if math.isnan(v) else float(v)
    elif isinstance(v, np.integer):
        v = int(v)
    return cell(ws, r, c, "n/a" if v is None else v, fmt, color="0000FF")


def head(ws, r, names, height=45):
    for c, name in enumerate(names, 1):
        x = ws.cell(r, c, name)
        x.font = Font(name="Arial", bold=True, color="FFFFFF")
        x.fill = NAVY
        x.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[r].height = height


def widths(ws, values):
    for i, w in enumerate(values, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def add_stop_lab(wb, dates, p, starts, months, limit, reps, fixed_levels, trail_levels,
                 ma_filter="all", ma_period=200):
    print("Running stop tests...")
    candidates = candidate_list(fixed_levels, trail_levels)
    e = np.flatnonzero(starts >= 0)
    s = starts[e]
    keep = s >= VOL_DAYS          # every candidate is judged on the same windows
    s, e = s[keep], e[keep]
    if len(s) < 750:
        print("Stop lab skipped: not enough history.")
        return
    W, last, vol = window_matrix(p, s, e), e - s, entry_vol(p, s, months)
    R, H, X = run_stops(W, last, vol, candidates)
    R1, _, _ = run_stops(W, last, vol, candidates, delay=1)
    rows = np.arange(len(s))
    n_c = len(candidates)
    mean, cvar = R.mean(axis=1), tail_loss(R)

    # execution stress
    slip = [np.where(H, (1 + R) * (1 - x) - 1, R) for x in SLIPPAGE]
    cal_days = (dates.values[e] - dates.values[s[None, :] + X]) / np.timedelta64(1, "D")
    cash = np.where(H, (1 + R) * (1 + CASH_RATE) ** (cal_days / 365.25) - 1, R)
    stress = [("Next-day close fill", R1)] + [(f"{x:.0%} slippage", y) for x, y in zip(SLIPPAGE, slip)] \
        + [(f"Cash earns {CASH_RATE:.0%} a year", cash)]

    # decades: windows must start AND end inside the decade
    decades = []
    for d in range(dates[0].year // 10 * 10, dates[-1].year + 1, 10):
        m = (dates[s].year >= d) & (dates[e].year < d + 10)
        if m.sum() >= MIN_DECADE_WINDOWS:
            decades.append((f"{d}s", m))

    # crises: drop every window that touches the peak-to-low span
    crises, kept_all = [], np.ones(len(s), dtype=bool)
    for depth, a, b in crisis_list(p):
        m = ~((s < b) & (e > a))
        kept_all &= m
        if m.sum() >= 250:
            crises.append((f"{dates[a]:%Y-%m} to {dates[b]:%Y-%m}", depth, a, b, m))

    # non-overlapping schedules, one starting in each month
    schedules = []
    per = dates[s].to_period("M")
    for first_month in per.unique()[:12]:
        cand = np.flatnonzero(per >= first_month)
        picks, reach = [], -1
        for i in cand:
            if s[i] >= reach:
                picks.append(i)
                reach = e[i]
        if len(picks) >= 5:
            schedules.append(np.array(picks))
    sched_diff = (np.array([R[:, i].mean(axis=1) - R[0, i].mean() for i in schedules])
                  if schedules else np.full((1, n_c), np.nan))

    boot = (bootstrap(p, months, reps, np.random.default_rng(SEED), candidates, ma_filter, ma_period)
            if reps > 0 else {})
    folds, wf_ret, wf_base = walk_forward(dates, s, e, R, limit)

    at = wb.sheetnames.index("Stop Analysis") + 1 if "Stop Analysis" in wb.sheetnames else 1
    card = wb.create_sheet("Stop Scorecard", 1)
    tests = wb.create_sheet("Stop Stress Tests", at + 1)
    periods = wb.create_sheet("Stop Periods & Crises", at + 2)
    bs = wb.create_sheet("Stop Bootstrap", at + 3)
    wf = wb.create_sheet("Stop Walk Forward", at + 4)
    names = [label(c) for c in candidates]
    DATA0 = 5                      # first candidate row on the three test sheets

    def candidate_rows(ws, first=DATA0):
        for i, nm in enumerate(names):
            cell(ws, first + i, 1, nm)

    # ---- Stress tests ----
    cell(tests, 1, 1, "Execution stress tests (95% CVaR: loss positive, negative = gain)", bold=True)
    cell(tests, 2, 1, "Values calculated by the script. Slippage and next-day fill apply only to "
         "windows whose stop triggered.")
    cols = ["Stop rule", "% stopped", "Base mean", "Base 95% CVaR", "Base worst",
            "% ending with loss", "% of stopped that would have ended up"]
    for nm, _ in stress:
        cols += [f"{nm}: mean", f"{nm}: CVaR"]
    cols += ["Episodes in worst 5%", "Largest episode share of worst 5%",
             "Non-overlap schedules: lowest mean change", "Non-overlap schedules: highest mean change"]
    head(tests, 4, cols, 60)
    candidate_rows(tests)
    k5 = max(1, math.ceil(len(s) * 0.05))
    for c in range(n_c):
        r = DATA0 + c
        stopped = H[c].sum()
        val(tests, r, 2, H[c].mean())
        val(tests, r, 3, mean[c]); val(tests, r, 4, cvar[c]); val(tests, r, 5, R[c].min())
        val(tests, r, 6, (R[c] < 0).mean())
        val(tests, r, 7, (R[0][H[c]] > 0).mean() if stopped else np.nan)
        col = 8
        for _, y in stress:
            val(tests, r, col, y[c].mean()); val(tests, r, col + 1, tail_loss(y[c])); col += 2
        # include all windows tied with the cutoff
        cut = np.partition(R[c], k5 - 1)[k5 - 1]
        groups, share = episodes(np.flatnonzero(R[c] <= cut + 1e-12), s, e)
        val(tests, r, col, groups, "0"); val(tests, r, col + 1, share)
        val(tests, r, col + 2, sched_diff[:, c].min()); val(tests, r, col + 3, sched_diff[:, c].max())
    tests.freeze_panes = "B5"
    widths(tests, [20] + [14] * (len(cols) - 1))

    # ---- Periods and crises ----
    cell(periods, 1, 1, "Decades and leave-one-crisis-out", bold=True)
    cell(periods, 2, 1, "Decade tests use only windows starting AND ending in that decade. Crisis tests "
         "REMOVE windows touching the dates (diagnostic only). 'CVaR saved' = No stop CVaR minus rule CVaR.")
    cols = ["Stop rule"]
    for nm, _ in decades:
        cols += [f"{nm}: mean", f"{nm}: CVaR"]
    cols.append("Worst decade CVaR")
    for nm, depth, *_ in crises:
        cols += [f"Without {nm} ({depth:.0%}): CVaR saved", f"Without {nm}: mean change vs no stop"]
    cols += ["Without all listed: CVaR saved", "Without all listed: mean change vs no stop"]
    head(periods, 4, cols, 75)
    candidate_rows(periods)
    worst_decade_col = 2 + 2 * len(decades)
    for c in range(n_c):
        r, col = DATA0 + c, 2
        dc = []
        for _, m in decades:
            dc.append(tail_loss(R[c, m]))
            val(periods, r, col, R[c, m].mean()); val(periods, r, col + 1, dc[-1]); col += 2
        val(periods, r, col, max(dc) if dc else np.nan); col += 1
        for m in [x[-1] for x in crises] + [kept_all]:
            ok = m.sum() >= 250
            val(periods, r, col, tail_loss(R[0, m]) - tail_loss(R[c, m]) if ok else np.nan)
            val(periods, r, col + 1, R[c, m].mean() - R[0, m].mean() if ok else np.nan)
            col += 2
    r = DATA0 + n_c + 1
    cell(periods, r, 1, "Windows used per test:", bold=True)
    for i, (nm, m) in enumerate([(n, m) for n, m in decades] + [("Without " + x[0], x[-1]) for x in crises]
                                + [("Without all listed", kept_all)], 1):
        cell(periods, r + i, 1, nm); val(periods, r + i, 2, int(m.sum()), "#,##0")
    periods.freeze_panes = "B5"
    widths(periods, [24] + [15] * (len(cols) - 1))

    # ---- Bootstrap ----
    cell(bs, 1, 1, "Block bootstrap: uncertainty from resampled histories", bold=True)
    cell(bs, 2, 1, f"{reps:,} resamples per block length; random seed {SEED}; fixed "
         f"{int(round(months * 21))}-day windows entered every {BOOT_STEP} days. 'Picked' uses the "
         f"selection rule with the loss limit at run time ({limit:.1%}).")
    cols = ["Stop rule"]
    for b in boot:
        cols += [f"{b}d: CVaR 5th pct", f"{b}d: CVaR median",
                 f"{b}d: CVaR {BOOT_CONFIDENCE:.0%} pct", f"{b}d: CVaR 95th pct",
                 f"{b}d: mean change 5th pct", f"{b}d: mean change 95th pct",
                 f"{b}d: % with lower CVaR than no stop", f"{b}d: % picked as best"]
    cols.append(f"Highest {BOOT_CONFIDENCE:.0%}-pct CVaR across blocks")
    head(bs, 4, cols, 62)
    candidate_rows(bs)
    boot_gate = np.full(n_c, np.nan)
    col = 2
    for b, (bm, bc) in boot.items():
        gate = np.quantile(bc, BOOT_CONFIDENCE, axis=0)
        boot_gate = np.fmax(boot_gate, gate)
        picks = np.bincount([choose(bm[i], bc[i], limit)[0] for i in range(len(bm))], minlength=n_c)
        qs = np.quantile(bc, [0.05, 0.5, 0.95], axis=0)
        dq = np.quantile(bm - bm[:, :1], [0.05, 0.95], axis=0)
        for c in range(n_c):
            vals = [qs[0, c], qs[1, c], gate[c], qs[2, c], dq[0, c], dq[1, c],
                    (bc[:, c] < bc[:, 0] - 1e-12).mean(), picks[c] / len(bm)]
            for j, v in enumerate(vals):
                val(bs, DATA0 + c, col + j, v)
        col += 8
    for c in range(n_c):
        val(bs, DATA0 + c, col, boot_gate[c])
    boot_col = col
    bs.freeze_panes = "B5"
    widths(bs, [20] + [14] * (len(cols) - 1))

    # ---- Walk forward ----
    cell(wf, 1, 1, "Walk-forward: does CHOOSING a stop from past data work on later data?", bold=True)
    cell(wf, 2, 1, f"Loss limit at run time: {limit:.1%}. Each year the choice uses only windows that "
         "ended before 1 January. Uses base CVaR only, NOT the scorecard stress tests. --max-cvar changes the limit.")
    if folds:
        head(wf, 4, ["Out-of-sample summary", "Walk-forward choice", "No stop (same windows)"], 30)
        stats = [("Mean return", wf_ret.mean(), wf_base.mean()),
                 ("95% CVaR", tail_loss(wf_ret), tail_loss(wf_base)),
                 ("Worst return", wf_ret.min(), wf_base.min()),
                 ("% ending with a loss", (wf_ret < 0).mean(), (wf_base < 0).mean())]
        for i, (nm, a, b) in enumerate(stats):
            cell(wf, 5 + i, 1, nm); val(wf, 5 + i, 2, a); val(wf, 5 + i, 3, b)
        changes = sum(a["pick"] != b["pick"] for a, b in zip(folds, folds[1:]))
        cell(wf, 9, 1, "Times the choice changed"); val(wf, 9, 2, changes, "0")
        cell(wf, 10, 1, "Yearly tests"); val(wf, 10, 2, len(folds), "0")
        head(wf, 12, ["Entry year", "Last training end date", "First test entry date", "Chosen rule",
                      "Met limit in training?", "Training CVaR", "Test windows", "Test mean",
                      "No stop mean", "Test worst", "No stop worst"])
        for i, f in enumerate(folds):
            r = 13 + i
            cell(wf, r, 1, str(f["year"]))
            val(wf, r, 2, f["train_end"].to_pydatetime(), "yyyy-mm-dd")
            val(wf, r, 3, f["first_entry"].to_pydatetime(), "yyyy-mm-dd")
            val(wf, r, 4, names[f["pick"]], None)
            val(wf, r, 5, "Yes" if f["met"] else "NO (lowest CVaR used)", None)
            val(wf, r, 6, f["train_cvar"]); val(wf, r, 7, f["n"], "0")
            for j, key in enumerate(["test_mean", "base_mean", "test_worst", "base_worst"]):
                val(wf, r, 8 + j, f[key])
    else:
        cell(wf, 4, 1, f"Not run: needs more than {WF_MIN_TRAIN_YEARS} years plus test data.")
    widths(wf, [26, 22, 22, 20, 22, 14, 13, 12, 13, 12, 13])

    # ---- Scorecard (live formulas) ----
    first, end = 15, 15 + n_c - 1
    cell(card, 1, 1, f"Stop scorecard: {months}-month holding period").font = Font(
        name="Arial", bold=True, size=14)
    cell(card, 2, 1, "A rule-based shortlist from historical tests, not a recommendation. "
         "Yellow cells are yours to edit.")
    cell(card, 3, 1, "Maximum acceptable 95% CVaR (loss)", bold=True)
    cell(card, 3, 2, limit, PCT, color="0000FF").fill = INPUT
    yes_no = DataValidation(type="list", formula1='"Yes,No"', allow_blank=False)
    card.add_data_validation(yes_no)
    gates = [("Require: next-day close fill within limit", "Yes"),
             (f"Require: {SLIPPAGE[0]:.0%} slippage within limit", "Yes"),
             ("Require: every eligible decade within limit", "Yes" if decades else "No"),
             (f"Require: bootstrap {BOOT_CONFIDENCE:.0%} pct CVaR within limit",
              "Yes" if boot else "No")]
    for i, (text, default) in enumerate(gates):
        cell(card, 4 + i, 1, text)
        x = cell(card, 4 + i, 2, default, color="0000FF")
        x.fill = INPUT
        yes_no.add(x)
    cols = ["Stop rule", "% stopped", "Mean return", "Mean change vs no stop", "Base CVaR",
            "Worst return", "Next-day fill CVaR", f"{SLIPPAGE[0]:.0%} slippage CVaR",
            "Worst decade CVaR", f"Bootstrap {BOOT_CONFIDENCE:.0%} pct CVaR", "Tests failed",
            "Passes?", "Neighbors also pass?", "Rank by mean (passing)",
            "Rank by mean (passing + stable)"]
    head(card, first - 1, cols)
    L = get_column_letter

    def ref(sheet, col, row):
        return f"='{sheet}'!{L(col)}{row}"

    for c, (kind, _) in enumerate(candidates):
        r, src = first + c, DATA0 + c
        cell(card, r, 1, names[c])
        links = {2: ref(tests.title, 2, src), 3: ref(tests.title, 3, src),
                 5: ref(tests.title, 4, src), 6: ref(tests.title, 5, src),
                 7: ref(tests.title, 9, src), 8: ref(tests.title, 11, src),
                 9: ref(periods.title, worst_decade_col, src), 10: ref(bs.title, boot_col, src)}
        for col, f in links.items():
            cell(card, r, col, f, PCT, color="008000")
        cell(card, r, 4, f"=C{r}-$C${first}", PCT)
        # a required test with no result counts as failed
        fails = [f'IF(E{r}>$B$3+1E-12,1,0)'] + [
            f'IF($B${4 + i}="Yes",IF(ISNUMBER({x}{r}),IF({x}{r}>$B$3+1E-12,1,0),1),0)'
            for i, x in enumerate("GHIJ")]
        cell(card, r, 11, "=" + "+".join(fails), "0")
        cell(card, r, 12, f'=IF(K{r}=0,"YES","no")')
        # neighbors = adjacent distances within the same stop family
        if kind == "None":
            f = f"=L{r}"
        else:
            near = [first + j for j in (c - 1, c + 1)
                    if 0 <= j < n_c and candidates[j][0] == kind]
            test = ",".join(f'L{j}="YES"' for j in near)
            f = f'=IF(AND(L{r}="YES",{test}),"YES","no")'
        cell(card, r, 13, f)
        for col in (11, 12, 13):
            card.cell(r, col).alignment = Alignment(horizontal="center")
        for col, flag in ((14, "L"), (15, "M")):
            cell(card, r, col, f'=IF({flag}{r}="YES",COUNTIFS(${flag}${first}:${flag}${end},"YES",'
                 f'$C${first}:$C${end},">"&(C{r}+1E-10))+1,"")', "0")
    rng = lambda c: f"${c}${first}:${c}${end}"
    cell(card, 10, 1, "Result of your rule", bold=True)
    for r, text, col in ((11, "Highest mean among passing rules", "N"),
                         (12, "Highest mean among passing AND stable rules", "O")):
        cell(card, r, 1, text)
        cell(card, r, 2, f'=IFERROR(INDEX({rng("A")},MATCH(1,{rng(col)},0)),"No rule qualifies")',
             bold=True)
    cell(card, 3, 4, "How to read: a rule passes only if base CVaR AND every required test are within "
         "the limit.")
    cell(card, 4, 4, "Stable = adjacent distances in the same family also pass (end-of-range rules have one).")
    cell(card, 5, 4, "Ties go to the rule listed first (No stop is first).")
    cell(card, 6, 4, "Check '% stopped': a rule that rarely triggers behaves much like No stop.")
    cell(card, 7, 4, "Green = linked from test tabs. Those results are fixed until the script is re-run.")
    cell(card, 8, 4, "Changing yellow cells updates this tab only, not Walk Forward or 'picked as best'.")
    r = end + 2
    if ma_filter != "all":
        cell(card, 9, 1, "Entry filter: " + filter_text(ma_filter, ma_period) + ".", bold=True)
    notes = [
        f"Windows tested: {len(s):,}, ending {dates[e[0]]:%Y-%m-%d} to {dates[e[-1]]:%Y-%m-%d}. The first "
        f"{VOL_DAYS} days are used only to size volatility stops, so figures can differ slightly from Stop Analysis.",
        f"Volatility stop: distance = multiple x daily volatility of the prior {VOL_DAYS} days, scaled to the "
        "holding period; set at entry and kept fixed (limited to 1%-95%).",
        "Selecting from many rules using the same history flatters the winner. Passing is not proof "
        "of future performance.",
    ]
    for i, t in enumerate(notes):
        cell(card, r + i, 1, t)
    card.freeze_panes = "B15"
    widths(card, [46, 17] + [13] * 9 + [10, 13, 13, 15])
