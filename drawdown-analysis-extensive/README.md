# Drawdown, recovery and stop-loss analysis

`drawdown-dynamics.py` studies how stop-losses behave on a daily price series
(QQQ by default), independently of any entry signal. It runs on the observed
history or on resampled and simulated price paths.

This is a research tool, not investment advice.

## How the script works: overview

The script asks: **if you buy, how do prices tend to fall and recover, and how
does adding a stop-loss change the result compared with not using one?**

```
1. Load daily prices        QQQ download, or your CSV
          |
2. Create trades            historical: buy on selected past dates
   (--method)               stationary / circular / fhs: generate new price
          |                 paths and buy at the first Open of each
          |
3. Analyze the trades       recovery:    was a drop hit, and did price
   (--analysis)                          return to the entry price?
          |                 compare:     Stop versus Hold
          |                 walkforward: choose a stop from earlier data,
          |                              test it on later actual trades
          |
4. Save result CSVs
          |
5. Draw charts              python plot_results.py   (separate step)
          |
6. Review and decide        you, using section 9. "No stop" can be the answer
```

### How trades are bought

| Setting | Entries |
|---|---|
| `historical` (default) | 5 random days in each month; change with `--entries-per-month` |
| `historical --entries-per-month all` | Every day, so entry selection isn't random |
| `stationary`, `circular`, `fhs` | The first Open of each randomly generated path |

Every buy is at the Open. No entry signal is used. In walk-forward, test
trades are always historical entries, whichever method chose the stop.

### What Stop is compared with
In `compare`, each trade is run two ways from the same entry:

| Choice | What happens |
|---|---|
| **Hold** | Sell at the Close H trading days after entry (default 252, about one year), with no stop |
| **Stop** | If the stop is hit, sell and stay in cash to the end of the same period. Otherwise sell when Hold does |

**"Hold" is not buying once and holding for the whole history.** It is a
fixed-period hold of each individual trade.

**Example, 20% stop:** you buy at 100. The price falls to 80, then ends the
period at 110. With a fill at the stop price, Stop sells at 80 for about -20%,
while Hold makes +10%. If the price instead ended at 60, Stop would still lose
about 20% and Hold would lose 40%. This ignores slippage and assumes no
interest on cash. A gap below the stop would give a worse fill.

The results across all trades are summarized as average and median returns,
log growth, large-loss measures and how often Stop beats Hold. See
`qqq_stop_vs_hold.csv`. The recovery analysis is different: it describes drops
and recoveries and doesn't compare Stop with Hold.

### What the script does not do

| Not covered | Meaning |
|---|---|
| Full-history buy-and-hold | It doesn't show how one investment grows from the first date to the last |
| Portfolio simulation | Each trade is tested separately. Overlapping trades don't share capital, and their returns aren't compounded |
| Random entry versus another entry rule | The comparison is with and without a stop, using the same entry. It doesn't test whether random timing beats another way of buying |
| Re-entry | After a stop, that trade doesn't buy again |

Comparing a complete strategy with full-history buy-and-hold would need extra
rules for when to buy, how much to invest, how to handle open positions and
when to re-enter. See also section 11.

## 1. Install

```bash
pip install -r requirements.txt
python test_drawdown_dynamics.py     # offline tests on synthetic data
```

`matplotlib` is needed only for `plot_results.py`. To review results and make
a decision, see sections 7 to 9.

`scipy` is used by yfinance price repair. `arch` is needed only for `fhs` and
`--block-lengths auto`.

## 2. Quick start

```bash
# Default: historical series, 5 entries per month, recovery + compare
python drawdown-dynamics.py

# Every trading day instead of a monthly sample
python drawdown-dynamics.py --entries-per-month all

# Stationary vs circular bootstrap, 5 seeds
python drawdown-dynamics.py --method stationary circular \
    --analysis compare --block-lengths 21 63 126 252 --n-seeds 5

# All four methods plus walk-forward
python drawdown-dynamics.py --method all --analysis all

# Automatic block length
python drawdown-dynamics.py --method stationary circular --block-lengths auto

# Your own data, finer stop grid, loss limit
python drawdown-dynamics.py --csv prices.csv --ticker QQQ \
    --stops 0.08 0.10 0.12 0.15 0.18 0.20 0.25 \
    --horizon 252 --cash-rate 0.03 --min-es -0.30
```

A CSV needs the date in the first column, then `Open, High, Low, Close`.
Prices must be adjusted consistently for splits (and dividends, if you want
total return). `--ticker` only names the outputs when `--csv` is used.

Each download is saved as `results/qqq_prices.csv`. Adjusted history can change
between downloads, so to reproduce a run, copy that file elsewhere and pass it
with `--csv`. Runs with different settings overwrite the same output names, so
use `--output-dir` to keep them apart.

### Does the code run all methods?

**Not by default.** `python drawdown-dynamics.py` runs only the `historical`
method, with the `recovery` and `compare` (Stop vs Hold) analyses. To run all
methods and analyses, then draw the charts as a separate step:

```bash
python drawdown-dynamics.py --method all --analysis all
python plot_results.py
```

| Option | What `all` includes |
|---|---|
| `--method all` | `historical`, `stationary`, `circular` and `fhs` |
| `--analysis all` | `recovery`, `compare` and `walkforward` |

With the default block lengths (21, 63, 126, 252), this gives **10
scenarios**: 1 historical, 4 stationary, 4 circular and 1 FHS.

`all` doesn't change other settings:

* **Entries:** historical trades still use 5 per month. Add
  `--entries-per-month all` to use every day.
* **Seeds:** one is used. `--n-seeds 5` repeats `compare` for the simulated
  methods only; recovery and walk-forward aren't repeated.
* **Block lengths:** `auto` and the single-day shuffle (`1`) aren't included.
  `--block-lengths 1 auto` runs both for the bootstraps, but **replaces** the
  four default lengths. Use a separate `--output-dir`, and pass the same
  folder to `plot_results.py` with `--results-dir`.

A fuller base run:

```bash
python drawdown-dynamics.py --method all --analysis all \
    --entries-per-month all --n-seeds 5 --output-dir results_base
python plot_results.py --results-dir results_base
```

On your first run, run `python test_drawdown_dynamics.py` beforehand, then
check the output for warnings, especially about data quality or the FHS model
fit.

## 3. Methods (`--method`)

| Method | What it does | Strength | Main limitation |
|---|---|---|---|
| `historical` | Trades start on observed dates | No resampling assumption | One path; overlapping trades are dependent |
| `stationary` | Joins blocks of random (geometric) length, mean L, wrapping at the end | Less tied to one exact block length | Breaks dependence at joins |
| `circular` | Same, with every block exactly L days | Simple; typically more efficient for variance estimation in theory | Joins at fixed positions |
| `fhs` | Fits GJR-GARCH(1,1) and resamples standardized daily shocks | Models changing volatility | Depends on the model; shocks are drawn independently |

Both bootstraps keep each day's Open, High, Low and Close, relative to the
previous Close, together. Price levels are never resampled. They preserve the
order of days inside a block only.

For simulated methods, **each path is one trade**, entered at the first Open.
`--paths` sets their number. `--entries-per-month` does not affect them.

**Block length.** Try several. `--block-lengths 1` shuffles single days
(either method). It removes time dependence and is a useful reference: compare
it with longer blocks to see how much results depend on the ordering of days.
`auto` uses the Politis-White rule with the 2009 correction, giving separate
values for the two bootstraps. By default it is applied to squared returns
(`--auto-series returns` changes this). It is designed for estimating the
variance of a sample mean, not for choosing stops.

**FHS.** The model is fitted with Student-t errors; shocks are then drawn from
the observed standardized residuals, not from a t distribution. The same
day's O, H, L and C are scaled by that day's close-to-close volatility.
`--fhs-start random` (default) starts each path at a volatility drawn from
history, representing a typical entry date. `current` starts at the forecast
following the last observation. Check the printed persistence and any
convergence warning. This is one model; others may give different answers.

**Check the drift.** Stop results depend strongly on the market's average
return. In `stop_vs_hold.csv`, compare the Hold rows (`StopPct` = 0) across
scenarios. All methods are based on past returns, so they largely inherit the
past upward or downward drift. None of them forecasts it.

### How each method works

All four methods create **price paths**: a sequence of daily Open, High, Low
and Close values after an entry. The stop and recovery rules are then applied
to those paths in the same way. Only the way the paths are made differs.

#### The shared idea behind the three simulated methods
QQQ gives us only one history. These methods build many alternative sequences
from its past days. They don't use past price *levels*. Each day is stored as
its movement from the previous day's Close. For example, if yesterday closed
at 100 and today's Open, High, Low and Close were 101, 103, 99 and 102, the day
is stored as 1.01, 1.03, 0.99 and 1.02.

To build a path, the script chains these daily movements together. If the path
is at 200 and the next day used is the one above, its prices become 202, 206,
198 and 204. The next day then starts from 204. The path is finally rescaled
so that its first Open, the entry price, equals 1.

#### `historical`: what happened after real entry dates?
1. Pick entry dates: a few random days in each month, or every day.
2. Buy at that day's Open.
3. Follow the actual prices that came next.
4. Apply the stop and recovery rules. Entries without enough later data for
   the test are left out.

Nothing is reshuffled. But entries close together share most of their future
days, so thousands of trades are **not** thousands of independent examples.
In effect, a few big market episodes can be counted many times.

#### `stationary`: reshuffle history in blocks of varying length
1. Pick a random past day and use its movement.
2. For the next day, usually continue with the following day in history.
   With a chance of 1 in L, jump to a new random day instead.
3. If the end of history is reached, continue from the beginning.
4. Repeat until the path is long enough. Then build more paths.

L is the **average** block length. With L = 63, there is a 1-in-63 chance of a
jump before each new day, so some blocks are very short and others very long.

#### `circular`: reshuffle history in blocks of one fixed length
1. Pick a random past day.
2. Copy exactly L days in a row, wrapping to the beginning of history if
   needed.
3. Pick another random day and copy the next L days.
4. Repeat until the path is long enough, trimming the last block if needed.

With L = 63, a new block starts every 63 days. The only difference from
`stationary` is how block length is decided.

**Illustration** (letters are consecutive days; history runs A to J, then
wraps to A):

| Method | Example path, with `\|` marking a jump |
|---|---|
| `circular`, L = 3 | D E F \| I J A \| B C D |
| `stationary`, average L = 3 | D E \| I J A B C \| F \| B C |

A new random block can happen to start at the next day in history, so a jump
isn't always visible.

**Choosing L:** short blocks mix history more and break long trends and
selloffs. Long blocks keep more of the original order, but provide fewer
different combinations. Neither is automatically better, so compare several.

#### `fhs`: reuse past surprises, adjusted for changing volatility
Markets have calm and turbulent stretches. Filtered historical simulation
(FHS) separates **how volatile the market was** from **how unusual each day's
move was** for those conditions.

1. **Fit a volatility model** (GJR-GARCH) to daily Close-to-Close returns.
   It estimates each day's expected volatility from earlier information. It
   allows a fall to raise future volatility differently from an equal rise.
   The model's settings are estimated from the whole sample being fitted.
2. **Standardize each day:** subtract the model's average daily return, then
   divide by that day's volatility. For example, ignoring the small average,
   a 2% fall when volatility is 1% is a shock of about -2. A 4% fall when
   volatility is 2% is also about -2. (The script uses log returns.)
3. **Draw a random past day's shocks** (Open, High, Low and Close together).
   Days are drawn one at a time, not in blocks.
4. **Scale them to the path's current volatility** and add back the average,
   giving the new day's prices.
5. **Update volatility** using the new Close, then repeat from step 3.

A large simulated move can therefore raise volatility on the following days,
creating turbulent stretches even though shocks are drawn independently.
With `--fhs-start random` (default), each path starts at a volatility level
drawn from history. With `current`, all paths start at the model's forecast
for the day after the data ends.

#### Which question does each method help answer?
| Method | Question | What it relies on |
|---|---|---|
| `historical` | What happened to trades on the actual price history? | The past periods covered |
| `stationary` | What if stretches of past days came in a different order, with varying lengths? | Past days and the average block length |
| `circular` | What if equal-length stretches came in a different order? | Past days and the fixed block length |
| `fhs` | What if past shocks occurred under changing volatility? | Past shocks and the volatility model |

No method forecasts the next price path. All depend on past data, and the
simulated ones can't show a kind of daily surprise absent from that sample.
They can combine days into longer rises or falls than history contains. FHS can
also produce daily moves larger than any observed, by scaling a past shock to
higher volatility.

#### How the paths are used
* **Recovery:** checks whether each path falls to the stop level, then whether
  it gets back to the entry price.
* **Stop vs Hold (`compare`):** runs both choices on the same path and compares
  the final results.
* **Walk-forward:** for each test year, picks a stop using earlier data only
  (actual or simulated). It then tests that choice on **actual** entries in
  that year.

## 4. Analyses (`--analysis`)

### `recovery`: how drawdowns behave
For each stop, W1 and W2: was the stop reached within W1, and did price then
return to the entry price within W2? This **does not** show whether a stop is
profitable.

### `compare`: stop versus Hold
Both end at the same horizon H (`--horizon`, default 252 days).

* **Hold:** sell at the Close of day H.
* **Stop:** if a Low reaches the stop, sell at the stop price. If a later day
  opens at or below it, sell at that Open. Apply `--slippage`, then hold cash
  at `--cash-rate` until day H. No re-entry.

`StopPct = 0` is Hold. All rows within one scenario and seed use the same
trades or paths, so differences are paired.

### `walkforward`: out-of-sample check
For each calendar year, the stop is chosen from prices dated **before** that
year, then applied to the actual historical entries in that year.

* The training window expands; `--min-train-years` sets the first.
* Training trades must finish inside the training data.
* The historical method trains on every training day; the monthly sample
  applies to test trades.
* Simulated methods resample, or refit, using training data only.
* If no row meets `--min-es`, Hold is used and `ConstraintMet` is False. This
  fallback does not itself satisfy the constraint.
* Test trades need a complete horizon, so the latest year may be partial or
  absent.
* Trades overlap, including across adjacent test years. The number of trades
  greatly overstates the amount of independent evidence.
* The summary pools trades. It is **not** a portfolio equity curve.
* It uses `--seed` only. Repeat with other seeds to check sensitivity.

## 5. Conventions

* Entry is at the Open of day 0, and **day 0's Low counts** toward the stop.
* A window of N days means day 0 plus the next N days. For recovery, W2 means
  the N days after the stop-hit day. `MedianDaysToStop = 0` means entry day.
* Recovery on the stop-hit day is not counted, because daily bars don't show
  the order of High and Low. This can understate recovery.
* The recovery table measures from the stop **level**. Gap fills and slippage
  apply only to `compare` and `walkforward`.
* Daily data can't show intraday fills. A real stop can fill worse than
  modelled, even without an opening gap.

### Which entries are included
For `historical`, an entry is used only if enough future data exists.

| Setting | Recovery rows use | Trade-off |
|---|---|---|
| `--eligibility per-window` (default) | Entries with W1 + W2 days available | Keeps recent data; rows can cover different dates |
| `--eligibility common` | Entries with the longest W1 + W2 (1,008 days) | Comparable rows; drops about the last 4 years of entries |

With `per-window`, `StopHitProbability` for the same stop and W1 can differ
across W2 because the set of entries differs. `compare` needs only H days.
Simulated paths always have the full length.

## 6. Options

| Option | Default | Meaning |
|---|---|---|
| `--ticker`, `--start`, `--end` | QQQ, 1999-03-10, latest | Download settings; `--end` is exclusive |
| `--csv` | none | Use a local file |
| `--output-dir` | `results/` beside the script | Output folder |
| `--method` | historical | One or more methods, or `all` |
| `--analysis` | recovery compare | Also `walkforward`, or `all` |
| `--entries-per-month` | 5 | A number or `all` |
| `--eligibility` | per-window | See above |
| `--paths` | 5000 | Simulated paths |
| `--block-lengths` | 21 63 126 252 | Trading days, or `auto` |
| `--auto-series` | squared | Series used by `auto` |
| `--fhs-start` | random | `random` or `current` |
| `--seed`, `--n-seeds` | 42, 1 | `--n-seeds` repeats `compare` only |
| `--stops` | 0.05 to 0.50 by 0.05 | Fractions; Hold is always added to `compare` |
| `--horizon` | 252 | H for `compare` and `walkforward`; recovery windows are set by `WINDOWS` in the script |
| `--slippage` | 0.0005 | 0.05% on stop exits only |
| `--cash-rate` | 0.0 | Constant annual rate |
| `--objective` | mean_log | See below |
| `--min-es` | none | e.g. `-0.30`; also applies to Hold |
| `--es-alpha` | 0.05 | Tail share |
| `--periods` | 3 | Historical sub-periods; 0 turns off |
| `--min-train-years` | 8 | Walk-forward |

The default zero cash rate puts stops at a disadvantage when cash earns
interest. A constant rate does not reproduce historical rate changes. Taxes,
and costs common to Hold and Stop, are not modelled.

### Objectives
| Name | Maximizes | Note |
|---|---|---|
| `mean_log` | Average log of final wealth | Growth criterion; penalizes large losses more than equal-sized gains help |
| `median_log` | Typical outcome | Same ranking as median wealth; can ignore changes confined to the tails, and often stays equal to Hold when few trades stop |
| `return_over_es` | Mean return / size of expected shortfall | Undefined if ES is not negative; ignores cash as a baseline |
| `es` | Least severe tail | Ignores upside; tends to favor tight stops |

**Expected shortfall (ES)** is the mean of the worst 5% of final returns. It
is not a maximum loss, and it ignores losses along the way; see the maximum
drawdown columns (close-to-close). Hold wins exact ties.

Choose the objective and any limit **before** looking at results.

## 7. Result files and column definitions

Files are written to `results/` (or `--output-dir`). `qqq` is the ticker prefix.

**Units.** In CSV files, returns, probabilities, rates and stops are
fractions: `0.10` = 10%. A difference of `0.005` is 0.5 percentage points.
Log growth is a natural log over the whole horizon H (annual only when
H = 252). Losses and drawdowns are negative. A blank means undefined or not
applicable. "Trade" means one historical entry or one simulated path.

| File | One row per | Written by |
|---|---|---|
| `qqq_recovery_summary.csv` | scenario, stop, W1, W2 | `recovery` |
| `qqq_stop_vs_hold.csv` | scenario, seed, stop | `compare` |
| `qqq_best_stop.csv` | scenario, seed | `compare` |
| `qqq_by_period.csv` | historical period, stop | `compare` with `historical` and `--periods` above 1 |
| `qqq_walk_forward.csv` | scenario, test year | `walkforward` |
| `qqq_walk_forward_summary.csv` | scenario, policy | `walkforward` |
| `qqq_run_config.json` | run | every run |
| `qqq_prices.csv` | trading day | downloads only |
| `charts/qqq_decision_table.csv` | stop | `plot_results.py` |

The script does not delete result CSVs from earlier runs. Use a new
`--output-dir` for each configuration.

### What is ES?

**ES (expected shortfall) is the average return of the worst 5% of trades.**
It shows how bad things were, on average, among the worst outcomes.

**Example:** with 1,000 trades, average the 50 with the lowest returns. If
that average is -0.35, **ES = -35%**.

**How to read it**

* **Higher is better:** -20% is less severe than -35%.
* **It isn't the maximum loss:** some trades can lose much more.
  `WorstReturn` shows the worst one in the sample.
* **It differs from `P05Return`:** that marks roughly where the worst 5%
  begins, while ES averages the returns in that group.
* **It uses final returns:** it doesn't capture losses during a trade that
  later recovered. For those, see `MedianMaxDrawdown` and `P05MaxDrawdown`,
  which use closing prices.
* **It can be positive:** this happens if the worst 5% of trades made money on
  average.

**Related columns**

| Column | Meaning |
|---|---|
| `ES` | Average return of the worst group, for that stop or for Hold |
| `DiffESVsHold` | The stop's ES minus Hold's. **Positive means the stop did better** |
| `ReturnOverES` | Average return divided by the size of ES. Higher means more return relative to this large-loss measure; blank if ES is zero or positive |
| `BalancedESGain` | Decision table: ES improvement over Hold, averaged within each method type and then across types |
| `WorstES` | Decision table: lowest ES across scenarios, after averaging seeds |

If a stop has an ES of -22% and Hold has -35%, `DiffESVsHold` is `0.13`, a
**13 percentage point improvement**. CSV files store these as decimals, so
-35% appears as `-0.35`.

Change the group size with `--es-alpha`; `0.10` averages the worst 10%. If this
isn't a whole number of trades, the script rounds the count up. The 5% in
these examples is the default.

**A better ES doesn't make a stop better overall.** A tight stop can reduce
large losses but also lower average returns or growth. Compare ES with
`MeanLogGrowth`, and check how both differ from Hold.

### Shared columns
| Column | Definition |
|---|---|
| `Ticker` | Value of `--ticker` |
| `Scenario` | `historical`, `fhs`, or method and block label, e.g. `stationary_L63`, `circular_Lauto` |
| `Seed` | Random seed. For historical rows it controls the monthly entry sample; with all entries it has no effect on trades |
| `StopPct` | Drop below entry that triggers the stop. In comparison files, `0` means Hold |

### `qqq_recovery_summary.csv`
Uses the first seed only.

| Column | Definition |
|---|---|
| `W1` | Window in which the drop can first occur: entry day plus 21/63/126/252/504 days for 1M to 24M |
| `W2` | Days allowed for recovery after the hit day, same lengths |
| `Entries` | Trades with enough data for this row |
| `StopHits` | Entries whose Low reached the stop level within W1 |
| `StopHitProbability` | `StopHits / Entries` |
| `MedianDaysToStop` | Median trading days from entry to first hit; 0 = entry day |
| `Recovered` | Hits whose High reached the entry price within the W2 days after the hit day |
| `NotRecovered` | `StopHits - Recovered` |
| `RecoveryProbability` | `Recovered / StopHits`; conditional on a hit |
| `NonRecoveryProbability` | `NotRecovered / StopHits` |
| `MedianDaysToRecovery` | Median days from **hit day** to recovery, recovered cases only |
| `MedianWorstDropNonRecovered` | Non-recovered cases: median of the lowest Low from hit day through W2 end, relative to entry |
| `P10WorstDropNonRecovered` | 10th percentile of those worst drops (the more severe end) |
| `MedianAdditionalDropAfterStop` | Median of worst drop plus `StopPct`. `-0.12` means 12 points of entry price below the stop level |
| `MedianEndReturnNonRecovered` | Median Close return versus entry on the last day of W2, non-recovered cases |

### `qqq_stop_vs_hold.csv`
Return = final wealth / 1 - 1. Log growth = ln(final wealth).

| Column | Definition |
|---|---|
| `Trades` | Number of trades in the row |
| `TriggerRate` | Share of trades stopped within H; 0 for Hold |
| `WhipsawRate` | Among stopped trades, share where Hold's final price is above the stop fill (before slippage and interest). Blank with no triggers |
| `MeanReturn` | Average return |
| `MedianReturn` | Middle return |
| `MeanLogGrowth` | Average log growth |
| `MedianLogGrowth` | Median log growth |
| `ProbLoss` | Share of trades with return below 0 |
| `P05Return` | 5th percentile return (always 5%) |
| `ES` | Mean of the worst `--es-alpha` share of returns (count rounded up). Closer to or above 0 is better |
| `WorstReturn` | Lowest final return in this sample; not a worst possible loss |
| `ReturnOverES` | `MeanReturn / -ES`; blank when ES is 0 or positive |
| `MedianMaxDrawdown` | Median of each trade's largest peak-to-trough fall, using Closes, entry price and stop exit value |
| `P05MaxDrawdown` | 5th percentile of those drawdowns (severe end) |
| `ProbBeatsHold` | Share of trades ending with more wealth than Hold on the same path |
| `ProbWorseThanHold` | Share ending with less. The remainder tie |
| `DiffMeanLogVsHold` | `MeanLogGrowth` minus Hold's. Positive = stop better |
| `DiffESVsHold` | `ES` minus Hold's. Positive = less severe tail |
| `DiffCILow`, `DiffCIHigh` | Historical only: approximate 95% interval for `DiffMeanLogVsHold`, from resampling blocks of entry months (at least 12 months and twice H). Blank if history is too short and on the Hold row. Likely too narrow with few independent periods; not adjusted for testing many stops |
| `SimulationSE` | Simulated only: Monte Carlo standard error of `DiffMeanLogVsHold` for one seed. **It does not measure uncertainty from limited history or model error.** More paths shrink it |

### `qqq_best_stop.csv`
In-sample winners. Not evidence of out-of-sample performance.

| Column | Definition |
|---|---|
| `Objective` | Value of `--objective` |
| `MinES` | Value of `--min-es`; blank if none |
| `BestStopPct` | Highest-scoring stop; `0` = Hold; blank if no row qualifies |
| `Result` | The same in words |

### `qqq_by_period.csv`
Historical trades split in time order into `--periods` groups of nearly equal
trade count.

| Column | Definition |
|---|---|
| `FirstEntry`, `LastEntry` | First and last entry dates in the period |
| All remaining metric columns | As in `stop_vs_hold`, calculated for that period. There are no interval or SE columns |

Neighboring periods share future prices, because trades last H days.

### `qqq_walk_forward.csv`

| Column | Definition |
|---|---|
| `TestYear` | Calendar year of the test entries |
| `TrainEnd` | Last price date available for choosing the stop |
| `ChosenStopPct` | Stop applied; `0` = Hold |
| `ConstraintMet` | False if no row met `--min-es`, so Hold was applied as a fallback. Also False if no row had a defined score |
| `Block` | Block length used; blank for historical and FHS |
| `TestTrades` | Test entries with a complete horizon |
| `MeanReturn` | Mean test return using the chosen policy |
| `HoldMeanReturn` | Mean test return of Hold, same entries |
| `MeanLogGrowth` | Mean test log growth, chosen policy |
| `HoldMeanLogGrowth` | Same for Hold |

### `qqq_walk_forward_summary.csv`
All test years pooled. This is not a portfolio equity curve.

| Column | Definition |
|---|---|
| `Policy` | `WalkForwardStop` (yearly choice, which may be Hold) or `Hold` |
| `Trades` | Pooled test trades; they overlap |
| `MeanReturn`, `MedianReturn` | Mean and median return |
| `MeanLogGrowth`, `MedianLogGrowth` | Mean and median log growth |
| `ES` | Expected shortfall of pooled returns |
| `WorstReturn` | Lowest test return |

### `charts/qqq_decision_table.csv`
Summarizes `stop_vs_hold`. Seeds are averaged first. **Gain** means the
objective value for a stop minus Hold's value in the same scenario (positive =
better; for `es`, less severe tail). For `return_over_es` it is in ratio units.

**Family-balanced** means: average the scenarios within each family
(historical, stationary, circular, FHS), then average the families equally.
This stops four block lengths counting four times. It is a reporting choice,
not a statistical estimate.

| Column | Definition |
|---|---|
| `StopPct`, `Policy` | Stop as a fraction, and as a label |
| `Scenarios` | Scenarios containing the stop |
| `ScoredScenarios` | Scenarios where the gain is defined |
| `BalancedGain` | Family-balanced average gain |
| `MedianGain` | Median gain across scenarios (not balanced) |
| `WorstGain` | Smallest gain in any scored scenario |
| `ShareBeatsHold` | Share of scored scenarios with gain above 0; Hold is 0. Unlike `ProbBeatsHold`, it counts scenarios, not trades |
| `MaxRegret` | Largest shortfall from the best row (including Hold) in any scenario; lower is better. Blank if the score is undefined somewhere |
| `BalancedESGain` | Family-balanced ES minus Hold's ES |
| `ShareBetterES` | Share of scenarios with a less severe ES than Hold |
| `WorstES` | Most severe seed-averaged ES in any scenario |
| `BalancedTriggerRate` | Family-balanced trigger rate |
| `ShareMeetsMinES` | Only with `--min-es`: share of scenarios whose seed-averaged ES meets it |

Check `ScoredScenarios` equals `Scenarios` before comparing rows.

### `qqq_run_config.json` and `qqq_prices.csv`
The JSON holds every command-line option (same names, with underscores), plus
`data_first`, `data_last`, `rows`, `block_lengths_used` (full-sample compare
or recovery runs; walk-forward blocks are in its CSV), and `numpy`, `pandas`
and `python` versions. The price file has `Date, Open, High, Low, Close`
(adjusted).

### Should whipsaw rate be used to select a stop?

**No, not on its own.** `WhipsawRate` helps explain why a stop helped or hurt,
but it shouldn't pick the stop.

It is the share of **stopped trades** where Hold's final price was higher than
the stop's sale price. Hold may still have lost money.

**Why it can mislead**

* **It ignores size:** missing a 1% rebound and a 40% rebound each count as one
  whipsaw.
* **It ignores how often stops trigger:** 40% of 800 stopped trades is 320
  whipsaws, while 60% of 50 is only 30.
* **Small samples are unreliable:** deep stops may trigger rarely, and many
  historical triggers can come from the same few selloffs.
* **It excludes slippage and interest:** it compares Hold's final price with
  the sale price before either is applied.
* **No triggers means no rate:** it is blank, not a perfect score.

**Example:** you buy at 100 with a 20% stop. Ten trades are sold at 80:

| Outcome if held | Trades | Hold compared with Stop, per trade |
|---|---|---|
| Ends at 85 | 6 | 5 points better |
| Ends at 50 | 4 | 30 points worse |

The whipsaw rate is **60%**, which looks bad. But across these ten trades, Stop
ends **9 points better per trade on average**: (4 x 30 - 6 x 5) / 10. This
ignores slippage and interest. The reverse can also happen: a low whipsaw rate
can hide a few very large missed rebounds.

**What to use instead**

| Question | Look at |
|---|---|
| Does the stop improve growth? | `DiffMeanLogVsHold` |
| Does it reduce large losses? | `DiffESVsHold` |
| Is the result consistent? | Decision table, period results and walk-forward |
| Could it be noise? | `DiffCILow` and `DiffCIHigh` (historical mean log growth); variation across seeds (simulated) |
| Why did it help or hurt? | `TriggerRate` and `WhipsawRate` together |

Choose your goal first, improving growth or limiting large losses, then follow
section 9. If two stops perform similarly, whipsaw rate can help explain how
they behave, but **a lower rate alone doesn't make one better**.

## 8. Charts

Run the analysis first, then:

```bash
python plot_results.py
python plot_results.py --results-dir results_base --ticker QQQ
```

PNG files and the decision table go in `<results-dir>/charts/`. Re-running
replaces that ticker's charts. Charts for analyses not in the latest run are
skipped. `--objective` re-scores charts 1, 2 and the table without rerunning;
it does not change charts 3-8 (chart 5 stays in mean log growth). Choose the
objective in advance, not by trying each until you like the answer.

Charts 1-6 average seeds, so check seed-to-seed variation in the CSV. In
charts, values are percentages and differences are percentage points.

| File (after `qqq_`) | Question | Favorable sign | Warning sign |
|---|---|---|---|
| `1_scorecard.png` | Which stops do well overall? | Neighboring stops with positive balanced gain, high share, small worst loss | One isolated positive bar; tiny gains |
| `2_robustness_heatmap.png` | Does it hold across methods and block lengths? | A blue vertical band | Color flips by method or L |
| `3_growth_vs_tail_risk.png` | How much growth is traded for tail protection? | Up and right of Hold | Safety bought with large growth loss |
| `4_historical_uncertainty.png` | Is the historical gain distinguishable from zero? | Interval above zero | Interval spans zero |
| `5_period_stability.png` | Does one period drive it? | Similar sign in each period | One period supplies all the gain |
| `6_trigger_whipsaw.png` | Why do stops help or hurt? | Explains the pattern | Very few triggers |
| `7_walk_forward.png` | Did choosing from past data work later? | Stable choices; gains over Hold | Choices jump; no gain over Hold |
| `8_recovery_context.png` | How often do drops recover? | Context only | Do not choose a stop from it |

Colors in charts 1, 2 and 5 are relative to Hold: blue better, red worse.
Heatmap color intensity is scaled to each chart, so read the numbers; a deep
color in one chart need not be a large effect.

## 9. Step-by-step review and decision

The thresholds below are **suggested defaults based on judgment, not
statistical tests**. Change them if you like, but do it before seeing results.
Passing them doesn't prove a stop will work in future.

### Step 0. Write down your rules
| Item | Example | Yours |
|---|---|---|
| Purpose: **A** improve growth, or **B** limit tail loss | A | |
| Horizon H | 252 | |
| Objective | `mean_log` | |
| ES limit (required for B) | -0.30 | |
| Stop grid | 5% to 50% by 5% | |
| Slippage, cash rate | your estimates | |
| M: smallest gain worth acting on, in the objective's units | 0.005 (0.5 points of log growth per 252 days) | |
| For B: largest growth cost you accept | 0.01 | |

M = 0.005 only makes sense for log-growth objectives at H = 252. For another
objective, set M in its units. Purpose B should normally keep `mean_log` as
the objective, to measure the growth cost.

Optional final holdout: add `--end YYYY-MM-DD` to **every** development run,
reserving recent years. Look at them once, after deciding. With H = 252 a few
years hold very few independent trades, so this check is weak. If you have
already studied results for those years, they aren't a clean holdout.

### Step 1. Run the base analysis
```bash
python test_drawdown_dynamics.py
python drawdown-dynamics.py --method all --analysis all \
    --entries-per-month all --n-seeds 5 --output-dir results_base
python plot_results.py --results-dir results_base
```
Add your Step 0 settings (`--horizon`, `--cash-rate`, `--slippage`,
`--objective`, `--min-es`) to the analysis command.

For later runs with a downloaded series, add
`--csv results_base/qqq_prices.csv` so that all runs use identical data.

### Step 2. Check that the run is trustworthy
* Read all warnings: repaired bars, data gaps, GARCH convergence and
  persistence (including walk-forward refits).
* In the JSON, confirm date range, row count and settings.
* Investigate extreme daily moves in the price file.
* Compare Hold rows across scenarios. Large differences in mean log growth or
  ES mean the scenarios describe different markets. Find out why.
* Note `TriggerRate x Trades`. A result driven by a handful of triggers, or
  historical triggers from only one or two selloffs, is weak.

**Stop here if data or model problems are unresolved.**

### Step 3. Check simulation noise
In `stop_vs_hold`, compare `DiffMeanLogVsHold` across seeds.
* Is its sign the same in most seeds for each simulated scenario?
* Is the gain at least about 2 x `SimulationSE`?
* Do seed-to-seed changes stay small relative to M?

If not, rerun with `--paths 20000`. `SimulationSE` applies to mean log growth
only; for other objectives, use seed-to-seed variation. This step says nothing
about whether history will repeat.

### Step 4. Apply gates (decision table, charts 1 and 2)
**Purpose A.** A stop is a candidate if all are true:

| Gate | Suggested rule |
|---|---|
| A1 Size | `BalancedGain` >= M |
| A2 Consistency | `ShareBeatsHold` >= 0.75 |
| A3 Downside | `WorstGain` >= -M |
| A4 Neighbors | At least one adjacent tested stop also passes A1-A3 |

**Purpose B.** A stop is a candidate if all are true:

| Gate | Suggested rule |
|---|---|
| B1 Limit | `ShareMeetsMinES` = 1 |
| B2 Cost | With `mean_log`, `WorstGain` >= -(accepted growth cost) |
| B3 Neighbors | At least one adjacent stop also passes B1-B2 |

In chart 2, also check whether results trend with block length L. If only long
(or only short) blocks favor a stop, the answer rests on an assumption about
time dependence that you need to justify.

**No candidate? The result is "no stop supported" for A, or "these stops
cannot meet both requirements" for B.** For B, Hold is not automatically
acceptable: if it breaks your limit, reconsider exposure.

### Step 5. Examine the trade-off (chart 3)
* For each candidate, write down growth gained or lost and ES improvement
  against Hold, scenario by scenario.
* Drop candidates dominated by another (worse on both measures) in most
  scenarios.
* A better ES is expected from a stop that sells into cash, so it is not by
  itself evidence of skill. **Compare with simply holding a smaller position**
  (the rest in cash) sized to a similar ES. This script doesn't calculate that
  benchmark; do it before adopting a stop for risk reduction.
* Check drawdown columns too. Final-return ES ignores losses along the way.

### Step 6. Check historical uncertainty (chart 4)
| Interval | Reading |
|---|---|
| Entirely above 0 | Supportive, though still likely too optimistic after testing many stops |
| Includes 0 | Inconclusive; expected with limited history |
| Entirely below 0 | Evidence against, for purpose A |

The interval covers mean log growth only.

### Step 7. Check period stability (chart 5)
Suggested rule for A: gain positive in at least 2 of 3 periods, with no period
worse than -2M. For B, check each period's `ES` in `by_period` against your
limit and its growth cost. This chart covers historical data only.

### Step 8. Check walk-forward (chart 7)
* Does the yearly choice stay inside or near your candidate range?
* Is pooled log growth (A) at least Hold's in most scenarios? For B, does
  pooled ES meet the limit at an acceptable growth cost?
* Check `ConstraintMet` and yearly results; one year may dominate.

This tests the **rule for choosing stops**, not a fixed candidate. Stable
choices alone prove nothing: they can come from the same large historical
episode staying in the training data.

### Step 9. Stress-test the candidates
Run each into its **own** output folder, then repeat Steps 4-8.

| Test | Option |
|---|---|
| Worse execution | `--slippage 0.002`, then `0.01` |
| Cash assumptions | `--cash-rate 0.0`, then a higher plausible rate |
| Horizon | `--horizon 126`, `--horizon 504` |
| Start date | `--start` (downloads), or a trimmed CSV |
| Current volatility | `--method fhs --fhs-start current --analysis compare` |
| Time dependence | `--block-lengths 1 auto` with both bootstraps |
| Finer grid near candidates | e.g. `--stops 0.26 0.28 0.30 0.32 0.34` |

A candidate should survive plausible changes. A different horizon is really a
different question, so a different answer there matters only if your holding
period is uncertain. Current-volatility results are conditional on one date.
A finer grid checks smoothness; picking its best point adds more overfitting.
Neighbor gates are easier to pass on a fine grid, so don't use it for Step 4.
`--block-lengths 1` is a reference without time dependence; don't mix it into
the main decision table.

### Step 10. Decide
1. Keep only candidates that passed the earlier steps.
2. If they form a continuous range, choose a value near its **middle**, not
   the highest score.
3. If separate ranges pass, treat the evidence as unstable.
4. For A, if evidence is mixed, **use no stop**.
5. For B, if no stop works reliably, don't force one; consider less exposure.
6. Then, if you reserved a holdout, look once. To run it, compare the chosen
   stop and Hold on entries after the cutoff (filter by entry date; the
   command line has no holdout option). If you change the choice afterwards,
   the holdout is used up.

Possible conclusions: **supported range**, **risk choice with a known cost**,
**inconclusive**, or **no stop**.

### Step 11. Record and revisit
| Field | Entry |
|---|---|
| Date, data range, code version | |
| Step 0 rules | |
| Candidate range and chosen value | |
| Balanced gain, worst gain, ES change | |
| Historical interval and period results | |
| Walk-forward result | |
| Stress tests that failed | |
| Conclusion and reason | |
| Next review date and what would change the decision | |

Review on a fixed schedule, such as yearly. Frequent re-optimizing invites
chasing noise. Before using real money, test the stop with your actual entry,
exit and re-entry rules.

### A warning from the test data
In a development run on synthetic prices with **independent daily returns and
positive drift**, no stop should improve expected log growth. Even so, in that
one sample, a 30% stop beat Hold by about 0.7 points in the historical
scenario (and in `circular_L252`), the historical walk-forward chose 30% in
most years, and later periods looked favorable. The family-balanced gain was
only about 0.06 points and fewer than half the scenarios beat Hold, so the
suggested gates rejected it. One example doesn't show how often the gates
make mistakes, but it shows how convincing noise can look.

## 10. General cautions

* Agreement among methods is not independent confirmation: all reuse the same
  history.
* Trying many stops, objectives, horizons and methods, then reporting the
  best, overfits. Walk-forward results also stop being out-of-sample if you
  keep changing settings after seeing them. This script does not correct for
  this (see Bailey et al.).
* QQQ's history from 1999 starts shortly before a very large technology
  selloff. Try other `--start` dates.
* Results apply to these entry, exit and no-re-entry rules. Your strategy may
  give a different answer.

## 11. Not included
Trailing stops, re-entry rules, volatility-scaled stops, strategy entry
signals, position sizing (including the smaller-position benchmark), historical cash rates, and formal multiple-testing
corrections.

## References
* Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal
  of the American Statistical Association*, 89, 1303-1313.
  [Record](https://gnosis.library.ucy.ac.cy/handle/7/57533)
* Politis, D. N., & Romano, J. P. (1992). A circular block-resampling
  procedure for stationary data. In LePage & Billard (eds.), *Exploring the
  Limits of Bootstrap*, Wiley, 263-270.
* Politis, D. N., & White, H. (2004). Automatic block-length selection for the
  dependent bootstrap. *Econometric Reviews*, 23(1), 53-70.
  [Record](https://econpapers.repec.org/RePEc:taf:emetrv:v:23:y:2004:i:1:p:53-70)
* Patton, A., Politis, D. N., & White, H. (2009). Correction to "Automatic
  block-length selection for the dependent bootstrap". *Econometric Reviews*,
  28(4), 372-375.
  [PDF](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf)
* Lahiri, S. N. (1999). Theoretical comparisons of block bootstrap methods.
  *Annals of Statistics*, 27(1), 386-404.
  [PDF](https://projecteuclid.org/journals/annals-of-statistics/volume-27/issue-1/Theoretical-comparisons-of-block-bootstrap-methods/10.1214/aos/1018031117.pdf)
* Nordman, D. J. (2009). A note on the stationary bootstrap's variance.
  *Annals of Statistics*, 37(1), 359-370.
  [arXiv](https://arxiv.org/abs/0903.0474)
* Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999). VaR without
  correlations for portfolios of derivative securities. *Journal of Futures
  Markets*, 19(5), 583-602.
  [Record](https://econpapers.repec.org/RePEc:wly:jfutmk:v:19:y:1999:i:5:p:583-602)
* Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
  between the expected value and the volatility of the nominal excess return
  on stocks. *Journal of Finance*, 48(5), 1779-1801.
* Kaminski, K. M., & Lo, A. W. (2014). When do stop-loss rules stop losses?
  *Journal of Financial Markets*, 18, 234-254.
  [Author copy](https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf)
* Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. The
  probability of backtest overfitting. *Journal of Computational Finance*.
  [SSRN](https://papers.ssrn.com/abstract=2326253)
* `arch` documentation:
  [optimal block length](https://bashtage.github.io/arch/doc/bootstrap/generated/arch.bootstrap.optimal_block_length.html)