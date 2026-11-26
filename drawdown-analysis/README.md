# Drawdown and Stop-Tolerance Analyzer

This project downloads daily market data from Yahoo Finance for **QQQ or another symbol**, measures historical drawdowns and entry-based adverse movement, and summarizes the results in percentage bins. Its practical purpose is to estimate how wide a stop must be to tolerate a chosen share of historical adverse movement.

The program creates:

- One Excel workbook containing all detailed and summary tables
- One four-panel PNG dashboard
- No collection of small CSV files

## Files

- `drawdown_stop_analysis.py`: executable analysis program
- `README.md`: methodology, configuration, outputs, and interpretation guide

## Important distinction

The program deliberately performs two separate analyses.

### 1. Peak-based market drawdowns

A market drawdown begins at a running closing-price peak, reaches a subsequent trough, and ends when the closing price regains or exceeds the old peak.

This answers:

> What peak-to-trough declines has this market historically experienced, and how long did they take to decline and recover?

### 2. Entry-based maximum adverse excursion (MAE)

For hypothetical daily, weekly, or monthly entries, the program measures the lowest subsequent daily low within a user-defined holding window relative to the entry close.

This answers:

> After an entry, how far did price move against the position during the holding window?

Entry-based MAE is more directly relevant to stop selection. A 15% market drawdown from a prior peak is not necessarily a 15% adverse move from a later entry.

## Installation

Use Python 3.10 or later. Install the required packages:

    pip install yfinance pandas numpy matplotlib openpyxl

Run the program from its directory:

    python drawdown_stop_analysis.py

By default, results are written to a `RESULTS` subfolder.

## User configuration

All user-editable settings are at the top of the Python file.

### Market data

| Setting | Default | Meaning |
|---|---:|---|
| `SYMBOL` | `QQQ` | Any Yahoo Finance symbol, such as `SPY`, `EFA`, `EEM`, or `^IXIC` |
| `START_DATE` | `1999-03-10` | First requested date |
| `END_DATE` | `None` | Latest available date, or a specified `YYYY-MM-DD` date |
| `INTERVAL` | `1d` | Daily data; this version is intentionally designed for daily bars |
| `AUTO_ADJUST` | `True` | Adjust OHLC values for splits and distributions |

Adjusted prices are generally preferable for long historical studies. If the purpose is to model actual stop orders against contemporaneous traded prices, consider rerunning with `AUTO_ADJUST = False` and comparing the result. Long histories containing splits can otherwise be misleading when unadjusted.

### Distribution bins

| Setting | Default | Meaning |
|---|---:|---|
| `BIN_WIDTH_PCT` | `5.0` | Produces bins such as 0-5%, 5-10%, and 10-15% |
| `MAX_BIN_PCT` | `None` | Automatically covers the observed maximum; set a number to create a final overflow bin |
| `MIN_EPISODE_DRAWDOWN_PCT` | `0.0` | Optional filter for very small peak-based episodes |

A 5% bin width is useful for high-level interpretation. For stop calibration, 1% or 2.5% bins provide more precision, but percentiles and the stop-survival table are usually better than relying only on a histogram.

### Entry-based MAE

| Setting | Default | Meaning |
|---|---:|---|
| `ANALYZE_ENTRY_MAE` | `True` | Enables hypothetical-entry analysis |
| `ENTRY_FREQUENCY` | `monthly` | First available bar of each day, week, or month |
| `MAX_HOLDING_BARS` | `504` | Maximum observation window, about two trading years |
| `RECOVERY_THRESHOLD_PCT` | `0.0` | Recovery level relative to entry; zero means return to entry price |
| `PROFIT_TARGET_PCT` | `None` | Optional target measured from entry |

Monthly entries reduce overlap and fit a one-to-two-year position-trading horizon. Daily entries provide a larger sample but are highly overlapping, so the apparent sample size should not be mistaken for independent evidence.

The entry price is the selected bar's closing price. MAE begins on the next bar and uses subsequent daily lows. This avoids claiming that a position entered at the close could have been stopped by the same day's earlier low.

### ATR and regimes

| Setting | Default | Meaning |
|---|---:|---|
| `ATR_PERIOD` | `20` | Wilder ATR lookback |
| `TREND_MA_PERIOD` | `200` | Trend regime moving average |
| `HIGH_VOL_QUANTILE` | `0.75` | High-volatility entries are at or above the 75th percentile of ATRP |
| `ATR_STOP_MULTIPLES` | `2` to `10` | ATR-normalized stop candidates |
| `CANDIDATE_STOPS_PCT` | `5` to `40` | Percentage stop candidates |
| `SURVIVAL_PERCENTILES` | 50 to 99 | MAE and drawdown percentile levels |

ATRP is calculated as:

    ATRP = ATR / Close * 100

ATR-normalized adverse movement is:

    MAE ATR units = MAE percentage / entry ATRP

Example: if ATRP at entry is 2% and MAE is 10%, the adverse movement equals approximately 5 ATR units.

## Peak-based drawdown logic

For each daily close:

    Drawdown = 1 - Close / RunningPeak

An episode contains:

- **Peak date and price:** latest closing high before price moves below it
- **Trough date and price:** lowest close before recovery
- **Recovery date:** first close at or above the prior peak
- **Drawdown percentage:** peak-to-trough loss
- **Decline bars:** peak to trough
- **Recovery bars:** trough to recovery
- **Total bars:** peak to recovery
- **Calendar-day durations:** calendar equivalents of the bar counts
- **Drawdown ATR units:** drawdown percentage divided by ATRP at the peak

If the final episode has not recovered by the end of the data, it is retained as an **open or censored episode**. Its recovery fields remain blank. This prevents the current drawdown from disappearing merely because recovery is not yet observable.

Equal or higher closes reset the peak before a drawdown starts. A full recovery completes the old episode and establishes a new peak.

## Entry-based MAE logic

For each selected entry:

1. Enter at that bar's close.
2. Examine up to `MAX_HOLDING_BARS` subsequent bars.
3. Find the minimum daily low in that window.
4. Calculate MAE as the percentage fall from entry close to that low.
5. Find the first subsequent daily high that reaches the configured recovery level.
6. Optionally find the first daily high reaching the profit target.
7. Record entry-time ATRP, trend regime, volatility regime, terminal return, and durations.

Using the daily low makes the stop test conservative. However, daily OHLC data cannot establish the intraday sequence when a low breaches a stop and the high reaches recovery or target on the same bar. The workbook therefore supports historical tolerance analysis, but exact execution studies should use intraday data and explicit order rules.

## Workbook sheets

### Overview

A compact audit sheet with data dates, episode counts, current open drawdown, MAE percentiles, and recovery rate.

### Drawdown Episodes

One row per peak-based episode, including magnitude, dates, durations, ATR normalization, regimes, recovery status, and bin.

### Drawdown Distribution

One row per magnitude bin. It includes:

- Episode count and percentage
- Cumulative episode percentage
- Recovered episode count and recovery rate
- Median decline, recovery, and total duration
- Maximum total duration
- Median ATR-normalized drawdown

### Entry MAE

One row per hypothetical entry. It includes:

- Entry and window dates
- Entry price
- MAE, trough date, and bars to trough
- Recovery result and bars to recovery
- Optional target result
- Terminal return
- Entry ATRP and MAE in ATR units
- Trend and volatility regimes
- Whether the full requested window was observable

Rows near the end of the dataset may have shorter observation periods. Use `FullWindowObserved` to exclude them when a strictly comparable horizon is required.

### MAE Distribution

Entry observations summarized by percentage bin, including frequency, recovery rate, duration, terminal return, and ATR-normalized MAE.

### Stop Survival

For every configured percentage stop:

- Number and rate of entries whose MAE equaled or exceeded the stop
- Historical survival rate
- Median time to the worst point among breached entries
- Recovery rate among breached entries
- Median recovery time and terminal return among breached entries

A breach rule of `MAE >= stop` is intentionally conservative. Touching the stop level counts as a stop event.

### ATR Stop Survival

The same survival concept expressed in ATR units. ATR normalization is useful because a fixed 10% stop represents different levels of market stress in quiet and volatile regimes.

### Percentiles

Reports peak-based drawdown, entry MAE, and entry MAE ATR units at the configured percentiles.

If P90 entry MAE is 18%, then 90% of historical sampled entries had MAE at or below approximately 18% during the configured window. It does **not** mean an 18% stop will preserve 90% of future trades.

### Regime Analysis

Compares entries across:

- Above versus below the 200-day moving average
- High versus normal ATRP

It reports median, P90, and P95 MAE, recovery behavior, and terminal return. This shows whether one unconditional stop hides materially different market states.

## Dashboard

The PNG contains four panels:

1. Peak-based drawdown episode distribution
2. Entry-based MAE distribution
3. Historical survival rate by percentage stop
4. P90 and P95 MAE by trend and volatility regime

## How to interpret results for stop selection

Use the outputs in the following order.

### Step 1: Choose the relevant holding window and entry sampling

For a one-to-two-year QQQ position, `MAX_HOLDING_BARS = 252` to `504` is reasonable. Monthly entry sampling is a practical baseline. Rerun weekly entries as a robustness check.

### Step 2: Inspect entry MAE percentiles

The P90 and P95 MAE values translate “most drawdowns” into explicit historical coverage. Treat them as candidate tolerance levels, not automatic recommendations.

### Step 3: Inspect the stop-survival curve

Look for the point where widening the stop produces diminishing improvement. For example, moving from 10% to 15% may materially raise historical survival, while moving from 25% to 30% may add little. The exact values must come from the output, not assumption.

### Step 4: Compare regimes

If below-MA/high-ATRP entries have much larger P90 or P95 MAE, one unconditional stop may be inappropriate. Possible responses include:

- Avoiding entries in that regime
- Reducing position size while widening the stop
- Using an ATR-based stop
- Using a trend exit instead of allowing an extremely wide fixed stop

### Step 5: Compare fixed-percent and ATR results

A candidate stop should be stated both ways, for example:

> 18% at entry, equal to 6.2 entry ATR units.

If the implied ATR multiple varies strongly across entries, a fixed-percentage stop is not controlling risk consistently.

### Step 6: Test strategy expectancy separately

A stop that survives 95% of entries may still be poor because it creates infrequent but very large losses. Historical tolerance is a necessary-condition test, not a sufficient-condition test.

The next strategy-level test should include:

- Actual entry signal
- Position sizing
- Stop execution and gap assumptions
- Re-entry rule
- Profit-taking or trend exit
- Commission and slippage
- CAGR, maximum portfolio drawdown, expectancy, win rate, and time in market

## Recommended robustness checks

Run a small grid rather than trusting one configuration:

- Entry frequency: monthly and weekly
- Holding window: 252, 378, and 504 bars
- Bin width: 2.5% and 5%
- Price basis: adjusted and unadjusted
- Trend regime: 200-day MA
- ATR period: 14 and 20 days
- Symbols: QQQ and SPY, then an out-of-sample market such as EFA or EEM

A robust stop region should not depend on one exact bin edge, one sampling date, or one symbol.

## Important limitations

1. **Yahoo Finance is a convenient research source, not an exchange-grade institutional feed.** Check suspicious bars against another source before making a consequential conclusion.
2. **Overlapping entries are dependent.** Daily and weekly observations inflate the apparent sample size.
3. **Daily bars do not show intraday order.** Gap-through stops and same-bar stop/target conflicts require explicit assumptions or intraday data.
4. **No strategy edge is tested.** Hypothetical periodic entries are a market-behavior probe, not a trading strategy.
5. **The final holding windows may be censored.** Filter `FullWindowObserved = True` for equal-horizon comparisons.
6. **Recovery is not the same as good capital use.** A security may eventually recover after a long, deep decline while tying up capital.
7. **Historical coverage is not future probability.** Structural market changes and unprecedented events remain possible.
8. **A wider stop requires smaller position size** if portfolio risk per trade is held constant.

## Practical position-sizing connection

Once a stop is selected, position size should be linked to the permitted portfolio loss:

    Position weight = Allowed portfolio loss / Stop percentage

Example: risking 1% of portfolio capital with a 20% stop implies a 5% position weight before gap and slippage allowances:

    1% / 20% = 5%

For a broad index ETF such as QQQ, investors may intentionally use different portfolio construction rules, but the arithmetic makes the cost of a wide stop explicit.

## Suggested interpretation standard

Do not report only “the best stop.” Report a range and the trade-off:

- **Minimum practical tolerance:** stop width near P80 MAE
- **High-coverage tolerance:** stop width near P90 MAE
- **Exceptional-event tolerance:** stop width near P95 MAE
- Regime-specific values for above/below MA and normal/high ATRP
- Corresponding ATR multiples
- Survival-rate gain from each additional 5 percentage points of stop width

The final choice should be the narrowest stop region that is robust across reasonable windows and regimes **and** improves the complete strategy after position sizing and execution costs.
