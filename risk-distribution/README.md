# Holding-Period Risk Analyzer

`risk_distribution.py` takes the daily price history of one investment and writes a single Excel workbook describing its **historical holding-period risk**.

## Contents

1. [Objective](#1-objective)
2. [Quick start](#2-quick-start)
3. [Methodology](#3-methodology)
4. [Workbook guide](#4-workbook-guide)
5. [Using the results](#5-using-the-results)
6. [Settings](#6-settings)
7. [Editing and recalculation](#7-editing-and-recalculation)
8. [Limitations](#8-limitations)
9. [Troubleshooting](#9-troubleshooting)
10. [Testing status](#10-testing-status)
11. [Stop testing lab](#11-stop-testing-lab)

---

## 1. Objective

To quantify the risk of holding an investment for a set period, using every available window in its price history. The program helps answer:

- How often did a holding period end in a loss, and how large was it?
- How bad were the worst outcomes?
- How far did the price fall **during** the holding period?
- When did the worst periods occur?
- How would a fixed or trailing stop have changed the 1-year results?

The results describe the past. They are **not forecasts**, and the program does not recommend an investment or a stop.

Holding periods by default: **1M, 3M, 6M, 1Y, 3Y, 5Y, 10Y**.

---

## 2. Quick start

### Install

You need Python 3 and these packages:

```bash
pip install -U yfinance pandas openpyxl
```

Keep **`risk_distribution.py` and `stop_lab.py` in the same folder**.

NumPy is installed with pandas.

### Run

Open a terminal in the folder that contains the script.

| Task | Command |
|---|---|
| Default ticker (SPY) | `python risk_distribution.py` |
| Another ticker | `python risk_distribution.py --ticker QQQ` |
| An index | `python risk_distribution.py --ticker "^GSPC"` |
| Your own price file | `python risk_distribution.py --csv prices.csv` |
| Price file with a chart label | `python risk_distribution.py --csv prices.csv --ticker "My Fund"` |
| Run the stop testing lab | `python risk_distribution.py --ticker SPY --stop-lab` |
| Skip the stop testing lab | `python risk_distribution.py --ticker SPY --no-stop-lab` |
| Set your loss limit | `python risk_distribution.py --ticker SPY --max-cvar 0.12` |
| Faster trial run | `python risk_distribution.py --ticker SPY --bootstrap 100` |
| Skip the bootstrap | `python risk_distribution.py --ticker SPY --bootstrap 0` |
| Entries above the 200-day MA | `python risk_distribution.py --ticker SPY --ma-filter above --ma-period 200` |
| Entries at or below the 100-day MA | `python risk_distribution.py --ticker SPY --ma-filter below --ma-period 100` |
| Any entry day | `python risk_distribution.py --ticker SPY --ma-filter all` |
| Choose the output name | `python risk_distribution.py --ticker SPY --out spy_risk.xlsx` |

Keep the quotes around symbols that contain `^`, and around labels that contain spaces.

### Output

Workbooks are saved in a `results` folder **next to the script**, even if you run it from another folder. The script creates the folder if needed.

```text
your-folder/
├── risk_distribution.py
├── stop_lab.py
├── README.md
└── results/
    └── risk_distribution_SPY.xlsx
```

Without `--out`, the file is named `risk_distribution_<label>.xlsx`, for example `risk_distribution_SPY.xlsx`. With `--out`, a file name alone (`spy_risk.xlsx`) also goes into `results`. A name that includes another folder (`reports/spy_risk.xlsx`) is saved there instead, relative to the folder you run the command from. A `--csv` path is also relative to that folder.

 Symbols other than letters, numbers, `-` and `_` are removed from the name. Filtered runs add the filter to the default name, for example `risk_distribution_SPY_above200MA.xlsx` or `risk_distribution_SPY_atbelow200MA.xlsx`. A custom `--out` name is used as given. A file with the same name is **overwritten**, so rename any copy you have edited.

### CSV format

The file needs a `Date` column and an `Adj Close` or `Close` column. If both price columns exist, `Adj Close` is used. Column names are case-sensitive.

```csv
Date,Adj Close
2020-01-02,100.00
2020-01-03,101.25
```

The two rows only show the layout. A real file needs at least 300 usable **daily** prices. Use unambiguous dates (`YYYY-MM-DD`) and plain numbers without currency signs or thousands separators.

---

## 3. Methodology

### 3.1 Price data

| Item | Treatment |
|---|---|
| Download source | Yahoo Finance through the `yfinance` package, maximum available daily history |
| Price used | Adjusted close (`auto_adjust=True`), which accounts for splits and dividends where Yahoo's data does |
| Price indexes | An index such as `^GSPC` excludes dividends, so do not compare it directly with a dividend-adjusted fund |
| Today's price | Excluded from downloads, by your computer's date. CSV files are used as supplied |
| Cleaning | Prices sorted by date. Blank, non-numeric and non-positive prices are removed. For duplicate dates, one row is kept, so remove duplicates from a CSV yourself |

Missing trading days are not detected or filled. Check your data for gaps and unusual jumps.

### 3.2 Windows

Each window has a fixed **calendar length**. Its end date moves forward **one trading day at a time**.

1. Choose an end date.
2. Go back the holding period, in calendar months, to find the target start date.
3. If there is no price on that date, use the last earlier date that has one.
4. Calculate the return.
5. Move to the next end date.

No window is calculated if the history does not reach the target start date. Because windows are set by end date, some start dates are used more than once and others not at all. At a shorter month-end, the target moves to that month's last day.

A holding period is left out if it has fewer than 60 windows.

### 3.3 Returns

| Measure | Formula |
|---|---|
| Holding-period return | `end price / start price - 1` |
| Annualized return (`m` months, 12 or more) | `(1 + return) ^ (12 / m) - 1` |

Returns are **cumulative** unless labelled annualized. Periods under 1 year are not annualized, because that would assume the short-period result repeats all year.

### 3.4 Distribution

All windows of the same length, across the full history, form one distribution. Each window has equal weight.

- **Percentiles** use Excel's `PERCENTILE`, which interpolates between observations.
- **Standard deviation** is the sample standard deviation (`STDEV`) of holding-period returns. It is not annualized.
- **Skew** below zero indicates a longer left (loss-side) tail by this measure.
- **Excess kurtosis** above zero indicates heavier tails or a sharper peak relative to a normal distribution. Read it alongside the actual tail results.
- **Historical frequency** is the number of windows meeting a condition divided by all windows. The workbook calls this "chance".

### 3.5 Tail risk

| Measure | Meaning |
|---|---|
| 95% VaR | The loss at the 5th-percentile return |
| 95% CVaR | The average of the worst 5% of windows, shown as a loss |
| 99% VaR / CVaR | The same measures for the worst 1% |

For CVaR, the number of tail windows is rounded up, and returns tied at the cutoff are included.

These figures come directly from windows of the stated length. No normal distribution is assumed, and daily risk is not scaled to longer periods. On Risk Summary, losses are positive numbers, and a negative loss (a gain) is shown as 0%. VaR is **not** the maximum possible loss.

### 3.6 Risk during the window

Both measures use daily closing prices, from the start date to the end date.

| Measure | Formula | Question answered |
|---|---|---|
| Lowest vs entry | `lowest close / start price - 1` | How far below my purchase price did it go? |
| Maximum drawdown | Lowest value of `close / highest close so far - 1` | What was the largest fall from a high? |

"Highest close so far" restarts at the beginning of each window. Example with made-up prices **100 → 120 → 90 → 110**: ending return **+10%**, lowest vs entry **-10%**, maximum drawdown **-25%**.

### 3.7 Overlap

Neighboring windows share most of their prices, so one market crash can appear in many of them. **Thousands of windows are not thousands of independent observations.**

**Approx. non-overlapping periods** is the time between the first and last window end dates, in years (days / 365.25), divided by the holding period in years. It is a rough guide to how much separate history there is. It is not a statistical effective sample size.

### 3.8 Stop analysis

By default, this applies to 1-year windows. Each window is calculated with and without a stop.

| Item | Rule |
|---|---|
| Fixed stop | Triggers when the close is the chosen distance or more below the entry price |
| Trailing stop | Triggers when the close is the chosen distance or more below the highest close since entry |
| Exit price | The **first close** that meets the trigger, which can be well past the stop level |
| After exit | Cash earning 0% until the window ends. No re-entry |
| No trigger | Held to the end of the window |
| Default distances | Automatic, from the ladder 1%, 2%, 3%, 4%, 5%, 7.5%, 10%, 12.5%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 70%. **Minimum:** first level at or above 2 x daily volatility (same for fixed and trailing). **Maximum:** first level at or above the depth that about 5% of windows reached (separate for fixed and trailing). At least 3 levels are kept. Both use the tested history, so they include hindsight |

Costs, taxes, extra slippage and intraday triggers are not included.

### 3.9 Moving-average entry filter

Controls which windows are tested, based on the price **on the day the window starts**.

| Mode | A window is tested if |
|---|---|
| `all` | Any start day (no filter) |
| `above` | Start-day close is **above** its moving average |
| `below` | Start-day close is **at or below** its moving average |

- The average is a **simple** moving average of the last `MA_PERIOD` **trading days**, including the start day's close. It uses no later prices. The test assumes you can still enter at that same close. In practice, a signal confirmed at the close may only be traded later.
- The condition is checked **only at entry**. The price may cross the average during the window. Crossing does not end a window or trigger a stop.
- The first `MA_PERIOD - 1` days have no full average, so they are excluded in `above` and `below`. These two modes together cover every window that starts after that point.
- The filter applies to all holding periods and to Risk Summary, Distribution, Worst Periods, Stop Analysis and the stop testing lab. "Full history" and "all windows" then mean all **qualifying** windows.
- A holding period needs at least 60 qualifying windows. The stop lab needs at least 750 after its volatility warm-up. Otherwise they are left out. With fewer windows, some decades may not be eligible, and the walk-forward test may have fewer years or not run.
- **Approx. non-overlapping periods** becomes a count of qualifying windows that do not overlap, because dividing the date span by the holding period would overstate the evidence.
- **Bootstrap:** the average is recalculated on each resampled history and only matching start days are tested. A history with fewer than 20 matching starts is redrawn, and the number of redraws is printed. This favors histories with enough matching days. If redraws become too frequent, the bootstrap is skipped. Because it joins blocks of returns, it can break trends, so be especially cautious with filtered bootstrap results.
- **Crisis tests** still identify declines from the full price history.
- The **Data** tab keeps all prices. Non-qualifying windows are blank. Two columns at the far right show the average (a formula) and whether each day qualifies. Which windows are included is fixed when the script runs, so re-run after changing prices or the period.

Differences between `above` and `below` are historical, can come from only a few episodes, and do not show that the average caused them. Trying many periods and keeping the best one can fit chance patterns. Choose the period before looking at results, or check that nearby periods agree.

---

## 4. Workbook guide

| Tab | Purpose |
|---|---|
| Risk Summary | Main risk results for all holding periods |
| Stop Analysis | Comparison of fixed and trailing stops |
| Distribution | Histograms and cumulative frequencies |
| Worst Periods | Worst historical windows and their dates |
| Data | Prices and results for every window |
| Stop Data | Results of every stop for every window |
| Notes | Data source, run date and assumptions |
| Stop Scorecard, Stop Stress Tests, Stop Periods & Crises, Stop Bootstrap, Stop Walk Forward | Stop robustness tests, see [section 11](#11-stop-testing-lab) |

Stop Analysis and Stop Data appear only when the stop holding period is included and has enough history.

### 4.1 Risk Summary

One column per holding period. **Start here.**

| Section | What it shows | How to read it |
|---|---|---|
| Sample | Window count and approx. non-overlapping periods | Low non-overlapping counts mean limited evidence |
| Distribution of holding-period return | Mean, median, standard deviation, skew, excess kurtosis, worst, best, and percentiles 1, 5, 10, 25, 75, 90, 95, 99 | A 5th percentile of X means about 5% of windows were at or below X |
| Chance of ENDING with a loss | Any loss, and losses worse than 10%, 20%, 30%, 40% | Looks only at the end of each window |
| Tail risk | VaR and CVaR at 95% and 99% | Positive numbers are losses |
| Annualized returns | Median, 5th percentile, 1st percentile, worst | "n/a" below 1 year |
| Fall below entry price | Share of windows down 10%, 20%, 30%, 40% or more at some point; median, 5th-percentile and worst lowest point | Includes windows that later recovered |
| Peak-to-trough drawdown | Median, severe (5th percentile) and worst drawdown; share of windows with a drawdown of 10%, 20%, 30%, 50% or more | More negative means a deeper fall |

### 4.2 Stop Analysis

There is one table for **fixed** stops and one for **trailing** stops. Each begins with a **No stop** row for comparison.

| Column | Meaning |
|---|---|
| Stop distance | Percentage below entry (fixed) or below the high since entry (trailing) |
| % windows stopped | Share of windows where the stop triggered |
| % of stopped that would have ended up | Of stopped windows, the share with a gain if held without a stop |
| Mean return / Median return | Average and middle result using this rule |
| % ending with a loss | Share of all windows with a return below zero |
| 5th percentile | Lower-tail return cutoff |
| 95% CVaR | Average loss in the worst 5%. Here a **negative number means a gain** |
| Worst | Lowest return. A stop does not guarantee a better worst result |
| Mean change vs no stop | Mean with the stop minus mean without it, in percentage points |
| % of stopped where stop helped | Of stopped windows, the share ending higher than if held |
| Mean / CVaR | Mean return per unit of tail loss. "n/a" if CVaR is zero or negative |
| Mean / CVaR: 1st half, 2nd half | The same ratio for the earlier and later halves of the windows |

A dash means the measure does not apply or no window was stopped.

The lower table splits windows by how they ended **without a stop** ("Ended up" or "Ended flat or down"). It shows the share of each group that fell a given distance below entry. It applies to fixed stops, not trailing stops.

### 4.3 Distribution

Each holding period has a table and a full-history histogram.

| Item | Meaning |
|---|---|
| First edge | Upper limit of the lowest, open-ended bin (editable) |
| Bin width | Width of each regular bin (editable) |
| From / To (below) | A bin includes its lower limit and excludes its upper limit |
| Bin | Chart label, usually the lower limit. `<` and `>=` mark the two end bins |
| Windows | Number of windows in the bin |
| Frequency | Share of all windows in the bin |
| Cumulative | Share of all windows below the bin's upper limit; the last row is 100% |

The horizontal and vertical scales differ between charts, so check them before comparing holding periods.

### 4.4 Worst Periods

Up to five windows with the lowest ending returns for each holding period: rank, start date, end date, return, lowest vs entry and maximum drawdown.

Windows within each list do not overlap (one may start on the day another ends). This stops one episode filling the list with near-identical dates. A single long crisis can still appear more than once. The list is ranked by **ending return**, so it may miss the windows with the deepest drawdowns. It can include positive returns if there were few or no further losing windows to choose.

### 4.5 Data

Each row is a trading date. `Date` and `Adj. price` come first, followed by four columns per holding period. The window **ends** on the row's date.

| Column | Meaning |
|---|---|
| `return` | Cumulative holding-period return |
| `annualized` | Annualized return; blank below 1 year |
| `lowest vs entry` | Lowest close relative to the starting price |
| `max drawdown` | Largest peak-to-trough fall within the window |

Cells are blank where there is not enough earlier history. With CSV input, the `Adj. price` column holds whichever price column was used, even if it was not adjusted.

### 4.6 Stop Data

Each row is one window in the stop analysis. It shows the end date and the **No stop** return, then two columns for every stop rule:

- **return:** result using that rule
- **hit:** `1` if the stop triggered, `0` if not

### 4.7 Notes

Records the instrument, data source, run date, main definitions and exclusions. **Check it before relying on or sharing a workbook.** For CSV input, confirm yourself whether the prices include adjustments.

---

## 5. Using the results

### 5.1 Check the data

1. On Notes, check the instrument and source.
2. On Risk Summary, check the first and last price dates.
3. Compare several prices or annual returns with another source.
4. Investigate unusually large moves in Data.

### 5.2 Assess risk

1. **Sample:** how much separate history is there?
2. **Ending loss:** how often did losses occur?
3. **Tail risk:** review 95% CVaR and the worst return.
4. **During the window:** could you have tolerated the falls and drawdowns?
5. **Distribution:** look at the shape and the loss side.
6. **Worst Periods:** which market events drive the results?

A result of 0% means the event did not occur in these windows. It does **not** mean the risk is zero. To compare investments, run each one separately using the same date range (for example, with trimmed CSV files).

### 5.3 Compare stops

Start with **Stop Scorecard** ([section 11](#11-stop-testing-lab)), which applies these ideas to more stop rules.

1. Decide your loss limit **before** looking at results, for example a maximum 95% CVaR.
2. Remove choices that fail it, including No stop if applicable. Check the worst result too.
3. Compare the mean return given up and how often eventual gainers were stopped out.
4. Prefer stable results across neighboring distances over one isolated winner.
5. Compare both halves. This is a stability check, not a test on unseen data.

Do not pick a stop only because it has the highest Mean / CVaR. If no stop meets your limit, consider a smaller position or a different investment.

### 5.4 Position size

`position value = money you are prepared to lose / assumed loss if stopped`

Illustration: a $1,000 loss budget and a 15% assumed loss give a position of about $6,667. Allow extra for price gaps and costs. A worse exit can still exceed the budget. For a trailing stop, this measures loss from entry, using the stop distance as the initial assumed loss. This calculation is not part of the workbook.

---

## 6. Settings

Edit these near the top of `risk_distribution.py`, save, and run again.

| Setting | Default | Meaning |
|---|---|---|
| `TICKER` | `"SPY"` | Default symbol |
| `HOLDING_MONTHS` | `[1, 3, 6, 12, 36, 60, 120]` | Holding periods, as positive whole months |
| `STOP_MONTHS` | `12` | Holding period for stop analysis; must also be in `HOLDING_MONTHS` |
| `STOP_LEVELS` | `"auto"` | `"auto"`, or your own list of decimals (`0.10` = 10%) for both stop types. Command line: `--stop-levels 0.05,0.10,0.20` |
| `STOP_LADDER` | 1% to 70% | Levels that `auto` picks from, in increasing order |
| `STOP_MIN_VOL_MULTIPLE` | `2.0` | Sets the `auto` minimum: this many daily standard deviations |
| `STOP_MIN_REACH` | `0.05` | Share of windows used to set the `auto` maximum |
| `STOP_MIN_LEVELS` | `3` | Fewest levels `auto` keeps |
| `MA_FILTER` | `"all"` | Entry filter: `"all"`, `"above"` or `"below"` ([section 3.9](#39-moving-average-entry-filter)) |
| `MA_PERIOD` | `200` | Moving average length in trading days; used by `above` and `below` |
| `HIST_BINS` | `30` | Approximate number of histogram bins |
| `RUN_STOP_LAB` | `True` | Switch for the stop testing lab ([section 11](#11-stop-testing-lab)) |
| `MAX_CVAR` | `0.15` | Loss limit for the scorecard and walk-forward. A placeholder: set your own |
| `BOOTSTRAP_RESAMPLES` | `1000` | Resamples per block length; `0` turns the bootstrap off |

Settings for the stop tests (slippage, cash yield, volatility multiples, block lengths and so on) are at the top of `stop_lab.py`. With `STOP_LEVELS = "auto"`, Stop Analysis uses shortened lists, while the stop testing lab always tests the full `STOP_LADDER`, so Walk Forward's choices are not trimmed using later prices. A manual list applies to both.

The loss and drawdown thresholds on Risk Summary, and the 95% and 99% tail levels, are set inside the `build` function.

---

## 7. Editing and recalculation

**Re-run the script whenever prices, dates or settings change.** This is the only way to keep every tab consistent. To analyze corrected prices, edit a CSV file and run it again.

| Updates in Excel (formulas) | Fixed until you re-run (set by the script) |
|---|---|
| Returns, annualized returns, lowest vs entry | Which start row each window uses |
| Risk Summary statistics | Maximum drawdown values |
| Stop Analysis summaries of Stop Data | Stopped returns and hit flags |
| Histogram counts and frequencies | Which windows appear on Worst Periods |
| | Formula date ranges, the half-sample split and descriptive text |

Editing a price in the workbook therefore leaves it **partly updated**. Editing the histogram's **First edge** or **Bin width** is safe, but the number of bins stays the same, and label decimal places do not change.

Blue text marks stored values, such as prices, bin settings and script-calculated results. Blue does not always mean a cell should be edited.

---

## 8. Limitations

- **Past results only.** Future losses can be larger than any in the data.
- **Overlap.** Windows are not independent. Long-period and 99% tail results may rest on very few episodes.
- **History length.** A short history may miss major market stresses. Different holding periods also cover different sets of end dates.
- **Daily closes only.** Intraday falls are not measured.
- **Not included.** Inflation, taxes, fees, trading costs and currency conversion. Any effects already built into the price series remain.
- **Data quality.** Yahoo data can contain errors and can be revised. Save the workbook and, if you need to reproduce it, the source prices in a CSV.
- **One investment per run.** No portfolio, benchmark or multi-asset analysis.
- **Stop testing.** Real orders may fill differently. Trying many stops can pick one that fitted past chance patterns.

---

## 9. Troubleshooting

| Problem | What to try |
|---|---|
| `No data returned` | Check the symbol and internet connection, update `yfinance`, or wait if rate-limited. You can use `--csv` |
| `ModuleNotFoundError` | Run the install command with the same Python you use to run the script |
| `KeyError` with a CSV | Check the column names: `Date` and `Adj Close` or `Close` |
| `Need at least ~300 daily prices` | Provide a longer daily history and check for invalid prices |
| A holding period is missing | The history gives fewer than 60 windows for it |
| Only the lab tabs are missing | Check that the lab is on: `RUN_STOP_LAB = True` or `--stop-lab` |
| Stop tabs are missing | Check that `STOP_MONTHS` is in `HOLDING_MONTHS` and has enough history |
| Cannot save the file | Close the workbook in Excel, or use a different `--out` name |
| Slow or large workbook | Remove some holding periods or stop levels, or use a shorter history |
| Formula results look empty | Open in Excel and allow calculation. The script writes formulas without stored results, so some file previewers show blanks |

---

## 10. Testing status

- Tested with **simulated prices** only. The live Yahoo download has **not** been tested successfully, because the development environment could not reach Yahoo.
- Holding-period summary and Stop Analysis values matched separate Python calculations. For the stop lab, sampled rules (fixed, trailing and volatility) matched a brute-force calculation on Stop Stress Tests and the decade and single-crisis columns. Scorecard pass and ranking results were checked at four loss limits.
- Altering later prices did not change the sampled walk-forward training choices.
- The bootstrap was checked for plausibility against newly simulated and shuffled histories, not matched value by value. Non-overlap schedule columns and walk-forward test returns were not independently recalculated.
- Formula recalculation in LibreOffice reported no errors. These checks do not cover every value or possible input.
- **MA filter:** with the filter set to `all`, a run with the stop lab on and 5 resamples matched the previous code in every cell. For 200-day `above` and `below` and 50-day `above`, selected Risk Summary measures and the entry flags matched a separate calculation, as did sampled 10% stop results. Sampled bootstrap draws tested only qualifying entries. Other filtered outputs were not independently recalculated.
- A run with 1,000 resamples on about 8,800 simulated prices took about 2.5 minutes in the 2-CPU test environment.
- Selected layouts and charts were checked visually in LibreOffice. The workbook has **not** been tested in Microsoft Excel.
- Very long histories, such as `^GSPC`, have not been tested for speed or file size.
- `SAMPLE_risk_simulated.xlsx` is for layout only. **Do not use its numbers for any investment decision.**

---

## 11. Stop testing lab

These tests apply to the `STOP_MONTHS` holding period (default 1 year). They help rule out weak stop choices. They **cannot prove which stop will be best** in future.

### 11.0 On/off switch

| Method | On | Off |
|---|---|---|
| Setting in `risk_distribution.py` (every run) | `RUN_STOP_LAB = True` | `RUN_STOP_LAB = False` |
| Command line (this run only; overrides the setting) | `--stop-lab` | `--no-stop-lab` |

The default is **on**. When off, the five lab tabs (Stop Scorecard, Stop Stress Tests, Stop Periods & Crises, Stop Bootstrap, Stop Walk Forward) are not created, and `--max-cvar` and `--bootstrap` have no effect. **Stop Analysis and Stop Data are still created**; they are the basic stop comparison, not part of the lab. `stop_lab.py` must stay in the folder even when the switch is off.

When on, the lab runs only if `STOP_MONTHS` is included in the holding periods and there is enough history. Because a run overwrites a file with the same name, a run with the lab off replaces an earlier workbook that included it. Use `--out` to keep both.

### 11.1 Rules tested

| Family | Settings | Definition |
|---|---|---|
| No stop | - | Benchmark |
| Fixed | Full `STOP_LADDER` (1% to 70%) in auto mode; otherwise your manual list | Distance below entry |
| Trailing | Full `STOP_LADDER` (1% to 70%) in auto mode; otherwise your manual list | Distance below the highest close since entry |
| Volatility | 0.50x to 2.00x, every 0.25x | Fixed stop whose distance depends on volatility at entry |

Volatility stop distance:

`multiple x standard deviation of the prior 63 daily log returns x sqrt(252 x holding months / 12)`

It uses only returns known at entry, stays unchanged during the window, and is limited to 1%-95%. **1.00x is not 1%.** The actual percentage differs between entry dates.

All rules use the same windows. Windows starting in the first 63 days are left out, so numbers can differ slightly from Stop Analysis. Triggers and exits otherwise follow section 3.8.

### 11.2 Selection rule

1. Set a maximum 95% CVaR.
2. A rule passes if its base CVaR **and every required test** are within it.
3. Passing rules are ranked by base mean return. Ties go to the rule listed first (No stop is first).
4. A passing rule is **stable** if the adjacent distances in its family also pass. Rules at the end of a range have only one neighbor. No stop has none.

"No rule qualifies" is a valid result. Consider a smaller position or a different investment before loosening the limit.

### 11.3 Stop Scorecard (start here)

| Cell or column | Meaning |
|---|---|
| B3 | Your loss limit (editable) |
| B4:B7 | Whether each test is required: next-day fill, 1% slippage, every eligible decade, bootstrap (editable, Yes/No) |
| B11 | Highest-mean passing rule |
| B12 | Highest-mean passing and stable rule |
| % stopped | How often the rule triggered. If rarely, it behaves much like No stop |
| Mean return, Mean change vs no stop | Return, and what was gained or given up |
| Base CVaR, Worst return | Tail loss and lowest historical result |
| Next-day fill, slippage, worst decade, bootstrap CVaR | Stress results |
| Tests failed | Base test plus required tests that exceed the limit. A required test with no result counts as failed |
| Passes?, Neighbors also pass? | Pass and stability flags |
| Rank columns | Rank by mean among passing, or passing and stable, rules. Equal means share a rank |

Changing yellow cells updates the scorecard in Excel. It does **not** change Walk Forward or the bootstrap's "picked as best". For those, re-run with `--max-cvar`.

### 11.4 Stop Stress Tests

| Item | Meaning |
|---|---|
| Next-day close fill | The signal occurs at one close and the exit at the next (no later than the window end) |
| 1% and 2% slippage | The exit value is reduced by that percentage, for stopped windows only |
| Cash earns 3% a year | Interest from exit to window end. 3% is an **assumption**, not a market rate. For information only |
| Episodes in worst 5% | Groups of worst windows connected by overlapping dates. One group can span several market events |
| Largest episode share | Share of the worst windows in the biggest group. High values mean the tail is concentrated |
| Non-overlap schedules | Uses up to 12 starting months, taking each next window after the previous one ends. Shows the lowest and highest mean change against No stop. Starts can move past weekends or month-ends. Schedules are not independent of each other |

### 11.5 Stop Periods & Crises

- **Decades:** only windows starting and ending in the same decade. A decade needs at least 750 windows. Partial decades can qualify.
- **Worst decade CVaR:** the highest tail loss among eligible decades.
- **Leave one crisis out:** the program finds the 3 deepest declines from a running high to the low before the next new high. It removes every window touching that high-to-low span, then does the same for all listed together. At least 250 windows must remain.
- **CVaR saved:** No stop CVaR minus the rule's CVaR. Positive means the stop reduced tail loss.

If the benefit largely disappears without one crisis, it depends heavily on that event. These removals are diagnostics, not required tests. Crises are identified using the full history.

### 11.6 Stop Bootstrap

The program rebuilds many price histories from randomly chosen blocks of consecutive daily returns (a stationary bootstrap). Block lengths vary randomly, averaging 63, 126 or 252 days. Blocks may wrap from the end of the history to the start. Every rule is tested on the same rebuilt histories.

| Column | Meaning |
|---|---|
| CVaR 5th, median, 80th, 95th percentile | Spread of tail loss across resamples |
| Mean change 5th and 95th percentile | Spread of the return gained or given up against No stop |
| % with lower CVaR than no stop | Share of resamples where the stop reduced tail loss |
| % picked as best | Share of resamples where the rule won under the basic limit. This applies base CVaR only. If none qualifies, the lowest CVaR is picked |
| Highest 80th-percentile CVaR | The least favorable block length; used by the scorecard |

For speed, it uses fixed trading-day windows (252 for 1 year), entered every 5 days, so it does not exactly match the historical calendar-window calculation.

**Read with care:** if picks are spread across many rules, no single stop stands out. Resampling breaks longer patterns, such as recoveries or long declines, so bootstrap results can differ substantially from historical ones. It cannot create daily moves that never occurred. The "80%" gate means 80% of resamples, **not** an 80% probability for your future loss.

### 11.7 Stop Walk Forward

This tests the **process of choosing** a stop on later data.

1. At each 1 January, use only windows that had already ended.
2. Choose the highest mean among rules within the base CVaR limit. If none qualifies, choose the lowest CVaR and flag it.
3. Apply that choice to windows entered during the year, once they have completed.
4. Move forward a year. Training uses an expanding history.

Testing begins in the calendar year after 10 years of completed windows, and needs at least 750 training windows. The final year may be partial.

It uses **base CVaR only**, not the scorecard's stress, decade, bootstrap or stability tests. So it does not validate the scorecard's final pick. A choice can meet the limit in training and exceed it later. Frequent changes suggest the choice is unstable, but an unchanged choice does not prove it is good.

The loss limit, stop ranges and other design choices are yours. If you change them after seeing results, the test data is no longer unseen.

### 11.8 Suggested process

1. Set the loss limit and required tests **before** looking at results.
2. Run the program and open Stop Scorecard.
3. Prefer a stable range of passing rules to a single winner.
4. Check the finalists on the stress, crisis and episode tabs.
5. Check how widely the bootstrap results vary.
6. Check whether choosing stops worked in Walk Forward, compared with No stop.
7. If the evidence is mixed, treat it as **no clear best stop**.

### 11.9 Not included

Intraday triggers (daily highs and lows), opening-gap fills, re-entry rules, profit targets, taxes, commissions, and testing on other investments. Slippage tests do not model any particular price gap or guarantee a maximum loss.
