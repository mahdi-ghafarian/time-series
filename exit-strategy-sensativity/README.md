# Exit Strategy Sensitivity

This program repeats the Exit Strategy stop/target analysis over random entry
samples. Each seed is evaluated once using the entry-SMA setting selected by
`MA_ENTRY_ENABLED`.

By default it analyzes QQQ daily data from 2000-01-01 through 2026-12-31, uses
25 requested entries per seed, tests 200 seeds (0 through 199), and uses a
12-period ATRP calculated on 3-month bars. Stop and target grids are ATRP
multipliers. It defaults to long entries, a 200-bar entry SMA, conservative
stop-first handling, no entry-MA filter, and no SMA exit. Set
`MA_ENTRY_ENABLED = True` to run with the entry-MA filter instead; the program
never runs both modes in the same execution.

The script downloads data directly with `yfinance` and does not import or
require Streamlit. Its editable defaults (ticker, date range, seed count,
entry count, SMA and ATRP settings, costs, and output directory) are grouped
near the top of `exit_strategy_sensitivity.py`; command-line options can
override them. ATRP is Wilder's ATR divided by the ATR candle's close and
expressed as a percentage. With a timeframe different from the trading bars,
only the last completed ATR candle is used. Each entry's ATRP is held fixed for
the entire trade. ATRP exit grids are multipliers of that entry ATRP; for
example, a stop multiplier of `2` with entry ATRP of `3%` sets a stop distance
of `6%` from entry.

The CSVs report stop/target parameters and their units, average actual
stop/target distances, returns in percent and ATRP units, and expectancy per
initial risk (`ExpectancyR`). Raw R-per-bar expectancy is the mean of each
seed's total realized R divided by its total bars held across trades (equal to
that seed's `ExpectancyR` divided by its average bars held). The heatmap
displays this scaled as R per 100 bars for readability; the raw R/bar value
remains in the summary CSV.
R-based metrics are undefined for stop settings without an initial stop.
Robust ranking in ATRP mode uses the fraction of
positive-`ExpectancyATRP` runs, median expectancy in ATRP units, then its
standard deviation. Use `--percent-exits` to retain fixed-percentage stops and
targets with the original percentage grids.

Four heatmaps are written by default for the selected entry-MA-filter setting:
win rate, ExpectancyR, ExpectancyR per 100 bars, and average bars held. Each
cell contains only its heatmap's metric. Edit the `HEATMAP_CONFIGS` dictionary
near the top of the script to configure each heatmap's metric and `min`/`max`
color bounds. Unset bounds auto-scale. Color bounds use the selected metric's
units. Use `--heatmap-metrics win_rate expectancy_r` to choose configured
heatmaps, or `--heatmap-metric ExpectancyATRP_Median` to create one custom
metric heatmap. The `--heatmap-min` and `--heatmap-max` options override the
bounds for all selected heatmaps.

Install the standalone dependencies with
`python -m pip install -r exit-strategy-sensativity/requirements.txt`.

Run from the repository root:

```powershell
python "exit-strategy-sensativity/exit_strategy_sensitivity.py"
```

The program downloads trading data and ATR data, then writes two CSV files and
four PNG heatmaps to `exit-strategy-sensativity/results`. The file names identify
ticker, trading timeframe, exit mode, and entry-MA filter; ATRP mode also
includes ATR timeframe and period:

- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_runs.csv`: one row for
  each seed and stop/target combination.
- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_summary.csv`: mean, median, standard
  deviation, min, max, and count across seeds for each metric and combination.
  `ExpectancyATRP_PositivePct` shows how often ATRP expectancy was positive.
- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_heatmap_win_rate.png`:
  median win rate per stop/target cell.
- Similar `_heatmap_expectancy_r.png`, `_heatmap_expectancy_r_per_bar.png`, and
  `_heatmap_bars_held.png` files show their respective configured metrics.

Options can reduce the run for a quick check or customize it:

```powershell
python "exit-strategy-sensativity/exit_strategy_sensitivity.py" `
  --seed-count 10 --seed-start 100 --entry-count 25 `
  --atr-timeframe 3mo --atr-period 14 `
  --ma-entry-enabled `
  --stops 0.5,1,1.5,2 --targets 1,2,3 `
  --heatmap-metrics win_rate expectancy_r expectancy_r_per_bar bars_held
```

Other options include `--ticker`, `--start-date`, `--end-date`, `--interval`,
`--percent-exits`, `--atr-timeframe` (`1d`, `1wk`, `1mo`, or `3mo`),
`--atr-period`, `--ma-entry-enabled`, `--no-ma-entry-enabled`, `--sma-period`,
`--direction`, `--commission`, `--slippage`, `--ma-exit`,
`--non-conservative`, `--heatmap-metrics`, `--heatmap-metric`,
`--heatmap-min`, and `--heatmap-max`.
For example, use `--direction Both` to randomly assign long/short directions
when the entry filter is disabled; when enabled, direction follows the close's
position relative to the SMA.

Run the focused tests without downloading market data:

```powershell
python -m unittest discover -s "exit-strategy-sensativity" -p "test_*.py"
```
