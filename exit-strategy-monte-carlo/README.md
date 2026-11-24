# Exit Strategy Sensitivity

This program repeats the Exit Strategy stop/target analysis over random entry
samples. Each seed is evaluated once using the entry-SMA setting selected by
`MA_ENTRY_ENABLED`.

By default it analyzes QQQ daily data from 2000-01-01 through 2026-12-31, uses
25 requested random trades per calendar year per seed, tests 200 seeds
(0 through 199), and uses a 12-period ATRP calculated on 3-month bars. Stop and target grids are ATRP
multipliers. It defaults to long entries, a 200-bar entry SMA, one open trade
at a time, conservative stop-first handling, no entry-MA filter, and no SMA
exit. Set `MA_ENTRY_ENABLED = True` to run with the entry-MA filter instead; the program
never runs both modes in the same execution.

The script downloads data directly with `yfinance` and does not import or
require Streamlit. Its editable defaults (ticker, date range, seed count,
entry count, SMA and ATRP settings, costs, and output directory) are grouped
near the top of `exit_strategy_monte_carlo_analysis.py`; command-line options can
override them. ATRP is Wilder's ATR divided by the ATR candle's close and
expressed as a percentage. With a timeframe different from the trading bars,
only the last completed ATR candle is used. Each entry's ATRP is held fixed for
the entire trade. ATRP exit grids are multipliers of that entry ATRP; for
example, a stop multiplier of `2` with entry ATRP of `3%` sets a stop distance
of `6%` from entry.
Use `--use-trp` for exit distances normalized by the expanding median True
Range percentage from the first downloaded range bar through each entry. TRP is
`TrueRange / Close * 100`, then the median is taken across all bars from the
first bar through the entry bar and held fixed for that trade. `--range-period`
controls how many pre-start range bars are downloaded; it does not cap the
expanding median. TRP grids use the same stop/target multipliers as ATRP mode.

The CSVs report stop/target parameters and their units, average actual
stop/target distances, returns in percent and ATRP units, and expectancy per
initial risk (`ExpectancyR`). `MaxConsecutiveLosses` is the longest streak of
negative-return trades in chronological entry order; a winning or break-even
trade resets the streak. Raw R-per-bar expectancy is the seed's total realized
R divided by its total bars held across trades (equal to
that seed's `ExpectancyR` divided by its average bars held). The heatmap
displays this scaled as R per 100 bars for readability; the raw R/bar value
remains in the summary CSV.
For a no-stop (`-1`) setting, `ExpectancyR` uses an assumed initial risk of
100% of entry value for normalization only; the simulation still places no stop.
`MaxDrawdownR` is the largest peak-to-trough drop in cumulative realized R.
`MaxDrawdownPct` uses sequential, full-equity compounding of closed-trade
`NetReturnPct` values; it excludes unrealized losses while a trade is open.
An individual return at or below -100% is treated as account ruin (100%
drawdown). This percentage model is not a realistic account-equity model for
short trades or concurrent positions.
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
`python -m pip install -r exit-strategy-monte-carlo/requirements.txt`.

Run from the repository root:

```powershell
python "exit-strategy-monte-carlo/exit_strategy_monte_carlo_analysis.py"
```

The program downloads trading data and ATR data, then writes two CSV files and
four PNG heatmaps to `exit-strategy-monte-carlo/results/analysis`. The file names identify
ticker, trading timeframe, exit mode, and entry-MA filter; ATRP mode also
includes ATR timeframe and period:

- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_runs.csv`: one row for
  each seed and stop/target combination.
- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_summary.csv`: mean, median, standard
  deviation, min, max, count, and the 5th, 25th, 75th, and 95th percentiles
  across seeds for each metric and combination. `ExpectancyATRP_PositivePct`
  shows how often ATRP expectancy was positive.
- `qqq_1d_atrp_3mo_12_entry-ma-disabled_sensitivity_heatmap_win_rate.png`:
  median win rate per stop/target cell.
- Similar `_heatmap_expectancy_r.png`, `_heatmap_expectancy_r_per_bar.png`, and
  `_heatmap_bars_held.png` files show their respective configured metrics.

Options can reduce the run for a quick check or customize it:

```powershell
python "exit-strategy-monte-carlo/exit_strategy_monte_carlo_analysis.py" `
  --seed-count 10 --seed-start 100 --trades-per-year 25 `
  --atr-timeframe 3mo --atr-period 14 `
  --ma-entry-enabled `
  --stops 0.5,1,1.5,2 --targets 1,2,3 `
  --heatmap-metrics win_rate expectancy_r expectancy_r_per_bar bars_held
```

Other options include `--ticker`, `--start-date`, `--end-date`, `--interval`,
`--exit-mode` (`ATRP`, `TRP`, or `Percent`), `--use-atrp`, `--use-trp`,
`--percent-exits`, `--range-timeframe` (`1d`, `1wk`, `1mo`, or `3mo`),
and `--range-period`. The former `--atr-timeframe` and `--atr-period`
options remain aliases.
`--atr-period`, `--ma-entry-enabled`, `--no-ma-entry-enabled`, `--sma-period`,
`--direction`, `--trades-per-year`, `--max-open-trades`, `--commission`,
`--slippage`, `--ma-exit`, `--non-conservative`, `--heatmap-metrics`,
`--heatmap-metric`, `--heatmap-min`, and `--heatmap-max`.
For example, use `--direction Both` to randomly assign long/short directions
when the entry filter is disabled; when enabled, direction follows the close's
position relative to the SMA.

Run the focused tests without downloading market data:

```powershell
python -m unittest discover -s "exit-strategy-monte-carlo" -p "test_*.py"
```

## Which script should I use?

Both scripts run Monte Carlo exit-strategy analyses, but they serve different
reporting needs. Both default to one simultaneous open trade.

| Script | Use it when you want to... | Summary output |
| --- | --- | --- |
| `exit_strategy_monte_carlo_analysis.py` | Compare stop/target combinations systematically and assess the distribution and robustness of results across seeds. | One summary containing mean, median, standard deviation, min, max, count, and P5/P25/P75/P95 for every metric. |
| `exit_strategy_monte_carlo_visualization.py` | Explore an annotated heatmap and inspect the trades and entries from a representative scenario. | A summary using one selected statistic across scenarios (for example, median or P25), plus scenario and representative-trade detail CSVs. |

Use the analysis script for a broad comparison with multiple distribution
statistics in one report. Use the visualization script when you want to focus
on one statistic at a time and examine scenario-level trade details. They share
trade simulation and metric calculations, but run separate workflows and save
outputs in separate directories.

## Monte Carlo analysis

`exit_strategy_monte_carlo_visualization.py` is a standalone Monte Carlo command-line
analysis; it does not use Streamlit. It evaluates stop/target combinations
over random-entry scenarios, creates an annotated heatmap, and writes a
selected-statistic summary, per-scenario results, and representative-scenario
trade and entry details. By default, scenario 0 supplies the detailed trades
and entries, while the heatmap and ranked summary aggregate across all
requested scenarios.

### Monte Carlo parameters

- `--scenario-count` / `--seed-count`: Number of independent Monte Carlo
  runs. Each run uses a different seed and contributes one metric value per
  stop/target combination to the selected aggregation.
- `--scenario-start` / `--seed-start`: Seed for the first run. The runs use
  consecutive seeds: `scenario-start`, `scenario-start + 1`, and so on.
- `--trades-per-year`: Maximum number of eligible random trades selected from
  each calendar year within every run. It is independent of the scenario count.
  SMA filtering, unavailable ATRP, and years with limited data can reduce the
  actual count.
- `--entry-count`: Legacy alias for `--trades-per-year`. Do not specify both
  aliases; the last occurrence determines the value.
- `--max-open-trades`: Positive integer limit on simultaneous positions
  (defaults to `1`). Each stop/target pair accepts entries chronologically and
  skips candidates while the configured number of trades is still open.
  Values above one allow concurrent trades; accepted trades can differ between
  pairs because exit times vary. Drawdown metrics are closed-trade measures
  and do not mark overlapping positions to market.

A seed controls reproducible random sampling. Sampling does not use the seed
to repeat the same trades for every year. Instead, a separate random selection
is made for each year using that scenario's seed.

For example, `--scenario-start 0 --scenario-count 200 --trades-per-year 25`
creates seeds `0` through `199`. In each of these 200 scenarios, up to 25
eligible trades are randomly selected from every represented calendar year.

Choose the aggregation across scenarios used by the heatmap and ranked summary with
`--statistic`: `min`, `median`, `average`, `max`, `p1`, `p5`, `p25`, `p75`,
`p95`, or `p99`. For example:

```powershell
python "exit-strategy-monte-carlo/exit_strategy_monte_carlo_visualization.py" `
  --percent-exits --scenario-count 50 --trades-per-year 25 `
  --max-open-trades 1 `
  --stops 5,10,20 --targets 5,10,20 `
  --statistic p5 --top-metric ExpectancyR --bottom-metric WinRatePct `
  --color-metric ExpectancyR
```

By default, the script uses the same market-data, scenario, entry, and exit
settings as the sensitivity CLI and writes its PNG and CSV files under
`exit-strategy-monte-carlo/results/visualization`, separate from the analysis
outputs in `exit-strategy-monte-carlo/results/analysis`. The heatmap's top, bottom, and color
metrics can be changed with `--top-metric`, `--bottom-metric`, and
`--color-metric`; `--rank-metric` independently controls the ranked CSV.
Defaults for all of these settings, including stop/target grids and output
directory, are grouped under `USER CONFIGURATION` near the top of the script;
command-line options override those values. The color scale defaults to 0–1;
values outside the range are clipped and can be included by overriding
`--color-min` or `--color-max`. Output names use a compact ticker, exit mode,
scenario count/start, entry-MA flag, and statistic prefix. The former
`--seed-count` and `--seed-start` options remain as aliases for compatibility.
