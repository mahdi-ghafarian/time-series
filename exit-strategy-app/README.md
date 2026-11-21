# Exit Strategy

The Exit Strategy app calculates stop-loss and target combinations from
historical Yahoo Finance data. It also includes an optional precomputed startup
bundle for the default QQQ configuration so the app can display results quickly
on launch.

Run the app directly from the repository root:

```powershell
streamlit run exit-strategy-app/app.py
```

Or run it through the suite:

```powershell
streamlit run market-analysis-suite/app.py
```

## Changing the startup defaults

Update the defaults in the following places whenever the startup configuration
changes:

1. Edit the current UI defaults in `exit_strategy_ui.py`:
   - `Interval`
   - `Start date`
   - `End date`
   - SMA entry/exit options
   - Trade direction
   - Random entries
   - Random seed
   - Commission
   - Slippage
   - Conservative stop/target behavior
   - Top, bottom, color, and rank metrics
   - Highlight thresholds
   - `DEFAULT_STOPS`
   - `DEFAULT_TARGETS`
2. Update the old-to-new preference migration in
   `../market-analysis-suite/app.py` if existing suite sessions need the new
   values. Update it together with `exit_strategy_ui.py` for date, highlight,
   metric, and stop/target changes.
3. Ensure `exit_strategy_startup.py` still compares every startup setting that
   must match the bundle. In particular, the bundle must match the ticker,
   dates, interval, entry count, seed, costs, direction, SMA options,
   conservative behavior, and stop/target grids.
4. Update the informational text in `app.py` if the displayed bundle settings
   change.

The current startup configuration is:

- Ticker: `QQQ`
- Start date: `2000-01-01`
- End date: `2026-12-31`
- Interval: `1d`
- Random entries: `1000`
- Random seed: `27`
- Commission: `0`
- Slippage: `0`
- Conservative: enabled
- Trade direction: `Long`
- SMA entry: disabled
- SMA exit: disabled
- Top, color, and rank metric: `ExpectancyR`
- Bottom metric: `WinRatePct`
- Highlight win rate: `67.0`
- Highlight expectancy R: `1.0`
- Stops:
  `5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 90, -1`
- Targets:
  `5, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, -1`

## Recreate the startup bundle

The startup file is
[`startup_result.pkl`](./startup_result.pkl). The save helper refuses to
replace an existing bundle, so delete the old bundle before recreating it.

From the repository root, run:

```powershell
Remove-Item -LiteralPath "exit-strategy-app/startup_result.pkl" -Force

python -c "import sys; from datetime import date; sys.path.insert(0, r'exit-strategy-app'); from exit_strategy_simulation import run_simulation_now; from exit_strategy_startup import save_startup_result_if_missing; stops=[5,7.5,10,12.5,15,20,25,30,35,40,45,50,75,90,-1]; targets=[5,7.5,10,12.5,15,20,25,30,35,40,45,50,75,100,-1]; out=run_simulation_now('QQQ',date(2000,1,1),date(2026,12,31),'1d',1000,27,0.0,0.0,True,stops,targets,200,'Long',False,False,False,False); print(save_startup_result_if_missing(out))"
```

The simulation downloads QQQ data and may take several minutes. The command
prints `True` when `startup_result.pkl` was created successfully.

Verify the saved bundle and settings with:

```powershell
python -c "import sys; from datetime import date; sys.path.insert(0, r'exit-strategy-app'); import exit_strategy_startup as s; o=s.load_startup_result(); stops=', '.join(map(str,[5,7.5,10,12.5,15,20,25,30,35,40,45,50,75,90,-1])).replace(', ', ','); targets=', '.join(map(str,[5,7.5,10,12.5,15,20,25,30,35,40,45,50,75,100,-1])).replace(', ', ','); print(s.result_ticker(o)); print(s.result_has_directional_entries(o)); print(s.startup_result_matches_settings(o,'QQQ',date(2000,1,1),date(2026,12,31),'1d',1000,27,0.0,0.0,True,stops,targets,'Long',False,False))"
```

The final three lines should indicate `QQQ`, `True`, and `True`.

Before publishing:

1. Commit `exit_strategy_ui.py`, `exit_strategy_startup.py`, and the new
   `startup_result.pkl`.
2. Commit any suite migration changes in `market-analysis-suite/app.py`.
3. Redeploy the app.
4. Test both standalone Exit Strategy and the Market Analysis Suite.
