from io import BytesIO
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exit_strategy_monte_carlo_analysis import (
    align_atrp_to_trading_data,
    align_trp_to_trading_data,
    build_parser,
    build_heatmap_requests,
    calculate_atrp,
    calculate_trp,
    calculate_metrics,
    create_heatmap,
    download_data,
    HEATMAP_METRICS,
    limit_concurrent_trades,
    run_sensitivity,
    select_entries,
    simulate_trade,
    summarize_runs,
)


class ExitStrategySensitivityTests(unittest.TestCase):
    def test_atrp_is_default_and_percent_mode_can_be_selected(self):
        defaults = build_parser().parse_args([])
        percent_mode = build_parser().parse_args(["--percent-exits"])
        enabled_entry_filter = build_parser().parse_args(["--ma-entry-enabled"])

        self.assertTrue(defaults.use_atrp)
        self.assertFalse(defaults.ma_entry_enabled)
        self.assertEqual(defaults.interval, "1d")
        self.assertEqual(defaults.atr_timeframe, "3mo")
        self.assertEqual(defaults.atr_period, 12)
        self.assertEqual(defaults.max_open_trades, 1)
        self.assertEqual(
            build_parser().parse_args(["--max-open-trades", "2"]).max_open_trades,
            2,
        )
        self.assertFalse(percent_mode.use_atrp)
        self.assertTrue(enabled_entry_filter.ma_entry_enabled)

    def test_runs_only_the_selected_entry_filter_for_each_seed(self):
        dates = pd.date_range("2020-01-01", periods=12, freq="D")
        close = np.array([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 14], dtype=float)
        data = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=dates,
        )
        actual_entries = select_entries(
            data,
            number_of_entries=11,
            seed=3,
            sma_period=2,
            trade_direction="Both",
            ma_enabled=True,
        )
        self.assertEqual(actual_entries["EntryIndex"].tolist(), list(range(1, 11)))
        self.assertEqual(
            actual_entries["Direction"].tolist(),
            [
                "Long",
                "Long",
                "Short",
                "Short",
                "Short",
                "Long",
                "Long",
                "Long",
                "Long",
                "Short",
            ],
        )

        runs = run_sensitivity(
            data=data,
            seeds=[3, 4],
            number_of_entries=3,
            stops=[5],
            targets=[5],
            sma_period=2,
            trade_direction="Long",
            use_atrp=False,
        )

        self.assertEqual(len(runs), 2)
        self.assertEqual(set(runs["Seed"]), {3, 4})
        self.assertEqual(set(runs["EntryMAEnabled"]), {False})
        self.assertTrue((runs["SelectedEntries"] == 3).all())

        filtered_runs = run_sensitivity(
            data=data,
            seeds=[3, 4],
            number_of_entries=3,
            stops=[5],
            targets=[5],
            sma_period=2,
            trade_direction="Long",
            use_atrp=False,
            ma_entry_enabled=True,
        )
        self.assertEqual(len(filtered_runs), 2)
        self.assertEqual(set(filtered_runs["EntryMAEnabled"]), {True})

    def test_select_entries_samples_requested_count_in_each_calendar_year(self):
        dates = pd.to_datetime(
            ["2020-12-30", "2020-12-31", "2021-01-01", "2021-01-02", "2021-01-03"]
        )
        close = np.array([10, 11, 12, 13, 14], dtype=float)
        data = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=dates,
        )

        entries = select_entries(
            data,
            number_of_entries=100,
            seed=7,
            sma_period=2,
            trade_direction="Long",
            ma_enabled=False,
            trades_per_year=2,
        )

        selected_year_counts = entries["EntryDate"].dt.year.value_counts().sort_index()
        self.assertEqual(selected_year_counts.to_dict(), {2020: 2, 2021: 2})

    def test_limit_concurrent_trades_releases_slots_at_exit_bar(self):
        trades = [
            {"EntryIndex": 0, "ExitIndex": 3},
            {"EntryIndex": 1, "ExitIndex": 2},
            {"EntryIndex": 2, "ExitIndex": 4},
            {"EntryIndex": 3, "ExitIndex": 5},
        ]

        selected = limit_concurrent_trades(trades, max_open_trades=1)

        self.assertEqual(
            [(trade["EntryIndex"], trade["ExitIndex"]) for trade in selected],
            [(0, 3), (3, 5)],
        )
        self.assertEqual(limit_concurrent_trades(trades, None), trades)

    def test_run_sensitivity_applies_max_open_trades(self):
        close = np.array([10, 11, 12, 13, 14, 15, 16], dtype=float)
        data = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=pd.date_range("2020-01-01", periods=len(close), freq="D"),
        )

        runs = run_sensitivity(
            data=data,
            seeds=[3],
            number_of_entries=3,
            trades_per_year=3,
            stops=[-1],
            targets=[-1],
            sma_period=2,
            trade_direction="Long",
            use_atrp=False,
            max_open_trades=1,
        )

        self.assertEqual(runs.loc[0, "SelectedEntries"], 1)
        self.assertEqual(runs.loc[0, "NumberOfTrades"], 1)

    def test_summary_ranks_consistent_positive_expectancy_first(self):
        base = {
            "Seed": [1, 2, 3, 1, 2, 3],
            "Ticker": ["QQQ"] * 6,
            "TradingTimeframe": ["1d"] * 6,
            "EntryMAEnabled": [False] * 3 + [True] * 3,
            "ExitMode": ["Percent"] * 6,
            "ParameterUnit": ["Percent of entry price"] * 6,
            "ATRTimeframe": ["N/A"] * 6,
            "ATRPeriod": [np.nan] * 6,
            "StopParameter": [5] * 6,
            "TargetParameter": [5] * 6,
        }
        metrics = {
            column: [1.0] * 6
            for column in (
                "NumberOfTrades",
                "MaxConsecutiveLosses",
                "WinRatePct",
                "AverageWinPct",
                "AverageLossPct",
                "AverageWinATRP",
                "AverageLossATRP",
                "RealizedRRR",
                "BreakEvenWinRatePct",
                "PlannedRRR",
                "PlannedBreakEvenWinRatePct",
                "ExpectancyPct",
                "ExpectancyPctPerBar",
                "CumulativeReturnPct",
                "ExpectancyATRP",
                "ExpectancyATRPPerBar",
                "ExpectancyTRP",
                "ExpectancyTRPPerBar",
                "ExpectancyRPerBar",
                "ExpectancyRPer100Bars",
                "TotalRealizedR",
                "MaxDrawdownR",
                "MaxDrawdownPct",
                "AverageBarsHeld",
                "AverageEntryATRP",
                "AverageEntryTRP",
                "AverageStopDistancePct",
                "AverageTargetDistancePct",
                "StopExitPct",
                "TargetExitPct",
                "MovingAverageExitPct",
                "EndOfDataExitPct",
                "AverageCommissionPct",
            )
        }
        metrics["ExpectancyR"] = [0.2, 0.1, -0.1, 0.3, 0.3, 0.3]
        runs = pd.DataFrame({**base, **metrics})

        summary = summarize_runs(runs)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["RobustRank"].tolist(), [1, 1])
        self.assertAlmostEqual(
            summary.loc[~summary["EntryMAEnabled"], "ExpectancyR_PositivePct"].iloc[0],
            200 / 3,
        )
        self.assertEqual(
            summary.loc[summary["EntryMAEnabled"], "ExpectancyR_PositivePct"].iloc[0],
            100.0,
        )
        self.assertIn("ExpectancyPctPerBar_Median", summary.columns)
        self.assertIn("ExpectancyATRPPerBar_Median", summary.columns)
        self.assertIn("ExpectancyRPerBar_Median", summary.columns)
        self.assertIn("ExpectancyRPer100Bars_Median", summary.columns)
        no_ma_summary = summary.loc[~summary["EntryMAEnabled"]].iloc[0]
        self.assertAlmostEqual(no_ma_summary["ExpectancyR_P5"], -0.08)
        self.assertAlmostEqual(no_ma_summary["ExpectancyR_P25"], 0.0)
        self.assertAlmostEqual(no_ma_summary["ExpectancyR_P75"], 0.15)
        self.assertAlmostEqual(no_ma_summary["ExpectancyR_P95"], 0.19)
        for percentile in (5, 25, 75, 95):
            self.assertIn(f"ExpectancyR_P{percentile}", HEATMAP_METRICS)

    def test_heatmap_requests_select_four_requested_metrics(self):
        atrp_requests = build_heatmap_requests(use_atrp=True)
        self.assertEqual(
            [request["name"] for request in atrp_requests],
            [
                "win_rate",
                "expectancy_r",
                "expectancy_r_per_bar",
                "bars_held",
            ],
        )
        self.assertEqual(atrp_requests[0]["metric"], "WinRatePct_Median")
        self.assertEqual(
            atrp_requests[1]["metric"], "ExpectancyR_Median"
        )
        self.assertEqual(
            atrp_requests[2]["metric"], "ExpectancyRPer100Bars_Median"
        )
        self.assertEqual(
            build_heatmap_requests(
                use_atrp=False,
                selected_configs=["expectancy_r"],
            )[0]["metric"],
            "ExpectancyR_Median",
        )

    def test_expectancy_per_bar_is_mean_trade_return_per_held_bar(self):
        close = np.array([100.0, 104.0, 108.0])
        prices = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.1,
                "Low": close - 0.1,
                "Close": close,
            },
            index=pd.date_range("2020-01-01", periods=len(close), freq="D"),
        )
        trade = simulate_trade(
            data=prices,
            entry_index=0,
            stop_parameter=10,
            target_parameter=-1,
            commission_pct=0.0,
            slippage_pct=0.0,
            conservative=True,
            direction="Long",
            use_atrp=False,
        )
        metrics = calculate_metrics(
            pd.DataFrame([trade]),
            stop=10,
            target=-1,
            exit_mode="Percent",
        )

        self.assertEqual(trade["BarsHeld"], 2)
        self.assertAlmostEqual(
            trade["NetReturnPctPerBar"],
            trade["NetReturnPct"] / trade["BarsHeld"],
        )
        self.assertAlmostEqual(
            metrics["ExpectancyPctPerBar"],
            trade["NetReturnPctPerBar"],
        )
        self.assertAlmostEqual(
            metrics["ExpectancyRPerBar"],
            trade["RealizedR"] / trade["BarsHeld"],
        )
        self.assertAlmostEqual(
            metrics["ExpectancyRPer100Bars"],
            metrics["ExpectancyRPerBar"] * 100,
        )

    def test_no_stop_uses_100_percent_risk_for_r_without_placing_stop(self):
        close = np.array([100.0, 110.0])
        data = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
            },
            index=pd.date_range("2020-01-01", periods=len(close), freq="D"),
        )

        trade = simulate_trade(
            data=data,
            entry_index=0,
            stop_parameter=-1,
            target_parameter=-1,
            commission_pct=0.0,
            slippage_pct=0.0,
            conservative=True,
            direction="Long",
        )
        metrics = calculate_metrics(
            pd.DataFrame([trade]),
            stop=-1,
            target=-1,
            exit_mode="Percent",
        )

        self.assertEqual(trade["ExitReason"], "EndOfData")
        self.assertEqual(trade["StopLossPct"], -1)
        self.assertTrue(np.isnan(trade["StopDistancePct"]))
        self.assertAlmostEqual(trade["RealizedR"], 0.1)
        self.assertAlmostEqual(metrics["ExpectancyR"], 0.1)
        self.assertAlmostEqual(metrics["TotalRealizedR"], 0.1)
        self.assertAlmostEqual(metrics["CumulativeReturnPct"], 10.0)

    def test_maximum_consecutive_losses_is_counted_in_trade_order(self):
        trades = pd.DataFrame(
            {
                "NetReturnPct": [-1.0, -2.0, 1.0, 0.0, -3.0, -4.0, -5.0],
                "NetReturnATRP": [-0.5, -1.0, 0.5, 0.0, -1.5, -2.0, -2.5],
                "NetReturnPctPerBar": [-0.5, -1.0, 0.5, 0.0, -1.5, -2.0, -2.5],
                "NetReturnATRPPerBar": [-0.25, -0.5, 0.25, 0.0, -0.75, -1.0, -1.25],
                "RealizedR": [-0.5, -1.0, 0.5, 0.0, -1.5, -2.0, -2.5],
                "BarsHeld": [2, 2, 2, 2, 2, 2, 2],
                "EntryATRP": [2.0] * 7,
                "StopDistancePct": [2.0] * 7,
                "TargetDistancePct": [4.0] * 7,
                "ExitReason": ["Stop"] * 7,
                "CommissionPct": [0.0] * 7,
            }
        )

        metrics = calculate_metrics(
            trades,
            stop=2,
            target=4,
            exit_mode="ATRP",
        )

        self.assertEqual(metrics["MaxConsecutiveLosses"], 3)

    def test_r_per_bar_is_total_r_divided_by_total_bars_held(self):
        trades = pd.DataFrame(
            {
                "NetReturnPct": [2.0, -0.5],
                "NetReturnATRP": [np.nan, np.nan],
                "NetReturnPctPerBar": [0.2, -0.5],
                "NetReturnATRPPerBar": [np.nan, np.nan],
                "RealizedR": [2.0, -0.5],
                "BarsHeld": [10, 1],
                "EntryATRP": [np.nan, np.nan],
                "StopDistancePct": [1.0, 1.0],
                "TargetDistancePct": [np.nan, np.nan],
                "ExitReason": ["EndOfData", "Stop"],
                "CommissionPct": [0.0, 0.0],
            }
        )

        metrics = calculate_metrics(
            trades,
            stop=1.0,
            target=-1,
            exit_mode="Percent",
        )

        self.assertAlmostEqual(metrics["ExpectancyR"], 0.75)
        self.assertAlmostEqual(metrics["TotalRealizedR"], 1.5)
        self.assertAlmostEqual(metrics["CumulativeReturnPct"], 1.49)
        self.assertAlmostEqual(metrics["AverageBarsHeld"], 5.5)
        self.assertAlmostEqual(metrics["ExpectancyRPerBar"], 1.5 / 11)
        self.assertGreater(metrics["ExpectancyRPerBar"], 0)
        self.assertAlmostEqual(
            metrics["ExpectancyRPer100Bars"],
            metrics["ExpectancyRPerBar"] * 100,
        )

    def test_maximum_drawdown_uses_r_and_compounded_account_equity(self):
        trades = pd.DataFrame(
            {
                "NetReturnPct": [10.0, -20.0, 5.0],
                "NetReturnATRP": [np.nan] * 3,
                "NetReturnPctPerBar": [10.0, -20.0, 5.0],
                "NetReturnATRPPerBar": [np.nan] * 3,
                "RealizedR": [1.0, -2.0, 0.5],
                "BarsHeld": [1.0] * 3,
                "EntryATRP": [np.nan] * 3,
                "StopDistancePct": [1.0] * 3,
                "TargetDistancePct": [np.nan] * 3,
                "ExitReason": ["Target", "Stop", "Target"],
                "CommissionPct": [0.0] * 3,
            }
        )

        metrics = calculate_metrics(
            trades,
            stop=1.0,
            target=1.0,
            exit_mode="Percent",
        )

        self.assertAlmostEqual(metrics["MaxDrawdownR"], 2.0)
        self.assertAlmostEqual(metrics["MaxDrawdownPct"], 20.0)

        ruined_trades = trades.iloc[:2].copy()
        ruined_trades["NetReturnPct"] = [-110.0, 50.0]
        ruined_metrics = calculate_metrics(
            ruined_trades,
            stop=1.0,
            target=1.0,
            exit_mode="Percent",
        )
        self.assertEqual(ruined_metrics["CumulativeReturnPct"], -100.0)
        self.assertEqual(ruined_metrics["MaxDrawdownPct"], 100.0)

    def test_downloads_and_normalizes_yahoo_ohlc_data(self):
        index = pd.date_range("2020-01-01", periods=2, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close"], ["QQQ"]]
        )
        yahoo_data = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5],
                [10.5, 12.0, 10.0, 11.5],
            ],
            index=index,
            columns=columns,
        )
        with patch(
            "exit_strategy_monte_carlo_analysis.yf.download",
            return_value=yahoo_data,
        ) as yahoo_download:
            data = download_data(
                "QQQ",
                date(2020, 1, 1),
                date(2020, 1, 2),
                "1d",
            )

        yahoo_download.assert_called_once_with(
            "QQQ",
            start="2020-01-01",
            end="2020-01-03",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        self.assertEqual(data.columns.tolist(), ["Open", "High", "Low", "Close"])
        self.assertEqual(len(data), 2)

    def test_heatmap_shows_only_the_selected_metric_per_cell(self):
        summary = pd.DataFrame(
            {
                "EntryMAEnabled": [False] * 4,
                "StopParameter": [0.5, 1.0, 0.5, 1.0],
                "TargetParameter": [0.5, 0.5, 1.0, 1.0],
                "ExpectancyATRP_Median": [-1.0, 0.0, 1.0, 2.0],
            }
        )
        comparison_result = pd.concat(
            [
                summary,
                pd.DataFrame(
                    {
                        "EntryMAEnabled": [True],
                        "StopParameter": [0.5],
                        "TargetParameter": [0.5],
                        "ExpectancyATRP_Median": [10.0],
                    }
                ),
            ],
            ignore_index=True,
        )

        figure = create_heatmap(
            results=comparison_result,
            stops=[0.5, 1.0],
            targets=[0.5, 1.0],
            metric="ExpectancyATRP_Median",
            ticker="QQQ",
            trading_timeframe="1d",
            entry_ma_enabled=False,
            exit_mode="ATRP",
            atr_timeframe="3mo",
            atr_period=14,
            color_min=-2.0,
            color_max=3.0,
        )
        try:
            axis = figure.axes[0]
            self.assertEqual(
                [text.get_text() for text in axis.texts],
                ["-1.00", "0.00", "1.00", "2.00"],
            )
            self.assertEqual(len(axis.images), 1)
            self.assertEqual(axis.images[0].norm.vmin, -2.0)
            self.assertEqual(axis.images[0].norm.vmax, 3.0)
            image = BytesIO()
            figure.savefig(image, format="png")
            self.assertTrue(image.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            plt.close(figure)

        auto_scale_figure = create_heatmap(
            results=comparison_result,
            stops=[0.5, 1.0],
            targets=[0.5, 1.0],
            metric="ExpectancyATRP_Median",
            ticker="QQQ",
            trading_timeframe="1d",
            entry_ma_enabled=False,
            exit_mode="ATRP",
            atr_timeframe="3mo",
            atr_period=14,
        )
        try:
            self.assertEqual(auto_scale_figure.axes[0].images[0].norm.vmin, -1.0)
            self.assertEqual(auto_scale_figure.axes[0].images[0].norm.vmax, 10.0)
        finally:
            plt.close(auto_scale_figure)

    def test_trp_is_expanding_median_true_range_percentage(self):
        data = pd.DataFrame(
            {
                "High": [105.0, 106.0, 100.0],
                "Low": [95.0, 94.0, 90.0],
                "Close": [100.0, 100.0, 95.0],
            },
            index=pd.date_range("2020-01-01", periods=3),
        )

        trp = calculate_trp(data)

        self.assertTrue(np.isfinite(trp.to_numpy()).all())
        self.assertGreater(trp.iloc[-1], trp.iloc[0])

        trade = simulate_trade(
            data=pd.DataFrame(
                {
                    "Open": [100.0, 100.0, 100.0],
                    "High": [105.0, 106.0, 100.0],
                    "Low": [95.0, 94.0, 90.0],
                    "Close": [100.0, 100.0, 95.0],
                },
                index=pd.date_range("2020-01-01", periods=3),
            ),
            entry_index=0,
            stop_parameter=2,
            target_parameter=3,
            commission_pct=0,
            slippage_pct=0,
            conservative=True,
            direction="Long",
            entry_trp=10.0,
            use_trp=True,
        )

        self.assertEqual(trade["ExitMode"], "TRP")
        self.assertAlmostEqual(trade["StopDistancePct"], 20.0)
        self.assertAlmostEqual(trade["TargetDistancePct"], 30.0)
        self.assertAlmostEqual(trade["NetReturnTRP"], trade["NetReturnPct"] / 10.0)

    def test_aligned_trp_keeps_expanding_median_instead_of_rolling_lookback(self):
        range_dates = pd.date_range("2020-01-01", periods=25, freq="QS")
        true_ranges = np.array([10.0] * 12 + [2.0] * 13)
        range_data = pd.DataFrame(
            {
                "High": 100.0 + true_ranges / 2,
                "Low": 100.0 - true_ranges / 2,
                "Close": 100.0,
            },
            index=range_dates,
        )
        trading_data = pd.DataFrame(
            index=pd.to_datetime(["2025-10-01", "2026-01-01"])
        )

        aligned = align_trp_to_trading_data(
            trading_data=trading_data,
            range_data=range_data,
            trading_timeframe="1d",
            range_timeframe="3mo",
        )

        self.assertEqual(aligned.iloc[0], 10.0)
        self.assertEqual(aligned.iloc[1], 6.0)

    def test_quarterly_atrp_is_only_available_after_its_bar_completes(self):
        atr_dates = pd.to_datetime(
            ["2020-01-01", "2020-04-01", "2020-07-01"]
        )
        atr_data = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0],
                "High": [110.0, 120.0, 130.0],
                "Low": [90.0, 80.0, 70.0],
                "Close": [100.0, 100.0, 100.0],
            },
            index=atr_dates,
        )
        daily_data = pd.DataFrame(
            {"Close": [100.0, 100.0, 100.0]},
            index=atr_dates,
        )

        quarterly_atrp = calculate_atrp(atr_data, period=1)
        aligned = align_atrp_to_trading_data(
            trading_data=daily_data,
            atr_data=atr_data,
            trading_timeframe="1d",
            atr_timeframe="3mo",
            atr_period=1,
        )

        self.assertTrue(np.isnan(aligned.iloc[0]))
        self.assertEqual(aligned.iloc[1], quarterly_atrp.iloc[0])
        self.assertEqual(aligned.iloc[2], quarterly_atrp.iloc[1])

    def test_atrp_stop_uses_entry_atrp_as_fixed_distance(self):
        prices = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0],
                "High": [101.0, 103.0, 101.0],
                "Low": [99.0, 95.0, 99.0],
                "Close": [100.0, 100.0, 100.0],
            },
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        trade = simulate_trade(
            data=prices,
            entry_index=0,
            stop_parameter=2.0,
            target_parameter=1.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            conservative=True,
            direction="Long",
            entry_atrp=2.0,
            use_atrp=True,
        )

        self.assertEqual(trade["ExitReason"], "Stop")
        self.assertEqual(trade["StopDistancePct"], 4.0)
        self.assertEqual(trade["ExitReferencePrice"], 96.0)
        self.assertAlmostEqual(trade["NetReturnATRP"], -2.0)
        self.assertAlmostEqual(trade["RealizedR"], -1.0)

    def test_sensitivity_reports_atrp_exit_parameters_and_expectancy(self):
        close = np.array([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 14], dtype=float)
        data = pd.DataFrame(
            {
                "Open": close,
                "High": close + 0.5,
                "Low": close - 0.5,
                "Close": close,
            },
            index=pd.date_range("2020-01-01", periods=len(close), freq="D"),
        )
        runs = run_sensitivity(
            data=data,
            seeds=[3],
            number_of_entries=3,
            stops=[1.0],
            targets=[2.0],
            sma_period=2,
            use_atrp=True,
            atrp_values=np.full(len(data), 2.0),
            atr_timeframe="3mo",
            atr_period=14,
        )
        summary = summarize_runs(runs, robustness_metric="ExpectancyATRP")

        self.assertEqual(set(runs["ExitMode"]), {"ATRP"})
        self.assertEqual(set(runs["StopParameter"]), {1.0})
        self.assertEqual(set(runs["TargetParameter"]), {2.0})
        self.assertTrue(
            np.isfinite(
                runs.loc[~runs["EntryMAEnabled"], "ExpectancyATRP"]
            ).all()
        )
        self.assertTrue((summary["RobustMetric"] == "ExpectancyATRP").all())


if __name__ == "__main__":
    unittest.main()
