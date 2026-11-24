from argparse import Namespace
from datetime import date
from io import BytesIO
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exit_strategy_monte_carlo_visualization import (
    aggregate_scenarios,
    create_scenario_heatmap,
    simulate_representative_scenario,
)


class ExitStrategyMonteCarloTests(unittest.TestCase):
    def test_aggregate_scenarios_support_percentiles_and_mean_alias(self):
        records = []
        for scenario, expectancy in enumerate([1.0, 3.0, 5.0, 7.0]):
            record = {
                "Seed": scenario,
                "Ticker": "QQQ",
                "TradingTimeframe": "1d",
                "EntryMAEnabled": False,
                "ExitMode": "Percent",
                "ParameterUnit": "Percent of entry price",
                "ATRTimeframe": "N/A",
                "ATRPeriod": float("nan"),
                "StopParameter": 5.0,
                "TargetParameter": 10.0,
            }
            record.update(
                {
                    metric: 1.0
                    for metric in (
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
                        "ExpectancyATRP",
                        "ExpectancyATRPPerBar",
                        "ExpectancyTRP",
                        "ExpectancyTRPPerBar",
                        "ExpectancyRPerBar",
                        "ExpectancyRPer100Bars",
                        "AverageEntryATRP",
                        "AverageEntryTRP",
                        "AverageStopDistancePct",
                        "AverageTargetDistancePct",
                        "AverageBarsHeld",
                        "StopExitPct",
                        "TargetExitPct",
                        "MovingAverageExitPct",
                        "EndOfDataExitPct",
                        "AverageCommissionPct",
                    )
                }
            )
            record["ExpectancyR"] = expectancy
            records.append(record)

        runs = pd.DataFrame(records)
        p25 = aggregate_scenarios(runs, "p25")
        average = aggregate_scenarios(runs, "average")

        self.assertEqual(p25.loc[0, "ExpectancyR"], 2.5)
        self.assertEqual(average.loc[0, "ExpectancyR"], 4.0)
        self.assertEqual(p25.loc[0, "ScenarioCount"], 4)
        self.assertEqual(p25.loc[0, "ScenariosWithTrades"], 4)

    def test_representative_scenario_handles_indexed_atrp_and_renders_heatmap(self):
        closes = np.array(
            [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 14, 15, 14, 16],
            dtype=float,
        )
        data = pd.DataFrame(
            {
                "Open": closes,
                "High": closes + 0.5,
                "Low": closes - 0.5,
                "Close": closes,
            },
            index=pd.date_range("2020-01-01", periods=len(closes)),
        )
        settings = Namespace(
            entry_count=3,
            sma_period=2,
            direction="Long",
            ma_entry_enabled=False,
            ma_exit=False,
            use_atrp=True,
            commission=0.0,
            slippage=0.0,
            conservative=True,
        )

        entries, trades, directional = simulate_representative_scenario(
            data=data,
            scenario=3,
            args=settings,
            stops=[5.0],
            targets=[5.0, 10.0],
            atrp_values=pd.Series(
                np.full(len(data), 2.0),
                index=data.index,
            ),
        )
        summary = pd.DataFrame(
            {
                "EntryMAEnabled": [False, False, False, False],
                "StopLossPct": [5.0, 5.0, -1.0, -1.0],
                "TargetPct": [5.0, 10.0, 5.0, 10.0],
                "ExpectancyR": [-0.2, 1.4, -0.5, 0.8],
                "WinRatePct": [50.0, 75.0, 40.0, 60.0],
            }
        )
        figure = create_scenario_heatmap(
            results=summary,
            stops=[5.0, -1.0],
            targets=[5.0, 10.0],
            ticker="TEST",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 15),
            number_of_entries=len(entries),
            requested_entries=3,
            top_metric="ExpectancyR",
            bottom_metric="WinRatePct",
            color_metric="ExpectancyR",
            vmin=0.0,
            vmax=1.0,
            win_rate_minimum=67.0,
            expectancy_r_minimum=0.3,
        )
        try:
            image = BytesIO()
            figure.savefig(image, format="png")
            self.assertTrue(image.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(figure.axes[0].images[0].norm.vmin, 0.0)
            self.assertEqual(figure.axes[0].images[0].norm.vmax, 1.0)
            self.assertEqual(
                [tick.get_text() for tick in figure.axes[0].get_xticklabels()],
                ["5.0", "NS"],
            )
            self.assertEqual(len(entries), 3)
            self.assertEqual(len(trades), 6)
            self.assertEqual(len(directional), 6)
            self.assertTrue((entries["EntryATRP"] == 2.0).all())
        finally:
            plt.close(figure)

    def test_representative_scenario_limits_simultaneous_trades(self):
        closes = np.arange(10, 18, dtype=float)
        data = pd.DataFrame(
            {
                "Open": closes,
                "High": closes + 0.5,
                "Low": closes - 0.5,
                "Close": closes,
            },
            index=pd.date_range("2020-01-01", periods=len(closes)),
        )
        settings = Namespace(
            trades_per_year=4,
            sma_period=2,
            direction="Long",
            ma_entry_enabled=False,
            ma_exit=False,
            use_atrp=False,
            use_trp=False,
            commission=0.0,
            slippage=0.0,
            conservative=True,
            max_open_trades=1,
        )

        entries, trades, directional = simulate_representative_scenario(
            data=data,
            scenario=3,
            args=settings,
            stops=[-1],
            targets=[-1],
            atrp_values=None,
        )

        self.assertEqual(len(entries), 4)
        self.assertEqual(len(trades), 1)
        self.assertEqual(len(directional), 3)


if __name__ == "__main__":
    unittest.main()
