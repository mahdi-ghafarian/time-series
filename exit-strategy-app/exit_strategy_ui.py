"""Streamlit controls and result views for the Exit Strategy page.

``render_sidebar`` owns the app's ``exit_*`` widgets and returns their current
values to the entrypoint. Its callbacks are passed in so ticker search and
preference persistence remain coordinated with the parent Market Analysis Suite.
``render_results`` owns summary, heatmap, directional statistics, detail tables,
and CSV downloads; it does not manage simulation or suite session state.
"""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from exit_strategy_core import calculate_directional_statistics
from exit_strategy_visualization import create_heatmap


DEFAULT_STOPS = [
    1, 2, 3, 4, 5, 7.5, 10, 12.5, 15,
    20, 25, 30, 35, 40, 45, 50, -1,
]


DEFAULT_TARGETS = [
    1, 2, 3, 4, 5, 7.5, 10, 12.5, 15,
    20, 25, 30, 35, 40, 45, 50, 100, -1,
]


METRICS = [
    "StopLossPct",
    "TargetPct",
    "NumberOfTrades",
    "WinRatePct",
    "AverageWinPct",
    "AverageLossPct",
    "RealizedRRR",
    "BreakEvenWinRatePct",
    "PlannedRRR",
    "PlannedBreakEvenWinRatePct",
    "ExpectancyPct",
    "ExpectancyR",
    "AverageBarsHeld",
    "StopExitPct",
    "TargetExitPct",
    "MovingAverageExitPct",
    "EndOfDataExitPct",
    "AverageCommissionPct",
    "Score",
]


def render_sidebar(
    persist_exit_preferences,
    select_exit_ticker,
    search_tickers,
) -> dict:
    """Render the sidebar and return simulation and display settings."""

    with st.sidebar:

        st.header(
            "Data"
        )

        # --------------------------------------------------------
        # Search
        # --------------------------------------------------------

        ticker_query = (
            st.text_input(
                "Search for ticker",

                placeholder=(
                    "Type a ticker "
                    "or company name"
                ),

                key=
                    "exit_search_ticker",

                live=
                    "500ms",

                on_change=
                    persist_exit_preferences,
            )
            .strip()
        )


        selected_label = (
            st.session_state.get(
                "exit_search_ticker_label",
                "",
            )
        )


        if (
            ticker_query
            != selected_label

            and

            len(
                ticker_query
            ) >= 2
        ):

            search_failed = False

            try:

                with st.spinner(
                    "Searching Yahoo Finance..."
                ):

                    ticker_matches = (
                        search_tickers(
                            ticker_query
                        )
                    )

            except Exception as exc:

                search_failed = True

                ticker_matches = []

                st.warning(
                    "Yahoo Finance ticker "
                    "search failed: "
                    f"{exc}. "
                    "You can still enter "
                    "a ticker manually."
                )


            if ticker_matches:

                for match in ticker_matches:

                    symbol = (
                        match[
                            "symbol"
                        ]
                    )

                    label = (
                        f"{symbol} | "
                        f"{match['name']} | "
                        f"{match['exchange'] or 'Unknown exchange'}"
                    )

                    st.button(
                        label,

                        key=(
                            "_exit_ticker_result_"
                            f"{symbol}"
                        ),

                        on_click=
                            select_exit_ticker,

                        args=(
                            symbol,
                        ),

                        width=
                            "stretch",
                    )

            elif not search_failed:

                st.caption(
                    "No Yahoo Finance matches found."
                )


        elif (
            ticker_query
            != selected_label

            and

            ticker_query
        ):

            st.caption(
                "Enter at least two "
                "characters to search."
            )


        # --------------------------------------------------------
        # Ticker
        #
        # No value= argument because the state key was initialized
        # before widget creation.
        # --------------------------------------------------------

        ticker = (
            st.text_input(
                "Ticker",

                help=(
                    "Examples: AAPL, MSFT, SPY, "
                    "QQQ, BTC-USD, or ^GSPC."
                ),

                key=
                    "exit_ticker",

                on_change=
                    persist_exit_preferences,
            )
            .strip()
            .upper()
        )


        # --------------------------------------------------------
        # Simulation settings
        # --------------------------------------------------------

        interval = (
            st.selectbox(
                "Interval",

                [
                    "1d",
                    "1wk",
                    "1mo",
                ],

                index=0,

                key=
                    "exit_interval",

                on_change=
                    persist_exit_preferences,
            )
        )


        st.subheader(
            "Date range"
        )


        start_date = (
            st.date_input(
                "Start date",

                value=date(
                    2010,
                    1,
                    1,
                ),

                key=
                    "exit_start_date",

                on_change=
                    persist_exit_preferences,
            )
        )


        end_date = (
            st.date_input(
                "End date",

                value=date(
                    2026,
                    10,
                    30,
                ),

                key=
                    "exit_end_date",

                on_change=
                    persist_exit_preferences,
            )
        )


        st.divider()


        st.header(
            "Simulation"
        )


        sma_period = (
            st.number_input(
                "SMA period (bars)",

                min_value=1,

                value=200,

                step=1,

                help=(
                    "Random entries are long when the close is "
                    "above this simple moving average and short "
                    "when it is below."
                ),

                key=
                    "exit_sma_period",

                on_change=
                    persist_exit_preferences,
            )
        )

        ma_enabled = (
            st.checkbox(
                "Use SMA for entry direction",

                value=False,

                help=(
                    "When enabled, entries are long above the SMA "
                    "and short below it."
                ),

                key=
                    "exit_ma_enabled",

                on_change=
                    persist_exit_preferences,
            )
        )

        ma_exit_enabled = (
            st.checkbox(
                "Exit on opposite SMA crossover",

                value=False,

                help=(
                    "Close an open trade when price crosses the "
                    "SMA against its direction. Stop-loss and "
                    "target exits remain active and take priority."
                ),

                key=
                    "exit_ma_exit_enabled",

                on_change=
                    persist_exit_preferences,
            )
        )


        trade_direction = (
            st.selectbox(
                "Trade direction",

                [
                    "Both",
                    "Long",
                    "Short",
                ],

                index=1,

                help=(
                    "When SMA logic is on, entries follow the SMA. "
                    "When it is off, Both randomly assigns long "
                    "or short to each entry."
                ),

                key=
                    "exit_trade_direction",

                on_change=
                    persist_exit_preferences,
            )
        )


        number_of_entries = (
            st.number_input(
                "Random entries",

                min_value=1,

                value=1000,

                step=100,

                key=
                    "exit_number_of_entries",

                on_change=
                    persist_exit_preferences,
            )
        )


        seed = (
            st.number_input(
                "Random seed",

                min_value=0,

                value=27,

                step=1,

                key=
                    "exit_seed",

                on_change=
                    persist_exit_preferences,
            )
        )


        commission = (
            st.number_input(
                "Commission (%)",

                min_value=0.0,

                value=0.0,

                step=0.01,

                format=
                    "%.3f",

                key=
                    "exit_commission",

                on_change=
                    persist_exit_preferences,
            )
        )


        slippage = (
            st.number_input(
                "Slippage (%)",

                min_value=0.0,

                value=0.0,

                step=0.01,

                format=
                    "%.3f",

                key=
                    "exit_slippage",

                on_change=
                    persist_exit_preferences,
            )
        )


        conservative = (
            st.checkbox(
                "Stop first when stop and "
                "target hit same candle",

                value=True,

                key=
                    "exit_conservative",

                on_change=
                    persist_exit_preferences,
            )
        )


        st.divider()


        # --------------------------------------------------------
        # Heatmap controls
        # --------------------------------------------------------

        st.header(
            "Heatmap"
        )


        top_metric = (
            st.selectbox(
                "Top metric",

                METRICS,

                index=
                    METRICS.index(
                        "ExpectancyR"
                    ),

                key=
                    "exit_top_metric",

                on_change=
                    persist_exit_preferences,
            )
        )


        bottom_metric = (
            st.selectbox(
                "Bottom metric",

                METRICS,

                index=
                    METRICS.index(
                        "WinRatePct"
                    ),

                key=
                    "exit_bottom_metric",

                on_change=
                    persist_exit_preferences,
            )
        )


        color_metric = (
            st.selectbox(
                "Color metric",

                METRICS,

                index=
                    METRICS.index(
                        "ExpectancyR"
                    ),

                key=
                    "exit_color_metric",

                on_change=
                    persist_exit_preferences,
            )
        )


        automatic_color_scale = (
            st.checkbox(
                "Auto color scale "
                "(use min/max)",

                value=False,

                key=
                    "exit_auto_color_scale",

                on_change=
                    persist_exit_preferences,
            )
        )


        (
            color_column_1,
            color_column_2,
        ) = st.columns(2)


        color_minimum = (
            color_column_1
            .number_input(
                "Color min",

                value=0.0,

                step=0.1,

                disabled=
                    automatic_color_scale,

                key=
                    "exit_color_minimum",

                on_change=
                    persist_exit_preferences,
            )
        )


        color_maximum = (
            color_column_2
            .number_input(
                "Color max",

                value=1.0,

                step=0.1,

                disabled=
                    automatic_color_scale,

                key=
                    "exit_color_maximum",

                on_change=
                    persist_exit_preferences,
            )
        )


        if automatic_color_scale:

            color_minimum = None
            color_maximum = None


        (
            highlight_column_1,
            highlight_column_2,
        ) = st.columns(2)


        highlight_win_rate = (
            highlight_column_1
            .number_input(
                "Highlight WR min",

                value=50.0,

                step=1.0,

                key=
                    "exit_highlight_win_rate",

                on_change=
                    persist_exit_preferences,
            )
        )


        highlight_expectancy_r = (
            highlight_column_2
            .number_input(
                "Highlight ExpR min",

                value=0.5,

                step=0.1,

                key=
                    "exit_highlight_expectancy_r",

                on_change=
                    persist_exit_preferences,
            )
        )


        rank_metric = (
            st.selectbox(
                "Rank metric",

                METRICS,

                index=
                    METRICS.index(
                        "ExpectancyR"
                    ),

                key=
                    "exit_rank_metric",

                on_change=
                    persist_exit_preferences,
            )
        )


        st.divider()


        # --------------------------------------------------------
        # Stop / target combinations
        # --------------------------------------------------------

        st.header(
            "Combinations"
        )


        with st.expander(
            "Stop / target values"
        ):

            stop_text = (
                st.text_area(
                    "Stops (%)",

                    ", ".join(
                        map(
                            str,
                            DEFAULT_STOPS,
                        )
                    ),

                    help=(
                        "Use -1 for "
                        "no stop (NS)."
                    ),

                    key=
                        "exit_stop_text",

                    on_change=
                        persist_exit_preferences,
                )
            )


            target_text = (
                st.text_area(
                    "Targets (%)",

                    ", ".join(
                        map(
                            str,
                            DEFAULT_TARGETS,
                        )
                    ),

                    help=(
                        "Use -1 for "
                        "no target (NT)."
                    ),

                    key=
                        "exit_target_text",

                    on_change=
                        persist_exit_preferences,
                )
            )


        st.divider()


        # No exit_* key because button trigger state must not be
        # saved and restored as a preference.
        run_simulation = (
            st.button(
                "Run simulation",

                type=
                    "primary",

                width=
                    "stretch",
            )
        )

    return {
        "bottom_metric": bottom_metric,
        "color_maximum": color_maximum,
        "color_metric": color_metric,
        "color_minimum": color_minimum,
        "commission": commission,
        "conservative": conservative,
        "end_date": end_date,
        "highlight_expectancy_r": highlight_expectancy_r,
        "highlight_win_rate": highlight_win_rate,
        "interval": interval,
        "ma_enabled": ma_enabled,
        "ma_exit_enabled": ma_exit_enabled,
        "number_of_entries": number_of_entries,
        "rank_metric": rank_metric,
        "run_simulation": run_simulation,
        "seed": seed,
        "slippage": slippage,
        "sma_period": sma_period,
        "start_date": start_date,
        "stop_text": stop_text,
        "target_text": target_text,
        "ticker": ticker,
        "top_metric": top_metric,
        "trade_direction": trade_direction
    }


def render_results(
    show_results,
    output_to_display,
    run_simulation,
    top_metric,
    bottom_metric,
    color_metric,
    color_minimum,
    color_maximum,
    highlight_win_rate,
    highlight_expectancy_r,
    rank_metric,
    startup_ticker,
) -> None:
    """Render simulation summaries, rankings, details, and downloads."""

    if show_results:

        (
            data,
            entries,
            results,
            trades,
            result_stops,
            result_targets,
            elapsed,
            result_ticker_value,
            result_start_date,
            result_end_date,
            requested_entries,
        ) = output_to_display


        if (
            result_ticker_value
            .strip()
            .upper()
            == startup_ticker

            and

            not run_simulation
        ):

            st.caption(
                "Showing precomputed QQQ startup results."
            )


        else:

            st.caption(
                "Showing simulation results for "
                f"{result_ticker_value}."
            )


        # ========================================================
        # Summary
        # ========================================================

        summary_columns = (
            st.columns(4)
        )


        summary_columns[
            0
        ].metric(
            "Bars",
            f"{len(data):,}",
        )


        summary_columns[
            1
        ].metric(
            "Trades used / requested",
            f"{len(entries):,} / {requested_entries:,}",
        )


        summary_columns[
            2
        ].metric(
            "Combinations",
            f"{len(results):,}",
        )


        summary_columns[
            3
        ].metric(
            "Run time",
            f"{elapsed:.1f}s",
        )


        # ========================================================
        # Heatmap
        # ========================================================

        st.subheader(
            "Heatmap"
        )


        heatmap = (
            create_heatmap(
                results=
                    results,

                stops=
                    result_stops,

                targets=
                    result_targets,

                ticker=
                    result_ticker_value,

                start_date=
                    result_start_date,

                end_date=
                    result_end_date,

                number_of_entries=
                    len(entries),

                requested_entries=
                    requested_entries,

                top_metric=
                    top_metric,

                bottom_metric=
                    bottom_metric,

                color_metric=
                    color_metric,

                vmin=
                    color_minimum,

                vmax=
                    color_maximum,

                win_rate_minimum=
                    highlight_win_rate,

                expectancy_r_minimum=
                    highlight_expectancy_r,
            )
        )


        if heatmap is not None:

            st.pyplot(
                heatmap,
                width="stretch",
            )

            plt.close(
                heatmap
            )


        # ========================================================
        # Ranked results
        # ========================================================

        ranked = (
            results.copy()
        )


        if (
            "Rank"
            in ranked.columns
        ):

            ranked = (
                ranked.drop(
                    columns=
                        "Rank"
                )
            )


        ranked = (
            ranked
            .sort_values(
                rank_metric,

                ascending=
                    False,

                na_position=
                    "last",
            )

            .reset_index(
                drop=True
            )
        )


        ranked.insert(
            0,
            "Rank",

            np.arange(
                1,
                len(ranked) + 1,
            ),
        )


        display_columns = [
            "Rank",
            "StopLossPct",
            "TargetPct",
            "NumberOfTrades",
            "ExpectancyR",
            "ExpectancyPct",
            "WinRatePct",
            "AverageWinPct",
            "AverageLossPct",
            "RealizedRRR",
            "StopExitPct",
            "TargetExitPct",
            "MovingAverageExitPct",
            "EndOfDataExitPct",
            "Score",
        ]


        with st.expander(
            (
                "Ranked combinations "
                f"(by {rank_metric})"
            ),

            expanded=False,
        ):

            st.dataframe(
                ranked[
                    display_columns
                ].round(
                    2
                ),

                hide_index=True,

                width=
                    "stretch",

                height=
                    600,
            )


        # ========================================================
        # Directional statistics
        # ========================================================

        directional_statistics = (
            calculate_directional_statistics(
                trades
            )
        )

        statistics_columns = [
            "StopLossPct",
            "TargetPct",
            "NumberOfTrades",
            "ExpectancyR",
            "ExpectancyPct",
            "WinRatePct",
            "AverageWinPct",
            "AverageLossPct",
            "RealizedRRR",
            "StopExitPct",
            "TargetExitPct",
            "MovingAverageExitPct",
            "EndOfDataExitPct",
        ]

        st.subheader(
            "Statistics by direction"
        )
        st.caption(
            "NumberOfTrades reflects the actual entries used in this run; "
            "the SMA filter can reduce this below the requested count."
        )

        all_statistics, long_statistics, short_statistics = (
            st.tabs(
                [
                    "All trades",
                    "Long trades",
                    "Short trades",
                ]
            )
        )

        for tab, direction in [
            (all_statistics, "All"),
            (long_statistics, "Long"),
            (short_statistics, "Short"),
        ]:

            with tab:

                direction_statistics = (
                    directional_statistics.loc[
                        directional_statistics[
                            "Direction"
                        ]
                        == direction
                    ]
                    .sort_values(
                        rank_metric,
                        ascending=False,
                        na_position="last",
                    )
                )

                st.dataframe(
                    direction_statistics[
                        statistics_columns
                    ].round(
                        2
                    ),

                    hide_index=True,

                    width=
                        "stretch",

                    height=
                        400,
                )


        # ========================================================
        # Detail tables
        # ========================================================

        with st.expander(
            "Random entry bars"
        ):

            st.dataframe(
                entries,

                hide_index=True,

                width=
                    "stretch",
            )


        with st.expander(
            "Trade details"
        ):

            st.dataframe(
                trades.round(
                    4
                ),

                hide_index=True,

                width=
                    "stretch",

                height=
                    600,
            )


        # ========================================================
        # Downloads
        # ========================================================

        st.subheader(
            "Download"
        )


        download_columns = (
            st.columns(4)
        )


        download_columns[
            0
        ].download_button(
            "Results CSV",

            ranked
            .round(4)
            .to_csv(
                index=False
            )
            .encode(),

            (
                f"{result_ticker_value.lower()}"
                "_stop_target_results.csv"
            ),

            mime=
                "text/csv",

            width=
                "stretch",
        )


        download_columns[
            1
        ].download_button(
            "Trade details CSV",

            trades
            .round(4)
            .to_csv(
                index=False
            )
            .encode(),

            (
                f"{result_ticker_value.lower()}"
                "_trade_details.csv"
            ),

            mime=
                "text/csv",

            width=
                "stretch",
        )


        download_columns[
            2
        ].download_button(
            "Directional stats CSV",

            directional_statistics
            .round(4)
            .to_csv(
                index=False
            )
            .encode(),

            (
                f"{result_ticker_value.lower()}"
                "_directional_statistics.csv"
            ),

            mime=
                "text/csv",

            width=
                "stretch",
        )


        download_columns[
            3
        ].download_button(
            "Random entries CSV",

            entries
            .to_csv(
                index=False
            )
            .encode(),

            (
                f"{result_ticker_value.lower()}"
                "_random_entries.csv"
            ),

            mime=
                "text/csv",

            width=
                "stretch",
        )
