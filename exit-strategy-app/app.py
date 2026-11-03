"""Streamlit entry point for the Exit Strategy application.

This module coordinates ticker and preference state, runs simulations, and
selects either the compatible bundled QQQ result or newly calculated results.
Data and trade calculations, simulation orchestration, bundled-result
compatibility, visualization, and page rendering live in adjacent
``exit_strategy_*`` modules.

The Market Analysis Suite executes this file with ``runpy.run_path``. Imports
therefore temporarily add this app's directory to ``sys.path``; the suite's
session-state keys and ``exit_*`` widget keys remain the integration contract.
"""

from pathlib import Path
import sys

import streamlit as st
import yfinance as yf


_APP_DIRECTORY = str(Path(__file__).resolve().parent)
_ADDED_APP_DIRECTORY = _APP_DIRECTORY not in sys.path

# runpy from the suite does not add this app's directory to sys.path.
if _ADDED_APP_DIRECTORY:
    sys.path.insert(0, _APP_DIRECTORY)

try:
    from exit_strategy_core import parse_values
    from exit_strategy_simulation import run_simulation_now
    from exit_strategy_startup import (
        STARTUP_TICKER,
        load_startup_result,
        result_has_directional_entries,
        result_ticker,
        startup_result_matches_settings,
        upgrade_startup_result,
    )
    from exit_strategy_ui import render_results, render_sidebar
finally:
    if _ADDED_APP_DIRECTORY:
        sys.path.remove(_APP_DIRECTORY)


IN_SUITE = st.session_state.get(
    "_market_analysis_suite",
    False,
)


if not IN_SUITE:
    st.set_page_config(
        page_title="Exit Strategy",
        page_icon="📊",
        layout="wide",
    )


# ============================================================
# Exit Strategy preference persistence
# ============================================================

def persist_exit_preferences() -> None:
    """
    Save Exit Strategy widget values for the parent suite.

    Copy ``exit_*`` values to suite preferences before switching apps. Widget
    state is read only here because Streamlit does not allow assigning to a
    widget-owned key after that widget has been created.
    """

    st.session_state.pop(
        "_exit_output",
        None,
    )

    if not st.session_state.get(
        "_market_analysis_suite",
        False,
    ):
        return

    preferences = dict(
        st.session_state.get(
            "_suite_saved_preferences",
            {},
        )
    )

    # --------------------------------------------------------
    # Synchronize ticker
    #
    # Read exit_ticker but do not assign back to exit_ticker.
    # --------------------------------------------------------

    if "exit_ticker" in st.session_state:

        ticker_value = (
            str(
                st.session_state[
                    "exit_ticker"
                ]
            )
            .strip()
            .upper()
        )

        if ticker_value:

            st.session_state[
                "_suite_shared_ticker"
            ] = ticker_value

    # --------------------------------------------------------
    # Synchronize ticker-search state
    # --------------------------------------------------------

    if (
        "exit_search_ticker"
        in st.session_state
    ):

        st.session_state[
            "_suite_ticker_search"
        ] = st.session_state[
            "exit_search_ticker"
        ]

    if (
        "exit_search_ticker_label"
        in st.session_state
    ):

        st.session_state[
            "_suite_ticker_search_label"
        ] = st.session_state[
            "exit_search_ticker_label"
        ]

    # --------------------------------------------------------
    # Save every persistent Exit Strategy widget
    # --------------------------------------------------------

    for key in list(
        st.session_state.keys()
    ):

        if key.startswith("exit_"):

            preferences[key] = (
                st.session_state[key]
            )

    st.session_state[
        "_suite_saved_preferences"
    ] = preferences


# ============================================================
# Ticker selection callback
# ============================================================

def select_exit_ticker(
    symbol: str,
) -> None:
    """
    Select a ticker from search results and synchronize suite state.

    Streamlit runs this callback before creating the next set of widgets, so
    updating the ticker widget's session-state key is permitted.
    """

    symbol = (
        symbol
        .strip()
        .upper()
    )

    st.session_state[
        "exit_ticker"
    ] = symbol

    st.session_state[
        "exit_search_ticker"
    ] = ""

    st.session_state[
        "exit_search_ticker_label"
    ] = ""

    if IN_SUITE:

        st.session_state[
            "_suite_shared_ticker"
        ] = symbol

        st.session_state[
            "_suite_ticker_search"
        ] = ""

        st.session_state[
            "_suite_ticker_search_label"
        ] = ""

    persist_exit_preferences()


# ============================================================
# Ticker search
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def search_tickers(
    query: str,
) -> list[dict[str, str]]:
    """Return cached Yahoo Finance ticker matches for a search query."""

    quotes = yf.Search(
        query,
        max_results=8,
        news_count=0,
        lists_count=0,
        include_cb=False,
    ).quotes

    matches = []

    for quote in quotes:

        symbol = quote.get(
            "symbol"
        )

        if not symbol:
            continue

        matches.append(
            {
                "symbol": str(
                    symbol
                ),

                "name": str(
                    quote.get(
                        "longname"
                    )
                    or quote.get(
                        "shortname"
                    )
                    or symbol
                ),

                "exchange": str(
                    quote.get(
                        "exchDisp"
                    )
                    or quote.get(
                        "exchange"
                    )
                    or ""
                ),
            }
        )

    return matches


# ============================================================
# Heading
# ============================================================

st.title(
    "Exit Strategy"
)

st.write(
    "Test stop-loss and target combinations "
    "using random historical entries in the direction "
    "of the selected simple moving average."
)


# ============================================================
# Initialize widget state BEFORE widget creation
# ============================================================

if "exit_ticker" not in st.session_state:

    st.session_state[
        "exit_ticker"
    ] = st.session_state.get(
        "_suite_shared_ticker",
        STARTUP_TICKER,
    )


if (
    "exit_search_ticker"
    not in st.session_state
):

    st.session_state[
        "exit_search_ticker"
    ] = ""


if (
    "exit_search_ticker_label"
    not in st.session_state
):

    st.session_state[
        "exit_search_ticker_label"
    ] = ""


# ============================================================
# Sidebar
# ============================================================

sidebar_settings = render_sidebar(
    persist_exit_preferences=persist_exit_preferences,
    select_exit_ticker=select_exit_ticker,
    search_tickers=search_tickers,
)

bottom_metric = sidebar_settings["bottom_metric"]
color_maximum = sidebar_settings["color_maximum"]
color_metric = sidebar_settings["color_metric"]
color_minimum = sidebar_settings["color_minimum"]
commission = sidebar_settings["commission"]
conservative = sidebar_settings["conservative"]
end_date = sidebar_settings["end_date"]
highlight_expectancy_r = sidebar_settings["highlight_expectancy_r"]
highlight_win_rate = sidebar_settings["highlight_win_rate"]
interval = sidebar_settings["interval"]
ma_enabled = sidebar_settings["ma_enabled"]
ma_exit_enabled = sidebar_settings["ma_exit_enabled"]
number_of_entries = sidebar_settings["number_of_entries"]
rank_metric = sidebar_settings["rank_metric"]
run_simulation = sidebar_settings["run_simulation"]
seed = sidebar_settings["seed"]
slippage = sidebar_settings["slippage"]
sma_period = sidebar_settings["sma_period"]
start_date = sidebar_settings["start_date"]
stop_text = sidebar_settings["stop_text"]
target_text = sidebar_settings["target_text"]
ticker = sidebar_settings["ticker"]
top_metric = sidebar_settings["top_metric"]
trade_direction = sidebar_settings["trade_direction"]


# ============================================================
# Input validation
# ============================================================

valid_inputs = True


if not ticker:

    st.warning(
        "Enter a ticker symbol."
    )

    valid_inputs = False


elif start_date >= end_date:

    st.warning(
        "The start date must be "
        "earlier than the end date."
    )

    valid_inputs = False


# ============================================================
# Run simulation only after button click
# ============================================================

if (
    valid_inputs
    and run_simulation
):

    try:

        stops = parse_values(
            stop_text
        )

        targets = parse_values(
            target_text
        )


        if (
            not stops
            or not targets
        ):

            raise ValueError(
                "Provide at least one stop "
                "and one target value."
            )


        persist_exit_preferences()


        st.session_state[
            "_exit_output"
        ] = run_simulation_now(
            ticker=
                ticker,

            start_date=
                start_date,

            end_date=
                end_date,

            interval=
                interval,

            number_of_entries=
                number_of_entries,

            seed=
                seed,

            commission=
                commission,

            slippage=
                slippage,

            conservative=
                conservative,

            stops=
                stops,

            targets=
                targets,

            sma_period=
                sma_period,

            trade_direction=
                trade_direction,

            ma_enabled=
                ma_enabled,

            ma_exit_enabled=
                ma_exit_enabled,

            highlight_win_rate=
                highlight_win_rate,

            highlight_expectancy_r=
                highlight_expectancy_r,
        )


    except Exception as exc:

        st.error(
            str(exc)
        )


# ============================================================
# Load QQQ startup result only when current ticker is QQQ
# ============================================================

if (
    "_exit_output"
    not in st.session_state
):

    if (
        ticker
        == STARTUP_TICKER
    ):

        startup_output = (
            load_startup_result()
        )


        if (
            startup_output
            is not None

            and

            startup_result_matches_settings(
                output=startup_output,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                number_of_entries=number_of_entries,
                seed=seed,
                commission=commission,
                slippage=slippage,
                conservative=conservative,
                stop_text=stop_text,
                target_text=target_text,
                trade_direction=trade_direction,
                ma_enabled=ma_enabled,
                ma_exit_enabled=ma_exit_enabled,
            )
        ):

            st.session_state[
                "_exit_output"
            ] = startup_output

        elif (
            startup_output
            is None
        ):

            st.info(
                "No precomputed QQQ startup result "
                "is available. Click Run simulation."
            )

        else:

            st.info(
                "The bundled QQQ result does not match the "
                "current settings. Click Run simulation to "
                "generate matching results."
            )


    else:

        st.info(
            f"Ticker is {ticker}. "
            "The bundled startup results are for QQQ "
            "and are therefore hidden. "
            f"Click Run simulation to generate "
            f"results for {ticker}."
        )


# ============================================================
# Determine whether output matches current ticker
# ============================================================

output_to_display = (
    st.session_state.get(
        "_exit_output"
    )
)

if (
    output_to_display is not None

    and

    not result_has_directional_entries(
        output_to_display
    )

    and

    startup_result_matches_settings(
        output=output_to_display,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        number_of_entries=number_of_entries,
        seed=seed,
        commission=commission,
        slippage=slippage,
        conservative=conservative,
        stop_text=stop_text,
        target_text=target_text,
        trade_direction=trade_direction,
        ma_enabled=ma_enabled,
        ma_exit_enabled=ma_exit_enabled,
    )
):

    output_to_display = upgrade_startup_result(
        output_to_display
    )


output_ticker = (
    result_ticker(
        output_to_display
    )
)


show_results = (
    valid_inputs

    and

    output_to_display
    is not None

    and

    result_has_directional_entries(
        output_to_display
    )

    and

    output_ticker
    == ticker
)


if (
    valid_inputs

    and

    output_to_display
    is not None

    and

    not result_has_directional_entries(
        output_to_display
    )

):

    st.info(
        "The bundled result is for QQQ daily data from "
        "2010-01-01 to 2026-10-30, 1,000 Long entries with "
        "seed 27, SMA entry and exit disabled, zero "
        "commission/slippage, and the default stop/target "
        "values. Select those settings to display it, or "
        "click Run simulation."
    )


if (
    valid_inputs

    and

    output_to_display
    is not None

    and

    result_has_directional_entries(
        output_to_display
    )

    and

    output_ticker
    != ticker
):

    st.info(
        f"Ticker changed from "
        f"{output_ticker or 'the previous ticker'} "
        f"to {ticker}. "
        f"Click Run simulation to generate "
        f"results for {ticker}."
    )


# ============================================================
# Display results
# ============================================================

render_results(
    show_results=show_results,
    output_to_display=output_to_display,
    run_simulation=run_simulation,
    top_metric=top_metric,
    bottom_metric=bottom_metric,
    color_metric=color_metric,
    color_minimum=color_minimum,
    color_maximum=color_maximum,
    highlight_win_rate=highlight_win_rate,
    highlight_expectancy_r=highlight_expectancy_r,
    rank_metric=rank_metric,
    startup_ticker=STARTUP_TICKER,
)
