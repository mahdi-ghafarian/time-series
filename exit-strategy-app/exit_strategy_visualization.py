"""Matplotlib visualization for stop/target comparison results.

The heatmap maps a selected result metric to cell color, shows two additional
metrics in each cell, and can highlight combinations meeting win-rate and
expectancy thresholds. The function returns a Matplotlib figure for the Streamlit
page to display and close.
"""

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def create_heatmap(
    results: pd.DataFrame,
    stops: list[float],
    targets: list[float],
    ticker: str,
    start_date: date,
    end_date: date,
    number_of_entries: int,
    top_metric: str,
    bottom_metric: str,
    color_metric: str,
    vmin,
    vmax,
    win_rate_minimum: float,
    expectancy_r_minimum: float,
):
    """Build the stop/target heatmap figure from comparison metrics.

    ``results`` must contain the selected metric columns plus ``WinRatePct``
    and ``ExpectancyR``. ``stops`` and ``targets`` define the displayed axis
    order; ``vmin`` and ``vmax`` optionally bound the color scale.
    """

    color_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=color_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    top_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=top_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    bottom_values = (
        results
        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values=bottom_metric,
        )
        .reindex(
            index=targets,
            columns=stops,
        )
    )

    highlighted = (
        pd.concat(
            [
                results[
                    [
                        "TargetPct",
                        "StopLossPct",
                    ]
                ],

                (
                    (
                        results[
                            "WinRatePct"
                        ]
                        > win_rate_minimum
                    )

                    &

                    (
                        results[
                            "ExpectancyR"
                        ]
                        > expectancy_r_minimum
                    )

                ).rename(
                    "Condition"
                ),
            ],

            axis=1,
        )

        .pivot(
            index="TargetPct",
            columns="StopLossPct",
            values="Condition",
        )

        .reindex(
            index=targets,
            columns=stops,
        )
    )

    matrix = (
        color_values
        .to_numpy(
            float
        )
    )

    finite_values = (
        matrix[
            np.isfinite(
                matrix
            )
        ]
    )

    if not finite_values.size:

        return None

    if vmin is None:

        vmin = float(
            np.min(
                finite_values
            )
        )

    if vmax is None:

        vmax = float(
            np.max(
                finite_values
            )
        )

    if vmin == vmax:

        padding = max(
            abs(vmin)
            * 0.01,
            1e-9,
        )

        vmin -= padding
        vmax += padding

    figure, axis = (
        plt.subplots(
            figsize=(
                max(
                    12,
                    len(stops)
                    * 0.85,
                ),

                max(
                    8,
                    len(targets)
                    * 0.55,
                ),
            )
        )
    )

    image = axis.imshow(
        matrix,
        cmap="RdYlGn",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    figure.colorbar(
        image,
        ax=axis,
    ).set_label(
        color_metric
    )

    axis.set_title(
        f"{ticker} | "
        f"{start_date} to {end_date} | "
        f"{number_of_entries:,} Random Entries\n"
        f"{top_metric} (top), "
        f"{bottom_metric} (bottom)"
    )

    axis.set_xlabel(
        "Stop-Loss (%)"
    )

    axis.set_ylabel(
        "Target (%)"
    )

    axis.set_xticks(
        np.arange(
            len(stops)
        )
    )

    axis.set_xticklabels(
        [
            "NS"
            if value == -1
            else str(value)

            for value
            in stops
        ]
    )

    axis.set_yticks(
        np.arange(
            len(targets)
        )
    )

    axis.set_yticklabels(
        [
            "NT"
            if value == -1
            else str(value)

            for value
            in targets
        ]
    )

    formats = {
        "ExpectancyR":
            ".2f",

        "ExpectancyPct":
            ".2f",

        "WinRatePct":
            ".1f",

        "RealizedRRR":
            ".2f",

        "Score":
            ".0f",
    }

    top_format = (
        formats.get(
            top_metric,
            ".2f",
        )
    )

    bottom_format = (
        formats.get(
            bottom_metric,
            ".2f",
        )
    )

    for row in range(
        len(targets)
    ):

        for column in range(
            len(stops)
        ):

            top_value = (
                top_values.iloc[
                    row,
                    column,
                ]
            )

            bottom_value = (
                bottom_values.iloc[
                    row,
                    column,
                ]
            )

            color_value = (
                matrix[
                    row,
                    column,
                ]
            )

            if (
                np.isfinite(
                    top_value
                )
                and
                np.isfinite(
                    bottom_value
                )
            ):

                label = (
                    f"{format(top_value, top_format)}\n"
                    f"{format(bottom_value, bottom_format)}"
                )

            else:

                label = "N/A"

            if np.isfinite(
                color_value
            ):

                rgba = (
                    image.cmap(
                        image.norm(
                            color_value
                        )
                    )
                )

            else:

                rgba = (
                    1,
                    1,
                    1,
                    1,
                )

            brightness = (
                0.299
                * rgba[0]

                + 0.587
                * rgba[1]

                + 0.114
                * rgba[2]
            )

            axis.text(
                column,
                row,
                label,

                ha="center",
                va="center",

                color=(
                    "black"
                    if brightness
                    > 0.55
                    else "white"
                ),

                fontsize=8,
            )

            if bool(
                highlighted.iloc[
                    row,
                    column,
                ]
            ):

                axis.add_patch(
                    Rectangle(
                        (
                            column - 0.5,
                            row - 0.5,
                        ),

                        1,
                        1,

                        fill=False,

                        edgecolor=
                            "black",

                        linewidth=2,
                    )
                )

    axis.set_xticks(
        np.arange(
            -0.5,
            len(stops),
            1,
        ),
        minor=True,
    )

    axis.set_yticks(
        np.arange(
            -0.5,
            len(targets),
            1,
        ),
        minor=True,
    )

    axis.grid(
        which="minor",
        color="gray",
        linewidth=0.4,
        alpha=0.4,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    figure.tight_layout()

    return figure
