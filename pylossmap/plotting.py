from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def plot_loss_map(
    data: pd.Series,
    meta: pd.DataFrame,
    types: List[str] = ["cold", "warm", "coll", "xrp"],
    types_to_colour: Optional[Dict[str, str]] = None,
    xtick_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (20, 10),
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (1e-7, 1e1),
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a loss map from data.

    Args:
        data: Series with as index the blms and value the measurement [Gy/s].
        meta: BLM metadata.
        types: which types to plot.
        types_to_colour: map of the types to colours.
        xtick_labels: list of BLM names to add to the x axis.
        figsize: figure size in inches.
        title: figure title.
        xlim: x axis limits.
        ylim: y axis limits.
        ax: ax on which to plot.
        **kwargs: forwarded to plt.bar.

    Returns:
        Figure and ax objects of the figure.
    """

    if types_to_colour is None:
        types_to_colour = {
            "cold": "b",
            "warm": "r",
            "coll": "k",
            "xrp": "g",
            "other": "m",
        }

    if title is None:
        try:
            title = (
                f"Beam Mode: {data.name[0]}, "
                f'Timestamp: {data.name[2].strftime("%Y-%m-%d %H:%M:%S:%f")}'
            )
        except Exception:
            pass

    if xlim is None:
        xlim = [meta["dcum"].min(), meta["dcum"].max()]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for typ in types:
        mask = meta["type"] == typ
        if not mask.any():
            continue
        ax.stem(
            meta[mask]["dcum"],
            data[mask],
            markerfmt=" ",
            linefmt=f"{types_to_colour[typ]}-",
            use_line_collection=True,
            label=typ,
            **kwargs,
        )
    ax.legend()

    ax.set_yscale("log")
    ax.yaxis.grid(which="both")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.set_yticks(
            [10 ** p for p in range(int(np.log10(ylim[0])), int(np.log10(ylim[1])) + 1)]
        )
    else:
        tick_mask = data > 0
        ax.set_yticks(
            [
                10 ** p
                for p in range(
                    int(np.log10(data[tick_mask].min())),
                    int(np.log10(data[tick_mask].max())) + 1,
                )
            ]
        )
    if xlim is not None:
        # xlim = [0, meta['coord'].max()]
        # meta['coord'].max()
        ax.set_xlim(xlim)

    if title is not None:
        ax.set_title(title)

    if xtick_labels is not None:
        ax.set_xticks([meta[c]["dcum"] for c in xtick_labels])
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=7)
    return fig, ax


def plot_waterfall(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    title: str = None,
    figsize: Tuple[float, float] = (20, 10),
    min_max_quantile: float = 0.85,
    ax: Optional[plt.Axes] = None,
    fill_missing: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots a water plot of the data.

    Args:
        data: DataFrame containing BLM data.
        meta: BLM metadata.
        title: figure title.
        figsize: figure size in inches.
        min_max_quantile: colormap min/max threshold.

    Returns:
        Figure and ax objects of the plot.
    """
    if isinstance(min_max_quantile, tuple):
        min_quant = min_max_quantile[0]
        max_quant = min_max_quantile[1]
    elif isinstance(min_max_quantile, float):
        min_quant = 1 - min_max_quantile
        max_quant = min_max_quantile

    # set the time axis labels
    y_lims = [
        data.index.get_level_values("timestamp")[-1],
        data.index.get_level_values("timestamp")[0],
    ]
    y_lims = mdates.date2num(y_lims)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if fill_missing:
        # make sure the columns are sorted in incresing s coord.
        missing = set(meta.index.tolist()) - set(data.columns)
        if missing:
            data[list(missing)] = np.nan
    data = data[meta.loc[data.columns].sort_values("dcum").index.tolist()]

    # Set some xaxis to s coord
    x_lims = [0, meta["dcum"].max()]
    extent = [x_lims[0], x_lims[1], y_lims[0], y_lims[1]]
    ax.imshow(
        data,
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        vmin=data.min().quantile(min_quant),
        vmax=data.max().quantile(max_quant),
    )
    ax.yaxis_date(tz="Europe/Zurich")

    if title is not None:
        ax.set_title(title)

    return fig, ax
