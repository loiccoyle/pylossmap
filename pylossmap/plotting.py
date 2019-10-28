import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def plot_loss_map(data,
                  meta,
                  types=['cold', 'warm', 'coll', 'xrp'],
                  types_to_colour=None,
                  xtick_labels=None,
                  figsize=(20, 10),
                  title=None,
                  x_lim=None,
                  y_lim=[1e-7, 1e1],
                  **kwargs):
    '''Plots a loss map from data.

    Args:
        data (Series): Series with as index the blms and value the
        measurement [Gy/s]
        xtick_labels (list, optional): list of BLM names to add to the x
        axis.
        figsize (tuple, optional): figure size in inches.
        title (str, optional): figure title.
        **kwargs: forwarded to plt.bar.

    Returns:
        Figure, Ax: figure and ax objects of the figure.
    '''

    if types_to_colour is None:
        types_to_colour = {'cold': 'b',
                           'warm': 'r',
                           'coll': 'k',
                           'xrp': 'g',
                           'other': 'm'}

    if title is None:
        try:
            title = f'Beam Mode: {data.name[0]}, Timestamp: {data.name[2].strftime("%Y-%m-%d %H:%M:%S:%f")}'
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=figsize)
    for typ in types:
        mask = meta['type'] == typ
        if not mask.any():
            continue
        ax.stem(meta[mask]['dcum'],
                data[mask],
                markerfmt=' ',
                linefmt=f'{types_to_colour[typ]}-',
                use_line_collection=True,
                label=typ,
                **kwargs)
    ax.legend()

    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_lim is not None:
        # x_lim = [0, meta['coord'].max()]
        ax.set_xlim(x_lim)

    ax.set_yscale('log')
    ax.yaxis.grid(which='both')

    if title is not None:
        ax.set_title(title)

    if xtick_labels is not None:
        ax.set_xticks([meta[c]['dcum'] for c in xtick_labels])
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=7)
    return fig, ax


def plot_waterfall(data,
                   meta,
                   title=None,
                   figsize=(20, 10),
                   min_max_quantile=0.95):
    """Plots a water plot of the data.

    Args:
        data (DataFrame): DataFrame containing BLM data.
        title (str, optional): figure title.
        figsize (tuple, optional): figure size in inches.
        min_max_quantile (tuple/float, optional): colormap min/max
        threshold.

    Returns:
        Figure, Ax: figure and ax objects of the plot.
    """
    if isinstance(min_max_quantile, tuple):
        min_quant = min_max_quantile[0]
        max_quant = min_max_quantile[1]
    elif isinstance(min_max_quantile, float):
        min_quant = min_max_quantile
        max_quant = min_max_quantile

    # set the time axis labels
    y_lims = [data.index.get_level_values('timestamp')[-1],
              data.index.get_level_values('timestamp')[0]]
    y_lims = mdates.date2num(y_lims)

    # Set some xaxis to s coord
    x_lims = [0, meta['dcum'].max()]
    extent = [x_lims[0], x_lims[1], y_lims[0], y_lims[1]]

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(data,
              aspect='auto',
              extent=extent,
              vmin=data.min().quantile(min_quant),
              vmax=data.max().quantile(max_quant))
    ax.yaxis_date(tz='Europe/Zurich')

    if title is not None:
        ax.set_title(title)

    return fig, ax
