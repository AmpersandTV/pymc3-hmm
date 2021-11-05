from typing import Any, Callable, Dict, List, Optional, Sequence, Text, Tuple, Union

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy.special import logsumexp

try:  # pragma: no cover
    import aesara.tensor as at
    from aesara.tensor.extra_ops import broadcast_shape
    from aesara.tensor.extra_ops import broadcast_to as at_broadcast_to
    from aesara.tensor.var import TensorVariable
except ImportError:  # pragma: no cover
    import theano.tensor as at
    from theano.tensor.extra_ops import broadcast_shape
    from theano.tensor.extra_ops import broadcast_to as at_broadcast_to
    from theano.tensor.var import TensorVariable


vsearchsorted = np.vectorize(np.searchsorted, otypes=[int], signature="(n),()->()")


def compute_steady_state(P):
    """Compute the steady state of a transition probability matrix.

    Parameters
    ----------
    P: TensorVariable
        A transition probability matrix for `M` states with shape `(1, M, M)`.

    Returns
    -------
    A tensor representing the steady state probabilities.
    """

    P = P[0]
    N_states = P.shape[-1]
    Lam = (at.eye(N_states) - P + at.ones((N_states, N_states))).T
    u = at.slinalg.solve(Lam, at.ones((N_states,)))
    return u


def compute_trans_freqs(states, N_states, counts_only=False):
    """Compute empirical state transition frequencies.

    Each row, `r`, corresponds to transitions from state `r` to each other
    state.

    Parameters
    ----------
    states: a pymc object or ndarray
        Vector sequence of states.
    N_states: int
        Total number of observable states.
    counts_only: boolean
        Return only the transition counts for each state.

    Returns
    -------
    res: ndarray
        Unless `counts_only` is `True`, return the empirical state transition
        frequencies; otherwise, return the transition counts for each state.
    """
    states_ = getattr(states, "values", states).ravel()

    if any(np.isnan(states_)):
        states_ = np.ma.masked_invalid(states_).astype(np.uint)
        states_mask = np.ma.getmask(states_)
        valid_pairs = ~states_mask[:-1] & ~states_mask[1:]
        state_pairs = (states_[:-1][valid_pairs], states_[1:][valid_pairs])
    else:
        state_pairs = (states_[:-1], states_[1:])

    counts = np.zeros((N_states, N_states))
    flat_coords = np.ravel_multi_index(state_pairs, counts.shape)
    counts.flat += np.bincount(flat_coords, minlength=counts.size)
    counts = np.nan_to_num(counts, nan=0)

    if counts_only:
        res = counts
    else:
        res = counts / np.maximum(1, counts.sum(axis=1, keepdims=True))

    return res


def tt_logsumexp(x, axis=None, keepdims=False):
    """Construct a Theano graph for a log-sum-exp calculation."""
    x_max_ = at.max(x, axis=axis, keepdims=True)

    if x_max_.ndim > 0:
        x_max_ = at.set_subtensor(x_max_[at.isinf(x_max_)], 0.0)
    elif at.isinf(x_max_):
        x_max_ = at.as_tensor(0.0)

    res = at.sum(at.exp(x - x_max_), axis=axis, keepdims=keepdims)
    res = at.log(res)

    if not keepdims:
        # SciPy uses the `axis` keyword here, but Theano doesn't support that.
        # x_max_ = tt.squeeze(x_max_, axis=axis)
        axis = np.atleast_1d(axis) if axis is not None else range(x_max_.ndim)
        x_max_ = x_max_.dimshuffle(
            [
                i
                for i in range(x_max_.ndim)
                if not x_max_.broadcastable[i] or i not in axis
            ]
        )

    return res + x_max_


def tt_logdotexp(A, b):
    """Construct a Theano graph for a numerically stable log-scale dot product.

    The result is more or less equivalent to `tt.log(tt.exp(A).dot(tt.exp(b)))`

    """
    pattern: List[Any] = list(range(A.ndim))
    A_bcast = A.dimshuffle(pattern + ["x"])

    sqz = False
    pattern_b: List[Any] = list(range(b.ndim))
    shape_b = ["x"] + pattern_b
    if len(shape_b) < 3:
        shape_b += ["x"]
        sqz = True

    b_bcast = b.dimshuffle(shape_b)
    res = tt_logsumexp(A_bcast + b_bcast, axis=1)
    return res.squeeze() if sqz else res


def logdotexp(A, b):
    """Compute a numerically stable log-scale dot product of NumPy values.

    The result is more or less equivalent to `np.log(np.exp(A).dot(np.exp(b)))`

    """
    sqz = False
    b_bcast = np.expand_dims(b, 0)
    if b.ndim < 2:
        b_bcast = np.expand_dims(b_bcast, -1)
        sqz = True

    A_bcast = np.expand_dims(A, -1)

    res = logsumexp(A_bcast + b_bcast, axis=1)
    return res.squeeze() if sqz else res


def tt_expand_dims(x, dims):
    """Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    This is a Theano equivalent of `numpy.expand_dims`.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or tuple of ints
        Position in the expanded axes where the new axis (or axes) is placed.

    """
    dim_range: List[Any] = list(range(x.ndim))
    for d in sorted(np.atleast_1d(dims), reverse=True):
        offset = 0 if d >= 0 else len(dim_range) + 1
        dim_range.insert(d + offset, "x")

    return x.dimshuffle(dim_range)


def tt_broadcast_arrays(*args: TensorVariable):
    """Broadcast any number of arrays against each other.

    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.

    """
    bcast_shape = broadcast_shape(*args)
    return tuple(at_broadcast_to(a, bcast_shape) for a in args)


def multilogit_inv(ys):
    """Compute the multilogit-inverse function for both NumPy and Theano arrays.

    In other words, this function maps `M`-many real numbers to an `M +
    1`-dimensional simplex.  This is a reduced version of the "softmax"
    function that's suitable for use with multinomial regression.

    Parameters
    ----------
    ys: ndarray or TensorVariable
        An array of "Linear" values (i.e. in `[-inf, inf]`), with length `M`,
        that are mapped to the `M + 1`-categories logistic scale.  The elements in
        the array corresponds to categories 1 through M, and the `M + 1`th category
        is the determined via "normalization".

    """
    if isinstance(ys, np.ndarray):
        lib = np
        lib_logsumexp = logsumexp
    else:
        lib = at
        lib_logsumexp = tt_logsumexp

    # exp_ys = lib.exp(ys)
    # res = lib.concatenate([exp_ys, lib.ones(tuple(ys.shape)[:-1] + (1,))], axis=-1)
    # res = res / (1 + lib.sum(exp_ys, axis=-1))[..., None]

    res = lib.concatenate([ys, lib.zeros(tuple(ys.shape)[:-1] + (1,))], axis=-1)
    res = lib.exp(res - lib_logsumexp(res, axis=-1, keepdims=True))
    return res


def plot_split_timeseries(
    data: pd.DataFrame,
    axes_split_data: Optional[List] = None,
    split_freq: str = "W",
    split_max: int = 5,
    twin_column_name: Optional[str] = None,
    twin_plot_kwargs: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, ...] = (15, 15),
    title: Optional[str] = None,
    plot_fn: Optional[Callable[..., Any]] = None,
    **plot_kwds,
):  # pragma: no cover
    """Plot long timeseries by splitting them across multiple rows using a given time frequency.

    This function requires the Pandas and Matplotlib libraries.

    Parameters
    ----------
    data: DataFrame
        The timeseries to be plotted.
    axes_split_data: List (optional)
        The result from a previous call of `plot_split_timeseries`.
    split_freq: str
        A Pandas time frequency string by which the series is split.
    split_max: int
        The maximum number of splits/rows to plot.
    twin_column_name: str (optional)
        If this value is non-`None`, it is used to indicate a column in `data`
        that will be plotted as a twin axis.
    twin_plot_kwargs: dict (optional)
        The arguments to `plot` for the twin axis, if any.
    plot_fn: callable (optional)
        The function used to plot each split/row.  The expected signature is
        `(ax, data, **kwargs)`.  The default implementation simply calls
        `ax.data`.

    Returns
    -------
    axes : list of axes
        The generated plot axes.
    """  # noqa: E501
    import matplotlib.transforms as mtrans
    from matplotlib.dates import AutoDateFormatter, AutoDateLocator

    if plot_fn is None:

        def plot_fn_(ax, data, **kwargs):
            return ax.plot(data, **kwargs)

        plot_fn = plot_fn_

    data = pd.DataFrame(data)

    if twin_column_name and len(data.columns) < 2:
        raise ValueError(
            "Option `twin_column` is only applicable for a two column `DataFrame`."
        )

    split_offset = pd.tseries.frequencies.to_offset(split_freq)

    if axes_split_data is None:
        grouper = pd.Grouper(freq=split_offset.freqstr, closed="left")
        obs_splits = [y_split for n, y_split in data.groupby(grouper)]

        if split_max:
            obs_splits = obs_splits[:split_max]

        n_partitions = len(obs_splits)

        fig, axes = plt.subplots(
            nrows=n_partitions, sharey=True, sharex=False, figsize=figsize
        )

        major_offset = mtrans.ScaledTranslation(0, -10 / 72.0, fig.dpi_scale_trans)

    else:
        # access split data
        if isinstance(axes_split_data[0][0], Axes):
            axes, split_indices = zip(*axes_split_data)
        else:
            axes_, splits_ = zip(*axes_split_data)
            (axes, _), (splits, _) = zip(*axes_), zip(*splits_)
            split_indices = [split.index for split in splits]
        obs_splits = [data.loc[split_idx, :] for split_idx in split_indices]
        major_offset = None

    plt.suptitle(title)

    return_axes_data = []
    for i, ax in enumerate(axes):
        split_data = obs_splits[i]

        if twin_column_name:
            alt_data = split_data[twin_column_name].to_frame()
            split_data = split_data.drop(columns=[twin_column_name])

        plot_fn(ax, split_data, **plot_kwds)

        if major_offset is not None:
            locator = AutoDateLocator()
            formatter = AutoDateFormatter(locator)

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # Shift the major tick labels down
            for xlabel in ax.xaxis.get_majorticklabels():
                xlabel.set_transform(xlabel.get_transform() + major_offset)

        legend_lines, legend_labels = ax.get_legend_handles_labels()

        if twin_column_name:
            twin_plot_kwargs = twin_plot_kwargs or {}
            alt_ax = ax.twinx()
            alt_ax._get_lines.get_next_color()
            alt_ax.plot(alt_data, **twin_plot_kwargs)

            alt_ax.grid(False)

            twin_lines, twin_labels = alt_ax.get_legend_handles_labels()
            legend_lines += twin_lines
            legend_labels += twin_labels

            return_axes_data.append(((ax, alt_ax), (split_data, alt_data.index)))
        else:
            return_axes_data.append((ax, split_data.index))

        # Make sure Matplotlib shows the true date range and doesn't
        # choose its own
        split_start_date = split_offset.rollback(split_data.index.min())
        split_end_date = split_start_date + split_offset

        assert split_data.index.min() >= split_start_date
        assert split_data.index.max() <= split_end_date

        ax.set_xlim(split_start_date, split_end_date)

        ax.legend(legend_lines, legend_labels)

    plt.tight_layout()

    return return_axes_data


def plot_timeseries_histograms(
    axes: Axes,
    data: pd.DataFrame,
    bins: Union[str, int, np.ndarray, Sequence[Union[int, float]]] = "auto",
    colormap: Colormap = cm.Blues,
    **plot_kwargs,
) -> Axes:  # pragma: no cover
    """Generate a heat-map-like plot for time-series sample data.

    The kind of input this function expects can be obtained from an
    XArray object as follows:

    .. code:

        data = az_post_trace.posterior_predictive.Y_t[chain_idx].loc[
            {"dt": slice(t1, t2)}
        ]
        data = data.to_dataframe().Y_t.unstack(level=0)

    Parameters
    ==========
    axes
        The Matplotlib axes to use for plotting.
    data
        The sample data to be plotted.  This should be in "wide" format: i.e.
        the index should be "time" and the columns should correspond to each
        sample.
    bins
        The `bins` parameter passed to ``np.histogram``.
    colormap
        The Matplotlib colormap use to show relative frequencies within bins.
    plot_kwargs
        Keywords passed to ``fill_between``.

    """
    index = data.index
    y_samples = data.values

    n_t = len(index)

    # generate histograms and bins
    list_of_hist, list_of_bins = [], []
    for t in range(n_t):
        # TODO: determine proper range=(np.min(Y_t), np.max(Y_t))
        hist, bins_ = np.histogram(y_samples[t], bins=bins, density=True)
        if np.sum(hist > 0) == 1:
            hist, bins_ = np.array([1.0]), np.array([bins_[0], bins_[-1]])
        list_of_hist.append(hist)
        list_of_bins.append(bins_)

    if axes is None:
        _, (axes) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 4))
        axes.plot(index, np.mean(y_samples, axis=1), alpha=0.0, drawstyle="steps")

    for t in range(n_t):
        mask = index == index[t]
        hist, bins_ = list_of_hist[t], list_of_bins[t]
        # normalize bin weights for plotting
        hist = hist / np.max(hist) * 0.85 if len(hist) > 1 else hist
        n = len(hist)
        # construct predictive arrays to plot
        y_t_ = np.tile(bins_, (n_t, 1))
        # include consecutive time points to create grid-ish steps
        if t > 0:
            mask = np.logical_or(mask, index == index[t - 1])
        for i in range(n):
            color_val = hist[i]
            color = colormap(color_val) if color_val else (1, 1, 1, 1)
            plot_kwargs.setdefault("step", "pre")
            axes.fill_between(
                index,
                y_t_[:, i],
                y_t_[:, i + 1],
                where=mask,
                color=color,
                **plot_kwargs,
            )

    return axes


def plot_split_timeseries_histograms(
    plot_data: pd.DataFrame, sample_col: str, plot_fn: Callable, **split_ts_kwargs
):  # pragma: no cover
    """A wrapper function for `plot_split_timeseries` and `plot_timeseries_histograms`

    Parameters
    ==========
    plot_data: DataFrame
        The sample data to be plotted.  This should be in "wide" format: i.e.
        the index should be "time" and the columns should correspond to each
        sample.
    sample_col: str
        Sample column to be plotted via `plot_timeseries_histograms`.
    plot_fn: callable
        A user-defined function to plot non-sample column(s), juxtaposed with
        the histogram plot of the sample column.
    split_ts_kwargs
        Keywords passed to ``plot_split_timeseries``.

    """
    axes_split_data = plot_split_timeseries(
        plot_data,
        plot_fn=plot_fn,
        **split_ts_kwargs,
    )

    _ = plot_split_timeseries(
        plot_data[sample_col],
        axes_split_data=axes_split_data,
        plot_fn=plot_timeseries_histograms,
        **split_ts_kwargs,
    )

    return axes_split_data


def plot_ts_histograms(
    axes: Axes,
    data: pd.DataFrame,
    sample_col: Text,
    N_obs: Optional[int] = None,
    **canvas_kwargs,
) -> Axes:  # pragma: no cover
    """Plot time series histograms on `datashader`'s canvas

    Parameters
    ==========
    axes: Axes
        The Matplotlib axes to use for plotting.
    data: DataFrame
        The sample data to be plotted. This should be in "long" format: i.e.
        the index should be "time" and the columns should contain "draw" and
        `sample_col`.
    sample_col: str
        Sample column to be plotted.
    N_obs: int
        Number of observations to plot. If unspecified, plotting all oberservations
        in `data`.
    canvas_kwargs
        Keywords passed to ``plot_split_timeseries``.

    """
    if "draw" not in data.columns:
        raise ValueError("`data` does not have 'draw' number in its columns")
    n_draws = np.max(data["draw"]) + 1
    total_obs = data.shape[0] // n_draws
    if N_obs is None:
        N_obs = total_obs
    elif N_obs > total_obs:
        raise ValueError(f"`N_obs` ({N_obs}) must be <= `total_obs` ({total_obs})")

    if "plot_width" not in canvas_kwargs:
        canvas_kwargs["plot_width"] = N_obs

    # initialize canvas
    canvas = ds.Canvas(**canvas_kwargs)

    # get data to plot
    data = data.head(N_obs * n_draws).copy()
    dts = data.index.unique()[:N_obs]
    data["dt"] = np.repeat(dts.view(np.int64), n_draws)

    agg = canvas.points(data, "dt", sample_col)
    agg.coords.update({"dt": dts})

    shade_res = tf.shade(agg, cmap="black", how="eq_hist")
    shade_res.values = shade_res.values / np.max(shade_res.values) * np.max(agg.values)
    res_img = tf.Image(shade_res)

    # start painting
    _ = res_img.plot(cmap="Blues", ax=axes, label="posterior predictives")

    return axes


def plot_split_ts_histograms(
    sample_data: pd.DataFrame,
    sample_col: Text,
    nonsample_data: Union[pd.Series, pd.DataFrame],
    plot_fn: Callable,
    **split_ts_kwargs,
):  # pragma: no cover
    """A wrapper function for `plot_split_timeseries` and `plot_ts_histograms`

    Parameters
    ==========
    plot_data: DataFrame
        The sample data to be plotted. This should be in "long" format: i.e.
        the index should be "time" and the columns should contain "draw" and
        `sample_col`.
    observed_data: Series
        The observed values corresponding to the samples in `plot_data`. The
        index should be "time" and the column should be observed values.
    plot_fn: callable
        A user-defined function to plot non-sample column(s), juxtaposed with
        the histogram plot of the sample column.
    split_ts_kwargs
        Keywords passed to ``plot_split_timeseries``.

    """
    axes_split_data = plot_split_timeseries(
        nonsample_data, plot_fn=plot_fn, **split_ts_kwargs
    )

    _ = plot_split_timeseries(
        sample_data,
        axes_split_data=axes_split_data,
        plot_fn=plot_ts_histograms,
        sample_col=sample_col,
        **split_ts_kwargs,
    )

    return axes_split_data
