import numpy as np

import theano.tensor as tt

from scipy.special import logsumexp


vsearchsorted = np.vectorize(np.searchsorted, otypes=[np.int], signature="(n),()->()")


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
    Lam = (tt.eye(N_states) - P + tt.ones((N_states, N_states))).T
    u = tt.slinalg.solve(Lam, tt.ones((N_states,)))
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
    x_max_ = tt.max(x, axis=axis, keepdims=True)

    if x_max_.ndim > 0:
        x_max_ = tt.set_subtensor(x_max_[tt.isinf(x_max_)], 0.0)
    elif tt.isinf(x_max_):
        x_max_ = tt.as_tensor(0.0)

    res = tt.sum(tt.exp(x - x_max_), axis=axis, keepdims=keepdims)
    res = tt.log(res)

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
    A_bcast = A.dimshuffle(list(range(A.ndim)) + ["x"])

    sqz = False
    shape_b = ["x"] + list(range(b.ndim))
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
    dim_range = list(range(x.ndim))
    for d in sorted(np.atleast_1d(dims), reverse=True):
        offset = 0 if d >= 0 else len(dim_range) + 1
        dim_range.insert(d + offset, "x")

    return x.dimshuffle(dim_range)


def tt_broadcast_arrays(*args):
    """Broadcast any number of arrays against each other.

    This is a Theano emulation of `numpy.broadcast_arrays`.  It does *not* use
    memory views, and--as a result--it will not be nearly as efficient as the
    NumPy version.

    Parameters
    ----------
    `*args` : array_likes
        The arrays to broadcast.

    """
    p = max(a.ndim for a in args)

    args = [tt.shape_padleft(a, n_ones=p - a.ndim) if a.ndim < p else a for a in args]

    bcast_shape = [None] * p
    for i in range(p - 1, -1, -1):
        non_bcast_args = [tuple(a.shape)[i] for a in args if not a.broadcastable[i]]
        bcast_shape[i] = tt.max([1] + non_bcast_args)

    # TODO: This could be very costly?
    return [a * tt.ones(bcast_shape) for a in args]


def broadcast_to(x, shape):
    """Broadcast an array to a new shape.

    This implementation will use NumPy when an `ndarray` is given and an
    inefficient Theano variant otherwise.

    Parameters
    ----------
    x : array_like
        The array to broadcast.
    shape : tuple
        The shape of the desired array.
    """
    if isinstance(x, np.ndarray):
        return np.broadcast_to(x, shape)  # pragma: no cover
    else:
        # TODO: This could be very costly?
        return x * tt.ones(shape)


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
        lib = tt
        lib_logsumexp = tt_logsumexp

    # exp_ys = lib.exp(ys)
    # res = lib.concatenate([exp_ys, lib.ones(tuple(ys.shape)[:-1] + (1,))], axis=-1)
    # res = res / (1 + lib.sum(exp_ys, axis=-1))[..., None]

    res = lib.concatenate([ys, lib.zeros(tuple(ys.shape)[:-1] + (1,))], axis=-1)
    res = lib.exp(res - lib_logsumexp(res, axis=-1, keepdims=True))
    return res


def plot_split_timeseries(
    data,
    split_freq="W",
    split_max=5,
    twin_column_name=None,
    twin_plot_kwargs=None,
    figsize=(15, 15),
    title=None,
    drawstyle="steps-pre",
    linewidth=0.5,
    plot_fn=None,
    **plot_kwds
):  # pragma: no cover
    """Plot long timeseries by splitting them across multiple rows using a given time frequency.

    This function requires the Pandas and Matplotlib libraries.

    Parameters
    ----------
    data: DataFrame
        The timeseries to be plotted.
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
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.transforms as mtrans

    if plot_fn is None:

        def plot_fn(ax, data, **kwargs):
            return ax.plot(data, **kwargs)

    data = pd.DataFrame(data)

    if twin_column_name and len(data.columns) < 2:
        raise ValueError(
            "Option `twin_column` is only applicable for a two column `DataFrame`."
        )

    split_offset = pd.tseries.frequencies.to_offset(split_freq)

    grouper = pd.Grouper(freq=split_offset.freqstr, closed="left")
    obs_splits = [y_split for n, y_split in data.groupby(grouper)]

    if split_max:
        obs_splits = obs_splits[:split_max]

    n_partitions = len(obs_splits)

    fig, axes = plt.subplots(
        nrows=n_partitions, sharey=True, sharex=False, figsize=figsize
    )

    major_offset = mtrans.ScaledTranslation(0, -10 / 72.0, fig.dpi_scale_trans)

    axes[0].set_title(title)

    return_axes_data = []
    for i, ax in enumerate(axes):
        split_data = obs_splits[i]

        if twin_column_name:
            alt_data = split_data[twin_column_name].to_frame()
            split_data = split_data.drop(columns=[twin_column_name])

        plot_fn(ax, split_data, drawstyle=drawstyle, linewidth=linewidth, **plot_kwds)

        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 23, 3)))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=range(0, 7, 1)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %a"))

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
