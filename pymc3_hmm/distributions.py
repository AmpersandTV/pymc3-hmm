import numpy as np

import theano
import theano.tensor as tt

import pymc3 as pm

from copy import copy

from theano.gof.op import get_test_value

from pymc3.distributions.mixture import all_discrete, _conversion_map
from pymc3.distributions.distribution import draw_values, _DrawValuesContext

from pymc3_hmm.utils import (
    broadcast_to,
    tt_expand_dims,
    tt_broadcast_arrays,
    vsearchsorted,
)


def distribution_subset_args(dist, shape, idx, point=None):
    """Obtain subsets of a distribution parameters via indexing.

    This is used to effectively "lift" slices/`Subtensor` `Op`s up to a
    distribution's parameters.  In other words, `pm.Normal(mu, sigma)[idx]`
    becomes `pm.Normal(mu[idx], sigma[idx])`.  In computations, the former
    requires the entire evaluation of `pm.Normal(mu, sigma)` (e.g. its `.logp`
    or a sample from `.random`), which could be very complex, while the latter
    only evaluates the subset of interest.

    XXX: this lifting isn't appropriate for every distribution.  It's fine for
    most scalar distributions and even some multivariate distributions, but
    some required functionality is missing in order to handle even the latter.

    Parameters
    ----------
    dist : Distribution
        The distribution object with the parameters to be indexed.
    shape : tuple or Shape
        The shape of the distribution's output/support.  This is used
        to (naively) determine the parameters' broadcasting pattern.
    idx : ndarray or TensorVariable
        The indices applied to the parameters of `dist`.
    point : dict (optional)
        A dictionary keyed on the `str` names of each parameter in `dist`,
        which are mapped to NumPy values for the corresponding parameter.  When
        this is given, the Theano parameters are replaced by their values in the
        dictionary.

    Returns
    -------
    res: list
        An ordered set of broadcasted and indexed parameters for `dist`.


    """

    dist_param_names = dist._distr_parameters_for_repr()

    if point:
        # Try to get a concrete/NumPy value if a `point` parameter was
        # given.
        try:
            idx = get_test_value(idx)
        except AttributeError:  # pragma: no cover
            pass

    res = []
    for param in dist_param_names:

        # Use the (sampled) point, if present
        if point is None or param not in point:
            x = getattr(dist, param, None)

            if x is None:
                continue
        else:
            x = point[param]

            # Try to get a concrete/NumPy value if a `point` parameter was
            # given.
            try:
                x = get_test_value(x)
                shape = get_test_value(shape)
            except AttributeError:  # pragma: no cover
                pass

        res.append(broadcast_to(x, shape)[idx])

    return res


def get_and_check_comp_value(x):
    if isinstance(x, pm.Distribution):
        try:
            return x.default()
        except AttributeError:
            pass

        return x.random()
    else:
        raise TypeError(
            "Component distributions must be PyMC3 Distributions. "
            "Got {}".format(type(x))
        )


class SwitchingProcess(pm.Distribution):
    """A distribution that models a switching process over arbitrary univariate mixtures and a state sequence.

    This class is like `Mixture`, but without the mixture weights.

    """

    def __init__(self, comp_dists, states, *args, **kwargs):
        """Initialize a `SwitchingProcess` instance.

        Each `Distribution` object in `comp_dists` must have a
        `Distribution.random_subset` method that takes a list of indices and
        returns a sample for only that subset.  Unfortunately, since PyMC3
        doesn't provide such a method, you'll have to implement it yourself and
        monkey patch a `Distribution` class.

        Parameters
        ----------
        comp_dists : list of Distribution
            A list containing `Distribution` objects for each mixture component.
            These are essentially the emissions distributions.
        states : DiscreteMarkovChain
            The hidden state sequence.  It should have a number of states
            equal to the size of `comp_dists`.

        """
        self.states = tt.as_tensor_variable(pm.intX(states))

        states_tv = get_test_value(self.states)

        bcast_comps = np.broadcast(
            states_tv, *[get_and_check_comp_value(x) for x in comp_dists[:31]]
        )

        self.comp_dists = []
        for dist in comp_dists:
            d = copy(dist)
            d.shape = bcast_comps.shape
            self.comp_dists.append(d)

        # TODO: Not sure why we would allow users to set the shape if we're
        # just going to compute it.
        # shape = kwargs.pop("shape", bcast_comps.shape)
        shape = bcast_comps.shape

        defaults = kwargs.pop("defaults", [])

        if all_discrete(comp_dists):
            default_dtype = _conversion_map[theano.config.floatX]
        else:
            default_dtype = theano.config.floatX

            try:
                bcast_means = tt_broadcast_arrays(
                    *([self.states] + [d.mean for d in self.comp_dists])
                )
                self.mean = tt.choose(self.states, bcast_means)

                if "mean" not in defaults:
                    defaults.append("mean")

            except (AttributeError, ValueError, IndexError):  # pragma: no cover
                pass

        dtype = kwargs.pop("dtype", default_dtype)

        try:
            bcast_modes = tt_broadcast_arrays(
                *([self.states] + [d.mode for d in self.comp_dists])
            )
            self.mode = tt.choose(self.states, bcast_modes)

            if "mode" not in defaults:
                defaults.append("mode")

        except (AttributeError, ValueError, IndexError):  # pragma: no cover
            pass

        super().__init__(shape=shape, dtype=dtype, defaults=defaults, **kwargs)

    def logp(self, obs):
        """Return the scalar Theano log-likelihood at a point."""

        obs_tt = tt.as_tensor_variable(obs)

        logp_val = tt.alloc(-np.inf, *obs.shape)

        for i, dist in enumerate(self.comp_dists):
            i_mask = tt.eq(self.states, i)
            obs_i = obs_tt[i_mask]
            i_idx = tt.unravel_index(
                tt.arange(tt.mul(self.states.shape)[0])[i_mask.ravel()],
                self.states.shape,
            )
            subset_dist = dist.dist(*distribution_subset_args(dist, obs.shape, i_idx))
            logp_val = tt.set_subtensor(logp_val[i_idx], subset_dist.logp(obs_i))

        logp_val.name = "SwitchingProcess_logp"

        return logp_val

    def random(self, point=None, size=None):
        """Sample from this distribution conditional on a given set of values.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        with _DrawValuesContext() as draw_context:

            # TODO FIXME: Very, very lame...
            term_smpl = draw_context.drawn_vars.get((self.states, 1), None)
            if term_smpl is not None:
                point[self.states.name] = term_smpl

            # `draw_values` is inconsistent and will not use the `size`
            # parameter if the variables aren't random variables.
            if hasattr(self.states, "distribution"):
                (states,) = draw_values([self.states], point=point, size=size)
            else:
                states = pm.Constant.dist(self.states).random(point=point, size=size)

            # states = states.T

            samples = np.empty(states.shape)

            for i, dist in enumerate(self.comp_dists):
                # We want to sample from only the parts of our component
                # distributions that are active given the states.
                # This is only really relevant when the component distributions
                # change over the state space (e.g. Poisson means that change
                # over time).
                # We could always sample such components over the entire space
                # (e.g. time), but, for spaces with large dimension, that would
                # be extremely costly and wasteful.
                i_idx = np.where(states == i)
                i_size = len(i_idx[0])
                if i_size > 0:
                    subset_args = distribution_subset_args(
                        dist, states.shape, i_idx, point=point
                    )
                    samples[i_idx] = dist.dist(*subset_args).random(point=point)

        return samples


class PoissonZeroProcess(SwitchingProcess):
    """A Poisson-Dirac-delta (at zero) mixture process.

    The first mixture component (at index 0) is the Dirac-delta at zero, and
    the second mixture component is the Poisson random variable.
    """

    def __init__(self, mu=None, states=None, **kwargs):
        """Initialize a `PoissonZeroProcess` object.

        Parameters
        ----------
        mu: tensor
            The Poisson rate(s)
        states: tensor
            A vector of integer 0-1 states that indicate which component of
            the mixture is active at each point/time.
        """
        self.mu = tt.as_tensor_variable(pm.floatX(mu))
        self.states = tt.as_tensor_variable(states)

        super().__init__([pm.Constant.dist(0), pm.Poisson.dist(mu)], states, **kwargs)


class DiscreteMarkovChain(pm.Discrete):
    """A first-order discrete Markov chain distribution.

    This class characterizes vector random variables consisting of state
    indicator values (i.e. `0` to `M - 1`) that are driven by a discrete Markov
    chain.

    """

    def __init__(self, Gamma, gamma_0, shape, **kwargs):
        """Initialize an `DiscreteMarkovChain` object.

        Parameters
        ----------
        Gammas: TensorVariable
            An array of transition probability matrices.  `Gammas` takes the
            shape `... x N x M x M` for a state sequence of length `N` having
            `M`-many distinct states.  Each row, `r`, in a transition probability
            matrix gives the probability of transitioning from state `r` to each
            other state.
        gamma_0: TensorVariable
            The initial state probabilities.  The last dimension should be length `M`,
            i.e. the number of distinct states.
        shape: tuple of int
            Shape of the state sequence.  The last dimension is `N`, i.e. the
            length of the state sequence(s).
        """
        self.gamma_0 = tt.as_tensor_variable(pm.floatX(gamma_0))

        assert Gamma.ndim >= 3

        self.Gammas = tt.as_tensor_variable(pm.floatX(Gamma))

        shape = np.atleast_1d(shape)

        dtype = _conversion_map[theano.config.floatX]
        self.mode = tt.zeros(tuple(shape), dtype=dtype)

        super().__init__(shape=shape, **kwargs)

    def logp(self, states):
        r"""Create a Theano graph that computes the log-likelihood for a discrete Markov chain.

        This is the log-likelihood for the joint distribution of states, :math:`S_t`, conditional
        on state samples, :math:`s_t`, given by the following:

        .. math::

            \int_{S_0} P(S_1 = s_1 \mid S_0) dP(S_0) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

        The first term (i.e. the integral) simply computes the marginal :math:`P(S_1 = s_1)`, so
        another way to express this result is as follows:

        .. math::

            P(S_1 = s_1) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

        """

        Gammas = tt.shape_padleft(self.Gammas, states.ndim - (self.Gammas.ndim - 2))

        # Multiply the initial state probabilities by the first transition
        # matrix by to get the marginal probability for state `S_1`.
        # The integral that produces the marginal is essentially
        # `gamma_0.dot(Gammas[0])`
        Gamma_1 = Gammas[..., 0:1, :, :]
        gamma_0 = tt_expand_dims(self.gamma_0, (-3, -1))
        P_S_1 = tt.sum(gamma_0 * Gamma_1, axis=-2)

        # The `tt.switch`s allow us to broadcast the indexing operation when
        # the replication dimensions of `states` and `Gammas` don't match
        # (e.g. `states.shape[0] > Gammas.shape[0]`)
        S_1_slices = tuple(
            slice(
                tt.switch(tt.eq(P_S_1.shape[i], 1), 0, 0),
                tt.switch(tt.eq(P_S_1.shape[i], 1), 1, d),
            )
            for i, d in enumerate(states.shape)
        )
        S_1_slices = (tuple(tt.ogrid[S_1_slices]) if S_1_slices else tuple()) + (
            states[..., 0:1],
        )
        logp_S_1 = tt.log(P_S_1[S_1_slices]).sum(axis=-1)

        # These are slices for the extra dimensions--including the state
        # sequence dimension (e.g. "time")--along which which we need to index
        # the transition matrix rows using the "observed" `states`.
        trans_slices = tuple(
            slice(
                tt.switch(
                    tt.eq(Gammas.shape[i], 1), 0, 1 if i == states.ndim - 1 else 0
                ),
                tt.switch(tt.eq(Gammas.shape[i], 1), 1, d),
            )
            for i, d in enumerate(states.shape)
        )
        trans_slices = (tuple(tt.ogrid[trans_slices]) if trans_slices else tuple()) + (
            states[..., :-1],
        )

        # Select the transition matrix row of each observed state; this yields
        # `P(S_t | S_{t-1} = s_{t-1})`
        P_S_2T = Gammas[trans_slices]

        obs_slices = tuple(slice(None, d) for d in P_S_2T.shape[:-1])
        obs_slices = (tuple(tt.ogrid[obs_slices]) if obs_slices else tuple()) + (
            states[..., 1:],
        )
        logp_S_1T = tt.log(P_S_2T[obs_slices])

        res = logp_S_1 + tt.sum(logp_S_1T, axis=-1)
        res.name = "DiscreteMarkovChain_logp"

        return res

    def random(self, point=None, size=None):
        """Sample from a discrete Markov chain conditional on a given set of values.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        with _DrawValuesContext() as draw_context:
            terms = [self.gamma_0, self.Gammas]

            gamma_0, Gamma = draw_values(terms, point=point)

            # Sample state 0 in each state sequence
            state_n = pm.Categorical.dist(gamma_0, shape=self.shape[:-1]).random(
                point=point, size=size
            )
            state_shape = state_n.shape

            N = self.shape[-1]

            states = np.empty(state_shape + (N,), dtype=self.dtype)

            unif_samples = np.random.uniform(size=states.shape)

            # Make sure we have a transition matrix for each element in a state
            # sequence
            Gamma = np.broadcast_to(Gamma, tuple(states.shape) + Gamma.shape[-2:])

            # Slices across each independent/replication dimension
            slices = [slice(None, d) for d in state_shape]
            slices = tuple(np.ogrid[slices])

            for n in range(0, N):
                gamma_t = Gamma[..., n, :, :]
                gamma_t = gamma_t[slices + (state_n,)]
                state_n = vsearchsorted(gamma_t.cumsum(axis=-1), unif_samples[..., n])
                states[..., n] = state_n

            return states
