import warnings

import numpy as np

try:  # pragma: no cover
    import aesara
    import aesara.tensor as at
    from aesara.graph.op import get_test_value
    from aesara.scalar import upcast
    from aesara.tensor.extra_ops import broadcast_to as at_broadcast_to
except ImportError:  # pragma: no cover
    import theano as aesara
    import theano.tensor as at
    from theano.graph.op import get_test_value
    from theano.scalar import upcast
    from theano.tensor.extra_ops import broadcast_to as at_broadcast_to

import pymc3 as pm
from pymc3.distributions.distribution import (
    Distribution,
    _DrawValuesContext,
    draw_values,
    generate_samples,
)
from pymc3.distributions.mixture import _conversion_map, all_discrete

from pymc3_hmm.utils import tt_broadcast_arrays, vsearchsorted


def distribution_subset_args(dist, shape, idx):
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

    Returns
    -------
    A ``dict`` of the broadcasted and indexed parameters for `dist`.

    """

    dist_param_names = dist._distr_parameters_for_repr()

    res = dict()
    for param in dist_param_names:

        bcast_res = at_broadcast_to(getattr(dist, param), shape)

        res[param] = bcast_res[idx]

    return res


def get_and_check_comp_value(x):
    if isinstance(x, Distribution):
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


class SwitchingProcess(Distribution):
    """A distribution that models a switching process over arbitrary univariate mixtures and a state sequence.

    This class is like `Mixture`, but without the mixture weights.

    """  # noqa: E501

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
        self.states = at.as_tensor_variable(pm.intX(states))

        if len(comp_dists) > 31:
            warnings.warn(
                "There are too many mixture distributions to properly"
                " determine their combined shape."
            )

        self.comp_dists = comp_dists

        states_tv = get_test_value(self.states)
        bcast_comps = np.broadcast(
            states_tv, *[get_and_check_comp_value(x) for x in comp_dists[:31]]
        )
        shape = bcast_comps.shape

        defaults = kwargs.pop("defaults", [])

        out_dtype = upcast(*[x.type.dtype for x in comp_dists])
        dtype = kwargs.pop("dtype", out_dtype)

        if not all_discrete(comp_dists):
            try:
                bcast_means = tt_broadcast_arrays(
                    *([self.states] + [d.mean.astype(dtype) for d in self.comp_dists])
                )
                self.mean = at.choose(self.states, bcast_means[1:])

                if "mean" not in defaults:
                    defaults.append("mean")

            except (AttributeError, ValueError, IndexError):  # pragma: no cover
                pass

        try:
            bcast_modes = tt_broadcast_arrays(
                *([self.states] + [d.mode.astype(dtype) for d in self.comp_dists])
            )
            self.mode = at.choose(self.states, bcast_modes[1:])

            if "mode" not in defaults:
                defaults.append("mode")

        except (AttributeError, ValueError, IndexError):  # pragma: no cover
            pass

        super().__init__(shape=shape, dtype=dtype, defaults=defaults, **kwargs)

    def logp(self, obs):
        """Return the Theano log-likelihood at a point."""

        obs_tt = at.as_tensor_variable(obs)

        logp_val = at.alloc(-np.inf, *obs.shape)

        for i, dist in enumerate(self.comp_dists):
            i_mask = at.eq(self.states, i)
            obs_i = obs_tt[i_mask]
            subset_dist_kwargs = distribution_subset_args(dist, obs.shape, i_mask)
            subset_dist = dist.dist(**subset_dist_kwargs)
            logp_val = at.set_subtensor(logp_val[i_mask], subset_dist.logp(obs_i))

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
        with _DrawValuesContext():
            (states,) = draw_values([self.states], point=point, size=size)

        # This is a terrible thing to have to do here, but it's better than
        # having to (know to) update `Distribution.shape` when/if dimensions
        # change (e.g. when sampling new state sequences).
        bcast_comps = np.broadcast(
            states, *[dist.random(point=point) for dist in self.comp_dists]
        )
        self_shape = bcast_comps.shape

        if size:
            # `draw_values` will not honor the `size` parameter if its arguments
            # don't contain random variables, so, when our `self.states` are
            # constants, we have to broadcast `states` so that it matches `size +
            # self.shape`.
            expanded_states = np.broadcast_to(
                states, tuple(np.atleast_1d(size)) + self_shape
            )
        else:
            expanded_states = np.broadcast_to(states, self_shape)

        samples = np.empty(expanded_states.shape)

        for i, dist in enumerate(self.comp_dists):
            # We want to sample from only the parts of our component
            # distributions that are active given the states.
            # This is only really relevant when the component distributions
            # change over the state space (e.g. Poisson means that change
            # over time).
            # We could always sample such components over the entire space
            # (e.g. time), but, for spaces with large dimension, that would
            # be extremely costly and wasteful.
            i_idx = np.where(expanded_states == i)
            i_size = len(i_idx[0])
            if i_size > 0:
                subset_kwargs = distribution_subset_args(
                    dist, expanded_states.shape, i_idx
                )
                state_dist = dist.dist(**subset_kwargs)

                sample = state_dist.random(point=point)
                samples[i_idx] = sample

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
        self.mu = at.as_tensor_variable(pm.floatX(mu))
        self.states = at.as_tensor_variable(states)

        super().__init__(
            [Constant.dist(np.array(0, dtype=np.int64)), pm.Poisson.dist(mu)],
            states,
            **kwargs
        )


class DiscreteMarkovChain(pm.Discrete):
    """A first-order discrete Markov chain distribution.

    This class characterizes vector random variables consisting of state
    indicator values (i.e. `0` to `M - 1`) that are driven by a discrete Markov
    chain.

    """

    def __init__(self, Gammas, gamma_0, shape, **kwargs):
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
        self.gamma_0 = at.as_tensor_variable(pm.floatX(gamma_0))

        assert Gammas.ndim >= 3

        self.Gammas = at.as_tensor_variable(pm.floatX(Gammas))

        shape = np.atleast_1d(shape)

        dtype = _conversion_map[aesara.config.floatX]
        self.mode = np.zeros(tuple(shape), dtype=dtype)

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

        """  # noqa: E501

        states_tt = at.as_tensor(states)

        if states.ndim > 1 or self.Gammas.ndim > 3 or self.gamma_0.ndim > 1:
            raise NotImplementedError("Broadcasting not supported.")

        Gammas_tt = at_broadcast_to(
            self.Gammas, (states.shape[0],) + tuple(self.Gammas.shape)[-2:]
        )
        gamma_0_tt = self.gamma_0

        Gamma_1_tt = Gammas_tt[0]
        P_S_1_tt = at.dot(gamma_0_tt, Gamma_1_tt)[states_tt[0]]

        # def S_logp_fn(S_tm1, S_t, Gamma):
        #     return at.log(Gamma[..., S_tm1, S_t])
        #
        # P_S_2T_tt, _ = aesara.scan(
        #     S_logp_fn,
        #     sequences=[
        #         {
        #             "input": states_tt,
        #             "taps": [-1, 0],
        #         },
        #         Gammas_tt,
        #     ],
        # )
        P_S_2T_tt = Gammas_tt[at.arange(1, states.shape[0]), states[:-1], states[1:]]

        log_P_S_1T_tt = at.concatenate(
            [at.shape_padright(at.log(P_S_1_tt)), at.log(P_S_2T_tt)]
        )

        res = log_P_S_1T_tt.sum()
        res.name = "states_logp"

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
        terms = [self.gamma_0, self.Gammas]

        with _DrawValuesContext():
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
        slices_tuple = tuple(np.ogrid[[slice(None, d) for d in state_shape]])

        for n in range(0, N):
            gamma_t = Gamma[..., n, :, :]
            gamma_t = gamma_t[slices_tuple + (state_n,)]
            state_n = vsearchsorted(gamma_t.cumsum(axis=-1), unif_samples[..., n])
            states[..., n] = state_n

        return states

    def _distr_parameters_for_repr(self):
        return ["Gammas", "gamma_0"]


class Constant(Distribution):
    r"""Constant log-likelihood.

    Parameters
    ----------
    value: float or int
        Constant parameter.
    """

    def __init__(self, c, shape=(), defaults=("mode",), **kwargs):

        c = at.as_tensor_variable(c)

        dtype = c.dtype

        if kwargs.get("transform", None) is not None:
            raise ValueError(
                "Transformations for constant distributions are not allowed."
            )  # pragma: no cover

        super().__init__(shape, dtype, defaults=defaults, **kwargs)

        self.mean = self.median = self.mode = self.c = c

    def random(self, point=None, size=None):
        c = draw_values([self.c], point=point, size=size)[0]

        def _random(c, dtype=self.dtype, size=None):
            return np.full(size, fill_value=c, dtype=dtype)

        return generate_samples(_random, c=c, dist_shape=self.shape, size=size).astype(
            self.dtype
        )

    def logp(self, value):
        return at.switch(at.eq(value, self.c), 0.0, -np.inf)

    def _distr_parameters_for_repr(self):
        return ["c"]


class HorseShoe(pm.Normal):
    """
    A Horseshoe distribution

    """

    def __init__(self, tau, shape, **kwargs):
        self.tau = tau
        self.beta_raw = pm.Normal.dist(tau=1, shape=shape)
        self.lmbda = pm.HalfCauchy.dist(beta=1, shape=shape)
        super().__init__(shape=shape, **kwargs)

    def random(self, point=None, size=None):
        beta_raw = self.beta_raw.random(point=point, size=size)
        lmbda = self.lmbda.random(point=point, size=size)
        return beta_raw * lmbda * self.tau

    def logp(self, values):
        """
        XXX: This is only a placeholder for the log-probability.
        It is required to get past PyMC3 design restrictions.
        Do not use this distribution without the `HSStep` sampler!

        """
        warnings.warn(
            "logp of HorseShoe class is not implemented."
            "Do not use this distribution without the `HSStep` sampler!"
        )
        return 0
