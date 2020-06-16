import numpy as np

import theano
import theano.tensor as tt

import pymc3 as pm

from copy import copy

from theano.gof.op import get_test_value

from pymc3.distributions.mixture import all_discrete, _conversion_map
from pymc3.distributions.distribution import draw_values, _DrawValuesContext


vsearchsorted = np.vectorize(np.searchsorted, otypes=[np.int], signature="(n),()->()")


def broadcast_to(x, shape):
    if isinstance(x, np.ndarray):
        return np.broadcast_to(x, shape)  # pragma: no cover
    else:
        return x * tt.ones(shape)


# In general, these `*_subset_args` functions are such a terrible thing to
# have to use, but the `Distribution` class simply cannot handle Theano
# (yes, broadly speaking), so we need to hack around its shortcomings.
def normal_subset_args(self, shape, idx):
    return [
        (broadcast_to(self.mu, shape))[idx],
        (broadcast_to(self.sigma, shape))[idx],
    ]


pm.Normal.subset_args = normal_subset_args


def poisson_subset_args(self, shape, idx):
    return [(broadcast_to(self.mu, shape))[idx]]


pm.Poisson.subset_args = poisson_subset_args


def constant_subset_args(self, shape, idx):
    return [(broadcast_to(self.c, shape))[idx]]


pm.Constant.subset_args = constant_subset_args


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

        Hint: use `types.MethodType` for patches to class instances.

        """
        self.states = tt.as_tensor_variable(states)

        assert self.states.squeeze().ndim < 2

        states_tv = get_test_value(self.states)

        bcast_comps = np.broadcast(
            states_tv, *[d.sample() if hasattr(d, "sample") else d for d in comp_dists]
        )

        self.comp_dists = []
        for dist in comp_dists:

            assert hasattr(dist, "subset_args")

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
                self.mean = tt.choose(
                    self.states.squeeze(),
                    [
                        tt.cast(d.mean * tt.ones(d.shape).squeeze(), default_dtype)
                        for d in self.comp_dists
                    ],
                )
                self.mean = tt.reshape(self.mean, shape)

                if "mean" not in defaults:
                    defaults.append("mean")

            except (AttributeError, ValueError, IndexError):  # pragma: no cover
                pass

        dtype = kwargs.pop("dtype", default_dtype)

        try:
            self.mode = tt.choose(
                self.states.squeeze(),
                [
                    tt.cast(d.mode * tt.ones(d.shape).squeeze(), default_dtype)
                    for d in self.comp_dists
                ],
            )
            self.mode = tt.reshape(self.mode, shape)

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
            subset_dist = dist.dist(*dist.subset_args(obs.shape, i_idx))
            logp_val = tt.set_subtensor(logp_val[i_idx], subset_dist.logp(obs_i))

        logp_val.name = "pois_zero_logp"

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

            states = states.T

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
                    subset_args = dist.subset_args(states.shape, i_idx)
                    samples[i_idx] = dist.dist(*subset_args).random(point=point).T

        # PyMC3 expects the dimension order size + shape
        return samples.T


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


class HMMStateSeq(pm.Discrete):
    """A hidden markov state sequence distribution.

    The class characterizes a vector random variable of state indicator
    values (i.e. `0` to `M - 1`).

    """

    def __init__(self, N=None, Gamma=None, gamma_0=None, **kwargs):
        """Initialize an `HMMStateSeq` object.

        Parameters
        ----------
        N: int
            The length of the state sequence.
        Gamma: tensor matrix
            An `M x M` matrix of state transition probabilities.  Each row,
            `r`, should give the probability of transitioning from state `r`
        gamma_0: tensor vector
            The initial state probabilities.  Should be length `M`.

        """
        self.N = pm.intX(get_test_value(N))

        try:
            self.M = get_test_value(tt.shape(gamma_0)[-1])
        except AttributeError:
            self.M = tt.shape(gamma_0)[-1]

        self.gamma_0 = tt.as_tensor_variable(pm.floatX(gamma_0))
        self.Gamma = tt.as_tensor_variable(Gamma)

        shape = kwargs.pop("shape", tuple(np.atleast_1d(self.N)))
        self.mode = tt.zeros(*shape)

        super().__init__(shape=shape, **kwargs)

    def logp(self, obs):
        """Return the scalar Theano log-likelihood at a point."""
        V_0_logp = pm.Categorical.dist(self.gamma_0).logp(obs[0])
        V_0_logp.name = "V_0_logp"
        xi_t = self.Gamma[obs[:-1]]
        xi_t.name = "xi_t"
        V_t_logp = pm.Categorical.dist(xi_t, shape=(self.N - 1, self.M)).logp(obs[1:])
        V_t_logp.name = "V_t_logp"
        res = V_0_logp + tt.sum(V_t_logp)
        res.name = "hmmstateseq_logl"
        return res

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
            terms = [self.gamma_0, self.Gamma]

            # TODO: Would it be better to use `size` here instead?
            gamma_0, Gamma = draw_values(terms, point=point)

            N = self.N

            state_n = pm.Categorical.dist(gamma_0).random(point=point, size=size).T
            state_shape = state_n.shape

            states_shape = (N,) + state_shape
            states = np.empty(states_shape, dtype=self.dtype)
            states[0] = state_n

            unif_samples = np.random.uniform(size=states_shape)

            if Gamma.ndim > 2:
                Gamma_bcast = np.broadcast_to(
                    Gamma, Gamma.shape[:2] + tuple(state_shape)
                )

            for n in range(1, N):
                if Gamma.ndim > 2:
                    gamma_t = np.asarray(
                        [Gamma_bcast[i][state_n[i]] for i in np.ndindex(state_shape)]
                    )
                else:
                    gamma_t = Gamma[state_n, :]

                state_n = vsearchsorted(gamma_t.cumsum(axis=-1), unif_samples[n])
                states[n] = state_n.reshape(state_shape)

            # PyMC3 expects the dimension order size + shape
            return states.T
