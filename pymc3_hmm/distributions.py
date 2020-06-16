import numpy as np

import theano.tensor as tt

import pymc3 as pm

from theano.gof.op import get_test_value

from pymc3.distributions.distribution import draw_values, _DrawValuesContext


vsearchsorted = np.vectorize(np.searchsorted, otypes=[np.int], signature="(n),()->()")


class Mixture(pm.Mixture):
    """A version of PyMC3's `Mixture` that uses log-scale mixture probabilities."""

    def __init__(self, logw, comp_dists, *args, **kwargs):
        self.logw = logw
        self.w = tt.exp(logw)
        super().__init__(self.w, comp_dists, *args, **kwargs)

    def logp(self, value):
        logw = self.logw
        loglik = pm.math.logsumexp(logw + self._comp_logp(value), axis=-1)
        # return bound(loglik,
        #              logw >= -np.inf, logw <= 0,
        #              tt.allclose(logsumexp(logw, axis=-1), 0.0),
        #              broadcast_conditions=False)
        return loglik


class PoissonZeroProcess(pm.Discrete):
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
        self.mode = tt.zeros(states.shape)
        shape = kwargs.pop("shape", get_test_value(states).shape)
        super().__init__(shape=shape, **kwargs)

    def logp(self, obs):
        """Return the scalar Theano log-likelihood at a point."""

        mu = self.mu

        nonzero_idx = tt.gt(self.states, 0.5)
        # mu_nzo = mu[nonzero_idx]
        # obs_nzo = obs[nonzero_idx]
        #
        # logp_val = pm.Constant.dist(0.0).logp(obs)
        #
        # # P(Y | Y > 0)
        # # logp_val = obs_nzo * tt.log(mu_nzo) - mu_nzo - tt.log1p(-tt.exp(mu_nzo)) - factln(obs_nzo)
        # # XXX: Can't use boolean `nonzero_idx`!
        # logp_val = tt.set_subtensor(logp_val[np.arange(nonzero_idx)], pm.Poisson.dist(mu_nzo).logp(obs_nzo))

        logp_val = tt.where(
            nonzero_idx, pm.Poisson.dist(mu).logp(obs), pm.Constant.dist(0.0).logp(obs)
        )

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

            terms = [self.mu, self.states]

            # TODO: It really seems like we shouldn't have to do this manually.
            for t in terms:
                # TODO FIXME: Very, very lame...
                term_smpl = draw_context.drawn_vars.get((t, 1), None)
                if term_smpl is not None:
                    point[t.name] = term_smpl

            mu, states = draw_values(terms, point=point)

            mu = np.broadcast_to(mu, states.shape[0]).reshape(states.shape)
            nonzero_idx = states > 0

            nonzero_num = np.sum(nonzero_idx)

            if not size or size == 1:
                size_shape = ()
            else:
                size_shape = tuple(np.atleast_1d(size or []))

            res = np.zeros(states.shape + size_shape, dtype=self.dtype)

            if nonzero_num > 0:
                pois_samples = pm.Poisson.dist(
                    mu[nonzero_idx], shape=(nonzero_num,)
                ).random(size=size)
                res[nonzero_idx] = pois_samples.reshape((nonzero_num,) + size_shape)

        return res


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
        terms = [self.gamma_0, self.Gamma]

        # with _DrawValuesContext() as draw_context:
        #     for t in terms:
        #         term_smpl = draw_context.drawn_vars.get((t, None), None)
        #         if term_smpl is not None:
        #             point[t.name] = term_smpl

        gamma_0, Gamma = draw_values(terms, point=point)

        N = self.N

        # size_shape = np.atleast_1d(size or [])
        # gamma_0 = np.broadcast_to(gamma_0, tuple(size_shape) + gamma_0.shape)

        state_n = pm.Categorical.dist(gamma_0).random(size=size)
        state_shape = state_n.shape

        states_shape = (N,) + state_shape
        states = np.empty(states_shape, dtype=self.dtype)
        states[0] = state_n

        unif_samples = np.random.uniform(size=states_shape)

        if Gamma.ndim > 2:
            Gamma_bcast = np.broadcast_to(Gamma, Gamma.shape[:2] + tuple(state_shape))

        for n in range(1, N):
            if Gamma.ndim > 2:
                gamma_t = np.asarray(
                    [Gamma_bcast[i][state_n[i]] for i in np.ndindex(state_shape)]
                )
            else:
                gamma_t = Gamma[state_n, :]

            state_n = vsearchsorted(gamma_t.cumsum(axis=-1), unif_samples[n])
            # state_n = pm.Categorical.dist(gamma_t).random()
            states[n] = state_n.reshape(state_shape)

        return states
