import numpy as np

import theano.tensor as tt

import pymc3 as pm

from itertools import chain

from theano.gof.op import get_test_value as test_value

from pymc3.step_methods.arraystep import ArrayStep, Competence

from pymc3_hmm.distributions import HMMStateSeq
from pymc3_hmm.utils import compute_trans_freqs


big: np.float = 1e20
small: np.float = 1.0 / big


def ffbs_astep(gamma_0, Gamma, log_lik):
    # Number of observations
    N: int = log_lik.shape[0]
    # Number of states
    M: int = log_lik.shape[1]

    # Initial state probabilities
    gamma_0_normed: np.ndarray = gamma_0
    gamma_0_sum = np.sum(gamma_0)
    gamma_0_normed /= gamma_0_sum

    # "Forward" probabilities
    alphas: np.ndarray = np.empty((N, M), dtype=np.float)
    # Previous forward probability
    alpha_nm1: np.ndarray = gamma_0_normed

    # Forward filtering
    for n in range(N):
        log_lik_n = log_lik[n]
        lik_n = np.exp(log_lik_n - log_lik_n.max())
        alpha_n: np.ndarray = lik_n * np.dot(Gamma, alpha_nm1)
        alpha_n_sum = np.sum(alpha_n)

        # Rescale small values
        if alpha_n_sum < small:
            alpha_n *= big

        alpha_nm1 = alpha_n
        alphas[n] = alpha_n

    # Our sample results
    samples: np.ndarray = np.empty((N,), dtype=np.int8)
    # The uniform samples used to sample the categorical states
    unif_samples: np.ndarray = np.random.uniform(size=N)

    beta_N: np.ndarray = alphas[N - 1] / alphas[N - 1].sum()
    state_np1: np.ndarray = np.searchsorted(beta_N.cumsum(), unif_samples[N - 1])
    samples[N - 1] = state_np1

    # Backward sampling
    for n in range(N - 2, -1, -1):
        beta_n: np.ndarray = Gamma[state_np1] * alphas[n]
        beta_n /= np.sum(beta_n)

        state_np1 = np.searchsorted(beta_n.cumsum(), unif_samples[n])

        samples[n] = state_np1

    return samples


class FFBSStep(ArrayStep):
    r"""Forward-filtering backward-sampling."""

    name = "ffbs"

    def __init__(self, var, values=None, model=None):

        model = pm.modelcontext(model)

        (var,) = pm.inputvars(var)

        self.M = var.distribution.M.astype(int)
        self.N = var.distribution.N.astype(int)

        dependent_rvs = [
            v
            for v in model.basic_RVs
            if v is not var and var in tt.gof.graph.inputs([v.logpt])
        ]

        # We can use this function to get log-likelihood values for each state.
        dependents_log_lik = model.fn(
            tt.add(*[v.logp_elemwiset for v in dependent_rvs])
        )

        self.gamma_0_fn = model.fn(var.distribution.gamma_0)
        self.Gamma_fn = model.fn(var.distribution.Gamma)

        super().__init__([var], [dependents_log_lik], allvars=True)

    def astep(self, point, log_lik_fn, inputs):
        gamma_0 = self.gamma_0_fn(inputs)
        Gamma_t = self.Gamma_fn(inputs)
        log_lik_vals = [log_lik_fn(np.repeat(m, self.N)) for m in range(self.M)]
        log_lik_t = np.stack(log_lik_vals, 1)

        return ffbs_astep(gamma_0, Gamma_t, log_lik_t)

    @staticmethod
    def competence(var):
        distribution = getattr(var.distribution, "parent_dist", var.distribution)

        if isinstance(distribution, HMMStateSeq):
            return Competence.IDEAL
        # elif isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
        #     return Competence.COMPATIBLE

        return Competence.INCOMPATIBLE


class TransMatConjugateStep(ArrayStep):
    r"""Conjugate update step for a Dirichlet prior transition matrix."""

    name = "trans-mat-conjugate"

    def __init__(self, dir_priors, hmm_states, values=None, model=None, rng=None):
        r"""Initialize a `TransMatConjugateStep` object.

        Parameters
        ----------
        dir_priors: list of Dirichlets
            State-ordered from-to prior transition probabilities.
        hmm_states: random variable
            The HMM states variable using `dir_priors` as its transition matrix.
        """

        model = pm.modelcontext(model)

        dir_priors = list(chain.from_iterable([pm.inputvars(d) for d in dir_priors]))

        self.rng = rng
        self.dists = list(dir_priors)
        self.hmm_states = hmm_states.name
        # TODO: Perform a consistency check between `hmm_states.Gamma` and
        # `dir_priors`.

        super().__init__(dir_priors, [], allvars=True)

    def astep(self, point, inputs):
        states = inputs[self.hmm_states]
        N_mat = compute_trans_freqs(states, point.shape[0], counts_only=True)

        # res = [np.random.dirichlet(test_value(d.distribution.dist.a) + N_mat[i])
        #        for i, d in enumerate(self.dists)]
        # trans_res = [d.distribution.dist.transform.forward_val(r) for d, r in zip(self.dists, res)]

        trans_res = [
            d.distribution.dist.transform.forward_val(
                np.random.dirichlet(test_value(d.distribution.dist.a) + N_mat[i])
            )
            for i, d in enumerate(self.dists)
        ]

        sample = np.stack(trans_res, 1)

        return sample.reshape(point.shape)

    @staticmethod
    def competence(var):

        # TODO: Check that dependent term is a conjugate type.

        distribution = getattr(var.distribution, "parent_dist", var.distribution)

        if isinstance(distribution, pm.Dirichlet):
            return Competence.COMPATIBLE

        return Competence.INCOMPATIBLE
