import numpy as np

import theano.tensor as tt

import pymc3 as pm

from itertools import chain

from theano.gof.op import get_test_value as test_value

from pymc3.step_methods.arraystep import ArrayStep, Competence

from pymc3_hmm.distributions import DiscreteMarkovChain
from pymc3_hmm.utils import compute_trans_freqs


big: np.float = 1e20
small: np.float = 1.0 / big


def ffbs_astep(gamma_0: np.ndarray, Gammas: np.ndarray, log_lik: np.ndarray):
    """Sample a forward-filtered backward-sampled (FFBS) state sequence.

    Parameters
    ----------
    gamma_0: np.ndarray
        The initial state probabilities.
    Gamma: np.ndarray
        The transition probability matrices.  This array should take the shape
        `(N, M, M)`, where `N` is the state sequence length and `M` is the number
        of distinct states.  If `N` is `1`, the single transition matrix will
        broadcast across all elements of the state sequence.
    log_lik: np.ndarray
        An array of shape `(M, N)` consisting of the log-likelihood values for
        each state value at each point in the sequence.

    Returns
    -------
    samples: np.ndarray
        An array of shape `(N,)` containing the FFBS sampled state sequence.

    """
    # Number of observations
    N: int = log_lik.shape[-1]

    # Number of states
    M: int = gamma_0.shape[-1]
    # assert M == log_lik.shape[-2]

    # Initial state probabilities
    gamma_0_normed: np.ndarray = gamma_0.copy()
    gamma_0_normed /= np.sum(gamma_0)

    # "Forward" probabilities
    alphas: np.ndarray = np.empty((M, N), dtype=np.float)
    # Previous forward probability
    alpha_nm1: np.ndarray = gamma_0_normed

    # Make sure we have a transition matrix for each element in a state
    # sequence
    Gamma: np.ndarray = np.broadcast_to(Gammas, (N,) + Gammas.shape[-2:])

    lik_n: np.ndarray = np.empty((M,), dtype=np.float)
    alpha_n: np.ndarray = np.empty((M,), dtype=np.float)

    # Forward filtering
    for n in range(N):
        log_lik_n: np.ndarray = log_lik[..., n]
        np.exp(log_lik_n - log_lik_n.max(), out=lik_n)
        np.dot(alpha_nm1, Gamma[n], out=alpha_n)
        alpha_n *= lik_n
        alpha_n_sum: float = np.sum(alpha_n)

        # Rescale small values
        if alpha_n_sum < small:
            alpha_n *= big

        alpha_nm1 = alpha_n
        alphas[..., n] = alpha_n

    # The FFBS samples
    samples: np.ndarray = np.empty((N,), dtype=np.int8)

    # The uniform samples used to sample the categorical states
    unif_samples: np.ndarray = np.random.uniform(size=samples.shape)

    alpha_N: np.ndarray = alphas[..., N - 1]
    beta_N: np.ndarray = alpha_N / alpha_N.sum()

    state_np1: np.ndarray = np.searchsorted(beta_N.cumsum(), unif_samples[N - 1])

    samples[N - 1] = state_np1

    beta_n: np.ndarray = np.empty((M,), dtype=np.float)

    # Backward sampling
    for n in range(N - 2, -1, -1):
        np.multiply(alphas[..., n], Gamma[n, :, state_np1], out=beta_n)
        beta_n /= np.sum(beta_n)

        state_np1 = np.searchsorted(beta_n.cumsum(), unif_samples[n])
        samples[n] = state_np1

    return samples


class FFBSStep(ArrayStep):
    r"""Forward-filtering backward-sampling steps.

    For a hidden Markov model with state sequence :math:`S_t`, observations
    :math:`y_t`, and parameters :math:`\theta`, this step method samples

    .. math::

        S_T &\sim \operatorname{P}\left( S_T \mid y_{1:T}, \theta \right)
        \\
        S_t \mid S_{t+1} &\sim \operatorname{P}\left( S_{t+1} \mid S_t, \theta \right)
            \operatorname{P}\left( S_{t+1} \mid y_{1:T}, \theta \right)

    """

    name = "ffbs"

    def __init__(self, var, values=None, model=None):

        model = pm.modelcontext(model)

        (var,) = pm.inputvars(var)

        self.dependent_rvs = [
            v
            for v in model.basic_RVs
            if v is not var and var in tt.gof.graph.inputs([v.logpt])
        ]

        # We compile a function--from a Theano graph--that computes the
        # total log-likelihood values for each state in the sequence.
        dependents_log_lik = model.fn(
            tt.sum([v.logp_elemwiset for v in self.dependent_rvs], axis=0)
        )

        self.gamma_0_fn = model.fn(var.distribution.gamma_0)
        self.Gammas_fn = model.fn(var.distribution.Gammas)

        super().__init__([var], [dependents_log_lik], allvars=True)

    def astep(self, point, log_lik_fn, inputs):
        gamma_0 = self.gamma_0_fn(inputs)
        Gammas_t = self.Gammas_fn(inputs)

        M = gamma_0.shape[-1]
        N = point.shape[-1]

        # TODO: Why won't broadcasting work with `log_lik_fn`?  Seems like we
        # could be missing out on a much more efficient/faster approach to this
        # potentially large computation.
        # state_seqs = np.broadcast_to(np.arange(M, dtype=np.int)[..., None], (M, N))
        # log_lik_t = log_lik_fn(state_seqs)
        log_lik_t = np.stack([log_lik_fn(np.broadcast_to(m, N)) for m in range(M)])

        return ffbs_astep(gamma_0, Gammas_t, log_lik_t)

    @staticmethod
    def competence(var):
        distribution = getattr(var.distribution, "parent_dist", var.distribution)

        if isinstance(distribution, DiscreteMarkovChain):
            return Competence.IDEAL
        # elif isinstance(distribution, pm.Bernoulli) or (var.dtype in pm.bool_types):
        #     return Competence.COMPATIBLE

        return Competence.INCOMPATIBLE


class TransMatConjugateStep(ArrayStep):
    r"""Conjugate update steps for a transition matrix with Dirichlet distributed rows conditioned on a state sequence.

    For a hidden Markov model given by

    .. math::

        \Gamma_k &\sim \operatorname{Dir}\left( \alpha_k \right),
        \quad k \in \{1, \dots, M\} \; \text{and} \;
        \alpha_k \in \mathbb{R}^{M}, \; \Gamma_k \in \mathbb{R}^{M \times M}
        \\
        S_t &\sim \operatorname{Cat}\left( \Gamma^\top \pi_t \right)

    this step method samples

    .. math::

        \Gamma_j &\sim \operatorname{P}\left( \Gamma_j \mid S_{1:T}, y_{1:T} \right)
        \\
                 &\sim \operatorname{Dir}\left( \alpha_j + N_j \right)


    where :math:`N_j \in \mathbb{R}^{M}` are counts of observed state
    transitions :math:`j \to k` for :math:`k \in \{1, \dots, K\}` conditional
    on :math:`S_{1:T}`.

    """

    name = "trans-mat-conjugate"

    def __init__(self, dir_priors, hmm_states, values=None, model=None, rng=None):
        """Initialize a `TransMatConjugateStep` object.

        Parameters
        ----------
        dir_priors : list of Dirichlets
            State-ordered from-to prior transition probabilities.
        hmm_states : DiscreteMarkovChain
            The HMM state sequence that uses `dir_priors` as its transition matrix.
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
        N_mat = compute_trans_freqs(states, len(self.dists), counts_only=True)

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

        # TODO: Check that the dependent term is a conjugate type.

        distribution = getattr(var.distribution, "parent_dist", var.distribution)

        if isinstance(distribution, pm.Dirichlet):
            return Competence.COMPATIBLE

        return Competence.INCOMPATIBLE
