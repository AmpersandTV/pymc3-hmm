from itertools import chain

import numpy as np

try:  # pragma: no cover
    import aesara.scalar as aes
    import aesara.tensor as at
    from aesara.compile import optdb
    from aesara.graph.basic import Variable, graph_inputs
    from aesara.graph.fg import FunctionGraph
    from aesara.graph.op import get_test_value as test_value
    from aesara.graph.opt import OpRemove, pre_greedy_local_optimizer
    from aesara.graph.optdb import Query
    from aesara.tensor.elemwise import DimShuffle, Elemwise
    from aesara.tensor.subtensor import AdvancedIncSubtensor1
    from aesara.tensor.var import TensorConstant
except ImportError:  # pragma: no cover
    import theano.scalar as aes
    import theano.tensor as at
    from theano.compile import optdb
    from theano.graph.basic import Variable, graph_inputs
    from theano.graph.fg import FunctionGraph
    from theano.graph.op import get_test_value as test_value
    from theano.graph.opt import OpRemove, pre_greedy_local_optimizer
    from theano.graph.optdb import Query
    from theano.tensor.elemwise import DimShuffle, Elemwise
    from theano.tensor.subtensor import AdvancedIncSubtensor1
    from theano.tensor.var import TensorConstant

import pymc3 as pm
from pymc3.distributions.distribution import draw_values
from pymc3.step_methods.arraystep import ArrayStep, BlockedStep, Competence
from pymc3.util import get_untransformed_name

from pymc3_hmm.distributions import DiscreteMarkovChain, SwitchingProcess
from pymc3_hmm.utils import compute_trans_freqs

big: float = 1e20
small: float = 1.0 / big


def ffbs_step(
    gamma_0: np.ndarray,
    Gammas: np.ndarray,
    log_lik: np.ndarray,
    alphas: np.ndarray,
    out: np.ndarray,
):
    """Sample a forward-filtered backward-sampled (FFBS) state sequence.

    Parameters
    ----------
    gamma_0
        The initial state probabilities.
    Gamma
        The transition probability matrices.  This array should take the shape
        ``(N, M, M)``, where ``N`` is the state sequence length and ``M`` is
        the number of distinct states.  If ``N`` is ``1``, the single
        transition matrix will broadcast across all elements of the state
        sequence.
    log_lik
        An array of shape `(M, N)` consisting of the log-likelihood values for
        each state value at each point in the sequence.
    alphas
        An array in which to store the forward probabilities.
    out
        An output array to be updated in-place with the posterior sample
        states.

    """
    # Number of observations
    N: int = log_lik.shape[-1]

    # Number of states
    M: int = gamma_0.shape[-1]
    # assert M == log_lik.shape[-2]

    # Initial state probabilities
    gamma_0_normed: np.ndarray = gamma_0.copy()
    gamma_0_normed /= np.sum(gamma_0)

    # Previous forward probability
    alpha_nm1: np.ndarray = gamma_0_normed

    # Make sure we have a transition matrix for each element in a state
    # sequence
    Gamma: np.ndarray = np.broadcast_to(Gammas, (N,) + Gammas.shape[-2:])

    lik_n: np.ndarray = np.empty((M,), dtype=float)
    alpha_n: np.ndarray = np.empty((M,), dtype=float)

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

    # The uniform samples used to sample the categorical states
    unif_samples: np.ndarray = np.random.uniform(size=out.shape)

    alpha_N: np.ndarray = alphas[..., N - 1]
    beta_N: np.ndarray = alpha_N / alpha_N.sum()

    state_np1: np.ndarray = np.searchsorted(beta_N.cumsum(), unif_samples[N - 1])

    out[N - 1] = state_np1

    beta_n: np.ndarray = np.empty((M,), dtype=float)

    # Backward sampling
    for n in range(N - 2, -1, -1):
        np.multiply(alphas[..., n], Gamma[n, :, state_np1], out=beta_n)
        beta_n /= np.sum(beta_n)

        state_np1 = np.searchsorted(beta_n.cumsum(), unif_samples[n])
        out[n] = state_np1

    return out


class FFBSStep(BlockedStep):
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

    def __init__(self, vars, values=None, model=None):

        if len(vars) > 1:
            raise ValueError("This sampler only takes one variable.")

        (var,) = pm.inputvars(vars)

        if not isinstance(var.distribution, DiscreteMarkovChain):
            raise TypeError("This sampler only samples `DiscreteMarkovChain`s.")

        model = pm.modelcontext(model)

        self.vars = [var]

        self.dependent_rvs = [
            v
            for v in model.basic_RVs
            if v is not var and var in graph_inputs([v.logpt])
        ]

        dep_comps_logp_stacked = []
        for i, dependent_rv in enumerate(self.dependent_rvs):
            if isinstance(dependent_rv.distribution, SwitchingProcess):
                comp_logps = []

                # Get the log-likelihoood sequences for each state in this
                # `SwitchingProcess` observations distribution
                for comp_dist in dependent_rv.distribution.comp_dists:
                    comp_logps.append(comp_dist.logp(dependent_rv))

                comp_logp_stacked = at.stack(comp_logps)
            else:
                raise TypeError(
                    "This sampler only supports `SwitchingProcess` observations"
                )

            dep_comps_logp_stacked.append(comp_logp_stacked)

        comp_logp_stacked = at.sum(dep_comps_logp_stacked, axis=0)

        (M,) = draw_values([var.distribution.gamma_0.shape[-1]], point=model.test_point)
        N = model.test_point[var.name].shape[-1]
        self.alphas = np.empty((M, N), dtype=float)

        self.log_lik_states = model.fn(comp_logp_stacked)
        self.gamma_0_fn = model.fn(var.distribution.gamma_0)
        self.Gammas_fn = model.fn(var.distribution.Gammas)

    def step(self, point):
        gamma_0 = self.gamma_0_fn(point)
        # TODO: Can we update these in-place (e.g. using a shared variable)?
        Gammas_t = self.Gammas_fn(point)
        # TODO: Can we update these in-place (e.g. using a shared variable)?
        log_lik_state_vals = self.log_lik_states(point)
        ffbs_step(
            gamma_0, Gammas_t, log_lik_state_vals, self.alphas, point[self.vars[0].name]
        )
        return point

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

    Dirichlet priors can also be embedded in larger transition matrices through
    `theano.tensor.set_subtensor` `Op`s.  See
    `TransMatConjugateStep._set_row_mappings`.

    """  # noqa: E501

    name = "trans-mat-conjugate"

    def __init__(self, model_vars, values=None, model=None, rng=None):
        """Initialize a `TransMatConjugateStep` object."""

        model = pm.modelcontext(model)

        if isinstance(model_vars, Variable):
            model_vars = [model_vars]

        model_vars = list(chain.from_iterable([pm.inputvars(v) for v in model_vars]))

        # TODO: Are the rows in this matrix our `dir_priors`?
        dir_priors = []
        self.dir_priors_untrans = []
        for d in model_vars:
            untrans_var = model.named_vars[get_untransformed_name(d.name)]
            if isinstance(untrans_var.distribution, pm.Dirichlet):
                self.dir_priors_untrans.append(untrans_var)
                dir_priors.append(d)

        state_seqs = [
            v
            for v in model.vars + model.observed_RVs
            if isinstance(v.distribution, DiscreteMarkovChain)
            and all(d in graph_inputs([v.distribution.Gammas]) for d in dir_priors)
        ]

        if not self.dir_priors_untrans or not len(state_seqs) == 1:
            raise ValueError(
                "This step method requires a set of Dirichlet priors"
                " that comprise a single transition matrix"
            )

        (state_seq,) = state_seqs

        Gamma = state_seq.distribution.Gammas

        self._set_row_mappings(Gamma, dir_priors, model)

        if len(self.row_remaps) != len(dir_priors):
            raise TypeError(
                "The Dirichlet priors could not be found"
                " in the graph for {}".format(state_seq.distribution.Gammas)
            )

        if state_seq in model.observed_RVs:
            self.state_seq_obs = np.asarray(state_seq.distribution.data)

        self.rng = rng
        self.dists = list(dir_priors)
        self.state_seq_name = state_seq.name

        super().__init__(dir_priors, [], allvars=True)

    def _set_row_mappings(self, Gamma, dir_priors, model):
        """Create maps from Dirichlet priors parameters to rows and slices in the transition matrix.

        These maps are needed when a transition matrix isn't simply comprised
        of Dirichlet prior rows, but--instead--slices of Dirichlet priors.

        Consider the following:

        .. code-block:: python

            with pm.Model():
                d_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
                d_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

                p_0_rv = tt.as_tensor([0, 0, 1])
                p_1_rv = tt.zeros(3)
                p_1_rv = tt.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
                p_2_rv = tt.zeros(3)
                p_2_rv = tt.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

                P_tt = tt.stack([p_0_rv, p_1_rv, p_2_rv])

        The transition matrix `P_tt` has Dirichlet priors in only two of its
        three rows, and--even then--they're only present in parts of two rows.

        In this example, we need to know that Dirichlet prior 0, i.e. `d_0_rv`,
        is mapped to row 1, and prior 1 is mapped to row 2.  Furthermore, we
        need to know that prior 0 fills columns 0 and 2 in row 1, and prior 1
        fills columns 1 and 2 in row 2.

        These mappings allow one to embed Dirichlet priors in larger transition
        matrices with--for instance--fixed transition behavior.

        """  # noqa: E501

        # Remove unimportant `Op`s from the transition matrix graph
        Gamma = pre_greedy_local_optimizer(
            FunctionGraph([], []),
            [
                OpRemove(Elemwise(aes.Cast(aes.float32))),
                OpRemove(Elemwise(aes.Cast(aes.float64))),
                OpRemove(Elemwise(aes.identity)),
            ],
            Gamma,
        )

        # Canonicalize the transition matrix graph
        fg = FunctionGraph(
            list(graph_inputs([Gamma] + self.dir_priors_untrans)),
            [Gamma] + self.dir_priors_untrans,
            clone=True,
        )
        canonicalize_opt = optdb.query(Query(include=["canonicalize"]))
        canonicalize_opt.optimize(fg)
        Gamma = fg.outputs[0]
        dir_priors_untrans = fg.outputs[1:]
        fg.disown()

        Gamma_DimShuffle = Gamma.owner

        if not (isinstance(Gamma_DimShuffle.op, DimShuffle)):
            raise TypeError("The transition matrix should be non-time-varying")

        Gamma_Join = Gamma_DimShuffle.inputs[0].owner

        if not (isinstance(Gamma_Join.op, at.basic.Join)):
            raise TypeError(
                "The transition matrix should be comprised of stacked row vectors"
            )

        Gamma_rows = Gamma_Join.inputs[1:]

        self.n_rows = len(Gamma_rows)

        # Loop through the rows in the transition matrix's graph and determine
        # how our transformed Dirichlet RVs map to this transition matrix.
        self.row_remaps = {}
        self.row_slices = {}
        for i, dim_row in enumerate(Gamma_rows):
            if not dim_row.owner:
                continue

            # By-pass the `DimShuffle`s applied to the `AdvancedIncSubtensor1`
            # `Op`s in which we're actually interested
            gamma_row = dim_row.owner.inputs[0]

            if gamma_row in dir_priors_untrans:
                # This is a row that's simply a `Dirichlet`
                j = dir_priors_untrans.index(gamma_row)
                self.row_remaps[j] = i
                self.row_slices[j] = slice(None)

            if gamma_row.owner.inputs[1] not in dir_priors_untrans:
                continue

            # Parts of a row set by a `*Subtensor*` `Op` using a full
            # `Dirichlet` e.g. `P_row[idx] = dir_rv`
            j = dir_priors_untrans.index(gamma_row.owner.inputs[1])
            untrans_dirich = dir_priors_untrans[j]

            if (
                gamma_row.owner
                and isinstance(gamma_row.owner.op, AdvancedIncSubtensor1)
                and gamma_row.owner.inputs[1] == untrans_dirich
            ):
                self.row_remaps[j] = i

                rhand_val = gamma_row.owner.inputs[2]
                if not isinstance(rhand_val, TensorConstant):
                    # TODO: We could allow more types of `idx` (e.g. slices)
                    # Currently, `idx` can't be something like `2:5`
                    raise TypeError(
                        "Only array indexing allowed for mixed"
                        " Dirichlet/non-Dirichlet rows"
                    )
                self.row_slices[j] = rhand_val.data

    def astep(self, point, inputs):

        states = getattr(self, "state_seq_obs", None)
        if states is None:
            states = inputs[self.state_seq_name]

        N_mat = compute_trans_freqs(states, self.n_rows, counts_only=True)

        trans_res = [
            d.distribution.dist.transform.forward_val(
                np.random.dirichlet(
                    test_value(d.distribution.dist.a)
                    + N_mat[self.row_remaps[i]][self.row_slices[i]]
                )
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
