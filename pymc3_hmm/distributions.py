from copy import copy
from typing import Sequence

import aesara
import aesara.tensor as at
import numpy as np
import pymc3 as pm
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import compute_test_value
from aesara.graph.opt import local_optimizer, pre_greedy_local_optimizer
from aesara.scalar import upcast
from aesara.tensor.basic import make_vector
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.basic import categorical
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from aesara.tensor.random.utils import broadcast_params, normalize_size_param
from aesara.tensor.type_other import NoneConst
from aesara.tensor.var import TensorVariable
from pymc3.aesaraf import change_rv_size
from pymc3.distributions.logprob import _logp, logpt


@local_optimizer([BroadcastTo])
def naive_bcast_rv_lift(fgraph, node):
    """Lift a ``BroadcastTo`` through a ``RandomVariable`` ``Op``.

    XXX: This implementation simply broadcasts the ``RandomVariable``'s
    parameters, which won't always work (e.g. multivariate distributions).

    TODO: Instead, it should use ``RandomVariable.ndim_supp``--and the like--to
    determine which dimensions of each parameter need to be broadcasted.
    Also, this doesn't need to remove ``size`` to perform the lifting, like it
    currently does.
    """

    if not (
        isinstance(node.op, BroadcastTo)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, RandomVariable)
    ):
        return

    bcast_shape = node.inputs[1:]

    assert len(bcast_shape) > 0

    rv_var = node.inputs[0]
    rv_node = rv_var.owner

    if hasattr(fgraph, "dont_touch_vars") and rv_var in fgraph.dont_touch_vars:
        return

    size_lift_res = local_rv_size_lift.transform(fgraph, rv_node)
    if size_lift_res is None:
        lifted_node = rv_node
    else:
        _, lifted_rv = size_lift_res
        lifted_node = lifted_rv.owner

    rng, size, dtype, *dist_params = lifted_node.inputs

    new_dist_params = [at.broadcast_to(param, bcast_shape) for param in dist_params]
    bcasted_node = lifted_node.op.make_node(rng, size, dtype, *new_dist_params)

    if aesara.config.compute_test_value != "off":
        compute_test_value(bcasted_node)

    return [bcasted_node.outputs[1]]


def rv_pull_down(x: TensorVariable, dont_touch_vars=None) -> TensorVariable:
    """Pull a ``RandomVariable`` ``Op`` down through a graph, when possible."""
    if dont_touch_vars is None:
        dont_touch_vars = []

    fgraph = FunctionGraph(outputs=dont_touch_vars, clone=False)

    return pre_greedy_local_optimizer(
        fgraph,
        [
            local_dimshuffle_rv_lift,
            local_subtensor_rv_lift,
            naive_bcast_rv_lift,
        ],
        x,
    )


def non_constant(x):
    x = at.as_tensor_variable(x)
    if isinstance(x, Constant):
        # XXX: This isn't good for `size` parameters, because it could result
        # in `at.get_vector_length` exceptions.
        res = x.type()
        res.tag = copy(res.tag)
        if aesara.config.compute_test_value != "off":
            res.tag.test_value = x.data
        res.name = x.name
        return res
    else:
        return x


class SwitchingProcessFactory(OpFromGraph):
    default_output = 1
    # FIXME: This is just to appease `random_make_inplace`
    inplace = True

    def make_node(self, *inputs):
        # Make the `make_node` signature consistent with the node inputs
        # TODO: This is a hack; make it less so.
        num_expected_inps = len(self.local_inputs) - len(self.shared_inputs)
        if len(inputs) > num_expected_inps:
            inputs = inputs[:num_expected_inps]
        return super().make_node(*inputs)


# Allow `SwitchingProcessFactory`s to be typed as `RandomVariable`s
RandomVariable.register(SwitchingProcessFactory)


def create_switching_process_op(size, states, comp_rvs, output_shape=None):
    # We use `make_vector` to preserve the known/fixed-length of our
    # `size` parameter.  If we made this a simple `at.vector`, some
    # shape-related steps in `RandomVariable` would unnecessarily fail.
    size_param = make_vector(
        *[non_constant(size[i]) for i in range(at.get_vector_length(size))]
    )
    size_param.name = "size"

    # We need to make a copy of the state sequence, because we don't want
    # or need anything above this part of the graph.
    states_param = states.type()
    states_param.name = states.name

    # TODO: We should create shallow copies of the component distributions, as
    # well.  In other words, the inputs to the `Op` we're constructing should
    # be the inputs to these component distributions.
    comp_rv_params = [non_constant(rv) for rv in comp_rvs]

    dtype = upcast(*[rv.type.dtype for rv in comp_rv_params])
    comp_ndim_supp = comp_rv_params[0].owner.op.ndim_supp

    resized_states_param = at.broadcast_to(
        states_param, tuple(size_param) + tuple(states_param.shape)
    )

    def resize_rv(x, size):
        if at.get_vector_length(size):
            return change_rv_size(x, size, expand=True)
        else:
            return x

    resized_comp_rvs = [
        # XXX: This will create new component distributions that are
        # disconnected from the originals!  In other words,
        # any reference to the old ones will be invalidated.
        resize_rv(
            rv_pull_down(at.atleast_1d(comp_rv), comp_rv.owner.inputs), size_param
        )
        for comp_rv in comp_rv_params
    ]

    bcast_states, *bcast_comp_rvs = broadcast_params(
        (resized_states_param,) + tuple(resized_comp_rvs),
        (0,) + (comp_ndim_supp,) * len(resized_comp_rvs),
    )

    if output_shape is not None:
        if comp_ndim_supp > 0 and at.get_vector_length(output_shape) > comp_ndim_supp:
            bcast_states = at.broadcast_to(
                bcast_states, tuple(output_shape[:-comp_ndim_supp])
            )
        bcast_comp_rvs = [at.broadcast_to(rv, output_shape) for rv in bcast_comp_rvs]
    else:
        output_shape = bcast_comp_rvs[0].shape

    assert at.get_vector_length(output_shape) > 0

    res = at.empty(output_shape, dtype=dtype)

    for i, bcasted_comp_rv in enumerate(bcast_comp_rvs):
        i_idx = at.nonzero(at.eq(bcast_states, i))
        indexed_comp_rv = bcasted_comp_rv[i_idx]

        lifted_comp_rv = rv_pull_down(indexed_comp_rv, bcasted_comp_rv.owner.inputs)

        res = at.set_subtensor(res[i_idx], lifted_comp_rv)

    new_op = SwitchingProcessFactory(
        # The first and third parameters are simply placeholders so that the
        # arguments signature matches `RandomVariable`'s
        [at.iscalar(), size_param, at.iscalar(), states_param] + list(comp_rv_params),
        [NoneConst, res],
        inline=True,
        on_unused_input="ignore",
    )

    # Add `RandomVariable`-like "metadata"
    new_op.ndim_supp = comp_ndim_supp + 1
    new_op.ndims_params = (1,) + tuple(comp.ndim for comp in comp_rv_params)

    return new_op


class SwitchingProcess(pm.Distribution):
    """A distribution that models a switching process over arbitrary univariate mixtures and a state sequence.

    This class is like `Mixture`, but without the mixture weights.

    """  # noqa: E501

    def __new__(cls, *args, **kwargs):
        obs = kwargs.get("observed", None)

        if obs is not None:
            # XXX: This is a nasty hack to allow the "size changes" PyMC3
            # performs on observed `RandomVariable`s.
            kwargs["out_shape"] = tuple(obs.shape)

        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def dist(
        cls,
        comp_rvs: Sequence[TensorVariable],
        states: TensorVariable,
        *args,
        size=None,
        out_shape=None,
        rng=None,
        **kwargs
    ):
        """Initialize a `SwitchingProcess` instance.

        Parameters
        ----------
        comp_rvs:
            A list containing `RandomVariable` objects for each mixture component.
        states:
            The hidden state sequence.  It should have a number of states
            equal to the size of `comp_dists`.

        """

        size = normalize_size_param(size)

        out_shape = kwargs.pop("out_shape", None)

        states = at.as_tensor(states)

        new_comp_rvs = []
        for rv in comp_rvs:
            new_rv = at.as_tensor(rv)
            new_rv.tag.value_var = new_rv.type()
            new_comp_rvs.append(new_rv)

        # TODO: Make sure `comp_rvs` are not in the/a model.
        # This will help reduce any rewrite inconsistencies.
        SwitchingProcessOp = create_switching_process_op(
            size,
            states,
            new_comp_rvs,
            output_shape=out_shape,
        )

        rv_var = SwitchingProcessOp(*([0, size, 0, states] + list(new_comp_rvs)))

        testval = kwargs.pop("testval", None)

        if testval is not None:
            rv_var.tag.test_value = testval

        return rv_var


@_logp.register(SwitchingProcessFactory)
def switching_process_logp(op, var, rvs_to_values, *dist_params, **kwargs):
    obs = rvs_to_values.get(var, var)
    states, *comp_rvs = dist_params[: len(op.inputs[3:])]

    obs_tt = at.as_tensor_variable(obs)

    logp_val = at.alloc(-np.inf, *tuple(obs_tt.shape))

    for i, comp_rv in enumerate(comp_rvs):
        i_idx = at.nonzero(at.eq(states, i))
        obs_i = obs_tt[i_idx]

        bcasted_comp_rv = at.broadcast_to(comp_rv, obs_tt.shape)
        indexed_comp_rv = bcasted_comp_rv[i_idx]

        lifted_comp_rv = rv_pull_down(indexed_comp_rv, comp_rv.owner.inputs)

        logp_val = at.set_subtensor(logp_val[i_idx], logpt(lifted_comp_rv, obs_i))

    if kwargs.get("sum", False):
        logp_val = logp_val.sum()

    return logp_val


class PoissonZeroProcess(SwitchingProcess):
    """A Poisson-Dirac-delta (at zero) mixture process.

    The first mixture component (at index 0) is the Dirac-delta at zero, and
    the second mixture component is the Poisson random variable.
    """

    @classmethod
    def dist(cls, mu=None, states=None, rng=None, **kwargs):
        """Initialize a `PoissonZeroProcess` object.

        Parameters
        ----------
        mu: tensor
            The Poisson rate(s)
        states: tensor
            A vector of integer 0-1 states that indicate which component of
            the mixture is active at each point/time.
        """
        mu = at.as_tensor_variable(mu)
        states = at.as_tensor_variable(states)

        # NOTE: This creates distributions that are *not* part of a `Model`
        return super().dist(
            [pm.Constant.dist(0), pm.Poisson.dist(mu, rng=rng)],
            states,
            rng=rng,
            **kwargs
        )


class DiscreteMarkovChainFactory(OpFromGraph):
    # Add `RandomVariable`-like "metadata"
    ndim_supp = 1
    ndims_params = (3, 1)
    default_output = 1
    # FIXME: This is just to appease `random_make_inplace`
    inplace = True


# Allow `DiscreteMarkovChainFactory`s to be typed as `RandomVariable`s
RandomVariable.register(DiscreteMarkovChainFactory)


def create_discrete_mc_op(rng, size, Gammas, gamma_0):

    # Again, we need to preserve the length of this symbolic vector, so we do
    # this.
    size_param = make_vector(
        *[non_constant(size[i]) for i in range(at.get_vector_length(size))]
    )
    size_param.name = "size"

    # We make shallow copies so that unwanted ancestors don't appear in the
    # graph.
    Gammas_param = non_constant(Gammas).type()
    Gammas_param.name = "Gammas_param"

    gamma_0_param = non_constant(gamma_0).type()
    gamma_0_param.name = "gamma_0_param"

    bcast_Gammas_param, bcast_gamma_0_param = broadcast_params(
        (Gammas_param, gamma_0_param), (3, 1)
    )

    # Sample state 0 in each state sequence
    state_0 = categorical(
        bcast_gamma_0_param,
        size=tuple(size_param) + tuple(bcast_gamma_0_param.shape[:-1]),
        # size=at.join(0, size_param, bcast_gamma_0_param.shape[:-1]),
        rng=rng,
    )

    N = bcast_Gammas_param.shape[-3]
    states_shape = tuple(state_0.shape) + (N,)

    bcast_Gammas_param = at.broadcast_to(
        bcast_Gammas_param, states_shape + tuple(bcast_Gammas_param.shape[-2:])
    )

    def loop_fn(n, state_nm1, Gammas_inner, rng):
        gamma_t = Gammas_inner[..., n, :, :]
        idx = tuple(at.ogrid[[slice(None, d) for d in tuple(state_0.shape)]]) + (
            state_nm1.T,
        )
        gamma_t = gamma_t[idx]
        state_n = categorical(gamma_t, rng=rng)
        return state_n.T

    res, _ = aesara.scan(
        loop_fn,
        outputs_info=[{"initial": state_0.T, "taps": [-1]}],
        sequences=[at.arange(N)],
        non_sequences=[bcast_Gammas_param, rng],
        # strict=True,
    )

    return DiscreteMarkovChainFactory(
        [at.iscalar(), size_param, at.iscalar(), Gammas_param, gamma_0_param],
        [rng, res.T],
        inline=False,
        on_unused_input="ignore",
    )


class DiscreteMarkovChain(pm.Distribution):
    """A first-order discrete Markov chain distribution.

    This class characterizes vector random variables consisting of state
    indicator values (i.e. `0` to `M - 1`) that are driven by a discrete Markov
    chain.

    """

    @classmethod
    def dist(cls, Gammas, gamma_0, size=None, rng=None, **kwargs):
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
        """
        gamma_0 = at.as_tensor_variable(pm.floatX(gamma_0))

        assert Gammas.ndim >= 3

        Gammas = at.as_tensor_variable(pm.floatX(Gammas))

        size = normalize_size_param(size)

        if rng is None:
            rng = aesara.shared(np.random.RandomState(), borrow=True)

        # rv_var = create_discrete_mc_op(size, Gammas, gamma_0)
        DiscreteMarkovChainOp = create_discrete_mc_op(rng, size, Gammas, gamma_0)
        rv_var = DiscreteMarkovChainOp(0, size, 0, Gammas, gamma_0)

        testval = kwargs.pop("testval", None)

        if testval is not None:
            rv_var.tag.test_value = testval

        return rv_var


@_logp.register(DiscreteMarkovChainFactory)
def discrete_mc_logp(op, var, rvs_to_values, *dist_params, **kwargs):
    r"""Create a Aesara graph that computes the log-likelihood for a discrete Markov chain.

    This is the log-likelihood for the joint distribution of states, :math:`S_t`, conditional
    on state samples, :math:`s_t`, given by the following:

    .. math::

        \int_{S_0} P(S_1 = s_1 \mid S_0) dP(S_0) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    The first term (i.e. the integral) simply computes the marginal :math:`P(S_1 = s_1)`, so
    another way to express this result is as follows:

    .. math::

        P(S_1 = s_1) \prod^{T}_{t=2} P(S_t = s_t \mid S_{t-1} = s_{t-1})

    """  # noqa: E501

    states = rvs_to_values.get(var, var)
    Gammas, gamma_0 = dist_params[: len(op.inputs[3:])]

    Gammas = at.shape_padleft(Gammas, states.ndim - (Gammas.ndim - 2))

    # Multiply the initial state probabilities by the first transition
    # matrix by to get the marginal probability for state `S_1`.
    # The integral that produces the marginal is essentially
    # `gamma_0.dot(Gammas[0])`
    Gamma_1 = Gammas[..., 0:1, :, :]
    gamma_0 = at.expand_dims(gamma_0, (-3, -1))
    P_S_1 = at.sum(gamma_0 * Gamma_1, axis=-2)

    # The `tt.switch`s allow us to broadcast the indexing operation when
    # the replication dimensions of `states` and `Gammas` don't match
    # (e.g. `states.shape[0] > Gammas.shape[0]`)
    S_1_slices = tuple(
        slice(
            at.switch(at.eq(P_S_1.shape[i], 1), 0, 0),
            at.switch(at.eq(P_S_1.shape[i], 1), 1, d),
        )
        for i, d in enumerate(states.shape)
    )
    S_1_slices = (tuple(at.ogrid[S_1_slices]) if S_1_slices else tuple()) + (
        states[..., 0:1],
    )
    logp_S_1 = at.log(P_S_1[S_1_slices]).sum(axis=-1)

    # These are slices for the extra dimensions--including the state
    # sequence dimension (e.g. "time")--along which which we need to index
    # the transition matrix rows using the "observed" `states`.
    trans_slices = tuple(
        slice(
            at.switch(at.eq(Gammas.shape[i], 1), 0, 1 if i == states.ndim - 1 else 0),
            at.switch(at.eq(Gammas.shape[i], 1), 1, d),
        )
        for i, d in enumerate(states.shape)
    )
    trans_slices = (tuple(at.ogrid[trans_slices]) if trans_slices else tuple()) + (
        states[..., :-1],
    )

    # Select the transition matrix row of each observed state; this yields
    # `P(S_t | S_{t-1} = s_{t-1})`
    P_S_2T = Gammas[trans_slices]

    obs_slices = tuple(slice(None, d) for d in P_S_2T.shape[:-1])
    obs_slices = (tuple(at.ogrid[obs_slices]) if obs_slices else tuple()) + (
        states[..., 1:],
    )
    logp_S_1T = at.log(P_S_2T[obs_slices])

    res = logp_S_1 + at.sum(logp_S_1T, axis=-1)
    res.name = "DiscreteMarkovChain_logp"

    if kwargs.get("sum", False):
        res = res.sum()

    return res
