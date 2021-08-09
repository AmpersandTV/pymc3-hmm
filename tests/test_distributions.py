import aesara
import aesara.tensor as at
import numpy as np
import pymc3 as pm
from aesara.compile.mode import get_mode
from pymc3.distributions.logprob import logpt, logpt_sum

from pymc3_hmm.distributions import (
    DiscreteMarkovChain,
    PoissonZeroProcess,
    SwitchingProcess,
    discrete_mc_logp,
    switching_process_logp,
)
from tests.utils import assert_no_rvs, simulate_poiszero_hmm


def dmc_logp(rv_var, obs):
    value_var = rv_var.tag.value_var
    return discrete_mc_logp(
        rv_var.owner.op, value_var, {value_var: obs}, *rv_var.owner.inputs[3:]
    )


def sp_logp(rv_var, obs):
    value_var = rv_var.tag.value_var
    return switching_process_logp(
        rv_var.owner.op, value_var, {value_var: obs}, *rv_var.owner.inputs[3:]
    )


def test_DiscreteMarkovChain_random():
    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gamma_base = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    test_Gamma = np.broadcast_to(test_Gamma_base, (10, 2, 2))
    test_gamma_0 = np.r_[0.0, 1.0]

    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0).eval()
    assert np.all(test_sample == 1)

    test_sample = DiscreteMarkovChain.dist(
        test_Gamma, 1.0 - test_gamma_0, size=10
    ).eval()
    assert np.all(test_sample == 0)

    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=12).eval()
    assert test_sample.shape == (12, 10)

    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=2).eval()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )

    # Now, the same set-up, but--this time--generate two state sequences
    # samples
    test_Gamma_base = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    test_Gamma = np.broadcast_to(test_Gamma_base, (10, 2, 2))
    test_gamma_0 = np.r_[0.2, 0.8]
    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=2).eval()
    # TODO: Fix the seed, and make sure there's at least one 0 and 1?
    assert test_sample.shape == (2, 10)

    # Two transition matrices--for two distinct state sequences--and one vector
    # of initial probs.
    test_Gamma_base = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_Gamma = np.broadcast_to(test_Gamma_base, (2, 10, 2, 2))
    test_gamma_0 = np.r_[0.0, 1.0]

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=3)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample,
        np.tile(np.stack([np.ones(10), np.ones(10)], 0).astype(int), (3, 1, 1)),
    )
    assert test_sample.shape == (3, 2, 10)

    # Two transition matrices and initial probs. for two distinct state
    # sequences
    test_Gamma_base = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_Gamma = np.broadcast_to(test_Gamma_base, (2, 10, 2, 2))
    test_gamma_0 = np.stack([np.r_[0.0, 1.0], np.r_[1.0, 0.0]])
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.zeros(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=3)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample,
        np.tile(np.stack([np.ones(10), np.zeros(10)], 0).astype(int), (3, 1, 1)),
    )
    assert test_sample.shape == (3, 2, 10)

    # "Time"-varying transition matrices with a single vector of initial
    # probabilities
    test_Gamma = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.r_[1, 0]

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample, np.r_[1, 0, 0])

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=3)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample, np.tile(np.r_[1, 0, 0].astype(int), (3, 1)))

    # "Time"-varying transition matrices with two initial
    # probabilities vectors
    test_Gamma = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.array([[1, 0], [0, 1]])

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [0, 1, 1]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=3)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample, np.tile(np.array([[1, 0, 0], [0, 1, 1]]).astype(int), (3, 1, 1))
    )

    # Two "Time"-varying transition matrices with two initial
    # probabilities vectors
    test_Gamma = np.stack(
        [
            [
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
            ],
            [
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
            ],
        ],
        axis=0,
    )
    test_gamma_0 = np.array([[1, 0], [0, 1]])

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [1, 1, 0]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, size=3)
    test_sample = test_dist.eval()
    assert np.array_equal(
        test_sample, np.tile(np.array([[1, 0, 0], [1, 1, 0]]).astype(int), (3, 1, 1))
    )


def test_DiscreteMarkovChain_model():
    N = 50

    with pm.Model(rng_seeder=np.random.RandomState(202353)):
        p_0_rv = pm.Dirichlet("p_0", np.ones(2))
        p_1_rv = pm.Dirichlet("p_1", np.ones(2))

        P_tt = at.stack([p_0_rv, p_1_rv])
        P_tt = at.broadcast_to(P_tt, (N,) + tuple(P_tt.shape))
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = pm.Dirichlet("pi_0", np.ones(2))

        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt)

    mode = get_mode(None).excluding("random_make_inplace")
    test_sample_fn = aesara.function([], [S_rv], mode=mode)

    test_sample = test_sample_fn()

    assert len(np.unique(test_sample)) > 1


def test_DiscreteMarkovChain_logp():
    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gammas_base = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    test_gamma_0 = np.r_[1.0, 0.0]
    test_obs = np.r_[1, 0, 1, 0]
    test_Gammas = np.broadcast_to(test_Gammas_base, (test_obs.shape[-1], 2, 2))

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_logp_tt = assert_no_rvs(logpt(test_dist, test_obs))
    assert test_logp_tt.eval() == 0

    # "Time"-varying transition matrices with a single vector of initial
    # probabilities
    test_Gammas = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.r_[1.0, 0.0]
    test_obs = np.r_[1, 0, 1, 0]

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_logp_tt = assert_no_rvs(logpt(test_dist, test_obs))

    assert test_logp_tt.eval() == 0

    # Static transition matrix and two state sequences
    test_Gammas_base = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    test_obs = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    test_gamma_0 = np.r_[0.5, 0.5]
    test_Gammas = np.broadcast_to(test_Gammas_base, (test_obs.shape[-1], 2, 2))

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_dist.tag.value_var = test_dist.clone()

    test_logp_tt = assert_no_rvs(dmc_logp(test_dist, test_obs))

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # Time-varying transition matrices and two state sequences
    test_Gammas_base = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ],
        axis=0,
    )
    test_obs = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    test_gamma_0 = np.r_[0.5, 0.5]

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_dist.tag.value_var = test_dist.clone()

    test_logp_tt = assert_no_rvs(dmc_logp(test_dist, test_obs))

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # Two sets of time-varying transition matrices and two state sequences
    test_Gammas = np.stack(
        [
            [
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
                np.array([[0.0, 1.0], [1.0, 0.0]]),
            ],
            [
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [0.0, 1.0]]),
            ],
        ],
        axis=0,
    )
    test_obs = np.array([[1, 0, 1, 0], [0, 0, 0, 0]])
    test_gamma_0 = np.r_[0.5, 0.5]

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_dist.tag.value_var = test_dist.clone()

    test_logp_tt = assert_no_rvs(dmc_logp(test_dist, test_obs))

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # Two sets of time-varying transition matrices--via `gamma_0`
    # broadcasting--and two state sequences
    test_gamma_0 = np.array([[0.5, 0.5], [0.5, 0.5]])

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_dist.tag.value_var = test_dist.clone()

    test_logp_tt = assert_no_rvs(dmc_logp(test_dist, test_obs))

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # "Time"-varying transition matrices with a single vector of initial
    # probabilities, but--this time--with better test values
    test_Gammas = np.stack(
        [
            np.array([[0.1, 0.9], [0.5, 0.5]]),
            np.array([[0.2, 0.8], [0.6, 0.4]]),
            np.array([[0.3, 0.7], [0.7, 0.3]]),
            np.array([[0.4, 0.6], [0.8, 0.2]]),
        ],
        axis=0,
    )
    test_gamma_0 = np.r_[0.3, 0.7]
    test_obs = np.r_[1, 0, 1, 0]

    test_dist = DiscreteMarkovChain.dist(test_Gammas, test_gamma_0)
    test_dist.tag.value_var = test_dist.clone()

    test_logp_tt = assert_no_rvs(dmc_logp(test_dist, test_obs))

    logp_res = test_logp_tt.eval()

    logp_exp = np.concatenate(
        [
            test_gamma_0.dot(test_Gammas[0])[None, ...],
            test_Gammas[(np.ogrid[1:4], test_obs[:-1])],
        ],
        axis=-2,
    )
    logp_exp = logp_exp[(np.ogrid[:4], test_obs)]
    logp_exp = np.log(logp_exp).sum()
    assert np.allclose(logp_res, logp_exp)

    test_logp = assert_no_rvs(logpt_sum(test_dist, test_obs))
    test_logp_val = test_logp.eval()
    assert test_logp_val.shape == ()


def test_SwitchingProcess_random():
    test_states = np.r_[0, 0, 1, 1, 0, 1]
    mu_zero_nonzero = [pm.Constant.dist(0), pm.Constant.dist(1)]
    test_dist = SwitchingProcess.dist(mu_zero_nonzero, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)
    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states > 0] > 0)

    test_sample = SwitchingProcess.dist(mu_zero_nonzero, test_states, size=5).eval()
    assert np.array_equal(test_sample.shape, (5,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = at.lvector("states")
    test_states.tag.test_value = np.r_[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    test_dist = SwitchingProcess.dist(mu_zero_nonzero, test_states)
    assert np.array_equal(
        test_dist.shape.eval({test_states: test_states.tag.test_value}),
        test_states.tag.test_value.shape,
    )
    test_sample = SwitchingProcess.dist(mu_zero_nonzero, test_states, size=1).eval(
        {test_states: test_states.tag.test_value}
    )
    assert np.array_equal(test_sample.shape, (1,) + test_states.tag.test_value.shape)
    assert np.all(test_sample[..., test_states.tag.test_value > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_mus = [pm.Constant.dist(i) for i in range(6)]
    test_dist = SwitchingProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.c_[0, 0, 1, 1, 0, 1].T
    test_mus = np.arange(1, 6).astype(np.float64)
    # One of the states has emissions that are a sequence of five Dirac delta
    # distributions on the values 1 to 5 (i.e. the one with values
    # `test_mus`), and the other is just a single delta at 0.  A single state
    # sample from this emissions mixture is a length five array of zeros or the
    # values 1 to 5.
    # Instead of specifying a state sequence containing only one state, we use
    # six state sequences--each of length one.  This should give us six samples
    # of either five zeros or the values 1 to 5.
    test_dist = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(test_mus)], test_states
    )
    assert np.array_equal(test_dist.shape.eval(), (6, 5))
    test_sample = test_dist.eval()
    assert np.array_equal(test_sample.shape, test_dist.shape.eval())
    sample_mus = test_sample[np.where(test_states > 0)[0]]
    assert np.all(sample_mus == test_mus)

    test_states = np.c_[0, 0, 1, 1, 0, 1]
    test_mus = np.arange(1, 7).astype(np.float64)
    test_dist = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(test_mus)], test_states
    )
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_sample = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(test_mus)], test_states, size=3
    ).eval()
    assert np.array_equal(test_sample.shape, (3,) + test_mus.shape)
    assert np.all(test_sample.sum(0)[..., test_states > 0] > 0)

    # Some misc. tests
    rng = aesara.shared(np.random.RandomState(2023532), borrow=True)

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [
        pm.Constant.dist(0),
        pm.Poisson.dist(100.0, rng=rng),
        pm.Poisson.dist(1000.0, rng=rng),
    ]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] == 0)
    assert np.all(0 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 1000)
    assert np.all(100 < test_sample[test_states == 2])

    test_mus = np.r_[100, 100, 500, 100, 100, 100]
    test_dists = [
        pm.Constant.dist(0),
        pm.Poisson.dist(test_mus, rng=rng),
        pm.Poisson.dist(10000.0, rng=rng),
    ]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(200 < test_sample[2] < 600)
    assert np.all(0 < test_sample[5] < 200)
    assert np.all(5000 < test_sample[test_states == 2])

    # Try a continuous mixture
    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [
        pm.Normal.dist(0.0, 1.0, rng=rng),
        pm.Normal.dist(100.0, 1.0, rng=rng),
        pm.Normal.dist(1000.0, 1.0, rng=rng),
    ]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)

    test_sample = test_dist.eval()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] < 10)
    assert np.all(50 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 150)
    assert np.all(900 < test_sample[test_states == 2])

    # Make sure we can use a large number of distributions in the mixture
    test_states = np.ones(50)
    test_dists = [pm.Constant.dist(i) for i in range(50)]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape.eval(), test_states.shape)


def test_SwitchingProcess_logp():

    rng = aesara.shared(np.random.RandomState(2023532), borrow=True)

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_comp_dists = [
        pm.Constant.dist(0),
        pm.Poisson.dist(100.0, rng=rng),
        pm.Poisson.dist(1000.0, rng=rng),
    ]

    test_dist = SwitchingProcess.dist(test_comp_dists, test_states)
    test_dist.tag.value_var = test_dist.clone()

    for i in range(len(test_comp_dists)):
        obs = np.tile(test_comp_dists[i].owner.inputs[3].eval(), test_states.shape)
        test_logp = assert_no_rvs(sp_logp(test_dist, obs)).eval()
        assert test_logp[test_states != i].max() < test_logp[test_states == i].min()

    # Evaluate multiple observed state sequences in an extreme case
    test_states = at.imatrix("states")
    test_states.tag.test_value = np.zeros((10, 4)).astype("int32")

    test_dist = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(1)], test_states
    )
    test_dist.tag.value_var = test_dist.clone()

    test_obs = np.tile(np.arange(4), (10, 1)).astype("int32")
    test_logp = assert_no_rvs(sp_logp(test_dist, test_obs))
    exp_logp = np.tile(
        np.array([0.0] + [-np.inf] * 3, dtype=aesara.config.floatX), (10, 1)
    )
    assert np.array_equal(
        test_logp.eval({test_states: test_states.tag.test_value}), exp_logp
    )

    np.random.seed(4343)
    test_states = at.lvector("states")
    test_states.tag.test_value = np.random.randint(0, 2, size=10, dtype=np.int64)
    test_dist = SwitchingProcess.dist(
        [
            pm.Constant.dist(0),
            pm.Poisson.dist(at.arange(test_states.shape[0]), rng=rng),
        ],
        test_states,
    )
    test_dist.tag.value_var = test_dist.clone()

    test_obs = np.stack([np.zeros(10), np.random.poisson(np.arange(10))])
    test_obs = test_obs[
        (test_states.tag.test_value,) + tuple(np.ogrid[:2, :10])[1:]
    ].squeeze()

    test_logp = assert_no_rvs(sp_logp(test_dist, test_obs))
    test_logp_val = test_logp.eval({test_states: test_states.tag.test_value})
    assert test_logp_val.shape == (10,)

    test_logp = assert_no_rvs(logpt_sum(test_dist, test_obs))
    test_logp_val = test_logp.eval({test_states: test_states.tag.test_value})
    assert test_logp_val.shape == ()


def test_PoissonZeroProcess_model():
    with pm.Model(rng_seeder=np.random.RandomState(2023532)):
        test_mean = pm.Constant("c", 1000.0)
        states = pm.Bernoulli("states", 0.5, size=10)
        Y = PoissonZeroProcess.dist(test_mean, states)

    # We want to make sure that the sampled states and observations correspond,
    # because, if there are any zero states with non-zero observations, we know
    # that the sampled states weren't actually used to draw the observations,
    # and that's a big problem
    sample_fn = aesara.function([], [states, Y])

    fgraph = sample_fn.maker.fgraph
    nodes = list(fgraph.apply_nodes)
    bernoulli_nodes = set(
        n for n in nodes if isinstance(n.op, type(at.random.bernoulli))
    )
    assert len(bernoulli_nodes) == 1

    for i in range(100):
        test_states, test_Y = sample_fn()
        assert np.all(0 < test_Y[..., test_states > 0])
        assert np.all(test_Y[..., test_states > 0] < 10000)


def test_random_PoissonZeroProcess_DiscreteMarkovChain():
    rng = np.random.RandomState(230)

    poiszero_sim, test_model = simulate_poiszero_hmm(30, 5000, rng=rng)

    assert poiszero_sim.keys() == {"P_tt", "S_t", "p_1", "p_0", "Y_t", "pi_0"}

    y_test = poiszero_sim["Y_t"].squeeze()
    nonzeros_idx = poiszero_sim["S_t"] > 0

    assert np.all(y_test[nonzeros_idx] > 0)
    assert np.all(y_test[~nonzeros_idx] == 0)
