import numpy as np

try:
    import aesara
    import aesara.tensor as at
except ImportError:
    import theano as aesara
    import theano.tensor as at

import pymc3 as pm
import pytest

from pymc3_hmm.distributions import (
    DiscreteMarkovChain,
    PoissonZeroProcess,
    SwitchingProcess,
    distribution_subset_args,
)
from tests.utils import simulate_poiszero_hmm


def test_DiscreteMarkovChain_str():
    Gammas = at.as_tensor(np.eye(2)[None, ...], name="Gammas")
    gamma_0 = at.as_tensor(np.r_[0, 1], name="gamma_0")

    with pm.Model():
        test_dist = DiscreteMarkovChain("P_rv", Gammas, gamma_0, shape=(2,))

    assert str(test_dist) == "P_rv ~ DiscreteMarkovChain"


def test_DiscreteMarkovChain_random():
    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gamma = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    test_gamma_0 = np.r_[0.0, 1.0]
    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=10).random()
    assert np.all(test_sample == 1)

    test_sample = DiscreteMarkovChain.dist(
        test_Gamma, 1.0 - test_gamma_0, shape=10
    ).random()
    assert np.all(test_sample == 0)
    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=10).random(
        size=12
    )
    assert test_sample.shape == (
        12,
        10,
    )

    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=10).random(
        size=2
    )
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )

    # Now, the same set-up, but--this time--generate two state sequences
    # samples
    test_Gamma = np.array([[[0.8, 0.2], [0.2, 0.8]]])
    test_gamma_0 = np.r_[0.2, 0.8]
    test_sample = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=10).random(
        size=2
    )
    # TODO: Fix the seed, and make sure there's at least one 0 and 1?
    assert test_sample.shape == (2, 10)

    # Two transition matrices--for two distinct state sequences--and one vector
    # of initial probs.
    test_Gamma = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_gamma_0 = np.r_[0.0, 1.0]
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=(2, 10))
    test_sample = test_dist.random()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_sample = test_dist.random(size=3)
    assert np.array_equal(
        test_sample,
        np.tile(np.stack([np.ones(10), np.ones(10)], 0).astype(int), (3, 1, 1)),
    )
    assert test_sample.shape == (3, 2, 10)

    # Two transition matrices and initial probs. for two distinct state
    # sequences
    test_Gamma = np.stack(
        [np.array([[[1.0, 0.0], [0.0, 1.0]]]), np.array([[[1.0, 0.0], [0.0, 1.0]]])]
    )
    test_gamma_0 = np.stack([np.r_[0.0, 1.0], np.r_[1.0, 0.0]])
    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=(2, 10))
    test_sample = test_dist.random()
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.zeros(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_sample = test_dist.random(size=3)
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

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=3)
    test_sample = test_dist.random()
    assert np.array_equal(test_sample, np.r_[1, 0, 0])

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_sample = test_dist.random(size=3)
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

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=(2, 3))
    test_sample = test_dist.random()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [0, 1, 1]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_sample = test_dist.random(size=3)
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

    test_dist = DiscreteMarkovChain.dist(test_Gamma, test_gamma_0, shape=(2, 3))
    test_sample = test_dist.random()
    assert np.array_equal(test_sample, np.array([[1, 0, 0], [1, 1, 0]]))

    # Now, the same set-up, but--this time--generate three state sequence
    # samples
    test_sample = test_dist.random(size=3)
    assert np.array_equal(
        test_sample, np.tile(np.array([[1, 0, 0], [1, 1, 0]]).astype(int), (3, 1, 1))
    )


def test_DiscreteMarkovChain_point():
    test_Gammas = at.as_tensor_variable(np.array([[[1.0, 0.0], [0.0, 1.0]]]))

    with pm.Model():
        # XXX: `draw_values` won't use the `Deterministic`s values in the `point` map!
        # Also, `Constant` is only for integer types (?!), so we can't use that.
        test_gamma_0 = pm.Dirichlet("gamma_0", np.r_[1.0, 1000.0], shape=2)
        test_point = {"gamma_0": np.r_[1.0, 0.0]}
        assert np.all(
            DiscreteMarkovChain.dist(test_Gammas, test_gamma_0, shape=10).random(
                point=test_point
            )
            == 0
        )
        assert np.all(
            DiscreteMarkovChain.dist(test_Gammas, 1.0 - test_gamma_0, shape=10).random(
                point=test_point
            )
            == 1
        )


def test_DiscreteMarkovChain_logp():
    aesara.config.compute_test_value = "warn"

    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gammas = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    test_gamma_0 = np.r_[1.0, 0.0]
    test_obs = np.r_[1, 0, 1, 0]

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )
    test_logp_tt = test_dist.logp(test_obs)
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

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

    assert test_logp_tt.eval() == 0

    # Static transition matrix and two state sequences
    test_Gammas = np.array([[[0.0, 1.0], [1.0, 0.0]]])

    test_obs = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])

    test_gamma_0 = np.r_[0.5, 0.5]

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # Time-varying transition matrices and two state sequences
    test_Gammas = np.stack(
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

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

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

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

    test_logp = test_logp_tt.eval()
    assert test_logp[0] == test_logp[1]

    # Two sets of time-varying transition matrices--via `gamma_0`
    # broadcasting--and two state sequences
    test_gamma_0 = np.array([[0.5, 0.5], [0.5, 0.5]])

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

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

    test_dist = DiscreteMarkovChain.dist(
        test_Gammas, test_gamma_0, shape=test_obs.shape[-1]
    )

    test_logp_tt = test_dist.logp(test_obs)

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


def test_SwitchingProcess_random():
    test_states = np.r_[0, 0, 1, 1, 0, 1]
    mu_zero_nonzero = [pm.Constant.dist(0), pm.Constant.dist(1)]
    test_dist = SwitchingProcess.dist(mu_zero_nonzero, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states > 0] > 0)

    test_sample = test_dist.random(size=5)
    assert np.array_equal(test_sample.shape, (5,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    test_dist = SwitchingProcess.dist(mu_zero_nonzero, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random(size=1)
    assert np.array_equal(test_sample.shape, (1,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_mus = [pm.Constant.dist(i) for i in range(6)]
    test_dist = SwitchingProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.c_[0, 0, 1, 1, 0, 1].T
    test_mus = np.arange(1, 6).astype(float)
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
    assert np.array_equal(test_dist.shape, (6, 5))
    test_sample = test_dist.random()
    assert np.array_equal(test_sample.shape, test_dist.shape)
    assert np.all(test_sample[np.where(test_states > 0)[0]] == test_mus)

    test_states = np.c_[0, 0, 1, 1, 0, 1]
    test_mus = np.arange(1, 7).astype(float)
    test_dist = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(test_mus)], test_states
    )
    assert np.array_equal(test_dist.shape, test_states.shape)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_sample = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(test_mus)], test_states
    ).random(size=3)
    assert np.array_equal(test_sample.shape, (3,) + test_mus.shape)
    assert np.all(test_sample.sum(0)[..., test_states > 0] > 0)


def test_PoissonZeroProcess_point():
    test_states = np.r_[0, 0, 1, 1, 0, 1]

    with pm.Model():
        test_mean = pm.Constant("c", 1000.0)
        test_point = {"c": 100.0}
        test_sample = PoissonZeroProcess.dist(test_mean, test_states).random(
            point=test_point
        )

    assert np.all(0 < test_sample[..., test_states > 0])
    assert np.all(test_sample[..., test_states > 0] < 200)


def test_random_PoissonZeroProcess_DiscreteMarkovChain():
    poiszero_sim, test_model = simulate_poiszero_hmm(30, 5000)

    y_test = poiszero_sim["Y_t"].squeeze()
    nonzeros_idx = poiszero_sim["S_t"] > 0

    assert np.all(y_test[nonzeros_idx] > 0)
    assert np.all(y_test[~nonzeros_idx] == 0)


def test_SwitchingProcess():

    np.random.seed(2023532)

    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [pm.Constant.dist(0), pm.Poisson.dist(100.0), pm.Poisson.dist(1000.0)]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)

    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] == 0)
    assert np.all(0 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 1000)
    assert np.all(100 < test_sample[test_states == 2])

    test_mus = np.r_[100, 100, 500, 100, 100, 100]
    test_dists = [
        pm.Constant.dist(0),
        pm.Poisson.dist(test_mus),
        pm.Poisson.dist(10000.0),
    ]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)

    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(200 < test_sample[2] < 600)
    assert np.all(0 < test_sample[5] < 200)
    assert np.all(5000 < test_sample[test_states == 2])

    test_dists = [pm.Constant.dist(0), pm.Poisson.dist(100.0), pm.Poisson.dist(1000.0)]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    for i in range(len(test_dists)):
        test_logp = test_dist.logp(
            np.tile(test_dists[i].mode.eval(), test_states.shape)
        ).eval()
        assert test_logp[test_states != i].max() < test_logp[test_states == i].min()

    # Try a continuous mixture
    test_states = np.r_[2, 0, 1, 2, 0, 1]
    test_dists = [
        pm.Normal.dist(0.0, 1.0),
        pm.Normal.dist(100.0, 1.0),
        pm.Normal.dist(1000.0, 1.0),
    ]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)

    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states == 0] < 10)
    assert np.all(50 < test_sample[test_states == 1])
    assert np.all(test_sample[test_states == 1] < 150)
    assert np.all(900 < test_sample[test_states == 2])

    # Make sure we can use a large number of distributions in the mixture
    test_states = np.ones(50)
    test_dists = [pm.Constant.dist(i) for i in range(50)]
    test_dist = SwitchingProcess.dist(test_dists, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)

    with pytest.raises(TypeError):
        SwitchingProcess.dist([1], test_states)

    with aesara.change_flags(compute_test_value="off"):
        # Test for the case when a default can't be computed
        test_dist = pm.Poisson.dist(at.scalar())

        # Confirm that there's no default
        with pytest.raises(AttributeError):
            test_dist.default()

        # Let it try to sample using `Distribution.random` and fail
        with pytest.raises(ValueError):
            SwitchingProcess.dist([test_dist], test_states)

    # Evaluate multiple observed state sequences in an extreme case
    test_states = at.imatrix("states")
    test_states.tag.test_value = np.zeros((10, 4)).astype("int32")
    test_dist = SwitchingProcess.dist(
        [pm.Constant.dist(0), pm.Constant.dist(1)], test_states
    )
    test_obs = np.tile(np.arange(4), (10, 1)).astype("int32")
    test_logp = test_dist.logp(test_obs)
    exp_logp = np.tile(
        np.array([0.0] + [-np.inf] * 3, dtype=aesara.config.floatX), (10, 1)
    )
    assert np.array_equal(test_logp.tag.test_value, exp_logp)


def test_subset_args():

    test_dist = pm.Constant.dist(c=np.r_[0.1, 1.2, 2.3])
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx)
    assert np.array_equal(res[0].eval(), np.r_[0.1, 2.3])

    test_point = {"c": np.r_[2.0, 3.0, 4.0]}
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx, point=test_point)
    assert np.array_equal(res[0].eval(), np.r_[2.0, 4.0])

    test_dist = pm.Normal.dist(mu=np.r_[0.1, 1.2, 2.3], sigma=np.r_[10.0])
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx)
    assert np.array_equal(res[0].eval(), np.r_[0.1, 2.3])
    assert np.array_equal(res[1].eval(), np.r_[10.0, 10.0])

    test_point = {"mu": np.r_[2.0, 3.0, 4.0], "sigma": np.r_[20.0, 30.0, 40.0]}
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx, point=test_point)
    assert np.array_equal(res[0].eval(), np.r_[2.0, 4.0])
    assert np.array_equal(res[1].eval(), np.r_[20.0, 40.0])

    test_dist = pm.Poisson.dist(mu=np.r_[0.1, 1.2, 2.3])
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx)
    assert np.array_equal(res[0].eval(), np.r_[0.1, 2.3])

    test_point = {"mu": np.r_[2.0, 3.0, 4.0]}
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx, point=test_point)
    assert np.array_equal(res[0].eval(), np.r_[2.0, 4.0])

    test_dist = pm.NegativeBinomial.dist(mu=np.r_[0.1, 1.2, 2.3], alpha=2)
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx)
    assert np.array_equal(res[0].eval(), np.r_[0.1, 2.3])
    assert np.array_equal(res[1].eval(), np.r_[2.0, 2.0])

    test_point = {"mu": np.r_[2.0, 3.0, 4.0], "alpha": np.r_[10, 11, 12]}
    test_idx = np.r_[0, 2]
    res = distribution_subset_args(test_dist, shape=[3], idx=test_idx, point=test_point)
    assert np.array_equal(res[0].eval(), np.r_[2.0, 4.0])
    assert np.array_equal(res[1].eval(), np.r_[10, 12])
