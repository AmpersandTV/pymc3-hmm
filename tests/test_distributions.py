import numpy as np

import pymc3 as pm

import theano.tensor as tt

from tests.utils import simulate_poiszero_hmm
from pymc3_hmm.distributions import PoissonZeroProcess, HMMStateSeq, SwitchingProcess


def test_HMMStateSeq_random():
    test_gamma_0 = np.r_[0.0, 1.0]
    test_Gamma = np.stack([[1.0, 0.0], [0.0, 1.0]])
    assert np.all(HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random() == 1)
    assert np.all(HMMStateSeq.dist(10, test_Gamma, 1.0 - test_gamma_0).random() == 0)
    assert HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=12).shape == (
        12,
        10,
    )

    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=2)
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 0).astype(int)
    )

    # TODO: Fix the seed, and make sure there's at least one 0 and 1
    test_gamma_0 = np.r_[0.2, 0.8]
    test_Gamma = np.stack([[0.8, 0.2], [0.2, 0.8]])
    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=2)
    # test_sample
    assert test_sample.shape == (2, 10)

    test_gamma_0 = np.stack([np.r_[0.0, 1.0], np.r_[1.0, 0.0]])
    test_Gamma = np.stack(
        [np.stack([[1.0, 0.0], [0.0, 1.0]]), np.stack([[1.0, 0.0], [0.0, 1.0]])]
    )
    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random()
    # test_sample
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.zeros(10)], 0).astype(int)
    )
    assert test_sample.shape == (2, 10)


def test_HMMStateSeq_point():
    test_Gamma = tt.as_tensor_variable(np.stack([[1.0, 0.0], [0.0, 1.0]]))

    with pm.Model():
        # XXX: `draw_values` won't use the `Deterministic`s values in the `point` map!
        # Also, `Constant` is only for integer types (?!), so we can't use that.
        test_gamma_0 = pm.Dirichlet("gamma_0", np.r_[1.0, 1000.0])
        test_point = {"gamma_0": np.r_[1.0, 0.0]}
        assert np.all(
            HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(point=test_point) == 0
        )
        assert np.all(
            HMMStateSeq.dist(10, test_Gamma, 1.0 - test_gamma_0).random(
                point=test_point
            )
            == 1
        )


def test_PoissonZeroProcess_random():
    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_dist = PoissonZeroProcess.dist(10.0, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states > 0] > 0)

    test_sample = test_dist.random(size=5)
    assert np.array_equal(test_sample.shape, (5,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    test_dist = PoissonZeroProcess.dist(100.0, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random(size=1)
    assert np.array_equal(test_sample.shape, (1,) + test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_mus = np.r_[10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    test_dist = PoissonZeroProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[..., test_states > 0] > 0)

    test_states = np.c_[0, 0, 1, 1, 0, 1].T
    test_dist = PoissonZeroProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    # TODO: This seems bad, but also what PyMC3 would do
    assert np.array_equal(test_sample.shape, test_states.squeeze().shape)
    assert np.all(test_sample[..., test_states.squeeze() > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_sample = PoissonZeroProcess.dist(10.0, test_states).random(size=3)
    assert np.array_equal(test_sample.shape, (3,) + test_states.shape)
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


def test_random_PoissonZeroProcess_HMMStateSeq():
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
