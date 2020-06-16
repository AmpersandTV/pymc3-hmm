import numpy as np

from tests.utils import simulate_poiszero_hmm
from pymc3_hmm.distributions import PoissonZeroProcess, HMMStateSeq


def test_HMMStateSeq():
    test_gamma_0 = np.r_[0.0, 1.0]
    test_Gamma = np.stack([[1.0, 0.0], [0.0, 1.0]])
    assert np.all(HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random() == 1)
    assert np.all(HMMStateSeq.dist(10, test_Gamma, 1.0 - test_gamma_0).random() == 0)
    assert HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=12).shape == (
        10,
        12,
    )

    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=2)
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.ones(10)], 1).astype(int)
    )

    # TODO: Fix the seed, and make sure there's at least one 0 and 1
    test_gamma_0 = np.r_[0.2, 0.8]
    test_Gamma = np.stack([[0.8, 0.2], [0.2, 0.8]])
    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random(size=2)
    # test_sample
    assert test_sample.shape == (10, 2)

    test_gamma_0 = np.stack([np.r_[0.0, 1.0], np.r_[1.0, 0.0]])
    test_Gamma = np.stack(
        [np.stack([[1.0, 0.0], [0.0, 1.0]]), np.stack([[1.0, 0.0], [0.0, 1.0]])]
    )
    test_sample = HMMStateSeq.dist(10, test_Gamma, test_gamma_0).random()
    # test_sample
    assert np.array_equal(
        test_sample, np.stack([np.ones(10), np.zeros(10)], 1).astype(int)
    )
    assert test_sample.shape == (10, 2)


def test_PoissonZeroProcess():
    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_dist = PoissonZeroProcess.dist(10.0, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert test_sample.shape == (test_states.shape[0],)
    assert np.all(test_sample[test_states > 0] > 0)

    test_sample = test_dist.random(size=5)
    assert np.array_equal(test_sample.shape, test_states.shape + (5,))
    assert np.all(test_sample[test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    test_dist = PoissonZeroProcess.dist(100.0, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random(size=1)
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_mus = np.r_[10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
    test_dist = PoissonZeroProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[test_states > 0] > 0)

    test_states = np.c_[0, 0, 1, 1, 0, 1].T
    test_dist = PoissonZeroProcess.dist(test_mus, test_states)
    assert np.array_equal(test_dist.shape, test_states.shape)
    test_sample = test_dist.random()
    assert np.array_equal(test_sample.shape, test_states.shape)
    assert np.all(test_sample[test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_sample = PoissonZeroProcess.dist(10.0, test_states).random(size=3)
    assert np.array_equal(test_sample.shape, test_states.shape + (3,))
    assert np.all(test_sample.sum(-1)[test_states > 0] > 0)

    test_states = np.r_[0, 0, 1, 1, 0, 1]
    test_sample = PoissonZeroProcess.dist(10.0, test_states).random(size=3)
    assert np.array_equal(test_sample.shape, test_states.shape + (3,))
    assert np.all(test_sample.sum(-1)[test_states > 0] > 0)


def test_random_PoissonZeroProcess_HMMStateSeq():
    poiszero_sim, test_model = simulate_poiszero_hmm(30, 5000)

    y_test = poiszero_sim["Y_t"].squeeze()
    nonzeros_idx = poiszero_sim["S_t"] > 0

    assert np.all(y_test[nonzeros_idx] > 0)
    assert np.all(y_test[~nonzeros_idx] == 0)
