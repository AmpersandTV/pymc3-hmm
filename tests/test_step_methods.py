import numpy as np
import scipy as sp

import theano.tensor as tt

import pymc3 as pm

from theano.gof.op import get_test_value

from tests.utils import simulate_poiszero_hmm

from pymc3_hmm.utils import compute_steady_state, compute_trans_freqs
from pymc3_hmm.distributions import PoissonZeroProcess, HMMStateSeq
from pymc3_hmm.step_methods import ffbs_astep, FFBSStep, TransMatConjugateStep

np.seterr(over="raise", under="raise")


def test_ffbs_astep():

    np.random.seed(2032)

    test_lik_0 = np.stack([np.repeat(1.0, 10000), np.repeat(0.0, 10000)], 1)
    test_lik_1 = np.stack([np.repeat(0.0, 10000), np.repeat(1.0, 10000)], 1)

    test_Gamma_t = np.c_[[0.9, 0.1], [0.1, 0.9]].T
    test_gamma_0 = np.r_[0.5, 0.5]

    # A well-separated mixture with non-degenerate likelihoods
    test_seq = np.random.choice(2, size=10000)
    test_obs = np.where(
        np.logical_not(test_seq),
        np.random.poisson(10, 10000),
        np.random.poisson(50, 10000),
    )
    test_lik_p = np.stack(
        [sp.stats.poisson.pmf(test_obs, 10), sp.stats.poisson.pmf(test_obs, 50)], 1
    )

    # TODO FIXME: This is a statistically unsound/unstable check.
    assert np.mean(test_lik_p.argmax(1) - test_seq) < 1e-2

    res = ffbs_astep(test_gamma_0, test_Gamma_t, test_lik_0)
    assert np.all(res == 0)
    res = ffbs_astep(test_gamma_0, test_Gamma_t, test_lik_1)
    assert np.all(res == 1)
    res = ffbs_astep(test_gamma_0, test_Gamma_t, test_lik_p)
    # TODO FIXME: This is a statistically unsound/unstable check.
    assert np.mean(res - test_seq) < 1e-2


def test_FFBSStep():

    np.random.seed(2032)

    poiszero_sim, _ = simulate_poiszero_hmm(30, 150)
    y_test = poiszero_sim["Y_t"]

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = compute_steady_state(P_rv)

        S_rv = HMMStateSeq("S_t", y_test.shape[0], P_rv, pi_0_tt)

        Y_rv = PoissonZeroProcess("Y_t", 9.0, S_rv, observed=y_test)

    with test_model:
        ffbs = FFBSStep([S_rv])

    test_point = test_model.test_point.copy()
    test_point["p_0_stickbreaking__"] = poiszero_sim["p_0_stickbreaking__"]
    test_point["p_1_stickbreaking__"] = poiszero_sim["p_1_stickbreaking__"]

    res = ffbs.step(test_point)

    # TODO: Is this really a good unit test?
    # assert np.array_equal(res['S_t'], poiszero_sim['S_t'])
    assert (
        np.sum(res["S_t"] - poiszero_sim["S_t"]) / poiszero_sim["S_t"].shape[0] < 0.01
    )


def test_TransMatConjugateStep():

    np.random.seed(2032)

    poiszero_sim, _ = simulate_poiszero_hmm(30, 150)
    y_test = poiszero_sim["Y_t"]

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = compute_steady_state(P_rv)

        S_rv = HMMStateSeq("S_t", y_test.shape[0], P_rv, pi_0_tt)

        Y_rv = PoissonZeroProcess("Y_t", 9.0, S_rv, observed=y_test)

    with test_model:
        transmat = TransMatConjugateStep([p_0_rv, p_1_rv], S_rv)

    test_point = test_model.test_point.copy()
    test_point["S_t"] = (y_test > 0).astype(int)

    res = transmat.step(test_point)

    # states = res['S_t']
    # trans_freq = compute_trans_freqs(states, 2)

    p_0_smpl = get_test_value(
        p_0_rv.distribution.transform.backward(res[p_0_rv.transformed.name])
    )
    p_1_smpl = get_test_value(
        p_1_rv.distribution.transform.backward(res[p_1_rv.transformed.name])
    )

    sampled_trans_mat = np.stack([p_0_smpl, p_1_smpl])

    true_trans_mat = (
        compute_trans_freqs(poiszero_sim["S_t"], 2, counts_only=True)
        + np.c_[[1, 1], [1, 1]]
    )
    true_trans_mat = true_trans_mat / true_trans_mat.sum(0)[..., None]

    # TODO: Come up with a good test.
    assert np.allclose(sampled_trans_mat, true_trans_mat, atol=0.3)
