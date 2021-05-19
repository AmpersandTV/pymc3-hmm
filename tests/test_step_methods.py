import warnings

import aesara.tensor as at
import numpy as np
import pymc3 as pm
import pytest
import scipy as sp
from aesara.graph.op import get_test_value

from pymc3_hmm.distributions import DiscreteMarkovChain, PoissonZeroProcess
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep, ffbs_step
from pymc3_hmm.utils import compute_steady_state, compute_trans_freqs
from tests.utils import simulate_poiszero_hmm


@pytest.fixture()
def raise_under_overflow():
    with np.errstate(over="raise", under="raise"):
        yield


# All tests in this module will raise on over- and under-flows (unless local
# settings dictate otherwise)
pytestmark = pytest.mark.usefixtures("raise_under_overflow")


def test_ffbs_step():

    np.random.seed(2032)

    # A single transition matrix and initial probabilities vector for each
    # element in the state sequence
    test_Gammas = np.array([[[0.9, 0.1], [0.1, 0.9]]])
    test_gamma_0 = np.r_[0.5, 0.5]

    test_log_lik_0 = np.stack(
        [np.broadcast_to(0.0, 10000), np.broadcast_to(-np.inf, 10000)]
    )
    alphas = np.empty(test_log_lik_0.shape)
    res = np.empty(test_log_lik_0.shape[-1])
    ffbs_step(test_gamma_0, test_Gammas, test_log_lik_0, alphas, res)
    assert np.all(res == 0)

    test_log_lik_1 = np.stack(
        [np.broadcast_to(-np.inf, 10000), np.broadcast_to(0.0, 10000)]
    )
    alphas = np.empty(test_log_lik_1.shape)
    res = np.empty(test_log_lik_1.shape[-1])
    ffbs_step(test_gamma_0, test_Gammas, test_log_lik_1, alphas, res)
    assert np.all(res == 1)

    # A well-separated mixture with non-degenerate likelihoods
    test_seq = np.random.choice(2, size=10000)
    test_obs = np.where(
        np.logical_not(test_seq),
        np.random.poisson(10, 10000),
        np.random.poisson(50, 10000),
    )
    test_log_lik_p = np.stack(
        [sp.stats.poisson.logpmf(test_obs, 10), sp.stats.poisson.logpmf(test_obs, 50)],
    )

    # TODO FIXME: This is a statistically unsound/unstable check.
    assert np.mean(np.abs(test_log_lik_p.argmax(0) - test_seq)) < 1e-2

    alphas = np.empty(test_log_lik_p.shape)
    res = np.empty(test_log_lik_p.shape[-1])
    ffbs_step(test_gamma_0, test_Gammas, test_log_lik_p, alphas, res)
    # TODO FIXME: This is a statistically unsound/unstable check.
    assert np.mean(np.abs(res - test_seq)) < 1e-2

    # "Time"-varying transition matrices that specify strictly alternating
    # states--except for the second-to-last one
    test_Gammas = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ],
        axis=0,
    )

    test_gamma_0 = np.r_[1.0, 0.0]

    test_log_lik = np.tile(np.r_[np.log(0.9), np.log(0.1)], (4, 1))
    test_log_lik[::2] = test_log_lik[::2][:, ::-1]
    test_log_lik = test_log_lik.T

    alphas = np.empty(test_log_lik.shape)
    res = np.empty(test_log_lik.shape[-1])
    ffbs_step(test_gamma_0, test_Gammas, test_log_lik, alphas, res)
    assert np.array_equal(res, np.r_[1, 0, 0, 1])


def test_FFBSStep():

    with pm.Model(), pytest.raises(ValueError):
        P_rv = np.eye(2)[None, ...]
        S_rv = DiscreteMarkovChain("S_t", P_rv, np.r_[1.0, 0.0], shape=10)
        S_2_rv = DiscreteMarkovChain("S_2_t", P_rv, np.r_[0.0, 1.0], shape=10)
        PoissonZeroProcess(
            "Y_t", 9.0, S_rv + S_2_rv, observed=np.random.poisson(9.0, size=10)
        )
        # Only one variable can be sampled by this step method
        ffbs = FFBSStep([S_rv, S_2_rv])

    with pm.Model(), pytest.raises(TypeError):
        S_rv = pm.Categorical("S_t", np.r_[1.0, 0.0], shape=10)
        PoissonZeroProcess("Y_t", 9.0, S_rv, observed=np.random.poisson(9.0, size=10))
        # Only `DiscreteMarkovChains` can be sampled with this step method
        ffbs = FFBSStep([S_rv])

    with pm.Model(), pytest.raises(TypeError):
        P_rv = np.eye(2)[None, ...]
        S_rv = DiscreteMarkovChain("S_t", P_rv, np.r_[1.0, 0.0], shape=10)
        pm.Poisson("Y_t", S_rv, observed=np.random.poisson(9.0, size=10))
        # Only `SwitchingProcess`es can used as dependent variables
        ffbs = FFBSStep([S_rv])

    np.random.seed(2032)

    poiszero_sim, _ = simulate_poiszero_hmm(30, 150)
    y_test = poiszero_sim["Y_t"]

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        P_tt = at.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt))

        pi_0_tt = compute_steady_state(P_rv)

        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=y_test.shape[0])

        PoissonZeroProcess("Y_t", 9.0, S_rv, observed=y_test)

    with test_model:
        ffbs = FFBSStep([S_rv])

    test_point = test_model.test_point.copy()
    test_point["p_0_stickbreaking__"] = poiszero_sim["p_0_stickbreaking__"]
    test_point["p_1_stickbreaking__"] = poiszero_sim["p_1_stickbreaking__"]

    res = ffbs.step(test_point)

    assert np.array_equal(res["S_t"], poiszero_sim["S_t"])


def test_FFBSStep_extreme():
    """Test a long series with extremely large mixture separation (and, thus, very small likelihoods)."""  # noqa: E501

    np.random.seed(2032)

    mu_true = 5000
    poiszero_sim, _ = simulate_poiszero_hmm(9000, mu_true)
    y_test = poiszero_sim["Y_t"]

    with pm.Model() as test_model:
        p_0_rv = poiszero_sim["p_0"]
        p_1_rv = poiszero_sim["p_1"]

        P_tt = at.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt))

        pi_0_tt = poiszero_sim["pi_0"]

        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=y_test.shape[0])
        S_rv.tag.test_value = (y_test > 0).astype(int)

        # This prior is very far from the true value...
        E_mu, Var_mu = 100.0, 10000.0
        mu_rv = pm.Gamma("mu", E_mu ** 2 / Var_mu, E_mu / Var_mu)

        PoissonZeroProcess("Y_t", mu_rv, S_rv, observed=y_test)

    with test_model:
        ffbs = FFBSStep([S_rv])

    test_point = test_model.test_point.copy()
    test_point["p_0_stickbreaking__"] = poiszero_sim["p_0_stickbreaking__"]
    test_point["p_1_stickbreaking__"] = poiszero_sim["p_1_stickbreaking__"]

    with np.errstate(over="ignore", under="ignore"):
        res = ffbs.step(test_point)

    assert np.array_equal(res["S_t"], poiszero_sim["S_t"])

    with test_model, np.errstate(
        over="ignore", under="ignore"
    ), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        mu_step = pm.NUTS([mu_rv])
        ffbs = FFBSStep([S_rv])
        steps = [ffbs, mu_step]
        trace = pm.sample(
            20,
            step=steps,
            cores=1,
            chains=1,
            tune=100,
            n_init=100,
            progressbar=False,
        )

        assert not trace.get_sampler_stats("diverging").all()
        assert trace["mu"].mean() > 1000.0


def test_TransMatConjugateStep():

    with pm.Model() as test_model, pytest.raises(ValueError):
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        transmat = TransMatConjugateStep(p_0_rv)

    np.random.seed(2032)

    poiszero_sim, _ = simulate_poiszero_hmm(30, 150)
    y_test = poiszero_sim["Y_t"]

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        P_tt = at.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt))

        pi_0_tt = compute_steady_state(P_rv)

        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=y_test.shape[0])

        PoissonZeroProcess("Y_t", 9.0, S_rv, observed=y_test)

    with test_model:
        transmat = TransMatConjugateStep(P_rv)

    test_point = test_model.test_point.copy()
    test_point["S_t"] = (y_test > 0).astype(int)

    res = transmat.step(test_point)

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

    assert np.allclose(sampled_trans_mat, true_trans_mat, atol=0.3)


def test_TransMatConjugateStep_subtensors():

    # Confirm that Dirichlet/non-Dirichlet mixed rows can be
    # parsed
    with pm.Model():
        d_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        d_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        p_0_rv = at.as_tensor([0, 0, 1])
        p_1_rv = at.zeros(3)
        p_1_rv = at.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = at.zeros(3)
        p_2_rv = at.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = at.stack([p_0_rv, p_1_rv, p_2_rv])
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt))
        DiscreteMarkovChain("S_t", P_rv, np.r_[1, 0, 0], shape=(10,))

        transmat = TransMatConjugateStep(P_rv)

    assert transmat.row_remaps == {0: 1, 1: 2}
    exp_slices = {0: np.r_[0, 2], 1: np.r_[1, 2]}
    assert exp_slices.keys() == transmat.row_slices.keys()
    assert all(
        np.array_equal(transmat.row_slices[i], exp_slices[i]) for i in exp_slices.keys()
    )

    # Same thing, just with some manipulations of the transition matrix
    with pm.Model():
        d_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        d_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        p_0_rv = at.as_tensor([0, 0, 1])
        p_1_rv = at.zeros(3)
        p_1_rv = at.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = at.zeros(3)
        p_2_rv = at.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = at.horizontal_stack(
            p_0_rv[..., None], p_1_rv[..., None], p_2_rv[..., None]
        )
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt.T))
        DiscreteMarkovChain("S_t", P_rv, np.r_[1, 0, 0], shape=(10,))

        transmat = TransMatConjugateStep(P_rv)

    assert transmat.row_remaps == {0: 1, 1: 2}
    exp_slices = {0: np.r_[0, 2], 1: np.r_[1, 2]}
    assert exp_slices.keys() == transmat.row_slices.keys()
    assert all(
        np.array_equal(transmat.row_slices[i], exp_slices[i]) for i in exp_slices.keys()
    )

    # Use an observed `DiscreteMarkovChain` and check the conjugate results
    with pm.Model():
        d_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        d_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        p_0_rv = at.as_tensor([0, 0, 1])
        p_1_rv = at.zeros(3)
        p_1_rv = at.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = at.zeros(3)
        p_2_rv = at.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = at.horizontal_stack(
            p_0_rv[..., None], p_1_rv[..., None], p_2_rv[..., None]
        )
        P_rv = pm.Deterministic("P_tt", at.shape_padleft(P_tt.T))
        DiscreteMarkovChain(
            "S_t", P_rv, np.r_[1, 0, 0], shape=(4,), observed=np.r_[0, 1, 0, 2]
        )

        transmat = TransMatConjugateStep(P_rv)
