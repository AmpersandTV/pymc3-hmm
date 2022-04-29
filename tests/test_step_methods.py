import warnings

import numpy as np
import pymc3 as pm
import pytest
import scipy as sp
import theano.tensor as tt
from pymc3.exceptions import SamplingError
from theano import shared
from theano.graph.op import get_test_value
from theano.sparse import structured_dot as sp_dot

from pymc3_hmm.distributions import DiscreteMarkovChain, HorseShoe, PoissonZeroProcess
from pymc3_hmm.step_methods import (
    FFBSStep,
    HSStep,
    TransMatConjugateStep,
    ffbs_step,
    hs_step,
    large_p_mvnormal_sampler,
)
from pymc3_hmm.utils import compute_steady_state, compute_trans_freqs
from tests.utils import simulate_poiszero_hmm

# @pytest.fixture()
# def raise_under_overflow():
#     with np.errstate(over="raise", under="raise"):
#         yield

# All tests in this module will raise on over- and under-flows (unless local
# settings dictate otherwise)
# pytestmark = pytest.mark.usefixtures("raise_under_overflow")


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

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))

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

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))

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

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))

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

        p_0_rv = tt.as_tensor([0, 0, 1])
        p_1_rv = tt.zeros(3)
        p_1_rv = tt.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = tt.zeros(3)
        p_2_rv = tt.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = tt.stack([p_0_rv, p_1_rv, p_2_rv])
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))
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

        p_0_rv = tt.as_tensor([0, 0, 1])
        p_1_rv = tt.zeros(3)
        p_1_rv = tt.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = tt.zeros(3)
        p_2_rv = tt.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = tt.horizontal_stack(
            p_0_rv[..., None], p_1_rv[..., None], p_2_rv[..., None]
        )
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt.T))
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

        p_0_rv = tt.as_tensor([0, 0, 1])
        p_1_rv = tt.zeros(3)
        p_1_rv = tt.set_subtensor(p_0_rv[[0, 2]], d_0_rv)
        p_2_rv = tt.zeros(3)
        p_2_rv = tt.set_subtensor(p_1_rv[[1, 2]], d_1_rv)

        P_tt = tt.horizontal_stack(
            p_0_rv[..., None], p_1_rv[..., None], p_2_rv[..., None]
        )
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt.T))
        DiscreteMarkovChain(
            "S_t", P_rv, np.r_[1, 0, 0], shape=(4,), observed=np.r_[0, 1, 0, 2]
        )

        transmat = TransMatConjugateStep(P_rv)


def test_large_p_mvnormal_sampler():

    # test case for dense matrix
    np.random.seed(2032)
    X = np.random.normal(size=250).reshape((50, 5))
    beta_true = np.ones(5)
    y = np.random.normal(X.dot(beta_true), 1)

    samples = large_p_mvnormal_sampler(np.ones(5), X, y)
    assert samples.shape == (5,)

    sample_sigma = np.linalg.inv(X.T.dot(X) + np.eye(5))
    sample_mu = sample_sigma.dot(X.T).dot(y)
    np.testing.assert_allclose(sample_mu.mean(), 1, rtol=0.1, atol=0)

    # test case for sparse matrix
    samples_sp = large_p_mvnormal_sampler(np.ones(5), sp.sparse.csr_matrix(X), y)
    assert samples_sp.shape == (5,)
    np.testing.assert_allclose(samples_sp.mean(), 1, rtol=0.1, atol=0)


def test_hs_step():
    # test case for dense matrix
    np.random.seed(2032)
    M = 5
    X = np.random.normal(size=250).reshape((50, M))
    beta_true = np.random.normal(size=M)
    y = np.random.normal(X.dot(beta_true), 1)

    vi = np.full(M, 1)
    lambda2 = np.full(M, 1)
    tau2 = 1
    xi = 1
    beta, lambda2, tau2, vi, xi = hs_step(lambda2, tau2, vi, xi, X, y)
    assert beta.shape == beta_true.shape
    assert (np.abs(beta - beta_true) / beta_true).mean() < 0.5

    # test case for sparse matrix

    vi = np.full(M, 1)
    lambda2 = np.full(M, 1)
    tau2 = 1
    xi = 1
    beta, lambda2, tau2, vi, xi = hs_step(
        lambda2, tau2, vi, xi, sp.sparse.csr_matrix(X), y
    )
    assert beta.shape == beta_true.shape
    assert (np.abs(beta - beta_true) / beta_true).mean() < 0.5


def test_HSStep_Normal():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.random.normal(10, size=M)
    y = np.random.normal(X.dot(beta_true), 1)

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.Normal("y", mu=beta.dot(X.T), sigma=1, observed=y)
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=50,
            tune=0,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (50, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.3)


def test_HSStep_Normal_Deterministic():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.random.normal(10, size=M)
    y = np.random.normal(X.dot(beta_true), 1)

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        # Make `mu` a `Deterministic`
        mu = pm.Deterministic("mu", beta.dot(X.T))
        pm.Normal("y", mu=mu, sigma=1, observed=y)
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=50,
            tune=0,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (50, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.5)


def test_HSStep_unsupported():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.random.normal(10, size=M)
    y = np.random.normal(X.dot(beta_true), 1)

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.TruncatedNormal("alpha", mu=beta.dot(X.T), sigma=1, observed=y)

        with pytest.raises(NotImplementedError):
            HSStep([beta])

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        mu = pm.Deterministic("mu", beta.dot(X.T))
        pm.TruncatedNormal("y", mu=mu, sigma=1, observed=y)

        with pytest.raises(NotImplementedError):
            HSStep([beta])

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        mu = pm.Deterministic("mu", beta.dot(X.T))
        pm.Normal("y", mu=mu, sigma=1, observed=y)
        beta_2 = HorseShoe("beta_2", tau=1, shape=M)
        with pytest.raises(ValueError):
            HSStep([beta, beta_2])


def test_HSStep_sparse():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.random.normal(10, size=M)
    y = np.random.normal(X.dot(beta_true), 1)
    X = sp.sparse.csr_matrix(X)

    M = X.shape[1]
    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.Normal("y", mu=sp_dot(X, tt.shape_padright(beta)), sigma=1, observed=y)
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=50,
            tune=0,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (50, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.3)


def test_HSStep_NegativeBinomial():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.array([1, 1, 2, 2, 0])
    y_nb = pm.NegativeBinomial.dist(np.exp(X.dot(beta_true)), 1).random()

    N_draws = 500
    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.NegativeBinomial("y", mu=tt.exp(beta.dot(X.T)), alpha=1, observed=y_nb)
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=N_draws,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (N_draws, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.5)

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M, testval=beta_true * 0.1)
        pm.NegativeBinomial("y", mu=beta.dot(np.abs(X.T)), alpha=1, observed=y_nb)
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=N_draws,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (N_draws, M)

    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M, testval=beta_true * 0.1)
        eta = pm.NegativeBinomial("eta", mu=beta.dot(X.T), alpha=1, shape=N)
        pm.Normal("y", mu=tt.exp(eta), sigma=1, observed=y_nb)

        with pytest.raises(SamplingError):
            HSStep([beta])
            pm.sample(
                draws=N_draws,
                step=hsstep,
                chains=1,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )


def test_HSStep_NegativeBinomial_sparse():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.array([1, 1, 2, 2, 0])
    y_nb = pm.NegativeBinomial.dist(np.exp(X.dot(beta_true)), 1).random()

    X = sp.sparse.csr_matrix(X)

    N_draws = 500
    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.NegativeBinomial(
            "y", mu=tt.exp(sp_dot(X, tt.shape_padright(beta))), alpha=1, observed=y_nb
        )
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=N_draws,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (N_draws, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.5)


@pytest.mark.xfail(reason="potentially sensitive to sampling and dependency")
def test_HSStep_NegativeBinomial_sparse_shared_y():
    np.random.seed(2032)
    M = 5
    N = 50
    X = np.random.normal(size=N * M).reshape((N, M))
    beta_true = np.array([1, 1, 2, 2, 0])
    y_nb = pm.NegativeBinomial.dist(np.exp(X.dot(beta_true)), 1).random()

    X = sp.sparse.csr_matrix(X)

    X_tt = shared(X, name="X", borrow=True)
    y_tt = shared(y_nb, name="y_t", borrow=True)

    N_draws = 100
    with pm.Model():
        beta = HorseShoe("beta", tau=1, shape=M)
        pm.NegativeBinomial(
            "y",
            mu=tt.exp(sp_dot(X_tt, tt.shape_padright(beta))),
            alpha=1,
            observed=y_tt,
        )
        hsstep = HSStep([beta])
        trace = pm.sample(
            draws=N_draws,
            step=hsstep,
            chains=1,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )

    beta_samples = trace.posterior["beta"][0].values
    assert beta_samples.shape == (N_draws, M)
    np.testing.assert_allclose(beta_samples.mean(0), beta_true, atol=0.5)
