from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess
from tests.utils import simulate_poiszero_hmm, check_metrics
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np


def test_sampling(N: int = 200, off_param=1):
    kwargs = {
        "N": N,
        "mus": [5000, 7000],
        "pi_0_a": np.r_[1, 1, 1],
        "Gamma": np.r_["0,2,1", [5, 1, 1], [1, 3, 1], [1, 1, 5]],
    }
    simulation, _ = simulate_poiszero_hmm(**kwargs)

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1, 1])
        p_2_rv = pm.Dirichlet("p_2", np.r_[1, 1, 1])

        P_tt = tt.stack([p_0_rv, p_1_rv, p_2_rv])
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = simulation["pi_0"]
        y_test = simulation["Y_t"]

        S_rv = HMMStateSeq("S_t", y_test.shape[0], P_rv, pi_0_tt)
        S_rv.tag.test_value = (y_test > 0).astype(np.int)

        mu_1, mu_2 = kwargs["mus"]

        E_1_mu, Var_1_mu = mu_1 * off_param, mu_1 / 5
        E_2_mu, Var_2_mu = (mu_2) * off_param, (mu_2) * 0.9

        mu_1_rv = pm.Gamma("mu_1", E_1_mu ** 2 / Var_1_mu, E_1_mu / Var_1_mu)
        mu_2_rv = pm.Gamma("mu_2", E_2_mu ** 2 / Var_2_mu, E_2_mu / Var_2_mu)

        Y_rv = SwitchingProcess(
            "Y_t",
            [pm.Constant.dist(0), pm.Poisson.dist(mu_1_rv), pm.Poisson.dist(mu_2_rv),],
            S_rv,
            observed=y_test,
        )

    with test_model:
        mu_step = pm.NUTS([mu_1_rv, mu_2_rv])
        ffbs = FFBSStep([S_rv])
        transitions = TransMatConjugateStep([p_0_rv, p_1_rv, p_2_rv], S_rv)
        steps = [ffbs, mu_step, transitions]
        trace_ = pm.sample(N, step=steps, return_inferencedata=True, chains=1)
        posterior = pm.sample_posterior_predictive(trace_.posterior)
        check_metrics(trace_, posterior, simulation)


# def test_PriorRobust():
#     for j in np.linspace(0.8, 1.2, 3):
#         print(f'off pram : {j}')
#         test_sampling(200, j)
#     assert True
