from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess
from tests.utils import (
    simulate_poiszero_hmm,
    check_metrics,
)
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import random
import pandas as pd
import patsy

# %%
def gen_design_matrix(N):
    t = pd.date_range(end=pd.to_datetime("today"), periods=N, freq="H").to_frame()
    t["weekday"] = t[0].dt.dayofweek
    t["hour"] = t[0].dt.hour
    t.reset_index()
    formula_str = " 1 +  C(hour) + C(weekday)"
    X_df = patsy.dmatrix(formula_str, t, return_type="dataframe")
    return X_df.values


def test_seasonality_sampling(N: int = 200, off_param=1):
    random.seed(123)

    X_t = gen_design_matrix(N)
    betas_intercept = np.random.normal(3, 0.5, size=1)
    betas_hour = np.sort(np.random.normal(1, 0.5, size=23))
    betas_week = np.sort(np.random.normal(1, 0.5, size=6))

    betas = tt.concatenate([betas_intercept, betas_hour, betas_week])
    eta_r = tt.exp(tt.dot(X_t, betas))

    kwargs = {
        "N": N,
        "mus": np.r_[eta_r],
        "pi_0_a": np.r_[1, 1],
        "Gamma": np.r_["0,2,1", [10, 1], [5, 5]],
    }
    simulation, _ = simulate_poiszero_hmm(**kwargs)

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = simulation["pi_0"]
        y_test = simulation["Y_t"]

        S_rv = HMMStateSeq("S_t", y_test.shape[0], P_rv, pi_0_tt)
        S_rv.tag.test_value = (y_test > 0).astype(np.int)

        X = gen_design_matrix(N)
        beta_s_intercept = pm.Normal("beta_s_intercept", 3, 0.5, shape=(1,))
        beta_s_hour = pm.Normal("beta_s_hour", 1, 0.5, shape=(23,))
        beta_s_week = pm.Normal("beta_s_week", 1, 0.5, shape=(6,))

        beta_s = pm.Deterministic(
            "beta_s", tt.concatenate([beta_s_intercept, beta_s_hour, beta_s_week])
        )
        mu = tt.exp(tt.dot(X, beta_s))

        Y_rv = SwitchingProcess(
            "Y_t", [pm.Constant.dist(0), pm.Poisson.dist(mu)], S_rv, observed=y_test,
        )
    with test_model:
        mu_step = pm.NUTS([mu, beta_s])
        ffbs = FFBSStep([S_rv])
        transitions = TransMatConjugateStep([p_0_rv, p_1_rv], S_rv)
        steps = [ffbs, mu_step, transitions]
        trace_ = pm.sample(N, step=steps, return_inferencedata=True, chains=1)
        posterior = pm.sample_posterior_predictive(trace_.posterior)

    print(betas, trace_.posterior["beta_s"].values.mean(0))
    print(posterior["Y_t"].mean(axis=0))
    check_metrics(trace_, posterior, simulation)

    return trace_, test_model, simulation, kwargs, posterior
