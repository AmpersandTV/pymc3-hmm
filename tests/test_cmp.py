from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess, broadcast_to
from pymc3.distributions.dist_math import bound
from pymc3.distributions import draw_values, generate_samples
from tests.utils import (
    simulate_hmm_dist,
    check_metrics_for_sampling,
)
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd
import patsy


def CMPoisson_subset_args(self, shape, idx):
    return [
        (broadcast_to(self.nu, shape))[idx],
        (broadcast_to(self.lamda, shape))[idx],
    ]


class CMPoisson(pm.Discrete):
    ## Adapted from https://gist.github.com/dadaromeo/33e581d9e3bcbad83531b4a91a87509f
    subset_args = CMPoisson_subset_args

    def __init__(self, lamda, nu, *args, **kwargs):
        super(CMPoisson, self).__init__(*args, **kwargs)
        self.lamda = lamda
        self.nu = nu
        self.alpha = tt.power(self.lamda, 1 / self.nu)

    def logp(self, value):
        lamda = self.lamda
        nu = self.nu
        alpha = self.alpha
        pi = tt.constant(np.pi)

        log_Z = nu * alpha - ((nu - 1) / 2) * tt.log(2 * pi * alpha) - 0.5 * tt.log(nu)
        return bound(
            value * tt.log(lamda) - nu * tt.gammaln(value + 1) - log_Z,
            lamda > 0,
            nu > 0,
        )

    def _random(self, lamda, nu, size=None):
        size = size or 1

        nu = np.atleast_1d(nu)
        alpha = np.atleast_1d(np.power(lamda, 1 / nu))
        Z = np.exp(nu * alpha) / ((2 * np.pi * alpha) ** ((nu - 1) / 2) * np.sqrt(nu))

        U = np.random.uniform(low=0, high=1, size=size)
        values = np.empty(size, dtype=int)

        for i in range(U.shape[0]):
            p = 1 / Z
            cdf = p
            k = 0
            u = U[i]

            while any(u > cdf):
                k += 1
                p = (p * lamda) / k ** nu
                cdf += p

            values[i] = k
        return values

    def random(self, point=None, size=None, repeat=None):
        lamda, nu = draw_values([self.lamda, self.nu], point=point)
        return generate_samples(
            self._random, lamda, nu, dist_shape=self.shape, size=size
        )


def gen_design_matrix(N):
    t = pd.date_range(end=pd.to_datetime("today"), periods=N, freq="H").to_frame()
    t["weekday"] = t[0].dt.dayofweek
    t["hour"] = t[0].dt.hour
    t.reset_index()
    formula_str = " 1 +  C(hour) + C(weekday)"
    X_df = patsy.dmatrix(formula_str, t, return_type="dataframe")
    return X_df.values


def test_seasonality_cmp_sampling(N: int = 200, off_param=1):
    with np.errstate(over="warn", under="warn"):
        np.random.seed(2032)

        X_t = gen_design_matrix(N)
        betas_intercept = np.random.normal(np.log(3000), 1, size=1)
        betas_hour = np.sort(np.random.normal(0.5, 0.1, size=23))
        betas_week = np.sort(np.random.normal(1, 0.1, size=6))

        betas = np.concatenate([betas_intercept, betas_hour, betas_week])
        eta_r = tt.exp(tt.dot(X_t, betas))

        eta_d = {"lamda": eta_r, "nu": 1}

        cls = CMPoisson

        kwargs = {
            "N": N,
            "arg_dict": [eta_d],
            "pi_0_a": np.r_[1, 1],
            "Gamma": np.r_["0,2,1", [10, 1], [5, 5]],
            "cls": cls,
        }
        simulation, _ = simulate_hmm_dist(**kwargs)

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
            beta_s_intercept = pm.Normal(
                "beta_s_intercept", np.log(3000), 1, shape=(1,)
            )
            beta_s_hour = pm.Normal("beta_s_hour", 0.5, 0.1, shape=(23,))
            beta_s_week = pm.Normal("beta_s_week", 1, 0.1, shape=(6,))

            beta_s = pm.Deterministic(
                "beta_s", tt.concatenate([beta_s_intercept, beta_s_hour, beta_s_week])
            )
            lamda = tt.exp(tt.dot(X, beta_s))

            nu = pm.Normal("nu", 1, 0.5, shape=(1,))

            Y_rv = SwitchingProcess(
                "Y_t",
                [pm.Constant.dist(0), cls.dist(lamda, nu)],
                S_rv,
                observed=y_test,
            )
        with test_model:
            mu_step = pm.NUTS([lamda, nu, beta_s])
            ffbs = FFBSStep([S_rv])
            transitions = TransMatConjugateStep([p_0_rv, p_1_rv], S_rv)
            steps = [ffbs, mu_step, transitions]
            trace_ = pm.sample(N, step=steps, return_inferencedata=True, chains=1)

        check_metrics_for_sampling(trace_, simulation)
        betas_np = np.concatenate([betas_intercept, betas_hour, betas_week])
        beta_pred = trace_.posterior["beta_s"].values[0].mean(0)
        beta_mape = abs(beta_pred - betas_np) / betas_np
        assert beta_mape.mean() < 0.2
