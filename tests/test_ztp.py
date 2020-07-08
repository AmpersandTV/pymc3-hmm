from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess, poisson_subset_args
from pymc3.distributions.dist_math import bound, logpow, factln
from pymc3.distributions import draw_values, generate_samples
from tests.utils import (
    simulate_poiszero_hmm,
    check_metrics_for_sampling,
)
from pymc3_hmm.step_methods import FFBSStep, TransMatConjugateStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import pandas as pd
import patsy
import scipy.stats


class ZeroTruncatedPoisson(pm.Poisson):
    # adapted from https://gist.github.com/ririw/2e3a4415dc8271bd2d132c476b98b567
    sub_args = poisson_subset_args

    def __init__(self, mu, *args, **kwargs):
        super().__init__(mu, *args, **kwargs)

    def ztf_cdf(self, mu, size=None):
        mu = np.asarray(mu)
        dist = scipy.stats.poisson(mu)
        nrm = 1 - dist.cdf(0)
        sample = np.random.rand(size[0]) * nrm + dist.cdf(0)
        return dist.ppf(sample)

    def random(self, point=None, size=None):
        mu = draw_values([self.mu], point=point)[0]
        return generate_samples(self.ztf_cdf, mu, dist_shape=self.shape, size=size)

    def logp(self, value):
        mu = self.mu
        #              mu^k
        #     PDF = ------------
        #            k! (e^mu - 1)
        # log(PDF) = log(mu^k) - (log(k!) + log(e^mu - 1))
        #
        # See https://en.wikipedia.org/wiki/Zero-truncated_Poisson_distribution
        # transformed log(e^mu - 1) to more stable form log(1 - exp(-mu)) + mu and
        # implmented log1mexp(x) = log(1 - exp(-x)) according to
        # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
        #  log1mexp(a) := (
        #       log(−expm1(−a))         0 < a ≤ a0( := log 2 ≈ 0.693)
        #       log1p(− exp(−a))        a > a0

        def log1mexp(mu):
            if tt.lt(mu, np.log(2)):
                return pm.math.log(-tt.expm1(-mu))
            else:
                return tt.log1p(1 - pm.math.exp(-mu))

        p = logpow(mu, value) - (factln(value) + log1mexp(mu) + mu)
        log_prob = bound(p, mu >= 0, value >= 0)
        # Return zero when mu and value are both zero
        return tt.switch(1 * tt.eq(mu, 0) * tt.eq(value, 0), 0, log_prob)


def gen_design_matrix(N):
    t = pd.date_range(end=pd.to_datetime("today"), periods=N, freq="H").to_frame()
    t["weekday"] = t[0].dt.dayofweek
    t["hour"] = t[0].dt.hour
    t.reset_index()
    formula_str = " 1 +  C(hour) + C(weekday)"
    X_df = patsy.dmatrix(formula_str, t, return_type="dataframe")
    return X_df.values


def test_seasonality_ztp_sampling(N: int = 200, off_param=1):
    with np.errstate(over="warn", under="warn"):
        np.random.seed(2032)

        X_t = gen_design_matrix(N)
        betas_intercept = np.random.normal(np.log(3000), 1, size=1)
        betas_hour = np.sort(np.random.normal(0.5, 0.1, size=23))
        betas_week = np.sort(np.random.normal(1, 0.1, size=6))

        betas = tt.concatenate([betas_intercept, betas_hour, betas_week])
        eta_r = tt.exp(tt.dot(X_t, betas))

        cls = ZeroTruncatedPoisson

        kwargs = {
            "N": N,
            "mus": np.r_[eta_r],
            "pi_0_a": np.r_[1, 1],
            "Gamma": np.r_["0,2,1", [10, 1], [5, 5]],
            "cls": cls,
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
            beta_s_intercept = pm.Normal(
                "beta_s_intercept", np.log(3000), 1, shape=(1,)
            )
            beta_s_hour = pm.Normal("beta_s_hour", 0.5, 0.1, shape=(23,))
            beta_s_week = pm.Normal("beta_s_week", 1, 0.1, shape=(6,))

            beta_s = pm.Deterministic(
                "beta_s", tt.concatenate([beta_s_intercept, beta_s_hour, beta_s_week])
            )
            mu = tt.exp(tt.dot(X, beta_s))

            Y_rv = SwitchingProcess(
                "Y_t", [pm.Constant.dist(0), cls.dist(mu)], S_rv, observed=y_test,
            )
        with test_model:
            mu_step = pm.NUTS([mu, beta_s])
            ffbs = FFBSStep([S_rv])
            transitions = TransMatConjugateStep([p_0_rv, p_1_rv], S_rv)
            steps = [ffbs, mu_step, transitions]
            trace_ = pm.sample(N, step=steps, return_inferencedata=True, chains=1)
            posterior = pm.sample_posterior_predictive(trace_.posterior)

        check_metrics_for_sampling(trace_, posterior, simulation)
