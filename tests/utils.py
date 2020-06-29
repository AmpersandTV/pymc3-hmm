from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess

import numpy as np
import theano.tensor as tt
import pymc3 as pm
import pandas as pd
import numbers
import theano
import arviz as az

theano.config.compute_test_value = "warn"


def simulate_poiszero_hmm(
    N,
    mus=np.r_[10.0, 30.0],
    pi_0_a=np.r_[1, 1],
    Gamma=np.r_["0,2", [5, 1], [1, 3]],
    **kwargs,
):
    if isinstance(mus, numbers.Number):
        mus = np.r_[mus]
    assert pi_0_a.size == mus.size + 1 == Gamma.shape[0] == Gamma.shape[1]

    with pm.Model() as test_model:
        trans_rows = [pm.Dirichlet(f"p_{i}", r) for i, r in enumerate(Gamma)]
        P_tt = tt.stack(trans_rows)
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = pm.Dirichlet("pi_0", pi_0_a)

        S_rv = HMMStateSeq("S_t", N, P_rv, pi_0_tt)

        Y_rv = SwitchingProcess(
            "Y_t",
            [pm.Constant.dist(0)] + [pm.Poisson.dist(mu) for mu in mus],
            S_rv,
            observed=np.zeros(N),
        )

        y_test_point = pm.sample_prior_predictive(samples=1)

    return y_test_point, test_model


def time_series(N):
    t = pd.date_range(end=pd.to_datetime("today"), periods=N, freq="H")
    # month = pd.get_dummies(t.month)
    week = pd.get_dummies(t.dayofweek).values
    hour = pd.get_dummies(t.hour).values
    return np.concatenate([week, hour], 1)


def gen_defualt_params_seaonality(N):
    def rotate(l, n):
        l = list(l)
        return np.array(l[n:] + l[:n])

    week_effect = np.sort(np.random.gamma(shape=1, scale=1, size=7))
    day_effect = np.sort(np.random.gamma(shape=1, scale=1, size=24))
    day_effect = rotate(day_effect, 2)
    week_effect = rotate(week_effect, 1)

    betas = np.concatenate([week_effect, day_effect])

    seasonal = tt.dot(time_series(N), betas)

    return {
        "N": N,
        "mus": np.r_[3000.0 * seasonal, 1000.0 * seasonal],
        "pi_0_a": np.r_[1, 1, 1],
        "Gamma": np.r_["0,2,1", [10, 1, 5], [1, 10, 5], [5, 1, 20]],
        "beta_s": betas,
    }


def gen_defualt_param(N):
    return {
        "N": N,
        "mus": np.r_[5000, 7000],
        "pi_0_a": np.r_[1, 1, 1],
        "Gamma": np.r_["0,2,1", [5, 1, 1], [1, 3, 1], [1, 1, 5]],
    }


def check_metrics(trace_, posterior, simulation):

    ## checking for state prediction
    st_trace = trace_.posterior["S_t"].mean(axis=0).mean(axis=0)
    mean_error_rate = (
        1
        - np.sum(np.equal(st_trace == 0, simulation["S_t"] == 0) * 1)
        / len(simulation["S_t"])
    ).values.tolist()

    ## check for positive possion
    positive_index = simulation["Y_t"] > 0
    positive_sim = simulation["Y_t"][positive_index]
    ## point metric
    y_trace = posterior["Y_t"].mean(axis=0)
    MAPE = np.nanmean(abs(y_trace[positive_index] - positive_sim) / positive_sim)

    ## confidence_metrics_
    az_post_trace = az.from_pymc3(posterior_predictive=posterior)
    post_pred_imps_hpd_df = az.hdi(
        az_post_trace, hdi_prob=0.95, group="posterior_predictive", var_names=["Y_t"]
    ).to_dataframe()

    post_pred_imps_hpd_df = post_pred_imps_hpd_df.unstack(level="hdi")
    post_pred_imps_hpd_df.columns = post_pred_imps_hpd_df.columns.set_levels(
        ["upper", "lower"], level="hdi"
    )
    pred_range = post_pred_imps_hpd_df[positive_index]["Y_t"]
    pred_range["T_Y"] = simulation["Y_t"][positive_index]

    pred_95_CI = sum(
        (pred_range["T_Y"] < pred_range["upper"])
        & (pred_range["T_Y"] > pred_range["lower"]) * 1
    ) / len(pred_range)

    assert mean_error_rate < 0.05
    assert MAPE < 0.3
    assert pred_95_CI < 0.3
