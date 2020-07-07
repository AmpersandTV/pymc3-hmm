import numpy as np
import theano.tensor as tt
import pymc3 as pm
import numbers
import theano
import arviz as az

from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess

theano.config.compute_test_value = "warn"


def simulate_poiszero_hmm(
    N, mus=np.r_[10.0, 30.0], pi_0_a=np.r_[1, 1], Gamma=np.r_["0,2", [5, 1], [1, 3]]
):
    if isinstance(mus, numbers.Number):
        mus = [mus]
    assert pi_0_a.size == len(mus) + 1 == Gamma.shape[0] == Gamma.shape[1]

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


def check_metrics_for_sampling(trace_, posterior, simulation):
    """
    this function is to check the how the posterior generated matches matched with the simulated toy series

    Parameters
    ----------
    trace_ : trace object returned from pm.sampling,
    posterior : posterior prediction
    simulation : simulated toy series generated from `simulate_poiszero_hmm` funcion

    Returns None
    -------

    """

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
    mape = np.mean(abs(y_trace[positive_index] - positive_sim) / positive_sim)

    ## confidence_metrics_
    az_post_trace = az.from_pymc3(posterior_predictive=posterior)
    ci_conf = 0.95
    post_pred_imps_hpd_df = az.hdi(
        az_post_trace, hdi_prob=ci_conf, group="posterior_predictive", var_names=["Y_t"]
    ).to_dataframe()

    post_pred_imps_hpd_df = post_pred_imps_hpd_df.unstack(level="hdi")
    post_pred_imps_hpd_df.columns = post_pred_imps_hpd_df.columns.set_levels(
        ["upper", "lower"], level="hdi"
    )
    pred_range = post_pred_imps_hpd_df[positive_index]["Y_t"]
    pred_range["T_Y"] = simulation["Y_t"][positive_index]

    pred_ci = sum(
        (pred_range["T_Y"] <= pred_range["upper"])
        & (pred_range["T_Y"] >= pred_range["lower"]) * 1
    ) / len(pred_range)

    assert mean_error_rate <= 0.05
    assert mape <= 0.05
    assert pred_ci >= ci_conf - 0.05
