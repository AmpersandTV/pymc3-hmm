from datetime import date, timedelta

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import pandas as pd
import patsy
import pymc3 as pm
from aesara import shared

from pymc3_hmm.distributions import DiscreteMarkovChain, SwitchingProcess
from pymc3_hmm.step_methods import FFBSStep
from pymc3_hmm.utils import multilogit_inv


def gen_toy_data(days=-7 * 10):
    dates = date.today() + timedelta(days=days), date.today() + timedelta(days=2)
    date_rng = pd.date_range(start=min(dates), end=max(dates), freq="H")
    raw_data = {"date": date_rng}
    data = pd.DataFrame.from_dict(raw_data)
    data = data.assign(
        month=pd.Categorical(data.date.dt.month, categories=list(range(1, 13))),
        weekday=pd.Categorical(data.date.dt.weekday, categories=list(range(7))),
        hour=pd.Categorical(data.date.dt.hour, categories=list(range(24))),
    )
    return data


def create_dirac_zero_hmm(X, mu, xis, observed):
    S = 2
    z_tt = at.stack([at.dot(X, xis[..., s, :]) for s in range(S)], axis=1)
    Gammas_tt = pm.Deterministic("Gamma", multilogit_inv(z_tt))
    gamma_0_rv = pm.Dirichlet("gamma_0", np.ones((S,)), shape=S)

    if type(observed) == np.ndarray:
        T = X.shape[0]
    else:
        T = X.get_value().shape[0]

    V_rv = DiscreteMarkovChain("V_t", Gammas_tt, gamma_0_rv, shape=T)
    if type(observed) == np.ndarray:
        V_rv.tag.test_value = (observed > 0) * 1
    else:
        V_rv.tag.test_value = (observed.get_value() > 0) * 1
    Y_rv = SwitchingProcess(
        "Y_t",
        [pm.Constant.dist(0), pm.Constant.dist(mu)],
        V_rv,
        observed=observed,
    )
    return Y_rv


def test_only_positive_state():
    number_of_draws = 50
    S = 2
    mu = 10
    y_t = np.repeat(0, 100)

    with pm.Model():
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1], shape=2)
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1], shape=2)

        P_tt = at.stack([p_0_rv, p_1_rv])
        Gammas_tt = pm.Deterministic("P_tt", at.shape_padleft(P_tt))

        gamma_0_rv = pm.Dirichlet("gamma_0", np.ones((S,)), shape=S)

        V_rv = DiscreteMarkovChain("V_t", Gammas_tt, gamma_0_rv, shape=y_t.shape[0])
        V_rv.tag.test_value = (y_t > 0) * 1

        _ = SwitchingProcess(
            "Y_t",
            [pm.Constant.dist(0), pm.Constant.dist(mu)],
            V_rv,
            observed=y_t,
        )

        posterior_trace = pm.sample(
            chains=1,
            draws=number_of_draws,
            return_inferencedata=True,
            step=FFBSStep([V_rv]),
        )

        posterior_pred_trace = pm.sample_posterior_predictive(
            posterior_trace.posterior, var_names=["Y_t"]
        )
        assert np.all(posterior_pred_trace["Y_t"] == 0)


def test_time_varying_model():

    np.random.seed(1039)

    data = gen_toy_data()

    formula_str = "1 + C(weekday)"
    X_df = patsy.dmatrix(formula_str, data, return_type="dataframe")
    X_np = X_df.values

    xi_shape = X_np.shape[1]

    xi_0_true = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]).reshape(xi_shape, 1)
    xi_1_true = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]).reshape(xi_shape, 1)

    xis_rv_true = np.stack([xi_0_true, xi_1_true], axis=1)

    with pm.Model() as sim_model:
        _ = create_dirac_zero_hmm(
            X_np, mu=1000, xis=xis_rv_true, observed=np.zeros(X_np.shape[0])
        )

    sim_point = pm.sample_prior_predictive(samples=1, model=sim_model)

    y_t = sim_point["Y_t"].squeeze().astype(int)

    split = int(len(y_t) * 0.7)

    train_y, test_V = y_t[:split], sim_point["V_t"].squeeze()[split:]
    train_X, test_X = X_np[:split, :], X_np[split:, :]

    X = shared(train_X, name="X", borrow=True)
    Y = shared(train_y, name="y_t", borrow=True)

    with pm.Model() as model:
        xis_rv = pm.Normal("xis", 0, 10, shape=xis_rv_true.shape)
        _ = create_dirac_zero_hmm(X, 1000, xis_rv, Y)

    number_of_draws = 500

    with model:
        steps = [
            FFBSStep([model.V_t]),
            pm.NUTS(
                vars=[
                    model.gamma_0,
                    model.Gamma,
                ],
                target_accept=0.90,
            ),
        ]

    with model:
        posterior_trace = pm.sample(
            draws=number_of_draws,
            step=steps,
            random_seed=100,
            return_inferencedata=True,
            chains=1,
            cores=1,
            progressbar=True,
            idata_kwargs={"dims": {"Y_t": ["date"], "V_t": ["date"]}},
        )

    # Update the shared variable values
    Y.set_value(np.ones(test_X.shape[0], dtype=Y.dtype))
    X.set_value(test_X)

    model.V_t.distribution.shape = (test_X.shape[0],)

    hdi_data = az.hdi(posterior_trace, hdi_prob=0.95, var_names=["xis"]).to_dataframe()
    hdi_data = hdi_data.unstack(level="hdi")

    xis_true_flat = xis_rv_true.squeeze().flatten()
    check_idx = ~np.in1d(
        np.arange(len(xis_true_flat)), np.arange(3, len(xis_true_flat), step=4)
    )
    assert np.all(
        xis_true_flat[check_idx] <= hdi_data["xis", "higher"].values[check_idx]
    )
    assert np.all(
        xis_true_flat[check_idx] >= hdi_data["xis", "lower"].values[check_idx]
    )

    trace = posterior_trace.posterior.drop_vars(["Gamma", "V_t"])

    with aesara.config.change_flags(compute_test_value="off"):
        adds_pois_ppc = pm.sample_posterior_predictive(
            trace, var_names=["V_t", "Y_t", "Gamma"], model=model
        )

    assert (np.abs(adds_pois_ppc["V_t"] - test_V) / test_V.shape[0]).mean() < 1e-2
