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
    # z_at = at.stack([at.dot(X, xis[..., s, :]) for s in range(S)], axis=1)
    z_at = at.tensordot(X, xis, axes=((1,), (0,)))
    z_at.name = "z"

    Gammas_at = pm.Deterministic("Gamma", multilogit_inv(z_at))
    gamma_0_rv = pm.Dirichlet("gamma_0", np.ones((S,)))

    if type(observed) == np.ndarray:
        V_initval = (observed > 0) * 1
    else:
        V_initval = (observed.get_value() > 0) * 1

    V_rv = DiscreteMarkovChain("V_t", Gammas_at, gamma_0_rv, initval=V_initval)

    Y_rv = SwitchingProcess(
        "Y_t",
        [pm.Constant.dist(0), pm.Constant.dist(mu)],
        V_rv,
        observed=observed,
    )
    return Y_rv


def test_only_positive_state():
    rng = np.random.RandomState(4284)

    number_of_draws = 50
    S = 2
    mu = 10
    y_t = np.repeat(0, 100)

    with pm.Model(rng_seeder=rng):
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

        P_at = at.stack([p_0_rv, p_1_rv])
        Gammas_at = pm.Deterministic(
            "P", at.broadcast_to(P_at, (y_t.shape[0],) + tuple(P_at.shape))
        )

        gamma_0_rv = pm.Dirichlet("gamma_0", np.ones((S,)))

        V_rv = DiscreteMarkovChain("V_t", Gammas_at, gamma_0_rv, initval=(y_t > 0) * 1)

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
            step=FFBSStep([V_rv], rng=rng),
        )

        posterior_pred_trace = pm.sample_posterior_predictive(
            posterior_trace.posterior, var_names=["Y_t"]
        )
        assert np.all(posterior_pred_trace["Y_t"] == 0)


def test_time_varying_model():
    rng = np.random.RandomState(1039)

    data = gen_toy_data()

    formula_str = "1 + C(weekday)"
    X_df = patsy.dmatrix(formula_str, data, return_type="dataframe")
    X_np = X_df.values

    xi_shape = X_np.shape[1]

    xi_0_true = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]).reshape(xi_shape, 1)
    xi_1_true = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]).reshape(xi_shape, 1)

    xis_rv_true = np.stack([xi_0_true, xi_1_true], axis=1)

    with pm.Model(rng_seeder=rng) as sim_model:
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
        xis_rv = pm.Normal("xis", 0, 10, size=xis_rv_true.shape)
        _ = create_dirac_zero_hmm(X, 1000, xis_rv, Y)

    number_of_draws = 500

    with model:
        steps = [
            FFBSStep([model.V_t], rng=rng),
            pm.NUTS(
                vars=[
                    model.gamma_0,
                    # TODO FIXME: Using `model.Gamma` here fails.  This looks
                    # like a v4 bug.  It should provide a better error than
                    # the one it gave, at the very least.
                    model.xis,
                ],
                target_accept=0.90,
            ),
        ]

    with model:
        posterior_trace = pm.sample(
            draws=number_of_draws,
            step=steps,
            random_seed=1030,
            return_inferencedata=True,
            chains=1,
            cores=1,
            progressbar=True,
            idata_kwargs={"dims": {"Y_t": ["date"], "V_t": ["date"]}},
        )

    hdi_data = az.hdi(posterior_trace, hdi_prob=0.95, var_names=["xis"]).to_dataframe()
    hdi_data = hdi_data.unstack(level="hdi")

    xis_true_flat = xis_rv_true.squeeze().flatten()
    true_xis_under = xis_true_flat <= hdi_data["xis", "higher"].values
    true_xis_above = xis_true_flat >= hdi_data["xis", "lower"].values

    assert np.sum(~(true_xis_under ^ true_xis_above)) > int(
        len(true_xis_under) * 2 / 3.0
    )

    trace = posterior_trace.posterior.drop_vars(["Gamma", "V_t"])

    # Update the shared variable values for out-of-sample predictions
    Y.set_value(np.ones(test_X.shape[0], dtype=Y.dtype))
    X.set_value(test_X)

    with aesara.config.change_flags(compute_test_value="off"):
        adds_pois_ppc = pm.sample_posterior_predictive(
            trace, var_names=["V_t", "Y_t", "Gamma"], model=model
        )

    assert (np.abs(adds_pois_ppc["V_t"] - test_V) / test_V.shape[0]).mean() < 1e-2
