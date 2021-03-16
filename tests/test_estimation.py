import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3_hmm.distributions import DiscreteMarkovChain, SwitchingProcess
from pymc3_hmm.step_methods import FFBSStep


def test_only_positive_state():
    number_of_draws = 50
    S = 2
    mu = 10
    y_t = np.repeat(0, 100)

    with pm.Model():
        p_0_rv = pm.Dirichlet("p_0", np.r_[1, 1])
        p_1_rv = pm.Dirichlet("p_1", np.r_[1, 1])

        P_tt = tt.stack([p_0_rv, p_1_rv])
        Gammas_tt = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))

        gamma_0_rv = pm.Dirichlet("gamma_0", np.ones((S,)))

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
