import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3_hmm.distributions import DiscreteMarkovChain, PoissonZeroProcess
from pymc3_hmm.step_methods import FFBSStep


def simulate_poiszero_hmm(
    N,
    observed,
    mu=10.0,
    pi_0_a=np.r_[1, 1],
    p_0_a=np.r_[5, 1],
    p_1_a=np.r_[1, 1],
    pi_0=None,
):
    p_0_rv = pm.Dirichlet("p_0", p_0_a)
    p_1_rv = pm.Dirichlet("p_1", p_1_a)
    P_tt = tt.stack([p_0_rv, p_1_rv])
    P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))
    if pi_0 is not None:
        pi_0_tt = tt.as_tensor(pi_0)
    else:
        pi_0_tt = pm.Dirichlet("pi_0", pi_0_a)
    S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=N)
    S_rv.tag.test_value = (observed > 0) * 1
    return PoissonZeroProcess("Y_t", mu, S_rv, observed=observed)


def test_gamma_0_estimation():

    np.random.seed(2032)
    true_initial_states = np.array([0.5, 0.5])

    with pm.Model(theano_config={"compute_test_value": "ignore"}) as sim_model:
        _ = simulate_poiszero_hmm(30, np.zeros(30), 150, pi_0=true_initial_states)

    sim_point = pm.sample_prior_predictive(samples=1, model=sim_model)
    sim_point["Y_t"] = sim_point["Y_t"].squeeze()
    y_test = sim_point["Y_t"]

    with pm.Model() as test_model:
        _ = simulate_poiszero_hmm(
            30,
            y_test,
            150,
        )
        states_step = FFBSStep([test_model["S_t"]])

        posterior_trace = pm.sample(
            step=[states_step],
            draws=5,
            return_inferencedata=True,
            chains=1,
            progressbar=True,
        )

    estimated_initial_state_probs = posterior_trace.posterior.pi_0.values[0].mean(0)
    np.testing.assert_almost_equal(
        estimated_initial_state_probs, true_initial_states, 1
    )
