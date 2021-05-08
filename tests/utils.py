import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3_hmm.distributions import DiscreteMarkovChain, PoissonZeroProcess


def simulate_poiszero_hmm(
    N, mu=10.0, pi_0_a=np.r_[1, 1], p_0_a=np.r_[5, 1], p_1_a=np.r_[1, 1]
):

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", p_0_a, shape=np.shape(pi_0_a))
        p_1_rv = pm.Dirichlet("p_1", p_1_a, shape=np.shape(pi_0_a))

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", tt.shape_padleft(P_tt))

        pi_0_tt = pm.Dirichlet("pi_0", pi_0_a, shape=np.shape(pi_0_a))

        S_rv = DiscreteMarkovChain("S_t", P_rv, pi_0_tt, shape=N)

        PoissonZeroProcess("Y_t", mu, S_rv, observed=np.zeros(N))

        sample_point = pm.sample_prior_predictive(samples=1)

        # Remove the extra "sampling" dimension from the sample results
        sample_point = {k: v.squeeze(0) for k, v in sample_point.items()}
        # Remove the extra dimension added due to `pm.sample_prior_predictive`
        # forcing `size=1` in its call to `test_model.Y_t.random`.
        sample_point["Y_t"] = sample_point["Y_t"].squeeze(0)

    return sample_point, test_model
