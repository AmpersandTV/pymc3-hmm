from pymc3_hmm.distributions import HMMStateSeq
from pymc3_hmm.utils import compute_steady_state
import theano.tensor as tt
import pymc3 as pm
import numpy as np


def auto_set_hmm_seq(N_states, model, states):
    """
    Initiate a HMMStateSeq based on the length of the mixture component.

    This function require pymc3 and HMMStateSeq.

    Parameters
    ----------
    N_states : int
        Number of states in the mixture
    model : pymc3.model.Model
        Model object that we trained on
    states : ndarray
        Vector sequence of states to set the `test_value` for `HMMStateSeq`

    Returns
    -------
    locals(), a dict of local variables for reference in sampling steps.
    """
    with model:
        pp = [pm.Dirichlet(f"p_{i}", np.ones(N_states)) for i in range(N_states)]
        P_tt = tt.stack(pp)
        P_rv = pm.Deterministic("Gamma", tt.shape_padleft(P_tt))
        pi_0_tt = compute_steady_state(P_rv)

        S_rv = HMMStateSeq("V_t", P_rv, pi_0_tt, shape=states.shape[0])
        S_rv.tag.test_value = states

        return locals()
