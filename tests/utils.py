import numpy as np

import theano.tensor as tt

import pymc3 as pm

from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess

import pandas as pd


def simulate_poiszero_hmm(N, mus=np.r_[10.0, 30.0],
                          pi_0_a=np.r_[1, 1, 1],
                          Gamma=np.r_['0,2,1',
                                      [5, 1, 1],
                                      [1, 3, 1],
                                      [1, 1, 5]]
                          ):
    assert pi_0_a.size == mus.size + 1 == Gamma.shape[0] == Gamma.shape[1]

    with pm.Model() as test_model:
        trans_rows = [pm.Dirichlet(f'p_{i}', r) for i, r in enumerate(Gamma)]
        P_tt = tt.stack(trans_rows)
        P_rv = pm.Deterministic('P_tt', P_tt)

        pi_0_tt = pm.Dirichlet('pi_0', pi_0_a)

        S_rv = HMMStateSeq('S_t', N, P_rv, pi_0_tt)

        Y_rv = SwitchingProcess('Y_t',
                                [pm.Constant.dist(0)] + [pm.Poisson.dist(mu)
                                                         for mu in mus],
                                S_rv, observed=np.zeros(N))

        y_test_point = pm.sample_prior_predictive(samples=1)

    return y_test_point, test_model


def time_series(N):
    t = pd.date_range(end=pd.to_datetime('today'), periods=N, freq='H')
    # month = pd.get_dummies(t.month)
    week = pd.get_dummies(t.weekday).values
    hour = pd.get_dummies(t.hour).values
    return np.concatenate([week, hour], 1)


def gen_defualt_params_seaonality(N):
    def rotate(l, n):
        l = list(l)
        return np.array(l[n:] + l[:n])

    week_effect = np.sort(np.random.gamma(shape=1, scale=1, size=7))
    day_effect = np.sort(np.random.gamma(shape=1, scale=0.5, size=24))
    day_effect = rotate(day_effect, 2)
    week_effect = rotate(week_effect, 1)

    betas = np.concatenate([week_effect, day_effect])

    seasonal = tt.dot(time_series(N), betas)

    return {"N": N, "mus": np.r_[3000.0 * seasonal, 1000.0 * seasonal],
            "pi_0_a": np.r_[1, 1, 1],
            'Gamma': np.r_['0,2,1', [10, 1, 5], [1, 10, 5], [5, 1, 20]],
            }


def gen_defualt_param(N):
    return {"N": N,
            "mus": np.r_[5000, 7000],
            "pi_0_a": np.r_[1, 1, 1],
            'Gamma': np.r_['0,2,1', [5, 1, 1], [1, 3, 1], [1, 1, 5]],
            }
