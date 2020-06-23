import numpy as np

import theano.tensor as tt

import pymc3 as pm

from pymc3_hmm.distributions import PoissonZeroProcess, HMMStateSeq, SwitchingProcess

import pandas as pd

import matplotlib.pyplot as plt


def simulate_poiszero_hmm(
    N, mu=10.0, pi_0_a=np.r_[1, 1], p_0_a=np.r_[5, 1], p_1_a=np.r_[1, 1]
):

    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet("p_0", p_0_a)
        p_1_rv = pm.Dirichlet("p_1", p_1_a)

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic("P_tt", P_tt)

        pi_0_tt = pm.Dirichlet("pi_0", pi_0_a)

        S_rv = HMMStateSeq("S_t", N, P_rv, pi_0_tt)

        Y_rv = PoissonZeroProcess("Y_t", mu, S_rv, observed=np.zeros(N))

        y_test_point = pm.sample_prior_predictive(samples=1)

    return y_test_point, test_model


def time_series(N):
    t = pd.date_range(end=pd.to_datetime('today'), periods=N, freq='H')
    # month = pd.get_dummies(t.month)
    week = pd.get_dummies(t.weekday).values
    hour = pd.get_dummies(t.hour).values
    return np.concatenate([week, hour], 1)


def simulate_poiszero_hmm_seasonal(N, mus=np.r_[10.0, 30.0],
                                   pi_0_a=np.r_[1, 1, 1],
                                   Gamma=np.r_['0,2,1', [5, 1, 1], [1, 3, 3], [1, 5, 10]],
                                   betas=np.random.normal(800, 800, 31)
                                   ):
    assert pi_0_a.size == mus.size + 1 == Gamma.shape[0] == Gamma.shape[1]

    with pm.Model() as test_model:
        trans_rows = [pm.Dirichlet(f'p_{i}', r) for i, r in enumerate(Gamma)]
        P_tt = tt.stack(trans_rows)
        P_rv = pm.Deterministic('P_tt', P_tt)

        pi_0_tt = pm.Dirichlet('pi_0', pi_0_a)

        S_rv = HMMStateSeq('S_t', N, P_rv, pi_0_tt)

        seasonality_x = time_series(N)
        seasonal = tt.dot(seasonality_x, betas)
        Y_rv = SwitchingProcess('Y_t',
                                [pm.Constant.dist(0)]
                                + [pm.Poisson.dist(mu * seasonal) for mu in mus],
                                S_rv, observed=np.zeros(N))

        y_test_point = pm.sample_prior_predictive(samples=1)

    return y_test_point, test_model



with open('./amp_uri.txt', 'r') as f:
    conn_str = f.readlines()[0].strip()


class PredictorTester(object):

    def __init__(self):
        self.sample_dict, self.cur_d, self.sbp_d = {}, {}, {}
        self.sample_combo = []
        self._calc_ratio()

    def _calc_ratio(self):
        self.impression_ratio_df = pd.read_sql(
            '''select dma_code, network_id, demo_id,market, network, demo_gender, demo_age_range,
               1-(sum(cast(impressions = 0 as int)* 1.0))/count(*) as obs_ratio,
                min(dt) as min_date
                from nielsen.impressions_pop_dmn_temp
                where population_size is not null
                group by 1, 2, 3, 4, 5, 6, 7''',
            con=conn_str,
            index_col=['dma_code', 'network_id', 'demo_id'])

    def get_sample_dict(self, obs_ratio: float):
        ## read sample data in dictionary by non-zeros observation ratio
        sample_idx = self.impression_ratio_df.query(f'obs_ratio == {obs_ratio}').sample(5, random_state=123)
        self.sample_combo = sample_idx.index.values
        qry_str = {i: r'''
               select dt as t, impressions, population_size
               from nielsen.impressions_pop_dmn_temp
               where population_size is not null
                 and dma_code='{}'
                 and network_id='{}'
                 and demo_id='{}'
               order by dt
               '''.format(*i) for i in self.sample_combo}

        def read_with_drop_dup(str):
            df = pd.read_sql(str, con=conn_str)
            df = df.drop_duplicates(subset='t', keep='first').set_index('t')
            return df

        self.sample_dict = {i: read_with_drop_dup(qry_str[i]) for i in qry_str}


def plot_real_vs_simulation(pp: PredictorTester, y_test:np.array):
    combos = list(pp.sample_dict.keys())
    num_of_plot = len(combos)
    N = len(y_test)

    fig, ax = plt.subplots(figsize=(15, 6.0), nrows=num_of_plot + 1)

    for i in range(num_of_plot):
        example = pp.sample_dict[combos[i]]
        ax[i].plot(example.impressions[-N:],
                   label=combos[i], color='black',
                   drawstyle='steps-pre', linewidth=0.5)

    ax[-1].plot(y_test,
                label=r'$y_t$', color='black',
                drawstyle='steps-pre', linewidth=0.5)

    for ax_ in ax:
        ax_.legend()

    plt.tight_layout()

    plt.show()
