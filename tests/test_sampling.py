# %%
from pymc3_hmm.distributions import HMMStateSeq, PoissonZeroProcess
from pymc3_hmm.step_methods import FFBSStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from datetime import datetime

# %%
def simulate_poiszero_hmm(N, mu=10.0,
                          pi_0_a=np.r_[1, 1],
                          p_0_a=np.r_[5, 1],
                          p_1_a=np.r_[1, 1], **kwargs):
    with pm.Model() as test_model:
        p_0_rv = pm.Dirichlet('p_0', p_0_a)
        p_1_rv = pm.Dirichlet('p_1', p_1_a)

        P_tt = tt.stack([p_0_rv, p_1_rv])
        P_rv = pm.Deterministic('P_tt', P_tt)

        pi_0_tt = pm.Dirichlet('pi_0', pi_0_a)

        S_rv = HMMStateSeq('S_t', N, P_rv, pi_0_tt)

        Y_rv = PoissonZeroProcess('Y_t', mu, S_rv, observed=np.zeros(N))

        y_test_point = pm.sample_prior_predictive(samples=1)

    return y_test_point, test_model

# %%
class Tester(object):

    def __init__(self):
        self.simulation = None
        self.test_model_d = {}
        self.trace = None
        self.param = {}

    def generate_simulation(self, **kwargs):
        self.simulation, _ = simulate_poiszero_hmm(**kwargs)
        self.param = kwargs

    def set_up_model(self):
        with pm.Model() as test_model:
            p_0_rv = self.simulation['p_0']
            p_1_rv = self.simulation['p_1']

            P_tt = tt.stack([p_0_rv, p_1_rv])
            P_rv = pm.Deterministic('P_tt', P_tt)

            pi_0_tt = self.simulation['pi_0']
            y_test = self.simulation['Y_t']

            S_rv = HMMStateSeq('S_t', y_test.shape[0], P_rv, pi_0_tt)

            if 'off_param' in self.param:
                off_param = self.param['off_param']
            else:
                off_param = 2

            mu_rv = pm.Normal('mu', self.param['mu'] * off_param, 100.0)
            Y_rv = PoissonZeroProcess('Y_t', mu_rv, S_rv, observed=y_test)

        self.test_model_d = locals()

    def sampling(self, sample_num):
        self.set_up_model()
        test_model = self.test_model_d['test_model']
        with test_model:
            mu_step = pm.NUTS([self.test_model_d['mu_rv']])
            ffbs = FFBSStep([self.test_model_d['S_rv']])
            steps = [ffbs]
            start_time = datetime.now()
            trace = pm.sample(sample_num, step=steps, return_inferencedata=True, tune=2000)
            time_elapsed = datetime.now() - start_time
        self.test_model_d['test_model'] = test_model
        self.trace = trace
        import matplotlib.pyplot as plt
        # Plot the posterior sample chains and their marginal distributions
        pm.traceplot(trace, compact=True)
        plt.show()
        return trace, time_elapsed

    def metric(self):
        if self.trace is not None:
            trace_ = self.trace
        else:
            trace_, _ = self.sampling(self.param['N'])

        trace = trace_.posterior['S_t'].mean(axis=0).mean(axis=0)

        mean_error_rate = 1 - np.sum(np.equal(trace,
                                              self.simulation['S_t']) * 1) / len(self.simulation['S_t'])

        mu = trace_.posterior['mu'].mean()

        mu_mape = abs(mu - self.param['mu']) / self.param['mu']

        return {'S_t_me': mean_error_rate.data, 'mu_mape': mu_mape}

# %%
class TestAround(object):

    def __init__(self, test_pram_dict):
        self.tester = Tester()
        self.param_dict = test_pram_dict
        self.res_dict = {}

    def run(self):
        for i in self.param_dict:
            d = self.param_dict[i]
            self.tester.generate_simulation(**d)
            print(self.tester.param)
            trace_, time_lapsed = self.tester.sampling(d['N'])
            metric = self.tester.metric()
            self.res_dict[i] = {'trace': trace_, 'time_lapsed': time_lapsed, 'metric': metric}

# %%
def based_d(N):
    return {"N": N, "mu": 1000.0,
            "pi_0_a": np.r_[1, 1],
            "p_0_a": np.r_[4.5, 1],
            "p_1_a": np.r_[1, 4.5]}


numbers_sizes = (i * 10 ** exp for exp in range(2, 6) for i in [1, 5])
param_N = {i: based_d(i) for i in numbers_sizes}

# %%
def based_d_off_param(N):
    return {"N": 200, "mu": 1000.0,
            "pi_0_a": np.r_[1, 1],
            "p_0_a": np.r_[5, 1],
            "p_1_a": np.r_[1, 5],
            'off_param': N}


off_param_l = list(i * 10 ** exp for exp in range(0, 1) for i in range(1, 10))
param_N_off_param = {i: based_d_off_param(i) for i in off_param_l}

tester = TestAround(param_N_off_param)
tester.run()

# %%
res_d = tester.res_dict
metrics_state = [res_d[i]['metric']['S_t_me'] for i in res_d]
metrics_mu = [res_d[i]['metric']['mu_mape'] for i in res_d]
# %%
import matplotlib.pyplot as plt

poiszero_sim, _ = simulate_poiszero_hmm(**based_d(200))
y_test = poiszero_sim['Y_t']
sim_obs_ratio = 1- len(y_test[y_test==0])/len(y_test)
# %%
import pandas as pd

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

# %%
pp = PredictorTester()
pp.get_sample_dict(obs_ratio=sim_obs_ratio)
# %%

def based_d(N):
    return {"N": N, "mu": 1000.0,
            "pi_0_a": np.r_[1, 1],
            "p_0_a": np.r_[4.5, 1],
            "p_1_a": np.r_[1, 4.5]}

poiszero_sim, _ = simulate_poiszero_hmm(**based_d(200))

y_test = poiszero_sim['Y_t']

combos = list(pp.sample_dict.keys())
num_of_plot = len(combos)

fig, ax = plt.subplots(figsize=(15, 6.0), nrows=num_of_plot + 1)

for i in range(num_of_plot):
    example = pp.sample_dict[combos[i]]
    ax[i].plot(example.impressions[-200:],
               label=combos[i], color='black',
               drawstyle='steps-pre', linewidth=0.5)

ax[-1].plot(y_test,
           label=r'$y_t$', color='black',
           drawstyle='steps-pre', linewidth=0.5)

for ax_ in ax:
    ax_.legend()

plt.tight_layout()

plt.show()


# %%

combos = list(pp.sample_dict.keys())
num_of_plot = len(combos)

fig, ax = plt.subplots(figsize=(15, 6.0), ncols=num_of_plot )

for i in range(num_of_plot):
    y_test = pp.sample_dict[combos[i]]
    positive = y_test.impressions
    positive = positive[positive > 0]
    mean_ = np.mean(positive)
    sd = np.std(positive)

    _95_ci = mean_ - 2* sd, mean_ + 2*sd
    _95_range = positive[ (positive > _95_ci[0]) & (positive < _95_ci[1])]
    print(_95_range)
    ax[i].hist(_95_range,bins = 1000,
               label=combos[i], color='black')



for ax_ in ax:
    ax_.legend()

plt.tight_layout()

plt.show()


