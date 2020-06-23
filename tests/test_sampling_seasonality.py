# %%
from pymc3_hmm.distributions import HMMStateSeq, SwitchingProcess
from tests.test_sampling import Tester
from tests.utils import simulate_poiszero_hmm_seasonal, PredictorTester, plot_real_vs_simulation, time_series
from pymc3_hmm.step_methods import FFBSStep
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


# %%
def rotate(l, n):
    l = list(l)
    return np.array(l[n:] + l[:n])


week_effect = np.sort(np.random.gamma(shape=1, scale=1, size=7))
day_effect = np.sort(np.random.gamma(shape=1, scale=0.5, size=24))
day_effect = rotate(day_effect, 2)


def based_d(N):
    return {"N": N, "mus": np.r_[3000.0, 1000.0],
            "pi_0_a": np.r_[1, 1, 1],
            'Gamma': np.r_['0,2,1', [10, 1, 5], [1, 10, 5], [5, 1, 20]],
            "betas": np.concatenate([week_effect, day_effect])}


# %%
N = 200
poiszero_sim, _ = simulate_poiszero_hmm_seasonal(**based_d(N))

states = poiszero_sim['S_t']
sim_obs_ratio = sum((states == 0) * 1) / len(states)

plt.plot(poiszero_sim['Y_t'],
         label=r'$y_t$', color='black',
         drawstyle='steps-pre', linewidth=0.5)

plt.show()

# %%
pp = PredictorTester()
pp.get_sample_dict(obs_ratio=sim_obs_ratio)

# %%
plot_real_vs_simulation(pp,poiszero_sim['Y_t'])

# %%
class TesterSeasonality(Tester):
    def __init__(self):
        super().__init__()

    def generate_simulation(self, **kwargs):
        print(kwargs)
        self.simulation, _ = simulate_poiszero_hmm_seasonal(**kwargs)
        self.param = kwargs

    def set_up_model(self):
        with pm.Model() as test_model:
            p_0_rv = pm.Dirichlet('p_0', np.r_[1, 1, 1])
            p_1_rv = pm.Dirichlet('p_1', np.r_[1, 1, 1])
            p_2_rv = pm.Dirichlet('p_2', np.r_[1, 1, 1])

            P_tt = tt.stack([p_0_rv, p_1_rv, p_2_rv])
            P_rv = pm.Deterministic('P_tt', P_tt)

            pi_0_tt = self.simulation['pi_0']
            y_test = self.simulation['Y_t']

            S_rv = HMMStateSeq('S_t', y_test.shape[0], P_rv, pi_0_tt)
            S_rv.tag.test_value = (y_test > 0).astype(np.int)

            E_1_mu, Var_1_mu = 3000.0, 3000.0
            # mu_1_rv = pm.Gamma('mu_1', E_1_mu ** 2 / Var_1_mu, E_1_mu / Var_1_mu)

            E_2_mu, Var_2_mu = 1000.0, 1000.0
            # mu_2_rv = pm.Gamma('mu_2', E_2_mu ** 2 / Var_2_mu, E_2_mu / Var_2_mu)

            s = time_series(self.param['N'])
            beta_s = pm.Gamma('beta_s', 1, 1, shape=(s.shape[1],))
            seasonal = tt.dot(s, beta_s)

            Y_rv = SwitchingProcess('Y_t',
                                    [pm.Constant.dist(0),
                                     pm.Poisson.dist(E_1_mu * seasonal),
                                     pm.Poisson.dist((E_1_mu + E_2_mu) * seasonal)],
                                    S_rv, observed=y_test)

        self.test_model_d = locals()

    def sampling(self, sample_num):
        self.set_up_model()
        test_model = self.test_model_d['test_model']
        with test_model:
            # mu_step_1 = pm.NUTS([self.test_model_d['mu_1_rv']])
            # mu_step_2 = pm.NUTS([self.test_model_d['mu_2_rv']])
            ffbs = FFBSStep([self.test_model_d['S_rv']])
            steps = [ffbs]
            start_time = datetime.now()
            trace = pm.sample(sample_num, step=steps, return_inferencedata=True, tune=1000)
            time_elapsed = datetime.now() - start_time
        self.test_model_d['test_model'] = test_model
        self.trace = trace
        import matplotlib.pyplot as plt
        # Plot the posterior sample chains and their marginal distributions
        pm.traceplot(trace, compact=True)
        plt.show()
        return trace, time_elapsed


# %%
tester = TesterSeasonality()
tester.generate_simulation(**based_d(150))
# %%
tester.set_up_model()
trace, time_elapsed = tester.sampling(150)

# %%

with tester.test_model_d['test_model']:
    post = pm.sample_posterior_predictive(trace.posterior)

# %%

combos = list(pp.sample_dict.keys())
num_of_plot = len(combos)

fig, ax = plt.subplots(figsize=(15, 6.0), nrows=num_of_plot + 1)

for i in range(num_of_plot):
    example = pp.sample_dict[combos[i]]
    ax[i].plot(example.impressions[-N:],
               label=combos[i], color='black',
               drawstyle='steps-pre', linewidth=0.5)

ax[-1].plot(post['Y_t'].mean(0),
            label=r'$y_t$', color='black',
            drawstyle='steps-pre', linewidth=0.5)

for ax_ in ax:
    ax_.legend()

plt.tight_layout()

plt.show()
