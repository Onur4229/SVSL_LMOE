import numpy as np

from contextual_environments.mujoco.table_tennis_multi import get_env, ACTION_DIM, CTXT_DIM, CONTEXT_RANGE_BOUNDS
from SVSL import SVSL
from util.Logger import LoggerLogFlags

########################################################################################################################
# seed
########################################################################################################################

c_seed = 0
np.random.seed(c_seed)

########################################################################################################################
# create environment
########################################################################################################################

n_cores = 40
action_dim = ACTION_DIM
ctxt_dim = CTXT_DIM
context_range_bounds = CONTEXT_RANGE_BOUNDS

env = get_env(name='tt', n_cores=n_cores, env_bounds=context_range_bounds)
########################################################################################################################
# configurations
########################################################################################################################

config = SVSL.get_default_config()
config.verbose = True
config.exp_name = 'tabletennis'
config.action_dim = action_dim
config.ctxt_dim = ctxt_dim
config.train_epochs_p_comp = 1000  # 4D: 1000, 2D: 500
config.train_epochs_weights = 100
config.num_init_components = 1
config.rm_component_thresh = 1e-05
config.n_samples_buffer = int(1.5 * (1 + (ctxt_dim + action_dim) + int((ctxt_dim + action_dim) *
                                                                       ((ctxt_dim + action_dim) + 1) / 2)))

n_new_samples_ratio = 0.4
config.n_new_samples_comps = int(n_new_samples_ratio * config.n_samples_buffer)
config.n_new_samples_ctxt = int(
    n_new_samples_ratio * int(1.5 * (1 + config.ctxt_dim + config.ctxt_dim * (config.ctxt_dim + 1) / 2)))
config.n_cmp_addings = 50
config.fine_tune_every_it = 50
config.save_model_every_it = 50
config.context_range_bounds = context_range_bounds
# config.test_every_it = 15000000
config.test_every_it = 1000
config.del_loc_opt = False

########################################################################################################################
# Hyperparameters
########################################################################################################################
config.alpha = 1e-05
config.beta = 0.2
config.beta_weight = 0.2
config.context_kl_bound = 0.01  # 4D: 0.01, 2D: 0.05
config.weight_kl_bound = 0.5
config.component_kl_bound = 0.01  # 4D: 0.01. 2D: 0.2

# needed for ablation, here for code completeness ....
config.zero_resp_fac = False
if config.zero_resp_fac:
    config.alpha_resp = 0
    config.beta_resp = 0
else:
    config.alpha_resp = config.alpha
    config.beta_resp = config.beta

########################################################################################################################
# Initializations
########################################################################################################################
config.ctxt_init_mean_std = np.array([0.625, 0.75, 0.58, 0.75])  # 4D
# config.ctxt_init_mean_std = np.array([0.5, 0.6])  # 2D
config.ctxt_init_mean = np.mean(np.array(context_range_bounds), axis=0)
ctxt_init_cov_prior = [0.32, 0.375, 0.29, 0.375]  # 4D
# ctxt_init_cov_prior = [0.25, 0.3]  # 2D
config.ctxt_init_cov = np.eye(config.ctxt_dim) * np.array(ctxt_init_cov_prior)

config.cmp_init_cov_prior = 1
config.cmp_init_cov = np.eye(config.action_dim) * config.cmp_init_cov_prior
config.cmp_init_mean_params = np.zeros((config.ctxt_dim + 1, config.action_dim))
########################################################################################################################
# Logging, Plotting & Saving
########################################################################################################################
config.png_plot_saving = False
config.vis_plots = False
config.png_plot_save_every_it = 0
config.snapshot_data_saving_per_it = False
config.save_data_ever_it = 1000000

config.log = True

config.log_options = [LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e,
                      LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e,
                      LoggerLogFlags.CO_ENTROPIES_c_e,
                      LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e,
                      LoggerLogFlags.CC_ENTROPIES_e_c,
                      LoggerLogFlags.MG_WEIGHTS_c_e,
                      LoggerLogFlags.MG_ENTROPIES_e,
                      LoggerLogFlags.MG_REWARDS_e_c,
                      LoggerLogFlags.TEST_REWARD_e]
config.plot_options = [LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e,
                       LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e,
                       LoggerLogFlags.CO_ENTROPIES_c_e,
                       LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e,
                       LoggerLogFlags.CC_ENTROPIES_e_c,
                       LoggerLogFlags.MG_WEIGHTS_c_e, LoggerLogFlags.MG_ENTROPIES_e,
                       LoggerLogFlags.MG_REWARDS_e_c]
config.save_per_it_options = config.plot_options
config.final_save_options = [LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e,
                             LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e,
                             LoggerLogFlags.CO_ENTROPIES_c_e,
                             LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e,
                             LoggerLogFlags.CC_ENTROPIES_e_c,
                             LoggerLogFlags.MG_WEIGHTS_c_e,
                             LoggerLogFlags.MG_ENTROPIES_e,
                             LoggerLogFlags.MG_REWARDS_e_c,
                             LoggerLogFlags.TEST_REWARD_e,
                             LoggerLogFlags.TEST_MIXT_ENTR_e]

########################################################################################################################
# initial structure of policies
########################################################################################################################

np.random.seed(c_seed)

initial_cmp_p = np.tile(np.expand_dims(config.cmp_init_mean_params, 0), [config.num_init_components, 1, 1])
initial_cmp_cov = np.tile(np.expand_dims(config.cmp_init_cov, 0), [config.num_init_components, 1, 1])
initial_ctxt_cmp_means = np.random.normal(loc=config.ctxt_init_mean, scale=config.ctxt_init_mean_std,
                                          size=[config.num_init_components, config.ctxt_dim])
initial_ctxt_cmp_covars = np.tile(np.expand_dims(config.ctxt_init_cov, 0), [config.num_init_components, 1, 1])
########################################################################################################################
algo = SVSL(config=config, seed=c_seed, init_cmp_params=initial_cmp_p, init_cmp_covars=initial_cmp_cov,
            init_ctxt_cmp_means=initial_ctxt_cmp_means, init_ctxt_cmp_covars=initial_ctxt_cmp_covars)
algo.train(env)
algo.logger.save_me(config.final_save_options)

##############################
# Playing
##############################
from util.SaveAndLoad import load_model_linmoe
from contextual_environments.mujoco.table_tennis_multi import get_tabletennis_cost_func
from util.plot_results import plot_cmp_ctxts_4d, colors
import matplotlib.pyplot as plt

np.random.seed(0)
rep_path = './experiments/tabletennis/0'

env = get_tabletennis_cost_func(context_range_bounds)

n_samples_executed = np.cumsum(np.load(rep_path + '/n_ep_samples_executed.npz',allow_pickle=True)['arr_0'])
test_reward = np.load(rep_path + '/test_reward.npz',allow_pickle=True)['arr_0']


plt.plot(test_reward)
ctxt_list = [[-0.2, -0.4, -0.3, -0.4],
         [-0.2, 0.3, -1, 0.3],
         [-0.2, 0.5, -1, 0.4],
         [-0.5, -0.0, -0.7, 0.0]
        ]

model = load_model_linmoe(rep_path)
plot_cmp_ctxts_4d(model, fig=None, colors=colors, cmp_indices=None)

for j in range(len(ctxt_list)):
    ctxt = np.atleast_2d(ctxt_list[j])
    sampled_indices = model.sample(np.repeat(ctxt, 1000, axis=0))[1]
    unique_cmps = np.unique(sampled_indices)
    chosen_idx = []
    n_success = 0
    for i in range(unique_cmps.shape[0]):
        cmp_idx = unique_cmps[i]
        mean_c_cmp = model.components[int(cmp_idx)].means(ctxt)
        reward, success = env.step(mean_c_cmp, ctxt, render=False)

