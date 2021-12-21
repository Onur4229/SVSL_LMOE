import matplotlib.pyplot as plt
import numpy as np

from contextual_environments.mujoco.beerpong import get_env, ACTION_DIM, CTXT_DIM, CONTEXT_RANGE_BOUNDS
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

n_cores = 8
action_dim = ACTION_DIM
ctxt_dim = CTXT_DIM
context_range_bounds = CONTEXT_RANGE_BOUNDS

env = get_env(name='bp', n_cores=n_cores, env_bounds=context_range_bounds)
########################################################################################################################
# configurations
########################################################################################################################

config = SVSL.get_default_config()

config.exp_name = 'beerpong'
config.action_dim = action_dim
config.ctxt_dim = ctxt_dim
config.train_epochs_p_comp = 750
config.train_epochs_weights = 100
config.num_init_components=1
config.rm_component_thresh=1e-6
config.n_samples_buffer=int(1.5 * (1 + (ctxt_dim + action_dim) + int((ctxt_dim + action_dim) *
                                                                     ((ctxt_dim + action_dim) + 1) / 2)))
config.n_new_samples_comps=int(2*(4 + np.floor(3 * np.log(config.ctxt_dim + config.action_dim))))
config.n_new_samples_ctxt=int(2*(4 + np.floor(3 * np.log(config.ctxt_dim))))
config.n_cmp_addings=70
config.fine_tune_every_it=50
config.save_model_every_it = 50
config.context_range_bounds = context_range_bounds
config.test_every_it=15000000
config.del_loc_opt = False

########################################################################################################################
# Hyperparameters
########################################################################################################################
config.alpha=0.001
config.beta=0.5
config.beta_weight=2.5
config.context_kl_bound=0.005
config.weight_kl_bound=0.01
config.component_kl_bound=0.01

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
config.ctxt_init_mean_std = np.array([0.4, 0.8])
config.ctxt_init_mean = np.mean(np.array(context_range_bounds), axis=0)
ctxt_init_cov_prior = [0.09, 0.25]
config.ctxt_init_cov = np.eye(config.ctxt_dim)*np.array(ctxt_init_cov_prior)

turn_cov = 0.05
push_cov = 1
diag_elems = [turn_cov, push_cov, turn_cov, push_cov, turn_cov, push_cov, turn_cov]*2
diag_elems.append(0.25) # variance for choosing trajectory length ....

config.cmp_init_cov = np.diag(diag_elems)*5
config.cmp_init_cov[-1, -1] = 0.25      # correct scaling covariance with 5 for trajectory variance ....
config.cmp_init_mean_params = np.zeros((config.ctxt_dim + 1, config.action_dim))

########################################################################################################################
# Logging, Plotting & Saving
########################################################################################################################
config.png_plot_saving=False
config.vis_plots=False
config.png_plot_save_every_it=50
config.snapshot_data_saving_per_it=False
config.save_data_ever_it=10000

config.log=True

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
initial_ctxt_cmp_means = np.random.normal(config.ctxt_init_mean, config.ctxt_init_mean_std,
                                          size=[config.num_init_components, config.ctxt_dim])
initial_ctxt_cmp_covars = np.tile(np.expand_dims(config.ctxt_init_cov, 0), [config.num_init_components, 1, 1])
########################################################################################################################
algo = SVSL(config, c_seed, init_cmp_params=initial_cmp_p, init_cmp_covars=initial_cmp_cov,
            init_ctxt_cmp_means=initial_ctxt_cmp_means, init_ctxt_cmp_covars=initial_ctxt_cmp_covars)
algo.train(env)
algo.logger.save_me(config.final_save_options)


##############################
# Plotting and Playing
##############################
from util.SaveAndLoad import load_model_linmoe
from contextual_environments.mujoco.beerpong import get_beerpong_cost_func
from util.plot_results import plot_ball_trajs, colors, plot_cmp_ctxts

np.random.seed(0)
rep_path = './experiments/beerpong/0/'

env = get_beerpong_cost_func(context_range_bounds)

# example contexts for plotting different ball trajectories
ctxts_1 = np.array([[-0.2, -1.4]])
ctxts_2 = np.array([[0.0, -1.4]])
ctxts_3 = np.array([[0.2, -1.4]])
ctxts_4 = np.array([[-0.2, -1.6]])
ctxts_5 = np.array([[0.0, -1.6]])
ctxts_6 = np.array([[0.2, -1.6]])
ctxts_7 = np.array([[-0.2, -1.8]])
ctxts_8 = np.array([[0.0, -1.8]])
ctxts_9 = np.array([[0.2, -1.8]])
ctxts_10 = np.array([[-0.2, -2.0]])
ctxts_11 = np.array([[0.0, -2.0]])
ctxts_12 = np.array([[0.2, -2.0]])
ctxt_list = [ctxts_1, ctxts_2, ctxts_3, ctxts_4, ctxts_5, ctxts_6, ctxts_7, ctxts_8, ctxts_9, ctxts_10, ctxts_11,
             ctxts_12]

model = load_model_linmoe(rep_path)
plot_cmp_ctxts(model, fig=None, colors=colors, cmp_indices=None)
for j in range(len(ctxt_list)):
    ctxts = np.atleast_2d(ctxt_list[j])
    ball_trajs_all_cmps = []
    chosen_idx = []
    succesful_idx = []
    n_success=0
    for k in range(100):
        _, cmp_idx = model.sample(ctxts)
        chosen_idx.append(int(cmp_idx))
        actions = np.zeros((ctxts.shape[0], action_dim))
        for i in range(cmp_idx.shape[0]):
            actions[i] = model.components[cmp_idx[i]].means(ctxts)
        reward, success = env.step(actions, ctxts, render=False)
        if success:
            succesful_idx.append(int(cmp_idx))
            n_success += 1
            ball_trajs_all_cmps.append(env.ball_traj.copy())
    ball_trajs_fig = plot_ball_trajs(ball_trajs_all_cmps, idx=succesful_idx, fig=None, use_colors=colors)
plt.show()