import matplotlib.pyplot as plt
import numpy as np

from contextual_environments.PlanarRobotObstacle_2dim_ctxt import PlanarRobotObstacle_2dim
from SVSL import SVSL
from util.Logger import LoggerLogFlags
from util.SaveAndLoad import load_model_linmoe
from util.plot_results import colors, plot_cmp_ctxts

########################################################################################################################
# seed
########################################################################################################################

c_seed = 0
np.random.seed(c_seed)

########################################################################################################################
# create environment
########################################################################################################################

env = PlanarRobotObstacle_2dim.create_env()
context_range_bounds = env.context_range_bounds
action_dim = env.action_dim
ctxt_dim = env.ctxt_dim
########################################################################################################################
# configurations
########################################################################################################################

config = SVSL.get_default_config()

config.exp_name = 'pl2'
config.action_dim = action_dim
config.ctxt_dim = ctxt_dim
config.train_epochs_p_comp = 350
config.train_epochs_weights = 100
config.num_init_components=1
config.rm_component_thresh=1e-6
config.n_samples_buffer=int(1.5 * (1 + (ctxt_dim + action_dim) + int((ctxt_dim + action_dim) *
                                                                     ((ctxt_dim + action_dim) + 1) / 2)))
n_new_sample_ratio = 0.6
config.n_new_samples_comps=int(n_new_sample_ratio*config.n_samples_buffer)
config.n_new_samples_ctxt=int(n_new_sample_ratio*int(1.5 * (1 + ctxt_dim + ctxt_dim * (ctxt_dim + 1) / 2)))
config.n_cmp_addings=60
config.fine_tune_every_it=50
config.save_model_every_it = 50
config.context_range_bounds = context_range_bounds
config.test_every_it=150

########################################################################################################################
# Hyperparameters
########################################################################################################################
config.alpha=1e-4
config.beta=1
config.beta_weight = config.beta

# MORE Params
config.context_kl_bound=0.001
config.weight_kl_bound=0.01
config.component_kl_bound=0.01

# for ablation
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
config.ctxt_init_mean_std=np.array([1.5, 3])
config.ctxt_init_mean=np.mean(np.array(context_range_bounds), axis=0)
config.ctxt_init_cov=np.eye(config.ctxt_dim)*np.array([1, 1])
config.cmp_init_cov=np.eye(config.action_dim)*(1/2*3.8)*0.1
config.cmp_init_cov[0,0] = 5
config.cmp_init_mean_params=np.zeros((config.ctxt_dim + 1, config.action_dim))

########################################################################################################################
# Logging, Plotting & Saving
########################################################################################################################
config.png_plot_saving=False
config.vis_plots=False
config.png_plot_save_every_it=500000
config.snapshot_data_saving_per_it=False
config.save_data_ever_it=10000

config.log=True
config.log_options = [LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e, LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e,
                      LoggerLogFlags.CO_POSITIONS_c_i,
                      LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e, LoggerLogFlags.CC_LOC_C_SAMPLES_c_i,
                      LoggerLogFlags.CC_REWARDS_LOC_C_SAMPLES_c_i, LoggerLogFlags.MC_VALS_s_i,
                      LoggerLogFlags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c,
                      LoggerLogFlags.CC_ENTROPIES_e_c,
                      LoggerLogFlags.MC_PROBS_s_i,
                      LoggerLogFlags.MG_WEIGHTS_c_e, LoggerLogFlags.MG_ENTROPIES_e, LoggerLogFlags.MG_REWARDS_e_c,
                      LoggerLogFlags.TEST_REWARD_e]
config.plot_options = [LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e, LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e,
                      LoggerLogFlags.CO_POSITIONS_c_i,
                      LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e, LoggerLogFlags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c]
config.save_per_it_options = config.plot_options
config.final_save_options = [LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e, LoggerLogFlags.CO_POSITIONS_c_i,
                             LoggerLogFlags.CC_ENTROPIES_e_c, LoggerLogFlags.MG_WEIGHTS_c_e,
                             LoggerLogFlags.MG_ENTROPIES_e, LoggerLogFlags.MG_REWARDS_e_c,
                             LoggerLogFlags.TEST_REWARD_e, LoggerLogFlags.TEST_MIXT_ENTR_e]


########################################################################################################################
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

########################################################################################################################
# plot model
# example contexts
plot_ctxts = [[5.5, 0], [6.5, 0], [5.5, -2], [6.5, -2], [5.5,2], [6.5, 2]]
repetitions = 100
mask_ar = np.arange(len(plot_ctxts))
mask_ar = np.tile(mask_ar, repetitions)
plot_ctxts = np.array(plot_ctxts)
plot_ctxts = np.tile(plot_ctxts, [repetitions, 1])
path = "./experiments/" + str(config.exp_name) + '/0/'
test_rewards = np.load(path+'test_reward.npz', allow_pickle=True)['arr_0']

plt.figure()
plt.plot(test_rewards)
plt.xlabel('Test Iterations')
plt.ylabel('Test Rewards')

model = load_model_linmoe(path)
samples, cmp_ind = model.sample(plot_ctxts)
fig_w = None
sampled_cmp_idx_dict_w = {}
for i in range(samples.shape[0]):
    print(i)
    mask_idx = mask_ar[i]
    if cmp_ind[i] not in list(sampled_cmp_idx_dict_w.keys()):
        sampled_cmp_idx_dict_w[cmp_ind[i]] = []
    if mask_idx not in sampled_cmp_idx_dict_w[cmp_ind[i]]:
        sampled_cmp_idx_dict_w[cmp_ind[i]].append(mask_idx)
        act = model.components[cmp_ind[i]].means(plot_ctxts[i].reshape((1, -1)))
        fig_w = env.visualize(act.reshape((1, -1)), config=None, fig=fig_w, cond_gating=None, contexts=plot_ctxts[i].reshape((1, -1)),
                            all_contexts=plot_ctxts[i].reshape((1, -1)), color=colors[cmp_ind[i]])

ctxt_plot_fig = plot_cmp_ctxts(model)
plt.figure(ctxt_plot_fig.number)
# plot context range bounds
plt.plot([4.5, 4.5], [-6, 6], 'r--')
plt.plot([7, 7], [-6, 6], 'r--')
plt.plot([4.5, 7], [-6, -6], 'r--')
plt.plot([4.5, 7], [6, 6], 'r--')