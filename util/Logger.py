from os import listdir

import numpy as np
import json
import os

from enum import Flag, auto
from copy import deepcopy


class LoggerLogFlags(Flag):
    """
    Naming Convention:
        x_y_z:
            x -> Type:
                MARGINAL_GATING (MG): pi(o)
                MARGINAL_CONTEXT (MC): pi(s)
                CONDITIONAL_GATING (CG): pi(o|s)
                CONDITIONAL_CONTEXT (CC): pi(s|o)
                COMPONENT (CO): pi(a|s,o)
             (marginal_Gating, Conditional_Gating (pi(o|s)) Conditional_Context, Component)
            y -> Type to be stored (e.g. weights, entropy, ...)
            z -> Type of context:
                i-> Current values (snapshot at iteration i)
                s-> over context values
                e-> over epochs
                c-> over components -> each component has corresponding list storing the information
    """

    # pi(o)
    MG_WEIGHTS_i = auto()				    # weights snapshot at iteration i (snpashot)
    MG_WEIGHTS_c_e = auto()                 # weights for each component over epochs
    MG_ENTROPIES_e = auto()				    # entropies stored over all training iterations (epochs)
    MG_REWARDS_e_c = auto()				    # rewards (per comp) over epochs
    MG_REWARDS_UNDELETED_e_c = auto()				    # rewards (per comp) over epochs

    # pi(s)
    MC_PROBS_s_i = auto()				    # probs over context samples at iteration i - snapshot
    MC_VALS_s_i = auto()				    # values (global context vals) at iteration i - snapshot

    # pi(o|s)
    CG_WEIGHTS_s_c_i = auto()					    # at iteration i (snapshot at iteration i)
    CG_ENTROPIES_e = auto()

    # pi(s|o)
    CC_PROBS_s_c_i = auto()				    # probs for each comp over context samples at iteration i - snapshot
    CC_ENTROPIES_e_c = auto()			    # entropies for each comp over epochs
    CC_MEANREWARDS_LOC_C_SAMPLES_e_c = auto() # mean rewards for each comp over epoch
    CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c = auto() # pi(o|s) mean rwrd + beta*entropy for each comp over epoch
    CC_REWARDS_LOC_C_SAMPLES_c_i = auto()   # rewards as points for each comp at iteration i # - snapshot
    CC_LOC_C_SAMPLES_c_i = auto()			# local context samples for each component at iteration i - snapshot
    CC_QUAD_REW_PREDS_LOCAL_c_i = auto()    # reward preds from quad. model for local ctxts for each comp. at iter i
    CC_REWARDS_GLOBAL_SAMPLES_c_i = auto()  # reward points for global ctxts for each component at iter i -snapshot
    CC_QUAD_REW_PREDS_GLOBAL_c_i = auto()   # reward preds from quad. model for global ctxts for each comp. at iter i
    CC_MEAN_LOG_RESPS_c_e = auto()			# mean log resp (pi(o|s)) for local ctxt samples for each comp. over epoch
    CC_ADD_DELETE_INFO_c = auto()			# special information for every conditional context, when added and deleted

    # pi(a|s,o)
    CO_MEANREWARDS_WITH_LOG_RESPS_c_e = auto()  # mean reward including log resps for each comp over epochs
    CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e = auto()  # mean reward + alpha*entropy for each comp over epochs
    CO_MEAN_TASK_REWARDS_c_e = auto()		# mean task specific rewards only for each comp over epochs
    CO_MEAN_SMOOTHNESS_REWARDS_c_e = auto()	# mean smoothness rewards only for each comp over epochs
    CO_MEAN_DISTANCE_REWARDS_c_e = auto()   # mean distance rewards only for each comp over epochs
    CO_MEAN_COLLISION_REWARDS_c_e = auto()  # mean collision rewards only for each comp over epocs
    CO_ENTROPIES_c_e = auto()				# entropies for each component over epoch
    CO_MEAN_LOG_RESPS_c_e = auto()			# mean log resps for each component over epoch
    CO_UPDATE_TIME_c_e = auto()				# update time for each component over epoch
    CO_POSITIONS_c_i = auto()				# component positions (based on samples) for every comp at it i (snapshot)
    CO_ADD_DELETE_INFO_c = auto()			# special information for every component, when added and deleted...

    # Validation
    TEST_REWARD_e = auto()
    TEST_NUM_COMP_CHOSEN_c_e = auto()
    TEST_MIXT_ENTR_e =auto()


    MARGINAL_GATING = MG_WEIGHTS_i | MG_WEIGHTS_c_e | MG_ENTROPIES_e | MG_REWARDS_e_c | MG_REWARDS_UNDELETED_e_c | CG_WEIGHTS_s_c_i
    CONDITIONAL_CONTEXT = MC_PROBS_s_i | MC_VALS_s_i | CC_PROBS_s_c_i | CC_ENTROPIES_e_c |  \
                          CC_MEANREWARDS_LOC_C_SAMPLES_e_c | CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c | \
                          CC_REWARDS_LOC_C_SAMPLES_c_i | CC_LOC_C_SAMPLES_c_i | CC_QUAD_REW_PREDS_LOCAL_c_i | \
                          CC_REWARDS_GLOBAL_SAMPLES_c_i | CC_QUAD_REW_PREDS_GLOBAL_c_i | CC_MEAN_LOG_RESPS_c_e | \
                          CC_ADD_DELETE_INFO_c
    COMPONENT = CO_MEANREWARDS_WITH_LOG_RESPS_c_e | CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e | CO_MEAN_TASK_REWARDS_c_e | \
                CO_MEAN_SMOOTHNESS_REWARDS_c_e | CO_MEAN_DISTANCE_REWARDS_c_e | CO_MEAN_COLLISION_REWARDS_c_e | \
                CO_ENTROPIES_c_e | CO_MEAN_LOG_RESPS_c_e | CO_UPDATE_TIME_c_e | CO_POSITIONS_c_i | CO_ADD_DELETE_INFO_c
    EPOCHS_VALUED = MG_WEIGHTS_c_e | MG_ENTROPIES_e | MG_REWARDS_e_c | MG_REWARDS_UNDELETED_e_c | CG_ENTROPIES_e |CC_ENTROPIES_e_c | \
                    CC_MEANREWARDS_LOC_C_SAMPLES_e_c | CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c | \
                    CC_MEAN_LOG_RESPS_c_e | CO_MEANREWARDS_WITH_LOG_RESPS_c_e | CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e | \
                    CO_MEAN_TASK_REWARDS_c_e | CO_MEAN_SMOOTHNESS_REWARDS_c_e | CO_MEAN_DISTANCE_REWARDS_c_e | \
                    CO_MEAN_COLLISION_REWARDS_c_e | CO_ENTROPIES_c_e | CO_MEAN_LOG_RESPS_c_e | CO_UPDATE_TIME_c_e | \
                    TEST_REWARD_e | TEST_NUM_COMP_CHOSEN_c_e | TEST_MIXT_ENTR_e
    LIST_VALUED = CC_PROBS_s_c_i | CC_ENTROPIES_e_c |  \
                  CC_MEANREWARDS_LOC_C_SAMPLES_e_c | CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c | \
                  CC_REWARDS_LOC_C_SAMPLES_c_i | CC_LOC_C_SAMPLES_c_i | CC_QUAD_REW_PREDS_LOCAL_c_i | \
                  CC_REWARDS_GLOBAL_SAMPLES_c_i | CC_QUAD_REW_PREDS_GLOBAL_c_i | CC_MEAN_LOG_RESPS_c_e | \
                  CC_ADD_DELETE_INFO_c | COMPONENT | CG_WEIGHTS_s_c_i | MG_REWARDS_e_c | MG_REWARDS_UNDELETED_e_c | MG_WEIGHTS_c_e | \
                  TEST_NUM_COMP_CHOSEN_c_e
    SAVE_AT_RUNTIME = MARGINAL_GATING | CONDITIONAL_CONTEXT | COMPONENT


class Logger:
    def __init__(self, config):

        self.c = config
        self.save2path = None
        self.save2path_per_it = None
        self.figures = {}
        self.from_flag2_attr = {}
        self.not2save_list = ['c',  'save2path', 'save2path_per_it', 'figures', #'from_flag2_attr',
                              'not2save_list', 'comp_add_delete_info_c', 'cond_ctxt_add_delete_info_c', 'del_idx_cmp_loc_opt',
                              'n_samples', 'n_ep_samples_executed', 'n_env_interacts']

        self.n_samples = 0
        self.n_ep_samples_executed = []
        self.n_env_interacts = []

        self.del_idx_cmp_loc_opt = []
        # Flag ensemble

        self.SAVE_AT_RUNTIME = [LoggerLogFlags.MARGINAL_GATING, LoggerLogFlags.CONDITIONAL_CONTEXT,
                                LoggerLogFlags.COMPONENT]

        # (Validation Reward - sampled after every x iteration)
        # self.test_reward = np.zeros((int(self.c.n_tot_it/self.c.test_every_it), 1))
        self.test_reward = []
        self.from_flag2_attr[LoggerLogFlags.TEST_REWARD_e] = self.test_reward

        self.test_num_comps_chosen = []
        self.from_flag2_attr[LoggerLogFlags.TEST_NUM_COMP_CHOSEN_c_e] = self.test_num_comps_chosen

        self.test_mixture_model_entropy = []
        self.from_flag2_attr[LoggerLogFlags.TEST_MIXT_ENTR_e] = self.test_mixture_model_entropy

        # data structure

        # pi(o)
        self.marginal_weights_i = np.zeros((self.c.num_init_components, 1))
        self.from_flag2_attr[LoggerLogFlags.MG_WEIGHTS_i] = self.marginal_weights_i

        self.marginal_weights_c_e = []  # not given as flag, can directly sotre with maringal_weights_i per iteration
        self.from_flag2_attr[LoggerLogFlags.MG_WEIGHTS_c_e] = self.marginal_weights_c_e

        self.marginal_entropies_e = np.zeros((self.c.n_tot_it, 1))
        self.from_flag2_attr[LoggerLogFlags.MG_ENTROPIES_e] = self.marginal_entropies_e

        self.marginal_rewards_e_c = []
        self.from_flag2_attr[LoggerLogFlags.MG_REWARDS_e_c] = self.marginal_rewards_e_c

        self.marginal_rewards_undeleted_e_c = []
        self.from_flag2_attr[LoggerLogFlags.MG_REWARDS_UNDELETED_e_c] = self.marginal_rewards_undeleted_e_c


        # pi(s)
        self.marginal_ctxt_probs_s_i = np.zeros((self.c.n_samples_buffer, 1))
        self.from_flag2_attr[LoggerLogFlags.MC_PROBS_s_i] = self.marginal_ctxt_probs_s_i
        self.marginal_ctxt_vals_s_i = np.zeros((self.c.n_samples_buffer, self.c.ctxt_dim))
        self.from_flag2_attr[LoggerLogFlags.MC_VALS_s_i] = self.marginal_ctxt_vals_s_i

        # pi(o|s)
        self.cond_gating_weights_s_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CG_WEIGHTS_s_c_i] = self.cond_gating_weights_s_c_i

        self.cond_gating_entropies_e = np.zeros((self.c.n_tot_it, 1))
        self.from_flag2_attr[LoggerLogFlags.CG_ENTROPIES_e] = self.cond_gating_entropies_e

        # pi(s|o)
        self.cond_ctxt_probs_s_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_PROBS_s_c_i] = self.cond_ctxt_probs_s_c_i

        self.cond_ctxt_entropies_e_c = []
        self.from_flag2_attr[LoggerLogFlags.CC_ENTROPIES_e_c] = self.cond_ctxt_entropies_e_c

        self.cond_ctxt_mean_rewards_loc_ctxt_samples_e_c = []
        self.from_flag2_attr[LoggerLogFlags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c] = \
            self.cond_ctxt_mean_rewards_loc_ctxt_samples_e_c

        self.cond_ctxt_mean_rewards_with_entropy_bonus_loc_samples_e_c = []
        self.from_flag2_attr[LoggerLogFlags.CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c] = \
            self.cond_ctxt_mean_rewards_with_entropy_bonus_loc_samples_e_c

        self.cond_ctxt_rewards_loc_c_samples_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_REWARDS_LOC_C_SAMPLES_c_i] = self.cond_ctxt_rewards_loc_c_samples_c_i

        self.cond_ctxt_loc_c_samples_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_LOC_C_SAMPLES_c_i] = self.cond_ctxt_loc_c_samples_c_i

        self.cond_ctxt_quad_rew_preds_loc_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_QUAD_REW_PREDS_LOCAL_c_i] = self.cond_ctxt_quad_rew_preds_loc_c_i

        self.cond_ctxt_rewards_global_samples_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_REWARDS_GLOBAL_SAMPLES_c_i] = self.cond_ctxt_rewards_global_samples_c_i

        self.cond_ctxt_quad_rew_preds_global_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CC_QUAD_REW_PREDS_GLOBAL_c_i] = self.cond_ctxt_quad_rew_preds_global_c_i

        self.cond_ctxt_mean_log_resps_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e] = self.cond_ctxt_mean_log_resps_c_e

        self.cond_ctxt_add_delete_info_c = []
        self.from_flag2_attr[LoggerLogFlags.CC_ADD_DELETE_INFO_c] = self.cond_ctxt_add_delete_info_c


        # pi(a|s,o)
        self.comp_mean_rewards_with_log_resps_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEANREWARDS_WITH_LOG_RESPS_c_e] = \
            self.comp_mean_rewards_with_log_resps_c_e

        self.comp_mean_rewards_with_entropy_bonus_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e] = \
            self.comp_mean_rewards_with_entropy_bonus_c_e

        self.comp_mean_task_rewards_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e] = self.comp_mean_task_rewards_c_e

        self.comp_mean_smoothness_rewards_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEAN_SMOOTHNESS_REWARDS_c_e] = self.comp_mean_smoothness_rewards_c_e

        self.comp_mean_distance_rewards_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEAN_DISTANCE_REWARDS_c_e] = self.comp_mean_distance_rewards_c_e

        self.comp_mean_collision_rewards_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEAN_COLLISION_REWARDS_c_e] = self.comp_mean_collision_rewards_c_e
        self.comp_entropoies_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_ENTROPIES_c_e] = self.comp_entropoies_c_e

        self.comp_mean_log_resps_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e] = self.comp_mean_log_resps_c_e

        self.comp_update_time_c_e = []
        self.from_flag2_attr[LoggerLogFlags.CO_UPDATE_TIME_c_e] = self.comp_update_time_c_e

        self.comp_positions_c_i = []
        self.from_flag2_attr[LoggerLogFlags.CO_POSITIONS_c_i] = self.comp_positions_c_i

        self.comp_add_delete_info_c = []
        self.from_flag2_attr[LoggerLogFlags.CO_ADD_DELETE_INFO_c] = self.comp_add_delete_info_c

    def get_save_path(self):

        exp_numbers = list(range(100000))
        exp_num = None
        path = None
        for i in exp_numbers:
            if not os.path.isdir(self.save2path + '/' + str(i)):
                path = self.save2path + '/' + str(i)
                exp_num = str(i)
                break
        return path, exp_num

    def initialize(self):

        for c_comp in range(self.c.num_init_components):
            # pi(o)
            self.marginal_weights_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.marginal_rewards_e_c.append(np.zeros((self.c.n_tot_it, 1)))

            # pi(o|s)
            self.cond_gating_weights_s_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))

            # pi(s|o)
            self.cond_ctxt_probs_s_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
            self.cond_ctxt_entropies_e_c.append(np.zeros((self.c.n_tot_it, 1)))
            self.cond_ctxt_mean_rewards_loc_ctxt_samples_e_c.append(np.zeros((self.c.n_tot_it, 1)))
            self.cond_ctxt_mean_rewards_with_entropy_bonus_loc_samples_e_c.append(np.zeros((self.c.n_tot_it, 1)))
            self.cond_ctxt_rewards_loc_c_samples_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
            self.cond_ctxt_loc_c_samples_c_i.append(np.zeros((self.c.n_samples_buffer, self.c.ctxt_dim)))
            self.cond_ctxt_quad_rew_preds_loc_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
            self.cond_ctxt_rewards_global_samples_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
            self.cond_ctxt_quad_rew_preds_global_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
            self.cond_ctxt_mean_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))

            # pi(a|s,o)
            self.comp_mean_rewards_with_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_rewards_with_entropy_bonus_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_task_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_smoothness_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_distance_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_collision_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_entropoies_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_mean_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_update_time_c_e.append(np.zeros((self.c.n_tot_it, 1)))
            self.comp_positions_c_i.append(np.zeros((self.c.n_samples_buffer, self.c.action_dim)))

    def log(self, flag, val2be_stored, c_it=None, cmp_idx=None):

        if LoggerLogFlags.CC_ADD_DELETE_INFO_c not in flag and LoggerLogFlags.CO_ADD_DELETE_INFO_c not in flag:
            if len(val2be_stored.shape) == 1:
                val2be_stored=val2be_stored.reshape((-1, 1))
            attr = self.from_flag2_attr[flag]
            if cmp_idx is None:     # no list
                if c_it is None:    # not an epoch information
                    attr[:, :] = val2be_stored
                else:
                    attr[c_it, :] = val2be_stored
            else:
                if c_it is None:
                    attr[cmp_idx][:, :] = val2be_stored
                else:
                    attr[cmp_idx][c_it, :] = val2be_stored
        else:
            if LoggerLogFlags.CC_ADD_DELETE_INFO_c:
                raise ValueError("Currently storing nothing!!")
            if LoggerLogFlags.CO_ADD_DELETE_INFO_c:
                raise ValueError("Currently stroing nothing!!")

    def create_new_dict_add_delete_info(self, idx_undeleted_list, added_it, deleted_it, idx_deleted_list, add_type):

        co_infos = {}
        cc_infos = {}
        # extend here what you need to store ....
        keys = ['idx_undeleted_list', 'added_it', 'deleted_it', 'idx_deleted_list', 'add_type']
        args = [idx_undeleted_list, added_it, deleted_it, idx_deleted_list, add_type]
        for key, arg in zip(keys, args):
            co_infos[key] = arg
            cc_infos[key] = arg
        return co_infos, cc_infos

    def get_information_from_comp_delete_add_information(self, comp_idx):

        for i in range(len(self.comp_add_delete_info_c)):
            c_co_dict = self.comp_add_delete_info_c[i]
            c_cc_dict = self.cond_ctxt_add_delete_info_c[i]
            if c_co_dict['idx_deleted_list'] == comp_idx:
                idx_undeleted_list_co = c_co_dict['idx_undeleted_list']
                idx_undeleted_list_cc = c_cc_dict['idx_undeleted_list']

                added_it_co = c_co_dict['added_it']
                added_it_cc = c_cc_dict['added_it']

                deleted_it_co = c_co_dict['deleted_it']
                deleted_it_cc = c_cc_dict['deleted_it']

                idx_deleted_list_co = c_co_dict['idx_deleted_list']
                idx_deleted_list_cc = c_cc_dict['idx_deleted_list']

                add_type_co = c_co_dict['add_type']
                add_type_cc = c_cc_dict['add_type']

                return idx_undeleted_list_co, idx_undeleted_list_cc, added_it_co, added_it_cc, deleted_it_co, \
                       deleted_it_cc, idx_deleted_list_co, idx_deleted_list_cc, add_type_co, add_type_cc

    def add_component(self, added_it, add_type, idx_deleted_list):

        # pi(o)
        if LoggerLogFlags.MG_WEIGHTS_i in self.c.log_options:
            new_marginal_weights_i = np.zeros((self.marginal_weights_i.shape[0] + 1, 1))
            new_marginal_weights_i[:-1, :] = self.marginal_weights_i[:, :]
            self.marginal_weights_i = new_marginal_weights_i
            self.from_flag2_attr[LoggerLogFlags.MG_WEIGHTS_i] = self.marginal_weights_i
        if LoggerLogFlags.MG_WEIGHTS_c_e in self.c.log_options:
            self.marginal_weights_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.MG_REWARDS_e_c in self.c.log_options:
            self.marginal_rewards_e_c.append(np.zeros((self.c.n_tot_it, 1)))

        # pi(o|s)
        if LoggerLogFlags.CG_WEIGHTS_s_c_i in self.c.log_options:
            self.cond_gating_weights_s_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))

        # pi(s|o)
        if LoggerLogFlags.CC_PROBS_s_c_i in self.c.log_options:
            self.cond_ctxt_probs_s_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
        if LoggerLogFlags.CC_ENTROPIES_e_c in self.c.log_options:
            self.cond_ctxt_entropies_e_c.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c in self.c.log_options:
            self.cond_ctxt_mean_rewards_loc_ctxt_samples_e_c.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c in self.c.log_options:
            self.cond_ctxt_mean_rewards_with_entropy_bonus_loc_samples_e_c.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CC_REWARDS_LOC_C_SAMPLES_c_i in self.c.log_options:
            self.cond_ctxt_rewards_loc_c_samples_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
        if LoggerLogFlags.CC_LOC_C_SAMPLES_c_i in self.c.log_options:
            self.cond_ctxt_loc_c_samples_c_i.append(np.zeros((self.c.n_samples_buffer, self.c.ctxt_dim)))
        if LoggerLogFlags.CC_QUAD_REW_PREDS_LOCAL_c_i in self.c.log_options:
            self.cond_ctxt_quad_rew_preds_loc_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
        if LoggerLogFlags.CC_REWARDS_GLOBAL_SAMPLES_c_i in self.c.log_options:
            self.cond_ctxt_rewards_global_samples_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
        if LoggerLogFlags.CC_QUAD_REW_PREDS_GLOBAL_c_i in self.c.log_options:
            self.cond_ctxt_quad_rew_preds_global_c_i.append(np.zeros((self.c.n_samples_buffer, 1)))
        if LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e in self.c.log_options:
            self.cond_ctxt_mean_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))

        # pi(a|s,o)
        if LoggerLogFlags.CO_MEANREWARDS_WITH_LOG_RESPS_c_e in self.c.log_options:
            self.comp_mean_rewards_with_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e in self.c.log_options:
            self.comp_mean_rewards_with_entropy_bonus_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e in self.c.log_options:
            self.comp_mean_task_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEAN_SMOOTHNESS_REWARDS_c_e in self.c.log_options:
            self.comp_mean_smoothness_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEAN_DISTANCE_REWARDS_c_e in self.c.log_options:
            self.comp_mean_distance_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEAN_COLLISION_REWARDS_c_e in self.c.log_options:
            self.comp_mean_collision_rewards_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_ENTROPIES_c_e in self.c.log_options:
            self.comp_entropoies_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e in self.c.log_options:
            self.comp_mean_log_resps_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_UPDATE_TIME_c_e in self.c.log_options:
            self.comp_update_time_c_e.append(np.zeros((self.c.n_tot_it, 1)))
        if LoggerLogFlags.CO_POSITIONS_c_i in self.c.log_options:
            self.comp_positions_c_i.append(np.zeros((self.c.n_samples_buffer, self.c.action_dim)))

        # co_new_dict, cc_new_dict = self.create_new_dict_add_delete_info(len(self.comp_add_delete_info_c), added_it,
        #                                                                 deleted_it=None,
        #                                                                 idx_deleted_list=idx_deleted_list,
        #                                                                 add_type=add_type)
        # self.comp_add_delete_info_c.append(co_new_dict)
        # self.comp_add_delete_info_c.append(cc_new_dict)

    def remove_component(self, cmp_idx):

        # pi(o)
        if LoggerLogFlags.MG_WEIGHTS_i in self.c.log_options:
            self.marginal_weights_i = np.delete(self.marginal_weights_i, cmp_idx, axis=0)
        if LoggerLogFlags.MG_WEIGHTS_c_e in self.c.log_options:
            del self.marginal_weights_c_e[cmp_idx]
        # del self.marginal_weights_i[cmp_idx]
        if LoggerLogFlags.MG_REWARDS_e_c in self.c.log_options:
            del self.marginal_rewards_e_c[cmp_idx]

        # pi(o|s)
        if LoggerLogFlags.CG_WEIGHTS_s_c_i in self.c.log_options:
            del self.cond_gating_weights_s_c_i[cmp_idx]

        # pi(s|o)
        if LoggerLogFlags.CC_PROBS_s_c_i in self.c.log_options:
            del self.cond_ctxt_probs_s_c_i[cmp_idx]
        if LoggerLogFlags.CC_ENTROPIES_e_c in self.c.log_options:
            del self.cond_ctxt_entropies_e_c[cmp_idx]
        if LoggerLogFlags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c in self.c.log_options:
            del self.cond_ctxt_mean_rewards_loc_ctxt_samples_e_c[cmp_idx]
        if LoggerLogFlags.CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c in self.c.log_options:
            del self.cond_ctxt_mean_rewards_with_entropy_bonus_loc_samples_e_c[cmp_idx]
        if LoggerLogFlags.CC_REWARDS_LOC_C_SAMPLES_c_i in self.c.log_options:
            del self.cond_ctxt_rewards_loc_c_samples_c_i[cmp_idx]
        if LoggerLogFlags.CC_LOC_C_SAMPLES_c_i in self.c.log_options:
            del self.cond_ctxt_loc_c_samples_c_i[cmp_idx]
        if LoggerLogFlags.CC_QUAD_REW_PREDS_LOCAL_c_i in self.c.log_options:
            del self.cond_ctxt_quad_rew_preds_loc_c_i[cmp_idx]
        if LoggerLogFlags.CC_REWARDS_GLOBAL_SAMPLES_c_i in self.c.log_options:
            del self.cond_ctxt_rewards_global_samples_c_i[cmp_idx]
        if LoggerLogFlags.CC_QUAD_REW_PREDS_GLOBAL_c_i in self.c.log_options:
            del self.cond_ctxt_quad_rew_preds_global_c_i[cmp_idx]
        if LoggerLogFlags.CC_MEAN_LOG_RESPS_c_e in self.c.log_options:
            del self.cond_ctxt_mean_log_resps_c_e[cmp_idx]

        # pi(a|s,o)
        if LoggerLogFlags.CO_MEANREWARDS_WITH_LOG_RESPS_c_e in self.c.log_options:
            del self.comp_mean_rewards_with_log_resps_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e in self.c.log_options:
            del self.comp_mean_rewards_with_entropy_bonus_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEAN_TASK_REWARDS_c_e in self.c.log_options:
            del self.comp_mean_task_rewards_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEAN_SMOOTHNESS_REWARDS_c_e in self.c.log_options:
            del self.comp_mean_smoothness_rewards_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEAN_DISTANCE_REWARDS_c_e in self.c.log_options:
            del self.comp_mean_distance_rewards_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEAN_COLLISION_REWARDS_c_e in self.c.log_options:
            del self.comp_mean_collision_rewards_c_e[cmp_idx]
        if LoggerLogFlags.CO_ENTROPIES_c_e in self.c.log_options:
            del self.comp_entropoies_c_e[cmp_idx]
        if LoggerLogFlags.CO_MEAN_LOG_RESPS_c_e in self.c.log_options:
            del self.comp_mean_log_resps_c_e[cmp_idx]
        if LoggerLogFlags.CO_UPDATE_TIME_c_e in self.c.log_options:
            del self.comp_update_time_c_e[cmp_idx]
        if LoggerLogFlags.CO_POSITIONS_c_i in self.c.log_options:
            del self.comp_positions_c_i[cmp_idx]

    def remove_components(self, former_num_comps, cmp_idx_array):
        if LoggerLogFlags.MG_REWARDS_UNDELETED_e_c in self.c.log_options:
            self.marginal_rewards_undeleted_e_c = deepcopy(self.marginal_rewards_e_c)

        remove_list = [0]*former_num_comps
        for k in range(cmp_idx_array.shape[0]):
            remove_list[cmp_idx_array[k]] = 1

        completed_delete_list = False
        while not completed_delete_list:
            if len(remove_list) == 0:
                completed_delete_list = True
            else:
                completed_delete_list = True
                for k in range(len(remove_list)):
                    if remove_list[k] == 1:
                        completed_delete_list = False
            if not completed_delete_list:
                for k in range(len(remove_list)):
                    if remove_list[k] == 1:
                        del remove_list[k]
                        self.remove_component(k)
                        break

    def save_current_it(self, flags, it):

        path = self.save2path_per_it + '/it_' + str(it)
        if not os.path.isdir(path):
            os.makedirs(path)

        self.save_me(logger_save_list=self.c.final_save_options, path=path)

    def save_me(self, logger_save_list=None, path=None):

        if path is None:
            if not os.path.isdir(self.save2path):  # we are already setting the path.. see constructor ContextualVIPS
                os.makedirs(self.save2path)  # already containes the experiment's number
            path = self.save2path
        # with io.open(path + '/config.yaml', 'w') as stream:
        #     yaml.dump(self.c._c_dict, stream)
        np.savez_compressed(os.path.join(path, 'config' + '.npz'), **self.c._c_dict)

        with open(path + '/comp_add_delete_info_c.json', 'w') as fout:
            json.dump(self.comp_add_delete_info_c, fout)

        with open(path + '/cond_ctxt_add_delete_info_c.json', 'w') as fout:
            json.dump(self.cond_ctxt_add_delete_info_c, fout)

        obj_variables = vars(self)
        # for key in self.not2save_list:
        #     del obj_variables[key]

        for key in obj_variables.keys():
            if key not in self.not2save_list:
                if key != 'from_flag2_attr':
                    if logger_save_list is not None:
                        c_obj_var = obj_variables[key]
                        for flag in logger_save_list:
                            if c_obj_var is self.from_flag2_attr[flag]:
                                # np.save(path + '/' + key, obj_variables[key])
                                np.savez_compressed(path + '/' + key, obj_variables[key])
                    else:
                        np.save(path + '/' + key, obj_variables[key])
            elif key == 'del_idx_cmp_loc_opt':
                np.save(path + '/' + 'del_idx_cmp_loc_opt', obj_variables[key])
            elif key == 'n_samples':
                np.save(path + '/' + 'n_samples', obj_variables[key])
                print('number of samples taken:', self.n_samples)
            elif key == 'n_ep_samples_executed':
                np.savez_compressed(path + '/' + key, self.n_ep_samples_executed)
            elif key == 'n_env_interacts':
                np.savez_compressed(path + '/' + key, self.n_env_interacts)

    @staticmethod
    def load_logger(path2load):

        from util.ConfigDict import ConfigDict
        # with open(path2load+'/config.yaml') as config_input:
        #     config_dict = yaml.load(config_input, Loader=yaml.FullLoader)
        config_dict = dict(np.load(path2load+'/config.npz', allow_pickle=True))

        config = ConfigDict()
        config.set_dict(config_dict)
        logger = Logger(config)

        all_files_in_path = listdir(path2load)
        for file_name in all_files_in_path:
            attr_name, file_name_suffix = os.path.splitext(file_name)
            if file_name_suffix == '.npy':
                path2file = path2load+'/' + file_name
                print(attr_name)
                val = np.load(path2file, allow_pickle=True)
                if 'c' in attr_name.split('_'):
                    val = list(val)
                logger.__setattr__(attr_name, val)
            if file_name_suffix == '.json':
                path2file = path2load + '/' + file_name
                with open(path2file, "r") as read_file:
                    val = json.load(read_file)
                    if 'c' in attr_name.split('_'):
                        val = list(val)
                    logger.__setattr__(attr_name, val)
        return logger