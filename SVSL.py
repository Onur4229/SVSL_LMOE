from SVSL_Base import SVSL_Base
from model_learner.LinMOELearner import LinMOELearner
from util.Logger import LoggerLogFlags as Flags
from copy import deepcopy
from util.SaveAndLoad import save_model_linmoe

import numpy as np
import time
import logging


class SVSL(SVSL_Base):
    def __init__(self, config, seed, init_cmp_params, init_cmp_covars, init_ctxt_cmp_means, init_ctxt_cmp_covars,
                 path=None):
        super(SVSL, self).__init__(config, seed, path)
        self.learner = LinMOELearner(ctxt_dim=self.c.ctxt_dim, sample_dim=self.c.action_dim, surrogate_reg_fact=1e-10,
                                     eta_offset_weight=0, omega_offset_weight=self.c.beta_weight,
                                     eta_offset_ctxt=0, omega_offset_ctxt=self.c.beta, eta_offset_cmp=0,
                                     omega_offset_cmp=self.c.alpha, constrain_entropy=True)
        self.learner.initialize_model(cmp_params=init_cmp_params, cmp_covars=init_cmp_covars,
                                      ctxt_cmp_means=init_ctxt_cmp_means, ctxt_cmp_covars=init_ctxt_cmp_covars)
        self.data_base.bg_cond_gating = deepcopy(self.model.ctxt_components[0])
        self.data_base.bg_comp = deepcopy(self.model.components[0])
        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0
        # need to be initialized here as the model type might be different for every algo
        self.last_quad_model_for_ctxt_updates = []
        self.last_ctxt_cmps_means = []
        self.last_ctxt_cmps_covars = []
        self.plot_actions = []
        self.plot_loc_ctxt_samples = []
        self.last_cmp_task_rewards = []

        # introduced here fore now
        self.save_samples_for_val_func_ctxts = None
        self.save_samples_for_val_func_actions = None
        self.save_samples_for_val_func_rewards = None
        self.copies_of_last_pi_s_o = None
        self.copies_of_last_pi_a_s_o = None
        self.big_tilde_pi_o_s = None
        self.big_tilde_pi_a_s_o = None


    def save_model(self, it=None):
        save_model_linmoe(self.model, self.logger.save2path, it=it)

    @staticmethod
    def calc_entropy_loss_bound(entropy_pi_0, gamma, entropies):
        return gamma * (entropies - entropy_pi_0) + entropy_pi_0

    def update_weights(self):

        if self.c_iteration > self.n_tot_cmp_train_it:
            if self.big_tilde_pi_o_s is None:
                self.save_model(it=self.c_iteration)
                self.logger.save_current_it(flags=self.c.save_per_it_options, it=self.c_iteration)
                self.get_samples(cmp_idx=None, sample_reuse=False, n_new_samples=0)  # when updating the gating, we do not do sample reuse
                                                                                # but we should fill the data base with most recent samples

            rewards, task_specific_rewards, _, __, ___, ____, _____ = self.get_reward(context_update=False,
                                                                                      weight_update=True)
            rewards = np.stack(rewards, -1)
            mg_entropy = self.model.weight_distribution.entropy()

            entropy_pi_0 = 0
            gamma = 0.99
            beta = gamma * (mg_entropy - entropy_pi_0) + entropy_pi_0

            res = self.learner.update_prior_weights(rewards, self.c.weight_kl_bound, entropy_loss_bound=beta)
            res = list(res)
            res.append(beta)
            res.append(np.mean(rewards))

            if self.c.log:
                self.mg_prepare_data_and_log(rewards)
        else:
            res = None
        return res

    def update_ctxt_cmp(self, cmp_idx=None):
        if self.c_iteration <= self.c.n_tot_cmp_train_it:
            samples_local_ctxts, action_samples = self.get_samples(cmp_idx=cmp_idx, sample_reuse=True,
                                                                   n_new_samples=self.c.n_new_samples_ctxt)
            self.plot_loc_ctxt_samples = samples_local_ctxts
            rewards, task_specific_rewards, _, __, ___, ____, ctxt_log_resps = self.get_reward(context_update=True,
                                                                                               weight_update=False)
            entropies = np.array([comp.entropy() for comp in self.model.ctxt_components])
            if cmp_idx is None: # update all components
                targets = rewards
                is_weights = np.ones((self.c.n_samples_buffer, self.model.num_components))/self.c.n_samples_buffer
                beta = self.calc_entropy_loss_bound(entropy_pi_0=-75, gamma=0.99, entropies=entropies)
                res_list = self.learner.update_all_ctxt_cmps(samples_local_ctxts, targets, self.c.context_kl_bound,
                                                             is_weights=is_weights, entropy_loss_bound=beta)
                means = []
                covs = []
                results_list = []
                quad_lin_reg_models = []
                for idx, res in enumerate(res_list):
                    means.append(res[0])
                    covs.append((res[1]))
                    results_list.append(list(res[2:-1]))
                    quad_lin_reg_models.append(res[-1])
                    results_list[-1].append(beta[idx])
                    results_list[-1].append(np.mean(rewards[idx]))

                self.last_quad_model_for_ctxt_updates = quad_lin_reg_models
                self.last_ctxt_cmps_means = means
                self.last_ctxt_cmps_covars = covs

            else:
                targets = rewards[cmp_idx]
                beta = self.calc_entropy_loss_bound(entropy_pi_0=-75, gamma=0.99, entropies=entropies[cmp_idx])
                is_weights = np.ones(targets.shape[0])/targets.shape[0]
                res = self.learner.update_ctxt_cmp(samples_local_ctxts[cmp_idx], targets,
                                                   kl_bound=self.c.context_kl_bound, is_weights=is_weights,
                                                   entropy_loss_bound=beta, cmp_idx=cmp_idx)
                results_list = list(res)
                if len(self.last_quad_model_for_ctxt_updates) - 1 < cmp_idx:
                    self.last_quad_model_for_ctxt_updates.append(results_list[-1])
                    self.last_ctxt_cmps_means.append(results_list[0])
                    self.last_ctxt_cmps_covars.append(results_list[1])
                else:
                    self.last_quad_model_for_ctxt_updates[cmp_idx] = results_list[-1]
                    self.last_ctxt_cmps_means[cmp_idx] = results_list[0]
                    self.last_ctxt_cmps_covars[cmp_idx] = results_list[1]
                results_list = results_list[2:-1]
                results_list.append(beta)
                results_list.append(np.mean(targets))
                results_list = [results_list]
                quad_lin_reg_models = self.last_quad_model_for_ctxt_updates

            if self.c.log:
                self.cc_prepare_data_and_log(quad_lin_reg_models, samples_local_ctxts, entropies, rewards,
                                             ctxt_log_resps)
            return results_list
        else:
            return [None]

    def update_cmp(self, cmp_idx=None):
        if self.c_iteration <= self.c.n_tot_cmp_train_it:
            samples_contexts, samples_components = self.get_samples(cmp_idx=cmp_idx, sample_reuse=True,
                                                                    n_new_samples=self.c.n_new_samples_comps)

            rewards, task_specific_rewards, smoothnes_rewards, distance_rewards, collision_rewards, comp_log_resps, _ = \
                self.get_reward(context_update=False, weight_update=False)
            # stort task specific_rewards from last iteration
            self.last_cmp_task_rewards = task_specific_rewards
            entropies = np.array([comp.expected_entropy() for comp in self.model.components])

            self.plot_actions = samples_components

            if cmp_idx is None:
                if self.c.adv_func_reg_in_Cond_More:
                    V_s = np.zeros((samples_components[0].shape[0], len(samples_components)))
                    targets = []
                    for j in range(self.model.num_components):
                        self.comp_v_func_regressor.fit(rewards_train=rewards[j],
                                                             samples_train=samples_contexts[j])
                        V_s[:, j] = self.comp_v_func_regressor.predict(samples_contexts[j]).squeeze()
                        targets.append(rewards[j] - V_s[:, j])
                else:
                    targets = rewards
                is_weights = np.ones((self.c.n_samples_buffer, self.model.num_components))/self.c.n_samples_buffer
                beta = self.calc_entropy_loss_bound(entropy_pi_0=-75, gamma=0.99, entropies=entropies)
                res_list = self.learner.update_all_cmps(samples_contexts, samples_components, targets, is_weights,
                                                          self.c.component_kl_bound, entropy_loss_bound=beta)
                results_list = []
                for idx, res in enumerate(res_list):
                    if self.c.log:
                        if Flags.CO_UPDATE_TIME_c_e in self.c.log_options:
                            self.logger.log(flag=Flags.CO_UPDATE_TIME_c_e, val2be_stored=np.array(list(res)[-1]),
                                            cmp_idx=idx,
                                            c_it=self.c_iteration)
                    results_list.append(list(res))
                    results_list[-1].append(beta[idx])
                    results_list[-1].append(np.mean(rewards[idx]))
            else:
                comp_samples_ctxts = samples_contexts[cmp_idx]
                comp_actions = samples_components[cmp_idx]
                if self.c.adv_func_reg_in_Cond_More:
                    self.comp_v_func_regressor.fit(rewards_train=rewards[cmp_idx],
                                                         samples_train=comp_samples_ctxts)
                    V_s = self.comp_v_func_regressor.predict(comp_samples_ctxts).squeeze()
                    targets = rewards[cmp_idx] - V_s
                else:
                    targets = rewards[cmp_idx]
                is_weights = np.ones(targets.shape[0])/targets.shape[0]
                beta = self.calc_entropy_loss_bound(entropy_pi_0=-75, gamma=0.99, entropies=entropies[cmp_idx])
                res = self.learner.update_cmp(comp_samples_ctxts, comp_actions, targets, is_weights,
                                                    self.c.component_kl_bound, entropy_loss_bound=beta, cmp_idx=cmp_idx)
                results_list = list(res)
                results_list.append(beta)
                results_list.append(np.mean(rewards[cmp_idx]))
                if Flags.CO_UPDATE_TIME_c_e in self.c.log_options:
                    self.logger.log(flag=Flags.CO_UPDATE_TIME_c_e, val2be_stored=np.array(results_list[-1]),
                                    cmp_idx=cmp_idx, c_it=self.c_iteration)
                results_list = [results_list]
            if self.c.log:
                self.co_prepare_data_and_log(rewards, entropies, task_specific_rewards, smoothnes_rewards,
                                             distance_rewards, collision_rewards, comp_log_resps)
            return results_list
        else:
            return [None]

    def get_samples(self, cmp_idx, sample_reuse, n_new_samples):

        for i in range(len(self.model.components)):
            cmps = self.model.components[i]
            ctxt_cmps = self.model.ctxt_components[i]
            if cmp_idx is not None and i==cmp_idx:
                update_succesfull = False
                while not update_succesfull:
                    n_samples = self.c.n_samples_buffer if (self.data_base.need_to_add_comp(cmp_idx) or not sample_reuse)\
                        else n_new_samples
                        # else self.c.n_new_samples
                    self.logger.n_samples += n_samples
                    ctxts, actions = self.get_samples_from_comp(cmp_idx, n_samples=n_samples)
                    start_time = time.time()
                    task_rewards = self.env.step(actions, ctxts)[0]
                    samples_on_env = self.env.get_tot_n_samples()
                    self.n_ep_samples_executed += samples_on_env[0]
                    self.n_env_interacts += samples_on_env[1]
                    if self.verbose:
                        print('Environment Sampling with ', str(ctxts.shape[0]), ' samples took ', str(time.time()-start_time), ' s')
                    action_weights = cmps.density(ctxts, actions)
                    ctxt_weights = ctxt_cmps.density(ctxts)
                    update_succesfull = self.data_base.update_samples(
                                  comp_params=[cmps._params.copy(), cmps._covar.copy()],
                                  cond_gat_params=[ctxt_cmps.mean.copy(), ctxt_cmps.covar.copy()],
                                  ctxt=ctxts, actions=actions, rewards=task_rewards, ctxt_weights= ctxt_weights,
                                  action_weights=action_weights, use_imp_weights = False, idx=i)
            else:
                if cmp_idx is None:     # which means we update all components
                    update_succesfull = False
                    while not update_succesfull:
                        n_samples = self.c.n_samples_buffer if (self.data_base.need_to_add_comp(i) or not sample_reuse)\
                            else n_new_samples
                        self.logger.n_samples += n_samples
                        ctxts, actions = self.get_samples_from_comp(i, n_samples=n_samples)
                        start_time = time.time()
                        task_rewards = self.env.step(actions, ctxts)[0]
                        samples_on_env = self.env.get_tot_n_samples()
                        self.n_ep_samples_executed += samples_on_env[0]
                        self.n_env_interacts += samples_on_env[1]
                        action_weights = cmps.density(ctxts, actions)
                        ctxt_weights = ctxt_cmps.density(ctxts)
                        if self.verbose:
                            print('Environment Sampling with ', str(ctxts.shape[0]), ' samples took ', str(time.time()-start_time), ' s')
                        update_succesfull = self.data_base.update_samples(
                                    comp_params=[cmps._params.copy(), cmps._covar.copy()],
                                    cond_gat_params=[ctxt_cmps.mean.copy(), ctxt_cmps.covar.copy()],
                                    ctxt=ctxts, actions=actions, rewards=task_rewards,ctxt_weights= ctxt_weights,
                                  action_weights=action_weights, use_imp_weights = False, idx=i)

        ctxt_samples, action_samples = self.data_base.get_updated_samples()
        return ctxt_samples, action_samples

    def get_samples_from_comp(self, cmp_idx, n_samples=None):
        if n_samples is None:
            n_samples = self.c.n_samples_buffer
        ctxt_samples = self.model.ctxt_components[cmp_idx].sample(n_samples)
        actions = self.model.components[cmp_idx].sample(ctxt_samples)
        return ctxt_samples, actions

    def get_reward(self, context_update, weight_update):

        rewards = []
        task_specific_rewards = []
        smoothnes_rewards = []
        distance_rewards = []
        collision_rewards = []
        comp_log_resp = []
        ctxt_log_resp = []
        ctxt_samples = []
        action_samples = []

        for i in range(self.model.num_components):
            # prepare the data needed for log resps
            c_contexts, c_actions, env_reward = self.data_base.get_data(cmp_idx=i)
            ctxt_samples.append(c_contexts)
            action_samples.append(c_actions)
            log_resps, log_gating_probs = self.model.log_responsibilities(c_contexts, c_actions)
            cmp_log_resps = log_resps[:, i]
            cmp_log_gating = log_gating_probs[:, i]
            c_reward = 0

            smoothnes_reward = np.zeros(env_reward.shape)
            distance_reward = np.zeros(env_reward.shape)
            collision_reward = np.zeros(env_reward.shape)

            task_specific_rewards.append(env_reward.copy())
            smoothnes_rewards.append(smoothnes_reward.copy())
            distance_rewards.append(distance_reward.copy())
            collision_rewards.append(collision_reward.copy())

            c_reward += env_reward + self.c.alpha_resp * cmp_log_resps
            comp_log_resp.append(cmp_log_resps)

            if context_update or weight_update:
                c_reward += self.c.alpha * self.model.components[i].expected_entropy()
                c_reward += ((self.c.beta_resp - self.c.alpha_resp) * cmp_log_gating)
                ctxt_log_resp.append(cmp_log_gating)

            if weight_update:
                if self.c.adv_for_gating_update:
                    c_reward = c_reward
                else:
                    c_reward = np.mean(c_reward) + self.c.beta * self.model.ctxt_components[i].entropy()
            rewards.append(c_reward)

        if weight_update and self.c.adv_for_gating_update:
            if self.c_iteration > self.n_tot_cmp_train_it:
                if self.copies_of_last_pi_a_s_o is None:
                    self.copies_of_last_pi_s_o = [deepcopy(self.model.ctxt_components[i])
                                                  for i in range(self.model.num_components)]
                    self.copies_of_last_pi_a_s_o = [deepcopy(self.model.components[i])
                                                    for i in range(self.model.num_components)]
                    self.save_samples_for_val_func_rewards = deepcopy(task_specific_rewards)
                    self.save_samples_for_val_func_ctxts = deepcopy(ctxt_samples)
                    self.save_samples_for_val_func_actions = deepcopy(action_samples)

            gating_rewards = []

            self.get_value_function(samples_changed=False)  # computational bottleneck
            delta_t = 0
            all_ctxt_samples = np.concatenate(ctxt_samples)
            baseline = self.ctxt_v_func_regressor.predict(all_ctxt_samples)

            n_ctxt_samples_p_comp = ctxt_samples[0].shape[0]
            for j in range(self.model.num_components):
                start_time = time.time()
                c_baseline = baseline[j * n_ctxt_samples_p_comp:(j + 1) * n_ctxt_samples_p_comp, :]
                c_reward = rewards[j] - c_baseline
                delta_t += time.time() - start_time
                c_reward = np.mean(c_reward) + self.c.beta * self.model.ctxt_components[j].entropy()
                gating_rewards.append(c_reward)
            if self.verbose:
                logging.info('Nadaraya Watson prediction took :{:5f}'.format(delta_t))

            rewards = gating_rewards

        return rewards, task_specific_rewards, smoothnes_rewards, distance_rewards, collision_rewards, comp_log_resp, \
               ctxt_log_resp

    def get_value_function(self, samples_changed=True):

        max_num_of_train_samples = 10000
        n_of_train_samples = len(self.save_samples_for_val_func_ctxts) * self.save_samples_for_val_func_ctxts[0].shape[0]
        # max_num_of_train_samples = n_of_train_samples
        if n_of_train_samples > max_num_of_train_samples:
            n_samples_per_comp = int(max_num_of_train_samples/len(self.save_samples_for_val_func_ctxts))
            new_samples_for_Val_func_ctxts = []
            new_samples_for_val_func_actions = []
            new_samples_for_val_func_rewards = []

            for i in range(len(self.save_samples_for_val_func_ctxts)):
                index_samples = np.random.randint(0, self.save_samples_for_val_func_ctxts[0].shape[0]-1, n_samples_per_comp)
                new_samples_for_Val_func_ctxts.append(self.save_samples_for_val_func_ctxts[i][index_samples, :])
                new_samples_for_val_func_actions.append(self.save_samples_for_val_func_actions[i][index_samples, :])
                new_samples_for_val_func_rewards.append(self.save_samples_for_val_func_rewards[i][index_samples])
            self.save_samples_for_val_func_ctxts = new_samples_for_Val_func_ctxts
            self.save_samples_for_val_func_actions = new_samples_for_val_func_actions
            self.save_samples_for_val_func_rewards = new_samples_for_val_func_rewards
        if type(self.save_samples_for_val_func_rewards) is list:
            self.save_samples_for_val_func_rewards = np.vstack(self.save_samples_for_val_func_rewards).T

        if self.big_tilde_pi_o_s is None:
            big_tilde_pi_a_s_o = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
            big_tilde_pi_o_s = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
            for j in range(self.model.num_components):
                tilde_pi_s_o = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
                big_tilde_pi_a_s_o[:, j] = self.copies_of_last_pi_a_s_o[j].density(
                    self.save_samples_for_val_func_ctxts[j], self.save_samples_for_val_func_actions[j])
                for i in range(self.model.num_components):
                    tilde_pi_s_o[:, i] = self.copies_of_last_pi_s_o[i].density(self.save_samples_for_val_func_ctxts[j])

                tilde_pi_s = np.sum(tilde_pi_s_o, axis=1) / self.model.num_components
                big_tilde_pi_o_s[:, j] = tilde_pi_s_o[:, j] / (self.model.num_components * tilde_pi_s + 1e-25)
            self.big_tilde_pi_o_s = big_tilde_pi_o_s
            self.big_tilde_pi_a_s_o = big_tilde_pi_a_s_o

        pi_o_s = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
        pi_a_s_o = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
        for j in range(self.model.num_components):
            pi_s_o = np.zeros((self.save_samples_for_val_func_ctxts[0].shape[0], self.model.num_components))
            pi_a_s_o[:, j] = self.model.components[j].density(self.save_samples_for_val_func_ctxts[j],
                                                                              self.save_samples_for_val_func_actions[j])
            for i in range(self.model.num_components):
                pi_s_o[:, i] = self.model.ctxt_components[i].density(self.save_samples_for_val_func_ctxts[j])
            pi_s = np.sum(pi_s_o*self.model.weight_distribution.probabilities[None, :], axis=1)
            pi_o_s[:, j] = (pi_s_o[:, j]*self.model.weight_distribution.probabilities[j])/pi_s

        i_o_weights = pi_o_s/self.big_tilde_pi_o_s

        i_weights = i_o_weights
        i_weights = i_weights.reshape(-1)
        i_weights /= np.sum(i_weights)
        targets = np.concatenate(self.save_samples_for_val_func_rewards.T)

        self.ctxt_v_func_regressor.fit(rewards_train=targets,
                                     samples_train=np.concatenate(self.save_samples_for_val_func_ctxts),
                                     i_weights=i_weights, samples_changed=samples_changed)

        return 0

    def randomly_add_component(self):
        # initial_cp = np.random.normal(size=[self.c.context_dim + 1, self.c.action_dim])
        initial_cmp_params = self.c.cmp_init_mean_params.copy()
        initial_cmp_covars = self.c.cmp_init_cov.copy()

        init_ctxt_cmp_means = np.random.normal(loc=self.c.ctxt_init_mean, scale=self.c.ctxt_init_mean_std,
                                               size=[self.c.ctxt_dim])
        init_ctxt_cmp_covars = self.c.ctxt_init_cov.copy()

        self._add_component(initial_cmp_params, initial_cmp_covars, init_ctxt_cmp_means, init_ctxt_cmp_covars)

    def _add_component(self, init_cmp_params, init_cmp_covar, init_ctxt_cmp_mean, init_ctxt_cmp_covar):

        init_weight = 1e-3
        n_comps = self.learner.add_component(init_cmp_params, init_cmp_covar, init_ctxt_cmp_mean, init_ctxt_cmp_covar,
                                             init_weight)
        # self.model.weight_distribution.set_probabilities(np.ones(n_comps)/n_comps)
        self.model.weight_distribution.probabilities = np.ones(n_comps)/n_comps
        self.c.num_components += 1
        if self.model.num_components != n_comps:
            assert AssertionError("number components missmatch")

        added_it = self.c_iteration
        idx_deleted_list = self.model.num_components - 1
        if self.c.log:
            self.logger.add_component(added_it=added_it, add_type=1, idx_deleted_list= idx_deleted_list)
        self.last_cmp_added_at_i = np.copy(self.c_iteration)

    def clean_up_comps(self):

        weights = self.model.weight_distribution.probabilities
        idx = np.where(weights <= self.c.rm_component_thresh)[0]
        former_num_comps = self.model.num_components

        if len(idx) is not 0:
            self.learner.remove_components(idx)
            self.data_base.remove_components(idx)
            if self.c.log:
                self.logger.remove_components(former_num_comps=former_num_comps, cmp_idx_array=idx)
            self.c.num_components -= idx.shape[0]
            indexes = list(idx)
            for index in sorted(indexes, reverse=True):
                del self.plot_actions[index]
                del self.last_ctxt_cmps_means[index]
                del self.last_ctxt_cmps_covars[index]
            print('deleted components', idx)

    def ask_to_del_last_cmp(self):
        if len(self.last_cmp_task_rewards) == 0 or len(self.last_cmp_task_rewards) == 1:
            return False
        else:
            cmp_mean_task_rewards = np.mean(np.array(self.last_cmp_task_rewards), axis=1)
            avg_cmps = np.mean(cmp_mean_task_rewards[:-1])
            if cmp_mean_task_rewards[-1]<5*avg_cmps:
                self.logger.del_idx_cmp_loc_opt.append(self.model.num_components - 1)
                if self.c.verbose:
                    print('last cmp mean task reward:', cmp_mean_task_rewards[-1])
                    print('avg of cmps:', avg_cmps)
                    print('-> Deleting last cmp')
                return True
            else:
                return False

    def remove_cmp(self, idx):
        self.learner.remove_component(idx)
        self.data_base.remove_component(idx)
        self.c.num_components -= 1
        del self.last_quad_model_for_ctxt_updates[idx]
        del self.last_ctxt_cmps_means[idx]
        del self.last_ctxt_cmps_covars[idx]
        del self.plot_actions[idx]
        del self.plot_loc_ctxt_samples[idx]
        if self.c.log:
            self.logger.remove_component(idx)

    def eval_current_model(self, ctxt_samples):
        log_gating_probs = self.model.log_gating_probs(ctxt_samples)
        samples, comp_idx_samples = self.model.sample(ctxt_samples, gating_probs=np.exp(log_gating_probs))
        num_chosen_cmps = np.zeros(self.model.num_components)
        for i in range(self.model.num_components):
            num_chosen_cmps[i] = np.where(comp_idx_samples == i)[0].shape[0]
        rewards = self.env.step(samples, ctxt_samples)[0]
        # if self.nn_t_rew_regressor.training_input is not None:
        #     print('pred error:', self.nn_t_rew_regressor.pred_error(np.concatenate((ctxt_samples, samples), axis=1), rewards))
        entropies = self.model.expected_entropy(ctxt_samples, log_gating_probs=log_gating_probs)
        expected_entropies = np.mean(entropies)
        self.logger.test_mixture_model_entropy.append(expected_entropies)
        self.logger.test_reward.append(np.mean(rewards))
        self.logger.test_num_comps_chosen.append(num_chosen_cmps)

    def mg_prepare_data_and_log(self, rewards):
        mg_weights_c_e = self.model.weight_distribution.probabilities
        marginal_gating_entropy = self.model.weight_distribution.entropy()
        if Flags.MG_WEIGHTS_i in self.c.log_options:
            self.logger.log(flag=Flags.MG_WEIGHTS_i, val2be_stored=mg_weights_c_e)
        if Flags.MG_ENTROPIES_e in self.c.log_options:
            self.logger.log(flag=Flags.MG_ENTROPIES_e, val2be_stored=marginal_gating_entropy, c_it=self.c_iteration)

        if Flags.CG_WEIGHTS_s_c_i in self.c.log_options:
            cg_weights = self.model.gating_probs(self.static_ctxt_samples)
        for i in range(self.model.num_components):
            if Flags.CG_WEIGHTS_s_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CG_WEIGHTS_s_c_i, val2be_stored=cg_weights[:, i], cmp_idx=i)
            if Flags.MG_WEIGHTS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.MG_WEIGHTS_c_e, val2be_stored=mg_weights_c_e[i], cmp_idx=i,
                            c_it=self.c_iteration)
            if Flags.MG_REWARDS_e_c in self.c.log_options:
                self.logger.log(flag=Flags.MG_REWARDS_e_c, val2be_stored=rewards[i], cmp_idx=i,
                            c_it=self.c_iteration)

    def cc_prepare_data_and_log(self, quad_lin_reg_models, samples_local_ctxts, entropies, rewards, ctxt_log_resps):
        cc_quad_rew_preds_global_list = []
        cc_quad_rew_preds_local_list = []
        cc_densities = self.model.cmp_ctxt_densities(self.static_ctxt_samples)


        # log conditional context information
        if Flags.MC_PROBS_s_i in self.c.log_options:
            mc_probs = self.model.cmp_m_ctxt_densities(self.static_ctxt_samples)[1]
            self.logger.log(flag=Flags.MC_PROBS_s_i, val2be_stored=mc_probs)
        if Flags.MC_VALS_s_i in self.c.log_options:
            self.logger.log(flag=Flags.MC_VALS_s_i, val2be_stored=self.static_ctxt_samples)

        for i in range(self.model.num_components):
            cc_quad_rew_preds_global = quad_lin_reg_models[i].predict(self.static_ctxt_samples).flatten()
            cc_quad_rew_preds_local = quad_lin_reg_models[i].predict(samples_local_ctxts[i]).flatten()
            cc_quad_rew_preds_global_list.append(cc_quad_rew_preds_global)
            cc_quad_rew_preds_local_list.append(cc_quad_rew_preds_local)
            if Flags.CC_PROBS_s_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CC_PROBS_s_c_i, val2be_stored=cc_densities[:, i], cmp_idx=i)
            if Flags.CC_ENTROPIES_e_c in self.c.log_options:
                self.logger.log(flag=Flags.CC_ENTROPIES_e_c, val2be_stored=entropies[i], cmp_idx=i, c_it=self.c_iteration)
            if Flags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c in self.c.log_options:
                self.logger.log(flag=Flags.CC_MEANREWARDS_LOC_C_SAMPLES_e_c, val2be_stored=np.mean(rewards[i]),
                            c_it=self.c_iteration, cmp_idx=i)
            if Flags.CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c in self.c.log_options:
                self.logger.log(flag=Flags.CC_MEANREWARDS_WITH_ENTROPY_BONUS_LOC_SAMPLES_e_c,
                            val2be_stored=np.mean(rewards[i]) + self.c.beta * entropies[i], c_it=self.c_iteration,
                            cmp_idx=i)
            if Flags.CC_REWARDS_LOC_C_SAMPLES_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CC_REWARDS_LOC_C_SAMPLES_c_i, val2be_stored=rewards[i], cmp_idx=i)
            if Flags.CC_LOC_C_SAMPLES_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CC_LOC_C_SAMPLES_c_i, val2be_stored=samples_local_ctxts[i], cmp_idx=i)
            if Flags.CC_QUAD_REW_PREDS_LOCAL_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CC_QUAD_REW_PREDS_LOCAL_c_i, val2be_stored=cc_quad_rew_preds_local, cmp_idx=i)
            if Flags.CC_QUAD_REW_PREDS_GLOBAL_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CC_QUAD_REW_PREDS_GLOBAL_c_i, val2be_stored=cc_quad_rew_preds_global, cmp_idx=i)
            if Flags.CC_MEAN_LOG_RESPS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CC_MEAN_LOG_RESPS_c_e, val2be_stored=np.mean(ctxt_log_resps[i]), cmp_idx=i,
                            c_it=self.c_iteration)

    def co_prepare_data_and_log(self, rewards, entropies, task_specific_rewards, smoothness_rewards, distance_rewards,
                                collision_rewards, comp_log_resps):

        for i in range(len(rewards)):
            r_entr = np.mean(rewards[i] + self.c.alpha * entropies[i])
            r = np.mean(rewards[i])
            t_r = np.mean(task_specific_rewards[i])
            smooth_reward = np.mean(smoothness_rewards[i])
            dist_reward = np.mean(distance_rewards[i])
            coll_reward = np.mean(collision_rewards[i])
            if Flags.CO_POSITIONS_c_i in self.c.log_options:
                self.logger.log(flag=Flags.CO_POSITIONS_c_i, val2be_stored=self.data_base.bg_data[i]['a'], cmp_idx=i)
            if Flags.CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEANREWARDS_WITH_ENTROPY_BONUS_c_e, val2be_stored=r_entr,
                            c_it=self.c_iteration, cmp_idx=i)
            if Flags.CO_MEANREWARDS_WITH_LOG_RESPS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEANREWARDS_WITH_LOG_RESPS_c_e, val2be_stored=r, c_it=self.c_iteration,
                            cmp_idx=i)
            if Flags.CO_MEAN_TASK_REWARDS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEAN_TASK_REWARDS_c_e, val2be_stored=t_r, c_it=self.c_iteration, cmp_idx=i)
            if Flags.CO_MEAN_SMOOTHNESS_REWARDS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEAN_SMOOTHNESS_REWARDS_c_e, val2be_stored=smooth_reward,
                            c_it=self.c_iteration, cmp_idx=i)
            if Flags.CO_MEAN_DISTANCE_REWARDS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEAN_DISTANCE_REWARDS_c_e, val2be_stored=dist_reward, c_it=self.c_iteration,
                            cmp_idx=i)
            if Flags.CO_MEAN_COLLISION_REWARDS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEAN_COLLISION_REWARDS_c_e, val2be_stored=coll_reward, c_it=self.c_iteration,
                            cmp_idx=i)
            if Flags.CO_ENTROPIES_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_ENTROPIES_c_e,
                            val2be_stored=self.model.components[i].expected_entropy(), cmp_idx=i, c_it=self.c_iteration)
            if Flags.CO_MEAN_LOG_RESPS_c_e in self.c.log_options:
                self.logger.log(flag=Flags.CO_MEAN_LOG_RESPS_c_e, val2be_stored=np.mean(comp_log_resps[i]),
                            c_it=self.c_iteration, cmp_idx=i)