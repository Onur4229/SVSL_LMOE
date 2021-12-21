import numpy as np

from copy import deepcopy


class DataBase:

    def __init__(self, n_samples, c_dim, action_dim, bg_comp, bg_cond_gating, look_back_it=5):
        self.look_back_it = look_back_it                # how many iteratio,s we want to look back
        self.n_samples = n_samples
        self.c_dim = c_dim
        self.action_dim = action_dim
        self.comp_data={}        # 'context samples', 'action samples', '(task)reward samples', 'policy data'
                            # 'cond. context policy data'
        self.n_effective_samples_comps_des = 2*(0.5*(c_dim+action_dim+1)*(c_dim+action_dim) + (c_dim+action_dim+1))
        self.n_effective_samples_gating_des = 2*((c_dim+1)*c_dim + (c_dim+1))
        self.bg_data = []   # the whole data of the background model
        self.bg_comp = bg_comp
        self.bg_cond_gating = bg_cond_gating
        self._comp_params_buffer = []       # store here the data from the newly smapled distr
        self._cond_gating_params_buffer = [] # store here the data from the newly sampled distr

        self.bg_comp_tmp = deepcopy(bg_comp)
        self.bg_cond_gating_tmp = deepcopy(bg_cond_gating)

        self._i_weights_comps = None
        self._i_weights_cond_gating = None

    def need_to_add_comp(self, idx_comp):

        return True if (idx_comp > len(self.bg_data)-1) else False

    def add_component(self, comp_params, cond_gat_params, ctxt, actions, rewards, ctxt_weights, action_weights):

        if ctxt.shape[0] < self.n_samples:
            raise ValueError("The number of samples to be stored for a new component have to be", self.n_samples)

        c_comp_data = {'c': ctxt[:self.n_samples, :], 'a': actions[:self.n_samples, :], 'r': rewards[:self.n_samples],
                       'p': comp_params, 'cp': cond_gat_params, 'cw': ctxt_weights, 'aw': action_weights}
        if self._i_weights_comps is None:
            self._i_weights_comps = np.zeros((c_comp_data['c'].shape[0], 1))
            self._i_weights_cond_gating = np.zeros((c_comp_data['c'].shape[0], 1))
        else:
            self._i_weights_comps = np.concatenate((self._i_weights_comps, np.zeros((self._i_weights_comps.shape[0], 1))),
                                                   axis=1)
            self._i_weights_cond_gating = np.concatenate((self._i_weights_cond_gating,
                                                          np.zeros((self._i_weights_cond_gating.shape[0], 1))), axis=1)
        self.bg_data.append(c_comp_data)

    def get_weights_from_last_dist(self, idx):

        comp_params = self.bg_data[idx]['p']
        cond_gating_params = self.bg_data[idx]['cp']
        self.bg_comp.update_parameters(comp_params[0], comp_params[1])     # mean params, covar
        self.bg_cond_gating.update_parameters(cond_gating_params[0], cond_gating_params[1])     # mean params, covar
        w_pi_a_s_o = self.bg_comp.density(contexts=self.bg_data[idx]['c'], samples=self.bg_data[idx]['a'])# comp weights
        w_pi_s_o = self.bg_cond_gating.density(samples=self.bg_data[idx]['c'])      # conditional gating weights
        return w_pi_a_s_o, w_pi_s_o

    def get_normalized_importance_weights(self):
        return self._i_weights_comps, self._i_weights_cond_gating

    def _calc_effective_n_of_samples(self, new_dist_comp, new_dist_cond_gating, idx):

        if self._i_weights_comps is None:
            self._i_weights_comps = np.zeros((self.n_samples, len(self.bg_data)))
            self._i_weights_cond_gating = np.zeros((self.n_samples, len(self.bg_data)))

        comp_samples = self.bg_data[idx]['a']
        ctxt_samples = self.bg_data[idx]['c']

        # old distribution weights (not the newly sampled distribution)
        w_pi_a_s_o_old, w_pi_s_o_old = self.get_weights_from_last_dist(idx)
        w_pi_a_s_o_new = new_dist_comp.density(contexts=ctxt_samples, samples=comp_samples)
        w_pi_s_o_new = new_dist_cond_gating.density(samples=ctxt_samples)

        w_pi_a_s_o_old = self.bg_data[idx]['aw']
        w_pi_s_o_old = self.bg_data[idx]['cw']

        i_weights_comps = w_pi_a_s_o_new/(w_pi_a_s_o_old+1e-25)
        i_weights_gating = w_pi_s_o_new/(w_pi_s_o_old+1e-25)

        # i_weights_comps *= i_weights_gating
        i_weights_comps /= np.sum(i_weights_comps)

        i_weights_gating /= np.sum(i_weights_gating)

        # write into importance weights array which we will need for updating the components and cond gatings
        self._i_weights_comps[:, idx] = i_weights_comps
        self._i_weights_cond_gating[:, idx] = i_weights_gating

        n_eff_samples_comp = np.sum(i_weights_comps*i_weights_gating)**2/np.sum((i_weights_comps*i_weights_gating)**2)
        # n_eff_samples_gating = np.sum(i_weights_gating)**2/np.sum(i_weights_gating**2)
        n_eff_samples_gating = n_eff_samples_comp

        return n_eff_samples_comp, n_eff_samples_gating

    def _update_bg_params(self, comp_params, cond_gat_params, idx):

        self.bg_comp_tmp = deepcopy(self.bg_comp)
        self.bg_cond_gating = deepcopy(self.bg_cond_gating)

        c_comp_dict = self.bg_data[idx]
        c_comp_dict['p'] = comp_params
        c_comp_dict['cp'] = cond_gat_params     # cond gating params
        self.bg_comp.update_parameters(comp_params[0], comp_params[1])     # mean params, covar
        self.bg_cond_gating.update_parameters(cond_gat_params[0], cond_gat_params[1])     # mean params, covar

    def update_samples(self, comp_params, cond_gat_params, ctxt, actions, rewards, ctxt_weights, action_weights, use_imp_weights,
                       idx):
        # store all the new data into the top of the data array and update the component param

        n_new_data = ctxt.shape[0]
        new_comp_dist = deepcopy(self.bg_comp)
        new_cond_gating_dist = deepcopy(self.bg_cond_gating)
        new_comp_dist.update_parameters(comp_params[0], comp_params[1])
        new_cond_gating_dist.update_parameters(cond_gat_params[0], cond_gat_params[1])

        if self.need_to_add_comp(idx):
            self.add_component(comp_params, cond_gat_params, ctxt, actions, rewards, ctxt_weights, action_weights)
            self._update_bg_params(self.bg_data[idx]['p'], self.bg_data[idx]['cp'], idx)
        else:
            c_comp_dict = self.bg_data[idx]

            if n_new_data > c_comp_dict['c'].shape[0]:
                c_comp_dict['c'][:, :] = ctxt[:n_new_data, :]
                c_comp_dict['a'][:, :] = actions[:n_new_data, :]
                c_comp_dict['r'][:, :] = rewards[:n_new_data, :]
                c_comp_dict['cw'][:, :] = ctxt_weights[:n_new_data]
                c_comp_dict['aw'][:, :] = action_weights[:n_new_data]

            else:
                # move all array elements down with number of new samples
                c_comp_dict['c'] = np.roll(c_comp_dict['c'], n_new_data, axis=0)
                c_comp_dict['a'] = np.roll(c_comp_dict['a'], n_new_data, axis=0)
                c_comp_dict['r'] = np.roll(c_comp_dict['r'], n_new_data, axis=0)
                c_comp_dict['cw'] = np.roll(c_comp_dict['cw'], n_new_data, axis=0)
                c_comp_dict['aw'] = np.roll(c_comp_dict['aw'], n_new_data, axis=0)
                # write the new samples to the first n_new_Data indices
                c_comp_dict['c'][:n_new_data, :] = ctxt
                c_comp_dict['a'][:n_new_data, :] = actions
                c_comp_dict['r'][:n_new_data] = rewards
                c_comp_dict['cw'][:n_new_data] = ctxt_weights
                c_comp_dict['aw'][:n_new_data] = action_weights

        if use_imp_weights:
            n_eff_samples_comp, n_eff_samples_gating = self._calc_effective_n_of_samples(new_comp_dist,
                                                                                               new_cond_gating_dist, idx)
            print("")
            print('number of effective samples component' + str(idx), n_eff_samples_comp)
            print('number of effective samples cond.gating' + str(idx), n_eff_samples_gating)
            print("")
            # self._update_bg_params(comp_params, cond_gat_params, idx)
            # return True
            if n_eff_samples_comp < self.n_effective_samples_comps_des:
                update_successfull = False # we reject the update if the number of effective samples does not exceed a threshold
                print('Rejected, resample and add to the pool: number effective component samples is too low')
                return update_successfull

            elif n_eff_samples_gating < self.n_effective_samples_gating_des:
                update_successfull = False # we reject the update if the number of effective samples does not exceed a threshold
                print('Rejected, resample and add to the pool: number effective context samples is too low')
                return update_successfull

            else:
                update_successfull = True
        else:
            # print('No importance weights are considered. Adding samples to data base')
            update_successfull = True

        if update_successfull: # if the number of effective samples is sufficient, update the internal bg distributions
            self._update_bg_params(comp_params, cond_gat_params, idx)
            return update_successfull

    def get_updated_samples(self):
        ctxt_samples = []
        action_samples = []
        for c_dict in self.bg_data:
            ctxt_samples.append(c_dict['c'])
            action_samples.append(c_dict['a'])
        return ctxt_samples, action_samples

    def get_data(self, cmp_idx):
        ctxt = self.bg_data[cmp_idx]['c']
        actions = self.bg_data[cmp_idx]['a']
        rewards = self.bg_data[cmp_idx]['r']
        return ctxt, actions, rewards

    def get_all_data(self):
        ctxts = []
        actions = []
        rewards = []
        for cmp_idx in range(len(self.bg_data)):
            ctxts.append(self.bg_data[cmp_idx]['c'])
            actions.append(self.bg_data[cmp_idx]['a'])
            rewards.append(self.bg_data[cmp_idx]['r'])
        ctxts = np.stack(ctxts, -1).reshape((-1, self.c_dim))
        actions = np.stack(actions, -1).reshape((-1, self.action_dim))
        rewards = np.stack(rewards, -1).reshape((-1, 1))
        return ctxts, actions, rewards

    def remove_component(self, idx):
        del self.bg_data[idx]

    def remove_components(self, idx_array):
        # to remove all components at once
        # create another list with zeros (0=do not remove, 1=remove)
        remove_list = [0]*len(self.bg_data)
        for k in range(idx_array.shape[0]):
            remove_list[idx_array[k]] = 1

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

    def delete_components(self, index_list):
        pass # if we want to delete several components at once

    def draw_samples(self, idx):
        pass #for now we will use all samples in the array. Therefore we will not be able to use many data samples


class DataBaseBgDistr:

    def __init__(self, n_samples, c_dim, action_dim, n_max_samples_stored):
        self.n_samples = n_samples
        self.c_dim = c_dim
        self.action_dim = action_dim
        self.n_max_samples_stored = n_max_samples_stored
        self.max_num_comps = 300                    # tune that one?

        self.data_pool = None                       # store all ctxt, action, reward, comp_idx samples here

        self.higher_hierarchy_model = None
        self.lower_hierarchy_model = None

    def setup_sampling_distr(self, higher_hierarchy_model, lower_hierarchy_model):
        n_comps = len(higher_hierarchy_model.components)

        self.higher_hierarchy_model = deepcopy(higher_hierarchy_model)
        self.higher_hierarchy_model.weight_Distribution._p = np.ones(n_comps)/n_comps

        self.lower_hierarchy_model = deepcopy(lower_hierarchy_model)

    def update_sampling_distribution(self):
        return NotImplementedError

    def update_sample_pool(self):
        return NotImplementedError

    def draw_samples_from_pool(self):
        return NotImplementedError

    def get_dissimilarity2sample_distr(self, model):
        return NotImplementedError

