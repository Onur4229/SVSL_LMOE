from distributions.lin_conditional.LinMOE import LinMOE
from itps.MoreLinCondGaussian import MoreLinCondGaussian
from itps.MoreGaussian import MoreGaussian
from itps.RepsCategorical import RepsCategorical
from model_learner.updates import lin_cond_more_update, marginal_more_update, categorical_reps_update


class LinMOELearner:
    def __init__(self, ctxt_dim, sample_dim, surrogate_reg_fact, eta_offset_weight, omega_offset_weight, eta_offset_ctxt,
                 omega_offset_ctxt, eta_offset_cmp, omega_offset_cmp, constrain_entropy):

        ################################################################################################################
        # General
        ################################################################################################################
        self._ctxt_dim = ctxt_dim
        self._sample_dim = sample_dim
        self._surrogate_reg_fact = surrogate_reg_fact
        self._eta_offset_weight = eta_offset_weight
        self._omega_offset_weight = omega_offset_weight
        self._eta_offset_ctxt = eta_offset_ctxt
        self._omega_offset_ctxt = omega_offset_ctxt
        self._eta_offset_cmp = eta_offset_cmp
        self._omega_offset_cmp = omega_offset_cmp
        self._constrain_entropy = constrain_entropy

        self._ctxt_cmp_learners = []        # we use several lerners, for potential parallelization in future...
        self._cmp_learners = []             # we use several lerners, for potential parallelization in future...
        self._weight_learner = None
        self._model = None

        ################################################################################################################
        # Some bounds for component-wise context optimization for numerical stability
        ################################################################################################################
        # self._min_std_per_dim = 0.005
        self._min_std_per_dim = 0.00005
        self._min_var_per_dim = self._min_std_per_dim ** 2
        self._max_std_per_dim = 3
        self._max_var_per_dim = self._max_std_per_dim ** 2
        ################################################################################################################

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def initialize_model(self, cmp_params, cmp_covars, ctxt_cmp_means, ctxt_cmp_covars):
        self._model = LinMOE(cmp_params, cmp_covars, ctxt_cmp_means, ctxt_cmp_covars)

        for i in range(self._model.num_components):
            self._cmp_learners.append(MoreLinCondGaussian(self._ctxt_dim, self._sample_dim, self._eta_offset_cmp,
                                    self._omega_offset_cmp, self._constrain_entropy))
            self._ctxt_cmp_learners.append(MoreGaussian(self._ctxt_dim, self._eta_offset_ctxt, self._omega_offset_ctxt,
                                                         self._constrain_entropy))


        self._weight_learner =  RepsCategorical(self._eta_offset_weight, self._omega_offset_weight, self._constrain_entropy)

    def update_prior_weights(self, rewards, kl_bound, entropy_loss_bound):
        kl, entropy, last_eta, last_omega, success, update_time = \
        categorical_reps_update(learner=self._weight_learner,
                                categorical_distr=self._model.weight_distribution,
                                rewards=rewards,
                                kl_bound=kl_bound,
                                entropy_loss_bound=entropy_loss_bound)
        return kl, entropy, last_eta, last_omega, success, update_time

    def update_ctxt_cmp(self, samples, rewards, kl_bound, is_weights, entropy_loss_bound, cmp_idx):
        new_mean, new_covar, kl, entropy, last_eta, last_omega, success, update_time, preds = \
            marginal_more_update(learner=self._ctxt_cmp_learners[cmp_idx],
                                 marginal_distr=self.model.ctxt_components[cmp_idx],
                                 samples=samples,
                                 rewards=rewards,
                                 kl_bound=kl_bound,
                                 is_weights=is_weights,
                                 entropy_loss_bound=entropy_loss_bound,
                                 surrogate_reg_fact=self._surrogate_reg_fact,
                                 eta_offset=self._eta_offset_ctxt,
                                 omega_offset=self._omega_offset_ctxt,
                                 min_var_per_dim=self._min_var_per_dim,
                                 max_var_per_dim=self._max_var_per_dim)
        return new_mean, new_covar, kl, entropy, last_eta, last_omega, success, update_time, preds

    def update_all_ctxt_cmps(self, samples, rewards, kl_bound, is_weights, entropy_loss_bound):
        res_list = []
        for i in range(self._model.num_components):
            if is_weights is not None:
                is_weight = is_weights[:, i]
            else:
                is_weight = None

            new_mean, new_covar, kl, entropy, last_eta, last_omega, success, update_time, preds = \
                self.update_ctxt_cmp(samples=samples[i],
                                     rewards=rewards[i],
                                     kl_bound=kl_bound,
                                     is_weights=is_weight,
                                     entropy_loss_bound=entropy_loss_bound[i],
                                     cmp_idx=i)
            res_list.append((new_mean, new_covar, kl, entropy, last_eta, last_omega, success, update_time, preds))
        return res_list

    def update_cmp(self, ctxt, samples, rewards, is_weights, kl_bound, entropy_loss_bound, cmp_idx):
        expected_kl, expected_entropy, last_eta, last_omega, success, update_time = \
            lin_cond_more_update(learner=self._cmp_learners[cmp_idx],
                                 lin_cond_distr=self._model.components[cmp_idx],
                                 ctxt=ctxt,
                                 samples=samples,
                                 rewards=rewards,
                                 is_weights=is_weights,
                                 kl_bound=kl_bound,
                                 entropy_loss_bound=entropy_loss_bound,
                                 surrogate_reg_fact=self._surrogate_reg_fact,
                                 eta_offset=self._eta_offset_cmp,
                                 omega_offset=self._omega_offset_cmp)
        return expected_kl, expected_entropy, last_eta, last_omega, success, update_time

    def update_all_cmps(self, ctxts, samples, rewards, is_weights, kl_bound, entropy_loss_bound):
        res_vec = []
        for i, c in enumerate(self._model.components):
            expected_kl, expected_entropy, last_eta, last_omega, success, update_time = \
                self.update_cmp(ctxts[i], samples[i], rewards[i], is_weights[:, i], kl_bound, entropy_loss_bound[i], i)
            res_vec.append( (expected_kl, expected_entropy, last_eta, last_omega, success, update_time))
        return res_vec

    def add_component(self, init_cmp_params, init_cmp_covar, init_ctxt_cmp_mean, init_ctxt_cmp_covar,
                      init_weight):
        # add to model
        self._model.add_component(cmp_params=init_cmp_params,
                                  cmp_covar=init_cmp_covar,
                                  ctxt_cmp_mean=init_ctxt_cmp_mean,
                                  ctxt_cmp_covar=init_ctxt_cmp_covar,
                                  init_weight=init_weight)
        # add to the learners
        self._cmp_learners.append(MoreLinCondGaussian(self._ctxt_dim, self._sample_dim, self._eta_offset_cmp,
                                                      self._omega_offset_cmp, self._constrain_entropy))
        self._ctxt_cmp_learners.append(MoreGaussian(self._ctxt_dim, self._eta_offset_ctxt, self._omega_offset_ctxt,
                                                    self._constrain_entropy))

        return self._model.num_components

    def remove_component(self, idx):
        self._model.remove_component(idx)
        # self.higher_hierarchy_model.remove_component(idx)
        del self._cmp_learners[idx]
        del self._ctxt_cmp_learners[idx]
        return self._model.num_components

    def remove_components(self, idx_array):
        # to remove all components at once
        # create another (mask) list with zeros (0=do not remove, 1=remove)
        remove_list = [0]*self._model.num_components
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

