from util.ConfigDict import ConfigDict
from util.Logger import Logger
from util.DataBase import DataBase
from regression.NadarayaWatson import NadarayaWatson
from copy import copy

import logging
import numpy as np
import os
import sys
import time


class _CWFormatter(logging.Formatter):
    def __init__(self):
        #self.std_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)

sh = logging.StreamHandler(sys.stdout)
formatter = _CWFormatter()
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[sh])


class SVSL_Base:
    @staticmethod
    def get_default_config():
        c = ConfigDict(
            ############################################################################################################
            # General Configurations
            ############################################################################################################
            exp_name='env',
            action_dim=2,
            ctxt_dim=1,
            train_epochs_p_comp=250,    # after 250 a component is added
            train_epochs_weights=100,   # number iterations to update pi(o) after finishing components training
            num_init_components=1,      # initial number of components (should always be 1)
            rm_component_thresh=1e-6,   # threshold of pi(o) to delete component at the end
            n_samples_buffer=100,       # number of samples in buffer per component
            n_new_samples_comps=10,     # number of samples sampled newly in each component update iteration
            n_new_samples_ctxt=10,      # number of samples sampled newly in each ctxt component update iteration
            n_cmp_addings=10,           # number of components to add
            train_time=0,               # total training time
            fine_tune_every_it=50,      # tune all components
            save_model_every_it = 1000000,
            n_tot_it = 0,
            n_tot_cmp_train_it = 0,
            n_new_samples_ratio=0.1,
            test_every_it=50,
            del_loc_opt=True,
            adv_func_reg_in_Cond_More=False,
            adv_for_gating_update=True,
            ############################################################################################################
            # Hyperparameters
            ############################################################################################################
            alpha=1,
            beta=1,
            context_kl_bound=0.01,
            weight_kl_bound=0.01,
            component_kl_bound=0.01,
            verbose=True,

            ############################################################################################################
            # Initializations
            ############################################################################################################
            ctxt_init_mean_std=0.05,
            ctxt_init_mean=0,
            ctxt_init_cov=1,
            cmp_init_cov=0,
            cmp_init_mean_params=0,

            ############################################################################################################
            # Nadaraya Watson Predictor
            ############################################################################################################
            rbf_scale=0.5,
            ############################################################################################################
            # Logging, Plotting & Saving
            ############################################################################################################
            log=True,
            log_options=[],
            plot_options=[],
            final_save_options=[],
            save_per_it_options=[],

            png_plot_saving=False,
            vis_plots=False,
            png_plot_save_every_it=10000,
            png_plot_vis_every_it=10000000,
            snapshot_data_saving_per_it=False,
            save_data_ever_it=10000,
            adv_plotting = False
        )
        return c

    def __init__(self, config, seed, path=None):
        self.c = config
        self.seed = seed
        self.verbose = self.c.verbose
        self.learner = None
        self.env = None
        self.static_ctxt_samples = None
        self.c.num_components = copy(self.c.num_init_components)

        self.cmps_added = 0
        self.c_iteration = 0
        self.n_added_cmps = 0
        self.last_cmp_added_at_i = 0
        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0
        self.n_tot_cmp_train_it = int(self.c.train_epochs_p_comp*(self.c.n_cmp_addings+self.c.num_init_components))
        self.n_tot_it = int(self.n_tot_cmp_train_it + self.c.train_epochs_weights)
        self.c.n_tot_cmp_train_it = self.n_tot_cmp_train_it
        self.c.n_tot_it = self.n_tot_it

        self.logger = Logger(self.c)
        self.logger.initialize()
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__)) + '/experiments/' + str(self.c.exp_name)
        else:
            path = path
        self.logger.save2path = path
        _, exp_number = self.logger.get_save_path()
        self.logger.save2path_per_it = path + '/' + 'data_p_iteration/' + exp_number
        self.logger.save2path += '/' + exp_number
        if not os.path.isdir(self.logger.save2path):
            os.makedirs(self.logger.save2path)
        if not os.path.isdir(self.logger.save2path_per_it):
            os.makedirs(self.logger.save2path_per_it)
        self.data_base = DataBase(n_samples=self.c.n_samples_buffer, c_dim=self.c.ctxt_dim, action_dim=self.c.action_dim,
                                  bg_comp=None, bg_cond_gating=None)

        self.comp_v_func_regressor = NadarayaWatson(indim=self.c.ctxt_dim, outdim=1, seed=seed,
                                                    kernel_scale=self.c.rbf_scale)
        self.ctxt_v_func_regressor = NadarayaWatson(indim=self.c.ctxt_dim, outdim=1, seed=seed,
                                                    kernel_scale=self.c.rbf_scale)
        # plotting
        save2path_png = path + '/png_plots/' + exp_number
        if not os.path.isdir(save2path_png):
            os.makedirs(save2path_png)

    @property
    def model(self):
        return self.learner.model

    def set_model(self, model):
        self.learner.model = model
    # use only one function for updating and differentiate if only one component or all components are updated in this
    # func
    def update_ctxt_cmp(self, cmp_idx=None):
        raise NotImplementedError

    # use only one function for updating and differentiate if only one component or all components are updated in this
    # func
    def update_cmp(self, cmp_idx=None):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def randomly_add_component(self):
        raise NotImplementedError

    def clean_up_comps(self):
        raise NotImplementedError

    def ask_to_del_last_cmp(self):
        raise NotImplementedError # implements the heuristic for delelting last cmp during runtime

    def remove_cmp(self, idx):
        raise NotImplementedError

    def get_reward(self, context_update, weight_update):
        raise NotImplementedError

    def get_samples(self, cmp_idx, sample_reuse, n_new_samples):
        raise NotImplementedError

    def eval_current_model(self, ctxt_samples):
        raise NotImplementedError

    def save_model(self, it=None):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def train(self, env):
        self.env = env
        self.static_ctxt_samples = self.env.sample_contexts(n_samples=self.c.n_samples_buffer, context_range_bounds=
            self.c.context_range_bounds)
        start_time = time.time()
        for i in range(self.n_tot_it):

            self.c_iteration = i
            if self.verbose:
                logging.info('########## Iteration:{:.0f} ##########'.format(i))

            self.logger.n_ep_samples_executed.append(np.copy(self.n_ep_samples_executed))
            self.logger.n_env_interacts.append(np.copy(self.n_env_interacts))
            self.n_ep_samples_executed = 0
            self.n_env_interacts = 0

            self.train_iter(i)
            if i%self.c.save_model_every_it==0:
                self.save_model(it=i)

        test_context_samples = self.env.sample_contexts(n_samples=self.c.n_samples_buffer, context_range_bounds=
        self.c.context_range_bounds)
        self.eval_current_model(test_context_samples)

        self.c.train_time = (time.time() - start_time) / 3600
        logging.info('################# Training took:{:.3f} ############# '.format(self.c.train_time))
        self.save_model()
        logging.info('saved model final')

    def train_iter(self, i):

        if i % self.c.test_every_it == 0 or i == self.n_tot_it - 1:
            test_start_time = time.time()
            test_context_samples = self.env.sample_contexts(n_samples=self.c.n_samples_buffer, context_range_bounds=
                                                                                            self.c.context_range_bounds)
            self.eval_current_model(test_context_samples)
            if self.verbose:
                logging.info('Testing took {:5f} s '.format(time.time()-test_start_time))

        if i >= (self.cmps_added+1)*self.c.train_epochs_p_comp:
            if self.cmps_added < self.c.n_cmp_addings:
                if self.c.del_loc_opt:
                    if self.ask_to_del_last_cmp():
                        self.remove_cmp(self.model.num_components-1)
                self.randomly_add_component()
                self.cmps_added += 1

        if i % self.c.fine_tune_every_it == 0 or i > self.c.n_tot_cmp_train_it:
            cmp_idx = None  # update all components
        else:
            cmp_idx = self.model.num_components-1 # update last added component

        suffix = '_' + str(cmp_idx) if cmp_idx is not None else 's'

        start_time = time.time()
        res = self.update_ctxt_cmp(cmp_idx)
        if self.verbose:
            ctxt_cmp_update_msg = 'Ctxt component' + suffix
            component_update_msg = 'Component' + suffix
            self.log_results(res, ctxt_cmp_update_msg)
            logging.info(ctxt_cmp_update_msg + " update time: {:5f} ".format(time.time() - start_time))

        if self.model.num_components >= 1:
            start_time = time.time()
            w_res = self.update_weights()
            if self.verbose:
                self.log_results([w_res], 'Weight distr.')
                logging.info('Weight distr. update time:{:5f}'.format(time.time() - start_time))

        start_time = time.time()
        c_res = self.update_cmp(cmp_idx)
        if self.verbose:
            logging.info(component_update_msg + ' update time:{:5f}'.format(time.time() - start_time))
            self.log_results(c_res, component_update_msg)

        if i == self.n_tot_it - 1:
            self.save_model(it=i)
            self.clean_up_comps()

        if self.c.snapshot_data_saving_per_it:
            if i % self.c.save_data_every_it == 0:
                self.logger.save_current_it(flags=self.c.save_per_it_options, it=i)


    def log_results(self, res, key_prefix):
        log_string = "----- UPDATE: " + key_prefix + " -----"
        logging.info(log_string)
        if len(res) == 1:
            if res[0] is None:
                log_string = "No update"
                logging.info(log_string)
            else:
                self.log_string(res[0])
        else:
            for i, cur_res in enumerate(res):
                self.log_string(cur_res, key_prefix + '_' + str(i))
        logging.info('\n')

    def log_string(self, res, key_prefix=None):
        kl, entropy, last_eta, last_omega, add_text, comp_time, entropy_bound, mean_rew = [np.array(x) for x in res]
        if key_prefix is not None:
            log_string = key_prefix + ' :' + "KL: {:.5f}. ".format(kl)
        else:
            log_string = "KL: {:.5f}. ".format(kl)
        log_string += "Entropy: {:.5f} ".format(entropy)
        log_string += "Last Eta: {:.5f} ".format(last_eta)
        log_string += "Last Omega: {:.5f} ".format(last_omega)
        log_string += "Computation Time: {:.5f} ".format(comp_time)
        log_string += "Entropy Bound: {:5f} ".format(entropy_bound)
        log_string += 'Success: ' + str(add_text)
        log_string += ' Mean Reward: {:5f}'.format(mean_rew)
        logging.info(log_string)



