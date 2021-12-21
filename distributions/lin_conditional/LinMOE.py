import numpy as np

from distributions.lin_conditional.LinCondGaussian import LinCondGaussian
from distributions.marginal.Gaussian import Gaussian
from distributions.marginal.Categorical import Categorical
from util.functions import log_sum_exp

# Models the mixture of experts policy described in the paper
class LinMOE:

    def __init__(self, cmp_params, cmp_covars, ctxt_cmp_means, ctxt_cmp_covars):
        self._ctxt_dim = cmp_params.shape[1] - 1
        self._sample_dim = cmp_params.shape[2]
        self._weight_distr = Categorical(np.ones(cmp_params.shape[0])/cmp_params.shape[0])

        self._cmps = []
        self._ctxt_cmps = []

        for i in range(cmp_params.shape[0]):
            self._cmps.append(LinCondGaussian(cmp_params[i], cmp_covars[i]))
            self._ctxt_cmps.append(Gaussian(ctxt_cmp_means[i], ctxt_cmp_covars[i]))

    def sample(self, ctxts, gating_probs=None):
        if gating_probs is None:
            gating_probs = self.gating_probs(ctxts)
        thresh = np.cumsum(gating_probs, axis=1)
        thresh[:, -1] = np.ones(ctxts.shape[0])
        eps = np.random.uniform(size=[ctxts.shape[0], 1])
        comp_idx_samples = np.argmax(eps < thresh, axis=-1)
        samples = np.zeros((ctxts.shape[0], self._sample_dim))
        for i in range(self.num_components):
            ctxt_samples_cmp_i_idx = np.where(comp_idx_samples == i)[0]
            ctxt_samples_cmp_i = ctxts[ctxt_samples_cmp_i_idx, :]
            if ctxt_samples_cmp_i.shape[0] != 0:
                samples[ctxt_samples_cmp_i_idx, :] = self._cmps[i].sample(ctxt_samples_cmp_i)
        return samples, comp_idx_samples

    def expected_entropy(self, ctxts, log_gating_probs=None):
        n_comps = self.num_components
        if log_gating_probs is None:
            log_gating_probs = self.log_gating_probs(ctxts)
        action_expectation = np.zeros((ctxts.shape[0], n_comps))
        for j in range(ctxts.shape[0]):
            for k in range(n_comps):
                tmp_ctxt = np.repeat(ctxts[j].reshape((1, -1)), 10, axis=0)
                action_samples = self.components[k].sample(tmp_ctxt)
                log_arg = self.log_cmp_densities(tmp_ctxt, action_samples)
                log_arg += log_gating_probs[j, :][None, :]
                max_arg = np.max(log_arg)
                log_arg = max_arg + np.log(np.sum(np.exp(log_arg - max_arg), axis=1) + 1e-200)
                action_expectation[j, k] = np.mean(log_arg).copy()
        entropies = -np.sum(np.exp(log_gating_probs) * action_expectation, axis=1)
        return entropies

    def joint_entropy(self, num_samples=100):
        n_comps = self.num_components
        joint_entropy = 0
        for k in range(n_comps):
            c_ctxt_cmp = self.ctxt_components[k]
            c_ctxts = c_ctxt_cmp.sample(num_samples=num_samples)
            #pi(a|s)
            cond_entropy = self.expected_entropy(c_ctxts)
            # log pi(s|o)
            log_pi_s_o_dens = self.log_cmp_ctxt_densities(c_ctxts)
            # exp(log(pi(s|o) + log(pi(o))
            exp_arg = log_pi_s_o_dens + self.weight_distribution.log_probabilities.reshape((1, -1))
            log_part = log_sum_exp(exp_arg, axis=1)
            joint_entropy += self.weight_distribution.probabilities[k] * np.mean(cond_entropy - log_part)
        return joint_entropy

    def marginal_ctxt_entropy(self, num_samples=100):
        n_comps = self.num_components
        marginal_entropy = 0
        for k in range(n_comps):
            c_ctxt_cmp = self.ctxt_components[k]
            c_ctxts = c_ctxt_cmp.sample(num_samples=num_samples)
            log_pi_s_o_dens = self.log_cmp_ctxt_densities(c_ctxts)
            exp_arg = log_pi_s_o_dens + self.weight_distribution.log_probabilities.reshape((1, -1))
            log_part = log_sum_exp(exp_arg, axis=1)
            marginal_entropy += self.weight_distribution.probabilities[k] * np.mean(log_part)
        return -marginal_entropy

    # pi(a|s) = log(sum_o pi(a|s,o) pi(o|s))
    def log_density(self, ctxts, samples):
        log_cmp_densities = self.log_cmp_densities(ctxts, samples)
        log_gating_probs = self.log_gating_probs(ctxts)
        exp_arg = log_cmp_densities + log_gating_probs
        log_density = log_sum_exp(exp_arg, axis=1)
        return log_density

    # sum_o pi(a|s,o) pi(o|s)
    def density(self, ctxts, samples):
        return np.exp(self.log_density(ctxts, samples))

    def log_cmp_densities(self, ctxts, samples):
        n_comps = self.num_components
        log_probs = np.zeros((ctxts.shape[0], n_comps))
        for i in range(n_comps):
            log_probs[:, i] = self._cmps[i].log_density(ctxts, samples)
        return log_probs

    def cmp_densities(self, ctxts, samples):
        return np.exp(self.log_cmp_densities(ctxts, samples))

    def log_cmp_ctxt_densities(self, ctxts):
        n_comps = self.num_components
        log_probs = np.zeros((ctxts.shape[0], n_comps))
        for i in range(n_comps):
            log_probs[:, i] = self._ctxt_cmps[i].log_density(ctxts)
        return log_probs

    # pi(s|o)
    def cmp_ctxt_densities(self, ctxts):
        return self.log_cmp_ctxt_densities(ctxts)

    # log pi(s|o) and pi(s)
    # returning both to avoid redundant calculations
    def log_cmp_m_ctxt_densities(self, ctxts):
        log_weights = self._weight_distr.log_probabilities
        log_cmp_ctxt_densities = self.log_cmp_ctxt_densities(ctxts)
        # log sum exp
        exp_arg = log_cmp_ctxt_densities + log_weights[None, :]
        log_marg_ctxt_densities = log_sum_exp(exp_arg, axis=1)
        return log_cmp_ctxt_densities, log_marg_ctxt_densities

    # pi(s|o) and pi(s)
    # returning both to avoid redundant calculations
    def cmp_m_ctxt_densities(self, ctxts):
        log_cmp_ctxt_densities, log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(ctxts)
        return np.exp(log_cmp_ctxt_densities), np.exp(log_marg_ctxt_densities)

    # log pi(o|s)
    def log_gating_probs(self, ctxts):
        log_weights = self._weight_distr.log_probabilities
        log_cmp_ctxt_densities, log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(ctxts)
        log_gating_probs = log_cmp_ctxt_densities + log_weights[None, :] - log_marg_ctxt_densities[:, None]
        return log_gating_probs

    # pi(o|s)
    def gating_probs(self, ctxts):
        return np.exp(self.log_gating_probs(ctxts))

    # pi(o|a,s), pi(o|s)
    # return both because pi(o|s) is automatically calculated
    def log_responsibilities(self, ctxts, samples):
        log_weights = self._weight_distr.log_probabilities
        log_cmp_ctxt_densities, log_marg_ctxt_densities = self.log_cmp_m_ctxt_densities(ctxts)
        log_gating_probs = log_cmp_ctxt_densities + log_weights[None, :] - log_marg_ctxt_densities[:, None]
        log_cmp_densities = self.log_cmp_densities(ctxts, samples)
        log_model_density = log_sum_exp(log_cmp_densities+log_gating_probs, axis=1)
        log_resps = log_cmp_densities + log_gating_probs - log_model_density[:, None]
        return log_resps, log_gating_probs

    def get_means(self, ctxts, cmp_idx):
        means = np.zeros((cmp_idx.shape[0], self._sample_dim))
        for i in range(cmp_idx.shape[0]):
            means[i] = self.components[cmp_idx[i]].means(ctxts[i].reshape((1, -1)))
        return means

    @property
    def components(self):
        return self._cmps

    @property
    def ctxt_components(self):
        return self._ctxt_cmps

    @property
    def weight_distribution(self):
        return self._weight_distr

    @property
    def num_components(self):
        return len(self._cmps)

    @property
    def ctxt_dim(self):
        return self._ctxt_dim

    def add_component(self, cmp_params, cmp_covar, ctxt_cmp_mean, ctxt_cmp_covar, init_weight):
        self._cmps.append(LinCondGaussian(cmp_params, cmp_covar))
        self._ctxt_cmps.append(Gaussian(ctxt_cmp_mean, ctxt_cmp_covar))
        self._weight_distr.add_entry(init_weight)

    def remove_component(self, idx):
        del self._cmps[idx]
        del self._ctxt_cmps[idx]
        self._weight_distr.remove_entry(idx)

if __name__ == "__main__":
    np.random.seed(0)
    n_comps = 10
    action_dim = 30
    ctxt_dim = 20
    cmp_params = np.zeros((n_comps, ctxt_dim+1, action_dim))

    for i in range(n_comps):
        cmp_params[i] = np.random.normal(0, 1, size=(ctxt_dim+1, action_dim))
    init_cmp_cvr = np.eye(action_dim)*5
    cmp_covars = np.tile(np.expand_dims(init_cmp_cvr, 0), [n_comps, 1, 1])

    ctxt_cmp_means = np.random.normal(0, 2, size=(n_comps, ctxt_dim))
    init_ctxt_cmp_cvr = np.eye(ctxt_dim)*5
    ctxt_cmp_covars = np.tile(np.expand_dims(init_ctxt_cmp_cvr, 0), [n_comps, 1, 1])

    linmoe = LinMOE(cmp_params, cmp_covars, ctxt_cmp_means, ctxt_cmp_covars)
    ctxts = np.random.normal(0, 5, size=(100, ctxt_dim))
    samples = linmoe.sample(ctxts)
    log_po_s_o_log_pi_s = linmoe.log_cmp_m_ctxt_densities(ctxts)
    log_resps, log_gating_probs = linmoe.log_responsibilities(ctxts, samples=samples)