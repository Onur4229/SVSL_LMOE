import numpy as np
import time
from regression.LinRegression import QuadFuncJoint, QuadFunc
from distributions.lin_conditional.LinCondGaussian import LinCondGaussian
from distributions.marginal.Gaussian import Gaussian
from distributions.marginal.Categorical import Categorical


########################################################################################################################
# Linear Conditional Gaussian MORE Update
########################################################################################################################
def lin_cond_more_update(learner, lin_cond_distr, ctxt, samples, rewards, is_weights, kl_bound, entropy_loss_bound,
                         surrogate_reg_fact, eta_offset, omega_offset):
    start_time = time.time()

    old_dist = LinCondGaussian(lin_cond_distr.params, lin_cond_distr.covar)

    sample_mean, sample_chol_cov = compute_joint_mean_and_chol_cov(is_weights, ctxt, lin_cond_distr.params,
                                                                   lin_cond_distr.covar)
    surrogate = QuadFuncJoint(ctxt.shape[1], samples.shape[1], surrogate_reg_fact, True, False)
    surrogate.fit((ctxt, samples), rewards, is_weights, sample_mean, sample_chol_cov)

    learner.eta_offset = eta_offset / surrogate.o_std
    learner.omega_offset = omega_offset / surrogate.o_std
    entropy_bound = entropy_loss_bound

    new_params, new_ocvar = learner.more_step(kl_bound, entropy_bound, old_dist, surrogate, ctxt, is_weights)
    if learner.success:
        lin_cond_distr.update_parameters(new_params, new_ocvar)

    kls = lin_cond_distr.kls(ctxt, old_dist)
    expected_kl = np.sum(is_weights * kls)
    expected_entropy = lin_cond_distr.expected_entropy()
    update_time = time.time() - start_time
    return expected_kl, expected_entropy, learner.last_eta, learner.last_omega, learner.success, update_time

def compute_joint_mean_and_chol_cov(is_weights, ctxts, params, cond_covar):
    # context mean and covariance, weighted by context weights
    is_weights = np.expand_dims(is_weights, -1)
    context_mean = np.sum(is_weights * ctxts, axis=0)
    diff = ctxts - context_mean
    context_covar = diff.T @ (is_weights * diff)
    lin_mat, bias = params[:-1], params[-1]
    joint_mean = np.concatenate([context_mean, bias + context_mean @ lin_mat])
    # See https://scicomp.stackexchange.com/questions/5050/cholesky-factorization-of-block-matrices
    try:
        cc_u = np.linalg.cholesky(context_covar)
    except np.linalg.LinAlgError:
        cc_u = np.linalg.cholesky(context_covar + 1e-6 * np.eye(context_covar.shape[0]))
    cc_s = np.linalg.solve(cc_u, context_covar @ lin_mat).T
    cc_l = np.linalg.cholesky(cond_covar)
    cc = np.concatenate([np.concatenate([cc_u, np.zeros([context_mean.shape[0], bias.shape[0]])], axis=-1),
                         np.concatenate([cc_s, cc_l], axis=-1)], axis=0)
    return joint_mean, cc
########################################################################################################################


########################################################################################################################
# Gaussian MORE Update
########################################################################################################################
def marginal_more_update(learner, marginal_distr, samples, rewards, kl_bound, is_weights, entropy_loss_bound,
                         surrogate_reg_fact, eta_offset, omega_offset, min_var_per_dim, max_var_per_dim):
    start_time = time.time()

    old_dist = Gaussian(marginal_distr.mean, marginal_distr.covar)
    surrogate = QuadFunc(surrogate_reg_fact, normalize=True, unnormalize_output=False)
    data_mean = np.mean(samples, axis=0)
    data_covar = np.cov(samples, rowvar=False)
    if len(data_covar.shape) == 0:
        data_covar = data_covar.reshape((-1, 1))
    try:
        data_chol = np.linalg.cholesky(data_covar)
    except:
        data_chol = np.linalg.cholesky(data_covar + 1e-6 * np.eye(data_covar.shape[0]))
    surrogate.fit(samples, rewards, is_weights, data_mean, data_chol)

    testsur = QuadFunc(surrogate_reg_fact, normalize=True, unnormalize_output=True)
    testsur.fit(samples, rewards, is_weights, data_mean, data_chol)

    preds = testsur
    entropy_bound = entropy_loss_bound

    learner.eta_offset = eta_offset / surrogate.o_std
    learner.omega_offset = omega_offset / surrogate.o_std

    new_mean, new_covar = learner.more_step(kl_bound, entropy_bound, old_dist, surrogate)
    if learner.success:
        # calculate eigenvalues and eigenvectors
        eig_vals_E, eig_vec_E = np.linalg.eig(new_covar)
        eig_vec_E_inv = np.linalg.inv(eig_vec_E)
        idx_min = np.where(eig_vals_E < min_var_per_dim)[0]
        if idx_min.shape[0] != 0:
            eig_vals_E[idx_min] = min_var_per_dim
            # build new covariance which achieves desired std
            new_covar = eig_vec_E @ np.diag(eig_vals_E) @ eig_vec_E_inv
        eig_vals_E, eig_vec_E = np.linalg.eig(new_covar)
        eig_vec_E_inv = np.linalg.inv(eig_vec_E)

        idx_max = np.where(eig_vals_E >= max_var_per_dim)[0]
        if idx_max.shape[0] != 0:
            eig_vals_E[idx_max] = max_var_per_dim
            new_covar = eig_vec_E @ np.diag(eig_vals_E) @ eig_vec_E_inv

        marginal_distr.update_parameters(new_mean, new_covar)
        kl = marginal_distr.kl(old_dist)
        entropy = marginal_distr.entropy()
    else:
        kl = entropy = 0
        new_mean = old_dist.mean
        new_covar = old_dist.covar
        print("Failed optimizing component ")
        print("mean of used samples:", np.mean(samples, axis=0, keepdims=True))
        print("std of used samples:", np.std(samples, axis=0, keepdims=True))
        print("mean of rewards:", np.mean(rewards))
        print("std of rewards:", np.std(rewards))
        print("entropy of old distr:", old_dist.entropy())
        print("covar of old distr:", old_dist.covar)
        # time.sleep(10)
    update_time = time.time() - start_time
    return new_mean, new_covar, kl, entropy, learner.last_eta, learner.last_omega, learner.success, update_time, preds
########################################################################################################################

########################################################################################################################
# REPS Categorical Update
########################################################################################################################
def categorical_reps_update(learner, categorical_distr, rewards, kl_bound, entropy_loss_bound):

    start_time = time.time()

    old_dist = Categorical(categorical_distr.probabilities)
    entropy_bound = entropy_loss_bound

    new_probabilities = learner.reps_step(kl_bound, entropy_bound, old_dist, rewards)
    new_probabilities[np.where(new_probabilities < 1e-20)[0]] = 1e-20

    if learner.success:
        categorical_distr.probabilities = new_probabilities

    kl = categorical_distr.kl(old_dist)
    entropy = categorical_distr.entropy()
    update_time = time.time() - start_time
    return kl, entropy, learner.last_eta, learner.last_omega, learner.success, update_time
########################################################################################################################