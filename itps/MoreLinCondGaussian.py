import numpy as np
from itps.ITPS import ITPS


class MoreLinCondGaussian(ITPS):

    def __init__(self, context_dim, sample_dim, eta_offset, omega_offset, constrain_entropy):
        super().__init__(eta_offset, omega_offset, constrain_entropy)
        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._ld_const = self._sample_dim * np.log(2 * np.pi)


    def more_step(self, eps, beta, old_dist, reward_surrogate, contexts, is_weights):
        self._eps = eps
        self._beta = beta
        self._contexts = contexts
        self._is_weights = is_weights

        self._old_precision = old_dist.precision
        self._old_chol_precision = old_dist.chol_precision
        self._old_means = old_dist.means(self._contexts)

        old_params = old_dist.params
        self._W = old_params[:-1]
        self._w = old_params[-1]

        self._reward_quad_ss = reward_surrogate.quad_term_yy
        self._reward_quad_sc = reward_surrogate.quad_term_yx
        self._reward_lin_s = reward_surrogate.lin_term_y

        temp = self._W @ self._old_precision
        self._M_const = temp @ self._W.T
        self._m_const = temp @ self._w

        old_logdet = old_dist.covar_logdet()
        self._m_0_const = np.sum((self._old_precision @ self._w) * self._w) + old_logdet

        self._kl_const_part = old_logdet - self._sample_dim
        self._entropy_const_part = 0.5 * self._sample_dim * np.log(2 * np.pi * np.e)

        try:
            self.ftol_rel = 1e-8
            opt_eta, opt_omega = self.opt_dual()
            new_L, new_l, new_precision = self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            new_covar = np.linalg.inv(new_precision)
            new_params = np.concatenate([new_L, np.expand_dims(new_l, 0)], axis=0) @ new_covar
            self._succ = True
            return new_params, new_covar
        except Exception:
            self._succ = False
            return None, None

    def _new_params(self, eta, omega):
        new_L = (eta * self._W @ self._old_precision + self._reward_quad_sc.T) / (eta + omega)
        new_l = (eta * self._w @ self._old_precision + self._reward_lin_s) / (eta + omega)
        new_quad = (eta * self._old_precision + self._reward_quad_ss) / (eta + omega)
        return new_L, new_l, new_quad

    def _dual(self, eta_omega, grad):
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if self._constrain_entropy and eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset
        new_L, new_l, new_quad = self._new_params(eta_off, omega_off)
        try:
            new_covar = np.linalg.inv(new_quad)
            new_chol_covar = np.linalg.cholesky(new_covar)
            new_logdet = 2 * np.sum(np.log(np.diagonal(new_chol_covar)))

            temp = new_L @ new_covar
            M = 0.5 * ((eta_off + omega_off) * temp @ new_L.T - eta_off * self._M_const)
            m = (eta_off + omega_off) * temp @ new_l - eta_off * self._m_const
            lcl = np.sum((new_covar @ new_l) * new_l)
            m_0 = (eta_off + omega_off) * (lcl + new_logdet) + omega_off * self._ld_const
            m_0 = 0.5 * (m_0 - eta_off * self._m_0_const)

            q = np.sum((self._contexts @ M) * self._contexts, 1)
            l = self._contexts @ m
            int_term = np.sum(self._is_weights * (q + l + m_0))  # importance weights, p(x | z) / ( n * p(x))
            dual = eta * self._eps - omega * self._beta + int_term

            new_p = np.concatenate([new_L, np.expand_dims(new_l, 0)], axis=0) @ new_covar
            kl_cc_term = self._kl_const_part - new_logdet + np.sum(np.square(self._old_chol_precision.T @ new_chol_covar))
            diff = self._old_means - (self._contexts @ new_p[:-1] + new_p[-1])
            sample_wise_kl = 0.5 * (kl_cc_term + np.sum(np.square(diff @ self._old_chol_precision), axis=-1))
            kl = np.sum(self._is_weights * sample_wise_kl)

            entropy = self._entropy_const_part + 0.5 * new_logdet

            self.last_kl = kl.copy()
            self.last_entropy = entropy.copy()
            grad[0] = self._eps - kl
            grad[1] = entropy - self._beta if self._constrain_entropy else 0.0
            grad[0] = np.clip(grad[0], -1e8, 1e8)
            grad[1] = np.clip(grad[1], -1e8, 1e8)
            self._grad = grad
            dual = np.clip(dual, -1e11, 1e11)
            return dual
        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            grad[1] = 0.0
            self._grad = grad
            return 1e12

