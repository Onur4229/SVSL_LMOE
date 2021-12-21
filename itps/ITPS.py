import numpy as np
import nlopt


class ITPS:
    # grad_bound = 1e-5
    grad_bound = 1e-3
    value_bound = 1e-8

    def __init__(self, eta_offset, omega_offset, constrain_entropy):
        self._constrain_entropy = constrain_entropy
        self._eta_offset = eta_offset
        self._omega_offset = omega_offset

        self._eta = None
        self._omega = None
        self._grad = None
        self._succ = False
        self._fig_soft = None
        self._rt_obj = None
        self.ftol_rel = None
        self._eps = None
        self._beta = None
        self.last_kl = None
        self.last_entropy = None

    def opt_dual(self):
        opt = nlopt.opt(nlopt.LD_LBFGS, 2)
        # opt.set_lower_bounds(0.0)
        opt.set_lower_bounds(1e-8)
        opt.set_upper_bounds(1e8)
        if self.ftol_rel is not None:
            opt.set_ftol_rel(self.ftol_rel)
        opt.set_min_objective(self._dual)
        try:
            opt_eta_omega = opt.optimize([40.0, 40.0])
            opt_eta = opt_eta_omega[0]
            opt_omega = opt_eta_omega[1] if self._constrain_entropy else 0.0
            return opt_eta, opt_omega
        except Exception as e:
            if (np.sqrt(self._grad[0] ** 2 + self._grad[1] ** 2) < ITPS.grad_bound) or \
                    (self._eta < ITPS.value_bound and np.abs(self._grad[1]) < ITPS.grad_bound) or \
                    (self._omega < ITPS.value_bound and np.abs(self._grad[0]) < ITPS.grad_bound):
                return self._eta, self._omega
            else:
                raise e

    def _dual(self, eta_omega, grad):
        raise NotImplementedError

    @property
    def last_eta(self):
        return self._eta

    @property
    def last_omega(self):
        return self._omega

    @property
    def last_grad(self):
        return self._grad

    @property
    def success(self):
        return self._succ

    @property
    def eta_offset(self):
        return self._eta_offset

    @eta_offset.setter
    def eta_offset(self, new_eta_offset):
        self._eta_offset = new_eta_offset

    @property
    def omega_offset(self):
        return self._omega_offset

    @omega_offset.setter
    def omega_offset(self, new_omega_offset):
        self._omega_offset = new_omega_offset
