import numpy as np


class LinCondGaussian:

    def __init__(self, params, covar):
        self._context_dim = params.shape[0] - 1
        self._sample_dim = params.shape[1]
        self.update_parameters(params, covar)

    def sample(self, contexts):
        eps = np.random.normal(size=[contexts.shape[0], self._sample_dim])
        return self.means(contexts) + eps @ self._chol_covar.T

    def density(self, contexts, samples):
        return np.exp(self.log_density(contexts, samples))

    def log_density(self, contexts, samples):
        norm_term = self._sample_dim * np.log(2 * np.pi) + self.covar_logdet()
        diff = samples - self.means(contexts)
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        return - 0.5 * (norm_term + exp_term)

    def log_likelihood(self, contexts, samples):
        return np.mean(self.log_density(contexts, samples))

    def kls(self, contexts, other):
        trace_term = np.sum(np.square(other.chol_precision.T @ self._chol_covar))
        kl = other.covar_logdet() - self.covar_logdet() - self._sample_dim + trace_term
        diff = other.means(contexts) - self.means(contexts)
        kl = kl + np.sum(np.square(diff @ other.chol_precision), axis=-1)
        return 0.5 * kl

    def expected_kl(self, contexts, other):
        return np.mean(self.kls(contexts, other))

    def expected_entropy(self):
        return 0.5 * (self._sample_dim * np.log(2 * np.pi * np.e) + self.covar_logdet())

    def covar_logdet(self):
        return 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-125))

    def update_parameters(self, params, covar):
        try:

            chol_covar = np.linalg.cholesky(covar)
            inv_chol_covar = np.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar
            chol_precision = np.linalg.cholesky(precision)

            self._covar = covar
            self._chol_covar = chol_covar
            self._inv_chol_covar = inv_chol_covar
            self._precision = precision
            self._chol_precision = chol_precision
            self._params = params

        except np.linalg.LinAlgError as e:
            print("Linear Conditional Gaussian Paramameter update rejected:", e)

    def means(self, contexts):
        return contexts @ self._params[:-1] + self._params[-1]

    @property
    def params(self):
        return self._params

    @property
    def covar(self):
        return self._covar

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision