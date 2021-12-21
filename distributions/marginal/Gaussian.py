import numpy as np


class Gaussian:

    def __init__(self, mean, covar):
        self._dim = mean.shape[-1]
        self.update_parameters(mean, covar)

    def density(self, samples):
        return np.exp(self.log_density(samples))

    def log_density(self, samples):
        norm_term = self._dim * np.log(2 * np.pi) + self.covar_logdet()
        diff = samples - self._mean
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        return -0.5 * (norm_term + exp_term)

    def log_likelihood(self, samples):
        return np.mean(self.log_density(samples))

    def sample(self, num_samples):
        # np.random.seed(0)
        eps = np.random.normal(size=[num_samples, self._dim])
        return self._mean + eps @ self._chol_covar.T

    def entropy(self):
        return 0.5 * (self._dim * np.log(2 * np.pi * np.e) + self.covar_logdet())

    def kl(self, other):
        trace_term = np.sum(np.square(other.chol_precision.T @ self._chol_covar))
        kl = other.covar_logdet() - self.covar_logdet() - self._dim + trace_term
        diff = other.mean - self._mean
        kl = kl + np.sum(np.square(other.chol_precision.T @ diff))
        return 0.5 * kl

    def covar_logdet(self):
        return 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-125))

    def update_parameters(self, mean, covar):
        try:
            chol_covar = np.linalg.cholesky(covar)
            inv_chol_covar = np.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar

            self._chol_precision = np.linalg.cholesky(precision)
            self._mean = mean
            self._lin_term = precision @ mean
            self._covar = covar
            self._precision = precision

            self._chol_covar = chol_covar

        except Exception as e:
            print("Gaussian Paramameter update rejected:", e)
            print("Covar:", covar)
            print("")
            print("mean:", mean)
            print("")
            print("")


    def sample_without_sampling(self, samples, n_samples):
        """
        @param samples: samples from which the new samples should be picked
        @param n_samples: number of samples which should be extracted from samples. Note that we might have a sample more
                          often in the pool
        """
        # calculate importance weights over samples
        is_weights = self.density(samples)
        norm_is_weights = is_weights/(np.sum(is_weights, axis=0)+1e-12)

        thresholds = np.expand_dims(np.cumsum(norm_is_weights), axis=0)
        thresholds[0, -1] = 1.0
        # shapes: thresholds: (1xn_samples)
        #          eps: (n_samples x 1)
        # needed like this, as we want to compare every eps with each threshold value ....
        # np.random.seed(0)
        eps = np.random.uniform(size=[n_samples, 1])
        sample_indices = np.argmax(eps < thresholds, axis=-1)
        return samples[sample_indices, :]


    @property
    def mean(self):
        return self._mean

    @property
    def covar(self):
        return self._covar

    @property
    def lin_term(self):
        return self._lin_term

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision

if __name__ == "__main__":

    p = np.array([0.3, 0.2, 0.1, 0.4])
    gaussian = Gaussian(mean=np.array([0]), covar=np.array([[1]]))
    samples = np.ones((10, 1))
    samples[1, :] = 10
    samples[2, :] = 5
    samples[3, :] = 1
    samples[4, :] = 2
    samples[5, :] = 3
    samples[6, :] = 1
    samples[7, :] = 1.5
    samples[8, :] = 0.5
    samples[9, :] = 10000

    samples = gaussian.sample_without_sampling(samples, 10)
    print(samples)