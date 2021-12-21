import numpy as np

from regression.RegressionAbstract import RegressionAbstract


class NadarayaWatson(RegressionAbstract):
    def __init__(self, indim, outdim, seed, kernel_scale):
        super(NadarayaWatson, self).__init__(indim, outdim, seed)

        self.kernel_scale = kernel_scale
        self.gram_mat = None
        self.samples_changed = True

    def fit(self, rewards_train, samples_train, samples_changed=True, i_weights=None):

        self._preprocess(rewards_train, samples_train, samples_changed, i_weights)

        # if samples_changed or self.training_input_mean is None:
        # might be that i_weights changed! -> recompute!!
        self.training_input_mean = np.sum(self.training_input * self.i_weights, axis=0)
        diff_in = self.training_input - self.training_input_mean
        cov = diff_in.T @ (self.i_weights * diff_in)
        self.training_input_std = np.sqrt(np.diag(cov))

        # output mean and std (no need of importance weights here - already done in prediction step)
        self.training_output_mean = np.mean(self.training_output, axis=0)
        self.training_output_std = np.std(self.training_output, axis=0)

    def predict(self, inputs):

        if len(inputs.shape) == 1:
            inputs = inputs.reshape((-1, 1))
        if len(self.training_output.shape) == 1:
            self.training_output = self.training_output.reshape((-1, 1))

        if self.samples_changed:
            diff_mat = self.training_input[:, None, :] - inputs[None, :, :]           # faster than using np.einsum
            exp_term = np.exp(-np.linalg.norm(diff_mat, axis=2) ** 2 / self.kernel_scale)
            self.gram_mat = exp_term
        else:
            if self.gram_mat is None:
                diff_mat = self.training_input[:, None, :] - inputs[None, :, :]  # faster than using np.einsum
                exp_term = np.exp(-np.linalg.norm(diff_mat, axis=2) ** 2 / self.kernel_scale)
                self.gram_mat = exp_term
        exp_term = self.i_weights * self.gram_mat
        normalizer = np.sum(exp_term, axis=0)
        rew_weighted_kernels = exp_term * self.training_output
        rew_weighted_kernels = np.sum(rew_weighted_kernels, axis=0)
        predictions = rew_weighted_kernels / (normalizer + 1e-20)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape((-1, 1))
        return predictions


if __name__ == "__main__":
    x_train_1 = np.linspace(-3, -1, 50)
    y_train_1 = np.ones(x_train_1.shape) * 2

    x_train_2 = np.linspace(-1, 1, 50)
    y_train_2 = np.ones(x_train_2.shape) * 0

    x_train_3 = np.linspace(1, 3, 50)
    y_train_3 = np.ones(x_train_3.shape) * 2

    x_train = np.concatenate((x_train_1, x_train_2))
    y_train = np.concatenate((y_train_1, y_train_2))
    x_train = np.concatenate((x_train, x_train_3))
    y_train = np.concatenate((y_train, y_train_3))

    x_plot = np.linspace(-4, 5, 200)

    regressor = NadarayaWatson(indim=1, outdim=1, seed=0, kernel_scale=0.001)
    regressor.fit(y_train, x_train)
    import matplotlib.pyplot as plt

    plt.plot(x_train, y_train, 'or')
    plt.plot(x_train, regressor.predict(x_train), 'ob')
    plt.plot(x_plot, regressor.predict((x_plot)), 'og')
    plt.show()