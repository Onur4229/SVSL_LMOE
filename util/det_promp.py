########################################################################################################################
# This file is from https://github.com/psclklnk/self-paced-rl/blob/master/sprl/util/det_promp.py
# and slightly adapted to our case
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


class DeterministicProMP:

    def __init__(self, n_basis, width=None, off=0.2):
        self.n_basis = n_basis
        self.centers = np.linspace(-off, 1. + off, n_basis)
        if width is None:
            self.widths = np.ones(n_basis) * ((1. + off) / (2. * n_basis))
        else:
            self.widths = np.ones(n_basis) * width
        self.scale = None
        self.weights = None

    def _exponential_kernel(self, z):
        z_ext = z[:, None]
        diffs = z_ext - self.centers[None, :]
        w = np.exp(-(np.square(diffs) / (2 * self.widths[None, :])))
        w_der = -(diffs / self.widths[None, :]) * w
        w_der2 = -(1 / self.widths[None, :]) * w + np.square(diffs / self.widths[None, :]) * w
        sum_w = np.sum(w, axis=1)[:, None]
        sum_w_der = np.sum(w_der, axis=1)[:, None]
        sum_w_der2 = np.sum(w_der2, axis=1)[:, None]

        tmp = w_der * sum_w - w * sum_w_der
        return w / sum_w, tmp / np.square(sum_w), \
               ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / np.power(sum_w, 3)

    def learn(self, t, pos, lmbd=1e-6):
        self.scale = np.max(t)
        # We normalize the timesteps to be in the interval [0, 1]
        features = self._exponential_kernel(t / self.scale)[0]
        # get weights with ridge regression
        self.weights = np.linalg.solve(np.dot(features.T, features) + lmbd * np.eye(features.shape[1]),
                                       np.dot(features.T, pos))

    def compute_trajectory(self, frequency, scale=1):
        corrected_scale = self.scale / scale
        N = int(corrected_scale * frequency)
        t = np.linspace(0, 1, N)
        pos_features, vel_features, acc_features = self._exponential_kernel(t)
        return t * corrected_scale, np.dot(pos_features, self.weights), \
               np.dot(vel_features, self.weights) / corrected_scale, \
               np.dot(acc_features, self.weights) / np.square(corrected_scale)

    def get_weights(self):
        return np.copy(self.weights)

    def set_weights(self, scale, weights):
        self.scale = scale
        self.weights = weights

    @staticmethod
    def shape_weights(theta, promp_time):
        if promp_time:
            if theta.shape[0] - 1 == 21:
                weights = np.reshape(theta[:-1], (-1, 7))  # 7 joints
            elif theta.shape[0] - 1 == 14:
                weights = np.reshape(theta[:-1], (-1, 7))
            elif theta.shape[0] - 1 == 7:
                weights = np.reshape(theta[:-1], (-1, 7))
            elif theta.shape[0] - 1 == 18:
                weights = np.reshape(theta[:-1], (-1, 6))  # 6 joints (last missing)
            elif theta.shape[0] - 1 == 12:
                weights = np.reshape(theta[:-1], (-1, 6))  # 6 joints (last missing)
            elif theta.shape[0] - 1 == 6:
                weights = np.reshape(theta[:-1], (-1, 3))  # 3 joints
            else:
                weights = np.reshape(theta[:-1], (-1, 3))  # 3 joints
        else:
            if theta.shape[0] == 21:
                weights = np.reshape(theta, (-1, 7))  # 7 joints
            elif theta.shape[0] == 14:
                weights = np.reshape(theta, (-1, 7))
            elif theta.shape[0]  == 7:
                weights = np.reshape(theta, (-1, 7))
            elif theta.shape[0]  == 18:
                weights = np.reshape(theta, (-1, 6))
            elif theta.shape[0]  == 12:
                weights = np.reshape(theta, (-1, 6))
            elif theta.shape[0] == 6:
                weights = np.reshape(theta, (-1, 3))
            else:
                weights = np.reshape(theta, (-1, 3))  # 3 joints
        return weights

    def visualize(self, frequency, scale=1):
        corrected_scale = self.scale / scale
        N = int(corrected_scale * frequency)
        t = np.linspace(0, 1, N)
        pos_features, __, __ = self._exponential_kernel(t)
        plt.plot(t, pos_features)
        plt.show()
        plt.figure()
        # pos_feat_weights = pos_features*
        plt.subplot(7,1,1)
        pos_feat_weights = pos_features*self.weights[:, 0].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,2)
        pos_feat_weights = pos_features*self.weights[:, 1].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,3)
        pos_feat_weights = pos_features*self.weights[:, 2].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,4)
        pos_feat_weights = pos_features*self.weights[:, 3].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,5)
        pos_feat_weights = pos_features*self.weights[:, 4].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,6)
        pos_feat_weights = pos_features*self.weights[:, 5].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.subplot(7,1,7)
        pos_feat_weights = pos_features*self.weights[:, 6].reshape((1, -1))
        plt.plot(t, pos_feat_weights)

        plt.show()

if __name__ == "__main__":
    # pmp = DeterministicProMP(n_basis=3 + 4, width=0.0035, off=0.01)
    # pmp = DeterministicProMP(n_basis=5 + 4, width=0.0035, off=0.01)
    # pmp = DeterministicProMP(n_basis=1+ 4, width=0.02, off=0.008)
    # pmp = DeterministicProMP(n_basis=1+ 4, width=0.02, off=0.005)
    pmp = DeterministicProMP(n_basis=1+ 2, width=0.03, off=0.005)
    pmp.set_weights(scale=3.5, weights=None)
    pmp.visualize(1750)

