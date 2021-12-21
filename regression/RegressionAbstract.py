import numpy as np

class RegressionAbstract:
    def __init__(self, indim, outdim, seed):
        self.indim = indim
        self.outdim = outdim
        self.seed = seed
        self.training_input = None
        self.training_output = None
        self.training_input_mean = None
        self.training_input_std = None
        self.training_output_mean = None
        self.training_output_std = None

    def fit(self, rewards_train, samples_train, samples_changed=True, i_weights=None):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def _preprocess(self, rewards_train, samples_train, samples_changed=True, i_weights=None):
        self.samples_changed = samples_changed
        if len(rewards_train.shape) == 1:
            rewards_train = rewards_train.reshape((-1, 1))
        if len(samples_train.shape) == 1:
            samples_train = samples_train.reshape((-1, 1))
        if i_weights is None:
            i_weights = np.ones((samples_train.shape[0], 1)) / samples_train.shape[0]
        else:
            if len(i_weights.shape) == 1:
                i_weights = i_weights.reshape((-1, 1))
        self.i_weights = i_weights
        self.training_output = rewards_train
        self.training_input = samples_train

    @staticmethod
    def _normalize_feat(feat, feat_mean=None, feat_std=None, i_weights=None):
        if feat_mean is None or feat_std is None:
            if i_weights is None:
                i_weights = np.ones(feat.shape) / feat.shape[0]
            feat_mean = np.sum(feat * i_weights, axis=0)
            diff_feat = feat - feat_mean
            feat_cov = diff_feat.T @ (i_weights * diff_feat)
            feat_std = np.sqrt(np.diag(feat_cov))

        feat_normalized = (feat - feat_mean) / (feat_std+1e-20)
        return feat_normalized

    def _normalize_output(self, output, recompute=False):
        if recompute:
            output_mean = np.mean(output, axis=0)
            output_std = np.std(output, axis=0)
        else:
            output_mean = self.training_output_mean
            output_std = self.training_output_std
        output_normalized = (output - output_mean) / (output_std)
        return output_normalized

    @staticmethod
    def _unnormalize(output_normalized, mean_unnormalized, std_unnormalized):
        return output_normalized * std_unnormalized + mean_unnormalized