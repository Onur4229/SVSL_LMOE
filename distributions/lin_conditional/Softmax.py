import numpy as np
from util.LogisticRegression import fit_softmax


class Softmax:

    def __init__(self, params):
        self._params = params

    def sample(self, contexts):
        thresholds = np.cumsum(self.probabilities(contexts), axis=-1)
        thresholds[:, -1] = 1.0
        eps = np.random.uniform(size=[contexts.shape[0], 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    def sample_without_sampling(self, contexts):
        return contexts[self.sample(contexts), :]

    def probabilities(self, contexts):
        return np.exp(self.log_probabilities(contexts))

    def log_probabilities(self, contexts):
        logits = self.calc_logits(contexts)
        # logits = self.affine_mapping(contexts, self._params)
        max_logits = np.max(logits, axis=-1, keepdims=True)
        return logits - (max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True)))

    def entropies(self, contexts):
        p = self.probabilities(contexts)
        return - np.sum(p * np.log(p + 1e-125), axis=-1)

    def expected_entropy(self, contexts):
        return np.mean(self.entropies(contexts))

    def kls(self, contexts, other):
        p = self.probabilities(contexts)
        other_log_p = other.log_probabilities(contexts)
        return np.sum(p * (np.log(p + 1e-125) - other_log_p), axis=-1)

    def expected_kl(self, contexts, other):
        return np.mean(self.kls(contexts, other))

    def calc_logits(self, contexts):
        return self.affine_mapping(contexts, self._params)

    @staticmethod
    def affine_mapping(x, p):
        return x @ p[:-1] + p[-1]

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        ##TODO: Substract for stability?
        self._params = new_params

    def add_entry(self, contexts, initial_probs):
        if len(initial_probs.shape) == 1:
            initial_probs = np.expand_dims(initial_probs, -1)
        if initial_probs.shape[1] == 1:
            new_probs = np.concatenate([self.probabilities(contexts), initial_probs], axis=-1)
            new_probs /= np.sum(new_probs, axis=-1, keepdims=True)
        else:
            new_probs = initial_probs
        new_params = np.concatenate([self._params, np.random.uniform(0, 1, (self._params.shape[0], 1))], axis=-1)
        self._params = fit_softmax(contexts, new_probs, new_params)

    def remove_entry(self, contexts, idx):
        probs = self.probabilities(contexts)
        new_probs = np.concatenate([probs[:, :idx], probs[:, idx + 1:]], axis=-1)
        new_probs /= np.sum(new_probs, axis=-1, keepdims=True)
        new_params = np.concatenate([self._params[:, :idx], self._params[:, idx + 1:]], axis=1)
        self._params = fit_softmax(contexts, new_probs, new_params)
