import numpy as np


class Categorical:

    def __init__(self, probabilities):
        self._p = probabilities

    def sample(self, num_samples):
        thresholds = np.expand_dims(np.cumsum(self._p), 0)
        thresholds[0, -1] = 1.0
        eps = np.random.uniform(size=[num_samples, 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    @property
    def probabilities(self):
        return self._p

    @probabilities.setter
    def probabilities(self, new_probabilities):
        self._p = new_probabilities

    @property
    def log_probabilities(self):
        return np.log(self._p + 1e-125)

    def add_entry(self, new_prob):
        p = np.concatenate([self._p, new_prob * np.ones(1,)], axis=0)
        self._p = p / np.sum(p)

    def remove_entry(self, idx):
        if idx > self._p.size:
            raise AssertionError("Invalid Index for Categorical")
        p = np.concatenate([self._p[:idx], self._p[idx + 1:]], axis=0)
        self._p = p / np.sum(p)

    def entropy(self):
        return - np.sum(self._p * np.log(self._p + 1e-125))

    def kl(self, other):
        return np.sum(self._p * (np.log(self._p + 1e-125) - other.log_probabilities))

if __name__ == "__main__":

    p = np.array([0.3, 0.2, 0.1, 0.4])

    cat = Categorical(p)

    s = cat.sample(100000000)
    for i in range(4):
        print(i, np.count_nonzero(s == i) / 100000000)