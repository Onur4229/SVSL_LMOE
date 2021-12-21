import numpy as np
from numba import jit


class RegressionFunc:

    def __init__(self, reg_fact, normalize, unnormalize_output, bias_entry=None):
        self._reg_fact = reg_fact
        self._normalize = normalize
        self._unnormalize_output = unnormalize_output
        self._bias_entry = bias_entry
        self._params = None
        self.o_std = None
        self._normalized_features = None
        self._normalized_outputs = None

    def __call__(self, inputs):
        if self._params is None:
            raise AssertionError("Model not trained yet")
        return self._feature_fn(inputs) @ self._params

    def _feature_fn(self, x):
        raise NotImplementedError

    def _normalize_features(self, features):
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        # do not normalize bias
        if self._bias_entry is not None:
            mean[:, self._bias_entry] = 0.0
            std[:, self._bias_entry] = 1.0
        features = (features - mean) / (std + 1e-20)
        return features, np.squeeze(mean, axis=0), np.squeeze(std, axis=0)

    def _normalize_outputs(self, outputs):
        mean = np.mean(outputs, axis=0, keepdims=True)
        # std = np.std(outputs)
        outputs_cov = np.cov(outputs.T, bias=True)
        if len(outputs_cov.shape) == 0:
            std = np.sqrt(outputs_cov)
        else:
            std = np.sqrt(np.diag(outputs_cov))
        outputs = (outputs - mean) / (std + 1e-20)
        return outputs, mean, std

    def _undo_normalization(self, params, f_mean, f_std, o_mean, o_std):
        if self._unnormalize_output:
            if len(params.shape) == 2:
                if len(f_mean.shape) == 1:
                    f_mean = f_mean.reshape((-1, 1))
                if len(o_std.shape) == 0:
                    o_std = o_std.reshape(1)
                if len(params.shape) == 1:
                    params = params.reshape((-1, 1))
                tmp = params[self._bias_entry].copy()
                params = np.dot(np.dot(np.diag(1 / f_std).T, params), np.diag(o_std))
                params[self._bias_entry] = tmp.reshape((1, -1))
                params[self._bias_entry] = np.dot(params[self._bias_entry], np.diag(o_std)) + o_mean - np.dot(f_mean[:-1].T,
                                                                                                          params[:-1, :])

            else:
                params *= (o_std / f_std)               # orig
                params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean) + o_mean         # orig
        else:
            params *= (1.0 / f_std)
            params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean)                  # orig
        return params

    def fit(self, inputs, outputs, weights=None):
        if len(outputs.shape) > 1:
            outputs = np.squeeze(outputs)
        features = self._feature_fn(inputs)
        if self._normalize:
            features, f_mean, f_std = self._normalize_features(features)
            outputs, o_mean, o_std = self._normalize_outputs(outputs)
            self._normalized_features = features
            self._normalized_outputs = outputs
        if weights is not None:
            if len(weights.shape) == 1:
                weights = np.expand_dims(weights, 1)
            weighted_features = weights * features
            # self._normalized_features = weighted_features
        else:
            weighted_features = features

        # regression
        reg_mat = np.eye(weighted_features.shape[-1]) * self._reg_fact
        if self._bias_entry is not None:
            reg_mat[self._bias_entry, self._bias_entry] = 0.0
        try:
            self._params = np.linalg.solve(weighted_features.T @ features + reg_mat, weighted_features.T @ outputs)
            if self._normalize:
                self._params = self._undo_normalization(self._params, f_mean, f_std, o_mean, o_std)
                self.o_std = o_std
        except np.linalg.LinAlgError as e:
            print("Error during matrix inversion", e.what())


class LinFunc(RegressionFunc):

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, -1)

    def _feature_fn(self, x):
        return np.concatenate([x, np.ones([x.shape[0], 1], dtype=x.dtype)], 1)


class QuadFunc(RegressionFunc):
    # *Fits - 0.5 * x ^ T  Rx + x ^ T r + r_0 ** * /

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, bias_entry=-1)
        self.quad_term = None
        self.lin_term = None
        self.const_term = None

    @staticmethod
    @jit(nopython=True)
    def _feature_fn(x):
        num_quad_features = int(np.floor(0.5 * (x.shape[-1] + 1) * x.shape[-1]))
        num_features = num_quad_features + x.shape[-1] + 1
        features = np.ones((x.shape[0], num_features))
        write_idx = 0
        # quad features
        for i in range(x.shape[-1]):
            for j in range(x.shape[-1] - i):
                features[:, write_idx] = x[:, i] * x[:, j + i]
                write_idx += 1
        # linear features
        features[:, num_quad_features: -1] = x

        # last coloumn (bias) already 1
        return features

    def fit(self, inputs, outputs, weights=None, sample_mean=None, sample_chol_cov=None):
        if sample_mean is None:
            assert sample_chol_cov is None
        if sample_chol_cov is None:
            assert sample_mean is None

        # whithening
        if sample_mean is not None and sample_chol_cov is not None:
            inv_samples_chol_cov = np.linalg.inv(sample_chol_cov)
            inputs = (inputs - sample_mean) @ inv_samples_chol_cov.T

        dim = inputs.shape[-1]

        super().fit(inputs, outputs, weights)

        idx = np.triu(np.ones([dim, dim], np.bool))

        qt = np.zeros([dim, dim])
        qt[idx] = self._params[:- (dim + 1)]
        self.quad_term = - qt - qt.T

        self.lin_term = self._params[-(dim + 1): -1]
        self.const_term = self._params[-1]

        # unwhitening:
        if sample_mean is not None and sample_chol_cov is not None:
            self.quad_term = inv_samples_chol_cov.T @ self.quad_term @ inv_samples_chol_cov
            t1 = inv_samples_chol_cov.T @ self.lin_term
            t2 = self.quad_term @ sample_mean
            self.lin_term = t1 + t2
            self.const_term += np.dot(sample_mean, -0.5 * t2 - t1)

    # - 0.5 * x ^ T Rx + x ^ T r + r_0 ** * /
    def predict(self, inputs):
        quad_part = np.diag(inputs@self.quad_term@inputs.T)
        quad_part = -0.5*quad_part
        res = quad_part + inputs@self.lin_term + self.const_term
        return res.flatten()
        # feat = self._feature_fn(inputs)
        # return feat@self._params

class QuadFuncJoint(QuadFunc):
    # Fits function of the form R(x, y) = -0.5 * x^T R_xx x + x^T R_xy y - 0.5 * y^T R_yy y + l_x^T x + l_y^T y + c

    def __init__(self, x_dim, y_dim, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output)
        self._x_dim = x_dim
        self._y_dim = y_dim
        self.quad_term_xx = None
        self.quad_term_yx = None
        self.quad_term_yy = None
        self.lin_term_x = None
        self.lin_term_y = None

    def fit(self, inputs, outputs, weights=None, sample_mean=None, sample_chol_cov=None):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = np.concatenate(inputs, axis=-1)
        super().fit(inputs, outputs, weights, sample_mean, sample_chol_cov)
        self.quad_term_xx = self.quad_term[:self._x_dim, :self._x_dim]
        self.quad_term_yx = - self.quad_term[self._x_dim:, :self._x_dim]
        self.quad_term_yy = self.quad_term[self._x_dim:, self._x_dim:]
        self.lin_term_x = self.lin_term[:self._x_dim]
        self.lin_term_y = self.lin_term[self._x_dim:]

    def predict(self, ctxts, samples):
        r_aa = self.quad_term_yy
        r_cc = self.quad_term_xx
        r_ac = self.quad_term_yx
        r_a = self.lin_term_y
        r_c = self.lin_term_x

        pred = -0.5*np.diag(samples @r_aa @samples.T) - 0.5*np.diag(ctxts@r_cc@ctxts.T) + np.diag(samples @r_ac @ ctxts.T) + samples@r_a + ctxts@r_c + self.const_term
        # pred = np.diag(samples @r_aa @samples.T) +np.diag(ctxts@r_cc@ctxts.T) + np.diag(samples @r_ac @ ctxts.T) + samples@r_a + ctxts@r_c + self.const_term
        return pred

    def predict_features(self, inputs):
        feat = self._feature_fn(inputs)
        return feat@self._params