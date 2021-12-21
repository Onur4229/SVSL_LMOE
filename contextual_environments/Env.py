import numpy as np

class Environment:

    def __init__(self, action_dim, ctxt_dim):
        self.action_dim = action_dim
        self.ctxt_dim = ctxt_dim
        self.state = None
        self.action = None

    def step(self, actions, contexts):
        raise NotImplementedError

    def _reward(self, actions, contexts):
        raise NotImplementedError

    def _eval(self, actions):
        raise NotImplementedError

    def visualize(self, actions, **kwargs):
        raise NotImplementedError

    def sample_contexts(self, n_samples, context_range_bounds):
        ctxt_samples = np.random.uniform(context_range_bounds[0], context_range_bounds[1], size=(n_samples, self.ctxt_dim))
        return ctxt_samples

    def get_tot_n_samples(self):
        return 0, 0

    def check_where_invalid(self, ctxts, context_range_bounds, set_to_valid_region=False):
        idx_max = []
        idx_min = []
        dist_quad = np.zeros(ctxts.shape[0])
        for dim in range(self.ctxt_dim):
            min_dim = context_range_bounds[0][dim]
            max_dim = context_range_bounds[1][dim]
            idx_max_c = np.where(ctxts[:, dim] > max_dim)[0]
            idx_min_c = np.where(ctxts[:, dim] < min_dim)[0]
            if set_to_valid_region:
                if idx_max_c.shape[0] != 0:
                    ctxts[idx_max_c, dim] = max_dim
                if idx_min_c.shape[0] != 0:
                    ctxts[idx_min_c, dim] = min_dim
            if idx_max_c.shape[0] != 0:
                dist_quad[idx_max_c] += (max_dim - ctxts[idx_max_c, dim]) ** 2
            if idx_min_c.shape[0] != 0:
                dist_quad[idx_min_c] += (min_dim - ctxts[idx_min_c, dim]) ** 2
            idx_max.append(idx_max_c)
            idx_min.append(idx_min_c)
        return idx_max, idx_min, ctxts, dist_quad