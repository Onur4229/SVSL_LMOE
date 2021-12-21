import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, cost_fn_obj, env_bounds, render):
    parent_remote.close()
    try:
        while True:
            mode, contexts, thetas = remote.recv()
            if mode == "experiment":
                n = contexts.shape[0]
                rewards = []
                successes = []
                n_ep_samples_executed = 0
                n_env_interacts = 0
                for i in range(0, n):
                    r, s = cost_fn_obj(contexts[i, :], thetas[i, :], env_bounds, render= render)
                    rewards.append(r)
                    successes.append(s)
                    n_ep_samples_executed += cost_fn_obj.n_ep_samples_executed
                    n_env_interacts += cost_fn_obj.n_env_interacts
                    cost_fn_obj.n_ep_samples_executed = 0
                    cost_fn_obj.n_env_interacts = 0
                remote.send((rewards, successes, n_ep_samples_executed, n_env_interacts))
            elif mode == "seed":
                cost_fn_obj.set_seed(contexts)
                remote.send(True)
            else:
                raise RuntimeError("Unexpected mode '" + str(mode) + "'")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')


class AbstractEnvironment:

    def __init__(self, name, n_cores, cost_fn_constructor, env_bounds=None, render=False, promp_time=False):
        self.name = name
        self.n_cores = n_cores
        self.env_bounds = env_bounds
        self.render = render
        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0

        cost_fn_obj_list = []
        for i in range(self.n_cores):
            cost_fn_obj_list.append(cost_fn_constructor(i, name=self.name, promp_time=promp_time))

        mp.set_start_method('fork', force=True)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_cores)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote, cost_fn_obj, self.env_bounds, self.render))
                    for (work_remote, remote, cost_fn_obj) in zip(self.work_remotes, self.remotes, cost_fn_obj_list)]

        self.cost_fn_obj_list = cost_fn_obj_list
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def get_tot_n_samples(self):

        return self.n_ep_samples_executed, self.n_env_interacts


    def set_seed(self, seed):
        # We need to seed this process as well since the contexts and thetas get sampled in it
        np.random.seed(seed)
        # We need to seed the processes as well to get consistent samples
        count = 0
        for remote in self.remotes:
            remote.send(("seed", self.n_cores * seed + count, None))
            count += 1

        # This is just for synchronization
        [remote.recv() for remote in self.remotes]

    def sample_contexts(self, n_samples, context_range_bounds):
        ctxt_dim = context_range_bounds[0].shape[0]
        ctxt_samples = np.random.uniform(context_range_bounds[0], context_range_bounds[1], size=(n_samples, ctxt_dim))
        return ctxt_samples

    def check_where_invalid(self, ctxts, context_range_bounds, set_to_valid_region=False):
        idx_max = []
        idx_min = []
        ctxt_dim = ctxts.shape[1]
        for dim in range(ctxt_dim):
            min_dim = context_range_bounds[0][dim]
            max_dim = context_range_bounds[1][dim]
            idx_max_c = np.where(ctxts[:, dim] > max_dim)[0]
            idx_min_c = np.where(ctxts[:, dim] < min_dim)[0]
            if set_to_valid_region:
                if idx_max_c.shape[0] != 0:
                    ctxts[idx_max_c, dim] = max_dim
                if idx_min_c.shape[0] != 0:
                    ctxts[idx_min_c, dim] = min_dim
            idx_max.append(idx_max_c)
            idx_min.append(idx_min_c)
        return idx_max, idx_min, ctxts

    def step(self, xs, cs):
        n = xs.shape[0]

        # We split the load of computing the experiment evenly among the cores
        contexts = []
        thetas = []
        n_sub = int(n / self.n_cores)
        rem = n - n_sub * self.n_cores
        count = 0
        for i in range(0, self.n_cores):
            extra = 0
            if rem > 0:
                extra = 1
                rem -= 1

            count_new = count + n_sub + extra

            sub_contexts = cs[count:count_new, :]
            if len(sub_contexts.shape) == 1:
                sub_contexts = np.array([sub_contexts])
            contexts.append(sub_contexts)

            sub_thetas = xs[count:count + n_sub + extra, :]
            if len(sub_thetas.shape) == 1:
                sub_thetas = np.array([sub_thetas])
            thetas.append(sub_thetas)

            count = count_new

        for remote, sub_contexts, sub_thetas in zip(self.remotes, contexts, thetas):
            remote.send(("experiment", sub_contexts, sub_thetas))
        tmp = [remote.recv() for remote in self.remotes]

        rewards = []
        successes = []
        n_ep_samples_executed = 0
        n_env_interacts = 0
        for sub_tmp in tmp:
            rewards.append(sub_tmp[0])
            successes.append(sub_tmp[1])
            n_ep_samples_executed += sub_tmp[2]
            n_env_interacts += sub_tmp[3]
        self.n_ep_samples_executed = n_ep_samples_executed
        self.n_env_interacts = n_env_interacts
        return np.concatenate(rewards), np.concatenate(successes)
