import numpy as np

def log_sum_exp(exp_arg, axis):
    exp_arg_use = exp_arg.copy()
    max_arg = np.max(exp_arg_use)
    exp_arg_use = np.clip(exp_arg_use-max_arg, -700, 700)
    return max_arg + np.log(np.sum(np.exp(exp_arg_use), axis=axis))


