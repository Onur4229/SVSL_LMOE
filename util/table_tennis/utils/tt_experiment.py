import numpy as np
from gym.utils import seeding
import sys, os


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def EnablePrint():
    sys.stdout = sys.__stdout__


def enablePrint():
    sys.stdout = sys.__stdout__


def ball_gun_straight(pos_array):
    initial_z = 0.76 + 1
    v_x, v_z = [2.5, 1]
    # v_x, v_z = [4, 1]
    initial_x, initial_y = pos_array
    v_y = 0
    initial_ball_state = np.array([initial_x, initial_y, initial_z, v_x, v_y, v_z])
    return initial_ball_state


def ball_initialize(random=False, scale=False, context_range=None, scale_value=None):
    if random:
        if scale:
            # if scale_value is None:
            scale_value = context_scale_initialize(context_range)
            v_x, v_y, v_z = [2.5, 2, 0.5] * scale_value
            dx = 1
            dy = 0
            dz = 0.05
        else:
            seed = None
            np_random, seed = seeding.np_random(seed)
            dx = np_random.uniform(-0.1, 0.1)
            dy = np_random.uniform(-0.1, 0.1)
            dz = np_random.uniform(-0.1, 0.1)

            v_x = np_random.uniform(1.7, 1.8)
            v_y = np_random.uniform(0.7, 0.8)
            v_z = np_random.uniform(0.1, 0.2)
    else:
        if scale:
            v_x, v_y, v_z = [2.5, 2, 0.5] * scale_value
        else:
            v_x = 2.5
            v_y = 2
            v_z = 0.5
        dx = 1
        dy = 0
        dz = 0.05

    initial_x = 0 + dx - 1.2
    initial_y = -0.2 + dy - 0.6
    initial_z = 0.3 + dz + 1.5
    initial_ball_state = np.array([initial_x, initial_y, initial_z, v_x, v_y, v_z])
    return initial_ball_state


def context_scale_initialize(range):
    low, high = range
    scale = np.random.uniform(low, high, 1)
    return scale
