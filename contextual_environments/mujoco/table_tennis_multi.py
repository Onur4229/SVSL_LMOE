from contextual_environments.mujoco.abstractenv import AbstractEnvironment
from util.table_tennis.utils.hierarchical_reward import HierarchicalRewardTableTennis
from util.table_tennis.trajectory.sigmoid_promp_trajectory import SigmoidProMPTrajectory
from contextual_environments.mujoco.table_tennis_env import TableTennisEnv
from contextual_environments.mujoco.table_tennis_sim import TableTennisSimulation
from util.table_tennis.utils.tt_experiment import ball_initialize, ball_gun_straight
import numpy as np

# 4D
min_ctxts_coords = np.array([-1.2, -0.6, -1.2, -0.65])
max_ctxt_coords = np.array([-0.2, 0.6, -0.2, 0.65])

# 2D
# min_ctxts_coords = np.array([-1.2, -0.6])
# max_ctxt_coords = np.array([-0.2, 0.6])
CONTEXT_RANGE_BOUNDS = np.array([min_ctxts_coords, max_ctxt_coords])
ACTION_DIM = 22
CTXT_DIM = 4
# CTXT_DIM = 2


def get_env(name, n_cores, env_bounds, render=False):
    return TableTennisMul(name=name, n_cores=n_cores, env_bounds=env_bounds, render=render)


def get_tabletennis_cost_func(env_bounds):
    return TableTennisCostFunction(env_bounds)


class TableTennisCostFunction:
    def __init__(self, env_bounds=None, name=None, promp_time=False):
        self.name = name
        self.env = None
        self.env_bounds = env_bounds
        self.promp_time = promp_time

        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0

    def __call__(self, context, action, env_bounds, render):
        reward = None
        if context.shape[0] == 2:  # Target Position
            context_ball = ball_initialize(random=False, scale=False, context_range=None, scale_value=1)  # Fix Velocity
            reward = self._run_experiment(context_ball, action, target_pos=context, scale=False, ctxt_for_valid=context,
                                          render=render)
        elif context.shape[0] == 4:  # Straight Ball
            target_pos = context[0:2]
            initial_pos_array = context[2:]
            context_ball = ball_gun_straight(initial_pos_array)
            reward = self._run_experiment(context_ball, action, target_pos=target_pos, scale=False,
                                          ctxt_for_valid=context, render=render)
        success = 1
        return reward.highest_reward, success

    def _run_experiment(self, initial_ball_state, action, scale=True, target_pos=None,
                        ctxt_for_valid=None, render=False):
        reward_obj = HierarchicalRewardTableTennis(ctxt_dim=CTXT_DIM,
                                                   context_range_bounds=CONTEXT_RANGE_BOUNDS)

        weights = action
        # Step 0 Context Valid
        reward_obj.context_invalid_punishment(ctxt_for_valid)
        if not reward_obj.goal_achievement:
            return reward_obj
        # Step 1 Hit duration Valid
        reward_obj.action_durations_valid_upper_lower_bound_valid(weights[-1])
        # Step 2 Action Valid
        if reward_obj.goal_achievement:
            # Context Reward
            trajectory = SigmoidProMPTrajectory(weights=weights)
            reward_obj.joint_motion_mean_minus_penalty(trj=trajectory)
            if not reward_obj.goal_achievement:
                return reward_obj
            if scale:
                target_pos = np.array([-0.5, -0.5])
            # If there is no env. We generate a new MuJoCo Env.
            if self.env is None:
                self.env = TableTennisEnv(initial_ball_state=initial_ball_state,
                                          reward_obj=reward_obj,
                                          target_pos=target_pos)
            # If Env already exits. We update the config
            else:
                self.env.update(initial_ball_state=initial_ball_state, reward_obj=reward_obj,
                                target_pos=target_pos)
            #  Step 3 Hit and Step 4 Target Position Reward
            table_tennis_simulation = TableTennisSimulation(render=render)
            n_env_interacts = table_tennis_simulation.simulation(trajectory=trajectory, env=self.env)
            self.n_ep_samples_executed += 1
            self.n_env_interacts += n_env_interacts
            return reward_obj
        # Step 2.1 Action not Valid, return the reward
        else:
            return reward_obj

    def get_tot_n_samples(self):
        n_ep_samples_executed = np.copy(self.n_ep_samples_executed)
        n_env_interacts = np.copy(self.n_env_interacts)
        self.n_ep_samples_executed = 0
        self.n_env_interacts = 0
        return n_ep_samples_executed, n_env_interacts

    def step(self, xs, cs, render=False):
        rewards = []
        successes = []
        for i in range(cs.shape[0]):
            r, s = self(cs[i, :], xs[i, :], self.env_bounds, render=render)
            rewards.append(r)
            successes.append(s)
        return np.stack(rewards), np.stack(successes)


class TableTennisMul(AbstractEnvironment):

    def __init__(self, name, n_cores, env_bounds, render=False, promp_time=False):
        super(TableTennisMul, self).__init__(name, n_cores, TableTennisCostFunction, env_bounds=env_bounds,
                                             render=render,
                                             promp_time=promp_time)
