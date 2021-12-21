import numpy as np
import glfw
import os
import copy

from gym import spaces
from gym.envs.robotics import robot_env
from pathlib import Path


class TableTennisEnv(robot_env.RobotEnv):
    """Class for Table Tennis environment.
    """

    def __init__(self, n_substeps=1,
                 model_path=None,
                 initial_qpos=None,
                 initial_ball_state=None,
                 reward_obj=None,
                 target_pos=None,
                 ):
        """Initializes a new MuJoCo based Table Tennis environment.
        """
        if model_path is None:
            current_dir = Path(os.path.split(os.path.realpath(__file__))[0])
            table_tennis_env_xml_path = current_dir / "table_tennis" / "table_tennis_env.xml"

            self.model_path = str(table_tennis_env_xml_path)
        time_step = 0.002
        if initial_qpos is None:
            initial_qpos = {"wam/base_yaw_joint_right": 0,  # control 0
                            "wam/shoulder_pitch_joint_right": 0,  # control 1
                            "wam/shoulder_yaw_joint_right": 0,  # control 2
                            "wam/elbow_pitch_joint_right": 1.5,  # control 3
                            "wam/wrist_yaw_joint_right": 0,  # control 4
                            "wam/wrist_pitch_joint_right": 0,  # control 5
                            "wam/palm_yaw_joint_right": 1.5}  # control 6

        assert initial_qpos is not None, "Must initialize the initial q position of robot arm"
        n_actions = 7
        self.initial_qpos = initial_qpos
        self.initial_qpos_value = np.array(list(initial_qpos.values())).copy()
        self.target_pos = target_pos
        action_space_low = np.array([-2.6, -2.0, -2.8, -0.9, -4.8, -1.6, -2.2])
        action_space_high = np.array([2.6, 2.0, 2.8, 3.1, 1.3, 1.6, 2.2])

        self.n_substeps = n_substeps

        super(TableTennisEnv, self).__init__(
            model_path=self.model_path, n_substeps=self.n_substeps, n_actions=n_actions,
            initial_qpos=initial_qpos)

        arr1 = self.target_pos
        arr2 = np.array([0.77])
        self.sim.model.body_pos[5] = np.concatenate((arr1, arr2))

        self.action_space = spaces.Box(low=action_space_low,
                                       high=action_space_high,
                                       dtype='float64')

        self.scale = None
        self.desired_pos = None
        self.n_actions = n_actions
        self.action = None
        self.time_step = time_step
        self.paddle_center_pos = self.sim.data.get_site_xpos('wam/paddle_center')
        self.reward_obj = reward_obj

        self.initial_ball_state = initial_ball_state

        self.target_ball_pos = self.sim.data.get_site_xpos("target_ball")
        self.racket_center_pos = self.sim.data.get_site_xpos("wam/paddle_center")

    def update(self,
               initial_qpos=None,
               initial_ball_state=None,
               reward_obj=None,
               target_pos=None,
               ):

        if initial_qpos is None:
            initial_qpos = self.initial_qpos

        assert initial_qpos is not None, "Must initialize the initial q position of robot arm"
        self.initial_qpos_value = np.array(list(initial_qpos.values())).copy()
        self.target_pos = target_pos
        self.reward_obj = reward_obj
        self.initial_ball_state = initial_ball_state

        # Update Location Information.
        arr1 = self.target_pos
        arr2 = np.array([0.77])
        self.sim.model.body_pos[3] = np.array([0, 0, 0.5])
        self.sim.model.body_pos[4] = np.array([0, 0, 0.5])
        self.sim.model.body_pos[5] = np.concatenate((arr1, arr2))
        self.sim.set_state(self.initial_state)

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()

        self.paddle_center_pos = self.sim.data.get_site_xpos('wam/paddle_center')
        self.target_ball_pos = self.sim.data.get_site_xpos("target_ball")
        self.racket_center_pos = self.sim.data.get_site_xpos("wam/paddle_center")

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None
            self._viewers = {}

    def compute_reward(self, achieved_goal, goal, info):
        self.reward_obj.temp_reward = 0
        # Stage Hitting
        self.reward_obj.hitting(self)
        # if not contact between racket and ball, return the highest reward
        if not self.reward_obj.goal_achievement:
            return self.reward_obj.highest_reward
        self.reward_obj.target_achievement(self)
        return self.reward_obj.highest_reward

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        [initial_x, initial_y, initial_z, v_x, v_y, v_z] = self.initial_ball_state
        self.sim.data.set_joint_qpos('tar:x', initial_x)
        self.sim.data.set_joint_qpos('tar:y', initial_y)
        self.sim.data.set_joint_qpos('tar:z', initial_z)
        self.energy_corrected = True
        self.give_reflection_reward = False

        # velocity is positive direction
        self.sim.data.set_joint_qvel('tar:x', v_x)
        self.sim.data.set_joint_qvel('tar:y', v_y)
        self.sim.data.set_joint_qvel('tar:z', v_z)

        # Apply gravity compensation
        self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        # Apply gravity compensation
        self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        self.sim.forward()
        # Get the target position
        self.initial_paddle_center_xpos = self.sim.data.get_site_xpos('wam/paddle_center').copy()
        self.initial_paddle_center_vel = None

    def _sample_goal(self):
        goal = self.initial_paddle_center_xpos[:3] + self.np_random.uniform(-0.2, 0.2, size=3)
        return goal.copy()

    def _get_obs(self):
        # positions of racket center
        ball_pos = self.sim.data.get_site_xpos("target_ball")
        obs = np.concatenate([
            self.sim.data.qpos[:].copy(), ball_pos.copy(), self.target_pos.copy()
        ])
        out_dict = {
            'observation': obs.copy(),
            'achieved_goal': ball_pos.copy(),
            'desired_goal': self.target_pos.copy(),
            'q_pos': self.sim.data.qpos[:].copy(),
            "ball_pos": ball_pos.copy(),
        }

        return out_dict

    def _step_callback(self):
        pass

    def _set_action(self, action):
        # Apply gravity compensation
        self.sim.data.qfrc_applied[:7] = self.sim.data.qfrc_bias[:7]
        assert action.shape == (self.n_actions,)
        self.action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = self.action[:]  # limit maximum change in position
        pos_ctrl = np.clip(pos_ctrl, self.action_space.low, self.action_space.high)
        # Get desired trajectory
        self.sim.data.ctrl[:] = pos_ctrl
        self.sim.forward()
        self.desired_pos = self.sim.data.get_site_xpos('wam/paddle_center').copy()

    def _is_success(self, achieved_goal, desired_goal):
        pass
