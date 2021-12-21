import numpy as np


class Trajectroy:
    def __init__(self, T=None):

        self.hit_time_point = None
        self.hit_joint_pos = list()
        self.hit_EE_pos = None
        self.hit_end_EE_pos = None
        self.hit_duration = T
        self.decelerate_duration = 1.5
        self.whole_duration = np.round(self.hit_duration + self.decelerate_duration, 4)
        self.sim_args = {
            "time_step": 0.002,
            "start_kp": 50,
            "RENDER": False,
            "end_kp": 400,
            "plot": False,
            "kp_config": False,
        }
        self.time_step = 0.002
        self.hit_time = np.arange(0, self.hit_duration, self.time_step)
        self.hit_end_time = self.hit_duration
        initial_qpos_value = {"wam/base_yaw_joint_right": 0,  # control 0
                              "wam/shoulder_pitch_joint_right": 0,  # control 1
                              "wam/shoulder_yaw_joint_right": 0,  # control 2
                              "wam/elbow_pitch_joint_right": 1.5,  # control 3
                              "wam/wrist_yaw_joint_right": 0,  # control 4
                              "wam/wrist_pitch_joint_right": 0,  # control 5
                              "wam/palm_yaw_joint_right": 1.5}  # control 6
        # initialize qpos
        joint_name_list = ["wam/base_yaw_joint_right", "wam/shoulder_pitch_joint_right", "wam/shoulder_yaw_joint_right",
                           "wam/elbow_pitch_joint_right", "wam/wrist_yaw_joint_right", "wam/wrist_pitch_joint_right",
                           "wam/palm_yaw_joint_right"]
        self.initial_qpos_value = np.zeros(len(joint_name_list))
        for name in joint_name_list:
            self.initial_qpos_value[joint_name_list.index(name)] = initial_qpos_value[name]

    @classmethod
    def gaussian(cls, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @classmethod
    def sigmoid(cls, x):
        x = x.astype(np.float32)
        s = 1 / (1 + np.exp(-x))
        return s

    def hit_trajectory(self):
        raise NotImplementedError()

    def decelerate_trajectory(self, hit_trj):
        start_point = np.expand_dims(hit_trj[:, -1], axis=1)
        decelerate_duration = 2  # 4D: 2, 2D: 1.5
        nDof = 7
        whole_duration_with_deceleration = self.hit_duration + decelerate_duration
        self.decelerate_time = np.arange(self.hit_duration, whole_duration_with_deceleration, self.time_step)
        end_trajectory_v = (hit_trj[:, -1] - hit_trj[:, -2]) / self.time_step
        v_start = end_trajectory_v.reshape((nDof, 1))
        v_end = 0
        acc = (v_end - v_start) / decelerate_duration
        self.decelerate_time_from_zero = self.decelerate_time - self.hit_duration
        self.decelerate_trj = start_point + v_start * np.tile(self.decelerate_time_from_zero,
                                                              (nDof, 1)) + 0.5 * acc * np.power(
            self.decelerate_time_from_zero, 2)
        return self.decelerate_trj

    def whole_trajectory(self):
        hit_trj = self.hit_trajectory()
        self.hit_trj_end_q_pos = hit_trj[:, -1]
        decelerate_trj = self.decelerate_trajectory(hit_trj)
        self.whole_time = np.concatenate((self.hit_time, self.decelerate_time))
        whole_trj = np.concatenate((hit_trj, decelerate_trj), axis=1)
        self.whole_trj = whole_trj
        return whole_trj
