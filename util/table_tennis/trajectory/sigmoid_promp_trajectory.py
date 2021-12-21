from util.table_tennis.trajectory.promp_trajectory import ProMPTrajectory
import numpy as np


class SigmoidProMPTrajectory(ProMPTrajectory):
    def __init__(self, weights=None):
        super(SigmoidProMPTrajectory, self).__init__( weights=weights)
        self.whole_trajectory()

    def hit_trajectory(self):
        # basic function
        self.hit_time_start_with_zero = np.arange(0, self.hit_duration, self.time_step)
        self.BF = self.basisGenerator.basis(self.hit_time_start_with_zero)
        trajectory_flat = self.proMP.get_weighted_trajectories(self.hit_time_start_with_zero, self.weights)
        trajectory_flat = trajectory_flat.reshape((self.hit_time_start_with_zero.shape[0], self.nDof)).T
        self.hit_trj = trajectory_flat
        const = 6
        T1 = 0
        T2 = T1 + self.hit_duration
        alpha = ProMPTrajectory.sigmoid(2 * const * (self.hit_time - T1) / (T2 - T1) - const)
        self.hit_trj = alpha * trajectory_flat + self.initial_qpos_value.reshape(7, 1)
        return self.hit_trj
