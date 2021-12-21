from util.table_tennis.trajectory.promp.promps import ProMP
from util.table_tennis.trajectory.trajectory import Trajectroy
import util.table_tennis.trajectory.promp.phase as phase
import util.table_tennis.trajectory.promp.basis as basis
import numpy as np


class ProMPTrajectory(Trajectroy):
    """


    """

    def __init__(self, weights=None):
        """

        Args:
            config:
            weights:
        """
        super(ProMPTrajectory, self).__init__(T=weights[-1])

        self.numBasis = 3
        self.nDof = 7
        self.phaseGenerator = phase.LinearPhaseGenerator()
        self.basisGenerator = basis.NormalizedRBFBasisGenerator(self.phaseGenerator, numBasis=self.numBasis,
                                                                duration=self.hit_duration,
                                                                basisBandWidthFactor=1,
                                                                numBasisOutside=0)
        self.proMP = ProMP(self.basisGenerator, self.phaseGenerator, self.nDof)

        self.weights = weights[:-1]
        self.whole_trajectory()


    def hit_trajectory(self):
        self.hit_time_start_with_zero = np.arange(0, self.hit_duration, self.time_step)
        trajectory_flat = self.proMP.get_weighted_trajectories(self.hit_time_start_with_zero, self.weights)
        trajectory_flat = trajectory_flat.reshape((self.hit_time.shape[0], self.nDof)).T
        test = False
        if test:
            trajectory_flat = np.full_like(trajectory_flat, 1)
        self.hit_trj = trajectory_flat
        return trajectory_flat
