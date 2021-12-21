from scipy.interpolate import make_interp_spline
import numpy as np

class PhaseGenerator():

    def __init__(self):
        return

    def phase(self, time):

        # Base class...
        return



class LinearPhaseGenerator(PhaseGenerator):

    def __init__(self, phaseVelocity = 1.0):

        PhaseGenerator.__init__(self)
        self.phaseVelocity = phaseVelocity


    def phase(self, time):

        phase = np.array(time) * self.phaseVelocity

        return phase

class SmoothPhaseGenerator(PhaseGenerator):

    def __init__(self, duration = 1):

        PhaseGenerator.__init__(self)
        self.duration = duration

        l, r = [(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]

        self.bsplinef = make_interp_spline([0, self.duration], [0, 1], bc_type=(l, r), k=5)



    def phase(self, time):
        phase = self.bsplinef(time)

        return phase


class RhythmicPhaseGenerator(PhaseGenerator):

    def __init__(self, phasePeriod = 1.0, useModulo = False ):
        PhaseGenerator.__init__(self)
        self.phasePeriod = phasePeriod
        self.useModulo = useModulo

    def phase(self, time):

        phase = time / self.phasePeriod
        if (self.useModulo):
            phase = np.mod(phase, 1.0)

        return phase



class ExpDecayPhaseGenerator(PhaseGenerator):

    def __init__(self, duration = 1, alphaPhase = 2):

        PhaseGenerator.__init__(self)

        self.tau =  1.0 / duration
        self.alphaPhase = alphaPhase

    def phase(self, time):

        time = time * self.tau

        phase = np.exp(- time * self.alphaPhase)

        return phase

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    phaseGenerator = LinearPhaseGenerator()
    time = np.linspace(0,1, 100)
    phase = phaseGenerator.phase(time)

    plt.figure()
    plt.plot(time, phase)
    # plt.hold(True)

    phaseGenerator.tau = 2
    phase = phaseGenerator.phase(time)
    plt.plot(time, phase)

    phaseGenerator.tau = 0.5
    phase = phaseGenerator.phase(time)
    plt.plot(time, phase)

    plt.show()


    print('PhaseGeneration Done')

