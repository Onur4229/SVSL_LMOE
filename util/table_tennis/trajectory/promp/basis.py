from util.table_tennis.trajectory.promp import phase
import numpy as np
class BasisGenerator():

    def __init__(self, phaseGenerator, numBasis=10):
        self.numBasis = numBasis
        self.phaseGenerator = phaseGenerator

    def basis(self):
        return None

    def basisMultiDoF(self, time, numDoF):
        basisSingleDoF = self.basis(time)

        basisMultiDoF = np.zeros((basisSingleDoF.shape[0] * numDoF, basisSingleDoF.shape[1] * numDoF))

        for i in range(numDoF):
            rowIndices = slice(i * basisSingleDoF.shape[0], (i + 1) * basisSingleDoF.shape[0])
            columnIndices = slice(i * basisSingleDoF.shape[1], (i + 1) * basisSingleDoF.shape[1])

            basisMultiDoF[rowIndices, columnIndices] = basisSingleDoF

        return basisMultiDoF


class DMPBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator, numBasis=10, duration=1, basisBandWidthFactor=3):
        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.basisBandWidthFactor = basisBandWidthFactor

        timePoints = np.linspace(0, duration, self.numBasis)
        self.centers = self.phaseGenerator.phase(timePoints)

        tmpBandWidth = np.hstack((self.centers[1:] - self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)

    def basis(self, time):
        phase = self.phaseGenerator.phase(time)

        diffSqr = np.array([((x - self.centers) ** 2) * self.bandWidth for x in phase])
        basis = np.exp(- diffSqr / 2)

        sumB = np.sum(basis, axis=1)
        basis = [column * phase / sumB for column in basis.transpose()]
        return np.array(basis).transpose()


class NormalizedRBFBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator, numBasis=10, duration=1, basisBandWidthFactor=3, numBasisOutside=0):
        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.basisBandWidthFactor = basisBandWidthFactor
        self.numBasisOutside = numBasisOutside
        basisDist = duration / (self.numBasis - 2 * self.numBasisOutside - 1)

        timePoints = np.linspace(-self.numBasisOutside * basisDist, duration + self.numBasisOutside * basisDist,
                                 self.numBasis)
        self.centers = self.phaseGenerator.phase(timePoints)

        tmpBandWidth = np.hstack((self.centers[1:] - self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)

    def basis(self, time):
        if isinstance(time, (float, int)):
            time = np.array([time])

        phase = self.phaseGenerator.phase(time)

        diffSqr = np.array([((x - self.centers) ** 2) * self.bandWidth for x in phase])
        basis = np.exp(- diffSqr / 2)

        sumB = np.sum(basis, axis=1)
        basis = [column / sumB for column in basis.transpose()]
        return np.array(basis).transpose()


class NormalizedRhythmicBasisGenerator(BasisGenerator):

    def __init__(self, phaseGenerator, numBasis=10, basisBandWidthFactor=3):
        BasisGenerator.__init__(self, phaseGenerator, numBasis)

        self.numBandWidthFactor = basisBandWidthFactor
        self.centers = np.linspace(0, 1, self.numBasis)

        tmpBandWidth = np.hstack((self.centers[1:] - self.centers[0:-1], self.centers[-1] - self.centers[- 2]))

        # The Centers should not overlap too much (makes w almost random due to aliasing effect).Empirically chosen
        self.bandWidth = self.basisBandWidthFactor / (tmpBandWidth ** 2)

    def basis(self):
        phase = self.getInputTensorIndex(0)

        diff = np.arraay([np.cos((phase - self.centers) * self.bandWidth * 2 * np.pi)])
        basis = np.exp(diff)

        sumB = np.sum(basis, axis=1)
        basis = [column / sumB for column in basis.transpose()]
        return np.array(basis).transpose()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = NormalizedRBFBasisGenerator(phaseGenerator, numBasis=5, duration=1, basisBandWidthFactor=1,
                                                 numBasisOutside=0)

    time = np.linspace(0, 2, 100)
    basis = basisGenerator.basis(time)
    basisMultiDoF = basisGenerator.basisMultiDoF(time, 3)

    plt.figure()
    plt.plot(time, basis)

    plt.show()

    print('done')
