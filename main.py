from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import copy
from itertools import product, combinations
import os, __future__, random, math, scipy, seaborn
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Particle(object):
    """ This is basic container for radiation particles. """
    """ Every Particle has position, direction and energy """

    def __init__(self, particle_type="photon", energy=0, position=[0, 0, 0], direction=[0, 0, 0]):
        """ Particle Type takes the values: electron or photon """
        self.type = particle_type
        """ Particle mean free path in cm """
        self.mfp = 0.0
        """ Particle energy units is in MV/Mev """
        self.energy = energy
        """ Position is 3D spatial container for x, y and z """
        self.position = position
        """ If v is a Euclidean vector in three-dimensional Euclidean space, ℝ3, v = v_x e_x + v_y e_y + v_z e_z , 
        where e_x, e_y, e_z are the standard basis in Cartesian notation, then the direction cosines are  
        α, β and γ are the direction cosines and the Cartesian coordinates of the unit vector v/|v|. 
         Here α^2 + β^2 + γ^2 = 1 """
        self.direction = direction


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def CalculateRotationOperator(theta, phi):
    """ theta -> elevation, phi - > azimuth """
    return np.array([[math.cos(theta), -math.sin(theta)*math.sin(phi), math.sin(theta) * math.cos(phi)],
                     [math.sin(phi), math.cos(theta), math.sin(theta)*math.sin(phi)],
                     [-math.sin(theta)*math.cos(phi), -math.sin(theta)*math.cos(phi), math.cos(theta)]], np.float)


def ConvertCartesian2spherical(x, y, z):
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    elev = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    az = math.atan2(y, x)
    return r, elev, az


def CalculateStatistical_PI(SampleSize):
    """ Throw Darts to Estimate π """
    """ Area of a Circle = π R^2 = π (D/2)^2 = π / 4  x D^2 (= Area of Square) """
    """ π = 4 x Area of a Circle / Area of Square """

    """ Prepare Containers """
    DartsInside = 0
    xInside = [], yInside = [], xOutside = [], yOutside = []

    """ Throw darts and save the data """
    for _ in range(SampleSize):
        x = random.random()
        y = random.random()

        if math.sqrt(x * x + y * y) < 1:
            DartsInside += 1
            xInside.append(x)
            yInside.append(y)
        else:
            xOutside.append(x)
            yOutside.append(y)

    """" Plot the results """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(xInside, yInside, s=20, c='b')
    ax.scatter(xOutside, yOutside, s=20, c='r')

    π = (float(DartsInside) / SampleSize) * 4
    plt.text(0.1, 0.5, 'Sample Size: {0}.\nπ: {1}.'.format(SampleSize, π), fontsize=10, color='white')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.show()


def GeneratePhotonParticle(PartcileCount, energy):
    """ Make energy distribution using gamma distribution which mimic the actual particles energy distribution """
    Shape, Scale = 2., energy ** 0.2
    energies = np.random.normal(energy, math.sqrt(energy), PartcileCount)

    """ Make distribution of positions """
    """ uncomment and copy <[x[iParticle], y[iParticle], z[iParticle]]> to the function below """
    x = np.random.uniform(0.0, 0.0, PartcileCount)
    y = np.random.uniform(0.0, 0.0, PartcileCount)
    z = np.random.uniform(0.0, 0.0, PartcileCount)

    """ Make distribution of direction cosines """
    """ uncomment and copy <[uX[iParticle], vY[iParticle], wZ[iParticle]]> to the function below """
    uX = np.random.uniform(-0.1, 0.1, PartcileCount)
    vY = np.random.uniform(-0.1, 0.1, PartcileCount)
    wZ = np.random.uniform(1.0, 1.0, PartcileCount)

    """ Make particles """
    particleList = []
    for iParticle in range(PartcileCount):
        Norm = NormalizeCalculateDirectionVector(uX[iParticle], vY[iParticle], wZ[iParticle])
        particleList.append(Particle("photon",
                                     energies[iParticle],
                                     [x[iParticle], y[iParticle], z[iParticle]],
                                     [Norm[0], Norm[1], Norm[2]]))

    return particleList


def CalculateElectronRange(electron:Particle):
    """ This method uses Katz and Benfold empirical formula """
    """ this method calculates the electron range in g/cm^2. 
        So we divide by the water density = 1 and by 0.01 to convert it to m """

    if electron is None:
        return

    UnitsConversion_cm2m = 0.01

    if 0.01 <= electron.energy <= 2.5:
        electron.mfp = (0.412 * electron.energy ** (1.265-0.0954 * math.log10(electron.energy))) * UnitsConversion_cm2m
    elif electron.energy > 2.5:
        electron.mfp = (0.5320 * electron.energy - 0.106) * UnitsConversion_cm2m
    else:
        electron.mfp = 0.0


def NormalizeCalculateDirectionVector(u, v, w):
    if w is None:
        w = math.sqrt(1 - u ** 2 - v ** 2)
        return np.array([u, v, w], np.float)
    else:
        VectNorm = math.sqrt(u ** 2 + v ** 2 + w ** 2)
        return np.array([u / VectNorm, v / VectNorm, w / VectNorm], np.float)


def Calculate_MeanFreePath(photon, material="water"):
    """ Particle Travels a mean free path inside the medium (the distance of the next interaction).
    Mu is the medium linear attenuation coefficient (is the total cross section including all interactions).
    It must use uniform distribution 0 < # < 1 in random number generation  """

    if photon is None:
        return

    """ Linear attenuation coefficient for water. 
        from https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html. 
        The units are in cm^2 /g. """

    if material == "water":
        MuvsEn = [1.00000E-03, 4.078E+03, 4.065E+03, 1.50000E-03, 1.376E+03, 1.372E+03, 2.00000E-03, 6.173E+02, 6.152E+02,
                  3.00000E-03, 1.929E+02, 1.917E+02, 4.00000E-03, 8.278E+01, 8.191E+01, 5.00000E-03, 4.258E+01, 4.188E+01,
                  6.00000E-03, 2.464E+01, 2.405E+01, 8.00000E-03, 1.037E+01, 9.915E+00, 1.00000E-02, 5.329E+00, 4.944E+00,
                  1.50000E-02, 1.673E+00, 1.374E+00, 2.00000E-02, 8.096E-01, 5.503E-01, 3.00000E-02, 3.756E-01, 1.557E-01,
                  4.00000E-02, 2.683E-01, 6.947E-02, 5.00000E-02, 2.269E-01, 4.223E-02, 6.00000E-02, 2.059E-01, 3.190E-02,
                  8.00000E-02, 1.837E-01, 2.597E-02, 1.00000E-01, 1.707E-01, 2.546E-02, 1.50000E-01, 1.505E-01, 2.764E-02,
                  2.00000E-01, 1.370E-01, 2.967E-02, 3.00000E-01, 1.186E-01, 3.192E-02, 4.00000E-01, 1.061E-01, 3.279E-02,
                  5.00000E-01, 9.687E-02, 3.299E-02, 6.00000E-01, 8.956E-02, 3.284E-02, 8.00000E-01, 7.865E-02, 3.206E-02,
                  1.00000E+00, 7.072E-02, 3.103E-02, 1.25000E+00, 6.323E-02, 2.965E-02, 1.50000E+00, 5.754E-02, 2.833E-02,
                  2.00000E+00, 4.942E-02, 2.608E-02, 3.00000E+00, 3.969E-02, 2.281E-02, 4.00000E+00, 3.403E-02, 2.066E-02,
                  5.00000E+00, 3.031E-02, 1.915E-02, 6.00000E+00, 2.770E-02, 1.806E-02, 8.00000E+00, 2.429E-02, 1.658E-02,
                  1.00000E+01, 2.219E-02, 1.566E-02, 1.50000E+01, 1.941E-02, 1.441E-02, 2.00000E+01, 1.813E-02, 1.382E-02]
        MuvsEn = np.asarray(MuvsEn)
        MuvsEn = np.reshape(MuvsEn, (36, 3))
    elif material == "air":
        MuvsEn = np.fromfile("air.txt", sep=" ")
        MuvsEn = np.reshape(MuvsEn, (38, 3))
    elif material == "lead":
        MuvsEn = np.fromfile("lead.txt", sep=" ")
        MuvsEn = np.reshape(MuvsEn, (74, 3))
    """ Calculate the linear attenuation coefficient """

    EnergyKey = [row[0] for row in MuvsEn]
    MuValue = [row[1] for row in MuvsEn]
    Mu = np.interp(photon.energy, EnergyKey, MuValue)

    """ Calculate mean free path. Here we use direct sampling method. We use water density = 1 g/cm^3 """
    Rho_H2O = 1
    Rho_Air = 0.001275
    Rho_PB = 11.34
    UnitsConversion_cm2m = 0.01

    if material == "water":
        photon.mfp = -1 / (Mu * Rho_H2O) * math.log(1 - random.random()) * UnitsConversion_cm2m
    elif material == "air":
        photon.mfp = -1 / (Mu * Rho_Air) * math.log(1 - random.random()) * UnitsConversion_cm2m
    elif material == "lead":
        photon.mfp = -1 / (Mu * Rho_PB) * math.log(1 - random.random()) * UnitsConversion_cm2m
    return


def Calculate_KleinNishina_Coefficient(energy, angle):
    alpha = energy / 0.511
    F_KN = (1 / (1 + alpha * (1 - math.cos(angle)))) ** 2 * (1 + (alpha ** 2 * (1 - math.cos(angle)) ** 2) /
                                                             ((1 + alpha * (1 - math.cos(angle))) * (
                                                                     1 + math.cos(angle) ** 2))) ** 2
    return F_KN


def Perfom_ComptonInteraction(particle):
    """ Perform Compton Interactions """
    scatteredPhoton = Particle()
    scatteredPhoton.type = "photon"
    comptonElectron = Particle()
    comptonElectron.type = "electron"

    """ Electrons do not go through compton interactions """
    if particle.type == "electron":
        return None, particle

    """ Photon scattering angle ranges from 0 to 180. Here we use uniform distribution for simplicity """
    PhotonScatterTheta = np.random.uniform(0, math.pi)
    PhotonScatterPhi = np.random.uniform(0., 2*math.pi)
    ElectronScatterPhi = math.fmod(PhotonScatterPhi + math.pi, 2 * math.pi)

    """ energy has to be in MV """
    alpha_i = particle.energy / 0.511
    scatteredPhoton.energy = alpha_i / (1 + alpha_i * (1 - math.cos(PhotonScatterTheta))) * 0.511
    comptonElectron.energy = alpha_i ** 2 * (1 - math.cos(PhotonScatterTheta)) / \
                             (1 + alpha_i * (1 - math.cos(PhotonScatterTheta))) * 0.511

    """ Calculate the electron scattering angle """
    ElectronScatterTheta = math.atan2(math.sin(PhotonScatterTheta), (1+alpha_i)*(1 - math.cos(PhotonScatterTheta)))

    """ Update particle position """
    incidentParticleDir = NormalizeCalculateDirectionVector(particle.direction[0],
                                                            particle.direction[1],
                                                            particle.direction[2]).tolist()

    scatteredPhoton.position[0] = particle.position[0] + particle.mfp * incidentParticleDir[0]
    scatteredPhoton.position[1] = particle.position[1] + particle.mfp * incidentParticleDir[1]
    scatteredPhoton.position[2] = particle.position[2] + particle.mfp * incidentParticleDir[2]

    """ In a classical model both photons and electrons scatter from the same point in space """
    comptonElectron.position = [scatteredPhoton.position[0], scatteredPhoton.position[1], scatteredPhoton.position[2]]

    """ Update particles direction """
    """ scattered photon: """
    rotPhotonOperator = CalculateRotationOperator(PhotonScatterTheta, PhotonScatterPhi)
    scatteredPhoton.direction = rotPhotonOperator.dot(incidentParticleDir).tolist()

    """ scattered electron """
    rotElectronOperator = CalculateRotationOperator(ElectronScatterTheta, ElectronScatterPhi)
    comptonElectron.direction = rotElectronOperator.dot(incidentParticleDir).tolist()

    return [scatteredPhoton, comptonElectron]


def SimulateComptonScenario(ParticleCount, ParticleEnergy, NoComptronEventsPerParticle, material="water"):
    """ Simulate Compton Scattering interaction """

    """ Prepare containers to save particles trajectories and newly generated particles """
    PhotonWaves = []
    ElectronWaves = []

    """ Generate N Particles with uniform distribution of angles and gamma distribution of energies """
    photons = GeneratePhotonParticle(ParticleCount, ParticleEnergy)
    electrons = [None] * ParticleCount

    """ Calculate Mean free paths of all photons """
    for photon in photons:
        Calculate_MeanFreePath(photon, material)

    """ Calculate max range of all electrons """
    for electron in electrons:
        CalculateElectronRange(electron)

    """ Append initial seed of particles """
    PhotonWaves.append(photons)
    ElectronWaves.append(electrons)

    """ Perform compton interactions "GenCount" times """
    # fig, ax = plt.subplots()
    energyList = []
    for photon in photons:
        energyList.append(photon.energy)
    #ax.hist(energyList, 50)

    for iGen in range(NoComptronEventsPerParticle):

        """ Perform Compton interaction with particles. This function replaces particles in previous generations """
        NewPhotons = []
        NewElectrons = []
        energyList.clear()
        for photon in photons:
            newPhoton, newElectron = Perfom_ComptonInteraction(photon)

            if photon.mfp > 0.01:
                NewPhotons.append(copy.deepcopy(newPhoton))
            else:
                NewPhotons.append(copy.deepcopy(photon))

            NewElectrons.append(copy.deepcopy(newElectron))
            energyList.append(newPhoton.energy)
        # y, x = np.histogram(energyList, 30)
        # ax.plot(x[:-1],y)
        """ Calculate Mean free paths of all photons """
        for photon in NewPhotons:
            Calculate_MeanFreePath(photon, material)

        """ Calculate max range of all electrons """
        for electron in NewElectrons:
            CalculateElectronRange(electron)

        """ replace parents particles with offsprings """
        photons = NewPhotons
        electrons = NewElectrons

        """ Save results in the super containers. For consistency append empty electrons objects """
        PhotonWaves.append(photons)
        ElectronWaves.append(electrons)

    # plt.show()

    """ render photons and electrons independently """
    #RenderParticleTrajectories3D([PhotonWaves[0]], [ElectronWaves[0]])
    return PhotonWaves, ElectronWaves


def RenderParticleTrajectories3D(PhotonWaves, ElectronWaves):
    """ Prepare 3d rendering space """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Hide axes ticks
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.tick_params(bottom=False, top=True, left=True, right=True)
    #ax.view_init(elev=0, azim=180)
    plt.title('Particle Trajectories', loc='left')

    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.grid(False)

    """ Loop over particles and then draw an arrow between position and direction """
    if PhotonWaves is not None:
        for wave in PhotonWaves:
            for iParticle in wave:
                p = Arrow3D([iParticle.position[0], iParticle.position[0] + iParticle.mfp * iParticle.direction[0]],
                            [iParticle.position[1], iParticle.position[1] + iParticle.mfp * iParticle.direction[1]],
                            [iParticle.position[2], iParticle.position[2] + iParticle.mfp * iParticle.direction[2]],
                            mutation_scale=5, lw=1, arrowstyle="-|>", color="b")
                ax.add_artist(p)

    if ElectronWaves is not None:
        for wave in ElectronWaves:
            for iParticle in wave:
                if iParticle is not None:
                    p = Arrow3D([iParticle.position[0], iParticle.position[0] + iParticle.mfp * iParticle.direction[0]],
                                [iParticle.position[1], iParticle.position[1] + iParticle.mfp * iParticle.direction[1]],
                                [iParticle.position[2], iParticle.position[2] + iParticle.mfp * iParticle.direction[2]],
                                mutation_scale=5, lw=1, ls=':', arrowstyle="-|>", color="y")
                    ax.add_artist(p)

    plt.show()


def GenereateHistogram(yVector, BinCount, xLabel, yLabel, Title, Notes):

    plt.subplots(figsize=(12, 9))

    """ Raw Histogram """
    plt.hist(yVector, bins=BinCount)

    """ Normalized Histogram """
    # plt.hist(yVector, bins=BinCount), density=True)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(Title+"\n"+Notes)

    """ Show Histogram """
    plt.show()

def plotComptonEnergies(particlecount, energy, numinteractions, material="water"):

    x, y = SimulateComptonScenario(particlecount, energy, numinteractions, material)
    EnergyList = []
    for interaction in range(numinteractions):
        tmplist = []
        for i in range(len(x[interaction])):
            if x[interaction][i].energy < 0:
                pass
            else:
                tmplist.append(x[interaction][i].energy)
        EnergyList.append(tmplist)

    ElectronEnergyList = []
    for interaction in range(numinteractions):
        tmplist = []
        for i in range(len(x[interaction])):
            # if y[interaction][i].energy < 0:
            #     pass
            tmplist.append(y[interaction + 1][i].energy)
        ElectronEnergyList.append(tmplist)


    fig, axs = plt.subplots(numinteractions, sharex=True, sharey=True)
    for i in range(numinteractions):
        axs[i].hist(ElectronEnergyList[i], bins=25)
        axs[i].set_title("Interaction #{}".format(i))
    plt.suptitle("Avg Energy = {} MeV \n Material = {}".format(energy, material))
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Counts")
    plt.tight_layout()
    # plt.savefig("{}MeV_{}.png".format(energy, material))
    plt.show()

def main():
    energies = [0.1, 10, 100]
    materials = ["air", "water", "lead"]
    for material in materials:
        for energy in energies:
            plotComptonEnergies(1000, energy, 5, material=material)


if __name__ == "__main__":
    main()
