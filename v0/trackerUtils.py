"""Formulas and fixed inputs of trackerOptimisation v0.

The fixed inputs come first, each followed by its unit and its source. Then the mapping of the unbounded optimiser
parameters to physical ones, then the detector model: material, conversion, scattering, the spatial likelihood
ratio, the ACD, the constraints and the loss. Everything is a differentiable JAX function, so jax.grad gives the
gradient of the loss with respect to every design parameter.

The notebook imports this module. The functions read the inputs from this module's namespace when they are called
(or traced, for the jitted ones), so to change an input for an experiment set it on the module, for example
`import trackerUtils; trackerUtils.downstreamScatteringWeight = 0.0`, before the first call that uses it.

Naming: camelCase everywhere, no unit suffixes. Units: lengths in cm, energies in MeV, times in s, solid angles in sr,
fluxes in 1/(cm^2 s). Angular variances are in rad^2.
"""
import jax
import jax.numpy as jnp
import numpy as np

# 64-bit floats: the significance formula subtracts large, similar numbers, and float32 loses precision there.
jax.config.update("jax_enable_x64", True)


# ======================================================================================================
# Fixed inputs. These are not optimised. If the optimiser could change them it would simply run to the bounds.
# ======================================================================================================
# Every number is followed by its unit and its source: a physical constant, a link, or "Placeholder" when no source has
# been found yet. Bibliography keys (in brackets) are in references/bibliography.bib, with local copies of the sources.
# Reference values and the open decisions for each placeholder are in PARAMETERS.md.

# Radiation lengths of the materials.
radiationLengthTungsten = 0.3504       # cm. Physical constant: tungsten radiation length [pdg2024tungsten] https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/tungsten_W.html
radiationLengthScintillator = 42.4     # cm. Physical constant: polyvinyltoluene scintillator is 42.54 cm [pdg2024pvt] https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/polyvinyltoluene.html. 42.4 is slightly off, to update.

# Pair conversion probability after x radiation lengths is 1 - exp(-7/9 * x), the high energy limit.
pairConversionCoefficientPerRadiationLength = 7.0 / 9.0   # Physical constant: sigma = 7/9 A/(X0 N_A), PDG Eq. 34.32 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf. Accurate to a few percent only down to 1 GeV, so optimistic at 100 MeV.

numberOfLayers = 10                    # Discrete, so it is scanned by hand rather than optimised. Placeholder. Reference: LAT 18 x-y layers, 16 with tungsten [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.2.1; HERD FIT 7 double layers [farina2021herd] https://doi.org/10.22323/1.395.0651 section 2.
photonEnergy = 100.0                   # MeV. Only used by the monochromatic source (sourceType = "monochromatic"). Placeholder. The power-law source has an energy axis instead.
electronMass = 0.511                   # MeV. Physical constant: electron mass, PDG https://pdg.lbl.gov/2024/
detectorSideLength = 40.0              # cm. Placeholder. Reference: LAT is 1.8 m wide [atwood2009lat] https://arxiv.org/abs/0902.1089 Fig. 1. Decision pending (PARAMETERS.md).
passiveRadiationLengthsPerLayer = 0.014   # X0 per x-y layer. The material of a layer that is not the foil (silicon detectors, supports, electronics). It absorbs, converts and scatters like the foil. Value of the LAT tracker, 0.014 X0 per x-y plane [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 2. A different technology would differ, decision pending.

# Fluxes and observation conditions.
signalPhotonFlux = 1e-3                # 1/(cm^2 s), integral flux above energyMinimum for the power law. Placeholder, very bright. Reference: 1e-7 above 100 MeV for a faint high-latitude source [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 1 note d.
diffusePhotonFlux = 1e-2               # 1/(cm^2 s) over the field of view. Placeholder. Reference: 1.5e-5 per cm^2 s sr above 100 MeV at high latitude, index 2.1 [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 1 note e.
chargedParticleFlux = 1.0              # 1/(cm^2 s). Placeholder. Reference: about 0.1 inferred from the LAT raw trigger rate of 2-4 kHz over about 3.2e4 cm^2 [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.2.3.
exposureDuration = 1e4                 # s. Placeholder. Reference: LAT 1-year survey [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 1 note d; HERD 1, 5 and 10 years [farina2021herd] https://doi.org/10.22323/1.395.0651 Fig. 3.
fieldOfViewSolidAngle = 1.0            # sr. Placeholder. Reference: LAT 2.4 sr at 1 GeV [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.1.

# Source spectrum, as in the Fermi-LAT and HERD sensitivity calculations: a power law dN/dE ~ E^-index truncated to [energyMinimum, energyMaximum]
# and binned in energy. The energy is assumed known perfectly (no energy migration), so every energy bin is its own set of classes and the test
# statistic is the sum over the bins. With the power-law source the three fluxes above are integral fluxes above energyMinimum. With the
# monochromatic source they are total fluxes at photonEnergy, which reproduces the first version of the model.
sourceType = "powerLaw"                # "powerLaw" or "monochromatic". The old monochromatic source is kept to compare with.
energyMinimum = 100.0                  # MeV. Placeholder. Above the Compton and pair crossover with margin, see the README for the reasoning.
energyMaximum = 10000.0                # MeV. Placeholder. Shower containment, statistics and backsplash, see the README for the reasoning.
binsPerDecade = 4                      # Energy bins per decade, as in the HERD ICRC 2021 sensitivity [farina2021herd] https://doi.org/10.22323/1.395.0651 section 6.
signalSpectralIndex = 2.0              # Index 2 power law, as in the Fermi-LAT sensitivity [fermilat2013performance] https://s3df.slac.stanford.edu/data/fermi/groups/canda/archive/pass8v6/lat_Performance.htm and HERD [farina2021herd] https://doi.org/10.22323/1.395.0651 section 6.
diffuseSpectralIndex = 2.1             # Diffuse photon background, high-latitude index 2.1 [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 1 note e.
chargedSpectralIndex = 2.7             # Charged cosmic-ray background. Placeholder, no source found (a steep power law).

# PSF model. "kalman": the direction of each pair member comes from a backward Kalman filter over all the hits below the conversion vertex,
# with multiple scattering as process noise, the mean radiative energy loss along the track, the energy sharing of the pair integrated over a
# few nodes (a mixture of Gaussians, so the PSF has tails) and the two tracks averaged with equal weights, as in the HERD reconstruction.
# "window": the earlier heuristic (one combined scatterer with full weight for the layers that carry the fitted hits and a toy weight below),
# kept to compare with.
psfModel = "kalman"                    # "kalman" or "window". Use setPsfModel to change it.
minimumTrackEnergy = 10.0              # MeV. Placeholder, no source: a track below this energy is not reconstructed (it ranges out or scatters wildly). It cuts the energy sharing of the pair.
numberOfEnergySharingNodes = 6         # Quadrature nodes over the energy sharing of the pair. Numerical choice.
pairEnergySharingCoefficient = 4.0 / 3.0   # Energy sharing of the pair, dsigma/dx ~ 1 - 4/3 x (1 - x), complete screening, PDG Eq. 34.31 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf. Its integral over 0 to 1 is 7/9, the conversion coefficient.
highlandConstant = 13.6                # MeV. Highland-Lynch-Dahl, PDG Eq. 34.16 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf
highlandLogCoefficient = 0.038         # PDG Eq. 34.16 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf
trackFilterPriorPositionVariance = 100.0   # cm^2. Uninformative prior of the Kalman filter, far above any hit error. Numerical choice, no physical source.
trackFilterPriorSlopeVariance = 10.0       # rad^2. Uninformative prior of the Kalman filter, far above any direction error. Numerical choice, no physical source.

# Bounds of the energy range, checked against the design by computeBoundWarnings (not part of the loss).
comptonPairCrossoverEnergy = 10.0      # MeV. Placeholder, from memory: Compton and pair production cross sections are equal at about 10 MeV in tungsten. To check against NIST XCOM.
criticalEnergyTungsten = 7.97          # MeV. Physical constant: critical energy of tungsten for e- [pdg2024tungsten] https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/tungsten_W.html
showerMaximumPhotonOffset = 0.5        # C_gamma = +0.5 in t_max = ln(E/E_c) + C_gamma for a photon-induced shower, PDG Eq. 34.36 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf

# Constraints that keep the optimum finite. Without them the optimiser drives the strip pitch to its
# lower bound and makes the detector as tall and channel-rich as allowed.
channelBudget = 5e4                    # number of strips. Placeholder, no source found. Scales with detector size, decision pending.
maximumHeight = 30.0                   # cm. Placeholder. Reference: LAT whole instrument is 0.72 m high, tracker alone not found [atwood2009lat] https://arxiv.org/abs/0902.1089 Fig. 1.

# A track needs hits in the conversion layer and in the layers below it to be reconstructed.
hitsRequiredForTracking = 3            # Hits per track: the conversion layer and the two layers below. HERD needs at least 3 hits per particle in each of X and Y [farina2021herd] https://doi.org/10.22323/1.395.0651 section 4. LAT: the first 2 planes after the conversion must be measured [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 2.
minimumDownstreamLayersForReconstruction = hitsRequiredForTracking - 1   # Derived: layers below the conversion layer that carry the other hits. 1: the conversion layer carries the first hit.
fullWeightLayersBelow = hitsRequiredForTracking - 1   # Derived: the layers below that carry fitted hits, so their material scatters the measured direction with full weight. 1: as above.

# Multiple scattering: material below the layers that carry the fitted hits is counted with this reduced weight (toy value).
downstreamScatteringWeight = 1.0 / 3.0   # Toy value, no source. Applies only beyond the fitted hits. The fit-covariance model is meant to replace it.

# Signal region: a cone around the source with this radius, in units of the angular resolution.
signalConeRadiusInSigma = 2.0          # Only used by the single-cone check. The objective has no cone. Analysis convention, no source. The Gaussian 68% radius is 1.51 sigma (computed from 1 - exp(-r^2/2)).

# Spatial likelihood ratio (the objective). The sky around the source is integrated out to the field of view.
numberOfReconstructableLayers = numberOfLayers - minimumDownstreamLayersForReconstruction   # Layers that can be a conversion layer. Derived.
fieldOfViewRadius = float(np.arccos(1.0 - fieldOfViewSolidAngle / (2.0 * np.pi)))   # rad. Derived, exact: half-angle of a circular cap of that solid angle, Omega = 2 pi (1 - cos(theta)). Mathematical result, no source.
radialGridMinimum = 1e-4               # rad. Innermost radius of the logarithmic integration grid. Numerical choice, no physical source.
radialGridPoints = 2000                # Numerical choice, checked against the analytic limit in the checks below.
minimumSignalPhotons = 10.0            # Soft constraint. Fermi-LAT sensitivity definition: at least 10 photons [fermilat2013performance] https://s3df.slac.stanford.edu/data/fermi/groups/canda/archive/pass8v6/lat_Performance.htm; HERD [farina2021herd] https://doi.org/10.22323/1.395.0651 section 6.

# Conversion-layer classes ("pseudo-detectors"). confusionMatrix[assigned, true] is the probability that a photon
# that converted in layer `true` is assigned to layer `assigned`, so each column sums to one.
labelConfusionMatrix = jnp.eye(numberOfReconstructableLayers)   # Perfect labels. The default and an upper bound. Definition, no source.
noLabelConfusionMatrix = jnp.full((numberOfReconstructableLayers,) * 2, 1.0 / numberOfReconstructableLayers)   # No labels. A lower bound. Definition: every class equally likely.
# Charged-particle background that gets through the ACD. By assumption nothing rejects it afterwards: the tracker gives no further
# rejection. It is a minimum-ionising particle crossing the whole instrument, so the foils neither attenuate it nor change its total
# rate. Which conversion-layer class it falls in is an assumption (no source): it fakes a conversion by interacting in the material of
# a layer (delta rays, bremsstrahlung, hadronic), so its share follows the radiation lengths of each layer, not the conversion
# probability. An even split was exploitable: concentrating the signal in one class gave a free rejection by the number of classes.

# Figure of merit: the test statistic TS = 2 ln(L(signal + background) / L(background)), as in Fermi-LAT.
detectionTestStatistic = 25.0          # TS of a 5 sigma detection, Fermi-LAT [fermilat2013performance] https://s3df.slac.stanford.edu/data/fermi/groups/canda/archive/pass8v6/lat_Performance.htm and [fermipy2017sensitivity] https://fermipy.readthedocs.io/en/v1.2/advanced/sensitivity.html
constraintPenaltyWeight = 20.0         # Weight of the constraint penalty against -ln(TS). Numerical choice, no physical source. 20 = 2 x 10, the weight used with -ln(Z), so the optimum is unchanged because TS = Z^2.

# Plastic scintillator ACD.
minimumIonisingEnergyLossPerLength = 2.019   # MeV/cm, energy deposited by a minimum ionising particle. Physical constant: polyvinyltoluene, 1.956 MeV cm^2/g at 1.032 g/cm^3 [pdg2024pvt] https://pdg.lbl.gov/2024/AtomicNuclearProperties/HTML/polyvinyltoluene.html
maximumVetoEfficiency = 0.9995         # Approximation, chosen between the previous placeholder 0.999 and the LAT tile efficiency above 0.9997 averaged over the ACD area [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.2.3 and Table 4.
vetoTurnOnWidthFraction = 0.08         # Width of the efficiency turn-on, as a fraction of the deposit. Calibrated to LAT: a 1 cm tile (2.0 MeV for a MIP) at the ground-analysis threshold of about 0.3 MIP then has an inefficiency of 6.6e-4, within a factor 2 of the LAT tile requirement of 3e-4 (efficiency above 0.9997) [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.2.3 and Table 4. It was a toy 0.25, which capped the ACD efficiency near 98% whatever the thickness and threshold.
accidentalVetoScale = 0.1              # MeV. Sets how fast noise-induced dead time falls with threshold. Toy value, no source.

# Allowed range of each free parameter. Optimisation runs in an unbounded space (see below).
parameterBounds = {
    # cm, one value per layer. Lower bound 0: a bare layer, whose absorption, conversion and scattering come from the explicit passive material. Upper bound 0.2 is a placeholder (LAT back foils are 0.072 cm [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 2).
    "converterThickness": (0.0, 0.2),
    # cm. Placeholder range. Reference: LAT about 3.2 cm, from pitch 0.0228 cm / 0.0071 [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 2.
    "layerSpacing": (0.5, 5.0),
    # cm. Placeholder range. Reference: LAT 228 µm = 0.0228 cm [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 2; HERD FIT about 250 µm [farina2021herd] https://doi.org/10.22323/1.395.0651 section 2.
    "stripPitch": (0.005, 0.1),
    # cm. Placeholder range. Reference: LAT 1.0 cm [atwood2009lat] https://arxiv.org/abs/0902.1089 Table 4.
    "acdThickness": (0.5, 3.0),
    # MeV. Placeholder range. Reference: LAT 0.45 MIP on board and about 0.30 MIP on the ground [atwood2009lat] https://arxiv.org/abs/0902.1089 section 2.2.3.
    "acdThreshold": (0.05, 1.0),
}



# ======================================================================================================
# Mapping of the unbounded optimiser parameters to the allowed ranges.
# ======================================================================================================
def mapUnboundedToRange(unboundedValue, lowerBound, upperBound):
    return lowerBound + (upperBound - lowerBound) * jax.nn.sigmoid(unboundedValue)


def mapUnboundedToPhysicalParameters(unboundedParameters):
    return {
        parameterName: mapUnboundedToRange(unboundedParameters[parameterName], *parameterBounds[parameterName])
        for parameterName in parameterBounds
    }


# ======================================================================================================
# Detector model.
# ======================================================================================================
# Layers below each layer, used both to decide if a track can be reconstructed and for the lever arm.
numberOfLayersBelow = numberOfLayers - 1 - jnp.arange(numberOfLayers)
numberOfEnergyBins = int(round(binsPerDecade * np.log10(energyMaximum / energyMinimum)))   # 10: logarithm base, mathematical.


def computeEnergyGrid(sourceTypeName):
    """Energy bins of the source: the energy that sets the PSF of each bin, and the share of each integral flux that falls in it."""
    if sourceTypeName == "monochromatic":
        return dict(psfEnergy=np.array([photonEnergy]), signalFraction=np.ones(1), diffuseFraction=np.ones(1), chargedFraction=np.ones(1))
    edges = np.geomspace(energyMinimum, energyMaximum, numberOfEnergyBins + 1)

    def fractionOfIntegralFluxInBin(index):
        # The integral flux above energyMinimum of E^-index is proportional to energyMinimum^(1 - index) / (index - 1). The share in a bin follows.
        return (edges[:-1] / energyMinimum) ** (1.0 - index) - (edges[1:] / energyMinimum) ** (1.0 - index)

    # The PSF variance of a bin is its mean over the bin, weighted by the signal spectrum. It scales as 1/E^2, so the energy that sets
    # the PSF of the bin is the one with 1/E^2 equal to the mean of 1/E^2. (The strip term does not depend on the energy.)
    gamma = signalSpectralIndex
    meanInverseEnergySquared = ((edges[:-1] ** (-gamma - 1.0) - edges[1:] ** (-gamma - 1.0)) / (gamma + 1.0)) / (
        (edges[:-1] ** (1.0 - gamma) - edges[1:] ** (1.0 - gamma)) / (gamma - 1.0)
    )
    return dict(
        psfEnergy=meanInverseEnergySquared ** -0.5,
        signalFraction=fractionOfIntegralFluxInBin(signalSpectralIndex),
        diffuseFraction=fractionOfIntegralFluxInBin(diffuseSpectralIndex),
        chargedFraction=fractionOfIntegralFluxInBin(chargedSpectralIndex),
    )


def computeEnergySharing(binEnergies):
    """Energy sharing of the pair for each energy: quadrature nodes (the fraction of the photon energy in one track), normalised weights,
    and the efficiency of the minimum track energy cut, relative to all the pairs."""
    if psfModel == "window":
        return np.full((len(binEnergies), 1), 0.5), np.ones((len(binEnergies), 1)), np.ones(len(binEnergies))   # One equal-sharing pair, no cut.
    unitNodes, unitWeights = np.polynomial.legendre.leggauss(numberOfEnergySharingNodes)   # On [-1, 1].
    fractionMinimum = minimumTrackEnergy / np.asarray(binEnergies)[:, None]
    nodes = fractionMinimum + (1.0 - 2.0 * fractionMinimum) * (unitNodes[None, :] + 1.0) / 2.0   # Between the cuts, 2: the cut at both ends.
    densityWeights = unitWeights[None, :] * (1.0 - 2.0 * fractionMinimum) / 2.0 * (1.0 - pairEnergySharingCoefficient * nodes * (1.0 - nodes))
    efficiency = densityWeights.sum(axis=1) / (1.0 - pairEnergySharingCoefficient / 6.0)   # 6: integral of x (1 - x) over 0 to 1 is 1/6, so the total is 7/9.
    return nodes, densityWeights / densityWeights.sum(axis=1, keepdims=True), efficiency


def setPsfModel(psfModelName):
    """Choose "kalman" or "window". Call before the first call of a jitted function."""
    global psfModel
    psfModel = psfModelName
    setSourceType(sourceType)


def setSourceType(sourceTypeName):
    """Choose "powerLaw" or "monochromatic". Call before the first call of a jitted function: they read these arrays when traced."""
    global sourceType, binPsfEnergy, binSignalFraction, binDiffuseFraction, binChargedFraction
    global binSharingNodes, binSharingWeight, binSharingEfficiency
    sourceType = sourceTypeName
    energyGrid = computeEnergyGrid(sourceTypeName)
    binPsfEnergy = jnp.asarray(energyGrid["psfEnergy"])
    binSignalFraction = jnp.asarray(energyGrid["signalFraction"])
    binDiffuseFraction = jnp.asarray(energyGrid["diffuseFraction"])
    binChargedFraction = jnp.asarray(energyGrid["chargedFraction"])
    sharingNodes, sharingWeights, sharingEfficiency = computeEnergySharing(energyGrid["psfEnergy"])
    binSharingNodes, binSharingWeight, binSharingEfficiency = jnp.asarray(sharingNodes), jnp.asarray(sharingWeights), jnp.asarray(sharingEfficiency)


setSourceType(sourceType)


def computeLayerMaterial(converterThickness):
    """Material in each layer, and above and below it, in radiation lengths."""
    # Material in one layer: tungsten converter plus one silicon plane.
    converterRadiationLengths = converterThickness / radiationLengthTungsten
    radiationLengthsPerLayer = converterRadiationLengths + passiveRadiationLengthsPerLayer   # Foil and passive material of the layer.

    # Material the photon crosses before reaching a layer, and material the pair crosses after leaving it.
    radiationLengthsAboveLayer = jnp.cumsum(radiationLengthsPerLayer) - radiationLengthsPerLayer
    radiationLengthsBelowLayer = jnp.sum(radiationLengthsPerLayer) - jnp.cumsum(radiationLengthsPerLayer)
    return converterRadiationLengths, radiationLengthsAboveLayer, radiationLengthsBelowLayer


def computeConversionProbabilityPerLayer(converterRadiationLengths, radiationLengthsAboveLayer):
    """Probability that the photon converts in each layer and leaves a reconstructable track."""
    hasEnoughDownstreamLayers = (numberOfLayersBelow >= minimumDownstreamLayersForReconstruction).astype(float)

    # Photon survives the material above (foils and passive material), then converts in the foil or the passive material of this layer.
    return (
        jnp.exp(-pairConversionCoefficientPerRadiationLength * radiationLengthsAboveLayer)
        * (1.0 - jnp.exp(-pairConversionCoefficientPerRadiationLength * (converterRadiationLengths + passiveRadiationLengthsPerLayer)))
        * hasEnoughDownstreamLayers
    )


def computeExpectedLayerPathAfterConversion(layerRadiationLengths):
    """Radiation lengths of its own layer (foil and passive material) that the pair crosses, averaged over where the photon converts.

    The photon is absorbed as it goes, so the conversion depth t in the layer has density proportional to exp(-kappa t).
    With u = kappa x the mean depth is x g(u), where g(u) = (1 - (1 + u) exp(-u)) / (u (1 - exp(-u))), and g -> 1/2 for a thin layer.
    The passive material is always there, so x is never 0 and g is never 0/0.
    """
    kappa = pairConversionCoefficientPerRadiationLength
    u = kappa * layerRadiationLengths
    meanDepthFraction = (-jnp.expm1(-u) - u * jnp.exp(-u)) / (u * -jnp.expm1(-u))
    return layerRadiationLengths * (1.0 - meanDepthFraction)


def computeScatteringRadiationLengths(converterRadiationLengths):
    """Radiation lengths that scatter the measured direction of a pair that converted in each layer, as one combined scatterer."""
    radiationLengthsPerLayer = converterRadiationLengths + passiveRadiationLengthsPerLayer   # Foil and passive material of each layer.
    cumulative = jnp.concatenate([jnp.zeros(1), jnp.cumsum(radiationLengthsPerLayer)])   # cumulative[k] is the material of layers 0 to k-1.
    layerIndex = jnp.arange(numberOfLayers)
    endOfFittedLayers = jnp.minimum(layerIndex + 1 + fullWeightLayersBelow, numberOfLayers)   # 1: first layer below the conversion layer.
    materialInFittedLayers = cumulative[endOfFittedLayers] - cumulative[layerIndex + 1]   # Full weight: these layers carry the fitted hits.
    materialBeyondFittedLayers = cumulative[numberOfLayers] - cumulative[endOfFittedLayers]
    return (
        computeExpectedLayerPathAfterConversion(radiationLengthsPerLayer)   # The rest of its own layer, which carries the first hit.
        + materialInFittedLayers
        + downstreamScatteringWeight * materialBeyondFittedLayers
    )


def computePerLayerAngularVariance(converterRadiationLengths, layerSpacing, stripPitch, energy=None):
    """Angular variance (rad^2) of the reconstructed direction for a photon of the given energy that converted in each layer."""
    energy = photonEnergy if energy is None else energy   # The monochromatic energy by default.
    # Three independent contributions, added in quadrature (variances in rad^2), evaluated per conversion layer.
    # 1. Multiple scattering, Highland-Lynch-Dahl with the log term (PDG Eq. 34.16). PDG says to apply it once to the combined
    #    scatterer, because adding separate theta0 in quadrature is systematically too small. Each pair member carries half the energy.
    scatteringRadiationLengths = computeScatteringRadiationLengths(converterRadiationLengths)
    highlandLogCorrection = (1.0 + 0.038 * jnp.log(scatteringRadiationLengths)) ** 2   # 0.038: PDG Eq. 34.16 [pdg2024passage] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-passage-particles-matter.pdf
    multipleScatteringVariance = (13.6 / (energy / 2.0)) ** 2 * scatteringRadiationLengths * highlandLogCorrection   # 13.6 MeV: same equation. 2.0: equal energy sharing, an approximation.
    # 2. Intrinsic opening angle of the pair.
    pairOpeningAngleVariance = (electronMass / energy) ** 2   # m_e / E: characteristic scale of the pair opening angle, an approximation with no single source.
    # 3. Strip resolution (pitch / sqrt(12) for a uniform hit distribution) over the lever arm to the last layer.
    leverArm = jnp.maximum(numberOfLayersBelow, 1) * layerSpacing   # cm
    stripResolutionVariance = 2.0 * (stripPitch / jnp.sqrt(12.0) / leverArm) ** 2   # 12: variance of a uniform distribution over one pitch is pitch^2/12, standard result (see PDG detectors review [pdg2024detectors] https://pdg.lbl.gov/2024/reviews/rpp2024-rev-particle-detectors-accel.pdf). 2.0: toy factor, no source.

    return multipleScatteringVariance + pairOpeningAngleVariance + stripResolutionVariance


def computePerLayerAngularVarianceInEnergyBins(converterRadiationLengths, layerSpacing, stripPitch):
    """Angular variance (rad^2), one row per energy bin and one column per conversion layer."""
    return jax.vmap(
        lambda energy: computePerLayerAngularVariance(converterRadiationLengths, layerSpacing, stripPitch, energy)
    )(binPsfEnergy)


def computeTrackSlopeVarianceAtVertex(trackEnergy, layerRadiationLengths, layerSpacing, stripPitch):
    """Variance (rad^2, one projection) of the direction at the conversion vertex of a track of the given energy, for every conversion layer.

    A backward Kalman filter over the hits below the vertex. The state is the position and the slope at a plane, the hit error is
    pitch / sqrt(12), and the multiple scattering of the material of each layer (at its plane) is the process noise on the slope. The
    track loses energy by radiation, E(t) = E exp(-t) in radiation lengths, so the scattering grows along it. As in the PDG advice, the
    Highland log term is applied once to the whole material of the track. The covariance does not depend on the hit values, so it is
    computed by a deterministic recursion (no simulated events). Hit efficiency is ideal, pattern recognition is not modelled.
    """
    numberOfPlanes = layerRadiationLengths.shape[0]
    ownPath = computeExpectedLayerPathAfterConversion(layerRadiationLengths)   # Rest of its own layer after the conversion.
    hitVariance = stripPitch**2 / 12.0   # 12: variance of a uniform distribution over one pitch.
    layerIndex = jnp.arange(numberOfPlanes)
    propagator = jnp.array([[1.0, -layerSpacing], [0.0, 1.0]])   # One plane upstream, for the backward filter.
    noiseShape = jnp.array([[layerSpacing**2, -layerSpacing], [-layerSpacing, 1.0]])   # A kick of the slope at a plane, seen from the next plane upstream.
    prior = jnp.diag(jnp.array([trackFilterPriorPositionVariance, trackFilterPriorSlopeVariance]))

    def varianceForConversionLayer(conversionLayer):
        isBelow = layerIndex > conversionLayer
        materialBelow = jnp.where(isBelow, layerRadiationLengths, 0.0)
        traversedBefore = ownPath[conversionLayer] + jnp.cumsum(materialBelow) - materialBelow   # Radiation lengths crossed before entering each layer.
        energyIn = trackEnergy * jnp.exp(-traversedBefore)
        logFactor = (1.0 + highlandLogCoefficient * jnp.log(ownPath[conversionLayer] + materialBelow.sum())) ** 2
        # Kick variance of a layer: Highland with the energy falling along the layer, the integral of dt / E(t)^2 = (exp(2 x) - 1) / (2 E_in^2).
        kickVariance = jnp.where(isBelow, highlandConstant**2 * jnp.expm1(2.0 * layerRadiationLengths) / 2.0 / energyIn**2 * logFactor, 0.0)   # 2.0: from the integral.
        ownKickVariance = highlandConstant**2 * jnp.expm1(2.0 * ownPath[conversionLayer]) / 2.0 / trackEnergy**2 * logFactor
        planeDescending = jnp.arange(numberOfPlanes - 1, -1, -1)
        kickBelowPlane = jnp.where(planeDescending < numberOfPlanes - 1, kickVariance[jnp.minimum(planeDescending + 1, numberOfPlanes - 1)], 0.0)

        def filterOnePlane(covariance, planeAndKick):
            plane, kick = planeAndKick
            predicted = jnp.where(plane == numberOfPlanes - 1, prior, propagator @ covariance @ propagator.T + kick * noiseShape)
            gain = predicted[:, 0] / (predicted[0, 0] + hitVariance)
            updated = predicted - jnp.outer(gain, predicted[0, :])
            return updated, updated[1, 1]

        _, slopeVarianceDescending = jax.lax.scan(filterOnePlane, jnp.zeros((2, 2)), (planeDescending, kickBelowPlane))
        # The filtered slope at the conversion plane is the one after the scattering of its own layer. The direction at the vertex adds that kick.
        return slopeVarianceDescending[numberOfPlanes - 1 - conversionLayer] + ownKickVariance

    return jax.vmap(varianceForConversionLayer)(layerIndex)


def computePsfComponentVariance(converterRadiationLengths, layerSpacing, stripPitch):
    """Variance (rad^2, per axis) of the reconstructed direction of each PSF component: one array with energy bins, layers and energy-sharing nodes."""
    if psfModel == "window":
        return computePerLayerAngularVarianceInEnergyBins(converterRadiationLengths, layerSpacing, stripPitch)[:, :, None]
    layerRadiationLengths = converterRadiationLengths + passiveRadiationLengthsPerLayer
    binEnergy = binPsfEnergy[:, None]
    trackEnergies = jnp.stack([binSharingNodes * binEnergy, (1.0 - binSharingNodes) * binEnergy], axis=-1)   # Bins, nodes, the two tracks.
    trackVariance = jax.vmap(jax.vmap(jax.vmap(
        lambda energy: computeTrackSlopeVarianceAtVertex(energy, layerRadiationLengths, layerSpacing, stripPitch)
    )))(trackEnergies)   # Bins, nodes, tracks, layers.
    # The photon direction is the plain average of the two track directions (as in the HERD reconstruction): variance (V_a + V_b) / 4.
    # The intrinsic opening angle of the pair is added as before.
    pairVariance = (trackVariance[:, :, 0, :] + trackVariance[:, :, 1, :]) / 4.0 + ((electronMass / binPsfEnergy) ** 2)[:, None, None]   # 4.0: average of two tracks.
    return jnp.transpose(pairVariance, (0, 2, 1))   # Bins, layers, nodes.


def computeEffectiveVarianceInEnergyBins(converterRadiationLengths, layerSpacing, stripPitch):
    """Mean angular variance (rad^2) over the PSF components, one row per energy bin and one column per layer. For reporting."""
    return jnp.sum(computePsfComponentVariance(converterRadiationLengths, layerSpacing, stripPitch) * binSharingWeight[:, None, :], axis=2)


def computeEffectiveAngularVariance(converterRadiationLengths, conversionProbabilityPerLayer, layerSpacing, stripPitch):
    """Variance averaged over conversion layers. Only the single-cone check uses this, the objective keeps each layer."""
    angularVariancePerLayer = computePerLayerAngularVariance(converterRadiationLengths, layerSpacing, stripPitch)
    return jnp.sum(conversionProbabilityPerLayer * angularVariancePerLayer) / jnp.sum(conversionProbabilityPerLayer)


def computeSignalConeFractions(effectiveAngularVariance):
    """How much signal the signal cone keeps, and how much of the background it lets in."""
    signalConeSolidAngle = jnp.pi * signalConeRadiusInSigma**2 * effectiveAngularVariance   # sr
    backgroundFractionInsideCone = signalConeSolidAngle / fieldOfViewSolidAngle
    # Fraction of a 2D Gaussian inside a circle of the given radius in sigma.
    signalContainmentFraction = 1.0 - jnp.exp(-(signalConeRadiusInSigma**2) / 2.0)   # 2.0: from the 2D Gaussian containment, mathematical result.
    return signalContainmentFraction, backgroundFractionInsideCone


def computeAcdResponse(acdThickness, acdThreshold):
    """Charged-particle veto efficiency, photon survival through the ACD, and livetime lost to noise."""
    energyDepositedInAcd = minimumIonisingEnergyLossPerLength * acdThickness   # MeV
    # Smooth turn-on of the charged-particle detection efficiency around the threshold.
    vetoEfficiency = maximumVetoEfficiency * jax.nn.sigmoid(
        (energyDepositedInAcd - acdThreshold) / (vetoTurnOnWidthFraction * energyDepositedInAcd)
    )
    # A photon that converts inside the ACD is vetoed by mistake ("self-veto"), so a thick ACD costs signal.
    photonSurvivalProbabilityThroughAcd = jnp.exp(
        -pairConversionCoefficientPerRadiationLength * acdThickness / radiationLengthScintillator
    )
    # Low thresholds trigger the veto on noise, which removes livetime. Toy model, falls exponentially.
    livetimeFraction = jnp.exp(-jnp.exp(-acdThreshold / accidentalVetoScale))
    return vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction


def computeExpectedCounts(
    totalConversionProbability, signalContainmentFraction, backgroundFractionInsideCone,
    vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction,
):
    """Expected signal and background counts over the exposure."""
    detectorArea = detectorSideLength**2   # cm^2
    exposureFactor = detectorArea * exposureDuration * livetimeFraction

    signalCount = (
        signalPhotonFlux * exposureFactor * totalConversionProbability
        * photonSurvivalProbabilityThroughAcd * signalContainmentFraction
    )
    # Two backgrounds: diffuse photons that convert like the signal, and charged particles the veto misses.
    backgroundCount = exposureFactor * backgroundFractionInsideCone * (
        diffusePhotonFlux * totalConversionProbability * photonSurvivalProbabilityThroughAcd
        + chargedParticleFlux * (1.0 - vetoEfficiency)
    )
    return signalCount, backgroundCount


def computeCountsPerLayer(
    conversionProbabilityPerLayer, converterRadiationLengths, vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction
):
    """Expected signal and background counts over the field of view: one row per energy bin and one column per true conversion layer."""
    exposureFactor = detectorSideLength**2 * exposureDuration * livetimeFraction
    probabilityPerLayer = conversionProbabilityPerLayer[:numberOfReconstructableLayers]
    # Share of the charged background in each class: follows the material of the layer (foil and passive material), not attenuated from above.
    materialPerLayer = (converterRadiationLengths + passiveRadiationLengthsPerLayer)[:numberOfReconstructableLayers]
    chargedBackgroundLayerShare = materialPerLayer / jnp.sum(materialPerLayer)
    signalCount = (   # The efficiency of the energy-sharing cut applies to the photons, not to the charged background.
        signalPhotonFlux * binSignalFraction[:, None] * binSharingEfficiency[:, None] * exposureFactor * probabilityPerLayer[None, :]
        * photonSurvivalProbabilityThroughAcd
    )
    backgroundCount = exposureFactor * (
        diffusePhotonFlux * binDiffuseFraction[:, None] * binSharingEfficiency[:, None] * probabilityPerLayer[None, :] * photonSurvivalProbabilityThroughAcd
        + chargedParticleFlux * binChargedFraction[:, None] * (1.0 - vetoEfficiency) * chargedBackgroundLayerShare[None, :]
    )
    return signalCount, backgroundCount


def computeAsimovIntegrand(signalDensity, backgroundDensity):
    """2 [(s + b) ln(1 + s/b) - s]: the Asimov likelihood ratio contribution of one patch of sky."""
    ratio = signalDensity / backgroundDensity
    # For small s/b the subtraction cancels to rounding noise, so use the series there.
    smallRatioSeries = ratio**2 / 2.0 - ratio**3 / 6.0 + ratio**4 / 12.0
    directFormula = (1.0 + ratio) * jnp.log1p(ratio) - ratio
    return 2.0 * backgroundDensity * jnp.where(ratio < 1e-3, smallRatioSeries, directFormula)   # 2.0: Asimov likelihood ratio definition. 1e-3: switch to the series, numerical choice.


def computeSpatialSignificanceSquaredComponents(
    signalCountPerComponent, angularVariancePerComponent, backgroundCountPerLayer, signalConfusionMatrix, backgroundConfusionMatrix
):
    """Expected likelihood ratio (TS) of a point source against background alone, summed over assigned-layer classes.

    The signal of a true layer is a mixture of Gaussian PSF components (one per energy-sharing node), so the matrices differ in size:
    the signal confusion matrix maps components to assigned classes, the background one maps layers to assigned classes.
    """
    radius = jnp.geomspace(radialGridMinimum, fieldOfViewRadius, radialGridPoints)   # rad
    # One 2D Gaussian PSF (1/sr) per component.
    psfPerComponent = jnp.exp(-radius[None, :] ** 2 / (2.0 * angularVariancePerComponent[:, None])) / (
        2.0 * jnp.pi * angularVariancePerComponent[:, None]
    )
    # An assigned class sees each true layer with the probability in the confusion matrix.
    signalDensityPerClass = signalConfusionMatrix @ (signalCountPerComponent[:, None] * psfPerComponent)   # 1/sr
    backgroundDensityPerClass = (backgroundConfusionMatrix @ backgroundCountPerLayer) / fieldOfViewSolidAngle   # 1/sr, flat
    integrand = computeAsimovIntegrand(signalDensityPerClass, backgroundDensityPerClass[:, None])
    # Integral of the integrand over the circular cap, with the exact area element 2 pi sin(theta) dtheta, as a trapezoid in ln theta (dtheta = theta d ln theta).
    integralPerClass = jnp.trapezoid(integrand * 2.0 * jnp.pi * jnp.sin(radius[None, :]) * radius[None, :], jnp.log(radius), axis=1)
    return jnp.sum(integralPerClass)


def computeSpatialSignificanceSquared(signalCountPerLayer, backgroundCountPerLayer, angularVariancePerLayer, confusionMatrix):
    """The same with one Gaussian PSF per true layer (no energy-sharing components). Kept for the checks."""
    return computeSpatialSignificanceSquaredComponents(
        signalCountPerLayer, angularVariancePerLayer, backgroundCountPerLayer, confusionMatrix, confusionMatrix
    )


def computeSpatialTestStatistic(signalCountPerBinAndLayer, backgroundCountPerBinAndLayer, angularVariancePerBinLayerNode, nodeWeights, confusionMatrix):
    """Expected test statistic of the power-law source: the sum over the energy bins of the per-bin spatial TS (no energy migration).

    With the best-fit power law equal to the true one in the Asimov dataset, this is the TS of the fit of the signal power law
    against the background-only fit, as in Fermi-LAT. The PSF of a layer is a mixture over the energy-sharing nodes.
    """
    numberOfBins, numberOfClasses, numberOfNodes = angularVariancePerBinLayerNode.shape
    signalPerComponent = (signalCountPerBinAndLayer[:, :, None] * nodeWeights[:, None, :]).reshape(numberOfBins, -1)   # Layer-major, then node.
    variancePerComponent = angularVariancePerBinLayerNode.reshape(numberOfBins, -1)
    signalConfusionMatrix = jnp.repeat(confusionMatrix, numberOfNodes, axis=1)   # The same layer assignment for every node.
    return jnp.sum(jax.vmap(computeSpatialSignificanceSquaredComponents, in_axes=(0, 0, 0, None, None))(
        signalPerComponent, variancePerComponent, backgroundCountPerBinAndLayer, signalConfusionMatrix, confusionMatrix
    ))


def computeConeCheck(conversionProbabilityPerLayer, angularVariancePerBinLayerNode, vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction):
    """Single-cone check, one averaged Gaussian and a cone of fixed radius in each energy bin. Returns the TS summed over the bins and the counts in the cones."""
    exposureFactor = detectorSideLength**2 * exposureDuration * livetimeFraction
    probabilityPerLayer = conversionProbabilityPerLayer[:numberOfReconstructableLayers]
    totalConversionProbability = jnp.sum(probabilityPerLayer)
    effectiveVariancePerBin = jnp.sum(probabilityPerLayer[None, :, None] * binSharingWeight[:, None, :] * angularVariancePerBinLayerNode, axis=(1, 2)) / totalConversionProbability
    signalContainmentFraction, backgroundFractionInsideCone = computeSignalConeFractions(effectiveVariancePerBin)
    signalCountPerBin = (
        signalPhotonFlux * binSignalFraction * binSharingEfficiency * exposureFactor * totalConversionProbability
        * photonSurvivalProbabilityThroughAcd * signalContainmentFraction
    )
    backgroundCountPerBin = exposureFactor * backgroundFractionInsideCone * (
        diffusePhotonFlux * binDiffuseFraction * binSharingEfficiency * totalConversionProbability * photonSurvivalProbabilityThroughAcd
        + chargedParticleFlux * binChargedFraction * (1.0 - vetoEfficiency)
    )
    coneTestStatistic = jnp.sum(computeAsimovSignificance(signalCountPerBin, backgroundCountPerBin) ** 2)
    return coneTestStatistic, jnp.sum(signalCountPerBin), jnp.sum(backgroundCountPerBin)


def computeContainmentRadii(signalCountPerLayer, angularVariancePerLayer, containmentFractions=(0.68, 0.95)):   # Conventional PSF68 and PSF95 of Fermi-LAT [atwood2009lat] https://arxiv.org/abs/0902.1089
    """Containment radii (degrees) of the signal PSF, a mixture over conversion layers. Reported only, not optimised."""
    signalCountPerLayer = jnp.ravel(signalCountPerLayer)   # Energy bins and layers together: the PSF is a mixture over both.
    angularVariancePerLayer = jnp.ravel(angularVariancePerLayer)
    radius = jnp.geomspace(radialGridMinimum, fieldOfViewRadius, radialGridPoints)
    weights = signalCountPerLayer / jnp.sum(signalCountPerLayer)
    containment = jnp.sum(
        weights[:, None] * (1.0 - jnp.exp(-radius[None, :] ** 2 / (2.0 * angularVariancePerLayer[:, None]))), axis=0
    )
    return [jnp.degrees(jnp.interp(fraction, containment, radius)) for fraction in containmentFractions]


def computeConstraintPenalty(stripPitch, layerSpacing):
    """Channel and height budgets, as smooth penalties that are zero while satisfied."""
    numberOfChannels = 2 * numberOfLayers * detectorSideLength / stripPitch    # x and y strips. 2: x and y planes, geometry.
    totalHeight = (numberOfLayers - 1) * layerSpacing   # cm
    constraintPenalty = (
        jax.nn.relu(numberOfChannels / channelBudget - 1.0) ** 2
        + jax.nn.relu(totalHeight / maximumHeight - 1.0) ** 2
    )
    return constraintPenalty, numberOfChannels


def computeDetectorResponse(unboundedParameters):
    """Spatial likelihood-ratio significance for a design, plus the single-cone check and diagnostics. Fully differentiable."""
    physicalParameters = mapUnboundedToPhysicalParameters(unboundedParameters)
    converterThickness = physicalParameters["converterThickness"]      # Vector, one entry per layer.
    layerSpacing = physicalParameters["layerSpacing"]
    stripPitch = physicalParameters["stripPitch"]
    acdThickness = physicalParameters["acdThickness"]
    acdThreshold = physicalParameters["acdThreshold"]

    converterRadiationLengths, radiationLengthsAboveLayer, radiationLengthsBelowLayer = computeLayerMaterial(converterThickness)
    conversionProbabilityPerLayer = computeConversionProbabilityPerLayer(converterRadiationLengths, radiationLengthsAboveLayer)
    totalConversionProbability = conversionProbabilityPerLayer.sum()
    vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction = computeAcdResponse(acdThickness, acdThreshold)

    # Objective: each conversion layer is a pseudo-detector with its own PSF, and the spatial likelihood ratios add.
    angularVariancePerLayer = computePsfComponentVariance(
        converterRadiationLengths, layerSpacing, stripPitch
    )[:, :numberOfReconstructableLayers, :]   # Energy bins, layers, energy-sharing nodes.
    signalCountPerLayer, backgroundCountPerLayer = computeCountsPerLayer(
        conversionProbabilityPerLayer, converterRadiationLengths, vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction
    )
    significanceSquared = computeSpatialTestStatistic(
        signalCountPerLayer, backgroundCountPerLayer, angularVariancePerLayer, binSharingWeight, labelConfusionMatrix
    )
    significanceSquaredNoLabel = computeSpatialTestStatistic(
        signalCountPerLayer, backgroundCountPerLayer, angularVariancePerLayer, binSharingWeight, noLabelConfusionMatrix
    )
    signalCountPerComponent = signalCountPerLayer[:, :, None] * binSharingWeight[:, None, :]
    psf68, psf95 = computeContainmentRadii(signalCountPerComponent, angularVariancePerLayer)

    # Single-cone check: one averaged Gaussian and a cone of fixed radius. Not used by the loss.
    coneTestStatistic, signalCount, backgroundCount = computeConeCheck(
        conversionProbabilityPerLayer, angularVariancePerLayer, vetoEfficiency, photonSurvivalProbabilityThroughAcd, livetimeFraction
    )
    # Signal-weighted mean variance over the energy bins and layers, for the single number "angular resolution".
    effectiveAngularVariance = jnp.sum(signalCountPerComponent * angularVariancePerLayer) / jnp.sum(signalCountPerComponent)

    constraintPenalty, numberOfChannels = computeConstraintPenalty(stripPitch, layerSpacing)
    totalSignalPhotons = jnp.sum(signalCountPerLayer)
    constraintPenalty = constraintPenalty + jax.nn.relu(1.0 - totalSignalPhotons / minimumSignalPhotons) ** 2

    return dict(
        testStatistic=significanceSquared,                 # The objective: expected TS, which is Z^2.
        testStatisticNoLabel=significanceSquaredNoLabel,
        coneTestStatistic=coneTestStatistic,
        significance=jnp.sqrt(significanceSquared),        # Z = sqrt(TS), kept for comparison with the old figure of merit.
        significanceNoLabel=jnp.sqrt(significanceSquaredNoLabel),
        coneSignificance=jnp.sqrt(coneTestStatistic),
        signalCount=signalCount,                  # Single-cone check.
        backgroundCount=backgroundCount,          # Single-cone check.
        totalSignalPhotons=totalSignalPhotons,
        constraintPenalty=constraintPenalty,
        totalConversionProbability=totalConversionProbability,
        angularResolution=jnp.degrees(jnp.sqrt(effectiveAngularVariance)),
        psf68=psf68,
        psf95=psf95,
        vetoEfficiency=vetoEfficiency,
        numberOfChannels=numberOfChannels,
        conversionProbabilityPerLayer=conversionProbabilityPerLayer,
    )


def computeAsimovSignificance(signalCount, backgroundCount):
    # Median discovery significance for a counting experiment, valid also when the counts are small.
    return jnp.sqrt(2.0 * ((signalCount + backgroundCount) * jnp.log1p(signalCount / backgroundCount) - signalCount))


def computeLoss(unboundedParameters):
    detectorResponse = computeDetectorResponse(unboundedParameters)
    # Log of the test statistic keeps gradients well scaled over orders of magnitude; the penalty enforces the budgets.
    return -jnp.log(detectorResponse["testStatistic"]) + constraintPenaltyWeight * detectorResponse["constraintPenalty"]


def computeLossFromSignificance(unboundedParameters):
    """The previous loss, -ln(Z) with weight 10 on the penalty. The same optimum as computeLoss. Kept in case we go back."""
    detectorResponse = computeDetectorResponse(unboundedParameters)
    return -jnp.log(detectorResponse["significance"]) + 10.0 * detectorResponse["constraintPenalty"]   # 10.0: weight of the constraint penalty. Numerical choice, no physical source.


scalarMetricNames = (
    "testStatistic", "testStatisticNoLabel", "coneTestStatistic", "significance", "significanceNoLabel", "coneSignificance", "signalCount", "backgroundCount", "totalSignalPhotons",
    "constraintPenalty", "totalConversionProbability", "angularResolution", "psf68", "psf95", "vetoEfficiency",
    "numberOfChannels",
)


@jax.jit
def computeMetrics(unboundedParameters):
    return computeDetectorResponse(unboundedParameters)


def computeBoundWarnings(unboundedParameters):
    """Checks of the energy bounds against a design, as plain-text warnings. Not part of the loss, so the gradients stay clean."""
    warningMessages = []
    if sourceType != "powerLaw":
        return warningMessages
    physicalParameters = mapUnboundedToPhysicalParameters(unboundedParameters)
    converterRadiationLengths = computeLayerMaterial(physicalParameters["converterThickness"])[0]
    if energyMinimum < comptonPairCrossoverEnergy:
        warningMessages.append(f"energyMinimum = {energyMinimum:.0f} MeV is below the Compton and pair crossover ({comptonPairCrossoverEnergy:.0f} MeV): the pair model does not describe the events.")
    trackerRadiationLengths = float(jnp.sum(converterRadiationLengths + passiveRadiationLengthsPerLayer))
    showerMaximumDepth = float(np.log(energyMaximum / criticalEnergyTungsten) + showerMaximumPhotonOffset)
    if trackerRadiationLengths > showerMaximumDepth:
        warningMessages.append(f"the tracker is too large for energyMaximum = {energyMaximum:.0f} MeV: it has {trackerRadiationLengths:.2f} radiation lengths against a shower maximum at {showerMaximumDepth:.2f}. The highest-energy photons shower inside it and the two-track pair model fails.")
    # Signal photons that the truncation at energyMaximum discards: the tail above it relative to the part inside, from the power law.
    tailShare = (energyMaximum / energyMinimum) ** (1.0 - signalSpectralIndex)
    photonsAboveMaximum = float(computeDetectorResponse(unboundedParameters)["totalSignalPhotons"]) * tailShare / (1.0 - tailShare)
    if photonsAboveMaximum > minimumSignalPhotons:
        warningMessages.append(f"the truncation at energyMaximum discards about {photonsAboveMaximum:.0f} expected signal photons (more than {minimumSignalPhotons:.0f}).")
    return warningMessages
