# Parameters and what they stand for

Every input of v0 stands for one or more real physical things. This table ties each one to its physical meaning
and to a reference value. Nothing here has been applied to the notebook yet. Items marked **decision** need an
answer first because they set the scale of the problem.

Sources. Keys are those of [references/bibliography.bib](references/bibliography.bib), and local copies of all of
them are in [references/](references):
- **[atwood2009lat]** Atwood et al. 2009, "The Large Area Telescope on the Fermi Gamma-ray Space Telescope
  Mission", https://arxiv.org/abs/0902.1089 (Tables 1, 2, 4 and section 2.2). Values read from the paper's text.
- **[farina2021herd]** Fariña et al., "Gamma-ray performance study of the HERD payload",
  https://doi.org/10.22323/1.395.0651.
- **[cattaneo2020psd]** HERD plastic scintillator beam test, https://arxiv.org/abs/2005.09905. It describes the
  design and the SiPM readout, but the extracted text had no efficiency numbers.
- **[pdg2024passage]**, **[pdg2024detectors]**, **[pdg2024tungsten]**, **[pdg2024silicon]**, **[pdg2024pvt]**:
  Particle Data Group 2024, https://pdg.lbl.gov/2024/
- **[fermilat2013performance]**, **[fermipy2017sensitivity]**: Fermi-LAT sensitivity definition and tools.
- "memory" means I did not find it in these documents: verify before use.

## Free (optimised) parameters

| Input | Stands for | Current range | Reference | Note |
|---|---|---|---|---|
| `converterThickness` (per layer) | Tungsten foil thickness | 1e-3 to 0.2 cm | LAT front foils 0.010 cm (0.03 X0), back 0.072 cm (0.18 X0, 93% W) | Lower bound is the passive material that is always there. LAT's is 0.014 X0 per x-y plane (supports, detectors, electronics), about 0.005 cm of W-equivalent. The current 1e-3 cm is a factor of about 5 more optimistic. **decision** |
| `layerSpacing` | Distance between x-y layers | 0.5 to 5 cm | LAT: pitch / spacing = 0.0071 with 228 µm pitch, so about 3.2 cm | In range. |
| `stripPitch` | Strip or fibre pitch | 50 to 1000 µm | LAT 228 µm, HERD FIT about 250 µm | In range. 50 µm is optimistic for strips over large areas. |
| `acdThickness` | Plastic scintillator thickness | 0.5 to 3 cm | LAT 1.0 cm (0.06 X0 including the thermal blanket) | In range. LAT's 0.06 X0 includes the micrometeoroid blanket, the model counts the scintillator only (0.024 X0 per cm). |
| `acdThreshold` | Discriminator level for the veto | 0.05 to 1.0 MeV | LAT: 0.45 MIP on board, about 0.30 MIP on the ground | Physically this is a fraction of the MIP deposit, not MeV. With the threshold in MeV, changing the thickness silently changes the threshold in MIP units. Suggest reparametrising in MIP. **decision** |

## Fixed inputs

| Input | Stands for | Current | Reference | Note |
|---|---|---|---|---|
| Radiation lengths | W, Si, scintillator | 0.3504, 9.37, 42.4 cm | W 0.3504 cm [pdg2024tungsten] and Si 9.370 cm [pdg2024silicon] match PDG. Polyvinyltoluene scintillator 42.54 cm [pdg2024pvt] (polystyrene 41.31 cm) | Update the scintillator value to 42.54 cm. |
| `7/9` coefficient | High-energy pair conversion limit | 7/9 | Physical constant, PDG Eq. 34.32 [pdg2024passage] | PDG says it is accurate to a few percent only down to 1 GeV, so it is optimistic at the 100 MeV lower bound. |
| `numberOfLayers` | x-y layers | 10 | LAT 18 (16 with foils, 2 bare), HERD FIT 7 double layers | |
| `hitsRequiredForTracking` | Hits a track needs | 3 | HERD: at least 3 hits per particle in each of X and Y [farina2021herd]. LAT: the first 2 planes after the conversion [atwood2009lat] | The conversion layer and the two layers below. Gives `minimumDownstreamLayersForReconstruction = 2` (was a placeholder 3) and 2 full-weight layers in the scattering. |
| `siliconThickness` | Silicon per layer | 0.03 cm (one plane) | LAT SSD 400 µm = 0.04 cm, and a layer has two planes (x and y) | The model counts one plane per layer. A layer of two planes would be 0.08 cm. HERD's FIT is scintillating fibre, not silicon. |
| `detectorSideLength` | Active tracker width | 40 cm | LAT 1.8 m x 1.8 m x 0.72 m (whole instrument), tower of 4 x 8.95 cm SSDs | **decision**: instrument scale. |
| `maximumHeight` | Tracker height budget | 30 cm | LAT 0.72 m for the whole instrument | No source for the tracker alone. **decision** |
| `channelBudget` | Readout channels | 5e4 | LAT: 1536 channels per tower layer pair (from the text), total of the order 1e6 (memory) | Scales with the detector size. **decision** |
| `signalPhotonFlux` | Source flux | 1e-3 /cm2 s | LAT reference: 1e-7 /cm2 s above 100 MeV for a faint high-latitude source (Table 1 note d) | The current value is 4 orders of magnitude brighter. Sets the regime: background-dominated or not. **decision** |
| `diffusePhotonFlux` | Diffuse background | 1e-2 /cm2 s over 1 sr | LAT: 1.5e-5 /cm2 s sr above 100 MeV at high latitude, index 2.1 (Table 1 note e) | The current value is about 700 times higher. |
| `chargedParticleFlux` | Cosmic-ray rate before the ACD | 1 /cm2 s | LAT raw trigger rate 2-4 kHz over about 3.2e4 cm2, so about 0.1 /cm2 s (inferred) | Verify against AMS fluxes. |
| `exposureDuration` | Observation time | 1e4 s | LAT 1-year survey, HERD 1, 5 and 10 years (Fig. 3) | Add the duty cycle (survey mode, SAA). |
| `fieldOfViewSolidAngle` | Instrument field of view | 1 sr | LAT 2.4 sr at 1 GeV [atwood2009lat], HERD wider (top and four sides) | |
| `minimumIonisingEnergyLossPerLength` | MIP loss in plastic | 2.019 MeV/cm | Polyvinyltoluene 2.019 MeV/cm (1.956 MeV cm2/g at 1.032 g/cm3) [pdg2024pvt] | Updated from 1.95, which was the mass stopping power rather than per length. |
| `maximumVetoEfficiency` | Ceiling of ACD efficiency | 0.9995 (was 0.999) | LAT tile requirement above 0.9997 averaged over the ACD area. Fibre ribbons over the gaps have above 90%. Whole-LAT charged rejection requirement 0.99999 with the other subsystems | Lumps hermeticity and intrinsic efficiency. Optimistic choice: 0.9997. |
| `vetoTurnOnWidthFraction`, `accidentalVetoScale` | Turn-on of the efficiency and noise dead time | 0.25, 0.1 MeV | No source | Toy numbers. They need a real noise or dark-count model. |
| Dead time | Readout dead time | Only the noise term | LAT 26.5 us per event, so about 1% at 400 Hz | Missing from the model. |
| Strip hit efficiency | Efficiency of one strip plane | Not in the model | LAT above 99% per plane, noise occupancy 1e-6 | See below. |
| `signalConeRadiusInSigma` | Analysis cut | 2 | Not an instrument parameter | Only used by the single-cone check. |

## Source spectrum (power-law source)

| Input | Stands for | Current | Reference | Note |
|---|---|---|---|---|
| `sourceType` | Source model | `"powerLaw"` | | `"monochromatic"` is kept, with `photonEnergy` = 100 MeV, and reproduces the earlier numbers. |
| `energyMinimum` | Lower edge of the range | 100 MeV | Above the Compton and pair crossover (about 10 MeV in tungsten, from memory) | Placeholder. |
| `energyMaximum` | Upper edge of the range | 10 GeV | Shower containment: `t_max = ln(E/E_c) + 0.5` [pdg2024passage] | Placeholder. |
| `binsPerDecade` | Energy bins | 4 | HERD sensitivity, 4 bins per decade [farina2021herd] | |
| `signalSpectralIndex` | Source spectrum | 2.0 | Fermi-LAT and HERD sensitivity use an index 2 power law | |
| `diffuseSpectralIndex` | Diffuse photon spectrum | 2.1 | LAT Table 1 note e [atwood2009lat] | |
| `chargedSpectralIndex` | Charged cosmic-ray spectrum | 2.7 | No source found | Placeholder. |
| `criticalEnergyTungsten` | Shower scale | 7.97 MeV | PDG, e- [pdg2024tungsten] | |
| `comptonPairCrossoverEnergy` | Energy where pair production equals Compton | 10 MeV | From memory | To check against NIST XCOM. |

With the power law the three fluxes are integral fluxes above `energyMinimum`.

## Things the model lacks that real parameters would feed

- **Hit efficiency and the confusion matrix.** LAT quotes above 99% per plane, and says that missing one of the
  first hits after a conversion costs about a factor 2 in resolution at 100 MeV and gives PSF tails. That is the
  physical origin of off-diagonal terms in the confusion matrix `C[assigned, true]`: a missed first hit moves
  a conversion to the next layer. It gives a first, grounded confusion matrix to replace the identity.
- **Charged background after the ACD.** Decided: nothing rejects it afterwards, so the tracker gives no further
  rejection, and it is split evenly over the conversion-layer classes (assumption, no source). A later model
  could add a real tracker rejection. That needs the single-plane efficiency (LAT: above 99% per plane), the
  topology cut with its photon efficiency, and the share of tracks entering through the top, which needs
  directions.
- **ACD hermeticity.** LAT: tiles under 1000 cm2, overlapped in one direction, fibre ribbons over the other
  gaps, light collection above 95% over a tile and above 75% within 1-2 cm of the edges. Needs angle.
- **Backsplash.** LAT requirement: self-veto must not reject more than 20% of photons at 300 GeV. EGRET lost
  at least a factor 2 above 10 GeV compared with 1 GeV. Relevant for `E_max` and for segmentation.

## Decisions needed before applying values

1. **Instrument scale.** The notebook is a 40 cm instrument, LAT is 1.8 m wide. Counts scale with area, so this
   changes the whole regime.
2. **Which source.** A bright source (the current 1e-3) or a faint LAT-reference source (1e-7)? The optimum
   depends on this because the signal-dominated and background-dominated regimes optimise differently.
3. **Optimistic or realistic.** For passive material, ACD efficiency and pitch: the LAT value, or something
   better where an upgrade is plausible?
4. **Reparametrise the ACD threshold in MIP units?**
5. **Tracker charged rejection** (currently none, by decision), later with its photon efficiency.
