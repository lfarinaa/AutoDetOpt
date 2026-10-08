# Findings log

A running record of what the optimisation has told us, including what turned out to be an artifact of the model.
Entries are in the order we found them. Numbers are from the notebook runs at the time and change as the model
changes, so each one says which model version it belongs to. Add new entries at the end of the relevant section.

Created 2026-10-08.

## Status in one paragraph

**v0 is at an early stage.** The shape of the optimal converter distribution has changed qualitatively with almost
every modelling improvement: back-loaded, front-loaded, super front-loaded (one slab), and now one thick foil in
every third layer. Several nearly equal optima coexist (random restarts end in at least five different designs
within 7% of each other in test statistic). This instability under minor improvements is itself the finding: **no
conclusion about where the tungsten should go can be drawn yet.** The only quantities that stayed put are those
pinned by a bound or a budget (see "What has been stable").

## 1. How the converter profile changed with the model

Foil thickness per layer, layer 0 at the top. Only the layers that can be a conversion layer are listed.

| # | Model version | Objective | Profile of the optimum | Type |
|---|---|---|---|---|
| 1 | First notebook: closed-form, one averaged variance, 2 sigma cone, 3 downstream layers masked | single cone, Asimov Z (initial 14.5, optimum 21.5) | 512, 543, 578, 618, 662, 713, 771 µm, ACD 3 cm | **back-loaded** |
| 2 | Spatial integral, no layer labels, downstream scattering weight 1/3 | spatial TS | 723, 701, 670, 630, 579, 519, 448 µm | **front-loaded** |
| 3 | As 2 with perfect labels and all charged background in layer 0 | labelled spatial TS | 11, 663, 543, 425, 318, 225, 145 µm | front-loaded, but layer 0 was a free veto (artifact, see 2.1) |
| 4 | As 2 and 3 with downstream scattering weight 0 | no labels / labels | 527 to 1784 µm / 325 to 1450 µm, increasing with depth | **back-loaded again** |
| 5 | Scratch test: hard window of 2 or 3 layers for the scattering | cone, no labels, labels | foils every third layer, for example 11, 822, 241, 10, 10, 1526, 369 µm | **comb** (first sighting) |
| 6 | New scattering model (3 hits, own silicon, log term, expected conversion depth), even split of the charged background | labelled spatial TS | 1999 µm in layer 0 (the upper bound), then bare layers | **super front-loaded** (one slab), partly a loophole (see 2.3) |
| 7 | As 6 with the charged background following the material of each layer | labelled spatial TS | best basin: 1449, 10, 11, 1260, 11, 10, 881, 235 µm | **one thick foil in every third layer** (comb) |

So we have seen back-loaded, front-loaded, super front-loaded and one-in-three comb optima. Row 4 shows that the
sign of the slope is controlled by a single toy input, `downstreamScatteringWeight`, which has no source. LAT's
design is thin in front and thick in the back, which agrees with rows 1 and 4 and disagrees with rows 2, 3, 6, 7.
It is also an energy trade-off, and v0 is monoenergetic.

## 2. Artifacts found and fixed

1. **Layer 0 as a free charged-particle veto.** With perfect layer labels and all the charged background in layer 0,
   the optimiser made layer 0 nearly bare and threw its class away, rejecting 8 million charged counts for the cost
   of 0.6% of the signal. It then abandoned the ACD (0.5 cm, veto efficiency 0.51). The labelled significance rose
   while the no-label and single-cone ones fell. Unphysical: it ignores the plane efficiency (LAT: above 99% per
   plane), side entry and gaps, and that a poor ACD plus a veto layer is a worse instrument than a good ACD.
2. **Charged background spread like the photons.** It then behaved like a gamma in the tracker (attenuated, tied to
   the pair-conversion coefficient, growing with foil thickness) and like a proton in the ACD.
3. **Even split of the charged background over the classes.** With almost all the signal in one class (97% in
   layer 0), perfect labels left that class only 1/8 of the charged background: a free rejection by the number of
   classes (labelled TS 30 against 16 without labels). Closed by making the share follow the material of each layer.
4. **Sky curvature.** The flat-sky disc was replaced by the exact circular cap (half-angle `arccos(1 - Omega/2pi)`,
   area element `2 pi sin(theta)`). The effect was small (Z 36.13 to 35.98 at the start).

**Consistency check that exposed these:** the labelled, no-label and single-cone figures of merit must move in the
same direction during optimisation. When they diverge, suspect the model. This check is on the roadmap as an
automatic warning.

## 3. Local optima in the random restarts

Ten random restarts, current model (test statistic TS, best first):

| Basin | TS | Found by | Foils [µm], layers 0 to 7 |
|---|---|---|---|
| A | 611.4 | 5 of 10 | 1449, 10, 11, 1260, 11, 10, 881, 235 |
| B | 597.7 | 1 of 10 | 1548, 10, 10, 1270, 465, 10, 10, 403 |
| C | 584.2 | 2 of 10 | 1333, 701, 10, 10, 1074, 10, 10, 497 |
| D | 580.9 | 1 of 10 | 1264, 697, 10, 11, 1096, 348, 202, 164 |
| E | 570.2 | 1 of 10 | 1074, 682, 421, 10, 10, 1016, 397, 147 |

- **They are real local optima, not frozen layers.** At the bare layers the physical gradient is positive (thickening
  makes the loss worse), and the thick layers are at an interior optimum. Resetting the bare layers to 5% of the
  range and re-optimising returned to the same design (basin E) or moved to another existing basin (D to C).
- **Cause:** a foil adds full-weight scattering for the two layers above it, so which layers carry a foil is a
  combinatorial choice with several near-equal answers. The hard window of two full-weight layers followed by the toy
  weight 1/3 creates the one-in-three structure.
- **Flat directions:** the layer spacing ends anywhere between 2.7 and 3.3 cm at the same TS.
- Plots: the random-restart cell of the notebook (summary, overview heatmap, one panel per restart).

## 4. Other findings

- **The 7/9 pair coefficient** is accurate to a few percent only down to about 1 GeV (PDG Eq. 34.32), so it is
  optimistic at the 100 MeV lower bound.
- **PDG advises** applying the Highland formula once to the combined scatterer, because adding separate `theta0`
  in quadrature is systematically too small. The model now does that.
- **Conversion depth:** for a 0.57 X0 foil the exact expected path after conversion is 0.306 X0 against 0.285 for
  "halfway" (7% more). The closed form agrees with numerical integration.
- **Constants that differed from PDG:** the scintillator radiation length (42.4 against 42.54 cm) is still to be
  updated. The MIP energy loss was the mass value 1.95 MeV cm2/g and is now 2.019 MeV/cm.
- **ACD efficiency** is capped near 0.978 by the toy turn-on width, not by `maximumVetoEfficiency` (changing it from
  0.999 to 0.9995 moved nothing). The ACD always runs to its 3 cm upper bound, because backsplash and hermeticity are
  not modelled.
- **Regime:** the source flux (1e-3) is four orders of magnitude above the LAT faint-source reference and the diffuse
  background about 700 times above, so TS reaches 600 against the detection threshold of 25. We are far from the
  sensitivity limit, and the optimum may differ at realistic flux.
- **Analytic checks pass:** the spatial integral reproduces the closed form `S^2 / (4 pi sigma^2 b)` within the
  expected curvature of the sky, a 2 sigma cone loses the expected factor of 1.156, autodiff matches finite
  differences, and classes with labels beat the same sample without.

## 5. What has been stable

Only the quantities that are pinned by a bound or budget: the ACD thickness (at its 3 cm upper bound) and the strip
pitch (160 to 180 µm, set by the channel budget). The total conversion probability has stayed between 0.4 and 0.6,
and PSF68 near 10 to 14 degrees, but these also moved with every model change.

## 6. What would make the foil placement trustworthy

1. The fit-covariance PSF model, to replace the hard window and the toy weight below the fitted hits.
2. The energy axis, because thin-front, thick-back is an energy trade-off.
3. A realistic regime: the source flux, the background, the exposure and the instrument size.
4. Directions, which give the tracker's charged rejection a physical basis.
5. A restart protocol: always report the spread over restarts and the basins, never a single run, and test whether
   the conclusion survives each model change.
