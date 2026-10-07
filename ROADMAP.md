# Roadmap

Planned work, not yet started. Items are ordered within each section. v0 and v1 are developed separately.

## Both projects: command-line script

Turn each calculation into a script that reads its inputs from a text file and prints results to the terminal.
The output can be piped to a file by the caller, so the script writes nothing itself.

- [ ] Choose the input format (plain `key = value` text, or TOML) and document every key with its unit.
- [ ] Move the fixed inputs and `parameterBounds` out of the code and into the input file.
- [ ] Parse the input file with validation: unknown keys, missing keys, bad units and out-of-range values
      give a clear error and a non-zero exit code.
- [ ] Separate the model (pure functions) from the optimisation loop, the plotting and the command line.
- [ ] Print to stdout: the inputs echoed back, the optimised parameters in physical units, and the final
      metrics (significance, counts, angular resolution, channels). Progress goes to stderr so piping stays clean.
- [ ] Add options for number of steps, learning rate, random seed and number of restarts.
- [ ] Make runs reproducible (seeded) and print the library versions.
- [ ] Decide what to do with the plots: separate optional command, or drop them from the script.
- [ ] Add a few regression tests, for example the v0 notebook result for the default inputs.
- [ ] Add a requirements file.

## v0: refine the analytic model

- [x] Replace the cone count with the spatial Asimov integral (single objective, no multi-objective).
- [x] Treat each conversion layer as its own pseudo-detector, with its own PSF and background, and sum the
      per-class spatial integrals.
- [x] Add a confusion matrix `C[l', l]` for conversion-layer assignment. Default to the identity.
- [x] Add the no-label case (a mixture PSF) as a lower bound, and check that identity gives an upper bound.
- [x] Set the lower bound of `converterThickness` to a small positive value (placeholder `1e-3 cm`), because
      passive material always converts some photons. No empty classes. Later derive it from a passive-material
      budget.
- [x] Charged-particle background split: all in the first layer (decided). See the open issue below.
- [ ] **Open issue: perfect labels plus charged background in layer 0 makes the tracker a perfect charged-particle
      rejector, so the ACD becomes redundant in the optimum.** Decide how the charged background leaks into the
      other classes (leakage fraction, or a charged row in the confusion matrix) before trusting any ACD result.
- [ ] Keep the old cone objective for a side-by-side check, then remove it.
- [x] Add a soft constraint of at least 10 signal photons, as in the Fermi sensitivity definition.
- [x] Report PSF68 and PSF95 as outputs.
- [ ] Add angle (`1/cos(theta)` path length, theta-dependent PSF and effective area) as a quadrature axis.
- [ ] Replace the monoenergetic photon with a power law (`Gamma = 2`) truncated at `E_min` and `E_max`
      (placeholders 100 MeV and 10 GeV). See "Power-law source and energy bounds" in the README.
- [ ] Add energy as a quadrature axis (log-spaced bins, 4 per decade), with tabulated cross sections,
      multiple scattering `~ 1/E` and opening angle `~ m_e / E`.
- [ ] Check the bounds with warnings (to stderr, not part of the loss): `E_min` below the Compton and pair
      crossover, tracker material `X_tot > t_max(E_max)`, fewer than about 10 expected signal photons above
      `E_max`.
- [ ] Look up and record the Compton and pair crossover energies and the critical energies of tungsten and
      silicon (NIST XCOM, PDG), replacing the from-memory values.
- [ ] Quantify how far the `7/9` conversion coefficient is from the tabulated pair cross section at `E_min`.
- [ ] Default: energy is known perfectly (identity assignment, no energy confusion). Later: add an
      energy-assignment matrix and a calorimeter design, and check the no-energy (mixture) limit.
- [ ] Give the diffuse photon and charged-particle backgrounds their own spectra.
- [ ] Add Compton and photoelectric interactions.
- [ ] Add a second charged species, with tracker-based rejection.
- [ ] Replace the resolution formulas with a Fisher or Kalman covariance.
- [ ] Add a PSF with non-Gaussian tails (for example a King function).
- [ ] Free background normalisation in the likelihood ratio (profile likelihood).
- [ ] Add the missing trade-offs that stop parameters running to their bounds (backsplash, mass, cost).
- [ ] Check which parameters sit at their bounds in every optimum.

## v1: stochastic event-level model

- [ ] Pick the first gradient estimator (pathwise with common random numbers) and test it on one interaction.
- [ ] Event generator: species, energy, direction, interaction depth.
- [ ] Interaction models: pair, Compton, photoelectric, bremsstrahlung, ionisation.
- [ ] Reconstruction in the loop: differentiable track fit, soft pattern-recognition efficiency.
- [ ] Binned Poisson likelihood objective in energy, angle and species.
- [ ] Compare estimators (pathwise, score function, surrogate) on gradient bias and variance.
- [ ] Validate against Geant4 at the optimum and at perturbed designs.
- [ ] Memory and speed: `vmap` over events, `jax.checkpoint` where needed.

## Parameters

- [ ] Settle the decisions listed at the end of [PARAMETERS.md](PARAMETERS.md), then apply the reference values.
- [ ] Reparametrise the ACD threshold in MIP units.
- [ ] Add readout dead time and strip hit efficiency to the model.
- [ ] Derive a first confusion matrix from the strip hit efficiency.

- [ ] Update the scintillator radiation length to 42.54 cm (PDG, polyvinyltoluene). The MIP energy loss is already 2.019 MeV/cm.
- [ ] Keep `references/` in step: add a bibliography entry and a local copy for every new source.

## Housekeeping

- [ ] Decide whether the notebooks are kept next to the scripts or archived.
- [ ] Set realistic values for all placeholder inputs and record their sources.
