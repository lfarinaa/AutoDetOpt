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

- [ ] Add angle (`1/cos(theta)` path length, theta-dependent PSF and effective area) as a quadrature axis.
- [ ] Add energy as a quadrature axis, with a spectrum and tabulated cross sections.
- [ ] Add Compton and photoelectric interactions.
- [ ] Add a second charged species, with tracker-based rejection.
- [ ] Replace the resolution formulas with a Fisher or Kalman covariance.
- [ ] Replace the averaged-variance Gaussian with a mixture over conversion layers, and a PSF with tails.
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

## Housekeeping

- [ ] Decide whether the notebooks are kept next to the scripts or archived.
- [ ] Set realistic values for all placeholder inputs and record their sources.
