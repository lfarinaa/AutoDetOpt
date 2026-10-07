# AutoDetOpt

Differentiable (autodiff) design optimisation of a pair-conversion tracker with an anticoincidence
detector (ACD), in the spirit of the MODE collaboration's work on end-to-end optimisation of detectors.

The repository holds two separate projects that share a goal but not code:

| Project | Approach | Status |
|---|---|---|
| [`v0/`](v0) | Closed-form expectation, no events | Working notebook |
| [`v1/`](v1) | Stochastic, event-level simulation | Not started |

Each is refined on its own. Neither imports from the other.

---

## trackerOptimisation v0: the numeric toy model

Notebook: [`v0/trackerOptimisation.ipynb`](v0/trackerOptimisation.ipynb) (JAX, optax, matplotlib).

A monoenergetic photon beam at normal incidence hits a stack of tungsten converter layers, each followed by
two silicon strip planes (x and y), wrapped in a plastic scintillator ACD. The whole response is a smooth
closed-form function of the design, so `jax.grad` gives the gradient of the significance with respect to
every parameter and Adam climbs it.

**Free parameters:** converter thickness per layer, layer spacing, strip pitch, ACD thickness, ACD threshold.
**Fixed:** layer count (scanned by hand), photon energy, detector size, fluxes, exposure.
**Objective:** Asimov significance of a point source over diffuse photon and charged-particle backgrounds,
with smooth penalties for channel budget and height.

How it works:

1. Pair conversion probability per layer is `exp(-7/9 x_above) * (1 - exp(-7/9 x_layer))`.
2. The angular variance of the reconstructed direction is multiple scattering (Highland, no log term) plus
   pair opening angle plus strip resolution over the lever arm. It is averaged over conversion layers.
3. A single Gaussian cone of 2 sigma sets the signal containment and the background fraction inside it.
4. The ACD gives a veto efficiency (smooth sigmoid turn-on), a self-veto photon survival factor and a
   livetime loss from noise.
5. Parameters are optimised in an unbounded space and mapped to their ranges by a sigmoid, so there is no
   clipping and no zero-gradient region at the bounds.

All numbers are placeholders. Do not trust an optimum until they are set to realistic values.

### Known limitations of v0

- **No events.** Averaging the variance over conversion layers replaces a mixture of PSFs with one Gaussian.
- **One interaction, one species.** Only pair conversion in tungsten. Charged particles are a flat flux times
  `1 - vetoEfficiency`, and the tracker plays no part in rejecting them.
- **No direction or energy dependence.** Area, acceptance and PSF are constants, and the energy is fixed.
- **Reconstruction is a formula.** Pattern recognition is a hard mask on the number of downstream layers.
- **Parameters run to bounds.** The optimum has `acdThickness` at its upper bound (about 3 cm). This means
  a trade-off is missing (backsplash self-veto, mass, cost), not that 3 cm is right.

v0 will be refined as an analytic model (see [ROADMAP.md](ROADMAP.md)), for example adding angle and energy as
quadrature axes. It stays deterministic.

---

## trackerOptimisation v1: the stochastic, multi-everything workflow

Directory: [`v1/`](v1). Not started. This README section records the design thinking.

Goal: go from the averaged toy to a **multi-particle, multi-direction, multi-interaction** workflow with
reconstruction in the loop.

### What changes relative to v0

| Extension | What is needed |
|---|---|
| **Multiple directions** | Path length `1/cos(theta)`. Lateral leakage and side entry through the finite box, ACD side panels that are actually hit. PSF depends on theta (projected lever arm, scattering path). Separate x and y planes. Acceptance integrated over the field of view and pointing profile. |
| **Multiple particles** | Gamma, e+-, protons, albedo gammas, neutrons, each with a flux spectrum and interaction model, and its own pass through veto and tracker. Backgrounds depend on the design through the tracker (dE/dx, topology), not only the ACD. |
| **Multiple interactions** | Compton, photoelectric, bremsstrahlung, ionisation, delta rays, hadronic secondaries. Cross-section tables (NIST XCOM) as differentiable functions of energy, Z and thickness. Compton dominates below about 10 MeV and needs its own reconstruction. Backsplash gives the real ACD trade-off. |
| **Energy** | Energy as a variable with a spectrum and a binned likelihood. Without a calorimeter, energy comes from track scattering or Compton kinematics, a design-dependent resolution. |

### The central problem: gradients through discrete randomness

In v0 the response is an expectation, so gradients are trivial. With simulated events, outcomes such as
"which interaction", "did the strip fire", "did the track reconstruct" are discrete, and their gradients are
zero or biased. Options, roughly in the order to try them:

1. **Semi-analytic quadrature** over (E, theta, species, depth). Deterministic, exact gradient, fine while
   the dimension is low.
2. **Pathwise (reparametrised) sampling.** Free paths by inverse CDF `s = -ln(u)/mu`, scattering angles from
   Highland or Moliere, energy sharing from Bethe-Heitler. Differentiable in thickness and spacing. Use
   common random numbers so the loss is deterministic for a fixed batch.
3. **Score-function or stochastic-AD estimators** for discrete branches (interaction choice, veto firing).
   Higher variance. Gumbel-softmax relaxations are an option if the bias is acceptable.
4. **Surrogates.** Train a network or generative model on Geant4 across design points and differentiate
   through it (local generative surrogates, Shirobokov et al. 2020).
5. **Differentiate the real simulator.** Aehle et al. (2024) did pathwise derivatives of an EM shower
   simulation. Works but is heavy engineering.

**Geometry.** Moving a boundary changes which events cross it, and naive pathwise gradients with hard
geometry miss that term. Keeping interactions as attenuation integrals (`mu * L`) avoids it. Full ray tracing
needs smooth boundaries or an SDF-style approach, as in differentiable rendering.

### Reconstruction in the loop

- **Pair tracking:** differentiable least-squares or Kalman fit, or a Gluckstern/Billoir-style covariance.
  The Cramer-Rao bound gives a differentiable resolution with the full multiple-scattering covariance.
  Pattern recognition efficiency still needs a soft model.
- **Compton:** ordering and ring reconstruction are combinatorial. Bound with Fisher information, or co-train
  a learned reconstruction with the design (a central MODE theme; TomOpt is their worked example).

### Objective and constraints

- Replace the single cone count with a binned Poisson likelihood in (energy, angle, species). The Asimov
  significance becomes a sum over bins, or use the Fisher information of the source parameters.
- Realistic PSF with tails instead of a fixed 2 sigma cone.
- Cost, mass and power terms, and likely a Pareto front.
- Discrete choices (layer count, material) as outer scans or soft masks.

### Validation

Gradients can be correct for a wrong model. At each stage, validate the optimum against Geant4 (or a similar
full simulation), at the optimised design and at a few perturbed points.

### References (from memory, verify before citing)

- MODE collaboration, white paper on machine-learning optimised design of experiments (arXiv:2203.13818).
- MODE, "Toward the end-to-end optimization of particle physics instruments with differentiable programming"
  (arXiv:2310.05673).
- TomOpt, differentiable muon tomography optimisation.
- Shirobokov et al., "Black-box optimization with local generative surrogates" (2020).
- Aehle et al., pathwise derivatives of electromagnetic shower simulations (2024).

---

## Planned tooling

Both projects are to become command-line scripts that read inputs from a text file and print results to the
terminal. See [ROADMAP.md](ROADMAP.md).
