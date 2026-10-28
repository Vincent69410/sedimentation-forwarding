# Boundary stabilization of a sedimentation front by forwarding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vincent69410/sedimentation-forwarding/blob/main/notebooks/worked_example.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for the paper

> M. O. Amirat, V. Andrieu, M. Bajodek, C. Valentin, *Boundary stabilization of a moving
> interface in a 2×2 weakly hyperbolic system of balance laws: application to continuous
> sedimentation*, 2026.

A suspension settles in a column. Above a critical solid fraction the particles
touch and carry a stress, below it they do not, and the surface separating the
two regimes is a shock — the sludge blanket — whose position is what we want to
regulate, using the underflow pump at the bottom as the only input. The
linearized plant is an input-to-state stable distributed layer driving a scalar
in which the control does not appear, which is the configuration *forwarding*
was devised for.

The code here does four things: it builds a stationary operating point and the
forwarding design on top of it, it evaluates the small-gain condition and the
constants of the stability certificate, it computes the exact closed-loop
spectrum from a characteristic determinant that discretizes the generator
nowhere, and it simulates the nonlinear closed loop on two scenarios, a change
of set point and a step of the feed fraction.

## Run it now

Click the Colab badge. Nothing to install, about three minutes on a free CPU,
and the notebook reproduces every number of Section 6 and draws its two data
figures: the stationary profile (Figure 3 of the extended version) and the
closed-loop scenarios (Figure 3 of the paper, Figure 4 of the extended
version). The closed-loop runs are read from `sedimentation_forwarding/data/`;
recomputing them takes about ten minutes each.

## Run it locally

```bash
git clone https://github.com/Vincent69410/sedimentation-forwarding.git
cd sedimentation-forwarding
pip install -r requirements.txt
python scripts/reproduce_section6.py           # the numbers of Section 6
python scripts/make_figures.py                 # the two data figures, as PDF in figs/
python scripts/closed_loop_scenario.py feed    # recompute one scenario (about ten minutes)
```

## What is in here

| | |
|---|---|
| `sedimentation_forwarding/model.py` | the design chain: stationary profile by shooting, linearization, interface gain `G_o`, interface pole `c*` by a bracketing root finder (Brent), Sylvester kernel, input gain `G`, then the ISS constants, the small-gain ratio (`best`) and the constants of the stability certificate of Table 2 (`certificate`) |
| `sedimentation_forwarding/spectrum.py` | the 5×5 characteristic determinant `D(s)`. Ω₁ in closed form, Ω₂ by RK4 on the augmented flow, root finding by Newton, zero counting by the argument principle |
| `sedimentation_forwarding/examples.py` | the illustrative example of Section 6, and a constructor taking physical parameters directly |
| `sedimentation_forwarding/scenarios.py` | the nonlinear closed loop on the physical domain (shock-capturing scheme compiled with numba): change of set point, step of the feed fraction |
| `sedimentation_forwarding/closed_loop.py` | the same scheme in plain numpy; gives the open-loop drift of the scheme quoted in the paper |
| `sedimentation_forwarding/figures.py` | the two data figures, drawn exactly as printed |
| `sedimentation_forwarding/data/` | the two closed-loop runs and the open-loop drift run |
| `notebooks/worked_example.ipynb` | the Colab notebook |
| `scripts/reproduce_section6.py` | the numbers of Section 6, no notebook |
| `scripts/make_figures.py` | the two data figures as PDF; `--rerun` recomputes the closed-loop runs |
| `scripts/closed_loop_scenario.py` | one closed-loop scenario, `setpoint` or `feed` |

## The operating point

A column of depth 1 m fed with 2 mm polystyrene beads at 11 % solid fraction,
drained at the bottom, with a 20 cm sediment bed. Eight constants.

Beads nearly matched in density to water are chosen on purpose. The margin in
the small-gain condition is governed by `K = g (rho_s - rho_f) / rho_s`, which
is 0.467 m/s^2 here against 6.11 for a mineral grain, and that factor of
thirteen is what allows a bed of 20 cm rather than 2 cm.

```python
import sedimentation_forwarding as sf

P = sf.illustrative_example()
b = sf.best(P, ngrid=14)
c = sf.certificate(P)               # weights of Table 2, kappa = 0.3

print(P.cstar)       # -5.184904e-06   interface pole
print(b['ratio'])    #  0.2863         small-gain condition, satisfied
print(c['alpha_w'])  #  0.03107        certified decay rate
print(c['C'])        #  6.9e+31        constant of the exponential estimate
```

What the paper claims, and what the notebook checks:

* `c*` from the pole equation agrees with the zero of `det D` to eleven
  digits, and the two computations share only the stationary profile and the
  propagator;
* closing the loop moves that zero to `-κ` exactly, uniformly over
  `κ ∈ [3.16e-2, 1]`;
* with the weights of Table 2, the certified rate is `α_w = 3.11e-2 s⁻¹`, with
  `ρ ≥ 1.7e16` and `C ≥ 6.9e31`;
* the nonlinear closed loop, solved on the physical domain by a
  shock-capturing scheme, follows a 10 mm change of set point (within 1 mm
  after 8 s) and, when the reference is updated, returns the bed to its set
  point after a 10 % step of the feed fraction, once the new suspension has
  crossed the free-settling zone (about 21 s). The scheme drifts by about
  16 µm/s in open loop, which leaves a bias of at most 0.1 mm.

## Your own plant

`settling_column` takes the physical parameters:

```python
Q = sf.settling_column(rho_s=2650., d=5e-4, eps_c=0.55, n_s=5,
                       sigma_0=1e5, eps_in_over_eps_c=0.2, z_b=1.0, L=0.02)
Q.solve_cstar()
sf.best(Q)['ratio']
```

Three things govern whether the small-gain condition holds. The compressive
modulus `rho_s * Q.cs[0]**2` of the bed: on the operating point above the
condition fails below about 0.10 MPa. The bed thickness `L`: `alphabar` goes
like `1/L` and `a_o` like `L`, and the condition fails above about 60 cm. And
the density contrast, through `K`: at the same bead diameter but
`rho_s = 2650` the thickness limit drops to 26 cm, and on a 0.5 mm mineral
grain to 7 cm.

One subtlety in `model.py` worth knowing before you change it. The matrix
`B2` built in `Plant._build` carries three contributions: the Jacobian of the
source, the variation of the flux Jacobian along the profile
(`(D_yA[.]) d_x y*`), and the `x`-dependence of the diagonalizing matrix. The
second is easy to drop and costs about 2% on `B2`; it is checked against a
finite-difference linearization of the nonlinear right-hand side.

Two further traps. The argument-principle count
must use the **raw** determinant: normalizing the rows of the matrix is
non-analytic and silently corrupts the winding number. And the contour needs
enough points — a few hundred per oscillation — or `np.unwrap` aliases and
returns a plausible but wrong count.

## Scope

The code is general: the parameters of any settler can be passed to
`Plant` (see `examples.settling_column`). The only operating point provided is
the illustrative example of the paper. `worked_example()` is kept as an alias
of `illustrative_example()` for earlier versions of this code.

## Citing

See `CITATION.cff`, or cite the paper directly.

## License

MIT.
