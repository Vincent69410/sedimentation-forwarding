# Boundary stabilization of a sedimentation front by forwarding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vincent69410/sedimentation-forwarding/blob/main/notebooks/worked_example.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for the paper

> M. O. Amirat, V. Andrieu, M. Bajodek, C. Valentin, *Boundary stabilization of the sludge
> blanket of a continuous clarifier–thickener by forwarding*, 2026.

A suspension settles in a column. Above a critical solid fraction the particles
touch and carry a stress, below it they do not, and the surface separating the
two regimes is a shock — the sludge blanket — whose position is what we want to
regulate, using the underflow pump at the bottom as the only input. The
linearized plant is an input-to-state stable distributed layer driving a scalar
in which the control does not appear, which is the configuration *forwarding*
was devised for.

The code here does three things: it builds a stationary operating point and the
forwarding design on top of it, it evaluates the small-gain condition that
certifies closed-loop stability, and it computes the exact closed-loop spectrum
from a characteristic determinant that discretizes the generator nowhere.

## Run it now

Click the Colab badge. Nothing to install, about three minutes on a free CPU,
and the notebook reproduces every number of Section 6 and draws its two
figures (Figure 3, the stationary profile and the pole equation; Figure 4, the
nonlinear closed loop).

## Run it locally

```bash
git clone https://github.com/Vincent69410/sedimentation-forwarding.git
cd sedimentation-forwarding
pip install -r requirements.txt
python scripts/reproduce_section6.py   # the numbers of Section 6
python scripts/make_figures.py         # Figures 3 and 4, as PDF in figs/
```

## What is in here

| | |
|---|---|
| `sedimentation_forwarding/model.py` | the design chain: stationary profile by shooting, linearization, interface gain `G_o`, transmission pole `c*` by fixed point, Sylvester kernel, input gain `G`, then the ISS constants and the small-gain ratio |
| `sedimentation_forwarding/spectrum.py` | the 5×5 characteristic determinant `D(s)`. Ω₁ in closed form, Ω₂ by RK4 on the augmented flow, root finding by Newton, zero counting by the argument principle |
| `sedimentation_forwarding/examples.py` | the operating point of Section 6, and a constructor taking physical parameters directly |
| `sedimentation_forwarding/closed_loop.py` | the nonlinear plant on the physical domain, shock-capturing scheme, in open or closed loop |
| `sedimentation_forwarding/figures.py` | Figures 3 and 4, drawn exactly as printed |
| `sedimentation_forwarding/data/` | the three nonlinear runs of Figure 4 (about a quarter of an hour each to recompute) |
| `notebooks/worked_example.ipynb` | the Colab notebook |
| `scripts/reproduce_section6.py` | the same numbers, no notebook |
| `scripts/make_figures.py` | Figures 3 and 4 as PDF; `--rerun` recomputes the nonlinear runs |
| `scripts/nonlinear_closed_loop.py` | one nonlinear run, with its parameters on the command line |

## The operating point

A column of depth 1 m fed with 2 mm polystyrene beads at 11 % solid fraction,
drained at the bottom, with a 20 cm sediment bed. Eight constants.

Beads nearly matched in density to water are chosen on purpose. The margin in
the small-gain condition is governed by `K = g (rho_s - rho_f) / rho_s`, which
is 0.467 m/s^2 here against 6.11 for a mineral grain, and that factor of
thirteen is what allows a bed of 20 cm rather than 2 cm.

```python
import sedimentation_forwarding as sf

P = sf.worked_example()
b = sf.best(P, ngrid=14)

print(P.cstar)      # -5.184904e-06   transmission pole
print(b['ratio'])   #  0.2863         small-gain condition, satisfied
print(b['margin'])  #  0.053743       certified decay rate (also bounded by 2*kappa)
```

What the paper claims, and what the notebook checks:

* `c*` from the fixed-point equation agrees with the zero of `det D` to eleven
  digits, and the two computations share only the stationary profile and the
  propagator;
* closing the loop moves that zero to `-κ` exactly, uniformly over
  `κ ∈ [3.16e-2, 1]`;
* the nonlinear closed loop, solved on the physical domain by a shock-capturing
  scheme, returns a 10 mm displacement of the bed to its set point in about
  twelve seconds (Figure 4, `sedimentation_forwarding/closed_loop.py`).

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

The municipal clarifier that motivates the model is not in this repository.
The code is general, so its parameters can be set by anyone who has them, but
its calibration rests on plant data that we are not free to redistribute.

## Citing

See `CITATION.cff`, or cite the paper directly.

## License

MIT.
