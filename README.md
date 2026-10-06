# Boundary stabilization of a sedimentation front by forwarding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vincent69410/sedimentation-forwarding/blob/main/notebooks/worked_example.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Companion code for the paper

> I. Amirat, V. Andrieu, C. Valentin, *Boundary stabilization of the sludge
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
and the notebook reproduces every number of Section 6.

## Run it locally

```bash
git clone https://github.com/Vincent69410/sedimentation-forwarding.git
cd sedimentation-forwarding
pip install -r requirements.txt
python scripts/reproduce_section6.py
```

## What is in here

| | |
|---|---|
| `sedimentation_forwarding/model.py` | the design chain: stationary profile by shooting, linearization, interface gain `G_o`, transmission pole `c*` by fixed point, Sylvester kernel, input gain `G`, then the ISS constants and the small-gain ratio |
| `sedimentation_forwarding/spectrum.py` | the 5×5 characteristic determinant `D(s)`. Ω₁ in closed form, Ω₂ by RK4 on the augmented flow, root finding by Newton, zero counting by the argument principle |
| `sedimentation_forwarding/examples.py` | the operating point of Section 6, and a constructor taking physical parameters directly |
| `notebooks/worked_example.ipynb` | the Colab notebook |
| `scripts/reproduce_section6.py` | the same numbers, no notebook |

## The operating point

A column of depth 1 m fed with 0.5 mm silica grains at 11 % solid fraction,
drained at the bottom, with a 2 cm sediment bed. Eight constants, no data file.

```python
import sedimentation_forwarding as sf

P = sf.worked_example()
b = sf.best(P, ngrid=14)

print(P.cstar)      # -4.358965e-04   transmission pole
print(b['ratio'])   #  0.3598         small-gain condition, satisfied
print(b['margin'])  #  9.5074e-01     certified decay rate
```

What the paper claims, and what the notebook checks:

* `c*` from the fixed-point equation agrees with the zero of `det D` to eleven
  digits, and the two computations share only the stationary profile and the
  propagator;
* closing the loop moves that zero to `-κ` exactly, to a relative 3.5e-11,
  uniformly over `κ ∈ [3.16e-2, 1]`;
* the argument principle finds no zero in the closed right half-plane and
  exactly one in `[-3,0] × [-30,30]`, so the spectral abscissa is exactly `-κ`.

## Your own plant

`settling_column` takes the physical parameters:

```python
Q = sf.settling_column(rho_s=2650., d=5e-4, eps_c=0.55, n_s=5,
                       sigma_0=1e5, eps_in_over_eps_c=0.2, z_b=1.0, L=0.02)
Q.solve_cstar()
sf.best(Q)['ratio']
```

Two things govern whether the small-gain condition holds: the compressive
modulus `rho_s * Q.cs[0]**2` of the bed, and its thickness `L`. On this family
the condition fails below about 0.45 MPa.

Two traps worth knowing if you modify the code. The argument-principle count
must use the **raw** determinant: normalizing the rows of the matrix is
non-analytic and silently corrupts the winding number. And the contour needs
enough points — a few hundred per oscillation — or `np.unwrap` aliases and
returns a plausible but wrong count.

## Scope

The clarifier that motivates the model is not in this repository. The code is
general, so its parameters can be set by anyone who has them, but the
calibration used in the paper rests on plant data that we are not free to
redistribute.

## Citing

See `CITATION.cff`, or cite the paper directly.

## License

MIT.
