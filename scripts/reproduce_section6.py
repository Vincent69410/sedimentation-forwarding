"""Reproduce every number of Section 6 from the command line.

    python scripts/reproduce_section6.py

Takes about two minutes. The notebook in notebooks/ does the same thing with
commentary and a figure.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import sedimentation_forwarding as sf

np.seterr(all="ignore")

P = sf.worked_example()
b = sf.best(P, ngrid=14)

print("=== Table 1 ===")
print(f"  v_m*      {P.vm:.4e} m/s")
print(f"  lambda    {P.lam:.4e} m/s")
print(f"  [eps*]    {P.jump:.4e}")
print(f"  (l3,l4) at zs*   ({P.l3s:.3f}, {P.l4s:.3f}) m/s")
print(f"  (l3,l4) at zb    ({P.l3b:.3f}, {P.l4b:.3f}) m/s")
print(f"  -k^2      {-P.k ** 2:.6f}")
print(f"  mu2bar    {P.mu2bar:.4e} 1/m")
print(f"  T_1       {P.zs / P.lam:.3f} s")
print(f"  c*        {P.cstar:.5e} 1/s")
print(f"  Delta_z   {P.Dz:.5f}")
print(f"  gamma_v   {P.gv:.5f}")
print(f"  G         {P.G:.5f}")

print("\n=== Condition 17 ===")
print(f"  alphabar  {b['alphabar']:.4f}")
print(f"  a_eta     {b['aeta']:.4e}")
print(f"  a_o       {b['ao']:.4e}")
print(f"  ratio     {b['ratio']:.4f}   ({'satisfied' if b['ratio'] < 1 else 'violated'})")
print(f"  beta <=   {b['margin']:.4e} s^-1")

print("\n=== characteristic determinant ===")
for nst in (250, 500, 800):
    sp = sf.Spec(P, nst=nst)
    print(f"  {nst:4d} steps   zero of det D = {np.real(sp.newton(P.cstar * 1.02)):.12e}")
print(f"  fixed point   c*            = {P.cstar:.12e}")

print("\n=== pole placement ===")
sp = sf.Spec(P, nst=500)
worst = 0.0
for kap in np.geomspace(3.16e-2, 1.0, 9):
    Kg = sp.gain(kap)
    r = sp.newton(-kap * 1.00002, "forwarding", Kg)
    r = sp.newton(r, "forwarding", Kg)
    worst = max(worst, abs(np.real(r) / (-kap) - 1.0))
print(f"  worst relative error over kappa in [3.16e-2, 1] : {worst:.2e}")
