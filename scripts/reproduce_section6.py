"""Reproduce every number of Section 6 from the command line.

    python scripts/reproduce_section6.py

Takes about two minutes. The notebook in notebooks/ does the same thing with
commentary and draws the data figures; scripts/make_figures.py writes them
as PDF. The closed-loop numbers are read from the stored runs (see
scripts/closed_loop_scenario.py to recompute them).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import sedimentation_forwarding as sf

np.seterr(all="ignore")

P = sf.illustrative_example()
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

L = P.zb - P.zs
print(f"  |c*|^-1   {1 / abs(P.cstar):.3e} s")
dx = (P.e[1] - P.e[0]) / (P.xg[1] - P.xg[0])
print(f"  mass balance  -v_m* d_x eps*(zs+)/[eps*] = {-P.vm * dx / P.jump:.4e} 1/s")
print(f"  Sed at zs+    {P.fs / P.ep / P.cs[0]:.2e}   crossing time of the bed {L / P.l3s * 1e3:.1f} ms")
print(f"  compression margin eps*(zs+) - eps_c = {P.ep - P.eps_c:.2e}")

print("\n=== small-gain condition ===")
print(f"  alphabar  {b['alphabar']:.4f}")
print(f"  a_eta     {b['aeta']:.4e}")
print(f"  a_o       {b['ao']:.4e}")
print(f"  ratio     {b['ratio']:.4f}   ({'satisfied' if b['ratio'] < 1 else 'violated'})")
print(f"  alpha_sg  {b['margin']:.4e} s^-1")

c = sf.certificate(P)
print("\n=== Table 2 (extended version) and item (iii) of the stability theorem, kappa = 0.3 ===")
print(f"  mu1       {c['mu1']:.3f} 1/m      p2x  {c['p2x']:.2e}")
print(f"  nu2       {c['nu2']:.3e}    theta {c['theta']:.3e}   delta {c['delta']}")
print(f"  sigma_u   {c['sigma_u']:.2e}   sigma_udot {c['sigma_udot']:.2e}   b_w {c['b_w']:.2e}")
print(f"  alpha_w   {c['alpha_w']:.3e} s^-1")
print(f"  rho >=    {c['rho_min']:.2e}   c2/c1 {c['c2_over_c1']:.2e}   C >= {c['C']:.2e}")
print(f"  estimate informative after t = ln C / alpha_w = {c['t_info']:.2e} s")

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

print("\n=== simulation of the closed loop (stored runs) ===")
from sedimentation_forwarding import scenarios, closed_loop
for k, v in scenarios.summary(scenarios.load_runs()).items():
    print(f"  {k:36s} {v:.4g}")
print(f"  drift of the scheme in open loop     {closed_loop.drift() * 1e6:.1f} micrometres/s")
