"""The nonlinear plant in open or closed loop, in plain numpy.

The original 2x2 system is solved on the PHYSICAL domain (0, z_b) by a
shock-capturing scheme, without the fixed-domain reformulation of the paper:
the blanket is a shock captured by the scheme, not an unknown of the problem.
In closed loop, the control law uses the forwarding kernel: at each step the
state is mapped onto the moving frame, Y is formed, and

    w = eta - L[Y],   u = K w,   v_m = v_m^* + u.

Scheme: finite volumes, Rusanov flux, SSP-RK2, CFL 0.4.

This module is the reference implementation of the scheme. The paper uses it
for one number: started at its own discrete equilibrium and left in open
loop, the scheme thins the 20 cm bed by 0.62 mm in 40 s, a drift of
16 micrometres per second (:func:`drift`). The run is stored in ``data/``.
The closed-loop scenarios of the paper use the same scheme compiled with
numba, in :mod:`scenarios`. One 40 s run here takes about a quarter of an hour.
"""
import json
import os
import time

import numpy as np

from .examples import worked_example

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: the open-loop run used for the drift of the scheme
DRIFT_RUN = "nl_open_eq.json"


def simulate(P=None, nc=800, tend=40.0, kappa=0.3, eta0=1e-2, mode="closed",
             t_relax=10.0, verbose=False):
    """Run the nonlinear plant, in open or closed loop.

    Parameters
    ----------
    P : Plant, optional
        operating point; default ``worked_example(Nd=2000)``.
    nc : int
        number of finite-volume cells on (0, z_b).
    tend : float
        final time [s].
    kappa : float
        placed pole [1/s], used in closed loop only.
    eta0 : float
        initial displacement of the blanket [m], positive downwards.
    mode : {"closed", "open"}
        with or without the feedback.
    t_relax : float
        the scheme is first run in open loop for this time, so that it
        settles on its OWN discrete equilibrium; the state is then shifted by
        ``round(eta0 / dz)`` cells.  On a uniform grid a shifted steady state
        is still steady (up to the boundaries), so the initial condition is a
        displaced equilibrium in the discrete sense, which removes the
        numerical drift at start-up.

    Returns
    -------
    dict with lists ``t, zs, vm, w, eta``, sampled every 0.02 s.
    """
    if mode not in ("closed", "open"):
        raise ValueError("mode must be 'closed' or 'open'")
    if P is None:
        P = worked_example(Nd=2000)

    zb, epsc = P.zb, P.eps_c
    K, gam, vms, lam = P.K, P.gam, P.vm, P.lam
    ein, fss = P.eps_in, P.fs

    def S(e):
        return np.where(e > epsc, P.a * (e ** P.n - epsc ** P.n), 0.0)

    def Sp(e):
        return np.where(e > epsc, P.a * P.n * e ** (P.n - 1), 0.0)

    def R(e):
        return np.where(e > epsc, gam * e * (1 - epsc) / ((1 - e) * epsc), gam)

    dz = zb / nc
    zc = (np.arange(nc) + 0.5) * dz

    # ---------------------------------------------------------- numerics
    def flux(e, f):
        return f, f ** 2 / e + S(e)

    def speed(e, f):
        return np.abs(f / e) + np.sqrt(Sp(e))

    def rhs(e, f, vm):
        # ghost states given by the boundary conditions
        eL, fL = ein, ein * (vm + K / gam)        # feed at z = 0
        eR, fR = e[-1], e[-1] * vm                # underflow at z = z_b
        ee = np.concatenate(([eL], e, [eR]))
        ff = np.concatenate(([fL], f, [fR]))
        F1, F2 = flux(ee, ff)
        a = speed(ee, ff)
        am = np.maximum(a[:-1], a[1:])
        # Rusanov flux at the nc + 1 interfaces
        G1 = 0.5 * (F1[:-1] + F1[1:]) - 0.5 * am * (ee[1:] - ee[:-1])
        G2 = 0.5 * (F2[:-1] + F2[1:]) - 0.5 * am * (ff[1:] - ff[:-1])
        s2 = K * e - R(e) * (f - e * vm)
        return -(G1[1:] - G1[:-1]) / dz, -(G2[1:] - G2[:-1]) / dz + s2

    def step(e, f, vm, dt):
        k1e, k1f = rhs(e, f, vm)
        e1 = e + dt * k1e
        f1 = f + dt * k1f
        k2e, k2f = rhs(e1, f1, vm)
        e = 0.5 * (e + e1 + dt * k2e)
        f = 0.5 * (f + f1 + dt * k2f)
        return np.clip(e, 1e-6, 0.95), f

    ref = {"zs": P.zs}      # reference blanket position, updated after relaxation

    def level(e):
        """Blanket position: crossing of (eps_in + eps_c)/2, interpolated."""
        thr = 0.5 * (ein + epsc)
        i = np.argmax(e > thr)
        if i == 0:
            return ref["zs"]
        a, b = e[i - 1], e[i]
        return zc[i - 1] + (thr - a) / (b - a) * dz

    # forwarding kernel, on the grids of the model
    L1, zg1 = P.L1, P.zg1      # Omega_1, x from 0 to z_s^*
    L2, xg2 = P.L2, P.xg       # Omega_2, x from z_s^* to z_b
    Kfb = -(P.cstar + kappa) / P.G
    den = P.l4 - P.l3

    def control(e, f, zs):
        """u = K (eta - L[Y]), Y being the deviation mapped on the moving frame."""
        zs0 = ref["zs"]
        eta = zs - zs0
        # Omega_1 : z = x zs / zs0
        z1 = zg1 * zs / zs0
        e1 = np.interp(z1, zc, e)
        f1 = np.interp(z1, zc, f)
        Y1 = e1 - ein
        Y2 = (f1 - ein * lam) - lam * Y1
        I1 = np.trapezoid(L1[:, 0] * Y1 + L1[:, 1] * Y2, zg1)
        # Omega_2 : z = zb - (zb - x)(zb - zs)/(zb - zs0)
        z2 = zb - (zb - xg2) * (zb - zs) / (zb - zs0)
        e2 = np.interp(z2, zc, e)
        f2 = np.interp(z2, zc, f)
        de = e2 - P.e
        df = f2 - fss
        Y3 = (P.l4 * de - df) / den
        Y4 = (-P.l3 * de + df) / den
        I2 = np.trapezoid(L2[:, 0] * Y3 + L2[:, 1] * Y4, xg2)
        w = eta - (I1 + I2)
        return Kfb * w, w, eta

    # --------------------------------------- start from a displaced equilibrium
    zs_ref = P.zs
    e = np.where(zc < zs_ref, ein, np.interp(zc - zs_ref + P.zs, P.xg, P.e))
    f = np.where(zc < zs_ref, ein * lam, fss)
    t = 0.0
    while t < t_relax:
        dt = min(0.4 * dz / speed(e, f).max(), t_relax - t)
        e, f = step(e, f, vms, dt)
        t += dt
    zs_num = level(e)
    m = int(round(eta0 / dz))
    if m > 0:
        e = np.concatenate((np.full(m, ein), e[:-m]))
        f = np.concatenate((np.full(m, ein * lam), f[:-m]))
    ref["zs"] = zs_num       # reference = the discrete equilibrium of the scheme
    if verbose:
        print(f"  discrete equilibrium: zs = {zs_num:.6f} (target {P.zs:.6f}), "
              f"imposed offset {m} cells = {m * dz * 1e3:.2f} mm", flush=True)

    # --------------------------------------------------------- time loop
    t = 0.0
    hist = {"t": [], "zs": [], "vm": [], "w": [], "eta": []}
    nxt = 0.0
    t0 = time.time()
    nstep = 0
    vm = vms
    while t < tend:
        zs = level(e)
        u, w, eta = control(e, f, zs)
        vm = vms + u if mode == "closed" else vms
        vm = max(vm, 0.0)
        if t >= nxt:
            for k, v in zip(("t", "zs", "vm", "w", "eta"), (t, zs, vm, w, eta)):
                hist[k].append(float(v))
            nxt += 0.02
        dt = min(0.4 * dz / speed(e, f).max(), tend - t)
        e, f = step(e, f, vm, dt)
        t += dt
        nstep += 1
        if verbose and nstep % 200000 == 0:
            print(f"  t={t:7.3f}  zs={zs:.6f}  vm={vm:.6e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    zs = level(e)
    for k, v in zip(("t", "zs", "vm", "w", "eta"), (t, zs, vm, w, eta)):
        hist[k].append(float(v))
    if verbose:
        print(f"{mode}: nc={nc} kappa={kappa} eta0={eta0}  "
              f"eta(0)={hist['eta'][0]:.4e} -> eta(T)={hist['eta'][-1]:.4e}  "
              f"({nstep} steps, {time.time() - t0:.0f}s)")
    return hist


def load_drift_run():
    """The open-loop run from the discrete equilibrium, as computed by
    ``simulate(mode="open", eta0=0.0)``."""
    return json.load(open(os.path.join(_DATA, DRIFT_RUN)))


def drift(run=None, zb=1.0):
    """Drift of the bed thickness in open loop [m/s], over the whole run."""
    if run is None:
        run = load_drift_run()
    t = np.array(run["t"])
    bed = zb - np.array(run["zs"])
    return (bed[-1] - bed[0]) / (t[-1] - t[0])
