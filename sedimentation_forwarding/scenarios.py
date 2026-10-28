"""Closed-loop scenarios of the last subsection of Section 6.

The original 2x2 system is solved on the PHYSICAL domain (0, z_b) by a
shock-capturing scheme (finite volumes, Rusanov flux, SSP-RK2, CFL 0.4); the
blanket is a shock captured by the scheme, located at each step by the
crossing of (eps_in + eps_c)/2. At every step the control law

    w = eta - L[Y],   u = K w,   v_m = v_m^* + u

is evaluated, Y being the deviation of the state, mapped onto the reference
configuration, from the reference equilibrium. The reference (z_s^*, v_m^*,
eps_s^*, L, K) is the equilibrium of the current set point and feed
fraction, recomputed from scratch with :func:`settling_column`.

Two scenarios, both with kappa = 0.3 and 800 cells (160 in the bed), shown
in the closed-loop figure of the paper (Figure 3 of the paper, Figure 4 of
the extended version):

    "setpoint"  the set point of the bed goes from 20 to 21 cm at t = 5 s and
                back at t = 40 s (panels (a)-(b));
    "feed"      the feed fraction goes from eps_in to 1.1 eps_in at t = 5 s
                and the reference is switched to the equilibrium of the new
                feed fraction (panels (c)-(d)). With ``update_reference=False``
                the reference stays at the nominal equilibrium; the loop is
                still stable but the bed settles about 17 mm below its set
                point.

The scheme is compiled with numba; one 75 s run takes about ten minutes on one
core (without numba the module still loads, but a run is far slower). The two runs of the figure are stored in ``data/`` and :func:`load_runs`
returns them. :mod:`closed_loop` contains the same scheme in plain numpy,
which is used for the open-loop drift of the scheme.
"""
import json
import os
import time

import numpy as np

try:
    from numba import njit
except ImportError:            # the stored runs can still be loaded and drawn
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .examples import settling_column

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: the two runs of the closed-loop figure: name -> data file
RUNS = {"setpoint": "scenario_setpoint.json", "feed": "scenario_feed.json"}


# ---------------------------------------------------------------- closures
@njit(cache=True)
def _S(e, a, n, epsc):
    return a * (e ** n - epsc ** n) if e > epsc else 0.0


@njit(cache=True)
def _Sp(e, a, n, epsc):
    return a * n * e ** (n - 1) if e > epsc else 0.0


@njit(cache=True)
def _R(e, gam, epsc):
    return gam * e * (1 - epsc) / ((1 - e) * epsc) if e > epsc else gam


# ------------------------------------------------------------------ scheme
@njit(cache=True)
def _rhs(e, f, vm, ein, K, gam, a, n, epsc, dz, de, df):
    """Rusanov fluxes; ghost states from the feed and underflow conditions."""
    nc = e.shape[0]
    G1 = np.empty(nc + 1)
    G2 = np.empty(nc + 1)
    for i in range(nc + 1):
        if i == 0:
            eL, fL = ein, ein * (vm + K / gam)          # feed at z = 0
        else:
            eL, fL = e[i - 1], f[i - 1]
        if i == nc:
            eR, fR = e[nc - 1], e[nc - 1] * vm          # underflow at z = z_b
        else:
            eR, fR = e[i], f[i]
        aL = abs(fL / eL) + np.sqrt(_Sp(eL, a, n, epsc))
        aR = abs(fR / eR) + np.sqrt(_Sp(eR, a, n, epsc))
        am = max(aL, aR)
        F2L = fL * fL / eL + _S(eL, a, n, epsc)
        F2R = fR * fR / eR + _S(eR, a, n, epsc)
        G1[i] = 0.5 * (fL + fR) - 0.5 * am * (eR - eL)
        G2[i] = 0.5 * (F2L + F2R) - 0.5 * am * (fR - fL)
    for i in range(nc):
        de[i] = -(G1[i + 1] - G1[i]) / dz
        df[i] = (-(G2[i + 1] - G2[i]) / dz + K * e[i]
                 - _R(e[i], gam, epsc) * (f[i] - e[i] * vm))


@njit(cache=True)
def _maxspeed(e, f, a, n, epsc):
    m = 0.0
    for i in range(e.shape[0]):
        s = abs(f[i] / e[i]) + np.sqrt(_Sp(e[i], a, n, epsc))
        if s > m:
            m = s
    return m


@njit(cache=True)
def _step(e, f, vm, dt, ein, K, gam, a, n, epsc, dz):
    """One SSP-RK2 step."""
    nc = e.shape[0]
    ke = np.empty(nc)
    kf = np.empty(nc)
    _rhs(e, f, vm, ein, K, gam, a, n, epsc, dz, ke, kf)
    e1 = e + dt * ke
    f1 = f + dt * kf
    _rhs(e1, f1, vm, ein, K, gam, a, n, epsc, dz, ke, kf)
    e2 = 0.5 * (e + e1 + dt * ke)
    f2 = 0.5 * (f + f1 + dt * kf)
    for i in range(nc):
        e2[i] = min(max(e2[i], 1e-6), 0.95)
    return e2, f2


@njit(cache=True)
def _level(e, zc, dz, thr, zdef):
    """Blanket position: crossing of thr, linearly interpolated."""
    for i in range(e.shape[0]):
        if e[i] > thr:
            if i == 0:
                return zdef
            a, b = e[i - 1], e[i]
            return zc[i - 1] + (thr - a) / (b - a) * dz
    return zdef


@njit(cache=True)
def _trap(y, x):
    s = 0.0
    for i in range(x.shape[0] - 1):
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return s


@njit(cache=True)
def _control(e, f, zs, zs0, zc, zb, zg1, L1, xg2, L2, Pe, ein0, lam, fss, l3, l4, Kfb):
    """u = K (eta - L[Y]), Y being the deviation mapped on the reference frame."""
    eta = zs - zs0
    # Omega_1 : z = x zs / zs0
    z1 = zg1 * zs / zs0
    e1 = np.interp(z1, zc, e)
    f1 = np.interp(z1, zc, f)
    Y1 = e1 - ein0
    Y2 = (f1 - ein0 * lam) - lam * Y1
    I1 = _trap(L1[:, 0] * Y1 + L1[:, 1] * Y2, zg1)
    # Omega_2 : z = zb - (zb - x)(zb - zs)/(zb - zs0)
    z2 = zb - (zb - xg2) * (zb - zs) / (zb - zs0)
    e2 = np.interp(z2, zc, e)
    f2 = np.interp(z2, zc, f)
    dee = e2 - Pe
    dff = f2 - fss
    den = l4 - l3
    Y3 = (l4 * dee - dff) / den
    Y4 = (-l3 * dee + dff) / den
    I2 = _trap(L2[:, 0] * Y3 + L2[:, 1] * Y4, xg2)
    w = eta - (I1 + I2)
    return Kfb * w, w, eta


def _reference(P, kappa):
    """The quantities of an equilibrium that the control law needs."""
    c = np.ascontiguousarray
    return dict(zg1=c(P.zg1), L1=c(P.L1), xg2=c(P.xg), L2=c(P.L2), Pe=c(P.e),
                ein0=P.eps_in, lam=P.lam, fss=P.fs, l3=c(P.l3), l4=c(P.l4),
                Kfb=-(P.cstar + kappa) / P.G, vms=P.vm, zsP=P.zs)


def run(scenario, nc=800, tend=75.0, kappa=0.3, t_relax=10.0, tstep=5.0,
        tback=40.0, dL=0.01, dtheta=0.02, update_reference=True, Nd=2000,
        verbose=False):
    """Run one closed-loop scenario.

    Parameters
    ----------
    scenario : {"setpoint", "feed"}
    nc : number of finite-volume cells on (0, z_b).
    tend : final time [s].
    kappa : placed pole [1/s].
    t_relax : the scheme is first run in open loop for this time, so that it
        starts from its OWN discrete equilibrium.
    tstep : time of the change of set point or of feed fraction [s].
    tback : time at which the set point returns to 20 cm ("setpoint" only).
    dL : change of the bed thickness set point [m] ("setpoint").
    dtheta : change of eps_in / eps_c ("feed"); 0.02 is a 10 % step.
    update_reference : "feed" only; switch the reference to the equilibrium
        of the new feed fraction at tstep.
    Nd : grid of the design (points of Omega_2^*).

    Returns
    -------
    dict with lists ``t, zs, zref, vm, w, eta, ein``, sampled every 0.02 s,
    and ``zb``, ``vms``, ``meta``.
    """
    if scenario not in RUNS:
        raise ValueError(f"scenario must be one of {list(RUNS)}")
    P = settling_column(Nd=Nd)
    P.solve_cstar()
    zb, epsc, K, gam, a, n = P.zb, P.eps_c, P.K, P.gam, P.a, float(P.n)
    ein_nom = P.eps_in
    dz = zb / nc
    zc = (np.arange(nc) + 0.5) * dz
    thr = 0.5 * (ein_nom + epsc)
    C0 = _reference(P, kappa)
    C1 = None
    if scenario == "setpoint":
        P1 = settling_column(L=(P.zb - P.zs) + dL, Nd=Nd)
        P1.solve_cstar()
        C1 = _reference(P1, kappa)
    elif update_reference:
        P1 = settling_column(eps_in_over_eps_c=P.eps_in / P.eps_c + dtheta, Nd=Nd)
        P1.solve_cstar()
        C1 = _reference(P1, kappa)
    if verbose and C1 is not None:
        print(f"  second reference: v_m* = {P1.vm:.6e}, c* = {P1.cstar:.6e}, "
              f"G = {P1.G:.6f}", flush=True)

    # relax onto the discrete equilibrium of the scheme
    e = np.where(zc < P.zs, ein_nom, np.interp(zc, P.xg, P.e))
    f = np.where(zc < P.zs, ein_nom * P.lam, P.fs)
    t = 0.0
    while t < t_relax:
        dt = min(0.4 * dz / _maxspeed(e, f, a, n, epsc), t_relax - t)
        e, f = _step(e, f, P.vm, dt, ein_nom, K, gam, a, n, epsc, dz)
        t += dt
    off = _level(e, zc, dz, thr, P.zs) - P.zs     # discrete minus exact position

    hist = {k: [] for k in ("t", "zs", "zref", "vm", "w", "eta", "ein")}
    t, nxt, t0, nstep = 0.0, 0.0, time.time(), 0
    while t < tend:
        switched = (C1 is not None and t >= tstep
                    and (scenario == "feed" or t < tback))
        C = C1 if switched else C0
        zs0 = C["zsP"] + off
        ein = ein_nom + (dtheta * epsc if (scenario == "feed" and t >= tstep) else 0.0)
        zs = _level(e, zc, dz, thr, zs0)
        u, w, eta = _control(e, f, zs, zs0, zc, zb, C["zg1"], C["L1"], C["xg2"],
                             C["L2"], C["Pe"], C["ein0"], C["lam"], C["fss"],
                             C["l3"], C["l4"], C["Kfb"])
        vm = max(C["vms"] + u, 0.0)
        if t >= nxt:
            for k, v in zip(hist, (t, zs, zs0, vm, w, eta, ein)):
                hist[k].append(float(v))
            nxt += 0.02
        dt = min(0.4 * dz / _maxspeed(e, f, a, n, epsc), tend - t)
        e, f = _step(e, f, vm, dt, ein, K, gam, a, n, epsc, dz)
        t += dt
        nstep += 1
        if verbose and nstep % 500000 == 0:
            print(f"  {scenario}: t = {t:6.2f} s, eta = {eta * 1e3:+.3f} mm, "
                  f"v_m = {vm * 100:.4f} cm/s ({time.time() - t0:.0f} s)", flush=True)
    hist["zb"] = zb
    hist["vms"] = P.vm
    hist["meta"] = dict(scenario=scenario, nc=nc, kappa=kappa, tstep=tstep,
                        tback=tback, dL=dL, dtheta=dtheta,
                        update_reference=update_reference, off=off,
                        vm1=(C1["vms"] if C1 else None))
    if verbose:
        print(f"  {scenario}: done in {time.time() - t0:.0f} s, {nstep} steps", flush=True)
    return hist


def load_runs():
    """The two stored runs, as computed by :func:`run`."""
    return {name: json.load(open(os.path.join(_DATA, fname)))
            for name, fname in RUNS.items()}


def recompute_runs(verbose=False):
    """Recompute the two runs (about ten minutes each)."""
    return {name: run(name, verbose=verbose) for name in RUNS}


def summary(runs):
    """The numbers quoted in the last subsection of Section 6."""
    out = {}
    A = runs["setpoint"]
    t = np.array(A["t"])
    bed = (A["zb"] - np.array(A["zs"])) * 1e3
    ref = (A["zb"] - np.array(A["zref"])) * 1e3
    vm = np.array(A["vm"]) * 100
    tstep, tback = A["meta"]["tstep"], A["meta"]["tback"]
    for name, (t1, t2) in (("up", (tstep, tback)), ("down", (tback, t[-1] + 1))):
        m = (t >= t1) & (t < t2)
        err = np.abs(bed[m] - ref[m])
        late = np.where(err > 1.0)[0]
        out[f"setpoint_{name}_within_1mm_after_s"] = (t[m][late[-1] + 1] - t1) if len(late) else 0.0
    out["setpoint_vm_min_cm_s"] = vm.min()
    out["setpoint_vm_max_cm_s"] = vm.max()
    B = runs["feed"]
    t = np.array(B["t"])
    bed = (B["zb"] - np.array(B["zs"])) * 1e3
    vm = np.array(B["vm"]) * 100
    out["feed_vm_star_new_cm_s"] = B["meta"]["vm1"] * 100 if B["meta"]["vm1"] else None
    out["feed_vm_min_cm_s"] = vm.min()
    out["feed_bed_max_mm"] = bed.max()
    out["feed_t_bed_max_s"] = t[np.argmax(bed)]
    out["feed_final_error_mm"] = bed[-1] - (B["zb"] - B["zref"][-1]) * 1e3
    # arrival of the new suspension at the bed: x(t) = int (v_m + K/gamma) dt
    P = settling_column(Nd=400)
    m = t >= B["meta"]["tstep"]
    tm, vmm, zsm = t[m], np.array(B["vm"])[m], np.array(B["zs"])[m]
    X = np.concatenate(([0.0], np.cumsum(0.5 * (vmm[1:] + vmm[:-1] + 2 * P.K / P.gam)
                                         * np.diff(tm))))
    out["feed_front_arrival_s"] = tm[np.argmax(X >= zsm)]
    return out
