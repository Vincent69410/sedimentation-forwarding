"""Nonlinear closed-loop simulation on the settling column of Section 6.

Defauts : 800 mailles, 40 s, kappa = 0.3, decalage initial de 10 mm, ce qui
reproduit la figure 3 de l'article (le lit de 20 cm porte 160 mailles).

On resout le systeme 2x2 d'origine sur le domaine PHYSIQUE (0, z_b), en
capture de choc, sans la reformulation a domaine fixe de l'article : le
manteau est un choc capture par le schema, pas une inconnue du probleme.
La loi de commande, elle, utilise bien le noyau de forwarding : a chaque pas
on transporte l'etat sur le repere mobile, on forme Y, puis
    w = eta - Lambda(Y),   u = K w,   v_m = v_m^* + u.

Schema : volumes finis, flux de Rusanov, SSP-RK2, CFL 0.4.
Le choix du domaine physique est deliberé : si la synthese ne marchait que
dans son propre repere, ce test le dirait.
"""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import warnings
import numpy as np

warnings.filterwarnings('ignore')
np.seterr(all='ignore')
import sedimentation_forwarding as sf

# ---------------------------------------------------------------- plant
P = sf.worked_example(Nd=2000)
zb, zs0, epsc = P.zb, P.zs, P.eps_c
K, gam, vms, lam = P.K, P.gam, P.vm, P.lam
ein, fss = P.eps_in, P.fs

S = lambda e: np.where(e > epsc, P.a * (e ** P.n - epsc ** P.n), 0.0)
Sp = lambda e: np.where(e > epsc, P.a * P.n * e ** (P.n - 1), 0.0)
R = lambda e: np.where(e > epsc, gam * e * (1 - epsc) / ((1 - e) * epsc), gam)

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 800
TEND = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
ETA0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2
MODE = sys.argv[5] if len(sys.argv) > 5 else 'closed'

dz = zb / NC
zc = (np.arange(NC) + 0.5) * dz

# ------------------------------------------------- equilibrium, on the grid
def equilibrium(zs):
    """Profil d'equilibre dont le manteau est en zs (meme forme, translate)."""
    e = np.where(zc < zs, ein, np.interp(zc - zs + zs0, P.xg, P.e))
    f = np.where(zc < zs, ein * lam, fss)
    return e, f

e, f = equilibrium(zs0)

# --- pre-relaxation : on laisse le schema trouver SON equilibre discret, puis
#     on translate l'etat de m cellules.  Sur une grille uniforme la translation
#     d'un etat stationnaire est encore stationnaire, aux bords pres : la
#     condition initiale est donc un equilibre deplace au sens discret, et non
#     une interpolation, ce qui supprime la derive numerique de depart.

# ------------------------------------------------------------- numerics
def flux(e, f):
    return f, f ** 2 / e + S(e)

def speed(e, f):
    return np.abs(f / e) + np.sqrt(Sp(e))

def source(e, f, vm):
    return np.zeros_like(e), K * e - R(e) * (f - e * vm)

def rhs(e, f, vm):
    # etats etendus par les conditions aux limites
    eL, fL = ein, ein * (vm + K / gam)           # alimentation en z = 0
    eR, fR = e[-1], e[-1] * vm                   # soutirage en z = zb
    ee = np.concatenate(([eL], e, [eR]))
    ff = np.concatenate(([fL], f, [fR]))
    F1, F2 = flux(ee, ff)
    a = speed(ee, ff)
    am = np.maximum(a[:-1], a[1:])
    # flux de Rusanov aux NC+1 interfaces
    G1 = 0.5 * (F1[:-1] + F1[1:]) - 0.5 * am * (ee[1:] - ee[:-1])
    G2 = 0.5 * (F2[:-1] + F2[1:]) - 0.5 * am * (ff[1:] - ff[:-1])
    s1, s2 = source(e, f, vm)
    return -(G1[1:] - G1[:-1]) / dz + s1, -(G2[1:] - G2[:-1]) / dz + s2

# ------------------------------------------------- blanket level, control
def level(e):
    """Position du manteau : traversee de (eps_in + eps_c)/2, interpolee."""
    thr = 0.5 * (ein + epsc)
    i = np.argmax(e > thr)
    if i == 0:
        return zs0
    a, b = e[i - 1], e[i]
    return zc[i - 1] + (thr - a) / (b - a) * dz

# noyau de forwarding, sur les grilles du modele
L1, zg1 = P.L1, P.zg1          # Omega_1, x de 0 a zs0
L2, xg2 = P.L2, P.xg           # Omega_2, x de zs0 a zb
Kfb = -(P.cstar + KAPPA) / P.G

def control(e, f, zs):
    """u = K ( eta - Lambda(Y) ), Y etant la deviation TRANSPORTEE."""
    eta = zs - zs0
    # Omega_1 : z = x zs/zs0
    z1 = zg1 * zs / zs0
    e1 = np.interp(z1, zc, e); f1 = np.interp(z1, zc, f)
    Y1 = e1 - ein
    Y2 = (f1 - ein * lam) - lam * Y1
    I1 = np.trapezoid(L1[:, 0] * Y1 + L1[:, 1] * Y2, zg1)
    # Omega_2 : z = zb - (zb - x)(zb - zs)/(zb - zs0)
    z2 = zb - (zb - xg2) * (zb - zs) / (zb - zs0)
    e2 = np.interp(z2, zc, e); f2 = np.interp(z2, zc, f)
    de = e2 - P.e; df = f2 - fss
    # Y = M2 ytilde, M2^-1 = [[1,1],[l3,l4]]
    den = P.l4 - P.l3
    Y3 = (P.l4 * de - df) / den
    Y4 = (-P.l3 * de + df) / den
    I2 = np.trapezoid(L2[:, 0] * Y3 + L2[:, 1] * Y4, xg2)
    w = eta - (I1 + I2)
    return Kfb * w, w, eta

def march(e, f, T, vm_fun):
    t = 0.0
    while t < T:
        dt = min(0.4 * dz / speed(e, f).max(), T - t)
        vm = vm_fun(e, f, t)
        k1e, k1f = rhs(e, f, vm)
        e1 = e + dt * k1e; f1 = f + dt * k1f
        k2e, k2f = rhs(e1, f1, vm)
        e = 0.5 * (e + e1 + dt * k2e); f = 0.5 * (f + f1 + dt * k2f)
        e = np.clip(e, 1e-6, 0.95)
        t += dt
    return e, f


t_relax = 10.0
e, f = march(e, f, t_relax, lambda *_: vms)
zs_num = level(e)
m = int(round(ETA0 / dz))
if m > 0:
    e = np.concatenate((np.full(m, ein), e[:-m]))
    f = np.concatenate((np.full(m, ein * lam), f[:-m]))
zs0 = zs_num                      # reference = l'equilibre discret du schema
print(f"  equilibre discret : zs = {zs_num:.6f} (cible {P.zs:.6f}), "
      f"decalage impose {m} cellules = {m*dz*1e3:.2f} mm", flush=True)

# -------------------------------------------------------------- time loop
t = 0.0
hist = {'t': [], 'zs': [], 'vm': [], 'w': [], 'eta': []}
nxt = 0.0
t0 = time.time()
nstep = 0
while t < TEND:
    zs = level(e)
    u, w, eta = control(e, f, zs)
    vm = vms + u if MODE == 'closed' else vms
    vm = max(vm, 0.0)
    if t >= nxt:
        hist['t'].append(t); hist['zs'].append(zs); hist['vm'].append(vm)
        hist['w'].append(w); hist['eta'].append(eta)
        nxt += 0.02
    dt = 0.4 * dz / speed(e, f).max()
    dt = min(dt, TEND - t)
    k1e, k1f = rhs(e, f, vm)
    e1 = e + dt * k1e; f1 = f + dt * k1f
    k2e, k2f = rhs(e1, f1, vm)
    e = 0.5 * (e + e1 + dt * k2e); f = 0.5 * (f + f1 + dt * k2f)
    e = np.clip(e, 1e-6, 0.95)
    t += dt; nstep += 1
    if nstep % 200000 == 0:
        print(f"  t={t:7.3f}  zs={zs:.6f}  vm={vm:.6e}  ({time.time()-t0:.0f}s)", flush=True)

zs = level(e)
hist['t'].append(t); hist['zs'].append(zs); hist['vm'].append(vm)
hist['w'].append(w); hist['eta'].append(eta)
tag = f"{MODE}_{NC}_{KAPPA:g}_{ETA0:g}"
json.dump({k: list(map(float, v)) for k, v in hist.items()}, open(f'nl_{tag}.json', 'w'))
print(f"{MODE}: NC={NC} kappa={KAPPA} eta0={ETA0}  "
      f"eta(0)={hist['eta'][0]:.4e} -> eta(T)={hist['eta'][-1]:.4e}  "
      f"({nstep} pas, {time.time()-t0:.0f}s)")
