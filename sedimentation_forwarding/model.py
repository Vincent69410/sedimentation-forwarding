"""Linearized design chain for the 2x2 sedimentation model.

Stationary profile -> linearization -> interface gain G_o, transmission pole
c*, Sylvester kernel, input gain G -> ISS constants (alphabar, d1, d4, Sigma_b,
I_eta, I_o, J_L, chi) -> small-gain condition  a_eta + a_o < alphabar.

Model, in the transported frame of the paper (c_eta = 0, kappa_int = 0):

    d_t eps + d_z f = 0
    d_t f  + d_z( f^2/eps + sigma_e(eps) ) = K eps - R(eps) (f - eps v_m)

with the power-law closures

    sigma_e(eps) = a (eps^n - eps_c^n)_+
    R(eps)       = gamma * eps (1 - eps_c) / ((1 - eps) eps_c)   for eps > eps_c
                 = gamma                                          otherwise.

Reference
---------
M. O. Amirat, V. Andrieu, M. Bajodek, C. Valentin, "Boundary stabilization of a
moving interface in a 2x2 weakly hyperbolic system of balance laws:
application to continuous sedimentation".
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

_trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


class Plant:
    def __init__(self, *, zb, zs, eps_c, n, a, rho_s, gam, K, theta, Nd=1200):
        """All physical parameters are required (see ``examples.settling_column``)."""
        self.zb, self.zs, self.eps_c, self.n = zb, zs, eps_c, n
        self.a = a
        self.gam, self.K, self.theta = gam, K, theta
        self.eps_in = theta * eps_c
        self.Nd = Nd
        self.ok = False
        self._build()

    # ------------------------- constitutive laws ------------------------
    def S(self, e):   return np.where(e > self.eps_c, self.a * (e ** self.n - self.eps_c ** self.n), 0.0)
    def Sp(self, e):  return np.where(e > self.eps_c, self.a * self.n * e ** (self.n - 1), 0.0)
    def Spp(self, e): return np.where(e > self.eps_c, self.a * self.n * (self.n - 1) * e ** (self.n - 2), 0.0)
    def R(self, e):
        ec = self.eps_c
        return np.where(e > ec, self.gam * e * (1 - ec) / ((1 - e) * ec), self.gam)
    def Rp(self, e):
        ec = self.eps_c
        return np.where(e > ec, self.gam * (1 - ec) / ec / (1 - e) ** 2, 0.0)

    # ------------------------ stationary profile ------------------------
    def _eps_plus(self, fs):
        h = lambda e: fs ** 2 / self.eps_in - (fs ** 2 / e + float(self.S(e)))
        lo, hi = self.eps_c * (1 + 1e-12), 0.995
        if h(lo) * h(hi) > 0:
            raise ValueError('pas de trace de compression')
        return brentq(h, lo, hi, xtol=1e-15)

    def _rhs(self, z, e, fs, vm):
        e = e[0]
        num = self.K * e - float(self.R(e)) * (fs - e * vm)
        den = float(self.Sp(e)) - (fs / e) ** 2
        return [num / den]

    def _bottom(self, vm):
        fs = self.eps_in * (vm + self.K / self.gam)
        ep = self._eps_plus(fs)
        s = solve_ivp(self._rhs, [self.zs, self.zb], [ep], args=(fs, vm),
                      rtol=1e-11, atol=1e-14, max_step=(self.zb - self.zs) / 200)
        if not s.success:
            raise ValueError('integration')
        return s.y[0, -1] - fs / vm

    def _build(self):
        # shooting on v_m^* so that f_s^*/v_m = eps(z_b)
        lo, hi = 1e-7, 1.0
        vs = np.geomspace(lo, hi, 220)
        prev = None
        root = None
        for v in vs:
            try:
                r = self._bottom(v)
            except Exception:
                prev = None
                continue
            if prev is not None and np.isfinite(r) and np.isfinite(prev[1]) and prev[1] * r < 0:
                try:
                    root = brentq(self._bottom, prev[0], v, xtol=1e-15)
                except Exception:
                    root = None
                if root is not None:
                    break
            prev = (v, r)
        if root is None:
            return
        self.vm = root
        self.fs = self.eps_in * (self.vm + self.K / self.gam)
        self.lam = self.fs / self.eps_in
        self.ep = self._eps_plus(self.fs)
        self.jump = self.ep - self.eps_in
        if self.jump <= 0:
            return

        Nd = self.Nd
        self.xg = np.linspace(self.zs, self.zb, Nd)
        self.dx = self.xg[1] - self.xg[0]
        s = solve_ivp(self._rhs, [self.zs, self.zb], [self.ep], args=(self.fs, self.vm),
                      t_eval=self.xg, rtol=1e-11, atol=1e-14,
                      max_step=(self.zb - self.zs) / 200)
        if not s.success:
            return
        e = s.y[0]
        if e.min() <= self.eps_c or e.max() >= 0.999:
            return
        self.e = e
        vsd = self.fs / e
        csd = np.sqrt(self.Sp(e))
        self.cs, self.vs_ = csd, vsd
        self.l3, self.l4 = vsd + csd, vsd - csd
        if not (self.l4.max() < 0 < self.l3.min()):
            return                                   # not subcritical
        deps = (self.K * e - self.R(e) * (self.fs - e * self.vm)) / (self.Sp(e) - (self.fs / e) ** 2)
        self.deps = deps
        dv = -self.fs / e ** 2 * deps
        dcs = self.Spp(e) / (2 * csd) * deps
        self.l3p, self.l4p = dv + dcs, dv - dcs

        Rd, Rp = self.R(e), self.Rp(e)
        Psi0p = Rp * e + Rd
        # D_yA[.] d_x y* : the flux Jacobian varies along the profile.  With
        # d_x f_s* = 0 this contributes -deps * (sigma_e'' + 2 v_s (v_s - l_j)/eps)
        # to the j-th column, and v_s - l_3 = -c_s, v_s - l_4 = +c_s.
        gA = self.Spp(e) - 2 * vsd * csd / e
        gB = self.Spp(e) + 2 * vsd * csd / e
        b3 = self.K - Rp * self.fs + Psi0p * self.vm - Rd * self.l3 - deps * gA
        b4 = self.K - Rp * self.fs + Psi0p * self.vm - Rd * self.l4 - deps * gB
        self.B2 = np.zeros((Nd, 2, 2)); self.C1 = np.zeros((Nd, 2)); self.C2 = np.zeros((Nd, 2))
        self.D2 = np.zeros((Nd, 2)); self.A2inv = np.zeros((Nd, 2, 2)); self.A2invB2 = np.zeros((Nd, 2, 2))
        L = self.zb - self.zs
        for i in range(Nd):
            den = self.l4[i] - self.l3[i]
            B = (np.outer([-1, 1], [b3[i], b4[i]]) +
                 np.outer([self.l3[i], -self.l4[i]], [self.l3p[i], self.l4p[i]])) / den
            self.B2[i] = B
            pref = deps[i] / den
            w = (self.zb - self.xg[i]) / L
            self.C1[i] = pref * np.array([-self.l3[i] * self.l4[i] / L, self.l3[i] * self.l4[i] / L])
            self.C2[i] = pref * np.array([w * self.l4[i], -w * self.l3[i]])
            self.D2[i] = (Rd[i] * e[i]) / (2 * csd[i]) * np.array([1., -1.])
            Ai = np.diag([1 / self.l3[i], 1 / self.l4[i]])
            self.A2inv[i] = Ai; self.A2invB2[i] = Ai @ B

        self.l3s, self.l4s, self.l3b, self.l4b = self.l3[0], self.l4[0], self.l3[-1], self.l4[-1]
        self.k = -self.l4s / self.l3s          # reflection coefficient is -k^2
        self.Ma = self.vs_[0] / self.cs[0]
        self.mu2bar = np.log((self.l3b * self.l3s) / (abs(self.l4b) * abs(self.l4s))) / (4 * L)
        # transported frame: c_eta = 0, kappa_int = 0
        self.Go = np.array([self.lam * (self.lam - self.l3s) / self.l3s,
                            (2 * self.lam - self.l3s) / self.l3s,
                            self.l4s * (self.l3s - self.l4s) / self.l3s]) / self.jump
        self.Kint = np.array([self.lam ** 2 / self.l3s ** 2, 2 * self.lam / self.l3s ** 2,
                              -self.l4s ** 2 / self.l3s ** 2])
        self.ok = True

    # ------------------ design: kernel, c^*, input gain -----------------
    def _Psi1(self, x, c):
        lam, gam = self.lam, self.gam
        A = np.exp(c / lam * (x - self.zs)); B = np.exp((c + gam) / lam * (x - self.zs))
        return np.array([[A, (c + gam) / (gam * lam) * (A - B)], [0.0, B]])

    def design(self, c):
        Nd, dx = self.Nd, self.dx
        Xi = c * self.A2inv - self.A2invB2
        Psi = np.eye(2); Psis = np.empty((Nd, 2, 2)); Psis[0] = Psi
        for i in range(Nd - 1):
            M = 0.5 * (Xi[i] + Xi[i + 1]) * dx
            Psi = Psi @ (np.eye(2) + M + 0.5 * M @ M); Psis[i + 1] = Psi
        v1, v2 = Psis[-1] @ np.array([1., 1.])
        Dz = v1 - self.Kint[2] * v2
        N21 = -self.Go[2] * v2 / Dz; N22 = self.Go[2] - N21 * self.Kint[2]
        Nm1 = np.array([N21 * self.Kint[0] - self.Go[0], N21 * self.Kint[1] - self.Go[1]])
        N2x = np.einsum('j,ijk->ik', [N21, N22], Psis)
        L2 = np.einsum('ik,ikl->il', N2x, self.A2inv)
        m1 = _trap(np.einsum('il,il->i', L2, self.C1), self.xg)
        m2 = _trap(np.einsum('il,il->i', L2, self.C2), self.xg)
        iL2D2 = _trap(np.einsum('il,il->i', L2, self.D2), self.xg)
        gv = 1.0 / (1.0 + m2)
        F = gv * (-m1)                                   # transported frame
        return dict(F=F, gv=gv, Dz=Dz, N21=N21, N22=N22, Nm1=Nm1, L2=L2, iL2D2=iL2D2)

    def solve_cstar(self, lo=None, hi=None):
        if lo is None:
            lo = -50 * abs(self.lam / self.zs)
        if hi is None:
            hi = 1e-12
        f = lambda c: self.design(c)['F'] - c
        grid = -np.geomspace(abs(lo), 1e-12, 200)
        prev = None; root = None
        for c in grid:
            try:
                v = f(c)
            except Exception:
                prev = None; continue
            if not np.isfinite(v):
                prev = None; continue
            if prev is not None and prev[1] * v < 0:
                root = brentq(f, prev[0], c, xtol=1e-15); break
            prev = (c, v)
        if root is None:
            return None
        self.cstar = root
        d = self.design(root)
        self.d = d
        A1inv = np.linalg.inv(np.array([[self.lam, 1.0], [0.0, self.lam]]))
        Ng1 = 1200
        self.zg1 = np.linspace(0, self.zs, Ng1)
        self.L1 = np.array([(d['Nm1'] @ self._Psi1(x, root)) @ A1inv for x in self.zg1]) * d['gv']
        self.L2 = d['L2'] * d['gv']
        # input gain G, both zones (kernel lemma of the paper)
        N1_0 = d['Nm1'] @ self._Psi1(0.0, root)
        N2_zb = np.array([d['N21'], d['N22']]) @ self._PsiEnd(root)
        Dbot = self.e[-1] / (self.l4[-1] - self.vm)
        iL1D1 = _trap(self.L1[:, 1] * self.gam * self.eps_in, self.zg1)
        self.G = d['gv'] * (-N1_0[1] * self.eps_in + N2_zb[1] * Dbot - d['iL2D2']) - iL1D1
        self.Dz, self.gv = d['Dz'], d['gv']
        return root

    def _PsiEnd(self, c):
        Xi = c * self.A2inv - self.A2invB2
        Psi = np.eye(2)
        for i in range(self.Nd - 1):
            M = 0.5 * (Xi[i] + Xi[i + 1]) * self.dx
            Psi = Psi @ (np.eye(2) + M + 0.5 * M @ M)
        return Psi


# ===================== ISS / small-gain constants ======================
def constants(P, mu1, mu2, r, nui, p1, p2, p2x):
    """P must already have gone through solve_cstar().  Here p3 = 1, p4 = r."""
    xg, L = P.xg, P.zb - P.zs
    P11 = np.exp(-2 * mu2 * (xg - P.zs))
    P22 = r * np.exp(+2 * mu2 * (xg - P.zb))

    # alphabar
    q11 = 2 * mu2 * P.l3 - P.l3p - 2 * P.B2[:, 0, 0]
    q22 = 2 * mu2 * np.abs(P.l4) - P.l4p - 2 * P.B2[:, 1, 1]
    off = -(np.sqrt(P11 / P22) * P.B2[:, 0, 1] + np.sqrt(P22 / P11) * P.B2[:, 1, 0])
    tr = q11 + q22; det = q11 * q22 - off ** 2
    ab = (0.5 * (tr - np.sqrt(np.maximum(tr ** 2 - 4 * det, 0.0)))).min()

    # boundary dissipation
    # Y_3(z_s*) = K_int . Y_o  with  K_int = [lam^2/l3^2, 2 lam/l3^2, -l4^2/l3^2].
    # The Young expansion of p3 l3 Y_3^2 therefore carries K_int_j SQUARED:
    # (lam^2/l3^2)^2 on Y_1 and k^4 on Y_4, not K_int_j itself.
    Bint1 = (P.lam / P.l3s) ** 2
    d1 = p1 * P.lam * np.exp(-2 * mu1 * P.zs) - 2 * (1 + 1 / nui) * P.l3s * Bint1 ** 2
    d4 = r * np.exp(-2 * mu2 * L) * abs(P.l4s) - P.l3s * (1 + nui) * P.k ** 4
    Sig = P.Go[0] ** 2 / d1 + P.Go[2] ** 2 / d4 if (d1 > 0 and d4 > 0) else np.inf

    # domain gains
    Ie = _trap(P11 * P.C1[:, 0] ** 2 + P22 * P.C1[:, 1] ** 2, xg)
    Io = _trap(P11 * P.C2[:, 0] ** 2 + P22 * P.C2[:, 1] ** 2, xg)

    # J_L and chi
    e1 = np.exp(-2 * mu1 * P.zg1)
    JL1 = _trap(P.L1[:, 0] ** 2 / (p1 * e1) + P.L1[:, 1] ** 2 / (p2 * e1), P.zg1)
    JL2 = _trap(P.L2[:, 0] ** 2 / P11 + P.L2[:, 1] ** 2 / P22, xg)
    JL = JL1 + JL2
    chi = _trap(np.einsum('il,il->i', P.L2, P.C1), xg)

    aeta = chi + np.sqrt(JL * Ie)
    ao = Sig * Io
    return dict(alphabar=ab, d1=d1, d4=d4, Sigma_b=Sig, Ieta=Ie, Io=Io,
                JL=JL, JL1=JL1, JL2=JL2, chi=chi, aeta=aeta, ao=ao,
                margin=ab - aeta - ao, ratio=(aeta + ao) / ab if ab > 0 else np.inf)


def best(P, ngrid=14):
    """Sweep (mu2, r, nu_i, p1); mu1 and p2x are sent to their useful limit."""
    L = P.zb - P.zs
    out = None
    mu1 = 1e-6                      # p1 grand -> d1 > 0 sans cout : mu1 -> 0
    p2x = 1e12
    p2 = 1e12                       # J_L1 -> 0
    for th in np.linspace(0.02, 0.98, ngrid):
        mu2 = th * P.mu2bar
        for nui in np.geomspace(1e-8, 1e-1, 6):
            lo = (1 + nui) * P.k ** 3 * np.exp(2 * mu2 * L)
            hi = np.exp(-2 * mu2 * L) * P.l3b / abs(P.l4b)
            if not (lo < hi):
                continue
            for u in np.linspace(0.02, 0.98, 7):
                r = lo * (hi / lo) ** u
                for p1 in np.geomspace(1e3, 1e14, 5):
                    c = constants(P, mu1, mu2, r, nui, p1, p2, p2x)
                    if c['d1'] <= 0 or c['d4'] <= 0 or c['alphabar'] <= 0:
                        continue
                    if out is None or c['ratio'] < out['ratio']:
                        out = dict(c, mu1=mu1, mu2=mu2, r=r, nui=nui, p1=p1, p2=p2, p2x=p2x,
                                   theta=th, u=u)
    return out


def certificate(P, kappa=0.3, mu2_ratio=0.980, r=0.999989, nui=1e-8, nub=1e-5,
                nu1=0.03, p_scale=1e14):
    """The weights of Table 2 of the extended version and the constants of the
    proof of Theorem 'exponential stability', item (iii).

    mu2 = mu2_ratio * mu2bar and r = p4/p3 are the pair retained in the
    window; mu1 = (alphabar + nu1) / (2 lambda) and p2x are the conditions of
    Appendix B on Omega_1^* taken with equality; p1 e^{-2 mu1 z_s*} =
    p2 e^{-2 mu1 z_s*} = p_scale; nu2 = alphabar/10, theta = alpha_sg/4,
    delta = 1/2. Returns the weights, alpha_w, b_w, the bound on rho and the
    constant C of the exponential estimate.
    """
    zs, lam, gam, eps_in = P.zs, P.lam, P.gam, P.eps_in
    mu2 = mu2_ratio * P.mu2bar
    # alphabar does not depend on mu1 and p1, p2: first pass to get it
    c0 = constants(P, 1e-6, mu2, r, nui, p_scale, p_scale, 1e12)
    ab = c0['alphabar']
    mu1 = (ab + nu1) / (2 * lam)
    p1 = p2 = p_scale * np.exp(2 * mu1 * zs)
    p2x = p1 / (nu1 * (2 * (mu1 * lam + gam) - ab))
    c = constants(P, mu1, mu2, r, nui, p1, p2, p2x)
    ab, ae, ao = c['alphabar'], c['aeta'], c['ao']
    asg = ab - ae - ao
    nu2, th, delta = ab / 10, asg / 4, 0.5

    xg = P.xg
    Dbot = P.e[-1] / (P.l4[-1] - P.vm)
    P11 = np.exp(-2 * mu2 * (xg - zs))
    P22 = r * np.exp(2 * mu2 * (xg - P.zb))
    DPD = _trap(P11 * P.D2[:, 0] ** 2 + P22 * P.D2[:, 1] ** 2, xg)
    sig_u = ((1 - np.exp(-2 * mu1 * zs)) / (2 * mu1) * p2 * gam ** 2 * eps_in ** 2 / nu2
             + lam * p2 * eps_in ** 2 + (1 + 1 / nub) * r * abs(P.l4[-1]) * Dbot ** 2)
    sig_udot = p2x * eps_in ** 2 / lam
    sig_vol = DPD / nu2
    sig_w = c['Ieta'] / th
    Kfb = -(P.cstar + kappa) / P.G
    b_w = sig_w + (sig_u + sig_vol + sig_udot * kappa ** 2) * Kfb ** 2
    a_w = ab - nu2 - ae - ao / (1 - delta) - th
    rho = b_w / (2 * kappa - a_w)
    L = P.zb - zs
    c1 = min(np.exp(-2 * mu2 * L), r * np.exp(-2 * mu2 * L),
             p1 * np.exp(-2 * mu1 * zs), p2x * np.exp(-2 * mu1 * zs))
    c2 = max(p1, p2, p2x, 1.0, r)
    C = 2 * rho * c['JL'] * c2 / c1
    return dict(c, mu1=mu1, mu2=mu2, r=r, nui=nui, nub=nub, nu1=nu1, p1=p1, p2=p2,
                p2x=p2x, nu2=nu2, theta=th, delta=delta, alpha_sg=asg,
                sigma_u=sig_u, sigma_udot=sig_udot, sigma_vol=sig_vol, sigma_w=sig_w,
                b_w=b_w, alpha_w=a_w, rho_min=rho, c2_over_c1=c2 / c1, C=C,
                t_info=np.log(C) / a_w)
