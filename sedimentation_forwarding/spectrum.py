"""Exact characteristic determinant of the linearized closed loop.

The generator is nowhere discretized:
  * Omega_1 (the Jordan block) is solved in closed form;
  * Omega_2 is propagated by RK4 on the augmented 6x6 flow;
  * the eigenvalues are the zeros of a 5x5 determinant D(s).

Unknowns (a, b, eta, nu, u) with a = Y3(zs*), b = Y4(zs*), nu = G_o . Y_o.

Reference
---------
M. O. Amirat, V. Andrieu, M. Bajodek, C. Valentin, "Boundary stabilization of the
sludge blanket of a continuous clarifier-thickener by forwarding".
"""
import numpy as np


class Spec:
    def __init__(self, P, nst=600):
        self.P = P
        self.zs, self.zb = P.zs, P.zb
        self.lam, self.gam, self.eps_in = P.lam, P.gam, P.eps_in
        self.Go, self.Kint = P.Go, P.Kint
        self.Dbot = P.e[-1] / (P.l4[-1] - P.vm)
        self.T1 = P.zs / P.lam
        self.cstar, self.gv = P.cstar, P.gv
        self.G = P.G
        self.L2 = P.L2
        # --- J_1 en forme fermee
        n1, n2 = (P.d['Nm1'] * P.gv)
        self.n1, self.n2 = n1, n2
        self.kc = (P.cstar + P.gam) / (P.gam * P.lam)
        self.p = P.cstar / P.lam
        self.q = (P.cstar + P.gam) / P.lam
        self.g_a = n1 * (self.kc / P.lam - 1.0 / P.lam ** 2)
        self.g_b = (n2 - n1 * self.kc) / P.lam
        self.set_steps(nst)

    # ---------------- propagateur sur Omega_2 --------------------------
    def _coeff(self, xs):
        P = self.P
        ip = lambda y: np.interp(xs, P.xg, y)
        l3, l4 = ip(P.l3), ip(P.l4)
        n = len(xs)
        M0 = np.zeros((n, 6, 6)); N = np.zeros((n, 6, 6))
        M0[:, 0, 0] = ip(P.B2[:, 0, 0]) / l3; M0[:, 0, 1] = ip(P.B2[:, 0, 1]) / l3
        M0[:, 0, 3] = ip(P.C1[:, 0]) / l3;    M0[:, 0, 4] = ip(P.C2[:, 0]) / l3
        M0[:, 0, 5] = ip(P.D2[:, 0]) / l3
        M0[:, 1, 0] = ip(P.B2[:, 1, 0]) / l4; M0[:, 1, 1] = ip(P.B2[:, 1, 1]) / l4
        M0[:, 1, 3] = ip(P.C1[:, 1]) / l4;    M0[:, 1, 4] = ip(P.C2[:, 1]) / l4
        M0[:, 1, 5] = ip(P.D2[:, 1]) / l4
        M0[:, 2, 0] = ip(self.L2[:, 0]);      M0[:, 2, 1] = ip(self.L2[:, 1])
        N[:, 0, 0] = -1.0 / l3;               N[:, 1, 1] = -1.0 / l4
        return M0, N

    def set_steps(self, n):
        self.NST = n
        XE = np.linspace(self.zs, self.zb, n + 1)
        self.H = XE[1] - XE[0]
        XN = np.concatenate([XE[:-1], XE[:-1] + self.H / 2, XE[1:]])
        M0a, Na = self._coeff(XN)
        self._M0 = M0a.reshape(3, n, 6, 6); self._N = Na.reshape(3, n, 6, 6)

    def omega2(self, s):
        Ma = self._M0[0] + s * self._N[0]
        Mb = self._M0[1] + s * self._N[1]
        Mc = self._M0[2] + s * self._N[2]
        Phi = np.eye(6, dtype=complex); H = self.H
        for i in range(self.NST):
            A1, A2, A3 = Ma[i], Mb[i], Mc[i]
            k1 = A1 @ Phi
            k2 = A2 @ (Phi + 0.5 * H * k1)
            k3 = A2 @ (Phi + 0.5 * H * k2)
            k4 = A3 @ (Phi + H * k3)
            Phi = Phi + (H / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Psel = np.zeros((6, 5)); Psel[0, 0] = Psel[1, 1] = Psel[3, 2] = Psel[4, 3] = Psel[5, 4] = 1.0
        return (Phi @ Psel)[[0, 1, 2], :]

    # ---------------- Omega_1 en forme fermee --------------------------
    def _Im(self, m, nu):
        X = self.zs
        m = complex(m); nu = complex(nu); d = m + nu
        if abs(d * X) < 1e-8:
            return X * np.exp(nu * X) * (1.0 - d * X / 2.0)
        return (np.exp(nu * X) - np.exp(-m * X)) / d

    def J1(self, s):
        s = complex(s); lam, gam, ein = self.lam, self.gam, self.eps_in
        r1, r2 = -s / lam, -(s + gam) / lam
        A = ein * s / (lam * gam)
        B1, B0 = s / (s + gam), gam / (s + gam)
        Ia = lambda nu: self._Im(self.p, nu)
        Ib = lambda nu: self._Im(self.q, nu)
        t = (self.n1 / lam) * A * (Ia(r1) - Ia(r2))
        t += ein * self.g_a * (B1 * Ia(r2) + B0 * Ia(0.0))
        t += ein * self.g_b * (B1 * Ib(r2) + B0 * Ib(0.0))
        return t

    def omega1(self, s):
        lam, gam, ein, zs = self.lam, self.gam, self.eps_in, self.zs
        E1 = np.exp(-(s + gam) * zs / lam)
        Y2s = ein * ((s / (s + gam)) * E1 + gam / (s + gam))
        Y1s = ein * s / (lam * gam) * np.exp(-s * zs / lam) * (1.0 - np.exp(-gam * zs / lam))
        return Y1s, Y2s, self.J1(s)

    # ---------------- determinant --------------------------------------
    def gain(self, kappa):
        return -(self.cstar + kappa) / self.G

    def det_with(self, s, y1s, j1, mode, Kg):
        _, Y2s, _ = self.omega1(s)
        W = self.omega2(s)
        M = np.zeros((5, 5), dtype=complex)
        M[0, 0] = 1.0; M[0, 1] = -self.Kint[2]
        M[0, 4] = -(self.Kint[0] * y1s + self.Kint[1] * Y2s)
        M[1] = W[1] - W[0]; M[1, 4] -= self.Dbot
        M[2, 3] = 1.0; M[2, 1] = -self.Go[2]
        M[2, 4] = -(self.Go[0] * y1s + self.Go[1] * Y2s)
        M[3, 2] = s; M[3, 3] = -1.0
        if mode == 'open':
            M[4, 4] = 1.0
        elif mode == 'eta':
            M[4, 4] = 1.0; M[4, 2] = -Kg
        else:
            M[4] = Kg * W[2]; M[4, 4] += 1.0 + Kg * j1; M[4, 2] += -Kg
        return M

    def dsc(self, s, mode='open', Kg=None):
        y1s, _, j1 = self.omega1(s)
        M = self.det_with(s, y1s, j1, mode, Kg)
        nr = np.linalg.norm(M, axis=1); nr[nr == 0] = 1.0
        return np.linalg.det(M / nr[:, None])

    def newton(self, s0, mode='open', Kg=None, it=80, tol=1e-15):
        s = complex(s0); h = 1e-7 * max(abs(s), 1e-4)
        for _ in range(it):
            f = self.dsc(s, mode, Kg)
            fp = (self.dsc(s + h, mode, Kg) - self.dsc(s - h, mode, Kg)) / (2 * h)
            if fp == 0 or not np.isfinite(fp):
                break
            ds = -f / fp
            if abs(ds) > 0.3 * max(abs(s), 1e-3):
                ds *= 0.3 * max(abs(s), 1e-3) / abs(ds)
            s += ds
            if abs(ds) < tol * max(abs(s), 1e-9):
                break
        return s

    # ------------------- delay family: plateau ---------------------------
    def _pref(self, s):
        return self.eps_in * s / (self.lam * self.gam) * (1.0 - np.exp(-self.gam * self.T1))

    def sigma(self, om, Kg, mode='forwarding', sig=0.0):
        s = sig + 1j * om
        j1 = self.J1(s) if mode == 'forwarding' else 0.0
        A = np.linalg.det(self.det_with(s, 0.0, j1, mode, Kg))
        B = np.linalg.det(self.det_with(s, 1.0, j1, mode, Kg)) - A
        q = abs(A / (B * self._pref(s)))
        return -np.log(q) / self.T1

    def plateau(self, Kg, oms=None, mode='forwarding'):
        if oms is None:
            oms = np.geomspace(0.5 * 2 * np.pi / self.T1, 400.0, 40)
        v = []
        for om in oms:
            try:
                x = self.sigma(om, Kg, mode)
            except Exception:
                continue
            if np.isfinite(x):
                v.append(x)
        return max(v) if v else np.nan

    def kappa_star(self, lo=1e-4, hi=1e3, n=28):
        rows = []
        for kap in np.geomspace(lo, hi, n):
            Kg = self.gain(kap)
            pl = self.plateau(Kg)
            rows.append((kap, Kg, pl, max(-kap, pl)))
        R = np.array(rows)
        i = int(np.argmin(R[:, 3]))
        return R, R[i]
