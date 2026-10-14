"""The two data figures of the paper, drawn exactly as printed.

    figure3(P)      stationary profile and pole equation   (Figure 3)
    figure4(runs)   nonlinear closed loop                  (Figure 4)

Figures 1 and 2 of the paper are diagrams (the apparatus and the block
diagram of the interconnection) and carry no data.
"""
import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, MUTED, GRID, INK = "#2F6DB5", "#C4571F", "#5b5b5b", "#dddddd", "#1a1a1a"

STYLE = {
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": "#b0b0b0", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.titlesize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.fontsize": 8, "legend.frameon": False, "pdf.fonttype": 42,
}


def figure3(P, fname=None):
    """(a) the stationary admissible profile, with the compression zone
    magnified; (b) the pole equation c = F(c)."""
    with plt.rc_context(STYLE):
        cs = P.cstar
        fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.1))

        # (a) profile
        z1 = np.linspace(0, P.zs, 200)
        ax[0].plot(z1, np.full_like(z1, P.eps_in), color=BLUE, lw=1.6)
        ax[0].plot(P.xg, P.e, color=BLUE, lw=1.6)
        ax[0].plot([P.zs, P.zs], [P.eps_in, P.ep], color=BLUE, lw=1.0, ls=":")
        ax[0].axhline(P.eps_c, color=MUTED, lw=0.8, ls="--")
        ax[0].text(0.93 * P.zb, P.eps_c + 0.016, r"$\varepsilon_c$",
                   color=MUTED, fontsize=8)
        ax[0].text(0.04 * P.zb, 0.075, r"free settling $\Omega_1^*$",
                   fontsize=8, color=MUTED)
        ax[0].set_xlabel(r"depth $z$ [m]")
        ax[0].set_ylabel(r"$\varepsilon_s^*$")
        ax[0].set_title(r"(a) stationary admissible profile")
        ax[0].set_ylim(0.0, 0.78)

        dev = P.e - P.ep
        p = int(np.floor(np.log10(max(abs(dev).max(), 1e-300))))
        ins = ax[0].inset_axes([0.09, 0.46, 0.55, 0.36])
        ins.plot(P.xg, dev * 10.0 ** (-p), color=BLUE, lw=1.4)
        ins.set_xlim(P.zs, P.zb)
        ins.tick_params(labelsize=6.5)
        ins.set_title(r"compression $\Omega_2^*$:  $(\varepsilon_s^*-\varepsilon_s^{*+})"
                      r"\times10^{%d}$" % (-p), fontsize=6.8, pad=2.5)
        ins.grid(True, color=GRID, lw=0.4)

        # (b) pole equation
        q = int(np.floor(np.log10(abs(cs))))
        u = 10.0 ** (-q)
        cg = np.linspace(2.2 * cs, -0.07 * cs, 70)
        Fv = []
        for c in cg:
            try:
                Fv.append(P.design(c)["F"])
            except Exception:
                Fv.append(np.nan)
        ax[1].plot(cg * u, np.array(Fv) * u, color=BLUE, lw=1.5,
                   label=r"$\mathcal{F}(c)$")
        ax[1].plot(cg * u, cg * u, color=MUTED, lw=0.9, ls="--", label="identity")
        ax[1].plot([cs * u], [cs * u], "o", color=ORANGE, ms=5)
        ax[1].annotate(r"$c^*=-%.4f\times10^{%d}$" % (abs(cs) * u, q),
                       (cs * u, cs * u), textcoords="offset points",
                       xytext=(-4, -18), color=ORANGE, fontsize=8)
        ax[1].set_xlabel(r"$c\ [\times10^{%d}]$" % q)
        ax[1].set_title(r"(b) pole equation $c=\mathcal{F}(c)$")
        ax[1].legend(loc="lower right")

        if fname:
            fig.savefig(fname, bbox_inches="tight")
    return fig


def figure4(runs, zb=1.0, fname=None):
    """(a) bed thickness in the three runs, with the deviation from the set
    point magnified; (b) the underflow velocity in closed loop.

    ``runs`` is the dict returned by ``closed_loop.load_runs()`` or
    ``closed_loop.figure4_runs()``.
    """
    def bed(r):                       # bed thickness, mm
        return (zb - np.array(r["zs"])) * 1e3

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
        set_point = bed(runs["open_eq"])[0]

        ax[0].plot(runs["open_offset"]["t"], bed(runs["open_offset"]), color=MUTED,
                   lw=1.3, ls="--", label="open loop, $10$ mm offset")
        ax[0].plot(runs["open_eq"]["t"], bed(runs["open_eq"]), color=MUTED,
                   lw=1.0, ls=":", label="open loop, at equilibrium")
        ax[0].plot(runs["closed"]["t"], bed(runs["closed"]), color=BLUE,
                   lw=1.7, label="closed loop, $10$ mm offset")
        ax[0].axhline(set_point, color=ORANGE, lw=0.8, ls="-.", label="set point")
        ax[0].set_xlabel(r"$t$ [s]")
        ax[0].set_ylabel("bed thickness [mm]")
        ax[0].set_title("(a) bed thickness")
        lo = min(bed(r).min() for r in runs.values())
        hi = max(bed(r).max() for r in runs.values())
        if 188.6 <= lo and hi <= 201.4:          # the limits of the paper
            ax[0].set_ylim(188.6, 201.4)
        ax[0].legend(loc="center right", bbox_to_anchor=(1.0, 0.62))

        ins = ax[0].inset_axes([0.30, 0.12, 0.44, 0.28])
        for k, c, lw, ls in (("open_eq", MUTED, 1.0, ":"), ("closed", BLUE, 1.5, "-")):
            ins.plot(runs[k]["t"], bed(runs[k]) - set_point, color=c, lw=lw, ls=ls)
        ins.axhline(0, color=ORANGE, lw=0.7, ls="-.")
        ins.set_ylim(-0.9, 0.3)
        ins.tick_params(labelsize=6.5)
        ins.set_title("deviation from set point [mm]", fontsize=6.5, pad=2.0)
        ins.grid(True, color=GRID, lw=0.4)

        ax[1].plot(runs["closed"]["t"], 1e2 * np.array(runs["closed"]["vm"]),
                   color=BLUE, lw=1.7)
        ax[1].axhline(1e2 * runs["open_eq"]["vm"][0], color=ORANGE, lw=0.8,
                      ls="-.", label=r"$v_m^*$")
        ax[1].set_xlabel(r"$t$ [s]")
        ax[1].set_ylabel(r"$v_m$ [cm/s]")
        ax[1].set_title("(b) underflow velocity")
        ax[1].legend(loc="lower right")

        if fname:
            fig.savefig(fname, bbox_inches="tight")
    return fig
