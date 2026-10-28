"""The data figures of the paper, drawn exactly as printed.

    figure_profile(P)          the stationary admissible profile
                               (Figure 3 of the extended version)
    figure_scenarios(runs)     the closed-loop scenarios (Figure 3 of the
                               paper, Figure 4 of the extended version)

The other figures are diagrams (the unit, the structure of the linearized
plant) and carry no data.
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


def figure_profile(P, fname=None):
    """The stationary admissible profile; the inset shows the deviation
    eps_s^* - eps_s^*(z_s^{*+}) on the compression zone, magnified."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.5))
        z1 = np.linspace(0, P.zs, 200)
        ax.plot(z1, np.full_like(z1, P.eps_in), color=BLUE, lw=1.6)
        ax.plot(P.xg, P.e, color=BLUE, lw=1.6)
        ax.plot([P.zs, P.zs], [P.eps_in, P.ep], color=BLUE, lw=1.0, ls=":")
        ax.axhline(P.eps_c, color=MUTED, lw=0.8, ls="--")
        ax.text(0.93 * P.zb, P.eps_c + 0.016, r"$\varepsilon_c$", color=MUTED, fontsize=8)
        ax.text(0.04 * P.zb, 0.035, r"free settling $\Omega_1^*$", fontsize=8, color=MUTED)
        ax.set_xlabel(r"depth $z$ [m]")
        ax.set_ylabel(r"$\varepsilon_s^*$")
        ax.set_ylim(0.0, 0.78)

        dev = P.e - P.ep
        p = int(np.floor(np.log10(max(abs(dev).max(), 1e-300))))
        ins = ax.inset_axes([0.16, 0.30, 0.46, 0.30])
        ins.plot(P.xg, dev * 10.0 ** (-p), color=BLUE, lw=1.4)
        ins.set_xlim(P.zs, P.zb)
        ins.tick_params(labelsize=6.5)
        ins.set_title(r"compression $\Omega_2^*$:  $(\varepsilon_s^*-\varepsilon_s^{*+})"
                      r"\times10^{%d}$" % (-p), fontsize=6.8, pad=2.5)
        ins.grid(True, color=GRID, lw=0.4)
        if fname:
            fig.savefig(fname, bbox_inches="tight")
    return fig


def figure_scenarios(runs, fname=None):
    """(a)-(b) change of set point; (c)-(d) step of the feed fraction with the
    reference updated. ``runs`` is the dict returned by
    ``scenarios.load_runs()`` or ``scenarios.recompute_runs()``."""
    A, B = runs["setpoint"], runs["feed"]
    zb = A["zb"]

    def bed(R):                       # bed thickness, mm
        return (zb - np.array(R["zs"])) * 1e3

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.6), sharex=True)
        t = np.array(A["t"])
        ax[0, 0].plot(t, (zb - np.array(A["zref"])) * 1e3, color=ORANGE, lw=0.9,
                      ls="-.", label="set point")
        ax[0, 0].plot(t, bed(A), color=BLUE, lw=1.5, label="closed loop")
        ax[0, 0].set_title("(a) set-point change: bed thickness")
        ax[0, 0].legend(loc="center right")
        ax[0, 1].plot(t, 1e2 * np.array(A["vm"]), color=BLUE, lw=1.5)
        ax[0, 1].axhline(1e2 * A["vms"], color=ORANGE, lw=0.8, ls="-.", label=r"$v_m^*$")
        ax[0, 1].legend(loc="lower right")
        ax[0, 1].set_title("(b) set-point change: underflow velocity")

        t = np.array(B["t"])
        ax[1, 0].plot(t, bed(B), color=BLUE, lw=1.5, label="closed loop")
        ax[1, 0].axhline(bed(A)[0], color=ORANGE, lw=0.9, ls="-.", label="set point")
        ax[1, 0].set_title(r"(c) feed step $\varepsilon_{in}\to1.1\,\varepsilon_{in}$"
                           r" at $t=5$ s: bed thickness")
        ax[1, 0].legend(loc="upper right")
        ax[1, 1].plot(t, 1e2 * np.array(B["vm"]), color=BLUE, lw=1.5)
        ax[1, 1].axhline(1e2 * B["vms"], color=ORANGE, lw=0.8, ls="-.",
                         label=r"$v_m^*(\varepsilon_{in})$")
        if B["meta"].get("vm1"):
            ax[1, 1].axhline(1e2 * B["meta"]["vm1"], color=ORANGE, lw=0.8, ls=":",
                             label=r"$v_m^*(1.1\,\varepsilon_{in})$")
        ax[1, 1].set_title("(d) feed step: underflow velocity")
        ax[1, 1].legend(loc="lower right", fontsize=7)

        for a in ax[:, 0]:
            a.set_ylabel("bed thickness [mm]")
        for a in ax[:, 1]:
            a.set_ylabel(r"$v_m$ [cm/s]")
        for a in ax[1]:
            a.set_xlabel(r"$t$ [s]")
        fig.tight_layout()
        if fname:
            fig.savefig(fname, bbox_inches="tight")
    return fig
