"""Operating points.

``worked_example`` returns the plant of Section 6: a 1 m laboratory settling
column fed with 2 mm polystyrene beads, with a 20 cm bed.
"""
import numpy as np

from .model import Plant

G = 9.81          # m/s^2
RHO_F = 1000.0    # kg/m^3, carrier fluid
MU_F = 1e-3       # Pa.s, carrier fluid


def terminal_velocity(rho_s, d, rho_f=RHO_F, mu=MU_F, it=80):
    """Terminal velocity of an isolated sphere, Schiller-Naumann drag.

    Solves v = v_Stokes / (1 + 0.15 Re^0.687) by fixed point, Re = rho_f v d/mu.
    """
    v_stokes = (rho_s - rho_f) * G * d ** 2 / (18 * mu)
    v = v_stokes
    for _ in range(it):
        re = rho_f * v * d / mu
        v = v_stokes / (1 + 0.15 * max(re, 1e-12) ** 0.687)
    return v


def settling_column(rho_s=1050.0, d=2e-3, eps_c=0.55, n_s=5, sigma_0=1e5,
                    eps_in_over_eps_c=0.2, z_b=1.0, L=0.20, Nd=1200):
    """A continuous settling column.

    The default is the operating point of Section 6: 2 mm polystyrene beads
    in water, a 20 cm bed at the bottom of a 1 m column. Nearly neutrally
    buoyant beads keep K = g(rho_s - rho_f)/rho_s small, which is what makes
    the small-gain condition hold on a bed one can actually see.

    Parameters
    ----------
    rho_s : particle density, kg/m^3.
    d : particle diameter, m. Sets the drag through the terminal velocity.
    eps_c : solid fraction above which the particles are in contact.
    n_s, sigma_0 : exponent and stress scale of the power law for sigma_e.
    eps_in_over_eps_c : feed fraction, as a multiple of eps_c.
    z_b : column depth, m.
    L : target thickness of the sediment bed, m.
    """
    K = G * (rho_s - RHO_F) / rho_s
    v0 = terminal_velocity(rho_s, d)
    gamma = K / v0
    return Plant(zb=z_b, zs=z_b - L, eps_c=eps_c, n=n_s,
                 a=sigma_0 / (rho_s * eps_c ** n_s), rho_s=rho_s,
                 gam=gamma, K=K, theta=eps_in_over_eps_c, Nd=Nd)


def worked_example(Nd=1200):
    """The operating point of Section 6, with the pole c* already solved.

    2 mm polystyrene beads, 20 cm bed, 1 m column.
    """
    P = settling_column(Nd=Nd)
    P.solve_cstar()
    return P


def reynolds(rho_s=1050.0, d=2e-3, rho_f=RHO_F, mu=MU_F):
    """Particle Reynolds number at the terminal velocity."""
    return rho_f * terminal_velocity(rho_s, d, rho_f, mu) * d / mu
