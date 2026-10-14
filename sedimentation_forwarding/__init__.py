"""Boundary stabilization of a sedimentation front by forwarding.

Companion code for the paper. Four modules:

    model        the design chain, from the stationary profile to the
                 small-gain constants;
    spectrum     the exact characteristic determinant of the closed loop;
    closed_loop  the nonlinear plant in open or closed loop (Figure 4);
    figures      Figures 3 and 4, drawn as printed.

A ready-made operating point, the worked example of Section 6, is returned by
:func:`worked_example`.
"""
from .model import Plant, constants, best
from .spectrum import Spec
from .examples import worked_example, settling_column
from . import closed_loop, figures

__all__ = ["Plant", "constants", "best", "Spec",
           "worked_example", "settling_column",
           "closed_loop", "figures"]
__version__ = "1.1.0"
