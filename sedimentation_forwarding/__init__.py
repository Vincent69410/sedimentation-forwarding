"""Boundary stabilization of a sedimentation front by forwarding.

Companion code for the paper. Five modules:

    model        the design chain, from the stationary profile to the
                 small-gain constants and the stability certificate;
    spectrum     the exact characteristic determinant of the closed loop;
    scenarios    the closed-loop scenarios of the last subsection of Section 6
                 (change of set point, step of the feed fraction);
    closed_loop  the same scheme in plain numpy, used for the open-loop drift;
    figures      the data figures, drawn as printed.

The operating point of Section 6, the illustrative example, is returned by
:func:`illustrative_example` (alias :func:`worked_example`).
"""
from .model import Plant, constants, best, certificate
from .spectrum import Spec
from .examples import illustrative_example, worked_example, settling_column
from . import closed_loop, scenarios, figures

__all__ = ["Plant", "constants", "best", "certificate", "Spec",
           "illustrative_example", "worked_example", "settling_column",
           "closed_loop", "scenarios", "figures"]
__version__ = "1.2.0"
