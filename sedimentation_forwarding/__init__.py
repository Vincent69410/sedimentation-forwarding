"""Boundary stabilization of a sedimentation front by forwarding.

Companion code for the paper. Two modules:

    model     the design chain, from the stationary profile to the
              small-gain constants;
    spectrum  the exact characteristic determinant of the closed loop.

A ready-made operating point, the worked example of Section 6, is returned by
:func:`worked_example`.
"""
from .model import Plant, constants, best
from .spectrum import Spec
from .examples import worked_example, settling_column

__all__ = ["Plant", "constants", "best", "Spec",
           "worked_example", "settling_column"]
__version__ = "1.0.0"
