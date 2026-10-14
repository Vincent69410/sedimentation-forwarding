"""Nonlinear closed-loop simulation of the worked example, one run.

    python scripts/nonlinear_closed_loop.py [NC] [TEND] [KAPPA] [ETA0] [MODE]

Defaults: 800 cells, 40 s, kappa = 0.3, initial offset 10 mm, closed loop,
which is the closed-loop run of Figure 4 (the 20 cm bed carries 160 cells).
MODE is "closed" or "open".  The result is written to nl_<tag>.json.

The scheme and the control law are in sedimentation_forwarding/closed_loop.py.
One run takes about a quarter of an hour; to redraw Figure 4 from the stored runs, use
scripts/make_figures.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from sedimentation_forwarding.closed_loop import simulate

NC = int(sys.argv[1]) if len(sys.argv) > 1 else 800
TEND = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
ETA0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2
MODE = sys.argv[5] if len(sys.argv) > 5 else "closed"

hist = simulate(nc=NC, tend=TEND, kappa=KAPPA, eta0=ETA0, mode=MODE, verbose=True)
tag = f"{MODE}_{NC}_{KAPPA:g}_{ETA0:g}"
with open(f"nl_{tag}.json", "w") as fh:
    json.dump(hist, fh)
print(f"written nl_{tag}.json")
