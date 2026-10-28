"""One closed-loop scenario of Section 6, with its parameters on the command line.

    python scripts/closed_loop_scenario.py setpoint
    python scripts/closed_loop_scenario.py feed
    python scripts/closed_loop_scenario.py feed --nominal   # reference not updated

The defaults (800 cells, kappa = 0.3, 75 s, change at t = 5 s) are those of
the paper. The result is written to scenario_<name>.json. One run takes about
ten minutes; the scheme and the control law are in
sedimentation_forwarding/scenarios.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from sedimentation_forwarding.scenarios import run

name = sys.argv[1] if len(sys.argv) > 1 else "setpoint"
nominal = "--nominal" in sys.argv
hist = run(name, update_reference=not nominal, verbose=True)
tag = name + ("_nominal" if nominal else "")
with open(f"scenario_{tag}.json", "w") as fh:
    json.dump(hist, fh)
bed = (hist["zb"] - np.array(hist["zs"])) * 1e3
ref = (hist["zb"] - np.array(hist["zref"])) * 1e3
print(f"written scenario_{tag}.json; final bed minus set point: {bed[-1] - ref[-1]:+.3f} mm")
