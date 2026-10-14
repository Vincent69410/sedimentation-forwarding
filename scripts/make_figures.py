"""Draw Figures 3 and 4 of the paper.

    python scripts/make_figures.py            # Figure 4 from the stored runs
    python scripts/make_figures.py --rerun    # recompute the three runs first
                                              # (about a quarter of an hour each)

Writes figs/fig_min.pdf (Figure 3) and figs/fig_nl.pdf (Figure 4).
Figures 1 and 2 are diagrams and carry no data.
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
np.seterr(all="ignore")

import sedimentation_forwarding as sf
from sedimentation_forwarding import closed_loop, figures

out = os.path.join(ROOT, "figs")
os.makedirs(out, exist_ok=True)

P = sf.worked_example(Nd=2000)
figures.figure3(P, fname=os.path.join(out, "fig_min.pdf"))
print("figs/fig_min.pdf written (Figure 3)")

if "--rerun" in sys.argv:
    runs = closed_loop.figure4_runs(P, verbose=True)
else:
    runs = closed_loop.load_runs()
figures.figure4(runs, zb=P.zb, fname=os.path.join(out, "fig_nl.pdf"))
print("figs/fig_nl.pdf written (Figure 4)")
