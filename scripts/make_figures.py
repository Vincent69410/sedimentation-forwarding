"""Draw the data figures of the paper.

    python scripts/make_figures.py            # closed-loop figure from the stored runs
    python scripts/make_figures.py --rerun    # recompute the two runs first
                                              # (about ten minutes each)

Writes figs/fig_min.pdf (the stationary profile, Figure 3 of the extended
version) and figs/fig_nl.pdf (the closed-loop scenarios, Figure 3 of the
paper, Figure 4 of the extended version). The other figures are diagrams and
carry no data.
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
from sedimentation_forwarding import scenarios, figures

out = os.path.join(ROOT, "figs")
os.makedirs(out, exist_ok=True)

P = sf.illustrative_example(Nd=2000)
figures.figure_profile(P, fname=os.path.join(out, "fig_min.pdf"))
print("figs/fig_min.pdf written (stationary profile)")

runs = scenarios.recompute_runs(verbose=True) if "--rerun" in sys.argv else scenarios.load_runs()
figures.figure_scenarios(runs, fname=os.path.join(out, "fig_nl.pdf"))
print("figs/fig_nl.pdf written (closed-loop scenarios)")
