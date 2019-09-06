#!/usr/bin/env python
import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import mock

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIM_KWARGS = dict(target_dir=OUT_DIR)

if __name__ == "__main__":
    with LOGMAN as logger:
        q_c = np.geomspace(1e2, 1e10, 10000)
        q_i = 1e6

        r = q_c / q_i
        t = ((1 - r) / (1 + r)) ** 2

        si.vis.xy_plot(
            "transmission",
            r,
            t,
            x_label=r"$r = Q^{\mathrm{Coupling}} / Q^{\mathrm{Intrinsic}}$",
            y_label=r"Transmission $= \left(\frac{1-r}{1+r}\right)^2$",
            x_log_axis=True,
            **PLOT_KWARGS
        )
