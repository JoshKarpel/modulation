import itertools
import logging
from pathlib import Path

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman, analysis
from modulation.resonators import mock

import matplotlib.pyplot as plt

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]
LINESTYLES = ["-", "-.", "--", ":"]


def wavelength_scan():
    ps = analysis.ParameterScan.from_file(
        Path(__file__).parent / "test_wavelength_scan.sims"
    )
    print(ps, len(ps))

    pump_wavelengths = np.array(sorted(ps.parameter_set("_pump_wavelength")))

    modes = set()
    for sim in ps.sims:
        modes.update(sim.spec.modes)

    print("total unique modes", len(modes))

    modes = sorted(modes, key=lambda m: m.wavelength)
    for mode in modes:
        print(mode)

    print(len(pump_wavelengths), pump_wavelengths / u.nm)

    x = np.empty(len(pump_wavelengths), dtype=np.float64)
    mode_to_y = {m: np.zeros_like(x) * np.NaN for m in modes}
    print(len(mode_to_y))
    for s, sim in enumerate(sorted(ps.sims, key=lambda s: s.spec._pump_wavelength)):
        x[s] = sim.spec._pump_wavelength
        for mode, energy in zip(sim.spec.modes, sim.mode_energies(sim.lookback.mean)):
            print(mode)
            mode_to_y[mode][s] = energy

    print(x / u.nm)
    for k, v in mode_to_y.items():
        print(k, v / u.pJ)

    modes = list(mode_to_y.keys())
    y = list(mode_to_y.values())

    styles = itertools.cycle(LINESTYLES)
    colors = itertools.cycle(COLORS)
    mode_kwargs = [dict(linestyle=next(styles), color=next(colors)) for mode in modes]

    si.vis.xy_plot(
        "wavelength_scan",
        x,
        *y,
        line_labels=[rf"${mode.tex}$" for mode in modes],
        line_kwargs=mode_kwargs,
        x_unit="nm",
        y_unit="pJ",
        x_label="Pump Wavelength",
        y_label="Mode Energies",
        y_log_axis=True,
        legend_on_right=True,
        legend_kwargs=dict(fontsize=8, ncol=2),
        vlines=[mode.wavelength for mode in modes],
        vline_kwargs=[dict(alpha=0.8, linestyle=":") for mode in modes],
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        wavelength_scan()
