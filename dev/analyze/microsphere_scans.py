import collections
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
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
LINESTYLES = ["-", "-.", "--"]


def wavelength_scan(path):
    ps = analysis.ParameterScan.from_file(path)
    print(ps, len(ps))

    pump_wavelengths = np.array(sorted(ps.parameter_set("pump_wavelength")))

    modes = set()
    for sim in ps.sims:
        modes.update(sim.spec.modes)
    modes = sorted(modes, key=lambda m: m.wavelength)

    print("total unique modes", len(modes))

    x_wavelength = np.empty(len(pump_wavelengths), dtype=np.float64)
    x_frequency = np.empty(len(pump_wavelengths), dtype=np.float64)
    mode_to_y = {m: np.zeros_like(x_wavelength) * np.NaN for m in modes}
    for s, sim in enumerate(sorted(ps.sims, key=lambda s: s.spec.pump_wavelength)):
        x_wavelength[s] = sim.spec.pump_wavelength
        x_frequency[s] = sim.spec.pump_frequency
        for mode, energy in zip(sim.spec.modes, sim.mode_energies(sim.lookback.mean)):
            mode_to_y[mode][s] = energy

    modes = list(mode_to_y.keys())
    y = list(mode_to_y.values())

    styles = itertools.cycle(LINESTYLES)
    colors = itertools.cycle(COLORS)
    mode_kwargs = [dict(linestyle=next(styles), color=next(colors)) for mode in modes]

    mode_vlines_kwargs = dict(alpha=0.8, linestyle=":")

    si.vis.xy_plot(
        f"{path.stem}__wavelength_scan",
        x_wavelength,
        *y,
        line_labels=[rf"${mode.tex}$" for mode in modes],
        line_kwargs=mode_kwargs,
        x_unit="nm",
        y_unit="pJ",
        x_label="Pump Wavelength",
        y_label="Mode Energies",
        y_log_axis=True,
        legend_on_right=True,
        font_size_legend=8,
        vlines=[mode.wavelength for mode in modes],
        vline_kwargs=[
            collections.ChainMap(mode_vlines_kwargs, kw)
            for mode, kw in zip(modes, mode_kwargs)
        ],
        y_lower_limit=1e-9 * u.pJ,
        **PLOT_KWARGS,
    )
    si.vis.xy_plot(
        f"{path.stem}__frequency_scan",
        x_frequency - (u.c / (1064 * u.nm)),
        *y,
        line_labels=[rf"${mode.tex}$" for mode in modes],
        line_kwargs=mode_kwargs,
        x_unit="MHz",
        y_unit="pJ",
        x_label=r"Pump Frequency Detuning from $1064 \, \mathrm{nm}}$",
        y_label="Mode Energies",
        y_log_axis=True,
        legend_on_right=True,
        font_size_legend=8,
        vlines=[mode.wavelength for mode in modes],
        vline_kwargs=[
            collections.ChainMap(mode_vlines_kwargs, kw)
            for mode, kw in zip(modes, mode_kwargs)
        ],
        y_lower_limit=1e-9 * u.pJ,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        paths = [Path(__file__).parent / "sym_wavelength_scan_v2.sims"]

        for path in paths:
            wavelength_scan(path)