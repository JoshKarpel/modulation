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

    y_lower_lim = 1e-3 * u.pJ
    y_upper_lim = 1e1 * u.pJ

    pump_powers = sorted(ps.parameter_set("pump_power"))
    pump_wavelengths = np.array(sorted(ps.parameter_set("pump_wavelength")))

    for pump_power in pump_powers:
        print(f"pump power is {pump_power / u.mW:.6f} mW")
        sims = sorted(
            ps.select(pump_power=pump_power), key=lambda s: s.spec.pump_wavelength
        )
        pump_mode = sims[0].spec.pump_mode

        modes = set()
        for sim in sims:
            modes.update(sim.spec.modes)
        modes = sorted(modes, key=lambda m: m.wavelength)

        print("total unique modes", len(modes))

        x_wavelength = np.empty(len(pump_wavelengths), dtype=np.float64)
        x_frequency = np.empty(len(pump_wavelengths), dtype=np.float64)
        mode_to_y = {m: np.zeros_like(x_wavelength) * np.NaN for m in modes}
        for s, sim in enumerate(sorted(sims, key=lambda s: s.spec.pump_wavelength)):
            x_wavelength[s] = sim.spec.pump_wavelength
            x_frequency[s] = sim.spec.pump_frequency
            for mode, energy in zip(
                sim.spec.modes, sim.mode_energies(sim.lookback.mean)
            ):
                mode_to_y[mode][s] = energy

        mode_to_y = {
            mode: y for mode, y in mode_to_y.items() if np.any(y >= y_lower_lim)
        }

        modes = list(mode_to_y.keys())
        y = list(mode_to_y.values())

        colors = itertools.cycle(COLORS)
        # styles = itertools.cycle(LINESTYLES)
        mode_kwargs = [
            dict(linestyle="-", color=next(colors), linewidth=1, alpha=0.8)
            for mode in modes
        ]

        mode_vlines_kwargs = dict(alpha=0.8, linestyle=":")

        postfix = f"{pump_power / u.mW:.6f}mW"
        si.vis.xy_plot(
            f"{path.stem}__wavelength_scan__{postfix}",
            x_wavelength - pump_mode.wavelength,
            *y,
            line_labels=[rf"${mode.tex}$" for mode in modes],
            line_kwargs=mode_kwargs,
            x_unit="pm",
            y_unit="pJ",
            x_label=r"Pump Detuning from Pump Mode",
            y_label="Mode Energies",
            y_log_axis=True,
            legend_on_right=True,
            font_size_legend=8,
            vlines=[mode.wavelength - pump_mode.wavelength for mode in modes],
            vline_kwargs=[
                collections.ChainMap(mode_vlines_kwargs, kw)
                for mode, kw in zip(modes, mode_kwargs)
            ],
            y_lower_limit=y_lower_lim,
            y_upper_limit=y_upper_lim,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"{path.stem}__frequency_scan__{postfix}",
            x_frequency - pump_mode.frequency,
            *y,
            line_labels=[rf"${mode.tex}$" for mode in modes],
            line_kwargs=mode_kwargs,
            x_unit="MHz",
            y_unit="pJ",
            x_label=r"Pump Detuning from Pump Mode",
            y_label="Mode Energies",
            y_log_axis=True,
            legend_on_right=True,
            font_size_legend=8,
            vlines=[mode.frequency - pump_mode.frequency for mode in modes],
            vline_kwargs=[
                collections.ChainMap(mode_vlines_kwargs, kw)
                for mode, kw in zip(modes, mode_kwargs)
            ],
            y_lower_limit=y_lower_lim,
            y_upper_limit=y_upper_lim,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with LOGMAN as logger:
        paths = [Path(__file__).parent / "wavelength_scan.sims"]

        for path in paths:
            wavelength_scan(path)
