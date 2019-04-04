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

COLORS = itertools.cycle(
    [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
)

LINESTYLES = ["-", "-.", "--", ":"]


def mode_kwargs(mode, pump_mode, wavelength_bounds):
    base = {}

    if mode == pump_mode:
        return {"color": "black", **base}

    i = 0
    for bound in wavelength_bounds:
        if mode.wavelength in bound:
            return {
                "linestyle": LINESTYLES[i % len(LINESTYLES)],
                "color": next(COLORS),
                **base,
            }
        i += 1


def mode_energy_plot_by_mixing_power(path):
    lower_lim = 1e-7 * u.pJ

    all_sims = analysis.ParameterScan.from_file(path)
    s = all_sims[0].spec
    modes = s.modes
    idxs = [all_sims[0].mode_to_index[mode] for mode in modes]
    pump_mode = s.pump_mode
    wavelength_bounds = s.wavelength_bounds

    for mixing_power in all_sims.parameter_set("mixing_power"):
        sims = all_sims.select(mixing_power=mixing_power)
        scan_powers = np.array([sim.spec.pump_power for sim in sims])
        means = [sim.mode_energies(sim.lookback.mean) for sim in sims]

        lines = []
        line_kwargs = []
        line_labels = []
        for idx, mode, color in zip(idxs, modes, COLORS):
            line = np.array([mean[idx] for mean in means])

            if not np.any(line >= lower_lim):
                continue

            lines.append(line)
            line_kwargs.append(mode_kwargs(mode, pump_mode, wavelength_bounds))
            line_labels.append(fr"${mode.tex}$")

        si.vis.xy_plot(
            f"{path.stem}__mode_energies__mixing={mixing_power / u.mW:.6f}mW",
            scan_powers,
            *lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Launched Pump Power",
            y_label="Steady-State Mode Energy",
            x_unit="mW",
            y_unit="pJ",
            x_log_axis=scan_powers[0] != 0,
            y_log_axis=True,
            y_lower_limit=lower_lim,
            # y_upper_limit=1e3 * u.pJ,
            legend_on_right=True,
            **PLOT_KWARGS,
        )


def conversion_efficiency_plot(path):
    sims = analysis.ParameterScan.from_file(path)
    mixing_powers = set()
    without_mixing = list(sims.select(mixing_power=0))
    with_mixings = [list(sims.select(mixing))]

    for with_mixing in with_mixings:
        s = with_mixing[0].spec
        modes = (s._pump_mode, s._stokes_mode, s._mixing_mode, s._modulated_mode)
        idxs = [with_mixing[0].mode_to_index[mode] for mode in modes]
        idxs_by_mode = dict(zip(modes, idxs))

        scan_powers = np.array([sim.spec._scan_power for sim in with_mixing])

        launched_mixing_power = s.mode_pumps[idxs_by_mode[s._mixing_mode]]._power

        getter = lambda sim: sim.mode_output_powers(sim.lookback.mean)[
            idxs_by_mode[s._modulated_mode]
        ]

        efficiency = np.array(
            [
                (getter(w) - getter(wo)) / launched_mixing_power
                for w, wo in zip(with_mixing, without_mixing)
            ]
        )

        si.vis.xy_plot(
            f"{path.stem}__conversion_efficiency",
            scan_powers,
            efficiency,
            x_label=f"Launched {scan_mode.label} Power",
            y_label="Conversion Efficiency",
            x_unit="uW",
            # y_unit = 'pJ',
            # title = rf'Mode Energies for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
            x_log_axis=scan_powers[0] != 0,
            y_log_axis=True,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with LOGMAN as logger:
        paths = [Path(__file__).parent / "meff_test.sims"]

        for path in paths:
            mode_energy_plot_by_mixing_power(path)
