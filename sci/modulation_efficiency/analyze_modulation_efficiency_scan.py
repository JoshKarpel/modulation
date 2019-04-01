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


def mode_energy_plot(path):
    sims = analysis.ParameterScan.from_file(path)
    s = sims[0].spec
    modes = s.modes
    idxs = [sims[0].mode_to_index[mode] for mode in modes]
    idxs_by_mode = dict(zip(modes, idxs))
    pump_mode = s.pump_mode
    wavelength_bounds = s.wavelength_bounds

    scan_powers = np.array([sim.spec.pump_power for sim in sims])

    means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
    mean_mags = [np.abs(sim.lookback.mean) for sim in sims]
    mins = [sim.mode_energies(sim.lookback.min) for sim in sims]
    maxs = [sim.mode_energies(sim.lookback.max) for sim in sims]

    lines = []
    mag_lines = []
    line_kwargs = []
    for idx, mode, color in zip(idxs, modes, COLORS):
        lines.append(np.array([mean[idx] for mean in means]))
        mag_lines.append(np.array([mean[idx] for mean in mean_mags]))
        line_kwargs.append(mode_kwargs(mode, pump_mode, wavelength_bounds))

    si.vis.xy_plot(
        f"{path.stem}__mode_energies",
        scan_powers,
        *lines,
        # line_labels=[str(mode) for mode in modes],
        line_kwargs=line_kwargs,
        x_label=f"Launched Pump Power",
        y_label="Steady-State Mode Energy",
        x_unit="mW",
        y_unit="pJ",
        x_log_axis=scan_powers[0] != 0,
        y_log_axis=True,
        y_lower_limit=1e-9 * u.pJ,
        y_upper_limit=1e3 * u.pJ,
        **PLOT_KWARGS,
    )

    # si.vis.xy_plot(
    #     f"mode_magnitudes___{name}",
    #     scan_powers,
    #     *mag_lines,
    #     line_labels=[mode.label for mode in modes],
    #     line_kwargs=line_kwargs,
    #     x_label=f"Launched {scan_mode.label} Power",
    #     y_label="Steady-State Mode Magnitudes (V/m)",
    #     x_unit="uW",
    #     y_unit=u.V_per_m,
    #     title=rf'Mode Magnitudes for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
    #     x_log_axis=scan_powers[0] != 0,
    #     y_log_axis=True,
    #     # save = False,
    #     # close = False,
    #     **PLOT_KWARGS,
    # )


def conversion_efficiency_plot(name, with_mixing, without_mixing):
    s = with_mixing[0].spec
    modes = (s._pump_mode, s._stokes_mode, s._mixing_mode, s._modulated_mode)
    idxs = [with_mixing[0].mode_to_index[mode] for mode in modes]
    idxs_by_mode = dict(zip(modes, idxs))
    scan_mode = s._scan_mode
    fixed_mode = s._fixed_mode

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
        f"conversion_efficiency___{name}",
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
        paths = [Path(__file__).parent / "cascaded_srs_threshold_scan.sims"]

        for path in paths:
            mode_energy_plot(path)
