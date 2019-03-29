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
            return {"linestyle": LINESTYLES[i], **base}
        i += 1


def time_step_scan(path):
    ps = analysis.ParameterScan.from_file(path)
    print(ps, len(ps))

    sim = ps[0]

    modes = set()
    for sim in ps:
        modes.update(sim.spec.modes)
    modes = sorted(modes, key=lambda m: -m.wavelength)
    print("total unique modes", len(modes))

    x = np.empty(len(ps))
    mode_to_y = {m: np.zeros_like(x) * np.NaN for m in modes}

    for idx, sim in enumerate(sorted(ps, key=lambda s: s.spec.time_step)):
        time_step = sim.spec.time_step
        print(f"time step is {time_step / u.psec:.6f} ps")
        print(f"pump power is {sim.spec.pumps[0].power / u.mW:.6f} mW")
        x[idx] = time_step

        for mode, energy in zip(sim.spec.modes, sim.mode_energies(sim.lookback.mean)):
            mode_to_y[mode][idx] = energy

    kwargs = [
        mode_kwargs(mode, sim.spec.pump_mode, sim.spec.wavelength_bounds)
        for mode in modes
    ]

    y_lower_lim = 1e-12 * u.pJ
    y_upper_lim = 1e1 * u.pJ

    y = list(mode_to_y.values())
    si.vis.xy_plot(
        f"{path.stem}__time_step_scan",
        x,
        *y,
        line_labels=[rf"${mode.tex}$" for mode in modes],
        line_kwargs=kwargs,
        x_unit="psec",
        y_unit="pJ",
        x_label=r"Time Step",
        y_label="Mode Energies",
        x_log_axis=True,
        y_log_axis=True,
        legend_on_right=True,
        font_size_legend=8,
        y_lower_limit=y_lower_lim,
        y_upper_limit=y_upper_lim,
        **PLOT_KWARGS,
    )


def final_time_and_time_step_scan(path):
    ps = analysis.ParameterScan.from_file(path)
    print(ps, len(ps))
    sim = ps[0]

    final_times = np.array(sorted(ps.parameter_set("time_final"), key=lambda x: -x))
    time_steps = np.array(sorted(ps.parameter_set("time_step")))
    final_time_mesh, time_step_mesh = np.meshgrid(
        final_times, time_steps, indexing="xy"
    )

    print(final_times)
    print(time_steps)

    print(len(final_times), len(time_steps))

    sims = {(sim.spec.time_final, sim.spec.time_step): sim for sim in ps}

    modes = sim.spec.modes
    mti = sim.mode_to_index
    print(len(modes))
    final_amplitude_by_mode = {mode: np.empty_like(final_time_mesh) for mode in modes}
    for mode, amplitude_mesh in final_amplitude_by_mode.items():
        for i, final_time in enumerate(final_times):
            for j, time_step in enumerate(time_steps):
                # print(i, j, final_time, time_step)
                mode_index = mti[mode]
                amplitude_mesh[i, j] = sims[(final_time, time_step)].lookback.mean[
                    mode_index
                ]

    shared = dict(
        x_label="Time Final",
        y_label="Time Step",
        x_unit="usec",
        y_unit="psec",
        x_log_axis=True,
        y_log_axis=True,
        z_log_axis=True,
    )

    for mode, amplitude_mesh in final_amplitude_by_mode.items():
        # si.vis.xyz_plot(
        #     f"{path.stem}__ABSOLUTE__time_final_and_time_step__{mode}",
        #     final_time_mesh,
        #     time_step_mesh,
        #     np.abs(amplitude_mesh),
        #     **shared,
        #     **PLOT_KWARGS,
        # )

        best = np.abs(amplitude_mesh[0, 0])  # longest final time, shortest time step

        si.vis.xyz_plot(
            f"{path.stem}__FRACDIFF__time_final_and_time_step__{mode}",
            final_time_mesh,
            time_step_mesh,
            np.abs(amplitude_mesh) / best,
            **shared,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with LOGMAN as logger:
        paths = [Path(__file__).parent / "srs_convergence_test.sims"]

        for path in paths:
            time_step_scan(path)
