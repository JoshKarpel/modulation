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
        paths = [Path(__file__).parent / "cascaded_srs_convergence_test.sims"]

        for path in paths:
            final_time_and_time_step_scan(path)
