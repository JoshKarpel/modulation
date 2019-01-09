import simulacra as si
import simulacra.units as u

import modulation

import numpy as np

from pathlib import Path

import gzip
import pickle

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / 'out' / THIS_FILE.stem
SIM_LIB = OUT_DIR / 'SIMLIB'

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def load_sims(path):
    with gzip.open(path, mode = 'rb') as f:
        return pickle.load(f)


COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']


def make_plot(name, sims):
    s = sims[0].spec
    modes = (
        s._pump_mode,
        s._stokes_mode,
        s._mixing_mode,
        s._modulated_mode,
    )
    idxs = [sims[0].mode_to_index[mode] for mode in modes]
    scan_mode = s._scan_mode
    fixed_mode = s._fixed_mode

    scan_power = np.array([sim.spec._scan_power for sim in sims]) / u.twopi

    means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
    mins = [sim.mode_energies(sim.lookback.min) for sim in sims]
    maxs = [sim.mode_energies(sim.lookback.max) for sim in sims]

    lines = []
    line_kwargs = []
    for idx, mode, color in zip(idxs, modes, COLORS):
        lines.append(np.array([mean[idx] for mean in means]))
        line_kwargs.append(dict(
            color = color,
        ))

        lines.append(np.array([m[idx] for m in mins]))
        line_kwargs.append(dict(
            color = color,
            linestyle = ':',
        ))

        lines.append(np.array([m[idx] for m in maxs]))
        line_kwargs.append(dict(
            color = color,
            linestyle = '--',
        ))

    si.vis.xy_plot(
        name,
        scan_power,
        *lines,
        # line_labels = [mode.label for mode in modes],
        line_kwargs = line_kwargs,
        x_label = f'Launched {scan_mode.label} Power',
        y_label = 'Steady-State Mode Energy',
        x_unit = 'uW',
        y_unit = 'pJ',
        title = rf'Modulation Efficiency for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.pumps[fixed_mode].power / u.uW:.1f} \, \mathrm{{\muW}} $',
        y_log_axis = True,
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    scans = [
    ]

    for scan in scans:
        sims = load_sims(Path(__file__).parent / 'data' / scan)
        make_plot(scan, sims)
