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
    idxs_by_mode = dict(zip(modes, idxs))
    scan_mode = s._scan_mode
    fixed_mode = s._fixed_mode

    scan_powers = np.array([sim.spec._scan_power for sim in sims])

    means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
    mins = [sim.mode_energies(sim.lookback.min) for sim in sims]
    maxs = [sim.mode_energies(sim.lookback.max) for sim in sims]

    lines = []
    fills = []
    line_kwargs = []
    for idx, mode, color in zip(idxs, modes, COLORS):
        lines.append(np.array([mean[idx] for mean in means]))
        line_kwargs.append(dict(
            color = color,
        ))

        fills.append((np.array([m[idx] for m in mins]), np.array([m[idx] for m in maxs])))

    figman = si.vis.xy_plot(
        name,
        scan_powers,
        *lines,
        line_labels = [mode.label for mode in modes],
        line_kwargs = line_kwargs,
        x_label = f'Launched {scan_mode.label} Power',
        y_label = 'Steady-State Mode Energy',
        x_unit = 'uW',
        y_unit = 'pJ',
        title = rf'Modulation Efficiency for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
        y_log_axis = True,
        legend_on_right = True,
        save = False,
        close = False,
        **PLOT_KWARGS,
    )

    ax = figman.fig.gca()
    for (lower, upper), color in zip(fills, COLORS):
        ax.fill_between(
            scan_powers / u.uW,
            lower / u.pJ,
            upper / u.pJ,
            facecolor = color,
            alpha = 0.5,
            linewidth = 0,
        )

    figman.save()
    figman.cleanup()


if __name__ == '__main__':
    scans = [
        'mod_eff_test_2.sims',
        'mod_eff_test_3.sims',
    ]

    for scan in scans:
        sims = load_sims(Path(__file__).parent / 'data' / scan)
        make_plot(scan, sims)
