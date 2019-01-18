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


def make_mode_energy_plot(name, sims):
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
    mean_mags = [np.abs(sim.lookback.mean) for sim in sims]
    mins = [sim.mode_energies(sim.lookback.min) for sim in sims]
    maxs = [sim.mode_energies(sim.lookback.max) for sim in sims]

    lines = []
    mag_lines = []
    fills = []
    line_kwargs = []
    for idx, mode, color in zip(idxs, modes, COLORS):
        lines.append(np.array([mean[idx] for mean in means]))
        mag_lines.append(np.array([mean[idx] for mean in mean_mags]))
        line_kwargs.append(dict(
            color = color,
        ))

        fills.append((np.array([m[idx] for m in mins]), np.array([m[idx] for m in maxs])))

    figman = si.vis.xy_plot(
        f'mode_energies___{name}',
        scan_powers,
        *lines,
        line_labels = [mode.label for mode in modes],
        line_kwargs = line_kwargs,
        x_label = f'Launched {scan_mode.label} Power',
        y_label = 'Steady-State Mode Energy',
        x_unit = 'uW',
        y_unit = 'pJ',
        title = rf'Mode Energies for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
        x_log_axis = scan_powers[0] != 0,
        y_log_axis = True,
        y_lower_limit = 1e-8 * u.pJ,
        y_upper_limit = 1e6 * u.pJ,
        # save = False,
        # close = False,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        f'mode_magnitudes___{name}',
        scan_powers,
        *mag_lines,
        line_labels = [mode.label for mode in modes],
        line_kwargs = line_kwargs,
        x_label = f'Launched {scan_mode.label} Power',
        y_label = 'Steady-State Mode Magnitudes (V/m)',
        x_unit = 'uW',
        y_unit = u.V_per_m,
        title = rf'Mode Magnitudes for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
        x_log_axis = scan_powers[0] != 0,
        y_log_axis = True,
        # save = False,
        # close = False,
        **PLOT_KWARGS,
    )

    # ax = figman.fig.gca()
    # for (lower, upper), color in zip(fills, COLORS):
    #     ax.fill_between(
    #         scan_powers / u.uW,
    #         lower / u.pJ,
    #         upper / u.pJ,
    #         facecolor = color,
    #         alpha = 0.5,
    #         linewidth = 0,
    #     )

    # figman.save()
    # figman.cleanup()


def make_conversion_efficiency_plot(name, with_mixing, without_mixing):
    s = with_mixing[0].spec
    modes = (
        s._pump_mode,
        s._stokes_mode,
        s._mixing_mode,
        s._modulated_mode,
    )
    idxs = [with_mixing[0].mode_to_index[mode] for mode in modes]
    idxs_by_mode = dict(zip(modes, idxs))
    scan_mode = s._scan_mode
    fixed_mode = s._fixed_mode

    scan_powers = np.array([sim.spec._scan_power for sim in with_mixing])

    launched_mixing_power = s.mode_pumps[idxs_by_mode[s._mixing_mode]]._power

    getter = lambda sim: sim.mode_output_powers(sim.lookback.mean)[idxs_by_mode[s._modulated_mode]]

    efficiency = np.array([
        (getter(w) - getter(wo)) / launched_mixing_power
        for w, wo in zip(with_mixing, without_mixing)
    ])

    si.vis.xy_plot(
        f'conversion_efficiency___{name}',
        scan_powers,
        efficiency,
        x_label = f'Launched {scan_mode.label} Power',
        y_label = 'Conversion Efficiency',
        x_unit = 'uW',
        # y_unit = 'pJ',
        # title = rf'Mode Energies for $ P_{{\mathrm{{{fixed_mode.label}}}}} = {s.mode_pumps[idxs_by_mode[fixed_mode]]._power / u.uW:.1f} \, \mathrm{{\mu W}} $',
        x_log_axis = scan_powers[0] != 0,
        y_log_axis = True,
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    # scans = [
    #     'unstim_test.sims',
    # ]
    # scans = [
    #     f'modeff__scan_pump__Q=1e{n}__mixing={m}.sims'
    #     for n in [4, 5, 6]
    #     for m in ['0', '1uW']
    # ]
    scans = [
        f'unstim_Q=1e{n}_crit_mixing={m}uW.sims'
        for n in [4, 5, 6]
        for m in [0, 1]
    ]

    for scan in scans:
        sims = load_sims(Path(__file__).parent / 'data' / scan)
        make_mode_energy_plot(scan, sims)

    paired_scans = [
        tuple(f'unstim_Q=1e{n}_crit_mixing={m}uW.sims' for m in [1, 0])
        for n in [4, 5, 6]
    ]
    for (w, wo), n in zip(paired_scans, [4, 5, 6]):
        make_conversion_efficiency_plot(
            f'Q=1e{n}_crit',
            load_sims(Path(__file__).parent / 'data' / w),
            load_sims(Path(__file__).parent / 'data' / wo),
        )
