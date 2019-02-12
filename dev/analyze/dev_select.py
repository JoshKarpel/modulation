import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman, analysis
from modulation.resonators import mock

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = "png",
    fig_dpi_scale = 6,
)


def test():
    ps = analysis.ParameterScan.from_file(Path(__file__).parent / 'test.sims')
    print(ps, len(ps))

    pump_powers = ps.parameter_set('_pump_power')
    mixing_powers = ps.parameter_set('_mixing_power')

    print(len(pump_powers), pump_powers)
    print(len(mixing_powers), mixing_powers)

    x = []
    y = []
    sims_by_mixing_power = {mp: ps.select(_mixing_power = mp) for mp in mixing_powers}
    for mixing_power, sims in sims_by_mixing_power.items():
        x.append([s.spec._pump_power for s in sims])
        y.append([s.mode_energies(s.lookback.mean)[s.mode_to_index[s.spec._modulated_mode]] for s in sims])

    si.vis.xxyy_plot(
        'test',
        x,
        y,
        x_unit = 'uW',
        y_unit = 'pJ',
        x_label = 'Pump Power',
        y_label = 'Energy in Modulated Mixing Mode',
        **PLOT_KWARGS,
    )


def more_test():
    ps = analysis.ParameterScan.from_file(Path(__file__).parent / 'pump_power_vs_mmm_q.sims')
    print(ps, len(ps))

    pump_powers = ps.parameter_set('_pump_power')
    mixing_powers = ps.parameter_set('_mixing_power')

    print(f'# pump powers: {len(pump_powers)}')
    print(f'# mixing powers: {len(mixing_powers)}')
    pump_mode = ps.sims[0].spec._pump_mode
    stokes_mode = ps.sims[0].spec._stokes_mode
    mixing_mode = ps.sims[0].spec._mixing_mode
    modulated_mode = ps.sims[0].spec._modulated_mode
    mti = ps.sims[0].mode_to_index

    print(pump_mode)
    print(stokes_mode)
    print(mixing_mode)
    print(modulated_mode)

    x = []
    y = []
    sims_by_mixing_power = {
        mp: ps.select(_mixing_power = mp)
        for mp in mixing_powers
    }
    for mixing_power, sims in sims_by_mixing_power.items():
        print(len(sims))
        x.append([s.spec._pump_power for s in sims])
        y.append([s.mode_energies(s.lookback.mean)[mti[modulated_mode]] for s in sims])

    si.vis.xxyy_plot(
        'pump_power_vs_mmm_q',
        x,
        y,
        x_unit = 'uW',
        y_unit = 'pJ',
        x_label = 'Pump Power',
        y_label = 'Energy in Modulated Mixing Mode',
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        # test()
        more_test()
