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


def make_plot(name, sims):
    idx_pump = sims[0].mode_to_index[sims[0].spec._pump_mode]
    idx_stokes = sims[0].mode_to_index[sims[0].spec._stokes_mode]
    idx_mixing = sims[0].mode_to_index[sims[0].spec._mixing_mode]
    idx_modulated = sims[0].mode_to_index[sims[0].spec._modulated_mode]

    pump_power = np.array([sim.spec._pump_power for sim in sims]) / u.twopi

    mode_energies = [sim.mode_energies(sim.mode_amplitudes) for sim in sims]
    pump_energies = np.array([energies[idx_pump] for energies in mode_energies])
    stokes_energies = np.array([energies[idx_stokes] for energies in mode_energies])
    mixing_energies = np.array([energies[idx_mixing] for energies in mode_energies])
    modulated_energies = np.array([energies[idx_modulated] for energies in mode_energies])

    si.vis.xy_plot(
        name,
        pump_power,
        pump_energies,
        stokes_energies,
        mixing_energies,
        modulated_energies,
        line_labels = [
            'Pump',
            'Stokes',
            'Mixing',
            'Modulated',
        ],
        x_label = 'Launched Pump Power',
        y_label = 'Steady-State Mode Energy',
        x_unit = 'uW',
        y_unit = 'pJ',
        y_log_axis = True,
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    scans = [
        'modulation_efficiency_qm=qmm=1e8.sims',
    ]

    for scan in scans:
        sims = load_sims(Path(__file__).parent / 'data' / scan)
        make_plot(scan, sims)
