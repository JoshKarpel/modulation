import simulacra as si
import simulacra.units as u

import modulation

import numpy as np

from pathlib import Path

import gzip
import pickle

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def load_sims(path):
    with gzip.open(path, mode="rb") as f:
        return pickle.load(f)


def make_plot(name, sims):
    idx_pump = sims[0].mode_to_index[sims[0].spec._pump_mode]
    idx_stokes = sims[0].mode_to_index[sims[0].spec._stokes_mode]

    detunings = np.array([sim.spec._detuning_omega for sim in sims]) / u.twopi

    mode_energies = [sim.mode_energies(sim.mode_amplitudes) for sim in sims]
    pump_energies = np.array([energies[idx_pump] for energies in mode_energies])
    stokes_energies = np.array([energies[idx_stokes] for energies in mode_energies])

    si.vis.xy_plot(
        name,
        detunings,
        pump_energies,
        stokes_energies,
        line_labels=["Pump", "Stokes"],
        x_label="Stokes Mode Detuning",
        y_label="Steady-State Mode Energy",
        x_unit="GHz",
        y_unit="pJ",
        y_lower_limit=0,
        y_pad=0.05,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    scans = [
        "stokes_detuning__800nm_pump__fwm_.01ns_2.sims",
        "stokes_detuning__800nm_pump__fixed_quality.sims",
        "stokes_detuning__800nm_pump__fixed_timescale.sims",
        "stokes_detuning__1064nm_pump__fixed_quality.sims",
        "stokes_detuning__1064nm_pump__fixed_timescale.sims",
    ]

    for scan in scans:
        sims = load_sims(Path(__file__).parent / "data" / scan)
        make_plot(scan, sims)
