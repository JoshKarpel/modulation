import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import microspheres

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = "png",
    fig_dpi_scale = 6,
)


def find_modes(wavelength_bounds, microsphere, max_radial_mode_number):
    mode_locations = microspheres.find_mode_locations(
        wavelength_bounds = wavelength_bounds,
        microsphere = microsphere,
        max_radial_mode_number = max_radial_mode_number,
    )

    modes = [
        microspheres.MicrosphereMode.from_mode_location(
            mode_location,
            m = mode_location.l,
        )
        for mode_location in mode_locations
    ]

    return modes


if __name__ == "__main__":
    R = 50 * u.um
    max_radial_mode_number = 5
    material = raman.RamanMaterial.from_name("silica")
    pump_wavelength = 800 * u.nm
    pump_power = 1 * u.mW
    pump_start_time = 1 * u.usec
    time_final = 10 * u.usec
    time_step = 1 * u.nsec
    stokes_orders = 5
    antistokes_orders = 0
    intrinsic_q = 1e8
    coupling_q = 1e8

    wavelength_bounds = microspheres.sideband_bounds(
        pump_wavelength = pump_wavelength,
        stokes_orders = stokes_orders,
        antistokes_orders = antistokes_orders,
        sideband_frequency = material.modulation_frequency,
        bandwidth_frequency = .2 * material.raman_linewidth,
    )

    for b in wavelength_bounds:
        print(b)
    print()

    microsphere = microspheres.Microsphere(
        radius = R,
        index_of_refraction = material.index_of_refraction,
    )

    modes = find_modes(
        wavelength_bounds,
        microsphere,
        max_radial_mode_number = max_radial_mode_number,
    )

    for m in modes:
        print(m)

    print(f'found {len(modes)} modes')

    pump_mode = microspheres.find_mode_with_closest_wavelength(modes, pump_wavelength)
    print(f'pump mode is {pump_mode}')

    spec = raman.StimulatedRamanScatteringSpecification(
        'test',
        material = material,
        modes = modes,
        mode_volume_integrator = microspheres.FixedGridSimpsonMicrosphereVolumeIntegrator(
            microsphere = microsphere
        ),
        mode_initial_amplitudes = {m: 1 for m in modes},  # very important!
        mode_pumps = {pump_mode: raman.RectangularPump(
            start_time = pump_start_time,
            power = pump_power,
        )},
        mode_intrinsic_quality_factors = {m: intrinsic_q for m in modes},
        mode_coupling_quality_factors = {m: coupling_q for m in modes},
        time_final = time_final,
        time_step = time_step,
        store_mode_amplitudes_vs_time = True,
    )

    sim = spec.to_sim()
    print(sim.info())

    sim.run(progress_bar = True)
    print(sim.info())

    sim.plot.mode_magnitudes_vs_time(
        y_lower_limit = 1e7 * u.V_per_m,
        **PLOT_KWARGS,
    )
