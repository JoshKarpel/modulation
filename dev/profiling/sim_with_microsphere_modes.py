import logging
import itertools
import time
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import microspheres

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def find_modes(wavelength_bounds, microsphere, max_radial_mode_number):
    mode_locations = microspheres.find_mode_locations(
        wavelength_bounds=wavelength_bounds,
        microsphere=microsphere,
        max_radial_mode_number=max_radial_mode_number,
    )

    modes = [
        microspheres.MicrosphereMode.from_mode_location(
            mode_location, m=mode_location.l
        )
        for mode_location in mode_locations
    ]

    return modes


linestyles = ["-", "-.", "--", ":"]


def mode_kwargs(sim, q, mode, pump_mode, wavelength_bounds):
    base = {"alpha": 0.8}

    if mode is pump_mode:
        return {"color": "black", **base}

    i = 0
    for bound in wavelength_bounds:
        if mode.wavelength in bound:
            return {"linestyle": linestyles[i], **base}
        i += 1


def run():
    R = 50 * u.um
    max_radial_mode_number = 5
    material = raman.RamanMaterial.from_name("silica")
    pump_wavelength = 800 * u.nm
    pump_start_time = 1 * u.usec
    time_step = 1 * u.psec
    time_final = 10000 * u.psec
    stokes_orders = 2
    antistokes_orders = 0  # you won't see any amplitude on these unless you have fwm
    bnd = 0.2
    intrinsic_q = 1e8
    coupling_q = 1e8
    pump_power = 1 * u.mW

    wavelength_bounds = microspheres.sideband_bounds(
        center_wavelength=pump_wavelength,
        stokes_orders=stokes_orders,
        antistokes_orders=antistokes_orders,
        sideband_frequency=material.modulation_frequency,
        bandwidth_frequency=bnd * material.raman_linewidth,
    )

    for b in wavelength_bounds:
        print(b)
    print()

    microsphere = microspheres.Microsphere(
        radius=R, index_of_refraction=material.index_of_refraction
    )

    modes = find_modes(
        wavelength_bounds, microsphere, max_radial_mode_number=max_radial_mode_number
    )

    for m in modes:
        print(m)

    print(f"found {len(modes)} modes")

    pump_mode = microspheres.find_mode_with_closest_wavelength(modes, pump_wavelength)
    print(f"pump mode is {pump_mode}")

    # spec = raman.StimulatedRamanScatteringSpecification(
    spec = raman.FourWaveMixingSpecification(
        f"pump={pump_power / u.mW:.3f}mW_dt={time_step / u.psec:.3f}ps",
        material=material,
        modes=modes,
        mode_volume_integrator=microspheres.FixedGridSimpsonMicrosphereVolumeIntegrator(
            microsphere=microsphere
        ),
        mode_initial_amplitudes={m: 1 for m in modes},  # very important!
        pumps=[
            raman.RectangularMonochromaticPump(
                start_time=pump_start_time,
                frequency=pump_mode.frequency,
                power=pump_power,
            )
        ],
        mode_intrinsic_quality_factors={m: intrinsic_q for m in modes},
        mode_coupling_quality_factors={m: coupling_q for m in modes},
        time_final=time_final,
        time_step=time_step,
        store_mode_amplitudes_vs_time=False,
        _pump_mode=pump_mode,
    )

    sim = spec.to_sim()
    print(sim.info())

    sim.run(progress_bar=True)
    print(sim.info())

    # sim.plot.mode_energies_vs_time(
    #     y_lower_limit=1e-5 * u.pJ,
    #     y_upper_limit=2e1 * u.pJ,
    #     average_over=10 * u.nsec,
    #     y_log_pad=1,
    #     mode_kwargs=lambda sim, q, mode: mode_kwargs(
    #         sim, q, mode, pump_mode, wavelength_bounds
    #     ),
    #     font_size_legend=8,
    #     **PLOT_KWARGS,
    # )


if __name__ == "__main__":
    run()
    time.sleep(1)
