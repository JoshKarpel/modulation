import itertools
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


def mode_kwargs(sim, q, mode, pump_mode, mixing_mode, wavelength_bounds):
    base = {"alpha": 0.8}

    if mode == pump_mode:
        return {"color": "black", **base}
    elif mode == mixing_mode:
        return {"color": "black", **base}

    i = 0
    for bound in wavelength_bounds:
        if mode.wavelength in bound:
            return {"linestyle": linestyles[i], **base}
        i += 1


def run(pump_power, mixing_power, time_step, cutoff):
    pump_wavelength = 1064 * u.nm
    mixing_wavelength = 800 * u.nm

    R = 50 * u.um
    max_radial_mode_number = 5
    material = raman.RamanMaterial.from_name("silica")

    pump_start_time = 0 * u.usec
    time_final = 600 * u.nsec
    intrinsic_q = 1e8
    coupling_q = 1e8

    wavelength_bounds = microspheres.sideband_bounds(
        center_wavelength=pump_wavelength,
        stokes_orders=1,
        antistokes_orders=0,
        sideband_frequency=material.modulation_frequency,
        bandwidth_frequency=0.05 * material.raman_linewidth / u.twopi,
    )
    wavelength_bounds += microspheres.sideband_bounds(
        center_wavelength=mixing_wavelength,
        stokes_orders=0,
        antistokes_orders=1,
        sideband_frequency=material.modulation_frequency,
        bandwidth_frequency=0.05 * material.raman_linewidth / u.twopi,
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
    mixing_mode = microspheres.find_mode_with_closest_wavelength(
        modes, mixing_wavelength
    )
    print("pump mode is", pump_mode)
    print("mixing mode is", mixing_mode)

    cutoff_str = (
        f"cutoff={cutoff / u.THz:.3f}THz" if cutoff is not None else "cutoff=none"
    )
    spec = raman.FourWaveMixingSpecification(
        f"pump={pump_power / u.mW:.3f}mW_mixing={mixing_power / u.uW:.3f}uW_dt={time_step / u.psec:.3f}ps_{cutoff_str}",
        material=material,
        modes=modes,
        mode_volume_integrator=microspheres.FixedGridSimpsonMicrosphereVolumeIntegrator(
            microsphere=microsphere
        ),
        mode_initial_amplitudes={m: 0 for m in modes},  # very important!
        pumps=[
            raman.RectangularMonochromaticPump(
                start_time=pump_start_time,
                frequency=pump_mode.frequency,
                power=pump_power,
            ),
            raman.RectangularMonochromaticPump(
                start_time=pump_start_time,
                frequency=mixing_mode.frequency,
                power=mixing_power,
            ),
        ],
        mode_intrinsic_quality_factors={m: intrinsic_q for m in modes},
        mode_coupling_quality_factors={m: coupling_q for m in modes},
        time_final=time_final,
        time_step=time_step,
        store_mode_amplitudes_vs_time=True,
        pump_mode=pump_mode,
        mixing_mode=mixing_mode,
        four_mode_detuning_cutoff=cutoff,
    )

    try:
        sim = si.Simulation.load(OUT_DIR / f"{spec.name}.sim")
    except FileNotFoundError:
        sim = spec.to_sim()
        print(sim.info())

        sim.run(progress_bar=True)
        print(sim.info())

        sim.save(target_dir=OUT_DIR)

    # sim.plot.mode_magnitudes_vs_time(
    #     # y_lower_limit=1e5 * u.V_per_m,
    #     # y_upper_limit=1e10 * u.V_per_m,
    #     average_over=10 * u.nsec,
    #     mode_kwargs=lambda s, q, mode: mode_kwargs(
    #         s, q, mode, sim.spec.pump_mode, sim.spec.mixing_mode, wavelength_bounds
    #     ),
    #     font_size_legend=8,
    #     **PLOT_KWARGS,
    # )
    sim.plot.mode_energies_vs_time(
        y_lower_limit=1e-12 * u.pJ,
        y_upper_limit=1e4 * u.pJ,
        average_over=0.1 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(
            s, q, mode, sim.spec.pump_mode, sim.spec.mixing_mode, wavelength_bounds
        ),
        font_size_legend=8,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    time_steps = [10 * u.psec, 1 * u.psec, 0.1 * u.psec, 0.05 * u.psec]
    cutoffs = [None, 10 * u.THz, 20 * u.THz]
    for time_step, cutoff in itertools.product(time_steps, cutoffs):
        run(
            pump_power=1 * u.W,
            mixing_power=1 * u.uW,
            time_step=time_step,
            cutoff=cutoff,
        )
