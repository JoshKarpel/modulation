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


def run():
    R = 50 * u.um
    max_radial_mode_number = 5
    material = raman.RamanMaterial.from_name("silica")
    pump_wavelength = 1064 * u.nm
    pump_start_time = 1 * u.usec
    time_final = 10 * u.usec
    stokes_orders = 1
    antistokes_orders = 1  # you won't see any amplitude on these unless you have fwm
    intrinsic_q = 1e8
    coupling_q = 1e8
    pump_power = 10 * u.mW
    time_step = 10 * u.psec

    mixing_wavelength = 800 * u.nm

    wavelength_bounds = microspheres.sideband_bounds(
        center_wavelength=pump_wavelength,
        stokes_orders=stokes_orders,
        antistokes_orders=antistokes_orders,
        sideband_frequency=material.modulation_frequency,
        bandwidth_frequency=0.1 * material.raman_linewidth / u.twopi,
    )
    wavelength_bounds += microspheres.sideband_bounds(
        center_wavelength=mixing_wavelength,
        stokes_orders=stokes_orders,
        antistokes_orders=antistokes_orders,
        sideband_frequency=material.modulation_frequency,
        bandwidth_frequency=0.1 * material.raman_linewidth / u.twopi,
    )

    # for b in wavelength_bounds:
    #     print(b)
    # print()

    microsphere = microspheres.Microsphere(
        radius=R, index_of_refraction=material.index_of_refraction
    )

    modes = find_modes(
        wavelength_bounds, microsphere, max_radial_mode_number=max_radial_mode_number
    )

    for m in modes:
        print(m, m.mode_volume_inside_resonator)

    mode_volumes = np.array([m.mode_volume_inside_resonator for m in modes])

    print(f"found {len(modes)} modes")
    print("min", np.min(mode_volumes))
    print("max", np.max(mode_volumes))
    print("median", np.median(mode_volumes))
    print("mean", np.mean(mode_volumes))
    print("std", np.std(mode_volumes))

    pump_mode = microspheres.find_mode_with_closest_wavelength(modes, pump_wavelength)
    # print(f'pump mode is {pump_mode}')

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
        store_mode_amplitudes_vs_time=True,
        _pump_mode=pump_mode,
    )

    sim = spec.to_sim()
    print(sim.info())

    # print(sim.polarization_sum_factors.shape)
    # print(sim.polarization_sum_factors.shape[0] ** 4)

    all_factors = np.abs(sim.mode_volume_coupling_factors.flatten())
    # print(all_factors)

    print("min", np.min(all_factors))
    print("max", np.max(all_factors))
    print("median", np.median(all_factors))
    print("mean", np.mean(all_factors))
    print("std", np.std(all_factors))

    with si.vis.FigureManager("coupling_factors_hist", **PLOT_KWARGS) as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        ax.hist(np.log10(all_factors / np.mean(all_factors)), bins=20)

        # ax.set_yscale("log")

        ax.set_ylabel("Frequency")
        ax.set_xlabel(
            r"$\mathrm{Log_{10}}$ of Normalized Mode Coupling Factor Magnitude"
        )

    omega_q, omega_r, omega_s, omega_t = np.meshgrid(
        sim.mode_omegas,
        sim.mode_omegas,
        sim.mode_omegas,
        sim.mode_omegas,
        indexing="ij",
        sparse=True,
    )
    omega_differences = omega_r + omega_t - omega_s - omega_q
    all_freq_diffs = (
        omega_differences[np.nonzero(omega_differences)].flatten() / u.twopi
    )

    with si.vis.FigureManager("omega_diffs_hist", **PLOT_KWARGS) as fm:
        fig = fm.fig
        ax = fig.add_subplot(111)

        ax.hist(np.log10(np.abs(all_freq_diffs)), bins=40, log=True)

        # ax.set_yscale("log")

        ax.set_ylabel("Frequency")
        ax.set_xlabel(r"$\mathrm{Log_{10}}$ of Frequency Differences")

        dt = np.array([1, 10, 100]) * u.psec
        vlines = np.log10(1 / dt)
        for vline in vlines:
            ax.axvline(vline, color="black")

    # sim.run(progress_bar=True)
    # print(sim.info())
    #
    # # sim.plot.mode_magnitudes_vs_time(
    # #     y_lower_limit = 1e5 * u.V_per_m,
    # #     y_upper_limit = 1e10 * u.V_per_m,
    # #     average_over = 10 * u.nsec,
    # #     mode_kwargs = lambda sim, q, mode: mode_kwargs(sim, q, mode, pump_mode, wavelength_bounds),
    # #     **PLOT_KWARGS,
    # # )
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
    # postfix = "OFF_RESONANT"
    # powers = np.array([1, 2, 4]) * u.mW
    # time_steps = np.array([100, 10, 1]) * u.psec
    #
    # for power, time_step in itertools.product(powers, time_steps):
    #     run(postfix, power, time_step)

    run()
