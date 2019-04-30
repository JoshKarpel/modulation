import itertools
import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import mock

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def make_modes(
    material,
    pump_wavelength,
    mixing_wavelength=None,
    pump_stokes_orders=1,
    pump_antistokes_orders=0,
    pump_mode_detunings=None,
    mixing_stokes_orders=0,
    mixing_antistokes_orders=1,
    mixing_mode_detunings=None,
    mode_volume=1e-20,
):
    if pump_mode_detunings is None:
        pump_mode_detunings = [0] * (pump_stokes_orders + pump_antistokes_orders + 1)
    if mixing_mode_detunings is None:
        mixing_mode_detunings = [0] * (
            mixing_stokes_orders + mixing_antistokes_orders + 1
        )

    modes = {}

    if pump_wavelength is not None:
        pump_omega = u.twopi * u.c / pump_wavelength

        for i, detuning in zip(
            range(-pump_antistokes_orders, pump_stokes_orders + 1), pump_mode_detunings
        ):
            modes[f"pump_{i}"] = mock.MockMode(
                label=f"\mathrm{{pump}}_{{{i}}}",
                omega=pump_omega - (i * material.modulation_omega) + detuning,
                mode_volume_inside_resonator=mode_volume,
                index_of_refraction=material.index_of_refraction,
            )

    if mixing_wavelength is not None:
        mixing_omega = u.twopi * u.c / mixing_wavelength

        for i, detuning in zip(
            range(-mixing_antistokes_orders, mixing_stokes_orders + 1),
            mixing_mode_detunings,
        ):
            modes[f"mixing_{i}"] = mock.MockMode(
                label=f"\mathrm{{mixing}}_{{{i}}}",
                omega=mixing_omega - (i * material.modulation_omega) + detuning,
                mode_volume_inside_resonator=mode_volume,
                index_of_refraction=material.index_of_refraction,
            )

    return modes


linestyles = ["-", "-.", "--", ":"]


def mode_kwargs(sim, q, mode, pump_mode, mixing_mode):
    kwargs = {}

    if "pump" in mode.label:
        kwargs["linestyle"] = "-"
    elif "mixing" in mode.label:
        kwargs["linestyle"] = "--"

    if mode == pump_mode or mode == mixing_mode:
        kwargs["color"] = "black"

    return kwargs


def run(time_step, cutoff):
    pump_power = 1 * u.W
    mixing_power = 0 * u.uW
    pump_wavelength = 1064 * u.nm
    # mixing_wavelength = 800 * u.nm
    mixing_wavelength = None

    material = raman.RamanMaterial.from_name("silica")

    pump_start_time = 50 * u.nsec
    time_final = 500 * u.nsec
    intrinsic_q = 1e8
    coupling_q = 1e8

    modes = make_modes(
        material,
        pump_wavelength=pump_wavelength,
        mixing_wavelength=mixing_wavelength,
        pump_stokes_orders=1,
        mixing_antistokes_orders=1,
    )

    for l, m in modes.items():
        print(l, m)

    pump_mode = modes.get("pump_0", None)
    mixing_mode = modes.get("mixing_0", None)

    pumps = []
    if pump_mode is not None:
        pumps.append(
            raman.RectangularMonochromaticPump(
                start_time=pump_start_time,
                frequency=pump_mode.frequency,
                power=pump_power,
            )
        )
    if mixing_mode is not None:
        pumps.append(
            raman.RectangularMonochromaticPump(
                start_time=pump_start_time,
                frequency=mixing_mode.frequency,
                power=mixing_power,
            )
        )

    cutoff_str = (
        f"cutoff={cutoff / u.THz:.1f}THz"
        if cutoff is not raman.AUTO_CUTOFF
        else "cutoff=AUTO"
    )
    mode_list = list(modes.values())
    spec = raman.FourWaveMixingSpecification(
        f"pump={pump_power / u.mW:.3f}mW_mixing={mixing_power / u.uW:.3f}uW_dt={time_step / u.psec:.3f}ps_{cutoff_str}",
        material=material,
        modes=mode_list,
        mode_volume_integrator=mock.MockVolumeIntegrator(volume_integral_result=1e-25),
        mode_initial_amplitudes={m: 1e-15 for m in modes},  # very important!
        pumps=pumps,
        mode_intrinsic_quality_factors={m: intrinsic_q for m in mode_list},
        mode_coupling_quality_factors={m: coupling_q for m in mode_list},
        time_final=time_final,
        time_step=time_step,
        store_mode_amplitudes_vs_time=True,
        pump_mode=pump_mode,
        mixing_mode=mixing_mode,
        four_mode_detuning_cutoff=cutoff,
    )

    sim = spec.to_sim()

    print(sim.info())
    sim.run(progress_bar=True)
    print(sim.info())

    sim.plot.mode_energies_vs_time(
        y_lower_limit=1e-12 * u.pJ,
        y_upper_limit=1e4 * u.pJ,
        average_over=5 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(
            s, q, mode, sim.spec.pump_mode, sim.spec.mixing_mode
        ),
        font_size_legend=8,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    time_steps = [
        # 1 * u.psec,
        # 1.1 * u.psec,
        # 10 * u.psec,
        # 11 * u.psec,
        # 100 * u.psec,
        # 110 * u.psec,
        1 * u.nsec,
        # 1.1 * u.nsec,
    ]
    cutoffs = [raman.AUTO_CUTOFF, 10 * u.THz, np.Inf]
    for time_step, cutoff in itertools.product(time_steps, cutoffs):
        run(time_step=time_step, cutoff=cutoff)
