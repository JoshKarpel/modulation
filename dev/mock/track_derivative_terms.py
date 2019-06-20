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
ANIM_KWARGS = dict(target_dir=OUT_DIR, length=30, fps=60)


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
            if i == 0:
                label = r"\mathrm{Pump}"
            elif i == 1:
                label = "\mathrm{Stokes}"
            else:
                label = f"\mathrm{{pump}}_{{{i}}}"
            modes[f"pump_{i}"] = mock.MockMode(
                label=label,
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
            if i == 0:
                label = r"\mathrm{Mixing}"
            elif i == -1:
                label = "\mathrm{Target}"
            else:
                label = f"\mathrm{{mixing}}_{{{i}}}"
            modes[f"mixing_{i}"] = mock.MockMode(
                label=label,
                omega=mixing_omega - (i * material.modulation_omega) + detuning,
                mode_volume_inside_resonator=mode_volume,
                index_of_refraction=material.index_of_refraction,
            )

    return modes


linestyles = ["-", "-.", "--", ":"]


def mode_kwargs(sim, q, mode, pump_mode, mixing_mode):
    kwargs = {}

    if mode == pump_mode:
        kwargs["linestyle"] = "-"
    elif mode == mixing_mode:
        kwargs["linestyle"] = "--"

    if mode == pump_mode or mode == mixing_mode:
        kwargs["color"] = "black"

    return kwargs


def run(pump_power, mixing_power, **kwargs):
    pump_wavelength = 1064 * u.nm

    mixing_wavelength = 800 * u.nm
    # mixing_wavelength = None

    material = raman.RamanMaterial.from_name("silica")

    time_step = 100 * u.psec
    pump_start_time = 100 * u.nsec
    mixing_start_time = 1 * u.usec
    time_final = 3 * u.usec
    intrinsic_q = 1e8
    coupling_q = 1e8

    modes = make_modes(
        material,
        pump_wavelength=pump_wavelength,
        mixing_wavelength=mixing_wavelength,
        pump_stokes_orders=1,
        mixing_antistokes_orders=1,
        pump_antistokes_orders=0,
        mixing_stokes_orders=0,
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
                start_time=mixing_start_time,
                frequency=mixing_mode.frequency,
                power=mixing_power,
            )
        )

    postfix = "_".join(f"{k}={v}" for k, v in kwargs.items())
    if len(postfix) > 0:
        postfix = "__" + postfix

    mode_list = list(modes.values())
    spec = raman.FourWaveMixingSpecification(
        f"pump={pump_power / u.mW:.3f}mW_mixing={mixing_power / u.uW:.3f}uW{postfix}",
        material=material,
        modes=mode_list,
        mode_volume_integrator=mock.MockVolumeIntegrator(volume_integral_result=1e-25),
        mode_initial_amplitudes={m: 1 for m in mode_list},  # very important!
        pumps=pumps,
        mode_intrinsic_quality_factors={m: intrinsic_q for m in mode_list},
        mode_coupling_quality_factors={m: coupling_q for m in mode_list},
        time_final=time_final,
        time_step=time_step,
        store_mode_amplitudes_vs_time=True,
        pump_mode=pump_mode,
        mixing_mode=mixing_mode,
        four_mode_detuning_cutoff=raman.AUTO_CUTOFF,
        evolution_algorithm=raman.RungeKutta4(),
        **kwargs,
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

    return sim


if __name__ == "__main__":
    # pump_powers = [1 * u.mW]
    pump_powers = [1 * u.mW, 10 * u.mW, 100 * u.mW, 1 * u.W]
    mixing_powers = [1 * u.uW]

    for (pp, mp) in itertools.product(pump_powers, mixing_powers):
        sim = run(pump_power=pp, mixing_power=mp)
        decay, pump, pol = sim.extract_derivatives(
            sim.mode_amplitudes, sim.current_time
        )
        print(decay)
        print(pump)
        total_pol = np.einsum("qrst->q", pol)
        print(total_pol)
        print((decay + pump + total_pol) / sim.mode_amplitudes)
        # print(pol)
