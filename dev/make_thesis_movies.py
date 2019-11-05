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
                label = r"\mathrm{Stokes}"
            else:
                label = fr"\mathrm{{pump}}_{{{i}}}"
            modes[f"pump_{-i}"] = mock.MockMode(
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
                label = r"\mathrm{Target}"
            else:
                label = fr"\mathrm{{mixing}}_{{{i}}}"
            modes[f"mixing_{-i}"] = mock.MockMode(
                label=label,
                omega=mixing_omega - (i * material.modulation_omega) + detuning,
                mode_volume_inside_resonator=mode_volume,
                index_of_refraction=material.index_of_refraction,
            )

    return modes


COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]

MODE_TO_COLOR = {
    "pump_-1": "#377eb8",
    "pump_0": "#e41a1c",
    "mixing_0": "#4daf4a",
    "mixing_1": "#984ea3",
}


def mode_kwargs(sim, q, mode, pump_mode, mixing_mode):
    kwargs = {}

    kwargs["color"] = COLORS[q]

    return kwargs


def run(pump_power, mixing_power, material_name):
    material = raman.RamanMaterial.from_name(material_name)

    pump_wavelength = 1064 * u.nm

    mixing_wavelength = 800 * u.nm

    time_step = 1 * u.nsec
    pump_start_time = 0 * u.usec
    mixing_start_time = 2 * u.usec
    time_final = 5 * u.usec
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

    pump_mode = modes.get("pump_0", None)
    mixing_mode = modes.get("mixing_0", None)

    colors = list(MODE_TO_COLOR.values())

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

    tag = f"pump={pump_power / u.mW:.3f}mW_mixing={mixing_power / u.mW:.3f}mW_material={material_name}"
    mode_list = list(modes.values())
    spec = raman.FourWaveMixingSpecification(
        tag,
        material=material,
        modes=mode_list,
        mode_volume_integrator=mock.MockVolumeIntegrator(volume_integral_result=1e-25),
        mode_initial_amplitudes={m: 1e-15 for m in mode_list},  # very important!
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
        animators=[
            raman.anim.SquareAnimator(
                axman=raman.anim.PolarComplexAmplitudeAxis(
                    r_log_lower_limit=8, r_log_upper_limit=11, mode_colors=colors
                ),
                **ANIM_KWARGS,
            )
        ],
    )

    sim = spec.to_sim()

    print(sim.info())
    sim.run(progress_bar=True)
    print(sim.info())


if __name__ == "__main__":
    pump_powers = [10 * u.mW]
    mixing_powers = [100 * u.uW]
    materials = ["silica", "silica-narrow"]

    for pp, mp, mat in itertools.product(pump_powers, mixing_powers, materials):
        run(pump_power=pp, mixing_power=mp, material_name=mat)
