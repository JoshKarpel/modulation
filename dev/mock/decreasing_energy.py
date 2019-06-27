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
ANIM_KWARGS = dict(target_dir=OUT_DIR, length=20, fps=30)


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

MODE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def mode_kwargs(sim, q, mode, pump_mode, mixing_mode):
    kwargs = {}

    kwargs["color"] = MODE_COLORS[q]

    return kwargs


def run(pump_power, mixing_power, **kwargs):
    pump_wavelength = 1064 * u.nm

    mixing_wavelength = 800 * u.nm

    material = raman.RamanMaterial.from_name("silica")

    time_step = 50 * u.psec
    pump_start_time = 50 * u.nsec
    mixing_start_time = 2 * u.usec
    time_final = 5 * u.usec
    intrinsic_q = 1e8
    coupling_q = 1e8

    # stokes_start_time = 1 * u.usec
    # stokes_power = pump_power

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
    # stokes_mode = modes.get("pump_1", None)
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
        # pumps.append(
        #     raman.RectangularMonochromaticPump(
        #         start_time=stokes_start_time,
        #         frequency=stokes_mode.frequency,
        #         power=stokes_power,
        #     )
        # )
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

    tag = f"pump={pump_power / u.mW:.3f}mW_mixing={mixing_power / u.mW:.3f}mW{postfix}__dt={time_step / u.psec:.3f}ps"
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
                    r_log_lower_limit=5, r_log_upper_limit=12, mode_colors=MODE_COLORS
                ),
                **ANIM_KWARGS,
            )
        ],
        **kwargs,
    )

    sim = spec.to_sim()

    print(sim.info())
    sim.run(progress_bar=True)
    print(sim.info())

    sim.plot.mode_complex_amplitudes_vs_time(
        # y_lower_limit=1e-20 * u.pJ,
        # y_upper_limit=1e4 * u.pJ,
        average_over=5 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(
            s, q, mode, sim.spec.pump_mode, sim.spec.mixing_mode
        ),
        font_size_legend=8,
        **PLOT_KWARGS,
    )
    sim.plot.mode_energies_vs_time(
        y_lower_limit=1e-20 * u.pJ,
        y_upper_limit=1e4 * u.pJ,
        average_over=5 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(
            s, q, mode, sim.spec.pump_mode, sim.spec.mixing_mode
        ),
        font_size_legend=8,
        **PLOT_KWARGS,
    )

    line_kwargs = [
        {"linestyle": "-", "color": "#1b9e77"},
        {"linestyle": "-", "color": "#d95f02"},
        {"linestyle": "-", "color": "#7570b3"},
        {"linestyle": "--", "color": "#1b9e77"},
        {"linestyle": "--", "color": "#d95f02"},
        {"linestyle": "--", "color": "#7570b3"},
        {"linestyle": "-", "color": "black"},
        {"linestyle": "--", "color": "black"},
    ]
    line_labels = [
        "re decay",
        "re pump",
        "re total polarization",
        "im decay",
        "im pump",
        "im total polarization",
        "re combined",
        "im combined",
    ]

    skip = 100
    for q, mode in enumerate(spec.modes):
        times = sim.times[::skip]
        amps = sim.mode_amplitudes_vs_time[::skip]

        decays = np.empty_like(times, dtype=np.complex128)
        pumps = np.empty_like(times, dtype=np.complex128)
        pols = np.empty_like(times, dtype=np.complex128)

        for idx, (time, amp) in enumerate(zip(times, amps)):
            decay, pump, pol = sim.extract_derivatives(amp, time)

            decays[idx] = decay[q] / amp[q]
            pumps[idx] = pump[q] / amp[q]
            pols[idx] = np.einsum("qrst->q", pol)[q] / amp[q]

        si.vis.xy_plot(
            f"{tag}__derivative_terms_mode_{mode.label}",
            times,
            np.real(decays),
            np.real(pumps),
            np.real(pols),
            np.imag(decays),
            np.imag(pumps),
            np.imag(pols),
            np.real(decays + pumps + pols),
            np.imag(decays + pumps + pols),
            title=rf"${mode.label}$ Relative Derivative Terms",
            x_label="Time $t$",
            x_unit="nsec",
            y_log_axis=True,
            y_upper_limit=10,
            sym_log_linear_threshold=1e-6,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            legend_on_right=True,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    pump_powers = [2 * u.mW, 10 * u.mW, 40 * u.mW, 100 * u.mW]
    mixing_powers = [1 * u.mW]
    ignore_self_interactions = [False]
    ignore_tripletss = [False]
    ignore_doubletss = [False]

    for (
        pp,
        mp,
        ignore_self_interaction,
        ignore_triplets,
        ignore_doublets,
    ) in itertools.product(
        pump_powers,
        mixing_powers,
        ignore_self_interactions,
        ignore_tripletss,
        ignore_doubletss,
    ):
        run(
            pump_power=pp,
            mixing_power=mp,
            ignore_self_interaction=ignore_self_interaction,
            ignore_triplets=ignore_triplets,
            ignore_doublets=ignore_doublets,
        )
