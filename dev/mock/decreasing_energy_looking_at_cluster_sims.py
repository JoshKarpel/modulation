import itertools
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import simulacra as si
import simulacra.units as u

from modulation import raman, analysis
from modulation.resonators import mock

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)
ANIM_KWARGS = dict(target_dir=OUT_DIR, length=20, fps=30)


MODE_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]


def mode_kwargs(sim, q, mode):
    kwargs = {}

    kwargs["color"] = MODE_COLORS[q]

    return kwargs


def make_plots(sim):
    sim.plot.mode_complex_amplitudes_vs_time(
        # y_lower_limit=1e-20 * u.pJ,
        # y_upper_limit=1e4 * u.pJ,
        average_over=5 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(s, q, mode),
        font_size_legend=8,
        **PLOT_KWARGS,
    )
    sim.plot.mode_energies_vs_time(
        y_lower_limit=1e-20 * u.pJ,
        y_upper_limit=1e4 * u.pJ,
        average_over=5 * u.nsec,
        y_log_pad=1,
        mode_kwargs=lambda s, q, mode: mode_kwargs(s, q, mode),
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
    for q, mode in enumerate(sim.spec.modes):
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
            f"{sim.name}__derivative_terms_mode_{mode.label}",
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
            y_lower_limit=-10,
            sym_log_linear_threshold=1e-6,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            legend_on_right=True,
            **PLOT_KWARGS,
        )


def make_animations(ps):
    available = np.array(sorted(ps.parameter_set("launched_pump_power")))

    sims_by_launched_pump_power = {s.spec.launched_pump_power: s for s in ps}
    # for k, v in sims_by_launched_pump_power.items():
    #     print(k, v)

    target_pump_powers = np.array([2, 10, 30, 100]) * u.mW
    launched_pump_powers = [
        si.utils.find_nearest_entry(available, tpp).value for tpp in target_pump_powers
    ]
    sims = [sims_by_launched_pump_power[lpp] for lpp in launched_pump_powers]

    for sim in sims:
        print()
        print(sim.info())
        print(sim.spec.mode_info())

        spec = sim.spec
        spec.lookback = raman.Lookback(lookback_time=sim.spec.lookback.lookback_time)
        spec.store_mode_amplitudes_vs_time = True
        spec.animators = [
            raman.anim.SquareAnimator(
                axman=raman.anim.PolarComplexAmplitudeAxis(
                    r_log_lower_limit=5, r_log_upper_limit=12, mode_colors=MODE_COLORS
                ),
                **ANIM_KWARGS,
            )
        ]
        spec.checkpoints = False

        new_sim = spec.to_sim()

        print(new_sim.info())
        new_sim.run(progress_bar=True)
        make_plots(new_sim)


def make_final_amp_plot(ps):
    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    modes = ps[0].spec.modes

    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    r = np.log10(np.abs(amplitudes))
    theta = np.angle(amplitudes)

    with si.vis.FigureManager("complex_amplitudes", **PLOT_KWARGS) as figman:
        fig = figman.fig
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction("counterclockwise")

        for mode_idx, mode in enumerate(modes):
            if mode_idx in (0, 3):  # skip stokes and target, which rotate
                continue

            ax.plot(
                theta[:, mode_idx],
                r[:, mode_idx],
                color=MODE_COLORS[mode_idx],
                label=mode.label,
                linewidth=0.5,
            )

            ax.scatter(
                theta[0, mode_idx],
                r[0, mode_idx],
                color=MODE_COLORS[mode_idx],
                alpha=0.5,
            )

        ax.set_ylim(7.5, 10)

        ax.grid(True)
        ax.legend(loc="lower right")


def make_phase_plot(ps):
    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    launched_powers = np.array(sorted(ps.parameter_set("launched_pump_power")))

    si.vis.xy_plot(
        "phases",
        launched_powers,
        np.angle(amplitudes[:, 1]),
        np.angle(amplitudes[:, 0] * amplitudes[:, 3]),
        np.angle(amplitudes[:, 2]),
        line_labels=[r"$\phi_{P}$", r"$\phi_{ST}$", r"$\phi_{M}$"],
        x_unit="mW",
        x_label="Launched Pump Power",
        x_log_axis=True,
        y_unit="rad",
        y_label="Phase",
        y_lower_limit=-u.pi,
        y_upper_limit=+u.pi,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    ps = analysis.ParameterScan.from_file(Path.cwd() / "de_dense_pump_power_scan.sims")

    # make_animations(ps)
    # make_final_amp_plot(ps)
    make_phase_plot(ps)
