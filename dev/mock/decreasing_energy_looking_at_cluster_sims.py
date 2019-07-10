import itertools
import logging
from pathlib import Path

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
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


def make_animations(ps, scan):
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
                postfix=f"_{scan}",
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


def make_final_amp_plot_for_mixing(ps):
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


def make_phase_plot_for_mixing(ps, scan):
    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    launched_powers = np.array(sorted(ps.parameter_set("launched_pump_power")))

    phi_p = np.angle(amplitudes[:, 1])
    phi_st = np.angle(amplitudes[:, 0] * amplitudes[:, 3])
    phi_m = np.angle(amplitudes[:, 2])

    # phi_p_interp = interp.interp1d(launched_powers, phi_p)
    # phi_st_interp = interp.interp1d(launched_powers, phi_st)
    # phi_m_interp = interp.interp1d(launched_powers, phi_m)
    # interps = {"p": phi_p_interp, "st": phi_st_interp, "m": phi_m_interp}
    #
    # intersections = {}
    # for (a, a_interp), (b, b_interp) in itertools.combinations(interps.items(), 2):
    #     print(a, b)
    #     intersections[a, b] = opt.root_scalar(
    #         lambda x: a_interp(x) - b_interp(x), bracket=(10 * u.mW, 100 * u.mW)
    #     ).root
    #
    # for k, v in intersections.items():
    #     print(f"{'-'.join(k):<5} -> {v / u.mW:.3f} mW")

    sim = ps[0]
    pol_sum_factors = sim._calculate_polarization_sum_factors()
    print(pol_sum_factors.shape)

    # MSPT -> 2013 and MTPS -> 2310 are the two terms that should enter
    MSPT = pol_sum_factors[2, 0, 1, 3]
    phi_MSPT = np.angle(MSPT)
    MTPS = pol_sum_factors[2, 3, 1, 0]
    phi_MTPS = np.angle(MTPS)
    print("2013 MSPT", MSPT, phi_MSPT / u.pi)
    print("2310 MTPS", MTPS, phi_MTPS / u.pi)
    print(
        "2222 MMMM", pol_sum_factors[2, 2, 2, 2], np.angle(pol_sum_factors[2, 2, 2, 2])
    )

    si.vis.xy_plot(
        f"{scan}__phases",
        launched_powers,
        phi_m,
        phi_p,
        phi_st,
        np.ones_like(phi_p) * phi_MSPT,
        np.ones_like(phi_p) * phi_MTPS,
        phi_MSPT + phi_st - phi_p,
        phi_MTPS + phi_st - phi_p,
        line_labels=[
            r"$\phi_{M}$",
            r"$\phi_{P}$",
            r"$\phi_{ST}$",
            r"$\phi_{G_{MSPT}}$",
            r"$\phi_{G_{MTPS}}$",
            r"$\phi_{G_{MSPT}} + \phi_{ST} - \phi_{P}$",
            r"$\phi_{G_{MTPS}} + \phi_{ST} - \phi_{P}$",
        ],
        line_kwargs=[
            {"color": "black"},
            {"color": "red"},
            {"color": "blue"},
            {"color": "purple"},
            {"color": "orange"},
            {"color": "purple", "linestyle": "--"},
            {"color": "orange", "linestyle": "--"},
        ],
        x_unit="mW",
        x_label="Launched Pump Power",
        x_log_axis=True,
        # x_lower_limit=1 * u.mW,
        # x_upper_limit=400 * u.mW,
        y_unit="rad",
        y_label="Phase",
        legend_on_right=True,
        **PLOT_KWARGS,
    )


def make_pump_and_pol_for_mixing(ps, scan):
    mixing_idx = 2

    sim = ps[0]
    pol_sum_factors = sim._calculate_polarization_sum_factors()

    # MSPT -> 2013 and MTPS -> 2310 are the two terms that should enter
    MSPT = pol_sum_factors[2, 0, 1, 3]
    MTPS = pol_sum_factors[2, 3, 1, 0]
    MMMM = pol_sum_factors[2, 2, 2, 2]
    MMSS = pol_sum_factors[2, 2, 0, 0]
    MMPP = pol_sum_factors[2, 2, 1, 1]
    MMTT = pol_sum_factors[2, 2, 3, 3]

    launched_powers = np.array(sorted(ps.parameter_set("launched_pump_power")))
    sims = sorted(ps, key=lambda s: s.spec.launched_pump_power)

    pump_term = np.array(
        [
            sim.pump_prefactor[mixing_idx]
            * 0.5
            * np.sqrt(sim.spec.launched_mixing_power)
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )

    mspt_term = np.array(
        [
            MSPT
            * (
                sim.mode_amplitudes[0]
                * np.conj(sim.mode_amplitudes[1])
                * sim.mode_amplitudes[3]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )
    mtps_term = np.array(
        [
            MTPS
            * (
                sim.mode_amplitudes[0]
                * np.conj(sim.mode_amplitudes[1])
                * sim.mode_amplitudes[3]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )
    mmmm_term = np.array(
        [
            MMMM
            * (
                sim.mode_amplitudes[2]
                * np.conj(sim.mode_amplitudes[2])
                * sim.mode_amplitudes[2]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )
    mmss_term = np.array(
        [
            MMSS
            * (
                sim.mode_amplitudes[2]
                * np.conj(sim.mode_amplitudes[0])
                * sim.mode_amplitudes[0]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )
    mmpp_term = np.array(
        [
            MMPP
            * (
                sim.mode_amplitudes[2]
                * np.conj(sim.mode_amplitudes[1])
                * sim.mode_amplitudes[1]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )
    mmtt_term = np.array(
        [
            MMTT
            * (
                sim.mode_amplitudes[2]
                * np.conj(sim.mode_amplitudes[3])
                * sim.mode_amplitudes[3]
            )
            / sim.mode_amplitude_decay_rates[mixing_idx]
            for sim in sims
        ]
    )

    actual = [sim.mode_amplitudes[2] for sim in sims]

    si.vis.xy_plot(
        f"{scan}__pump_and_pol",
        launched_powers,
        np.real(pump_term),
        np.abs(pump_term + mtps_term + mmss_term + mmpp_term),
        np.abs(actual),
        np.real(mtps_term),
        np.imag(mtps_term),
        np.real(mmss_term),
        np.imag(mmss_term),
        np.real(mmpp_term),
        np.imag(mmpp_term),
        np.real(mmtt_term),
        np.imag(mmtt_term),
        line_kwargs=[
            {"color": "black", "linestyle": "-"},
            {"color": "purple", "linestyle": "--"},
            {"color": "purple", "linestyle": "-"},
            {"color": "blue", "linestyle": "-"},
            {"color": "blue", "linestyle": "--"},
            {"color": "red", "linestyle": "-"},
            {"color": "red", "linestyle": "--"},
            {"color": "green", "linestyle": "-"},
            {"color": "green", "linestyle": "--"},
            {"color": "pink", "linestyle": "-"},
            {"color": "pink", "linestyle": "--"},
        ],
        line_labels=[
            "re pump",
            "abs sum",
            "abs actual",
            "re MTPS",
            "im MTPS",
            "re MMSS",
            "im MMSS",
            "re MMPP",
            "im MMPP",
            "im MMTT",
            "im MMTT",
        ],
        x_unit="mW",
        x_label="Launched Pump Power",
        x_log_axis=True,
        # x_lower_limit=1 * u.mW,
        # x_upper_limit=400 * u.mW,
        y_pad=0.2,
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    modes = ps[0].spec.modes

    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    r = np.abs(amplitudes)
    theta = np.angle(amplitudes)

    with si.vis.FigureManager(
        f"{scan}__complex_amplitudes_with_terms", **PLOT_KWARGS
    ) as figman:
        fig = figman.fig
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction("counterclockwise")

        ax.scatter(np.zeros_like(pump_term), (pump_term), color="purple", label="Pump")

        ax.plot(
            np.angle(mtps_term),
            (np.abs(mtps_term)),
            label="MTPS",
            linewidth=1,
            color="blue",
            linestyle=":",
        )

        ax.plot(
            np.angle(mmss_term),
            (np.abs(mmss_term)),
            label="MMSS",
            linewidth=1,
            color="red",
            linestyle=":",
        )

        ax.plot(
            np.angle(mmpp_term),
            (np.abs(mmpp_term)),
            label="MMPP",
            linewidth=1,
            color="green",
            linestyle=":",
        )

        ax.plot(
            np.angle(mmtt_term),
            (np.abs(mmtt_term)),
            label="MMTT",
            linewidth=1,
            color="pink",
            linestyle=":",
        )

        ax.plot(
            np.angle(pump_term + mtps_term + mmss_term + mmpp_term),
            (np.abs(pump_term + mtps_term + mmss_term + mmpp_term)),
            label="Sum",
            linewidth=1,
            color="black",
            linestyle="--",
        )

        for mode_idx, mode in enumerate(modes):
            if mode_idx in (0, 1, 3):  # skip stokes and target, which rotate
                continue

            ax.plot(
                theta[:, mode_idx],
                r[:, mode_idx],
                color=MODE_COLORS[mode_idx],
                label=mode.label,
                linewidth=1,
            )

            ax.scatter(
                theta[0, mode_idx],
                r[0, mode_idx],
                color=MODE_COLORS[mode_idx],
                alpha=0.5,
            )

        # ax.set_ylim(7.25, 8.75)

        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))


def make_pump_and_pol_for_target(ps, scan):
    target_idx = 3

    sim = ps[0]
    pol_sum_factors = sim._calculate_polarization_sum_factors()

    # MSPT -> 2013 and MTPS -> 2310 are the two terms that should enter
    TPSM = pol_sum_factors[3, 1, 0, 2]
    TMSP = pol_sum_factors[3, 3, 0, 1]

    launched_powers = np.array(sorted(ps.parameter_set("launched_pump_power")))
    sims = sorted(ps, key=lambda s: s.spec.launched_pump_power)

    def term(r, s, t):
        return np.array(
            [
                pol_sum_factors[target_idx, r, s, t]
                * (
                    sim.mode_amplitudes[r]
                    * np.conj(sim.mode_amplitudes[s])
                    * sim.mode_amplitudes[t]
                )
                / sim.mode_amplitude_decay_rates[target_idx]
                for sim in sims
            ]
        )

    tpsm_term = term(1, 0, 2)
    tmsp_term = term(2, 0, 1)

    actual = [sim.mode_amplitudes[target_idx] for sim in sims]

    si.vis.xy_plot(
        f"{scan}__pump_and_pol__for_target",
        launched_powers,
        np.abs(tpsm_term + tmsp_term),
        np.abs(actual),
        np.real(tpsm_term),
        np.imag(tpsm_term),
        np.real(tmsp_term),
        np.imag(tmsp_term),
        line_kwargs=[
            {"color": "black", "linestyle": "-"},
            {"color": "purple", "linestyle": "--"},
            {"color": "blue", "linestyle": "-"},
            {"color": "blue", "linestyle": "--"},
            {"color": "red", "linestyle": "-"},
            {"color": "red", "linestyle": "--"},
        ],
        line_labels=[
            "abs sum",
            "abs actual",
            "re TPSM",
            "im TPSM",
            "re TMSP",
            "im TMSP",
        ],
        x_unit="mW",
        x_label="Launched Pump Power",
        x_log_axis=True,
        # x_lower_limit=1 * u.mW,
        # x_upper_limit=400 * u.mW,
        y_pad=0.2,
        legend_on_right=True,
        **PLOT_KWARGS,
    )

    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    modes = ps[0].spec.modes

    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    r = np.abs(amplitudes)
    theta = np.angle(amplitudes)

    with si.vis.FigureManager(
        f"{scan}__complex_amplitudes_with_terms__for_target", **PLOT_KWARGS
    ) as figman:
        fig = figman.fig
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection="polar")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction("counterclockwise")

        ax.plot(
            np.angle(tpsm_term),
            (np.abs(tpsm_term)),
            label="TPSM",
            linewidth=1,
            color="blue",
            linestyle=":",
        )

        ax.plot(
            np.angle(tmsp_term),
            (np.abs(tmsp_term)),
            label="TMSP",
            linewidth=1,
            color="red",
            linestyle=":",
        )

        ax.plot(
            np.angle(tpsm_term + tmsp_term),
            (np.abs(tpsm_term + tmsp_term)),
            label="Sum",
            linewidth=1,
            color="black",
            linestyle="--",
        )

        for mode_idx, mode in enumerate(modes):
            if mode_idx in (0, 1, 3):  # skip stokes and target, which rotate
                continue

            ax.plot(
                theta[:, mode_idx],
                r[:, mode_idx],
                color=MODE_COLORS[mode_idx],
                label=mode.label,
                linewidth=1,
            )

            ax.scatter(
                theta[0, mode_idx],
                r[0, mode_idx],
                color=MODE_COLORS[mode_idx],
                alpha=0.5,
            )

        # ax.set_ylim(7.25, 8.75)

        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))


def plot_final_amplitudes(ps, scan):
    amplitudes = np.empty((len(ps), 4), dtype=np.complex128)
    modes = ps[0].spec.modes

    x = np.array(sorted(s.spec.launched_pump_power for s in ps))
    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

    y = amplitudes[:, 2] * np.conj(amplitudes[:, 0]) * amplitudes[:, 1]
    si.vis.xy_plot(
        f"{scan}__MSP_product",
        x,
        np.abs(y),
        # np.real(y),
        # np.imag(y),
        # line_labels=["abs", "re", "im"],
        # y_log_axis=True,
        x_log_axis=True,
        **PLOT_KWARGS,
    )


def foobar(ps, scan):
    print(ps)

    with_mixing = [sim for sim in ps if sim.spec.launched_mixing_power != 0]
    without_mixing = [sim for sim in ps if sim.spec.launched_mixing_power == 0]

    print(len(with_mixing))
    print(len(without_mixing))

    x = np.array(sorted(s.spec.launched_pump_power for s in with_mixing))

    amplitudes_with_mixing = np.empty((len(x), 4), dtype=np.complex128)
    for idx, sim in enumerate(
        sorted(with_mixing, key=lambda sim: sim.spec.launched_pump_power)
    ):
        amplitudes_with_mixing[idx] = sim.mode_amplitudes

    amplitudes_without_mixing = np.empty_like(amplitudes_with_mixing)
    for idx, sim in enumerate(
        sorted(without_mixing, key=lambda sim: sim.spec.launched_pump_power)
    ):
        amplitudes_without_mixing[idx] = sim.mode_amplitudes

    amplitudes_with_mixing = np.abs(amplitudes_with_mixing)
    energies_with_mixing = np.abs(amplitudes_with_mixing) ** 2
    amplitudes_without_mixing = np.abs(amplitudes_without_mixing)
    energies_without_mixing = np.abs(amplitudes_without_mixing) ** 2

    STOKES = 0
    PUMP = 1
    MIXING = 2
    TARGET = 3

    si.vis.xy_plot(
        "foobar_1",
        x,
        energies_with_mixing[:, PUMP] - energies_without_mixing[:, PUMP],
        energies_with_mixing[:, STOKES] - energies_without_mixing[:, STOKES],
        energies_with_mixing[:, MIXING] - energies_with_mixing[0, MIXING],
        line_labels=[
            "Pump with mixing - Pump no mixing",
            "Stokes with mixing - Stokes no mixing",
            "Mixing with pump - Mixing no pump",
        ],
        x_log_axis=True,
        x_label="Launched Pump Power",
        x_unit="mW",
        legend_on_right=True,
        **PLOT_KWARGS,
    )


def wizbang(ps, scan):
    x = np.array(sorted(ps.parameter_set("launched_pump_power")))

    STOKES = 0
    PUMP = 1
    MIXING = 2
    TARGET = 3

    sim = ps[0]
    pol_sum_factors = sim._calculate_polarization_sum_factors()

    amplitudes = np.empty((len(x), 4), dtype=np.complex128)
    mixing_amp = np.empty(len(x), dtype=np.complex128)
    target_amp = np.empty(len(x), dtype=np.complex128)
    for idx, sim in enumerate(sorted(ps, key=lambda sim: sim.spec.launched_pump_power)):
        amplitudes[idx] = sim.mode_amplitudes

        num = sim.pump_prefactor[MIXING] * 0.5 * np.sqrt(sim.spec.launched_mixing_power)
        den_var = (
            pol_sum_factors[TARGET, MIXING, STOKES, PUMP]
            * pol_sum_factors[MIXING, TARGET, PUMP, STOKES]  # order matters!
            * (np.abs(sim.mode_amplitudes[STOKES]) ** 2)
            * (np.abs(sim.mode_amplitudes[PUMP]) ** 2)
            / (sim.mode_amplitude_decay_rates[TARGET])
        )
        den_fixed = sim.mode_amplitude_decay_rates[MIXING]
        mixing_amp[idx] = num / (den_var + den_fixed)

        target_amp[idx] = (
            -pol_sum_factors[TARGET, MIXING, STOKES, PUMP]
            * sim.mode_amplitudes[MIXING]
            * sim.mode_amplitudes[STOKES]
            * sim.mode_amplitudes[PUMP]
            / sim.mode_amplitude_decay_rates[TARGET]
        )
        print(target_amp[idx])

    amplitudes = np.abs(amplitudes)
    mixing_amp = np.abs(mixing_amp)
    target_amp = np.abs(target_amp)

    # print(amplitudes[:, MIXING])
    # print(mixing_amp)

    si.vis.xy_plot(
        "wizbang",
        x,
        amplitudes[:, MIXING],
        mixing_amp,
        amplitudes[:, TARGET],
        target_amp,
        line_kwargs=[
            {"color": "blue", "linestyle": "-"},
            {"color": "green", "linestyle": "--"},
            {"color": "red", "linestyle": "-"},
            {"color": "purple", "linestyle": "--"},
        ],
        line_labels=[
            "Mixing (Simulation)",
            "Mixing (Theory)",
            "Target (Simulation)",
            "Target (Theory)",
        ],
        x_log_axis=True,
        x_label="Launched Pump Power",
        x_unit="mW",
        y_label=r"$ \left| \mathcal{E}_q \right| $",
        title="Mode Magnitude for Narrow Raman Linewidth",
        # legend_on_right=True,
        # y_pad=0.1,
        y_log_axis=True,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    # for scan in ["narrow", "very-narrow", "pump_power_scan_redux"]:
    #     ps = analysis.ParameterScan.from_file(Path.cwd() / f"{scan}.sims")
    #
    #     # make_final_amp_plot(ps)
    #     # make_phase_plot_for_mixing(ps, scan)
    #     # make_pump_and_pol_for_mixing(ps, scan)
    #     # make_pump_and_pol_for_target(ps, scan)
    #     plot_final_amplitudes(ps, scan)
    #     # make_animations(ps, scan)

    # ps = analysis.ParameterScan.from_file(Path.cwd() / "narrow_v2.sims")
    # foobar(ps, "narrow_v2")

    ps = analysis.ParameterScan.from_file(Path.cwd() / "narrow.sims")
    wizbang(ps, "narrow")
