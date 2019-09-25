import collections
import functools
import itertools
import logging
import operator
from pathlib import Path

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman, analysis
from modulation.resonators import microspheres

import matplotlib.pyplot as plt

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(img_format="png", fig_dpi_scale=6)

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

LINESTYLES = ["-", "-.", "--", ":"]

ORDER_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]
ORDERS = [
    "pump|+0",
    "pump|+1",
    "pump|-1",
    "mixing|+0",
    "mixing|-1",
    "mixing|+1",
    "pump|-2",
    "pump|near",
]
ORDER_TO_COLOR = dict(zip(ORDERS, ORDER_COLORS))


def mode_kwargs(idx, mode):
    kwargs = {}

    kwargs["color"] = ORDER_TO_COLOR.get(mode.label, "black")

    # if "+0" in mode.label:
    #     kwargs["color"] = "black"

    if "mixing" in mode.label:
        kwargs["linestyle"] = "--"

    return kwargs


def mode_energy_and_modulation_efficiency_plots_vs_attribute(
    path,
    attr,
    x_unit=None,
    x_label=None,
    x_log=True,
    per_attrs=None,
    target_mode_order="mixing|+1",
):
    if x_label is None:
        x_label = attr.replace("_", " ").title()

    if per_attrs is None:
        per_attrs = []

    get_attr_from_sim = lambda s: getattr(s.spec, attr)

    ps = analysis.ParameterScan.from_file(path)

    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]
    modulated_mode_index = s.modes.index(s.order_to_mode[target_mode_order])

    per_attr_sets = [ps.parameter_set(attr) for attr in per_attrs]
    for per_attr_values in tqdm(list(itertools.product(*per_attr_sets))):
        per_attr_key_value = dict(zip(per_attrs, per_attr_values))
        postfix = "_".join(f"{k}={v:.3e}" for k, v in per_attr_key_value.items())
        sims = sorted(ps.select(**per_attr_key_value), key=get_attr_from_sim)

        scan_variable = np.array([get_attr_from_sim(sim) for sim in sims])
        means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
        powers = [sim.mode_output_powers(sim.lookback.mean) for sim in sims]

        energy_lines = []
        power_lines = []
        line_kwargs = []
        line_labels = []
        for idx, mode in zip(idxs, modes):
            energy = np.array([energies[idx] for energies in means])
            power = np.array([pow[idx] for pow in powers])

            energy_lines.append(energy)
            power_lines.append(power)
            line_kwargs.append(mode_kwargs(idx, mode))
            line_labels.append(fr"${mode.tex}$")

        si.vis.xy_plot(
            f"mode_energies__{postfix}".replace("|", "_"),
            scan_variable,
            *energy_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=x_label,
            y_label="Steady-State Mode Energy",
            x_unit=x_unit,
            y_unit="pJ",
            x_log_axis=x_log,
            y_log_axis=True,
            y_lower_limit=1e-10 * u.pJ,
            y_upper_limit=1e4 * u.pJ,
            legend_on_right=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )

        getter = lambda s: s.mode_output_powers(s.lookback.mean)[modulated_mode_index]
        efficiences = [getter(s) / s.spec.launched_mixing_power for s in sims]

        si.vis.xy_plot(
            f"modulation_efficiency__{postfix}".replace("|", "_"),
            scan_variable,
            efficiences,
            title=rf"Modulation Efficiency",
            x_label=x_label,
            y_label="Modulation Efficiency",
            x_unit=x_unit,
            x_log_axis=x_log,
            y_log_axis=True,
            y_lower_limit=1e-10,
            y_upper_limit=1,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


def mode_energy_and_power_differential_against_zero_mixing_plots_vs_attribute(
    path, attr, x_unit=None, x_label=None, x_log=True, per_attrs=None
):
    if x_label is None:
        x_label = attr.replace("_", " ").title()

    if per_attrs is None:
        per_attrs = []

    get_attr_from_sim = lambda s: getattr(s.spec, attr)

    ps = analysis.ParameterScan.from_file(path)

    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]

    per_attr_sets = [ps.parameter_set(attr) for attr in per_attrs]
    for per_attr_values in tqdm(list(itertools.product(*per_attr_sets))):
        per_attr_key_value = dict(zip(per_attrs, per_attr_values))
        postfix = "_".join(f"{k}={v:.3e}" for k, v in per_attr_key_value.items())
        sims = sorted(ps.select(**per_attr_key_value), key=get_attr_from_sim)
        no_mixing_sims = sorted(
            ps.select(**{**per_attr_key_value, **dict(launched_mixing_power=0)}),
            key=get_attr_from_sim,
        )

        print(len(sims))
        print(len(no_mixing_sims))
        assert len(sims) == len(no_mixing_sims)

        scan_variable = np.array([get_attr_from_sim(sim) for sim in sims])

        energies = [sim.mode_energies(sim.lookback.mean) for sim in sims]
        powers = [sim.mode_output_powers(sim.lookback.mean) for sim in sims]

        no_mixing_means = [
            sim.mode_energies(sim.lookback.mean) for sim in no_mixing_sims
        ]
        no_mixing_powers = [
            sim.mode_output_powers(sim.lookback.mean) for sim in no_mixing_sims
        ]

        energy_lines = []
        power_lines = []
        energy_diff_lines = []
        power_diff_lines = []
        line_kwargs = []
        line_labels = []
        for idx, mode in zip(idxs, modes):
            energy = np.array([energy[idx] for energy in energies])
            power = np.array([pow[idx] for pow in powers])

            energy_diff = np.array(
                [
                    energy[idx] - no_mixing_energy[idx]
                    for energy, no_mixing_energy in zip(energies, no_mixing_means)
                ]
            )
            power_diff = np.array(
                [
                    pow[idx] - no_mixing_pow[idx]
                    for pow, no_mixing_pow in zip(powers, no_mixing_powers)
                ]
            )

            energy_lines.append(energy)
            power_lines.append(power)
            energy_diff_lines.append(energy_diff)
            power_diff_lines.append(power_diff)
            line_kwargs.append(mode_kwargs(idx, mode))
            line_labels.append(fr"${mode.tex}$")

        si.vis.xy_plot(
            f"mode_energies__{postfix}",
            scan_variable,
            *energy_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=x_label,
            y_label="Steady-State Mode Energy",
            x_unit=x_unit,
            y_unit="pJ",
            x_log_axis=x_log,
            y_log_axis=True,
            y_lower_limit=1e-10 * u.pJ,
            y_upper_limit=1e4 * u.pJ,
            legend_on_right=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"mode_powers__{postfix}",
            scan_variable,
            *power_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=x_label,
            y_label="Steady-State Mode Output Power",
            x_unit=x_unit,
            y_unit="uW",
            x_log_axis=x_log,
            y_log_axis=True,
            y_upper_limit=1 * u.W,
            y_lower_limit=1e-1 * u.pW,
            legend_on_right=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            f"mode_energy_diff__{postfix}",
            scan_variable,
            *energy_diff_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=x_label,
            y_label="Steady-State Mode Energy",
            x_unit=x_unit,
            y_unit="pJ",
            x_log_axis=x_log,
            # y_log_axis=True,
            sym_log_linear_threshold=1e-3,
            legend_on_right=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"mode_power_diff__{postfix}",
            scan_variable,
            *power_diff_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=x_label,
            y_label="Steady-State Mode Output Power",
            x_unit=x_unit,
            y_unit="uW",
            x_log_axis=x_log,
            # y_log_axis=True,
            sym_log_linear_threshold=1e-3,
            legend_on_right=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


def modulation_efficiency_vs_attribute_by_mixing_power(
    path, attr, x_unit=None, x_label=None, modulated_mode_order="mixing|+1", x_log=True
):
    if x_label is None:
        x_label = attr.replace("_", " ").title()

    get_attr_from_sim = lambda s: getattr(s.spec, attr)

    ps = analysis.ParameterScan.from_file(path)

    without_mixing = sorted(ps.select(launched_mixing_power=0), key=get_attr_from_sim)
    s = without_mixing[0].spec

    nonzero_mixing_powers = sorted(ps.parameter_set("launched_mixing_power") - {0})

    modulated_mode_index = s.modes.index(s.order_to_mode[modulated_mode_order])
    launched_mixing_to_xy = {}
    for launched_mixing_power in nonzero_mixing_powers:
        postfix = f""
        sims = sorted(
            ps.select(launched_mixing_power=launched_mixing_power),
            key=get_attr_from_sim,
        )

        scan_variable = np.array([get_attr_from_sim(sim) for sim in sims])

        getter = lambda s: s.mode_output_powers(s.lookback.mean)[modulated_mode_index]
        efficiency = np.empty_like(scan_variable)
        for idx, (no_mixing_sim, with_mixing_sim) in enumerate(
            zip(without_mixing, sims)
        ):
            efficiency[idx] = (
                getter(with_mixing_sim) - getter(no_mixing_sim)
            ) / launched_mixing_power

        launched_mixing_to_xy[launched_mixing_power] = (scan_variable, efficiency)

    xx = [x for x, y in launched_mixing_to_xy.values()]
    yy = [y for x, y in launched_mixing_to_xy.values()]

    si.vis.xxyy_plot(
        f"modulation_efficiency__{postfix}",
        xx,
        yy,
        line_labels=[
            fr'$P_{{\mathrm{{mixing}}}} = {mixing_power / u.uW:.1f} \, \mathrm{{\mu W}}$'
            for mixing_power in nonzero_mixing_powers
        ],
        title=rf"Modulation Efficiency",
        x_label=x_label,
        y_label="Modulation Efficiency",
        x_unit=x_unit,
        x_log_axis=x_log,
        y_log_axis=True,
        font_size_legend=8,
        y_lower_limit=1e-10,
        y_upper_limit=1,
        target_dir=OUT_DIR / path.stem,
        **PLOT_KWARGS,
    )


def mode_energy_2d(path, mode="mixing|+1"):
    ps = analysis.ParameterScan.from_file(path)

    launched_pump = np.array(sorted(ps.parameter_set("launched_pump_power")))
    launched_mixing = np.array(sorted(ps.parameter_set("launched_mixing_power")))

    x_mesh, y_mesh = np.meshgrid(launched_pump, launched_mixing, indexing="ij")

    sims = {
        (sim.spec.launched_pump_power, sim.spec.launched_mixing_power): sim
        for sim in ps
    }

    get_energy = lambda sim: sim.mode_energies(sim.lookback.mean)[
        sim.spec.modes.index(sim.spec.order_to_mode[mode])
    ]
    get_efficiency = (
        lambda sim: sim.mode_output_powers(sim.lookback.mean)[
            sim.spec.modes.index(sim.spec.order_to_mode[mode])
        ]
        / sim.spec.launched_mixing_power
    )

    energy_mesh = np.empty_like(x_mesh)
    efficiency_mesh = np.empty_like(x_mesh)
    for i, x in enumerate(launched_pump):
        for j, y in enumerate(launched_mixing):
            energy_mesh[i, j] = get_energy(sims[(x, y)])
            efficiency_mesh[i, j] = get_efficiency(sims[(x, y)])

    si.vis.xyz_plot(
        f"mode_energy__{mode}",
        x_mesh,
        y_mesh,
        energy_mesh,
        x_label="Launched Pump Power",
        x_unit="mW",
        y_label="Launched Mixing Power",
        y_unit="mW",
        z_unit="pJ",
        x_log_axis=True,
        y_log_axis=True,
        z_log_axis=True,
        z_lower_limit=1e-6 * u.pJ,
        # z_upper_limit=1e-2 * u.pJ,
        target_dir=OUT_DIR / path.stem,
        **PLOT_KWARGS,
    )

    si.vis.xyz_plot(
        f"modulation_efficiency__{mode}",
        x_mesh,
        y_mesh,
        efficiency_mesh,
        x_label="Launched Pump Power",
        x_unit="mW",
        y_label="Launched Mixing Power",
        y_unit="mW",
        x_log_axis=True,
        y_log_axis=True,
        z_log_axis=True,
        z_lower_limit=1e-6,
        z_upper_limit=1e-2,
        contours=[1e-5, 1e-4, 1e-3],
        contour_kwargs={"colors": "white"},
        contour_label_kwargs={"fmt": "%.1e", "colors": "white", "inline_spacing": 50},
        target_dir=OUT_DIR / path.stem,
        **PLOT_KWARGS,
    )


def derivatives(path, attr, x_unit=None, x_label=None, x_log=True, per_attrs=None):
    if x_label is None:
        x_label = attr.replace("_", " ").title()

    if per_attrs is None:
        per_attrs = []

    get_attr_from_sim = lambda s: getattr(s.spec, attr)

    ps = analysis.ParameterScan.from_file(path)

    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]

    sum_factors = ps[0]._calculate_polarization_sum_factors()

    per_attr_sets = [ps.parameter_set(attr) for attr in per_attrs]
    for per_attr_values in tqdm(list(itertools.product(*per_attr_sets))):
        per_attr_key_value = dict(zip(per_attrs, per_attr_values))
        postfix = "_".join(f"{k}={v:.3e}" for k, v in per_attr_key_value.items())
        sims = sorted(ps.select(**per_attr_key_value), key=get_attr_from_sim)

        scan_variable = np.array([get_attr_from_sim(sim) for sim in sims])
        # means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
        # powers = [sim.mode_output_powers(sim.lookback.mean) for sim in sims]

        for mode_idx, mode in zip(idxs, modes):
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
            amps = []
            decays = []
            rel_decays = []
            pumps = []
            rel_pumps = []
            total_pols = []
            rel_total_pols = []

            for sim_idx, sim in enumerate(sims):
                sim.polarization_sum_factors = sum_factors
                decay, pump, pol = sim.extract_derivatives(
                    sim.mode_amplitudes, sim.current_time
                )
                amps.append(sim.mode_amplitudes[mode_idx])
                decays.append(decay[mode_idx])
                rel_decays.append(decay[mode_idx] / sim.mode_amplitudes[mode_idx])
                pumps.append(pump[mode_idx])
                rel_pumps.append(pump[mode_idx] / sim.mode_amplitudes[mode_idx])
                total_pols.append(np.einsum("qrst->q", pol)[mode_idx])
                rel_total_pols.append(
                    np.einsum("qrst->q", pol)[mode_idx] / sim.mode_amplitudes[mode_idx]
                )

            amps = np.array(amps)
            decays = np.array(decays)
            pumps = np.array(pumps)
            total_pols = np.array(total_pols)
            rel_decays = np.array(rel_decays)
            rel_pumps = np.array(rel_pumps)
            rel_total_pols = np.array(rel_total_pols)

            si.vis.xy_plot(
                f"relative_derivatives__mode_{mode.label}__{postfix}",
                scan_variable,
                np.real(rel_decays),
                np.real(rel_pumps),
                np.real(rel_total_pols),
                np.imag(rel_decays),
                np.imag(rel_pumps),
                np.imag(rel_total_pols),
                np.real(rel_decays + rel_pumps + rel_total_pols),
                np.imag(rel_decays + rel_pumps + rel_total_pols),
                line_kwargs=line_kwargs,
                line_labels=line_labels,
                legend_on_right=True,
                sym_log_linear_threshold=1e-6,
                x_label=x_label,
                x_unit=x_unit,
                x_log_axis=x_log,
                y_log_axis=True,
                target_dir=OUT_DIR / path.stem,
                **PLOT_KWARGS,
            )
            si.vis.xy_plot(
                f"derivatives__mode_{mode.label}__{postfix}",
                scan_variable,
                np.real(amps),
                np.imag(amps),
                np.real(decays),
                np.real(pumps),
                np.real(total_pols),
                np.imag(decays),
                np.imag(pumps),
                np.imag(total_pols),
                np.real(decays + pumps + total_pols),
                np.imag(decays + pumps + total_pols),
                line_kwargs=[
                    {"linestyle": "-", "color": "grey"},
                    {"linestyle": "--", "color": "grey"},
                ]
                + line_kwargs,
                line_labels=["re amplitude", "im amplitude"] + line_labels,
                legend_on_right=True,
                # sym_log_linear_threshold = 1e-6,
                x_label=x_label,
                x_unit=x_unit,
                x_log_axis=x_log,
                y_log_axis=True,
                target_dir=OUT_DIR / path.stem,
                **PLOT_KWARGS,
            )


if __name__ == "__main__":
    with LOGMAN as logger:
        BASE = Path(__file__).parent

        mode_energy_and_modulation_efficiency_plots_vs_attribute(
            BASE / "sep-scan.sims",
            attr="fiber_separation",
            x_unit="um",
            x_label="Fiber Separation",
            x_log=False,
            per_attrs=["launched_pump_power", "launched_mixing_wavelength"],
        )

        mode_energy_and_modulation_efficiency_plots_vs_attribute(
            BASE / "test-near-pump.sims",
            attr="pump|near_mode_detuning",
            x_unit="MHz",
            x_label="Extra Near-Pump Mode Detuning from Pump",
            x_log=True,
            per_attrs=["pump|-2_mode_detuning", "launched_pump_power"],
        )
