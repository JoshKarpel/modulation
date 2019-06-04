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


def mode_kwargs(idx, mode):
    kwargs = {}

    kwargs["color"] = COLORS[idx % len(COLORS)]

    if "+0" in mode.label:
        kwargs["color"] = "black"

    if "mixing" in mode.label:
        kwargs["linestyle"] = "--"

    return kwargs


def mode_energy_and_power_plots_vs_attribute(
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
            font_size_legend=6,
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
            font_size_legend=6,
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


if __name__ == "__main__":
    with LOGMAN as logger:
        BASE = Path(__file__).parent

        # names = [
        #     # "test_mock_multiorder.sims",
        #     # "mock_multiorder_meff.sims",
        #     # "just_two_stokes_orders.sims",
        #     # "mock_cascaded_srs.sims",
        #     # "mock_cascaded_srs_using_fwm.sims",
        #     "mock_launched_mixing_detuning_scan.sims",
        #     "mock_launched_mixing_wavelength_scan.sims",
        # ]
        # paths = (BASE / name for name in names)
        #
        # for func in (
        #     mode_energy_and_power_plots_vs_attribute_per_mixing_power,
        #     modulation_efficiency_vs_attribute_by_mixing_power,
        # ):
        #     # func(
        #     #     BASE / "mock_launched_mixing_detuning_scan.sims",
        #     #     attr="launched_mixing_detuning",
        #     #     x_unit="Hz",
        #     # )
        #     func(
        #         BASE / "mock_launched_mixing_wavelength_scan.sims",
        #         attr="launched_mixing_wavelength",
        #         x_unit="nm",
        #         x_log=False,
        #     )

        # mode_energy_and_power_plots_vs_attribute_per_mixing_power(
        #     BASE / "mock_cascaded_srs_final_time_scan_using_srs.sims",
        #     attr="time_final",
        #     x_unit="us",
        #     x_log=True,
        # )
        # mode_energy_and_power_plots_vs_attribute_per_mixing_power(
        #     BASE / "mock_cascaded_srs_final_time_scan_using_fwm.sims",
        #     attr="time_final",
        #     x_unit="us",
        #     x_log=True,
        # )

        # mode_energy_and_power_plots_vs_attribute(
        #     BASE / "mock_cascaded_srs_p1_detuning.sims",
        #     attr="pump|-1_mode_detuning",
        #     x_unit="Hz",
        #     x_log=True,
        #     per_attrs=["time_final", "launched_pump_power", "launched_mixing_power"],
        # )

        # broken_ladder = [
        #     "mock_cascaded_srs_detuned_ladder.sims",
        #     "mock_cascaded_srs_more_detuned.sims",
        # ]
        # for x in broken_ladder:
        #     mode_energy_and_power_plots_vs_attribute(
        #         BASE / x,
        #         attr="launched_pump_power",
        #         x_unit="mW",
        #         x_log=True,
        #         per_attrs=["time_final"],
        #     )

        mode_energy_and_power_plots_vs_attribute(
            BASE / "paper__modeff_vs_launched_pump_power.sims",
            attr="launched_pump_power",
            x_unit="mW",
            x_log=True,
            per_attrs=["launched_mixing_wavelength", "launched_mixing_power"],
        )

        mode_energy_and_power_plots_vs_attribute(
            BASE / "paper__modeff_vs_mixing_wavelength.sims",
            attr="launched_mixing_wavelength",
            x_unit="nm",
            x_log=False,
            per_attrs=["launched_pump_power", "launched_mixing_power"],
        )

        mode_energy_and_power_plots_vs_attribute(
            BASE / "paper__modeff_vs_target_detuning.sims",
            attr="mixing|+1_mode_detuning",
            x_unit="Hz",
            x_log=True,
            per_attrs=[
                "launched_mixing_wavelength",
                "launched_pump_power",
                "launched_mixing_power",
            ],
        )
