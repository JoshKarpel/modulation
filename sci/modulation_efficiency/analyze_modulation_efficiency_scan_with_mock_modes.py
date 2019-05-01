import collections
import itertools
import logging
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


def mode_energy_plot_vs_pump_power_per_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]

    for launched_mixing_power in ps.parameter_set("launched_mixing_power"):
        sims = ps.select(launched_mixing_power=launched_mixing_power)

        scan_powers = np.array([sim.spec.launched_pump_power for sim in sims])
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
            # line_kwargs.append(mode_kwargs(idx, mode, pump_mode))
            line_labels.append(fr"${mode.tex}$")

        postfix = f"mixing={launched_mixing_power / u.mW:.6f}mW"

        si.vis.xy_plot(
            f"mode_energies__{postfix}",
            scan_powers,
            *energy_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Launched Pump Power",
            y_label="Steady-State Mode Energy",
            x_unit="mW",
            y_unit="pJ",
            x_log_axis=scan_powers[0] != 0,
            y_log_axis=True,
            y_lower_limit=1e-7 * u.pJ,
            y_upper_limit=1e5 * u.pJ,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"mode_powers__{postfix}",
            scan_powers,
            *power_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Launched Pump Power",
            y_label="Steady-State Mode Output Power",
            x_unit="mW",
            y_unit="uW",
            y_lower_limit=1 * u.pW,
            y_upper_limit=1 * u.W,
            x_log_axis=scan_powers[0] != 0,
            y_log_axis=True,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


def modulation_efficiency_vs_pump_power_by_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    without_mixing = sorted(
        ps.select(launched_mixing_power=0), key=lambda s: s.spec.launched_pump_power
    )
    s = without_mixing[0].spec

    nonzero_mixing_powers = sorted(ps.parameter_set("launched_mixing_power") - {0})

    modulated_mode = s.modulated_mode
    modulated_mode_index = s.modes.index(modulated_mode)
    mixing_power_to_xy = {}
    for launched_mixing_power in nonzero_mixing_powers:
        sims = sorted(
            ps.select(launched_mixing_power=launched_mixing_power),
            key=lambda s: s.spec.launched_pump_power,
        )

        pump_powers = np.array([sim.spec.launched_pump_power for sim in sims])

        getter = lambda s: s.mode_output_powers(s.lookback.mean)[modulated_mode_index]
        efficiency = np.empty_like(pump_powers)
        for idx, (no_mixing_sim, with_mixing_sim) in enumerate(
            zip(without_mixing, sims)
        ):
            efficiency[idx] = (
                getter(with_mixing_sim) - getter(no_mixing_sim)
            ) / launched_mixing_power

        mixing_power_to_xy[launched_mixing_power] = (pump_powers, efficiency)

    xx = [x for x, y in mixing_power_to_xy.values()]
    yy = [y for x, y in mixing_power_to_xy.values()]

    si.vis.xxyy_plot(
        f"modulation_efficiency",
        xx,
        yy,
        line_labels=[
            fr'$P_{{\mathrm{{mixing}}}} = {mixing_power / u.uW:.1f} \, \mathrm{{\mu W}}$'
            for mixing_power in nonzero_mixing_powers
        ],
        title=rf"Modulation Efficiency",
        x_label="Launched Pump Power",
        y_label="Modulation Efficiency",
        x_unit="mW",
        x_log_axis=True,
        y_log_axis=True,
        font_size_legend=8,
        y_lower_limit=1e-10,
        y_upper_limit=1,
        target_dir=OUT_DIR / path.stem,
        **PLOT_KWARGS,
    )


def mode_energy_plot_vs_modulated_mode_detuning_per_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]

    for launched_mixing_power in ps.parameter_set("launched_mixing_power"):
        sims = ps.select(launched_mixing_power=launched_mixing_power)

        scan_detunings = np.array([sim.spec.modulated_mode_detuning for sim in sims])
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
            # line_kwargs.append(mode_kwargs(idx, mode, pump_mode))
            line_labels.append(fr"${mode.tex}$")

        postfix = f"mixing={launched_mixing_power / u.mW:.6f}mW"

        vlines = [s.modes[3].frequency / s.mode_total_quality_factors[3]]
        energy_hlines = [energy_lines[3][0] / 2]  # half-power point
        power_hlines = [power_lines[3][0] / 2]
        kw = [{"linestyle": "--"}]

        si.vis.xy_plot(
            f"mode_energies__{postfix}",
            scan_detunings,
            *energy_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Modulated Mode Detuning",
            y_label="Steady-State Mode Energy",
            x_unit="Hz",
            y_unit="pJ",
            x_log_axis=True,
            y_log_axis=True,
            y_lower_limit=1e-7 * u.pJ,
            y_upper_limit=1e5 * u.pJ,
            vlines=vlines,
            hlines=energy_hlines,
            vline_kwargs=kw,
            hline_kwargs=kw,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"mode_powers__{postfix}",
            scan_detunings,
            *power_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Modulated Mode Detuning",
            y_label="Steady-State Mode Output Power",
            x_unit="Hz",
            y_unit="uW",
            x_log_axis=True,
            y_log_axis=True,
            y_upper_limit=1 * u.W,
            y_lower_limit=1 * u.pW,
            vlines=vlines,
            hlines=power_hlines,
            vline_kwargs=kw,
            hline_kwargs=kw,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


def modulation_efficiency_vs_modulated_mode_detuning_by_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    without_mixing = sorted(
        ps.select(launched_mixing_power=0), key=lambda s: s.spec.launched_pump_power
    )
    s = without_mixing[0].spec

    nonzero_mixing_powers = sorted(ps.parameter_set("launched_mixing_power") - {0})

    modulated_mode = s.modulated_mode
    modulated_mode_index = s.modes.index(modulated_mode)
    detuning_to_xy = {}
    for launched_mixing_power in nonzero_mixing_powers:
        sims = sorted(
            ps.select(launched_mixing_power=launched_mixing_power),
            key=lambda s: s.spec.modulated_mode_detuning,
        )

        detunings = np.array([sim.spec.modulated_mode_detuning for sim in sims])

        getter = lambda s: s.mode_output_powers(s.lookback.mean)[modulated_mode_index]
        efficiency = np.empty_like(detunings)
        for idx, (no_mixing_sim, with_mixing_sim) in enumerate(
            zip(without_mixing, sims)
        ):
            efficiency[idx] = (
                getter(with_mixing_sim) - getter(no_mixing_sim)
            ) / launched_mixing_power

        detuning_to_xy[launched_mixing_power] = (detunings, efficiency)

    xx = [x for x, y in detuning_to_xy.values()]
    yy = [y for x, y in detuning_to_xy.values()]

    si.vis.xxyy_plot(
        f"modulation_efficiency",
        xx,
        yy,
        line_labels=[
            fr'$P_{{\mathrm{{mixing}}}} = {mixing_power / u.uW:.1f} \, \mathrm{{\mu W}}$'
            for mixing_power in nonzero_mixing_powers
        ],
        title=rf"Modulation Efficiency",
        x_label="Modulated Mode Detuning",
        y_label="Modulation Efficiency",
        x_unit="Hz",
        x_log_axis=True,
        y_log_axis=True,
        font_size_legend=8,
        y_lower_limit=1e-10,
        y_upper_limit=1,
        target_dir=OUT_DIR / path.stem,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        # names = ["mock_meff.sims"]
        # paths = (Path(__file__).parent / name for name in names)
        #
        # for path in paths:
        #     mode_energy_plot_vs_pump_power_per_mixing_power(path)
        #     modulation_efficiency_vs_pump_power_by_mixing_power(path)

        names = ["mock_detuning_scan.sims"]
        paths = (Path(__file__).parent / name for name in names)

        for path in paths:
            mode_energy_plot_vs_modulated_mode_detuning_per_mixing_power(path)
            modulation_efficiency_vs_modulated_mode_detuning_by_mixing_power(path)
