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


def mode_kwargs(q, mode, pump_mode, wavelength_bounds):
    base = {}

    if mode == pump_mode:
        return {"color": "black", **base}

    i = 0
    for bound in wavelength_bounds:
        if mode.wavelength in bound:
            return {
                "linestyle": LINESTYLES[i % len(LINESTYLES)],
                "color": COLORS[q % len(COLORS)],
                **base,
            }
        i += 1


def mode_energy_plot_by_mixing_power(path):
    lower_lim = 1e-7 * u.pJ

    all_sims = analysis.ParameterScan.from_file(path)
    s = all_sims[0].spec
    modes = s.modes
    idxs = [all_sims[0].mode_to_index[mode] for mode in modes]
    pump_mode = s.pump_mode
    wavelength_bounds = s.wavelength_bounds

    # for idx, mode in enumerate(modes):
    #     print(mode, f"{s.mode_coupling_quality_factors[idx]:.4g}")

    for mixing_power in all_sims.parameter_set("mixing_power"):
        sims = all_sims.select(mixing_power=mixing_power)

        scan_powers = np.array([sim.spec.pump_power for sim in sims])
        means = [sim.mode_energies(sim.lookback.mean) for sim in sims]
        powers = [sim.mode_output_powers(sim.lookback.mean) for sim in sims]

        energy_lines = []
        power_lines = []
        line_kwargs = []
        line_labels = []
        for idx, mode in zip(idxs, modes):
            energy = np.array([energies[idx] for energies in means])
            power = np.array([pow[idx] for pow in powers])

            if lower_lim is not None and not np.any(energy >= lower_lim):
                continue

            energy_lines.append(energy)
            power_lines.append(power)
            line_kwargs.append(mode_kwargs(idx, mode, pump_mode, wavelength_bounds))
            line_labels.append(fr"${mode.tex}$")

        si.vis.xy_plot(
            f"mode_energies__mixing={mixing_power / u.mW:.6f}mW",
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
            y_lower_limit=lower_lim,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            f"mode_powers__mixing={mixing_power / u.mW:.6f}mW",
            scan_powers,
            *power_lines,
            line_kwargs=line_kwargs,
            line_labels=line_labels,
            x_label=f"Launched Pump Power",
            y_label="Steady-State Mode Output Power",
            x_unit="mW",
            y_unit="nW",
            x_log_axis=scan_powers[0] != 0,
            y_log_axis=True,
            legend_on_right=True,
            font_size_legend=6,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


def modulation_efficiency_plot_by_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    without_mixing = sorted(ps.select(mixing_power=0), key=lambda s: s.spec.pump_power)
    s = without_mixing[0].spec

    sidebands = sorted(s.wavelength_bounds)

    mixing_sideband = microspheres.sideband_of_wavelength(
        s.mixing_wavelength, sidebands
    )
    modulated_mixing_sideband = sidebands[sidebands.index(mixing_sideband) - 1]

    sidebands_to_modes = microspheres.group_modes_by_sideband(
        s.modes, s.wavelength_bounds
    )

    nonzero_mixing_powers = sorted(ps.parameter_set("mixing_power") - {0})

    for modulated_mode in sidebands_to_modes[modulated_mixing_sideband]:
        modulated_mode_index = s.modes.index(modulated_mode)
        mixing_power_to_xy = {}
        for mixing_power in nonzero_mixing_powers:
            sims = sorted(
                ps.select(mixing_power=mixing_power), key=lambda s: s.spec.pump_power
            )

            pump_powers = np.array([sim.spec.pump_power for sim in sims])

            getter = lambda s: s.mode_output_powers(s.lookback.mean)[
                modulated_mode_index
            ]
            efficiency = np.empty_like(pump_powers)
            for idx, (no_mixing_sim, with_mixing_sim) in enumerate(
                zip(without_mixing, sims)
            ):
                efficiency[idx] = (
                    getter(with_mixing_sim) - getter(no_mixing_sim)
                ) / mixing_power

            mixing_power_to_xy[mixing_power] = (pump_powers, efficiency)

        xx = [x for x, y in mixing_power_to_xy.values()]
        yy = [y for x, y in mixing_power_to_xy.values()]

        si.vis.xxyy_plot(
            f"modulation_efficiency__mode_at_{modulated_mode.wavelength / u.nm:.6f}nm",
            xx,
            yy,
            line_labels=[
                fr'$P_{{\mathrm{{mixing}}}} = {mixing_power / u.uW:.3f} \, \mathrm{{\mu W}}$'
                for mixing_power in nonzero_mixing_powers
            ],
            title=rf"Modulation Efficiency for ${modulated_mode.tex}$",
            x_label="Launched Pump Power",
            y_label="Modulation Efficiency",
            x_unit="mW",
            x_log_axis=True,
            y_log_axis=True,
            target_dir=OUT_DIR / path.stem,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with LOGMAN as logger:
        paths = [
            # Path(__file__).parent / "meff_test.sims",
            # Path(__file__).parent / "meff_test_v2.sims",
            Path(__file__).parent
            / "meff_test_with_new_amps_coupling__minimal_set.sims"
        ]

        for path in paths:
            mode_energy_plot_by_mixing_power(path)
            modulation_efficiency_plot_by_mixing_power(path)
