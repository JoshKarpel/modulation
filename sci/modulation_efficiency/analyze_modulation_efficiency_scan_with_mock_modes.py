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


def mode_energy_plot_by_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)
    s = ps[0].spec
    modes = s.modes
    idxs = [ps[0].mode_to_index[mode] for mode in modes]

    for four_mode_detuning_cutoff in ps.parameter_set("four_mode_detuning_cutoff"):
        for launched_mixing_power in ps.parameter_set("launched_mixing_power"):
            sims = ps.select(
                launched_mixing_power=launched_mixing_power,
                four_mode_detuning_cutoff=four_mode_detuning_cutoff,
            )

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

            fmdc_str = (
                f"{four_mode_detuning_cutoff / u.THz:.1f}THz"
                if four_mode_detuning_cutoff is not None
                else "None"
            )
            postfix = f"mixing={launched_mixing_power / u.mW:.6f}mW__fmdc={fmdc_str}"

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


def modulation_efficiency_plot_by_mixing_power(path):
    ps = analysis.ParameterScan.from_file(path)

    without_mixing = sorted(
        ps.select(launched_mixing_power=0), key=lambda s: s.spec.launched_pump_power
    )
    s = without_mixing[0].spec

    sidebands = sorted(s.wavelength_bounds)

    mixing_sideband = microspheres.sideband_of_wavelength(
        s.mixing_wavelength, sidebands
    )
    modulated_mixing_sideband = sidebands[sidebands.index(mixing_sideband) - 1]

    sidebands_to_modes = microspheres.group_modes_by_sideband(
        s.modes, s.wavelength_bounds
    )

    nonzero_mixing_powers = sorted(ps.parameter_set("launched_mixing_power") - {0})

    for modulated_mode in sidebands_to_modes[modulated_mixing_sideband]:
        modulated_mode_index = s.modes.index(modulated_mode)
        mixing_power_to_xy = {}
        for mixing_power in nonzero_mixing_powers:
            sims = sorted(
                ps.select(mixing_power=mixing_power),
                key=lambda s: s.spec.launched_pump_power,
            )

            pump_powers = np.array([sim.spec.launched_pump_power for sim in sims])

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
        names = ["test_mock.sims"]
        paths = (Path(__file__).parent / name for name in names)

        for path in paths:
            mode_energy_plot_by_mixing_power(path)
            # modulation_efficiency_plot_by_mixing_power(path)
