import itertools
import logging
from pathlib import Path

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman, analysis
from modulation.resonators import mock

import matplotlib.pyplot as plt

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def test():
    ps = analysis.ParameterScan.from_file(Path(__file__).parent / "test.sims")
    print(ps, len(ps))

    pump_powers = ps.parameter_set("_pump_power")
    mixing_powers = ps.parameter_set("_mixing_power")

    print(len(pump_powers), pump_powers)
    print(len(mixing_powers), mixing_powers)

    x = []
    y = []
    sims_by_mixing_power = {mp: ps.select(_mixing_power=mp) for mp in mixing_powers}
    for mixing_power, sims in sims_by_mixing_power.items():
        x.append([s.spec._pump_power for s in sims])
        y.append(
            [
                s.mode_energies(s.lookback.mean)[
                    s.mode_to_index[s.spec._modulated_mode]
                ]
                for s in sims
            ]
        )

    si.vis.xxyy_plot(
        "test",
        x,
        y,
        x_unit="uW",
        y_unit="pJ",
        x_label="Pump Power",
        y_label="Energy in Modulated Mixing Mode",
        **PLOT_KWARGS,
    )


def more_test():
    ps = analysis.ParameterScan.from_file(
        Path(__file__).parent / "scan_pump_power_by_mmm_q_and_mixing_power.sims"
    )

    pump_powers = sorted(ps.parameter_set("_pump_power"))
    mixing_powers = sorted(ps.parameter_set("_mixing_power"))
    coupling_quality_factors = sorted(
        ps.parameter_set("_mixing_and_modulated_intrinsic_q")
    )
    intrinsic_quality_factors = sorted(
        ps.parameter_set("_mixing_and_modulated_coupling_q")
    )

    pump_mode = ps.sims[0].spec._pump_mode
    stokes_mode = ps.sims[0].spec._stokes_mode
    mixing_mode = ps.sims[0].spec._mixing_mode
    modulated_mode = ps.sims[0].spec._modulated_mode
    modes = (pump_mode, stokes_mode, mixing_mode, modulated_mode)
    mti = ps.sims[0].mode_to_index

    sims_by_mixing_power = {
        (mp, qi, qc): ps.select(
            _mixing_power=mp,
            _mixing_and_modulated_intrinsic_q=qi,
            _mixing_and_modulated_coupling_q=qc,
        )
        for mp in mixing_powers
        for qi in intrinsic_quality_factors
        for qc in coupling_quality_factors
    }
    for (mixing_power, intrinsic_q, coupling_q), sims in tqdm(
        sims_by_mixing_power.items()
    ):
        x = [s.spec._pump_power for s in sims]
        ys = [
            [s.mode_energies(s.lookback.mean)[mti[mode]] for s in sims]
            for mode in modes
        ]

        si.vis.xy_plot(
            f"Pm={mixing_power / u.uW:.6f}uW_Qi=10^{int(np.log10(intrinsic_q))}_Qc=10^{int(np.log10(coupling_q))}",
            x,
            *ys,
            line_labels=[mode.label for mode in modes],
            x_unit="uW",
            y_unit="pJ",
            y_log_axis=True,
            y_lower_limit=1e-10 * u.pJ,
            y_upper_limit=1e2 * u.pJ,
            x_label="Pump Power",
            y_label=r"$\mathcal{U}_q$",
            title=f'$P_m = {mixing_power / u.uW:.6f} \, \mathrm{{\mu W}}, \, Q_i = 10^{int(np.log10(intrinsic_q))}, \, Q_c = 10^{int(np.log10(coupling_q))}$',
            **PLOT_KWARGS,
        )


COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
STYLES = ["-", "--", ":"]


def all_in_one():
    ps = analysis.ParameterScan.from_file(
        Path(__file__).parent
        / "scan_pump_power_by_mmm_q_and_mixing_power_with_ref.sims"
    )

    pump_powers = sorted(ps.parameter_set("_pump_power"))
    mixing_powers = sorted(ps.parameter_set("_mixing_power"))
    coupling_quality_factors = sorted(
        ps.parameter_set("_mixing_and_modulated_intrinsic_q")
    )
    intrinsic_quality_factors = sorted(
        ps.parameter_set("_mixing_and_modulated_coupling_q")
    )

    pump_mode = ps.sims[0].spec._pump_mode
    stokes_mode = ps.sims[0].spec._stokes_mode
    mixing_mode = ps.sims[0].spec._mixing_mode
    modulated_mode = ps.sims[0].spec._modulated_mode
    modes = (pump_mode, stokes_mode, mixing_mode, modulated_mode)
    mti = ps.sims[0].mode_to_index

    with si.vis.FigureManager("modulation_efficiency_matrix", **PLOT_KWARGS) as fm:
        fig = fm.fig
        axes = [[None for _ in range(3)] for _ in range(3)]
        for row, col in itertools.product(range(3), repeat=2):
            ax = fig.add_subplot(
                3,
                3,
                (row * 3) + col + 1,
                sharex=axes[0][0] if row != 0 and col != 0 else None,
                sharey=axes[0][0] if row != 0 and col != 0 else None,
            )
            axes[row][col] = ax

            ax.set_xlim(1e-4, 1e2)
            ax.set_xscale("log")

            ax.set_ylim(1e-10, 1)
            ax.set_yscale("log")

            ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
            ax.set_yticklabels(
                ["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"]
            )

            ax.set_xticks([1e-2, 1, 1e2])
            ax.set_xticklabels(["$10^{-2}$", "$1$", "$10^2$"])

            ax.grid(True)
            ax.tick_params(
                left=col == 0,
                right=False,
                top=False,
                bottom=row == 2,
                labelleft=col == 0,
                labelright=False,
                labeltop=False,
                labelbottom=row == 2,
                labelsize=8,
            )

        fig.subplots_adjust(wspace=0.02, hspace=0.035)

        mp_to_color = dict(zip(mixing_powers[1:], COLORS))
        mp_to_syle = dict(zip(mixing_powers[1:], STYLES))

        zero_mixing_power_sims = {
            (qi, qc): sorted(
                ps.select(
                    _mixing_power=mixing_powers[0],
                    _mixing_and_modulated_intrinsic_q=qi,
                    _mixing_and_modulated_coupling_q=qc,
                ),
                key=lambda sim: sim.spec._pump_power,
            )
            for qi in intrinsic_quality_factors
            for qc in coupling_quality_factors
        }
        for mp in mixing_powers[1:]:
            sims_by_mixing_power = {
                (qi, qc): sorted(
                    ps.select(
                        _mixing_power=mp,
                        _mixing_and_modulated_intrinsic_q=qi,
                        _mixing_and_modulated_coupling_q=qc,
                    ),
                    key=lambda sim: sim.spec._pump_power,
                )
                for qi in intrinsic_quality_factors
                for qc in coupling_quality_factors
            }

            for idx, ((intrinsic_q, coupling_q), sims) in enumerate(
                tqdm(sims_by_mixing_power.items())
            ):
                zero = zero_mixing_power_sims[(intrinsic_q, coupling_q)]

                row, col = divmod(idx, 3)
                ax = axes[row][col]

                x = np.array([s.spec._pump_power for s in sims])
                y = np.array(
                    [
                        (
                            s.mode_output_powers(s.lookback.mean)[mti[modulated_mode]]
                            - z.mode_output_powers(z.lookback.mean)[mti[modulated_mode]]
                        )
                        / mp
                        for s, z in zip(sims, zero)
                    ]
                )

                ax.plot(
                    x / u.mW,
                    y,
                    color=mp_to_color[mp],
                    linestyle=mp_to_syle[mp],
                    linewidth=1.5,
                    label=rf'$P_m = {mp / u.uW:.0f} \, \mathrm{{\mu W}}$',
                )

                if col == 0:
                    ax.set_ylabel(
                        rf"$Q_i = 10^{int(np.log10(intrinsic_q))}$", fontsize=9
                    )
                if row == 2:
                    ax.set_xlabel(
                        rf"$Q_c = 10^{int(np.log10(coupling_q))}$", fontsize=9
                    )

        bigax = fig.add_subplot(111, frameon=False)
        bigax.tick_params(
            labelcolor="none", top=False, bottom=False, left=False, right=False
        )
        bigax.set_xlabel(r"Pump Power ($\mathrm{{mW}}$)", labelpad=17, fontsize=12)
        bigax.set_ylabel(r"Modulation Efficiency", labelpad=22, fontsize=12)

        axes[2][0].set_xticks([1e-4, 1e-2, 1, 1e2])
        axes[2][0].set_xticklabels(["$10^{-4}$", "$10^{-2}$", "$1$", "$10^2$"])

        axes[2][0].set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
        axes[2][0].set_yticklabels(
            ["$10^{-10}$", "$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"]
        )

        axes[0][1].legend(
            bbox_to_anchor=(0, 1.12, 1, 0.1),
            loc="center",
            ncol=3,
            borderaxespad=0.0,
            fontsize=8,
            frameon=False,
            handletextpad=0.5,
        )


if __name__ == "__main__":
    with LOGMAN as logger:
        # test()
        # more_test()
        all_in_one()
