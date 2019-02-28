import logging
from pathlib import Path

import numpy as np

import simulacra as si
import simulacra.units as u

from modulation import raman
from modulation.resonators import microspheres

THIS_FILE = Path(__file__)
OUT_DIR = THIS_FILE.parent / "out" / THIS_FILE.stem
SIM_LIB = OUT_DIR / "SIMLIB"

LOGMAN = si.utils.LogManager("simulacra", "modulation", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    t = np.geomspace(1 * u.fsec, 1 * u.sec, 10000)
    pump_omega = 375 * u.THz
    mode_omega = 374 * u.THz

    explicit = np.exp(1j * mode_omega * t) * np.cos(pump_omega * t)
    implicit = (
        np.exp(1j * (mode_omega - pump_omega) * t)
        + np.exp(1j * (mode_omega + pump_omega) * t)
    ) / 2

    # si.vis.xy_plot(
    #     'explicit_and_implicit',
    #     t,
    #     np.real(explicit),
    #     np.imag(explicit),
    #     np.real(implicit),
    #     np.imag(implicit),
    #     line_kwargs = [
    #         {'color': 'blue', },
    #         {'color': 'red', },
    #         {'color': 'teal', 'linestyle': '--', },
    #         {'color': 'pink', 'linestyle': '--', },
    #     ],
    #     line_labels = [
    #         'explicit re',
    #         'explicit im',
    #         'implicit re',
    #         'implicit im',
    #     ],
    #     x_unit = 'fsec',
    #     **PLOT_KWARGS,
    # )

    si.vis.xy_plot(
        "diff",
        t,
        np.abs(np.real(explicit) - np.real(implicit)),
        np.abs(np.imag(explicit) - np.imag(implicit)),
        line_kwargs=[{"alpha": 0.8}, {"alpha": 0.8}],
        line_labels=["Real", "Imag"],
        x_unit="usec",
        x_log_axis=True,
        y_log_axis=True,
        x_label=r"Time $t$",
        y_label="Absolute Difference",
        y_lower_limit=1e-16,
        y_upper_limit=1,
        **PLOT_KWARGS,
    )
