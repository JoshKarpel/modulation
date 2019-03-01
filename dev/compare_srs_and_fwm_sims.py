#!/usr/bin/env python
import itertools
import logging
from pathlib import Path
from copy import deepcopy

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

ANIM_KWARGS = dict(target_dir=OUT_DIR)


def find_mode(modes, omega: float):
    return sorted(modes, key=lambda m: abs(m.omega - omega))[0]


def compare(dt):
    with LOGMAN as logger:
        pump_wavelength = 1064 * u.nm
        pump_power = 100 * u.uW

        ###

        material = raman.material.RamanMaterial.from_name("silica")
        pump_omega = u.twopi * u.c / pump_wavelength

        modes = [
            mock.MockMode(
                label=f"pump",
                omega=pump_omega,
                index_of_refraction=1.45,
                mode_volume_inside_resonator=1e-20,
            ),
            mock.MockMode(
                label=f"stokes",
                omega=pump_omega - material.modulation_omega,
                # + (0.079_384_839_103 * u.THz),
                index_of_refraction=1.45,
                mode_volume_inside_resonator=1e-20,
            ),
        ]

        pump_mode = modes[0]
        stokes_mode = modes[1]
        df = abs(pump_mode.frequency - stokes_mode.frequency)
        pumps = [
            raman.pump.ConstantMonochromaticPump(
                frequency=pump_mode.frequency, power=pump_power
            )
        ]

        spec_kwargs = dict(
            material=material,
            mode_volume_integrator=mock.MockVolumeIntegrator(
                volume_integral_result=1e-25
            ),
            modes=modes,
            mode_initial_amplitudes={stokes_mode: 1},
            mode_intrinsic_quality_factors=dict(zip(modes, itertools.repeat(1e8))),
            mode_coupling_quality_factors=dict(zip(modes, itertools.repeat(1e8))),
            pumps=pumps,
            time_initial=0 * u.nsec,
            time_final=1 * u.usec,
            time_step=dt,
            store_mode_amplitudes_vs_time=True,
        )

        srs = raman.StimulatedRamanScatteringSpecification(
            "srs", **deepcopy(spec_kwargs)
        ).to_sim()
        fwm = raman.FourWaveMixingSpecification("fwm", **deepcopy(spec_kwargs)).to_sim()

        srs.run(progress_bar=True)
        # srs.plot.mode_magnitudes_vs_time(y_log_axis=False, **PLOT_KWARGS)

        fwm.run(progress_bar=True)
        # fwm.plot.mode_magnitudes_vs_time(y_log_axis=False, **PLOT_KWARGS)

        si.vis.xy_plot(
            f"compare__dt={spec_kwargs['time_step'] / u.psec:.6f}ps__{df * dt}",
            srs.times,
            srs.mode_magnitudes_vs_time[:, 0],
            srs.mode_magnitudes_vs_time[:, 1],
            fwm.mode_magnitudes_vs_time[:, 0],
            fwm.mode_magnitudes_vs_time[:, 1],
            line_labels=["srs stokes", "srs pump", "fwm stokes", "fwm pump"],
            line_kwargs=[None, None, {"linestyle": "--"}, {"linestyle": "--"}],
            x_unit="nsec",
            y_lower_limit=0,
            y_upper_limit=4e9,
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    # compare(10 * u.psec)

    dts = (
        np.array(
            [
                10,
                10.1,
                10.01,
                10.001,
                10.0001,
                10.00001,
                1,
                1.1,
                1.01,
                1.001,
                1.0001,
                1.00001,
            ]
        )
        * u.psec
    )
    si.utils.multi_map(compare, dts, processes=2)
